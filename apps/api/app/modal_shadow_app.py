"""The shadow workload as Modal runs it, on Modal's GPU, in Modal's container (R6-T2).

This file is the *remote* half of Modal shadow execution and the only part of this
repository that ever executes on somebody else's machine. It is deployed with

    modal deploy apps/api/app/modal_shadow_app.py

and after that it is never imported by the API or the worker again — they reach the deployed
function by name over Modal's API, which is what `app.modal_client` does.

Two properties of this file are deliberate and worth stating, because both are the kind of
thing a later edit would undo without noticing:

- **it imports nothing from `app`.** Not `app.limits`, not `app.config`, not `app.shadow`.
  `modal deploy` imports this module in whatever shell the operator ran it from, and
  `app/__init__.py` refuses to import without the full production credential set — so a
  single `from app...` here would mean a deploy that only works on a machine configured to
  run the service. It would also mean the *container* Modal builds needs those credentials.
  The one thing this module shares with the local half is the three names at the top, and
  `app.modal_client` restates them rather than importing them, precisely so that this stays
  true. `tests/test_modal_shadow.py` asserts the two agree;

- **it is a stub, and it stays a stub in R6-T2.** No detector, no weights, no media. The
  question this task asks is whether a shadow run can be executed on a remote GPU and land
  back in `shadow_runs` without production noticing, and answering it with a real model
  would have meant reviewing an execution backend and a detector at once. What the stub does
  do is prove the *GPU* — it reads the device out of `nvidia-smi` and reports it, so a run
  that completed is evidence that an NVIDIA GPU was really attached rather than evidence
  that a `sleep` returned.

The image is deliberately the bare Debian slim one. A shadow workload that loads a model
will need its own image with its own pinned dependencies, and building that image now — for
a stub that would not use it — would be paying a cold-start cost to import nothing.
"""

import subprocess
import time
from typing import Optional

import modal

# What the deployed app is called on Modal, and what `app.modal_client` looks it up by. These
# two strings are the entire interface between the local worker and this file.
APP_NAME = "deepguard-shadow"
FUNCTION_NAME = "run_shadow_stub"

# Which deployment of the workload produced an observation, written to
# `shadow_runs.provider_version`. It identifies the *stub*, and a real workload replacing it
# must change this — a corpus that cannot tell one workload's observations from another's is
# not a corpus anything can be calibrated against.
WORKLOAD_VERSION = "modal-stub-1"

# R6 asks for a cost-effective NVIDIA target and names L4 and A10 as the starting points.
# L4 is the cheaper of the two and is more GPU than a stub that runs `nvidia-smi` needs; it
# is chosen now so the number this task reports is one a real workload would also see. At
# Modal's published L4 rate a stub run is a fraction of a cent, and a real detector at ~140 s
# is about $0.03 — which is the figure that makes shadow evaluation affordable at all.
GPU = "L4"

# Modal's own ceiling on one execution, enforced on Modal's side. It bounds the container,
# where the local deadline in `app.limits.shadow_modal_timeout_seconds` bounds the *wait* —
# two different failures needing two different bounds. This one is the smaller: a remote
# container that has hung should be killed by Modal before the worker gives up on it, so the
# usual failure is a clean remote error rather than a local timeout over a container still
# burning GPU time.
TIMEOUT_SECONDS = 600

# How long Modal keeps a warm container after a run. Shadow runs arrive one per completed
# analysis and irregularly, so most of them will pay a cold start regardless; two minutes
# costs nothing when idle and spares a burst of analyses from paying it repeatedly.
#
# `min_containers` is deliberately not set, so this scales to zero. A GPU held warm around
# the clock is the failure mode worth naming: an A10G kept up continuously runs to roughly
# $790 a month, against a shadow corpus that accumulates a handful of rows a day. Paying a
# cold start every time is the correct trade here, and it is why the local half waits with a
# heartbeat instead of a short timeout.
SCALEDOWN_WINDOW_SECONDS = 120

# Reading the attached device. `nvidia-smi` is present in Modal's GPU containers and needs no
# package of ours, which is the whole reason the stub proves the GPU this way rather than by
# importing torch — a two-gigabyte dependency to answer a question a 20 ms subprocess answers.
NVIDIA_SMI = (
    "nvidia-smi",
    "--query-gpu=name,driver_version,memory.total",
    "--format=csv,noheader",
)

# The subprocess above is a header read, not work. Anything approaching this is a container
# whose GPU is not answering, and the empty device below is the honest thing to report.
NVIDIA_SMI_TIMEOUT_SECONDS = 30

image = modal.Image.debian_slim(python_version="3.12")

app = modal.App(APP_NAME, image=image)


def attached_gpu() -> Optional[str]:
    """The GPU this container actually has, as the driver names it, or nothing.

    Never raises. The stub's job is to report what it saw, and "the device could not be read"
    is one of the things it might have seen — turning that into an exception would fail a
    shadow run that did in fact execute remotely, which is the observation this task wants.
    """
    try:
        completed = subprocess.run(
            NVIDIA_SMI,
            capture_output=True,
            text=True,
            timeout=NVIDIA_SMI_TIMEOUT_SECONDS,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None

    device = completed.stdout.strip().splitlines()
    return device[0].strip() if device else None


@app.function(
    name=FUNCTION_NAME,
    gpu=GPU,
    timeout=TIMEOUT_SECONDS,
    scaledown_window=SCALEDOWN_WINDOW_SECONDS,
)
def run_shadow_stub(analysis_id: str) -> dict:
    """Execute the shadow stub on a remote GPU and return what it observed.

    The return value is written into `shadow_runs.evidence` verbatim and interpreted by
    nothing, so it has to be a plain JSON-serializable document — no dataclasses, no numpy
    scalars, nothing whose repr survives the wire but whose type does not survive `JSONB`.

    `workload_version` is the one key the local half insists on: `app.modal_client` refuses a
    result without it, because an observation whose deployment identity is unknown is an
    observation nothing can be calibrated from later.
    """
    started = time.monotonic()
    device = attached_gpu()

    return {
        "observed": True,
        "analysis_id": analysis_id,
        "backend": "modal",
        "workload_version": WORKLOAD_VERSION,
        "gpu_requested": GPU,
        "gpu_attached": device,
        "remote_seconds": round(time.monotonic() - started, 3),
    }
