"""Reaching the shadow workload that runs on Modal, and refusing to when it isn't there (R6-T2).

R6-T1 built the shadow queue and ran a stub inside the worker process. This module is the
other end of the same experiment: the stub now runs on a remote NVIDIA GPU that Modal rents
by the second, and this is the local half that starts it, waits for it and hands back what it
said. `app.modal_shadow_app` is the remote half — deployed once, never imported here.

Everything in this module exists to keep one promise: a shadow run reaching Modal is an
experiment, and an experiment must not be able to cost a customer anything. Four properties
make that structural rather than careful:

- **it is opt-in twice over.** `DEEPGUARD_SHADOW_MODAL` has to be set *and* both Modal tokens
  have to be present before `configured()` says yes. A deployment that has shadow mode on but
  no Modal configuration keeps running the local stub exactly as R6-T1 left it;

- **`modal` is imported inside a function, never at module scope.** The API imports `app.*`
  eagerly and the worker imports this module through `app.shadow`; a top-level `import modal`
  would make a missing or broken Modal SDK a process that will not start. Here it is one
  `ModalUnavailable` on one shadow run. The same reasoning applies to Modal's configuration:
  the SDK reads its tokens when it is imported, so importing it only after `configured()` has
  said the tokens are there is what stops it from quietly authenticating as whatever
  `~/.modal.toml` happens to hold on a developer's machine;

- **every failure is this module's own exception type.** Modal's SDK raises from `modal.exception`,
  gRPC raises its own, a hung network raises `OSError`. All of them are caught and re-raised
  as a `ModalShadowError`, so `app.shadow` can fail one run without knowing what a Modal
  error looks like, and so a class name written into `shadow_runs.error_message` names a
  DeepGuard concept rather than leaking a vendor's internals into a column;

- **nothing here waits.** The remote call is `spawn`ed and the handle handed back; `collect`
  asks with `timeout=0` and says "not yet" rather than blocking. This is the property the
  worker actually depends on, and it was added after a runtime test caught its absence: with
  a blocking wait, a production job submitted while a Modal call was in flight was claimed
  8 ms *after* that call returned, having waited 4.1 s for an experiment. A cold start would
  have made that 30-90 seconds and a hung Modal the full local deadline. Waiting is bounded
  by a deadline carried on the handle, and the lease is renewed by whoever is doing the
  asking.

What this module deliberately is not: a provider abstraction. There is one backend, reached
by two names, and a second one would be a second module rather than a registry this one is
made to fit into.
"""

import json
import logging
import os
import time
import uuid
from dataclasses import dataclass
from typing import Any

from app.limits import shadow_modal_timeout_seconds

logger = logging.getLogger(__name__)

# Whether this deployment wants shadow runs executed on Modal rather than in-process. Off
# unless set, and separate from `DEEPGUARD_SHADOW_MODE`: shadow mode is whether experiments
# run at all, this is where they run. A stack can have the first without the second, and that
# is the R6-T1 behaviour, unchanged.
MODAL_SHADOW_VARIABLE = "DEEPGUARD_SHADOW_MODAL"

# The values that turn it on, spelled out for the same reason `app.shadow` spells them out:
# a compose file eventually carries `=false`, and "any non-empty value is on" reads that
# exactly backwards.
ENABLED_VALUES = frozenset({"1", "true", "yes", "on"})

# Modal's own credential pair, under the names Modal's SDK already reads them from. Not
# renamed into a `DEEPGUARD_` prefix: these are handed to somebody else's client, and a
# variable whose name differs from the one the vendor documents is a variable an operator
# sets in the wrong place.
TOKEN_ID_VARIABLE = "MODAL_TOKEN_ID"
TOKEN_SECRET_VARIABLE = "MODAL_TOKEN_SECRET"

# The deployed app and function, restated rather than imported from `app.modal_shadow_app` —
# importing that module here would drag `import modal` back to module scope, which is the one
# thing this file is arranged to avoid. `tests/test_modal_shadow.py` asserts the two agree,
# so the duplication is checked rather than trusted.
APP_NAME = "deepguard-shadow"
FUNCTION_NAME = "run_shadow_stub"

# The key the remote result has to carry. A result without it is refused: an observation
# whose deployment identity is unknown cannot be told apart from the next workload's, which
# makes it useless to the offline calibration these rows exist for.
VERSION_KEY = "workload_version"

# There is deliberately no poll interval here. Nothing in this module waits: `spawn_stub`
# returns as soon as Modal has accepted the call and `collect` asks with `timeout=0`. How
# often the question gets asked is `app.shadow`'s business, and it is answered by the worker's
# own idle poll — which is also where the lease gets renewed, so a cold start on this
# workspace (measured at 30-90 seconds for a container that has scaled to zero) is a handful
# of instant checks rather than one long blind wait.


class ModalShadowError(Exception):
    """Base class for every failure raised by this module.

    One base class because `app.shadow` treats them identically — any of them fails one
    shadow run and nothing else. They are distinguished at all so the class name written to
    `shadow_runs.error_message` says which kind of thing went wrong when the rows are read
    back offline.
    """


class ModalUnavailable(ModalShadowError):
    """Modal could not be reached or used at all. Nothing was executed remotely."""


class ModalNotConfigured(ModalUnavailable):
    """This deployment did not ask for Modal, or did not give it credentials."""


class ModalTimeout(ModalShadowError):
    """The remote execution outlived the local deadline. It may still be running on Modal."""


class ModalExecutionError(ModalShadowError):
    """Modal ran the workload and it did not produce a usable result."""


@dataclass(frozen=True)
class ModalResult:
    """What the remote workload returned, validated but uninterpreted.

    Deliberately not `app.shadow.ShadowObservation`: returning that type would mean importing
    `app.shadow` here, and `app.shadow` imports this module. The conversion is one line at the
    caller, which is cheaper than a circular import or a shared types module holding two
    fields.
    """

    provider_version: str
    evidence: dict[str, Any]


@dataclass(frozen=True)
class ModalCall:
    """A remote call that has been started and not yet collected.

    Held in the worker's memory rather than written to the shadow row, and that is a decision
    worth stating. Persisting the call id would let a *different* worker collect a call this
    one started, which sounds like an improvement and is really a second scheduler: two
    workers would then need to agree about who owns an in-flight remote call, on top of the
    lease they already use to agree about who owns the row.

    What happens instead when this worker dies is what already happened in R6-T1: the lease
    stops being renewed, `recover_stale_runs` fails the row, and one analysis is missing from
    a corpus that is measured offline. The remote container is left to Modal's own timeout.
    That is the correct cost for an experiment, and it needs no new column and no migration.
    """

    call: Any
    analysis_id: uuid.UUID
    deadline: float


def enabled() -> bool:
    """Whether this deployment has asked for Modal-backed shadow execution."""
    return os.getenv(MODAL_SHADOW_VARIABLE, "").strip().lower() in ENABLED_VALUES


def credentials_present() -> bool:
    """Whether both Modal tokens are set to something.

    Empty is missing, exactly as `app.config` treats it and for the same reason: a compose
    file listing a variable the host never set passes it through as the empty string, and a
    client authenticating with an empty token would fail on the wire instead of here.
    """
    return all(
        os.getenv(variable, "").strip()
        for variable in (TOKEN_ID_VARIABLE, TOKEN_SECRET_VARIABLE)
    )


def configured() -> bool:
    """Whether a shadow run should be executed on Modal rather than in-process.

    Both halves, because either one alone is a misconfiguration that should degrade rather
    than fail: a flag without credentials is a deployment that meant to enable Modal and did
    not finish, and credentials without a flag is a machine that happens to have a Modal
    login. Neither is a reason to stop running the local stub.
    """
    return enabled() and credentials_present()


def _load_modal():
    """Import Modal's SDK now, or say it is unavailable.

    Inside a function on purpose — see the module docstring. `Exception` rather than
    `ImportError`, because a broken installation of a package this large fails in more ways
    than one and every one of them means the same thing here.
    """
    try:
        import modal
        import modal.exception
    except Exception as error:
        raise ModalUnavailable(
            "The modal SDK could not be imported; no shadow run was executed remotely."
        ) from error

    return modal


def _deployed_function(modal):
    """A handle on the function `modal deploy` published, or `ModalUnavailable`.

    Looking it up by name rather than holding a reference to `app.modal_shadow_app` is what
    keeps the remote definition out of this process entirely: the worker does not need the
    workload's code, its image or its dependencies to ask for it to be run.
    """
    try:
        return modal.Function.from_name(APP_NAME, FUNCTION_NAME)
    except Exception as error:
        raise ModalUnavailable(
            f"The Modal function {APP_NAME}/{FUNCTION_NAME} could not be resolved; "
            "it may not be deployed."
        ) from error


def _validated(result: Any) -> ModalResult:
    """Turn whatever came back off the wire into an observation, or refuse it.

    Three checks, each of which is a real failure mode of a remote call rather than a
    defensive habit: a workload that returned something other than a document, a document
    holding values `JSONB` cannot store, and a document that does not say which deployment
    produced it. All three are `ModalExecutionError` — Modal did run something; what it ran
    did not answer the question.
    """
    if not isinstance(result, dict):
        raise ModalExecutionError(
            f"The Modal workload returned {type(result).__name__}, not a document."
        )

    try:
        json.dumps(result)
    except (TypeError, ValueError) as error:
        raise ModalExecutionError(
            "The Modal workload returned a document that is not JSON-serializable."
        ) from error

    version = result.get(VERSION_KEY)
    if not isinstance(version, str) or not version.strip():
        raise ModalExecutionError(
            f"The Modal workload returned no usable {VERSION_KEY}."
        )

    return ModalResult(provider_version=version.strip(), evidence=result)


def spawn_stub(analysis_id: uuid.UUID) -> "ModalCall":
    """Start the deployed shadow stub on Modal's GPU and return immediately.

    Starting and collecting are two functions rather than one for a reason that is about the
    *worker*, not about Modal. The worker is a single loop that claims production jobs; any
    function here that waited for a remote GPU would hold that loop for the length of a cold
    start, and an analysis submitted in the meantime would sit queued behind an experiment.
    That is the exact inversion shadow mode exists to prevent, and it was measured happening
    before this split: a production job was claimed 8 ms after an in-flight Modal call
    returned, having waited 4.1 s for it.

    So nothing here blocks. The handle goes back to `app.shadow`, which asks about it again
    on later idle polls, and the loop is free the whole time.

    Raises only `ModalShadowError` subclasses, so the caller does not need to know what a
    Modal exception is.
    """
    if not configured():
        raise ModalNotConfigured(
            f"{MODAL_SHADOW_VARIABLE} is not enabled, or the Modal credentials are absent."
        )

    modal = _load_modal()
    function = _deployed_function(modal)

    try:
        call = function.spawn(str(analysis_id))
    except Exception as error:
        raise ModalUnavailable("Starting the Modal shadow workload failed.") from error

    logger.info(
        "Spawned Modal shadow workload %s/%s for analysis %s.",
        APP_NAME,
        FUNCTION_NAME,
        analysis_id,
    )

    return ModalCall(
        call=call,
        analysis_id=analysis_id,
        deadline=time.monotonic() + shadow_modal_timeout_seconds(),
    )


def collect(pending: "ModalCall") -> ModalResult | None:
    """Ask whether the remote call has finished. `None` means it has not.

    `timeout=0` is the whole point: one round trip that returns whatever is ready and does not
    wait for anything that is not. A collect that blocked even briefly would put the worker
    loop back inside Modal's SDK, which is the thing this module is arranged to keep it out
    of.

    The local deadline is checked here rather than by a timer, because here is where time is
    noticed. It bounds the *waiting*, not the remote container — `app.modal_shadow_app`
    carries Modal's own ceiling on execution — and the two are different failures: a workload
    that hangs, and a Modal that stopped answering.

    Raises `ModalShadowError` subclasses. A raise means this shadow run is over; `None` means
    ask again later.
    """
    modal = _load_modal()

    try:
        return _validated(pending.call.get(timeout=0))
    except modal.exception.FunctionTimeoutError as error:
        raise ModalExecutionError(
            "The Modal shadow workload exceeded its own remote timeout."
        ) from error
    except modal.exception.OutputExpiredError as error:
        raise ModalExecutionError(
            "The Modal shadow workload's result expired before it was read."
        ) from error
    except TimeoutError:
        # Python's *builtin* TimeoutError, and this clause is the one that actually fires.
        # `modal/_functions.py` does not import `TimeoutError` from `modal.exception`, so the
        # bare `raise TimeoutError()` in `poll_function` — "there is no output yet" — raises
        # the builtin. Which matters enormously here, because the builtin is a subclass of
        # `OSError`: without this clause the clause below catches every unfinished call and
        # reports a healthy cold start as a lost connection.
        #
        # That is not a hypothetical. It shipped for the length of one runtime test: every
        # Modal shadow run failed with `ModalUnavailable` about 200 ms after being spawned,
        # and the mocked tests missed it because they raised the documented
        # `modal.exception.TimeoutError` rather than the one Modal really raises.
        pass
    except modal.exception.TimeoutError:
        # The documented type, and the base class of the two handled above. Kept beside the
        # builtin rather than instead of it: this is what the SDK's own exception hierarchy
        # says this condition is, and a version that starts raising it should not be a
        # version that breaks this.
        pass
    except modal.exception.Error as error:
        raise ModalExecutionError(
            f"The Modal shadow workload failed: {type(error).__name__}."
        ) from error
    except OSError as error:
        # A socket that went away underneath the SDK. Unavailable rather than an execution
        # failure: the workload may well be running, we simply cannot hear it.
        raise ModalUnavailable(
            "The connection to Modal was lost while asking about the shadow workload."
        ) from error

    if time.monotonic() > pending.deadline:
        raise ModalTimeout(
            "The Modal shadow workload did not finish within the local deadline."
        )

    return None


def cancel(pending: "ModalCall") -> None:
    """Stop a remote call nobody is going to read, and never raise doing it.

    Called whenever a shadow run ends without its result being written — a lost lease, a local
    deadline, a worker shutting down. A container left running is GPU time billed for an
    answer that is already discarded.

    Best-effort by nature: the reason we are cancelling may be that Modal is unreachable, and
    a cancel that raised would replace a precise failure with whatever the SDK threw while
    giving up.
    """
    try:
        pending.call.cancel()
    except Exception:
        logger.debug("Cancelling the Modal shadow workload failed.", exc_info=True)
