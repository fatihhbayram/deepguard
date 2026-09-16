"""Deep Evidence Enrichment: the path that runs after the verdict, and may not touch it.

R10 splits one job into two paths over the same `Analysis`. `app.worker` is the Fast Decision
Path — media acquisition, NVIDIA SVD, EfficientNet-B7, `evaluate_v5`, verdict persistence — and
it is finished the moment the verdict is durable. This module is the other path: the
evidence-only detectors, run afterwards, writing evidence nobody's verdict depends on.

The whole point is the asymmetry. `docs/architecture/R10_EXECUTION_CONTRACT.md` §1.1:

> The automated verdict is final once the Fast Decision Path completes. Deep Evidence may
> enrich the report, but it must never change `risk_level`, `risk_rule_id`,
> `risk_rules_version`, `risk_calibration_id`, or decision coverage.

Four properties make that structural here rather than conventional:

1. **This module holds no reference to the risk engine.** There is no import of
   `app.risk_engine` below, and `tests/test_enrichment.py` asserts that absence over the
   module rather than trusting it — the shape `tests/test_shadow_mode.py` already uses to
   assert the shadow isolation. §7.2 requires it: "not to confirm the verdict, not to log a
   comparison, not behind a feature flag, not in shadow mode against the persisted value".
   The ruleset version this module handles is read off the decided analysis and never asked
   of the engine, which is why the role table below transcribes its key instead of importing
   it.
2. **Every write goes through `app.enrichment_guard`.** A session that refuses `analyses`,
   `analysis_jobs`, `analysis_reviews`, `media_files` and the deciding detectors' signal rows
   outright. A defect here fails at the boundary instead of quietly rewriting a verdict.
3. **Enrichment is claimed only on a poll that found no decision work.** `app.worker.run`
   asks for a job first, every time, so a queued analysis is never behind an enrichment task
   (§4.2, and the ordering that already keeps shadow mode behind production work).
4. **A failure here is recorded here.** Per component, on its own row, with its own error
   message. Nothing in this module can move an analysis out of `DECIDED` — it cannot reach
   the column (invariant I11).

**Per component, not per analysis.** The unit of *state* is the component: LipForensics has
its own row, its own terminal status and its own error message, because `ENRICHMENT_PARTIAL`
is a claim about which components succeeded and a summary column cannot name either group
(§5.4). The unit of *execution* is the analysis: one claim takes every queued component of one
analysis and runs them against one downloaded artifact. Claiming per component would have
downloaded the same derivative four times and extracted the same WAV twice — `analyse_audio`
answers both audio questions off one extraction, and splitting it would have doubled the
expensive half of it for no state that these rows do not already carry.

**Enrichment reads the derivative the fast path already made.** Not the original, and it does
not transcode again: `media_files.derivative_storage_key` names an object the decision path
uploaded and provably read. Media that needed no derivative is read as the original, which is
the artifact in that case. A transcode per enrichment would have been the expensive step of
the pipeline run twice for evidence nobody is waiting on.

**On-demand is a state, not an absence.** A task written `not_requested` says enrichment was
deferred; an analysis with no task rows at all ran before R10 existed. §5.2 and §8.1 need those
to be different facts, and `app.db.models.AnalysisEnrichmentTask` says more about why the
deferral is a row.

Run from `app.worker`'s loop. This module has no process of its own — §5.4 asks for the
existing async job architecture to be extended rather than for a new queue or a new
orchestration framework, and a second container polling a second table would have been both.
"""

import logging
import os
import tempfile
import threading
import uuid
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db.models import (
    ANALYSIS_STATUS_COMPLETED,
    ANALYSIS_STATUS_FAILED,
    ANALYSIS_STATUS_QUEUED,
    ENRICHMENT_TASK_STATUS_ABSTAINED,
    ENRICHMENT_TASK_STATUS_COMPLETED,
    ENRICHMENT_TASK_STATUS_FAILED,
    ENRICHMENT_TASK_STATUS_NOT_REQUESTED,
    ENRICHMENT_TASK_STATUS_PROCESSING,
    ENRICHMENT_TASK_STATUS_QUEUED,
    PRODUCT_MODE_DEEP_ANALYSIS,
    PRODUCT_MODE_QUICK_SCAN,
    SIGNAL_STATUS_FAILED,
    SIGNAL_STATUS_SUCCESS,
    Analysis,
    AnalysisEnrichmentTask,
    AnalysisSignal,
    MediaFile,
)
from app.db.session import SessionLocal
from app.detection import (
    AASIST_PROVIDER,
    ABSTENTION_ERRORS,
    ACTIVE_SPEAKER_SIGNAL,
    AUDIO_AUTHENTICITY_SIGNAL,
    EFFORT_PROVIDER,
    FACE_FORGERY_SIGNAL,
    LIP_FORENSICS_SIGNAL,
    LIPFORENSICS_PROVIDER,
    NVIDIA_PROVIDER,
)
from app.enrichment_guard import enrichment_session
from app.storage import fetch_object

logger = logging.getLogger(__name__)

# One Deep Evidence component, named as the `analysis_signals` row it writes. A pair rather
# than a name of its own, so the task row, the guard's allowlist, the signal it writes and
# §4.1's membership table all spell the same component the same way.
Component = tuple[str, str]

# Which detectors are evidence-only, per ruleset version (§4.1). The membership of the Deep
# Evidence path, and the reason it is a table keyed by version rather than a flat list:
#
# §3.2 is explicit that a ruleset which makes a detector decision-eligible again — as
# `r5-v3.0.0` did for LipForensics — moves that detector into the Fast Decision Path, and that
# R10-T2 "must not hard-code a detector list that can drift away from it". Keyed by version,
# the membership follows the ruleset by construction: a future `r11-v6.0.0` that promotes
# LipForensics adds an entry without it, and every analysis decided under it stops queueing a
# LipForensics task on the day that entry lands. A flat tuple would have kept queueing one.
#
# The keys are written as literals rather than imported from `app.risk_engine`, and that is the
# one place this module's isolation costs something. It is worth the cost: §7.2 requires that
# the enrichment path hold no reference to the engine at all, and an import for a version
# string is still an import that a later reader could widen into a call. The literal cannot
# drift silently — `tests/test_enrichment.py` asserts these keys against the engine's own
# `RULES_VERSION_V5`, which is a test that may import both because a test decides nothing.
EVIDENCE_ONLY_COMPONENTS: dict[str, tuple[Component, ...]] = {
    "r9-v5.0.0": (
        (LIPFORENSICS_PROVIDER, LIP_FORENSICS_SIGNAL),
        (EFFORT_PROVIDER, FACE_FORGERY_SIGNAL),
        (NVIDIA_PROVIDER, ACTIVE_SPEAKER_SIGNAL),
        (AASIST_PROVIDER, AUDIO_AUTHENTICITY_SIGNAL),
    ),
}

# The two axes an analysis is read along under R10 (§5). Two independent states rather than one
# scalar status, because a single status cannot express "verdict ready, evidence still
# arriving" without lying about one of the two.
#
# These are read-time projections and no column holds one. The decision axis is projected from
# `analyses.status` and the risk columns; the enrichment axis from the task rows. R10-T3 is
# what renders them.
DECISION_PENDING = "DECISION_PENDING"
DECIDED = "DECIDED"
DECISION_FAILED = "DECISION_FAILED"

ENRICHMENT_NOT_REQUESTED = "ENRICHMENT_NOT_REQUESTED"
ENRICHMENT_PENDING = "ENRICHMENT_PENDING"
ENRICHMENT_PROCESSING = "ENRICHMENT_PROCESSING"
ENRICHMENT_COMPLETE = "ENRICHMENT_COMPLETE"
ENRICHMENT_PARTIAL = "ENRICHMENT_PARTIAL"
ENRICHMENT_FAILED = "ENRICHMENT_FAILED"

# The task statuses that mean a component is finished, and the subset of those that mean it
# finished *successfully*. Two sets rather than one because `abstained` is in both and
# `failed` is in only the first, which is the entire §5.2 distinction expressed as data.
#
# Written as sets the derivation reads rather than as conditions inlined into it, so that a
# sixth status cannot be added to `app.db.models` and silently fall out of `ENRICHMENT_COMPLETE`
# by being absent from a comparison nobody remembered to update.
TERMINAL_TASK_STATUSES = frozenset(
    {
        ENRICHMENT_TASK_STATUS_COMPLETED,
        ENRICHMENT_TASK_STATUS_ABSTAINED,
        ENRICHMENT_TASK_STATUS_FAILED,
    }
)

# "Every Deep Evidence component reached a terminal state and all of them succeeded" is what
# `ENRICHMENT_COMPLETE` asserts, and an abstention satisfies it: the component was asked, it
# answered, and its answer is evidence. An analysis of a clip with no audio is fully enriched,
# not partially enriched — there was nothing more to get.
SUCCEEDED_TASK_STATUSES = frozenset(
    {ENRICHMENT_TASK_STATUS_COMPLETED, ENRICHMENT_TASK_STATUS_ABSTAINED}
)

# Never a live state and never written by anything (§5.2). It is what an analysis with no task
# rows is read as: it ran the full detector chain in one stage, before R10 existed. Deliberately
# not `ENRICHMENT_COMPLETE` — that would assert every Deep Evidence component terminated *and
# succeeded*, which no legacy row can support, since a pre-R10 analysis could always carry a
# failed evidence-only signal and still reach `completed` (§8.1).
LEGACY_SINGLE_STAGE = "LEGACY_SINGLE_STAGE"

# The enrichment axis does not apply to an analysis that has no verdict. §8.1 gives `failed` and
# `queued` rows no enrichment reading at all, and a state that said `ENRICHMENT_FAILED` beside a
# decision that never happened would invite reading the two as one outcome.
ENRICHMENT_NOT_APPLICABLE = "ENRICHMENT_NOT_APPLICABLE"

# Whether a deployment queues enrichment with the decision or waits to be asked (§10, question
# 5, which §4.2 leaves open to this task).
#
# `immediate` is the default, and the default is the whole of R10's compatibility story: under
# it, exactly the same six signals are produced for exactly the same media as before this task,
# and the only thing that changed is that the verdict no longer waits for four of them. A
# default of `deferred` would have shipped R10 as a silent reduction in what every report
# contains, which is a product decision and not a latency one.
#
# Deliberately execution vocabulary and not product vocabulary. §4.2 and the task packet both
# reserve mode naming — Quick Scan, Deep Analysis — for R10-T3, and a knob named after a
# product mode would fix that naming here, in the layer that should outlive it.
ENRICHMENT_POLICY_VARIABLE = "DEEPGUARD_ENRICHMENT_POLICY"
POLICY_IMMEDIATE = "immediate"
POLICY_DEFERRED = "deferred"

# How long a claim on an enrichment task is believed for without being renewed. Longer than the
# decision job's three minutes because the work behind it is the slow half of the pipeline —
# LipForensics alone costs minutes — and because nothing waits on the recovery: a decision job's
# stale lease holds a customer's analysis open, while a stale enrichment task holds only
# evidence that was always going to arrive late.
ENRICHMENT_LEASE_SECONDS = 600

# How often the heartbeat pushes that lease forward. The same relationship to the lease the
# decision job keeps — many renewals inside one lease — so a component that runs long is told
# apart from a worker that died.
ENRICHMENT_HEARTBEAT_SECONDS = 60

# What a recovered task records, as a class name like every other failure in this codebase.
# Nobody caught an exception to produce it — the worker that would have is gone — so it is
# written literally.
STALE_LEASE_ERROR = "StaleEnrichmentLease"

# What a component records when the artifact it was supposed to read is not there. A class name
# for the same reason, and its own name rather than a detector's: nothing was asked about the
# media, so no detector's failure would be the truthful thing to write down.
MISSING_ARTIFACT_ERROR = "EnrichmentArtifactUnavailable"

MAX_ERROR_MESSAGE = 200

TEMP_FILE_PREFIX = "deepguard-enrichment-"


class UnknownRuleset(Exception):
    """An analysis was decided under a ruleset whose component roles are not declared here.

    Raised rather than defaulted, and that is the point of it. A default — an empty membership,
    or the newest version's — would either silently stop enriching every analysis under a new
    ruleset or silently enrich under role assignments that ruleset never made. Both are wrong
    quietly; this is wrong loudly, on the deployment that introduced the ruleset, which is the
    only moment anybody can fix it.
    """


def evidence_only_components(rules_version: str) -> tuple[Component, ...]:
    """Which detectors are Deep Evidence under one ruleset version (§4.1).

    Read from the version the analysis was *decided under*, never from the version this
    deployment happens to be running. An analysis decided under an older ruleset keeps that
    ruleset's role assignment for the life of its enrichment, exactly as it keeps that
    ruleset's verdict — §2 requires the roles be read "from the frozen ruleset version, never
    from the runtime detector registry".
    """
    try:
        return EVIDENCE_ONLY_COMPONENTS[rules_version]
    except KeyError as error:
        raise UnknownRuleset(
            f"No Deep Evidence membership is declared for ruleset {rules_version!r}. "
            "A ruleset version assigns every detector a role; declare its evidence-only set "
            "in `app.enrichment.EVIDENCE_ONLY_COMPONENTS`."
        ) from error


def _component_enabled(component: Component) -> bool:
    """Whether this deployment runs a component at all, as distinct from whether it decides.

    Two different questions, and keeping them apart is the point. `EVIDENCE_ONLY_COMPONENTS`
    answers "what role does the frozen ruleset give this detector" — a forensic fact, fixed
    when the analysis was decided. This answers "is this detector switched on in this
    deployment" — an operational fact about right now. A detector that is off is not
    evidence-only-and-failing; it is a detector nothing asked anything.

    Effort is the only component with such a switch, and `app.effort.is_enabled` is a rollback
    lever rather than a policy: it is on unless a deployment turns it off, and turning it off
    returns this system to the exact SVD + B7 decisional baseline with nothing to migrate. It
    moved here with the detector in R10-T2 — `app.worker.local_readings` used to hold it — and
    it does the same thing in the same direction: a disabled detector gets no task row, so it
    writes no signal and counts in no enrichment state. `app.worker.local_readings` explained
    why that is absence rather than a `FAILED` row, and the reasoning did not change when the
    detector moved paths.

    Imported inside the function so a deployment that never enriches does not load the Effort
    stack to answer a question about it.
    """
    if component != (EFFORT_PROVIDER, FACE_FORGERY_SIGNAL):
        return True

    from app.effort import is_enabled as effort_enabled

    return effort_enabled()


def policy() -> str:
    """Whether this deployment queues enrichment with the decision or waits to be asked.

    Anything that is not exactly `deferred` is `immediate`. The fail-open direction is
    deliberate and is the opposite of how a security switch would be written: the failure mode
    of a misspelled value here is that a report carries all of its evidence, which is what this
    system did before R10 and what it does today.
    """
    configured = os.getenv(ENRICHMENT_POLICY_VARIABLE, "").strip().lower()
    return POLICY_DEFERRED if configured == POLICY_DEFERRED else POLICY_IMMEDIATE


def lease_deadline():
    """How far ahead of *the database's* clock a fresh lease reaches.

    `now()` and the arithmetic in PostgreSQL, exactly as `app.worker.lease_deadline` and
    `app.shadow.lease_deadline` do it, and for the same reason: the comparison that decides
    staleness happens there too, so two workers whose machine clocks disagree must not write
    deadlines on one timeline and have them judged on another.
    """
    return func.now() + timedelta(seconds=ENRICHMENT_LEASE_SECONDS)


def initial_status(mode: str | None) -> str:
    """The status this submission's Deep Evidence rows are written in (§4.2, §5.2).

    Three answers from two sources, and the ordering between them is the point. A named
    product mode is a decision the submission made and it wins; null is a submission that
    made no such decision, and it falls through to the deployment's policy — which is what
    every submission got before modes existed and what every submission through the public
    API still gets. That fall-through is the whole of the backward compatibility story.

    A mode this function does not recognise cannot reach it: the API refuses one with a 422
    and the check constraint on `analysis_jobs.enrichment_mode` refuses one below that. If a
    row somehow held one, it is treated as no mode at all — the deployment's policy — rather
    than as a deferral, because the fail-open direction here is the one that produces a
    report with all of its evidence.
    """
    if mode == PRODUCT_MODE_DEEP_ANALYSIS:
        return ENRICHMENT_TASK_STATUS_QUEUED

    if mode == PRODUCT_MODE_QUICK_SCAN:
        return ENRICHMENT_TASK_STATUS_NOT_REQUESTED

    return (
        ENRICHMENT_TASK_STATUS_QUEUED
        if policy() == POLICY_IMMEDIATE
        else ENRICHMENT_TASK_STATUS_NOT_REQUESTED
    )


def enqueue(
    session: Session,
    analysis_id: uuid.UUID,
    rules_version: str,
    mode: str | None = None,
) -> int:
    """Write this analysis's Deep Evidence tasks. Returns how many were written.

    Called by `app.worker` on the way out of a decision it completed — after the verdict is
    committed and the job closed, never before. The ordering is the contract's: enrichment is
    not a precondition of the decision (§4.2) and an analysis that never gets here is a
    decision that failed, which §8.1 gives no enrichment reading at all.

    Every component gets a row whether or not this deployment wants it run yet. A deferred
    component is `not_requested`, which is a state and not an absence — the module docstring
    and `AnalysisEnrichmentTask` both say why that distinction is what keeps legacy analyses
    readable.

    **Never raises**, deliberately, and it is the load-bearing property of this function rather
    than loose error handling. It runs in its own transaction after the analysis is already
    complete; a failure here rolls back these inserts and nothing else. An analysis that had
    been decided, published and closed must not retroactively fail because its supplementary
    evidence could not be *scheduled* — that would let the enrichment path destroy a fast
    decision through the one door §7.4 did not think to name.

    A duplicate is not a failure either. The unique constraint refuses a second set of tasks
    for the same analysis — a job concluded twice, two workers racing — and that refusal is the
    intended outcome rather than an error to report.

    `mode` is the product mode the submission asked for, or null for one that asked for none
    (R10-T3). It decides the status the rows are written in and nothing else: the same
    components are written for the same analysis either way, so a Quick Scan is a Deep
    Analysis whose evidence has not been asked for yet rather than an analysis with less of
    it. It arrives here, after the verdict is published, and is read nowhere earlier — which
    is what makes "the two modes decide identically" a property of the code's shape.
    """
    try:
        components = evidence_only_components(rules_version)
    except UnknownRuleset:
        logger.warning(
            "No Deep Evidence membership is declared for the ruleset analysis %s was decided "
            "under; no enrichment was queued. The analysis is unaffected.",
            analysis_id,
            exc_info=True,
        )
        return 0

    components = tuple(
        component for component in components if _component_enabled(component)
    )

    if not components:
        return 0

    status = initial_status(mode)

    try:
        session.add_all(
            [
                AnalysisEnrichmentTask(
                    analysis_id=analysis_id,
                    provider=provider,
                    signal_type=signal_type,
                    status=status,
                    rules_version=rules_version,
                )
                for provider, signal_type in components
            ]
        )
        session.commit()
    except Exception:
        session.rollback()
        logger.warning(
            "Queueing Deep Evidence for analysis %s failed; the analysis is unaffected.",
            analysis_id,
            exc_info=True,
        )
        return 0

    logger.info(
        "Queued %s Deep Evidence component(s) for analysis %s as %s.",
        len(components),
        analysis_id,
        status,
    )

    return len(components)


def request(session: Session, analysis_id: uuid.UUID) -> int:
    """Ask for enrichment that was deferred. Returns how many components it queued.

    The internal trigger, and the whole of it: `not_requested` becomes `queued` and the worker
    picks it up on its next idle poll. Deliberately the only transition this function performs
    — it does not re-run a component that already reached a terminal state, and it does not
    reach a component another worker is holding, because neither is what "please start the
    enrichment that was deferred" means.

    Internal on purpose. R10-T3 is what exposes a way to call it and what decides the product
    vocabulary around it; this task owes that task a mechanism and deliberately not a name.
    Asking for enrichment on an analysis that has none deferred queues nothing and says so by
    returning zero, which is the honest answer for a legacy analysis and for one already
    enriched alike.
    """
    queued = session.execute(
        AnalysisEnrichmentTask.__table__.update()
        .where(
            AnalysisEnrichmentTask.analysis_id == analysis_id,
            AnalysisEnrichmentTask.status == ENRICHMENT_TASK_STATUS_NOT_REQUESTED,
        )
        .values(status=ENRICHMENT_TASK_STATUS_QUEUED)
        .returning(AnalysisEnrichmentTask.id)
    ).scalars().all()

    session.commit()

    return len(queued)


def recover_stale_tasks(session: Session) -> int:
    """Fail every `processing` task whose worker stopped saying it was alive.

    The test `app.worker.recover_stale_jobs` applies, against this table: a lease that has run
    out is a worker that is gone, and age is not the test — a component in the middle of
    LipForensics writes nothing for minutes.

    Terminal, like every other recovery here, and for the same reason: a component that
    reliably kills the process would otherwise be handed to worker after worker.

    **No analysis and no job is touched.** Not their status, not their risk columns, not their
    signals. This is the recovery path §7.4 has in mind when it says "a crashed or leaked
    enrichment worker must never leave an analysis looking undecided" — the decision axis is
    not reachable from this statement, and the guarded session would refuse it even if this
    statement tried.
    """
    recovered = session.execute(
        AnalysisEnrichmentTask.__table__.update()
        .where(
            AnalysisEnrichmentTask.status == ENRICHMENT_TASK_STATUS_PROCESSING,
            # Never null on a claimed task, but stated rather than assumed: `NULL < now()` is
            # null, so a row that somehow had no lease would never be reached.
            AnalysisEnrichmentTask.lease_expires_at.is_not(None),
            AnalysisEnrichmentTask.lease_expires_at < func.now(),
        )
        .values(
            status=ENRICHMENT_TASK_STATUS_FAILED,
            error_message=STALE_LEASE_ERROR,
            lease_expires_at=None,
        )
        .returning(AnalysisEnrichmentTask.id)
    ).scalars().all()

    if not recovered:
        session.rollback()
        return 0

    session.commit()

    logger.warning(
        "Recovered %s stale Deep Evidence task(s): %s.",
        len(recovered),
        ", ".join(str(task_id) for task_id in recovered),
    )

    return len(recovered)


@dataclass(frozen=True)
class ClaimedComponent:
    """One component this worker owns, as plain values rather than a row."""

    task_id: uuid.UUID
    provider: str
    signal_type: str

    @property
    def component(self) -> Component:
        return (self.provider, self.signal_type)


@dataclass(frozen=True)
class ClaimedEnrichment:
    """Every queued component of one analysis, claimed together, with what running them needs.

    Everything is read inside the claim transaction and carried out of it by value, exactly as
    `app.worker.ClaimedJob` is and for the same reason: after the commit there is no session
    and no lock, and an ORM row here would be an invitation to hold a connection open across
    minutes of inference.
    """

    analysis_id: uuid.UUID
    rules_version: str
    components: tuple[ClaimedComponent, ...]
    # The object these components read. The derivative the fast decision path produced and
    # uploaded, when there is one, and the original when the media needed no derivative — in
    # which case the original *is* the artifact every detector saw. Null when normalization was
    # required and no derivative was ever recorded, which is a decision that completed without
    # a readable artifact; the components record that rather than transcoding a second time.
    storage_key: str | None
    # The rate NVIDIA's frame indices are read against, as `app.worker` reads it: the rate the
    # derivative was transcoded to hold constant, which for un-normalized media is the
    # original's own probed rate.
    frame_rate: float

    @property
    def task_ids(self) -> tuple[uuid.UUID, ...]:
        return tuple(component.task_id for component in self.components)


def claim(session: Session) -> ClaimedEnrichment | None:
    """Take ownership of every queued component of one analysis, or return nothing.

    Two statements in one transaction. The first finds the oldest queued task and locks it with
    `FOR UPDATE ... SKIP LOCKED`, which is what makes several workers safe here for the reason
    it makes `app.worker.claim_job` safe: two of them racing for the same row do not queue
    behind each other and do not both get it. The second takes that task's *analysis* and locks
    whatever else of it is still queued, so one download and one WAV extraction serve every
    component that is owed.

    A component another worker is already holding is skipped rather than waited for, so two
    workers may legitimately split one analysis's components between them. That is safe because
    the unit of state is the component: each row is claimed, leased, run and concluded on its
    own, and neither worker can see or overwrite the other's.

    Oldest first, so a backlog stays a queue rather than a stack.

    The claim starts the lease in this same transaction. A row that was `processing` for even
    one commit without a deadline would be a task no recovery could ever reach.
    """
    first = session.execute(
        select(AnalysisEnrichmentTask)
        .where(AnalysisEnrichmentTask.status == ENRICHMENT_TASK_STATUS_QUEUED)
        .order_by(AnalysisEnrichmentTask.created_at)
        .limit(1)
        .with_for_update(skip_locked=True)
    ).scalar_one_or_none()

    if first is None:
        session.rollback()
        return None

    siblings = session.execute(
        select(AnalysisEnrichmentTask)
        .where(
            AnalysisEnrichmentTask.analysis_id == first.analysis_id,
            AnalysisEnrichmentTask.status == ENRICHMENT_TASK_STATUS_QUEUED,
            AnalysisEnrichmentTask.id != first.id,
        )
        .order_by(AnalysisEnrichmentTask.created_at)
        .with_for_update(skip_locked=True)
    ).scalars().all()

    media = session.execute(
        select(
            MediaFile.original_storage_key,
            MediaFile.derivative_storage_key,
            MediaFile.was_normalized,
            MediaFile.frame_rate,
        ).where(MediaFile.analysis_id == first.analysis_id)
    ).first()

    claimed_rows = [first, *siblings]
    for task in claimed_rows:
        task.status = ENRICHMENT_TASK_STATUS_PROCESSING
        task.lease_expires_at = lease_deadline()

    claimed = ClaimedEnrichment(
        analysis_id=first.analysis_id,
        rules_version=first.rules_version,
        components=tuple(
            ClaimedComponent(
                task_id=task.id, provider=task.provider, signal_type=task.signal_type
            )
            for task in claimed_rows
        ),
        storage_key=_artifact_key(media),
        # Every analysis has media — the upload commits the two together — so the absence of a
        # row here is a broken database rather than a case to carry a default through. A rate
        # of zero would reach the active-speaker chain as a real figure.
        frame_rate=media.frame_rate if media is not None else 0.0,
    )
    session.commit()

    return claimed


def _artifact_key(media) -> str | None:
    """Which stored object this analysis's evidence-only detectors should read.

    The derivative when one was made, because that is the artifact the deciding detectors
    actually saw and the one every contract in `app.detection` is written against. The original
    when no derivative was needed, because then the original is that artifact and re-encoding it
    would hand these detectors a file the decision never read.

    Nothing when normalization was required and no derivative was recorded. That is a decision
    that completed without a readable artifact — a transcode that failed, whose signals are
    already `FAILED` rows saying so — and the honest answer is that there is nothing for these
    components to read. Transcoding here instead would run the pipeline's most expensive step a
    second time, for a file the verdict was not taken from.
    """
    if media is None:
        return None

    if media.was_normalized:
        return media.derivative_storage_key

    return media.original_storage_key


def renew_lease(session: Session, task_ids: Sequence[uuid.UUID]) -> bool:
    """Push this claim's deadline forward. Returns whether any of it is still ours.

    Conditional on the rows still being `processing`, which is what stops a worker that was
    already recovered from quietly taking its work back. One statement for every component of
    the claim, because they were claimed together and are being kept alive by one heartbeat.
    """
    renewed = session.execute(
        AnalysisEnrichmentTask.__table__.update()
        .where(
            AnalysisEnrichmentTask.id.in_(task_ids),
            AnalysisEnrichmentTask.status == ENRICHMENT_TASK_STATUS_PROCESSING,
        )
        .values(lease_expires_at=lease_deadline())
    )
    session.commit()

    return renewed.rowcount > 0


def _renew_until_stopped(
    task_ids: Sequence[uuid.UUID], stopped: threading.Event
) -> None:
    """Renew this claim's lease on its own connection until told to stop.

    Its own session, not the one running the components: a `Session` is not safe to use from
    two threads, and the thread doing the work spends the whole of it with no session open.
    `app.worker._renew_until_stopped` is the same mechanism against the other table, and its
    docstring says why a failed renewal is retried rather than escalated.
    """
    while not stopped.wait(ENRICHMENT_HEARTBEAT_SECONDS):
        try:
            with SessionLocal() as session:
                if not renew_lease(session, task_ids):
                    logger.warning(
                        "Deep Evidence tasks %s are no longer this worker's to renew; they "
                        "were recovered as stale. Their results will be discarded.",
                        ", ".join(str(task_id) for task_id in task_ids),
                    )
                    return
        except Exception:
            logger.exception("Renewing a Deep Evidence lease failed; will retry.")


@contextmanager
def leased(task_ids: Sequence[uuid.UUID]) -> Iterator[None]:
    """Keep a claim's lease alive for the length of the block.

    A daemon thread, for the reason `app.worker.leased` uses one: the whole mechanism exists
    for the case where this process dies without unwinding, and a non-daemon thread would make
    that shutdown hang.
    """
    stopped = threading.Event()
    heartbeat = threading.Thread(
        target=_renew_until_stopped,
        args=(tuple(task_ids), stopped),
        name=f"enrichment-lease-{task_ids[0] if task_ids else 'none'}",
        daemon=True,
    )
    heartbeat.start()

    try:
        yield
    finally:
        stopped.set()
        heartbeat.join(timeout=ENRICHMENT_HEARTBEAT_SECONDS)


@contextmanager
def fetched_artifact(storage_key: str) -> Iterator[Path]:
    """Download one stored object to a temp file for the block, and remove it after.

    Deliberately a second small copy of `app.worker.fetched_artifact` rather than a shared
    helper imported from it. `app.worker` imports this module — it is what queues enrichment
    and what routes a poll to it — so importing back would close a cycle, and the alternative
    is a third module holding eleven lines. `app.shadow` already owns its own lease machinery
    beside the worker's for the same reason, and AGENTS.md prefers a small duplication to a
    premature abstraction.
    """
    handle = tempfile.NamedTemporaryFile(prefix=TEMP_FILE_PREFIX, delete=False)
    handle.close()
    path = Path(handle.name)

    try:
        fetch_object(storage_key, path)
        yield path
    finally:
        path.unlink(missing_ok=True)


@dataclass(frozen=True)
class ComponentEvidence:
    """What running one component produced, before any of it is persisted.

    The signal and its segments travel together because they must not drift apart: segments
    hang off their signal's id, and a list that lost track of which signal produced it would be
    attached to the wrong evidence. `app.worker.SignalEvidence` is the same pair on the other
    path; this one is not imported from there because importing `app.worker` would close a
    cycle, and the type is three lines.
    """

    signal: AnalysisSignal
    segments: list


def run_components(
    claimed: ClaimedEnrichment, path: Path
) -> tuple[dict[Component, ComponentEvidence], dict[Component, str]]:
    """Ask each claimed component about the artifact, one at a time.

    Returns what was produced and, separately, why anything missing is missing.

    Written out rather than dispatched through a registry, exactly as
    `app.worker.local_readings` is and for the reason stated there: these detectors are not
    interchangeable behind an abstraction — one judges how a mouth moves, one judges aligned
    face crops through a CLIP subspace decomposition, one finds who is speaking and one judges
    whether audio was synthesised — and AGENTS.md forbids a provider registry built for the
    detector that might exist later.

    Sequential, for the reason the fast path's local readings are: all of this is blocking CPU
    inference in a container with a CPU quota and a 6 GiB cap, so running them at once would
    not finish sooner — it would contend for the same cores and hold every model resident at
    the same peak.

    **The audio pair shares one call.** `analyse_audio` extracts one WAV and answers both
    questions off it, and splitting them into two claims would have extracted it twice. They
    remain two components with two rows and two terminal states; what they share is an
    extraction, not an answer, and neither informs the other. They therefore also fail
    together, and only with each other: the extraction is what they share, so an extraction
    that broke is a fact about both of them and about nothing else.

    **One component's collapse costs only that component.** Each is invoked inside its own
    `try`, so a detector that raises outright — a timeout, a torch that broke, a checkpoint
    that is not there — costs its own row and leaves the readings taken before it standing.
    That is rule 11 and §7.4 together, and it is the same judgement `app.worker.AnalysisTimedOut`
    makes on the other path: a complete, independent reading is not made untrue by something
    unrelated failing afterwards, and discarding real forensic findings to tidy up after a
    failure would destroy evidence over a scheduling event.

    Most failures never reach these handlers, because `app.detection` records them as `FAILED`
    signals instead — a missing checkpoint, an unreadable clip, an abstention. What lands here
    is what that layer does not catch, and the point of catching it is containment rather than
    interpretation: the component records the exception's class name and nothing about the
    media, because a limit this worker hit is a statement about this worker.

    Imported inside the function rather than at module scope. `app.detection` pulls in the
    detector stack, and this module is imported by `app.worker` at startup; keeping the import
    here means a deployment whose enrichment is deferred does not pay for loading it until
    something is actually claimed.
    """
    from app.detection import analyse_audio, detect_effort, detect_lip_forensics
    import asyncio

    claimed_components = {component.component for component in claimed.components}
    produced: dict[Component, ComponentEvidence] = {}
    failed: dict[Component, str] = {}

    lip = (LIPFORENSICS_PROVIDER, LIP_FORENSICS_SIGNAL)
    if lip in claimed_components:
        try:
            produced[lip] = ComponentEvidence(
                signal=detect_lip_forensics(path), segments=[]
            )
        except Exception as error:
            logger.exception("Deep Evidence component %s/%s failed outright.", *lip)
            failed[lip] = type(error).__name__[:MAX_ERROR_MESSAGE]

    effort = (EFFORT_PROVIDER, FACE_FORGERY_SIGNAL)
    if effort in claimed_components:
        try:
            produced[effort] = ComponentEvidence(
                signal=detect_effort(path), segments=[]
            )
        except Exception as error:
            logger.exception("Deep Evidence component %s/%s failed outright.", *effort)
            failed[effort] = type(error).__name__[:MAX_ERROR_MESSAGE]

    speaker = (NVIDIA_PROVIDER, ACTIVE_SPEAKER_SIGNAL)
    audio = (AASIST_PROVIDER, AUDIO_AUTHENTICITY_SIGNAL)
    wanted_audio = tuple(
        component
        for component in (speaker, audio)
        if component in claimed_components
    )

    if wanted_audio:
        try:
            # One event loop for the one call, as the fast path used to run it: the whole
            # chain — extraction, diarization, NVIDIA, the local checkpoint — happens inside
            # it, so the temporary WAV never outlives it.
            (speaker_signal, speaker_segments), (audio_signal, audio_segments) = (
                asyncio.run(analyse_audio(path, claimed.frame_rate))
            )
        except Exception as error:
            logger.exception("The Deep Evidence audio chain failed outright.")
            reason = type(error).__name__[:MAX_ERROR_MESSAGE]
            for component in wanted_audio:
                failed[component] = reason
        else:
            if speaker in claimed_components:
                produced[speaker] = ComponentEvidence(
                    signal=speaker_signal, segments=speaker_segments
                )
            if audio in claimed_components:
                produced[audio] = ComponentEvidence(
                    signal=audio_signal, segments=audio_segments
                )

    return produced, failed


def persist(
    claimed: ClaimedEnrichment,
    produced: dict[Component, ComponentEvidence],
    failed: dict[Component, str] | None = None,
) -> None:
    """Commit each component's evidence and conclude its task, one component per transaction.

    **One transaction per component, not one for the claim.** Each is an independent forensic
    reading (rule 11) and §7.5 requires that a component's signal and its segments become
    visible *together*, so that no reader ever observes a half-written detector result. It does
    not require — and must not have — that four components become visible together: a report
    read between two of them is supposed to show the one that has landed, which is the whole of
    "readers must tolerate evidence appearing between two successive reads".

    **Replace, never accumulate.** A re-run component's previous signal row is deleted and
    rewritten, which takes its segments with it through the `ON DELETE CASCADE` on
    `analysis_segments.signal_id`. §7.3 requires exactly this: evidence from two different
    executions of the same detector never coexists, and the `(analysis_id, provider,
    signal_type)` uniqueness that makes enrichment retries idempotent is preserved rather than
    worked around.

    Everything here goes through `app.enrichment_guard.enrichment_session`, which refuses any
    write outside §6.2's allowance. The guard is given this analysis's ruleset membership, so
    what counts as a writable signal row is decided by the ruleset the analysis was decided
    under rather than by this function.

    The task's terminal status follows whether the detector produced a usable reading, which is
    what §7.4 needs in order for `ENRICHMENT_PARTIAL` to mean anything: a claim in which
    LipForensics failed and AASIST succeeded has to be readable as partial, and it can only be
    if the two rows differ. The evidence is committed either way — a `FAILED` signal is a
    finding ("this source was asked and had no answer") and is kept for the same reason
    `app.worker.abandon_job` keeps what a timed-out job proved.
    """
    allowed = frozenset(evidence_only_components(claimed.rules_version))
    failed = failed or {}

    for component in claimed.components:
        evidence = produced.get(component.component)

        if evidence is None:
            # Claimed but produced nothing: the detector raised outright, or was never
            # reached. No signal row is written — a `FAILED` row is a finding ("this source
            # was asked and had no answer"), and one written for a detector that collapsed
            # before answering would be recording a finding nobody made. That is the same
            # distinction `app.worker.analyse` draws for the readings a transcode failure
            # never reaches.
            #
            # The reason is the one `run_components` caught, so an operator reads what
            # actually happened rather than a generic fallback. `MISSING_ARTIFACT_ERROR`
            # stands only where nothing recorded a reason at all.
            _conclude(
                component,
                ENRICHMENT_TASK_STATUS_FAILED,
                failed.get(component.component, MISSING_ARTIFACT_ERROR),
                allowed,
            )
            continue

        with enrichment_session(allowed) as session:
            # Fetched and deleted through the ORM rather than with a Core `DELETE`, because the
            # guard refuses Core DML against `analysis_signals` — it cannot read a provider out
            # of a WHERE clause, and an allowlist that waved through what it could not inspect
            # would not be one. Fetching first also means the delete is checked against the
            # provider the row actually has.
            superseded = session.execute(
                select(AnalysisSignal).where(
                    AnalysisSignal.analysis_id == claimed.analysis_id,
                    AnalysisSignal.provider == component.provider,
                    AnalysisSignal.signal_type == component.signal_type,
                )
            ).scalars().all()

            for row in superseded:
                session.delete(row)

            if superseded:
                # Before the insert, so the unique constraint sees the old row gone. One
                # transaction still, so a reader never observes the component missing.
                session.flush()

            evidence.signal.analysis_id = claimed.analysis_id
            session.add(evidence.signal)

            if evidence.segments:
                # Segments hang off their own signal, so its id has to exist before any of
                # them can name one — a flush inside this same transaction, not a second one.
                session.flush()
                for segment in evidence.segments:
                    segment.signal_id = evidence.signal.id
                session.add_all(evidence.segments)

            status, error_message = _component_outcome(evidence.signal)

            session.execute(
                AnalysisEnrichmentTask.__table__.update()
                .where(
                    AnalysisEnrichmentTask.id == component.task_id,
                    # Conditional on the task still being this worker's, exactly as
                    # `app.worker._set_status` is conditional on its job still being
                    # `processing`. A component recovered as stale while it ran is no longer
                    # ours to conclude, and a worker back from the dead must not overwrite the
                    # recovery that replaced it.
                    AnalysisEnrichmentTask.status == ENRICHMENT_TASK_STATUS_PROCESSING,
                )
                .values(
                    status=status, error_message=error_message, lease_expires_at=None
                )
            )

            session.commit()


def _component_outcome(signal: AnalysisSignal) -> tuple[str, str | None]:
    """How one component's task ends, read from the signal its detector wrote.

    Three outcomes, not two, and the middle one is the whole of the R10-T1 §5.2 distinction:

    - the detector produced a reading — `completed`, with nothing to explain;
    - the detector was asked and answered that there was nothing here to score — `abstained`,
      a **success** on the enrichment axis, carrying the documented reason;
    - anything else — `failed`, carrying what went wrong.

    **The signal row is not consulted for its verdict value and is not modified.** What is read
    is the identity of the failure `app.detection` already recorded in the row's metadata, which
    is durable, already persisted, and already distinguishes these cases — see
    `app.detection.ABSTENTION_ERRORS`, which binds to the exception classes themselves rather
    than to their spellings. No forensic evidence changes shape to make this distinction; it was
    already there to be read.

    `TIMEOUT` is a failure and can never be an abstention, stated rather than left to follow
    from the metadata. A timeout is a statement about this worker — a machine under load, a
    limit set too tight — and an abstention is a statement about the media. Letting a timeout
    reach the abstention branch on the strength of a metadata key would let this worker's own
    condition be reported to an analyst as a fact about someone's video.

    The reverse guard matters as much: only a `FAILED` signal can abstain. A `SUCCESS` row
    carrying an abstention-shaped metadata key is a defect in a detector, and it is read as the
    reading it claims to be rather than being quietly downgraded here.
    """
    if signal.status == SIGNAL_STATUS_SUCCESS:
        return ENRICHMENT_TASK_STATUS_COMPLETED, None

    reason = _signal_error(signal)

    if signal.status == SIGNAL_STATUS_FAILED and reason in ABSTENTION_ERRORS:
        # The reason travels with it. An abstention is not a failure, but it is still something
        # an operator asks "why is there no lip evidence on this report" about, and the answer
        # — no trackable face — is the one this row can give.
        return ENRICHMENT_TASK_STATUS_ABSTAINED, reason

    return ENRICHMENT_TASK_STATUS_FAILED, reason


def _signal_error(signal: AnalysisSignal) -> str:
    """What a component records as the reason its detector produced no reading.

    The error class name `app.detection` already put in the signal's metadata, so the task row
    and the evidence row say the same thing about the same outcome. The signal's status when
    there is none — a detector that failed without naming a reason still failed, and an empty
    string would read as a failure with nothing to say.
    """
    metadata = signal.signal_metadata or {}
    reason = metadata.get("error") or signal.status

    return str(reason)[:MAX_ERROR_MESSAGE]


def _conclude(
    component: ClaimedComponent,
    status: str,
    error_message: str | None,
    allowed: frozenset[Component],
) -> None:
    """Close one component out without writing evidence for it.

    The path for a component that was claimed and never asked anything — the artifact was not
    there, or the claim failed before it was reached. Its task reaches a terminal state, which
    is what the enrichment axis needs in order to settle, and no signal row is invented to
    explain it.
    """
    with enrichment_session(allowed) as session:
        session.execute(
            AnalysisEnrichmentTask.__table__.update()
            .where(
                AnalysisEnrichmentTask.id == component.task_id,
                AnalysisEnrichmentTask.status == ENRICHMENT_TASK_STATUS_PROCESSING,
            )
            .values(status=status, error_message=error_message, lease_expires_at=None)
        )
        session.commit()


def fail_claim(claimed: ClaimedEnrichment, error: BaseException) -> None:
    """Record that this whole claim could not be run, and why.

    Reached for failures that are ours rather than a detector's: the object store unreachable,
    the fetched artifact unreadable, a bug here. Every claimed component is failed, because
    none of them was asked anything.

    **Nothing about the analysis moves.** Its status, its verdict and its evidence are exactly
    as the fast decision left them, and the guarded session would refuse this function the
    columns even if it asked. That is invariant I11, and it is the difference between an
    enrichment that failed and an analysis that failed.
    """
    allowed = frozenset(evidence_only_components(claimed.rules_version))

    for component in claimed.components:
        _conclude(
            component,
            ENRICHMENT_TASK_STATUS_FAILED,
            type(error).__name__[:MAX_ERROR_MESSAGE],
            allowed,
        )


def process_one(session: Session) -> bool:
    """Claim and run one analysis's Deep Evidence. Returns whether there was any to run.

    Called by `app.worker.run` only on a poll that found no decision job, which is where the
    Fast Decision Path's strict priority actually lives (§4.2): enrichment is never claimed
    while a queued job exists, so it cannot delay a verdict by one poll, let alone by the
    minutes a component takes.

    Recovery runs first, on the same poll, for the reason `app.worker.process_one` runs its own
    first: it is one statement that normally matches nothing, on a loop that was going to query
    this table anyway.

    Swallows its own failures, exactly as `app.shadow.process_one` does. A broken enrichment
    must not reach the worker loop's backoff, must not stop the next decision job being
    claimed, and must not turn an idle poll into an error — the fast path's independence from
    this one is the property R10 exists to create, and it would be a strange way to lose it.
    """
    try:
        recover_stale_tasks(session)

        claimed = claim(session)
        if claimed is None:
            return False
    except Exception:
        logger.exception("Claiming Deep Evidence work failed; the decision path is unaffected.")
        return False

    logger.info(
        "Claimed %s Deep Evidence component(s) for analysis %s.",
        len(claimed.components),
        claimed.analysis_id,
    )

    try:
        with leased(claimed.task_ids):
            if claimed.storage_key is None:
                # A decision that completed without a readable artifact. Every component is
                # closed out as failed with that reason, and no detector is asked about a file
                # that is not there.
                logger.warning(
                    "Analysis %s has no artifact for Deep Evidence to read; its components "
                    "are recorded as failed.",
                    claimed.analysis_id,
                )
                for component in claimed.components:
                    _conclude(
                        component,
                        ENRICHMENT_TASK_STATUS_FAILED,
                        MISSING_ARTIFACT_ERROR,
                        frozenset(evidence_only_components(claimed.rules_version)),
                    )
                return True

            with fetched_artifact(claimed.storage_key) as path:
                produced, failed = run_components(claimed, path)

            persist(claimed, produced, failed)
    except Exception as error:
        logger.exception(
            "Deep Evidence for analysis %s failed; its verdict is untouched.",
            claimed.analysis_id,
        )
        try:
            fail_claim(claimed, error)
        except Exception:
            logger.exception(
                "Recording the Deep Evidence failure for analysis %s failed; its tasks are "
                "left to their lease.",
                claimed.analysis_id,
            )
        return True

    logger.info(
        "Completed Deep Evidence for analysis %s: %s.",
        claimed.analysis_id,
        ", ".join(
            f"{component.provider}/{component.signal_type}"
            for component in claimed.components
        ),
    )

    return True


def decision_state(analysis_status: str, risk_level: str | None) -> str:
    """Where an analysis is on the decision axis (§5.1).

    A projection over columns that already exist, never a column of its own. `analyses.status`
    keeps meaning decision status exactly as it did before R10 — §5.4 requires that every
    existing reader keep its current meaning for free — and this function is what names the
    meaning it already had.

    `INCONCLUSIVE` is a verdict and not a decision failure, so it lands on `DECIDED` like any
    other verdict. Conflating the two would let a real classification be presented as a system
    fault (§5.1).
    """
    if analysis_status == ANALYSIS_STATUS_COMPLETED:
        return DECIDED

    if analysis_status == ANALYSIS_STATUS_FAILED:
        return DECISION_FAILED

    if analysis_status == ANALYSIS_STATUS_QUEUED:
        return DECISION_PENDING

    # An analysis in a status this projection does not know is pending rather than decided. The
    # conservative direction: a verdict claimed for a row whose status nobody recognises would
    # be a verdict invented by a reader, and `risk_level` is checked rather than assumed for
    # exactly that reason.
    return DECIDED if risk_level is not None else DECISION_PENDING


def enrichment_state(analysis_decision_state: str, task_statuses: Sequence[str]) -> str:
    """Where an analysis is on the enrichment axis (§5.2), derived from its component rows.

    Derived rather than stored, which is §5.4's preference and the only representation under
    which `ENRICHMENT_PARTIAL` can name which components failed: the summary is recomputed from
    the rows an operator can read, so it can never assert something those rows do not say.

    **No task rows means legacy, not complete.** An analysis with none of these rows ran the
    full detector chain in one stage, before R10 existed (§8.1). It is emphatically not
    `ENRICHMENT_COMPLETE`, which asserts every Deep Evidence component terminated *and*
    succeeded — a claim no legacy row can support, since a pre-R10 analysis could carry a
    failed evidence-only signal and still reach `completed`. What its components actually did
    is read from the signal rows, as it always was.

    The decision state is taken as an argument because §8.1 gives an undecided analysis no
    enrichment reading at all. An analysis that never got a verdict has nothing for
    supplementary evidence to supplement, and a state that said `ENRICHMENT_FAILED` beside a
    decision that never happened would invite reading the two as one outcome.
    """
    if analysis_decision_state != DECIDED:
        return ENRICHMENT_NOT_APPLICABLE

    statuses = list(task_statuses)

    if not statuses:
        return LEGACY_SINGLE_STAGE

    if all(status == ENRICHMENT_TASK_STATUS_NOT_REQUESTED for status in statuses):
        return ENRICHMENT_NOT_REQUESTED

    if any(status == ENRICHMENT_TASK_STATUS_PROCESSING for status in statuses):
        return ENRICHMENT_PROCESSING

    if not all(status in TERMINAL_TASK_STATUSES for status in statuses):
        # Something is still owed: a queued component, or a component left `not_requested`
        # beside others that were asked for. Pending rather than processing — nothing is
        # running yet — and pending rather than partial, because §5.2 reserves the terminal
        # states for the case where *every* component has terminated.
        return ENRICHMENT_PENDING

    succeeded = sum(1 for status in statuses if status in SUCCEEDED_TASK_STATUSES)

    if succeeded == len(statuses):
        return ENRICHMENT_COMPLETE

    if succeeded == 0:
        return ENRICHMENT_FAILED

    return ENRICHMENT_PARTIAL


def analysis_states(session: Session, analysis_id: uuid.UUID) -> tuple[str, str]:
    """Both axes for one analysis, as `(decision_state, enrichment_state)` (§5.3).

    One reader for the two, because they are read together and independently: every combination
    is legal, and the one R10 exists to create — `DECIDED` beside `ENRICHMENT_PROCESSING` — is
    only expressible if neither is derived from the other.

    Returns `DECISION_PENDING` and `ENRICHMENT_NOT_APPLICABLE` for an analysis that does not
    exist rather than raising. A caller asking about a row it cannot see is a reader's problem
    to report, not this projection's to guess at.
    """
    analysis = session.execute(
        select(Analysis.status, Analysis.risk_level).where(Analysis.id == analysis_id)
    ).first()

    if analysis is None:
        return DECISION_PENDING, ENRICHMENT_NOT_APPLICABLE

    decision = decision_state(analysis.status, analysis.risk_level)

    statuses = session.execute(
        select(AnalysisEnrichmentTask.status).where(
            AnalysisEnrichmentTask.analysis_id == analysis_id
        )
    ).scalars().all()

    return decision, enrichment_state(decision, statuses)


@dataclass(frozen=True)
class ComponentState:
    """One Deep Evidence component's own state, as the API projects it (R10-T3 §3.A).

    The component is named by the `(provider, signal_type)` pair everything else names it by
    — the task row, the guard's allowlist, the signal it writes and §4.1's membership table —
    so a reader joining this to the evidence below it has one spelling to match on.

    `state` is the task's status verbatim: `not_requested`, `queued`, `processing`,
    `completed`, `abstained` or `failed`. Not translated, not grouped and in particular not
    collapsed — `abstained` beside `failed` is the distinction §5.2 exists to keep, and a
    projection that folded the two would have thrown it away before any renderer could get
    it wrong.
    """

    provider: str
    signal_type: str
    state: str


def component_states(
    session: Session, analysis_ids: Sequence[uuid.UUID]
) -> dict[uuid.UUID, tuple[ComponentState, ...]]:
    """Every analysis's per-component states, in one statement (§5.2).

    Batched because its caller is a listing as well as a report: a statement per analysis
    would put a query per row on the dashboard's read path, which is the shape
    `app.api.analyses` already refuses for segments, timelines and audio windows.

    An analysis with no task rows is absent from the result rather than present with an empty
    tuple, and the difference matters to nobody downstream: `enrichment_state` reads an empty
    sequence as `LEGACY_SINGLE_STAGE`, which is what a missing key resolves to through
    `dict.get(..., ())`.

    Ordered by component so two reads of the same analysis list its components in the same
    order. Nothing depends on which order it is; something does depend on it not changing
    between two renders of the same report.
    """
    if not analysis_ids:
        return {}

    rows = session.execute(
        select(
            AnalysisEnrichmentTask.analysis_id,
            AnalysisEnrichmentTask.provider,
            AnalysisEnrichmentTask.signal_type,
            AnalysisEnrichmentTask.status,
        )
        .where(AnalysisEnrichmentTask.analysis_id.in_(analysis_ids))
        .order_by(
            AnalysisEnrichmentTask.provider,
            AnalysisEnrichmentTask.signal_type,
        )
    ).all()

    states: dict[uuid.UUID, list[ComponentState]] = {}
    for row in rows:
        states.setdefault(row.analysis_id, []).append(
            ComponentState(
                provider=row.provider,
                signal_type=row.signal_type,
                state=row.status,
            )
        )

    return {
        analysis_id: tuple(components) for analysis_id, components in states.items()
    }
