"""Running an uncalibrated workload on real traffic without letting it reach anyone (R6-T1).

Phase 5 left the risk engine deterministic and calibrated: three detectors, three measured
operating points, one rule fires. Anything new has to earn a place in that before it can move
a band, and earning it means being run over real media and measured offline. Shadow mode is
where that running happens.

The whole design is about what a shadow observation is *not* allowed to touch, and every one
of those guarantees is structural rather than a rule somebody has to remember:

- **it is not evidence.** Shadow observations are written to `shadow_runs` and to nothing
  else. `analysis_signals` — the forensic record the report renders and the risk engine reads
  — gains no row, no provider and no column from anything here;
- **no reader can see it.** Neither API module, and nothing in `app.risk_engine`, names
  `ShadowRun`. There is no join to traverse and no filter to forget, which is a stronger
  statement than "the query excludes it";
- **the risk engine cannot consume it.** `app.risk_engine.evaluate` accepts three specific
  evidence types and rejects everything else outright, so even a future caller that tried to
  hand it a shadow observation would raise rather than classify;
- **production never waits for it.** A shadow run is enqueued *after* the production job has
  committed its decision and closed itself out, in a separate transaction whose failure is
  logged and swallowed. It is then executed on a later poll, and only on a poll where there
  was no production work to do. An analysis is complete, decided and readable whether the
  shadow run succeeds, fails, times out or never happens at all.

The execution model is the one already in the repository. `shadow_runs` is claimed with the
same `FOR UPDATE ... SKIP LOCKED` statement `app.worker.claim_job` uses, holds the same kind
of lease, and is polled by the same process on the same loop. No queue broker, no scheduler
and no second service: what this needed from an execution framework, PostgreSQL and the
existing loop already provide.

What runs is a stub. R6-T1 is the infrastructure and nothing else — `run_stub_workload` does
no inference, loads no model and reads no media; it records that the machinery worked. The
task that integrates a real experimental detector replaces that one function and the constant
naming it, and finds every boundary above already in place and already tested.

R6-T2 adds a second place for that stub to run: Modal's GPU, reached through
`app.modal_client`. It is still a stub — no detector, no weights, no media — because the
question that task asks is whether a *remote* execution backend can be plugged in underneath
this module without any of the guarantees above weakening, and the answer has to be readable
without a model in the way. Three things change here and nothing else does:

- **which workload is queued is a configuration question.** `workload()` names `modal-stub`
  when Modal is configured and `stub` when it is not, the name is written on the row at
  enqueue time, and execution dispatches on the name the row carries. A run therefore
  executes as what it was queued as, and the two kinds of observation stay distinguishable in
  a table that is going to be read offline months later;

- **a workload is started on one poll and collected on another.** This is the change that
  matters most and it was not in the first draft of R6-T2. A remote GPU answers in tens of
  seconds, and the worker runs one loop; a `process_one` that waited for Modal would hold that
  loop, and a production job submitted meanwhile would sit queued behind an experiment. That
  was measured before the split: a job was claimed 8 ms after an in-flight Modal call
  returned, having waited 4.1 s for it. So `start_workload` spawns and returns, and
  `_collect_pending` asks — with no timeout, one round trip — on later idle polls. R6-T1's
  guarantee was that an experiment is never *claimed* ahead of a queued job; this is the
  guarantee that one already running cannot get in front of one either;

- **the lease is renewed rather than only set.** A remote call can outlive
  `SHADOW_LEASE_SECONDS`, and `renew_lease` pushes the deadline out on every visit to a
  pending run. The renewal is also the ownership check: a refusal means the row was recovered
  or taken while the GPU was busy, so the remote call is cancelled and its answer discarded
  instead of written over somebody else's recovery;

- **Modal failing is a shadow run failing.** Every way the remote half can go wrong —
  unreachable, not deployed, no credentials, a timeout, a result that is not a document —
  arrives here as a `ModalShadowError` and is handled by the `except` that was already
  catching a stub that raised. Nothing about production changes, including in the case where
  Modal is enabled and completely down.

Off unless switched on. `DEEPGUARD_SHADOW_MODE` gates enqueueing, so a deployment that has
not asked for shadow mode does not accumulate rows for a workload nobody is measuring.
Execution is deliberately *not* gated by the same flag: runs already queued when the flag went
away are still finished rather than stranded `queued` forever.
"""

import logging
import os
import uuid
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app import modal_client
from app.db.models import (
    SHADOW_RUN_STATUS_COMPLETED,
    SHADOW_RUN_STATUS_FAILED,
    SHADOW_RUN_STATUS_PROCESSING,
    SHADOW_RUN_STATUS_QUEUED,
    ShadowRun,
)

logger = logging.getLogger(__name__)

# Whether completed analyses are enqueued for shadow execution. Absent means off: shadow mode
# is an experiment being run deliberately, and a deployment that never asked for one should
# not be quietly building a corpus for it.
SHADOW_MODE_VARIABLE = "DEEPGUARD_SHADOW_MODE"

# The values that turn it on. Spelled out rather than "anything non-empty", so
# `DEEPGUARD_SHADOW_MODE=false` — which a compose file will eventually contain — means false.
ENABLED_VALUES = frozenset({"1", "true", "yes", "on"})

# The only workload there is, and it is a stub (R6-T1). Named as a constant because the
# `shadow_runs` rows carry it: when a real workload replaces this one, its observations must
# be distinguishable from the observations of the placeholder that proved the pipeline.
STUB_WORKLOAD = "stub"

# What the stub reports as its deployment identity, in the column a real workload will record
# a commit and a weights digest in.
STUB_WORKLOAD_VERSION = "stub-1"

# The same stub, executed on Modal's GPU instead of in this process (R6-T2). A second name
# rather than a flag on the row, because the name is what a calibration reads a year later:
# `stub` and `modal-stub` are observations of the same workload made by two different
# backends, and a corpus that cannot separate them cannot answer whether the backend mattered.
#
# It also makes the unique constraint do something useful. `(analysis_id, workload)` means one
# analysis can carry both, so turning Modal on does not orphan the local runs already queued
# and does not refuse a remote run for an analysis that has a local one.
MODAL_STUB_WORKLOAD = "modal-stub"

# How long a claim on a shadow run is believed for. Ten minutes. Generous rather than tight
# because nothing depends on a shadow run being recovered quickly: the cost of a stale row
# sitting a little longer is one analysis missing from a corpus that is measured offline, not
# a customer waiting.
#
# Since R6-T2 it is also renewed rather than only set — see `renew_lease`. The figure did not
# have to move for remote execution, and that is the point of renewing instead: a lease long
# enough to cover the worst cold start anybody ever sees would be a lease too long to recover
# a dead worker's row from in reasonable time.
SHADOW_LEASE_SECONDS = 600

# What a recovered run records, as a class name like every other failure written to a status
# column. Nobody caught an exception to produce it — the worker that would have is gone.
STALE_LEASE_ERROR = "StaleShadowLease"

# What a failed run records at most, for the same reason `app.worker` bounds its own: the
# exception's text can quote credentials, endpoints or SQL, so the class name is written down
# and the traceback stays in the log.
MAX_ERROR_MESSAGE = 200


# The remote call this worker started and has not collected yet, keyed by run id. At most one
# entry: the worker claims one shadow run at a time, and a second would need a second lease to
# renew and a second deadline to watch for no gain — a shadow corpus grows by a handful of
# rows a day.
#
# Process memory rather than a column on the row, which `app.modal_client.ModalCall` explains:
# persisting the call id would let another worker collect a call this one started, and the two
# would then need to agree about who owns an in-flight remote call on top of the lease they
# already use to agree about who owns the row. A worker that dies instead loses the call, and
# the lease it stops renewing fails the row exactly as R6-T1 arranged.
_pending: dict[uuid.UUID, tuple["ClaimedShadowRun", Any]] = {}


class UnknownShadowWorkload(Exception):
    """A queued run names a workload this worker cannot execute."""


@dataclass(frozen=True)
class ShadowObservation:
    """What a shadow workload saw, as the workload reported it.

    Deliberately not shaped like `SvdEvidence`, `FaceEvidence` or `LipEvidence`. Those three
    types are the risk engine's input contract and carry a calibrated score, a status the
    rules branch on and a count the rules test for degeneracy; this one carries an opaque
    document nothing in this codebase interprets. Making it resemble them would be an
    invitation to pass one where the other belongs, which is exactly the mistake the whole
    module is arranged to make impossible — and which `app.risk_engine.evaluate` now refuses
    outright.
    """

    provider_version: str
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ClaimedShadowRun:
    """A shadow run this worker owns, as plain values rather than a row.

    Read inside the claim transaction and carried out of it, for the reason
    `app.worker.ClaimedJob` gives: after the commit there is no session and no lock, and an
    ORM row here would invite database access during the workload itself.
    """

    run_id: uuid.UUID
    analysis_id: uuid.UUID
    workload: str


def enabled() -> bool:
    """Whether this deployment has asked for shadow mode."""
    return os.getenv(SHADOW_MODE_VARIABLE, "").strip().lower() in ENABLED_VALUES


def workload() -> str:
    """Which workload a run queued right now is a run of (R6-T2).

    Decided at enqueue time and written on the row, rather than decided again at execution
    time from configuration that may have changed since. That ordering is what makes the
    dispatch in `run_workload` honest: a row says what it is, and a `modal-stub` row queued
    while Modal was configured fails cleanly if Modal has been turned off since, instead of
    silently completing as a local stub and putting an observation nobody made into a corpus
    labelled `modal-stub`.
    """
    return MODAL_STUB_WORKLOAD if modal_client.configured() else STUB_WORKLOAD


def lease_deadline():
    """How far ahead of *the database's* clock a fresh lease reaches.

    `now()` and the arithmetic in PostgreSQL, exactly as `app.worker.lease_deadline` does it
    and for the same reason: the comparison that decides staleness happens there too, so two
    workers whose machine clocks disagree must not be writing deadlines on one timeline and
    having them judged on another.
    """
    return func.now() + timedelta(seconds=SHADOW_LEASE_SECONDS)


def enqueue(session: Session, analysis_id: uuid.UUID) -> bool:
    """Queue a shadow run for a completed analysis. Returns whether one was queued.

    Called from the worker on the way out of a job it finished, and structured so that it can
    fail for any reason at all without the analysis noticing. It runs in its own transaction,
    after the decision has been committed and the job closed; a failure here rolls back this
    insert and nothing else, and is logged rather than raised.

    Never raises, deliberately, and that is the load-bearing property of this function rather
    than an oversight in its error handling. A production analysis that failed because an
    experiment could not be scheduled would invert the entire point of shadow mode.

    A duplicate is not a failure. The unique constraint refuses a second run of the same
    workload against the same analysis — a job concluded twice, two workers racing — and that
    refusal is the intended outcome, not an error to report.
    """
    if not enabled():
        return False

    queued_workload = workload()

    try:
        session.add(
            ShadowRun(
                analysis_id=analysis_id,
                workload=queued_workload,
                status=SHADOW_RUN_STATUS_QUEUED,
            )
        )
        session.commit()
    except Exception:
        session.rollback()
        logger.warning(
            "Queueing a shadow run for analysis %s failed; the analysis is unaffected.",
            analysis_id,
            exc_info=True,
        )
        return False

    logger.info(
        "Queued a %s shadow run for analysis %s.", queued_workload, analysis_id
    )
    return True


def recover_stale_runs(session: Session) -> int:
    """Fail every `processing` shadow run whose worker stopped being alive.

    The same test `app.worker.recover_stale_jobs` applies, against this table: a lease that
    has run out is a worker that is gone. Terminal for the same reason too — a workload that
    reliably kills the process would otherwise be handed to worker after worker — and cheap
    for the same reason, one statement that normally matches nothing.

    No analysis and no job is touched here. A shadow run's failure is a fact about an
    experiment and about nothing else.
    """
    recovered = session.execute(
        ShadowRun.__table__.update()
        .where(
            ShadowRun.status == SHADOW_RUN_STATUS_PROCESSING,
            # Never null on a claimed run, but stated rather than assumed: `NULL < now()` is
            # null, so a row that somehow had no lease would never be reached.
            ShadowRun.lease_expires_at.is_not(None),
            ShadowRun.lease_expires_at < func.now(),
        )
        .values(
            status=SHADOW_RUN_STATUS_FAILED,
            error_message=STALE_LEASE_ERROR,
            lease_expires_at=None,
        )
        .returning(ShadowRun.id)
    ).scalars().all()

    if not recovered:
        session.rollback()
        return 0

    session.commit()

    logger.warning(
        "Recovered %s stale shadow run(s): %s.",
        len(recovered),
        ", ".join(str(run_id) for run_id in recovered),
    )

    return len(recovered)


def claim_run(session: Session) -> ClaimedShadowRun | None:
    """Take exclusive ownership of one queued shadow run, or return nothing if there is none.

    `FOR UPDATE ... SKIP LOCKED`, oldest first, status moved before the commit releases the
    lock — the same three properties that make `app.worker.claim_job` safe against several
    workers, applied to this table. The lease starts in the same transaction, so a row is
    never `processing` without a deadline something could eventually recover it by.
    """
    run = session.execute(
        select(ShadowRun)
        .where(ShadowRun.status == SHADOW_RUN_STATUS_QUEUED)
        .order_by(ShadowRun.created_at)
        .limit(1)
        .with_for_update(skip_locked=True)
    ).scalar_one_or_none()

    if run is None:
        session.rollback()
        return None

    run.status = SHADOW_RUN_STATUS_PROCESSING
    run.lease_expires_at = lease_deadline()
    claimed = ClaimedShadowRun(
        run_id=run.id, analysis_id=run.analysis_id, workload=run.workload
    )
    session.commit()

    return claimed


def renew_lease(session: Session, claimed: ClaimedShadowRun) -> bool:
    """Push this run's lease out by a full period. Returns whether it is still ours.

    The whole of the heartbeat, in one conditional update. Conditional on the row still being
    `processing`, which is what turns "renew the lease" into "renew the lease *and* tell me
    whether I still have it": a row that `recover_stale_runs` already failed matches nothing
    and the caller learns it has lost the run, which is the only useful thing to know halfway
    through a remote call.

    Not a background thread, and that is a deliberate limit on how much machinery this task
    adds. The only workload long enough to need this is one that spends its time waiting in a
    poll loop, and a loop is somewhere to put a heartbeat. A workload that blocked instead
    would need a thread — and would need it designed, because a thread renewing a lease for a
    call that has already failed is a lease that never expires.
    """
    renewed = session.execute(
        ShadowRun.__table__.update()
        .where(
            ShadowRun.id == claimed.run_id,
            ShadowRun.status == SHADOW_RUN_STATUS_PROCESSING,
        )
        .values(lease_expires_at=lease_deadline())
    )

    if renewed.rowcount != 1:
        session.rollback()
        return False

    session.commit()
    return True


def run_stub_workload(claimed: ClaimedShadowRun) -> ShadowObservation:
    """The placeholder workload R6-T1 proves the infrastructure with.

    It loads no model, reads no media and reaches no GPU. What it produces is a record that a
    queued run was claimed and executed by a worker, which is the only thing this task set out
    to demonstrate — integrating a real experimental detector is a later task, and doing it
    here would have meant reviewing the isolation boundaries and a detector at once.

    The one thing it is careful about is shape: it returns a `ShadowObservation` holding an
    opaque document, which is what a real workload will return too, so replacing this function
    changes nothing else in the module.
    """
    return ShadowObservation(
        provider_version=STUB_WORKLOAD_VERSION,
        evidence={"observed": True, "analysis_id": str(claimed.analysis_id)},
    )


def start_workload(
    session: Session, claimed: ClaimedShadowRun
) -> ShadowObservation | None:
    """Begin the workload this run was queued as. `None` means it is running remotely.

    Dispatch on the name the row carries rather than on configuration read now, for the
    reason `workload` gives: a row is a run *of* something, and what it is a run of was
    decided when it was queued.

    The local stub finishes here and returns its observation. The Modal one cannot: it is
    *started* here and collected on a later poll, because this function runs on the worker's
    only thread and a remote GPU takes tens of seconds to answer. Returning `None` is what
    hands the loop back — see `_collect_pending`.

    An unknown name is an error rather than a fallback. A row naming a workload this worker
    does not have is either a deployment mid-rollout or a workload that has been removed, and
    in both cases failing the run says so, where quietly running the stub instead would put a
    `stub` observation into a corpus under some other workload's name.
    """
    if claimed.workload == MODAL_STUB_WORKLOAD:
        _pending[claimed.run_id] = (claimed, modal_client.spawn_stub(claimed.analysis_id))
        return None

    if claimed.workload == STUB_WORKLOAD:
        return run_stub_workload(claimed)

    raise UnknownShadowWorkload(claimed.workload)


def _collect_pending(session: Session) -> bool:
    """Ask about the remote call this worker started, and finish the run if it has answered.

    Returns whether anything is still in flight afterwards, which is how `process_one` knows
    not to claim a second run.

    Three things happen on every visit, in this order and for three different reasons:

    - **the lease is renewed first.** A remote call outliving `SHADOW_LEASE_SECONDS` is the
      case this exists for, and renewing before asking means a slow answer is never the reason
      a run is recovered as stale. If the renewal is refused the run is no longer this
      worker's — recovered while the GPU was busy, or taken — so the call is cancelled and
      forgotten rather than waited on for an observation `complete_run` would refuse anyway;
    - **the call is asked, never waited for.** `modal_client.collect` returns `None` if there
      is no answer yet, and this returns straight back to the loop, which polls production
      again in a couple of seconds. That is the whole non-blocking guarantee: the worker is
      inside Modal's SDK for one round trip per idle poll and never for the length of a run;
    - **a failure ends the run and nothing else.** Every way Modal can go wrong arrives as a
      `ModalShadowError`, fails this row, and leaves the worker polling.
    """
    run_id, (claimed, pending) = next(iter(_pending.items()))

    if not renew_lease(session, claimed):
        logger.warning(
            "Shadow run %s was recovered as stale while its Modal workload ran; the remote "
            "call is cancelled and its observation discarded.",
            run_id,
        )
        _pending.pop(run_id, None)
        modal_client.cancel(pending)
        return False

    try:
        result = modal_client.collect(pending)
    except Exception as error:
        logger.exception("Shadow run %s failed on Modal.", run_id)
        _pending.pop(run_id, None)
        modal_client.cancel(pending)
        _record_failure(session, claimed, error)
        return False

    if result is None:
        return True

    _pending.pop(run_id, None)
    _record_observation(
        session,
        claimed,
        ShadowObservation(
            provider_version=result.provider_version, evidence=result.evidence
        ),
    )
    return False


def abandon_pending() -> None:
    """Cancel and forget whatever this worker was waiting on. Used when it is shutting down.

    The row is left `processing` with a lease that stops being renewed, which
    `recover_stale_runs` fails on some later worker's poll — the R6-T1 behaviour for a worker
    that goes away, reached deliberately here instead of by dying. What this adds is the
    cancellation: a container still running for an answer nobody will collect is GPU time
    billed for nothing.
    """
    while _pending:
        _, (_, pending) = _pending.popitem()
        modal_client.cancel(pending)


def complete_run(
    session: Session, claimed: ClaimedShadowRun, observation: ShadowObservation
) -> bool:
    """Record what the workload observed and close the run out. Returns whether it counted.

    Conditional on the run still being `processing`, which is what stops a worker whose lease
    expired from writing an observation over a recovery that already happened. A run that is
    no longer this worker's is rolled back and reported, exactly as `app.worker._set_status`
    treats a job that was recovered out from under it.
    """
    updated = session.execute(
        ShadowRun.__table__.update()
        .where(
            ShadowRun.id == claimed.run_id,
            ShadowRun.status == SHADOW_RUN_STATUS_PROCESSING,
        )
        .values(
            status=SHADOW_RUN_STATUS_COMPLETED,
            provider_version=observation.provider_version,
            evidence=observation.evidence,
            error_message=None,
            lease_expires_at=None,
        )
    )

    if updated.rowcount != 1:
        session.rollback()
        return False

    session.commit()
    return True


def fail_run(session: Session, claimed: ClaimedShadowRun, error: BaseException) -> None:
    """Record that the workload failed, without recording anything it might have said.

    The class name and not the message, bounded, for the reason `app.worker.fail_job` gives:
    an exception's own text can quote a credential or an endpoint, and the traceback is
    already in the log.

    No evidence is written. A workload that raised produced no observation, and an empty
    document in that column would be an observation nobody made.
    """
    session.execute(
        ShadowRun.__table__.update()
        .where(
            ShadowRun.id == claimed.run_id,
            ShadowRun.status == SHADOW_RUN_STATUS_PROCESSING,
        )
        .values(
            status=SHADOW_RUN_STATUS_FAILED,
            error_message=type(error).__name__[:MAX_ERROR_MESSAGE],
            lease_expires_at=None,
        )
    )
    session.commit()


def _discard_transaction(session: Session) -> None:
    """Roll back whatever this module left open, and never raise doing it.

    The promise `process_one` makes is that nothing propagates out of it, and a rollback is
    reached on exactly the paths where the session may itself be the broken thing — a
    connection that has gone away cannot be rolled back either. A failure to clean up must not
    become the exception the cleanup was there to contain.
    """
    try:
        session.rollback()
    except Exception:
        logger.debug("Rolling back after a failed shadow run failed too.", exc_info=True)


def _record_failure(
    session: Session, claimed: ClaimedShadowRun, error: BaseException
) -> None:
    """Write a run's failure down, and never raise doing it.

    A failure that cannot be recorded is not a failure that propagates: the lease is still
    there, and `recover_stale_runs` will close the row out on some later poll. Losing the
    precise reason is a worse record of an experiment, not a worse outcome for anybody.
    """
    try:
        fail_run(session, claimed, error)
    except Exception:
        logger.exception(
            "Recording the failure of shadow run %s failed; it will be recovered by its "
            "lease.",
            claimed.run_id,
        )
        _discard_transaction(session)


def _record_observation(
    session: Session, claimed: ClaimedShadowRun, observation: ShadowObservation
) -> None:
    """Write what a workload saw, and never raise doing it."""
    try:
        if complete_run(session, claimed, observation):
            logger.info(
                "Completed shadow run %s (%s) for analysis %s.",
                claimed.run_id,
                claimed.workload,
                claimed.analysis_id,
            )
        else:
            logger.warning(
                "Shadow run %s was recovered as stale while it ran; its observation is "
                "discarded.",
                claimed.run_id,
            )
    except Exception:
        logger.exception(
            "Recording the observation of shadow run %s failed.", claimed.run_id
        )
        _discard_transaction(session)


def process_one(session: Session) -> bool:
    """Advance shadow execution by one step. Returns whether there was anything to do.

    The shadow half of `app.worker.process_one`, and called by the worker loop only when the
    production half found nothing to do. That ordering was R6-T1's non-blocking guarantee: an
    experiment is never *claimed* while a queued job exists.

    R6-T2 had to make a second guarantee, because the first one turned out not to cover the
    case that matters once a workload runs on a remote GPU. An experiment claimed on an idle
    poll used to finish in microseconds; a Modal call takes tens of seconds, and while the
    worker was inside it, a job submitted a moment later sat queued. It was measured: a
    production job was claimed 8 ms after an in-flight Modal call returned, having waited
    4.1 s behind it, and a cold start or a hung Modal would have made that far worse.

    So a step is never a wait. There are exactly three of them, and each returns to the loop
    promptly:

    - a remote call is outstanding — renew its lease, ask once whether it has answered, and
      hand the loop straight back if it has not;
    - nothing is outstanding and a run is queued — claim it and *start* it. The local stub
      finishes inside this step; the Modal one is spawned and collected on a later visit;
    - nothing is outstanding and nothing is queued — say so, and the loop goes to sleep.

    Nothing propagates out of here. Every failure — the workload's, the database's, a defect
    in this module — is logged and swallowed, so a broken experiment cannot reach the loop
    that runs production work, cannot trigger its error backoff and cannot stop it claiming
    the next job. This is the one place in the codebase where a bare `except Exception` that
    reports `False` is the correct behaviour rather than a smell: the caller has production
    work to get on with and there is nothing it could usefully do about an experiment that
    went wrong.
    """
    if _pending:
        try:
            return _collect_pending(session)
        except Exception:
            logger.exception(
                "Collecting a remote shadow run failed; production work is unaffected."
            )
            _discard_transaction(session)
            return False

    try:
        recover_stale_runs(session)

        claimed = claim_run(session)
        if claimed is None:
            return False
    except Exception:
        logger.exception("Claiming a shadow run failed; production work is unaffected.")
        _discard_transaction(session)
        return False

    try:
        observation = start_workload(session, claimed)
    except Exception as error:
        logger.exception("Shadow run %s failed.", claimed.run_id)
        _record_failure(session, claimed, error)
        return True

    # `None` means the workload is running somewhere else and will be collected on a later
    # poll. The run stays `processing`, holding the lease `_collect_pending` renews.
    if observation is not None:
        _record_observation(session, claimed, observation)

    return True
