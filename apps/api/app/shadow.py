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

# How long a claim on a shadow run is believed for. Ten minutes, and no heartbeat renewing it
# — see `ShadowRun.lease_expires_at`. Generous rather than tight because nothing depends on a
# shadow run being recovered quickly: the cost of a stale row sitting a little longer is one
# analysis missing from a corpus that is measured offline, not a customer waiting.
SHADOW_LEASE_SECONDS = 600

# What a recovered run records, as a class name like every other failure written to a status
# column. Nobody caught an exception to produce it — the worker that would have is gone.
STALE_LEASE_ERROR = "StaleShadowLease"

# What a failed run records at most, for the same reason `app.worker` bounds its own: the
# exception's text can quote credentials, endpoints or SQL, so the class name is written down
# and the traceback stays in the log.
MAX_ERROR_MESSAGE = 200


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

    try:
        session.add(
            ShadowRun(
                analysis_id=analysis_id,
                workload=STUB_WORKLOAD,
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
        "Queued a %s shadow run for analysis %s.", STUB_WORKLOAD, analysis_id
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


def process_one(session: Session) -> bool:
    """Claim and execute a single shadow run. Returns whether there was one to run.

    The shadow half of `app.worker.process_one`, and called by the worker loop only when the
    production half found nothing to do. That ordering is the non-blocking guarantee at the
    level of the queue rather than of one job: no analysis waits behind an experiment, because
    an experiment is only ever picked up on a poll where there was no analysis to run.

    Nothing propagates out of here. Every failure — the workload's, the database's, a defect
    in this module — is logged and swallowed, so a broken experiment cannot reach the loop that
    runs production work, cannot trigger its error backoff and cannot stop it claiming the next
    job. This is the one place in the codebase where a bare `except Exception` that reports
    `False` is the correct behaviour rather than a smell: the caller has production work to get
    on with and there is nothing it could usefully do about an experiment that went wrong.
    """
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
        observation = run_stub_workload(claimed)
    except Exception as error:
        logger.exception("Shadow run %s failed.", claimed.run_id)
        try:
            fail_run(session, claimed, error)
        except Exception:
            logger.exception(
                "Recording the failure of shadow run %s failed; it will be recovered by "
                "its lease.",
                claimed.run_id,
            )
            _discard_transaction(session)
        return True

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

    return True
