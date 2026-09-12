"""The one route an administrator inspects detection work through.

Mounted under `/api/v1/admin` beside the account routes and behind the same `require_admin`,
because it answers the same kind of question: not "what did my analysis conclude" — which is
the workspace's, narrowed to the reader's own rows — but "what is this deployment's queue
doing", which is everybody's rows and therefore nobody's but an operator's.

It is read-only, and that is a decision rather than an omission. The obvious next control on
a failure table is a retry button, and this system cannot honour one: `persist_evidence`
writes a signal per provider with `session.add()`, so re-running a job that already got as
far as writing evidence would insert a second row per provider — duplicate forensic record
where the schema promises one, on a table whose whole purpose is that a stored signal is what
the detector actually said. Requeuing safely means deciding what happens to the existing
evidence first, which is a schema question and not a button. Until that is answered there is
no retry endpoint here, and no POST or PATCH of any kind.

What an operator is given instead is the two facts the job row already holds. `error_message`
is the worker's own diagnostic text, written by `fail_job` and by the stale-lease recovery,
and it is carried across unchanged — nothing here classifies a failure or guesses at a cause,
because the only honest answer to "why did this fail" is the one the process that failed
wrote down. And `is_stale` is computed here rather than left to the reader, for the reason
given on `stale_lease`.
"""

import logging
import uuid
from datetime import datetime

from fastapi import APIRouter, Depends
from pydantic import BaseModel, ConfigDict
from sqlalchemy import Boolean, case, func, select
from sqlalchemy.orm import Session
from sqlalchemy.sql.elements import ColumnElement

from app.db.models import JOB_STATUS_PROCESSING, AnalysisJob
from app.db.session import get_session
from app.web_auth import require_admin

logger = logging.getLogger(__name__)

# How many jobs one read returns. A ceiling and not a page size: there is no cursor here and
# no `?limit=`, because the question this screen answers is "what has the queue been doing
# lately" and that is the top of the table.
#
# The account listing next door has no limit at all, and the difference is the table. Accounts
# are tens of rows in an internal deployment; jobs are one row per analysis ever submitted and
# grow for as long as the service runs, so an unbounded `SELECT` here is a query that works in
# development and returns a hundred thousand rows in a year. Paging back through history is a
# real requirement the day somebody has it — against a real number, with a real cursor.
RECENT_JOB_LIMIT = 100

# Every route below is behind `require_admin`, declared once on the router rather than per
# route, so that a route added to this file later cannot be an administrative route somebody
# forgot to guard. Same shape as `admin_users.py`, and the same dependency — not a second
# role check written here, which would be a second answer to the only question that matters.
#
# `require_admin` is layered on `require_user`, so an anonymous caller is refused with 401 and
# an authenticated non-administrator with 403.
router = APIRouter(
    prefix="/api/v1/admin",
    tags=["admin"],
    dependencies=[Depends(require_admin)],
)


class AdminJob(BaseModel):
    """One unit of detection work as an administrator is shown it.

    The job row's operational fields and nothing from the analysis behind it. No storage key,
    no source URL, no filename, no owner: this screen is about the queue, and an operator
    reading it does not need the media's whereabouts to see that a worker died.

    Built field by field in `visible_job` rather than by `from_attributes`, the convention
    `AdminUser` sets — a column added to `AnalysisJob` later reaches this payload when
    somebody decides it should, not by having been added.
    """

    model_config = ConfigDict(from_attributes=False)

    id: uuid.UUID
    analysis_id: uuid.UUID
    status: str
    request_id: str | None
    lease_expires_at: datetime | None
    error_message: str | None
    created_at: datetime
    updated_at: datetime

    # Whether this job's claim has expired while it still says it is running — see
    # `stale_lease`. Derived, never stored: there is no such column, and the answer changes
    # with the clock rather than with a write.
    is_stale: bool


def stale_lease() -> ColumnElement[bool]:
    """The `is_stale` expression: a claimed job whose claim has run out.

    Three conditions, matching `recover_stale_jobs` in `app/worker.py` exactly, because the
    two must agree about what stale means — this route shows the operator the rows that
    recovery is about to fail, and a looser test here would label jobs the worker will leave
    running. The `lease_expires_at IS NOT NULL` clause is redundant against a real claimed job
    and stated anyway for the same reason it is stated there: `NULL < now()` is null, not
    false, and a `processing` row without a lease would otherwise fall out of the comparison
    silently.

    Evaluated by PostgreSQL, against `now()`, in the same statement that reads the rows. That
    is what makes this the server's answer rather than an answer about the server: the clock
    is the one the worker leases against, so a machine running the API with a skewed clock
    cannot invent a stale job or hide one. The browser's clock never enters it — the API sends
    a boolean, not a deadline for a page to compare against `Date.now()`.
    """
    return case(
        (
            (
                (AnalysisJob.status == JOB_STATUS_PROCESSING)
                & AnalysisJob.lease_expires_at.is_not(None)
                & (AnalysisJob.lease_expires_at < func.now())
            ),
            True,
        ),
        else_=False,
    ).cast(Boolean)


def visible_job(job: AnalysisJob, is_stale: bool) -> AdminJob:
    """The stored job narrowed to what an administrator is told about it.

    The one place an `AnalysisJob` becomes a response in this module, so there is one place to
    read to know what leaves it.
    """
    return AdminJob(
        id=job.id,
        analysis_id=job.analysis_id,
        status=job.status,
        request_id=job.request_id,
        lease_expires_at=job.lease_expires_at,
        error_message=job.error_message,
        created_at=job.created_at,
        updated_at=job.updated_at,
        is_stale=is_stale,
    )


@router.get("/jobs", response_model=list[AdminJob])
def list_jobs(session: Session = Depends(get_session)) -> list[AdminJob]:
    """The most recent jobs, newest first, whatever state they are in.

    Unfiltered by status on purpose. An operator arrives here after a report did not appear,
    and the answer is as often "still queued" or "running" as it is "failed" — a table that
    showed only failures would answer one of those three and leave the other two looking like
    nothing had been submitted at all.

    Ordered by `created_at` descending with the id as the tiebreaker, so the listing is stable
    between renders: two jobs committed in the same transaction share a timestamp, and a sort
    with ties in it reshuffles itself on every read.
    """
    rows = session.execute(
        select(AnalysisJob, stale_lease().label("is_stale"))
        .order_by(AnalysisJob.created_at.desc(), AnalysisJob.id.desc())
        .limit(RECENT_JOB_LIMIT)
    ).all()

    return [visible_job(job, is_stale) for job, is_stale in rows]
