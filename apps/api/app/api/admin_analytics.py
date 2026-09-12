"""What this deployment has been doing for the last seven days, counted rather than judged.

Mounted under `/api/v1/admin` beside the account and job routes, behind the same
`require_admin`, and read-only for the same reason `admin_jobs.py` is: these are everybody's
rows, which makes them an operator's question and nobody else's, and there is nothing on this
screen an administrator could act on if it offered them a control.

What separates this route from the job listing next door is that it returns no rows at all.
Every number below is produced by a `GROUP BY` that PostgreSQL evaluates, and what crosses
into Python is one short list of `(category, count)` pairs per question. That is not a
performance preference — it is the only way the counts can be trusted. A route that read the
week's analyses into a list and counted them would be counting whatever the reader happened
to be entitled to see, in whatever order the driver handed them over, with every semantic
decision about what a row *means* re-made in application code. Here the database counts, and
this module only decides which bucket a name belongs in.

Three things it deliberately does not do.

*It does not average, compare, or expose a detector score.* `AnalysisSignal.score` is never
selected. Provider health here is a count of the `status` values the detectors' own runs
persisted — how often each provider answered at all — and that is an operational fact about
this deployment, not forensic evidence about any media. A column of mean scores per provider
would read as a leaderboard of which detector is right, which is a question this system does
not answer and which nothing in the schema entitles anyone to answer.

*It does not treat a missing signal as a failure.* Providers are not all invoked for every
analysis, so an analysis with no row for a provider means that provider was not asked. Only
persisted rows are counted, and the arithmetic is per provider rather than against the week's
analysis total, so there is no denominator anywhere that a never-invoked provider could fall
short of.

*It does not re-decide a risk level.* The distribution is `Analysis.risk_level` as stored. A
null there is not a missing value to be filled in or a decision still in flight — it is the
absence of a persisted decision, and it is reported under its own name (`UNDECIDED`) rather
than folded into `UNKNOWN`, which is a decision the risk engine reached and wrote down.

The window is fixed at seven days and there is no `?from=`. The question this screen answers
is "is the pipeline healthy right now", and the answer to that is recent; historical slicing
is a different screen with a different shape, to be built the day somebody needs one.
"""

import logging
from collections.abc import Iterable, Sequence
from datetime import timedelta

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy import ColumnElement, Row, func, select
from sqlalchemy.orm import Session

from app.db.models import (
    ACQUISITION_METHOD_UPLOAD,
    ACQUISITION_METHOD_URL,
    ANALYSIS_STATUS_COMPLETED,
    ANALYSIS_STATUS_FAILED,
    ANALYSIS_STATUS_QUEUED,
    JOB_STATUS_COMPLETED,
    JOB_STATUS_FAILED,
    JOB_STATUS_PROCESSING,
    JOB_STATUS_QUEUED,
    SIGNAL_STATUS_FAILED,
    SIGNAL_STATUS_SUCCESS,
    SIGNAL_STATUS_TIMEOUT,
    Analysis,
    AnalysisJob,
    AnalysisSignal,
    MediaFile,
)
from app.db.session import get_session
from app.risk_engine import RISK_HIGH, RISK_MEDIUM, RISK_UNKNOWN
from app.web_auth import require_admin

logger = logging.getLogger(__name__)

# How far back every count below reaches. Seven days, hardcoded, and the only period this
# route has: see the module docstring for why there is no parameter for it.
WINDOW_DAYS = 7

# The window's name on the wire. A label for the reader, not a parseable value — a client that
# wanted the boundary as a timestamp would be a client computing its own window, which is the
# thing this route exists to have exactly one of.
WINDOW_LABEL = f"{WINDOW_DAYS}d"

# The bucket a row with no persisted `risk_level` is counted under.
#
# Spelled in the same case as the risk engine's own levels because it sits beside them in one
# distribution and a lone lowercase key would read as a different kind of thing. It is not one
# of them: `RISK_UNKNOWN` means the engine looked and could not conclude, and this means the
# engine has not written an answer at all. Collapsing the two would turn "we have not decided"
# into "we decided we could not tell", which is a claim about media nobody has assessed.
RISK_UNDECIDED = "UNDECIDED"

# The bucket a media row with no persisted `acquisition_method` is counted under.
#
# `app/db/models.py` states what a null there means: the analysis predates the column, and is
# read as "not recorded" rather than resolved to a guess. Lowercase, because the two methods it
# sits beside are lowercase, for the same reason `RISK_UNDECIDED` is capitalised.
ACQUISITION_UNRECORDED = "unrecorded"

# The categories each distribution reports even when the week produced none of them.
#
# A key that vanishes when its count reaches zero is the failure mode these lists exist to
# prevent: a quiet week has no `failed` jobs, the key disappears, and the screen that should
# be saying "nothing failed" instead says nothing at all — indistinguishable, to a reader, from
# a metric that was never wired up. So the vocabulary the schema defines is always present and
# a category nobody produced reads as the zero it is.
#
# These are floors and not filters. A status outside them is carried through under its own
# name by `tallied`, because a value the database holds and this module does not recognise is
# exactly the thing an operator needs to see, not the thing to drop.
ANALYSIS_STATUSES = (
    ANALYSIS_STATUS_QUEUED,
    ANALYSIS_STATUS_COMPLETED,
    ANALYSIS_STATUS_FAILED,
)
JOB_STATUSES = (
    JOB_STATUS_QUEUED,
    JOB_STATUS_PROCESSING,
    JOB_STATUS_COMPLETED,
    JOB_STATUS_FAILED,
)
ACQUISITION_METHODS = (
    ACQUISITION_METHOD_UPLOAD,
    ACQUISITION_METHOD_URL,
    ACQUISITION_UNRECORDED,
)

# The three outcomes a detector run can persist, and what every provider in the health table
# is reported against.
#
# All three, never two. `app/db/models.py` is explicit that a provider that refused and a
# provider that never answered in time are different forensic facts, and a table that showed
# only SUCCESS and FAILED would be collapsing the timeouts into one or other of them — or, if
# it dropped them, quietly under-reporting how often a detector is too slow to be useful,
# which is the single most actionable thing on this page.
SIGNAL_STATUSES = (
    SIGNAL_STATUS_SUCCESS,
    SIGNAL_STATUS_FAILED,
    SIGNAL_STATUS_TIMEOUT,
)

# Every risk level the engine can write, plus the absence of one.
#
# Imported from `app/risk_engine.py` rather than spelled out here, because these are the
# engine's own vocabulary and this route is only reporting it. A literal `"HIGH"` written
# here would be a second definition that goes on looking correct for as long as it takes
# somebody to rename a level — at which point this screen would report zero of the new name
# under a key for the old one, which is worse than failing.
#
# `RISK_UNDECIDED` is deliberately not among them and is defined in this module: it is not a
# level the engine can write. It is what this route calls the absence of one.
RISK_LEVELS = (RISK_HIGH, RISK_MEDIUM, RISK_UNKNOWN, RISK_UNDECIDED)


class AnalyticsWindow(BaseModel):
    """The seven-day picture, as one payload.

    Every value is a count of persisted rows. There is no rate, no ratio and no average
    anywhere in this shape, which is deliberate: a derived number is an interpretation, and the
    moment one appears here it becomes this route's interpretation rather than the reader's.
    `failed` and `completed` side by side say everything a failure rate would, and say it
    without deciding whether in-flight work belongs in the denominator.
    """

    # Which window produced these counts. Present so a payload read on its own — logged,
    # pasted into a ticket — still says what it is a count of.
    window: str

    # Analyses submitted in the window, by their stored status, plus the total. The total is a
    # separate aggregate rather than a sum of the map, so it stays right if a status outside
    # `ANALYSIS_STATUSES` ever appears.
    analyses_total: int
    analyses_by_status: dict[str, int]

    # Detection work queued in the window, by job status. Distinct from the analyses above:
    # they are the same population today, and nothing guarantees they stay that way, so they
    # are counted separately rather than one being presented as the other.
    jobs_by_status: dict[str, int]

    # The window's analyses by the risk level persisted on them, with `UNDECIDED` for the ones
    # carrying none.
    risk_distribution: dict[str, int]

    # How the window's media arrived. Keyed by `MediaFile.acquisition_method`, with
    # `unrecorded` for rows predating the column.
    acquisition: dict[str, int]

    # Provider health: for each detector that persisted at least one signal in the window, how
    # many of each status it wrote. Providers appear because they have rows, never because a
    # list here says they should — a detector nobody invoked this week is absent, not zeroed,
    # and the difference is the difference between "asked and silent" and "not asked".
    detectors: dict[str, dict[str, int]]


def within_window(created_at: ColumnElement) -> ColumnElement[bool]:
    """The window predicate, built once and applied to each table's own `created_at`.

    The boundary is `func.now() - interval`, computed by PostgreSQL inside each statement,
    rather than a `datetime.now(timezone.utc)` resolved in this process and sent as a bound
    parameter. Same reasoning as `stale_lease` in `app/api/admin_jobs.py`, and it matters more
    here than it does there: these rows were timestamped by the database's own `now()` default,
    so comparing them against the API container's clock would be comparing two clocks. An API
    host drifting a few minutes would include or exclude a sliver of rows at the boundary, and
    nothing about the result would look wrong. One clock, and it is the one that wrote the
    column.
    """
    return created_at >= func.now() - timedelta(days=WINDOW_DAYS)


def tallied(
    rows: Sequence[Row], *, categories: Iterable[str], absent: str | None = None
) -> dict[str, int]:
    """`(category, count)` pairs from the database, as a map with the known keys present.

    The counting already happened in PostgreSQL; this only decides what the keys are called and
    which ones appear. Three things it does, in order:

    Seeds every category in `categories` at zero, so a quiet week reports zeros rather than
    gaps — see the comment on `ANALYSIS_STATUSES`.

    Renames a null group to `absent`, when the caller has a name for it. A `GROUP BY` over a
    nullable column returns null as its own group, and that group is a real answer with a real
    count; it needs a key, and the caller is the one that knows what its absence means.

    Carries through any category the seed list did not anticipate. That is the case worth being
    careful about: a status this module has never heard of is a genuine signal — a migration
    half-applied, a writer using a spelling the model does not define — and dropping it would
    hide the one row that explains why the numbers do not add up.
    """
    tally = {category: 0 for category in categories}

    for category, count in rows:
        if category is None:
            # A null group with no name for it is not silently merged into anything. The only
            # callers that omit `absent` are the ones whose column is `NOT NULL`, where this
            # cannot happen; if it ever does, it is a schema change and it should be loud.
            if absent is None:
                logger.warning("A null group was counted under a column that forbids null.")
                continue
            category = absent

        tally[category] = tally.get(category, 0) + count

    return tally


def grouped_counts(
    session: Session, column: ColumnElement, created_at: ColumnElement
) -> Sequence[Row]:
    """One `GROUP BY` over one column, narrowed to the window.

    The single shape every distribution below is built from, so there is one place to read to
    know what these statements do — and, more to the point, one place that could ever be
    changed into something that returns rows. `func.count()` is evaluated by PostgreSQL and
    what arrives here is one row per distinct value, however many rows went into it.
    """
    return session.execute(
        select(column, func.count())
        .where(within_window(created_at))
        .group_by(column)
    ).all()


router = APIRouter(
    prefix="/api/v1/admin",
    tags=["admin"],
    dependencies=[Depends(require_admin)],
)


@router.get("/analytics", response_model=AnalyticsWindow)
def read_analytics(session: Session = Depends(get_session)) -> AnalyticsWindow:
    """The deployment's last seven days, counted by the database.

    Six statements, each a `GROUP BY` or a `COUNT`, none of them returning a row of data. They
    are issued separately rather than assembled into one query with five joined subselects
    because they answer five unrelated questions over four tables, and a join between them
    would multiply rows against each other — an analysis with three signals would be counted
    three times in the risk distribution. Five cheap independent scans of a week's rows, read
    inside one transaction and therefore one consistent snapshot of it.

    An empty window is not a special case and is not checked for. Every statement returns no
    rows, `tallied` seeds its categories at zero, `analyses_total` is `COUNT(*)` over nothing,
    and the payload comes out fully formed with zeros in it. That is the difference between a
    screen that says "nothing happened this week" and one that fails to load on a quiet Monday.
    """
    analyses_total = session.execute(
        select(func.count()).select_from(Analysis).where(within_window(Analysis.created_at))
    ).scalar_one()

    analyses_by_status = tallied(
        grouped_counts(session, Analysis.status, Analysis.created_at),
        categories=ANALYSIS_STATUSES,
    )

    jobs_by_status = tallied(
        grouped_counts(session, AnalysisJob.status, AnalysisJob.created_at),
        categories=JOB_STATUSES,
    )

    # Grouped on the stored column, null group and all, and named `UNDECIDED` afterwards rather
    # than `COALESCE`d to that string in SQL. A coalesce would merge a row that genuinely held
    # the literal text "UNDECIDED" into the null bucket and there would be no way to tell from
    # the result that it had happened. Nothing writes that string today; the point is that this
    # route's answer does not depend on nothing ever writing it.
    risk_distribution = tallied(
        grouped_counts(session, Analysis.risk_level, Analysis.created_at),
        categories=RISK_LEVELS,
        absent=RISK_UNDECIDED,
    )

    # Windowed on the *analysis*, joined, because `media_files` has no `created_at` of its own —
    # its row is written in the same transaction as the analysis it belongs to, so the parent's
    # timestamp is when this media arrived. An inner join, which is what makes the arithmetic
    # right: an analysis whose media row was never written contributes nothing here rather than
    # a null that would land in `unrecorded` and misreport an incomplete upload as an old one.
    acquisition = tallied(
        session.execute(
            select(MediaFile.acquisition_method, func.count())
            .join(Analysis, Analysis.id == MediaFile.analysis_id)
            .where(within_window(Analysis.created_at))
            .group_by(MediaFile.acquisition_method)
        ).all(),
        categories=ACQUISITION_METHODS,
        absent=ACQUISITION_UNRECORDED,
    )

    # The only two-column grouping here: provider and status together, so one pass produces the
    # whole table. Windowed on the signal's own `created_at` rather than the analysis's, because
    # a detector that answered this week about media submitted last week is this week's
    # operational fact — the page is about what the providers have been doing lately.
    #
    # `score` is not selected, here or anywhere in this module.
    detectors: dict[str, dict[str, int]] = {}
    signal_rows = session.execute(
        select(AnalysisSignal.provider, AnalysisSignal.status, func.count())
        .where(within_window(AnalysisSignal.created_at))
        .group_by(AnalysisSignal.provider, AnalysisSignal.status)
    ).all()

    for provider, status, count in signal_rows:
        # A provider's row appears the first time one of its signals does, and is seeded with
        # all three statuses at zero. So a detector that answered forty times and failed none
        # reads as `FAILED 0` rather than as a blank the reader has to interpret — while a
        # detector nobody invoked this week stays out of the table entirely, because it has no
        # rows and an absent provider is not a failing one.
        tally = detectors.setdefault(provider, {name: 0 for name in SIGNAL_STATUSES})
        tally[status] = tally.get(status, 0) + count

    return AnalyticsWindow(
        window=WINDOW_LABEL,
        analyses_total=analyses_total,
        analyses_by_status=analyses_by_status,
        jobs_by_status=jobs_by_status,
        risk_distribution=risk_distribution,
        acquisition=acquisition,
        detectors=detectors,
    )
