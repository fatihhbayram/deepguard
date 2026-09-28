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

*It does not re-decide a risk level.* `Analysis.risk_level` is one column written in two
vocabularies, and which one a row holds is named by the `risk_rules_version` stored beside it.
So the two are grouped together and every row is placed by *both* (R13-T3): a `r9-v5.0.0`
verdict under `decisions`, a level written by one of the four pinned earlier rulesets under
`recorded_risk_levels`, and any pairing that is neither — a v5 row carrying `MEDIUM`, a legacy
row carrying a verdict, a ruleset this build has never heard of — under `unrecognised`, by its
own version and value. Nothing is translated across. A null level on a row with no ruleset, or
on a v5 row, is the absence of a persisted decision and is reported as `UNDECIDED` rather than
folded into `UNKNOWN`, which is a decision the risk engine reached and wrote down.

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
    FEEDBACK_ASSESSMENTS,
    FEEDBACK_CLAIMED_LABELS,
    Analysis,
    AnalysisJob,
    AnalysisSignal,
    MediaFile,
    UserFeedback,
)
from app.db.session import get_session
from app.risk_engine import (
    RISK_HIGH,
    RISK_MEDIUM,
    RISK_UNKNOWN,
    RULES_VERSION_V5,
    VERDICT_INCONCLUSIVE,
    VERDICT_MANIPULATION_DETECTED,
    VERDICT_NO_SIGNAL,
)
from app.risk_trace import RULESET_V1, RULESET_V2, RULESET_V3, RULESET_V4
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

# The two vocabularies `analyses.risk_level` is written in, each bound to the rulesets that
# write it (R13-T3).
#
# Imported from `app/risk_engine.py` and `app/risk_trace.py` rather than spelled out here,
# because these are the engine's own vocabulary and this route is only reporting it. A literal
# `"HIGH"` written here would be a second definition that goes on looking correct for as long as
# it takes somebody to rename a level — at which point this screen would report zero of the new
# name under a key for the old one, which is worse than failing.
#
# `r9-v5.0.0` writes the three verdicts. `RISK_UNDECIDED` sits in the same list and is defined in
# this module: it is not a verdict the engine can write, it is what this route calls the absence
# of any decision.
DECISION_VERDICTS = (
    VERDICT_MANIPULATION_DETECTED,
    VERDICT_NO_SIGNAL,
    VERDICT_INCONCLUSIVE,
)
DECISIONS = (*DECISION_VERDICTS, RISK_UNDECIDED)

# Every ruleset before v5 wrote `HIGH`, `MEDIUM` or `UNKNOWN`. The rulesets are pinned by
# exact version — the four `app/risk_trace.py` holds frozen tables for, which are the four this
# deployment has ever run — and *not* as "anything other than v5". A version outside this set
# is one no build of this application wrote, and a `HIGH` stamped with it is not a legacy
# level this screen can vouch for; it is carried under `unrecognised` instead.
LEGACY_RULES_VERSIONS = frozenset(
    ruleset.rules_version for ruleset in (RULESET_V1, RULESET_V2, RULESET_V3, RULESET_V4)
)
RECORDED_RISK_LEVELS = (RISK_HIGH, RISK_MEDIUM, RISK_UNKNOWN)

# How a null half of an unrecognised `(ruleset, value)` pair is spelled in that pair's key.
UNRECOGNISED_ABSENT = "(none)"

# The assessment `disagreement_rate` counts (R14-T2). One of `FEEDBACK_ASSESSMENTS`, which names
# no constant per value; the test suite holds this spelling to that tuple.
FEEDBACK_DISAGREE = "DISAGREE"


class AnalyticsWindow(BaseModel):
    """The seven-day picture, as one payload.

    Every value is a count of persisted rows. There is no rate, no ratio and no average
    anywhere in this shape, which is deliberate: a derived number is an interpretation, and the
    moment one appears here it becomes this route's interpretation rather than the reader's.
    `failed` and `completed` side by side say everything a failure rate would, and say it
    without deciding whether in-flight work belongs in the denominator.

    One exception, required by R14-T2: `disagreement_rate` in the feedback buckets. It is
    defined in `feedback_buckets` and computed here, so the browser never divides anything, and
    its denominator is fixed — the bucket's own feedback — so there is nothing left to decide.
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

    # The window's analyses by the risk level persisted on them, placed by the ruleset version
    # persisted beside it — see `risk_buckets`. Three maps rather than one, because the column
    # holds two vocabularies and a single map would sit a `MEDIUM` one row under a
    # `MANIPULATION_DETECTED` as though they answered the same question.
    #
    # `decisions`: `r9-v5.0.0` verdicts, plus `UNDECIDED` for analyses with no decision yet.
    decisions: dict[str, int]
    # `recorded_risk_levels`: `HIGH`/`MEDIUM`/`UNKNOWN` as written by a pinned earlier ruleset.
    recorded_risk_levels: dict[str, int]
    # `unrecognised`: every other pairing, keyed `"<ruleset>/<value>"` so neither half is lost.
    # Empty on a healthy deployment, and never seeded: a key here is always a real row.
    unrecognised: dict[str, int]

    # How the window's media arrived. Keyed by `MediaFile.acquisition_method`, with
    # `unrecorded` for rows predating the column.
    acquisition: dict[str, int]

    # Provider health: for each detector that persisted at least one signal in the window, how
    # many of each status it wrote. Providers appear because they have rows, never because a
    # list here says they should — a detector nobody invoked this week is absent, not zeroed,
    # and the difference is the difference between "asked and silent" and "not asked".
    detectors: dict[str, dict[str, int]]

    # User feedback (R14-T2): what owners said about the results they were shown. Unverified
    # opinion — not Ground Truth, not a Human Review, not an evaluation of any detector.
    #
    # Windowed on `UserFeedback.updated_at`, not on the analysis: this is the current state of
    # every piece of feedback created or changed in the last seven days, about an analysis of any
    # age. A revision overwrites the row, so this is not a count of submissions and there is no
    # event history behind it; a resubmission that changed nothing does not move `updated_at`
    # and does not bring a stale row back into the window.
    feedback_total: int
    feedback_by_assessment: dict[str, int]
    # The label users claimed, with no null key: feedback that claimed no label is counted in
    # `feedback_without_claimed_label` instead.
    feedback_by_claimed_label: dict[str, int]
    feedback_without_claimed_label: int
    # Feedback by the verdict on the analysis it is about, placed by `risk_bucket` — the same
    # three maps as `decisions`, `recorded_risk_levels` and `unrecognised` above. The first two
    # carry every known key even with no feedback (`total_feedback` 0, `disagreement_rate`
    # null); the third holds only pairings some feedback was about.
    feedback_by_decision: dict[str, dict[str, float | int | None]]
    feedback_by_recorded_risk_level: dict[str, dict[str, float | int | None]]
    feedback_by_unrecognised_risk_state: dict[str, dict[str, float | int | None]]


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


def unrecognised_key(rules_version: str | None, risk_level: str | None) -> str:
    """The name an unrecognised pairing is counted under: both halves, as stored.

    Both, because either half on its own would misfile the row in the reader's head — a bare
    `MEDIUM` here reads as a legacy level, and a bare `r9-v5.0.0` says nothing about what was
    wrong with it.
    """
    return (
        f"{UNRECOGNISED_ABSENT if rules_version is None else rules_version}"
        f"/{UNRECOGNISED_ABSENT if risk_level is None else risk_level}"
    )


BUCKET_DECISIONS = "decisions"
BUCKET_RECORDED = "recorded_risk_levels"
BUCKET_UNRECOGNISED = "unrecognised"


def risk_bucket(risk_level: str | None, rules_version: str | None) -> tuple[str, str]:
    """Which of the three maps one stored `(risk_level, risk_rules_version)` pair belongs in,
    and under which key: the single place that decision is made.

    `risk_buckets` places the week's analyses with it and `feedback_buckets` places the week's
    feedback with it (R14-T2), so a pairing cannot be a decision in one list and unrecognised
    in the other. It decides by exact match on the pair and on nothing else; the rules are
    listed on `risk_buckets`, which was their only caller before R14-T2.
    """
    if rules_version == RULES_VERSION_V5 and risk_level in DECISION_VERDICTS:
        return BUCKET_DECISIONS, risk_level
    if risk_level is None and rules_version in (None, RULES_VERSION_V5):
        return BUCKET_DECISIONS, RISK_UNDECIDED
    if rules_version in LEGACY_RULES_VERSIONS and risk_level in RECORDED_RISK_LEVELS:
        return BUCKET_RECORDED, risk_level
    return BUCKET_UNRECOGNISED, unrecognised_key(rules_version, risk_level)


def risk_buckets(
    rows: Sequence[Row],
) -> tuple[dict[str, int], dict[str, int], dict[str, int]]:
    """`(risk_level, risk_rules_version, count)` groups, placed by both columns together.

    The counting already happened in PostgreSQL; this only decides which of three maps each
    group belongs in, by `risk_bucket`, which decides by exact match on the pair and on nothing
    else:

    - a `r9-v5.0.0` row holding one of the three verdicts is a **decision**;
    - a row with no level, and either no ruleset or `r9-v5.0.0`, is **`UNDECIDED`** — no
      decision has been written. The four risk columns are written in one transaction, so the
      row an in-flight or failed analysis actually leaves behind has neither; a v5 version
      beside a null level is the same absence and is counted the same way;
    - a row from one of the pinned legacy rulesets holding `HIGH`, `MEDIUM` or `UNKNOWN` is a
      **recorded risk level**;
    - everything else is **unrecognised**, by its own ruleset and value. That includes a legacy
      ruleset beside a null level: no earlier ruleset defines what that absence means, so it
      is not given `UNDECIDED`'s meaning either. And it includes a legacy row holding a v5
      verdict, a v5 row holding a legacy level, and any ruleset this build did not write — each
      is data that contradicts its own stamp, and none is repaired into a list it does not
      belong to.

    Every known key is seeded at zero, for the reason `ANALYSIS_STATUSES` gives. `unrecognised`
    is not seeded: it has no vocabulary, and a key in it is always a count of real rows.
    """
    buckets: dict[str, dict[str, int]] = {
        BUCKET_DECISIONS: {name: 0 for name in DECISIONS},
        BUCKET_RECORDED: {level: 0 for level in RECORDED_RISK_LEVELS},
        BUCKET_UNRECOGNISED: {},
    }

    for risk_level, rules_version, count in rows:
        bucket, key = risk_bucket(risk_level, rules_version)
        buckets[bucket][key] = buckets[bucket].get(key, 0) + count

    return buckets[BUCKET_DECISIONS], buckets[BUCKET_RECORDED], buckets[BUCKET_UNRECOGNISED]


def feedback_tally() -> dict[str, float | int | None]:
    """One verdict bucket's feedback, before anything has been counted into it.

    Every assessment is present at zero and the rate is null, so a verdict nobody gave feedback
    on still has the whole shape — the same floor `ANALYSIS_STATUSES` is, applied one level down.
    """
    return {
        "total_feedback": 0,
        **{assessment: 0 for assessment in FEEDBACK_ASSESSMENTS},
        "disagreement_rate": None,
    }


def feedback_buckets(
    rows: Sequence[Row],
) -> tuple[
    dict[str, dict[str, float | int | None]],
    dict[str, dict[str, float | int | None]],
    dict[str, dict[str, float | int | None]],
]:
    """`(risk_level, risk_rules_version, assessment, count)` groups, placed by `risk_bucket`.

    Each piece of feedback lands in the bucket of the verdict persisted on the analysis it is
    about, placed exactly as the analysis itself is placed in `risk_buckets` — by the same
    function, not a copy of its rules. `decisions` and `recorded_risk_levels` are seeded with
    every known key; `unrecognised` only ever holds pairings some feedback was actually about.

    `disagreement_rate` is `DISAGREE / total_feedback` for that one bucket, and null when the
    bucket has no feedback: nothing was said, so there is no rate, and a zero would claim that
    everybody agreed. It is the share of users who pushed back on a verdict they were shown —
    an unverified opinion. It is not a false-positive rate or an error rate of any kind, and
    nothing here compares it with Ground Truth.
    """
    buckets: dict[str, dict[str, dict[str, float | int | None]]] = {
        BUCKET_DECISIONS: {name: feedback_tally() for name in DECISIONS},
        BUCKET_RECORDED: {level: feedback_tally() for level in RECORDED_RISK_LEVELS},
        BUCKET_UNRECOGNISED: {},
    }

    for risk_level, rules_version, assessment, count in rows:
        bucket, key = risk_bucket(risk_level, rules_version)
        tally = buckets[bucket].setdefault(key, feedback_tally())
        # `.get`, not `+=` on a seeded key: an assessment outside the vocabulary cannot pass the
        # table's check constraint, but if one ever did it is carried under its own name — the
        # rule `tallied` follows — and it is still in `total_feedback`.
        tally[assessment] = tally.get(assessment, 0) + count
        tally["total_feedback"] += count

    for bucket in buckets.values():
        for tally in bucket.values():
            total = tally["total_feedback"]
            tally["disagreement_rate"] = tally[FEEDBACK_DISAGREE] / total if total else None

    return buckets[BUCKET_DECISIONS], buckets[BUCKET_RECORDED], buckets[BUCKET_UNRECOGNISED]


def claimed_label_counts(rows: Sequence[Row]) -> tuple[dict[str, int], int]:
    """`(claimed_label, count)` groups, as the labels users claimed and how many claimed none.

    A null claimed label is counted as its own integer rather than under a key. A `"null"` key
    beside `GENUINE` and `FACE_SWAP` would read as a fifth thing a user could claim; "the user
    did not say" is a different kind of fact from any label.
    """
    labels = {label: 0 for label in FEEDBACK_CLAIMED_LABELS}
    without_label = 0

    for label, count in rows:
        if label is None:
            without_label += count
        else:
            labels[label] = labels.get(label, 0) + count

    return labels, without_label


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

    Six statements, each a `GROUP BY` or a `COUNT`, none of them returning a row of data — and
    since R14-T2 four more of the same kind over `user_feedback`. They
    are issued separately rather than assembled into one query with five joined subselects
    because they answer five unrelated questions over four tables, and a join between them
    would multiply rows against each other — an analysis with three signals would be counted
    three times in the risk buckets. Five cheap independent scans of a week's rows, read
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

    # Grouped on the level *and* the ruleset that wrote it, null groups and all, and named
    # afterwards rather than `COALESCE`d in SQL. A coalesce would merge a row that genuinely held
    # the literal text "UNDECIDED" into the null bucket and there would be no way to tell from
    # the result that it had happened; here such a row has no ruleset that writes that string
    # and lands in `unrecognised` under its own name.
    decisions, recorded_risk_levels, unrecognised = risk_buckets(
        session.execute(
            select(Analysis.risk_level, Analysis.risk_rules_version, func.count())
            .where(within_window(Analysis.created_at))
            .group_by(Analysis.risk_level, Analysis.risk_rules_version)
        ).all()
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

    # Provider and status grouped together, so one pass produces the whole table. Windowed on the signal's own `created_at` rather than the analysis's, because
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

    # User feedback, windowed on its own `updated_at` — see the comment on `feedback_total`. The
    # window predicate is the same `within_window`, applied to a different column on purpose.
    feedback_total = session.execute(
        select(func.count())
        .select_from(UserFeedback)
        .where(within_window(UserFeedback.updated_at))
    ).scalar_one()

    feedback_by_assessment = tallied(
        grouped_counts(session, UserFeedback.assessment, UserFeedback.updated_at),
        categories=FEEDBACK_ASSESSMENTS,
    )

    feedback_by_claimed_label, feedback_without_claimed_label = claimed_label_counts(
        grouped_counts(session, UserFeedback.claimed_label, UserFeedback.updated_at)
    )

    # Joined to the analysis for the verdict persisted on it, and grouped on both halves of that
    # verdict so `risk_bucket` can place it. An inner join is exact: `analysis_id` is `NOT NULL`
    # and cascades, so every feedback row has its analysis. The analysis is not windowed — only
    # the feedback is.
    feedback_by_decision, feedback_by_recorded_risk_level, feedback_by_unrecognised = (
        feedback_buckets(
            session.execute(
                select(
                    Analysis.risk_level,
                    Analysis.risk_rules_version,
                    UserFeedback.assessment,
                    func.count(),
                )
                .join(Analysis, Analysis.id == UserFeedback.analysis_id)
                .where(within_window(UserFeedback.updated_at))
                .group_by(
                    Analysis.risk_level, Analysis.risk_rules_version, UserFeedback.assessment
                )
            ).all()
        )
    )

    return AnalyticsWindow(
        window=WINDOW_LABEL,
        analyses_total=analyses_total,
        analyses_by_status=analyses_by_status,
        jobs_by_status=jobs_by_status,
        decisions=decisions,
        recorded_risk_levels=recorded_risk_levels,
        unrecognised=unrecognised,
        acquisition=acquisition,
        detectors=detectors,
        feedback_total=feedback_total,
        feedback_by_assessment=feedback_by_assessment,
        feedback_by_claimed_label=feedback_by_claimed_label,
        feedback_without_claimed_label=feedback_without_claimed_label,
        feedback_by_decision=feedback_by_decision,
        feedback_by_recorded_risk_level=feedback_by_recorded_risk_level,
        feedback_by_unrecognised_risk_state=feedback_by_unrecognised,
    )
