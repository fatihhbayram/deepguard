"""The human review an administrator attaches to a forensic result (R8-T7).

Mounted under `/api/v1/admin`, the sixth module on that prefix and the first that writes
anything about an analysis. What it manages is an *opinion* about a result, and the entire
design of this file is about keeping that opinion from ever being mistaken for the result.

**Nothing here writes to `analyses` or to `analysis_signals`.** Not a column, not a flag, not
a status. `risk_level`, `risk_rules_version`, `risk_calibration_id` and `risk_rule_id` are
written once by the worker under a named ruleset and a named calibration, and the signal rows
are what each detector actually answered; all of them are evidence, and evidence that an
operator can revise is not evidence. The review lives in its own table — `AnalysisReview`,
keyed by the analysis — so the isolation is structural: there is no statement in this module
that names a forensic column, and the only row it can update is the review.

That is why this is a separate table and not two more fields on `analyses`. With the fields on
the row, this endpoint would be issuing an UPDATE against the record it exists not to touch,
and the guarantee would rest on the `SET` clause listing the right two columns forever.

**The review has two axes and neither is forensic.** The status is workflow — has anybody
looked. The analyst assessment, added in R9-T7, is what they made of the automated assessment:
`AGREES_WITH_AUTOMATED_ASSESSMENT`, `DISAGREES_WITH_AUTOMATED_ASSESSMENT` or `UNDETERMINED`.
They are separate columns and separate diffs in the audit log because they answer separate
questions — a case can be closed by somebody who disagreed with it.

**An assessment is an opinion about the automated assessment, and cannot become a verdict.**
Disagreeing writes one nullable column on the review row. It does not replace `risk_level`, does
not produce a second classification, does not mark the analysis as correctly or incorrectly
decided, and is not a ground truth label — the system holds no ground truth, and an operator
pressing a button in a form does not create any. Nothing in the verdict, coverage or provenance
path reads this column. The spellings are long for exactly this reason: they cannot be read as
an answer about the media, only as a statement about an answer.

**The status vocabulary is operational and never forensic.** `REVIEWED` means somebody looked;
`NEEDS_FOLLOW_UP` means somebody looked and wants it looked at again. There is no `FAKE`, no
`GENUINE`, no `CONFIRMED` — those are answers about the media, the risk engine gives them under
a ruleset that can be named and replayed, and a second set of them written by hand here would
be an unversioned classification sitting beside the real one with nothing to say which the
report meant. `AnalysisReview` states this at the column.

**Unreviewed is the absence of a row.** There is no third stored value and no backfill: every
analysis that existed before this table did is already in the correct state, and the GET below
answers for one by reporting `UNREVIEWED` rather than 404. So "has anybody looked at this" is a
question this API can answer about the entire history of the deployment on the day it ships.

**A change is recorded in the same transaction as the change itself.** One `AdminAuditEvent`
per mutation, added to the session that holds the review, exactly as `admin_users.py` and
`admin_api_keys.py` do — and, like both of them, a request that changes nothing writes nothing
at all. The event never carries the note's text: `note_changed` says the wording moved, and the
wording itself stays on the review row, where one read of one screen is what exposes it.

**The note is plain text, and this module is where that becomes true.** It is bounded,
stripped, and refused if it carries control characters; it is never parsed, never rendered, and
never interpreted as markup anywhere. The screen above prints it as text — see
`app/admin/analyses/[id]/page.tsx` — which is what keeps a note containing `<script>` a note
containing `<script>` rather than one administrator's injection into another's browser.
"""

import logging
import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, ConfigDict, field_validator
from sqlalchemy import ColumnElement, and_, exists, false, func, or_, select
from sqlalchemy.orm import Session

from app.api.admin_analytics import (
    BUCKET_DECISIONS,
    BUCKET_RECORDED,
    DECISIONS,
    FEEDBACK_DISAGREE,
    risk_bucket,
)
from app.db.models import (
    ANALYST_ASSESSMENTS,
    AUDIT_ACTION_REVIEW_CREATED,
    AUDIT_ACTION_REVIEW_UPDATED,
    AUDIT_TARGET_ANALYSIS,
    FEEDBACK_ASSESSMENTS,
    MAX_REVIEW_NOTE_LENGTH,
    REVIEW_STATUS_REVIEWED,
    REVIEW_STATUS_UNREVIEWED,
    REVIEW_STATUSES,
    SIGNAL_STATUS_FAILED,
    SIGNAL_STATUS_TIMEOUT,
    AdminAuditEvent,
    Analysis,
    AnalysisReview,
    AnalysisSignal,
    MediaFile,
    User,
    UserFeedback,
)
from app.db.session import get_session
from app.observability import current_request_id
from app.web_auth import require_admin, require_same_origin

logger = logging.getLogger(__name__)

# What a note may not contain. Every C0 control character except the two that are ordinary
# text — a tab and a newline — plus DEL.
#
# The NUL byte is the one that has to be refused rather than merely discouraged: PostgreSQL
# cannot store it in a text column at all, so a note carrying one would surface as a 500 on a
# request whose only fault was a stray byte. The rest go with it because a note is plain text
# and a vertical tab in the middle of one is not text somebody typed on purpose — it is the
# residue of a paste from somewhere else, and it renders as nothing while still counting
# towards the length.
FORBIDDEN_NOTE_CHARACTERS = frozenset(
    chr(code) for code in list(range(0x00, 0x20)) + [0x7F]
) - {"\t", "\n"}

# Every route below is behind `require_admin`, declared once on the router rather than per
# route, so a route added to this file later cannot be an administrative route somebody forgot
# to guard. The same shape the five admin modules beside it use, and the same dependency — not
# a second role check written here, which would be a second answer to the only question that
# matters.
#
# There is no reviewer role. An administrator is the reviewer, which is the whole of the
# authorization model: inventing a role with exactly one member and exactly one permission
# would be a grant system built for a requirement nobody has stated.
#
# `require_admin` is layered on `require_user`, so an anonymous caller is refused with 401 and
# an authenticated non-administrator with 403 — and that applies to the GET as much as to the
# PUT. Who has looked at which case is itself operational information.
router = APIRouter(
    prefix="/api/v1/admin",
    tags=["admin"],
    dependencies=[Depends(require_admin)],
)


class AnalysisReviewState(BaseModel):
    """Where one analysis stands in the review workflow, reviewed or not.

    One response model for both routes and for both states, which is what lets the screen above
    render an unreviewed analysis and a reviewed one with the same code path instead of
    branching on a 404.

    The four nullable fields are null together: they are the review row's, and an unreviewed
    analysis has no review row. A null `reviewer_id` therefore means "nobody has reviewed this",
    never "somebody did and we lost who".

    Built field by field in the two helpers below rather than by `from_attributes`, keeping the
    convention its siblings set: a column added to `AnalysisReview` later reaches this payload
    when somebody decides it should, not by having been added.
    """

    model_config = ConfigDict(from_attributes=False)

    analysis_id: uuid.UUID

    # `REVIEWED`, `NEEDS_FOLLOW_UP`, or `UNREVIEWED` for an analysis with no review row. The
    # third is the only value here that is never stored anywhere — see `AnalysisReview`.
    status: str

    # What the reviewer made of the automated assessment, or null where they recorded nothing.
    # Null is the honest answer for every review written before R9-T7 and for any review since
    # whose author did not answer the question — it is not `UNDETERMINED`, which is a position
    # somebody took. Readers must render the two differently and must never treat either as a
    # statement about the media.
    analyst_assessment: str | None

    # The reviewer's own words, as text. Empty when there is no review and when the reviewer
    # wrote none; the two are told apart by `status`, not by this field.
    note: str

    # Who last wrote this review, and the address they had when they wrote it. The id is not a
    # foreign key — `AnalysisReview.reviewer_id` says why — so it may name an account that no
    # longer exists, and the snapshot beside it is what keeps the row readable when it does.
    reviewer_id: uuid.UUID | None
    reviewer_email_snapshot: str | None

    # When the review was first written and when it last actually moved. Null on an unreviewed
    # analysis. `updated_at` advances only on a change that stuck: opening the form and saving
    # it untouched is a no-op here and leaves this alone.
    created_at: datetime | None
    updated_at: datetime | None


class ReviewChange(BaseModel):
    """What a review request may say: a status, a note, and nothing else.

    `extra="forbid"` so a body naming a field this model does not have is refused rather than
    quietly ignored. That is the ordinary reason to forbid extras, and here it carries the
    weight of the whole feature: a request that tried to supply `risk_level`, `reviewer_id`, or
    `analysis_id` is a 422, not a request accepted with the field dropped. The caller is told
    the field does not exist instead of believing they set it — and the two that would matter
    most are a forensic column and the identity of the reviewer, neither of which any request
    body in this application may name.
    """

    model_config = ConfigDict(extra="forbid")

    status: str

    # Optional, and `None` is a value a caller may send rather than only an omission: clearing
    # an assessment somebody recorded by mistake is a legitimate revision, and the form above
    # offers it. It is not a fourth vocabulary word — see the validator.
    analyst_assessment: str | None = None

    # Defaulted, so "reviewed, nothing to add" is a body with one field rather than one that
    # has to spell out an empty string. The default is `""` and not `None` because the column
    # is not nullable and because two spellings of "no note" would make the no-op comparison
    # below answer wrongly for one of them.
    note: str = ""

    @field_validator("status")
    @classmethod
    def operational_status(cls, value: str) -> str:
        """One of the two workflow states, and nothing that reads as a verdict.

        Checked against `REVIEW_STATUSES` rather than against a list written here, so the
        model, the database's check constraint and the test that proves they agree all name one
        source. A value outside it is a 422.

        This is the validator that keeps the feature honest. `FAKE` and `GENUINE` are refused
        here not because they are misspelled but because they are answers about the media, and
        the only place in this system entitled to give one is the risk engine.
        """
        if value not in REVIEW_STATUSES:
            raise ValueError(f"status must be one of {', '.join(REVIEW_STATUSES)}")

        return value

    @field_validator("analyst_assessment")
    @classmethod
    def opinion_about_the_automated_assessment(cls, value: str | None) -> str | None:
        """One of the three assessments, or nothing at all.

        Checked against `ANALYST_ASSESSMENTS` for the reason the status validator gives: the
        model, the database's check constraint and the test that proves they agree name one
        source rather than three lists.

        `None` passes through untouched and is not coerced to `UNDETERMINED`. The two mean
        different things — nobody recorded an opinion, against somebody recorded that they
        could not form one — and the coercion would have quietly attributed the second to every
        reviewer who did the first.

        This is the validator that keeps the new axis from becoming a verdict. `TRUE`, `FALSE`,
        `CONFIRMED`, `TP` and `FP` are refused here not as misspellings but because they are
        answers about the media or claims about correctness, and neither is a thing a review may
        assert: the first belongs to the risk engine under a named ruleset, and the second to
        ground truth this system does not hold.
        """
        if value is None:
            return None

        if value not in ANALYST_ASSESSMENTS:
            raise ValueError(
                f"analyst_assessment must be one of {', '.join(ANALYST_ASSESSMENTS)}"
            )

        return value

    @field_validator("note")
    @classmethod
    def plain_text_note(cls, value: str) -> str:
        """Plain text, bounded, and normalized exactly once.

        Stripped first, which is what makes `"   "` an empty note rather than a stored note of
        three spaces — a review that would read as blank on the screen and be indistinguishable
        from a rendering fault. The stripped value is what is returned and therefore what is
        stored *and* what the no-op comparison sees, so there is one normalization rather than
        two that could disagree about whether anything changed.

        The length is checked after stripping and against the column's own bound, so a note
        this model accepted can always be written — a value refused by PostgreSQL rather than
        here would surface as a 500 on a request that was merely too long.

        Control characters are refused rather than stripped out. Silently rewriting somebody's
        note is worse than telling them it was not accepted: the difference matters when the
        note is a record of a judgement somebody will be held to.
        """
        note = value.strip()

        if len(note) > MAX_REVIEW_NOTE_LENGTH:
            raise ValueError(f"note must be at most {MAX_REVIEW_NOTE_LENGTH} characters")

        if any(character in FORBIDDEN_NOTE_CHARACTERS for character in note):
            raise ValueError("note must be plain text")

        return note


def unreviewed(analysis_id: uuid.UUID) -> AnalysisReviewState:
    """What an analysis nobody has reviewed looks like.

    Constructed rather than fetched, because there is nothing to fetch: `UNREVIEWED` is the
    absence of a row. It is built in one place so the shape of "not reviewed" is identical
    everywhere it is returned, and so the day a field is added to the payload there is one
    place that has to decide what it says when there is no review.
    """
    return AnalysisReviewState(
        analysis_id=analysis_id,
        status=REVIEW_STATUS_UNREVIEWED,
        analyst_assessment=None,
        note="",
        reviewer_id=None,
        reviewer_email_snapshot=None,
        created_at=None,
        updated_at=None,
    )


def visible_review(review: AnalysisReview) -> AnalysisReviewState:
    """The stored review as a response.

    The one place an `AnalysisReview` becomes a payload in this module, so there is one place
    to read to know what leaves it. The field list is written out rather than spread off the
    ORM object, which is the convention `visible_key` and `visible_entry` set.
    """
    return AnalysisReviewState(
        analysis_id=review.analysis_id,
        status=review.status,
        analyst_assessment=review.analyst_assessment,
        note=review.note,
        reviewer_id=review.reviewer_id,
        reviewer_email_snapshot=review.reviewer_email_snapshot,
        created_at=review.created_at,
        updated_at=review.updated_at,
    )


def analysis_or_404(session: Session, analysis_id: uuid.UUID) -> Analysis:
    """The analysis, locked for the duration of this request, or a 404.

    **`FOR UPDATE` on the analysis, and this is the one statement in the module that touches
    that table — a lock, never a write.** It does two things at once. It establishes that the
    analysis exists, which is what makes a review of a non-existent id a 404 instead of an
    orphan insert that the foreign key would reject as a 500. And it serializes the two
    administrators who press save on the same analysis at the same moment: without it both
    would read "no review", both would insert, and the second would fail on the primary key —
    a duplicate-key 500 for a request that was merely second. With it, the second waits and
    reads what the first actually committed, which puts it on the update path where it belongs.

    Locking the forensic row does not compromise its immutability: a lock writes nothing, and
    the transaction that holds it issues no statement naming a forensic column. What it costs
    is that a worker committing a risk decision for this exact analysis would wait behind the
    review, for the sub-millisecond the review's transaction lives.

    404 for an id that names nothing. Not the concealing 404 the public API gives an
    unreachable analysis, and for the reason `update_user` gives: there is nothing to hide from
    this caller, who is entitled to read every analysis in the system already.
    """
    analysis = session.execute(
        select(Analysis).where(Analysis.id == analysis_id).with_for_update()
    ).scalar_one_or_none()

    if analysis is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="analysis not found"
        )

    return analysis


def stored_review(session: Session, analysis_id: uuid.UUID) -> AnalysisReview | None:
    """The review for one analysis, or None where nobody has written one."""
    return session.execute(
        select(AnalysisReview).where(AnalysisReview.analysis_id == analysis_id)
    ).scalar_one_or_none()


def review_changes(
    previous_status: str,
    new_status: str,
    previous_assessment: str | None,
    new_assessment: str | None,
    note_moved: bool,
) -> dict:
    """What the audit row says moved, without ever saying what the note now says.

    Two deliberate departures from the payload `admin_users.py` writes, and they pull in
    opposite directions:

    `status` appears only when it moved, which is the house convention — an event restating an
    untouched field makes every row look like a change to everything. On a first review the
    old value is `UNREVIEWED`, the state the analysis was in by having no row at all, so the
    diff reads as a real transition rather than as a null.

    `analyst_assessment` is a second, separate diff under the same rule, and separate is the
    requirement rather than the tidy option: the two axes answer different questions, and a
    combined entry would make "closed the case" and "disagreed with the detector" indistinguish-
    able in the one record that is supposed to tell them apart. Its old value is a JSON null on
    a review that carried none, which is distinct from `"UNDETERMINED"` appearing there.

    Neither key is a forensic diff. This event describes a change to `analysis_reviews`, and
    `target_id` names the analysis only because that is what the review is about — no row on
    `analyses` moved, and reading this payload as a history of the verdict would be reading it
    as the opposite of what it records.

    `note_changed` appears either way, which is not. It is there precisely *because* the note's
    text is withheld: with the wording absent from the event, a reader who saw no `note` key
    could not tell "the wording is unchanged" from "this log does not record wording". A
    `false` here is a statement, not a restatement — the status moved and the words did not —
    and it is the only thing the audit log will ever say about the contents of a note.

    The text stays on the review row alone. An audit table is read whole by one screen and
    grows without bound; copying a thousand characters of prose into every revision would
    reprint the same paragraph across a dozen rows and bury the field that actually moved.
    """
    changes: dict = {"note_changed": note_moved}

    if previous_status != new_status:
        changes["status"] = {"old": previous_status, "new": new_status}

    if previous_assessment != new_assessment:
        changes["analyst_assessment"] = {
            "old": previous_assessment,
            "new": new_assessment,
        }

    return changes


# --- the review queue (R14-T3) ------------------------------------------------------------
#
# Which analyses an administrator might want to open next, filtered on operational facts that
# already exist: what the owner said in feedback, what the engine decided, whether a detector
# failed, and whether anybody has reviewed it. **Being in the queue says nothing about the
# media.** There is no score, no priority and no new state here — the order is chronological,
# and every flag on an item states which stored fact it matched, never what that fact means.

# How many items one page carries by default, and the most a caller may ask for.
REVIEW_QUEUE_DEFAULT_LIMIT = 50
REVIEW_QUEUE_MAX_LIMIT = 200

# The two signal states that mean a detector did not answer. `SUCCESS` is the third and the
# only other one the column holds.
SIGNAL_ERROR_STATUSES = (SIGNAL_STATUS_FAILED, SIGNAL_STATUS_TIMEOUT)


class ReviewQueueItem(BaseModel):
    """One analysis in the queue, with the facts it was filtered on.

    **No single feedback record.** An analysis can carry feedback from several accounts, and
    picking one of them to show would be picking arbitrarily; the item carries a count per
    assessment instead, and `has_disagree_feedback` is read off that count.

    **The verdict is placed by `risk_bucket`**, the classifier R13-T3 established and R14-T2
    extracted, so exactly one of `decision`, `recorded_risk_level` and `unrecognised_risk_state`
    is set, and a legacy `HIGH` can never be read as a v5 verdict or the other way round.
    """

    model_config = ConfigDict(from_attributes=False)

    analysis_id: uuid.UUID
    created_at: datetime
    status: str

    # Every media hash on the analysis, sorted. A list, because `media_files.analysis_id` is not
    # unique in the schema and choosing one row would be choosing arbitrarily. Empty for an
    # analysis with no media row.
    media_sha256s: list[str]

    risk_rules_version: str | None
    decision: str | None
    recorded_risk_level: str | None
    unrecognised_risk_state: str | None

    # `UNREVIEWED` where there is no review row, as `get_analysis_review` answers.
    review_status: str
    analyst_assessment: str | None

    # Every assessment present, at zero where nobody gave it.
    feedback_counts: dict[str, int]
    has_disagree_feedback: bool

    # `FAILED` and `TIMEOUT` signals, each present, at zero where there were none.
    signal_error_counts: dict[str, int]
    has_signal_errors: bool


class ReviewQueuePage(BaseModel):
    """One page of the queue and the size of the whole filtered set it came from."""

    items: list[ReviewQueueItem]
    total: int
    limit: int
    offset: int


def pairs_for_decision(session: Session, decision: str) -> list[tuple[str | None, str | None]]:
    """Every stored `(risk_level, risk_rules_version)` pair that `risk_bucket` places under
    this decision.

    The pairs actually present in `analyses` are grouped by PostgreSQL — a handful of rows —
    and each is classified here by `risk_bucket` itself. The SQL filter built from the result
    only matches pairs exactly; it does not restate the taxonomy, so it cannot disagree with the
    operational summary about what a decision is.
    """
    stored = session.execute(
        select(Analysis.risk_level, Analysis.risk_rules_version).group_by(
            Analysis.risk_level, Analysis.risk_rules_version
        )
    ).all()

    return [
        (risk_level, rules_version)
        for risk_level, rules_version in stored
        if risk_bucket(risk_level, rules_version) == (BUCKET_DECISIONS, decision)
    ]


def review_queue_filters(
    pairs: list[tuple[str | None, str | None]] | None,
    feedback_assessment: str | None,
    signal_error: bool,
    unreviewed_only: bool,
) -> list[ColumnElement[bool]]:
    """The WHERE clauses over `analyses`, one per filter the caller set.

    **Every relationship is an `EXISTS`, never a join.** Feedback, signals and reviews are each
    one-to-many (or one-to-one) against the analysis, and a join would multiply the analysis row
    by its children; an `EXISTS` keeps the statement over `analyses` alone, so the count and the
    page are counted and cut on one row per analysis without a `DISTINCT` to undo anything.

    `pairs` is `None` when no decision filter was asked for, and an empty list when one was and
    no stored pair matches — which filters everything out rather than nothing.
    """
    clauses: list[ColumnElement[bool]] = []

    if pairs is not None:
        clauses.append(
            or_(
                *(
                    and_(
                        Analysis.risk_level.is_not_distinct_from(risk_level),
                        Analysis.risk_rules_version.is_not_distinct_from(rules_version),
                    )
                    for risk_level, rules_version in pairs
                )
            )
            if pairs
            else false()
        )

    if feedback_assessment is not None:
        clauses.append(
            exists().where(
                UserFeedback.analysis_id == Analysis.id,
                UserFeedback.assessment == feedback_assessment,
            )
        )

    if signal_error:
        clauses.append(
            exists().where(
                AnalysisSignal.analysis_id == Analysis.id,
                AnalysisSignal.status.in_(SIGNAL_ERROR_STATUSES),
            )
        )

    # `NEEDS_FOLLOW_UP` stays in the queue: somebody looked and asked for another look, so the
    # case is still open. Only `REVIEWED` takes an analysis out.
    if unreviewed_only:
        clauses.append(
            ~exists().where(
                AnalysisReview.analysis_id == Analysis.id,
                AnalysisReview.status == REVIEW_STATUS_REVIEWED,
            )
        )

    return clauses


def review_queue_items(session: Session, analyses: list) -> list[ReviewQueueItem]:
    """The page's analyses, with their feedback, signal errors, review and media attached.

    One grouped statement per relationship, each restricted to the ids on this page, so the
    number of statements is fixed whatever the page size and no child row ever multiplies an
    analysis.
    """
    ids = [analysis.id for analysis in analyses]

    feedback: dict[uuid.UUID, dict[str, int]] = {
        analysis_id: {name: 0 for name in FEEDBACK_ASSESSMENTS} for analysis_id in ids
    }
    for analysis_id, assessment, count in session.execute(
        select(UserFeedback.analysis_id, UserFeedback.assessment, func.count())
        .where(UserFeedback.analysis_id.in_(ids))
        .group_by(UserFeedback.analysis_id, UserFeedback.assessment)
    ):
        feedback[analysis_id][assessment] = count

    signal_errors: dict[uuid.UUID, dict[str, int]] = {
        analysis_id: {name: 0 for name in SIGNAL_ERROR_STATUSES} for analysis_id in ids
    }
    for analysis_id, signal_status, count in session.execute(
        select(AnalysisSignal.analysis_id, AnalysisSignal.status, func.count())
        .where(
            AnalysisSignal.analysis_id.in_(ids),
            AnalysisSignal.status.in_(SIGNAL_ERROR_STATUSES),
        )
        .group_by(AnalysisSignal.analysis_id, AnalysisSignal.status)
    ):
        signal_errors[analysis_id][signal_status] = count

    reviews = {
        review.analysis_id: review
        for review in session.execute(
            select(AnalysisReview).where(AnalysisReview.analysis_id.in_(ids))
        ).scalars()
    }

    media: dict[uuid.UUID, list[str]] = {analysis_id: [] for analysis_id in ids}
    for analysis_id, sha256 in session.execute(
        select(MediaFile.analysis_id, MediaFile.original_sha256)
        .where(MediaFile.analysis_id.in_(ids))
        .order_by(MediaFile.original_sha256)
    ):
        media[analysis_id].append(sha256)

    items = []
    for analysis in analyses:
        bucket, key = risk_bucket(analysis.risk_level, analysis.risk_rules_version)
        review = reviews.get(analysis.id)
        feedback_counts = feedback[analysis.id]
        error_counts = signal_errors[analysis.id]

        items.append(
            ReviewQueueItem(
                analysis_id=analysis.id,
                created_at=analysis.created_at,
                status=analysis.status,
                media_sha256s=media[analysis.id],
                risk_rules_version=analysis.risk_rules_version,
                decision=key if bucket == BUCKET_DECISIONS else None,
                recorded_risk_level=key if bucket == BUCKET_RECORDED else None,
                unrecognised_risk_state=(
                    key if bucket not in (BUCKET_DECISIONS, BUCKET_RECORDED) else None
                ),
                review_status=REVIEW_STATUS_UNREVIEWED if review is None else review.status,
                analyst_assessment=None if review is None else review.analyst_assessment,
                feedback_counts=feedback_counts,
                has_disagree_feedback=feedback_counts[FEEDBACK_DISAGREE] > 0,
                signal_error_counts=error_counts,
                has_signal_errors=sum(error_counts.values()) > 0,
            )
        )

    return items


# Declared before every `/analyses/{analysis_id}/...` route in this file, so the static path is
# matched first whatever the dynamic routes on this prefix become.
@router.get("/analyses/queue", response_model=ReviewQueuePage)
def get_review_queue(
    feedback_assessment: str | None = None,
    decision: str | None = None,
    signal_error: bool = False,
    unreviewed_only: bool = True,
    limit: int = Query(REVIEW_QUEUE_DEFAULT_LIMIT, ge=1, le=REVIEW_QUEUE_MAX_LIMIT),
    offset: int = Query(0, ge=0),
    session: Session = Depends(get_session),
) -> ReviewQueuePage:
    """The analyses matching every filter set, newest first, one page at a time.

    Read-only: nothing here writes, and no statement names a forensic column except to read it.

    `decision` is one of the v5 verdicts or `UNDECIDED`, exactly the keys the operational
    summary's `decisions` map carries, and it matches the same rows that map counts under that
    key. `feedback_assessment` is one of the feedback vocabulary. Anything else is a 422 rather
    than a filter that silently matches nothing.

    `total` is counted by the same WHERE clauses the page is cut with, over `analyses` alone.
    """
    if decision is not None and decision not in DECISIONS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"decision must be one of {', '.join(DECISIONS)}",
        )

    if feedback_assessment is not None and feedback_assessment not in FEEDBACK_ASSESSMENTS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"feedback_assessment must be one of {', '.join(FEEDBACK_ASSESSMENTS)}",
        )

    clauses = review_queue_filters(
        pairs_for_decision(session, decision) if decision is not None else None,
        feedback_assessment,
        signal_error,
        unreviewed_only,
    )

    total = session.execute(
        select(func.count()).select_from(Analysis).where(*clauses)
    ).scalar_one()

    # The id breaks ties: two analyses committed in one transaction share `created_at`, and
    # without it a row could appear on two pages or on none.
    analyses = list(
        session.execute(
            select(Analysis)
            .where(*clauses)
            .order_by(Analysis.created_at.desc(), Analysis.id.desc())
            .limit(limit)
            .offset(offset)
        ).scalars()
    )

    return ReviewQueuePage(
        items=review_queue_items(session, analyses),
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get(
    "/analyses/{analysis_id}/review",
    response_model=AnalysisReviewState,
)
def get_analysis_review(
    analysis_id: uuid.UUID,
    session: Session = Depends(get_session),
) -> AnalysisReviewState:
    """Where this analysis stands in the review workflow.

    **An analysis nobody has reviewed is a 200, not a 404.** The absence of a review row is a
    real state with a name, and answering 404 would make "this analysis has never been looked
    at" indistinguishable from "this analysis does not exist" — the first is the ordinary case
    for the entire history of the deployment, and the second is a bad id. Only an id that names
    no analysis is a 404 here.

    No lock and no `FOR UPDATE`: this is a read, and a read that took a row lock would make
    opening the screen contend with the worker writing a risk decision.

    The reviewer's address is the snapshot frozen on the review row, never a join to `users`.
    Resolving it live would mean renaming an account rewrote who a past review was signed by,
    and `AnalysisReview.reviewer_email_snapshot` carries the full reasoning. An address here
    answers "who was this, then".
    """
    analysis = session.execute(
        select(Analysis.id).where(Analysis.id == analysis_id)
    ).scalar_one_or_none()

    if analysis is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="analysis not found"
        )

    review = stored_review(session, analysis_id)

    return unreviewed(analysis_id) if review is None else visible_review(review)


@router.put(
    "/analyses/{analysis_id}/review",
    response_model=AnalysisReviewState,
    # A state change, so it carries the same origin check every other dashboard mutation does.
    # The session cookie is `SameSite=Lax` and would not be attached to a cross-site request in
    # the first place; this is the independent check behind that, made by the server that owns
    # the data.
    dependencies=[Depends(require_same_origin)],
)
def set_analysis_review(
    analysis_id: uuid.UUID,
    change: ReviewChange,
    administrator: User = Depends(require_admin),
    session: Session = Depends(get_session),
) -> AnalysisReviewState:
    """Write or revise the human review of one analysis.

    A PUT and not a PATCH, because the body is the whole review: every field is a complete
    statement of where the reviewer now stands rather than an edit to one part of it — which is
    what makes an absent `analyst_assessment` mean "none recorded" rather than "leave whatever
    is there". That also makes it idempotent, which is what the no-op below turns from a
    property of the verb into a property of the database.

    **The analysis is untouched.** The only rows this function writes are the review and the
    audit event. `analysis_or_404` reads the analysis and locks it, and no statement here
    assigns to a forensic column — there is no attribute assignment against `analysis` anywhere
    below, which is what makes the immutability structural rather than careful. That holds for
    the assessment exactly as it does for the status: an analyst who disagrees writes one
    nullable column on `analysis_reviews`, and `risk_level` still says what the engine decided.

    **A request that changes nothing writes nothing.** Not the review, and above all not an
    audit row. The screen submits both controls on every save, so "open the form and press
    save" is the ordinary case rather than a strange one, and a log in which a dozen entries
    say a review was updated to what it already said is a log that cannot answer when the
    review actually changed. It returns 200 with the review as it stands: the caller asked for
    a state and that is the state.

    The comparison is against the stripped, validated note — one normalization, applied by the
    model — so a save that differs only in trailing whitespace is correctly a no-op rather than
    a revision that changed nothing visible.

    **The reviewer is the session, never the body.** `administrator` comes from
    `require_admin`, and `ReviewChange` has no field a caller could put an identity in; the
    address is copied in as it reads right now and frozen on the row. A later revision by
    somebody else overwrites both, which is why `admin_audit_events` — not this row — is what
    holds the history of who said what.

    The event is added to the session that holds the review so that one `commit()` writes both
    or neither. An audit row written after the commit, or by a listener, is a row that survives
    a rolled-back change and is missing for one that stuck — which is the only time anybody
    reads it.
    """
    analysis_or_404(session, analysis_id)

    review = stored_review(session, analysis_id)

    # The state the analysis is in right now, which on a first review is the one it has by
    # having no row: `UNREVIEWED`. Captured before anything is assigned, because after the
    # mutation there is nothing left to compare against.
    previous_status = REVIEW_STATUS_UNREVIEWED if review is None else review.status
    previous_assessment = None if review is None else review.analyst_assessment
    previous_note = "" if review is None else review.note

    status_moved = previous_status != change.status
    assessment_moved = previous_assessment != change.analyst_assessment
    note_moved = previous_note != change.note

    if not status_moved and not assessment_moved and not note_moved:
        # Nothing to write. Returns before the mutation rather than assigning the same values
        # over themselves and relying on the diff coming out empty — the emptiness test would
        # work today and would put the guarantee in a comparison instead of in the control
        # flow, one edit away from somebody adding a field that always differs.
        #
        # `review` cannot be None here: an unreviewed analysis has status `UNREVIEWED`, which
        # `ReviewChange` refuses, so no request can match every arm of the no-op against it.
        return visible_review(review) if review is not None else unreviewed(analysis_id)

    action = (
        AUDIT_ACTION_REVIEW_UPDATED if review is not None else AUDIT_ACTION_REVIEW_CREATED
    )

    if review is None:
        review = AnalysisReview(analysis_id=analysis_id)
        session.add(review)

    review.status = change.status
    # Assigned on every write, including when it is `None`. A PUT is a complete statement of
    # where the reviewer now stands, so clearing an assessment is expressed by sending none —
    # and skipping the assignment for `None` would have made the field the one part of this
    # body that could only ever be set and never unset.
    review.analyst_assessment = change.analyst_assessment
    review.note = change.note
    review.reviewer_id = administrator.id
    review.reviewer_email_snapshot = administrator.email

    changes = review_changes(
        previous_status,
        change.status,
        previous_assessment,
        change.analyst_assessment,
        note_moved,
    )

    # The record of the change, added to the session that holds the change itself.
    #
    # `target_email_snapshot` is null and not left out: an analysis is not a person and has no
    # address, and the column means "the address of the account this event was about" rather
    # than "an address". Filling it with the reviewer's would make the target of the event read
    # as the administrator who wrote it, which is what `actor_email_snapshot` already says.
    session.add(
        AdminAuditEvent(
            actor_id=administrator.id,
            actor_email_snapshot=administrator.email,
            action=action,
            target_type=AUDIT_TARGET_ANALYSIS,
            target_id=str(analysis_id),
            target_email_snapshot=None,
            changes=changes,
            request_id=current_request_id(),
        )
    )

    session.commit()
    session.refresh(review)

    # The administrator, the analysis, the status and the assessment — and deliberately not the
    # note. The note is the one piece of this that is free prose written by a person, and a log
    # line is read by more eyes and kept in more places than the screen it came from.
    logger.info(
        "Administrator %s set the review of analysis %s to %s (analyst assessment: %s).",
        administrator.id,
        analysis_id,
        review.status,
        review.analyst_assessment,
    )

    return visible_review(review)
