"""What the owner of an analysis says about its result, kept apart from every truth (R13-T1).

Two routers. The owner's, under `/api/v1`, writes and reads *their own* feedback on *their own*
analysis. The administrator's, under `/api/v1/admin`, reads every feedback row on one analysis
and writes nothing.

**Owner only, admin included.** Writing and reading here requires `Analysis.owner_id` to equal
the session's user. That is narrower than `visible_to`, which lets an administrator reach every
analysis: feedback is what the person who submitted the media says, and an administrator
writing "user feedback" on somebody else's analysis would be a Human Review under another name.
An analysis nobody owns — one submitted through an API key, or before there were accounts — has
no owner to give feedback, so it is out of reach here. Every refusal is the same 404
`visible_to`'s routes give, so this route confirms no id the read routes would not.

**Nothing here writes to `analyses`, `analysis_signals`, `analysis_reviews` or `ground_truth`.**
The only statement that writes is the upsert onto `user_feedback`. A `claimed_label` of
`GENUINE` is what somebody says; it is never Ground Truth, it never reaches the evaluator, and
it never moves a verdict.

**No audit event.** `admin_audit_events` records what administrators did, and there is no
end-user action log. The feedback row's `created_at` and `updated_at` are the whole record; an
immutable history of feedback is not provided by R13-T1.
"""

import logging
import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, ConfigDict, field_validator
from sqlalchemy import func, or_, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from app.api.admin_analyses import FORBIDDEN_NOTE_CHARACTERS
from app.db.models import (
    FEEDBACK_ASSESSMENTS,
    FEEDBACK_CLAIMED_LABELS,
    FEEDBACK_ONE_PER_USER_CONSTRAINT,
    MAX_FEEDBACK_NOTES_LENGTH,
    Analysis,
    User,
    UserFeedback,
)
from app.db.session import get_session
from app.web_auth import require_admin, require_same_origin, require_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["feedback"])

# Read-only, and admin-only for every route on it.
admin_router = APIRouter(
    prefix="/api/v1/admin",
    tags=["admin"],
    dependencies=[Depends(require_admin)],
)


class FeedbackChange(BaseModel):
    """The complete feedback state the caller is submitting — not a partial edit.

    Every submission replaces the stored state as a whole: an omitted or null `claimed_label` or
    `notes` means "none", and clears a value stored earlier. There is no PATCH semantics and no
    "leave unchanged". The answer is 200 whether the row was created, updated or left as it was.

    `extra="forbid"`: a body naming `risk_level`, `user_id` or `analysis_id` is a 422, not a field
    quietly dropped."""

    model_config = ConfigDict(extra="forbid")

    assessment: str
    claimed_label: str | None = None
    notes: str | None = None

    @field_validator("assessment")
    @classmethod
    def known_assessment(cls, value: str) -> str:
        if value not in FEEDBACK_ASSESSMENTS:
            raise ValueError(f"assessment must be one of {', '.join(FEEDBACK_ASSESSMENTS)}")
        return value

    @field_validator("claimed_label")
    @classmethod
    def known_claimed_label(cls, value: str | None) -> str | None:
        if value is not None and value not in FEEDBACK_CLAIMED_LABELS:
            raise ValueError(
                f"claimed_label must be one of {', '.join(FEEDBACK_CLAIMED_LABELS)}"
            )
        return value

    @field_validator("notes")
    @classmethod
    def plain_text_notes(cls, value: str | None) -> str | None:
        """Stripped, bounded and plain text, with empty stored as null — the review note's rules,
        so one normalization feeds both the write and the no-op comparison."""
        if value is None:
            return None

        notes = value.strip()

        if len(notes) > MAX_FEEDBACK_NOTES_LENGTH:
            raise ValueError(f"notes must be at most {MAX_FEEDBACK_NOTES_LENGTH} characters")

        if any(character in FORBIDDEN_NOTE_CHARACTERS for character in notes):
            raise ValueError("notes must be plain text")

        return notes or None


class FeedbackState(BaseModel):
    """The caller's own feedback on one analysis. With nothing submitted, `assessment` and both
    timestamps are null."""

    analysis_id: uuid.UUID
    assessment: str | None
    claimed_label: str | None
    notes: str | None
    created_at: datetime | None
    updated_at: datetime | None


class FeedbackEntry(BaseModel):
    """One user's feedback as an administrator reads it."""

    user_id: uuid.UUID
    user_email: str
    assessment: str
    claimed_label: str | None
    notes: str | None
    created_at: datetime
    updated_at: datetime


def owned_analysis_or_404(session: Session, analysis_id: uuid.UUID, user: User) -> None:
    """The analysis exists and this user owns it, or a 404.

    `owner_id = user.id` with no role exemption. A null owner can equal no id, so an API-key or
    legacy analysis is refused by the same comparison.
    """
    owned = session.execute(
        select(Analysis.id).where(Analysis.id == analysis_id, Analysis.owner_id == user.id)
    ).first()

    if owned is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="analysis not found")


def stored_feedback(
    session: Session, analysis_id: uuid.UUID, user: User
) -> UserFeedback | None:
    return session.execute(
        select(UserFeedback).where(
            UserFeedback.analysis_id == analysis_id, UserFeedback.user_id == user.id
        )
    ).scalar_one_or_none()


def visible_feedback(analysis_id: uuid.UUID, feedback: UserFeedback | None) -> FeedbackState:
    if feedback is None:
        return FeedbackState(
            analysis_id=analysis_id,
            assessment=None,
            claimed_label=None,
            notes=None,
            created_at=None,
            updated_at=None,
        )

    return FeedbackState(
        analysis_id=feedback.analysis_id,
        assessment=feedback.assessment,
        claimed_label=feedback.claimed_label,
        notes=feedback.notes,
        created_at=feedback.created_at,
        updated_at=feedback.updated_at,
    )


@router.get("/analyses/{analysis_id}/feedback", response_model=FeedbackState)
def get_feedback(
    analysis_id: uuid.UUID,
    session: Session = Depends(get_session),
    user: User = Depends(require_user),
) -> FeedbackState:
    """The caller's own feedback on an analysis they own."""
    owned_analysis_or_404(session, analysis_id, user)

    return visible_feedback(analysis_id, stored_feedback(session, analysis_id, user))


@router.post(
    "/analyses/{analysis_id}/feedback",
    response_model=FeedbackState,
    # 200 for create, update and no-op alike: an upsert of the complete state, not a creation.
    status_code=status.HTTP_200_OK,
    dependencies=[Depends(require_same_origin)],
)
def submit_feedback(
    analysis_id: uuid.UUID,
    change: FeedbackChange,
    session: Session = Depends(get_session),
    user: User = Depends(require_user),
) -> FeedbackState:
    """Create or revise the caller's feedback on an analysis they own.

    **One statement, resolved by the database.** `INSERT ... ON CONFLICT (user_id, analysis_id)
    DO UPDATE ... WHERE <any field IS DISTINCT FROM the stored one>`. Two concurrent submissions
    cannot both insert — the unique constraint turns the second into the update arm — and an
    identical resubmission matches the conflict but not the `WHERE`, so it writes nothing and
    `updated_at` stays where it was. Either way the answer is 200 with the feedback as it stands.

    No lock on the analysis: this transaction reads `analyses` once and writes only
    `user_feedback`, so it never waits on, or holds up, a worker writing a verdict.
    """
    owned_analysis_or_404(session, analysis_id, user)

    submitted = insert(UserFeedback).values(
        id=uuid.uuid4(),
        analysis_id=analysis_id,
        user_id=user.id,
        assessment=change.assessment,
        claimed_label=change.claimed_label,
        notes=change.notes,
    )
    session.execute(
        submitted.on_conflict_do_update(
            constraint=FEEDBACK_ONE_PER_USER_CONSTRAINT,
            set_={
                "assessment": submitted.excluded.assessment,
                "claimed_label": submitted.excluded.claimed_label,
                "notes": submitted.excluded.notes,
                "updated_at": func.now(),
            },
            where=or_(
                UserFeedback.assessment.is_distinct_from(submitted.excluded.assessment),
                UserFeedback.claimed_label.is_distinct_from(submitted.excluded.claimed_label),
                UserFeedback.notes.is_distinct_from(submitted.excluded.notes),
            ),
        )
    )
    session.commit()

    feedback = stored_feedback(session, analysis_id, user)

    # Never the notes: free prose stays on the row.
    logger.info(
        "User %s submitted feedback on analysis %s (assessment: %s, claimed label: %s).",
        user.id,
        analysis_id,
        feedback.assessment,
        feedback.claimed_label,
    )

    return visible_feedback(analysis_id, feedback)


@admin_router.get(
    "/analyses/{analysis_id}/feedback", response_model=list[FeedbackEntry]
)
def list_feedback(
    analysis_id: uuid.UUID,
    session: Session = Depends(get_session),
) -> list[FeedbackEntry]:
    """Every user's feedback on one analysis, oldest first. 404 only for an id naming nothing.

    The address is read live from `users`: `user_id` is a `RESTRICT` foreign key, so the account
    always exists while its feedback does.
    """
    exists = session.execute(select(Analysis.id).where(Analysis.id == analysis_id)).first()
    if exists is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="analysis not found")

    rows = session.execute(
        select(UserFeedback, User.email)
        .join(User, User.id == UserFeedback.user_id)
        .where(UserFeedback.analysis_id == analysis_id)
        .order_by(UserFeedback.created_at, UserFeedback.id)
    ).all()

    return [
        FeedbackEntry(
            user_id=feedback.user_id,
            user_email=email,
            assessment=feedback.assessment,
            claimed_label=feedback.claimed_label,
            notes=feedback.notes,
            created_at=feedback.created_at,
            updated_at=feedback.updated_at,
        )
        for feedback, email in rows
    ]
