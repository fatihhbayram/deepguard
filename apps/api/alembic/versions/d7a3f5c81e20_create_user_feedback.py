"""create user feedback

Revision ID: d7a3f5c81e20
Revises: c5e2a7d49f18
Create Date: 2026-09-26 22:00:00.000000

Creates `user_feedback` (R13-T1): what the owner of an analysis said about its result — an
assessment (`AGREE`, `DISAGREE`, `UNSURE`), an optional claimed label and optional notes.

A claim, never Ground Truth and never a Human Review. The table references `analyses` and
`users` only; it has no reference to `ground_truth`, `analysis_reviews` or `analysis_signals`.

`UNIQUE(user_id, analysis_id)`: one row per person per analysis, which the endpoint's upsert
resolves on. `analysis_id` cascades as every analysis child does; `user_id` is `RESTRICT` as
`analyses.owner_id` is.

**No existing table is touched** and nothing is backfilled.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = 'd7a3f5c81e20'
down_revision: Union[str, Sequence[str], None] = 'c5e2a7d49f18'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "user_feedback",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column(
            "analysis_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("analyses.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "user_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column("assessment", sa.String(length=16), nullable=False),
        sa.Column("claimed_label", sa.String(length=32), nullable=True),
        sa.Column("notes", sa.String(length=1000), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.UniqueConstraint("user_id", "analysis_id", name="uq_user_feedback_user_analysis"),
        sa.CheckConstraint(
            "assessment IN ('AGREE', 'DISAGREE', 'UNSURE')",
            name="ck_user_feedback_assessment",
        ),
        sa.CheckConstraint(
            "claimed_label IS NULL OR claimed_label IN "
            "('GENUINE', 'AI_GENERATED', 'FACE_SWAP', 'OTHER')",
            name="ck_user_feedback_claimed_label",
        ),
    )
    op.create_index("ix_user_feedback_analysis_id", "user_feedback", ["analysis_id"])


def downgrade() -> None:
    op.drop_index("ix_user_feedback_analysis_id", table_name="user_feedback")
    op.drop_table("user_feedback")
