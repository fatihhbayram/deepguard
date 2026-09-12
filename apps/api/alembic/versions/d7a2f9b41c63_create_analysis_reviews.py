"""create analysis reviews

Revision ID: d7a2f9b41c63
Revises: b3f7a21c9d84
Create Date: 2026-09-12 21:10:00.000000

Creates `analysis_reviews` (R8-T7): the human governance layer over a forensic result, kept
in its own table so that the result itself stays untouched.

**No column is added to `analyses` and none is altered.** That is the point of a separate
table rather than two more fields: `risk_level`, `risk_rules_version`, `risk_calibration_id`
and `risk_rule_id` are written once by the worker under a named ruleset, and putting a mutable
opinion in the same row would have placed every one of them within reach of an UPDATE meant
for the opinion. This migration cannot change a forensic value because it does not name one.

Nothing is backfilled and nothing needs to be. Every analysis that already exists is
unreviewed, and unreviewed is the absence of a row here — so the state is already correct for
the entire table the moment this runs, with no rows written and no default to choose.

`analysis_id` is both the foreign key and the primary key, which is what makes "one review per
analysis" a property of the schema rather than a rule the endpoint remembers. `ON DELETE
CASCADE`: a review of a deleted analysis annotates nothing.

`reviewer_id` holds a `users.id` and is deliberately not constrained to one, the same shape
`admin_audit_events.actor_id` has and for a related reason. A reference would make this table
an obstacle to removing an account — `RESTRICT` blocking it, `CASCADE` destroying governance
attached to analyses that are still live, `SET NULL` leaving a review nobody signed. The row
stays readable without the account because the address is snapshotted onto it. The column is
indexed, because "everything this reviewer looked at" is the one query beyond the per-analysis
read that this table will be asked for.

The check constraint on `status` is the vocabulary itself. Only `REVIEWED` and
`NEEDS_FOLLOW_UP` are operational workflow states; a forensic-sounding value in this column
would be a second, unversioned classification of the same media sitting beside the real one,
and the database refuses one rather than trusting every future writer not to insert it.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = 'd7a2f9b41c63'
down_revision: Union[str, Sequence[str], None] = 'b3f7a21c9d84'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "analysis_reviews",
        sa.Column(
            "analysis_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("analyses.id", ondelete="CASCADE"),
            primary_key=True,
            nullable=False,
        ),
        sa.Column("status", sa.String(length=32), nullable=False),
        # Bounded, and matching `AnalysisReview.note`. A note the model accepted and the
        # database refused would surface as a 500 on a request that was merely too long.
        sa.Column("note", sa.String(length=1000), nullable=False),
        sa.Column("reviewer_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("reviewer_email_snapshot", sa.String(length=255), nullable=True),
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
        sa.CheckConstraint(
            "status IN ('REVIEWED', 'NEEDS_FOLLOW_UP')",
            name="ck_analysis_reviews_status",
        ),
    )
    # Not a constraint, just a lookup: "everything this reviewer has looked at".
    op.create_index(
        "ix_analysis_reviews_reviewer_id", "analysis_reviews", ["reviewer_id"]
    )


def downgrade() -> None:
    op.drop_index("ix_analysis_reviews_reviewer_id", table_name="analysis_reviews")
    op.drop_table("analysis_reviews")
