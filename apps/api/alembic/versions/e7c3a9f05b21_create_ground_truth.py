"""create ground truth

Revision ID: e7c3a9f05b21
Revises: a1c6f24b78e9
Create Date: 2026-09-22 12:00:00.000000

Creates `ground_truth` (R12-T2): what a set of bytes actually is, recorded by an administrator
and kept apart from every verdict, review and provenance value it is used to measure.

Keyed by `media_sha256` — the `original_sha256` of the bytes — rather than by a media row or an
analysis. The same bytes can be uploaded under many analyses, so a truth attached to one media
row would belong to one upload of the file instead of to the file. For the same reason the
column is not a foreign key: there is no single row it could reference.

No `manipulation_family` column. It is derived from `label` on read and never stored.

**No existing table is touched.** Nothing on `analyses`, `analysis_signals`,
`analysis_reviews` or `media_files` is added or altered, and nothing is backfilled: a hash with
no row here has no recorded Ground Truth, which is the correct state for every file that exists
the moment this runs.

`actor_id` is not a foreign key either, for the reason `admin_audit_events.actor_id` is not.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = 'e7c3a9f05b21'
down_revision: Union[str, Sequence[str], None] = 'a1c6f24b78e9'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "ground_truth",
        sa.Column("media_sha256", sa.String(length=64), primary_key=True, nullable=False),
        sa.Column("source_class", sa.String(length=32), nullable=False),
        sa.Column("label", sa.String(length=32), nullable=False),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("actor_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("actor_email_snapshot", sa.String(length=255), nullable=True),
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
    )
    op.create_index("ix_ground_truth_actor_id", "ground_truth", ["actor_id"])


def downgrade() -> None:
    op.drop_index("ix_ground_truth_actor_id", table_name="ground_truth")
    op.drop_table("ground_truth")
