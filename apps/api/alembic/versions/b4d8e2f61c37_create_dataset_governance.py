"""create dataset governance

Revision ID: b4d8e2f61c37
Revises: e7c3a9f05b21
Create Date: 2026-09-24 12:00:00.000000

Creates `lineage_splits` and `media_governance` (R12-T5): where a set of bytes came from, and
the one evaluation split its source lineage belongs to.

The split is a column of `lineage_splits` only. `media_governance` names a lineage by foreign
key and has no split column, so files of one lineage cannot be in two splits.

A derived file must be in its parent's lineage, and that is enforced here as well as in the
route: `(derived_from_sha256, source_lineage_id)` references `media_governance`'s own
`(media_sha256, source_lineage_id)`, which is unique because `media_sha256` is the key.

`media_sha256` is not a foreign key to `media_files`, for the reason `ground_truth.media_sha256`
is not. `actor_id` is not a foreign key either.

**No existing table is touched** and nothing is backfilled: a file with no governance row has no
recorded lineage or split, which is the correct state for every file that exists today.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = 'b4d8e2f61c37'
down_revision: Union[str, Sequence[str], None] = 'e7c3a9f05b21'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "lineage_splits",
        sa.Column("source_lineage_id", sa.String(length=128), primary_key=True, nullable=False),
        sa.Column("dataset_split", sa.String(length=16), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "dataset_split IN ('CALIBRATION', 'VALIDATION', 'TEST', 'HOLDOUT')",
            name="ck_lineage_splits_dataset_split",
        ),
    )

    op.create_table(
        "media_governance",
        sa.Column("media_sha256", sa.String(length=64), primary_key=True, nullable=False),
        sa.Column(
            "source_lineage_id",
            sa.String(length=128),
            sa.ForeignKey("lineage_splits.source_lineage_id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column("recording_identity", sa.String(length=128), nullable=True),
        sa.Column("transformations", postgresql.JSONB(), nullable=True),
        sa.Column("derived_from_sha256", sa.String(length=64), nullable=True),
        sa.Column("generation_pipeline", sa.String(length=128), nullable=True),
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
        sa.UniqueConstraint(
            "media_sha256", "source_lineage_id", name="uq_media_governance_sha256_lineage"
        ),
        sa.ForeignKeyConstraint(
            ["derived_from_sha256", "source_lineage_id"],
            ["media_governance.media_sha256", "media_governance.source_lineage_id"],
            name="fk_media_governance_parent_same_lineage",
            ondelete="RESTRICT",
        ),
        sa.CheckConstraint(
            "media_sha256 ~ '^[0-9a-f]{64}$'",
            name="ck_media_governance_media_sha256",
        ),
        sa.CheckConstraint(
            "derived_from_sha256 IS NULL OR derived_from_sha256 ~ '^[0-9a-f]{64}$'",
            name="ck_media_governance_derived_from_sha256",
        ),
        sa.CheckConstraint(
            "derived_from_sha256 IS NULL OR derived_from_sha256 <> media_sha256",
            name="ck_media_governance_not_self_derived",
        ),
    )
    op.create_index(
        "ix_media_governance_source_lineage_id", "media_governance", ["source_lineage_id"]
    )
    op.create_index(
        "ix_media_governance_recording_identity", "media_governance", ["recording_identity"]
    )
    op.create_index(
        "ix_media_governance_derived_from_sha256", "media_governance", ["derived_from_sha256"]
    )
    op.create_index("ix_media_governance_actor_id", "media_governance", ["actor_id"])


def downgrade() -> None:
    op.drop_index("ix_media_governance_actor_id", table_name="media_governance")
    op.drop_index("ix_media_governance_derived_from_sha256", table_name="media_governance")
    op.drop_index("ix_media_governance_recording_identity", table_name="media_governance")
    op.drop_index("ix_media_governance_source_lineage_id", table_name="media_governance")
    op.drop_table("media_governance")
    op.drop_table("lineage_splits")
