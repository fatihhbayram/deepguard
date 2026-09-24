"""media governance carries corpus metadata

Revision ID: c5e2a7d49f18
Revises: b4d8e2f61c37
Create Date: 2026-09-24 18:00:00.000000

Adds to `media_governance` (R12-T5A) the eight fields `scripts/eval/corpus.CorpusItem` requires
and no table recorded: `license`, `permission_status`, `stratum_primary`, `source`,
`acquisition_type`, `benchmark_family`, `redistributable` and `private`.

All eight are nullable, have no server default and are not backfilled: null is "not recorded",
which is the truth for every existing row. That includes the two booleans — a default of
`redistributable = false` / `private = true` would read as a statement nobody made, while null
keeps the row out of any export until somebody does.

No existing column, key or constraint is touched; the split, lineage and parent invariants of
`b4d8e2f61c37` are unchanged.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'c5e2a7d49f18'
down_revision: Union[str, Sequence[str], None] = 'b4d8e2f61c37'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("media_governance", sa.Column("license", sa.String(length=255), nullable=True))
    op.add_column(
        "media_governance", sa.Column("permission_status", sa.String(length=255), nullable=True)
    )
    op.add_column(
        "media_governance", sa.Column("stratum_primary", sa.String(length=128), nullable=True)
    )
    op.add_column("media_governance", sa.Column("source", sa.String(length=512), nullable=True))
    op.add_column(
        "media_governance", sa.Column("acquisition_type", sa.String(length=128), nullable=True)
    )
    op.add_column(
        "media_governance", sa.Column("benchmark_family", sa.String(length=128), nullable=True)
    )
    op.add_column("media_governance", sa.Column("redistributable", sa.Boolean(), nullable=True))
    op.add_column("media_governance", sa.Column("private", sa.Boolean(), nullable=True))


def downgrade() -> None:
    op.drop_column("media_governance", "private")
    op.drop_column("media_governance", "redistributable")
    op.drop_column("media_governance", "benchmark_family")
    op.drop_column("media_governance", "acquisition_type")
    op.drop_column("media_governance", "source")
    op.drop_column("media_governance", "stratum_primary")
    op.drop_column("media_governance", "permission_status")
    op.drop_column("media_governance", "license")
