"""analysis jobs carry the product mode

Revision ID: a1c6f24b78e9
Revises: e4b9c27a51f0
Create Date: 2026-09-16 14:10:00.000000

The schema half of R10-T3's product modes. One nullable column on the execution record, and
the check constraint that keeps it to the two modes that exist.

**Nothing here backfills and nothing here mutates a row.** Every job already in this table
was queued by a submission that named no mode, and null is exactly that fact. Writing
`deep_analysis` onto them would record a choice nobody made, and would also be the wrong
choice to record: what those submissions actually got was the deployment's own enrichment
policy, which is what null continues to mean.

The column is on `analysis_jobs` and not on `analyses` because a mode is not a forensic
fact. It selects the enrichment trigger — whether Deep Evidence is queued with the decision
or left to be asked for — and it reaches no detector, no threshold, no ruleset and no
coverage denominator. `analyses` records what was measured and what was decided; this
records how one submission asked for its work to be scheduled, beside the request id that is
there for the same kind of reason.

Nullable also keeps this a metadata-only alteration of a populated table rather than a
rewrite.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'a1c6f24b78e9'
down_revision: Union[str, Sequence[str], None] = 'e4b9c27a51f0'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

# Restated rather than imported from `app.db.models`, as every migration here restates what
# it wrote: a migration records what was done to a database on a particular day, and an
# import would let a later edit rewrite that history.
PRODUCT_MODES = ('quick_scan', 'deep_analysis')

PRODUCT_MODE_CONSTRAINT = 'ck_analysis_jobs_enrichment_mode'


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column(
        'analysis_jobs',
        sa.Column('enrichment_mode', sa.String(length=16), nullable=True),
    )
    op.create_check_constraint(
        PRODUCT_MODE_CONSTRAINT,
        'analysis_jobs',
        "enrichment_mode IS NULL OR enrichment_mode IN ('%s')" % "', '".join(PRODUCT_MODES),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_constraint(PRODUCT_MODE_CONSTRAINT, 'analysis_jobs', type_='check')
    op.drop_column('analysis_jobs', 'enrichment_mode')
