"""analysis jobs carry a worker lease

Revision ID: 2c24bbb92649
Revises: c7a4e0b91f56
Create Date: 2026-08-26 15:15:33.033440

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '2c24bbb92649'
down_revision: Union[str, Sequence[str], None] = 'c7a4e0b91f56'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

# Must match `app.worker.LEASE_SECONDS`. Restated rather than imported: a migration is a
# record of what was done to a database on a particular day, and importing application code
# would let a later edit rewrite the history of what this migration actually ran.
LEASE_SECONDS = 180


def upgrade() -> None:
    """Upgrade schema.

    Nullable, because most jobs have no lease and never will: `queued` has not been claimed
    and a terminal job is finished. Nullable is also what keeps this a metadata-only change
    rather than a rewrite of the table.

    The backfill is the part worth reading. Jobs already sitting in `processing` were claimed
    by a worker that predates leases, so `NULL < now()` is `NULL` and recovery would step
    over them forever — the abandoned jobs this change exists for would be exactly the ones
    it could never reach. They are given one full lease from now rather than an expired one:
    if a pre-lease worker is somehow still running one, it survives long enough for the
    deploy to replace it, and if none is, the job is recovered a few minutes from now instead
    of instantly. Being three minutes late to fail a dead job costs nothing; failing a live
    one costs an analysis.
    """
    op.add_column(
        'analysis_jobs',
        sa.Column('lease_expires_at', sa.DateTime(timezone=True), nullable=True),
    )
    op.execute(
        sa.text(
            "UPDATE analysis_jobs "
            f"SET lease_expires_at = now() + interval '{LEASE_SECONDS} seconds' "
            "WHERE status = 'processing'"
        )
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('analysis_jobs', 'lease_expires_at')
