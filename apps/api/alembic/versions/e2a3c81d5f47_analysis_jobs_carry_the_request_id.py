"""analysis jobs carry the request id

Revision ID: e2a3c81d5f47
Revises: d9f1a4c6802e
Create Date: 2026-09-01 09:41:12.884517

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'e2a3c81d5f47'
down_revision: Union[str, Sequence[str], None] = 'd9f1a4c6802e'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

# Must match `app.observability.MAX_REQUEST_ID_LENGTH`, which is what refuses a longer value
# before it ever reaches this column. Restated rather than imported: a migration records what
# was done to a database on a particular day, and importing application code would let a
# later edit rewrite that history.
MAX_REQUEST_ID_LENGTH = 64


def upgrade() -> None:
    """Upgrade schema.

    The one column that carries a request's identity across the queue (R1-T4). The API
    writes it in the same insert as the job; the worker reads it back at claim time and
    binds it to its own logs.

    Nullable, and deliberately not backfilled. Every job already in this table was queued by
    a request nobody recorded an id for, and writing one now would fabricate the very fact
    the column exists to carry — an id that correlates to no log line anywhere. Nullable is
    also what makes this a metadata-only alteration of a populated table rather than a
    rewrite.

    No index. The column is read one job at a time, by primary key, in the statement that
    claims it; nothing searches by request id in the database, because the place an operator
    follows a request id is the log stream this migration exists to make greppable. An index
    added for a query nobody issues is write cost on the queue's hot path.
    """
    op.add_column(
        'analysis_jobs',
        sa.Column('request_id', sa.String(length=MAX_REQUEST_ID_LENGTH), nullable=True),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('analysis_jobs', 'request_id')
