"""segments carry audio window evidence

Revision ID: f4c9a17b60de
Revises: d3b81a6f42c7
Create Date: 2026-08-25 10:12:04.331847

The local AASIST checkpoint is the third source to write into `analysis_segments`, and the
first that emits *two* raw figures for one scored unit: the graph produces a pair of logits
per 64600-sample window and publishes no softmax, threshold or class over them. The existing
`logit` column holds one of them; discarding the other would throw away half of what the
model actually said, and deriving one from the other is not possible.

So one nullable column is added and nothing else. The window index goes in `clip_index` and
the preprocessing bounds in `start_time`/`end_time`, both of which already exist and already
mean the right thing for this source — see `AnalysisSegment` for what those bounds do and do
not assert.

Nothing is backfilled and nothing is rewritten. Every existing row belongs to a source that
reports a single logit or none at all, and reads back exactly as it did before. Adding a
nullable column with no default is a catalog-only change in PostgreSQL, so this is safe on a
table that already holds evidence.

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'f4c9a17b60de'
down_revision: Union[str, Sequence[str], None] = 'd3b81a6f42c7'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # The second of AASIST's two raw outputs, in graph order: `logit` is output column 0 and
    # this is column 1, the one upstream reads as the bona fide score. Null on every row from
    # a source that emits a single figure or none.
    op.add_column(
        'analysis_segments', sa.Column('bona_fide_logit', sa.Float(), nullable=True)
    )


def downgrade() -> None:
    """Downgrade schema."""
    # Audio window rows would survive the drop as half their own evidence — a window index
    # and one unlabelled logit, for a signal type that does not exist below this revision.
    # A row that can no longer be read as what it is is worse than no row, so they go with
    # the column that described them. Nothing else in the table carries this value.
    op.execute(
        """
        DELETE FROM analysis_segments
        WHERE signal_id IN (
            SELECT id FROM analysis_signals WHERE signal_type = 'audio_authenticity'
        )
        """
    )

    op.drop_column('analysis_segments', 'bona_fide_logit')
