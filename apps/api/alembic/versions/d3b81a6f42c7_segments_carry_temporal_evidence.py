"""segments carry temporal evidence

Revision ID: d3b81a6f42c7
Revises: a4c7e2f18b60
Create Date: 2026-08-21 23:05:41.882706

`analysis_segments` was built for one evidence source. NVIDIA's synthetic-video detector
scores the video in clips, so the table required a clip index and a logit on every row.
Active Speaker Detection is the second source to write here and it reports neither: its
evidence is a time range, a tracked face and the diarized voice that face was matched to.

So the two clip columns become nullable and four columns are added. Nothing is
backfilled and nothing is rewritten — every existing row is clip evidence, keeps both of
its figures, and reads back exactly as it did before.

Widening `NOT NULL` to nullable takes no table rewrite in PostgreSQL, and the added
columns are all nullable with no default, so this is a catalog-only change on a table
that already holds evidence.

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'd3b81a6f42c7'
down_revision: Union[str, Sequence[str], None] = 'a4c7e2f18b60'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Clip evidence is no longer the only evidence. An active-speaker row has no clip and
    # no logit, and filling these in for one would fabricate provider output.
    op.alter_column('analysis_segments', 'clip_index', existing_type=sa.BigInteger(), nullable=True)
    op.alter_column('analysis_segments', 'logit', existing_type=sa.Float(), nullable=True)

    # Seconds from the start of the analysed video. Null on clip evidence, which NVIDIA
    # reports no times for.
    op.add_column('analysis_segments', sa.Column('start_time', sa.Float(), nullable=True))
    op.add_column('analysis_segments', sa.Column('end_time', sa.Float(), nullable=True))

    # Who a time range is about. `face_id` is NVIDIA's tracked-face identifier, stored
    # wide because the provider's field is a uint32; `speaker_label` is pyannote's own
    # label for the matched voice, and is null when NVIDIA matched the face to none.
    op.add_column('analysis_segments', sa.Column('face_id', sa.BigInteger(), nullable=True))
    op.add_column('analysis_segments', sa.Column('speaker_label', sa.String(length=64), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    # Active-speaker rows carry no clip index or logit, so restoring the NOT NULL
    # constraints would fail against them. They are evidence for a signal type that does
    # not exist below this revision, so they are removed with the columns that described
    # them rather than left behind as rows nothing can read.
    op.execute(
        """
        DELETE FROM analysis_segments
        WHERE clip_index IS NULL OR logit IS NULL
        """
    )

    op.drop_column('analysis_segments', 'speaker_label')
    op.drop_column('analysis_segments', 'face_id')
    op.drop_column('analysis_segments', 'end_time')
    op.drop_column('analysis_segments', 'start_time')

    op.alter_column('analysis_segments', 'logit', existing_type=sa.Float(), nullable=False)
    op.alter_column('analysis_segments', 'clip_index', existing_type=sa.BigInteger(), nullable=False)
