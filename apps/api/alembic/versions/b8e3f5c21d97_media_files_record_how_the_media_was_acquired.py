"""media files record how the media was acquired

Revision ID: b8e3f5c21d97
Revises: c5e19b4a70d3
Create Date: 2026-09-05 11:04:37.219640

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'b8e3f5c21d97'
down_revision: Union[str, Sequence[str], None] = 'c5e19b4a70d3'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema.

    One column recording how the analysed artifact came to exist (R7-T1). False is the
    acquisition every analysis before this had: an uploaded file, or a URL whose source
    served one already-muxed file, which the pipeline stored byte-for-byte. True is the
    acquisition YouTube's DASH/HLS catalogue forces: a video stream and an audio stream
    fetched separately and muxed into one container by ffmpeg on DeepGuard's own machine.

    It exists because the difference outlives the request that created it. C2PA provenance
    is read minutes later, by another process, off the stored original, and the sentence
    "these are the bytes the source served" is true of one of those artifacts and false of
    the other. Without a column the reader has no way to know which it is holding.

    An acquisition fact and nothing else. It is not evidence of tampering, not an input to
    the risk engine, and no rule, threshold or verdict reads it: media assembled from two
    streams is neither more nor less authentic than media served as one file, and the only
    claim this column supports is about what DeepGuard did, not about what the video is.

    `NOT NULL` with a `false` server default rather than nullable, because unlike a request
    id there is no unknown case to represent. Every existing row was acquired as a single
    file — that is what the code that wrote them could do — so the default states a fact
    about them rather than guessing one, and existing analyses keep reading exactly as they
    did. PostgreSQL stores the default in the catalogue, so this stays a metadata-only
    alteration of a populated table.

    No index: nothing searches by it. It is read on the row it belongs to, alongside the
    rest of that media's facts.
    """
    op.add_column(
        'media_files',
        sa.Column(
            'was_assembled',
            sa.Boolean(),
            nullable=False,
            server_default=sa.false(),
        ),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('media_files', 'was_assembled')
