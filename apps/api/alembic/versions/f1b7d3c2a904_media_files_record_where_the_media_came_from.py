"""media files record where the media came from

Revision ID: f1b7d3c2a904
Revises: b8e3f5c21d97
Create Date: 2026-09-08 10:12:44.108921

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'f1b7d3c2a904'
down_revision: Union[str, Sequence[str], None] = 'b8e3f5c21d97'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

# The two doors media comes through. `upload` is a file a client sent; `url` is a file this
# service fetched from an address a client named. Widths match the longer of the two with
# room to spare rather than being sized to the exact string.
ACQUISITION_METHOD_LENGTH = 16

# A hostname and nothing else. 253 is the DNS ceiling on a fully qualified name, so a value
# this column cannot hold is a value that is not a hostname.
SOURCE_HOST_LENGTH = 253


def upgrade() -> None:
    """Upgrade schema.

    Two columns saying how the analysed artifact reached DeepGuard (R7-T12), beside the
    `was_assembled` column R7-T1 added for how it was put together.

    `acquisition_method` is `upload` or `url`. `source_host` is the normalized hostname of a
    URL submission — lowercased, `www.` stripped, no scheme, no port, no path, no query, no
    fragment, no credentials — and null for an upload, where there is no host and inventing
    one would be a fabrication.

    Deliberately the hostname alone. The submitted URL can carry a signed expiry, an access
    token, a session id or a private path in its query string, and a column holding the whole
    thing would put those in the database, in every backup of it, and in front of every reader
    of an analysis. The host answers the question a report needs to ask — where did this come
    from — and is the largest part of the URL that carries no secret.

    Both nullable, and neither backfilled. Every row written before this column existed was
    acquired by a request that recorded nothing about how, and an analysis from that era is
    honestly *unknown* on both counts: it could have been an upload or a URL, and no evidence
    survives to say which. A `NOT NULL DEFAULT 'upload'` would be a guess written into the
    record as a fact, which is exactly the error this task exists to prevent.

    This is why R7-T1's `was_assembled = false` must not be read alone. On a row where
    `acquisition_method` is null, `false` means only what that migration's default meant — the
    artifact was not muxed here — and says nothing about whether a source served it or a
    client sent it. `was_assembled` becomes an acquisition claim only when read together with
    a non-null `acquisition_method`.

    An acquisition fact and nothing else. No detector, threshold, ruleset or verdict reads
    either column, and neither is evidence about the media: a file fetched from a host is
    neither more nor less authentic than one uploaded, and the only claim these columns
    support is about what DeepGuard did.

    No index: nothing searches or aggregates by host. Both are read on the row they belong
    to, alongside the rest of that media's facts. Adding nullable columns with no default is
    a metadata-only alteration, so this does not rewrite the populated table.
    """
    op.add_column(
        'media_files',
        sa.Column(
            'acquisition_method',
            sa.String(length=ACQUISITION_METHOD_LENGTH),
            nullable=True,
        ),
    )
    op.add_column(
        'media_files',
        sa.Column(
            'source_host',
            sa.String(length=SOURCE_HOST_LENGTH),
            nullable=True,
        ),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('media_files', 'source_host')
    op.drop_column('media_files', 'acquisition_method')
