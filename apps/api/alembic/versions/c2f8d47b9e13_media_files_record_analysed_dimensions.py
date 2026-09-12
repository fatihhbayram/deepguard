"""media files record analysed dimensions

Revision ID: c2f8d47b9e13
Revises: f1b7d3c2a904
Create Date: 2026-09-12 09:41:03.517204

Separates the two dimension pairs a rotated video has. `width`/`height` have always been
the *original's* coded size; what a detector actually receives is the derivative, and
ffmpeg bakes the display matrix into it — so a phone video encoded 1920x1080 with a
quarter turn is analysed as a genuine 1080x1920 picture. Reporting the first pair as the
analysed geometry was a false statement about what was examined.

All three columns are nullable and none is backfilled here. Every row written before this
migration was analysed by a worker that never measured its derivative, and null is read as
"not recorded" — the report degrades to showing the original's encoded size, labelled as
the original's, rather than passing it off as the analysed one.

Recovering what can be recovered is `backfill_analysed_dimensions.py`, deliberately not
this file: it downloads and probes stored artifacts, which is not work to do inside a DDL
transaction, and it is interruptible and re-runnable in a way a migration is not. Its
docstring carries the evaluation of which historical rows are answerable and why the rest
must stay null.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'c2f8d47b9e13'
down_revision: Union[str, Sequence[str], None] = 'f1b7d3c2a904'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "media_files",
        sa.Column("display_rotation", sa.Integer(), nullable=True),
    )
    op.add_column(
        "media_files",
        sa.Column("analyzed_width", sa.Integer(), nullable=True),
    )
    op.add_column(
        "media_files",
        sa.Column("analyzed_height", sa.Integer(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("media_files", "analyzed_height")
    op.drop_column("media_files", "analyzed_width")
    op.drop_column("media_files", "display_rotation")
