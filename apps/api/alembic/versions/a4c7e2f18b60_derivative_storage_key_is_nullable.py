"""derivative storage key is nullable

Revision ID: a4c7e2f18b60
Revises: 051bbc6f3851
Create Date: 2026-08-21 18:35:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'a4c7e2f18b60'
down_revision: Union[str, Sequence[str], None] = '051bbc6f3851'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema.

    Normalization moved off the upload request and into the worker, so an upload that
    needs a derivative now commits before one exists. The column has to be able to say
    that: a placeholder key would name an object nobody ever stored.

    Existing rows all carry a real key and are untouched — every analysis written before
    this migration was normalized, if at all, on the request that created it.
    """
    op.alter_column(
        'media_files',
        'derivative_storage_key',
        existing_type=sa.String(length=512),
        nullable=True,
    )


def downgrade() -> None:
    """Downgrade schema.

    Rows whose derivative is still owed cannot satisfy the restored constraint and are
    not invented into shape; the migration fails on them instead, which is the honest
    outcome for a schema that can no longer represent the state they are in.
    """
    op.alter_column(
        'media_files',
        'derivative_storage_key',
        existing_type=sa.String(length=512),
        nullable=False,
    )
