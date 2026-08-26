"""analyses carry their api key owner

Revision ID: c7a4e0b91f56
Revises: b6f0a92c47d1
Create Date: 2026-08-26 09:41:07.552118

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'c7a4e0b91f56'
down_revision: Union[str, Sequence[str], None] = 'b6f0a92c47d1'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema.

    Nullable, and nullable permanently. Every analysis stored before this migration came in
    through the dashboard, which authenticates nobody, so there is no key to backfill and no
    honest default to invent — null is the accurate statement that no API key submitted the
    row. Adding the column nullable is also what keeps this a metadata-only change on a
    populated table rather than a rewrite of every existing analysis.

    The index is not decoration: every public read filters on this column, and without it a
    customer polling one analysis would scan every analysis on file.

    `RESTRICT` on delete, so removing an API key cannot silently take the forensic records
    it authenticated with it. Keys are retired with `is_active = false`, which leaves the
    analyses and their attribution intact.
    """
    op.add_column('analyses', sa.Column('api_key_id', sa.UUID(), nullable=True))
    op.create_index(op.f('ix_analyses_api_key_id'), 'analyses', ['api_key_id'], unique=False)
    op.create_foreign_key(
        'fk_analyses_api_key_id_api_keys',
        'analyses',
        'api_keys',
        ['api_key_id'],
        ['id'],
        ondelete='RESTRICT',
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_constraint('fk_analyses_api_key_id_api_keys', 'analyses', type_='foreignkey')
    op.drop_index(op.f('ix_analyses_api_key_id'), table_name='analyses')
    op.drop_column('analyses', 'api_key_id')
