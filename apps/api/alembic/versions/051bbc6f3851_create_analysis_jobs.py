"""create analysis jobs

Revision ID: 051bbc6f3851
Revises: c93e05a7f1b2
Create Date: 2026-08-20 18:51:01.775224

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '051bbc6f3851'
down_revision: Union[str, Sequence[str], None] = 'c93e05a7f1b2'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema.

    Analyses that predate this table keep no job. They were detected synchronously and
    are already finished, so inventing `completed` job rows for them would be recording
    work this queue never did.
    """
    op.create_table('analysis_jobs',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('analysis_id', sa.UUID(), nullable=False),
    sa.Column('status', sa.String(length=16), nullable=False),
    sa.Column('error_message', sa.Text(), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.ForeignKeyConstraint(['analysis_id'], ['analyses.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('analysis_id')
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table('analysis_jobs')
