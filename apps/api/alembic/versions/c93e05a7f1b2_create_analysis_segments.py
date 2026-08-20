"""create analysis segments

Revision ID: c93e05a7f1b2
Revises: b47f21c9d3a8
Create Date: 2026-08-20 18:41:02.554311

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'c93e05a7f1b2'
down_revision: Union[str, Sequence[str], None] = 'b47f21c9d3a8'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table('analysis_segments',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('signal_id', sa.UUID(), nullable=False),
    sa.Column('clip_index', sa.BigInteger(), nullable=False),
    sa.Column('logit', sa.Float(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.ForeignKeyConstraint(['signal_id'], ['analysis_signals.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_analysis_segments_signal_id'), 'analysis_segments', ['signal_id'], unique=False)


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(op.f('ix_analysis_segments_signal_id'), table_name='analysis_segments')
    op.drop_table('analysis_segments')
