"""create analysis signals

Revision ID: b47f21c9d3a8
Revises: e58c4d97c70a
Create Date: 2026-08-20 10:12:44.118207

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = 'b47f21c9d3a8'
down_revision: Union[str, Sequence[str], None] = 'e58c4d97c70a'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table('analysis_signals',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('analysis_id', sa.UUID(), nullable=False),
    sa.Column('provider', sa.String(length=64), nullable=False),
    sa.Column('signal_type', sa.String(length=64), nullable=False),
    sa.Column('score', sa.Float(), nullable=True),
    sa.Column('risk_level', sa.String(length=16), nullable=True),
    sa.Column('provider_version', sa.String(length=128), nullable=True),
    sa.Column('status', sa.String(length=16), nullable=False),
    sa.Column('metadata', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.ForeignKeyConstraint(['analysis_id'], ['analyses.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_analysis_signals_analysis_id'), 'analysis_signals', ['analysis_id'], unique=False)


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(op.f('ix_analysis_signals_analysis_id'), table_name='analysis_signals')
    op.drop_table('analysis_signals')
