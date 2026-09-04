"""create shadow runs

Revision ID: c5e19b4a70d3
Revises: a7f2c93e51b8
Create Date: 2026-09-04 21:10:00.000000

Shadow mode's isolated persistence (R6-T1). One table, holding both the queued work and the
observation the workload produced, and referenced by nothing that answers a customer or feeds
the risk engine — see `app.db.models.ShadowRun` for why the separation is structural.

Deliberately not a column on `analyses` and not a row in `analysis_signals`. Either would put
an uncalibrated experimental figure inside the record every customer-facing reader already
selects, and the isolation would then be a filter that one forgetful query undoes.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = 'c5e19b4a70d3'
down_revision: Union[str, Sequence[str], None] = 'a7f2c93e51b8'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table('shadow_runs',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('analysis_id', sa.UUID(), nullable=False),
    sa.Column('workload', sa.String(length=64), nullable=False),
    sa.Column('status', sa.String(length=16), nullable=False),
    sa.Column('lease_expires_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('error_message', sa.Text(), nullable=True),
    sa.Column('provider_version', sa.String(length=255), nullable=True),
    sa.Column('evidence', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.ForeignKeyConstraint(['analysis_id'], ['analyses.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('analysis_id', 'workload', name='uq_shadow_runs_analysis_workload')
    )
    op.create_index(op.f('ix_shadow_runs_analysis_id'), 'shadow_runs', ['analysis_id'], unique=False)


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(op.f('ix_shadow_runs_analysis_id'), table_name='shadow_runs')
    op.drop_table('shadow_runs')
