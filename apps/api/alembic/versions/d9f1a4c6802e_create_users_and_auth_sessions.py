"""create users and auth sessions

Revision ID: d9f1a4c6802e
Revises: 2c24bbb92649
Create Date: 2026-08-31 11:04:52.117903

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'd9f1a4c6802e'
down_revision: Union[str, Sequence[str], None] = '2c24bbb92649'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema.

    Three changes, in the only order that works: `users` first, because the column added to
    `analyses` at the end points at it.

    `users.email` is unique on the *normalized* address — lowercased and stripped, which is
    the form the application stores and the form login looks up by. Unique on anything else
    would let two accounts differ only by case and both answer to the same sign-in.

    `users.role` defaults to `USER` in the database as well as in the model, so an account
    inserted by hand is an ordinary one. The safe direction: the mistake this prevents is an
    accidentally administrative account.

    `auth_sessions` holds no token, only its SHA-256 digest — 64 hex characters, unique
    because authentication looks the row up by it directly. `expires_at` is required and
    `revoked_at` is not: every session has a deadline from the moment it is created, and
    only a deliberately ended one has a revocation time.

    `analyses.owner_id` is nullable and permanently so, exactly like `api_key_id` beside it:
    every analysis stored before this migration was submitted by nobody, and null says that
    accurately where a backfilled value would be invented. Nullable also keeps this a
    metadata-only alteration rather than a rewrite of the table.

    The check constraint is the point of the whole column. It admits a row owned by a user,
    a row owned by an API key and a row owned by nobody, and refuses only the fourth case —
    a row owned by both, which is what would make one customer's analysis reachable through
    another authentication path. It is written as `NOT (both present)` rather than as an
    exclusive-or so the unowned dashboard rows already in the table stay legal; an XOR would
    fail to create against existing data and would be the wrong rule besides.
    """
    op.create_table('users',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('email', sa.String(length=320), nullable=False),
    sa.Column('password_hash', sa.String(length=255), nullable=False),
    sa.Column('role', sa.String(length=16), server_default='USER', nullable=False),
    sa.Column('is_active', sa.Boolean(), server_default=sa.text('true'), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_users_email'), 'users', ['email'], unique=True)

    op.create_table('auth_sessions',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('user_id', sa.UUID(), nullable=False),
    sa.Column('token_hash', sa.String(length=64), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('expires_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('revoked_at', sa.DateTime(timezone=True), nullable=True),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_auth_sessions_user_id'), 'auth_sessions', ['user_id'], unique=False)
    op.create_index(op.f('ix_auth_sessions_token_hash'), 'auth_sessions', ['token_hash'], unique=True)

    op.add_column('analyses', sa.Column('owner_id', sa.UUID(), nullable=True))
    op.create_index(op.f('ix_analyses_owner_id'), 'analyses', ['owner_id'], unique=False)
    op.create_foreign_key(
        'fk_analyses_owner_id_users',
        'analyses',
        'users',
        ['owner_id'],
        ['id'],
        ondelete='RESTRICT',
    )
    op.create_check_constraint(
        'ck_analyses_single_owner',
        'analyses',
        'NOT (owner_id IS NOT NULL AND api_key_id IS NOT NULL)',
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_constraint('ck_analyses_single_owner', 'analyses', type_='check')
    op.drop_constraint('fk_analyses_owner_id_users', 'analyses', type_='foreignkey')
    op.drop_index(op.f('ix_analyses_owner_id'), table_name='analyses')
    op.drop_column('analyses', 'owner_id')

    op.drop_index(op.f('ix_auth_sessions_token_hash'), table_name='auth_sessions')
    op.drop_index(op.f('ix_auth_sessions_user_id'), table_name='auth_sessions')
    op.drop_table('auth_sessions')

    op.drop_index(op.f('ix_users_email'), table_name='users')
    op.drop_table('users')
