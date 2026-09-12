"""admin audit events

Revision ID: b3f7a21c9d84
Revises: c2f8d47b9e13
Create Date: 2026-09-12 20:40:00.000000

Creates `admin_audit_events` (R8-T5): an append-only record of the privileged changes an
administrator makes to somebody else's account.

Nothing is backfilled. Every role change made before this table existed was made by a route
that had nowhere to write it down, and inventing rows for them would put statements in an
audit log that nobody witnessed — the log starts here and is silent about what came before,
which is the only honest thing it can say.

`actor_id` holds a `users.id` and is deliberately not a foreign key to it, which is the same
shape `target_id` has. An audit event is a historical fact and must outlive the lifecycle of
the account it names: a reference would make this table an obstacle to deleting a user —
`RESTRICT` blocking the deletion, `CASCADE` destroying the very evidence the log exists to
keep, `SET NULL` erasing who did it — and all three let account housekeeping edit the past.
The row stays readable without the account because it does not depend on it: the address is
snapshotted onto the event itself. The column is indexed, because correlating events by actor
is the one query beyond the listing that this table will be asked for.

`changes` is `JSON` and not `JSONB`. The payload is read back whole by one screen and is
never queried into, so the containment indexing JSONB exists for buys nothing here, while its
key reordering would lose the one piece of structure the document has.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = 'b3f7a21c9d84'
down_revision: Union[str, Sequence[str], None] = 'c2f8d47b9e13'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "admin_audit_events",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("actor_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("actor_email_snapshot", sa.String(length=255), nullable=True),
        sa.Column("action", sa.String(length=64), nullable=False),
        sa.Column("target_type", sa.String(length=64), nullable=False),
        sa.Column("target_id", sa.String(length=255), nullable=False),
        sa.Column("target_email_snapshot", sa.String(length=255), nullable=True),
        sa.Column("changes", sa.JSON(), nullable=False),
        sa.Column("request_id", sa.String(length=64), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
    )
    # Not a constraint, just a lookup: "everything this administrator has done" is the one
    # question beyond the listing that this table will be asked.
    op.create_index(
        "ix_admin_audit_events_actor_id", "admin_audit_events", ["actor_id"]
    )
    # The only read this table has orders by this column, newest first.
    op.create_index(
        "ix_admin_audit_events_created_at", "admin_audit_events", ["created_at"]
    )


def downgrade() -> None:
    op.drop_index("ix_admin_audit_events_created_at", table_name="admin_audit_events")
    op.drop_index("ix_admin_audit_events_actor_id", table_name="admin_audit_events")
    op.drop_table("admin_audit_events")
