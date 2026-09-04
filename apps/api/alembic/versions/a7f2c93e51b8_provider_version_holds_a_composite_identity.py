"""provider version holds a composite identity

Revision ID: a7f2c93e51b8
Revises: e2a3c81d5f47
Create Date: 2026-09-04 19:40:12.884213

Every detector wired in before R5-T2 has an identity that fits in 128 characters: an NVCF
function id is a uuid, and a Hugging Face checkpoint is `repository@revision`. LipForensics
does not, and not because its name is long.

Its architecture and its weights come from two different places. The network is executed from
a pinned commit of `ahaliassos/LipForensics`; the FF++ forgery weights are a file the upstream
README links on Google Drive, which carries no revision of its own and whose only identity is
its digest. Neither half pins the model on its own — the same checkpoint loaded into a
different network is a different model, and the same network with different weights is a
different model — so the signal records both, and both at full length: a truncated SHA-256
beside a full one would be the only abbreviated digest in this schema, and R5-T3 will compare
this column with exact string equality when it calibrates an operating point for the detector.

So the column is widened rather than the identity being cut to fit it. 255 is the next
ordinary width above what is needed and leaves room for the next such model without a second
migration for the same reason.

Widening a `varchar` in PostgreSQL rewrites no rows and takes no table rewrite — it is a
catalog change and an `ALTER TABLE ... TYPE` that the planner treats as a no-op resize — so
this is safe on a table that already holds evidence, and every stored value reads back
unchanged.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'a7f2c93e51b8'
down_revision: Union[str, Sequence[str], None] = 'e2a3c81d5f47'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.alter_column(
        'analysis_signals',
        'provider_version',
        existing_type=sa.String(length=128),
        type_=sa.String(length=255),
        existing_nullable=True,
    )


def downgrade() -> None:
    """Downgrade schema."""
    # Narrowing is not the mirror image of widening: PostgreSQL refuses the change outright if
    # any stored value is longer than the new width, so a database carrying mouth-dynamics evidence
    # cannot go back through this. That is the correct failure — the alternative is truncating
    # a model identity, which would leave signals naming weights nobody can check them against
    # — and it is left to fail loudly rather than being papered over with a `USING` clause.
    op.alter_column(
        'analysis_signals',
        'provider_version',
        existing_type=sa.String(length=255),
        type_=sa.String(length=128),
        existing_nullable=True,
    )
