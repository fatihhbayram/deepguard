"""reviews carry an analyst assessment

Revision ID: a2d8f61c95b4
Revises: d7a2f9b41c63
Create Date: 2026-09-14 10:20:00.000000

Adds `analysis_reviews.analyst_assessment` (R9-T7): what a reviewer made of the automated
assessment, recorded on its own axis from the workflow status already on the row.

**Nothing on `analyses` is added, altered or read.** The verdict, the ruleset it was decided
under, the calibration and the rule that fired stay exactly as the engine wrote them — this
migration names one table and one new column on it, which is what keeps a human opinion from
ever being written into the forensic record.

**No row is backfilled and no default is chosen.** Every review that already exists keeps the
status it was written with and gets a null here, which is the correct state for it: its author
was never asked this question. That is deliberately distinct from `UNDETERMINED`, which means
somebody was asked and could not say. A `server_default` would have collapsed the two and
written an opinion on behalf of every reviewer in the table's history.

The column is nullable for the same reason, permanently rather than as a migration step. There
is no follow-up that makes it `NOT NULL`: reviews written from now on may still legitimately
carry no assessment, and a review is not invalid for lacking one.

The check constraint is the vocabulary. Only `AGREES_WITH_AUTOMATED_ASSESSMENT`,
`DISAGREES_WITH_AUTOMATED_ASSESSMENT` and `UNDETERMINED` are storable, and the long spellings
are the point — a value here can only be read as a statement about the automated assessment,
never as a second answer about the media. `TRUE`, `FALSE`, `CONFIRMED` and every correctness
label are refused by the database, not merely by the application.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'a2d8f61c95b4'
down_revision: Union[str, Sequence[str], None] = 'd7a2f9b41c63'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "analysis_reviews",
        # 64 rather than the 32 `status` carries: the longest value here is 35 characters, and
        # the length is what it is because the vocabulary is spelled out in full on purpose.
        sa.Column("analyst_assessment", sa.String(length=64), nullable=True),
    )
    op.create_check_constraint(
        "ck_analysis_reviews_analyst_assessment",
        "analysis_reviews",
        "analyst_assessment IS NULL OR analyst_assessment IN "
        "('AGREES_WITH_AUTOMATED_ASSESSMENT', "
        "'DISAGREES_WITH_AUTOMATED_ASSESSMENT', 'UNDETERMINED')",
    )


def downgrade() -> None:
    op.drop_constraint(
        "ck_analysis_reviews_analyst_assessment",
        "analysis_reviews",
        type_="check",
    )
    op.drop_column("analysis_reviews", "analyst_assessment")
