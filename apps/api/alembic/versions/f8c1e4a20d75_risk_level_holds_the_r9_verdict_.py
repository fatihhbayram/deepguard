"""risk level holds the r9 verdict vocabulary

Revision ID: f8c1e4a20d75
Revises: a2d8f61c95b4
Create Date: 2026-09-14 19:26:19.548139

`analyses.risk_level` has been `varchar(16)` since the column was created, which was wide
enough for every verdict any ruleset through `r7-v4.0.0` can produce: `HIGH`, `MEDIUM`,
`UNKNOWN`. `r9-v5.0.0` speaks a different vocabulary and two of its three verdicts do not
fit — `MANIPULATION_DETECTED` is 21 characters and `NO_CALIBRATED_MANIPULATION_SIGNAL` is
33.

Those names are long deliberately (R9-T1 invariant 3): `NO_CALIBRATED_MANIPULATION_SIGNAL`
says what was and was not established, where a short label like `LOW` invites every reader
downstream of it to hear "authentic". Abbreviating the vocabulary to fit the column would
undo the thing the vocabulary exists to do, so the column is widened instead.

64 rather than `text`: it holds the longest verdict today with room for a fourth that is
longer, and it keeps the column a bounded domain — a width a bad write runs into rather
than a field that silently accepts anything. Both columns that declare the risk-level type
move together, so neither is left as a trap for whoever writes to it next.

**This migration activates nothing.** `evaluate_v5` still has no caller in the production
path; the worker writes v4 verdicts under `p7-v1.0.0` exactly as it did yesterday. What
changes is that the schema stops being the reason v5 cannot be persisted, which is the
prerequisite for the task that wires it up and not that task itself.

Nothing is rewritten. Widening a `varchar` in PostgreSQL is a catalog change — no table
rewrite, no row touched, no lock beyond the brief `ACCESS EXCLUSIVE` the `ALTER` takes — so
every stored `HIGH`, `MEDIUM`, `UNKNOWN` and null reads back byte for byte as it was.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'f8c1e4a20d75'
down_revision: Union[str, Sequence[str], None] = 'a2d8f61c95b4'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


# The two columns that declare the risk-level type, and the only two this migration touches.
# Other `varchar(16)` columns in this schema — `analysis_jobs.status`, `shadow_runs.status`,
# `users.role`, `media_files.acquisition_method`, `analyses.risk_rule_id` — hold unrelated
# vocabularies that the R9 verdicts have nothing to do with, and every one of them still fits
# what it holds (`R9-100`, the longest new rule id, is six characters).
VERDICT_COLUMNS = ("analyses", "analysis_signals")

# What the column was, and what a downgrade would put back.
LEGACY_WIDTH = 16
WIDTH = 64


def upgrade() -> None:
    """Upgrade schema."""
    for table in VERDICT_COLUMNS:
        op.alter_column(
            table,
            'risk_level',
            existing_type=sa.VARCHAR(length=LEGACY_WIDTH),
            type_=sa.String(length=WIDTH),
            existing_nullable=True,
        )


def downgrade() -> None:
    """Downgrade schema.

    Refused outright if any stored verdict is longer than the old width.

    PostgreSQL would refuse the `ALTER` itself in that case, so this check is not what makes
    the downgrade safe — it is what makes it *legible*, and what makes it atomic in the way
    that matters. The server's own error names a column and a value and leaves the operator
    to work out that a v5 verdict is now persisted and that going back means deciding what
    to do with it; this one says so, names the table and counts the rows, and it runs before
    any DDL is emitted, so a two-column downgrade cannot narrow the first column and then
    fail on the second.

    A `USING left(risk_level, 16)` clause would make this succeed, and it is exactly what
    must not be written here: `NO_CALIBRATED_MANIPULATION_SIGNAL` truncated to sixteen
    characters is `NO_CALIBRATED_MA`, which is not a verdict, not a value any rule can
    explain, and silently destroys a forensic record. Losing the ability to downgrade is the
    correct price of having persisted something the old schema cannot hold.
    """
    bind = op.get_bind()

    for table in VERDICT_COLUMNS:
        overlong = bind.execute(
            sa.text(
                f"SELECT count(*) FROM {table} "  # noqa: S608 - a literal from the tuple above
                "WHERE risk_level IS NOT NULL AND char_length(risk_level) > :width"
            ),
            {"width": LEGACY_WIDTH},
        ).scalar_one()

        if overlong:
            longest = bind.execute(
                sa.text(
                    f"SELECT DISTINCT risk_level FROM {table} "  # noqa: S608
                    "WHERE char_length(risk_level) > :width ORDER BY risk_level"
                ),
                {"width": LEGACY_WIDTH},
            ).scalars().all()

            raise RuntimeError(
                f"Refusing to downgrade {revision}: {table}.risk_level holds {overlong} "
                f"row(s) whose verdict is longer than {LEGACY_WIDTH} characters "
                f"({', '.join(longest)}). Narrowing the column would truncate a persisted "
                "forensic verdict into a value no ruleset can explain. Resolve those rows "
                "deliberately before downgrading."
            )

    for table in reversed(VERDICT_COLUMNS):
        op.alter_column(
            table,
            'risk_level',
            existing_type=sa.String(length=WIDTH),
            type_=sa.VARCHAR(length=LEGACY_WIDTH),
            existing_nullable=True,
        )
