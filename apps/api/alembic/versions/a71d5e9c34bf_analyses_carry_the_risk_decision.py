"""analyses carry the risk decision

Revision ID: a71d5e9c34bf
Revises: f4c9a17b60de
Create Date: 2026-08-25 14:02:51.118409

P7-T3 gives DeepGuard its first classification, and a classification that cannot be
explained afterwards is not forensic evidence. So four columns go on `analyses` together:
the level, the immutable ruleset that produced it, the calibration the thresholds were
measured under, and the single rule that fired. Reading a row a year from now then answers
"what was concluded, by which sentence, under which measurement" without re-running
anything — and without trusting that the rules in the working tree are still the rules that
decided it.

Four columns and no fifth table. A generalized rule-engine schema — rule definitions,
versions, per-rule outcome rows — would be a framework for a ruleset that currently has
four rules and one input signal (AGENTS.md: YAGNI, no speculative abstraction). If a second
direct-risk source ever arrives, that is when the shape it needs will be known.

Nothing is copied here from the evidence. No score, no threshold, no clip count: those stay
in `analysis_signals`, which remains the forensic record, and a figure duplicated into this
table could silently drift from the row it was copied out of (rule 11).

All four are nullable with no default and nothing is backfilled. Null means no decision has
been taken — every analysis that completed before this revision, plus every analysis still
queued — and it is deliberately not the same as `UNKNOWN`, which is a real classification
with rule `R010` or `R012` behind it. Inventing `UNKNOWN` for historical rows would fabricate
decisions no ruleset ever made. Adding nullable columns with no default is a catalog-only
change in PostgreSQL, so this is safe on a table that already holds analyses.

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'a71d5e9c34bf'
down_revision: Union[str, Sequence[str], None] = 'f4c9a17b60de'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # `HIGH`, `MEDIUM` or `UNKNOWN`. Ruleset v1 emits no `LOW` — the value was measured
    # (T_LOW = 0.05) and deliberately not activated as a boundary — so the width is for the
    # vocabulary the model declares, not for what v1 writes.
    op.add_column('analyses', sa.Column('risk_level', sa.String(length=16), nullable=True))
    # The ruleset that decided, as an opaque name: `p7-v1.0.0`. Not a package version and
    # not a commit — it changes when a rule, threshold or ordering changes, and only then.
    op.add_column(
        'analyses', sa.Column('risk_rules_version', sa.String(length=32), nullable=True)
    )
    # SHA-256 of the calibration artifact's identity fields, so a stored verdict is traceable
    # to the corpus, provider deployment and error policy its thresholds came from.
    op.add_column(
        'analyses', sa.Column('risk_calibration_id', sa.String(length=64), nullable=True)
    )
    # Which single rule fired — `R010`, `R012`, `R100` or `R200`. The level says what was
    # concluded; this says why, and the two are stored together or neither is readable.
    op.add_column('analyses', sa.Column('risk_rule_id', sa.String(length=16), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    # The trace goes with the level. Keeping a classification below this revision would
    # leave a verdict with no ruleset, calibration or rule behind it — exactly the
    # unexplainable decision these columns were added to prevent — and the evidence it was
    # derived from is untouched in `analysis_signals` either way.
    op.drop_column('analyses', 'risk_rule_id')
    op.drop_column('analyses', 'risk_calibration_id')
    op.drop_column('analyses', 'risk_rules_version')
    op.drop_column('analyses', 'risk_level')
