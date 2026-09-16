"""separate enrichment execution

Revision ID: e4b9c27a51f0
Revises: f8c1e4a20d75
Create Date: 2026-09-16 12:40:00.000000

The schema half of R10-T2: the execution record Deep Evidence Enrichment runs against, and
the database-level guard that makes a published verdict immutable.

**Nothing here backfills and nothing here mutates a row** (contract §8.2). The table is
created empty and stays empty for every analysis decided before this migration ran, which is
exactly the fact those analyses carry: they were decided by the single-stage worker, under
`p7-v1.0.0`, `r5-v3.0.0`, `r7-v4.0.0` or `r9-v5.0.0`, at a time when the distinction between
a decision and an enrichment did not exist. There is no two-axis state to backfill onto them
because the axis being recorded was not being measured. Readers project them as
`DECIDED + LEGACY_SINGLE_STAGE` at read time, from the absence of rows here — see
`app.enrichment.enrichment_state`, and §8.1, which is explicit that a legacy `completed` row
is *not* `ENRICHMENT_COMPLETE`.

`abstained` is a terminal status beside `completed` and `failed` for a related reason: a
detector that ran and found nothing to score — no trackable face, no audio stream — succeeded
on the enrichment axis (§5.2), and collapsing it into `failed` would have made a silent clip
look like an enrichment that broke. It changes nothing about the evidence: the signal row is
written exactly as it always was, and R9's coverage arithmetic still treats the abstention as
unresolved coverage. See `app.db.models.ENRICHMENT_TASK_STATUS_ABSTAINED`.

That absence is load-bearing, and it is why `not_requested` is a task *status* rather than
the lack of a task. An analysis decided under R10 with enrichment deferred holds four rows
saying so; a legacy analysis holds none. Had deferral been represented as "no rows", the two
would have been indistinguishable and every pre-R10 report would have started claiming that
supplementary evidence had been deliberately declined.

The trigger is the database half of the immutability declaration in §6.3, and it is the
second lock rather than the first. The first is `app.enrichment_guard`, which refuses the
enrichment path any write to `analyses` at all — including an idempotent rewrite, because
§6.1 puts the prohibition on the write and not on the delta. What this trigger adds is the
guarantee that does not depend on which code path is executing: once an analysis holds a
verdict, no statement from any connection may change its five decision fields. A second
verdict publication — two workers racing, a recovered worker coming back to life, a future
caller nobody has written yet — is refused by PostgreSQL rather than by a convention.

It fires on `risk_level IS NOT NULL`, which is precisely "this analysis has been decided"
(§6.3 freezes the fields "after the first successful fast decision"). A `DECISION_FAILED`
analysis has null decision fields and is not covered here; nothing writes to one either, and
inventing a second condition to protect columns that are already null would be protecting
nothing.

The trigger also protects every historical row, which is a consequence worth naming rather
than a side effect to discover: after this migration, invariant I9 — historical analyses are
byte-for-byte unchanged — is enforced by the database for the decision fields, not merely
promised by the code.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'e4b9c27a51f0'
down_revision: Union[str, Sequence[str], None] = 'f8c1e4a20d75'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


# The guard, as one function and one trigger. Written out here rather than assembled from the
# model's constants because a migration is a historical statement: it has to keep doing what
# it did on the day it ran, even after the columns it names have been renamed in `models.py`.
DECISION_GUARD_FUNCTION = "deepguard_reject_decided_analysis_update"
DECISION_GUARD_TRIGGER = "analyses_decision_is_immutable"

CREATE_DECISION_GUARD_FUNCTION = f"""
CREATE FUNCTION {DECISION_GUARD_FUNCTION}() RETURNS trigger AS $$
BEGIN
    IF OLD.risk_level IS NOT NULL AND (
           NEW.risk_level          IS DISTINCT FROM OLD.risk_level
        OR NEW.risk_rules_version  IS DISTINCT FROM OLD.risk_rules_version
        OR NEW.risk_calibration_id IS DISTINCT FROM OLD.risk_calibration_id
        OR NEW.risk_rule_id        IS DISTINCT FROM OLD.risk_rule_id
        OR NEW.status              IS DISTINCT FROM OLD.status
    ) THEN
        RAISE EXCEPTION
            'analysis % is already decided; its decision fields are immutable', OLD.id
            USING ERRCODE = 'raise_exception';
    END IF;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;
"""

CREATE_DECISION_GUARD_TRIGGER = f"""
CREATE TRIGGER {DECISION_GUARD_TRIGGER}
BEFORE UPDATE ON analyses
FOR EACH ROW
EXECUTE FUNCTION {DECISION_GUARD_FUNCTION}();
"""


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        'analysis_enrichment_tasks',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('analysis_id', sa.UUID(), nullable=False),
        sa.Column('provider', sa.String(length=64), nullable=False),
        sa.Column('signal_type', sa.String(length=64), nullable=False),
        sa.Column('status', sa.String(length=16), nullable=False),
        sa.Column('rules_version', sa.String(length=32), nullable=False),
        sa.Column('lease_expires_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['analysis_id'], ['analyses.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint(
            'analysis_id', 'provider', 'signal_type',
            name='uq_analysis_enrichment_tasks_component',
        ),
        sa.CheckConstraint(
            "status IN ('not_requested', 'queued', 'processing', 'completed', "
            "'abstained', 'failed')",
            name='ck_analysis_enrichment_tasks_status',
        ),
    )
    op.create_index(
        op.f('ix_analysis_enrichment_tasks_analysis_id'),
        'analysis_enrichment_tasks',
        ['analysis_id'],
        unique=False,
    )

    op.execute(CREATE_DECISION_GUARD_FUNCTION)
    op.execute(CREATE_DECISION_GUARD_TRIGGER)


def downgrade() -> None:
    """Downgrade schema.

    Drops what this migration created and nothing else. No analysis row is touched on the way
    down any more than on the way up: an enrichment execution record ceasing to exist says
    nothing about the verdict the analysis holds, and a downgrade that "tidied up" by writing
    to `analyses` would be the mutation the whole phase is built to avoid.

    A downgrade therefore returns every analysis to being read as legacy single-stage, which
    is the honest reading of a database with no enrichment records in it.
    """
    op.execute(f"DROP TRIGGER IF EXISTS {DECISION_GUARD_TRIGGER} ON analyses")
    op.execute(f"DROP FUNCTION IF EXISTS {DECISION_GUARD_FUNCTION}()")

    op.drop_index(
        op.f('ix_analysis_enrichment_tasks_analysis_id'),
        table_name='analysis_enrichment_tasks',
    )
    op.drop_table('analysis_enrichment_tasks')
