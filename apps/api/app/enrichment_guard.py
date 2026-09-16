"""The persistence boundary Deep Evidence Enrichment is allowed to write through (R10-T2).

R10 splits one job into two paths over the same `Analysis`, and the whole value of that split
rests on one asymmetry: the fast path decides, and the enrichment path may never touch what it
decided. §6.1 of `docs/architecture/R10_EXECUTION_CONTRACT.md` lists what is denied; §6.3
requires that the denial be enforced "in depth rather than by convention — a database-level
guard, a conditional `WHERE` clause on the verdict write, or both", and is explicit that what
may *not* be chosen is "to rely on the enrichment code simply not doing it".

This module is the application half of that. It hands the enrichment worker a `Session` that
physically refuses the writes the contract denies, so that a defect in the enrichment path — a
careless `session.add`, a future detector that decides it should mark the analysis complete, a
copy-paste from `app.worker` — fails loudly at the boundary instead of quietly rewriting a
verdict. The migration `e4b9c27a51f0` is the other half: a PostgreSQL trigger that refuses a
second verdict publication from any connection at all, including one that never came through
here.

The two are deliberately not the same check. The trigger can only see values, so it enforces
§6.3 — a decided analysis's fields may not *change*. This guard can see intent, so it enforces
§6.1 — the enrichment path may not write those rows *at all*, "with a different value" or "with
the same value", because "the prohibition is on the write, not on the delta". A rewrite that
happens to be idempotent still means the enrichment path holds code that touches the verdict,
and it is that code this is built to make impossible to ship.

**What is allowed, in full:**

- `analysis_enrichment_tasks` — the path's own execution records;
- `analysis_signals`, through the ORM, for a `(provider, signal_type)` pair that is
  evidence-only under the ruleset the analysis was decided under;
- `analysis_segments`, for a signal row this same session has already been allowed to write.

Everything else is refused, and the refusal is by table rather than by column wherever the
whole table is out of reach. `analyses`, `analysis_jobs`, `analysis_reviews` and `media_files`
have no legitimate enrichment write between them — the decision fields, the decision job's
status, an analyst's assessment and the media identity columns are the four things §6.1 names —
so the rule is the simple one and there is no column list to keep in step with a schema that
moves.

**Why the allowlist for signals is not a denylist of SVD and B7.** §3.2 requires that
enrichment membership follow the ruleset's role assignment rather than a detector list held in
the orchestration code, because a future ruleset can promote a detector back into the decision
(as `r5-v3.0.0` did for LipForensics). A denylist naming today's two deciding detectors would
keep letting a promoted detector through; an allowlist built from the ruleset's evidence-only
set stops naming it the moment the ruleset stops. The pair the guard admits is therefore asked
of `app.enrichment`, per analysis, from the version frozen on its task rows.

**Core DML against `analysis_signals` is refused outright**, including DML that would have been
legitimate. The guard can read a provider off an ORM object it is handed; it cannot read one
out of the `WHERE` clause of an arbitrary `UPDATE`, and a check that silently passed statements
it could not inspect would be an allowlist with a hole in the shape of the attack it exists to
stop. The enrichment path writes signals through the ORM, so the rule costs it nothing.

**This guard is not where enrichment is stopped from deciding.** It cannot be: `evaluate_v5`
writes nothing, so no persistence check can see it run. That prohibition (§7.2, invariant I4)
is structural instead — `app.enrichment` holds no import of `app.risk_engine`, and
`tests/test_enrichment.py` asserts that absence over the module rather than trusting it, in the
manner `tests/test_shadow_mode.py` already asserts the shadow isolation.
"""

import logging
from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import event
from sqlalchemy.orm import Session

from app.db.models import (
    AnalysisEnrichmentTask,
    AnalysisSegment,
    AnalysisSignal,
)
from app.db.session import SessionLocal

logger = logging.getLogger(__name__)

# The one table the enrichment path owns outright, and the two it owns conditionally. Named as
# table names rather than as classes because the check has to work for Core DML, where there is
# no class to compare against.
ENRICHMENT_TASKS_TABLE = AnalysisEnrichmentTask.__tablename__
SIGNALS_TABLE = AnalysisSignal.__tablename__
SEGMENTS_TABLE = AnalysisSegment.__tablename__

# Every table a write may reach through this session at all. Anything absent from this set is
# refused before its columns are even looked at — see the module docstring for why the coarse
# rule is the right one for `analyses`, `analysis_jobs`, `analysis_reviews` and `media_files`.
WRITABLE_TABLES = frozenset({ENRICHMENT_TASKS_TABLE, SIGNALS_TABLE, SEGMENTS_TABLE})


class ProtectedWriteRejected(Exception):
    """The enrichment path tried to write something §6.1 denies it.

    A defect, always, and raised rather than logged-and-skipped: a write that was attempted and
    silently dropped would leave the enrichment path believing it had done something, and the
    next reader would have to guess which of the two happened. It carries what was attempted so
    the traceback names the table and the reason without quoting a row.

    It is not an enrichment *failure* in the sense §7.4 gives that word. A component that
    records `FAILED` has been asked a forensic question and had no answer; this is code doing
    something it is not allowed to do, and it is the enrichment worker's own bug to fix.
    """


class EnrichmentWriteBoundary:
    """Tracks what one guarded session has been allowed to write, and refuses the rest.

    Stateful, and it has to be: a segment is legitimate only if the signal it hangs off is one
    this same session was allowed to write, and that is a fact about this session's history
    rather than about the segment in isolation. The alternative — reading the parent signal
    back out of the database to check its provider — would ask the database a question this
    object already knows the answer to, on every segment of every component.

    One boundary per session, created by `enrichment_session` and not reachable otherwise. It
    is deliberately not a module-level singleton: two enrichment sessions in one process must
    not be able to authorise each other's writes.
    """

    def __init__(self, allowed_components: frozenset[tuple[str, str]]) -> None:
        # The `(provider, signal_type)` pairs that are evidence-only under the ruleset this
        # analysis was decided under. Passed in rather than computed here, because the roles
        # belong to the frozen ruleset and this module is not entitled to an opinion about
        # which detector decides — see §3.2 and `app.enrichment.evidence_only_components`.
        self.allowed_components = allowed_components
        # Signal rows this session has already approved. Held as the objects rather than as
        # their ids, because an id is not the stable thing here: `AnalysisSignal.id` has a
        # Python-side default, so a freshly constructed row carries `None` until the flush
        # that writes it assigns one. Recording the id at approval time would have recorded
        # `None` for every new signal and then matched no segment at all — while recording it
        # for a row that already had one would have worked, which is the worst shape a bug
        # like this can take. The object is identical across both moments; its id is not.
        self._owned_signals: list[AnalysisSignal] = []

    def check_instance(self, instance: object, operation: str) -> None:
        """Refuse one ORM object this session is about to write, unless §6.2 allows it."""
        if isinstance(instance, AnalysisEnrichmentTask):
            return

        if isinstance(instance, AnalysisSignal):
            self._check_signal(instance, operation)
            return

        if isinstance(instance, AnalysisSegment):
            self._check_segment(instance, operation)
            return

        raise ProtectedWriteRejected(
            f"The enrichment path may not {operation} {type(instance).__name__}. "
            "Deep Evidence writes its own execution records, its own signal rows and their "
            "segments, and nothing else (R10 execution contract §6.1)."
        )

    def _check_signal(self, signal: AnalysisSignal, operation: str) -> None:
        """Refuse a signal row that is not this analysis's enrichment evidence to write."""
        component = (signal.provider, signal.signal_type)

        if component not in self.allowed_components:
            raise ProtectedWriteRejected(
                f"The enrichment path may not {operation} the "
                f"{signal.provider}/{signal.signal_type} signal: it is not evidence-only "
                "under the ruleset this analysis was decided under. The deciding detectors' "
                "rows are the evidence the verdict rests on and are written once, by the "
                "fast decision path (R10 execution contract §6.1)."
            )

        # §6.2, and the one column-level rule here. `analysis_signals.risk_level` is null on
        # every row in this database and stays null: a level written beside a provider's score
        # would be the per-detector verdict the risk engine exists to refuse (rule 11). The
        # fast path does not write it either; enrichment is refused it explicitly because
        # enrichment is the path that could plausibly be tempted.
        if signal.risk_level is not None:
            raise ProtectedWriteRejected(
                "The enrichment path may not write a risk level onto a signal row. "
                "Risk is a decision about the analysis, taken under a named ruleset, and it "
                "lives on `analyses` (R10 execution contract §6.2)."
            )

        # Approved, so its segments may name it. Recorded after the checks, never before: a
        # signal that was refused must not leave a row behind that would let its segments in.
        self._owned_signals.append(signal)

    def _check_segment(self, segment: AnalysisSegment, operation: str) -> None:
        """Refuse a segment that does not hang off a signal this session was allowed to write.

        The owned ids are resolved now rather than at approval time, because that is the moment
        they exist: the signal was flushed between the two, and the flush is what assigned it
        one. A `None` on either side never matches, so a segment presented before its parent
        has been written is refused rather than admitted by two absent ids comparing equal.
        """
        owned = {signal.id for signal in self._owned_signals if signal.id is not None}

        if segment.signal_id is None or segment.signal_id not in owned:
            raise ProtectedWriteRejected(
                f"The enrichment path may not {operation} a segment of signal "
                f"{segment.signal_id}: this session has not been allowed to write that "
                "signal. Segments are writable only beneath an enrichment signal row of "
                "this same analysis (R10 execution contract §6.2)."
            )

    def check_statement(self, statement: object, operation: str) -> None:
        """Refuse one Core DML statement this session is about to execute.

        Table-level, because that is what a statement reliably exposes. `analysis_signals` is
        refused here even though some updates to it would be legitimate — the module docstring
        says why an allowlist that waved through statements it could not read would not be an
        allowlist at all.
        """
        table = getattr(statement, "table", None)
        name = getattr(table, "name", None)

        if name == ENRICHMENT_TASKS_TABLE:
            return

        if name in WRITABLE_TABLES:
            raise ProtectedWriteRejected(
                f"The enrichment path may not {operation} `{name}` through a Core "
                "statement. Evidence rows are written through the ORM, where the guard can "
                "read the provider off the row it is given; a statement's WHERE clause is "
                "not something this boundary can check (R10 execution contract §6.1)."
            )

        raise ProtectedWriteRejected(
            f"The enrichment path may not {operation} `{name}`. Deep Evidence writes its own "
            "execution records, its own signal rows and their segments, and nothing else "
            "(R10 execution contract §6.1)."
        )


@contextmanager
def enrichment_session(
    allowed_components: frozenset[tuple[str, str]],
) -> Iterator[Session]:
    """A database session the enrichment path may use, and the only one it may use.

    The guard is installed on this session alone rather than on `SessionLocal`, which the API
    and the fast decision path also build from. A global listener would have had to decide from
    inside a flush which caller it was serving, and the answer to that is a thread-local or a
    flag — either of which is a switch, and a switch protecting a verdict is a switch somebody
    eventually turns off.

    Two listeners, because ORM work and Core DML arrive through different doors and closing one
    would leave the other open:

    - `before_flush` sees every object added, modified or deleted through the unit of work, and
      is where a signal's provider and a segment's parent can actually be read;
    - `do_orm_execute` sees `session.execute(...)` of an `INSERT`, `UPDATE` or `DELETE`, which
      is how `app.worker` writes statuses and is therefore the shape a copy-paste into the
      enrichment path would take.

    Reads are untouched. The enrichment worker has to be able to select the analysis, its media
    and its own tasks, and nothing about a `SELECT` can change a verdict.

    A rejected write raises out of the flush or the execute, which rolls the transaction back
    with nothing written. The caller is expected not to catch it as an enrichment failure — it
    is a defect in this codebase, not a detector that had no answer.
    """
    boundary = EnrichmentWriteBoundary(allowed_components)
    session = SessionLocal()

    @event.listens_for(session, "before_flush")
    def _guard_flush(flush_session: Session, flush_context, instances) -> None:
        for instance in flush_session.new:
            boundary.check_instance(instance, "insert")
        for instance in flush_session.dirty:
            # `dirty` is optimistic — it includes objects whose attributes were touched
            # without changing. That is the right side to err on here: §6.1 puts the
            # prohibition on the write rather than on the delta, so an object presented for
            # update is checked whether or not the update turns out to be a no-op.
            boundary.check_instance(instance, "update")
        for instance in flush_session.deleted:
            boundary.check_instance(instance, "delete")

    @event.listens_for(session, "do_orm_execute")
    def _guard_execute(state) -> None:
        if state.is_select:
            return

        if state.is_insert or state.is_update or state.is_delete:
            operation = (
                "insert into"
                if state.is_insert
                else "update" if state.is_update else "delete from"
            )
            boundary.check_statement(state.statement, operation)

    try:
        yield session
    finally:
        session.close()
