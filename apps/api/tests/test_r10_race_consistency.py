"""The seven Race/Read consistency scenarios R10-T4 mandates.

`tests/test_enrichment.py` proves what the enrichment path *cannot* do, one claim at a time,
against a system that is standing still. This module asks the other question: what a reader
can *see* while the system is moving. Every test here runs a reader on its own connection,
concurrently with a writer on another, and keeps every sample it took — an invariant that
holds at rest and breaks in the window between two commits is exactly the defect R10's
two-axis projection could introduce, and the only way to find it is to be looking during the
window rather than after it.

The seven scenarios, as R10-T4 names them:

1. a report read while a component transitions `processing` -> `completed`;
2. a report read while another component fails;
3. repeated reads during enrichment never alter the verdict;
4. no half-written signal/segment state;
5. a deferred trigger concurrent with a report read;
6. an enrichment request concurrent with enrichment completion;
7. no duplicate task, and no stale projection crash.

Three rules the whole module follows.

**The reader reads production's read path.** `app.api.analyses.analysis_evidence_select` and
`analysis_payloads` are what the report route calls, and they are what the samples below are
taken through. A test that sampled `analysis_states` alone would be checking the projection
and not the report, and the report is where a half-written state would be shown to somebody.
The ownership filter is the one thing omitted — these fixtures' analyses have no owner, and
`visible_to` is not what any of these races is about.

**The writer is production's writer.** `enrichment.persist`, `enrichment.process_one`,
`enrichment.request` and `enrichment.recover_stale_tasks`, unmodified. Only the detectors are
stubbed, for the reason `tests/test_enrichment.py` stubs them: nothing here is about a score.

**A window that was never entered proves nothing.** Every concurrent test asserts that its
reader saw the state both before and after the transition it was watching. Without that, a
run whose writer finished before the reader's first sample would pass by having missed the
race entirely, which is the failure mode a concurrency test is likeliest to have.
"""

import threading
import time
import uuid

import pytest
from sqlalchemy import event, func, select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError

from app import detection, enrichment
from app.api.analyses import analysis_evidence_select, analysis_payloads
from app.db.models import (
    ENRICHMENT_TASK_COMPONENT_CONSTRAINT,
    Analysis,
    AnalysisEnrichmentTask,
    AnalysisJob,
    AnalysisSegment,
    AnalysisSignal,
    MediaFile,
)
from app.db.session import SessionLocal, engine

RULES_V5 = "r9-v5.0.0"

LIP_COMPONENT = ("lipforensics", "lip_forensics")
EFFORT_COMPONENT = ("effort", "face_forgery")
ASD_COMPONENT = ("nvidia", "active_speaker")
AASIST_COMPONENT = ("aasist", "audio_authenticity")

# How many segments the stubbed AASIST writes. More than one on purpose: scenario 4 is about
# a reader landing *between* two segment inserts, and a single-segment signal cannot express
# the difference between "none yet" and "some but not all".
AASIST_SEGMENTS = 8

# The five columns §6.1 write-protects, read as one tuple so a sample is comparable with `==`.
DECISION_FIELDS = (
    "status",
    "risk_level",
    "risk_rules_version",
    "risk_calibration_id",
    "risk_rule_id",
)

# Every value the enrichment axis is allowed to take. Scenario 7 asserts membership rather
# than a particular state: what it is hunting is a projection that returns something outside
# its own vocabulary, or raises, while the rows underneath it are being rewritten.
LEGAL_ENRICHMENT_STATES = frozenset(
    {
        enrichment.ENRICHMENT_NOT_REQUESTED,
        enrichment.ENRICHMENT_PENDING,
        enrichment.ENRICHMENT_PROCESSING,
        enrichment.ENRICHMENT_COMPLETE,
        enrichment.ENRICHMENT_PARTIAL,
        enrichment.ENRICHMENT_FAILED,
        enrichment.LEGACY_SINGLE_STAGE,
        enrichment.ENRICHMENT_NOT_APPLICABLE,
    }
)

LEGAL_DECISION_STATES = frozenset(
    {enrichment.DECISION_PENDING, enrichment.DECIDED, enrichment.DECISION_FAILED}
)


# --------------------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------------------


@pytest.fixture
def database():
    """The live engine, or a skip when this environment has no PostgreSQL.

    Every test in this module is about two connections disagreeing, so there is nothing here
    that can be demonstrated without a real server.
    """
    try:
        with engine.connect():
            pass
    except SQLAlchemyError as error:
        pytest.skip(f"PostgreSQL is not reachable: {error.__class__.__name__}")

    return engine


@pytest.fixture
def decided(database):
    """A decided analysis with its media, exactly as the Fast Decision Path leaves one.

    `completed`, with all five decision fields populated and both deciding detectors' evidence
    already committed — the only state §4.2 ever queues enrichment against. The deciding
    signals are written because the report read path reads them, and a race test whose reader
    was looking at a report with no evidence on it would be watching the wrong page.
    """
    created = []

    def make(*, mode=None):
        with SessionLocal() as session:
            row = Analysis(
                status="completed",
                risk_level="NO_CALIBRATED_MANIPULATION_SIGNAL",
                risk_rules_version=RULES_V5,
                risk_calibration_id="c" * 64,
                risk_rule_id="R9-200",
            )
            session.add(row)
            session.flush()

            digest = uuid.uuid4().hex + uuid.uuid4().hex
            session.add(
                MediaFile(
                    analysis_id=row.id,
                    original_filename="clip.mp4",
                    content_type="video/mp4",
                    size_bytes=4096,
                    original_sha256=digest,
                    original_storage_key=f"originals/{digest}",
                    format_name="mov,mp4,m4a,3gp,3g2,mj2",
                    codec_name="h264",
                    width=1920,
                    height=1080,
                    duration=12.34,
                    frame_rate=30.0,
                    pix_fmt="yuv420p",
                    constant_frame_rate=True,
                    was_normalized=False,
                )
            )
            session.add(
                AnalysisSignal(
                    analysis_id=row.id,
                    provider="nvidia",
                    signal_type="synthetic_video",
                    status="SUCCESS",
                    score=0.1234,
                    provider_version="svd-1",
                    signal_metadata={"total_clips": 7},
                )
            )
            session.add(
                AnalysisSignal(
                    analysis_id=row.id,
                    provider="efficientnet-b7",
                    signal_type="face_manipulation",
                    status="SUCCESS",
                    score=0.0456,
                    provider_version="b7-1",
                    signal_metadata={"frames_scored": 6},
                )
            )
            session.add(
                AnalysisJob(analysis_id=row.id, status="completed", enrichment_mode=mode)
            )
            session.commit()
            created.append(row.id)

            return row.id

    yield make

    with SessionLocal() as session:
        for analysis_id in created:
            # Media, job, tasks, signals and segments all follow through ON DELETE CASCADE.
            session.query(Analysis).filter(Analysis.id == analysis_id).delete()
        session.commit()


@pytest.fixture
def components(monkeypatch):
    """Stand in for the four evidence-only detectors, so no test here loads a checkpoint.

    The same shape `tests/test_enrichment.py` uses, with two additions this module needs: a
    latch that holds one detector inside its call so a reader can be run against a component
    that is genuinely `processing`, and a segment-bearing AASIST result so scenario 4 has a
    signal whose children could in principle land apart from it.
    """

    class Recorder:
        def __init__(self):
            self.effort_fails = True
            self.lip_gate = None
            self.lip_entered = threading.Event()
            self.calls = []

        def detect_lip_forensics(self, path):
            self.calls.append("lip")
            self.lip_entered.set()
            if self.lip_gate is not None:
                # Held here, inside the detector, with the task row committed `processing`
                # and its lease live. This is the only way to give a reader a window that is
                # wide enough to sample rather than one it has to win a coin toss to enter.
                assert self.lip_gate.wait(timeout=60), "the lip gate was never released"
            return AnalysisSignal(
                provider=LIP_COMPONENT[0],
                signal_type=LIP_COMPONENT[1],
                status="SUCCESS",
                score=0.31,
                provider_version="lip-1",
            )

        def detect_effort(self, path):
            self.calls.append("effort")
            return AnalysisSignal(
                provider=EFFORT_COMPONENT[0],
                signal_type=EFFORT_COMPONENT[1],
                status="FAILED",
                signal_metadata={
                    "error": (
                        "EffortModelUnavailable"
                        if self.effort_fails
                        else "EffortNoFaceDetected"
                    )
                },
            )

        async def analyse_audio(self, path, frame_rate):
            self.calls.append("audio")
            speaker = AnalysisSignal(
                provider=ASD_COMPONENT[0],
                signal_type=ASD_COMPONENT[1],
                status="SUCCESS",
                provider_version="asd-1",
            )
            audio = AnalysisSignal(
                provider=AASIST_COMPONENT[0],
                signal_type=AASIST_COMPONENT[1],
                status="SUCCESS",
                score=0.12,
                provider_version="aasist-1",
            )
            # Shaped as the real detector writes them, `clip_index` and `bona_fide_logit`
            # included: the report's audio-window projection reads both and refuses a row
            # missing either, so a thinner stub would fail this module's reader on its own
            # fixture rather than on anything the writer did.
            segments = [
                AnalysisSegment(
                    clip_index=index,
                    start_time=float(index),
                    end_time=float(index) + 1.0,
                    logit=0.5,
                    bona_fide_logit=-0.5,
                )
                for index in range(AASIST_SEGMENTS)
            ]
            return (speaker, []), (audio, segments)

    recorder = Recorder()
    monkeypatch.setattr(detection, "detect_lip_forensics", recorder.detect_lip_forensics)
    monkeypatch.setattr(detection, "detect_effort", recorder.detect_effort)
    monkeypatch.setattr(detection, "analyse_audio", recorder.analyse_audio)
    return recorder


@pytest.fixture
def artifact(monkeypatch, tmp_path):
    """Hand the enrichment path a local file instead of reaching MinIO."""
    path = tmp_path / "artifact.mp4"
    path.write_bytes(b"not really a video")

    def fetch(storage_key, destination):
        destination.write_bytes(path.read_bytes())

    monkeypatch.setattr(enrichment, "fetch_object", fetch)
    return path


@pytest.fixture
def immediate(monkeypatch):
    """The default deployment policy, stated rather than inherited from the environment."""
    monkeypatch.delenv(enrichment.ENRICHMENT_POLICY_VARIABLE, raising=False)


# --------------------------------------------------------------------------------------
# The reader
# --------------------------------------------------------------------------------------


def read_report(analysis_id):
    """One report read, through the statements the report route actually issues.

    A session of its own every time, so what comes back is what a *different* transaction can
    see rather than anything this process has staged. `visible_to` is the one thing the route
    does that is left out — see the module docstring.
    """
    with SessionLocal() as session:
        rows = session.execute(
            analysis_evidence_select().where(Analysis.id == analysis_id)
        ).all()
        payloads = analysis_payloads(session, rows)

    return payloads[0] if payloads else None


def sample(analysis_id):
    """One report read reduced to the facts these tests compare.

    A payload carries far more than any of this is about, and comparing whole payloads would
    make every assertion below fail for the wrong reason the first time a field is added.
    """
    payload = read_report(analysis_id)
    if payload is None:
        return None

    return {
        "decision_state": payload.decision_state,
        "enrichment_state": payload.aggregate_enrichment_state,
        "components": tuple(
            (component.provider, component.signal_type, component.state)
            for component in payload.per_component_state
        ),
        "verdict": (
            payload.status,
            payload.risk_level,
            payload.risk_rules_version,
            payload.risk_calibration_id,
            payload.risk_rule_id,
        ),
    }


def evidence_shape(analysis_id):
    """Every evidence-only signal on this analysis, with how many segments it has.

    Read in one transaction on one connection, which is what makes it a statement about
    atomicity: PostgreSQL's default READ COMMITTED gives each *statement* its own snapshot, so
    the two selects below are issued inside one explicitly-opened transaction to be certain
    they see the same one. A signal counted here without its segments would be a half-written
    detector result, which is what §7.5 forbids.
    """
    with SessionLocal() as session:
        with session.begin():
            session.connection(
                execution_options={"isolation_level": "REPEATABLE READ"}
            )
            signals = session.execute(
                select(AnalysisSignal.id, AnalysisSignal.provider, AnalysisSignal.signal_type)
                .where(AnalysisSignal.analysis_id == analysis_id)
            ).all()
            counts = dict(
                session.execute(
                    select(AnalysisSegment.signal_id, func.count())
                    .where(
                        AnalysisSegment.signal_id.in_([row.id for row in signals] or [None])
                    )
                    .group_by(AnalysisSegment.signal_id)
                ).all()
            )

    return {
        (row.provider, row.signal_type): counts.get(row.id, 0) for row in signals
    }



def report_and_evidence(analysis_id):
    """The report and the raw evidence rows, read in one transaction at one snapshot.

    Scenario 1 compares a component's task status against whether its signal row exists, and
    that comparison is only meaningful if both facts come from the same instant. Two separate
    sessions would give two snapshots, and a commit landing between them would report a torn
    state that no transaction ever saw — §7.5 explicitly permits evidence to appear "between
    two successive reads", so a reader that straddled the commit would be failing the writer
    for the reader's own choice of where to look.

    REPEATABLE READ rather than the default, for the same reason `evidence_shape` uses it: it
    is per-*statement* snapshots under READ COMMITTED that would tear, and there is more than
    one statement in a report read.
    """
    with SessionLocal() as session:
        with session.begin():
            session.connection(
                execution_options={"isolation_level": "REPEATABLE READ"}
            )
            rows = session.execute(
                analysis_evidence_select().where(Analysis.id == analysis_id)
            ).all()
            payloads = analysis_payloads(session, rows)
            present = {
                (row.provider, row.signal_type)
                for row in session.execute(
                    select(AnalysisSignal.provider, AnalysisSignal.signal_type).where(
                        AnalysisSignal.analysis_id == analysis_id
                    )
                ).all()
            }

    payload = payloads[0]

    return {
        "components": {
            (component.provider, component.signal_type): component.state
            for component in payload.per_component_state
        },
        "evidence": present,
        "verdict": (
            payload.status,
            payload.risk_level,
            payload.risk_rules_version,
            payload.risk_calibration_id,
            payload.risk_rule_id,
        ),
    }


def stage_and_commit(analysis_id, mode=None):
    """Stage this analysis's components, as the transaction publishing a verdict does."""
    with SessionLocal() as session:
        staged = enrichment.stage(session, analysis_id, RULES_V5, mode)
        session.commit()

    return staged


def task_rows(analysis_id):
    with SessionLocal() as session:
        return {
            (task.provider, task.signal_type): task.status
            for task in session.query(AnalysisEnrichmentTask)
            .filter_by(analysis_id=analysis_id)
            .all()
        }


def decision_columns(analysis_id):
    with SessionLocal() as session:
        row = session.get(Analysis, analysis_id)
        return tuple(getattr(row, field) for field in DECISION_FIELDS)


class Sampler(threading.Thread):
    """A reader that samples the report on its own connection until it is told to stop.

    Kept as a thread rather than a loop inside each test because all seven need the same
    thing: start sampling, let a writer run, stop, and then reason over everything that was
    seen. It records exceptions rather than raising them into a thread nobody is joining on,
    so scenario 7's "no stale projection crash" is an assertion about `self.error` instead of
    a test that hangs.
    """

    def __init__(self, analysis_id, read=sample, interval=0.0):
        super().__init__(daemon=True)
        self.analysis_id = analysis_id
        self.read = read
        self.interval = interval
        self.samples = []
        self.error = None
        self.started_reading = threading.Event()
        self._stopping = threading.Event()

    def run(self):
        try:
            while not self._stopping.is_set():
                self.samples.append(self.read(self.analysis_id))
                self.started_reading.set()
                if self.interval:
                    time.sleep(self.interval)
            # One last sample after the writer is known to be finished, so every test can
            # assert on the settled state without a second read path.
            self.samples.append(self.read(self.analysis_id))
        except BaseException as error:  # noqa: BLE001 — recorded, then asserted on
            self.error = error
            self.started_reading.set()

    def stop(self):
        self._stopping.set()
        self.join(timeout=60)
        assert not self.is_alive(), "the sampler did not stop"


def run_sampled(analysis_id, writer, read=sample):
    """Run `writer` with a reader sampling throughout, and hand back everything it saw."""
    sampler = Sampler(analysis_id, read=read)
    sampler.start()
    assert sampler.started_reading.wait(timeout=30), "the sampler never took a sample"

    try:
        writer()
    finally:
        sampler.stop()

    assert sampler.error is None, f"the reader raised: {sampler.error!r}"
    assert len(sampler.samples) > 1, "the reader took no useful samples"

    return sampler.samples


# --------------------------------------------------------------------------------------
# 1. A report read while a component transitions processing -> completed
# --------------------------------------------------------------------------------------


@pytest.mark.integration
def test_rc1_a_read_never_catches_a_component_between_processing_and_completed(
    decided, components, artifact, immediate
):
    """Scenario 1. The component's task status and its evidence move together, or not at all.

    LipForensics is held inside its own detector call, so the window this samples is real and
    wide rather than a few microseconds that a run might never enter: the task row is
    committed `processing` before the detector is entered, and the transition to `completed`
    happens in `persist`, in the same transaction as the signal insert.

    The invariant is the biconditional. For every sample, LipForensics reads `completed` if
    and only if its signal row is there. A sample showing `completed` with no signal would be
    a report claiming evidence that has not landed; a sample showing the signal while the task
    still said `processing` would be the same defect with the two halves swapped.
    """
    analysis_id = decided()
    stage_and_commit(analysis_id)

    gate = threading.Event()
    components.lip_gate = gate

    sampler = Sampler(analysis_id, read=report_and_evidence)
    worker_done = threading.Event()

    def enrich():
        with SessionLocal() as session:
            enrichment.process_one(session)
        worker_done.set()

    thread = threading.Thread(target=enrich, daemon=True)
    thread.start()

    # The detector is now inside the call, which means the task row is committed `processing`.
    assert components.lip_entered.wait(timeout=60), "the enrichment never started"
    sampler.start()
    assert sampler.started_reading.wait(timeout=30), "the sampler never took a sample"

    # Let it sample the held state for a moment, then release the transition underneath it.
    time.sleep(0.5)
    gate.set()
    assert worker_done.wait(timeout=120), "the enrichment never finished"
    thread.join(timeout=30)
    sampler.stop()

    assert sampler.error is None, f"the reader raised: {sampler.error!r}"
    samples = sampler.samples
    assert len(samples) > 1, "the reader took no useful samples"

    for index, seen in enumerate(samples):
        lip_completed = seen["components"][LIP_COMPONENT] == "completed"
        lip_evidence = LIP_COMPONENT in seen["evidence"]
        assert lip_completed == lip_evidence, (
            f"sample {index} caught LipForensics half-written: "
            f"task={seen['components'][LIP_COMPONENT]} evidence_present={lip_evidence}"
        )

    # The window really was spanned, so the absence above is a finding rather than a miss.
    seen_states = [seen["components"][LIP_COMPONENT] for seen in samples]
    assert "processing" in seen_states, seen_states
    assert "completed" in seen_states, seen_states

    # And nothing about the verdict moved across the transition.
    assert len({seen["verdict"] for seen in samples}) == 1


# --------------------------------------------------------------------------------------
# 2. A report read while another component fails
# --------------------------------------------------------------------------------------


@pytest.mark.integration
def test_rc2_a_component_failing_is_invisible_to_the_verdict_a_reader_sees(
    decided, components, artifact, immediate
):
    """Scenario 2. Effort fails throughout; nothing a reader sees about the verdict moves.

    The failure is a real one on the production path — `persist` concludes the task `failed`
    with the reason `app.detection` recorded — and it happens while the reader is mid-flight.
    Two claims: the five decision fields are byte-identical in every single sample, and the
    failing component is never seen in a state outside its own legal progression. A report
    that qualified or withdrew a verdict because supplementary evidence failed would be §7.4's
    exact prohibition, and it would show up here as a verdict tuple that changed.
    """
    analysis_id = decided()
    stage_and_commit(analysis_id)
    before = decision_columns(analysis_id)

    def enrich():
        with SessionLocal() as session:
            assert enrichment.process_one(session) is True

    samples = run_sampled(analysis_id, enrich)

    verdicts = {seen["verdict"] for seen in samples}
    assert verdicts == {before}, f"the verdict moved under a failing component: {verdicts}"

    effort_states = [
        dict(
            ((provider, signal_type), state)
            for provider, signal_type, state in seen["components"]
        )[EFFORT_COMPONENT]
        for seen in samples
    ]
    assert set(effort_states) <= {"queued", "processing", "failed"}, effort_states
    # Both ends of the transition were observed.
    assert effort_states[0] in {"queued", "processing"}, effort_states
    assert effort_states[-1] == "failed", effort_states

    # And the failure is readable per component rather than as a summary.
    assert task_rows(analysis_id)[EFFORT_COMPONENT] == "failed"
    assert decision_columns(analysis_id) == before


# --------------------------------------------------------------------------------------
# 3. Repeated reads during enrichment never alter the verdict
# --------------------------------------------------------------------------------------


@pytest.mark.integration
def test_rc3_repeated_reads_during_enrichment_write_nothing_and_change_nothing(
    decided, components, artifact, immediate
):
    """Scenario 3. Reading is not a write, asserted at the cursor rather than by inspection.

    Two independent proofs, because the weaker one alone would be easy to satisfy by accident.

    The observable one: the five decision fields are identical before, throughout and after,
    across every one of the reads this test performs while enrichment runs underneath them.

    The structural one: a statement-level hook is attached to the engine for the duration, and
    every statement the *read* path issues is required to be a SELECT. A read path that
    lazily wrote something — a cached projection, a touched timestamp, a backfilled column —
    would be caught here even if its write happened to be idempotent, which is exactly the
    distinction §6.1 draws when it prohibits the write rather than the delta.
    """
    analysis_id = decided()
    stage_and_commit(analysis_id)
    before = decision_columns(analysis_id)

    reading = threading.local()
    offending = []

    def watch_statements(conn, cursor, statement, parameters, context, executemany):
        if getattr(reading, "active", False):
            head = statement.lstrip().split(None, 1)[0].upper()
            if head not in {"SELECT", "BEGIN", "COMMIT", "ROLLBACK", "SET", "SHOW"}:
                offending.append(statement.strip()[:200])

    event.listen(engine, "before_cursor_execute", watch_statements)

    def reader_sample(analysis_id):
        reading.active = True
        try:
            return sample(analysis_id)
        finally:
            reading.active = False

    try:
        def enrich():
            with SessionLocal() as session:
                assert enrichment.process_one(session) is True

        samples = run_sampled(analysis_id, enrich, read=reader_sample)
    finally:
        event.remove(engine, "before_cursor_execute", watch_statements)

    assert offending == [], f"the read path issued a write: {offending}"

    verdicts = {seen["verdict"] for seen in samples}
    assert verdicts == {before}, f"repeated reads moved the verdict: {verdicts}"
    assert decision_columns(analysis_id) == before

    # The reads really did span the enrichment rather than all landing on one side of it.
    enrichment_states = [seen["enrichment_state"] for seen in samples]
    assert len(set(enrichment_states)) > 1, enrichment_states
    assert enrichment_states[-1] == enrichment.ENRICHMENT_PARTIAL, enrichment_states


# --------------------------------------------------------------------------------------
# 4. No half-written signal/segment state
# --------------------------------------------------------------------------------------


@pytest.mark.integration
def test_rc4_a_signal_and_its_segments_become_visible_together(
    decided, components, artifact, immediate
):
    """Scenario 4. §7.5's "no reader ever observes a half-written detector result".

    AASIST is the component with children: the stub gives it eight segments, written after the
    flush that gives the signal row its id and before the single commit that publishes both.
    A reader sampling throughout must see the signal with zero segments in no sample at all —
    the intermediate state exists inside the writing transaction and must never escape it.

    The read is deliberately taken inside one REPEATABLE READ transaction. Under READ
    COMMITTED each statement gets its own snapshot, so a signal select and a segment count
    issued back to back could straddle the commit and report a torn state that no transaction
    ever actually saw — the test would then be failing on its own reading rather than on the
    writer's atomicity.
    """
    analysis_id = decided()
    stage_and_commit(analysis_id)

    def enrich():
        with SessionLocal() as session:
            assert enrichment.process_one(session) is True

    samples = run_sampled(analysis_id, enrich, read=evidence_shape)

    for index, seen in enumerate(samples):
        if AASIST_COMPONENT in seen:
            assert seen[AASIST_COMPONENT] == AASIST_SEGMENTS, (
                f"sample {index} caught AASIST with {seen[AASIST_COMPONENT]} of "
                f"{AASIST_SEGMENTS} segments"
            )

    # The window was spanned: the component was absent at first and whole at the end.
    assert AASIST_COMPONENT not in samples[0], samples[0]
    assert samples[-1][AASIST_COMPONENT] == AASIST_SEGMENTS, samples[-1]


# --------------------------------------------------------------------------------------
# 5. A deferred trigger concurrent with a report read
# --------------------------------------------------------------------------------------


@pytest.mark.integration
def test_rc5_the_deferred_trigger_moves_every_component_or_none_of_them(
    decided, components, artifact, immediate
):
    """Scenario 5. `enrichment.request` is one statement, and a reader can prove it.

    A Quick Scan's four components sit `not_requested`. The trigger moves them with a single
    conditional UPDATE, so no reader may ever see two of them queued beside two that are not:
    that mixed set projects to `ENRICHMENT_PENDING` and would read, on a report, as "some of
    your supplementary evidence was asked for", which is a state the product does not have and
    an operator could not act on.

    The assertion is on the component set rather than on the aggregate, because the aggregate
    would hide the very thing being looked for — a half-applied trigger and a legitimately
    pending one project to the same word.
    """
    analysis_id = decided(mode="quick_scan")
    stage_and_commit(analysis_id, mode="quick_scan")
    assert set(task_rows(analysis_id).values()) == {"not_requested"}

    def trigger():
        # A beat, so the reader is demonstrably mid-flight when the UPDATE lands rather than
        # racing it from a standing start.
        time.sleep(0.2)
        with SessionLocal() as session:
            assert enrichment.request(session, analysis_id) == 4

    samples = run_sampled(analysis_id, trigger)

    for index, seen in enumerate(samples):
        states = {state for _, _, state in seen["components"]}
        assert states in ({"not_requested"}, {"queued"}), (
            f"sample {index} caught the trigger half-applied: {seen['components']}"
        )
        assert seen["enrichment_state"] in (
            enrichment.ENRICHMENT_NOT_REQUESTED,
            enrichment.ENRICHMENT_PENDING,
        ), seen
        assert seen["decision_state"] == enrichment.DECIDED

    aggregates = [seen["enrichment_state"] for seen in samples]
    assert enrichment.ENRICHMENT_NOT_REQUESTED in aggregates, aggregates
    assert enrichment.ENRICHMENT_PENDING in aggregates, aggregates


# --------------------------------------------------------------------------------------
# 6. An enrichment request concurrent with enrichment completion
# --------------------------------------------------------------------------------------


@pytest.mark.integration
def test_rc6_a_request_arriving_as_enrichment_completes_creates_no_work(
    decided, components, artifact, immediate
):
    """Scenario 6. A request is not a retry, including when it lands on the finish line.

    The trigger is fired repeatedly on its own connection while the worker is concluding the
    four components on another. `request` reaches `not_requested` rows and nothing else, so
    every one of those calls must queue zero: a component that is `processing` is not its to
    take, and one that has just reached `completed` or `failed` is finished. A single
    non-zero return here would mean a detector was silently re-run by somebody pressing the
    only button there is.

    The counterpart claim is that nothing was duplicated by the contention: four task rows at
    the end, one per component, and the terminal states the run actually produced.
    """
    analysis_id = decided()
    stage_and_commit(analysis_id)

    queued_counts = []
    stop = threading.Event()

    def trigger():
        while not stop.is_set():
            with SessionLocal() as session:
                queued_counts.append(enrichment.request(session, analysis_id))

    trigger_thread = threading.Thread(target=trigger, daemon=True)
    trigger_thread.start()

    try:
        with SessionLocal() as session:
            assert enrichment.process_one(session) is True
    finally:
        stop.set()
        trigger_thread.join(timeout=60)

    assert queued_counts, "the trigger never ran"
    assert set(queued_counts) == {0}, (
        f"a request created enrichment work during a run: {sorted(set(queued_counts))}"
    )

    rows = task_rows(analysis_id)
    assert len(rows) == 4, rows
    assert rows[LIP_COMPONENT] == "completed"
    assert rows[AASIST_COMPONENT] == "completed"
    assert rows[ASD_COMPONENT] == "completed"
    assert rows[EFFORT_COMPONENT] == "failed"

    with SessionLocal() as session:
        assert enrichment.analysis_states(session, analysis_id) == (
            enrichment.DECIDED,
            enrichment.ENRICHMENT_PARTIAL,
        )


# --------------------------------------------------------------------------------------
# 7. No duplicate task, and no stale projection crash
# --------------------------------------------------------------------------------------


@pytest.mark.integration
def test_rc7_contention_duplicates_no_task_and_crashes_no_projection(
    decided, components, artifact, immediate
):
    """Scenario 7, both halves, under the same contention.

    **No duplicate task.** `stage` is called repeatedly from several threads against an
    analysis that already has its rows, concurrently with triggers and with stale recovery.
    The `(analysis_id, provider, signal_type)` uniqueness is what enrichment retry idempotency
    rests on (§7.3), and a second row for a component would make `ENRICHMENT_PARTIAL`
    ambiguous about which execution it was describing. Exactly four rows must exist at the
    end, and exactly four at every point a reader looked.

    A second `stage` against an already-staged analysis is *expected* to raise, and the
    assertion below says so rather than tolerating a quiet success. The database refusing the
    insert is the mechanism, not a symptom: this test asserts every one of those refusals is a
    unique violation on that named constraint, because a `stage` that returned normally under
    this contention would mean the duplicate had been absorbed somewhere above the constraint
    — and absorbed is exactly what it must not be. Production never calls `stage` twice (it
    runs once, inside the transaction that publishes the verdict), so what is being proven
    here is the floor under that, not the ordinary path.

    The trigger and the recovery are held to the stricter standard: both are statements
    production issues repeatedly and concurrently by design, and neither may raise at all.

    **No stale projection crash.** Recovery rewrites task rows out from under a reader that is
    mid-projection. `analysis_states` and `component_states` must keep answering — with a
    state from their own declared vocabulary and without raising — for every sample. A reader
    that threw here would take the report down for an analysis whose verdict was never in
    doubt, which is §7.4's "a crashed or leaked enrichment worker must never leave an analysis
    looking undecided" seen from the reading side.
    """
    analysis_id = decided()
    stage_and_commit(analysis_id)
    before = decision_columns(analysis_id)

    refused = []
    errors = []
    restaged_without_refusal = []
    stop = threading.Event()

    def restage():
        while not stop.is_set():
            try:
                with SessionLocal() as session:
                    enrichment.stage(session, analysis_id, RULES_V5, None)
                    session.commit()
                restaged_without_refusal.append(True)
            except IntegrityError as error:
                refused.append(error)
            except Exception as error:  # noqa: BLE001 — recorded and asserted on below
                errors.append(("stage", error))

    def retrigger():
        while not stop.is_set():
            try:
                with SessionLocal() as session:
                    enrichment.request(session, analysis_id)
            except Exception as error:  # noqa: BLE001
                errors.append(("request", error))

    def recover():
        while not stop.is_set():
            try:
                with SessionLocal() as session:
                    enrichment.recover_stale_tasks(session)
            except Exception as error:  # noqa: BLE001
                errors.append(("recover", error))

    def projection(analysis_id):
        with SessionLocal() as session:
            states = enrichment.analysis_states(session, analysis_id)
            per_component = enrichment.component_states(session, [analysis_id])
        return states, per_component.get(analysis_id, ())

    sampler = Sampler(analysis_id, read=projection)
    writers = [
        threading.Thread(target=restage, daemon=True),
        threading.Thread(target=restage, daemon=True),
        threading.Thread(target=retrigger, daemon=True),
        threading.Thread(target=recover, daemon=True),
    ]

    sampler.start()
    assert sampler.started_reading.wait(timeout=30), "the sampler never took a sample"
    for writer in writers:
        writer.start()

    # Long enough for the four writers to interleave many times over; short enough that the
    # suite does not pay for it. The contention is continuous, not a single collision.
    time.sleep(3.0)

    stop.set()
    for writer in writers:
        writer.join(timeout=60)
        assert not writer.is_alive(), "a writer did not stop"
    sampler.stop()

    assert errors == [], f"the trigger or the recovery raised under contention: {errors}"
    assert sampler.error is None, f"the projection raised: {sampler.error!r}"

    # Every duplicate `stage` was stopped at the constraint, by name, and none slipped past.
    assert refused, "the contention never produced a second stage"
    assert restaged_without_refusal == [], (
        "a duplicate stage was absorbed above the constraint rather than refused"
    )
    for error in refused:
        assert ENRICHMENT_TASK_COMPONENT_CONSTRAINT in str(error), str(error)
    assert len(sampler.samples) > 1, "the reader took no useful samples"

    for index, ((decision, aggregate), per_component) in enumerate(sampler.samples):
        assert decision in LEGAL_DECISION_STATES, (index, decision)
        assert aggregate in LEGAL_ENRICHMENT_STATES, (index, aggregate)
        # Four components, never three and never five, at every instant a reader looked.
        assert len(per_component) == 4, (index, per_component)
        assert len({(c.provider, c.signal_type) for c in per_component}) == 4, (
            index,
            per_component,
        )

    rows = task_rows(analysis_id)
    assert len(rows) == 4, rows
    assert decision_columns(analysis_id) == before
