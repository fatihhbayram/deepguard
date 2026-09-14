"""The live decision path, after R9-T8B moved it onto `r9-v5.0.0`.

`app.worker.conclude_job` is the one place in the system where a verdict is produced. Until
this task it called `evaluate`; it now calls `evaluate_v5`, and that single swap is what this
file is about. Everything here runs the real function against real PostgreSQL, because what is
under test is not the arithmetic — `tests/test_risk_engine_v5.py` proves the rules as pure
functions, all 36 rows of them — but the production behaviour that arithmetic is wired into:
which analyses get a v5 verdict, which must never get one, and what a reader sees afterwards.

**The worker holds no rules, and that is the first thing proved below.** It fetches three
signals, hands them over, and persists what comes back. It compares no score against any
threshold, counts no coverage, and knows nothing about which detector may decide. A worker
that held a copy of any of that would be a second decision model, and a second model is one
that can disagree with the engine while both look right in isolation.

**The cutover is a boundary in time, not a migration.** An analysis is decided once, when it
reaches `conclude_job`, under whichever ruleset that deployment calls. An analysis already
holding a verdict was decided under the rules that were live when *it* passed through, and
nothing in this deployment goes back for it: no backfill, no re-evaluation, and no mapping of
a stored `HIGH`, `MEDIUM` or `UNKNOWN` onto the R9 vocabulary (R9-T1 invariant 4). The last
section stages a completed `r7-v4.0.0` analysis beside live v5 work and asserts every column
of it, and its rendered trace, comes back exactly as it went in.

**No threshold is mocked anywhere in this module**, for the reason both engine suites give:
`SVD_T_HIGH` and `FACE_T_HIGH` are the R4-T1 study's measured operating points, not knobs. The
scores below are chosen relative to them and the signal rows are written by hand, so the
evidence under test is exact and everything around it can be varied freely.
"""

import hashlib
import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.exc import SQLAlchemyError

from app import risk_engine, worker
from app.db.models import (
    Analysis,
    AnalysisJob,
    AnalysisSignal,
    MediaFile,
    User,
    USER_ROLE_ADMIN,
)
from app.db.session import SessionLocal, engine, get_session
from app.main import app
from app.risk_engine import RiskDecision
from app.web_auth import hash_password, require_user

pytestmark = pytest.mark.integration

# The three deployments the calibrations bind to, restated rather than imported, on the same
# terms as both engine suites: the binding is a fact about a measurement that was taken, not a
# value the code may choose, so an edit to a constant must fail here rather than be agreed with.
VALIDATED_FUNCTION_ID = "847b6e53-0133-452d-ab85-d7acf3ace723"
VALIDATED_FACE_CHECKPOINT = (
    "tomas-gajarsky/facetorch-deepfake-efficientnet-b7"
    "@4acc494f37eb63d7457166eff2acb45c5b04b9a6"
)
VALIDATED_LIP_MODEL = (
    "https://github.com/ahaliassos/LipForensics"
    "@d0bf5553bfb9676f1771d590472b26a3a76de894"
    "+4b7790bc8e02d0c25ecfa0d8d6a2907123c2206cc32e2bad6044e50f013c253d"
)

# What the live path must now stamp on every analysis it concludes, and what it must never
# stamp again. Both are literals: reading them out of the module under test would let a mistaken
# edit to either string pass unnoticed.
RULES_VERSION_V5 = "r9-v5.0.0"
RULES_VERSION_V4 = "r7-v4.0.0"
CALIBRATION_ID = "a74f6b9dbc64cead34cb8e31a03791228cdeb19497e8e5e0bc1a67c0337fc5f7"

MANIPULATION_DETECTED = "MANIPULATION_DETECTED"
NO_SIGNAL = "NO_CALIBRATED_MANIPULATION_SIGNAL"
INCONCLUSIVE = "INCONCLUSIVE"

V5_VERDICTS = frozenset({MANIPULATION_DETECTED, NO_SIGNAL, INCONCLUSIVE})
V4_VERDICTS = frozenset({"HIGH", "MEDIUM", "LOW", "UNKNOWN"})

RULE_V5_HIGH_MULTIPLE = "R9-100"
RULE_V5_HIGH_SVD = "R9-101"
RULE_V5_HIGH_FACE = "R9-102"
RULE_V5_NO_SIGNAL = "R9-200"
RULE_V5_INCONCLUSIVE_PARTIAL = "R9-300"
RULE_V5_INCONCLUSIVE_ALL = "R9-301"

# The operating points R4-T1 selected, restated for the same reason as the deployments above.
EXPECTED_SVD_T_HIGH = 0.9550971388816833
EXPECTED_FACE_T_HIGH = 0.9867589175701141

# One score clear of each threshold and one clear below it. Boundary behaviour is the engine
# suite's business and is proved there against the constants themselves; what these are for is
# to make each staged analysis unambiguously a hit or unambiguously not one.
SVD_ABOVE = 0.9931
SVD_BELOW = 0.4646
FACE_ABOVE = 0.9931
FACE_BELOW = 0.4646

# The aggregate figures a successful call writes into each signal's metadata, as
# `detect_synthetic_video`, `detect_face_manipulation` and `detect_lip_forensics` write them.
TOTAL_CLIPS = 7
LOGIT = 1.9142135381698608
FRAMES_SCORED = 8
WINDOWS_SCORED = 3

# Two real clips from the R4-T1 corpus, and the clearest evidence in the study that neither
# detector may quieten the other: both are genuine manipulations, and on each one detector is
# emphatic while the other sits at its floor.
#
#   sonic_en_03 (talking-head synthesis) — NVIDIA 0.9961, face model 0.0053
#   ffpp_dev_Deepfakes_106_198 (face swap) — NVIDIA 0.1648, face model 0.9943
CORPUS_SYNTHETIC_SVD = 0.9961
CORPUS_SYNTHETIC_FACE = 0.0053
CORPUS_FACESWAP_SVD = 0.1648
CORPUS_FACESWAP_FACE = 0.9943

# The highest score any genuine clip reached in R5-T3's corpus, and the lowest any face swap
# did. Both are mouth-dynamics readings, and under v5 neither may move anything at all.
CORPUS_LIP_GENUINE_MAX = 0.016813907772302628
CORPUS_LIP_FACESWAP_MIN = 0.4424368441104889


# --------------------------------------------------------------------------------------
# Staging
# --------------------------------------------------------------------------------------


@pytest.fixture
def database():
    """The live engine, or a skip when this environment has no PostgreSQL."""
    try:
        with engine.connect():
            pass
    except SQLAlchemyError as error:
        pytest.skip(f"PostgreSQL is not reachable: {error.__class__.__name__}")

    return engine


def persisted_svd_signal(
    analysis_id,
    *,
    score=SVD_BELOW,
    status="SUCCESS",
    provider_version=VALIDATED_FUNCTION_ID,
    total_clips=TOTAL_CLIPS,
) -> AnalysisSignal:
    """A synthetic-video row exactly as `detect_synthetic_video` writes one."""
    return AnalysisSignal(
        analysis_id=analysis_id,
        provider="nvidia",
        signal_type="synthetic_video",
        status=status,
        score=score,
        provider_version=provider_version,
        signal_metadata={"logit": LOGIT, "total_clips": total_clips},
    )


def persisted_face_signal(
    analysis_id,
    *,
    score=FACE_BELOW,
    status="SUCCESS",
    provider_version=VALIDATED_FACE_CHECKPOINT,
    frames_scored=FRAMES_SCORED,
) -> AnalysisSignal:
    """A face-manipulation row exactly as `detect_face_manipulation` writes one."""
    return AnalysisSignal(
        analysis_id=analysis_id,
        provider="efficientnet-b7",
        signal_type="face_manipulation",
        status=status,
        score=score,
        provider_version=provider_version,
        signal_metadata={
            "frames_requested": 8,
            "frames_decoded": frames_scored if isinstance(frames_scored, int) else 8,
            "frames_scored": frames_scored,
            "frame_scores": [],
        },
    )


def persisted_lip_signal(
    analysis_id,
    *,
    score=0.05,
    status="SUCCESS",
    provider_version=VALIDATED_LIP_MODEL,
    windows_scored=WINDOWS_SCORED,
) -> AnalysisSignal:
    """A mouth-dynamics row exactly as `detect_lip_forensics` writes one."""
    return AnalysisSignal(
        analysis_id=analysis_id,
        provider="lipforensics",
        signal_type="lip_forensics",
        status=status,
        score=score,
        provider_version=provider_version,
        signal_metadata={
            "windows_requested": 4,
            "windows_read": 4,
            "windows_scored": windows_scored,
            "window_logits": [],
        },
    )


def context_signals(*, provenance=True, active_speaker=True, audio=True, failed=False):
    """The three signals that must never touch a classification, in a stated arrangement.

    None of them has a calibration of any kind, which is the whole reason they are excluded —
    under v5 as under every ruleset before it. They are varied here so that "excluded" can be
    demonstrated rather than asserted about the code.
    """
    status = "FAILED" if failed else "SUCCESS"
    signals = []

    if provenance:
        signals.append(
            AnalysisSignal(
                provider="c2pa",
                signal_type="provenance",
                status=status,
                provider_version="0.90.14",
                signal_metadata={
                    "manifest_exists": not failed,
                    "validation_state": "Valid" if not failed else None,
                    "signature_issuer": "Test Signing Cert" if not failed else None,
                },
            )
        )
    if active_speaker:
        signals.append(
            AnalysisSignal(
                provider="nvidia",
                signal_type="active_speaker",
                status=status,
                provider_version="f286f937-05c4-454b-8312-fba67a2a6fa7",
                signal_metadata={"total_speaking_segments": 0 if failed else 12},
            )
        )
    if audio:
        signals.append(
            AnalysisSignal(
                provider="aasist",
                signal_type="audio_authenticity",
                status=status,
                provider_version="SpeechAntiSpoofingBenchmarks/AASIST@16774d45",
                signal_metadata={"total_audio_windows": 0 if failed else 3},
            )
        )

    return signals


def detached(signals):
    """Fresh ORM instances of one arrangement, since an instance cannot be attached twice."""
    return [
        AnalysisSignal(
            provider=signal.provider,
            signal_type=signal.signal_type,
            status=signal.status,
            score=signal.score,
            provider_version=signal.provider_version,
            signal_metadata=signal.signal_metadata,
        )
        for signal in signals
    ]


@pytest.fixture
def analysed(database):
    """An analysis whose evidence is already persisted and whose job is still `processing`.

    Exactly the state `conclude_job` is called in: the detectors have run, every signal row is
    committed, and the only thing left is the classification. A media row goes in beside it
    because the API reads below join onto one — the evidence under test is the signals.
    """
    created = []

    def stage(
        *,
        svd_signal=True,
        face_signal=True,
        lip_signal=False,
        svd_kwargs=None,
        face_kwargs=None,
        lip_kwargs=None,
        context=(),
        owner=None,
    ):
        with SessionLocal() as session:
            analysis = Analysis(status="queued", owner_id=owner.id if owner else None)
            session.add(analysis)
            session.flush()

            if svd_signal:
                session.add(persisted_svd_signal(analysis.id, **(svd_kwargs or {})))
            if face_signal:
                session.add(persisted_face_signal(analysis.id, **(face_kwargs or {})))
            if lip_signal:
                session.add(persisted_lip_signal(analysis.id, **(lip_kwargs or {})))
            for signal in context:
                signal.analysis_id = analysis.id
                session.add(signal)

            session.add(media_row(analysis.id))
            session.add(AnalysisJob(analysis_id=analysis.id, status="processing"))
            session.commit()
            created.append(analysis.id)

            return worker.ClaimedJob(
                job_id=session.query(AnalysisJob)
                .filter_by(analysis_id=analysis.id)
                .one()
                .id,
                analysis_id=analysis.id,
                original_storage_key="originals/unused",
                normalization_required=False,
                frame_rate=30.0,
            )

    yield stage

    with SessionLocal() as session:
        for analysis_id in created:
            # Signals, media and the job go with it through ON DELETE CASCADE.
            session.query(Analysis).filter(Analysis.id == analysis_id).delete()
        session.commit()


def media_row(analysis_id) -> MediaFile:
    """The probed media the dashboard read joins onto, with a digest unique to the analysis."""
    digest = hashlib.sha256(str(analysis_id).encode()).hexdigest()

    return MediaFile(
        analysis_id=analysis_id,
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


def read_analysis(analysis_id) -> Analysis:
    with SessionLocal() as reader:
        return reader.query(Analysis).filter_by(id=analysis_id).one()


def read_job(job_id) -> AnalysisJob:
    with SessionLocal() as reader:
        return reader.query(AnalysisJob).filter_by(id=job_id).one()


def read_signals(analysis_id) -> dict[str, AnalysisSignal]:
    with SessionLocal() as reader:
        rows = reader.query(AnalysisSignal).filter_by(analysis_id=analysis_id).all()

    return {row.signal_type: row for row in rows}


def stored_decision(analysis_id) -> tuple:
    """The whole decision as four columns, which is how it is written and how it is read."""
    analysis = read_analysis(analysis_id)

    return (
        analysis.risk_level,
        analysis.risk_rule_id,
        analysis.risk_rules_version,
        analysis.risk_calibration_id,
    )


def conclude(claimed) -> RiskDecision | None:
    with SessionLocal() as session:
        return worker.conclude_job(session, claimed)


@pytest.fixture
def administrator(database):
    """A real administrator account, for the API reads below."""
    with SessionLocal() as session:
        user = User(
            email=f"{uuid.uuid4().hex}@example.com",
            password_hash=hash_password("test-account-password"),
            role=USER_ROLE_ADMIN,
        )
        session.add(user)
        session.commit()
        session.refresh(user)

        yield user

        # Analyses first: `analyses.owner_id` is `ON DELETE RESTRICT`, so the account cannot
        # be removed while one still names it — and the staging fixtures are torn down after
        # this one, not before it.
        session.rollback()
        session.query(Analysis).filter(Analysis.owner_id == user.id).delete()
        session.flush()
        session.query(User).filter(User.id == user.id).delete()
        session.commit()


@pytest.fixture
def reader(administrator):
    """A client over a live session, signed in as an administrator."""
    with SessionLocal() as session:
        app.dependency_overrides[get_session] = lambda: session
        app.dependency_overrides[require_user] = lambda: administrator

        with TestClient(app) as client:
            yield client

        app.dependency_overrides.clear()


# --------------------------------------------------------------------------------------
# Routing: the live path is v5, and the worker holds none of it
# --------------------------------------------------------------------------------------


def test_the_worker_calls_the_v5_engine_and_not_the_v4_one(analysed, monkeypatch):
    """Routing, asserted at the call rather than inferred from the answer.

    Two rulesets can agree on a verdict for one arrangement of evidence, so a test that only
    read the stored `rules_version` could pass while the wrong function was being called for
    the right-looking reason. This one records which function ran and fails `evaluate` outright
    if it is reached from the production path at all.
    """
    calls = []

    real_v5 = risk_engine.evaluate_v5

    def recorded(*evidence):
        calls.append(evidence)
        return real_v5(*evidence)

    monkeypatch.setattr(worker, "evaluate_v5", recorded)
    monkeypatch.setattr(
        risk_engine,
        "evaluate",
        lambda *_, **__: pytest.fail("the live path called the v4 engine"),
    )

    claimed = analysed(svd_kwargs={"score": SVD_ABOVE})
    conclude(claimed)

    assert len(calls) == 1
    assert read_analysis(claimed.analysis_id).risk_rules_version == RULES_VERSION_V5


def test_the_worker_hands_the_engine_the_three_persisted_signals_unchanged(
    analysed, monkeypatch
):
    """Three arguments, in the engine's order, carrying exactly what the database holds.

    The mouth-dynamics evidence is passed even though v5 reads nothing from it. A worker that
    started skipping it because the current ruleset ignores it would be holding a rule of its
    own — and would break the day a ruleset reads it again, silently, by passing `None`.
    """
    captured = []

    real_v5 = risk_engine.evaluate_v5
    monkeypatch.setattr(
        worker,
        "evaluate_v5",
        lambda *evidence: (captured.append(evidence), real_v5(*evidence))[1],
    )

    claimed = analysed(
        svd_kwargs={"score": SVD_ABOVE},
        face_kwargs={"score": FACE_BELOW},
        lip_signal=True,
        lip_kwargs={"score": CORPUS_LIP_FACESWAP_MIN},
    )
    conclude(claimed)

    (svd, face, lip), = captured

    assert (svd.provider, svd.signal_type, svd.score) == (
        "nvidia",
        "synthetic_video",
        SVD_ABOVE,
    )
    assert (face.provider, face.signal_type, face.score) == (
        "efficientnet-b7",
        "face_manipulation",
        FACE_BELOW,
    )
    assert (lip.provider, lip.signal_type, lip.score) == (
        "lipforensics",
        "lip_forensics",
        CORPUS_LIP_FACESWAP_MIN,
    )


def test_the_worker_persists_what_the_engine_returned_without_inspecting_it(
    analysed, monkeypatch
):
    """The constraint this task is built around, stated as an experiment.

    The engine is replaced by one returning a decision no ruleset in this codebase can produce
    — an unknown verdict, an unknown rule, an unknown ruleset — over evidence whose real v5
    answer is `MANIPULATION_DETECTED`. All four columns come back as the engine gave them.

    If the worker held a threshold comparison, a coverage count, a vocabulary check or a
    fallback of its own, this would be where it showed: the stored row would be something
    other than what was handed over. It persists; it does not decide.
    """
    fabricated = RiskDecision(
        risk_level="A_VERDICT_NO_RULESET_DEFINES",
        rules_version="not-a-ruleset",
        calibration_id="not-a-calibration",
        rule_id="not-a-rule",
    )
    monkeypatch.setattr(worker, "evaluate_v5", lambda *_: fabricated)

    claimed = analysed(svd_kwargs={"score": SVD_ABOVE}, face_kwargs={"score": FACE_ABOVE})
    conclude(claimed)

    assert stored_decision(claimed.analysis_id) == (
        fabricated.risk_level,
        fabricated.rule_id,
        fabricated.rules_version,
        fabricated.calibration_id,
    )


def test_the_returned_decision_and_the_stored_row_are_the_same_decision(analysed):
    """What `conclude_job` hands back is what it wrote, in all four fields."""
    claimed = analysed(svd_kwargs={"score": SVD_ABOVE})

    decision = conclude(claimed)

    assert decision is not None
    assert stored_decision(claimed.analysis_id) == (
        decision.risk_level,
        decision.rule_id,
        decision.rules_version,
        decision.calibration_id,
    )


def test_a_pending_analysis_is_decided_only_when_it_reaches_the_worker(analysed):
    """Nothing is classified while the job is in progress, and nothing completes unclassified.

    This is the cutover boundary from the other side: an analysis submitted before this
    deployment and still queued carries no verdict at all until `conclude_job` runs, and the
    verdict it then gets is this deployment's — which is why a pending job crossing the
    deployment receives v5 and a completed one keeps what it had.
    """
    claimed = analysed(svd_kwargs={"score": SVD_BELOW})

    before = read_analysis(claimed.analysis_id)
    assert before.status == "queued"
    assert stored_decision(claimed.analysis_id) == (None, None, None, None)

    conclude(claimed)

    assert read_job(claimed.job_id).status == "completed"
    after = read_analysis(claimed.analysis_id)
    assert after.status == "completed"
    assert after.risk_level == NO_SIGNAL
    assert after.risk_rules_version == RULES_VERSION_V5


# --------------------------------------------------------------------------------------
# The verdicts the live path now produces
# --------------------------------------------------------------------------------------

# Every permutation of what the two decision-eligible detectors can be in production, and the
# verdict the live path must persist for it. Written out row by row rather than derived, so a
# defect cannot be agreed with by recomputing it the same wrong way the engine did.
#
# The unusable states are the real ones the pipeline writes: a `FAILED` row with a null score is
# how both a failure and an abstention are recorded; a `SUCCESS` row from a deployment the
# thresholds were never measured against is R6-T1's shadow-mode risk; and a mean taken over zero
# units is a `SUCCESS` row carrying no reading.
SVD_STATES = {
    "hit": {"svd_kwargs": {"score": SVD_ABOVE}},
    "usable, below": {"svd_kwargs": {"score": SVD_BELOW}},
    "failed": {"svd_kwargs": {"score": None, "status": "FAILED"}},
    "absent": {"svd_signal": False},
    "uncalibrated": {
        "svd_kwargs": {"score": SVD_ABOVE, "provider_version": "some-other-function"}
    },
    "no clips": {"svd_kwargs": {"score": SVD_ABOVE, "total_clips": 0}},
}

FACE_STATES = {
    "hit": {"face_kwargs": {"score": FACE_ABOVE}},
    "usable, below": {"face_kwargs": {"score": FACE_BELOW}},
    "failed": {"face_kwargs": {"score": None, "status": "FAILED"}},
    "absent": {"face_signal": False},
    "uncalibrated": {
        "face_kwargs": {"score": FACE_ABOVE, "provider_version": "some-other-checkpoint"}
    },
    "no frames": {"face_kwargs": {"score": FACE_ABOVE, "frames_scored": 0}},
}

UNRESOLVED_SVD = ("failed", "absent", "uncalibrated", "no clips")
UNRESOLVED_FACE = ("failed", "absent", "uncalibrated", "no frames")


def expected_v5(svd_state: str, face_state: str) -> tuple[str, str]:
    """The verdict and rule the contract names for one pair of detector states.

    A table rather than a re-implementation: it branches on the *names* of the states, never
    on a score, a threshold or a count, so it cannot reach agreement with the engine by
    repeating its arithmetic.
    """
    svd_hit = svd_state == "hit"
    face_hit = face_state == "hit"

    if svd_hit and face_hit:
        return MANIPULATION_DETECTED, RULE_V5_HIGH_MULTIPLE
    if svd_hit:
        return MANIPULATION_DETECTED, RULE_V5_HIGH_SVD
    if face_hit:
        return MANIPULATION_DETECTED, RULE_V5_HIGH_FACE

    svd_usable = svd_state not in UNRESOLVED_SVD
    face_usable = face_state not in UNRESOLVED_FACE

    if svd_usable and face_usable:
        return NO_SIGNAL, RULE_V5_NO_SIGNAL
    if svd_usable or face_usable:
        return INCONCLUSIVE, RULE_V5_INCONCLUSIVE_PARTIAL

    return INCONCLUSIVE, RULE_V5_INCONCLUSIVE_ALL


@pytest.mark.parametrize("face_state", list(FACE_STATES))
@pytest.mark.parametrize("svd_state", list(SVD_STATES))
def test_every_permutation_of_the_two_deciders_lands_where_the_contract_says(
    analysed, svd_state, face_state
):
    """All 36 of them, through the database, with the whole trace asserted each time.

    The rule id is asserted beside the verdict because it is what tells a later reader *how
    much was read*: `R9-200` and `R9-300` can both sit under an analysis nobody flagged, and
    the difference between "both detectors were read and neither flagged" and "we could barely
    read anything" is exactly what the v4 vocabulary could not say.
    """
    claimed = analysed(**SVD_STATES[svd_state], **FACE_STATES[face_state])

    conclude(claimed)

    verdict, rule_id = expected_v5(svd_state, face_state)
    assert stored_decision(claimed.analysis_id) == (
        verdict,
        rule_id,
        RULES_VERSION_V5,
        CALIBRATION_ID,
    )


def test_a_synthetic_video_finding_alone_is_manipulation_detected(analysed):
    """`sonic_en_03`, through the database: NVIDIA emphatic, the face model at its floor.

    The face model reporting 0.0053 does not soften the verdict and does not reduce it to a
    coverage statement. A rule that let the quiet detector speak would have missed this clip,
    which is a real manipulation.
    """
    claimed = analysed(
        svd_kwargs={"score": CORPUS_SYNTHETIC_SVD},
        face_kwargs={"score": CORPUS_SYNTHETIC_FACE},
        context=context_signals(),
    )

    conclude(claimed)

    assert stored_decision(claimed.analysis_id) == (
        MANIPULATION_DETECTED,
        RULE_V5_HIGH_SVD,
        RULES_VERSION_V5,
        CALIBRATION_ID,
    )


def test_a_face_finding_alone_is_manipulation_detected_with_no_svd_row_at_all(analysed):
    """`ffpp_dev_Deepfakes_106_198`, with the synthetic-video detector missing entirely.

    A hit is never held back by missing coverage: the detector that did read reached its own
    operating point, and that is a finding whatever the other detector did or did not do.
    """
    claimed = analysed(
        svd_signal=False,
        face_kwargs={"score": CORPUS_FACESWAP_FACE},
        context=context_signals(),
    )

    conclude(claimed)

    assert stored_decision(claimed.analysis_id) == (
        MANIPULATION_DETECTED,
        RULE_V5_HIGH_FACE,
        RULES_VERSION_V5,
        CALIBRATION_ID,
    )


def test_full_coverage_with_no_hit_is_not_a_finding_of_authenticity(analysed):
    """Both detectors read, neither flagged — `NO_CALIBRATED_MANIPULATION_SIGNAL`, by name.

    The assertion is on the exact stored string, and on it not being any of the words this
    verdict is not. A row that came back `MEDIUM`, or as anything meaning "clean", would be
    reporting a conclusion the detectors did not reach (R9-T1 invariant 3).
    """
    claimed = analysed(
        svd_kwargs={"score": SVD_BELOW},
        face_kwargs={"score": FACE_BELOW},
        context=context_signals(),
    )

    conclude(claimed)

    analysis = read_analysis(claimed.analysis_id)

    assert analysis.risk_level == NO_SIGNAL
    assert analysis.risk_rule_id == RULE_V5_NO_SIGNAL
    assert analysis.risk_level not in V4_VERDICTS


def test_zero_coverage_is_inconclusive_and_no_context_signal_rescues_it(analysed):
    """A perfect provenance chain, a full speaker timeline and clean audio change nothing.

    `INCONCLUSIVE` is not a gap for uncalibrated evidence to fill in. Those three signals are
    forensic evidence in their own right and none of them has an operating point, so none of
    them may complete coverage that the calibrated detectors did not produce.
    """
    claimed = analysed(
        svd_kwargs={"score": 0.9999, "provider_version": f"{VALIDATED_FUNCTION_ID}-v2"},
        face_kwargs={
            "score": 0.9999,
            "provider_version": f"{VALIDATED_FACE_CHECKPOINT}-v2",
        },
        lip_signal=True,
        lip_kwargs={"score": 0.9999},
        context=context_signals(),
    )

    conclude(claimed)

    assert stored_decision(claimed.analysis_id) == (
        INCONCLUSIVE,
        RULE_V5_INCONCLUSIVE_ALL,
        RULES_VERSION_V5,
        CALIBRATION_ID,
    )


def test_partial_coverage_is_inconclusive_rather_than_a_quiet_no_signal(analysed):
    """One detector read and did not flag; the other never answered.

    Under v4 both this and full coverage landed on `MEDIUM`, separated only by the rule id.
    Here they are different verdicts, so the distinction survives a reader that shows the
    level and drops the trace — which is the whole reason v5 exists.
    """
    claimed = analysed(
        svd_kwargs={"score": SVD_BELOW},
        face_kwargs={"score": None, "status": "FAILED"},
        context=context_signals(),
    )

    conclude(claimed)

    assert stored_decision(claimed.analysis_id) == (
        INCONCLUSIVE,
        RULE_V5_INCONCLUSIVE_PARTIAL,
        RULES_VERSION_V5,
        CALIBRATION_ID,
    )


def test_the_decision_never_lands_on_a_signal_row(analysed):
    """The verdict goes on the analysis. No provider is labelled with one."""
    claimed = analysed(
        svd_kwargs={"score": SVD_ABOVE},
        face_kwargs={"score": FACE_ABOVE},
        lip_signal=True,
        lip_kwargs={"score": 0.99},
        context=context_signals(),
    )

    before = {
        signal_type: (row.status, row.score, row.provider_version, row.risk_level)
        for signal_type, row in read_signals(claimed.analysis_id).items()
    }

    conclude(claimed)

    after = {
        signal_type: (row.status, row.score, row.provider_version, row.risk_level)
        for signal_type, row in read_signals(claimed.analysis_id).items()
    }

    assert after == before
    assert all(row.risk_level is None for row in read_signals(claimed.analysis_id).values())


# --------------------------------------------------------------------------------------
# Orthogonality: what may not move a v5 verdict
# --------------------------------------------------------------------------------------

# Every state the mouth-dynamics detector can be in, including the two that would be findings
# if it were allowed to make one. Under v5 it is `evidence_only` and this is total: its score,
# its status, its deployment identity and its very presence are outside the arithmetic.
LIP_STATES = {
    "absent": {"lip_signal": False},
    "at the top of its scale": {"lip_signal": True, "lip_kwargs": {"score": 1.0}},
    "lowest face swap in R5-T3": {
        "lip_signal": True,
        "lip_kwargs": {"score": CORPUS_LIP_FACESWAP_MIN},
    },
    "highest genuine in R5-T3": {
        "lip_signal": True,
        "lip_kwargs": {"score": CORPUS_LIP_GENUINE_MAX},
    },
    "failed": {
        "lip_signal": True,
        "lip_kwargs": {"score": None, "status": "FAILED"},
    },
    "timed out": {
        "lip_signal": True,
        "lip_kwargs": {"score": 0.6584, "status": "TIMEOUT"},
    },
    "uncalibrated deployment": {
        "lip_signal": True,
        "lip_kwargs": {"score": 0.99, "provider_version": "some-other-checkpoint"},
    },
    "no windows scored": {
        "lip_signal": True,
        "lip_kwargs": {"score": 0.99, "windows_scored": 0},
    },
}


@pytest.mark.parametrize(
    ("svd_kwargs", "face_kwargs"),
    [
        ({"score": SVD_ABOVE}, {"score": FACE_BELOW}),
        ({"score": SVD_BELOW}, {"score": FACE_ABOVE}),
        ({"score": SVD_BELOW}, {"score": FACE_BELOW}),
        ({"score": SVD_BELOW}, {"score": None, "status": "FAILED"}),
        ({"score": None, "status": "FAILED"}, {"score": None, "status": "FAILED"}),
    ],
)
def test_no_state_of_the_mouth_dynamics_detector_moves_a_persisted_verdict(
    analysed, svd_kwargs, face_kwargs
):
    """Eight states of a detector that may not decide, over five arrangements that can.

    Including `uncalibrated deployment`, which is the one that could have raised: v5 refuses
    uncalibrated evidence in the two slots that decide and deliberately does not check the
    third. Had it checked, that detector's deployment metadata would decide whether this
    analysis gets a verdict at all — which is the influence R9-T1 invariant 1 denies it.

    Every column is compared, not just the verdict. A mouth-dynamics reading that changed only
    the rule id would still be an evidence-only detector altering the record.
    """
    decisions = {}

    for name, lip_state in LIP_STATES.items():
        claimed = analysed(svd_kwargs=svd_kwargs, face_kwargs=face_kwargs, **lip_state)
        conclude(claimed)
        decisions[name] = stored_decision(claimed.analysis_id)

    assert len(set(decisions.values())) == 1, decisions
    assert set(decisions) == set(LIP_STATES)


# Identical calibrated evidence, each entry a different arrangement of everything without a
# calibration. Removed, present, failed, and every combination in between.
CONTEXT_ARRANGEMENTS = {
    "all present and successful": context_signals(),
    "all present and failed": context_signals(failed=True),
    "none at all": [],
    "no provenance": context_signals(provenance=False),
    "no active speaker": context_signals(active_speaker=False),
    "no audio authenticity": context_signals(audio=False),
    "provenance only": context_signals(active_speaker=False, audio=False),
    "provenance failed": context_signals(active_speaker=False, audio=False, failed=True),
    "active speaker only": context_signals(provenance=False, audio=False),
    "active speaker failed": context_signals(provenance=False, audio=False, failed=True),
    "audio only": context_signals(provenance=False, active_speaker=False),
    "audio failed": context_signals(provenance=False, active_speaker=False, failed=True),
}


@pytest.mark.parametrize(
    ("svd_score", "face_score", "verdict", "rule_id"),
    [
        (SVD_ABOVE, FACE_BELOW, MANIPULATION_DETECTED, RULE_V5_HIGH_SVD),
        (SVD_BELOW, FACE_ABOVE, MANIPULATION_DETECTED, RULE_V5_HIGH_FACE),
        (SVD_ABOVE, FACE_ABOVE, MANIPULATION_DETECTED, RULE_V5_HIGH_MULTIPLE),
        (SVD_BELOW, FACE_BELOW, NO_SIGNAL, RULE_V5_NO_SIGNAL),
    ],
)
def test_no_arrangement_of_the_uncalibrated_signals_changes_a_v5_verdict(
    analysed, svd_score, face_score, verdict, rule_id
):
    """The isolation requirement, against the live path, exhaustively.

    Every arrangement of C2PA, the speaker timeline and AASIST — present, absent, successful,
    failed — over one unchanged pair of calibrated scores. Nothing here is averaged, voted on,
    weighted or combined, and this is the test that would fail the moment something started
    to be.
    """
    decisions = {}

    for name, context in CONTEXT_ARRANGEMENTS.items():
        claimed = analysed(
            svd_kwargs={"score": svd_score},
            face_kwargs={"score": face_score},
            context=detached(context),
        )
        conclude(claimed)
        decisions[name] = stored_decision(claimed.analysis_id)

    expected = (verdict, rule_id, RULES_VERSION_V5, CALIBRATION_ID)

    assert decisions == {name: expected for name in CONTEXT_ARRANGEMENTS}


# --------------------------------------------------------------------------------------
# Round trip: a v5 verdict, as its readers get it
# --------------------------------------------------------------------------------------


def test_a_newly_decided_v5_analysis_reads_back_through_the_api_intact(
    analysed, administrator, reader
):
    """Decided by the worker, read by the dashboard: the same four values, unabbreviated.

    Deliberately end to end rather than staged: `tests/test_verdict_persistence.py` already
    proves a hand-written v5 verdict survives the column. What this adds is that the verdict
    the *worker* produced is the one a reader is shown — no truncation, no re-labelling and no
    mapping onto a v4 word on the way out.
    """
    claimed = analysed(
        svd_kwargs={"score": SVD_BELOW},
        face_kwargs={"score": FACE_BELOW},
        lip_signal=True,
        owner=administrator,
    )

    conclude(claimed)

    response = reader.get(f"/api/v1/analyses/{claimed.analysis_id}")

    assert response.status_code == 200
    body = response.json()
    assert body["risk_level"] == NO_SIGNAL
    assert body["risk_rules_version"] == RULES_VERSION_V5
    assert body["risk_rule_id"] == RULE_V5_NO_SIGNAL
    assert body["risk_calibration_id"] == CALIBRATION_ID


def test_the_trace_explains_a_worker_produced_v5_verdict_under_v5_rules(
    analysed, administrator, reader
):
    """The RiskTrace over a live decision: coverage, the eligible pair, and the evidence-only one.

    `decision_coverage` is 2/2 here because both deciding detectors were read — a count of
    readings, not of findings, which is why it sits under a `MANIPULATION_DETECTED` exactly as
    it would under any other verdict. The mouth-dynamics detector is present in the trace and
    on the supplementary side of the split, which is v5's own `decisional` flag showing through
    rather than a choice this reader made.
    """
    claimed = analysed(
        svd_kwargs={"score": SVD_ABOVE},
        face_kwargs={"score": FACE_BELOW},
        lip_signal=True,
        lip_kwargs={"score": CORPUS_LIP_FACESWAP_MIN},
        owner=administrator,
    )

    conclude(claimed)

    body = reader.get(f"/api/v1/analyses/{claimed.analysis_id}").json()
    trace = body["risk_trace"]

    assert trace["risk_level"] == MANIPULATION_DETECTED
    assert trace["rules_version"] == RULES_VERSION_V5
    assert trace["rule_id"] == RULE_V5_HIGH_SVD
    assert trace["calibration_id"] == CALIBRATION_ID
    assert trace["interpreted"] is True
    assert trace["rule_summary"]

    assert trace["decision_coverage"] == {
        "usable": 2,
        "total": 2,
        "is_complete": True,
        "status": "complete",
    }

    eligible = {
        contribution["signal"]: contribution
        for contribution in trace["decision_eligible_detectors"]
    }
    supplementary = {
        contribution["signal"] for contribution in trace["supplementary_evidence"]
    }

    assert set(eligible) == {"synthetic_video", "face_manipulation"}
    assert supplementary == {"lip_forensics"}

    assert eligible["synthetic_video"]["condition"] == "threshold_reached"
    assert eligible["synthetic_video"]["role"] == "decisive"
    assert eligible["synthetic_video"]["threshold"] == EXPECTED_SVD_T_HIGH
    assert eligible["face_manipulation"]["condition"] == "threshold_not_reached"
    assert eligible["face_manipulation"]["role"] == "considered"
    assert eligible["face_manipulation"]["threshold"] == EXPECTED_FACE_T_HIGH


def test_partial_coverage_is_visible_in_the_trace_of_a_live_decision(
    analysed, administrator, reader
):
    """1 of 2, beside `INCONCLUSIVE`, with the reason the second detector contributed nothing.

    The coverage fraction and the verdict are read from the same persisted row, so a reader is
    never left to infer how much was read from the word alone.
    """
    claimed = analysed(
        svd_kwargs={"score": SVD_BELOW},
        face_kwargs={"score": None, "status": "FAILED"},
        owner=administrator,
    )

    conclude(claimed)

    trace = reader.get(f"/api/v1/analyses/{claimed.analysis_id}").json()["risk_trace"]

    assert trace["risk_level"] == INCONCLUSIVE
    assert trace["rule_id"] == RULE_V5_INCONCLUSIVE_PARTIAL
    assert trace["decision_coverage"] == {
        "usable": 1,
        "total": 2,
        "is_complete": False,
        "status": "partial",
    }

    face = next(
        contribution
        for contribution in trace["decision_eligible_detectors"]
        if contribution["signal"] == "face_manipulation"
    )
    assert face["condition"] == "unavailable"
    assert face["unavailable_reason"] == "detector_did_not_report"
    assert face["score"] is None


# --------------------------------------------------------------------------------------
# Immutable history: what the cutover may not touch
# --------------------------------------------------------------------------------------


@pytest.fixture
def completed_v4_analysis(database, administrator):
    """A finished `r7-v4.0.0` analysis, exactly as one decided before the cutover stands.

    Its job is `completed` and its analysis is `completed`, which is what makes it history
    rather than work: the pipeline is done with it and no worker holds a claim on it.
    """
    with SessionLocal() as session:
        analysis = Analysis(
            status="completed",
            owner_id=administrator.id,
            risk_level="HIGH",
            risk_rules_version=RULES_VERSION_V4,
            risk_rule_id="R100",
            risk_calibration_id=CALIBRATION_ID,
        )
        session.add(analysis)
        session.flush()

        session.add(persisted_svd_signal(analysis.id, score=CORPUS_SYNTHETIC_SVD))
        session.add(persisted_face_signal(analysis.id, score=CORPUS_SYNTHETIC_FACE))
        session.add(persisted_lip_signal(analysis.id, score=CORPUS_LIP_GENUINE_MAX))
        session.add(media_row(analysis.id))
        session.add(AnalysisJob(analysis_id=analysis.id, status="completed"))
        session.commit()
        analysis_id = analysis.id

    yield analysis_id

    with SessionLocal() as session:
        session.query(Analysis).filter(Analysis.id == analysis_id).delete()
        session.commit()


def test_a_completed_v4_analysis_is_untouched_while_v5_work_runs_beside_it(
    analysed, completed_v4_analysis
):
    """The cutover rewrites nothing. Deploying v5 is not a migration of what came before.

    A finished v4 analysis is staged, several v5 analyses are then decided in the same
    database, and every column of the old row is compared before and after — including its
    status and the evidence under it. Nothing in this deployment goes looking for history.
    """
    before = stored_decision(completed_v4_analysis)
    signals_before = {
        signal_type: (row.status, row.score, row.provider_version)
        for signal_type, row in read_signals(completed_v4_analysis).items()
    }

    for svd_kwargs, face_kwargs in (
        ({"score": SVD_ABOVE}, {"score": FACE_BELOW}),
        ({"score": SVD_BELOW}, {"score": FACE_BELOW}),
        ({"score": None, "status": "FAILED"}, {"score": None, "status": "FAILED"}),
    ):
        conclude(analysed(svd_kwargs=svd_kwargs, face_kwargs=face_kwargs))

    assert stored_decision(completed_v4_analysis) == before
    assert before == ("HIGH", "R100", RULES_VERSION_V4, CALIBRATION_ID)
    assert read_analysis(completed_v4_analysis).status == "completed"
    assert {
        signal_type: (row.status, row.score, row.provider_version)
        for signal_type, row in read_signals(completed_v4_analysis).items()
    } == signals_before


def test_a_v4_verdict_is_never_relabelled_into_the_r9_vocabulary(completed_v4_analysis):
    """`HIGH` stays `HIGH`. It does not become `MANIPULATION_DETECTED`.

    The two vocabularies are resolved through `risk_rules_version` and are never merged: a
    stored v4 word carries what v4 meant by it, and mapping it onto a v5 verdict would be
    asserting a decision nobody took (R9-T1 invariant 4).
    """
    verdict, _, rules_version, _ = stored_decision(completed_v4_analysis)

    assert verdict in V4_VERDICTS
    assert verdict not in V5_VERDICTS
    assert rules_version == RULES_VERSION_V4


def test_a_completed_analysis_cannot_be_re_decided_by_a_worker(completed_v4_analysis):
    """Even driven at it directly, `conclude_job` refuses to overwrite a finished analysis.

    The guard is the one that already keeps stale recovery honest: the status update is
    conditional on the job still being `processing`, so a job that has already finished
    matches nothing, `_set_status` reports the loss, and the transaction is rolled back. That
    is what makes historical immutability a property of the code rather than of the fact that
    nothing currently calls this — a replay, a re-queue or a resurrected worker all land here.
    """
    with SessionLocal() as session:
        job = session.query(AnalysisJob).filter_by(analysis_id=completed_v4_analysis).one()
        claimed = worker.ClaimedJob(
            job_id=job.id,
            analysis_id=completed_v4_analysis,
            original_storage_key="originals/unused",
            normalization_required=False,
            frame_rate=30.0,
        )

    assert conclude(claimed) is None

    assert stored_decision(completed_v4_analysis) == (
        "HIGH",
        "R100",
        RULES_VERSION_V4,
        CALIBRATION_ID,
    )
    assert read_job(claimed.job_id).status == "completed"


def test_a_v4_analysis_still_reads_and_explains_itself_under_v4_rules(
    completed_v4_analysis, reader
):
    """The report of an old analysis is unchanged by the cutover, trace included.

    `app.risk_trace` resolves a decision through the `risk_rules_version` on its own row, so
    this one is explained with `r7-v4.0.0` thresholds and `r7-v4.0.0` rule meanings. Under
    that ruleset the mouth-dynamics detector is supplementary and no coverage was stated at
    all, and `decision_coverage` is null rather than zero: a decision taken without a
    denominator makes no coverage claim, which is a different fact from a coverage of none.
    """
    body = reader.get(f"/api/v1/analyses/{completed_v4_analysis}").json()

    assert body["risk_level"] == "HIGH"
    assert body["risk_rules_version"] == RULES_VERSION_V4

    trace = body["risk_trace"]
    assert trace["risk_level"] == "HIGH"
    assert trace["rules_version"] == RULES_VERSION_V4
    assert trace["rule_id"] == "R100"
    assert trace["interpreted"] is True
    assert trace["decision_coverage"] is None

    assert {
        contribution["signal"] for contribution in trace["decision_eligible_detectors"]
    } == {"synthetic_video", "face_manipulation"}
    assert {
        contribution["signal"] for contribution in trace["supplementary_evidence"]
    } == {"lip_forensics"}
