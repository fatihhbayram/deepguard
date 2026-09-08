"""Effort is persisted, and Effort decides nothing (R7-T11).

R7-T10 evaluated three integration postures for the Effort detector against the measurements of
R7-T7, R7-T8 and R7-T9, and adopted the evidence-only one: Effort runs in the production analysis
path, records what it observed, and is read by no rule. This module is the proof of the second
half, which is the half that is easy to lose by accident.

The tests below fall into two groups, and the split is the point:

- **that the reading is recorded properly** — under an identity that collides with nothing, with
  the checkpoint digest and upstream revision that produced it, and with abstention, failure and
  success kept as three different facts rather than three shades of one number;

- **that the reading reaches nothing** — no ruleset lists it, no threshold exists for it, no
  `calibration_id` was minted for it, the risk engine cannot be handed it, and a successful,
  failed or abstained Effort leaves the decision on an otherwise identical analysis bit-for-bit
  where it was.

The second group is deliberately written as *comparisons against the same analysis without
Effort* rather than as assertions about what the level ought to be. A test that says "still
UNKNOWN" passes for the wrong reason the day a rule starts reading this signal and happens to
produce UNKNOWN anyway; a test that says "the same as it was without this row" does not.
"""

from pathlib import Path

import pytest

from app import detection, effort, risk_engine, risk_trace
from app.db.models import (
    SIGNAL_STATUS_FAILED,
    SIGNAL_STATUS_SUCCESS,
)
from app.detection import (
    EFFICIENTNET_B7_PROVIDER,
    EFFORT_PROVIDER,
    FACE_FORGERY_SIGNAL,
    FACE_MANIPULATION_SIGNAL,
    LIP_FORENSICS_SIGNAL,
    LIPFORENSICS_PROVIDER,
    NVIDIA_PROVIDER,
    SYNTHETIC_VIDEO_SIGNAL,
    detect_effort,
)
from app.effort import (
    EffortEvidence,
    EffortInferenceError,
    EffortMediaError,
    EffortModelUnavailable,
    EffortNoFaceDetected,
)
from app.risk_engine import FaceEvidence, LipEvidence, SvdEvidence

# The identity R7-T7 froze and R7-T11 transcribed. Restated here rather than imported from
# `app.effort` for the reason the freeze document gives about its own duplication: a test that
# imports the value it is checking cannot notice the value changing.
EFFORT_REPOSITORY = "https://github.com/YZY-stack/Effort-AIGI-Detection"
EFFORT_REVISION = "96f5dea2b534d400cfd7003f053c7e93c8e16461"
EFFORT_CHECKPOINT_SHA256 = (
    "8d86711f098d16b49c048962fc3e16a906380f7bb17b8a0e89bd545b926943ee"
)
EFFORT_LANDMARK_SHA256 = (
    "8cae4375589dd915d9a0a881101bed1bbb4e9887e35e63b024388f1ca25ff869"
)

# The candidate operating point R7-T9 measured, and the one number this integration may not
# contain. It is an offline artifact of one confirmation on 388 calibration lineages — the
# highest readable genuine score in that sample, which is an order statistic of one sample — and
# R7-T10 §5 declined to adopt it. Named here only so the tests can assert it never appears.
R7T9_EVALUATION_THRESHOLD = 0.9574909508228302


def evidence(**overrides) -> EffortEvidence:
    """One ordinary Effort reading: eight frames sampled, six of them with a face."""
    fields = {
        "upstream_repository": EFFORT_REPOSITORY,
        "upstream_revision": EFFORT_REVISION,
        "checkpoint_filename": "effort_clip_L14_trainOn_FaceForensic.pth",
        "checkpoint_sha256": EFFORT_CHECKPOINT_SHA256,
        "landmark_model_sha256": EFFORT_LANDMARK_SHA256,
        "clip_repository": "https://huggingface.co/openai/clip-vit-large-patch14",
        "clip_revision": "32bd64288804d66eefd0ccbe215aa642df71cc41",
        "clip_sha256": {"config.json": "8a09b467" + "0" * 56},
        "runtime": {
            "python": "3.12.14",
            "torch": "2.13.0+cpu",
            "torchvision": "0.28.0+cpu",
            "dlib": "20.0.1",
            "transformers": "5.16.1",
        },
        "torch_version": "2.13.0+cpu",
        "device": "cpu",
        "missing_forward_path_key_count": 0,
        "frames_per_clip": 8,
        "frames_requested": 8,
        "frames_decoded": 8,
        "frames_with_face": 6,
        "score": 0.4787200441,
        "frame_scores": (0.41, 0.52, 0.48, 0.44, 0.51, 0.51),
    }
    fields.update(overrides)

    return EffortEvidence(**fields)


@pytest.fixture
def scored(monkeypatch, tmp_path):
    """`detect_effort` over a stubbed model that returns `evidence()`."""

    def analyze(_path, **_kwargs):
        return evidence()

    monkeypatch.setattr(detection, "analyze_effort", analyze)
    artifact = tmp_path / "clip.mp4"
    artifact.write_bytes(b"video")

    return detect_effort(artifact)


def refusing(monkeypatch, tmp_path, error):
    """`detect_effort` over a model that raises `error`."""

    def analyze(_path, **_kwargs):
        raise error

    monkeypatch.setattr(detection, "analyze_effort", analyze)
    artifact = tmp_path / "clip.mp4"
    artifact.write_bytes(b"video")

    return detect_effort(artifact)


# --- 1. The signal is persisted under an identity of its own -------------------------------


def test_effort_is_written_under_its_own_provider_and_signal_type(scored):
    assert scored.provider == EFFORT_PROVIDER == "effort"
    assert scored.signal_type == FACE_FORGERY_SIGNAL == "face_forgery"


def test_effort_does_not_collide_with_any_other_detector_identity(scored):
    """The pair is unique across every detector this pipeline persists.

    R7-T10 §3 is why this is a test rather than a naming convention: `app.risk_engine` selects
    evidence by exact `(provider, signal_type)` pair, so a shared pair would make every existing
    reader ambiguous about which row it had selected — and a reader that picked the wrong one
    would band an Effort score against the B7's threshold, which was measured on a 54-clip corpus
    against a different model's score distribution.
    """
    others = {
        (NVIDIA_PROVIDER, SYNTHETIC_VIDEO_SIGNAL),
        (NVIDIA_PROVIDER, detection.ACTIVE_SPEAKER_SIGNAL),
        (detection.C2PA_PROVIDER, detection.PROVENANCE_SIGNAL),
        (detection.AASIST_PROVIDER, detection.AUDIO_AUTHENTICITY_SIGNAL),
        (EFFICIENTNET_B7_PROVIDER, FACE_MANIPULATION_SIGNAL),
        (LIPFORENSICS_PROVIDER, LIP_FORENSICS_SIGNAL),
    }

    assert (EFFORT_PROVIDER, FACE_FORGERY_SIGNAL) not in others


def test_effort_does_not_reuse_the_b7_signal_type(scored):
    """The specific collision R7-T10 §3 names, asserted on its own.

    The `provider` differing is not enough and is not what is checked here. B7's `signal_type` is
    the string a threshold lookup keys on, and the two detectors measure the same target on
    different scales — which is exactly the pair of properties that makes a shared type dangerous
    rather than merely untidy.
    """
    assert scored.signal_type != FACE_MANIPULATION_SIGNAL
    assert FACE_FORGERY_SIGNAL != FACE_MANIPULATION_SIGNAL


def test_a_successful_reading_carries_the_score_on_its_own_scale(scored):
    assert scored.status == SIGNAL_STATUS_SUCCESS
    assert scored.score == 0.4787200441
    # Never a per-signal verdict, exactly as for every other detector here: a risk level beside a
    # provider's number would be a second, unvalidated classifier (rule 11).
    assert scored.risk_level is None


# --- 5. Provider and model provenance --------------------------------------------------------


def test_provider_version_names_the_upstream_revision_and_the_checkpoint_digest(scored):
    """Both halves, because either alone names something that is not a model.

    The same composite shape LipForensics uses. The revision fixes the architecture the network
    is executed from and the digest fixes the weights loaded into it; the same checkpoint in a
    different network is a different model with no operating point of its own.
    """
    assert scored.provider_version == (
        f"{EFFORT_REPOSITORY}@{EFFORT_REVISION}+{EFFORT_CHECKPOINT_SHA256}"
    )


def test_the_metadata_records_every_artifact_by_digest(scored):
    metadata = scored.signal_metadata

    assert metadata["upstream_repository"] == EFFORT_REPOSITORY
    assert metadata["upstream_revision"] == EFFORT_REVISION
    assert metadata["checkpoint_sha256"] == EFFORT_CHECKPOINT_SHA256
    # The landmark model and the backbone are inputs to every score — a different alignment or a
    # different CLIP is a different number — so they are pinned on the row exactly as the
    # checkpoint is.
    assert metadata["landmark_model_sha256"] == EFFORT_LANDMARK_SHA256
    assert metadata["clip_revision"] == "32bd64288804d66eefd0ccbe215aa642df71cc41"
    assert metadata["clip_sha256"] == {"config.json": "8a09b467" + "0" * 56}


def test_the_metadata_records_the_frozen_protocol_the_reading_was_taken_under(scored):
    """The three degrees of freedom R7-T7 froze, written onto the row itself.

    A stored reading has to say what produced it rather than refer to a document, because the
    document is what a future preprocessing change would edit.
    """
    metadata = scored.signal_metadata

    assert metadata["frames_per_clip"] == 8
    assert metadata["frame_sampling"] == "evenly spaced over [0, frame_count - 1], deduplicated"
    assert "dlib frontal detector" in metadata["face_detection_and_cropping"]
    assert "81-landmark" in metadata["face_detection_and_cropping"]
    assert metadata["aggregation"] == "arithmetic mean over frames that yielded a face"


def test_the_metadata_records_how_much_of_the_clip_could_be_read(scored):
    """The three counts, which differ on ordinary media and must not be interchangeable.

    They are what keeps the score from reading as a statement about the whole clip: this reading
    is a mean over six frames of an eight-frame sample, and the row says so.
    """
    metadata = scored.signal_metadata

    assert metadata["frames_requested"] == 8
    assert metadata["frames_decoded"] == 8
    assert metadata["frames_with_face"] == 6
    assert metadata["frame_scores"] == [0.41, 0.52, 0.48, 0.44, 0.51, 0.51]


def test_the_metadata_records_the_runtime_the_number_was_produced_on(scored):
    """The full stack, not only torch — because the stack is what moved the number.

    R7-T11 compared this pipeline against R7-T9's own run and found semantic preprocessing parity
    with measured numerical drift: identical frame accounting and identical abstention behaviour,
    and readings differing by a mean of about 0.0065 with a maximum of about 0.028, on a different
    Python, a different torch and the CPU rather than a GPU. That makes the runtime part of what a
    stored reading means, so it is persisted beside the checkpoint digest rather than assumed.
    """
    runtime = scored.signal_metadata["runtime"]

    assert set(runtime) == {"python", "torch", "torchvision", "dlib", "transformers"}
    assert runtime["torch"] == "2.13.0+cpu"
    assert runtime["dlib"] == "20.0.1"

    metadata = scored.signal_metadata
    assert metadata["torch_version"] == "2.13.0+cpu"
    assert metadata["device"] == "cpu"
    # Zero, or the forward path was partly randomly initialised and the score beside it is
    # meaningless rather than merely uncertain.
    assert metadata["missing_forward_path_key_count"] == 0


def test_the_runtime_is_recorded_even_when_a_version_cannot_be_read():
    """Provenance must not be able to fail a reading.

    A stack whose version cannot be read is worth recording as exactly that; refusing to answer
    because the detector could not describe itself would trade a reading for a label.
    """
    assert effort._version_of("a-package-that-is-not-installed") == "unknown"
    assert effort._version_of("pytest") != "unknown"


def test_the_drift_is_survivable_only_because_nothing_bands_this_score():
    """The measured drift and the evidence-only posture, asserted as one fact.

    R7-T11 measured a maximum absolute difference of about 0.028 between this runtime and the one
    R7-T9 evaluated on. That is acceptable here for exactly one reason: no production threshold
    exists for Effort, so there is no comparison whose outcome a difference of that size could
    change. This test fails the moment that stops being true — which is the moment a calibration
    measured against *this* environment becomes a prerequisite rather than a nicety.
    """
    for ruleset in risk_trace.RULESETS.values():
        for signal in ruleset.signals:
            assert signal.provider != EFFORT_PROVIDER

    assert not hasattr(effort, "THRESHOLD")
    assert not hasattr(effort, "OPERATING_THRESHOLD")

    engine_source = Path(risk_engine.__file__).read_text()
    assert "face_forgery" not in engine_source


# --- Abstention and failure are explicit states, never numbers -------------------------------


def test_an_abstention_is_recorded_without_a_score(monkeypatch, tmp_path):
    """No face in any sampled frame: a non-observation, and never a zero.

    A fabricated 0.0 would be a negative reading the model never produced, and it would let the
    detector buy a better false-positive rate by failing to run. R7-T9 abstained on 10.73% of
    genuine validation lineages and excluded them from both sides of every rate it reported.
    """
    signal = refusing(
        monkeypatch,
        tmp_path,
        EffortNoFaceDetected("no face", frames_requested=8, frames_decoded=8),
    )

    assert signal.status == SIGNAL_STATUS_FAILED
    assert signal.score is None
    assert signal.signal_metadata["error"] == "EffortNoFaceDetected"
    # The accounting behind the abstention, so "eight frames looked at, none with a face" is
    # distinguishable from "the decoder returned nothing".
    assert signal.signal_metadata["frames_requested"] == 8
    assert signal.signal_metadata["frames_decoded"] == 8
    assert signal.signal_metadata["frames_with_face"] == 0


def test_an_abstention_is_told_apart_from_the_failures_beside_it(monkeypatch, tmp_path):
    """Four different facts, four different reasons on the row.

    An abstention is an ordinary property of an upload; the other three are faults on this side.
    Collapsing them would make a clip with nobody in it indistinguishable from a machine whose
    checkpoint is missing.
    """
    reasons = set()
    for error in (
        EffortNoFaceDetected("no face", frames_requested=8, frames_decoded=8),
        EffortModelUnavailable("checkpoint digest mismatch"),
        EffortMediaError("undecodable"),
        EffortInferenceError("torch broke"),
    ):
        signal = refusing(monkeypatch, tmp_path, error)
        assert signal.status == SIGNAL_STATUS_FAILED
        assert signal.score is None
        reasons.add(signal.signal_metadata["error"])

    assert reasons == {
        "EffortNoFaceDetected",
        "EffortModelUnavailable",
        "EffortMediaError",
        "EffortInferenceError",
    }


def test_a_failure_never_becomes_negative_evidence(monkeypatch, tmp_path):
    """A failed detector has no score, and specifically not a low one.

    The temptation this refuses is writing 0.0 for "we found nothing", which would read as the
    model reporting genuine media — the one thing an unreached model cannot report.
    """
    signal = refusing(monkeypatch, tmp_path, EffortModelUnavailable("dlib is not installed"))

    assert signal.score is None
    assert signal.score != 0.0
    assert signal.risk_level is None


def test_the_failure_message_never_reaches_the_row(monkeypatch, tmp_path):
    """Only the kind. The messages quote the local artifacts' paths."""
    signal = refusing(
        monkeypatch, tmp_path, EffortModelUnavailable("/models/effort/secret.pth is wrong")
    )

    assert signal.signal_metadata == {"error": "EffortModelUnavailable"}


def test_detect_effort_never_raises(monkeypatch, tmp_path):
    """Every failure is this signal's, and costs no other reading and no analysis.

    `app.worker.local_readings` relies on this: a detector that raised would take the job down
    and with it the SVD and B7 rows the analysis is actually decided on.
    """
    signal = refusing(monkeypatch, tmp_path, EffortInferenceError("torch broke"))

    assert signal.provider == EFFORT_PROVIDER
    assert signal.status == SIGNAL_STATUS_FAILED


# --- 6. No production threshold, and no calibration identity ---------------------------------


def test_the_r7t9_evaluation_threshold_is_no_production_value():
    """`0.9574909508228302` may be discussed in production code, never evaluated by it.

    R7-T10 §5 gives three disqualifying reasons for adopting it: it was confirmed once on 388
    lineages, it is the highest readable genuine calibration score itself — an order statistic of
    one sample, which is the least stable thing a sample produces — and it was measured on curated
    corpora rather than on this system's traffic.

    Checked as a *literal in the parsed syntax tree* rather than as a substring, which is the
    distinction that makes the test mean what it says. The modules below do name the number, in
    comments and docstrings that explain why it was not adopted; that prose is the record of the
    decision and deleting it would lose the reason. What must not exist is the number as a value
    something could compare a score against.
    """
    import ast

    for module in (effort, detection, risk_engine, risk_trace):
        tree = ast.parse(Path(module.__file__).read_text())
        literals = [
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, float)
        ]

        assert R7T9_EVALUATION_THRESHOLD not in literals, module.__name__


def test_no_ruleset_holds_a_threshold_for_effort():
    """Every frozen ruleset, checked by identity rather than by count.

    A ruleset that listed Effort would have to name a threshold, and no threshold for this
    detector has ever been measured on this system's traffic.
    """
    for ruleset in risk_trace.RULESETS.values():
        listed = {(signal.provider, signal.signal_type) for signal in ruleset.signals}
        assert (EFFORT_PROVIDER, FACE_FORGERY_SIGNAL) not in listed
        assert EFFORT_PROVIDER not in {signal.provider for signal in ruleset.signals}


def test_no_new_calibration_id_was_minted_for_effort():
    """The analysis-level calibration identity is unchanged by this integration.

    It is the digest over the artifacts the *decision* rests on. Effort contributes to no
    decision, so it contributes to no identity: minting one would assert a measurement nobody
    took, and would silently invalidate every stored decision that names the old one.
    """
    assert risk_engine.CALIBRATION_ID == (
        "a74f6b9dbc64cead34cb8e31a03791228cdeb19497e8e5e0bc1a67c0337fc5f7"
    )
    assert risk_trace.RULESET_V4.calibration_id == risk_engine.CALIBRATION_ID
    assert risk_engine.RULES_VERSION == "r7-v4.0.0"


def test_the_effort_module_holds_no_threshold_at_all():
    """Not merely "not R7-T9's number" — no operating point of any kind.

    Every other detector module here is paired with a threshold somewhere in `app.risk_engine`.
    This one has none anywhere, which is what evidence-only means mechanically.
    """
    assert not hasattr(effort, "THRESHOLD")
    assert not hasattr(effort, "OPERATING_THRESHOLD")
    assert not hasattr(effort, "CALIBRATION_ID")

    engine_source = Path(risk_engine.__file__).read_text()
    assert "EFFORT" not in engine_source
    assert "face_forgery" not in engine_source


# --- 3. No existing ruleset consumes Effort --------------------------------------------------


def test_the_risk_engine_refuses_an_effort_signal_outright():
    """Handing it over is an error rather than a classification.

    `evaluate` admits three evidence types and rejects everything else before a rule is read, so
    a future caller that tried to route Effort into a decision raises instead of banding it
    against a threshold that was never measured for it.
    """

    class EffortDecisionEvidence:
        provider = EFFORT_PROVIDER
        signal_type = FACE_FORGERY_SIGNAL
        status = SIGNAL_STATUS_SUCCESS
        provider_version = "irrelevant"
        score = 0.99
        frames_scored = 8

    with pytest.raises(risk_engine.UncalibratedEvidence):
        risk_engine.evaluate(face=EffortDecisionEvidence())

    with pytest.raises(risk_engine.UncalibratedEvidence):
        risk_engine.evaluate(svd=EffortDecisionEvidence())

    with pytest.raises(risk_engine.UncalibratedEvidence):
        risk_engine.evaluate(lip=EffortDecisionEvidence())


def test_the_engine_takes_no_effort_argument():
    """There is no parameter to pass it through, which is the structural half of the guarantee."""
    import inspect

    parameters = set(inspect.signature(risk_engine.evaluate).parameters)

    assert parameters == {"svd", "face", "lip"}


def test_the_worker_reads_only_the_three_deciding_signals():
    """No reader fetches the Effort row for a decision.

    `app.worker` has one `persisted_*_evidence` function per deciding detector and each names one
    provider and one signal type, so the queries cannot see Effort even by accident.
    """
    from app import worker

    source = Path(worker.__file__).read_text()
    _, after = source.split("def persisted_svd_evidence", 1)
    readers, _ = after.split("def conclude_job", 1)

    assert "EFFORT_PROVIDER" not in readers
    assert "FACE_FORGERY_SIGNAL" not in readers


# --- 2. Effort changes no decision, in any of its three outcomes -----------------------------


def calibrated_svd(score: float) -> SvdEvidence:
    return SvdEvidence(
        provider=NVIDIA_PROVIDER,
        signal_type=SYNTHETIC_VIDEO_SIGNAL,
        status=SIGNAL_STATUS_SUCCESS,
        provider_version=risk_engine.SVD_PROVIDER_VERSION,
        score=score,
        total_clips=12,
    )


def calibrated_face(score: float) -> FaceEvidence:
    return FaceEvidence(
        provider=EFFICIENTNET_B7_PROVIDER,
        signal_type=FACE_MANIPULATION_SIGNAL,
        status=SIGNAL_STATUS_SUCCESS,
        provider_version=risk_engine.FACE_PROVIDER_VERSION,
        score=score,
        frames_scored=16,
    )


def calibrated_lip(score: float) -> LipEvidence:
    return LipEvidence(
        provider=LIPFORENSICS_PROVIDER,
        signal_type=LIP_FORENSICS_SIGNAL,
        status=SIGNAL_STATUS_SUCCESS,
        provider_version=risk_engine.LIP_PROVIDER_VERSION,
        score=score,
        windows_scored=4,
    )


# The three decisions this pipeline can reach, each with the evidence that reaches it. Scores
# either side of the two deciding thresholds — SVD at 0.9550971388816833 and the B7 at
# 0.9867589175701141 — so the table exercises a HIGH, a MEDIUM and an UNKNOWN rather than one
# band three times.
DECISIONS = [
    ("high from svd", calibrated_svd(0.99), calibrated_face(0.01), calibrated_lip(0.01)),
    ("high from face", calibrated_svd(0.10), calibrated_face(0.99), calibrated_lip(0.01)),
    ("both flagged", calibrated_svd(0.99), calibrated_face(0.99), calibrated_lip(0.90)),
    ("neither flagged", calibrated_svd(0.10), calibrated_face(0.01), calibrated_lip(0.01)),
    ("nothing readable", None, None, None),
]


@pytest.mark.parametrize("name, svd, face, lip", DECISIONS)
def test_the_decision_is_what_it_was_before_effort_existed(name, svd, face, lip):
    """The baseline itself: these are the only inputs a decision has.

    Effort cannot appear in this call because there is nowhere to put it — which is the whole of
    the rollback property. Turning Effort off returns the system to exactly this, with no ruleset
    to migrate and no stored row to rewrite, because this is what was already running.
    """
    decision = risk_engine.evaluate(svd=svd, face=face, lip=lip)

    assert decision.rules_version == "r7-v4.0.0"
    assert decision.calibration_id == risk_engine.CALIBRATION_ID
    assert decision.risk_level in {"HIGH", "MEDIUM", "UNKNOWN"}


@pytest.mark.parametrize("name, svd, face, lip", DECISIONS)
@pytest.mark.parametrize(
    "outcome",
    ["success_low", "success_high", "abstained", "failed", "absent"],
    ids=["effort scored low", "effort scored high", "effort abstained", "effort failed", "no effort"],
)
def test_no_effort_outcome_moves_a_decision(name, svd, face, lip, outcome, monkeypatch, tmp_path):
    """The central claim, checked as a comparison rather than as an expected level.

    For every decision the engine can reach, the analysis is decided identically whether Effort
    scored 0.02, scored 0.99, abstained, failed outright, or never ran. The Effort signal is
    produced for real in each case — `detect_effort` runs and returns a row — and then the
    decision is taken, and the two are compared against the decision taken without it.

    `effort scored high` is the case worth stating on its own: 0.99 is above the point R7-T9
    measured as a candidate, and it still moves nothing, because nothing in this system compares
    this number to that one.
    """
    baseline = risk_engine.evaluate(svd=svd, face=face, lip=lip)

    if outcome == "success_low":
        signal = refusing_or_scoring(monkeypatch, tmp_path, evidence(score=0.02))
    elif outcome == "success_high":
        signal = refusing_or_scoring(monkeypatch, tmp_path, evidence(score=0.99))
    elif outcome == "abstained":
        signal = refusing(
            monkeypatch,
            tmp_path,
            EffortNoFaceDetected("no face", frames_requested=8, frames_decoded=8),
        )
    elif outcome == "failed":
        signal = refusing(monkeypatch, tmp_path, EffortInferenceError("torch broke"))
    else:
        signal = None

    if signal is not None:
        assert signal.provider == EFFORT_PROVIDER

    after = risk_engine.evaluate(svd=svd, face=face, lip=lip)

    assert after.risk_level == baseline.risk_level
    assert after.rule_id == baseline.rule_id
    assert after.rules_version == baseline.rules_version
    assert after.calibration_id == baseline.calibration_id


def refusing_or_scoring(monkeypatch, tmp_path, produced):
    """`detect_effort` over a model that returns `produced`."""

    def analyze(_path, **_kwargs):
        return produced

    monkeypatch.setattr(detection, "analyze_effort", analyze)
    artifact = tmp_path / "clip.mp4"
    artifact.write_bytes(b"video")

    return detect_effort(artifact)


def test_effort_contributes_to_no_decisional_count():
    """The count of flagged detectors is what `rule_id` encodes, and Effort is in none of it.

    Under `r7-v4.0.0` exactly two detectors can flag, so `R102` — both of them — is the ceiling
    however high an Effort score sits. There is no arithmetic anywhere that could carry a third
    contributor into it, because the engine is never given a third.
    """
    both = risk_engine.evaluate(
        svd=calibrated_svd(0.99), face=calibrated_face(0.99), lip=calibrated_lip(0.99)
    )

    assert both.rule_id == "R102"
    assert both.risk_level == "HIGH"

    # And the rule that fires is decided entirely by the two deciding scores: the mouth-dynamics
    # score above its own threshold does not add a third, and neither could Effort.
    one = risk_engine.evaluate(
        svd=calibrated_svd(0.99), face=calibrated_face(0.01), lip=calibrated_lip(0.99)
    )

    assert one.rule_id == "R100"


# --- The risk trace shows Effort as nothing --------------------------------------------------


def test_effort_is_not_a_contribution_in_any_ruleset_trace():
    """Not in `contributions`, at any stored version.

    R7-T10 §8.1: the trace's contributions are built from the ruleset's `CalibratedSignal` list,
    and a signal with no production threshold has nothing to be banded against. An entry for it
    would require inventing a threshold to band it with, which is the failure the whole posture
    exists to avoid.
    """
    for ruleset in risk_trace.RULESETS.values():
        for signal in ruleset.signals:
            assert signal.provider != EFFORT_PROVIDER
            assert signal.signal_type != FACE_FORGERY_SIGNAL


def test_no_ruleset_was_edited_to_mention_effort():
    """Historical immutability, asserted on the two rulesets a stored decision can name.

    A decision taken under `r7-v4.0.0` was taken by rules that did not read Effort and must keep
    reading that way forever — including for an analysis whose row later acquires an Effort signal
    because the detector was run on a re-analysis (R7-T10 §8.2).
    """
    assert len(risk_trace.RULESET_V4.signals) == 3
    assert {signal.signal_type for signal in risk_trace.RULESET_V4.signals} == {
        SYNTHETIC_VIDEO_SIGNAL,
        FACE_MANIPULATION_SIGNAL,
        LIP_FORENSICS_SIGNAL,
    }
    assert len(risk_trace.RULESET_V3.signals) == 3
    assert len(risk_trace.RULESET_V2.signals) == 2
    assert len(risk_trace.RULESET_V1.signals) == 1


# --- 4. The three existing detectors are untouched -------------------------------------------


def test_the_existing_detector_identities_are_unchanged():
    """The strings every stored row and every threshold lookup depends on."""
    assert (NVIDIA_PROVIDER, SYNTHETIC_VIDEO_SIGNAL) == ("nvidia", "synthetic_video")
    assert (EFFICIENTNET_B7_PROVIDER, FACE_MANIPULATION_SIGNAL) == (
        "efficientnet-b7",
        "face_manipulation",
    )
    assert (LIPFORENSICS_PROVIDER, LIP_FORENSICS_SIGNAL) == ("lipforensics", "lip_forensics")


def test_the_existing_operating_points_are_unchanged():
    """The two deciding thresholds and the one evidence-only one, to the last digit.

    They clear the highest genuine score observed by 0.0010 and 0.0003 respectively, so a
    difference far below what would look like a rounding change is a different operating point.
    """
    assert risk_engine.SVD_T_HIGH == 0.9550971388816833
    assert risk_engine.FACE_T_HIGH == 0.9867589175701141
    assert risk_engine.LIP_T_HIGH == 0.22962537594139576


def test_lipforensics_is_still_evidence_only_and_still_read():
    """R7-T5's demotion is untouched by this task.

    Effort arriving as a second non-decisional detector must not disturb the first one: it is
    still banded in the trace with `decisional=False`, and still read by the engine for how much
    of the evidence could be read at all.
    """
    lip = [
        signal
        for signal in risk_trace.RULESET_V4.signals
        if signal.signal_type == LIP_FORENSICS_SIGNAL
    ]

    assert len(lip) == 1
    assert lip[0].decisional is False
    assert lip[0].threshold == 0.22962537594139576


def test_the_deciding_detectors_are_still_decisional():
    decisional = {
        signal.signal_type
        for signal in risk_trace.RULESET_V4.signals
        if signal.decisional
    }

    assert decisional == {SYNTHETIC_VIDEO_SIGNAL, FACE_MANIPULATION_SIGNAL}


# --- The rollback switch ---------------------------------------------------------------------


def test_effort_runs_unless_a_deployment_turns_it_off(monkeypatch):
    """On by default, off only when said so — and the off values are spelled out.

    The opposite default from `DEEPGUARD_SHADOW_MODE`, deliberately: a shadow experiment is opted
    into and this is a detector the image ships.
    """
    monkeypatch.delenv(effort.ENABLED_VARIABLE, raising=False)
    assert effort.is_enabled() is True

    for value in ("false", "FALSE", "0", "no", "off"):
        monkeypatch.setenv(effort.ENABLED_VARIABLE, value)
        assert effort.is_enabled() is False

    for value in ("true", "1", "yes", "on", ""):
        monkeypatch.setenv(effort.ENABLED_VARIABLE, value)
        assert effort.is_enabled() is True


def test_a_disabled_effort_writes_no_row_at_all(monkeypatch, tmp_path):
    """Absence, not a `FAILED` row: nothing asked the detector anything.

    A `FAILED` row is a finding — "this source was asked and could not answer" — and writing one
    for a detector that was never reached would record a finding nobody made. It is also what
    makes the rollback clean: an analysis run with Effort off is indistinguishable from one run
    before Effort existed.
    """
    from app import worker

    monkeypatch.setenv(effort.ENABLED_VARIABLE, "false")
    monkeypatch.setattr(detection, "detect_face_manipulation", lambda _p: _row("b7"))
    monkeypatch.setattr(detection, "detect_lip_forensics", lambda _p: _row("lip"))
    monkeypatch.setattr(worker, "detect_face_manipulation", lambda _p: _row("b7"))
    monkeypatch.setattr(worker, "detect_lip_forensics", lambda _p: _row("lip"))
    monkeypatch.setattr(
        worker, "detect_effort", lambda _p: pytest.fail("Effort ran while disabled")
    )

    readings = worker.local_readings(tmp_path / "clip.mp4")

    assert [reading.signal.provider for reading in readings] == ["b7", "lip"]


def test_an_enabled_effort_is_the_third_reading(monkeypatch, tmp_path):
    """And it runs between the two, cheapest first — the order `local_readings` documents."""
    from app import worker

    monkeypatch.delenv(effort.ENABLED_VARIABLE, raising=False)
    monkeypatch.setattr(worker, "detect_face_manipulation", lambda _p: _row("b7"))
    monkeypatch.setattr(worker, "detect_lip_forensics", lambda _p: _row("lip"))
    monkeypatch.setattr(worker, "detect_effort", lambda _p: _row(EFFORT_PROVIDER))

    readings = worker.local_readings(tmp_path / "clip.mp4")

    assert [reading.signal.provider for reading in readings] == ["b7", EFFORT_PROVIDER, "lip"]
    # No timeline evidence: the frozen protocol is one clip to one reading, and a per-frame row
    # would claim the sampled frames were detections of manipulation at those moments.
    assert all(reading.segments == [] for reading in readings)


def _row(provider: str):
    from app.db.models import AnalysisSignal

    signal = AnalysisSignal(provider=provider, signal_type="stub")
    signal.status = SIGNAL_STATUS_SUCCESS

    return signal


# --- The frozen preprocessing ----------------------------------------------------------------


def test_the_production_transcription_is_byte_identical_to_the_benchmarked_one():
    """`app/effort_upstream.py` is the file the three studies ran, copied rather than adapted.

    It has to be a copy: `apps/api` cannot import from `scripts/`, which is not on the API's path
    and not in the image's build context. A copy can drift, which is what this test exists to
    stop — the alignment in particular is a pile of magic numbers, and one of them retyped
    slightly differently produces a face crop that is *almost* the one the checkpoint was trained
    on, which does not fail loudly but returns plausible numbers about a different input
    distribution.

    Compared as bytes on disk rather than through an import, so the check still runs in an image
    that has not installed the detector's libraries.
    """
    import app

    production = Path(app.__file__).parent / "effort_upstream.py"

    assert production.exists()

    # Walked up rather than indexed, because this suite runs both from a checkout and from an
    # image where `apps/api` is the filesystem root and there is no `scripts/` above it.
    benchmarked = None
    for ancestor in production.parents:
        candidate = ancestor / "scripts" / "eval" / "models" / "effort_upstream.py"
        if candidate.exists():
            benchmarked = candidate
            break

    if benchmarked is None:
        pytest.skip("the benchmark harness is not present in this checkout")

    assert production.read_bytes() == benchmarked.read_bytes()


def test_the_frozen_sampling_rule_is_eight_evenly_spaced_frames():
    """The rule `effort_freeze.json` fixed, checked at the boundaries that make it a rule.

    `linspace` over the closed range rather than a stride, so the last frame is always included
    and a short clip and a long one are sampled the same way in proportion. Clips shorter than
    eight frames contribute every frame they have.
    """
    assert effort.FRAMES_PER_CLIP == 8
    assert effort._sample_indices(0) == []
    assert effort._sample_indices(3) == [0, 1, 2]
    assert effort._sample_indices(8) == [0, 1, 2, 3, 4, 5, 6, 7]

    sampled = effort._sample_indices(100)
    assert len(sampled) == 8
    assert sampled[0] == 0
    assert sampled[-1] == 99
    assert sampled == sorted(sampled)


def test_the_aggregation_is_the_arithmetic_mean():
    """Upstream's own: `get_video_metrics` computes a video's prediction as `pred_sum / leng`.

    Named as a constant and written onto every row, because a median, a maximum or a trimmed mean
    would be a different detector wearing this one's evidence — and the three studies measured
    this one.
    """
    assert effort.AGGREGATION == "arithmetic mean over frames that yielded a face"


def test_the_pinned_artifacts_are_the_ones_the_studies_measured():
    assert effort.CHECKPOINT_SHA256 == EFFORT_CHECKPOINT_SHA256
    assert effort.LANDMARK_SHA256 == EFFORT_LANDMARK_SHA256
    assert effort.UPSTREAM_REVISION == EFFORT_REVISION
    assert effort.LANDMARK_FILENAME == "shape_predictor_81_face_landmarks.dat"


def test_a_swapped_artifact_is_refused_rather_than_loaded(tmp_path, monkeypatch):
    """The digest is checked on every call, not trusted from the build.

    R7-T7's first download of this checkpoint was silently truncated at 73 MB of 1,158 MB, which
    loads far enough to produce numbers rather than errors. A bind mount, a rebuilt layer or a
    developer pointing `DEEPGUARD_EFFORT_MODEL_DIR` at a benchmark cache all put bytes in front
    of this process that no build ever saw.
    """
    monkeypatch.setenv(effort.MODEL_DIR_ENV, str(tmp_path))
    (tmp_path / effort.CHECKPOINT_FILENAME).write_bytes(b"not the checkpoint")

    with pytest.raises(EffortModelUnavailable) as refusal:
        effort._load(tmp_path)

    assert "sha256" in str(refusal.value)


def test_a_missing_artifact_is_refused_rather_than_loaded(tmp_path, monkeypatch):
    monkeypatch.setenv(effort.MODEL_DIR_ENV, str(tmp_path))

    with pytest.raises(EffortModelUnavailable) as refusal:
        effort._load(tmp_path)

    assert "not found" in str(refusal.value)


def test_the_model_directory_is_configuration_and_not_request_data(monkeypatch):
    monkeypatch.delenv(effort.MODEL_DIR_ENV, raising=False)
    assert effort._model_dir() == Path("/models/effort")

    monkeypatch.setenv(effort.MODEL_DIR_ENV, "/somewhere/else")
    assert effort._model_dir() == Path("/somewhere/else")


# --- The API surface: Effort is persisted without numeric exposure ---------------------------
#
# R7-T10 §11 A5 makes the exposure task conditional — "required only if the implementation
# exposes Effort in the report or API surface. If the score is persisted internally without
# numeric exposure, A5 does not exist." This integration takes that branch, and these tests are
# what hold it there.
#
# It is not a limitation of the schema; it is the smaller change. Both responses are built from
# an explicit per-detector list rather than from a generic signals collection, so Effort does not
# appear unless somebody adds it — and R7-T10 §12 is emphatic about what adding it costs without
# a wording task first: "a raw score rendered beside a detector name reads to a user as a
# probability that the media is fake", which is the exact claim rule 11 forbids and which would
# be a particularly bad claim for this detector, uninformative in 32 of 32 cases on the
# in-the-wild family R7-T7 measured.


def test_the_dashboard_response_has_no_effort_field():
    """The analysis response names its detectors one by one, and Effort is not among them."""
    from app.api import analyses

    fields = set(analyses.AnalysisSummary.model_fields)

    assert "effort" not in fields
    assert "face_forgery" not in fields
    # And the five that were there are still there, named exactly as they were.
    assert {
        "synthetic_video",
        "provenance",
        "active_speaker",
        "audio_authenticity",
        "face_manipulation",
        "lip_forensics",
    } <= fields


def test_the_public_api_exposes_no_effort_signal():
    """The public collection is a fixed list of six, and this integration did not lengthen it."""
    from app.api.public_v1 import analyses as public

    source = Path(public.__file__).read_text()

    assert "effort" not in source.lower()
    assert "face_forgery" not in source


def test_no_api_module_reads_the_effort_row():
    """Neither response is built from a generic signals query that would pick it up."""
    from app.api import analyses

    source = Path(analyses.__file__).read_text()

    assert "EFFORT_PROVIDER" not in source
    assert "face_forgery" not in source


# --- The wording the row carries -------------------------------------------------------------


def test_the_row_never_calls_the_score_a_probability_or_a_confidence(scored):
    """R7-T10 §12: a raw score rendered beside a detector name reads as "probability it is fake".

    On the in-the-wild family R7-T7 measured, that number is uninformative in 32 of 32 cases, so
    the claim would be both forbidden by rule 11 and specifically wrong here. The metadata keys
    are the API surface — the analysis response renders the persisted document — so the wording
    is checked where it is stored.
    """
    keys = " ".join(scored.signal_metadata).lower()

    assert "confidence" not in keys
    assert "probability" not in keys
    assert "fake" not in keys
    assert "real" not in keys
    assert "verdict" not in keys
    assert "threshold" not in keys
