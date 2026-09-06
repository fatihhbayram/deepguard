"""The offline replay: that it is the production rules, and that it feeds them honest evidence."""

import pytest

from eval import replay as replay_module
from eval.corpus import SPLIT_EVALUATION, CorpusItem

engine = replay_module.load_risk_engine()


def clip_item(clip_id="clip", label="real", private=False):
    return CorpusItem(
        clip_id=clip_id,
        path=f"clips/{clip_id}.mp4",
        label=label,
        family="mavos_real_en",
        stratum_primary="genuine_in_the_wild",
        strata=["genuine_in_the_wild"],
        source_lineage_id="mavos-real:en:1",
        base_media_id=clip_id,
        derivative_of=None,
        derivation="none",
        source="test",
        acquisition_type="test",
        label_provenance="test",
        license="test",
        permission_status="test",
        redistributable=False,
        private=private,
        sha256="0" * 64,
        bytes=1,
        split=SPLIT_EVALUATION,
    )


def run(clip_id, score, counts_key, count, status="ok"):
    """A `results.json` in the shape `benchmark.cli` writes, carrying one clip."""
    return {
        "clips": {
            clip_id: {"clip_id": clip_id, "status": status, "score": score, "error": None}
        },
        "provenance": {counts_key: {clip_id: count}},
        "ran": True,
    }


def empty_run():
    return {"clips": {}, "provenance": {}, "ran": False}


def versions():
    """The production version strings, so evidence is eligible unless a test says otherwise."""
    return {
        "svd": engine.SVD_PROVIDER_VERSION,
        "face": engine.FACE_PROVIDER_VERSION,
        "lip": engine.LIP_PROVIDER_VERSION,
    }


def test_the_rules_are_the_production_module_not_a_copy():
    """If this file ever grew its own rule table, this is the test that would have to be deleted."""
    assert replay_module.RISK_ENGINE_PATH.name == "risk_engine.py"
    assert "apps/api/app" in replay_module.RISK_ENGINE_PATH.as_posix()
    assert engine.RULES_VERSION == "r5-v3.0.0"
    assert engine.LIP_T_HIGH == 0.22962537594139576


def test_a_lip_score_over_the_threshold_alone_reaches_high_on_r103():
    """The rule the regression case fires, reproduced offline from evidence alone."""
    svd, face, lip = replay_module.build_evidence(
        "clip",
        engine,
        run("clip", 0.04, "total_clips_by_clip_id", 3),
        run("clip", 0.05, "frames_scored_by_clip_id", 8),
        run("clip", 0.3995, "windows_scored_by_clip_id", 4),
        versions(),
    )
    decision = engine.evaluate(svd=svd, face=face, lip=lip)
    assert decision.risk_level == "HIGH"
    assert decision.rule_id == "R103"
    assert replay_module.responsible_detectors(engine, svd, face, lip) == ["lipforensics"]


def test_a_lip_score_under_the_threshold_does_not_reach_high():
    svd, face, lip = replay_module.build_evidence(
        "clip",
        engine,
        run("clip", 0.04, "total_clips_by_clip_id", 3),
        run("clip", 0.05, "frames_scored_by_clip_id", 8),
        run("clip", 0.2, "windows_scored_by_clip_id", 4),
        versions(),
    )
    decision = engine.evaluate(svd=svd, face=face, lip=lip)
    assert decision.risk_level == "MEDIUM"
    assert decision.rule_id == "R200"


def test_an_abstention_is_silence_and_never_a_negative():
    """A clip the detector could not read must not be classified as though it had been read."""
    svd, face, lip = replay_module.build_evidence(
        "clip",
        engine,
        run("clip", 0.04, "total_clips_by_clip_id", 3),
        run("clip", 0.05, "frames_scored_by_clip_id", 8),
        run("clip", None, "windows_scored_by_clip_id", 0, status="error"),
        versions(),
    )
    assert lip.status == "FAILED"
    assert engine.is_usable_lip(lip) is False
    decision = engine.evaluate(svd=svd, face=face, lip=lip)
    assert decision.rule_id == "R201"


def test_a_zero_unit_count_is_not_a_usable_reading():
    """`windows_scored = 0` is a mean over nothing; the rules already refuse it and so must we."""
    _, _, lip = replay_module.build_evidence(
        "clip",
        engine,
        empty_run(),
        empty_run(),
        run("clip", 0.9, "windows_scored_by_clip_id", 0),
        versions(),
    )
    assert engine.is_usable_lip(lip) is False


def test_a_missing_count_is_not_invented():
    """A run without the counted wrapper yields `None`, which the rules treat as unreadable.

    The alternative — defaulting to some plausible count — would have the replay assert
    evidence the detector never reported.
    """
    _, _, lip = replay_module.build_evidence(
        "clip",
        engine,
        empty_run(),
        empty_run(),
        {"clips": {"clip": {"clip_id": "clip", "status": "ok", "score": 0.9}}, "provenance": {},
         "ran": True},
        versions(),
    )
    assert lip.windows_scored is None
    assert engine.is_usable_lip(lip) is False


def test_a_deployment_that_is_not_the_calibrated_one_is_ineligible():
    """A score from another checkpoint is a measurement of something else, and must not decide."""
    _, _, lip = replay_module.build_evidence(
        "clip",
        engine,
        empty_run(),
        empty_run(),
        run("clip", 0.99, "windows_scored_by_clip_id", 4),
        {**versions(), "lip": "https://example.invalid@deadbeef+0000"},
    )
    assert engine.is_eligible_lip(lip) is False
    assert engine.evaluate(lip=lip).risk_level == "UNKNOWN"


def test_provider_version_strings_are_rebuilt_to_the_production_constants():
    """The replay derives them from observed provenance; they must land on the real strings."""
    lip_provenance = {
        "upstream": {
            "repository": "https://github.com/ahaliassos/LipForensics",
            "revision": "d0bf5553bfb9676f1771d590472b26a3a76de894",
        },
        "classifier": {
            "sha256": "4b7790bc8e02d0c25ecfa0d8d6a2907123c2206cc32e2bad6044e50f013c253d"
        },
    }
    face_provenance = {
        "classifier": {
            "artifact": (
                "tomas-gajarsky/facetorch-deepfake-efficientnet-b7/model-torch2.11.pt2"
            ),
            "revision": "4acc494f37eb63d7457166eff2acb45c5b04b9a6",
        }
    }
    assert replay_module.lip_provider_version(lip_provenance) == engine.LIP_PROVIDER_VERSION
    assert replay_module.face_provider_version(face_provenance) == engine.FACE_PROVIDER_VERSION


def test_a_detector_that_was_not_run_is_absent_rather_than_failed():
    """Not running a detector is not the same as a detector that ran and abstained."""
    svd, _, _ = replay_module.build_evidence(
        "clip", engine, empty_run(), empty_run(), empty_run(), versions()
    )
    assert svd is None


def test_the_replay_writes_a_trace_that_names_its_rule_set():
    lip_run = run("clip", 0.9, "windows_scored_by_clip_id", 4)
    lip_run["provenance"].update(
        {
            "upstream": {
                "repository": "https://github.com/ahaliassos/LipForensics",
                "revision": "d0bf5553bfb9676f1771d590472b26a3a76de894",
            },
            "classifier": {
                "sha256": (
                    "4b7790bc8e02d0c25ecfa0d8d6a2907123c2206cc32e2bad6044e50f013c253d"
                )
            },
        }
    )
    report = replay_module.replay([clip_item()], engine, empty_run(), empty_run(), lip_run)
    assert report["eligibility"]["lip"]["matches_calibrated_deployment"] is True
    assert report["rules"]["rules_version"] == engine.RULES_VERSION
    assert report["rules"]["lip_t_high"] == engine.LIP_T_HIGH
    assert report["rules"]["modified_by_this_task"] is False
    assert report["decisions"][0]["rule_id"] == "R103"
    assert report["decisions"][0]["responsible_detectors"] == ["lipforensics"]
