"""R12-T6: the operational export against real PostgreSQL rows, then replayed.

Rows are seeded through the real models into the suite's own database (`deepguard_test`) and are
test data: every hash, lineage and analysis this file creates is removed afterwards. The export
is read inside a `READ ONLY` transaction, as `main` reads it, and `scripts/eval/replay.py` then
consumes the artifacts it wrote.
"""

import hashlib
import importlib.util
import json
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from app.db.models import (
    ANALYSIS_STATUS_COMPLETED,
    Analysis,
    AnalysisSignal,
    GroundTruth,
    LineageSplit,
    MediaFile,
    MediaGovernance,
)
from app.db.session import SessionLocal, engine
from app.risk_engine import (
    FACE_PROVIDER,
    FACE_PROVIDER_VERSION,
    FACE_SIGNAL_TYPE,
    SVD_PROVIDER,
    SVD_PROVIDER_VERSION,
    SVD_SIGNAL_TYPE,
)

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[3]
OTHER_SVD_VERSION = "r12t6-test-other-function-id"

CORPUS_METADATA = {
    "license": "test fixture licence",
    "permission_status": "test fixture; not real media",
    "redistributable": False,
    "private": True,
    "stratum_primary": "r12t6_test_stratum",
    "source": "r12t6 integration test fixture",
    "acquisition_type": "test_fixture",
    "benchmark_family": "r12t6_test_family",
}


def load_script(name: str):
    scripts = REPO_ROOT / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    spec = importlib.util.spec_from_file_location(
        f"r12t6_{name}", scripts / "eval" / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def database():
    try:
        with engine.connect():
            pass
    except SQLAlchemyError as error:
        pytest.skip(f"PostgreSQL is not reachable: {error.__class__.__name__}")
    return engine


@pytest.fixture
def seeded(database):
    """Seed the rows, yield their hashes, and remove every one afterwards."""
    analyses: list[uuid.UUID] = []
    hashes: list[str] = []
    lineages: list[str] = []
    actor = uuid.uuid4()
    base_time = datetime(2026, 9, 24, 10, 0, tzinfo=timezone.utc)

    def new_hash() -> str:
        sha = hashlib.sha256(uuid.uuid4().bytes).hexdigest()
        hashes.append(sha)
        return sha

    def upload(db, sha: str, minutes: int) -> uuid.UUID:
        analysis = Analysis(
            status=ANALYSIS_STATUS_COMPLETED,
            risk_level="NO_CALIBRATED_MANIPULATION_SIGNAL",
            risk_rules_version="r9-v5.0.0",
            risk_calibration_id="c" * 64,
            risk_rule_id="R1",
            created_at=base_time + timedelta(minutes=minutes),
        )
        db.add(analysis)
        db.flush()
        analyses.append(analysis.id)
        db.add(
            MediaFile(
                analysis_id=analysis.id,
                original_filename="clip.mp4",
                content_type="video/mp4",
                size_bytes=4096,
                original_sha256=sha,
                original_storage_key=f"originals/{sha}",
                format_name="mov,mp4,m4a,3gp,3g2,mj2",
                codec_name="h264",
                width=640,
                height=360,
                duration=4.0,
                frame_rate=25.0,
                pix_fmt="yuv420p",
                constant_frame_rate=True,
                was_normalized=False,
                was_assembled=False,
                acquisition_method="upload",
            )
        )
        return analysis.id

    def add_signal(db, analysis_id, provider, signal_type, version, score, count_key, minutes):
        db.add(
            AnalysisSignal(
                analysis_id=analysis_id,
                provider=provider,
                signal_type=signal_type,
                provider_version=version,
                status="SUCCESS",
                score=score,
                signal_metadata={count_key: 5},
                created_at=base_time + timedelta(minutes=minutes),
            )
        )

    with SessionLocal() as db:
        genuine, swap, ineligible, unlabelled = (new_hash() for _ in range(4))
        lineage_test, lineage_holdout = f"r12t6-{uuid.uuid4().hex}", f"r12t6-{uuid.uuid4().hex}"
        lineages.extend([lineage_test, lineage_holdout])

        genuine_first = upload(db, genuine, 0)
        upload(db, genuine, 5)  # the same bytes again: one corpus item, one duplicate
        swap_analysis = upload(db, swap, 1)
        upload(db, ineligible, 2)
        upload(db, unlabelled, 3)

        add_signal(db, genuine_first, SVD_PROVIDER, SVD_SIGNAL_TYPE, SVD_PROVIDER_VERSION,
                   0.10, "total_clips", 0)
        add_signal(db, swap_analysis, SVD_PROVIDER, SVD_SIGNAL_TYPE, SVD_PROVIDER_VERSION,
                   0.20, "total_clips", 1)
        add_signal(db, swap_analysis, SVD_PROVIDER, SVD_SIGNAL_TYPE, OTHER_SVD_VERSION,
                   0.30, "total_clips", 1)
        add_signal(db, swap_analysis, FACE_PROVIDER, FACE_SIGNAL_TYPE, FACE_PROVIDER_VERSION,
                   0.999, "frames_scored", 1)

        db.add_all([
            LineageSplit(source_lineage_id=lineage_test, dataset_split="TEST"),
            LineageSplit(source_lineage_id=lineage_holdout, dataset_split="HOLDOUT"),
        ])
        db.flush()
        db.add_all([
            MediaGovernance(media_sha256=genuine, source_lineage_id=lineage_test,
                            actor_id=actor, **CORPUS_METADATA),
            MediaGovernance(media_sha256=swap, source_lineage_id=lineage_holdout,
                            actor_id=actor, **CORPUS_METADATA),
            MediaGovernance(media_sha256=ineligible, source_lineage_id=lineage_test,
                            actor_id=actor, **CORPUS_METADATA),
            MediaGovernance(media_sha256=unlabelled, source_lineage_id=lineage_test,
                            actor_id=actor, **CORPUS_METADATA),
            GroundTruth(media_sha256=genuine, source_class="CONTROLLED_TEST", label="GENUINE",
                        actor_id=actor),
            GroundTruth(media_sha256=swap, source_class="CONTROLLED_TEST", label="FACE_SWAP",
                        actor_id=actor),
            GroundTruth(media_sha256=ineligible, source_class="UNKNOWN", label="GENUINE",
                        actor_id=actor),
            GroundTruth(media_sha256=unlabelled, source_class="OWNER_KNOWN", label="UNKNOWN",
                        actor_id=actor),
        ])
        db.commit()

        yield {"genuine": genuine, "swap": swap, "ineligible": ineligible,
               "unlabelled": unlabelled}

        db.rollback()
        db.query(GroundTruth).filter(GroundTruth.media_sha256.in_(hashes)).delete(
            synchronize_session=False
        )
        db.query(MediaGovernance).filter(MediaGovernance.media_sha256.in_(hashes)).delete(
            synchronize_session=False
        )
        db.query(LineageSplit).filter(LineageSplit.source_lineage_id.in_(lineages)).delete(
            synchronize_session=False
        )
        for analysis_id in analyses:
            db.query(AnalysisSignal).filter(AnalysisSignal.analysis_id == analysis_id).delete(
                synchronize_session=False
            )
            db.query(MediaFile).filter(MediaFile.analysis_id == analysis_id).delete(
                synchronize_session=False
            )
            db.query(Analysis).filter(Analysis.id == analysis_id).delete(
                synchronize_session=False
            )
        db.commit()


def test_the_database_export_is_replayed_under_each_explicit_provider_version(
    seeded, tmp_path
):
    exporter = load_script("export_operational_dataset")
    replay = load_script("replay")
    out = tmp_path / "export"

    with SessionLocal() as db:
        db.execute(text("SET TRANSACTION READ ONLY"))
        metadata = exporter.export(db, out, None, {"database": "deepguard_test"})
        db.rollback()

    corpus = json.loads((out / "corpus.json").read_text())
    by_id = {item["clip_id"]: item for item in corpus["items"]}
    assert metadata["exported_sha_count"] >= 2
    assert seeded["genuine"] in by_id and seeded["swap"] in by_id
    assert seeded["ineligible"] not in by_id and seeded["unlabelled"] not in by_id
    assert metadata["exclusion_counts"]["ground_truth_ineligible"] >= 1
    assert metadata["exclusion_counts"]["unknown_label"] >= 1
    assert metadata["duplicate_counts"]["media_files"] >= 1
    assert by_id[seeded["genuine"]]["label"] == "real"
    assert by_id[seeded["swap"]]["label"] == "face_swap"
    assert by_id[seeded["swap"]]["family"] == CORPUS_METADATA["benchmark_family"]
    assert by_id[seeded["swap"]]["split"] == "HOLDOUT"
    assert by_id[seeded["genuine"]]["redistributable"] is False
    assert by_id[seeded["genuine"]]["label_provenance"] == "CONTROLLED_TEST"

    files = {}
    for path in out.glob("results_*.json"):
        payload = json.loads(path.read_text())
        files[(payload["provider"], payload["provider_version"], payload["signal_type"])] = path
    svd_calibrated = files[(SVD_PROVIDER, SVD_PROVIDER_VERSION, SVD_SIGNAL_TYPE)]
    svd_other = files[(SVD_PROVIDER, OTHER_SVD_VERSION, SVD_SIGNAL_TYPE)]
    face = files[(FACE_PROVIDER, FACE_PROVIDER_VERSION, FACE_SIGNAL_TYPE)]
    assert svd_calibrated.name != svd_other.name

    for svd_file, expected_version, calibrated in (
        (svd_calibrated, SVD_PROVIDER_VERSION, True),
        (svd_other, OTHER_SVD_VERSION, False),
    ):
        report_path = tmp_path / f"replay-{svd_file.stem}.json"
        assert replay.main([
            "--corpus", str(out / "corpus.json"),
            "--svd-run", str(svd_file),
            "--face-run", str(face),
            "--output", str(report_path),
        ]) == 0
        report = json.loads(report_path.read_text())
        assert report["eligibility"]["svd"]["observed_provider_version"] == expected_version
        assert report["eligibility"]["svd"]["matches_calibrated_deployment"] is calibrated
        assert report["eligibility"]["face"]["matches_calibrated_deployment"] is True
        decisions = {d["clip_id"]: d for d in report["decisions"]}
        assert decisions[seeded["swap"]]["scores"]["face"] == 0.999
        assert decisions[seeded["swap"]]["counts"]["face_frames_scored"] == 5
