"""R12-T7: the cross-component invariants no single R12 task's tests reach.

Everything else R12-T7 checks is already asserted where it belongs, and is not repeated here:
Ground Truth leaving verdict, signals, provenance and the human review untouched, and its audit
trail, in `test_ground_truth_api.py`; the governance audit trail including the corpus metadata in
`test_dataset_governance_api.py`; exclusions staying out of the confusion matrix in
`scripts/eval/tests/test_operational_metrics.py`; the artifact hashes and replay without
`model_provenance` in `scripts/eval/tests/test_export_operational_dataset.py`.

Three claims were left:

*A human review never becomes Ground Truth.* The existing tests hold the direction Ground Truth →
review. This holds the other: agreeing or disagreeing with the automated assessment neither
creates a Ground Truth record nor changes or audits one that exists.

*Dataset governance leaves the forensic record alone.* Creating and revising a governance
statement changes neither the verdict columns, nor any signal (provenance included), nor the
human review, of any analysis of those bytes.

*The export is deterministic.* The existing tests hash one export. This runs it twice against the
same database, and once more with the rows it reads in reverse order, and requires
`corpus.json`, `manifest.csv` and every `results_*.json` to be identical byte for byte, and
`export_metadata.json` to differ in `export_timestamp` alone.
"""

import hashlib
import json
import uuid

import pytest
from sqlalchemy import select, text

from app.db.models import (
    ANALYST_ASSESSMENTS,
    AUDIT_TARGET_GROUND_TRUTH,
    REVIEW_STATUS_REVIEWED,
    USER_ROLE_ADMIN,
    AnalysisReview,
    GroundTruth,
    LineageSplit,
    MediaGovernance,
)
from app.db.session import SessionLocal
from tests.conftest import DASHBOARD_ORIGIN
from tests.test_dataset_governance_api import CORPUS_METADATA, GOVERNANCE_URL
from tests.test_export_operational_dataset import load_script, seeded  # noqa: F401
from tests.test_ground_truth_api import (  # noqa: F401
    STATEMENT,
    database,
    events_for,
    forensic_state,
    make_analysis,
    make_user,
    new_hash,
    plain_http_environment,
    put_ground_truth,
    session,
    signed_in,
    stored_count,
)

pytestmark = pytest.mark.integration


# --- a human review is not Ground Truth ----------------------------------------------------


def put_review(client, analysis_id, assessment: str):
    return client.put(
        f"/api/v1/admin/analyses/{analysis_id}/review",
        json={
            "status": REVIEW_STATUS_REVIEWED,
            "analyst_assessment": assessment,
            "note": f"Assessed: {assessment}.",
        },
        headers={"Origin": DASHBOARD_ORIGIN},
    )


def ground_truth_state(session, sha256: str) -> tuple:
    db, _, _, _ = session
    db.commit()
    db.expire_all()
    record = db.get(GroundTruth, sha256)
    return (
        record.source_class,
        record.label,
        record.notes,
        record.actor_id,
        record.actor_email_snapshot,
        record.created_at,
        record.updated_at,
    )


def stored_assessment(session, analysis_id) -> str | None:
    db, _, _, _ = session
    db.commit()
    db.expire_all()
    return db.execute(
        select(AnalysisReview.analyst_assessment).where(
            AnalysisReview.analysis_id == analysis_id
        )
    ).scalar_one()


@pytest.mark.parametrize("assessment", ANALYST_ASSESSMENTS)
def test_a_human_review_does_not_create_ground_truth(session, assessment):
    sha256 = new_hash(session)
    analysis = make_analysis(session, sha256)
    client = signed_in(make_user(session, role=USER_ROLE_ADMIN))

    assert put_review(client, analysis.id, assessment).status_code == 200

    assert stored_assessment(session, analysis.id) == assessment
    assert stored_count(session, sha256) == 0
    assert events_for(session, sha256) == []


@pytest.mark.parametrize("assessment", ANALYST_ASSESSMENTS)
def test_a_human_review_does_not_change_or_audit_existing_ground_truth(session, assessment):
    """The review contradicts nothing here on purpose: whatever it says, the recorded label and
    its single creation event stand."""
    sha256 = new_hash(session)
    analysis = make_analysis(session, sha256)
    client = signed_in(make_user(session, role=USER_ROLE_ADMIN))
    assert put_ground_truth(client, sha256, STATEMENT).status_code == 200
    before = ground_truth_state(session, sha256)

    assert put_review(client, analysis.id, assessment).status_code == 200

    assert stored_assessment(session, analysis.id) == assessment
    assert ground_truth_state(session, sha256) == before
    [event] = events_for(session, sha256)
    assert event.target_type == AUDIT_TARGET_GROUND_TRUTH


# --- dataset governance is not forensic state ---------------------------------------------


@pytest.fixture
def lineages(session):
    """Lineage ids this test names; its governance rows and their splits go before `session`
    removes the media, audit events and analyses."""
    named: list[str] = []
    yield named

    db, _, _, hashes = session
    db.rollback()
    db.query(MediaGovernance).filter(MediaGovernance.media_sha256.in_(hashes)).delete(
        synchronize_session=False
    )
    db.query(LineageSplit).filter(LineageSplit.source_lineage_id.in_(named)).delete(
        synchronize_session=False
    )
    db.commit()


def put_governance(client, sha256: str, body: dict):
    return client.put(
        f"{GOVERNANCE_URL}/{sha256}", json=body, headers={"Origin": DASHBOARD_ORIGIN}
    )


def test_governing_media_leaves_every_analysis_of_it_untouched(session, lineages):
    """Verdict, signals (provenance included) and the human review, before and after both a
    creation and a revision of the governance statement, for every analysis of the bytes. The
    governance audit events are expected; they are counted only to show both writes stuck."""
    sha256 = new_hash(session)
    analyses = [make_analysis(session, sha256), make_analysis(session, sha256)]
    before = [forensic_state(session, analysis) for analysis in analyses]
    client = signed_in(make_user(session, role=USER_ROLE_ADMIN))
    lineage = f"r12t7-{uuid.uuid4().hex}"
    lineages.append(lineage)
    created = {"source_lineage_id": lineage, "dataset_split": "HOLDOUT", **CORPUS_METADATA}
    revised = {
        **created,
        "transformations": ["reencode"],
        "private": not CORPUS_METADATA["private"],
        "benchmark_family": "r12t7-revised-family",
    }

    assert put_governance(client, sha256, created).status_code == 200
    assert [forensic_state(session, analysis) for analysis in analyses] == before

    assert put_governance(client, sha256, revised).status_code == 200
    assert [forensic_state(session, analysis) for analysis in analyses] == before

    db, _, _, _ = session
    db.expire_all()
    record = db.get(MediaGovernance, sha256)
    assert (record.benchmark_family, record.private) == (
        "r12t7-revised-family",
        not CORPUS_METADATA["private"],
    )
    assert len(events_for(session, sha256)) == 2


# --- the export is deterministic -----------------------------------------------------------


DETERMINISTIC_ARTIFACTS = ("corpus.json", "manifest.csv")


def read_only_export(exporter, out):
    with SessionLocal() as db:
        db.execute(text("SET TRANSACTION READ ONLY"))
        metadata = exporter.export(db, out, None, {"database": "deepguard_test"})
        db.rollback()
    return metadata


def artifact_names(out) -> list[str]:
    return [*DETERMINISTIC_ARTIFACTS, *sorted(p.name for p in out.glob("results_*.json"))]


def assert_identical_exports(first, first_metadata, second, second_metadata):
    names = artifact_names(first)
    assert names == artifact_names(second)
    assert len(names) > len(DETERMINISTIC_ARTIFACTS), "no results file was written"
    for name in names:
        assert (first / name).read_bytes() == (second / name).read_bytes(), name

    for out, metadata in ((first, first_metadata), (second, second_metadata)):
        assert json.loads((out / "export_metadata.json").read_text()) == metadata
        assert set(metadata["artifact_hashes"]) == set(names)
        for name in names:
            digest = hashlib.sha256((out / name).read_bytes()).hexdigest()
            assert metadata["artifact_hashes"][name] == digest, name

    strip = lambda metadata: {k: v for k, v in metadata.items() if k != "export_timestamp"}
    assert strip(first_metadata) == strip(second_metadata)


def test_two_exports_of_one_database_are_byte_identical(seeded, tmp_path):
    exporter = load_script("export_operational_dataset")

    first_metadata = read_only_export(exporter, tmp_path / "first")
    second_metadata = read_only_export(exporter, tmp_path / "second")

    assert seeded["swap"] in (tmp_path / "first" / "corpus.json").read_text()
    assert_identical_exports(
        tmp_path / "first", first_metadata, tmp_path / "second", second_metadata
    )


def test_the_export_does_not_depend_on_the_order_rows_are_read(seeded, tmp_path):
    """`load_rows` carries no `ORDER BY`, so PostgreSQL promises no row order; two runs agreeing
    could be luck of the heap. Handing the exporter every row list reversed is not."""
    exporter = load_script("export_operational_dataset")

    with SessionLocal() as db:
        db.execute(text("SET TRANSACTION READ ONLY"))
        truths, governance, media, signals, facts = exporter.load_rows(db)
        db.rollback()

    def write(out, order):
        corpus = exporter.build_corpus(order(truths), order(governance), order(media), None)
        results = exporter.build_results(
            order(signals), {item.clip_id for item in corpus["items"]}
        )
        return exporter.write_export(
            out, corpus, results, split_filter=None,
            exporter_identity={"script": "r12-t7"}, source_db={"database": "deepguard_test"},
            query_facts=facts,
        )

    forward = write(tmp_path / "forward", list)
    reverse = write(tmp_path / "reverse", lambda rows: list(reversed(rows)))

    assert seeded["genuine"] in (tmp_path / "forward" / "corpus.json").read_text()
    assert_identical_exports(tmp_path / "forward", forward, tmp_path / "reverse", reverse)
