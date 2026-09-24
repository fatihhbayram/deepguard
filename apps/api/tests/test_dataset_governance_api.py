"""What the dataset governance routes persist, what they refuse, and what they audit (R12-T5).

Written to the conventions of `test_ground_truth_api.py`: real PostgreSQL, real sessions opened
through the real login route, and every row this file creates removed afterwards.

The claims divide into six groups.

*Identity.* Only the canonical 64-lowercase-hex spelling of a hash is accepted, in the path and
in `derived_from_sha256`; a hash no media row carries is refused.

*Splits.* One lineage, one split: a second file stated into an existing lineage with another
split is a 409, with the same split it is accepted. A record never changes lineage.

*Recordings.* Every record of one `recording_identity` is in one lineage.

*Derivation.* A parent must be uploaded *and* governed, must share the child's lineage, and the
derivation graph never gains a cycle — including one closed by updating an old record.

*The audit trail.* One event per change, with old and new for every field, `dataset_split`
included.

*The evaluator.* `scripts/eval/operational_metrics.py` reads the split through the real join.

*Corpus metadata (R12-T5A).* The eight fields a benchmark corpus record needs are written and
read back unchanged, default to "not recorded" — the booleans included, which have three states
— are refused when they would need converting, are audited with every other field, and `benchmark_family` is
independent of Ground Truth. The migration adds them without touching an existing row's
meaning, forwards and back.
"""

import ast
import hashlib
import importlib.util
import os
import subprocess
import sys
import uuid
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select, text
from sqlalchemy.engine import make_url
from sqlalchemy.exc import IntegrityError, SQLAlchemyError

from app.db.models import (
    ANALYSIS_STATUS_COMPLETED,
    AUDIT_ACTION_DATASET_GOVERNANCE_CREATED,
    AUDIT_ACTION_DATASET_GOVERNANCE_UPDATED,
    AUDIT_TARGET_DATASET_GOVERNANCE,
    USER_ROLE_ADMIN,
    USER_ROLE_USER,
    AdminAuditEvent,
    Analysis,
    AuthSession,
    GroundTruth,
    LineageSplit,
    MediaFile,
    MediaGovernance,
    User,
)
from app.dataset_governance import chain_reaches
from app.db.session import SessionLocal, engine
from app.main import app
from app.web_auth import ENVIRONMENT_VARIABLE, SESSION_COOKIE_NAME, hash_password
from tests.conftest import DASHBOARD_ORIGIN, TEST_DATABASE_URL

pytestmark = pytest.mark.integration

GOVERNANCE_URL = "/api/v1/admin/dataset-governance"

# A test fixture, not a credential.
PASSWORD = "correct-horse-battery-staple"

SEMANTIC_FIELDS = (
    "source_lineage_id",
    "dataset_split",
    "recording_identity",
    "transformations",
    "derived_from_sha256",
    "generation_pipeline",
    *(
        "license",
        "permission_status",
        "redistributable",
        "private",
        "stratum_primary",
        "source",
        "acquisition_type",
        "benchmark_family",
    ),
)

# The eight corpus metadata fields R12-T5A adds, with values shaped like the ones real corpora
# carry (`deepguard-corpus/r7t9/corpus.json`), including the longest kind of `source` line.
CORPUS_METADATA = {
    "license": "MAVOS-DD dataset terms (research use)",
    "permission_status": "private; local evaluation only",
    "redistributable": True,
    "private": False,
    "stratum_primary": "face_manipulation_in_the_wild_social_transcode_shorts",
    "source": (
        "34data/social-media-deepfakes-fake@f4b76fc47eb9e5bff66eceaa89185e51821f943d"
        "#v22-fake-social-media-deepfakes_001.zip:smd_fake_00a7fdb4.mp4"
    ),
    "acquisition_type": "platform_transcode_download",
    "benchmark_family": "inswapper",
}
CORPUS_METADATA_FIELDS = tuple(CORPUS_METADATA)

REPO_ROOT = Path(__file__).resolve().parents[3]
API_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def plain_http_environment(monkeypatch):
    monkeypatch.setenv(ENVIRONMENT_VARIABLE, "development")


@pytest.fixture(scope="module")
def database():
    try:
        with engine.connect():
            pass
    except SQLAlchemyError as error:
        pytest.skip(f"PostgreSQL is not reachable: {error.__class__.__name__}")

    return engine


@pytest.fixture
def session(database):
    """A real session that removes every row this file created.

    Governance rows go children first (the parent reference is RESTRICT), then the lineages this
    file named, then audit events, media, analyses and users.
    """
    users: list[uuid.UUID] = []
    analyses: list[uuid.UUID] = []
    hashes: list[str] = []
    lineages: list[str] = []

    with SessionLocal() as db:
        yield db, users, analyses, hashes, lineages

        db.rollback()
        remaining = set(hashes)
        while remaining:
            parents = {
                parent
                for (parent,) in db.execute(
                    select(MediaGovernance.derived_from_sha256).where(
                        MediaGovernance.media_sha256.in_(remaining)
                    )
                )
                if parent is not None
            }
            leaves = remaining - parents or remaining
            db.query(MediaGovernance).filter(
                MediaGovernance.media_sha256.in_(leaves)
            ).delete(synchronize_session=False)
            db.flush()
            remaining -= leaves
        for sha256 in hashes:
            db.query(AdminAuditEvent).filter(
                AdminAuditEvent.target_id == sha256
            ).delete(synchronize_session=False)
        db.query(LineageSplit).filter(LineageSplit.source_lineage_id.in_(lineages)).delete(
            synchronize_session=False
        )
        for analysis_id in analyses:
            db.query(MediaFile).filter(MediaFile.analysis_id == analysis_id).delete(
                synchronize_session=False
            )
            db.query(Analysis).filter(Analysis.id == analysis_id).delete()
        for user_id in users:
            db.query(AdminAuditEvent).filter(
                AdminAuditEvent.actor_id == user_id
            ).delete(synchronize_session=False)
            db.query(AuthSession).filter(AuthSession.user_id == user_id).delete()
            db.query(User).filter(User.id == user_id).delete()
        db.commit()


def make_user(session, *, role: str = USER_ROLE_USER) -> User:
    db, users, _, _, _ = session

    user = User(
        email=f"{uuid.uuid4().hex}@example.com",
        password_hash=hash_password(PASSWORD),
        role=role,
    )
    db.add(user)
    db.commit()
    users.append(user.id)

    return user


def new_hash(session) -> str:
    _, _, _, hashes, _ = session
    sha256 = hashlib.sha256(uuid.uuid4().bytes).hexdigest()
    hashes.append(sha256)

    return sha256


def new_lineage(session) -> str:
    _, _, _, _, lineages = session
    lineage = f"lineage-{uuid.uuid4().hex}"
    lineages.append(lineage)

    return lineage


def make_media(session, sha256: str | None = None) -> str:
    """A completed, decided analysis with a media row carrying these bytes."""
    db, _, analyses, _, _ = session
    sha256 = sha256 or new_hash(session)

    analysis = Analysis(
        status=ANALYSIS_STATUS_COMPLETED,
        risk_level="MANIPULATION_DETECTED",
        risk_rules_version="r9-v5.0.0",
        risk_calibration_id="c" * 64,
        risk_rule_id="R1",
    )
    db.add(analysis)
    db.flush()
    analyses.append(analysis.id)

    db.add(
        MediaFile(
            analysis_id=analysis.id,
            original_filename="clip.mov",
            content_type="video/quicktime",
            size_bytes=4096,
            original_sha256=sha256,
            original_storage_key=f"originals/{sha256}",
            format_name="mov,mp4,m4a,3gp,3g2,mj2",
            codec_name="h264",
            width=1920,
            height=1080,
            duration=12.34,
            frame_rate=30000 / 1001,
            pix_fmt="yuv420p",
            constant_frame_rate=True,
            was_normalized=False,
            was_assembled=False,
            acquisition_method="upload",
        )
    )
    db.commit()

    return sha256


def signed_in(user: User) -> TestClient:
    client = TestClient(app)
    response = client.post(
        "/api/v1/auth/login", json={"email": user.email, "password": PASSWORD}
    )

    assert response.status_code == 200, "the fixture account could not sign in"
    assert client.cookies.get(SESSION_COOKIE_NAME), "no session cookie was issued"

    return client


@pytest.fixture
def admin(session) -> TestClient:
    return signed_in(make_user(session, role=USER_ROLE_ADMIN))


def put(client: TestClient, sha256: str, body: dict):
    return client.put(
        f"{GOVERNANCE_URL}/{sha256}", json=body, headers={"Origin": DASHBOARD_ORIGIN}
    )


def statement(lineage: str, split: str = "CALIBRATION", **fields) -> dict:
    return {"source_lineage_id": lineage, "dataset_split": split, **fields}


def stored(session, sha256: str) -> MediaGovernance | None:
    db = session[0]
    db.commit()
    db.expire_all()

    return db.get(MediaGovernance, sha256)


def events_for(session, sha256: str) -> list[AdminAuditEvent]:
    db = session[0]
    db.commit()

    return list(
        db.execute(
            select(AdminAuditEvent)
            .where(AdminAuditEvent.target_id == sha256)
            .order_by(AdminAuditEvent.created_at, AdminAuditEvent.id)
        )
        .scalars()
        .all()
    )


def governed(session, admin, lineage: str, split: str = "CALIBRATION", **fields) -> str:
    """Upload and govern one file, returning its hash."""
    sha256 = make_media(session)
    response = put(admin, sha256, statement(lineage, split, **fields))
    assert response.status_code == 200, response.text

    return sha256


# --- identity -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "spelling",
    [
        lambda h: h.upper(),
        lambda h: h[:63],
        lambda h: h + "0",
        lambda h: h[:63] + "g",
        lambda h: "sha256:" + h,
    ],
    ids=["uppercase", "short", "long", "non-hex", "prefixed"],
)
def test_a_non_canonical_hash_is_refused_in_the_path(session, admin, spelling):
    sha256 = make_media(session)
    lineage = new_lineage(session)

    assert put(admin, spelling(sha256), statement(lineage)).status_code == 422
    assert admin.get(f"{GOVERNANCE_URL}/{spelling(sha256)}").status_code == 422
    assert stored(session, sha256) is None
    assert events_for(session, sha256) == []


@pytest.mark.parametrize(
    "spelling",
    [lambda h: h.upper(), lambda h: h[:63], lambda h: h[:63] + "g"],
    ids=["uppercase", "short", "non-hex"],
)
def test_a_non_canonical_parent_hash_is_refused(session, admin, spelling):
    lineage = new_lineage(session)
    parent = governed(session, admin, lineage)
    child = make_media(session)

    response = put(admin, child, statement(lineage, derived_from_sha256=spelling(parent)))

    assert response.status_code == 422
    assert stored(session, child) is None


def test_governance_for_bytes_nobody_uploaded_is_refused(session, admin):
    sha256 = new_hash(session)
    lineage = new_lineage(session)

    response = put(admin, sha256, statement(lineage))

    assert response.status_code == 404
    assert response.json() == {"detail": "media not found"}
    assert stored(session, sha256) is None
    assert session[0].get(LineageSplit, lineage) is None


def test_known_bytes_with_no_record_are_not_given_a_split(session, admin):
    sha256 = make_media(session)

    response = admin.get(f"{GOVERNANCE_URL}/{sha256}")

    assert response.status_code == 404
    assert response.json() == {"detail": "dataset governance not recorded"}


def test_the_contract_forbids_undeclared_fields(session, admin):
    sha256 = make_media(session)
    lineage = new_lineage(session)

    for extra in ({"manipulation_family": "FACE_SWAP"}, {"label": "GENUINE"}, {"actor_id": "x"}):
        assert put(admin, sha256, statement(lineage, **extra)).status_code == 422

    assert put(admin, sha256, statement(lineage, split="TRAINING")).status_code == 422
    assert put(admin, sha256, statement(lineage, transformations="reencode")).status_code == 422
    assert stored(session, sha256) is None


# --- splits ---------------------------------------------------------------------------------


def test_same_lineage_with_a_different_split_is_a_conflict(session, admin):
    lineage = new_lineage(session)
    governed(session, admin, lineage, "CALIBRATION")
    second = make_media(session)

    response = put(admin, second, statement(lineage, "HOLDOUT"))

    assert response.status_code == 409
    assert "split reassignment is not supported" in response.json()["detail"]
    assert stored(session, second) is None
    assert session[0].get(LineageSplit, lineage).dataset_split == "CALIBRATION"


def test_same_lineage_with_the_same_split_is_accepted(session, admin):
    lineage = new_lineage(session)
    first = governed(session, admin, lineage, "HOLDOUT")
    second = governed(session, admin, lineage, "HOLDOUT")

    assert stored(session, first).source_lineage_id == lineage
    assert stored(session, second).source_lineage_id == lineage
    response = admin.get(f"{GOVERNANCE_URL}/{second}")
    assert response.status_code == 200
    assert response.json()["dataset_split"] == "HOLDOUT"


def test_an_existing_record_cannot_change_its_split(session, admin):
    lineage = new_lineage(session)
    sha256 = governed(session, admin, lineage, "TEST")

    response = put(admin, sha256, statement(lineage, "CALIBRATION"))

    assert response.status_code == 409
    assert session[0].get(LineageSplit, lineage).dataset_split == "TEST"
    assert len(events_for(session, sha256)) == 1


def test_an_existing_record_cannot_move_to_another_lineage(session, admin):
    sha256 = governed(session, admin, new_lineage(session), "CALIBRATION")
    other = new_lineage(session)

    response = put(admin, sha256, statement(other, "HOLDOUT"))

    assert response.status_code == 409
    assert response.json() == {"detail": "source lineage cannot be changed"}
    assert session[0].get(LineageSplit, other) is None


def test_the_database_holds_one_split_per_lineage(session, admin):
    """The structural half: a second split row for one lineage is a duplicate key."""
    lineage = new_lineage(session)
    governed(session, admin, lineage, "CALIBRATION")
    db = session[0]

    db.add(LineageSplit(source_lineage_id=lineage, dataset_split="HOLDOUT"))
    with pytest.raises(IntegrityError):
        db.commit()
    db.rollback()


# --- recordings -----------------------------------------------------------------------------


def test_one_recording_cannot_span_two_lineages(session, admin):
    recording = f"rec-{uuid.uuid4().hex}"
    governed(session, admin, new_lineage(session), "CALIBRATION", recording_identity=recording)
    other = make_media(session)
    other_lineage = new_lineage(session)

    response = put(
        admin, other, statement(other_lineage, "HOLDOUT", recording_identity=recording)
    )

    assert response.status_code == 409
    assert response.json() == {
        "detail": "recording identity already belongs to another source lineage"
    }
    assert stored(session, other) is None
    assert session[0].get(LineageSplit, other_lineage) is None


def test_one_recording_in_one_lineage_is_accepted(session, admin):
    recording = f"rec-{uuid.uuid4().hex}"
    lineage = new_lineage(session)
    governed(session, admin, lineage, recording_identity=recording)
    second = governed(session, admin, lineage, recording_identity=recording)

    assert stored(session, second).recording_identity == recording


# --- derivation -----------------------------------------------------------------------------


def test_a_file_cannot_be_derived_from_itself(session, admin):
    sha256 = make_media(session)
    lineage = new_lineage(session)

    response = put(admin, sha256, statement(lineage, derived_from_sha256=sha256))

    assert response.status_code == 422
    assert stored(session, sha256) is None


def test_an_update_that_closes_a_cycle_is_refused(session, admin):
    """A → B exists; re-pointing A at B would make A → B → A. Creation order was safe; the
    update is what has to be caught, by walking the chain."""
    lineage = new_lineage(session)
    a = governed(session, admin, lineage)
    b = governed(session, admin, lineage, derived_from_sha256=a)

    response = put(admin, a, statement(lineage, derived_from_sha256=b))

    assert response.status_code == 409
    assert response.json() == {"detail": "derivation would create a cycle"}
    assert stored(session, a).derived_from_sha256 is None


def test_a_longer_cycle_is_refused(session, admin):
    lineage = new_lineage(session)
    a = governed(session, admin, lineage)
    b = governed(session, admin, lineage, derived_from_sha256=a)
    c = governed(session, admin, lineage, derived_from_sha256=b)

    assert put(admin, a, statement(lineage, derived_from_sha256=c)).status_code == 409
    # Re-pointing within the tree without a loop is fine.
    assert put(admin, c, statement(lineage, derived_from_sha256=a)).status_code == 200


def test_the_chain_walk_sees_every_ancestor():
    parents = {"c": "b", "b": "a", "a": None}
    assert chain_reaches("c", "a", parents.get)
    assert not chain_reaches("a", "c", parents.get)
    # A stored loop does not spin the walk forever.
    looped = {"x": "y", "y": "x"}
    assert not chain_reaches("x", "z", looped.get)


def test_a_parent_must_share_the_childs_lineage(session, admin):
    parent = governed(session, admin, new_lineage(session), "CALIBRATION")
    child = make_media(session)
    other = new_lineage(session)

    response = put(admin, child, statement(other, "HOLDOUT", derived_from_sha256=parent))

    assert response.status_code == 409
    assert response.json() == {"detail": "derived media must share its parent's source lineage"}
    assert stored(session, child) is None
    assert session[0].get(LineageSplit, other) is None


def test_a_parent_that_was_uploaded_but_not_governed_is_refused(session, admin):
    parent = make_media(session)
    child = make_media(session)

    response = put(admin, child, statement(new_lineage(session), derived_from_sha256=parent))

    assert response.status_code == 409
    assert response.json() == {"detail": "parent governance not recorded"}
    assert stored(session, child) is None


def test_a_parent_nobody_uploaded_is_refused(session, admin):
    child = make_media(session)

    response = put(
        admin, child, statement(new_lineage(session), derived_from_sha256=new_hash(session))
    )

    assert response.status_code == 409
    assert response.json() == {"detail": "parent media not found"}


def test_derived_media_is_in_its_parents_split_structurally(session, admin):
    """The child names no split of its own that could differ: it reaches the split through the
    shared lineage, and the database refuses a child in another lineage from its parent."""
    lineage = new_lineage(session)
    parent = governed(session, admin, lineage, "HOLDOUT")
    child = governed(
        session, admin, lineage, "HOLDOUT",
        derived_from_sha256=parent, transformations=["reencode", "resize"],
    )
    db = session[0]

    split = db.execute(
        select(LineageSplit.dataset_split)
        .join(MediaGovernance, MediaGovernance.source_lineage_id == LineageSplit.source_lineage_id)
        .where(MediaGovernance.media_sha256 == child)
    ).scalar_one()
    assert split == "HOLDOUT"

    # Bypassing the route: a child row in another lineage breaks the composite foreign key.
    elsewhere = new_lineage(session)
    db.add(LineageSplit(source_lineage_id=elsewhere, dataset_split="CALIBRATION"))
    db.commit()
    db.add(
        MediaGovernance(
            media_sha256=make_media(session),
            source_lineage_id=elsewhere,
            derived_from_sha256=parent,
            actor_id=uuid.uuid4(),
        )
    )
    with pytest.raises(IntegrityError):
        db.commit()
    db.rollback()


# --- the audit trail ------------------------------------------------------------------------


def test_every_change_is_audited_with_every_field_including_the_split(session, admin):
    lineage = new_lineage(session)
    sha256 = governed(session, admin, lineage, "VALIDATION", generation_pipeline="Sora")

    body = statement(
        lineage, "VALIDATION", generation_pipeline="Sora", transformations=["reencode"]
    )
    assert put(admin, sha256, body).status_code == 200
    # Unchanged: nothing is written.
    assert put(admin, sha256, body).status_code == 200

    created, updated = events_for(session, sha256)

    assert created.action == AUDIT_ACTION_DATASET_GOVERNANCE_CREATED
    assert created.target_type == AUDIT_TARGET_DATASET_GOVERNANCE
    assert set(created.changes) == set(SEMANTIC_FIELDS)
    assert all(change["old"] is None for change in created.changes.values())
    assert created.changes["dataset_split"] == {"old": None, "new": "VALIDATION"}

    assert updated.action == AUDIT_ACTION_DATASET_GOVERNANCE_UPDATED
    assert set(updated.changes) == set(SEMANTIC_FIELDS)
    assert updated.changes["dataset_split"] == {"old": "VALIDATION", "new": "VALIDATION"}
    assert updated.changes["transformations"] == {"old": None, "new": ["reencode"]}
    assert updated.changes["generation_pipeline"] == {"old": "Sora", "new": "Sora"}


def test_a_refused_statement_writes_nothing(session, admin):
    lineage = new_lineage(session)
    sha256 = governed(session, admin, lineage, "CALIBRATION")

    assert put(admin, sha256, statement(lineage, "TEST")).status_code == 409
    assert len(events_for(session, sha256)) == 1


# --- access ---------------------------------------------------------------------------------


def test_both_routes_are_for_administrators_only(session):
    sha256 = make_media(session)
    body = statement(new_lineage(session))

    anonymous = TestClient(app)
    assert anonymous.get(f"{GOVERNANCE_URL}/{sha256}").status_code == 401
    assert put(anonymous, sha256, body).status_code == 401

    user = signed_in(make_user(session))
    assert user.get(f"{GOVERNANCE_URL}/{sha256}").status_code == 403
    assert put(user, sha256, body).status_code == 403
    assert stored(session, sha256) is None


# --- the evaluator --------------------------------------------------------------------------


def load_operational_metrics():
    scripts = REPO_ROOT / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    spec = importlib.util.spec_from_file_location(
        "r12t5_operational_metrics", scripts / "eval" / "operational_metrics.py"
    )
    module = importlib.util.module_from_spec(spec)
    # Its dataclasses resolve string annotations through `sys.modules`.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    return module


def test_the_evaluator_reads_the_recorded_split_and_unavailable_without_one(session, admin):
    om = load_operational_metrics()
    governed_sha = governed(session, admin, new_lineage(session), "HOLDOUT")
    ungoverned_sha = make_media(session)

    db = session[0]
    db.commit()
    rows, facts = om.load_rows(db)
    db.rollback()

    by_sha = {row.media_sha256: row for row in rows}
    assert by_sha[governed_sha].split == "HOLDOUT"
    assert by_sha[ungoverned_sha].split == "unavailable"
    assert "lineage_splits" in facts["join"]
    assert facts["media_governance_records_total"] >= 1


# --- corpus metadata (R12-T5A) --------------------------------------------------------------


def test_corpus_metadata_is_written_and_read_back_unchanged(session, admin):
    lineage = new_lineage(session)
    sha256 = make_media(session)

    response = put(admin, sha256, statement(lineage, "TEST", **CORPUS_METADATA))

    assert response.status_code == 200, response.text
    for name, value in CORPUS_METADATA.items():
        assert response.json()[name] == value
    record = stored(session, sha256)
    for name, value in CORPUS_METADATA.items():
        assert getattr(record, name) == value
    fetched = admin.get(f"{GOVERNANCE_URL}/{sha256}").json()
    assert {name: fetched[name] for name in CORPUS_METADATA} == CORPUS_METADATA


def test_unstated_corpus_metadata_is_not_recorded(session, admin):
    """Nothing is filled in: every field, booleans included, stays null."""
    sha256 = governed(session, admin, new_lineage(session))

    record = stored(session, sha256)
    fetched = admin.get(f"{GOVERNANCE_URL}/{sha256}").json()

    for name in CORPUS_METADATA_FIELDS:
        assert getattr(record, name) is None
        assert fetched[name] is None


@pytest.mark.parametrize("value", [True, False, None])
def test_the_booleans_have_three_states(session, admin, value):
    """True and false are statements; null is the absence of one, and all three round-trip."""
    lineage = new_lineage(session)
    sha256 = governed(session, admin, lineage, redistributable=value, private=value)

    record = stored(session, sha256)
    fetched = admin.get(f"{GOVERNANCE_URL}/{sha256}").json()

    assert record.redistributable is value and fetched["redistributable"] is value
    assert record.private is value and fetched["private"] is value
    assert events_for(session, sha256)[0].changes["private"] == {"old": None, "new": value}


def test_a_put_is_the_whole_statement_for_corpus_metadata_too(session, admin):
    """An omitted field means "not recorded", not "leave as it was" — the T5 PUT semantics."""
    lineage = new_lineage(session)
    sha256 = governed(session, admin, lineage, **CORPUS_METADATA)

    assert put(admin, sha256, statement(lineage, license="CC-BY-4.0")).status_code == 200

    record = stored(session, sha256)
    assert record.license == "CC-BY-4.0"
    assert record.benchmark_family is None
    assert record.redistributable is None
    assert record.private is None


@pytest.mark.parametrize(
    "field, value",
    [
        ("redistributable", "true"),
        ("redistributable", 1),
        ("private", "false"),
        ("private", 0),
        ("license", ""),
        ("license", "x" * 256),
        ("source", "x" * 513),
        ("stratum_primary", "two\nlines"),
        ("benchmark_family", 7),
    ],
    ids=[
        "bool-from-string", "bool-from-int", "private-from-string", "private-from-int",
        "empty-string", "license-too-long", "source-too-long",
        "control-character", "family-from-int",
    ],
)
def test_corpus_metadata_that_would_need_converting_is_refused(session, admin, field, value):
    sha256 = make_media(session)

    response = put(admin, sha256, statement(new_lineage(session), **{field: value}))

    assert response.status_code == 422
    assert stored(session, sha256) is None
    assert events_for(session, sha256) == []


def test_corpus_metadata_changes_are_audited_old_and_new(session, admin):
    lineage = new_lineage(session)
    sha256 = governed(session, admin, lineage, **CORPUS_METADATA)
    revised = {**CORPUS_METADATA, "private": True, "benchmark_family": "blendface"}

    assert put(admin, sha256, statement(lineage, **revised)).status_code == 200
    # Unchanged: nothing is written.
    assert put(admin, sha256, statement(lineage, **revised)).status_code == 200

    created, updated = events_for(session, sha256)

    assert set(created.changes) == set(SEMANTIC_FIELDS)
    for name in CORPUS_METADATA_FIELDS:
        assert created.changes[name] == {"old": None, "new": CORPUS_METADATA[name]}

    assert updated.action == AUDIT_ACTION_DATASET_GOVERNANCE_UPDATED
    assert set(updated.changes) == set(SEMANTIC_FIELDS)
    for name in CORPUS_METADATA_FIELDS:
        assert updated.changes[name] == {"old": CORPUS_METADATA[name], "new": revised[name]}
    assert updated.changes["private"] == {"old": False, "new": True}
    assert updated.changes["benchmark_family"] == {"old": "inswapper", "new": "blendface"}


def test_corpus_metadata_leaves_the_split_rules_as_they_were(session, admin):
    """Metadata rides on the statement; it opens no way around a T5 guard."""
    lineage = new_lineage(session)
    sha256 = governed(session, admin, lineage, "CALIBRATION")

    response = put(admin, sha256, statement(lineage, "HOLDOUT", **CORPUS_METADATA))

    assert response.status_code == 409
    assert stored(session, sha256).license is None
    assert session[0].get(LineageSplit, lineage).dataset_split == "CALIBRATION"


def test_benchmark_family_is_independent_of_ground_truth(session, admin):
    """Stated by hand and stored as stated, whatever Ground Truth says or later says."""
    db = session[0]
    lineage = new_lineage(session)
    sha256 = make_media(session)
    db.add(
        GroundTruth(
            media_sha256=sha256,
            source_class="CONTROLLED_TEST",
            label="GENUINE",
            actor_id=uuid.uuid4(),
        )
    )
    db.commit()

    try:
        # A face-swap family on bytes Ground Truth calls genuine is accepted, not corrected.
        response = put(admin, sha256, statement(lineage, benchmark_family="inswapper"))
        assert response.status_code == 200, response.text
        assert stored(session, sha256).benchmark_family == "inswapper"

        # Revising the Ground Truth does not touch the governance record.
        truth = db.get(GroundTruth, sha256)
        truth.label = "AI_GENERATED"
        db.commit()
        assert stored(session, sha256).benchmark_family == "inswapper"
        assert len(events_for(session, sha256)) == 1

        # And the family is never offered from Ground Truth when none is stated.
        other = make_media(session)
        db.add(
            GroundTruth(
                media_sha256=other,
                source_class="CONTROLLED_TEST",
                label="FACE_SWAP",
                actor_id=uuid.uuid4(),
            )
        )
        db.commit()
        assert put(admin, other, statement(lineage)).status_code == 200
        assert stored(session, other).benchmark_family is None
        assert admin.get(f"{GOVERNANCE_URL}/{other}").json()["benchmark_family"] is None
    finally:
        db.rollback()
        db.query(GroundTruth).filter(
            GroundTruth.media_sha256.in_(session[3])
        ).delete(synchronize_session=False)
        db.commit()


def test_the_governance_module_does_not_read_ground_truth():
    """The structural half of independence: no import path from a label to a family."""
    for module in ("dataset_governance.py", "api/admin_dataset_governance.py"):
        tree = ast.parse((API_ROOT / "app" / module).read_text(encoding="utf-8"))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported |= {alias.name for alias in node.names}
            elif isinstance(node, ast.ImportFrom):
                imported.add(node.module or "")
                imported |= {alias.name for alias in node.names}
        assert "app.ground_truth" not in imported
        assert "GroundTruth" not in imported
        assert "derive_manipulation_family" not in imported


# --- the corpus metadata migration, forwards and back ---------------------------------------
#
# On a disposable database of its own, for the reason `test_verdict_persistence.py` gives: a
# downgrade run against the shared test database would leave the rest of the suite on a schema
# that does not ship.

METADATA_REVISION = "c5e2a7d49f18"
GOVERNANCE_REVISION = "b4d8e2f61c37"


def alembic(command: list[str], url) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "alembic", *command],
        cwd=API_ROOT,
        env={**os.environ, "DATABASE_URL": url.render_as_string(hide_password=False)},
        capture_output=True,
        text=True,
    )


@pytest.fixture
def database_before_the_metadata(database):
    """A fresh database migrated to head and wound back to the T5 governance revision."""
    name = f"deepguard_migration_{uuid.uuid4().hex[:12]}"
    url = make_url(TEST_DATABASE_URL).set(database=name)
    server = create_engine(
        make_url(TEST_DATABASE_URL).set(database="postgres"), isolation_level="AUTOCOMMIT"
    )

    try:
        with server.connect() as connection:
            connection.execute(text(f'CREATE DATABASE "{name}"'))

        upgraded = alembic(["upgrade", "head"], url)
        assert upgraded.returncode == 0, upgraded.stderr
        wound_back = alembic(["downgrade", GOVERNANCE_REVISION], url)
        assert wound_back.returncode == 0, wound_back.stderr

        yield url
    finally:
        with server.connect() as connection:
            connection.execute(text(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)'))
        server.dispose()


def governance_columns(url) -> dict[str, dict]:
    target = create_engine(url)
    try:
        with target.connect() as connection:
            rows = connection.execute(
                text(
                    "SELECT column_name, data_type, is_nullable, character_maximum_length, "
                    "column_default "
                    "FROM information_schema.columns WHERE table_name = 'media_governance'"
                )
            ).all()
    finally:
        target.dispose()
    return {
        row.column_name: {
            "type": row.data_type,
            "nullable": row.is_nullable == "YES",
            "length": row.character_maximum_length,
            "default": row.column_default,
        }
        for row in rows
    }


def test_the_metadata_migration_leaves_existing_rows_unrecorded(
    database_before_the_metadata,
):
    url = database_before_the_metadata
    sha256 = hashlib.sha256(uuid.uuid4().bytes).hexdigest()
    target = create_engine(url)

    try:
        # A T5-era row, written as that schema allowed — raw SQL, ORM uninvolved, because the
        # mapped class already carries columns this database does not have yet.
        with target.begin() as connection:
            connection.execute(
                text(
                    "INSERT INTO lineage_splits (source_lineage_id, dataset_split) "
                    "VALUES ('lineage-pre-t5a', 'CALIBRATION')"
                )
            )
            connection.execute(
                text(
                    "INSERT INTO media_governance (media_sha256, source_lineage_id, actor_id) "
                    "VALUES (:sha, 'lineage-pre-t5a', :actor)"
                ),
                {"sha": sha256, "actor": uuid.uuid4()},
            )

        upgraded = alembic(["upgrade", METADATA_REVISION], url)
        assert upgraded.returncode == 0, upgraded.stderr

        with target.connect() as connection:
            row = connection.execute(
                text("SELECT * FROM media_governance WHERE media_sha256 = :sha"),
                {"sha": sha256},
            ).mappings().one()
    finally:
        target.dispose()

    # Nothing is stated on the row's behalf — not even a "safe" boolean.
    for name in CORPUS_METADATA_FIELDS:
        assert row[name] is None
    # The T5 half of the row is as it was.
    assert row["source_lineage_id"] == "lineage-pre-t5a"
    assert row["derived_from_sha256"] is None

    columns = governance_columns(url)
    for name in ("redistributable", "private"):
        assert columns[name] == {
            "type": "boolean", "nullable": True, "length": None, "default": None
        }
    assert columns["license"] == {
        "type": "character varying", "nullable": True, "length": 255, "default": None
    }
    assert columns["source"] == {
        "type": "character varying", "nullable": True, "length": 512, "default": None
    }


def test_the_metadata_migration_downgrades_to_the_t5_schema(database_before_the_metadata):
    url = database_before_the_metadata
    before = set(governance_columns(url))

    assert alembic(["upgrade", METADATA_REVISION], url).returncode == 0
    assert set(governance_columns(url)) - before == set(CORPUS_METADATA_FIELDS)

    downgraded = alembic(["downgrade", GOVERNANCE_REVISION], url)
    assert downgraded.returncode == 0, downgraded.stderr
    assert set(governance_columns(url)) == before
