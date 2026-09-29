"""R14-T5 (D1): the analysis read reports every media hash, read-only, beside `original_sha256`.

`media_files.analysis_id` is not unique, so an analysis can carry several media rows. The admin
promotion form must let the analyst choose among all of them, and it builds that choice from
`media_sha256s` on the analysis read. What is held here:

* several media rows → every distinct hash, sorted, once each;
* one media row → a list of that one hash;
* `original_sha256` is still reported and is one of the list;
* only the detail read carries it: the listing and the public contract do not gain it;
* an analysis the caller may not see reveals no hashes.
"""

import hashlib
import uuid

import pytest

from app.api.public_v1.analyses import PublicAnalysis
from app.db.models import (
    ANALYSIS_STATUS_COMPLETED,
    USER_ROLE_USER,
    Analysis,
    AuthSession,
    MediaFile,
    User,
)
from app.db.session import SessionLocal
from app.web_auth import hash_password
from tests.test_admin_analytics import (  # noqa: F401
    PASSWORD,
    database,
    plain_http_environment,
    signed_in,
)

pytestmark = pytest.mark.integration


@pytest.fixture
def session(database):
    """A real session that removes the analyses (media rows cascade) and then the users."""
    users: list[uuid.UUID] = []
    analyses: list[uuid.UUID] = []

    with SessionLocal() as db:
        yield db, users, analyses

        db.rollback()
        for analysis_id in analyses:
            db.query(Analysis).filter(Analysis.id == analysis_id).delete()
        for user_id in users:
            db.query(AuthSession).filter(AuthSession.user_id == user_id).delete()
            db.query(User).filter(User.id == user_id).delete()
        db.commit()


def make_owner(session) -> User:
    db, users, _ = session
    user = User(
        email=f"{uuid.uuid4().hex}@example.com",
        password_hash=hash_password(PASSWORD),
        role=USER_ROLE_USER,
        is_active=True,
    )
    db.add(user)
    db.commit()
    users.append(user.id)
    return user


def new_hash() -> str:
    return hashlib.sha256(uuid.uuid4().bytes).hexdigest()


def make_analysis(session, owner: User, hashes: list[str]) -> Analysis:
    """A completed analysis owned by `owner`, with one media row per entry of `hashes`."""
    db, _, analyses = session

    analysis = Analysis(status=ANALYSIS_STATUS_COMPLETED, owner_id=owner.id)
    db.add(analysis)
    db.flush()
    analyses.append(analysis.id)

    for index, sha256 in enumerate(hashes):
        db.add(
            MediaFile(
                analysis_id=analysis.id,
                original_filename=f"clip-{index}.mp4",
                content_type="video/mp4",
                size_bytes=4096,
                original_sha256=sha256,
                original_storage_key=f"originals/{sha256}-{index}",
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

    return analysis


def test_an_analysis_with_several_media_reports_every_distinct_hash_sorted(session):
    owner = make_owner(session)
    first, second = sorted([new_hash(), new_hash()])
    # Inserted out of order, and one hash twice: the list is sorted and each hash appears once.
    analysis = make_analysis(session, owner, [second, first, second])

    response = signed_in(owner).get(f"/api/v1/analyses/{analysis.id}")

    assert response.status_code == 200
    body = response.json()
    assert body["media_sha256s"] == [first, second]
    # The existing field is still there, still one string, and one of the list.
    assert isinstance(body["original_sha256"], str)
    assert body["original_sha256"] in body["media_sha256s"]


def test_an_analysis_with_one_media_reports_a_list_of_that_hash(session):
    owner = make_owner(session)
    sha256 = new_hash()
    analysis = make_analysis(session, owner, [sha256])

    body = signed_in(owner).get(f"/api/v1/analyses/{analysis.id}").json()

    assert body["media_sha256s"] == [sha256]
    assert body["original_sha256"] == sha256


def test_the_listing_does_not_gain_the_field(session):
    """Detail only: the listing's field set and per-page query count are pinned contracts
    (`test_analysis_listing.py`), and nothing that reads the listing needs the list."""
    owner = make_owner(session)
    analysis = make_analysis(session, owner, [new_hash(), new_hash()])

    items = signed_in(owner).get("/api/v1/analyses").json()

    mine = [item for item in items if item["id"] == str(analysis.id)]
    assert mine
    assert all("media_sha256s" not in item for item in mine)


def test_an_analysis_the_caller_cannot_see_reveals_no_hashes(session):
    owner = make_owner(session)
    stranger = make_owner(session)
    analysis = make_analysis(session, owner, [new_hash()])

    response = signed_in(stranger).get(f"/api/v1/analyses/{analysis.id}")

    assert response.status_code == 404
    assert "media_sha256s" not in response.json()


def test_the_public_contract_does_not_gain_the_field():
    assert "media_sha256s" not in PublicAnalysis.model_fields
