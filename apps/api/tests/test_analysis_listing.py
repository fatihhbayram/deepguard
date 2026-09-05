"""Unit coverage for the dashboard listing endpoint.

The session is faked here so the suite needs no live database: what is under test is the
query this route builds and how it maps rows onto the response. The query is asserted by
compiling it, because a fake driver cannot prove ordering or a limit.
`tests/test_persistence.py` covers the same endpoint against real PostgreSQL.

The route has demanded a signed-in account since R1-T2, so the caller here is an
administrator: an administrator's listing is the unnarrowed one, which is what leaves every
existing assertion in this file about the *shape* of the query rather than about who asked
for it. The ownership filter is its own small group of tests below, and it is proven against
real users and real rows in `tests/test_dashboard_authorization.py`.
"""

import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.exc import OperationalError

from app.api.analyses import DASHBOARD_SEGMENTS, RECENT_ANALYSES_LIMIT
from app.db.models import USER_ROLE_ADMIN, USER_ROLE_USER, User
from app.db.session import get_session
from app.main import app
from app.web_auth import require_user

CREATED_AT = datetime(2026, 8, 19, 18, 8, 1, tzinfo=timezone.utc)

# Exactly the fields the dashboard renders. Anything else appearing here would be a leak
# of storage internals or of a phase that has not happened yet.
EXPECTED_FIELDS = {
    "id",
    "status",
    "created_at",
    "risk_level",
    "risk_rules_version",
    "risk_rule_id",
    "risk_calibration_id",
    "original_filename",
    "declared_content_type",
    "size_bytes",
    "original_sha256",
    "was_normalized",
    "was_assembled",
    "media",
    "synthetic_video",
    "provenance",
    "active_speaker",
    "audio_authenticity",
    "face_manipulation",
    "lip_forensics",
}

# What ffprobe established about the original, as the database kept it. `major_brand` is
# absent because no column holds it — the listing reports the stored evidence, not the
# probe's full output.
EXPECTED_MEDIA_FIELDS = {
    "format_name",
    "codec_name",
    "width",
    "height",
    "duration",
    "frame_rate",
    "pix_fmt",
    "constant_frame_rate",
}

# The signal object the dashboard receives: the provider's own identity, state and
# figures. `risk_level` is not among them — risk is a product-level decision on the
# analysis, never a per-detector verdict (rule 11) — and neither is the raw metadata
# document, which carries diagnostic detail on a failure.
EXPECTED_SIGNAL_FIELDS = {
    "provider",
    "signal_type",
    "status",
    "score",
    "provider_version",
    "logit",
    "total_clips",
    "segments",
}

# The provenance object the dashboard receives: what the file itself claims, and the
# state of the reading that established it. No score — a signature is not a figure on a
# scale — and no raw metadata document, which holds diagnostic detail on a failure.
EXPECTED_PROVENANCE_FIELDS = {
    "provider",
    "signal_type",
    "status",
    "provider_version",
    "manifest_exists",
    "validation_state",
    "claim_generator",
    "signature_issuer",
    "remote_manifest_url",
}

# The active-speaker object the dashboard receives. No score: this detector reports a
# timeline, not a figure on a scale, and a number here would sit beside NVIDIA's synthetic
# probability as though the two could be compared.
EXPECTED_ACTIVE_SPEAKER_FIELDS = {
    "provider",
    "signal_type",
    "status",
    "provider_version",
    "total_speaking_segments",
    "segments_truncated",
    "segments",
}

# The audio object the dashboard receives. No score, and no field that could stand in for
# one: the checkpoint publishes no threshold and no calibration, so there is no file-level
# figure to report and the windows are the whole of the evidence.
EXPECTED_AUDIO_FIELDS = {
    "provider",
    "signal_type",
    "status",
    "provider_version",
    "total_audio_windows",
    "persisted_audio_windows",
    "windows_truncated",
    "windows",
}

# One clip of provider evidence: the frame index NVIDIA scored and the logit it gave it.
# No time range and no probability, because NVIDIA reports neither per clip.
EXPECTED_SEGMENT_FIELDS = {"clip_index", "logit"}

# One speaking segment: a real time range and the identity it is about. No clip index and
# no logit, because an active-speaker result has neither.
EXPECTED_SPEAKING_FIELDS = {"start_time", "end_time", "face_id", "speaker_label"}

# One audio window: where in the sequence it sits, the preprocessing bounds it covers and
# both of the model's raw outputs. No score and no label, because the model emits neither.
EXPECTED_AUDIO_WINDOW_FIELDS = {
    "clip_index",
    "start_time",
    "end_time",
    "logit",
    "bona_fide_logit",
}

PROBABILITY = 0.7929722666740417
FUNCTION_ID = "847b6e53-0133-452d-ab85-d7acf3ace723"

# The risk trace, spelled out rather than imported from `app.risk_engine`. These tests are
# about the listing handing back what the row holds; reading the engine's own constants
# here would let an endpoint that recomputed the decision pass anyway.
RULES_VERSION = "p7-v1.0.0"
CALIBRATION_ID = "3e362e8edfe253437234e3c291230a2921a6344555ab0861ee5871c53d20949c"
RULE_CALIBRATED_HIGH = "R100"
RULE_INDETERMINATE_BAND = "R200"
RULE_UNVALIDATED_PROVIDER = "R010"
ASD_FUNCTION_ID = "9e93cf1e-de1e-4f1f-8b1e-0c3f1a2d5b77"
C2PA_SDK_VERSION = "0.90.14"
AASIST_CHECKPOINT = (
    "SpeechAntiSpoofingBenchmarks/AASIST@16774d458d86d2a021ae31646c1bf66a5331b53e"
)
FACETORCH_CHECKPOINT = (
    "tomas-gajarsky/facetorch-deepfake-efficientnet-b7"
    "@4acc494f37eb63d7457166eff2acb45c5b04b9a6"
)

# The face classifier's own figure for the clip. Deliberately above `T_HIGH`, the boundary
# the *calibrated* signal is banded on: the listing must hand this number back untouched
# beside a MEDIUM decision, which is what proves it is not being read as a risk input.
FACE_SCORE = 0.9931

LIPFORENSICS_MODEL = (
    "https://github.com/ahaliassos/LipForensics"
    "@d0bf5553bfb9676f1771d590472b26a3a76de894"
    "+4b7790bc8e02d0c25ecfa0d8d6a2907123c2206cc32e2bad6044e50f013c253d"
)

# The mouth-dynamics model's own figure for the clip. High on its own scale, for the same reason the
# face score above is: the listing must hand it back untouched beside a decision no rule let it
# near, which is what proves it is not being read as a risk input.
LIP_FORENSICS_SCORE = 0.9612


def listing_row(**overrides):
    """A row shaped like the one the select emits, with column names, not model names.

    The default carries a successful NVIDIA signal, since that is what an analysis run
    through the current pipeline has, and the risk decision that signal's score earns it:
    0.79 sits below `T_HIGH`, so the indeterminate band is what the engine wrote.
    """
    values = {
        "id": uuid.uuid4(),
        "status": "completed",
        "created_at": CREATED_AT,
        "risk_level": "MEDIUM",
        "risk_rules_version": RULES_VERSION,
        "risk_rule_id": RULE_INDETERMINATE_BAND,
        "risk_calibration_id": CALIBRATION_ID,
        "original_filename": "clip.mp4",
        # The column is `content_type`; the response renames it to `declared_content_type`.
        "content_type": "video/mp4",
        "size_bytes": 13054,
        "original_sha256": "a" * 64,
        "was_normalized": False,
        # How the artifact was acquired (R7-T1). False is every upload and every URL whose
        # source served one file; the listing reports it so a reader is never left to assume
        # bytes were published as they are stored.
        "was_assembled": False,
        # What ffprobe established about the original, exactly as `media_files` holds it.
        "format_name": "mov,mp4,m4a,3gp,3g2,mj2",
        "codec_name": "h264",
        "width": 1280,
        "height": 720,
        "duration": 5.0,
        "frame_rate": 30.0,
        "pix_fmt": "yuv420p",
        "constant_frame_rate": True,
        "signal_id": uuid.uuid4(),
        "signal_provider": "nvidia",
        "signal_type": "synthetic_video",
        "signal_status": "SUCCESS",
        "signal_score": PROBABILITY,
        "signal_provider_version": FUNCTION_ID,
        "signal_metadata": {"logit": 1.9142135381698608, "total_clips": 7},
        "provenance_provider": "c2pa",
        "provenance_signal_type": "provenance",
        "provenance_status": "SUCCESS",
        "provenance_provider_version": C2PA_SDK_VERSION,
        "provenance_metadata": {
            "manifest_exists": True,
            "validation_state": "Valid",
            "validation_failures": ["signingCredential.untrusted"],
            "is_embedded": True,
            "remote_manifest_url": None,
            "active_manifest_label": "urn:c2pa:6f0e1a2b",
            "claim_generator": "test-camera",
            "signature_issuer": "Test Signing Cert",
            "signature_time": None,
            "assertion_labels": ["c2pa.actions.v2"],
        },
        "active_speaker_id": uuid.uuid4(),
        "active_speaker_provider": "nvidia",
        "active_speaker_signal_type": "active_speaker",
        "active_speaker_status": "SUCCESS",
        "active_speaker_provider_version": ASD_FUNCTION_ID,
        "active_speaker_metadata": {
            "frame_rate": 30.0,
            "total_frames": 150,
            "total_speaking_segments": 2,
            "segments_truncated": False,
            "speaker_detection_threshold": 0.5,
            "diarized_speakers": {"SPEAKER_00": 0},
        },
        "audio_id": uuid.uuid4(),
        "audio_provider": "aasist",
        "audio_signal_type": "audio_authenticity",
        "audio_status": "SUCCESS",
        "audio_provider_version": AASIST_CHECKPOINT,
        "audio_metadata": {
            "model_repository": "SpeechAntiSpoofingBenchmarks/AASIST",
            "model_revision": "16774d458d86d2a021ae31646c1bf66a5331b53e",
            "model_sha256": "130e5362" + "0" * 56,
            "sample_rate": 16000,
            "channels": 1,
            "window_samples": 64600,
            "window_padding_scheme": "repeat-tile",
            "total_samples": 129200,
            "total_audio_windows": 2,
            "persisted_audio_windows": 2,
            "windows_truncated": False,
            "window_bounds": "deepguard_preprocessing",
            "bona_fide_logit_index": 1,
        },
        "face_provider": "efficientnet-b7",
        "face_signal_type": "face_manipulation",
        "face_status": "SUCCESS",
        "face_score": FACE_SCORE,
        "face_provider_version": FACETORCH_CHECKPOINT,
        "face_metadata": {
            "classifier_repository": "tomas-gajarsky/facetorch-deepfake-efficientnet-b7",
            "classifier_revision": "4acc494f37eb63d7457166eff2acb45c5b04b9a6",
            "classifier_sha256": "97b49a70" + "0" * 56,
            "locator_repository": "opencv/opencv_zoo",
            "locator_revision": "47534e27c9851bb1128ccc0102f1145e27f23f98",
            "locator_sha256": "8f2383e4" + "0" * 56,
            "torch_version": "2.13.0+cpu",
            "input_size": 380,
            "crop_margin": 1 / 3,
            "face_score_threshold": 0.6,
            "frames_requested": 8,
            "frames_decoded": 8,
            "frames_scored": 6,
            "frame_scores": [
                {"frame_index": 0, "probability": 0.91},
                {"frame_index": 14, "probability": 0.88},
            ],
        },
        "lip_forensics_provider": "lipforensics",
        "lip_forensics_signal_type": "lip_forensics",
        "lip_forensics_status": "SUCCESS",
        "lip_forensics_score": LIP_FORENSICS_SCORE,
        "lip_forensics_provider_version": LIPFORENSICS_MODEL,
        "lip_forensics_metadata": {
            "weights_origin": "https://drive.google.com/file/d/xyz (upstream README)",
            "weights_sha256": "4b7790bc" + "0" * 56,
            "upstream_repository": "https://github.com/ahaliassos/LipForensics",
            "upstream_revision": "d0bf5553bfb9676f1771d590472b26a3a76de894",
            "source_sha256": {"models/tcn.py": "35f57b7b" + "0" * 56},
            "landmark_library": "face-alignment 1.5.0",
            "landmark_compiled": False,
            "face_detector_sha256": "619a3168" + "0" * 56,
            "landmark_model_sha256": "11f355bf" + "0" * 56,
            "torch_version": "2.13.0+cpu",
            "device": "cpu",
            "frames_per_window": 25,
            "crop_size": 96,
            "input_size": 88,
            "windows_requested": 4,
            "windows_read": 4,
            "windows_scored": 3,
            "window_logits": [
                {"start_frame": 0, "logit": 3.1},
                {"start_frame": 58, "logit": 2.4},
            ],
        },
    }

    return SimpleNamespace(**{**values, **overrides})


def unsignalled_row(**overrides):
    """A row for an analysis neither outer join found a signal for.

    The risk decision is `UNKNOWN`, which is what the engine concludes when there is no
    eligible direct evidence to weigh — a decision that was taken, not a missing one.
    """
    return listing_row(
        risk_level="UNKNOWN",
        risk_rule_id=RULE_UNVALIDATED_PROVIDER,
        signal_id=None,
        signal_provider=None,
        signal_type=None,
        signal_status=None,
        signal_score=None,
        signal_provider_version=None,
        signal_metadata=None,
        provenance_provider=None,
        provenance_signal_type=None,
        provenance_status=None,
        provenance_provider_version=None,
        provenance_metadata=None,
        active_speaker_id=None,
        active_speaker_provider=None,
        active_speaker_signal_type=None,
        active_speaker_status=None,
        active_speaker_provider_version=None,
        active_speaker_metadata=None,
        audio_id=None,
        audio_provider=None,
        audio_signal_type=None,
        audio_status=None,
        audio_provider_version=None,
        audio_metadata=None,
        face_provider=None,
        face_signal_type=None,
        face_status=None,
        face_score=None,
        face_provider_version=None,
        face_metadata=None,
        lip_forensics_provider=None,
        lip_forensics_signal_type=None,
        lip_forensics_status=None,
        lip_forensics_score=None,
        lip_forensics_provider_version=None,
        lip_forensics_metadata=None,
        **overrides,
    )


def provenance_row(metadata, status: str = "SUCCESS", **overrides):
    """A row whose provenance reading ended in `status`, carrying `metadata`."""
    return listing_row(
        provenance_status=status,
        provenance_metadata=metadata,
        **overrides,
    )


def failed_signal_row(status: str, **overrides):
    """A row whose detection did not produce a number, as the pipeline persists it."""
    return listing_row(
        signal_status=status,
        signal_score=None,
        signal_provider_version=None,
        signal_metadata={"error": "NvidiaProviderTimeout"},
        **overrides,
    )


def failed_active_speaker_row(status: str, **overrides):
    """A row whose active-speaker chain did not produce a timeline."""
    return listing_row(
        active_speaker_status=status,
        active_speaker_provider_version=None,
        active_speaker_metadata={"error": "NvidiaActiveSpeakerTimeout"},
        **overrides,
    )


def failed_audio_row(status: str, **overrides):
    """A row whose audio reading did not produce any windows."""
    return listing_row(
        audio_status=status,
        audio_provider_version=None,
        audio_metadata={"error": "AudioDetectorModelUnavailable"},
        **overrides,
    )


def failed_face_row(status: str, **overrides):
    """A row whose face-manipulation reading did not produce a score.

    `FaceDetectorNoFaceFound` is the ordinary case rather than a fault: the classifier was
    never asked, so there is no number, and the absent score is the whole point.
    """
    return listing_row(
        face_status=status,
        face_score=None,
        face_provider_version=None,
        face_metadata={"error": "FaceDetectorNoFaceFound"},
        **overrides,
    )


def segment_row(signal_id, clip_index: int, logit: float):
    """A row shaped like the one the segment select emits."""
    return SimpleNamespace(signal_id=signal_id, clip_index=clip_index, logit=logit)


def audio_window_row(signal_id, index: int, bounds, logits):
    """A row shaped like the one the audio-window select emits."""
    return SimpleNamespace(
        signal_id=signal_id,
        clip_index=index,
        start_time=bounds[0],
        end_time=bounds[1],
        logit=logits[0],
        bona_fide_logit=logits[1],
    )


def speaking_row(signal_id, start: float, end: float, face_id: int, label: str | None):
    """A row shaped like the one the speaking-timeline select emits."""
    return SimpleNamespace(
        signal_id=signal_id,
        start_time=start,
        end_time=end,
        face_id=face_id,
        speaker_label=label,
    )


class FakeSession:
    """Stand-in for a SQLAlchemy session that records the statements it was given.

    The route issues up to four: the listing, the clip evidence for the detection signals it
    found, the speaking timeline for the active-speaker signals it found, and the windows
    for the audio signals it found. Any evidence query is skipped when nothing was found to
    look up, so they are told apart by the columns they read rather than by their position.
    """

    def __init__(self, rows=(), segment_rows=(), speaking_rows=(), audio_rows=()):
        self.rows = list(rows)
        self.segment_rows = list(segment_rows)
        self.speaking_rows = list(speaking_rows)
        self.audio_rows = list(audio_rows)
        self.execute_error = None
        self.statements = []

    def execute(self, statement):
        self.statements.append(statement)
        if self.execute_error is not None:
            raise self.execute_error

        # Resolved now rather than inside the lambda: by the time the route reads the rows,
        # a later statement may already have been recorded.
        rows = self.answer(len(self.statements), statement)

        return SimpleNamespace(all=lambda: rows)

    def answer(self, position: int, statement):
        if position == 1:
            return self.rows

        # `bona_fide_logit` is checked first and is the only discriminator that works: the
        # audio query reads `clip_index` too, so testing for that column first would hand
        # the audio statement the clip rows.
        sql = str(statement)
        if "bona_fide_logit" in sql:
            return self.audio_rows

        return self.segment_rows if "clip_index" in sql else self.speaking_rows


def account(role: str = USER_ROLE_ADMIN) -> User:
    """A signed-in account, never persisted.

    The routes take the user their dependency returns and read two things off it — the id
    and the role — so an unsaved instance is the whole of what these tests need. Signing in
    for real, against real rows, is what `tests/test_dashboard_authorization.py` does.
    """
    return User(
        id=uuid.uuid4(),
        email=f"{uuid.uuid4().hex}@example.com",
        password_hash="unused",
        role=role,
        is_active=True,
    )


@pytest.fixture
def fake_session():
    session = FakeSession()
    app.dependency_overrides[get_session] = lambda: session
    yield session
    app.dependency_overrides.clear()


@pytest.fixture
def admin():
    """The account every request in this module is made as, unless a test says otherwise."""
    return account(USER_ROLE_ADMIN)


@pytest.fixture
def client(fake_session, admin):
    app.dependency_overrides[require_user] = lambda: admin

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def anonymous_client(fake_session):
    """A client with no session at all, for the tests about being refused one."""
    with TestClient(app) as test_client:
        yield test_client


def compiled(session, index: int = 0) -> str:
    """One statement the route issued, as literal SQL. The listing is the first."""
    statement = session.statements[index]

    return str(statement.compile(compile_kwargs={"literal_binds": True}))


# --- who may read the listing at all ---------------------------------------------------


def test_an_unauthenticated_listing_is_refused(anonymous_client, fake_session):
    """No cookie, no listing — and, just as importantly, no query.

    The refusal has to come from the dependency, before the route body runs. A route that
    read the analyses and then decided whether to show them would still have read them, and
    the next mistake in that shape is one that forgets the second half.
    """
    response = anonymous_client.get("/api/v1/analyses")

    assert response.status_code == 401
    assert response.json() == {"detail": "Not authenticated"}
    assert fake_session.statements == []


def test_an_unauthenticated_detail_read_is_refused(anonymous_client, fake_session):
    response = anonymous_client.get(f"/api/v1/analyses/{uuid.uuid4()}")

    assert response.status_code == 401
    assert fake_session.statements == []


# --- what the account's role narrows the listing to -------------------------------------


def test_a_user_listing_is_filtered_to_that_users_analyses(fake_session):
    """The narrowing is a `WHERE` clause on the one statement the listing issues.

    Asserted against the compiled SQL rather than against the rows that come back, because
    the fake returns whatever it is given: what has to be true is that the *database* was
    asked for this user's analyses, not that the route filtered a wider answer afterwards.
    A filter applied after the read would leave the wider read in place for the next change
    to expose.
    """
    user = account(USER_ROLE_USER)
    app.dependency_overrides[require_user] = lambda: user

    with TestClient(app) as client:
        assert client.get("/api/v1/analyses").status_code == 200

    sql = compiled(fake_session)
    assert "analyses.owner_id = " in sql
    assert user.id.hex in sql


def test_an_admin_listing_is_not_filtered_by_owner(client, fake_session):
    """An administrator reads the whole system, so no ownership clause is added at all."""
    client.get("/api/v1/analyses")

    assert "owner_id" not in compiled(fake_session)


def test_a_user_detail_read_is_filtered_to_that_users_analyses(fake_session):
    """The same narrowing on the report's route, so the two readers cannot disagree.

    This is the pair that matters: a listing that hides another account's analysis while the
    detail route serves it to anyone holding the id would leak every record whose id ever
    appeared in a link.
    """
    user = account(USER_ROLE_USER)
    wanted = uuid.uuid4()
    app.dependency_overrides[require_user] = lambda: user

    with TestClient(app) as client:
        client.get(f"/api/v1/analyses/{wanted}")

    sql = compiled(fake_session)
    assert "analyses.owner_id = " in sql
    assert user.id.hex in sql
    assert wanted.hex in sql


def test_an_admin_detail_read_is_not_filtered_by_owner(client, fake_session):
    client.get(f"/api/v1/analyses/{uuid.uuid4()}")

    assert "owner_id" not in compiled(fake_session)


def test_empty_database_returns_an_empty_list(client, fake_session):
    response = client.get("/api/v1/analyses")

    assert response.status_code == 200
    assert response.json() == []


def test_persisted_analysis_is_returned_with_the_dashboard_fields(client, fake_session):
    row = listing_row()
    fake_session.rows = [row]

    response = client.get("/api/v1/analyses")

    assert response.status_code == 200
    assert response.json() == [
        {
            "id": str(row.id),
            "status": "completed",
            "created_at": "2026-08-19T18:08:01Z",
            "risk_level": "MEDIUM",
            "risk_rules_version": RULES_VERSION,
            "risk_rule_id": RULE_INDETERMINATE_BAND,
            "risk_calibration_id": CALIBRATION_ID,
            "original_filename": "clip.mp4",
            "declared_content_type": "video/mp4",
            "size_bytes": 13054,
            "original_sha256": "a" * 64,
            "was_normalized": False,
            "was_assembled": False,
            "media": {
                "format_name": "mov,mp4,m4a,3gp,3g2,mj2",
                "codec_name": "h264",
                "width": 1280,
                "height": 720,
                "duration": 5.0,
                "frame_rate": 30.0,
                "pix_fmt": "yuv420p",
                "constant_frame_rate": True,
            },
            "synthetic_video": {
                "provider": "nvidia",
                "signal_type": "synthetic_video",
                "status": "SUCCESS",
                "score": PROBABILITY,
                "provider_version": FUNCTION_ID,
                "logit": 1.9142135381698608,
                "total_clips": 7,
                "segments": [],
            },
            "provenance": {
                "provider": "c2pa",
                "signal_type": "provenance",
                "status": "SUCCESS",
                "provider_version": C2PA_SDK_VERSION,
                "manifest_exists": True,
                "validation_state": "Valid",
                "claim_generator": "test-camera",
                "signature_issuer": "Test Signing Cert",
                "remote_manifest_url": None,
            },
            "active_speaker": {
                "provider": "nvidia",
                "signal_type": "active_speaker",
                "status": "SUCCESS",
                "provider_version": ASD_FUNCTION_ID,
                "total_speaking_segments": 2,
                "segments_truncated": False,
                "segments": [],
            },
            "audio_authenticity": {
                "provider": "aasist",
                "signal_type": "audio_authenticity",
                "status": "SUCCESS",
                "provider_version": AASIST_CHECKPOINT,
                "total_audio_windows": 2,
                "persisted_audio_windows": 2,
                "windows_truncated": False,
                "windows": [],
            },
            # The face classifier's own figure, handed back exactly as the row holds it —
            # beside a MEDIUM decision, which is what a score above `T_HIGH` looks like when
            # nothing is reading it as risk.
            "face_manipulation": {
                "provider": "efficientnet-b7",
                "signal_type": "face_manipulation",
                "status": "SUCCESS",
                "score": FACE_SCORE,
                "provider_version": FACETORCH_CHECKPOINT,
                "frames_requested": 8,
                "frames_decoded": 8,
                "frames_scored": 6,
            },
            # The mouth-dynamics model's own figure, handed back exactly as the row holds it. No
            # rule reads it under any ruleset, so it sits beside the decision above having
            # taken no part in it — and it is never set beside the face score as though the
            # two were comparable.
            "lip_forensics": {
                "provider": "lipforensics",
                "signal_type": "lip_forensics",
                "status": "SUCCESS",
                "score": LIP_FORENSICS_SCORE,
                "provider_version": LIPFORENSICS_MODEL,
                "windows_requested": 4,
                "windows_read": 4,
                "windows_scored": 3,
            },
        }
    ]


def test_listing_exposes_no_field_beyond_the_dashboard_set(client, fake_session):
    fake_session.rows = [listing_row()]

    response = client.get("/api/v1/analyses")

    # Storage keys, ffprobe geometry and derivative identity are not part of this view.
    assert set(response.json()[0]) == EXPECTED_FIELDS


def test_declared_content_type_reports_the_declared_mime(client, fake_session):
    fake_session.rows = [listing_row(content_type="video/quicktime")]

    response = client.get("/api/v1/analyses")

    assert response.json()[0]["declared_content_type"] == "video/quicktime"


def test_a_missing_filename_is_returned_as_null(client, fake_session):
    fake_session.rows = [listing_row(original_filename=None)]

    response = client.get("/api/v1/analyses")

    assert response.json()[0]["original_filename"] is None


def test_normalized_analysis_reports_it(client, fake_session):
    fake_session.rows = [listing_row(was_normalized=True)]

    response = client.get("/api/v1/analyses")

    assert response.json()[0]["was_normalized"] is True


# The risk decision. What the engine concluded and committed, read back off the analysis
# row. The listing reports it; it never takes it, and never reconstructs one from the
# detector scores sitting beside it in the same response.


def test_a_high_risk_decision_is_returned_with_its_whole_trace(client, fake_session):
    fake_session.rows = [
        listing_row(
            risk_level="HIGH",
            risk_rule_id=RULE_CALIBRATED_HIGH,
            signal_score=0.9912,
        )
    ]

    analysis = client.get("/api/v1/analyses").json()[0]

    assert analysis["risk_level"] == "HIGH"
    assert analysis["risk_rules_version"] == RULES_VERSION
    assert analysis["risk_rule_id"] == RULE_CALIBRATED_HIGH
    assert analysis["risk_calibration_id"] == CALIBRATION_ID


def test_a_medium_risk_decision_is_returned_with_its_whole_trace(client, fake_session):
    fake_session.rows = [listing_row()]

    analysis = client.get("/api/v1/analyses").json()[0]

    assert analysis["risk_level"] == "MEDIUM"
    assert analysis["risk_rules_version"] == RULES_VERSION
    assert analysis["risk_rule_id"] == RULE_INDETERMINATE_BAND
    assert analysis["risk_calibration_id"] == CALIBRATION_ID


def test_an_unknown_decision_stays_explicitly_unknown(client, fake_session):
    """`UNKNOWN` is an answer, not the absence of one.

    The engine ran, a rule fired, and what it concluded is that the evidence does not
    support a classification. Rendering that as no decision would throw away the fact that
    the question was asked and answered.
    """
    fake_session.rows = [
        listing_row(risk_level="UNKNOWN", risk_rule_id=RULE_UNVALIDATED_PROVIDER)
    ]

    analysis = client.get("/api/v1/analyses").json()[0]

    assert analysis["risk_level"] == "UNKNOWN"
    assert analysis["risk_rule_id"] == RULE_UNVALIDATED_PROVIDER
    assert analysis["risk_rules_version"] == RULES_VERSION
    assert analysis["risk_calibration_id"] == CALIBRATION_ID


def test_an_analysis_with_no_decision_reports_null_rather_than_unknown(
    client, fake_session
):
    """A queued or in-flight analysis has no decision, and null is how that is said."""
    fake_session.rows = [
        listing_row(
            status="queued",
            risk_level=None,
            risk_rules_version=None,
            risk_rule_id=None,
            risk_calibration_id=None,
        )
    ]

    analysis = client.get("/api/v1/analyses").json()[0]

    assert analysis["risk_level"] is None
    assert analysis["risk_rules_version"] is None
    assert analysis["risk_rule_id"] is None
    assert analysis["risk_calibration_id"] is None


def test_a_pre_p7_analysis_stays_distinguishable_from_an_unknown_decision(
    client, fake_session
):
    """The distinction the whole risk column rests on, asserted on one response.

    An analysis completed before the engine existed carries null: nothing ever classified
    it. An analysis the engine classified as `UNKNOWN` carries `UNKNOWN`. Both are listed
    here at once, and the listing must not hand back the same thing for the two.
    """
    legacy = listing_row(
        risk_level=None,
        risk_rules_version=None,
        risk_rule_id=None,
        risk_calibration_id=None,
    )
    classified = listing_row(
        risk_level="UNKNOWN", risk_rule_id=RULE_UNVALIDATED_PROVIDER
    )
    fake_session.rows = [legacy, classified]

    analyses = client.get("/api/v1/analyses").json()

    assert analyses[0]["risk_level"] is None
    assert analyses[0]["risk_rules_version"] is None
    assert analyses[1]["risk_level"] == "UNKNOWN"
    assert analyses[1]["risk_rules_version"] == RULES_VERSION


def test_the_stored_decision_is_reported_even_when_the_score_beside_it_disagrees(
    client, fake_session
):
    """The N+1 guard's cousin: the listing reads the decision, it does not re-derive it.

    The score here is well above `T_HIGH`, so an endpoint that classified from the signal
    would answer `HIGH`. The stored decision says `MEDIUM` — taken under whichever ruleset
    was in force at the time — and that is what a forensic record means. Re-deriving it
    would let this response contradict the row it is reporting.
    """
    fake_session.rows = [listing_row(signal_score=0.999)]

    analysis = client.get("/api/v1/analyses").json()[0]

    assert analysis["risk_level"] == "MEDIUM"
    assert analysis["risk_rule_id"] == RULE_INDETERMINATE_BAND


def test_the_decision_is_read_in_the_same_statement_as_the_listing(client, fake_session):
    """The risk columns cost no extra query: they are on the analysis row already."""
    fake_session.rows = [listing_row() for _ in range(20)]

    client.get("/api/v1/analyses")

    sql = compiled(fake_session)
    for column in (
        "analyses.risk_level",
        "analyses.risk_rules_version",
        "analyses.risk_rule_id",
        "analyses.risk_calibration_id",
    ):
        assert column in sql

    assert len(fake_session.statements) == 4


# Media facts. What ffprobe read out of the original, as distinct from what the client
# declared about it. Established before any detector ran and stored since P1.


def test_the_probed_media_facts_are_exposed(client, fake_session):
    fake_session.rows = [listing_row()]

    media = client.get("/api/v1/analyses").json()[0]["media"]

    assert set(media) == EXPECTED_MEDIA_FIELDS
    assert media["format_name"] == "mov,mp4,m4a,3gp,3g2,mj2"
    assert media["codec_name"] == "h264"
    assert media["width"] == 1280
    assert media["height"] == 720
    assert media["duration"] == 5.0
    assert media["frame_rate"] == 30.0
    assert media["pix_fmt"] == "yuv420p"
    assert media["constant_frame_rate"] is True


def test_the_container_is_reported_as_ffprobe_worded_it(client, fake_session):
    """The demuxer family, not a container name narrowed out of it.

    One ffprobe format name covers MOV and MP4 alike. Only `major_brand` separates them and
    no column holds it, so calling this row an MP4 would be the listing claiming something
    the stored evidence does not establish.
    """
    fake_session.rows = [listing_row(format_name="matroska,webm", codec_name="vp9")]

    media = client.get("/api/v1/analyses").json()[0]["media"]

    assert media["format_name"] == "matroska,webm"
    assert media["codec_name"] == "vp9"


def test_media_facts_are_reported_untransformed(client, fake_session):
    """A fractional NTSC rate is not rounded on the way out; presentation owns that."""
    fake_session.rows = [listing_row(frame_rate=30000 / 1001, width=1920, height=1080)]

    media = client.get("/api/v1/analyses").json()[0]["media"]

    assert media["frame_rate"] == pytest.approx(30000 / 1001, rel=1e-12)
    assert (media["width"], media["height"]) == (1920, 1080)


def test_media_without_a_pixel_format_reports_null(client, fake_session):
    fake_session.rows = [listing_row(pix_fmt=None)]

    # ffprobe reported none. An empty string would read as a format called "".
    assert client.get("/api/v1/analyses").json()[0]["media"]["pix_fmt"] is None


def test_variable_frame_rate_media_reports_it(client, fake_session):
    fake_session.rows = [listing_row(constant_frame_rate=False)]

    assert client.get("/api/v1/analyses").json()[0]["media"]["constant_frame_rate"] is False


def test_the_media_facts_ride_the_listing_statement(client, fake_session):
    """The N+1 guard for the media columns: they are already on the joined row."""
    fake_session.rows = [listing_row() for _ in range(20)]

    client.get("/api/v1/analyses")

    assert len(fake_session.statements) == 4
    sql = compiled(fake_session)
    for column in ("format_name", "codec_name", "media_files.width", "media_files.height"):
        assert column in sql


def test_every_persisted_analysis_is_returned(client, fake_session):
    fake_session.rows = [
        listing_row(created_at=CREATED_AT - timedelta(minutes=offset)) for offset in range(3)
    ]

    response = client.get("/api/v1/analyses")

    assert len(response.json()) == 3


def test_the_query_joins_media_onto_the_analysis(client, fake_session):
    client.get("/api/v1/analyses")

    sql = compiled(fake_session)
    # An analysis and its media are written in one transaction, so an inner join cannot
    # hide a row.
    assert "JOIN media_files ON media_files.analysis_id = analyses.id" in sql
    assert "LEFT OUTER JOIN media_files" not in sql


def test_the_query_returns_the_most_recent_analyses_first(client, fake_session):
    client.get("/api/v1/analyses")

    sql = compiled(fake_session)
    # The id breaks ties, because two analyses committed together share `created_at`.
    assert "ORDER BY analyses.created_at DESC, analyses.id DESC" in sql


def test_the_query_applies_the_fixed_limit(client, fake_session):
    client.get("/api/v1/analyses")

    assert f"LIMIT {RECENT_ANALYSES_LIMIT}" in compiled(fake_session)


def test_the_query_selects_only_the_columns_the_listing_needs(client, fake_session):
    client.get("/api/v1/analyses")

    sql = compiled(fake_session)
    for absent in ("original_storage_key", "derivative_storage_key", "derivative_sha256"):
        assert absent not in sql


def test_the_signal_is_read_in_the_same_statement_as_the_listing(client, fake_session):
    fake_session.rows = [listing_row(), listing_row()]

    client.get("/api/v1/analyses")

    # The signal rides along on the listing itself rather than being fetched per analysis.
    assert "LEFT OUTER JOIN analysis_signals" in compiled(fake_session)


def test_the_page_costs_the_same_number_of_queries_however_many_analyses_it_holds(
    client, fake_session
):
    """The N+1 guard: four statements for one analysis, and four for twenty."""
    fake_session.rows = [listing_row()]
    client.get("/api/v1/analyses")
    assert len(fake_session.statements) == 4

    fake_session.statements.clear()
    fake_session.rows = [listing_row() for _ in range(20)]
    client.get("/api/v1/analyses")

    assert len(fake_session.statements) == 4


def test_no_segment_query_is_issued_when_no_analysis_carries_a_signal(client, fake_session):
    fake_session.rows = [unsignalled_row(), unsignalled_row()]

    client.get("/api/v1/analyses")

    # Nothing to look up for either evidence kind, so neither round trip is worth it.
    assert len(fake_session.statements) == 1


def test_the_signal_join_is_restricted_to_the_nvidia_synthetic_video_signal(
    client, fake_session
):
    client.get("/api/v1/analyses")

    sql = compiled(fake_session)
    # Narrowing the join keeps a second provider's future signal from multiplying rows.
    assert "analysis_signals.provider = 'nvidia'" in sql
    assert "analysis_signals.signal_type = 'synthetic_video'" in sql


def test_the_query_selects_no_signal_column_the_dashboard_must_not_show(client, fake_session):
    client.get("/api/v1/analyses")

    sql = compiled(fake_session)
    # `analysis_signals` carries a `risk_level` column of its own. It is a per-detector
    # figure and must never reach the dashboard: risk is one product-level decision over
    # all the evidence, and a verdict per signal beside it would be the very collapse rule
    # 11 refuses. Only the analysis row's decision is read — no signal join, aliased or
    # not, may select its own.
    assert "analyses.risk_level" in sql
    for signal_table in (
        "analysis_signals",
        "analysis_signals_1",
        "analysis_signals_2",
        "analysis_signals_3",
    ):
        assert f"{signal_table}.risk_level" not in sql


def test_a_successful_signal_is_exposed_with_the_provider_figures(client, fake_session):
    fake_session.rows = [listing_row()]

    signal = client.get("/api/v1/analyses").json()[0]["synthetic_video"]

    assert set(signal) == EXPECTED_SIGNAL_FIELDS
    assert signal["provider"] == "nvidia"
    assert signal["signal_type"] == "synthetic_video"
    assert signal["status"] == "SUCCESS"
    # Untransformed and unrounded: NVIDIA's number, on NVIDIA's scale.
    assert signal["score"] == PROBABILITY
    assert signal["provider_version"] == FUNCTION_ID
    assert signal["total_clips"] == 7


@pytest.mark.parametrize("status", ["FAILED", "TIMEOUT"])
def test_a_failed_signal_is_exposed_with_a_null_score(client, fake_session, status):
    fake_session.rows = [failed_signal_row(status)]

    signal = client.get("/api/v1/analyses").json()[0]["synthetic_video"]

    assert signal["status"] == status
    # Never 0.0: the detector produced no number, and zero would be a fabricated answer.
    assert signal["score"] is None
    assert signal["logit"] is None
    assert signal["total_clips"] is None


@pytest.mark.parametrize("status", ["FAILED", "TIMEOUT"])
def test_a_failed_signal_never_exposes_the_provider_error(client, fake_session, status):
    fake_session.rows = [failed_signal_row(status)]

    body = client.get("/api/v1/analyses").text

    # The stored metadata holds the exception class name; it is diagnostic, not evidence.
    assert "NvidiaProviderTimeout" not in body
    assert "error" not in body


def test_an_analysis_without_a_signal_is_listed_with_none(client, fake_session):
    fake_session.rows = [unsignalled_row()]

    row = client.get("/api/v1/analyses").json()[0]

    assert row["synthetic_video"] is None
    # The analysis itself still lists in full.
    assert row["original_filename"] == "clip.mp4"


def test_unusable_metadata_does_not_break_the_listing(client, fake_session):
    fake_session.rows = [
        listing_row(signal_metadata=None),
        listing_row(signal_metadata={"total_clips": "seven"}),
        listing_row(signal_metadata={"logit": True, "total_clips": True}),
    ]

    response = client.get("/api/v1/analyses")

    assert response.status_code == 200
    # A figure that is missing or of the wrong type is reported as absent, not guessed at.
    for row in response.json():
        assert row["synthetic_video"]["logit"] is None
        assert row["synthetic_video"]["total_clips"] is None
        assert row["synthetic_video"]["score"] == PROBABILITY


def test_a_whole_logit_survives_the_json_round_trip(client, fake_session):
    fake_session.rows = [listing_row(signal_metadata={"logit": 2, "total_clips": 7})]

    signal = client.get("/api/v1/analyses").json()[0]["synthetic_video"]

    assert signal["logit"] == 2.0


def test_database_failure_returns_a_controlled_503(client, fake_session):
    fake_session.execute_error = OperationalError("SELECT", None, Exception("connection lost"))

    response = client.get("/api/v1/analyses")

    assert response.status_code == 503
    # No statement, connection string or driver detail reaches the client.
    assert response.json() == {"detail": "analyses are temporarily unavailable"}


def test_segment_evidence_is_exposed_on_the_signal(client, fake_session):
    row = listing_row()
    fake_session.rows = [row]
    fake_session.segment_rows = [
        segment_row(row.signal_id, 45, 2.5),
        segment_row(row.signal_id, 12, 1.25),
    ]

    signal = client.get("/api/v1/analyses").json()[0]["synthetic_video"]

    assert signal["segments"] == [
        {"clip_index": 45, "logit": 2.5},
        {"clip_index": 12, "logit": 1.25},
    ]


def test_a_segment_exposes_no_field_beyond_what_the_provider_reported(client, fake_session):
    row = listing_row()
    fake_session.rows = [row]
    fake_session.segment_rows = [segment_row(row.signal_id, 45, 2.5)]

    segment = client.get("/api/v1/analyses").json()[0]["synthetic_video"]["segments"][0]

    # No start_time, no end_time, no score, no risk_level: NVIDIA reports a frame index
    # and a logit for a clip, and anything else here would have been invented.
    assert set(segment) == EXPECTED_SEGMENT_FIELDS


def test_segment_logits_are_exposed_untransformed(client, fake_session):
    row = listing_row()
    fake_session.rows = [row]
    # A raw model logit: negative, and on no bounded scale.
    fake_session.segment_rows = [segment_row(row.signal_id, 3, -1.4362945556640625)]

    segment = client.get("/api/v1/analyses").json()[0]["synthetic_video"]["segments"][0]

    assert segment["logit"] == -1.4362945556640625


def test_segments_are_capped_at_the_display_count(client, fake_session):
    row = listing_row()
    fake_session.rows = [row]
    fake_session.segment_rows = [
        segment_row(row.signal_id, index, 3.0 - index) for index in range(DASHBOARD_SEGMENTS + 4)
    ]

    signal = client.get("/api/v1/analyses").json()[0]["synthetic_video"]

    assert len(signal["segments"]) == DASHBOARD_SEGMENTS
    # The strongest survive the cap; the weakest are the ones dropped.
    assert [segment["clip_index"] for segment in signal["segments"]] == list(
        range(DASHBOARD_SEGMENTS)
    )


def test_each_analysis_receives_only_its_own_segments(client, fake_session):
    first, second = listing_row(), listing_row()
    fake_session.rows = [first, second]
    fake_session.segment_rows = [
        segment_row(first.signal_id, 1, 2.0),
        segment_row(second.signal_id, 9, 1.0),
        segment_row(second.signal_id, 8, 0.5),
    ]

    body = client.get("/api/v1/analyses").json()

    assert body[0]["synthetic_video"]["segments"] == [{"clip_index": 1, "logit": 2.0}]
    assert [s["clip_index"] for s in body[1]["synthetic_video"]["segments"]] == [9, 8]


def test_a_signal_with_no_stored_segments_reports_an_empty_list(client, fake_session):
    fake_session.rows = [listing_row()]
    fake_session.segment_rows = []

    signal = client.get("/api/v1/analyses").json()[0]["synthetic_video"]

    # Empty, not null: the signal exists and simply carries no clip evidence.
    assert signal["segments"] == []


def test_a_failed_signal_carries_no_segments(client, fake_session):
    fake_session.rows = [failed_signal_row("FAILED")]

    signal = client.get("/api/v1/analyses").json()[0]["synthetic_video"]

    assert signal["segments"] == []


def test_the_segment_query_asks_only_for_the_listed_signals(client, fake_session):
    row = listing_row()
    fake_session.rows = [row, unsignalled_row()]

    client.get("/api/v1/analyses")

    sql = compiled(fake_session, index=1)
    assert "analysis_segments.signal_id IN" in sql
    # Rendered without its dashes by the literal bind.
    assert row.signal_id.hex in sql
    # The analysis carrying no signal contributes no id to look up.
    assert sql.count("'") == 2


def test_the_segment_query_orders_the_strongest_clip_first(client, fake_session):
    fake_session.rows = [listing_row()]

    client.get("/api/v1/analyses")

    sql = compiled(fake_session, index=1)
    # The clip index breaks ties, so the same evidence always reads back in one order.
    assert (
        "ORDER BY analysis_segments.signal_id, analysis_segments.logit DESC, "
        "analysis_segments.clip_index" in sql
    )


def test_a_segment_query_failure_returns_the_same_controlled_503(client, fake_session):
    fake_session.rows = [listing_row()]

    def fail_on_the_segment_query(statement):
        fake_session.statements.append(statement)
        if len(fake_session.statements) == 1:
            return SimpleNamespace(all=lambda: fake_session.rows)
        raise OperationalError("SELECT", None, Exception("connection lost"))

    fake_session.execute = fail_on_the_segment_query

    response = client.get("/api/v1/analyses")

    assert response.status_code == 503
    assert response.json() == {"detail": "analyses are temporarily unavailable"}


# Provenance. The other evidence source on the same listing: no score, and three states a
# reader has to be able to tell apart — read and signed, read and unsigned, not read.


def test_the_provenance_join_is_restricted_to_the_c2pa_provenance_signal(client, fake_session):
    client.get("/api/v1/analyses")

    sql = compiled(fake_session)
    assert "provider = 'c2pa'" in sql
    assert "signal_type = 'provenance'" in sql
    # Each further signal joins the table again under its own alias, so no two of them
    # can multiply each other.
    assert sql.count("LEFT OUTER JOIN analysis_signals") == 6


def test_both_signals_ride_the_same_statement(client, fake_session):
    """The N+1 guard again, now that further signals join the same rows."""
    fake_session.rows = [listing_row() for _ in range(20)]

    client.get("/api/v1/analyses")

    assert len(fake_session.statements) == 4


def test_provenance_is_exposed_with_the_facts_the_file_carries(client, fake_session):
    fake_session.rows = [listing_row()]

    provenance = client.get("/api/v1/analyses").json()[0]["provenance"]

    assert set(provenance) == EXPECTED_PROVENANCE_FIELDS
    assert provenance["provider"] == "c2pa"
    assert provenance["signal_type"] == "provenance"
    assert provenance["status"] == "SUCCESS"
    assert provenance["manifest_exists"] is True
    # The SDK's own word for what it made of the signature, untranslated.
    assert provenance["validation_state"] == "Valid"
    assert provenance["claim_generator"] == "test-camera"
    assert provenance["signature_issuer"] == "Test Signing Cert"
    assert provenance["provider_version"] == C2PA_SDK_VERSION


def test_provenance_carries_no_score(client, fake_session):
    fake_session.rows = [listing_row()]

    # A signature is not a figure on a scale. A number here would end up beside NVIDIA's
    # probability as though the two could be compared.
    assert "score" not in client.get("/api/v1/analyses").json()[0]["provenance"]


def test_media_carrying_no_credentials_is_a_successful_reading(client, fake_session):
    fake_session.rows = [
        provenance_row({"manifest_exists": False, "validation_state": None})
    ]

    provenance = client.get("/api/v1/analyses").json()[0]["provenance"]

    # The reading worked and found nothing, which is what most media looks like. It is
    # distinguishable from a failure by its status and from a signed file by the flag.
    assert provenance["status"] == "SUCCESS"
    assert provenance["manifest_exists"] is False
    assert provenance["validation_state"] is None


@pytest.mark.parametrize("state", ["Trusted", "Valid", "Invalid"])
def test_the_validation_state_is_passed_through_as_the_sdk_worded_it(
    client, fake_session, state
):
    fake_session.rows = [provenance_row({"manifest_exists": True, "validation_state": state})]

    provenance = client.get("/api/v1/analyses").json()[0]["provenance"]

    assert provenance["validation_state"] == state


def test_a_failed_reading_is_not_reported_as_absent_credentials(client, fake_session):
    fake_session.rows = [provenance_row({"error": "C2paUnsupportedFormat"}, status="FAILED")]

    provenance = client.get("/api/v1/analyses").json()[0]["provenance"]

    assert provenance["status"] == "FAILED"
    # Null, not false: whether the file carries credentials is exactly what is unknown.
    assert provenance["manifest_exists"] is None
    assert provenance["validation_state"] is None


def test_a_failed_reading_never_exposes_the_extraction_error(client, fake_session):
    fake_session.rows = [provenance_row({"error": "C2paUnsupportedFormat"}, status="FAILED")]

    body = client.get("/api/v1/analyses").text

    assert "C2paUnsupportedFormat" not in body
    assert "error" not in body


def test_an_analysis_read_before_provenance_existed_is_listed_with_none(client, fake_session):
    fake_session.rows = [unsignalled_row()]

    row = client.get("/api/v1/analyses").json()[0]

    # No signal row at all — a different fact from a reading that found no credentials.
    assert row["provenance"] is None


def test_unusable_provenance_metadata_does_not_break_the_listing(client, fake_session):
    fake_session.rows = [
        provenance_row(None),
        provenance_row({"manifest_exists": "yes", "validation_state": 7}),
        provenance_row({}),
    ]

    response = client.get("/api/v1/analyses")

    assert response.status_code == 200
    # A truthy string must not become `true`: that flag is what separates "no credentials"
    # from "could not tell", and coercing it would erase the difference.
    for row in response.json():
        assert row["provenance"]["manifest_exists"] is None
        assert row["provenance"]["validation_state"] is None


def test_a_remote_manifest_url_is_exposed_beside_the_absent_manifest(client, fake_session):
    """Claimed provenance kept outside the file is not the same as none at all.

    The file named a manifest; it simply is not in these bytes, and the reader deliberately
    did not go and get it. Reporting only `manifest_exists: false` would leave the two
    indistinguishable and credit the file with a claim it never made.
    """
    url = "https://provenance.example/manifest.c2pa"
    fake_session.rows = [
        provenance_row({"manifest_exists": False, "remote_manifest_url": url})
    ]

    provenance = client.get("/api/v1/analyses").json()[0]["provenance"]

    assert provenance["status"] == "SUCCESS"
    assert provenance["manifest_exists"] is False
    assert provenance["remote_manifest_url"] == url
    # Nothing was fetched, so there is no signature to have a state.
    assert provenance["validation_state"] is None


def test_media_with_no_credentials_at_all_carries_no_remote_url(client, fake_session):
    fake_session.rows = [
        provenance_row({"manifest_exists": False, "remote_manifest_url": None})
    ]

    provenance = client.get("/api/v1/analyses").json()[0]["provenance"]

    # The pair that separates the two: no manifest, and no URL claiming one elsewhere.
    assert provenance["manifest_exists"] is False
    assert provenance["remote_manifest_url"] is None


def test_an_embedded_manifest_carries_no_remote_url(client, fake_session):
    fake_session.rows = [listing_row()]

    assert client.get("/api/v1/analyses").json()[0]["provenance"]["remote_manifest_url"] is None


def test_an_unusable_remote_manifest_url_reads_as_absent(client, fake_session):
    fake_session.rows = [
        provenance_row({"manifest_exists": False, "remote_manifest_url": 7}),
        provenance_row({"manifest_exists": False}),
    ]

    response = client.get("/api/v1/analyses")

    assert response.status_code == 200
    # A figure that is not a URL is not turned into one; the row still lists.
    for row in response.json():
        assert row["provenance"]["remote_manifest_url"] is None


# Active speaker. The third evidence source on the same listing: no score, a timeline
# instead, and four states a reader has to be able to tell apart — no signal at all, a
# chain that failed, a detector that saw nobody speaking, and a real timeline.


def test_the_active_speaker_join_is_restricted_to_the_nvidia_active_speaker_signal(
    client, fake_session
):
    client.get("/api/v1/analyses")

    sql = compiled(fake_session)
    assert "signal_type = 'active_speaker'" in sql
    # Same provider as the synthetic-video signal, so only the signal type separates the
    # two joins onto it.
    assert sql.count("provider = 'nvidia'") == 2


def test_active_speaker_is_exposed_with_the_signal_facts(client, fake_session):
    fake_session.rows = [listing_row()]

    signal = client.get("/api/v1/analyses").json()[0]["active_speaker"]

    assert set(signal) == EXPECTED_ACTIVE_SPEAKER_FIELDS
    assert signal["provider"] == "nvidia"
    assert signal["signal_type"] == "active_speaker"
    assert signal["status"] == "SUCCESS"
    assert signal["provider_version"] == ASD_FUNCTION_ID
    assert signal["total_speaking_segments"] == 2
    assert signal["segments_truncated"] is False


def test_active_speaker_carries_no_score(client, fake_session):
    fake_session.rows = [listing_row()]

    # A timeline is not a figure on a scale. A number here would end up beside NVIDIA's
    # synthetic probability as though the two could be compared.
    assert "score" not in client.get("/api/v1/analyses").json()[0]["active_speaker"]


def test_an_analysis_without_an_active_speaker_signal_is_listed_with_none(
    client, fake_session
):
    fake_session.rows = [unsignalled_row()]

    row = client.get("/api/v1/analyses").json()[0]

    # No signal row at all — a different fact from a detector that ran and saw nobody.
    assert row["active_speaker"] is None


def test_the_speaking_timeline_is_exposed_on_the_signal(client, fake_session):
    row = listing_row()
    fake_session.rows = [row]
    fake_session.speaking_rows = [
        speaking_row(row.active_speaker_id, 0.2, 4.4, 0, "SPEAKER_00"),
        speaking_row(row.active_speaker_id, 5.0, 6.5, 1, "SPEAKER_01"),
    ]

    signal = client.get("/api/v1/analyses").json()[0]["active_speaker"]

    assert signal["segments"] == [
        {"start_time": 0.2, "end_time": 4.4, "face_id": 0, "speaker_label": "SPEAKER_00"},
        {"start_time": 5.0, "end_time": 6.5, "face_id": 1, "speaker_label": "SPEAKER_01"},
    ]


def test_a_speaking_segment_exposes_no_field_beyond_what_the_provider_reported(
    client, fake_session
):
    row = listing_row()
    fake_session.rows = [row]
    fake_session.speaking_rows = [speaking_row(row.active_speaker_id, 0.0, 1.0, 0, "SPEAKER_00")]

    segment = client.get("/api/v1/analyses").json()[0]["active_speaker"]["segments"][0]

    # No clip index, no logit, no score: an active-speaker result has none of them, and
    # anything else here would have been invented.
    assert set(segment) == EXPECTED_SPEAKING_FIELDS


def test_a_face_matched_to_no_diarized_voice_keeps_its_segment(client, fake_session):
    row = listing_row()
    fake_session.rows = [row]
    fake_session.speaking_rows = [speaking_row(row.active_speaker_id, 1.0, 2.0, 3, None)]

    segment = client.get("/api/v1/analyses").json()[0]["active_speaker"]["segments"][0]

    # "This face was speaking and no diarized voice matched it" is an observation, not
    # missing data, so the segment lists with a null label rather than being dropped.
    assert segment["face_id"] == 3
    assert segment["speaker_label"] is None


def test_a_successful_detection_with_no_speaking_face_reports_an_empty_timeline(
    client, fake_session
):
    fake_session.rows = [
        listing_row(
            active_speaker_metadata={
                "total_speaking_segments": 0,
                "segments_truncated": False,
            }
        )
    ]
    fake_session.speaking_rows = []

    signal = client.get("/api/v1/analyses").json()[0]["active_speaker"]

    # The detector ran and saw nobody speaking. That is a real result, told apart from a
    # failure by the status beside it, and it says nothing about the media.
    assert signal["status"] == "SUCCESS"
    assert signal["segments"] == []
    assert signal["total_speaking_segments"] == 0


@pytest.mark.parametrize("status", ["FAILED", "TIMEOUT"])
def test_a_failed_active_speaker_signal_carries_no_timeline(client, fake_session, status):
    fake_session.rows = [failed_active_speaker_row(status)]

    signal = client.get("/api/v1/analyses").json()[0]["active_speaker"]

    assert signal["status"] == status
    assert signal["segments"] == []
    # No figure is invented for a chain that did not get to produce one.
    assert signal["total_speaking_segments"] is None
    assert signal["segments_truncated"] is None


@pytest.mark.parametrize("status", ["FAILED", "TIMEOUT"])
def test_a_failed_active_speaker_signal_never_exposes_the_provider_error(
    client, fake_session, status
):
    fake_session.rows = [failed_active_speaker_row(status)]

    body = client.get("/api/v1/analyses").text

    # The stored metadata holds the exception class name; it is diagnostic, not evidence.
    assert "NvidiaActiveSpeakerTimeout" not in body
    assert "error" not in body


def test_a_truncated_timeline_says_so(client, fake_session):
    row = listing_row(
        active_speaker_metadata={"total_speaking_segments": 84, "segments_truncated": True}
    )
    fake_session.rows = [row]
    fake_session.speaking_rows = [speaking_row(row.active_speaker_id, 0.0, 1.0, 0, "SPEAKER_00")]

    signal = client.get("/api/v1/analyses").json()[0]["active_speaker"]

    # What was persisted, against how many runs there were: without the pair, a partial
    # timeline would read as the whole one.
    assert signal["total_speaking_segments"] == 84
    assert signal["segments_truncated"] is True


def test_unusable_active_speaker_metadata_does_not_break_the_listing(client, fake_session):
    fake_session.rows = [
        listing_row(active_speaker_metadata=None),
        listing_row(active_speaker_metadata={"total_speaking_segments": "two"}),
        listing_row(active_speaker_metadata={"segments_truncated": "yes"}),
    ]

    response = client.get("/api/v1/analyses")

    assert response.status_code == 200
    # A figure that is missing or of the wrong type is reported as absent, not guessed at —
    # and a truthy string must not become a `true` that hides a truncated timeline.
    for row in response.json():
        assert row["active_speaker"]["total_speaking_segments"] is None
        assert row["active_speaker"]["segments_truncated"] is None


def test_each_analysis_receives_only_its_own_speaking_segments(client, fake_session):
    first, second = listing_row(), listing_row()
    fake_session.rows = [first, second]
    fake_session.speaking_rows = [
        speaking_row(first.active_speaker_id, 0.0, 1.0, 0, "SPEAKER_00"),
        speaking_row(second.active_speaker_id, 2.0, 3.0, 1, "SPEAKER_01"),
    ]

    body = client.get("/api/v1/analyses").json()

    assert [s["start_time"] for s in body[0]["active_speaker"]["segments"]] == [0.0]
    assert [s["start_time"] for s in body[1]["active_speaker"]["segments"]] == [2.0]


def test_the_timeline_query_asks_only_for_the_listed_active_speaker_signals(
    client, fake_session
):
    row = listing_row()
    fake_session.rows = [row, unsignalled_row()]

    client.get("/api/v1/analyses")

    sql = compiled(fake_session, index=2)
    assert "analysis_segments.signal_id IN" in sql
    # Rendered without its dashes by the literal bind.
    assert row.active_speaker_id.hex in sql
    # The analysis carrying no signal contributes no id to look up.
    assert sql.count("'") == 2


def test_the_timeline_query_reads_the_segments_chronologically(client, fake_session):
    fake_session.rows = [listing_row()]

    client.get("/api/v1/analyses")

    sql = compiled(fake_session, index=2)
    # Two faces can begin speaking on the same frame, so the end time and the face break
    # the tie and the timeline reads back the same way every call.
    assert (
        "ORDER BY analysis_segments.signal_id, analysis_segments.start_time, "
        "analysis_segments.end_time, analysis_segments.face_id" in sql
    )


def test_no_timeline_query_is_issued_when_no_analysis_carries_an_active_speaker_signal(
    client, fake_session
):
    fake_session.rows = [listing_row(active_speaker_id=None, active_speaker_status=None)]

    client.get("/api/v1/analyses")

    # The listing, the clip evidence and the audio windows — and nothing to look a timeline
    # up for.
    assert len(fake_session.statements) == 3


def test_a_timeline_query_failure_returns_the_same_controlled_503(client, fake_session):
    fake_session.rows = [listing_row()]

    def fail_on_the_timeline_query(statement):
        fake_session.statements.append(statement)
        if "start_time" in str(statement):
            raise OperationalError("SELECT", None, Exception("connection lost"))
        return SimpleNamespace(all=lambda: fake_session.rows if len(fake_session.statements) == 1 else [])

    fake_session.execute = fail_on_the_timeline_query

    response = client.get("/api/v1/analyses")

    assert response.status_code == 503
    assert response.json() == {"detail": "analyses are temporarily unavailable"}


# The local audio evidence. A fourth signal on the listing, with the same four states the
# others keep apart and one more thing to be careful about: the window bounds it carries
# are DeepGuard's own preprocessing boundaries, not something the model reported.


def test_a_successful_audio_reading_is_exposed_with_its_counts(client, fake_session):
    fake_session.rows = [listing_row()]

    signal = client.get("/api/v1/analyses").json()[0]["audio_authenticity"]

    assert set(signal) == EXPECTED_AUDIO_FIELDS
    assert signal["provider"] == "aasist"
    assert signal["signal_type"] == "audio_authenticity"
    assert signal["status"] == "SUCCESS"
    # Which checkpoint produced the figures. A different revision is a different reading.
    assert signal["provider_version"] == AASIST_CHECKPOINT
    assert signal["total_audio_windows"] == 2
    assert signal["persisted_audio_windows"] == 2
    assert signal["windows_truncated"] is False


def test_the_audio_signal_carries_no_file_level_score(client, fake_session):
    """The one field it must never grow: the model publishes no calibration."""
    fake_session.rows = [listing_row()]

    signal = client.get("/api/v1/analyses").json()[0]["audio_authenticity"]

    assert "score" not in signal
    assert "risk_level" not in signal
    # Nor the stored metadata document, which holds the exception class name on a failure.
    assert "metadata" not in signal


def test_audio_windows_reach_the_listing_with_both_raw_logits(client, fake_session):
    row = listing_row()
    fake_session.rows = [row]
    fake_session.audio_rows = [
        audio_window_row(row.audio_id, 0, (0.0, 4.0375), (-2.89, 1.65)),
        audio_window_row(row.audio_id, 1, (4.0375, 8.075), (2.44, -2.73)),
    ]

    windows = client.get("/api/v1/analyses").json()[0]["audio_authenticity"]["windows"]

    assert [set(window) for window in windows] == [EXPECTED_AUDIO_WINDOW_FIELDS] * 2
    # Both outputs in graph order, untransformed. Neither can be derived from the other.
    assert [(window["logit"], window["bona_fide_logit"]) for window in windows] == [
        (-2.89, 1.65),
        (2.44, -2.73),
    ]
    # The bounds of the windows DeepGuard cut, in the order it cut them.
    assert [(window["start_time"], window["end_time"]) for window in windows] == [
        (0.0, 4.0375),
        (4.0375, 8.075),
    ]
    assert [window["clip_index"] for window in windows] == [0, 1]


def test_a_truncated_audio_sweep_reaches_the_listing_as_truncated(client, fake_session):
    row = listing_row(
        audio_metadata={
            "total_audio_windows": 149,
            "persisted_audio_windows": 50,
            "windows_truncated": True,
        }
    )
    fake_session.rows = [row]
    fake_session.audio_rows = [
        audio_window_row(row.audio_id, index, (float(index), index + 1.0), (0.1, 0.2))
        for index in range(50)
    ]

    signal = client.get("/api/v1/analyses").json()[0]["audio_authenticity"]

    # Without the pair, a chronological prefix would read as the whole recording.
    assert len(signal["windows"]) == 50
    assert signal["total_audio_windows"] == 149
    assert signal["windows_truncated"] is True


def test_the_whole_stored_sweep_is_handed_on_rather_than_trimmed_again(client, fake_session):
    """Unlike the clip evidence, which the listing trims to the display count.

    Trimming here would move the boundary `persisted_audio_windows` describes, and leave
    the response contradicting its own count.
    """
    row = listing_row()
    fake_session.rows = [row]
    fake_session.audio_rows = [
        audio_window_row(row.audio_id, index, (float(index), index + 1.0), (0.1, 0.2))
        for index in range(DASHBOARD_SEGMENTS + 4)
    ]

    windows = client.get("/api/v1/analyses").json()[0]["audio_authenticity"]["windows"]

    assert len(windows) == DASHBOARD_SEGMENTS + 4


def test_an_analysis_without_an_audio_signal_reports_null(client, fake_session):
    fake_session.rows = [unsignalled_row()]

    # Not the same fact as a reading that ran: this analysis was stored before the
    # checkpoint was wired in, and nothing ever looked at its audio.
    assert client.get("/api/v1/analyses").json()[0]["audio_authenticity"] is None


@pytest.mark.parametrize("status", ["FAILED", "TIMEOUT"])
def test_an_audio_reading_that_did_not_happen_carries_its_status_and_no_windows(
    client, fake_session, status
):
    fake_session.rows = [failed_audio_row(status)]

    signal = client.get("/api/v1/analyses").json()[0]["audio_authenticity"]

    assert signal["status"] == status
    assert signal["windows"] == []
    # No counts either: nothing was swept, so a zero here would be a fabricated figure.
    assert signal["total_audio_windows"] is None
    assert signal["persisted_audio_windows"] is None
    assert signal["windows_truncated"] is None


def test_a_failed_audio_reading_never_leaks_its_diagnostic_detail(client, fake_session):
    fake_session.rows = [failed_audio_row("FAILED")]

    response = client.get("/api/v1/analyses")

    # The stored metadata holds the exception class name, which is internal detail.
    assert "AudioDetectorModelUnavailable" not in response.text


def test_a_successful_audio_reading_with_no_windows_is_not_a_failure(client, fake_session):
    """The state the dashboard has to tell apart from a reading that did not happen."""
    fake_session.rows = [listing_row()]
    fake_session.audio_rows = []

    signal = client.get("/api/v1/analyses").json()[0]["audio_authenticity"]

    assert signal["status"] == "SUCCESS"
    assert signal["windows"] == []


def test_the_audio_signal_is_read_in_the_same_statement_as_the_listing(client, fake_session):
    fake_session.rows = [listing_row(), listing_row()]

    client.get("/api/v1/analyses")

    sql = compiled(fake_session)
    # Narrowed like every other join, so a fourth signal cannot multiply the listing's rows.
    assert "provider = 'aasist'" in sql
    assert "signal_type = 'audio_authenticity'" in sql


def test_the_audio_window_query_names_only_the_signals_the_listing_found(
    client, fake_session
):
    row = listing_row()
    fake_session.rows = [row, unsignalled_row()]

    client.get("/api/v1/analyses")

    sql = compiled(fake_session, index=3)
    assert row.audio_id.hex in sql
    # The analysis carrying no signal contributes no id to look up.
    assert sql.count("'") == 2


def test_the_audio_window_query_reads_the_windows_in_the_order_they_were_cut(
    client, fake_session
):
    fake_session.rows = [listing_row()]

    client.get("/api/v1/analyses")

    sql = compiled(fake_session, index=3)
    # The window index is the only order this evidence has, and it is total within a
    # signal — so the same stored sweep reads back the same way on every call. Ordering by
    # logit would impose a ranking the checkpoint gives no basis for.
    assert (
        "ORDER BY analysis_segments.signal_id, analysis_segments.clip_index" in sql
    )
    assert "logit DESC" not in sql


def test_no_audio_query_is_issued_when_no_analysis_carries_an_audio_signal(
    client, fake_session
):
    fake_session.rows = [listing_row(audio_id=None, audio_status=None)]

    client.get("/api/v1/analyses")

    # The listing, the clip evidence and the timeline — nothing to look windows up for.
    assert len(fake_session.statements) == 3


def test_an_audio_query_failure_returns_the_same_controlled_503(client, fake_session):
    fake_session.rows = [listing_row()]

    def fail_on_the_audio_query(statement):
        fake_session.statements.append(statement)
        if "bona_fide_logit" in str(statement):
            raise OperationalError("SELECT", None, Exception("connection lost"))
        return SimpleNamespace(
            all=lambda: fake_session.rows if len(fake_session.statements) == 1 else []
        )

    fake_session.execute = fail_on_the_audio_query

    response = client.get("/api/v1/analyses")

    assert response.status_code == 503
    assert response.json() == {"detail": "analyses are temporarily unavailable"}
