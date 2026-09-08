"""Where an analysed artifact came from, and the exact sentences a report may say about it.

R7-T12 records two things beside R7-T1's `was_assembled`: which door the media came through,
and — for a URL submission — the host it was fetched from. Everything here is about the two
ways that record can go wrong.

The first is *leakage*. A submitted media URL routinely carries a signed expiry, an access
token, a session id or a private path in its query string, and the only defence that cannot be
forgotten later is never to extract them. So the normalization is tested as an absence over
the whole answer rather than as an equality on the part that was meant to survive.

The second is *over-claiming*, and it is the harder one. DeepGuard fetched bytes from a host.
It did not establish that the host is the publisher, that the bytes are the file the publisher
issued, or that anything upstream of the fetch is as it appears — so a report calling an
acquired file "the original" would assert something no evidence in this system supports. The
wording tests pin the sentences the report is allowed to say, and check that none of them
contains the words that would turn an acquisition into a provenance claim.

Those sentences are checked by *running* them. `acquisitionStatement` and
`credentialsAbsentStatement` are pure functions over three fields, so the block of
`app/analysis.ts` that holds them is transpiled and executed here and the assertions are
against what a reader would actually see. Asserting on the source text would only prove the
sentences exist somewhere in the file; this proves which one each acquisition produces, which
is the property that matters — a correct sentence on the wrong branch is the whole failure
mode this task exists to prevent.

The route-level halves of this feature are tested where their fixtures already live: what each
door persists, and what a secret-bearing URL leaves behind, are in `test_url_ingestion.py` and
`test_upload.py`.

Nothing here asserts anything about risk, except the section that proves nothing may. These
are acquisition facts; no rule reads them.
"""

import json
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from shutil import which
from types import SimpleNamespace

import pytest

from app import downloader, risk_engine
from app.api.analyses import analysis_payloads
from app.db.models import ACQUISITION_METHOD_UPLOAD, ACQUISITION_METHOD_URL

# --------------------------------------------------------------------------------------
# Host normalization
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://videos.example.com/clip", "videos.example.com"),
        # Case is not identity. `EXAMPLE.COM` and `example.com` are one host, and two
        # spellings of it in the column would read as two origins in a report.
        ("https://VIDEOS.EXAMPLE.COM/clip", "videos.example.com"),
        # The legacy subdomain, stripped once and only as a whole label.
        ("https://www.youtube.com/watch?v=abc", "youtube.com"),
        ("https://www.www.example.com/clip", "www.example.com"),
        ("https://wwwexample.com/clip", "wwwexample.com"),
        # The root dot names the same host and would otherwise store a second spelling.
        ("https://example.com./clip", "example.com"),
        # A port is not part of the origin this records.
        ("https://videos.example.com:8443/clip", "videos.example.com"),
        # Credentials in the authority are exactly what must never be persisted.
        ("https://user:secret@videos.example.com/clip", "videos.example.com"),
        # An IPv6 literal is a host like any other; the brackets are syntax, not name.
        ("https://[2001:db8::1]/clip", "2001:db8::1"),
    ],
)
def test_a_host_is_recorded_in_one_normalized_form(url, expected):
    assert downloader.normalized_host(url) == expected


def test_the_host_is_parsed_rather_than_scanned_for():
    """A URL that looks like one host and is another resolves to the host that is fetched.

    `https://youtube.com@evil.example/x` has the authority `evil.example`; everything before
    the `@` is userinfo. A normalization that split on characters would record `youtube.com`
    and put a name the request never contacted into the report — which is a worse failure
    than recording nothing, because it would be believed.
    """
    assert downloader.normalized_host("https://youtube.com@evil.example/x") == "evil.example"


@pytest.mark.parametrize(
    "url",
    [
        "not a url at all",
        "https:///clip",
        # `urlsplit` refuses to answer for both of these rather than guessing.
        "https://[::1/clip",
    ],
)
def test_a_string_with_no_readable_host_records_nothing(url):
    """`None`, never a fragment of the string and never a placeholder.

    A host this function cannot read is a host it does not know, and the record says so. It
    is not a validation verdict: whether a URL may be fetched at all is `validate_url`'s
    question, and none of these would survive it either.
    """
    assert downloader.normalized_host(url) is None


def test_an_unparseable_port_does_not_cost_the_host():
    """The port is not part of what is recorded, so a broken one changes nothing.

    `urlsplit` reads the host without ever parsing the port, and this function never asks for
    one. Written down because the opposite implementation — reading the port and letting its
    `ValueError` escape — would throw away a host that is perfectly readable.
    """
    assert downloader.normalized_host("https://example.com:notaport/clip") == "example.com"


def test_normalization_keeps_no_part_of_the_url_but_the_host():
    """Every other component is gone, including all the ones that carry secrets.

    Asserted as an absence over the whole answer rather than as an equality, so a later
    normalization that started appending a path or a port fails here rather than quietly
    widening what the database holds.
    """
    host = downloader.normalized_host(
        "https://user:s3cr3t@www.example.com:8443/private/clip.mp4"
        "?token=abcdef123456&Expires=99999999#t=42"
    )

    assert host == "example.com"

    for fragment in (
        "s3cr3t", "user", "8443", "private", "clip.mp4", "token", "abcdef123456",
        "Expires", "99999999", "42", "/", "?", "#", ":", "@",
    ):
        assert fragment not in host


# --------------------------------------------------------------------------------------
# Historical rows
# --------------------------------------------------------------------------------------
#
# An analysis stored before R7-T12 recorded nothing about its acquisition. Both columns are
# null on it, and null is a third state rather than a synonym for `upload`: nobody
# established which door that analysis came through, and a read path that resolved it to one
# would be manufacturing a provenance fact out of a gap in the record.

WEB_ROOT = Path(__file__).resolve().parents[2] / "web"
WEB_ANALYSIS = WEB_ROOT / "app" / "analysis.ts"

requires_web = pytest.mark.skipif(
    not WEB_ROOT.exists(), reason="the web application is not present"
)


def summary_row(**overrides):
    """A row shaped like the listing's select emits, carrying only what this file reads.

    Deliberately minimal. The listing's own full-row fixture lives in
    `test_analysis_listing.py`; what is under test here is two columns and the way the read
    path passes them through, so everything else is the least that makes the projection run.
    """
    values = {
        "id": uuid.uuid4(),
        "status": "completed",
        "created_at": datetime(2026, 9, 8, 12, 0, tzinfo=timezone.utc),
        "risk_level": "UNKNOWN",
        "risk_rules_version": "p7-v1.0.0",
        "risk_rule_id": "R200",
        "risk_calibration_id": "c" * 64,
        "original_filename": "clip.mp4",
        "content_type": "video/mp4",
        "size_bytes": 4096,
        "original_sha256": "a" * 64,
        "was_normalized": False,
        "was_assembled": False,
        "acquisition_method": None,
        "source_host": None,
        "format_name": "mov,mp4,m4a,3gp,3g2,mj2",
        "codec_name": "h264",
        "width": 1920,
        "height": 1080,
        "duration": 12.34,
        "frame_rate": 30.0,
        "pix_fmt": "yuv420p",
        "constant_frame_rate": True,
        "signal_id": None,
        "signal_provider": None,
        "signal_type": None,
        "signal_status": None,
        "signal_score": None,
        "signal_provider_version": None,
        "signal_metadata": None,
        "provenance_provider": None,
        "provenance_signal_type": None,
        "provenance_status": None,
        "provenance_provider_version": None,
        "provenance_metadata": None,
        "active_speaker_id": None,
        "active_speaker_provider": None,
        "active_speaker_signal_type": None,
        "active_speaker_status": None,
        "active_speaker_provider_version": None,
        "active_speaker_metadata": None,
        "audio_id": None,
        "audio_provider": None,
        "audio_signal_type": None,
        "audio_status": None,
        "audio_provider_version": None,
        "audio_metadata": None,
        "face_provider": None,
        "face_signal_type": None,
        "face_status": None,
        "face_score": None,
        "face_provider_version": None,
        "face_metadata": None,
        "lip_forensics_provider": None,
        "lip_forensics_signal_type": None,
        "lip_forensics_status": None,
        "lip_forensics_score": None,
        "lip_forensics_provider_version": None,
        "lip_forensics_metadata": None,
    }

    return SimpleNamespace(**{**values, **overrides})


class NoEvidenceSession:
    """A session that answers every evidence query with nothing.

    `analysis_payloads` issues three statements for clip evidence, speaking timelines and
    audio windows. None of the rows here carries a signal, so all three are asked for an
    empty list of ids and this is what an empty answer looks like.
    """

    def execute(self, statement):
        return SimpleNamespace(all=lambda: [])


def test_an_analysis_from_before_the_columns_reads_back_as_not_recorded():
    """It renders, and it renders as unknown rather than as an upload.

    Both halves matter. A read path that raised on a null would make every analysis stored
    before R7-T12 unreachable; one that substituted `upload` would make them all readable and
    some of them wrong.
    """
    [summary] = analysis_payloads(NoEvidenceSession(), [summary_row()])

    assert summary.acquisition_method is None
    assert summary.source_host is None
    # Everything else about the row still reads exactly as it did.
    assert summary.original_sha256 == "a" * 64
    assert summary.media.codec_name == "h264"


def test_an_old_assembled_row_is_not_reinterpreted_as_a_publisher_original():
    """`was_assembled = false` on a null-method row keeps meaning only what it meant.

    R7-T1 gave every pre-existing row `false` by server default, so on a row with no recorded
    method it says "not muxed here" and nothing about who served the bytes. The read path
    passes it through beside a null method and leaves the reading to the wording, which is
    where the distinction is drawn.
    """
    [summary] = analysis_payloads(NoEvidenceSession(), [summary_row(was_assembled=False)])

    assert summary.was_assembled is False
    assert summary.acquisition_method is None


def test_a_recorded_acquisition_reaches_the_read_path_unchanged():
    """Passed through as stored — no defaulting, no normalization done a second time."""
    [summary] = analysis_payloads(
        NoEvidenceSession(),
        [
            summary_row(
                acquisition_method=ACQUISITION_METHOD_URL,
                source_host="youtube.com",
                was_assembled=True,
            )
        ],
    )

    assert summary.acquisition_method == ACQUISITION_METHOD_URL
    assert summary.source_host == "youtube.com"
    assert summary.was_assembled is True


# --------------------------------------------------------------------------------------
# The wording contract (R7-T12)
# --------------------------------------------------------------------------------------
#
# The report's two acquisition sentences, executed rather than read. The block of
# `app/analysis.ts` between the two constants and the end of `credentialsAbsentStatement` is
# self-contained — no imports, no React, two pure functions over three fields — so it is
# transpiled with the web application's own TypeScript and run under node.
#
# Executed rather than pattern-matched because the failure this guards against is a correct
# sentence on the wrong branch. A source-text assertion would pass just as happily if the
# assembled wording were returned for a single-served acquisition.

NODE = "node"
TYPESCRIPT = WEB_ROOT / "node_modules" / "typescript"

# Skipped rather than failed where the toolchain is absent, which is the same rule the other
# web-reading tests in this suite follow: a backend checkout with no `npm install` is a
# legitimate state, and these tests have nothing to read there.
requires_wording = pytest.mark.skipif(
    not (WEB_ANALYSIS.exists() and TYPESCRIPT.exists() and which(NODE) is not None),
    reason="node and the web application's TypeScript are needed to run the report's wording",
)


def _wording_source() -> str:
    """The self-contained block of `analysis.ts` that holds the two sentence functions.

    Sliced by name rather than by line number, so moving the block within the file does not
    break this and removing either function fails loudly here instead of silently skipping.
    """
    source = WEB_ANALYSIS.read_text(encoding="utf-8")

    start = source.index("export const ACQUISITION_METHOD_UPLOAD")
    tail = source.index("export function credentialsAbsentStatement", start)
    end = source.index("\n}\n", tail) + len("\n}\n")

    return source[start:end]


def _run_wording(cases: list[dict]) -> list[dict]:
    """Transpile the block and ask it for its sentence for each case.

    One node process for the whole parametrized set rather than one per case: the transpile
    is the expensive part and it is identical every time.
    """
    driver = f"""
        const ts = require({str(TYPESCRIPT)!r});
        const module_ = {{ exports: {{}} }};
        const compiled = ts.transpileModule(
            {_wording_source()!r},
            {{ compilerOptions: {{ module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2020 }} }}
        ).outputText;
        new Function("exports", "module", compiled)(module_.exports, module_);
        const cases = JSON.parse(process.argv[1]);
        console.log(JSON.stringify(cases.map((analysis) => ({{
            acquisition: module_.exports.acquisitionStatement(analysis),
            credentials: module_.exports.credentialsAbsentStatement(analysis),
        }}))));
    """

    result = subprocess.run(
        [NODE, "-e", driver, json.dumps(cases)],
        capture_output=True,
        text=True,
        timeout=60,
    )

    if result.returncode != 0:
        raise RuntimeError(f"node exited {result.returncode}:\n{result.stderr.strip()}")

    return json.loads(result.stdout)


# The four acquisitions the report has to describe, and the one sentence each may produce.
# Written out in full rather than assembled from fragments: these strings are the contract,
# and a test that built them the same way the code does would agree with any change to it.
UPLOAD = {
    "acquisition_method": ACQUISITION_METHOD_UPLOAD,
    "source_host": None,
    "was_assembled": False,
}
URL_SINGLE_SERVED = {
    "acquisition_method": ACQUISITION_METHOD_URL,
    "source_host": "videos.example.com",
    "was_assembled": False,
}
URL_ASSEMBLED = {
    "acquisition_method": ACQUISITION_METHOD_URL,
    "source_host": "youtube.com",
    "was_assembled": True,
}
UNRECORDED = {"acquisition_method": None, "source_host": None, "was_assembled": False}
UNRECORDED_ASSEMBLED = {
    "acquisition_method": None,
    "source_host": None,
    "was_assembled": True,
}


@requires_wording
def test_a_single_served_url_acquisition_says_only_that_it_was_acquired():
    """The wording contract for the commonest URL case, to the character.

    "acquired from" and "a single served file" are the whole of the claim. It says a file was
    fetched from that host and that the host served it as one file — both facts about the
    fetch — and stops there, because DeepGuard did not establish that the host is the
    publisher or that the bytes are what the publisher issued.
    """
    [rendered] = _run_wording([URL_SINGLE_SERVED])

    assert rendered["acquisition"] == (
        "Media was acquired from videos.example.com as a single served file."
    )


@requires_wording
def test_an_assembled_url_acquisition_says_it_was_assembled_from_components():
    """The DASH/HLS case, which is the one a reader is likeliest to misread.

    The source published no single file, so the stored artifact was muxed here. The sentence
    says so in the same breath as the host, which is what stops a reader taking the hash
    beside it for the hash of something the source served whole.
    """
    [rendered] = _run_wording([URL_ASSEMBLED])

    assert rendered["acquisition"] == (
        "Media was acquired from youtube.com and assembled from "
        "multiple served media components."
    )


@requires_wording
def test_an_upload_says_only_that_it_was_uploaded():
    """No host, no origin, and no claim about the file's history.

    DeepGuard knows a client sent these bytes and nothing whatever about where the client got
    them. "Uploaded by the submitter" is the entire extent of what the record supports.
    """
    [rendered] = _run_wording([UPLOAD])

    assert rendered["acquisition"] == "The analysed artifact was uploaded by the submitter."


@requires_wording
def test_an_analysis_with_no_recorded_acquisition_says_so():
    """The historical row, described as unrecorded rather than as an upload.

    This is the sentence that keeps the whole feature honest. Every analysis stored before
    R7-T12 reaches this branch, and the alternative — falling through to the upload wording —
    would state a provenance fact about thousands of rows that nobody ever established.
    """
    [rendered] = _run_wording([UNRECORDED])

    assert rendered["acquisition"] == (
        "How this artifact was acquired was not recorded for this analysis."
    )


@requires_wording
def test_an_old_row_that_was_assembled_still_reports_the_assembly():
    """`was_assembled = true` is a recorded fact even where the method is not.

    True was only ever written by an acquisition that really did mux two streams here, so it
    can be reported on a row with no method. False on such a row is only R7-T1's server
    default and is reported as nothing at all — which is the asymmetry the previous test
    fixes in place, and the difference between reading a record and reinterpreting a default.
    """
    [rendered] = _run_wording([UNRECORDED_ASSEMBLED])

    assert rendered["acquisition"] == (
        "How this artifact was acquired was not recorded for this analysis. "
        "It was assembled by DeepGuard from separate video and audio streams."
    )


@requires_wording
def test_no_acquisition_sentence_claims_the_artifact_is_a_publisher_original():
    """The words that would turn an acquisition into a provenance claim, in every branch.

    A vocabulary check rather than a sentence check, so a later rewording that reads naturally
    and claims too much fails here. "Original", "publisher", "authentic", "unmodified",
    "verified" and "genuine" are each a claim about the file's history, and DeepGuard's
    evidence reaches the fetch and no further.

    `credentialsAbsentStatement` is exempt from exactly one of them, and only in the phrase
    that *denies* the claim — see the test below.
    """
    rendered = _run_wording(
        [UPLOAD, URL_SINGLE_SERVED, URL_ASSEMBLED, UNRECORDED, UNRECORDED_ASSEMBLED]
    )

    for sentence in (case["acquisition"] for case in rendered):
        lowered = sentence.lower()
        for claim in (
            "original",
            "publisher",
            "authentic",
            "unmodified",
            "verified",
            "genuine",
            "as received",
        ):
            assert claim not in lowered, sentence


@requires_wording
def test_the_credentials_absence_is_scoped_to_the_acquired_artifact():
    """The C2PA sentence names the artifact it read and disclaims everything upstream.

    A manifest can be stripped by any hop between a publisher and this fetch — a re-encode, a
    CDN, a platform's own pipeline — so its absence in the acquired file is evidence about
    that file only. The second sentence exists to say that outright, because a reader who
    stopped at the first would draw exactly the wrong conclusion.
    """
    [rendered] = _run_wording([URL_SINGLE_SERVED])

    assert rendered["credentials"] == (
        "No Content Credentials were found in the artifact acquired from "
        "videos.example.com. This does not establish whether credentials were present in "
        "an upstream or publisher-original file."
    )


@requires_wording
def test_the_credentials_absence_names_no_host_it_does_not_have():
    """An upload and an unrecorded acquisition get the same limit, without a host.

    The claim shrinks to the artifact itself rather than reaching for an origin. There is no
    upstream file this service can point at, so it points at none — a sentence naming a host
    for an upload would be inventing the one fact an upload does not have.
    """
    rendered = _run_wording([UPLOAD, UNRECORDED])

    for case in rendered:
        assert case["credentials"] == (
            "No Content Credentials were found in the analysed artifact. This does not "
            "establish whether credentials were present in any file it was derived from."
        )
        assert "acquired from" not in case["credentials"]


@requires_wording
def test_the_assembled_acquisition_also_disclaims_the_upstream_file():
    """Assembly does not change what the absence of credentials establishes.

    The manifest is as absent, and as uninformative about an upstream file, on an artifact
    muxed here as on one served whole. The sentence is the same one, with the same host.
    """
    [rendered] = _run_wording([URL_ASSEMBLED])

    assert rendered["credentials"] == (
        "No Content Credentials were found in the artifact acquired from youtube.com. "
        "This does not establish whether credentials were present in an upstream or "
        "publisher-original file."
    )


@requires_wording
def test_no_sentence_can_interpolate_a_url_into_the_report():
    """The functions read `source_host` and nothing else, so there is nothing else to leak.

    Defence in depth against a leak the API is already supposed to prevent. If a full URL ever
    did reach the browser in that field, the sentence would print it — so this asserts that
    the sentences carry the field and that the field is the only input they have.
    """
    [rendered] = _run_wording(
        [
            {
                "acquisition_method": ACQUISITION_METHOD_URL,
                "source_host": "videos.example.com",
                "was_assembled": False,
                # Fields the report never reads. Present here because a function that reached
                # for one of them would put it on the page.
                "submitted_url": "https://videos.example.com/clip?token=sk-live-secret",
                "original_filename": "clip.mp4",
            }
        ]
    )

    for sentence in rendered.values():
        assert "token" not in sentence
        assert "sk-live-secret" not in sentence
        assert "https://" not in sentence


# --------------------------------------------------------------------------------------
# Isolation
# --------------------------------------------------------------------------------------


def test_reading_a_host_is_not_a_permission_to_fetch_it(monkeypatch):
    """Normalization answers for a host the SSRF guard would refuse, and refuses it anyway.

    The two are separate questions and must stay separate. `normalized_host` says what a URL
    names; `validate_url` says whether this server may fetch it. A normalization that quietly
    doubled as an allowlist, or a validation that trusted a name because it parsed, would each
    be the same bug from a different direction.

    So: the loopback URL has a perfectly readable host, and `validate_url` still says no.
    """
    assert downloader.normalized_host("http://localhost:8000/clip") == "localhost"

    monkeypatch.setattr(
        downloader.socket,
        "getaddrinfo",
        lambda *args, **kwargs: [(None, None, None, "", ("127.0.0.1", 8000))],
    )

    with pytest.raises(downloader.BlockedAddress):
        downloader.validate_url("http://localhost:8000/clip")


def test_recording_a_source_never_touches_the_network():
    """Normalization is string work, and nothing else.

    It runs on the request path, and a version of it that resolved a name would be a second,
    unguarded resolution beside the one the SSRF defence installs its guard around. Proven by
    removing `getaddrinfo` entirely: an implementation that reached for the network raises,
    and this one answers.
    """
    import socket as socket_module

    real = socket_module.getaddrinfo
    socket_module.getaddrinfo = None
    try:
        assert downloader.normalized_host("https://www.example.com/clip") == "example.com"
    finally:
        socket_module.getaddrinfo = real


def test_no_risk_rule_reads_an_acquisition_fact():
    """The engine's inputs, read from the engine.

    An acquisition is not evidence. Media fetched from a host is neither more nor less
    authentic than media uploaded, and an artifact muxed here is neither more nor less
    authentic than one served whole — so a rule that read any of the three would be deriving
    suspicion from DeepGuard's own plumbing.

    Asserted against the module's source rather than by running a decision, because what is
    being ruled out is a *reference*: a rule that read the column only under some condition
    would still be a rule that reads it, and no fixture would reliably reach that branch.
    """
    source = Path(risk_engine.__file__).read_text(encoding="utf-8")

    for column in ("acquisition_method", "source_host", "was_assembled"):
        assert column not in source
