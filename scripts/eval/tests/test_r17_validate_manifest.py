"""R17-T1 manifest validation: what it catches, and that it freezes all or nothing."""

import csv
import io
import json

import pytest

from eval.r17_validate_manifest import (
    COLUMNS,
    ORIGINAL_COLUMNS,
    ManifestError,
    diff_findings,
    first_free_shot,
    format_ts,
    freeze,
    overlaps,
    parse_changelog,
    parse_ts,
    render_csv,
    render_json,
    shift_window,
    timestamp_findings,
    validate,
    zero_gaps,
)

SHA = "a" * 64
LAYOUT = [  # (prefix, count, truth, generator, publisher, channel, url, evidence_url)
    ("AI", 17, "AI", "Sora", "OpenAI", "OpenAI", "https://www.youtube.com/watch?v=sora0000000",
     "https://www.youtube.com/watch?v=sora0000000"),
    ("AI", 11, "AI", "Veo", "Google DeepMind", "Google DeepMind",
     "https://www.youtube.com/watch?v=veo00000000", "https://deepmind.google/veo"),
    ("AI", 1, "AI", "Runway Gen-3", "Runway", "Runway", "https://cdn.example/gen3.mp4",
     "https://runway.com/research/introducing-gen-3-alpha"),
    ("AI", 1, "AI", "Runway Gen-4", "Runway", "Runway",
     "https://www.youtube.com/watch?v=gen40000000", "https://www.youtube.com/watch?v=gen40000000"),
    ("REAL", 10, "REAL", "NASA Artemis II Orion SAW1", "NASA", "NASA",
     "https://images.nasa.gov/details/saw1", "https://images.nasa.gov/details/saw1"),
    ("REAL", 5, "REAL", "NASA Artemis II Orion ascent", "NASA", "NASA",
     "https://images.nasa.gov/details/saj", "https://images.nasa.gov/details/saj"),
]


def make_rows():
    """45 rows that pass every check: 8 s clips, 10 s apart, all evidence present."""
    rows, n = [], {"AI": 0, "REAL": 0}
    for cluster, (prefix, count, truth, gen, pub, chan, url, ev) in enumerate(LAYOUT, 1):
        for i in range(count):
            n[prefix] += 1
            start = i * 10_000
            rows.append({
                "case_id": f"{prefix}-{n[prefix]:03d}", "truth": truth,
                "generator_or_source": gen, "publisher": pub, "source_url": url,
                "clip_start": format_ts(start), "clip_end": format_ts(start + 8000),
                "notes": "scene", "provenance": "official_showcase",
                "validation_status": "PROVISIONAL",
                "source_title": "title", "source_channel": chan, "source_duration_s": "300.000",
                "source_upload_date": "2024-01-01", "accessed_at_utc": "2026-10-15T00:00:00Z",
                "url_status": "OK", "clip_duration_s": "8.000", "clip_sha256": SHA,
                "clip_format_id": "136", "clip_resolution": "1280x720", "clip_fps": "30",
                "clip_has_audio": "false", "source_cluster_id": f"C{cluster}",
                "timestamp_status": "ORIGINAL",
                "truth_evidence_type": "official_page" if "youtube" not in ev
                else "official_statement_description",
                "truth_evidence_url": ev, "truth_evidence_quote": '"generated"',
                "truth_evidence_accessed_at_utc": "2026-10-15T00:00:00Z",
                "truth_assertion_scope": "whole_video", "visual_gt_check": "PASS",
                "visual_gt_note": "clean", "gt_status": "CONFIRMED",
            })
    return rows


def write_dataset(path, rows, provisional=None, changelog="# CHANGELOG\n"):
    provisional = provisional if provisional is not None else [
        {c: r[c] for c in ORIGINAL_COLUMNS} for r in rows]
    (path / "manifest.csv").write_text(render_csv(rows), encoding="utf-8")
    (path / "manifest.json").write_text(render_json(rows), encoding="utf-8")
    (path / "manifest.provisional.csv").write_text(render_csv(provisional, ORIGINAL_COLUMNS))
    (path / "manifest.provisional.json").write_text(render_json(provisional, ORIGINAL_COLUMNS))
    (path / "CHANGELOG.md").write_text(changelog, encoding="utf-8")
    return path


TOOLS = {"yt_dlp": "x", "ffmpeg": "y", "tool_image": "z"}


# --- timestamps -----------------------------------------------------------------------------

@pytest.mark.parametrize("text,ms", [("00:13", 13_000), ("00:13.750", 13_750), ("01:31.5", 91_500),
                                     ("123:00.001", 7_380_001)])
def test_parse_ts_accepts_whole_and_subsecond(text, ms):
    assert parse_ts(text) == ms


@pytest.mark.parametrize("text", ["", "13", "0:13", "00:60", "00:13.7500", "00:13,750", "-1:00"])
def test_parse_ts_rejects_malformed(text):
    with pytest.raises(ManifestError):
        parse_ts(text)


def test_format_ts_round_trips():
    assert format_ts(13_750) == "00:13.750"
    assert parse_ts(format_ts(91_750)) == 91_750


def test_overlaps_found_like_ai018_019_and_ai027_028():
    rows = [
        {"case_id": "AI-018", "source_url": "u", "clip_start": "00:06", "clip_end": "00:14"},
        {"case_id": "AI-019", "source_url": "u", "clip_start": "00:13", "clip_end": "00:21"},
        {"case_id": "AI-027", "source_url": "u", "clip_start": "01:24", "clip_end": "01:32"},
        {"case_id": "AI-028", "source_url": "u", "clip_start": "01:31", "clip_end": "01:39"},
        {"case_id": "OTHER", "source_url": "v", "clip_start": "00:13", "clip_end": "00:21"},
    ]
    assert overlaps(rows) == [("AI-018", "AI-019", 1000), ("AI-027", "AI-028", 1000)]


def test_zero_gap_is_flagged_not_overlap():
    rows = [
        {"case_id": "AI-021", "source_url": "u", "clip_start": "00:33", "clip_end": "00:41"},
        {"case_id": "AI-022", "source_url": "u", "clip_start": "00:41", "clip_end": "00:49"},
    ]
    assert overlaps(rows) == []
    assert zero_gaps(rows) == [("AI-021", "AI-022")]


def test_timestamp_findings_duration_and_source_end():
    good = make_rows()[0]
    short = dict(good, clip_end="00:07.999")
    past = dict(good, clip_start="04:55.000", clip_end="05:03.000")
    assert timestamp_findings([good]) == []
    assert any("expected 8.000s" in f for f in timestamp_findings([short]))
    assert any("past source end" in f for f in timestamp_findings([past]))


# --- correction rules -----------------------------------------------------------------------

def test_shift_window_minimal_shift_out_of_a_title_card():
    # Sora-like: card until 11.067, shot 11.067–21.067, clip asked for at 9.000.
    assert shift_window(9_000, (11_067, 21_067)) == (11_567, 19_567)


def test_shift_window_no_shift_when_already_clean():
    assert shift_window(12_000, (11_067, 21_067)) == (12_000, 20_000)


def test_shift_window_exact_eight_second_shot_gets_no_margin():
    # Veo-like: every scene is exactly 8.000 s and starts at x.750.
    assert shift_window(6_000, (5_750, 13_750)) == (5_750, 13_750)


def test_shift_window_avoids_occupied_and_never_overlaps():
    s, e = shift_window(13_000, (10_000, 40_000), occupied=[(12_000, 20_000)])
    assert e <= 12_000 or s >= 20_000
    assert (s, e) == (20_000, 28_000)


def test_shift_window_none_when_shot_too_short():
    # Veo dog-with-balloon: 6.2 s — no 8 s window, rule 2's case.
    assert shift_window(91_000, (91_750, 97_958)) is None


def test_first_free_shot_skips_used_dirty_and_short_shots():
    shots = [
        {"start_ms": 0, "end_ms": 10_000, "clean": True},       # holds a used clip
        {"start_ms": 10_000, "end_ms": 16_000, "clean": True},  # too short
        {"start_ms": 16_000, "end_ms": 30_000, "clean": False},  # title card
        {"start_ms": 30_000, "end_ms": 38_000, "clean": True},
    ]
    assert first_free_shot(shots, used=[(1_000, 9_000)]) == (30_000, 38_000)
    assert first_free_shot(shots[:3], used=[(1_000, 9_000)]) is None


# --- evidence, clusters, CSV/JSON -----------------------------------------------------------

def test_valid_dataset_has_no_findings(tmp_path):
    _, findings = validate(write_dataset(tmp_path, make_rows()))
    assert findings == []


@pytest.mark.parametrize("field,value,needle", [
    ("truth_evidence_quote", "  ", "empty truth_evidence_quote"),
    ("truth_evidence_url", "https://some-blog.example/sora", "not an official host"),
    ("truth_evidence_type", "channel_title", "truth_evidence_type"),
    ("truth_assertion_scope", "", "truth_assertion_scope"),
    ("gt_status", "AMBIGUOUS", "gt_status"),
    ("visual_gt_check", "AMBIGUOUS", "visual_gt_check"),
    ("clip_sha256", "", "clip_sha256"),
    ("url_status", "UNAVAILABLE", "url_status"),
])
def test_one_bad_case_is_a_finding(tmp_path, field, value, needle):
    rows = make_rows()
    rows[3][field] = value
    _, findings = validate(write_dataset(tmp_path, rows))
    assert any(needle in f for f in findings)


def test_youtube_evidence_needs_official_channel():
    from eval.r17_validate_manifest import evidence_findings
    row = dict(make_rows()[0], source_channel="Some Reuploader")
    assert any("non-official channel" in f for f in evidence_findings(row))


def test_cluster_must_follow_source_url(tmp_path):
    rows = make_rows()
    rows[0]["source_cluster_id"] = "C2"
    _, findings = validate(write_dataset(tmp_path, rows))
    assert any("spans sources" in f or "has clusters" in f for f in findings)


def test_csv_json_disagreement_is_found(tmp_path):
    write_dataset(tmp_path, make_rows())
    data = json.loads((tmp_path / "manifest.json").read_text())
    data[5]["notes"] = "different"
    (tmp_path / "manifest.json").write_text(json.dumps(data))
    _, findings = validate(tmp_path)
    assert any("csv and json differ" in f for f in findings)


def test_counts_must_match_the_agreed_dataset(tmp_path):
    rows = make_rows()[:-1]
    _, findings = validate(write_dataset(tmp_path, rows))
    assert any("44 rows" in f for f in findings)


def test_malformed_row_is_rejected(tmp_path):
    rows = make_rows()
    rows[7]["clip_start"] = "7 seconds"
    _, findings = validate(write_dataset(tmp_path, rows))
    assert any("bad timestamp" in f for f in findings)


# --- CHANGELOG diff -------------------------------------------------------------------------

CHANGE = "| AI-001 | clip_start | `00:00.000` | `00:01.500` | rule 1 | reason |\n"


def test_parse_changelog_reads_only_change_rows():
    text = "# CHANGELOG\n\n| case | field | old | new |\n|---|---|---|---|\n" + CHANGE
    assert parse_changelog(text) == {("AI-001", "clip_start"): ("00:00.000", "00:01.500")}


def test_parse_changelog_rejects_duplicate_rows():
    with pytest.raises(ManifestError):
        parse_changelog(CHANGE + CHANGE)


def test_diff_only_in_changelog_cells():
    prov = [{c: "x" for c in ORIGINAL_COLUMNS} | {"case_id": "AI-001"}]
    frozen = [dict(prov[0], clip_start="00:01.500", validation_status="VALIDATED")]
    prov[0]["clip_start"] = "00:00.000"
    logged = {("AI-001", "clip_start"): ("00:00.000", "00:01.500")}
    assert diff_findings(prov, frozen, logged) == []
    assert diff_findings(prov, frozen, {}) == ["AI-001.clip_start: changed without a CHANGELOG row"]
    wrong = {("AI-001", "clip_start"): ("00:00.000", "00:02.000")}
    assert "CHANGELOG says" in diff_findings(prov, frozen, wrong)[0]


def test_unlogged_change_against_provisional_blocks(tmp_path):
    rows = make_rows()
    provisional = [{c: r[c] for c in ORIGINAL_COLUMNS} for r in rows]
    rows[0]["notes"] = "silently edited"
    _, findings = validate(write_dataset(tmp_path, rows, provisional))
    assert any("AI-001.notes" in f for f in findings)


# --- freeze ---------------------------------------------------------------------------------

def test_freeze_writes_validated_manifest_and_record(tmp_path):
    write_dataset(tmp_path, make_rows())
    assert freeze(tmp_path, TOOLS, now="2026-10-15T00:00:00Z") == []
    rows = list(csv.DictReader(io.StringIO((tmp_path / "manifest.csv").read_text())))
    assert {r["validation_status"] for r in rows} == {"VALIDATED"}
    assert list(rows[0]) == list(COLUMNS)
    record = json.loads((tmp_path / "FREEZE.json").read_text())
    assert record["counts"]["cases"] == 45
    assert record["counts"]["source_clusters"] == 6
    assert len(record["clips"]) == 45
    assert any("not generalisable" in s for s in record["limitations"])


def test_freeze_is_idempotent(tmp_path):
    write_dataset(tmp_path, make_rows())
    freeze(tmp_path, TOOLS, now="2026-10-15T00:00:00Z")
    first = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    assert freeze(tmp_path, TOOLS, now="2027-01-01T00:00:00Z") == []
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == first


def test_no_partial_freeze(tmp_path):
    rows = make_rows()
    rows[44]["gt_status"] = "AMBIGUOUS"
    write_dataset(tmp_path, rows)
    before = (tmp_path / "manifest.csv").read_bytes()
    findings = freeze(tmp_path, TOOLS, now="2026-10-15T00:00:00Z")
    assert findings
    assert not (tmp_path / "FREEZE.json").exists()
    assert (tmp_path / "manifest.csv").read_bytes() == before
