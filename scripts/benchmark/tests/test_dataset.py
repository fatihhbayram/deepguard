"""Manifest ingestion: what a valid corpus yields, and what an invalid one reports."""

import pytest

from benchmark.dataset import (
    Clip,
    ManifestError,
    label_counts,
    load_manifest,
)


def write_corpus(root, rows, *, header="clip_id,path,label,audio_path", media=()):
    for name in media:
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_bytes(b"not really media")
    manifest = root / "manifest.csv"
    manifest.write_text("\n".join([header, *rows]) + "\n", encoding="utf-8")
    return manifest


def test_loads_labels_paths_and_optional_audio(tmp_path):
    manifest = write_corpus(
        tmp_path,
        [
            "b_synth,clips/b.mp4,synthetic,clips/b.wav",
            "a_real,clips/a.mp4,real,",
        ],
        media=["clips/a.mp4", "clips/b.mp4", "clips/b.wav"],
    )

    clips = load_manifest(manifest).clips

    # Sorted by clip_id, not by file order, so two runs line up record for record.
    assert [c.clip_id for c in clips] == ["a_real", "b_synth"]
    assert clips[0] == Clip(
        clip_id="a_real", path=tmp_path / "clips/a.mp4", label="real", audio_path=None
    )
    assert clips[1].audio_path == tmp_path / "clips/b.wav"
    assert clips[0].is_manipulated is False
    assert clips[1].is_manipulated is True


def test_every_non_real_label_is_the_positive_class(tmp_path):
    manifest = write_corpus(
        tmp_path,
        [
            "s,s.mp4,synthetic,",
            "f,f.mp4,face_swap,",
            "a,a.wav,audio_spoof,",
            "r,r.mp4,real,",
        ],
        media=["s.mp4", "f.mp4", "a.wav", "r.mp4"],
    )

    clips = load_manifest(manifest).clips

    assert {c.label: c.is_manipulated for c in clips} == {
        "synthetic": True,
        "face_swap": True,
        "audio_spoof": True,
        "real": False,
    }
    assert label_counts(clips) == {
        "audio_spoof": 1,
        "face_swap": 1,
        "real": 1,
        "synthetic": 1,
    }


def test_clip_id_defaults_to_the_path_as_written(tmp_path):
    manifest = write_corpus(
        tmp_path, [",clips/a.mp4,real,"], media=["clips/a.mp4"]
    )
    assert load_manifest(manifest).clips[0].clip_id == "clips/a.mp4"


def test_paths_resolve_against_the_manifest_not_the_working_directory(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    manifest = write_corpus(corpus, ["a,a.mp4,real,"], media=["a.mp4"])
    assert load_manifest(manifest).clips[0].path == corpus / "a.mp4"


def test_absolute_paths_are_left_alone(tmp_path):
    media = tmp_path / "elsewhere" / "a.mp4"
    media.parent.mkdir()
    media.write_bytes(b"x")
    manifest = write_corpus(tmp_path, [f"a,{media},real,"])
    assert load_manifest(manifest).clips[0].path == media


def test_unknown_columns_are_ignored(tmp_path):
    manifest = write_corpus(
        tmp_path,
        ["a,a.mp4,real,,GAN paper 2024"],
        header="clip_id,path,label,audio_path,provenance",
        media=["a.mp4"],
    )
    assert load_manifest(manifest).clips[0].clip_id == "a"


def test_a_missing_manifest_is_reported(tmp_path):
    with pytest.raises(ManifestError, match="manifest not found"):
        load_manifest(tmp_path / "nope.csv")


def test_missing_required_columns_are_named(tmp_path):
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("clip_id,audio_path\na,\n", encoding="utf-8")
    with pytest.raises(ManifestError, match="missing required column\\(s\\): label, path"):
        load_manifest(manifest)


def test_a_manifest_with_no_rows_is_rejected(tmp_path):
    manifest = write_corpus(tmp_path, [])
    with pytest.raises(ManifestError, match="describes no clips"):
        load_manifest(manifest)


def test_every_problem_is_reported_at_once(tmp_path):
    manifest = write_corpus(
        tmp_path,
        [
            "good,a.mp4,real,",
            "bad_label,a.mp4,deepfake,",
            "missing,gone.mp4,real,",
            "good,a.mp4,real,",
            ",,real,",
        ],
        media=["a.mp4"],
    )

    with pytest.raises(ManifestError) as raised:
        load_manifest(manifest)

    message = str(raised.value)
    assert "4 problem(s)" in message
    assert "line 3: unknown label 'deepfake'" in message
    assert "line 4: media file not found" in message
    assert "line 5: duplicate clip_id 'good'" in message
    assert "line 6: empty `path`" in message


def test_a_declared_audio_track_must_exist(tmp_path):
    manifest = write_corpus(
        tmp_path, ["a,a.mp4,audio_spoof,a.wav"], media=["a.mp4"]
    )
    with pytest.raises(ManifestError, match="audio file not found"):
        load_manifest(manifest)


def test_fingerprint_tracks_the_manifest_contents(tmp_path):
    manifest = write_corpus(tmp_path, ["a,a.mp4,real,"], media=["a.mp4"])
    original = load_manifest(manifest).manifest_sha256

    assert load_manifest(manifest).manifest_sha256 == original

    # A relabelled corpus is a different ground truth and must fingerprint differently.
    write_corpus(tmp_path, ["a,a.mp4,synthetic,"], media=["a.mp4"])
    assert load_manifest(manifest).manifest_sha256 != original


def test_byte_order_mark_does_not_hide_the_first_column(tmp_path):
    """Excel's "CSV UTF-8" export prefixes a BOM, which must not reach a column name.

    Both halves matter. A BOM on a required column made the manifest look like it was
    missing `path` while `path` was plainly there; a BOM on `clip_id` was worse, because
    the run *succeeded* with every clip silently identified by its path instead.
    """
    manifest = tmp_path / "manifest.csv"
    (tmp_path / "a.mp4").write_bytes(b"not really media")
    (tmp_path / "b.mp4").write_bytes(b"not really media")
    manifest.write_text(
        "﻿clip_id,path,label\r\na_real,a.mp4,real\r\nb_synth,b.mp4,synthetic\r\n",
        encoding="utf-8",
    )

    clips = load_manifest(manifest).clips

    assert [c.clip_id for c in clips] == ["a_real", "b_synth"]
    assert clips[0].path == tmp_path / "a.mp4"


def test_byte_order_mark_before_a_required_column(tmp_path):
    manifest = tmp_path / "manifest.csv"
    (tmp_path / "a.mp4").write_bytes(b"not really media")
    manifest.write_text("﻿path,label\r\na.mp4,real\r\n", encoding="utf-8")

    assert [c.clip_id for c in load_manifest(manifest).clips] == ["a.mp4"]


def test_fingerprint_still_separates_the_two_encodings(tmp_path):
    """The parse tolerates a BOM; the fingerprint must not paper over it.

    Byte-identical ground truth is the claim `manifest_sha256` makes, so two files that
    differ by three bytes have to hash differently even though they load alike.
    """
    plain = tmp_path / "plain.csv"
    bommed = tmp_path / "bommed.csv"
    (tmp_path / "a.mp4").write_bytes(b"not really media")
    plain.write_text("clip_id,path,label\na,a.mp4,real\n", encoding="utf-8")
    bommed.write_text("﻿clip_id,path,label\na,a.mp4,real\n", encoding="utf-8")

    assert [c.clip_id for c in load_manifest(plain).clips] == ["a"]
    assert [c.clip_id for c in load_manifest(bommed).clips] == ["a"]
    assert (
        load_manifest(plain).manifest_sha256
        != load_manifest(bommed).manifest_sha256
    )
