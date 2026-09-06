"""The split guarantee and the leakage audit: what they catch, and what they refuse to guess."""

import pytest

from eval.corpus import (
    SPLIT_CALIBRATION,
    SPLIT_EVALUATION,
    CorpusError,
    CorpusItem,
    leakage_findings,
    split_by_lineage,
    summarise,
)


def item(clip_id, lineage, label="real", sha=None, derivative_of=None, split=None):
    """One corpus record with the fields these tests care about and defaults for the rest."""
    return CorpusItem(
        clip_id=clip_id,
        path=f"clips/{clip_id}.mp4",
        label=label,
        family="family",
        stratum_primary="stratum",
        strata=["stratum"],
        source_lineage_id=lineage,
        base_media_id=derivative_of or clip_id,
        derivative_of=derivative_of,
        derivation="none" if derivative_of is None else "social_transcode_shorts",
        source="test",
        acquisition_type="test",
        label_provenance="test",
        license="test",
        permission_status="test",
        redistributable=False,
        private=False,
        sha256=(sha or clip_id).ljust(64, "0"),
        bytes=1,
        split=split,
    )


def test_split_refuses_an_unassigned_lineage():
    """A lineage nobody named must stop the build, not default into a split.

    This is the failure the whole task is written against: a clip that lands in the evaluation
    set because a rule did not cover it is a hold-out nobody chose.
    """
    with pytest.raises(CorpusError) as error:
        split_by_lineage([item("a", "lineage:1")], {})
    assert "lineage:1" in str(error.value)


def test_split_refuses_an_unknown_split_name():
    with pytest.raises(CorpusError):
        split_by_lineage([item("a", "lineage:1")], {"lineage:1": "holdout"})


def test_clean_splits_report_no_leakage():
    items = [item("a", "l:1"), item("b", "l:2"), item("c", "l:3")]
    stamped = split_by_lineage(
        items, {"l:1": SPLIT_CALIBRATION, "l:2": SPLIT_EVALUATION, "l:3": SPLIT_EVALUATION}
    )
    findings = leakage_findings(stamped)
    assert findings["leaked"] is False
    assert findings["shared_lineage_ids"] == []
    assert findings["lineages_calibration"] == 1
    assert findings["lineages_evaluation"] == 2


def test_a_lineage_on_both_sides_is_caught():
    """The check the task requires, stated as the thing it must not miss."""
    items = [
        item("a", "l:1", split=SPLIT_CALIBRATION),
        item("b", "l:1", split=SPLIT_EVALUATION),
    ]
    findings = leakage_findings(items)
    assert findings["leaked"] is True
    assert findings["shared_lineage_ids"] == ["l:1"]


def test_identical_bytes_across_splits_are_caught_even_under_different_lineages():
    """Two lineage ids cannot make one file into two independent observations.

    This is the case a lineage-id comparison alone would pass: the build mislabelled the
    lineage, so only the bytes can say the two records are one measurement.
    """
    items = [
        item("a", "l:1", sha="dupe", split=SPLIT_CALIBRATION),
        item("b", "l:2", sha="dupe", split=SPLIT_EVALUATION),
    ]
    findings = leakage_findings(items)
    assert findings["leaked"] is True
    assert findings["shared_lineage_ids"] == []
    assert len(findings["cross_split_sha256"]) == 1


def test_a_derivative_separated_from_its_base_is_caught():
    items = [
        item("a", "l:1", split=SPLIT_CALIBRATION),
        item("a__t", "l:1", derivative_of="a", split=SPLIT_EVALUATION),
    ]
    findings = leakage_findings(items)
    assert findings["leaked"] is True
    assert findings["derivatives_split_from_base"] == ["a__t"]


def test_a_derivative_whose_base_is_absent_is_caught():
    items = [item("a__t", "l:1", derivative_of="a", split=SPLIT_EVALUATION)]
    assert leakage_findings(items)["derivatives_without_base"] == ["a__t"]


def test_an_unassigned_clip_counts_as_leakage():
    """An unsplit clip has not been kept out of anything, so the audit may not pass it."""
    assert leakage_findings([item("a", "l:1")])["leaked"] is True


def test_summarise_counts_lineages_and_clips_separately():
    items = [
        item("a", "l:1", split=SPLIT_EVALUATION),
        item("a__t", "l:1", derivative_of="a", split=SPLIT_EVALUATION),
        item("b", "l:2", split=SPLIT_EVALUATION),
    ]
    counts = summarise(items)[SPLIT_EVALUATION]
    assert counts["clips"] == 3
    assert counts["base_clips"] == 2
    assert counts["lineages"] == 2
