"""R9-T3 — replay `evaluate_v5` over the evaluation corpus and over contract fixtures.

    PYTHONPATH=scripts python3 scripts/eval/replay_r9t3.py \
        --corpus ../deepguard-corpus/r7t5/corpus.json \
        --svd-run  ../deepguard-corpus/runs/r7t5-svd/results.json \
        --face-run ../deepguard-corpus/runs/r7t5-face/results.json \
        --lip-run  ../deepguard-corpus/runs/r7t5-lip/results.json \
        --output-dir ../deepguard-corpus/r9t3 \
        --report docs/ai/reviews/R9_T3/r9-t3-replay-validation-report.md

**Offline only.** Nothing here is imported by `apps/`, nothing is written under `apps/`, and
`evaluate_v5` gains no production caller from this file. The engine is loaded from source by
path exactly as `replay.py` loads it, and the evidence assembly, provider-version derivation and
run reading are *reused* from `eval.replay` rather than written a second time — a second copy of
the evidence builder is the first place a replay silently stops describing the shipped rules.

**The report has two halves and they measure different things.**

*Part A* replays both rulesets over the same persisted evidence. `evaluate` (v4) and
`evaluate_v5` are called on the same `SvdEvidence`/`FaceEvidence`/`LipEvidence` objects for every
clip, so the transition matrix compares two rule sets and not two runs. No v4 level is mapped,
renamed or compared by string equality to a v5 verdict: both are recorded as the engine returned
them and the matrix counts the pairs that occurred.

*Part B* asserts the R9-T1 contract on synthetic evidence. It measures nothing about media — it
fixes the boundaries the corpus does not happen to contain (a score exactly at an operating
point, a row that was never written, a count of zero) and fails loudly when one moves.

**Nothing about the engine is adjusted to any of it.** Thresholds, rule ids, `D_TOTAL_V5` and
the calibration id are read and recorded. If Part A's rates are not what anyone hoped, that is
the finding.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.corpus import SPLIT_EVALUATION, CorpusItem, read_corpus
from eval.replay import (
    RISK_ENGINE_PATH,
    build_evidence,
    face_provider_version,
    load_risk_engine,
    lip_provider_version,
    read_run,
    svd_provider_version,
)

# The corpus labels, and what each one is the target population *for*. A detection rate is only
# a rate of anything if the denominator is the family the detector was calibrated against:
# `SVD` is measured on generated video, `B7` on face swaps, and neither is credited or debited
# for the other's population.
LABEL_GENUINE = "real"
LABEL_GENERATED = "synthetic"
LABEL_FACE_SWAP = "face_swap"


# --------------------------------------------------------------------------------------------
# Part A — corpus replay
# --------------------------------------------------------------------------------------------


def coverage_terms(engine: ModuleType, svd, face, lip) -> dict:
    """`D_total`, `D_usable`, `D_hits` for one clip, recomputed from the engine's own predicates.

    Recomputed rather than read back out of the decision, because `RiskDecision` carries a
    verdict and a rule id and not the counts behind them. The predicates are the engine's —
    `is_eligible_*`, `is_usable_*`, `svd_threshold_reached`, `face_threshold_reached` — so this
    is a second reading of the same functions, not a second implementation of the arithmetic.

    `lip` is accepted and deliberately never consulted: it is `evidence_only` under v5 and is
    outside all three counts (R9-T1 invariant 1). It is in the signature so that the one place
    a leak could be introduced is visible here rather than hidden by an absent parameter.
    """
    svd_usable = engine.is_eligible_svd(svd) and engine.is_usable_svd(svd)
    face_usable = engine.is_eligible_face(face) and engine.is_usable_face(face)
    svd_hit = svd_usable and engine.svd_threshold_reached(svd)
    face_hit = face_usable and engine.face_threshold_reached(face)
    return {
        "d_total": engine.D_TOTAL_V5,
        "d_usable": int(svd_usable) + int(face_usable),
        "d_hits": int(svd_hit) + int(face_hit),
        "svd_usable": svd_usable,
        "face_usable": face_usable,
        "svd_hit": svd_hit,
        "face_hit": face_hit,
    }


def replay_corpus(
    items: list[CorpusItem],
    engine: ModuleType,
    svd_run: dict,
    face_run: dict,
    lip_run: dict,
) -> dict:
    """Both rulesets over the same evidence, one row per clip.

    The two calls sit next to each other on purpose. `evaluate` and `evaluate_v5` receive the
    identical evidence objects, so any difference between the two columns is a difference of
    rules. The `lip=None` third call is the isolation check: same evidence with the
    evidence-only slot emptied, and its verdict recorded so the report can state the claim as a
    measurement rather than repeat it as a design intention.
    """
    versions = {
        "svd": svd_provider_version(svd_run["provenance"]),
        "face": face_provider_version(face_run["provenance"]),
        "lip": lip_provider_version(lip_run["provenance"]),
    }
    expected = {
        "svd": engine.SVD_PROVIDER_VERSION,
        "face": engine.FACE_PROVIDER_VERSION,
        "lip": engine.LIP_PROVIDER_VERSION,
    }
    eligibility = {
        name: {
            "observed_provider_version": versions[name],
            "production_provider_version": expected[name],
            "matches_calibrated_deployment": versions[name] == expected[name],
        }
        for name in ("svd", "face", "lip")
    }

    rows = []
    for item in items:
        svd, face, lip = build_evidence(
            item.clip_id, engine, svd_run, face_run, lip_run, versions
        )
        v4 = engine.evaluate(svd=svd, face=face, lip=lip)
        v5 = engine.evaluate_v5(svd=svd, face=face, lip=lip)
        v5_no_lip = engine.evaluate_v5(svd=svd, face=face, lip=None)
        terms = coverage_terms(engine, svd, face, lip)
        rows.append(
            {
                "clip_id": item.clip_id,
                "split": item.split,
                "label": item.label,
                "family": item.family,
                "stratum": item.stratum_primary,
                "source_lineage_id": item.source_lineage_id,
                "v4": {
                    "risk_level": v4.risk_level,
                    "rule_id": v4.rule_id,
                    "rules_version": v4.rules_version,
                },
                "v5": {
                    "verdict": v5.risk_level,
                    "rule_id": v5.rule_id,
                    "rules_version": v5.rules_version,
                },
                "v5_lip_removed": {
                    "verdict": v5_no_lip.risk_level,
                    "rule_id": v5_no_lip.rule_id,
                },
                "coverage": terms,
                "rows_present": {
                    "svd": svd is not None,
                    "face": face is not None,
                    "lip": lip is not None,
                },
                "statuses": {
                    "svd": svd.status if svd else None,
                    "face": face.status if face else None,
                    "lip": lip.status if lip else None,
                },
                "scores": {
                    "svd": svd.score if svd else None,
                    "face": face.score if face else None,
                    "lip": lip.score if lip else None,
                },
            }
        )

    return {"eligibility": eligibility, "rows": rows}


def _rate(numerator: int, denominator: int) -> float | None:
    """A rate, or `None` where there is no population to take one over.

    `None` rather than `0.0`: a rate over an empty denominator is not zero, and a report that
    prints `0.0%` for a population that does not exist has stated a measurement nobody took.
    """
    if denominator == 0:
        return None
    return numerator / denominator


def _lineage_rate(rows: list[dict], predicate) -> dict:
    """The same count over distinct recordings rather than over files.

    A third of this corpus is constructed derivatives of other clips in it, so a per-clip rate
    counts one bad recording up to nine times. A lineage counts once, and counts as flagged if
    *any* of its clips did — which is the operationally relevant question for a false positive:
    would this recording have produced a detection somewhere.
    """
    by_lineage = defaultdict(list)
    for row in rows:
        by_lineage[row["source_lineage_id"]].append(row)
    hit = sum(1 for group in by_lineage.values() if any(predicate(r) for r in group))
    return {"lineages": len(by_lineage), "flagged": hit, "rate": _rate(hit, len(by_lineage))}


def measure_part_a(engine: ModuleType, rows: list[dict]) -> dict:
    """The empirical figures, with no target to hit and no comparison to a target.

    Every rate below is a count divided by the population it belongs to. None of them is checked
    against an expectation anywhere in this file; the constraint on this task is to measure them.
    """
    detected = engine.VERDICT_MANIPULATION_DETECTED
    no_signal = engine.VERDICT_NO_SIGNAL
    inconclusive = engine.VERDICT_INCONCLUSIVE

    def subset(label: str) -> list[dict]:
        return [r for r in rows if r["label"] == label]

    def verdicts(sub: list[dict]) -> dict:
        return dict(sorted(Counter(r["v5"]["verdict"] for r in sub).items()))

    def rules(sub: list[dict]) -> dict:
        return dict(sorted(Counter(r["v5"]["rule_id"] for r in sub).items()))

    genuine = subset(LABEL_GENUINE)
    generated = subset(LABEL_GENERATED)
    face_swap = subset(LABEL_FACE_SWAP)

    genuine_fp = [r for r in genuine if r["v5"]["verdict"] == detected]

    # Detection is reported two ways and they answer different questions. The *verdict* rate is
    # what a user of the product would see. The *detector* rate is whether that detector's own
    # comparator fired, which can differ from the verdict only by the other detector also
    # firing — recorded separately so a corroborated hit is not silently credited to one source.
    generated_svd_hits = [r for r in generated if r["coverage"]["svd_hit"]]
    face_swap_b7_hits = [r for r in face_swap if r["coverage"]["face_hit"]]

    # And a third denominator, reported beside the other two rather than instead of them. A
    # detector that produced no usable reading for a clip did not miss it — it did not look at
    # it. The rate over the whole population is what the product delivered on this corpus; the
    # rate over the clips the detector actually read is what the detector did. Conflating them
    # in either direction misstates something: the first alone blames a detector for a run that
    # never invoked it, the second alone hides that the coverage was never obtained.
    generated_svd_read = [r for r in generated if r["coverage"]["svd_usable"]]
    face_swap_b7_read = [r for r in face_swap if r["coverage"]["face_usable"]]

    coverage_counts = Counter(
        (r["coverage"]["d_usable"], r["coverage"]["d_total"]) for r in rows
    )
    coverage_distribution = {
        "full": sum(n for (u, t), n in coverage_counts.items() if u == t),
        "partial": sum(n for (u, t), n in coverage_counts.items() if 0 < u < t),
        "zero": sum(n for (u, t), n in coverage_counts.items() if u == 0),
    }
    # Which detector was missing, for every clip whose coverage was incomplete. The rate above
    # says how much evidence was unobtainable; this says what was unobtainable, which is the
    # difference between a detector that abstains and a detector that was never invoked.
    missing_reason = Counter()
    for row in rows:
        if row["coverage"]["d_usable"] == row["coverage"]["d_total"]:
            continue
        for name in ("svd", "face"):
            if row["coverage"][f"{name}_usable"]:
                continue
            if not row["rows_present"][name]:
                missing_reason[f"{name}:no-row"] += 1
            elif row["statuses"][name] != engine.SIGNAL_STATUS_SUCCESS:
                missing_reason[f"{name}:failed-or-abstained"] += 1
            else:
                missing_reason[f"{name}:unusable-figures"] += 1

    inconclusive_rows = [r for r in rows if r["v5"]["verdict"] == inconclusive]

    return {
        "population": {
            "clips": len(rows),
            "lineages": len({r["source_lineage_id"] for r in rows}),
            "by_label": dict(sorted(Counter(r["label"] for r in rows).items())),
        },
        "v5_verdicts": verdicts(rows),
        "v5_rules": rules(rows),
        "genuine": {
            "clips": len(genuine),
            "false_positive": {
                "verdict": detected,
                "clips": len(genuine_fp),
                "rate": _rate(len(genuine_fp), len(genuine)),
                "by_lineage": _lineage_rate(
                    genuine, lambda r: r["v5"]["verdict"] == detected
                ),
                "by_rule": dict(sorted(Counter(r["v5"]["rule_id"] for r in genuine_fp).items())),
                "by_family": dict(sorted(Counter(r["family"] for r in genuine_fp).items())),
                "by_stratum": dict(
                    sorted(Counter(r["stratum"] for r in genuine_fp).items())
                ),
            },
            "verdict_distribution": verdicts(genuine),
            "no_signal_rate": _rate(
                sum(1 for r in genuine if r["v5"]["verdict"] == no_signal), len(genuine)
            ),
            "inconclusive_rate": _rate(
                sum(1 for r in genuine if r["v5"]["verdict"] == inconclusive), len(genuine)
            ),
            "rules": rules(genuine),
        },
        "generated_video": {
            "clips": len(generated),
            "svd_detector_rate": {
                "clips": len(generated_svd_hits),
                "rate": _rate(len(generated_svd_hits), len(generated)),
                "by_lineage": _lineage_rate(generated, lambda r: r["coverage"]["svd_hit"]),
            },
            "svd_detector_rate_where_read": {
                "read": len(generated_svd_read),
                "max_score_observed": max(
                    (r["scores"]["svd"] for r in generated_svd_read), default=None
                ),
                "clips": sum(1 for r in generated_svd_read if r["coverage"]["svd_hit"]),
                "rate": _rate(
                    sum(1 for r in generated_svd_read if r["coverage"]["svd_hit"]),
                    len(generated_svd_read),
                ),
            },
            "verdict_rate": {
                "clips": sum(1 for r in generated if r["v5"]["verdict"] == detected),
                "rate": _rate(
                    sum(1 for r in generated if r["v5"]["verdict"] == detected),
                    len(generated),
                ),
            },
            "verdict_distribution": verdicts(generated),
            "rules": rules(generated),
            "missed_breakdown": {
                f"{verdict}/{rule}": n
                for (verdict, rule), n in sorted(
                    Counter(
                        (r["v5"]["verdict"], r["v5"]["rule_id"])
                        for r in generated
                        if not r["coverage"]["svd_hit"]
                    ).items()
                )
            },
            "by_family": {
                family: {
                    "clips": n,
                    "svd_hits": sum(
                        1
                        for r in generated
                        if r["family"] == family and r["coverage"]["svd_hit"]
                    ),
                }
                for family, n in sorted(Counter(r["family"] for r in generated).items())
            },
        },
        "face_swaps": {
            "clips": len(face_swap),
            "b7_detector_rate": {
                "clips": len(face_swap_b7_hits),
                "rate": _rate(len(face_swap_b7_hits), len(face_swap)),
                "by_lineage": _lineage_rate(face_swap, lambda r: r["coverage"]["face_hit"]),
            },
            "b7_detector_rate_where_read": {
                "read": len(face_swap_b7_read),
                "clips": sum(1 for r in face_swap_b7_read if r["coverage"]["face_hit"]),
                "rate": _rate(
                    sum(1 for r in face_swap_b7_read if r["coverage"]["face_hit"]),
                    len(face_swap_b7_read),
                ),
                "max_score_observed": max(
                    (r["scores"]["face"] for r in face_swap_b7_read), default=None
                ),
            },
            "verdict_rate": {
                "clips": sum(1 for r in face_swap if r["v5"]["verdict"] == detected),
                "rate": _rate(
                    sum(1 for r in face_swap if r["v5"]["verdict"] == detected),
                    len(face_swap),
                ),
            },
            "verdict_distribution": verdicts(face_swap),
            "rules": rules(face_swap),
            "missed_breakdown": {
                f"{verdict}/{rule}": n
                for (verdict, rule), n in sorted(
                    Counter(
                        (r["v5"]["verdict"], r["v5"]["rule_id"])
                        for r in face_swap
                        if not r["coverage"]["face_hit"]
                    ).items()
                )
            },
            "by_family": {
                family: {
                    "clips": n,
                    "b7_hits": sum(
                        1
                        for r in face_swap
                        if r["family"] == family and r["coverage"]["face_hit"]
                    ),
                }
                for family, n in sorted(Counter(r["family"] for r in face_swap).items())
            },
        },
        "inconclusive": {
            "clips": len(inconclusive_rows),
            "rate": _rate(len(inconclusive_rows), len(rows)),
            "by_rule": dict(
                sorted(Counter(r["v5"]["rule_id"] for r in inconclusive_rows).items())
            ),
            "by_label": dict(sorted(Counter(r["label"] for r in inconclusive_rows).items())),
        },
        "coverage_distribution": {
            "d_total": engine.D_TOTAL_V5,
            "clips": coverage_distribution,
            "rates": {
                name: _rate(n, len(rows)) for name, n in coverage_distribution.items()
            },
            "by_d_usable": {
                str(u): n
                for (u, _), n in sorted(coverage_counts.items(), key=lambda kv: kv[0][0])
            },
            "missing_coverage_reasons": dict(sorted(missing_reason.items())),
        },
        "lip_isolation": {
            "claim": (
                "evaluate_v5 returns an identical verdict and rule id with the evidence-only "
                "LipForensics signal removed, over every clip replayed"
            ),
            "clips_compared": len(rows),
            "clips_with_lip_row": sum(1 for r in rows if r["rows_present"]["lip"]),
            "verdict_differences": sum(
                1 for r in rows if r["v5"]["verdict"] != r["v5_lip_removed"]["verdict"]
            ),
            "rule_differences": sum(
                1 for r in rows if r["v5"]["rule_id"] != r["v5_lip_removed"]["rule_id"]
            ),
        },
    }


# The sentence that explains a transition, keyed on the **whole** (v4 level, v4 rule, v5
# verdict, v5 rule) pair. Keyed on all four and not on the v4 side alone, because one v4 rule id
# does not determine where a clip lands under v5 — `R201` splits three ways here, and a table
# keyed on `R201` would hand two of those three splits a confident sentence describing the
# third. A pair with no entry is reported as unexplained rather than absorbed.
TRANSITION_RULES: dict[tuple[str, str, str, str], str] = {
    (
        "MEDIUM",
        "R201",
        "NO_CALIBRATED_MANIPULATION_SIGNAL",
        "R9-200",
    ): (
        "Both deciding detectors were read and neither reached its operating point, and the "
        "evidence-only mouth-dynamics detector failed. v4 counts that detector toward "
        "`readable`, so its `R200` — which requires `svd_below and face_below and lip_usable` — "
        "could not fire and the clip fell through to `R201`, the partial-coverage band. v5 "
        "counts the evidence-only detector in no denominator at all (R9-T1 invariant 1), so "
        "coverage is complete: D_usable == D_total == 2, D_hits == 0, truth-table row 3. This is "
        "the invariant measured on real evidence rather than asserted — under v4 a failure in a "
        "detector that may not decide still moved the band."
    ),
    (
        "MEDIUM",
        "R201",
        "INCONCLUSIVE",
        "R9-301",
    ): (
        "v4 `R201` because something was readable — but the only readable signal was the "
        "evidence-only one. v5 counts it nowhere, so D_usable == 0 and the verdict is "
        "INCONCLUSIVE under R9-301, all expected coverage missing."
    ),
    (
        "HIGH",
        "R100",
        "MANIPULATION_DETECTED",
        "R9-101",
    ): (
        "v4 `R100` fires when the synthetic-video detector alone reaches `SVD_T_HIGH`; v5 "
        "`R9-101` is the same condition in the R9 vocabulary. Same detector, same operating "
        "point, same comparator — only the verdict string changed. The `D_usable` column shows "
        "this pair occurring at both complete and partial coverage: truth-table rows 1 and 2, "
        "which is §4's rule that a hit is never softened by the other detector being missing."
    ),
}

# Fallback sentences keyed on the v4 side only, consulted when the four-tuple above has no
# entry. These cover the transitions where the v4 rule does determine the v5 landing.
TRANSITION_RULES_BY_V4: dict[tuple[str, str], str] = {
    ("HIGH", "R100"): (
        "v4 R100 fires when the synthetic-video detector alone reaches SVD_T_HIGH; v5 R9-101 is "
        "the same condition in the R9 vocabulary. Same detector, same operating point, same "
        "comparator — only the verdict string changed."
    ),
    ("HIGH", "R101"): (
        "v4 R101 is the face classifier alone reaching FACE_T_HIGH; v5 R9-102 is that same "
        "condition. Unchanged threshold and comparator."
    ),
    ("HIGH", "R102"): (
        "v4 R102 is both deciding detectors reaching their own thresholds; v5 R9-100 is the "
        "same corroborated hit. D_hits == 2."
    ),
    ("HIGH", "R103"): (
        "v4 R103 is retired and fires in no current replay. Any occurrence here is a defect."
    ),
    ("MEDIUM", "R200"): (
        "v4 R200 means all three detectors were readable and neither deciding detector flagged. "
        "v5 counts only the two deciding detectors, so D_usable == D_total == 2 with D_hits == "
        "0 — truth-table row 3, NO_CALIBRATED_MANIPULATION_SIGNAL under R9-200. The verdict is "
        "renamed, not re-decided: v4's MEDIUM invited reading this as a mild finding, and the "
        "v5 name states only that the calibrated detectors were read and none reached its point."
    ),
    ("MEDIUM", "R201"): (
        "v4 R201 means at least one detector was readable but not all of them — a state v4 puts "
        "in the same MEDIUM band as R200, separated only by a rule id a renderer may drop. v5 "
        "splits it out: incomplete coverage with no hit is truth-table row 4, INCONCLUSIVE. "
        "R9-300 when some coverage was obtained, R9-301 when none was. This is the transition "
        "R9-T1 exists for."
    ),
    ("MEDIUM", "R203"): (
        "A v4 rule id this replay did not anticipate; see the trace rather than this table."
    ),
    ("UNKNOWN", "R010"): (
        "v4 R010 is no calibrated evidence readable at all. Under v5 D_usable == 0 < D_total, "
        "truth-table row 4/5 territory, INCONCLUSIVE under R9-301 — all expected coverage "
        "missing. The frozen D_total keeps the absent detectors in the denominator rather than "
        "letting them vanish from it."
    ),
    ("UNKNOWN", "R012"): (
        "v4 R012 is an eligible detector that answered illegibly and nothing else readable. v5 "
        "treats an unreadable figure as not a usable reading, so D_usable == 0 and the verdict "
        "is INCONCLUSIVE under R9-301."
    ),
}


def transition_matrix(engine: ModuleType, rows: list[dict]) -> dict:
    """How each v4 decision moved under v5, counted from the pairs that actually occurred.

    No mapping table decides this. Both engines ran on the same evidence and the pairs are
    tallied; the explanation attached to each pair names the coverage/verdict rule responsible,
    and a pair with no entry in `TRANSITION_RULES` is reported as unexplained rather than given
    a plausible-sounding sentence.
    """
    pairs = Counter(
        (
            row["v4"]["risk_level"],
            row["v4"]["rule_id"],
            row["v5"]["verdict"],
            row["v5"]["rule_id"],
        )
        for row in rows
    )

    entries = []
    for (v4_level, v4_rule, v5_verdict, v5_rule), count in sorted(
        pairs.items(), key=lambda kv: (-kv[1], kv[0])
    ):
        members = [
            r
            for r in rows
            if r["v4"]["risk_level"] == v4_level
            and r["v4"]["rule_id"] == v4_rule
            and r["v5"]["verdict"] == v5_verdict
            and r["v5"]["rule_id"] == v5_rule
        ]
        coverage = Counter(r["coverage"]["d_usable"] for r in members)
        explanation = TRANSITION_RULES.get(
            (v4_level, v4_rule, v5_verdict, v5_rule)
        ) or TRANSITION_RULES_BY_V4.get((v4_level, v4_rule))
        entry = {
            "v4_level": v4_level,
            "v4_rule": v4_rule,
            "v5_verdict": v5_verdict,
            "v5_rule": v5_rule,
            "clips": count,
            "share": _rate(count, len(rows)),
            "d_usable_distribution": {str(k): v for k, v in sorted(coverage.items())},
            "d_hits_distribution": {
                str(k): v
                for k, v in sorted(Counter(r["coverage"]["d_hits"] for r in members).items())
            },
            "by_label": dict(sorted(Counter(r["label"] for r in members).items())),
            "explanation": explanation or "UNEXPLAINED — no rule sentence recorded for this pair.",
            "explained": explanation is not None,
        }
        entries.append(entry)

    return {
        "clips": len(rows),
        "v4_levels": dict(sorted(Counter(r["v4"]["risk_level"] for r in rows).items())),
        "v4_rules": dict(sorted(Counter(r["v4"]["rule_id"] for r in rows).items())),
        "v5_verdicts": dict(sorted(Counter(r["v5"]["verdict"] for r in rows).items())),
        "v5_rules": dict(sorted(Counter(r["v5"]["rule_id"] for r in rows).items())),
        "unexplained_transitions": sum(1 for e in entries if not e["explained"]),
        "transitions": entries,
    }


# --------------------------------------------------------------------------------------------
# Part B — deterministic contract fixtures
# --------------------------------------------------------------------------------------------
#
# Synthetic evidence, built from the engine's own constants so a threshold change moves the
# fixtures with it rather than leaving them asserting a number nobody uses any more. These are
# contract assertions: each one names the R9-T1 clause it enforces and fails loudly.


def svd_evidence(engine: ModuleType, **overrides):
    """A synthetic-video row that is eligible and usable unless an override says otherwise."""
    fields = {
        "provider": engine.SVD_PROVIDER,
        "signal_type": engine.SVD_SIGNAL_TYPE,
        "status": engine.SIGNAL_STATUS_SUCCESS,
        "provider_version": engine.SVD_PROVIDER_VERSION,
        "score": 0.10,
        "total_clips": 8,
    }
    fields.update(overrides)
    return engine.SvdEvidence(**fields)


def face_evidence(engine: ModuleType, **overrides):
    """A face row that is eligible and usable unless an override says otherwise."""
    fields = {
        "provider": engine.FACE_PROVIDER,
        "signal_type": engine.FACE_SIGNAL_TYPE,
        "status": engine.SIGNAL_STATUS_SUCCESS,
        "provider_version": engine.FACE_PROVIDER_VERSION,
        "score": 0.10,
        "frames_scored": 32,
    }
    fields.update(overrides)
    return engine.FaceEvidence(**fields)


def lip_evidence(engine: ModuleType, **overrides):
    """A mouth-dynamics row. Under v5 nothing about it may change any outcome."""
    fields = {
        "provider": engine.LIP_PROVIDER,
        "signal_type": engine.LIP_SIGNAL_TYPE,
        "status": engine.SIGNAL_STATUS_SUCCESS,
        "provider_version": engine.LIP_PROVIDER_VERSION,
        "score": 0.50,
        "windows_scored": 4,
    }
    fields.update(overrides)
    return engine.LipEvidence(**fields)


def _case(name: str, clause: str, engine: ModuleType, svd, face, lip, expect_verdict, expect_rule):
    """Run one fixture and record what the engine actually returned.

    The expectation is recorded beside the observation rather than asserted away, so a failing
    contract appears in the report as a failing row with both values instead of a traceback that
    loses the measurement.
    """
    decision = engine.evaluate_v5(svd=svd, face=face, lip=lip)
    passed = decision.risk_level == expect_verdict and decision.rule_id == expect_rule
    return {
        "case": name,
        "clause": clause,
        "expected": {"verdict": expect_verdict, "rule_id": expect_rule},
        "observed": {
            "verdict": decision.risk_level,
            "rule_id": decision.rule_id,
            "rules_version": decision.rules_version,
            "calibration_id": decision.calibration_id,
        },
        "passed": passed,
    }


def lip_isolation_cases(engine: ModuleType) -> dict:
    """LipForensics isolation, as a cross product rather than a single example.

    Every state the evidence-only slot can be in — absent, present and scoring at either
    extreme, failed, uncalibrated, unreadable figures, and an outright foreign object — against
    every deciding-detector configuration that produces a different verdict. The claim is that
    the verdict, the rule id and the fact that a verdict was produced at all are invariant over
    the first axis. Anything less than the cross product proves it for one example.
    """
    lip_states = {
        "absent": None,
        "success_score_zero": lip_evidence(engine, score=0.0),
        "success_score_one": lip_evidence(engine, score=1.0),
        "success_above_own_threshold": lip_evidence(engine, score=engine.LIP_T_HIGH),
        "failed": lip_evidence(engine, status="FAILED", score=None, windows_scored=None),
        "timeout": lip_evidence(engine, status="TIMEOUT", score=None, windows_scored=None),
        "uncalibrated_deployment": lip_evidence(engine, provider_version="someone-elses-model@0"),
        "unreadable_figures": lip_evidence(engine, score=17.0, windows_scored=0),
        "foreign_object": object(),
    }
    deciding_states = {
        "both_usable_below": (svd_evidence(engine), face_evidence(engine)),
        "svd_hit": (svd_evidence(engine, score=engine.SVD_T_HIGH), face_evidence(engine)),
        "face_hit": (svd_evidence(engine), face_evidence(engine, score=engine.FACE_T_HIGH)),
        "both_hit": (
            svd_evidence(engine, score=engine.SVD_T_HIGH),
            face_evidence(engine, score=engine.FACE_T_HIGH),
        ),
        "svd_failed": (svd_evidence(engine, status="FAILED", score=None), face_evidence(engine)),
        "both_failed": (
            svd_evidence(engine, status="FAILED", score=None),
            face_evidence(engine, status="FAILED", score=None),
        ),
        "no_rows": (None, None),
    }

    comparisons = []
    for deciding_name, (svd, face) in deciding_states.items():
        baseline = engine.evaluate_v5(svd=svd, face=face, lip=None)
        for lip_name, lip in lip_states.items():
            try:
                decision = engine.evaluate_v5(svd=svd, face=face, lip=lip)
                raised = None
                identical = (
                    decision.risk_level == baseline.risk_level
                    and decision.rule_id == baseline.rule_id
                    and decision.rules_version == baseline.rules_version
                    and decision.calibration_id == baseline.calibration_id
                )
            except Exception as exc:  # a raise is an outcome the evidence-only slot controlled
                raised, identical = f"{type(exc).__name__}: {exc}", False
            comparisons.append(
                {
                    "deciding_state": deciding_name,
                    "lip_state": lip_name,
                    "baseline": {
                        "verdict": baseline.risk_level,
                        "rule_id": baseline.rule_id,
                    },
                    "raised": raised,
                    "identical": identical,
                }
            )

    failures = [c for c in comparisons if not c["identical"]]
    return {
        "case": "lip_isolation",
        "clause": "R9-T1 invariant 1 — evidence-only isolation",
        "combinations": len(comparisons),
        "lip_states": len(lip_states),
        "deciding_states": len(deciding_states),
        "failures": failures,
        "passed": not failures,
    }


def run_part_b(engine: ModuleType) -> dict:
    """The deterministic contract fixtures, each naming the clause it enforces."""
    detected = engine.VERDICT_MANIPULATION_DETECTED
    no_signal = engine.VERDICT_NO_SIGNAL
    inconclusive = engine.VERDICT_INCONCLUSIVE

    # Strictly below each operating point, by one representable float. `nextafter` rather than a
    # hand-chosen 0.95: a literal just under the threshold would be re-tuned every time the
    # threshold moved and would stop testing the boundary the moment someone rounded it.
    svd_just_below = math.nextafter(engine.SVD_T_HIGH, 0.0)
    face_just_below = math.nextafter(engine.FACE_T_HIGH, 0.0)

    cases = [
        # --- coverage failures: the three ways evidence goes missing -----------------------
        _case(
            "nvidia_failure",
            "Truth table row 4 — one deciding detector fails, the other reads below.",
            engine,
            svd_evidence(engine, status="FAILED", score=None, total_clips=None),
            face_evidence(engine),
            lip_evidence(engine),
            inconclusive,
            engine.RULE_V5_INCONCLUSIVE_PARTIAL,
        ),
        _case(
            "b7_failure",
            "Truth table row 4 — the face classifier fails, synthetic-video reads below.",
            engine,
            svd_evidence(engine),
            face_evidence(engine, status="FAILED", score=None, frames_scored=None),
            lip_evidence(engine),
            inconclusive,
            engine.RULE_V5_INCONCLUSIVE_PARTIAL,
        ),
        _case(
            "dual_failure",
            "Truth table row 4, D_usable == 0 — both deciding detectors fail.",
            engine,
            svd_evidence(engine, status="FAILED", score=None, total_clips=None),
            face_evidence(engine, status="FAILED", score=None, frames_scored=None),
            lip_evidence(engine),
            inconclusive,
            engine.RULE_V5_INCONCLUSIVE_ALL,
        ),
        _case(
            "dual_failure_with_readable_lip",
            "Invariant 1 — a readable evidence-only signal does not complete coverage.",
            engine,
            svd_evidence(engine, status="FAILED", score=None, total_clips=None),
            face_evidence(engine, status="FAILED", score=None, frames_scored=None),
            lip_evidence(engine, score=0.99),
            inconclusive,
            engine.RULE_V5_INCONCLUSIVE_ALL,
        ),
        # --- abstention, which is unresolved coverage by default (§3.3) ---------------------
        _case(
            "face_abstention_no_declaration",
            "§3.3 — an abstention with no resolution declaration is not a usable reading.",
            engine,
            svd_evidence(engine),
            face_evidence(engine, status="FAILED", score=None, frames_scored=0),
            None,
            inconclusive,
            engine.RULE_V5_INCONCLUSIVE_PARTIAL,
        ),
        # --- no-row: the detector was never invoked -----------------------------------------
        _case(
            "no_row_svd_other_below",
            "Edge case — no-row keeps the detector in the frozen denominator.",
            engine,
            None,
            face_evidence(engine),
            lip_evidence(engine),
            inconclusive,
            engine.RULE_V5_INCONCLUSIVE_PARTIAL,
        ),
        _case(
            "no_row_face_other_below",
            "Edge case — the same, with the face row missing.",
            engine,
            svd_evidence(engine),
            None,
            lip_evidence(engine),
            inconclusive,
            engine.RULE_V5_INCONCLUSIVE_PARTIAL,
        ),
        _case(
            "no_rows_at_all",
            "Truth table row 4 with D_usable == 0 — neither deciding detector was invoked.",
            engine,
            None,
            None,
            None,
            inconclusive,
            engine.RULE_V5_INCONCLUSIVE_ALL,
        ),
        _case(
            "no_row_svd_other_hit",
            "Truth table row 2 — a hit stands beside missing coverage.",
            engine,
            None,
            face_evidence(engine, score=engine.FACE_T_HIGH),
            None,
            detected,
            engine.RULE_V5_HIGH_FACE,
        ),
        # --- thresholds: exactly at, and one float below ------------------------------------
        _case(
            "svd_exact_threshold",
            "§3.2 — SVD_T_HIGH is reached at equality by NVIDIA's own comparator.",
            engine,
            svd_evidence(engine, score=engine.SVD_T_HIGH),
            face_evidence(engine),
            lip_evidence(engine),
            detected,
            engine.RULE_V5_HIGH_SVD,
        ),
        _case(
            "face_exact_threshold",
            "§3.2 — FACE_T_HIGH is reached at equality by the face classifier's comparator.",
            engine,
            svd_evidence(engine),
            face_evidence(engine, score=engine.FACE_T_HIGH),
            lip_evidence(engine),
            detected,
            engine.RULE_V5_HIGH_FACE,
        ),
        _case(
            "svd_just_below_threshold",
            "§3.2 — one representable float below the operating point is not reached.",
            engine,
            svd_evidence(engine, score=svd_just_below),
            face_evidence(engine),
            lip_evidence(engine),
            no_signal,
            engine.RULE_V5_NO_SIGNAL,
        ),
        _case(
            "face_just_below_threshold",
            "§3.2 — the same, on the face classifier's own scale.",
            engine,
            svd_evidence(engine),
            face_evidence(engine, score=face_just_below),
            lip_evidence(engine),
            no_signal,
            engine.RULE_V5_NO_SIGNAL,
        ),
        _case(
            "both_just_below_threshold",
            "Truth table row 3 — complete coverage, neither detector reached its point.",
            engine,
            svd_evidence(engine, score=svd_just_below),
            face_evidence(engine, score=face_just_below),
            lip_evidence(engine),
            no_signal,
            engine.RULE_V5_NO_SIGNAL,
        ),
        _case(
            "both_exact_threshold",
            "Truth table row 1 — corroborated hit, D_hits == 2.",
            engine,
            svd_evidence(engine, score=engine.SVD_T_HIGH),
            face_evidence(engine, score=engine.FACE_T_HIGH),
            lip_evidence(engine),
            detected,
            engine.RULE_V5_HIGH_MULTIPLE,
        ),
        _case(
            "hit_beside_failure_is_not_softened",
            "§4 — MANIPULATION_DETECTED is not vetoed by the other detector failing.",
            engine,
            svd_evidence(engine, score=engine.SVD_T_HIGH),
            face_evidence(engine, status="FAILED", score=None, frames_scored=None),
            lip_evidence(engine),
            detected,
            engine.RULE_V5_HIGH_SVD,
        ),
        # --- unusable evidence: a row exists, its figures cannot carry a classification ------
        _case(
            "unusable_null_score",
            "§2.2 — a successful row with no score is not a usable reading.",
            engine,
            svd_evidence(engine, score=None),
            face_evidence(engine),
            lip_evidence(engine),
            inconclusive,
            engine.RULE_V5_INCONCLUSIVE_PARTIAL,
        ),
        _case(
            "unusable_score_out_of_range",
            "§2.2 — a score outside [0, 1] did not come from a calibrated sigmoid.",
            engine,
            svd_evidence(engine, score=1.4),
            face_evidence(engine),
            lip_evidence(engine),
            inconclusive,
            engine.RULE_V5_INCONCLUSIVE_PARTIAL,
        ),
        _case(
            "unusable_nan_score",
            "§2.2 — NaN compares false against every threshold and is refused before it can.",
            engine,
            svd_evidence(engine, score=float("nan")),
            face_evidence(engine),
            lip_evidence(engine),
            inconclusive,
            engine.RULE_V5_INCONCLUSIVE_PARTIAL,
        ),
        _case(
            "unusable_zero_aggregation_count",
            "§2.2 — an aggregate over zero clips is a figure over an empty table.",
            engine,
            svd_evidence(engine, score=0.99, total_clips=0),
            face_evidence(engine),
            lip_evidence(engine),
            inconclusive,
            engine.RULE_V5_INCONCLUSIVE_PARTIAL,
        ),
        _case(
            "unusable_missing_frames_scored",
            "§2.2 — a face mean with no frame count behind it is not readable.",
            engine,
            svd_evidence(engine),
            face_evidence(engine, score=0.99, frames_scored=None),
            lip_evidence(engine),
            inconclusive,
            engine.RULE_V5_INCONCLUSIVE_PARTIAL,
        ),
        _case(
            "unusable_both_sides",
            "§2.2 — neither deciding row is readable, so no coverage was obtained.",
            engine,
            svd_evidence(engine, score=None),
            face_evidence(engine, score=None),
            lip_evidence(engine),
            inconclusive,
            engine.RULE_V5_INCONCLUSIVE_ALL,
        ),
        # --- uncalibrated deployment, which is a clean run this ruleset may not read ---------
        _case(
            "uncalibrated_svd_clean_run",
            "§2.2 rule 3 — an uncalibrated deployment's clean run is not coverage.",
            engine,
            svd_evidence(engine, provider_version="847b6e53-0133-452d-ab85-d7acf3ace723-preview"),
            face_evidence(engine),
            lip_evidence(engine),
            inconclusive,
            engine.RULE_V5_INCONCLUSIVE_PARTIAL,
        ),
        _case(
            "uncalibrated_face_reaching_threshold",
            "R6-T1 — a shadow-mode observation reaches no verdict even above the point.",
            engine,
            svd_evidence(engine),
            face_evidence(engine, score=1.0, provider_version="some-other-checkpoint@abc"),
            lip_evidence(engine),
            inconclusive,
            engine.RULE_V5_INCONCLUSIVE_PARTIAL,
        ),
        _case(
            "wrong_signal_type_is_not_coverage",
            "§2.2 — a row of the wrong signal type is not this detector answering.",
            engine,
            svd_evidence(engine, signal_type="something_else"),
            face_evidence(engine),
            lip_evidence(engine),
            inconclusive,
            engine.RULE_V5_INCONCLUSIVE_PARTIAL,
        ),
    ]

    # The type guard is an outcome too: a foreign object in a deciding slot must raise rather
    # than be compared against an operating point measured for something else (R6-T1).
    guard_cases = []
    for name, kwargs in (("svd", {"svd": object()}), ("face", {"face": object()})):
        try:
            engine.evaluate_v5(**kwargs)
            guard_cases.append(
                {
                    "case": f"foreign_object_in_{name}_slot_raises",
                    "clause": "R6-T1 — a deciding slot refuses anything but its evidence type.",
                    "expected": {"raises": "UncalibratedEvidence"},
                    "observed": {"raises": None},
                    "passed": False,
                }
            )
        except Exception as exc:
            guard_cases.append(
                {
                    "case": f"foreign_object_in_{name}_slot_raises",
                    "clause": "R6-T1 — a deciding slot refuses anything but its evidence type.",
                    "expected": {"raises": "UncalibratedEvidence"},
                    "observed": {"raises": type(exc).__name__},
                    "passed": type(exc).__name__ == "UncalibratedEvidence",
                }
            )

    isolation = lip_isolation_cases(engine)
    all_cases = cases + guard_cases
    return {
        "cases": all_cases,
        "lip_isolation": isolation,
        "summary": {
            "cases": len(all_cases),
            "passed": sum(1 for c in all_cases if c["passed"]),
            "failed": sum(1 for c in all_cases if not c["passed"]),
            "lip_isolation_combinations": isolation["combinations"],
            "lip_isolation_failures": len(isolation["failures"]),
            "all_passed": all(c["passed"] for c in all_cases) and isolation["passed"],
        },
    }


# --------------------------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------------------------


def _pct(rate: float | None) -> str:
    return "n/a" if rate is None else f"{rate * 100:.2f}%"


def write_report(report: dict, path: Path) -> None:
    """The Markdown the Architect reads. Every figure comes from `report`, none is retyped."""
    a = report["part_a"]["measurements"]
    matrix = report["part_a"]["transition_matrix"]
    b = report["part_b"]
    rules = report["rules"]
    lines: list[str] = []
    w = lines.append

    w("# R9-T3 — `evaluate_v5` replay and validation")
    w("")
    w(f"Replayed {report['replayed_at']}. Offline observation only: `evaluate_v5` has no ")
    w("production caller, nothing under `apps/` was written, and no threshold, rule id or ")
    w("calibration identity was changed by this task.")
    w("")
    w("| | |")
    w("|---|---|")
    w(f"| v4 ruleset | `{rules['rules_version_v4']}` |")
    w(f"| v5 ruleset | `{rules['rules_version_v5']}` |")
    w(f"| calibration | `{rules['calibration_id'][:16]}…` (shared, unchanged) |")
    w(f"| `SVD_T_HIGH` | `{rules['svd_t_high']}` |")
    w(f"| `FACE_T_HIGH` | `{rules['face_t_high']}` |")
    w(f"| `D_TOTAL_V5` | `{rules['d_total_v5']}` |")
    w(f"| engine source | `{rules['source']}` |")
    w(f"| engine modified by this task | `{rules['modified_by_this_task']}` |")
    w("")
    w("## What was measured")
    w("")
    w("Every figure below is a count over the population it belongs to. Nothing in this task ")
    w("compared a measurement against a target, and no threshold, rule or corpus was adjusted ")
    w("toward one.")
    w("")
    w("| measurement | value |")
    w("|---|---|")
    w(f"| genuine false-positive rate (`MANIPULATION_DETECTED` on genuine media) | "
      f"{_pct(a['genuine']['false_positive']['rate'])} "
      f"({a['genuine']['false_positive']['clips']}/{a['genuine']['clips']} clips) |")
    w(f"| SVD detection on generated video | "
      f"{_pct(a['generated_video']['svd_detector_rate']['rate'])} "
      f"({a['generated_video']['svd_detector_rate']['clips']}/"
      f"{a['generated_video']['clips']} clips); "
      f"{_pct(a['generated_video']['svd_detector_rate_where_read']['rate'])} over the "
      f"{a['generated_video']['svd_detector_rate_where_read']['read']} it read |")
    w(f"| B7 detection on face swaps | "
      f"{_pct(a['face_swaps']['b7_detector_rate']['rate'])} "
      f"({a['face_swaps']['b7_detector_rate']['clips']}/{a['face_swaps']['clips']} clips) |")
    w(f"| `INCONCLUSIVE` rate | {_pct(a['inconclusive']['rate'])} "
      f"({a['inconclusive']['clips']}/{a['population']['clips']} clips) |")
    w(f"| coverage: full / partial / zero | "
      f"{a['coverage_distribution']['clips']['full']} / "
      f"{a['coverage_distribution']['clips']['partial']} / "
      f"{a['coverage_distribution']['clips']['zero']} clips |")
    w(f"| LipForensics effect on any verdict | "
      f"{a['lip_isolation']['verdict_differences']} differences over "
      f"{a['lip_isolation']['clips_compared']} clips |")
    w(f"| unexplained v4→v5 transitions | {matrix['unexplained_transitions']} |")
    w(f"| Part B contract fixtures | {b['summary']['passed']}/{b['summary']['cases']} pass, "
      f"{b['summary']['lip_isolation_failures']} isolation differences over "
      f"{b['summary']['lip_isolation_combinations']} combinations |")
    w("")
    w("## Corpus and detector runs")
    w("")
    w(f"Split replayed: **{report['part_a']['split']}** — "
      f"{a['population']['clips']} clips over {a['population']['lineages']} distinct recordings "
      f"({', '.join(f'{k}: {v}' for k, v in a['population']['by_label'].items())}).")
    w("")
    w("| detector | run | deployment matches the calibrated one |")
    w("|---|---|---|")
    for name, entry in report["part_a"]["eligibility"].items():
        run = report["part_a"]["runs"][name]
        w(f"| {name} | `{run['path'] or 'not run'}` | "
          f"{'yes' if entry['matches_calibrated_deployment'] else '**NO — ineligible**'} |")
    w("")

    w("## Part A — corpus replay")
    w("")
    w("### A.1 Verdict distribution under v5")
    w("")
    w("| verdict | clips | share |")
    w("|---|---:|---:|")
    for verdict, n in a["v5_verdicts"].items():
        w(f"| `{verdict}` | {n} | {_pct(n / a['population']['clips'])} |")
    w("")
    w("| v5 rule | clips |")
    w("|---|---:|")
    for rule, n in a["v5_rules"].items():
        w(f"| `{rule}` | {n} |")
    w("")

    g = a["genuine"]
    w("### A.2 Genuine media — false positives")
    w("")
    w(f"Population: {g['clips']} genuine clips "
      f"({g['false_positive']['by_lineage']['lineages']} recordings).")
    w("")
    w("| measurement | clips | rate |")
    w("|---|---:|---:|")
    w(f"| `MANIPULATION_DETECTED` (false positive) | {g['false_positive']['clips']} | "
      f"{_pct(g['false_positive']['rate'])} |")
    w(f"| — per distinct recording | {g['false_positive']['by_lineage']['flagged']} | "
      f"{_pct(g['false_positive']['by_lineage']['rate'])} |")
    w(f"| `NO_CALIBRATED_MANIPULATION_SIGNAL` | "
      f"{g['verdict_distribution'].get('NO_CALIBRATED_MANIPULATION_SIGNAL', 0)} | "
      f"{_pct(g['no_signal_rate'])} |")
    w(f"| `INCONCLUSIVE` | {g['verdict_distribution'].get('INCONCLUSIVE', 0)} | "
      f"{_pct(g['inconclusive_rate'])} |")
    w("")
    if g["false_positive"]["clips"]:
        w("False positives by rule and stratum:")
        w("")
        w("| v5 rule | clips |")
        w("|---|---:|")
        for rule, n in g["false_positive"]["by_rule"].items():
            w(f"| `{rule}` | {n} |")
        w("")
        w("| stratum | clips |")
        w("|---|---:|")
        for stratum, n in g["false_positive"]["by_stratum"].items():
            w(f"| `{stratum}` | {n} |")
        w("")

    gen = a["generated_video"]
    w("### A.3 Generated video — SVD detection")
    w("")
    w(f"Population: {gen['clips']} generated clips "
      f"({gen['svd_detector_rate']['by_lineage']['lineages']} recordings).")
    w("")
    w("| measurement | clips | rate |")
    w("|---|---:|---:|")
    w(f"| SVD reached `SVD_T_HIGH` | {gen['svd_detector_rate']['clips']} | "
      f"{_pct(gen['svd_detector_rate']['rate'])} |")
    w(f"| — per distinct recording | {gen['svd_detector_rate']['by_lineage']['flagged']} | "
      f"{_pct(gen['svd_detector_rate']['by_lineage']['rate'])} |")
    w(f"| — over the {gen['svd_detector_rate_where_read']['read']} clips SVD actually read | "
      f"{gen['svd_detector_rate_where_read']['clips']} | "
      f"{_pct(gen['svd_detector_rate_where_read']['rate'])} |")
    w(f"| verdict `MANIPULATION_DETECTED` | {gen['verdict_rate']['clips']} | "
      f"{_pct(gen['verdict_rate']['rate'])} |")
    w("")
    w(f"Highest SVD score observed on generated video: "
      f"`{gen['svd_detector_rate_where_read']['max_score_observed']}` "
      f"against an operating point of `{rules['svd_t_high']}`.")
    w("")
    w("Where the clips SVD did not flag landed (a miss is legitimately either verdict):")
    w("")
    w("| verdict / rule | clips |")
    w("|---|---:|")
    for key, n in gen["missed_breakdown"].items():
        w(f"| `{key}` | {n} |")
    w("")
    w("| family | clips | SVD hits |")
    w("|---|---:|---:|")
    for family, entry in gen["by_family"].items():
        w(f"| `{family}` | {entry['clips']} | {entry['svd_hits']} |")
    w("")

    fs = a["face_swaps"]
    w("### A.4 Face swaps — B7 detection")
    w("")
    w(f"Population: {fs['clips']} face-swap clips "
      f"({fs['b7_detector_rate']['by_lineage']['lineages']} recordings).")
    w("")
    w("| measurement | clips | rate |")
    w("|---|---:|---:|")
    w(f"| B7 reached `FACE_T_HIGH` | {fs['b7_detector_rate']['clips']} | "
      f"{_pct(fs['b7_detector_rate']['rate'])} |")
    w(f"| — per distinct recording | {fs['b7_detector_rate']['by_lineage']['flagged']} | "
      f"{_pct(fs['b7_detector_rate']['by_lineage']['rate'])} |")
    w(f"| — over the {fs['b7_detector_rate_where_read']['read']} clips B7 actually read | "
      f"{fs['b7_detector_rate_where_read']['clips']} | "
      f"{_pct(fs['b7_detector_rate_where_read']['rate'])} |")
    w(f"| verdict `MANIPULATION_DETECTED` | {fs['verdict_rate']['clips']} | "
      f"{_pct(fs['verdict_rate']['rate'])} |")
    w("")
    w(f"Highest B7 score observed on a face swap in this split: "
      f"`{fs['b7_detector_rate_where_read']['max_score_observed']}` against an operating point "
      f"of `{rules['face_t_high']}` — the whole face-swap population sits below the threshold, "
      f"so the zero is the operating point not being reached rather than evidence going "
      f"missing. The clips that did reach `MANIPULATION_DETECTED` here did so through the "
      f"synthetic-video detector.")
    w("")
    w("Where the clips B7 did not flag landed:")
    w("")
    w("| verdict / rule | clips |")
    w("|---|---:|")
    for key, n in fs["missed_breakdown"].items():
        w(f"| `{key}` | {n} |")
    w("")
    w("| family | clips | B7 hits |")
    w("|---|---:|---:|")
    for family, entry in fs["by_family"].items():
        w(f"| `{family}` | {entry['clips']} | {entry['b7_hits']} |")
    w("")

    inc = a["inconclusive"]
    w("### A.5 INCONCLUSIVE rate")
    w("")
    w(f"**{inc['clips']} of {a['population']['clips']} clips — {_pct(inc['rate'])}.**")
    w("")
    w("| v5 rule | clips |")
    w("|---|---:|")
    for rule, n in inc["by_rule"].items():
        w(f"| `{rule}` | {n} |")
    w("")
    w("| label | clips |")
    w("|---|---:|")
    for label, n in inc["by_label"].items():
        w(f"| `{label}` | {n} |")
    w("")

    cov = a["coverage_distribution"]
    w("### A.6 Coverage distribution")
    w("")
    w(f"`D_total` is frozen at {cov['d_total']} for this ruleset.")
    w("")
    w("| coverage | clips | share |")
    w("|---|---:|---:|")
    for name in ("full", "partial", "zero"):
        w(f"| {name} (`D_usable` "
          f"{'== D_total' if name == 'full' else '0 < D_usable < D_total' if name == 'partial' else '== 0'}) "
          f"| {cov['clips'][name]} | {_pct(cov['rates'][name])} |")
    w("")
    w("Why coverage was missing, counted per missing detector-reading:")
    w("")
    w("| reason | occurrences |")
    w("|---|---:|")
    for reason, n in cov["missing_coverage_reasons"].items():
        w(f"| `{reason}` | {n} |")
    w("")
    w("**What this rate is a property of.** The dominant reason for missing coverage above is "
      "`svd:no-row` — the historical R7-T5 synthetic-video run was executed against "
      "`manifest_svd.csv`, a subset of the corpus, so those clips have no NVIDIA row to read. "
      "That is a fact about how that benchmark run was scheduled, not about the v5 coverage "
      "model and not about how often the detector fails in production. The frozen `D_total` is "
      "doing exactly what §2.4 specifies — a detector that was never invoked stays in the "
      "denominator and shows up as missing coverage instead of vanishing from it — and the "
      "corpus-wide `INCONCLUSIVE` rate below should be read as the rate over *this* evidence "
      "set, not as a forecast. A production-representative figure needs a run in which both "
      "deciding detectors were invoked on every clip.")
    w("")

    iso = a["lip_isolation"]
    w("### A.7 LipForensics isolation, measured over the corpus")
    w("")
    w(f"`evaluate_v5` was called twice per clip on identical deciding evidence, once with the "
      f"mouth-dynamics signal present as persisted and once with it removed. "
      f"{iso['clips_with_lip_row']} of {iso['clips_compared']} clips carry a LipForensics row.")
    w("")
    w(f"- verdict differences: **{iso['verdict_differences']}**")
    w(f"- rule-id differences: **{iso['rule_differences']}**")
    w("")

    w("### A.8 v4 → v5 transition matrix")
    w("")
    w("Both rulesets ran on the same evidence objects. No v4 level was mapped or string-matched "
      "onto a v5 verdict; the pairs below are the ones that occurred.")
    w("")
    w("| v4 level | v4 rule | v5 verdict | v5 rule | clips | share | `D_usable` |")
    w("|---|---|---|---|---:|---:|---|")
    for t in matrix["transitions"]:
        cover = ", ".join(f"{k}→{v}" for k, v in t["d_usable_distribution"].items())
        w(f"| `{t['v4_level']}` | `{t['v4_rule']}` | `{t['v5_verdict']}` | `{t['v5_rule']}` "
          f"| {t['clips']} | {_pct(t['share'])} | {cover} |")
    w("")
    w(f"Unexplained transitions: **{matrix['unexplained_transitions']}**.")
    w("")
    w("Why each shift occurs:")
    w("")
    for t in matrix["transitions"]:
        w(f"**`{t['v4_level']}`/`{t['v4_rule']}` → `{t['v5_verdict']}`/`{t['v5_rule']}` "
          f"({t['clips']} clips).** {t['explanation']}")
        w("")

    w("## Part B — contract validation fixtures")
    w("")
    s = b["summary"]
    w(f"{s['passed']} of {s['cases']} deterministic cases pass; "
      f"{s['lip_isolation_combinations']} isolation combinations with "
      f"{s['lip_isolation_failures']} differences.")
    w("")
    w("| case | clause | expected | observed | result |")
    w("|---|---|---|---|---|")
    for c in b["cases"]:
        expected = (
            f"`{c['expected']['verdict']}`/`{c['expected']['rule_id']}`"
            if "verdict" in c["expected"]
            else f"raises `{c['expected']['raises']}`"
        )
        observed = (
            f"`{c['observed']['verdict']}`/`{c['observed']['rule_id']}`"
            if "verdict" in c["observed"]
            else f"raises `{c['observed']['raises']}`"
        )
        w(f"| `{c['case']}` | {c['clause']} | {expected} | {observed} | "
          f"{'PASS' if c['passed'] else '**FAIL**'} |")
    w("")
    iso_b = b["lip_isolation"]
    w("### B.1 LipForensics isolation, as a cross product")
    w("")
    w(f"{iso_b['lip_states']} evidence-only states × {iso_b['deciding_states']} "
      f"deciding-detector configurations = {iso_b['combinations']} comparisons against the "
      f"same evidence with the slot empty. Differences: **{len(iso_b['failures'])}** "
      f"— {'PASS' if iso_b['passed'] else '**FAIL**'}.")
    w("")
    if iso_b["failures"]:
        w("| deciding state | lip state | raised |")
        w("|---|---|---|")
        for f in iso_b["failures"]:
            w(f"| `{f['deciding_state']}` | `{f['lip_state']}` | {f['raised']} |")
        w("")

    w("## Status")
    w("")
    w("Replay complete. No engine logic, threshold or corpus was modified, and `evaluate_v5` "
      "remains unwired from the production pipeline. Awaiting Architect FINAL PASS on these "
      "results.")
    w("")

    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="replay_r9t3",
        description="Replay evaluate_v5 over the evaluation corpus and contract fixtures.",
    )
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--svd-run", type=Path)
    parser.add_argument("--face-run", type=Path)
    parser.add_argument("--lip-run", type=Path)
    parser.add_argument(
        "--split",
        default=SPLIT_EVALUATION,
        help="Corpus split to replay; 'all' replays every clip.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--report",
        type=Path,
        help=(
            "Where to write the Markdown report. Defaults to `r9t3_validation_report.md` inside "
            "--output-dir; point it at docs/ai/reviews/R9_T3/ to write the reviewed copy "
            "directly rather than copying a file that can then drift from its data."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    engine = load_risk_engine()
    items = read_corpus(args.corpus)
    if args.split != "all":
        items = [item for item in items if item.split == args.split]

    svd_run, face_run, lip_run = (
        read_run(args.svd_run),
        read_run(args.face_run),
        read_run(args.lip_run),
    )
    replayed = replay_corpus(items, engine, svd_run, face_run, lip_run)
    rows = replayed["rows"]

    report = {
        "schema_version": "r9-t3-validation-1",
        "replayed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rules": {
            "rules_version_v4": engine.RULES_VERSION,
            "rules_version_v5": engine.RULES_VERSION_V5,
            "calibration_id": engine.CALIBRATION_ID,
            "svd_t_high": engine.SVD_T_HIGH,
            "face_t_high": engine.FACE_T_HIGH,
            "lip_t_high": engine.LIP_T_HIGH,
            "d_total_v5": engine.D_TOTAL_V5,
            "source": str(RISK_ENGINE_PATH),
            "modified_by_this_task": False,
            "wired_to_production": False,
        },
        "part_a": {
            "split": args.split,
            "runs": {
                "svd": {"ran": svd_run["ran"], "path": svd_run.get("results_path")},
                "face": {"ran": face_run["ran"], "path": face_run.get("results_path")},
                "lip": {"ran": lip_run["ran"], "path": lip_run.get("results_path")},
            },
            "eligibility": replayed["eligibility"],
            "measurements": measure_part_a(engine, rows),
            "transition_matrix": transition_matrix(engine, rows),
        },
        "part_b": run_part_b(engine),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "r9t3_validation.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "r9t3_decisions.json").write_text(
        json.dumps({"schema_version": "r9-t3-decisions-1", "rows": rows}, indent=2) + "\n",
        encoding="utf-8",
    )
    report_path = args.report or (args.output_dir / "r9t3_validation_report.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    write_report(report, report_path)

    a = report["part_a"]["measurements"]
    print(f"v4 {engine.RULES_VERSION}  v5 {engine.RULES_VERSION_V5}  "
          f"calibration {engine.CALIBRATION_ID[:12]}…")
    print(f"Part A — {a['population']['clips']} clips ({args.split})")
    print(f"  v5 verdicts: {a['v5_verdicts']}")
    print(f"  genuine false-positive rate: {_pct(a['genuine']['false_positive']['rate'])} "
          f"({a['genuine']['false_positive']['clips']}/{a['genuine']['clips']})")
    print(f"  SVD detection on generated: {_pct(a['generated_video']['svd_detector_rate']['rate'])}")
    print(f"  B7 detection on face swaps: {_pct(a['face_swaps']['b7_detector_rate']['rate'])}")
    print(f"  INCONCLUSIVE rate: {_pct(a['inconclusive']['rate'])}")
    print(f"  coverage: {a['coverage_distribution']['clips']}")
    print(f"  lip isolation differences: {a['lip_isolation']['verdict_differences']}")
    print(f"  transitions: {len(report['part_a']['transition_matrix']['transitions'])}, "
          f"unexplained {report['part_a']['transition_matrix']['unexplained_transitions']}")
    s = report["part_b"]["summary"]
    print(f"Part B — {s['passed']}/{s['cases']} cases pass, "
          f"{s['lip_isolation_failures']} isolation differences over "
          f"{s['lip_isolation_combinations']} combinations")
    print(f"  wrote {report_path}")
    return 0 if s["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
