# R17-T2 — Frozen Scoring Mapping

Frozen before any R17 result exists. `scripts/eval/r17_t2_score.py` applies this document as written and decides nothing new. The T2 closure commit makes it immutable for R17-T3.

Authority: `r17_t2_implementation_plan.md` rev 3 §5, with the Architect's cluster ruling.

## 1. Population and truth

- The population is the frozen R17-T1 dataset, after its trust anchors check out (`scripts/eval/specs/r17_t2_expected_digests.json`).
- **N = 45:** 30 `AI` and 15 `REAL`.
- `truth` is read only from the verified `manifest.json`:
  - `AI` is the synthetic class.
  - `REAL` is the genuine class.
- `source_cluster_id` is read from the verified `manifest.json`.
- `blind_map.json` contributes only the `blind_id → case_id` join. It must agree with the cluster id in `manifest.json`.

## 2. One final state per case

Each case gets exactly one final state. The rules are checked in order and the first one that matches wins.

| Order | Final state | Rule |
|---|---|---|
| 1 | `hash_blocked` | The clip's sha256 differs from its frozen value. The clip is not submitted. Any `hash_blocked` case also blocks T3. |
| 2 | `operational_failure` | The case was submitted, but one of these holds: `status` ≠ `completed`, a decision timeout, or an upload, transport or provider/system failure that left no completed analysis. |
| 3 | `protocol_violation` | `status = completed`, but `risk_rules_version` ≠ `r9-v5.0.0`, or `risk_level` is missing or not one of the three values below. |
| 4 | `abstention` | `status = completed`, `risk_level = INCONCLUSIVE` |
| 5 | `positive` | `status = completed`, `risk_level = MANIPULATION_DETECTED` |
| 5 | `negative` | `status = completed`, `risk_level = NO_CALIBRATED_MANIPULATION_SIGNAL` |

Notes on the states:
- A case that never reached the runner, with no record at all, is `operational_failure`.
- An enrichment timeout after a completed decision is not an operational failure. Enrichment is recorded separately as `enrichment_state`.

**Invariant.** The scorer checks these and refuses to emit results if any fails:

- `hash_blocked + operational_failure + protocol_violation + abstention + positive + negative = 45`
- The same sum is 30 over the `AI` cases and 15 over the `REAL` cases.

**The four non-decided states are never merged.** `abstention`, `operational_failure`, `protocol_violation` and `hash_blocked` are each counted and reported on their own line.

## 3. Confusion

Confusion applies only to `positive` and `negative`. Every other final state has `confusion = null`.

| truth | final state | confusion |
|---|---|---|
| AI | positive | TP |
| AI | negative | FN |
| REAL | positive | FP |
| REAL | negative | TN |

## 4. Metrics and denominators

| Metric | Formula | Denominator |
|---|---|---|
| TPR (recall) | TP / (TP+FN) | decided AI cases |
| FNR | FN / (TP+FN) | decided AI cases |
| FPR | FP / (FP+TN) | decided REAL cases |
| TNR | TN / (FP+TN) | decided REAL cases |
| coverage | (TP+FN+FP+TN) / N | all cases |
| coverage_AI | (TP+FN) / N_AI | AI cases |
| coverage_REAL | (FP+TN) / N_REAL | REAL cases |
| abstention_rate | abstention / N | all cases |
| abstention_rate_AI | abstention_AI / N_AI | AI cases |
| abstention_rate_REAL | abstention_REAL / N_REAL | REAL cases |
| operational_failure_rate | operational_failure / N | all cases |
| protocol_violation_rate | protocol_violation / N | all cases |
| hash_blocked_rate | hash_blocked / N | all cases |
| **secondary_recall_abstention_as_miss** | TP / (TP+FN+abstention_AI) | decided AI cases plus AI abstentions |

- The secondary recall leaves `operational_failure`, `protocol_violation` and `hash_blocked` **out** of its denominator. They are reported on their own lines.
- The secondary recall is descriptive. It is always labelled **"secondary, abstention-as-miss"** and never replaces the primary TPR.
- **Intervals:** exact two-sided 95% Clopper-Pearson, with each tail at 0.975, using `eval/stats.py`.
- **Zero denominator:** the rate and its interval are `N/A`. That is JSON `null` and CSV `N/A`. It is never shown as `0%`.
- **k = 0 with n > 0:** the observed rate is shown with its Clopper-Pearson interval, never as a bare `0%`.

## 5. Reporting levels (never pooled)

1. **Case level:** N = 45, with the formulas of §4.
2. **Source-cluster level:** the formulas of §4, applied to the cases of each `source_cluster_id`.
   - A REAL-only cluster has no TPR, FNR, coverage_AI or secondary recall. Each is `N/A`.
   - An AI-only cluster has no FPR, TNR or coverage_REAL. Each is `N/A`.
   - **Fixed label:** "`<K>` source clusters, not independent, not generalisable."
   - `<K>` is read from `counts.source_clusters` in the verified `FREEZE.json` (6 at freeze) and never hard-coded.
   - The scorer checks that `<K>` equals the number of distinct `source_cluster_id` values in the joined rows, and refuses if they differ.

Breakdown by `generator_or_source` is descriptive only and derives no rule.

## 6. Order of verification in the scorer

Any failure aborts before a single verdict is read.

1. Check the live `FREEZE.json`, `manifest.csv` and `manifest.json` against `scripts/eval/specs/r17_t2_expected_digests.json`.
2. Verify the Phase A seal against the `--expected-seal-root` argument (plan §4.5).
3. Check that the hashes of this document and of `r17_t2_result_schema.json`, as recorded in Phase A `environment.json`, equal the files in the repository.
4. Only then read the records and join `blind_map.json` and `truth`. This step also checks that every clip_sha256 in the execution manifest equals the frozen `clip_sha256` of the case it maps to.

## 7. Smoke population (harness check only, not a measurement)

The T2 smoke run uses 1–2 non-dataset synthetic clips. Its `blind_map.json` declares `population = "smoke"` and carries its own placeholder `truth` and cluster ids.

The rules are the same as for the dataset, with three differences:
- The invariant is checked against the declared smoke counts, not 45/30/15.
- `<K>` is the number of distinct smoke clusters.
- Every output is labelled `SMOKE — harness verification, not a measurement`.

No smoke case id may equal an R17 case id. Step 1 of §6 still runs against the live dataset files.
