# R17-T5 — Preregistered Challenger Specification

Frozen before any challenger or diagnostic outcome is computed. `scripts/eval/r17_t5_challenger.py` and `scripts/eval/r17_t5_c3_diagnostic.py` apply this document as written and decide nothing new. Both refuse to run unless the sha256 of this file equals the value passed on their command line. No rule, threshold, population or decision criterion below may be changed after an outcome is seen (stop S3).

Authority: `r17_t5_implementation_plan.md` rev 3 §3–§5 (Architect PASS; D1–D4 approved) and its task packet.

## 0. Prior visibility (disclosure)

- The R17-T4 record (`scripts/eval/baselines/r17_t4_case_causes.json`, committed in `0c4f2cf`) already holds the stored `synthetic_video` and `face_manipulation` values for every case, including the 5 cases where `synthetic_video` succeeded. The plan itself quotes the two `lip_forensics` values above `LIP_T_HIGH` (AI-018 0.349, AI-024 0.883).
- The implementer did not compute any C1 or C2 outcome before this file was frozen. No threshold below is chosen on this dataset: every threshold is a production constant read from the committed `apps/api/app/risk_engine.py`. Prior visibility therefore cannot have tuned an operating point. It is disclosed because the T4 record makes several C1/C2 inputs public, so C1/C2 are not blind.

## 1. Population, inputs and integrity

- **Population:** the 45 R17-T3 baseline cases (30 `AI`, 15 `REAL`), 6 source clusters. Read only from the sealed T3 Phase A directory and the anchored Phase B `results.json`. Truth and `source_cluster_id` come from those records (as frozen by R17-T2).
- **Face-absence reason:** read from the committed T4 record, field `face_manipulation.metadata.error` (provenance `production_committed_row`). It is not sealed T3 evidence, and this is stated wherever it is used.
- **Integrity before a single case is read** (any mismatch is a refusal, stop S2):
  1. The T4 closure commit `0c4f2cf9f75fa4ef33f5cebd6616eda71d98b47b` exists, and the tracked T4 record is byte-identical to the one it holds.
  2. The tracked T3 anchor is byte-identical to the one in the T3 closure commit `74e91e76f567d280947af882f5724e77a8a1cfbf`.
  3. The Phase A seal recomputes to the anchor's `phase_a_seal_root`.
  4. `sha256(results.json)` equals the anchor's `results_json_sha256`.
  5. The dataset `manifest.json` and `FREEZE.json` equal the anchor's `dataset_anchors`.
  6. This file hashes to the value given on the command line.
- **C0 reproduction gate:** the committed `evaluate_v5`, loaded by path and unchanged, run on the sealed `synthetic_video` and `face_manipulation` fields, must reproduce the stored `risk_level` and `risk_rule_id` for 45/45 cases. Any mismatch is a refusal.

## 2. Shared definitions

- **Usable `synthetic_video`:** `is_eligible_svd(e) and is_usable_svd(e)` from the committed module, on the sealed fields `provider, signal_type, status, provider_version, score, total_clips`.
- **SVD hit:** usable and `svd_threshold_reached(e)`, i.e. `score >= SVD_T_HIGH`. `SVD_T_HIGH` is read from the committed module and must equal `0.9550971388816833` (reported rounded as `0.9551`). Any other value is a refusal.
- **Usable `lip_forensics`:** `is_eligible_lip(e) and is_usable_lip(e)` from the committed module, on the sealed fields `provider, signal_type, status, provider_version, score, windows_scored`.
- **Lip hit:** usable and `score >= LIP_T_HIGH`. `LIP_T_HIGH` is read from the committed module and must equal `0.22962537594139576`. The committed module has no lip comparator (v5 compares lip against nothing), so the evaluator states the `>=` comparison itself, with the same equality convention as the two native comparators.
- **Faceless case:** `face_manipulation.status == FAILED` and `face_manipulation.metadata.error == "FaceDetectorNoFaceFound"`.
- **Final states and confusion:** exactly as frozen in `r17_t2_scoring_freeze.md` §2–§3. A challenger produces one of the three canonical verdicts per case. `MANIPULATION_DETECTED_AUXILIARY` (C2) is scored as `positive` and additionally flagged `auxiliary: true`.

## 3. Candidates

### C0 — Baseline (`r9-v5.0.0`, unchanged)
The stored production verdict, reproduced by the gate in §1. C0 is the comparison point for every candidate. It is never replaced.

### C1 — Faceless Coverage Fallback (threshold unchanged)
For each case, in order:
1. If C0 is not `INCONCLUSIVE`: C1 = C0.
2. Else, if the case is faceless and `synthetic_video` is usable:
   - SVD hit → `MANIPULATION_DETECTED`;
   - otherwise → `NO_CALIBRATED_MANIPULATION_SIGNAL`.
3. Else: C1 = C0 (`INCONCLUSIVE`).

**No threshold tuning.** `SVD_T_HIGH` is the production constant. No other or adjusted threshold is computed or reported.

**Structural property (derived from the rules, before any data).** Under `evaluate_v5` any usable SVD hit fires `R9-101` (`MANIPULATION_DETECTED`) whatever the face signal holds. So a C0 `INCONCLUSIVE` case with usable SVD has an SVD score below `SVD_T_HIGH`, and step 2 can only ever yield `NO_CALIBRATED_MANIPULATION_SIGNAL`. Consequences, stated before data: C1 adds no positive, so `TP(C1) = TP(C0)` and `FP(C1) = FP(C0)`. Every case C1 changes moves from `abstention` to `negative`: a `TN` on `REAL`, an `FN` on `AI`. C1 coverage gain on `AI` cases is a gain in decided misses, not in recall. The evaluator checks this property on the data and refuses if it is violated.

### C2 — LipForensics Auxiliary (exploratory only, promotion-ineligible)
Applied on top of a visual path X ∈ {C0, C1}, giving two variants `C2@C0` and `C2@C1`. For each case:
1. If X is not `INCONCLUSIVE`: C2@X = X.
2. Else, if lip hit → `MANIPULATION_DETECTED_AUXILIARY` (scored `positive`, `auxiliary: true`).
3. Else: C2@X = X.

C2 has no safety metric. The 15 REAL controls contain no human face, so lip-based FPR is untestable here. Any auxiliary positive on a REAL case is still counted and reported as FP. No safety or generalisation claim is permitted.

### C3 — NVIDIA Availability Diagnostic (separate external diagnostic)
- **Population:** the cases whose sealed `synthetic_video.status` is `FAILED` or `TIMEOUT` in the T3 record (40 by the T4 inventory). The evaluator refuses if this count is not 40.
- **Input:** the frozen clip of each case (`~/deepguard-corpus/r17_t3_work/phase_a_input/<blind_id>.mp4`). Its sha256 must equal both the frozen `clip_sha256` and the sealed `media_sha256s`, because T3 analysed the clip un-normalised (`was_normalized = false`). A mismatch is not submitted and counts as `hash_blocked`.
- **Procedure (fixed):** one attempt per case, no retries. Sequential in `case_id` order. Each call is the committed `app.nvidia_video.analyze_video`, unchanged, with the production deadline from `app.limits`. It runs inside an ephemeral `docker compose run --rm --no-deps` container of the `api` image, with the clip directory mounted read-only. No running service is restarted or modified. There is a fixed 30 s pause between the end of one call and the start of the next.
- **Per-case outcome (exactly one, first match wins):**

| Outcome | Condition |
|---|---|
| `hash_blocked` | the clip failed the sha256 gate and was not submitted |
| `usable_response` | the call returned, and `SvdEvidence("nvidia", "synthetic_video", "SUCCESS", function_id, probability, total_clips)` is eligible and usable by the committed predicates |
| `ineligible_response` | the call returned, but that evidence is not eligible or not usable (e.g. a different function id) |
| `input_format_incompatibility` | `NvidiaProviderError` whose gRPC status is `INVALID_ARGUMENT`, `FAILED_PRECONDITION`, `OUT_OF_RANGE` or `UNIMPLEMENTED` |
| `provider_unavailable` | `NvidiaProviderUnavailable` (`UNAVAILABLE`, `RESOURCE_EXHAUSTED`) |
| `provider_timeout` | `NvidiaProviderTimeout` |
| `authentication_error` | `NvidiaAuthenticationError` |
| `invalid_response` | `NvidiaProviderInvalidResponse` |
| `local_file_error` | `NvidiaLocalFileError` |
| `other_provider_error` | any other `NvidiaProviderError` (e.g. `INTERNAL`) or other exception |

- **Provenance and storage:** every C3 record carries `provenance: "diagnostic_revalidation"` and its own UTC timestamps. It is written only under `~/deepguard-corpus/r17_t5_diagnostic/`. Nothing is written into, or merged with, the sealed T3 Phase A/B directories, the T3 anchor or the T4 record. C3 never replaces C0 as the baseline.
- **Counterfactual (diagnostic only):** for each of the 45 cases, the counterfactual SVD evidence is the sealed one when it was `SUCCESS` in T3. Otherwise it is the C3 response when that is `usable_response`, and otherwise the sealed failure. C0′ is the committed `evaluate_v5` on that evidence with the sealed face evidence. C1′ and C2′ apply §3 unchanged on top of C0′. Every counterfactual figure is labelled `counterfactual — diagnostic_revalidation, not a measurement of production`.

## 4. Metrics

- **Formulas, denominators and intervals:** exactly `r17_t2_scoring_freeze.md` §4 (exact two-sided 95% Clopper-Pearson; zero denominator → N/A, never 0%). Computed by the committed `metrics()` of `scripts/eval/r17_t2_score.py`.
- **Levels:** case level (N = 45) and source-cluster level, never pooled, with the fixed label "`<K>` source clusters, not independent, not generalisable". `<K>` is read from `FREEZE.json`.
- **Variants measured:** C0, C1, C2@C0 and C2@C1. For C3: C0′, C1′, C2′@C0′ and C2′@C1′.
- **Per-candidate figures used by §5:**
  - `faceless_cases`: the number of faceless cases.
  - `faceless_decided(X)`: the faceless cases whose X state is `positive` or `negative`.
  - `faceless_coverage_gain = faceless_decided(C1) − faceless_decided(C0)`.
  - `FP(C1)`: REAL cases whose C1 state is `positive`, over **all 15 REAL**. This is the strictest reading. The faceless-usable subset is reported beside it.
  - `usable_real_denominator`: faceless REAL cases with usable SVD.
  - `c2_delta_tp(X) = TP(C2@X) − TP(X)`, and `c2_aux_fp(X)`: auxiliary positives on REAL.
  - `c3_usable`, `c3_format_incompatible`: the counts of `usable_response` and `input_format_incompatibility` over the 40.

## 5. Exhaustive decision rules (applied in the order listed; the first match wins)

### C1
1. `INCONCLUSIVE` — `usable_real_denominator == 0`.
2. `FAIL-SAFETY` — `FP(C1) > 0`.
3. `PASS` — `faceless_coverage_gain >= 4` (and `FP(C1) == 0`, by step 2).
4. `FAIL-BENEFIT` — otherwise (`FP(C1) == 0` and `faceless_coverage_gain < 4`).

A C1 `PASS` makes C1 eligible **only for R17-T6 further promotion review**. The 15 faceless NASA REAL controls are **not sufficient by themselves for production promotion**. T6 must preserve this limitation and may require broader REAL validation. A `PASS` is never reported as a recall improvement, because of the §3 structural property.

### C2
`DESCRIPTIVE ONLY`. Report `c2_delta_tp` for C2@C0 and C2@C1 exactly, plus any auxiliary FP. **Ineligible for promotion in this phase.** The verdict is informative only, for future data collection.

### C3
1. `DIAGNOSTIC PASS` — `c3_usable >= 35` (supports a transient, provider-availability-related failure pattern).
2. `DIAGNOSTIC FAIL` — `c3_format_incompatible >= 10` (supports persistent clip incompatibility).
3. `INCONCLUSIVE` — every other outcome, including a provider still broadly unavailable.

Steps 1 and 2 cannot both hold (35 + 10 > 40). **Ineligible for promotion** (diagnostic counterfactual only).

## 6. Dataset limitation notice (mandatory in every output)

> The 15 REAL controls are a single narrow technical domain (NASA Artemis II spacecraft cameras, 2 source clusters) with zero human faces. Zero false positives on this set is not general-population FPR or safety evidence. C2's lip-based FPR cannot be tested here at all. Cluster-level results come from 6 source clusters, which are not independent and not generalisable.

## 7. Outputs

- `scripts/eval/baselines/r17_t5_challenger_results.json` (tracked). It holds the verification record, this file's sha256, per-case states for every variant, case- and cluster-level metrics, the §5 verdicts, and a separate `c3_diagnostic` section carrying `provenance: "diagnostic_revalidation"` and the sha256 of the diagnostic output it was read from.
- `~/deepguard-corpus/r17_t5_diagnostic/` (untracked, outside the repository): the raw C3 per-case records and the run log.
- `docs/ai/r17_t5_challenger_report.md`: the report.
