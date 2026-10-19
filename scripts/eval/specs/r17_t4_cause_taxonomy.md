# R17-T4 — Frozen Cause Taxonomy

Frozen before any R17-T3 case is classified. `scripts/eval/r17_t4_analyze.py` applies this document as written and decides nothing new. The analyser refuses to run unless the sha256 of this file equals the value passed on its command line. The T4 closure commit makes the file immutable for R17-T5.

Authority: `r17_t4_implementation_plan.md` rev 2 §4, §5 and §8 (Architect PASS WITH FIXES; D1–D4 approved).

## 1. Population and inputs

- **Population:** the 45 R17-T3 baseline cases (30 `AI`, 15 `REAL`). They are read only from the sealed T3 Phase A directory and the anchored Phase B `results.json`. Truth is joined, because T4 is not blind.
- **Integrity before anything is read:**
  - The T3 closure commit exists.
  - The tracked T3 anchor at `HEAD` is byte-identical to the anchor in that commit.
  - The Phase A seal recomputes to `phase_a_seal_root` from the tracked anchor.
  - `sha256(results.json)` equals `results_json_sha256` from the tracked anchor.
  - The dataset `manifest.json` and `FREEZE.json` equal the anchor's `dataset_anchors`.
  - Any mismatch is a refusal (stop S2).
- **No new measurement.** The analyser runs no detector and submits no clip. It computes no alternative verdict and no alternative threshold.

## 2. Evidence provenance

Every evidence pointer is `{analysis_id, source, field, value, provenance}`. It carries exactly one provenance label:

| Label | What it is | Sealed by T3 |
|---|---|---|
| `sealed_t3_record` | A file inside the sealed Phase A directory (`cases/<blind_id>/record.json`, `analysis.json`), or a row of the anchored Phase B `results.json` | Yes |
| `production_committed_row` | A row of production `analysis_signals` for one of the 45 analysis ids. It is read once with a read-only query and exported with its query time and the export's sha256. | No |
| `worker_log_supplemental` | An api or worker log line from the analysis time window | No |

- `production_committed_row` and `worker_log_supplemental` are **never** treated as sealed T3 evidence.
- A primary cause whose deciding pointer is not `sealed_t3_record` is flagged `rests_on_non_sealed_evidence: true`.
- When the evidence a class needs is absent or rotated, that class is not assigned. If no other class applies, the case is `unclassifiable_from_stored_evidence`. Nothing is forced.

## 3. Decision path (read from the committed ruleset code)

The **decision path** of a fired rule is the set of signals its condition reads in `apps/api/app/risk_engine.py::evaluate_v5` (`r9-v5.0.0`).

| Rule | Verdict | Condition | Decision path |
|---|---|---|---|
| R9-200 | `NO_CALIBRATED_MANIPULATION_SIGNAL` | `D_usable == D_TOTAL_V5 (2)` and `D_hits == 0` | `synthetic_video`, `face_manipulation` |
| R9-300 | `INCONCLUSIVE` | `0 < D_usable < 2` and `D_hits == 0` | `synthetic_video`, `face_manipulation` |
| R9-301 | `INCONCLUSIVE` | `D_usable == 0` (so `D_hits == 0`) | `synthetic_video`, `face_manipulation` |

- `lip_forensics` is accepted as a parameter of `evaluate_v5` but never read.
- `audio_authenticity`, `active_speaker` and `provenance` are not parameters of `evaluate_v5`.
- No signal outside the decision path can be a primary cause of class 2 or 3.

**Usable reading.** This is per decision-path signal, with the same predicates as `is_eligible_*` and `is_usable_*`:

- the provider, the signal type, `status == SUCCESS` and `provider_version` are exactly the calibrated deployment;
- the score is a finite value in [0, 1];
- the count is a positive int: `total_clips` for `synthetic_video`, `frames_scored` for `face_manipulation`.

All of these fields are read from the sealed `analysis.json`. A decision-path signal that is not a usable reading is **unusable**.

## 4. Failure reason categories

A FAILED or TIMEOUT signal's reason is the `metadata.error` exception class name of its `production_committed_row`. Status `TIMEOUT` is `runtime` whatever its error. The category comes from the class's definition in the committed code. It does not come from which classes happened to occur.

| Category | Exception classes (committed code) |
|---|---|
| `input` (property of the clip: no face, no tracked face, no audio stream, no usable speech or input, undecodable media) | `FaceDetectorNoFaceFound`, `FaceDetectorMediaError`, `LipForensicsNoTrackedFace`, `LipForensicsMediaError`, `EffortNoFaceDetected`, `EffortMediaError`, `NvidiaActiveSpeakerInputError`, `SpeakerDiarizationAudioError`, `AudioDetectorAudioError` |
| `runtime` (server, deployment or provider side: unavailable or quota-exhausted provider, authentication, timeout, invalid response, model unavailable, inference error, local file or pipeline fault) | `NvidiaProviderError`, `NvidiaProviderUnavailable`, `NvidiaAuthenticationError`, `NvidiaProviderTimeout`, `NvidiaProviderInvalidResponse`, `NvidiaLocalFileError`, `NvidiaActiveSpeakerProviderError`, `NvidiaActiveSpeakerUnavailable`, `NvidiaActiveSpeakerAuthenticationError`, `NvidiaActiveSpeakerTimeout`, `NvidiaActiveSpeakerInvalidResponse`, `NvidiaActiveSpeakerLocalFileError`, `FaceDetectorModelUnavailable`, `FaceDetectorInferenceError`, `LipForensicsModelUnavailable`, `LipForensicsInferenceError`, `EffortModelUnavailable`, `EffortInferenceError`, `SpeakerDiarizationUnavailable`, `SpeakerDiarizationModelError`, `AudioDetectorModelUnavailable`, `AudioDetectorInferenceError`, `AudioDetectorInputError` |
| `unknown` | Any other class name, a missing `metadata.error`, or no `production_committed_row` |

- `AudioDetectorInputError` is `runtime` because the code defines it as a pipeline-shape mistake (sample rate, channels, encoding), not a property of the clip.
- `NvidiaProviderUnavailable` covers both gRPC `UNAVAILABLE` and `RESOURCE_EXHAUSTED`. The stored row does not tell them apart.

## 5. Primary cause: exactly one per case, first match wins

| Order | Primary cause | Assigned when |
|---|---|---|
| 1 | `decision_contradicts_truth` | A canonical decision was produced (R9-1xx or R9-200) and it disagrees with truth: R9-200 on `AI`, or R9-1xx on `REAL` |
| 2 | `detector_input_ineligible` | The fired rule is an abstention (R9-300 or R9-301), and at least one **decision-path** signal is FAILED or TIMEOUT with reason category `input` |
| 3 | `detector_runtime_failure` | The fired rule is an abstention, and at least one **decision-path** signal is FAILED or TIMEOUT with reason category `runtime` |
| 4 | `rule_requires_missing_signal` | The fired rule is an abstention, at least one decision-path signal is usable, and at least one decision-path signal is unusable (FAILED or absent) with no `input` or `runtime` reason available. The rule abstains because it requires the missing signal. |
| 5 | `rule_uncertain_band` | The fired rule is an abstention, every decision-path signal is usable, and the rule's condition defines an uncertain or inconclusive score band containing the scores |
| 6 | `rule_conflicting_signals` | The fired rule is an abstention, every decision-path signal is usable, and the rule's condition abstains because the usable signals disagree |
| 7 | `unclassifiable_from_stored_evidence` | None of the above can be established from stored evidence. This includes a needed `production_committed_row` or `worker_log_supplemental` that is absent or rotated. |

- **Decision-path restriction (classes 2 and 3).** Only signals in the fired rule's decision path (§3) are considered. A FAILED signal outside the decision path never yields class 2 or 3 as primary, whatever its reason.
- **Order is binding.** When both decision-path signals failed, one with `input` and one with `runtime`, the primary cause is `detector_input_ineligible` and `detector_runtime_failure` is recorded as contributing. The report states that the order decided this.
- **Classes 5 and 6 under `r9-v5.0.0`.** `evaluate_v5` defines no score band between its thresholds and no conflict rule. Two usable readings with no hit fire R9-200, and any hit fires R9-1xx. Classes 5 and 6 are therefore not reachable under this ruleset. They stay in the order so the taxonomy is unchanged across rulesets. The analyser never assigns them under `r9-v5.0.0`.
- **A correct canonical decision** (R9-200 on `REAL`, R9-1xx on `AI`) is not a failure. It has no primary failure cause and would be recorded as `none_decision_agrees_with_truth`. The R17-T3 baseline has none.

## 6. Contributing causes

These are zero or more per case, from the same class names, each with evidence pointers:

- `detector_input_ineligible` or `detector_runtime_failure`, for every FAILED or TIMEOUT signal not already used as the primary cause. Off-path signals are included: `lip_forensics`, `audio_authenticity`, `active_speaker`, `provenance`, and the enrichment-only `face_forgery` (`effort`) component. Each such contribution records `on_decision_path: false`.
- `unclassifiable_from_stored_evidence`, for a FAILED signal whose reason category is `unknown`.
- Contributing counts may sum above 45. Primary counts sum to exactly 45.

## 7. Aggregation-rule attribution (every abstention)

For each abstention the analyser records:

- the fired rule id;
- its condition (§3);
- each decision-path signal's status and usability;
- exactly one attribution:

| Attribution | Assigned when |
|---|---|
| `attributable_to_detector_evidence` | At least one decision-path signal is unusable (FAILED, TIMEOUT, absent or unreadable) in the sealed record |
| `attributable_to_aggregation_rule` | Every decision-path signal is a usable reading in the sealed record and the rule still abstained |
| `indeterminate` | The sealed record cannot establish usability for a decision-path signal (missing field) |

- Signals outside the decision path are never used for the attribution.
- The attribution is static. It computes no alternative verdict.

## 8. Reproduction check (D2, optional)

`evaluate_v5` is a pure function. Its module imports only `math` and `dataclasses`. It may be run, unchanged, in an ephemeral `docker compose run --rm --no-deps` container of the api image, on evidence built only from the sealed `analysis.json` fields in §3. The check passes when the reproduced `risk_level` and `rule_id` equal the stored ones for N / N cases. Before it runs:

- the image's `app/risk_engine.py` sha256 is compared with the committed file;
- any difference skips the check, and the reason is recorded.

## 9. Breakdowns and labels

The breakdowns are descriptive:

- by signal;
- by provider;
- by `generator_or_source` (from the anchored `manifest.json`);
- by `source_cluster_id`;
- `AI` vs `REAL`.

Cluster tables carry the label `<K> source clusters, not independent, not generalisable`, with K from the anchored `FREEZE.json`. Nothing in the T4 output is a threshold, ruleset, detector or provider recommendation.
