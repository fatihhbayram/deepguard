# R17-T6 — Promotion Decision Record

Formal decision record for Phase R17. It decides one question: does any R17 challenger enter production? It adds no code, threshold, rule or forensic meaning. It only reads frozen evidence.

Authority: `r17_t6_implementation_plan.md` rev 2 (Architect PASS; D1–D3 approved) and its task packet.

**Ruling: Option A — NO PROMOTION / CLOSE WITHOUT RULESET CHANGE.** C1, C2 and C3 are rejected for production. `r9-v5.0.0` is preserved intact.

## 0. Evidence basis (verified before drafting)

Each item was checked byte for byte at drafting time. A mismatch would have stopped the task (stop S2).

| Evidence | Location | sha256 / value |
|---|---|---|
| T5 closure commit | `git log -1` | `68092a2ec390226be5166197060f8f2024c9d39c` |
| T3 baseline anchor | `scripts/eval/baselines/r17_t3_baseline_anchor.json` (closed in `74e91e76…`) | `e5209d030a979a876827910fd9fe2a8e04b39b1388dcc9a03a9c369c5a9e30de` |
| T4 case causes | `scripts/eval/baselines/r17_t4_case_causes.json` (closed in `0c4f2cf9…`) | `80633792fb9274c3157a9d6c75265c475761804196d296dee68ef6715ef68f00` |
| T5 challenger spec | `scripts/eval/specs/r17_t5_challenger_spec.md` | `b1dc9991e80cdcebcd974517b071ed519d93f7a859002f5c81a284ab5f454f3e` |
| T5 challenger results | `scripts/eval/baselines/r17_t5_challenger_results.json` | `479677630662c5b49874ab84ada445e823acb1b2b50b7fc7156050208756380f` |
| C3 diagnostic records | `deepguard-corpus/r17_t5_diagnostic/c3_records.json` (untracked corpus) | `2bb32470f7c0f575c5218701d03ea06620e61260c150664a45e5ee91969fc748` |
| Dataset `FREEZE.json` | `deepguard-corpus/r17-ai-visual-v1/` | `c57220b755afe67a09fbc34f90f3ab7ff06e748e75ef8985de27bc07471d4c0d` |
| Dataset `manifest.json` | `deepguard-corpus/r17-ai-visual-v1/` | `5ce7d8363696b5370edd54f923ef816b60f671be2fa1ec9a67d0b488684d73e7` |
| Phase A seal root | T3 anchor | `6fc23d408dc4da918fc85968acc364b700d1d770114d718735e2862687d2f579` |
| Production thresholds | `apps/api/app/risk_engine.py` (unchanged since `68092a2`) | `SVD_T_HIGH` 0.9550971388816833, `LIP_T_HIGH` 0.22962537594139576 |

`scripts/eval/`, `apps/` and `docker-compose.yml` are identical to the T5 closure commit.

> **Dataset limitation (mandatory).** The 15 REAL controls are a single narrow technical domain: NASA Artemis II spacecraft cameras, 2 source clusters, zero human faces. Zero false positives on this set is **not** general-population FPR or safety evidence. Lip-based FPR cannot be tested here at all. Cluster-level results come from 6 source clusters, which are not independent and not generalisable.

## 1. Evidence table (case level, N = 45; exact two-sided 95% Clopper-Pearson)

| Variant | Source | positive | negative | abstention | TP | FN | FP | TN | TPR | FPR | Promotion status (T5) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| C0 `r9-v5.0.0` | T3 anchor, reproduced 45/45 in T5 | 0 | 1 | 44 | 0 | 1 | 0 | 0 | 0/1 [0, 0.975] | N/A | baseline |
| C1 Faceless Coverage Fallback | T5 factual | 0 | 5 | 40 | 0 | 1 | 0 | 4 | 0/1 [0, 0.975] | 0/4 [0, 0.602] | PASS; eligible only for T6 review |
| C2@C0 LipForensics Auxiliary | T5 factual | 2 | 1 | 42 | 2 | 1 | 0 | 0 | 2/3 [0.094, 0.992] | N/A | DESCRIPTIVE ONLY; ineligible |
| C2@C1 | T5 factual | 2 | 5 | 38 | 2 | 1 | 0 | 4 | 2/3 [0.094, 0.992] | 0/4 [0, 0.602] | DESCRIPTIVE ONLY; ineligible |
| C3 NVIDIA Availability Diagnostic | T5 `diagnostic_revalidation` | — | — | — | — | — | — | — | — | — | INCONCLUSIVE; ineligible |

C3 outcome: 5/40 usable responses, 35/40 `RESOURCE_EXHAUSTED` ("exceeded rate limit") from submission #6 on, 0/40 input/format incompatibility. All 5 recovered SVD scores are below `SVD_T_HIGH` (AI-001 0.755, AI-003 0.021, AI-004 0.185, AI-005 0.173, AI-006 0.124).

C3 counterfactual (diagnostic, not a measurement of production): C1′ decides 6 cases, AI-004 and AI-005 (CL-SORA, faceless) and 4 REAL. Case-level FN goes from 4 under C0′ to 6 under C1′, so C1′ adds **+2 FN**. FP stays 0.

## 2. Candidates not reviewed against the gates

- **C2 — rejected.** Preregistered as promotion-ineligible (T5 spec §3). Its +2 TP (AI-018, AI-024) is descriptive. The REAL controls contain no human face, so its lip-based FPR on genuine media is untestable on this dataset.
- **C3 — rejected.** Preregistered as a diagnostic only and promotion-ineligible. Its verdict is INCONCLUSIVE. The T5 diagnostic **proved** `RESOURCE_EXHAUSTED` rate limiting during C3. T3's historical failure pattern (a time-correlated `NvidiaProviderUnavailable` run, T4) is **consistent with** this. T3 stored only the failure class and no gRPC code, so the same root cause is **not proven retroactively** for T3.

## 3. C1 against the four promotion gates

| Gate | Requirement | Evidence | Status |
|---|---|---|---|
| **1. Metric efficacy** | Did C1 meet its preregistered primary metric? | T5 results `figures`, `verdicts.C1` | **PASS.** Faceless coverage gain 4 (threshold ≥ 4). All 4 are REAL abstentions turned TN (REAL-001, -003, -005, -010; all CL-NASA-A2-SAW1). FP 0. By construction C1 adds no positive, so it brings **no AI recall gain**. |
| **2. Safety & FPR robustness** | Does zero FPR generalise beyond the test set? | T1 dataset composition, T5 `dataset_limitation` | **FAIL / DEFICIENT.** The 4 TNs come from one source cluster in a single faceless NASA spacecraft-camera domain. FPR 0/4 has a 95% upper bound of **60.2%**. General-population safety cannot be established. |
| **3. Asymmetric error risk** | Does C1 turn undetected AI into decided negatives? | T5 spec §3 structural property; C3 counterfactual | **CONCERN.** Every case C1 changes goes from abstention to `NO_CALIBRATED_MANIPULATION_SIGNAL`: a TN on REAL, an FN on AI. With the provider answering, the C3 counterfactual shows this on AI: AI-004 and AI-005 (Sora, faceless, SVD below `SVD_T_HIGH`) become decided misses, +2 FN. Where false clearance is more harmful than abstention, C1's error asymmetry is unfavorable. |
| **4. Forensic semantic impact** | Implementation complexity versus forensic impact | Rule architecture (`evaluate_v5`), forensic verdict contract | **CONCERN / NOT SUFFICIENT FOR PROMOTION.** Implementation complexity is low (one fallback branch). The forensic semantic impact is material: faceless media with sub-threshold SVD moves from `INCONCLUSIVE` to `NO_CALIBRATED_MANIPULATION_SIGNAL`. That changes what the system tells a user about a whole class of media. |

C1 passes only the narrow metric gate. It fails the safety gate and raises concerns on the error-asymmetry and forensic-semantics gates. A pass on Gate 1 alone does not qualify a candidate for promotion.

## 4. Ruling

- **Option A — NO PROMOTION / CLOSE WITHOUT RULESET CHANGE.** Approved by Architect decision D1.
- C1, C2 and C3 are rejected for production deployment.
- `r9-v5.0.0` is preserved intact. No new ruleset is defined. No threshold, rule, DB schema, DB data or forensic semantics change.
- The R17 exit criterion "a promotion decision (promotion, or an explicit 'no promotion')" is satisfied by this record.

## 5. R17 closure statements

1. No challenger is promoted into production.
2. `r9-v5.0.0` remains the active production ruleset.
3. AI-visual recall is still insufficiently validated/improved.
4. Provider reliability remains an unresolved operational concern.
5. Broader REAL safety validation is required before reconsidering C1.

## 6. Entry criteria for any future recall-improvement work

These are preconditions, not a schedule. They name no roadmap phase. Any later work that reconsiders C1 or a similar candidate must first meet all three:

1. **Broader multi-domain REAL validation.** REAL controls from several independent source domains, including footage with human faces and the faceless content C1 would decide. The set must be large enough for the FPR upper bound to support a safety claim. Validation must also cover the faceless AI clips that C1 would turn into negatives.
2. **Provider/quota-stable `synthetic_video` access.** A `synthetic_video` provider path whose rate limits or quotas cannot cut a full evaluation short. Availability must be shown before any recall measurement relies on it.
3. **Additional AI recall validation.** Evidence on why AI clips score below `SVD_T_HIGH` even when the provider answers (5/5 in C3), measured on frozen, truth-evidenced AI clips beyond the 30 here. Any threshold or detector change must create a new calibration and ruleset context. It must never rewrite historical analyses.

Two separate problems carry forward from T5. Criterion 2 targets provider availability. Criterion 3 targets sub-threshold SVD scores on AI clips. Fixing one does not fix the other.
