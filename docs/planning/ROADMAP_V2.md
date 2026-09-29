# DeepGuard Post-MVP Roadmap (v2)

## ROADMAP_V2 EXECUTION RULE

Follow `ROADMAP_V2.md` strictly in sequence.

Work on exactly one task at a time:
- PM defines the next task within the current roadmap phase.
- Architect reviews the implementation plan.
- PM produces the executable Claude Code task packet.
- Claude implements only that approved task.
- Claude stops for Architect review.
- After Architect approval, the task is committed.
- Only then may PM prepare the next task.

Do NOT:
- ask Claude to automatically continue to the next task
- skip tasks or phases
- re-plan ROADMAP_V2 unless explicitly requested
- bundle multiple roadmap tasks into one implementation
- start the next task before the previous task has been reviewed and committed

Example:
R1-T1 → Architect review → commit → PM prepares R1-T2.

---
## Detector Roadmap Principles

- NVIDIA SVD remains an independent baseline signal.
- New detectors must be benchmarked before integration.
- Detector outputs remain independent evidence.
- No arbitrary score averaging.
- No Fake/Real interpretation without explicit support.
- New detectors must not influence Risk Engine until calibrated.
- Preserve rule/calibration traceability.
- Prefer Rule of Three before generic provider/plugin abstractions.
- Shadow mode comes later for experimental models.

---

# R1 — Production Readiness

## Objective
Harden the MVP infrastructure for live deployment, establishing robust identity, access control, and operational stability.

## Scope
**Identity, Access & Role Management:**
- Minimal authentication/session architecture
- Backend-enforced USER / ADMIN authorization
- Analysis ownership and isolation
- Role-aware navigation with login/logout
- Separate USER dashboard and ADMIN dashboard
- Report authorization
- Maintain existing public API-key isolation (from P9)

**Production Hardening:**
- Deployment manifests and secrets/configuration management
- Observability and logging
- Worker schema readiness (gate startup against schema state)
- Execution, resource, and time limits
- Database backup and restore processes
- API documentation
- Production smoke testing

## Non-goals
- Adding new forensic models or detectors.
- Advanced billing or organizational/team hierarchies.
- Real-time/live stream analysis.

## Dependencies
- Frozen MVP (P0–P10) Baseline.

## Exit criteria
- Users can log in, view their isolated dashboard, and log out.
- Admins can access an admin dashboard.
- API keys from P9 continue to function correctly and remain isolated.
- The system can be deployed to a production environment with proper secrets management.
- Worker nodes respect schema readiness and resource limits.
- Backup/restore processes are documented and verified.
- Production smoke tests pass.

## Carry-forward items
- Global/cross-key download concurrency limit.
- Web build-time font fetch (vendor fonts locally).
- Report Page Visual Polish.

---

# R2 — Detector Benchmark Framework

## Objective
Establish a rigorous, reproducible framework for evaluating new deepfake and manipulation detectors before they enter the product pipeline.

## Scope
- Offline benchmarking pipeline.
- Dataset ingestion (real, synthetic, face-swapped, and an optional audio anti-spoof evaluation track).
- Accuracy, false positive, and false negative measurement.
- Performance (latency/memory) profiling.
- Output of reproducible evaluation metrics and artifacts.

## Non-goals
- Establishing calibration thresholds or rules (this belongs to R4).
- Integrating the benchmarked models into the live Risk Engine yet.
- Building a UI for the benchmark tool (CLI/scripts are sufficient).

## Dependencies
- R1 Production Readiness.

## Exit criteria
- A standardized benchmark script/tool exists.
- The tool can run a proposed model against a known test dataset and output reproducible evaluation metrics.
- The framework enforces the principle that new detectors must be benchmarked before integration.

## Carry-forward items
- Shadow mode infrastructure (deferred to R6).

---

# R3 — Face Manipulation Detector

## Objective
Introduce a specialized detector targeting face-swap and localized facial manipulations to complement the NVIDIA SVD baseline.

## Scope
- Select and integrate a proven face manipulation model.
- Process video frames specifically for facial anomalies.
- Output independent facial manipulation evidence.

## Non-goals
- Altering the Risk Engine immediately (the new detector runs independently first).
- Generic provider abstractions (Wait until Rule of Three).

## Dependencies
- R2 Detector Benchmark Framework (the model must pass the benchmark first).

## Exit criteria
- The face manipulation detector processes videos async via the worker pipeline.
- Facial manipulation evidence is persisted independently.
- The dashboard/report displays the new independent evidence without conflating it with NVIDIA SVD.

## Carry-forward items
- Risk Engine integration (deferred to R4).

---

# R4 — Calibration + Risk Engine v2

## Objective
Upgrade the Risk Engine to deterministically synthesize signals from validated detectors (e.g., NVIDIA SVD, Face Manipulation) into a calibrated risk tier.

## Scope
- Establish calibration thresholds and rules based on R2 evaluation metrics.
- Update risk rules to handle multi-detector disagreements.
- Preserve full rule and calibration traceability.

## Non-goals
- Assuming AASIST contributes to Risk Engine v2; it may influence risk only if a dedicated R2 benchmark establishes supported semantics and thresholds.
- Arbitrary score averaging (Risk is rule-based, not a math average).
- Emitting a generic "Fake/Real" verdict without explicit support.

## Dependencies
- R3 Face Manipulation Detector.

## Exit criteria
- Risk Engine v2 emits HIGH, MEDIUM, or UNKNOWN based on multi-source evidence.
- The specific rule/version that triggered the risk tier is persisted and traceable.
- The report clearly explains how the multiple signals contributed to the final risk tier.

## Carry-forward items
- Additional detector support.

---

# R5 — Third Risk-Eligible Manipulation Detector

## Objective
Satisfy the "Rule of Three" by adding a third major risk-eligible detector (e.g., general artifacts, lighting inconsistencies, or lip-sync anomalies) to broaden forensic coverage without confusing it with existing AASIST, Active Speaker, or C2PA signals.

## Scope
- Benchmark and integrate a third risk-eligible manipulation detector.
- Refactor the ingestion and execution pipeline slightly if a generic abstraction is now justified by the Rule of Three.

## Non-goals
- Over-engineering plugin abstractions if three hardcoded paths remain cleaner.

## Dependencies
- R4 Calibration + Risk Engine v2.

## Exit criteria
- A third detector runs in the pipeline and produces independent evidence.
- The Risk Engine safely consumes the third signal (after calibration).
- The pipeline architecture proves it can scale beyond two visual detectors.

## Carry-forward items
- Automated plugin loading.

---

# R6 — Model Operations / Shadow Mode

## Objective
Safely evaluate experimental or uncalibrated models against live production traffic without impacting the user-facing forensic report or Risk Engine.

## Scope
- Implement a "shadow mode" flag for specific worker tasks.
- Asynchronously run experimental detectors on incoming media.
- Persist shadow evidence isolated from the public API, report, and Risk Engine (implementation leaves exact persistence design to the task).

## Non-goals
- Showing shadow data to users.
- Slowing down the primary analysis pipeline.

## Dependencies
- R1 Production Readiness (Worker execution/resource limits are critical here).

## Exit criteria
- Shadow models run on live traffic without blocking the main pipeline.
- Shadow data is persisted separately for internal benchmark analysis.
- Risk Engine output is completely isolated from shadow models.

## Carry-forward items
- Automated shadow-to-production promotion.

---

# R7 — DeepGuard v2 (COMPLETED)

## Objective
Finalize the transition to a multi-model, enterprise-grade media authenticity platform and declare DeepGuard v2.

## Scope
- Comprehensive UI/UX polish incorporating all multi-detector evidence.
- "Risk trace" terminology clarity and Report Page visual polish.
- Finalization of any remaining carry-forward items (Instagram auth, high-quality YouTube DASH/HLS).

## Non-goals
- Pushing unproven features; v2 is a stability and maturity milestone.

## Dependencies
- R1 through R6 completion.

## Exit criteria
- The platform operates stably with multiple detectors and shadow mode.
- Reports elegantly present complex, multi-source forensic evidence.
- DeepGuard v2 is officially tagged and released.

## Carry-forward items
- Ongoing model retraining and MLOps lifecycle.

# R9 — Verdict Semantics & Decision Coverage

## Goal

Make InspectRoot operationally understandable for newsroom/media users without weakening forensic correctness.

R9 separates three concepts that must never be conflated:

1. Manipulation assessment
2. Decision coverage
3. Authenticity / provenance

The system must clearly answer:

- Did any calibrated decision-eligible detector detect manipulation?
- Did all decision-eligible detectors successfully produce usable readings?
- Is there independent provenance/source evidence establishing authenticity?

A detector failing to find manipulation is not proof that media is authentic.

Historical decisions remain immutable and version-aware.

---

## R9-T1 — Verdict Semantic Contract

Define the formal semantic contract before any production implementation.

Must define:

- `decision_eligible`
- `evidence_only`
- `usable_reading`
- `decision_coverage`
- `complete_coverage`

User-facing verdicts:

- `MANIPULATION_DETECTED`
- `NO_CALIBRATED_MANIPULATION_SIGNAL`
- `INCONCLUSIVE`

Rules:

- `MANIPULATION_DETECTED`:
  At least one decision-eligible detector produced a usable reading and reached its own calibrated threshold.

- `NO_CALIBRATED_MANIPULATION_SIGNAL`:
  All decision-eligible detectors produced usable readings and none reached its own calibrated threshold.

- `INCONCLUSIVE`:
  At least one decision-eligible detector failed to provide a usable reading and no other decision-eligible detector reached its calibrated threshold.

Evidence-only detector availability must not reduce decision coverage.

Deliverables:
- formal truth table
- explicit invariants
- status semantics for SUCCESS / FAILED / TIMEOUT / unavailable / abstained / no-row
- no production-code changes

---

## R9-T2 — Versioned Decision Engine

Introduce the next immutable decision ruleset.

Decision-eligible detectors:

- NVIDIA synthetic-video detector
- EfficientNet-B7 face-manipulation detector

LipForensics remains evidence-only.

Core logic:

```text
if any decision-eligible detector reaches its calibrated threshold:
    MANIPULATION_DETECTED

else if every decision-eligible detector produced a usable reading:
    NO_CALIBRATED_MANIPULATION_SIGNAL

else:
    INCONCLUSIVE
```

Constraints:

- no threshold changes
- no score averaging
- no voting
- no confidence reinterpretation
- historical rulesets remain immutable
- existing persisted decisions are never silently recalculated

---

## R9-T3 — Replay & Validation

Replay the new semantics against the existing evaluation corpus before activation.

Required cases include:

- genuine media
- generated video
- face swaps
- NVIDIA failure
- B7 failure
- dual failure
- LipForensics failure
- threshold boundary cases

Required outputs:

- genuine false-positive rate
- generated-video detection rate
- face-swap detection rate
- INCONCLUSIVE rate
- decision coverage distribution
- comparison with the current ruleset

No production activation before Architect review.

---

## R9-T4 — Decision Trace API

Expose persisted verdict semantics directly through the API.

The frontend and PDF must not reconstruct forensic decision logic.

API must expose:

- final persisted verdict
- ruleset version
- calibration identity
- decision coverage
- decision-eligible detector states
- supplementary evidence states

The API/persisted trace remains the single source of truth.

---

## R9-T5 — Operational Report Summary

Rework the report summary for newsroom/media users.

Primary summary must answer:

1. What did InspectRoot detect?
2. Was decision coverage complete?
3. What does the result not prove?

Example:

```text
No calibrated manipulation signal detected
Decision coverage: 2/2 complete
```

Required clarification:

```text
This does not prove that the media is authentic.
It means InspectRoot found no calibrated evidence of the manipulation
types covered by the completed decision detectors.
```

Detailed forensic evidence remains available below.

---

## R9-T6 — Authenticity / Provenance Status

Separate authenticity/provenance from manipulation detection.

Initial provenance states may include:

- `UNVERIFIED`
- `PROVENANCE_PRESENT`

Rules:

- absence of C2PA is not manipulation evidence
- presence of C2PA does not automatically prove authenticity
- detector scores cannot produce `VERIFIED_AUTHENTIC`
- provenance and manipulation assessment remain separate

---

## R9-T7 — Human Review Semantics

Integrate the existing Human Review workflow with the new verdict model.

Constraints:

- human review does not mutate forensic verdict fields
- human review does not rewrite detector evidence
- audit history remains append-only
- automated verdict and human review remain visually and semantically separate

---

## R9-T8 — UX & Report Consolidation

After semantics are stable, consolidate the report hierarchy.

Primary order:

1. Verdict
2. Decision coverage
3. Authenticity / provenance
4. Human review
5. Technical forensic evidence

Newsroom users should understand the operational result within seconds.

Forensic users must still be able to inspect raw evidence.

Do not use:

- Fake/Real probability
- confidence percentage
- score averaging
- unsupported authenticity claims

---

## R9-T9 — Final QA / Closure

R9 closes only after proving:

- evidence-only detector failure cannot reduce decision coverage
- decision-eligible detector failure produces INCONCLUSIVE when required
- calibrated threshold hit produces MANIPULATION_DETECTED
- complete decision coverage with no threshold hit produces NO_CALIBRATED_MANIPULATION_SIGNAL
- historical rulesets remain immutable
- frontend does not independently calculate verdict
- PDF does not independently calculate verdict
- API/persisted trace is the single source of truth
- detector scores are never described as InspectRoot confidence
- provenance remains independent from manipulation assessment
- human review cannot alter forensic evidence
- regression suite passes
- migrations are clean
- report rendering remains correct

No new feature work during closure.

---

## R9 Execution Rule

Execute strictly one task at a time:

R9-T1
→ Architect review
→ implementation
→ Architect FINAL PASS
→ commit

Then proceed to R9-T2.

No task may begin before the previous task is committed.

## R10 — Fast Decision + On-Demand Deep Evidence Architecture

### Goal

Reduce production decision latency by separating the decision-critical path from supplementary forensic enrichment.

The automated verdict must become available as soon as the decision-eligible detectors complete, while evidence-only analysis continues separately and must never mutate the persisted verdict.

### Core invariant

> The automated verdict is final once the Fast Decision Path completes. Deep Evidence may enrich the report, but it must never change `risk_level`, `risk_rule_id`, `risk_rules_version`, `risk_calibration_id`, or decision coverage.

### R10-T1 — Execution Contract

Define the production execution contract for:

- Fast Decision Path
- Deep Evidence Enrichment
- transition/state semantics
- retry/failure behavior
- historical compatibility

Fast Decision Path must include only what is required for the automated decision:

- media acquisition / normalization required by deciding detectors
- NVIDIA SVD
- EfficientNet-B7
- `evaluate_v5`
- verdict persistence
- RiskTrace-compatible evidence persistence
- provenance where it is sufficiently inexpensive and independent

Deep Evidence includes supplementary signals such as:

- LipForensics
- Effort
- ASD
- AASIST
- future evidence-only detectors

Do not redesign the forensic data model unless required.

---

### R10-T2 — Worker / Job Separation

Separate execution orchestration without creating two independent analysis systems.

Requirements:

- preserve a single `Analysis`
- preserve existing detector evidence models
- preserve versioned decision semantics
- introduce the smallest job/state separation necessary
- Fast Decision must not wait for Deep Evidence
- Deep Evidence must not write or recompute automated verdict fields
- retries and failures in enrichment must not invalidate a completed Fast Decision
- avoid speculative abstractions and unnecessary queues

Prefer extending the existing async job architecture over introducing a new orchestration framework.

---

### R10-T3 — API / UI State Separation

Expose decision completion and enrichment completion as distinct states.

User-facing behavior should clearly distinguish:

- automated assessment complete
- supplementary evidence processing
- supplementary evidence complete
- supplementary evidence unavailable / failed

Requirements:

- report may be opened immediately after Fast Decision completes
- verdict wording must remain unchanged while enrichment runs
- evidence sections may progressively appear
- no UI logic may recompute forensic decisions
- legacy reports remain readable
- print/PDF output must represent the actual state at render time

Potential product modes may include:

- Quick Scan
- Deep Analysis

but mode naming and UX must remain secondary to the underlying execution contract.

---

### R10-T4 — Regression & Latency QA

Validate that the new orchestration improves latency without changing forensic semantics.

Required checks:

- Fast Decision result matches the current synchronous/full-chain decision for identical SVD/B7 evidence
- Deep Evidence completion cannot mutate the verdict
- LipForensics remains evidence-only
- provenance remains orthogonal
- Human Review remains orthogonal
- historical analyses remain immutable
- retries/failures do not produce duplicate verdict writes
- race conditions between report reads and enrichment writes are safe

Latency measurements:

- upload → SVD/B7 complete
- upload → persisted verdict
- upload → report available
- upload → full enrichment complete

Target product direction:

- automated verdict should normally be available in approximately 30–90 seconds where detector/runtime conditions allow
- Deep Evidence latency is reported separately and is not part of Fast Decision SLA

R10 closes only after semantic equivalence and latency improvements are demonstrated on representative media.


---

## R11 — External Genuine Regression: Real-TurnTurk

### Goal

Add an external Turkish genuine conversational-video regression set to evaluate false positives and detector stability on natural speech, facial movement, and audio/video dynamics.

Dataset:

- `tugrulbayrak/Real-TurnTurk`
- external dataset
- must retain source attribution, license metadata, dataset version, and lineage

This dataset supplements the existing genuine corpus; it does not replace it.

### R11-T1 — Dataset Intake & Lineage

Inspect and freeze:

- exact dataset revision
- available media files
- participant/conversation structure
- audio/video availability
- codec/container information
- source metadata
- license
- usable clip duration

Create a deterministic manifest.

Do not invent ground-truth labels beyond what the dataset actually supports.

---

### R11-T2 — Deterministic Genuine Clip Set

Create approximately 30–60 deterministic clips representing natural Turkish conversational video.

Requirements:

- deterministic extraction
- stable clip IDs
- preserve source recording lineage
- avoid overlapping clips where practical
- record timestamps and preprocessing
- do not mix these clips into calibration data

The set is an external genuine regression population.

---

### R11-T3 — Detector Regression Run

Run the current production decision path over the frozen clip set.

Measure:

- SVD invocation / usability / threshold hits
- B7 invocation / usability / abstentions / threshold hits
- automated verdict distribution
- full / partial / zero decision coverage
- false-positive count/rate
- RiskTrace consistency

Do not call non-hits proof of authenticity.

Where useful, run a small Deep Evidence smoke subset to verify natural Turkish speech/facial motion does not create pathological supplementary evidence behavior.

---

### R11-T4 — Codec / Transcode Stability

Create controlled transformations from the genuine source clips where legally and technically possible:

- H.264 transcode
- bitrate reduction
- resolution reduction
- social-media-like compression
- frame-rate variation

Compare detector stability against the original external genuine clips.

Purpose:

- identify compression-induced false positives
- identify detector availability regressions
- identify preprocessing sensitivity

This task is regression analysis, not calibration.

---

### R11-T5 — External Genuine Regression Report

Produce a versioned report containing:

- dataset revision
- sample manifest
- lineage
- runtime/model identities
- decision metrics
- false positives
- codec/transcode findings
- anomalies
- comparison with existing genuine validation populations

If significant false-positive or stability problems appear, open a separate capability/improvement task rather than silently modifying thresholds.


---

## R12 — Ground Truth / Feedback & Evaluation Governance

### Goal

Introduce explicit Ground Truth / Feedback capability without conflating analyst opinion, automated verdicts, or provenance.

Ground Truth is separate from Human Review.

Human Review remains analyst opinion.

Automated forensic evidence remains immutable.

### R12-T1 — Ground Truth Contract & Taxonomy

Define the Ground Truth contract.

Candidate source classes:

- OWNER_KNOWN
- CONTROLLED_TEST
- EXTERNAL_VERIFIED
- UNKNOWN

Candidate labels:

- GENUINE
- AI_GENERATED
- FACE_SWAP
- AUDIO_MANIPULATION
- OTHER_MANIPULATION
- UNKNOWN

Manipulation-family taxonomy may include:

- NONE
- GENERATED_VIDEO
- FACE_SWAP
- AUDIO_MANIPULATION
- OTHER
- UNKNOWN

Do not derive Ground Truth from the automated verdict.

---

### R12-T2 — Persistence, Audit & API

Add Ground Truth persistence separately from:

- detector evidence
- automated verdict
- provenance
- Human Review

Requirements:

- explicit source
- explicit label
- actor
- timestamp
- audit history
- nullable/unknown states
- no mutation of forensic evidence

---

### R12-T3 — Feedback Capture UI

Add controlled UI for authorized users to record known truth or evaluation feedback.

UI must clearly distinguish:

- automated assessment
- Human Review
- Ground Truth

Do not label analyst disagreement as false positive/false negative unless Ground Truth exists.

---

### R12-T4 — Evaluation Mapping & Metrics

Only where Ground Truth exists, derive:

- TP
- TN
- FP
- FN
- precision
- recall
- specificity
- FPR
- FNR

Metrics must remain version-aware by:

- ruleset
- calibration
- detector deployment
- dataset split

Historical evaluation results must remain reproducible.

---

### R12-T5 — Dataset Governance & Split Integrity

Introduce dataset governance for:

- calibration
- validation
- test
- holdout

Track:

- source lineage
- recording identity
- transformations
- derived clips
- manipulation family
- generation pipeline where known

Prevent recording-level or lineage-level leakage between splits.

---

### R12-T6 — Export / Replay Integration

Allow Ground Truth-bearing evaluation sets to participate in versioned replay/export workflows.

Requirements:

- deterministic manifests
- ruleset/model identity capture
- immutable historical outputs
- no automatic threshold tuning
- no contamination between calibration and evaluation sets

---

### R12-T7 — Final QA / Closure

Verify:

- Ground Truth cannot alter forensic evidence
- Ground Truth cannot silently alter persisted automated verdicts
- Human Review remains separate from truth
- FP/FN/TP/TN are only produced where sufficient Ground Truth exists
- auditability is complete
- historical results remain reproducible
- API/UI semantics are consistent

R12 closes only after Ground Truth, Human Review, automated assessment, and provenance are demonstrably separate concepts throughout the product.


## R13 — Post-R12 Product Improvements

### R13-T1 — User Feedback Capture

Allow regular users to give feedback on results from their own analyses without turning that feedback into Ground Truth.

Requirements:
- User Feedback is a separate domain concept from Ground Truth, Human Review, automated verdict, and forensic evidence.
- A regular user may only submit feedback for an analysis they own/authorize.
- User feedback must NEVER modify Ground Truth, `Analysis.risk_*` (persisted verdict), `AnalysisSignal`, or `AnalysisReview` (Human Review).
- User feedback does not participate in FP/FN/TP/TN evaluation.
- Show feedback on admin pages separately.
- Vocabulary: AGREE, DISAGREE, UNSURE. Optional claimed label: GENUINE, AI_GENERATED, FACE_SWAP, OTHER.

---

### R13-T2 — Operational Summary Taxonomy Cleanup

Separate or remove the legacy risk taxonomy from the operational verdict summary to avoid confusing R9 decision values (MANIPULATION_DETECTED, etc.) with legacy risk levels (HIGH, MEDIUM, UNKNOWN).

Requirements:
- Primary card shows only Decision / Verdict (MANIPULATION_DETECTED, NO_CALIBRATED_MANIPULATION_SIGNAL, INCONCLUSIVE, UNDECIDED).
- Legacy risk level, if kept, shown separately as "Recorded Risk Level".
- No detector changes, threshold changes, or ruleset changes. UI/reporting semantics cleanup only.

## R14 — Production Monitoring & Review Operations

### R14-T1 — Baseline Test Failures Cleanup (CLOSED)

Determine the exact root cause of both baseline failures and fix them with the smallest correct change, without weakening test coverage or changing intended product semantics.

---

### R14-T2 — User Feedback Analytics (CLOSED)

Add operational analytics for UserFeedback without treating feedback as Ground Truth.
- Examples: total feedback count, AGREE / DISAGREE / UNSURE distribution, disagreement rate by persisted verdict, claimed-label distribution, recent feedback activity.
- Invariants: UserFeedback remains separate from GroundTruth, Human Review, and automated verdicts. No FP/FN/TP/TN calculations from user feedback. No verdict or detector changes.

---

### R14-T3 — Admin Review Queue (CLOSED)

Allow admins/analysts to identify analyses that may need review using operational signals (e.g. user DISAGREE feedback, INCONCLUSIVE decisions, missing/failed detector signals).
- Invariants: Queue membership is not Ground Truth. Must not modify forensic evidence or verdicts. Do not invent a risk score unless separately calibrated/approved.

---

### R14-T4 — Detector Failure / Missing-Signal Monitoring (CLOSED)

Expose operational health of detector execution (FAILED, TIMEOUT, missing expected signals, provider/version breakdown, recent failure rates).
- Invariants: Monitoring only. No detector or threshold changes in this task.

---

### R14-T5 — Live Dataset Governance & Ground Truth Acquisition Workflow (CLOSED)

Make it practical to build verified live evaluation data from production cases.
- Cover: candidate selection, governance completeness, Ground Truth recording workflow, lineage/split assignment, export eligibility visibility, explicit distinction between user feedback, analyst review and verified Ground Truth.

---

### R14-T6 — Production QA / Closure (CLOSED)

Run final cross-component QA for R14 and close the phase.

---

## Deferred Capability Backlog

### Face-Swap Detector Improvement

Current R9 evidence shows EfficientNet-B7 produces usable readings on the face-swap validation population but does not reach its current calibrated operating threshold.

This is a detector capability / operating-point limitation, not an R9 semantic or pipeline failure.

Do not block current productization work on this item.

Future work may include:

- B7 score-distribution analysis
- ROC / PR analysis
- threshold recalibration feasibility
- challenger face-swap detector benchmark
- additional manipulation families
- recompression / low-resolution robustness
- versioned recalibration and replay if justified

Any threshold or detector change must create a new calibration/ruleset context and must never rewrite historical analyses.