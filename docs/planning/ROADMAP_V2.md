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

# R7 — DeepGuard v2

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
