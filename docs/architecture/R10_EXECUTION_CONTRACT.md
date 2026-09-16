# R10-T1 — Execution Contract: Fast Decision + On-Demand Deep Evidence

**Status:** Awaiting Architect FINAL PASS
**Phase:** R10 — Fast Decision + On-Demand Deep Evidence Architecture
**Applies to:** every analysis executed under the R10 orchestration, from the task that first implements it (R10-T2) onward
**Does not apply to:** any analysis persisted before that point — `p7-v1.0.0`, `r5-v3.0.0`, `r7-v4.0.0`, and `r9-v5.0.0` analyses decided by the single-stage worker (see §8)

**No application code is modified by this task.** `worker.py`, `models.py`, `detection.py`,
`risk_engine.py`, `risk_trace.py`, the API layer and the frontend are untouched. This
document is the contract that R10-T2 implements, R10-T3 exposes, and R10-T4 verifies.
Where it describes current behaviour it does so to constrain those tasks, not to authorise
a change now.

---

## 1. Purpose

R10 exists to decouple **decision latency** from **evidence completeness**. Today one job
runs every detector before an analysis is allowed to reach `completed`, so the automated
verdict waits on detectors that, under `r9-v5.0.0`, cannot affect it.

R10 splits execution into two paths over the *same* `Analysis`:

| Path | Question it answers | Latency budget |
|---|---|---|
| **Fast Decision Path** | What is the automated verdict, and what coverage stands behind it? | Target 30–90 s (product direction, R10-T4) |
| **Deep Evidence Enrichment** | What else can be shown to an analyst about this media? | Unbounded; reported separately, never part of the Fast Decision SLA |

### 1.1 Core invariant

> The automated verdict is final once the Fast Decision Path completes. Deep Evidence may
> enrich the report, but it must never change `risk_level`, `risk_rule_id`,
> `risk_rules_version`, `risk_calibration_id`, or decision coverage.

Everything below is an elaboration of that sentence. Where any other statement in this
document appears to conflict with it, the invariant wins and the other statement is a
defect in this document.

### 1.2 What R10 does not change

R10 is an **orchestration** change, not a forensic one. It does not change:

- the forensic data model (`Analysis`, `MediaFile`, `AnalysisSignal`, `AnalysisSegment`);
- the risk ruleset, its thresholds, its calibration, or its vocabulary;
- which detectors are `decision_eligible` and which are `evidence_only` — that remains a
  property of the frozen ruleset version, as fixed by the R9-T1 Verdict Semantic Contract;
- the meaning of any persisted verdict, past or future.

A verdict produced by the Fast Decision Path must be **bit-identical** to the verdict the
current single-stage worker would produce from the same decision-eligible evidence. R10-T4
demonstrates this; R10-T2 may not introduce any change that makes it untrue.

---

## 2. Vocabulary

| Term | Definition |
|---|---|
| **Fast Decision Path** | The bounded sequence of work required to persist an automated verdict and its RiskTrace-compatible evidence. Defined in §3. |
| **Deep Evidence Enrichment** | Execution of evidence-only detectors after, or concurrently with, the fast decision. Defined in §4. |
| **decision-complete** | The Fast Decision Path has terminated successfully and the decision fields are persisted. §5.1. |
| **analysis-fully-enriched** | Every Deep Evidence component for this analysis has reached a terminal state, and all of them succeeded. §5.2. |
| **decision field** | One of the five write-protected columns listed in §6.1. |
| **enrichment field** | Any persisted artefact a Deep Evidence component is permitted to write. §6.2. |
| **decision-eligible / evidence-only** | As defined by R9-T1 §2.1, read from the frozen ruleset version the analysis ran under — never from the runtime detector registry. |

**decision-complete does not imply analysis-fully-enriched, and
analysis-fully-enriched is never a precondition for decision-complete.** This is the whole
point of R10 and the single most important sentence in this section.

---

## 3. Fast Decision Path

### 3.1 Membership

The Fast Decision Path contains **only** what the automated decision requires:

1. **Media acquisition and normalization** — but strictly the normalization the deciding
   detectors need. Derivative creation that exists solely to feed an evidence-only detector
   is Deep Evidence work, not fast-path work.
2. **NVIDIA SVD** — `provider="nvidia"`, `signal_type="synthetic_video"`.
3. **EfficientNet-B7** — `provider="efficientnet-b7"`, `signal_type="face_manipulation"`.
4. **`evaluate_v5` invocation** — exactly one invocation per analysis (§7.2).
5. **Verdict persistence** — the five decision fields of §6.1, written in one transaction.
6. **RiskTrace-compatible evidence persistence** — the signal rows and segment rows that
   make the decision replayable. A verdict must never become visible without the evidence
   that supports it already being durable (§7.5). Whether the two share one physical
   transaction is an implementation property, not the invariant.

Nothing else may be added to the Fast Decision Path without a ruleset change that makes a
detector decision-eligible. "It is quick" is not a membership criterion; "the verdict
cannot be computed without it" is.

### 3.2 The LipForensics argument to `evaluate_v5`

`evaluate_v5` currently takes three evidence arguments — SVD, face, and lip — and
`worker.py` supplies all three. Under `r9-v5.0.0` LipForensics is **evidence-only**: it is
counted in no coverage denominator, it can trip no rule, and its absence, failure or
abstention removes no coverage.

Under R10 the fast decision runs **before** LipForensics has executed. Therefore:

- `evaluate_v5` continues to receive its lip argument, unchanged in shape. The worker does
  not start deciding which arguments the engine is entitled to read — that has always been
  the engine's business, not the worker's.
- At fast-decision time that argument reflects the persisted lip evidence as it actually
  is, which under R10 will normally be **absent**. Absent is a state `evaluate_v5` already
  handles, and under `r9-v5.0.0` it is indistinguishable in outcome from present.
- Because lip evidence is counted nowhere under `r9-v5.0.0`, passing it as absent yields
  the same verdict, the same rule id and the same coverage as passing it populated. R10-T4
  must prove this on representative media rather than assume it.

**Consequence for any future ruleset:** if a ruleset version ever makes LipForensics
decision-eligible again — as `r5-v3.0.0` did — then LipForensics moves into the Fast
Decision Path for analyses run under that ruleset. Deep Evidence membership is derived from
the ruleset's role assignment, and R10-T2 must not hard-code a detector list that can drift
away from it.

### 3.3 Provenance is orthogonal

Provenance (`provider="c2pa"`, `signal_type="provenance"`) is neither Fast Decision nor
Deep Evidence in the sense that matters here: it answers a different question — *is there
independent source evidence?* — and it enters no coverage arithmetic under any ruleset.

- Provenance **must not block** the fast decision. It may run in parallel with the
  deciding detectors, and whether it completes quickly, fails, returns nothing, or times
  out, the verdict completes regardless.
- Provenance is read from the **original** artefact, never from a derivative, and R10 does
  not change that.
- A provenance result arriving after the verdict is persisted **does not** reopen the
  verdict, exactly as a Deep Evidence result does not.
- Provenance may remain inside the fast-path job where it is cheap and independent, or move
  out of it. That placement is an R10-T2 latency decision; it carries no semantic weight in
  either direction, precisely because provenance never touches a decision field.

### 3.4 Human Review is orthogonal

`AnalysisReview` is an analyst's record about an analysis. It has never been part of the
automated pipeline and R10 does not make it one.

- Human Review neither blocks nor is blocked by either path.
- A review may be opened as soon as the analysis is decision-complete; it does not wait for
  enrichment.
- An analyst assessment **never** writes a decision field, and enrichment **never** writes
  a review field. The two are disjoint, in both directions.

---

## 4. Deep Evidence Enrichment

### 4.1 Membership

Every detector that is `evidence_only` under the analysis's frozen ruleset version,
currently:

| Component | `provider` | `signal_type` |
|---|---|---|
| LipForensics | `lipforensics` | `lip_forensics` |
| Effort | `effort` | `face_forgery` |
| Active Speaker Detection (ASD) | `nvidia` | `active_speaker` |
| AASIST | `aasist` | `audio_authenticity` |
| future evidence-only detectors | — | — |

ASD produces speaker-activity evidence; it does not diarize and it assigns no identity. R10
does not alter that.

### 4.2 Scheduling

Deep Evidence may be scheduled concurrently with the Fast Decision Path or after it. R10-T2
chooses. Two constraints bind that choice:

1. Deep Evidence **must never** be a precondition of the fast decision. A concurrent
   enrichment that is still running when the deciding detectors finish does not delay the
   verdict by one millisecond.
2. Deep Evidence **must never** contend for a resource in a way that can starve the Fast
   Decision Path. If the two share a GPU or a concurrency slot, the fast path takes
   priority. Latency regression on the fast path is a failure of R10 even if every verdict
   is correct.

"On-demand" is permitted: a product mode may defer enrichment until a user asks for it
(R10-T3's Quick Scan / Deep Analysis direction). Deferred enrichment is an enrichment state,
not a failure, and §5.2 gives it one.

---

## 5. Two-Axis State Semantics

The `Analysis` entity stays **unified**. There is one analysis, one id, one report, one
forensic record. What R10 changes is that its lifecycle is read along two independent axes
instead of one scalar status, because a single status cannot express
"verdict ready, evidence still arriving" without lying about one of the two.

### 5.1 Decision axis

| Decision state | Meaning |
|---|---|
| `DECISION_PENDING` | The Fast Decision Path has not yet terminated. No decision field is populated. |
| `DECIDED` | The Fast Decision Path terminated successfully. All five decision fields are populated and immutable from this instant (§6.3). **This is `decision-complete`.** |
| `DECISION_FAILED` | The Fast Decision Path terminated without producing a verdict — a defect, an unreadable artefact, an infrastructure failure. Decision fields remain null. |

`DECIDED` and `DECISION_FAILED` are terminal. There is no transition out of either, and no
transition between them in either direction. In particular there is no path
`DECIDED → DECISION_PENDING`: nothing reopens a decided analysis.

`INCONCLUSIVE` is a **verdict**, not a decision failure. An analysis whose coverage was
incomplete and whose detectors found nothing usable is `DECIDED` with
`risk_level = "INCONCLUSIVE"`. Conflating the two would let a real classification be
presented as a system fault.

### 5.2 Enrichment axis

| Enrichment state | Meaning |
|---|---|
| `ENRICHMENT_NOT_REQUESTED` | Enrichment is deferred or was never asked for (on-demand mode). Not a failure. |
| `ENRICHMENT_PENDING` | Enrichment is owed and has not started. |
| `ENRICHMENT_PROCESSING` | At least one Deep Evidence component is running. |
| `ENRICHMENT_COMPLETE` | Every Deep Evidence component reached a terminal state and all succeeded. **This is `analysis-fully-enriched`.** |
| `ENRICHMENT_PARTIAL` | Every component reached a terminal state; at least one succeeded and at least one did not. |
| `ENRICHMENT_FAILED` | Every component reached a terminal state and none succeeded. |
| `LEGACY_SINGLE_STAGE` | Read-time projection only, never a live state. The analysis ran the full detector chain in one stage, before R10 existed. Says nothing about which components succeeded — that is read from the signal rows. §8.1. |

`LEGACY_SINGLE_STAGE` is reachable only by projection over a pre-R10 row (§8.1). No
analysis executed under R10 ever enters it, and no writer ever produces it.

A component that abstains for a documented forensic reason — no face track, no audio stream
— **succeeded**. It was asked, it answered, and its answer is evidence. Abstention is not
an enrichment failure, and R10-T3 must not render it as one.

### 5.3 The two axes are independent

Every combination of the two is legal, and the following are the ones that matter:

| Decision | Enrichment | Reading |
|---|---|---|
| `DECIDED` | `ENRICHMENT_PROCESSING` | Verdict is final and the report is open. Evidence sections are still filling. **The state R10 exists to create.** |
| `DECIDED` | `ENRICHMENT_FAILED` | Verdict is final. Supplementary evidence is unavailable. The analysis is complete as a *decision*. |
| `DECIDED` | `ENRICHMENT_COMPLETE` | Verdict is final and the forensic record is whole. |
| `DECISION_FAILED` | any | No verdict exists and none will. Enrichment may still hold evidence; it confers no verdict on anything. |

**Enrichment state never gates report availability.** The report opens the moment the
analysis is `DECIDED`. R10-T3 makes the distinction visible; it may not make it blocking.

### 5.4 Relationship to the existing `status` columns

`analyses.status` (`queued` / `completed` / `failed`) and `analysis_jobs.status`
(`queued` / `processing` / `completed` / `failed`) are today a single-axis encoding of the
decision axis alone. R10-T2 decides the persistence representation of the two axes —
whether the enrichment axis becomes a column, a derived projection over per-component
execution rows, or a separate job entity in the manner of `ShadowRun`.

That choice is constrained, not made, here:

- `analyses.status` must keep meaning **decision** status. `completed` must not come to
  mean "and enriched". Every existing reader — API, dashboard, print/PDF, tests — must keep
  its current meaning for free.
- The `analysis_jobs` one-row-per-analysis uniqueness must be preserved. Enrichment
  execution needs its own record; it must not be squeezed into the decision job's row.
- Enrichment state must be **derivable** from per-component execution records, so that
  `ENRICHMENT_PARTIAL` can name *which* components failed rather than asserting a summary
  nobody can trace back.
- Prefer extending the existing async job architecture over introducing a new orchestration
  framework or a new queue. Speculative abstraction is out of scope for R10.

---

## 6. Field-Level Write Permissions

### 6.1 Decision fields — Deep Evidence is DENIED

The Deep Evidence worker **may not write, update, recompute, or delete** any of:

| Field | Table |
|---|---|
| `risk_level` | `analyses` |
| `risk_rules_version` | `analyses` |
| `risk_calibration_id` | `analyses` |
| `risk_rule_id` | `analyses` |
| `status` | `analyses` |

It may not write these fields with a *different* value, and it may not write them with the
*same* value: an idempotent-looking rewrite is still a second write to a decision field, it
still moves `updated_at`-style metadata, and it still means the enrichment path holds code
that touches the verdict. The prohibition is on the write, not on the delta.

Additionally denied to the Deep Evidence worker:

- any invocation of `evaluate_v5` or any other risk-engine entry point (§7.2);
- any write to the deciding detectors' signal rows — `nvidia`/`synthetic_video` and
  `efficientnet-b7`/`face_manipulation` — including their `score`, `status`,
  `provider_version`, `metadata`, and their `analysis_segments` children;
- any write to `analyses.owner_id` or `analyses.api_key_id`;
- any write to `media_files` identity columns: `original_sha256`, `original_storage_key`,
  `derivative_storage_key`, `derivative_sha256`, and the probed-shape columns;
- any write to `analysis_reviews`.

### 6.2 Enrichment fields — Deep Evidence is ALLOWED

The Deep Evidence worker **may** write:

| Artefact | Scope of permission |
|---|---|
| `analysis_signals` rows for evidence-only providers | Insert and update rows whose `(provider, signal_type)` pair belongs to §4.1. Full ownership of `score`, `status`, `provider_version`, `metadata` on those rows. |
| `analysis_segments` rows | Only those whose `signal_id` references a signal row the enrichment path owns under the line above. |
| its own execution records | Status, lease, timestamps, error text and per-component evidence on whatever record R10-T2 introduces for enrichment execution. |

`analysis_signals.risk_level` stays **null** on every row, as it is today. Enrichment does
not get to write a per-signal classification; that is the per-detector verdict the risk
engine exists to refuse.

### 6.3 Immutability declaration

> **The decision fields are immutable after the first successful fast decision.**
>
> Once an analysis reaches `DECIDED`, the values of `risk_level`, `risk_rules_version`,
> `risk_calibration_id`, `risk_rule_id` and the `completed` decision status are frozen for
> the lifetime of that row. No enrichment outcome, no retry, no later-arriving detector, no
> provenance result, no human review, and no re-queue may alter them.

The only legitimate way to obtain a different verdict for the same media is to submit a
**new analysis**, which gets its own id, its own evidence, and its own decision under
whatever ruleset is frozen at that time. Re-deciding in place would destroy the property
that makes a DeepGuard verdict a forensic record: that it is explainable from the evidence
and the ruleset that existed when it was taken.

This should be enforced in depth rather than by convention — a database-level guard, a
conditional `WHERE` clause on the verdict write, or both. R10-T2 chooses the mechanism;
what it may not choose is to rely on the enrichment code simply not doing it.

---

## 7. Retry, Idempotency, and Partial Failure

### 7.1 Fast Decision retry

- A Fast Decision Path that fails before persisting a verdict may be retried under the
  existing job/lease recovery semantics. Nothing here changes stale-job recovery.
- The verdict write is conditional on the analysis still being undecided. A worker whose
  claim was recovered while it ran is no longer entitled to publish, and rolls back — this
  is today's behaviour and it remains correct under R10.
- **Exactly one successful verdict publication per analysis, ever.** A retry that re-runs
  the deciding detectors and re-evaluates is legitimate while no verdict has been
  published; what is bounded is the publication, not the attempt. Duplicate verdict
  publications are a defect, and R10-T4 tests for them explicitly.

### 7.2 Risk engine invocation

Three rules, in decreasing order of how often they will be tested:

1. **Deep Evidence never invokes the risk engine.** Not `evaluate_v5`, not any other
   entry point. Not to confirm the verdict, not to log a comparison, not behind a feature
   flag, not in shadow mode against the persisted value. The enrichment path holds no
   reference to the engine at all.
2. **The risk engine is never invoked again after a successful verdict publication.** Once
   an analysis is `DECIDED`, re-evaluation produces a second answer, and a second answer
   creates the question of which one is the verdict — a question this architecture exists
   to make unaskable.
3. **A Fast Decision retry may invoke the risk engine again while no verdict has been
   published.** This is the crash window: the engine can run, and the process can die
   before the verdict reaches durable storage. Re-evaluating on retry is the correct
   recovery, and it is safe precisely because evaluation has no persisted effect of its
   own — the only durable act is the publication, and §7.1 admits exactly one of those.

The alternative — durably persisting the evaluation result separately so it never needs
recomputing — would add a second place a verdict can live, ahead of the one place it is
supposed to live. R10 does not pay that cost. Evaluation is cheap, deterministic over the
persisted evidence, and therefore safely repeatable right up until the moment it is
published.
### 7.3 Deep Evidence retry and idempotency

- **Retrying Deep Evidence never reopens the verdict.** A retry re-runs evidence-only
  detectors and rewrites their own signal rows; the decision fields are not in its reach
  (§6.1) and not in its code path.
- Enrichment retries are **idempotent per component**. Re-running a component replaces that
  component's evidence for that analysis; it does not accumulate duplicate rows. The
  existing `(analysis_id, provider, signal_type)` uniqueness on `analysis_signals` is the
  invariant that makes this true, and it must be preserved.
- Enrichment retries carry no bound on when they may occur. An enrichment retried an hour
  after the decision is as legal as one retried a second after it, because neither can
  touch the decision.
- Segment rows belonging to a re-run component are replaced together with their parent
  signal, so evidence from two different executions of the same detector never coexists.

### 7.4 Partial failure

> **Deep Evidence failures never destroy Fast Decisions.**

Concretely:

- If LipForensics fails and AASIST succeeds, the analysis is `DECIDED` +
  `ENRICHMENT_PARTIAL`. The verdict stands, unchanged and unqualified. The report shows
  AASIST's evidence and says plainly that lip evidence is unavailable.
- If every Deep Evidence component fails, the analysis is `DECIDED` +
  `ENRICHMENT_FAILED`. The verdict still stands. An analysis with a verdict and no
  supplementary evidence is a complete decision with a thin report, not a broken analysis.
- A crashed or leaked enrichment worker must never leave an analysis looking undecided. The
  decision axis is not reachable from enrichment recovery code.
- Enrichment failure is recorded per component, with an error message an operator can read.
  A summary state alone would make `ENRICHMENT_PARTIAL` untraceable.

### 7.5 Read/write races

A report read may land at any moment during enrichment. Therefore:

- **A verdict is never visible without its supporting evidence already being durable.**
  This is the invariant; sharing one physical transaction is one way to satisfy it, not the
  requirement itself. Today the worker commits deciding evidence first
  (`worker.py:984`) and publishes the verdict in a later transaction
  (`worker.py:1179`), which satisfies the invariant in that order and must be confirmed
  rather than assumed when R10-T2 inspects the persistence flow. What is forbidden is the
  reverse order — a published verdict whose evidence has not landed.
- Each enrichment component's signal and segment rows become visible **together**, so no
  reader ever observes a half-written detector result. A signal and its segments are
  written in one transaction today, and that must be preserved.
- Readers must tolerate evidence appearing between two successive reads of the same
  analysis. This is expected behaviour, not an inconsistency.
- Print/PDF output represents the state at render time and says so. A PDF rendered during
  enrichment is a truthful document about a decided analysis with evidence outstanding —
  not a draft, and not something to be withheld until enrichment finishes.

---

## 8. Historical Compatibility

### 8.1 Legacy analyses are single-stage

Every analysis decided before R10-T2 ships was produced by the single-stage worker, under
one of `p7-v1.0.0`, `r5-v3.0.0`, `r7-v4.0.0`, or `r9-v5.0.0`. Those rows carry no two-axis
state because no two-axis state existed when they were written.

**Readers must interpret them as legacy single-stage completed**, projected onto the two
axes as follows:

| Legacy `analyses.status` | Decision axis | Enrichment axis |
|---|---|---|
| `completed` | `DECIDED` | `LEGACY_SINGLE_STAGE` |
| `failed` | `DECISION_FAILED` | not applicable |
| `queued` | `DECISION_PENDING` | not applicable |

A legacy `completed` row is **not** mapped to `ENRICHMENT_COMPLETE`. The two mean different
things and collapsing them would be a false claim: `ENRICHMENT_COMPLETE` asserts that every
Deep Evidence component terminated *and succeeded*, which no legacy row can support — a
pre-R10 analysis could always carry a failed evidence-only signal and still reach
`completed`. `LEGACY_SINGLE_STAGE` asserts only what actually happened: the full chain ran
in one stage. Which components succeeded is read from the signal rows, as it always was.

`LEGACY_SINGLE_STAGE` need not be a new database enum value. It is a read-time semantic
projection, derived from the absence of R10 enrichment execution records on the row. R10-T2
chooses the representation; R10-T3 must not render it with wording that claims supplementary
evidence succeeded.

### 8.2 No backfills, no mutation

- **No migration backfills two-axis state onto historical rows.** There is no fact to
  backfill: the distinction being recorded did not exist when they ran.
- **Historical rows are never mutated.** Not their decision fields, not their status, not
  their signals, not their segments. This is the same rule that has governed every prior
  phase and R10 does not weaken it.
- Legacy reports remain readable without any migration. If a reader needs the two-axis view
  it derives it at read time from §8.1, and derives nothing it cannot support.
- A pre-R10 analysis is never re-run to "upgrade" it into the new architecture. Re-running
  means a new analysis with a new id (§6.3).

### 8.3 Ruleset versions are unaffected

R10 introduces **no new ruleset version**. `r9-v5.0.0` continues to be the ruleset the Fast
Decision Path evaluates under, with the same thresholds, the same calibration, the same
coverage arithmetic and the same vocabulary — `MANIPULATION_DETECTED`,
`NO_CALIBRATED_MANIPULATION_SIGNAL`, `INCONCLUSIVE`. Changing where a detector runs is not
a semantic change, and R10 must not be allowed to become one by accident.

---

## 9. Invariants for R10-T4

The following are the testable claims this contract makes. R10-T4 verifies each one; a
failure of any is a failure of R10.

| # | Invariant |
|---|---|
| I1 | The Fast Decision verdict is identical to the current full-chain verdict for identical SVD and B7 evidence. |
| I2 | Passing lip evidence as absent to `evaluate_v5` yields the same verdict, rule id and coverage as passing it populated, under `r9-v5.0.0`. |
| I3 | No Deep Evidence outcome — success, failure, partial, retry — changes any of the five decision fields. |
| I4 | `evaluate_v5` is never invoked by Deep Evidence, and is never invoked again after a successful verdict publication. |
| I5 | Exactly one successful verdict publication occurs per analysis; retries before publication are permitted and produce no duplicate. |
| I6 | LipForensics remains evidence-only and enters no coverage arithmetic. |
| I7 | Provenance completing, failing or timing out does not delay or alter the verdict. |
| I8 | Human Review neither blocks nor is blocked by either path, and writes no decision field. |
| I9 | Historical analyses are byte-for-byte unchanged after R10 ships. |
| I10 | A report read concurrent with an enrichment write never observes a partial detector result, and never observes a published verdict whose supporting evidence is not yet durable. |
| I11 | An enrichment worker crash never moves an analysis out of `DECIDED`. |
| I12 | Fast Decision latency does not regress relative to the pre-R10 SVD+B7 segment. |

Latency figures reported separately, per R10-T4: upload → SVD/B7 complete; upload →
persisted verdict; upload → report available; upload → full enrichment complete. The first
three are the Fast Decision SLA. The fourth is not.

---

## 10. Open Questions for R10-T2

Deliberately left undecided here, because they are implementation choices this contract
constrains rather than makes:

1. **Enrichment execution record** — a new job table, an extension of `analysis_jobs`
   (subject to its one-row-per-analysis uniqueness), or a `ShadowRun`-shaped per-workload
   table. §5.4 lists the constraints on the answer.
2. **Enrichment state persistence** — stored column versus derived projection.
3. **Immutability enforcement mechanism** — database constraint, conditional write, or both
   (§6.3).
4. **Provenance placement** — inside the fast-path job or alongside enrichment. Semantically
   free either way (§3.3); a latency decision only.
5. **Trigger policy** — enrichment always queued at upload, or on demand per product mode.
   §5.2 gives on-demand a state; §4.2 permits both.

---

## 11. Acceptance

This document satisfies R10-T1 when the Architect confirms:

- [x] `docs/architecture/R10_EXECUTION_CONTRACT.md` created; no application code modified.
- [x] Fast Decision Path membership defined and bounded (§3.1).
- [x] Deep Evidence membership defined (§4.1).
- [x] *decision-complete* and *analysis-fully-enriched* distinguished (§2, §5.1, §5.2).
- [x] Two-axis state defined, with the axes independent and the unified `Analysis`
      preserved (§5).
- [x] Deep Evidence allowed/denied fields listed explicitly (§6.1, §6.2).
- [x] Decision-field immutability after first successful fast decision formally declared
      (§6.3).
- [x] Retry, idempotency and partial-failure semantics defined; Deep Evidence failures
      cannot destroy Fast Decisions (§7).
- [x] Architect PASS WITH FIXES applied: publication-bounded risk engine invocation (§7.1,
      §7.2, I4, I5); `LEGACY_SINGLE_STAGE` projection kept distinct from
      `ENRICHMENT_COMPLETE` (§5.2, §8.1); verdict/evidence stated as a visibility ordering
      invariant rather than a shared-transaction requirement (§3.1, §7.5, I10).
- [x] Provenance and Human Review confirmed orthogonal (§3.3, §3.4).
- [x] Historical compatibility defined: v1–v5.0.0 read as legacy single-stage, no backfills,
      no mutation (§8).
