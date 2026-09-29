/**
 * The human review of one analysis, shown beside the forensic result it is about (R8-T7).
 *
 * **The whole design of this page is the boundaries between its cards.** The automated
 * assessment — the verdict, the coverage it was taken from, the ruleset, the calibration and the
 * rule that fired — has no control of any kind on it. Authenticity and provenance is a second
 * card because it is a second question, answered from the file rather than from the detectors.
 * What a person said is the third, and it is the only part of this screen with a form. They are
 * drawn as separate cards with separate headings and separate explanatory prose, because an
 * operator reading a case has to be able to say which of the three statements in front of them
 * came from a detector, which from the file itself, and which from a colleague.
 *
 * **The order is the report's order (R9-T8).** Assessment, then coverage beside it, then
 * provenance, then human review. The two surfaces differ in density and in how much of the
 * record they print, and they must not differ in what they say: an operator comparing this
 * screen with the report a reader was sent has to find the same verdict, the same coverage and
 * the same provenance state on both.
 *
 * That separation is structural and not a styling decision. The forensic values come from
 * `fetchAnalysis` in `../../../analysis`; the review comes from `fetchReview` in `../../reviews`;
 * the form posts to `/admin/update-review`, which forwards to an endpoint that can write
 * nothing but the review row. There is no code path from this page to a risk column.
 *
 * **The note is rendered as text and never as markup.** No `dangerouslySetInnerHTML`, no
 * Markdown, no linkification — it goes into a `<p>` with `whitespace-pre-wrap` so the
 * reviewer's line breaks survive and nothing else about it is interpreted. The API stores a
 * note verbatim precisely because nothing renders it; a note containing `<script>` is a note
 * containing `<script>`, and it prints as those characters. This is the one paragraph of this
 * file that must survive every future edit to it.
 *
 * **The analyst assessment is a third statement and is labelled as one (R9-T7).** The card
 * below prints the automated assessment, the workflow status and the analyst's opinion of the
 * automated assessment as three separate lines with three separate words, because they are
 * three separate things and an operator has to be able to say which is which. Disagreement is
 * rendered as what a person thought; it does not restate the verdict, does not annotate it, and
 * does not appear anywhere near the card above.
 *
 * **Nothing is recomputed.** The risk classification is displayed exactly as the worker
 * committed it, with no detector score compared against a threshold here, and the review is
 * displayed exactly as the API returned it. A screen that re-derived either could contradict
 * the record it exists to show. Nor is anything scored: an assessment is not tallied, not
 * compared with the verdict to produce a judgement about it, and never presented as accuracy.
 *
 * **Ground Truth is a fourth card and a fourth kind of statement (R12-T3).** What the media
 * actually is, recorded against the original's SHA-256 by somebody in a position to know it. It
 * has its own form, its own route and its own endpoint, and it starts from the stored record or
 * from nothing — never from the verdict, the provenance reading or the review above it, because
 * Ground Truth filled in from any of those would mark them correct by construction. This screen
 * does not compare it with the verdict either: that comparison is evaluation, and belongs
 * elsewhere.
 *
 * **Promotion is a fifth card and writes through the three existing endpoints (R14-T5).** It
 * saves dataset governance, then Ground Truth, then the review status, each through its own route
 * and only after the one before succeeded (`../../promotion.ts`). A user feedback record may be
 * chosen as context: its claim prefills the label when it names one exactly, and never chooses a
 * source class. The review status opens on the current state, so `NEEDS_FOLLOW_UP` is retained
 * unless the analyst changes it.
 *
 * A Server Component with no client-side code at all. The controls are plain HTML forms, the
 * same as the account controls, so the page works with JavaScript disabled and the outcome of
 * a save arrives as a redirect carrying it in the query string.
 */

import Link from "next/link";
import { notFound, redirect } from "next/navigation";

import {
  ProvenanceSignal,
  RULES_VERSION_V5,
  RiskTrace,
  V5_VERDICT_WORDING,
  classificationLabel,
  fetchAnalysis,
  fetchSession,
  isV5Verdict,
  provenanceWording,
} from "../../../analysis";
import { LOGIN_PATH, adminAnalysisPath } from "../../../session";
import { AdminAlert } from "../../components/AdminAlert";
import { AdminPageHeader } from "../../components/AdminPageHeader";
import { AdminSection } from "../../components/AdminSection";
import { AdminStatusBadge } from "../../components/AdminStatusBadge";
import { AdminValue } from "../../components/AdminValue";
import {
  ANALYST_ASSESSMENT_AGREES,
  ANALYST_ASSESSMENT_DISAGREES,
  ANALYST_ASSESSMENT_LABELS,
  ANALYST_ASSESSMENT_NONE,
  ANALYST_ASSESSMENT_UNDETERMINED,
  AnalysisReview,
  MAX_REVIEW_NOTE_LENGTH,
  REVIEW_STATUS_LABELS,
  REVIEW_STATUS_NEEDS_FOLLOW_UP,
  REVIEW_STATUS_REVIEWED,
  REVIEW_STATUS_UNREVIEWED,
  ReviewResult,
  fetchReview,
} from "../../reviews";
import {
  GROUND_TRUTH_LABEL_LABELS,
  GROUND_TRUTH_SOURCE_CLASS_LABELS,
  GroundTruth,
  GroundTruthResult,
  fetchGroundTruth,
} from "../../ground-truth";
import { UserFeedbackEntry, UserFeedbackResult, fetchUserFeedback } from "../../user-feedback";
import {
  DATASET_SPLIT_LABELS,
  DatasetGovernance,
  DatasetGovernanceResult,
  fetchDatasetGovernance,
} from "../../dataset-governance";
import {
  BOOLEAN_NOT_RECORDED,
  PROMOTION_STEPS,
  ParsedOutcome,
  PromotionStep,
  REVIEW_TARGETS,
  defaultReviewTarget,
  isReviewOnlyFailure,
  parseOutcome,
  prefilledGroundTruthLabel,
  promotionCandidates,
  selectedPromotionMedia,
} from "../../promotion";
import {
  FEEDBACK_ASSESSMENT_LABELS,
  FEEDBACK_CLAIMED_LABEL_LABELS,
} from "../../../user-feedback";

// The sentence the Ground Truth card shows in place of its form when the analysis carries no
// hash of its original. Ground Truth is keyed by those bytes; without them there is nothing to
// attach it to, and the card says so rather than disappearing.
const NO_SHA256_FOR_GROUND_TRUTH =
  "Ground Truth cannot be recorded because this analysis has no original media SHA-256 identity.";

// What the risk column says when no decision was ever taken. Distinct from `UNKNOWN`, which is
// a decision, and phrased as a statement about the record rather than about the media — the
// same wording the report uses for the same reason.
const NO_DECISION = "No decision recorded";

/* ------------------------------------------------------------------ *
 * Primitives
 * ------------------------------------------------------------------ */

/*
 * The private `Legend`, `Heading`, `Alert` and `Value` this file carried — the sixth copy of the
 * first three, as the comment that stood here counted them — are `AdminPageHeader`, `AdminAlert`
 * and `AdminValue` in `../../components` since R8-T9.
 *
 * One behaviour changed in the move and it is worth naming: this page drew its success alert in
 * emerald while the other six drew theirs in accent. Accent is what `AdminAlert` uses, because in
 * this palette emerald means a person affirmed something — which is what the review badge below
 * uses it for, and what a save confirmation is not.
 *
 * `Fact` stays local. It is on two screens, this one and the account detail, and two is not
 * three. Recorded rather than left implicit, which is the habit that got the extraction made.
 */

/** One labelled fact, as a definition-list pair. */
function Fact({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div>
      <dt className="text-[11px] font-medium tracking-[0.08em] text-muted uppercase">
        {label}
      </dt>
      <dd className="mt-1 text-[13px] text-bone">{children}</dd>
    </div>
  );
}

/* ------------------------------------------------------------------ *
 * The forensic half
 * ------------------------------------------------------------------ */

/**
 * The risk classification as the worker committed it, and the trace that explains it.
 *
 * **No control, no form, no editable field — and no endpoint behind one.** An administrator
 * cannot change any value in this card from anywhere in this application; the review below is
 * the only thing on this page that can be written. The absence is the feature, and the prose
 * beside the card says so in the page's own voice rather than leaving an operator to infer it
 * from the lack of a button.
 *
 * A level the page does not recognize is printed as `Unsupported` rather than dropped. The
 * risk vocabulary can grow, and a screen that silently rendered an unknown classification as
 * blank would be a screen that hides exactly the case a reader needs to see.
 *
 * **Which level is recognized is not decided here.** `classificationLabel` resolves it through
 * the ruleset version the decision names, and the newsroom report calls the same function — so
 * one analysis cannot read as `Manipulation detected` on the report and `Unsupported` here. That
 * split is exactly what this card had while it knew only the pre-v5 levels, and two screens
 * contradicting each other about the same row is a worse failure than either alone: an operator
 * comparing them has no way to tell which one is lying.
 *
 * The coverage fact below is the API's `decision_coverage`, printed as it arrived. This card
 * makes no comparison — not `usable` against `total`, not a score against a threshold — and a
 * decision that states no coverage (every pre-v5 ruleset) simply has no coverage row rather than
 * a fabricated `0/0`.
 */
function ForensicResult({
  analysis,
}: {
  analysis: {
    id: string;
    status: string;
    created_at: string;
    risk_level: string | null;
    risk_rules_version: string | null;
    risk_rule_id: string | null;
    risk_calibration_id: string | null;
    // Read for one value only: the coverage the decision stated. The verdict itself is still the
    // `risk_level` column above — the trace explains a decision and never replaces it.
    risk_trace: RiskTrace | null;
  };
}) {
  const level = analysis.risk_level;
  const coverage = analysis.risk_trace?.decision_coverage ?? null;
  // The limit the verdict carries, for the one vocabulary that states one. Looked up from the
  // same frozen table the report reads (R9-T5), keyed by the ruleset the decision names and by
  // nothing else — so the two screens cannot word the same verdict differently, and a pre-v5
  // level gets no v5 sentence attached to it (R9-T1 invariant 4).
  const wording =
    analysis.risk_rules_version === RULES_VERSION_V5 &&
    level !== null &&
    isV5Verdict(level)
      ? V5_VERDICT_WORDING[level]
      : null;
  const legacyLevel = level !== null && analysis.risk_rules_version !== RULES_VERSION_V5;

  return (
    <AdminSection
      title="Automated assessment"
      description="Committed by the detection pipeline under the ruleset named below, and immutable. There is no control on this card and no route in this application that can alter any value on it — the review beneath is a separate record and does not change what the detectors concluded."
    >
      {/* The assessment first and the coverage beside it, in the order the report states them
          (R9-T8). The record fields follow, under their own rule: an operator opening this case
          needs the finding and how much of the reading it was taken from before they need the
          identity of the ruleset that took it. */}
      <dl className="mt-4 grid grid-cols-1 gap-5 sm:grid-cols-2">
        {/* One field, labelled by the vocabulary the ruleset wrote it in (R13-T2). A v5 row
            holds a verdict and is the decision; any earlier ruleset wrote a level, which is
            labelled as the recorded risk level it is rather than as a verdict. Chosen by the
            ruleset and never by the string, like the label itself. */}
        <Fact label={legacyLevel ? "Recorded risk level" : "Decision"}>
          {level === null
            ? NO_DECISION
            : classificationLabel(level, analysis.risk_rules_version)}
        </Fact>
        {coverage !== null && (
          <Fact label="Decision coverage">
            {/* Three API fields interpolated, including the word. `complete` is never reached
                here by comparing `usable` against `total`: that comparison is the coverage model
                and it lives in one place, on the other side of the API (R9-T4). */}
            <AdminValue>
              {`${coverage.usable}/${coverage.total} ${coverage.status}`}
            </AdminValue>
          </Fact>
        )}
      </dl>

      {/* What the assessment does not establish, in the verdict's own locked words. The report
          prints this sentence under the same verdict; an operator comparing the two screens has
          to find the same limits on both, or the denser screen becomes the one that reads as
          more certain. */}
      {wording !== null && (
        <p className="mt-4 max-w-[74ch] text-[13px] leading-relaxed text-muted">
          {wording.clarification}
        </p>
      )}

      <dl className="mt-5 grid grid-cols-1 gap-5 border-t border-hair pt-5 sm:grid-cols-2">
        <Fact label="Analysis status">{analysis.status}</Fact>
        <Fact label="Ruleset">
          <AdminValue className="break-all">{analysis.risk_rules_version}</AdminValue>
        </Fact>
        <Fact label="Rule fired">
          <AdminValue className="break-all">{analysis.risk_rule_id}</AdminValue>
        </Fact>
        <Fact label="Calibration">
          <AdminValue className="break-all">{analysis.risk_calibration_id}</AdminValue>
        </Fact>
        <Fact label="Submitted">
          {/* The API's own timestamp, shown as stored rather than reformatted into a local
              rendering the record does not hold — the convention every screen in this surface
              follows. */}
          <AdminValue className="break-all">{analysis.created_at}</AdminValue>
        </Fact>
      </dl>
    </AdminSection>
  );
}

/* ------------------------------------------------------------------ *
 * The provenance half — a separate question, from separate evidence
 * ------------------------------------------------------------------ */

/**
 * What the file itself claims about its origin, on the two axes the API states it on (R9-T6).
 *
 * Its own card and never a row on the one above, which is the whole point of the section. An
 * operator who met the provenance state as a field beside the verdict would read it as an input
 * to that verdict; it is not one in either direction — no assessment moved this state and this
 * state moved no assessment — and a card of its own is how this screen says so, the same way the
 * report says it with a heading of its own.
 *
 * **Both axes, always together.** `UNVERIFIED` alone cannot say whether anybody looked: a file
 * that was read and carries nothing and a file whose reading failed both answer it, and only
 * `provenance_availability` tells them apart. `provenanceWording` resolves the pair into the one
 * thing to say and this card prints it; it makes no reading of its own, and in particular it
 * never derives authenticity from the presence of a manifest.
 *
 * A state outside the vocabulary this build knows — an older API that states no axes, a newer one
 * that states something unfamiliar — gets the raw evidence with no caption rather than a sentence
 * written for a different state. The facts below the state are the record either way.
 */
function ProvenanceRecord({ signal }: { signal: ProvenanceSignal | null }) {
  const wording =
    signal === null
      ? null
      : provenanceWording(signal.provenance_status, signal.provenance_availability);

  return (
    <AdminSection
      title="Authenticity and provenance"
      description="A separate question from the assessment above, answered from separate evidence and recorded separately. Nothing here reports the media as authentic or manipulated: the absence of credentials is not evidence of manipulation, and their presence is not proof of authenticity."
    >
      {signal === null ? (
        <p className="mt-4 max-w-[74ch] text-[13px] leading-relaxed text-muted">
          No provenance reading is stored for this analysis. That is not a failed reading —
          nothing recorded one, so there is no evidence from this source either way.
        </p>
      ) : (
        <>
          {wording !== null && (
            <div className="mt-4">
              {/* The pair, as the pair it is. Neither axis is ever printed without the other. */}
              <p className="font-mono text-[11px] tracking-[0.08em] text-muted uppercase">
                {signal.provenance_status} · {signal.provenance_availability}
              </p>
              <p className="mt-1.5 text-[15px] font-semibold text-bone">{wording.title}</p>
              <p className="mt-1.5 max-w-[74ch] text-[13px] leading-relaxed text-muted">
                {wording.meaning}
              </p>
              <p className="mt-2 max-w-[74ch] text-[13px] leading-relaxed text-muted">
                {wording.clarification}
              </p>
            </div>
          )}

          <dl
            className={`grid grid-cols-1 gap-5 sm:grid-cols-2 ${
              wording === null ? "mt-4" : "mt-5 border-t border-hair pt-5"
            }`}
          >
            <Fact label="Reading status">
              <AdminValue className="break-all">{signal.status}</AdminValue>
            </Fact>
            <Fact label="Manifest present in the file">
              {signal.manifest_exists === null
                ? "unknown — the reading failed"
                : signal.manifest_exists
                  ? "yes"
                  : "no"}
            </Fact>
            <Fact label="Validation state">
              <AdminValue className="break-all">{signal.validation_state}</AdminValue>
            </Fact>
            <Fact label="Claim generator">
              <AdminValue className="break-all">{signal.claim_generator}</AdminValue>
            </Fact>
            <Fact label="Signature issuer">
              <AdminValue className="break-all">{signal.signature_issuer}</AdminValue>
            </Fact>
            <Fact label="Remote manifest URL">
              {/* Recorded and never fetched — the same limit the report states. */}
              <AdminValue className="break-all">{signal.remote_manifest_url}</AdminValue>
            </Fact>
          </dl>
        </>
      )}
    </AdminSection>
  );
}

/* ------------------------------------------------------------------ *
 * The review half
 * ------------------------------------------------------------------ */

/** The workflow state as a chip, including the state that is the absence of a record. */
function StatusBadge({ status }: { status: string }) {
  const tone =
    status === REVIEW_STATUS_NEEDS_FOLLOW_UP
      ? "warning"
      : status === REVIEW_STATUS_REVIEWED
        ? "positive"
        : "muted";

  return (
    <AdminStatusBadge tone={tone} mono>
      {/* Unknown statuses print as the API spelled them rather than as a blank chip: the day
          the taxonomy grows, this is what says so. */}
      {REVIEW_STATUS_LABELS[status] ?? status}
    </AdminStatusBadge>
  );
}

/**
 * What the reviewer made of the automated assessment, as a chip.
 *
 * Drawn in the muted tone for all four states, and that is the deliberate part. A green chip
 * for agreement and a red one for disagreement would read as a scorecard on the detector —
 * right and wrong — and this screen is not entitled to say either: nothing here knows what the
 * media actually was. The colour on this card belongs to the workflow status, which is
 * operational; an opinion gets a neutral chip and a label that says whose opinion it is.
 */
function AssessmentBadge({ assessment }: { assessment: string | null }) {
  return (
    <AdminStatusBadge tone="muted" mono>
      {/* Unknown assessments print as the API spelled them rather than as a blank chip, the
          same way the status badge above handles a taxonomy that has grown. */}
      {ANALYST_ASSESSMENT_LABELS[assessment ?? ANALYST_ASSESSMENT_NONE] ?? assessment}
    </AdminStatusBadge>
  );
}

/**
 * The reviewer's own words, printed as text.
 *
 * `whitespace-pre-wrap` so the line breaks somebody typed survive, and nothing else. There is
 * no markup rendering here and there must never be: the API accepts a note verbatim — angle
 * brackets, ampersands and all — on the understanding that every reader treats it as a string.
 * React escapes it by rendering it as a child; introducing `dangerouslySetInnerHTML` anywhere
 * in this component would turn one administrator's note into script in another's browser.
 */
function Note({ note }: { note: string }) {
  if (note === "") {
    return (
      <p className="text-[13px] text-muted">No note was written with this review.</p>
    );
  }

  return (
    <p className="max-w-[74ch] text-[13px] leading-relaxed whitespace-pre-wrap text-bone">
      {note}
    </p>
  );
}

/** What the record says about the review as it stands, including that there is none. */
function ReviewRecord({ review }: { review: AnalysisReview }) {
  if (review.status === REVIEW_STATUS_UNREVIEWED) {
    return (
      <div className="mt-4">
        <StatusBadge status={review.status} />
        <p className="mt-3 max-w-[74ch] text-[13px] leading-relaxed text-muted">
          Nobody has reviewed this analysis. That is the state every analysis is in until
          somebody writes a review here — it is not a record that something is missing, and it
          says nothing about the forensic result above.
        </p>
      </div>
    );
  }

  return (
    <div className="mt-4">
      {/* Two chips and two words, not one compound state. The first says where the case is in
          the workflow; the second says what the reviewer made of the automated assessment.
          Reading either as the other is the mistake this layout exists to prevent, so they are
          labelled rather than left to be inferred from the colour. */}
      <div className="flex flex-wrap items-center gap-x-5 gap-y-2">
        <span className="flex flex-wrap items-center gap-2">
          <span className="text-[13px] text-muted">Human review</span>
          <StatusBadge status={review.status} />
        </span>
        <span className="flex flex-wrap items-center gap-2">
          <span className="text-[13px] text-muted">Analyst assessment</span>
          <AssessmentBadge assessment={review.analyst_assessment} />
        </span>
      </div>

      {review.analyst_assessment === ANALYST_ASSESSMENT_DISAGREES && (
        <p className="mt-3 max-w-[74ch] text-[13px] leading-relaxed text-muted">
          This reviewer did not agree with the automated assessment. That is recorded here as
          their opinion and nothing else: the forensic result above is unchanged, no second
          classification has been stored, and nothing in this record says which of the two was
          correct.
        </p>
      )}

      {review.analyst_assessment === null && (
        <p className="mt-3 max-w-[74ch] text-[13px] leading-relaxed text-muted">
          No assessment was recorded with this review. Reviews written before this field existed
          carry none, and none has been assumed for them — that is not the same as{" "}
          {ANALYST_ASSESSMENT_LABELS[ANALYST_ASSESSMENT_UNDETERMINED].toLowerCase()}, which is a
          position somebody took.
        </p>
      )}

      <dl className="mt-4 grid grid-cols-1 gap-5 sm:grid-cols-2">
        <Fact label="Reviewer">
          {/* The address as it read when the review was last written, never looked up now. A
              rename does not change who signed a past review — see `admin/reviews.ts`. */}
          {review.reviewer_email_snapshot ?? (
            <span className="text-muted">not recorded</span>
          )}
          <div className="mt-1">
            <AdminValue className="break-all">{review.reviewer_id}</AdminValue>
          </div>
        </Fact>
        <Fact label="Last changed">
          <AdminValue className="break-all">{review.updated_at}</AdminValue>
          <div className="mt-1">
            <AdminValue className="break-all">
              {review.created_at === null ? null : `first written ${review.created_at}`}
            </AdminValue>
          </div>
        </Fact>
      </dl>

      <div className="mt-5">
        <dt className="text-[11px] font-medium tracking-[0.08em] text-muted uppercase">
          Note
        </dt>
        <div className="mt-2">
          <Note note={review.note} />
        </div>
      </div>
    </div>
  );
}

/**
 * The form: a status, a note, and one button.
 *
 * `defaultValue` rather than `value` on both controls: these are uncontrolled inputs in a
 * server-rendered form, and a `value` with no `onChange` is a field React will not let the
 * reader alter.
 *
 * The status select offers the two operational statuses and nothing else. `UNREVIEWED` is
 * absent because it is not a value — it is the absence of a review row, and there is no route
 * that un-reviews an analysis. Nothing that reads as a verdict is here either: whether the media
 * is genuine is the card above, decided under a named ruleset, and this control does not get a
 * second opinion on it.
 *
 * The assessment select offers the three assessments plus "Not recorded", and that fourth
 * option is load-bearing in two directions. It is what lets a review written before the field
 * existed be saved again without an opinion being invented for its author, and it is the only
 * way to clear an assessment somebody chose by mistake. It is the preselected option whenever
 * the stored value is null, so opening the form and pressing save records nothing new.
 *
 * **What this control cannot do is overrule the card above.** Choosing "Disagrees with
 * automated assessment" writes one nullable column on the review row; the verdict, the ruleset,
 * the calibration, the rule that fired and the decision coverage are all exactly as they were,
 * and there is no field on this form and no route behind it that reaches any of them. It is
 * also not a correctness label — this screen holds no ground truth against which the detector
 * could be marked right or wrong, and the value stored here must never be counted as though it
 * were.
 *
 * An unreviewed analysis opens with `REVIEWED` preselected, which is the ordinary first answer
 * — but the button says "Save review" rather than anything that implies agreement, and nothing
 * is written until somebody presses it. The assessment opens on "Not recorded" for the same
 * reason, one step further: a default of agreement would be the form answering the question on
 * the reviewer's behalf.
 *
 * `maxLength` on the textarea matches the API's own bound, so a reviewer is stopped at the
 * limit rather than losing the tail of what they wrote to a 422. It is a convenience and not
 * the enforcement: the API checks the length itself, and a request that got past this control
 * is refused there.
 *
 * The button is always enabled, including when nothing on the form has been touched. Saving an
 * unchanged review is a no-op the API applies and returns unchanged — it writes no row and no
 * audit event — which is a better outcome than a disabled state that would need JavaScript to
 * track.
 */
function ReviewForm({ review }: { review: AnalysisReview }) {
  const selected =
    review.status === REVIEW_STATUS_NEEDS_FOLLOW_UP
      ? REVIEW_STATUS_NEEDS_FOLLOW_UP
      : REVIEW_STATUS_REVIEWED;

  // Null becomes the empty option rather than a missing selection. A `<select>` with a
  // `defaultValue` matching none of its options falls back to the first one, which here would
  // silently preselect an opinion on behalf of a review that carried none.
  const selectedAssessment = review.analyst_assessment ?? ANALYST_ASSESSMENT_NONE;

  return (
    <form
      action="/admin/update-review"
      method="post"
      className="mt-6 border-t border-hair pt-6"
    >
      <input type="hidden" name="analysis_id" value={review.analysis_id} />

      <div className="flex flex-wrap items-center gap-2">
        <label className="text-[13px] text-muted" htmlFor="review-status">
          Status
        </label>
        <select
          id="review-status"
          name="status"
          defaultValue={selected}
          className="rounded-md border border-line bg-ink px-2.5 py-1.5 text-[12px] text-bone transition-colors duration-150 hover:border-rule"
        >
          <option value={REVIEW_STATUS_REVIEWED}>
            {REVIEW_STATUS_LABELS[REVIEW_STATUS_REVIEWED]}
          </option>
          <option value={REVIEW_STATUS_NEEDS_FOLLOW_UP}>
            {REVIEW_STATUS_LABELS[REVIEW_STATUS_NEEDS_FOLLOW_UP]}
          </option>
        </select>
      </div>

      <div className="mt-4 flex flex-wrap items-center gap-2">
        <label className="text-[13px] text-muted" htmlFor="review-analyst-assessment">
          Analyst assessment
        </label>
        <select
          id="review-analyst-assessment"
          name="analyst_assessment"
          defaultValue={selectedAssessment}
          className="rounded-md border border-line bg-ink px-2.5 py-1.5 text-[12px] text-bone transition-colors duration-150 hover:border-rule"
        >
          {/* First, so that "no opinion" is what the control reads as before anybody touches
              it, and so a legacy review round-trips unchanged. */}
          <option value={ANALYST_ASSESSMENT_NONE}>
            {ANALYST_ASSESSMENT_LABELS[ANALYST_ASSESSMENT_NONE]}
          </option>
          <option value={ANALYST_ASSESSMENT_AGREES}>
            {ANALYST_ASSESSMENT_LABELS[ANALYST_ASSESSMENT_AGREES]}
          </option>
          <option value={ANALYST_ASSESSMENT_DISAGREES}>
            {ANALYST_ASSESSMENT_LABELS[ANALYST_ASSESSMENT_DISAGREES]}
          </option>
          <option value={ANALYST_ASSESSMENT_UNDETERMINED}>
            {ANALYST_ASSESSMENT_LABELS[ANALYST_ASSESSMENT_UNDETERMINED]}
          </option>
        </select>
      </div>

      <p className="mt-2 max-w-[74ch] text-[12px] leading-relaxed text-muted">
        An opinion about the automated assessment, recorded alongside it and never in place of
        it. Disagreeing does not change the forensic result above, does not store a second
        classification, and is not a record of whether the detector was right.
      </p>

      <div className="mt-4">
        <label className="text-[13px] text-muted" htmlFor="review-note">
          Note
        </label>
        <textarea
          id="review-note"
          name="note"
          rows={4}
          maxLength={MAX_REVIEW_NOTE_LENGTH}
          defaultValue={review.note}
          className="mt-2 block w-full rounded-md border border-line bg-ink px-3 py-2 text-[13px] leading-relaxed text-bone transition-colors duration-150 hover:border-rule focus:border-rule focus:outline-none"
        />
        <p className="mt-2 text-[12px] text-muted">
          Plain text, at most {MAX_REVIEW_NOTE_LENGTH} characters. It is stored exactly as
          typed and shown as text — nothing here is interpreted as formatting or as a link.
        </p>
      </div>

      <button
        type="submit"
        className="mt-4 rounded-md border border-line px-3 py-1.5 text-[12px] font-medium text-muted transition-colors duration-150 hover:border-rule hover:text-bone"
      >
        Save review
      </button>
    </form>
  );
}

/** The review card: the record as it stands, and the form that changes it. */
function HumanReview({ review }: { review: AnalysisReview }) {
  return (
    // The one plate in this surface drawn in the accent tone. It is the boundary this whole
    // page is built around: everything above is what the detectors concluded, and this is what a
    // person said. The tint is a hairline and a wash, not a highlight.
    <AdminSection
      tone="accent"
      title="Human review"
      description="An operational record of whether somebody has looked at this case, and of what they made of the automated assessment — not a second answer about the media. Saving a review leaves the forensic result above exactly as it was, and every change is recorded in the audit log with who made it and when."
    >
      <ReviewRecord review={review} />
      <ReviewForm review={review} />
    </AdminSection>
  );
}

/* ------------------------------------------------------------------ *
 * The Ground Truth half — what the media is, not what anybody concluded
 * ------------------------------------------------------------------ */

/** What is stored for these bytes, or that nothing is. Notes print as text, like the review note. */
function GroundTruthRecord({ groundTruth }: { groundTruth: GroundTruth | null }) {
  if (groundTruth === null) {
    return (
      <p className="mt-4 max-w-[74ch] text-[13px] leading-relaxed text-muted">
        No Ground Truth has been recorded for this file. That is not a statement that its origin
        is unknown — nobody has made a statement about it at all.
      </p>
    );
  }

  return (
    <div className="mt-4">
      <dl className="grid grid-cols-1 gap-5 sm:grid-cols-2">
        <Fact label="Label">
          {/* Unknown values print as the API spelled them, as the review badges do. */}
          {GROUND_TRUTH_LABEL_LABELS[groundTruth.label] ?? groundTruth.label}
        </Fact>
        <Fact label="How it is known">
          {GROUND_TRUTH_SOURCE_CLASS_LABELS[groundTruth.source_class] ??
            groundTruth.source_class}
        </Fact>
        <Fact label="Recorded by">
          {groundTruth.actor_email_snapshot ?? <span className="text-muted">not recorded</span>}
          <div className="mt-1">
            <AdminValue className="break-all">{groundTruth.actor_id}</AdminValue>
          </div>
        </Fact>
        <Fact label="Last changed">
          <AdminValue className="break-all">{groundTruth.updated_at}</AdminValue>
          <div className="mt-1">
            <AdminValue className="break-all">{`first written ${groundTruth.created_at}`}</AdminValue>
          </div>
        </Fact>
      </dl>

      <div className="mt-5">
        <dt className="text-[11px] font-medium tracking-[0.08em] text-muted uppercase">Notes</dt>
        <div className="mt-2">
          {groundTruth.notes === null || groundTruth.notes === "" ? (
            <p className="text-[13px] text-muted">No notes were recorded.</p>
          ) : (
            <p className="max-w-[74ch] text-[13px] leading-relaxed whitespace-pre-wrap text-bone">
              {groundTruth.notes}
            </p>
          )}
        </div>
      </div>
    </div>
  );
}

/**
 * The form: how the label is known, what the media is, optional notes.
 *
 * **Seeded from the stored Ground Truth and from nothing else.** Not from the verdict, the
 * provenance reading or the review — their values are not passed to this component at all. With
 * no record, both selects open on an empty option and are `required`: a first option preselected
 * by default would be the form stating what the media is on somebody's behalf, and `GENUINE`
 * happens to be first.
 *
 * There is no family control. `manipulation_family` is derived from the label by the API and
 * refused if sent.
 *
 * Notes are optional and carry no `required`; an empty textarea is forwarded as no notes.
 */
function GroundTruthForm({
  analysisId,
  sha256,
  groundTruth,
}: {
  analysisId: string;
  sha256: string;
  groundTruth: GroundTruth | null;
}) {
  const selectClass =
    "rounded-md border border-line bg-ink px-2.5 py-1.5 text-[12px] text-bone transition-colors duration-150 hover:border-rule";

  return (
    <form
      action="/admin/update-ground-truth"
      method="post"
      className="mt-6 border-t border-hair pt-6"
    >
      <input type="hidden" name="analysis_id" value={analysisId} />
      <input type="hidden" name="sha256" value={sha256} />

      <div className="flex flex-wrap items-center gap-2">
        <label className="text-[13px] text-muted" htmlFor="ground-truth-label">
          Label
        </label>
        <select
          id="ground-truth-label"
          name="label"
          required
          defaultValue={groundTruth?.label ?? ""}
          className={selectClass}
        >
          <option value="" disabled>
            Choose…
          </option>
          {Object.entries(GROUND_TRUTH_LABEL_LABELS).map(([value, text]) => (
            <option key={value} value={value}>
              {text}
            </option>
          ))}
        </select>
      </div>

      <div className="mt-4 flex flex-wrap items-center gap-2">
        <label className="text-[13px] text-muted" htmlFor="ground-truth-source-class">
          How it is known
        </label>
        <select
          id="ground-truth-source-class"
          name="source_class"
          required
          defaultValue={groundTruth?.source_class ?? ""}
          className={selectClass}
        >
          <option value="" disabled>
            Choose…
          </option>
          {Object.entries(GROUND_TRUTH_SOURCE_CLASS_LABELS).map(([value, text]) => (
            <option key={value} value={value}>
              {text}
            </option>
          ))}
        </select>
      </div>

      <div className="mt-4">
        <label className="text-[13px] text-muted" htmlFor="ground-truth-notes">
          Notes <span className="text-muted">(optional)</span>
        </label>
        <textarea
          id="ground-truth-notes"
          name="notes"
          rows={3}
          defaultValue={groundTruth?.notes ?? ""}
          className="mt-2 block w-full rounded-md border border-line bg-ink px-3 py-2 text-[13px] leading-relaxed text-bone transition-colors duration-150 hover:border-rule focus:border-rule focus:outline-none"
        />
        <p className="mt-2 text-[12px] text-muted">
          Plain text, stored exactly as typed and shown as text. Leave empty for no notes.
        </p>
      </div>

      <button
        type="submit"
        className="mt-4 rounded-md border border-line px-3 py-1.5 text-[12px] font-medium text-muted transition-colors duration-150 hover:border-rule hover:text-bone"
      >
        Save Ground Truth
      </button>
    </form>
  );
}

/**
 * The Ground Truth card: its own plate, below the review, with its own outcome messages.
 *
 * With no hash the card still renders, read-only, with the one sentence that says why — hiding
 * it would leave an operator unable to tell "not recorded" from "cannot be recorded".
 */
function GroundTruthSection({
  analysisId,
  sha256,
  result,
  saved,
  error,
}: {
  analysisId: string;
  sha256: string | null;
  result: GroundTruthResult | null;
  saved: boolean;
  error: string | null;
}) {
  return (
    <AdminSection
      title="Ground Truth"
      description="What this media actually is, stated by somebody in a position to know it and recorded against the original file's SHA-256 — so it applies to every analysis of the same bytes. It is independent of the automated assessment, the provenance reading and the human review above: it is not filled in from them and saving it changes none of them."
    >
      {error && (
        <div className="mt-4">
          <AdminAlert tone="error">{error}</AdminAlert>
        </div>
      )}
      {saved && !error && (
        <div className="mt-4">
          <AdminAlert tone="success">
            Ground Truth saved. The automated assessment and the human review were not changed.
          </AdminAlert>
        </div>
      )}

      {sha256 === null ? (
        <p className="mt-4 max-w-[74ch] text-[13px] leading-relaxed text-muted">
          {NO_SHA256_FOR_GROUND_TRUTH}
        </p>
      ) : result === null || !result.ok ? (
        <div className="mt-4">
          <AdminAlert tone="error">
            {result === null ? "Ground Truth could not be read." : result.error}
          </AdminAlert>
        </div>
      ) : (
        <>
          <GroundTruthRecord groundTruth={result.groundTruth} />
          <GroundTruthForm
            analysisId={analysisId}
            sha256={sha256}
            groundTruth={result.groundTruth}
          />
        </>
      )}
    </AdminSection>
  );
}

/* ------------------------------------------------------------------ *
 * User feedback — what the analysis's owner claimed, read-only
 * ------------------------------------------------------------------ */

/** One owner's feedback. The claimed label is printed as a claim, never as a label of the media. */
function UserFeedbackItem({
  entry,
  promotionHref,
}: {
  entry: UserFeedbackEntry;
  promotionHref: string | null;
}) {
  return (
    <li className="border-t border-hair pt-4 first:border-t-0 first:pt-0">
      <dl className="grid grid-cols-1 gap-5 sm:grid-cols-2">
        <Fact label="Assessment of the result">
          {FEEDBACK_ASSESSMENT_LABELS[entry.assessment] ?? entry.assessment}
        </Fact>
        <Fact label="User's claim (unverified)">
          {entry.claimed_label === null ? (
            <span className="text-muted">no claim</span>
          ) : (
            `Claims: ${FEEDBACK_CLAIMED_LABEL_LABELS[entry.claimed_label] ?? entry.claimed_label}`
          )}
        </Fact>
        <Fact label="Submitted by">
          {entry.user_email}
          <div className="mt-1">
            <AdminValue className="break-all">{entry.user_id}</AdminValue>
          </div>
        </Fact>
        <Fact label="Last changed">
          <AdminValue className="break-all">{entry.updated_at}</AdminValue>
          <div className="mt-1">
            <AdminValue className="break-all">{`first written ${entry.created_at}`}</AdminValue>
          </div>
        </Fact>
      </dl>
      <div className="mt-4">
        {entry.notes === null ? (
          <p className="text-[13px] text-muted">No notes.</p>
        ) : (
          <p className="max-w-[74ch] text-[13px] leading-relaxed whitespace-pre-wrap text-bone">
            {entry.notes}
          </p>
        )}
      </div>
      {/* Per record, so the feedback used as context is the one the analyst chose — never "the
          first". It opens the promotion form; it saves nothing. */}
      {promotionHref !== null && (
        <Link
          href={promotionHref}
          className="mt-4 inline-block rounded-md border border-line px-3 py-1.5 text-[12px] font-medium text-muted transition-colors duration-150 hover:border-rule hover:text-bone"
        >
          Use this feedback as context for promotion
        </Link>
      )}
    </li>
  );
}

/**
 * Its own plate, after Ground Truth, with no form: administrators read user feedback and never
 * write it. It is not Ground Truth and not a review, and the card says so.
 */
function UserFeedbackSection({
  result,
  promotionHref,
}: {
  result: UserFeedbackResult;
  promotionHref: ((entry: UserFeedbackEntry) => string) | null;
}) {
  return (
    <AdminSection
      title="User feedback"
      description="What the owner of this analysis said about its result. It is an unverified claim by the submitter — not Ground Truth, not a human review, and never used to score the detectors. Giving feedback changes nothing above, and administrators cannot write or edit it."
    >
      {!result.ok ? (
        <div className="mt-4">
          <AdminAlert tone="error">{result.error}</AdminAlert>
        </div>
      ) : result.entries.length === 0 ? (
        <p className="mt-4 text-[13px] text-muted">No user feedback has been given.</p>
      ) : (
        <ul className="mt-4 space-y-4">
          {result.entries.map((entry) => (
            <UserFeedbackItem
              key={entry.user_id}
              entry={entry}
              promotionHref={promotionHref === null ? null : promotionHref(entry)}
            />
          ))}
        </ul>
      )}
    </AdminSection>
  );
}

/* ------------------------------------------------------------------ *
 * Promotion — governance, then Ground Truth, then the review (R14-T5)
 * ------------------------------------------------------------------ */

type PromotionMode = "manual" | "feedback";

/** This analysis, with the promotion panel open in one mode, scrolled to it. */
function promotionPath(analysisId: string, params: Record<string, string>): string {
  return `${adminAnalysisPath(analysisId)}?${new URLSearchParams(params)}#promotion`;
}

const PROMOTION_STEP_LABELS: Record<PromotionStep, string> = {
  governance: "dataset governance",
  ground_truth: "Ground Truth",
  review: "review status",
};

function stepList(steps: PromotionStep[]): string {
  return steps.length === 0 ? "nothing" : steps.map((step) => PROMOTION_STEP_LABELS[step]).join(", ");
}

/**
 * What the last promotion did, step by step. A failure after a save is called a partial success
 * and says so, because the saved steps are saved: three endpoints, three transactions.
 */
function PromotionOutcome({ outcome }: { outcome: ParsedOutcome }) {
  if (outcome.failed === null && outcome.error === null) {
    return (
      <AdminAlert tone="success">
        {`Saved: ${stepList(outcome.saved)}. The automated assessment and the user's feedback were not changed.`}
      </AdminAlert>
    );
  }

  if (outcome.failed === null) {
    return <AdminAlert tone="error">{`Nothing was saved. ${outcome.error ?? ""}`}</AdminAlert>;
  }

  // Every step after the one that failed: the sequence stops there.
  const notAttempted = PROMOTION_STEPS.slice(PROMOTION_STEPS.indexOf(outcome.failed) + 1);

  return (
    <AdminAlert tone="error">
      {outcome.saved.length > 0 ? "Partial success. " : "Nothing was saved. "}
      {`Saved: ${stepList(outcome.saved)}. Failed: ${PROMOTION_STEP_LABELS[outcome.failed]} — ${(outcome.error ?? "no reason given").replace(/\.$/, "")}.`}
      {notAttempted.length > 0 && ` Not attempted: ${stepList(notAttempted)}.`}
      {outcome.failed === "review"
        ? " Retry the review status below."
        : " Correct the form and submit it again; a step already saved is not written twice."}
    </AdminAlert>
  );
}

const inputClass =
  "mt-1 block w-full rounded-md border border-line bg-ink px-2.5 py-1.5 text-[12px] text-bone transition-colors duration-150 hover:border-rule focus:border-rule focus:outline-none";

function TextField({
  name,
  label,
  defaultValue,
  required = false,
  readOnly = false,
  hint,
}: {
  name: string;
  label: string;
  defaultValue: string;
  required?: boolean;
  readOnly?: boolean;
  hint?: string;
}) {
  return (
    <label className="block text-[12px] text-muted">
      {label}
      {!required && <span> (optional)</span>}
      <input
        name={name}
        type="text"
        defaultValue={defaultValue}
        required={required}
        readOnly={readOnly}
        className={inputClass}
      />
      {hint && <span className="mt-1 block text-[11px] text-muted">{hint}</span>}
    </label>
  );
}

function BooleanField({
  name,
  label,
  value,
}: {
  name: string;
  label: string;
  value: boolean | null;
}) {
  return (
    <label className="block text-[12px] text-muted">
      {label}
      <select
        name={name}
        defaultValue={value === null ? BOOLEAN_NOT_RECORDED : String(value)}
        className={inputClass}
      >
        {/* Unknown is not false: an unstated boolean stays unrecorded. */}
        <option value={BOOLEAN_NOT_RECORDED}>Not recorded</option>
        <option value="true">Yes</option>
        <option value="false">No</option>
      </select>
    </label>
  );
}

/** The review status control, opening on the current state so it is retained unless changed. */
function ReviewTargetField({ review }: { review: AnalysisReview }) {
  return (
    <>
      <input type="hidden" name="expected_review_status" value={review.status} />
      <label className="block text-[12px] text-muted">
        {`Leave the review as (currently ${REVIEW_STATUS_LABELS[review.status] ?? review.status})`}
        <select
          name="review_status"
          required
          defaultValue={defaultReviewTarget(review.status)}
          className={inputClass}
        >
          <option value="" disabled>
            Choose…
          </option>
          {REVIEW_TARGETS.map((status) => (
            <option key={status} value={status}>
              {REVIEW_STATUS_LABELS[status] ?? status}
            </option>
          ))}
        </select>
      </label>
    </>
  );
}

/**
 * The promotion form. Every control starts from the stored governance or from nothing, except the
 * label, which a chosen feedback record may prefill when its claim names a label exactly. The
 * source class and the media always start unchosen.
 */
function PromotionForm({
  analysisId,
  candidates,
  governance,
  groundTruth,
  review,
  feedback,
}: {
  analysisId: string;
  candidates: string[];
  governance: DatasetGovernance | null;
  groundTruth: GroundTruth | null;
  review: AnalysisReview;
  feedback: UserFeedbackEntry | null;
}) {
  const fieldsetClass = "mt-6 border-t border-hair pt-5";
  const legendClass = "text-[11px] font-medium tracking-[0.08em] text-muted uppercase";
  // An existing record keeps its lineage and its lineage keeps its split (409 otherwise), so
  // both are shown fixed rather than offered as a change the API would refuse.
  const fixed = governance !== null;

  return (
    <form action="/admin/promote-ground-truth" method="post" className="mt-4">
      <input type="hidden" name="analysis_id" value={analysisId} />
      <input type="hidden" name="feedback_user_id" value={feedback?.user_id ?? ""} />

      <fieldset className={fieldsetClass}>
        <legend className={legendClass}>Media</legend>
        <p className="mt-2 text-[12px] text-muted">
          Choose the file these records are about. Governance and Ground Truth are keyed by its
          SHA-256 and apply to every analysis of the same bytes.
        </p>
        {candidates.map((sha256) => (
          <label key={sha256} className="mt-2 flex items-center gap-2 text-[12px] text-bone">
            {/* Nothing preselected, even for one file: the analyst states which bytes. */}
            <input type="radio" name="sha256" value={sha256} required />
            <AdminValue className="break-all">{sha256}</AdminValue>
          </label>
        ))}
      </fieldset>

      <fieldset className={fieldsetClass}>
        <legend className={legendClass}>1 · Dataset governance</legend>
        <p className="mt-2 text-[12px] text-muted">
          Saved first, so a label never reaches evaluation without the lineage, split and licence
          it is exported under. The whole record is sent: an emptied field is cleared.
        </p>
        <div className="mt-3 grid grid-cols-1 gap-4 sm:grid-cols-2">
          <TextField
            name="source_lineage_id"
            label="Source lineage"
            required
            readOnly={fixed}
            defaultValue={governance?.source_lineage_id ?? ""}
            hint={fixed ? "Recorded; a record keeps its lineage." : undefined}
          />
          <label className="block text-[12px] text-muted">
            Dataset split
            {fixed ? (
              <>
                <input type="hidden" name="dataset_split" value={governance.dataset_split} />
                <span className={`${inputClass} text-muted`}>
                  {DATASET_SPLIT_LABELS[governance.dataset_split] ?? governance.dataset_split}
                </span>
              </>
            ) : (
              <select name="dataset_split" required defaultValue="" className={inputClass}>
                <option value="" disabled>
                  Choose…
                </option>
                {Object.entries(DATASET_SPLIT_LABELS).map(([value, text]) => (
                  <option key={value} value={value}>
                    {text}
                  </option>
                ))}
              </select>
            )}
          </label>
          <TextField name="recording_identity" label="Recording identity" defaultValue={governance?.recording_identity ?? ""} />
          <TextField name="derived_from_sha256" label="Derived from (SHA-256)" defaultValue={governance?.derived_from_sha256 ?? ""} />
          <TextField name="generation_pipeline" label="Generation pipeline" defaultValue={governance?.generation_pipeline ?? ""} />
          <TextField name="license" label="Licence" defaultValue={governance?.license ?? ""} />
          <TextField name="permission_status" label="Permission status" defaultValue={governance?.permission_status ?? ""} />
          <BooleanField name="redistributable" label="Redistributable" value={governance?.redistributable ?? null} />
          <BooleanField name="private" label="Private" value={governance?.private ?? null} />
          <TextField name="stratum_primary" label="Primary stratum" defaultValue={governance?.stratum_primary ?? ""} />
          <TextField name="source" label="Source" defaultValue={governance?.source ?? ""} />
          <TextField name="acquisition_type" label="Acquisition type" defaultValue={governance?.acquisition_type ?? ""} />
          <TextField name="benchmark_family" label="Benchmark family" defaultValue={governance?.benchmark_family ?? ""} />
        </div>
        <label className="mt-4 block text-[12px] text-muted">
          Transformations <span>(optional, one per line, in order)</span>
          <textarea
            name="transformations"
            rows={2}
            defaultValue={(governance?.transformations ?? []).join("\n")}
            className={inputClass}
          />
        </label>
      </fieldset>

      <fieldset className={fieldsetClass}>
        <legend className={legendClass}>2 · Ground Truth</legend>
        {groundTruth !== null && (
          <p className="mt-2 text-[12px] text-muted">
            {`Currently recorded: ${GROUND_TRUTH_LABEL_LABELS[groundTruth.label] ?? groundTruth.label}, ${GROUND_TRUTH_SOURCE_CLASS_LABELS[groundTruth.source_class] ?? groundTruth.source_class}. Saving replaces it.`}
          </p>
        )}
        <div className="mt-3 grid grid-cols-1 gap-4 sm:grid-cols-2">
          <label className="block text-[12px] text-muted">
            Label
            <select
              name="label"
              required
              defaultValue={feedback === null ? "" : prefilledGroundTruthLabel(feedback.claimed_label)}
              className={inputClass}
            >
              <option value="" disabled>
                Choose…
              </option>
              {Object.entries(GROUND_TRUTH_LABEL_LABELS).map(([value, text]) => (
                <option key={value} value={value}>
                  {text}
                </option>
              ))}
            </select>
          </label>
          <label className="block text-[12px] text-muted">
            How it is known
            {/* Never preselected, from feedback or otherwise. */}
            <select name="source_class" required defaultValue="" className={inputClass}>
              <option value="" disabled>
                Choose…
              </option>
              {Object.entries(GROUND_TRUTH_SOURCE_CLASS_LABELS).map(([value, text]) => (
                <option key={value} value={value}>
                  {text}
                </option>
              ))}
            </select>
          </label>
        </div>
        <p className="mt-3 max-w-[74ch] text-[12px] leading-relaxed text-muted">
          A user&apos;s claim is not provenance. Their feedback — even as the owner of the
          analysis — does not by itself make this <em>Known to the owner</em> or{" "}
          <em>Externally verified</em>. Without independent evidence, choose <em>Unknown</em>.
        </p>
        <label className="mt-3 flex items-start gap-2 text-[12px] text-bone">
          <input type="checkbox" name="provenance_attested" value="1" className="mt-0.5" />
          <span>
            The source class above rests on evidence independent of the user&apos;s feedback, the
            automated assessment and the human review, and that evidence is stated in the notes.
            (Required for every source class except Unknown.)
          </span>
        </label>
        <label className="mt-3 block text-[12px] text-muted">
          Notes <span>(the evidence, for any source class other than Unknown)</span>
          <textarea name="notes" rows={3} defaultValue="" className={inputClass} />
        </label>
      </fieldset>

      <fieldset className={fieldsetClass}>
        <legend className={legendClass}>3 · Review</legend>
        <p className="mt-2 text-[12px] text-muted">
          Set only after both records above are saved. The review&apos;s assessment and note are
          kept as they are.
        </p>
        <div className="mt-3 max-w-[40ch]">
          <ReviewTargetField review={review} />
        </div>
      </fieldset>

      <div className="mt-6 flex flex-wrap items-center gap-3">
        <button
          type="submit"
          className="rounded-md border border-line px-3 py-1.5 text-[12px] font-medium text-muted transition-colors duration-150 hover:border-rule hover:text-bone"
        >
          Save governance, Ground Truth and review
        </button>
        <Link
          href={adminAnalysisPath(analysisId)}
          className="text-[12px] text-muted underline-offset-2 hover:text-bone hover:underline"
        >
          Cancel
        </Link>
      </div>
    </form>
  );
}

/** The review step on its own, for the partial success where only the review failed. */
function ReviewRetryForm({
  analysisId,
  feedbackUserId,
  media,
  review,
}: {
  analysisId: string;
  feedbackUserId: string | null;
  media: string | null;
  review: AnalysisReview;
}) {
  return (
    <form action="/admin/promote-ground-truth" method="post" className="mt-4 max-w-[40ch]">
      <input type="hidden" name="analysis_id" value={analysisId} />
      <input type="hidden" name="feedback_user_id" value={feedbackUserId ?? ""} />
      <input type="hidden" name="retry" value="review" />
      {media !== null && <input type="hidden" name="media" value={media} />}
      <ReviewTargetField review={review} />
      <button
        type="submit"
        className="mt-3 rounded-md border border-line px-3 py-1.5 text-[12px] font-medium text-muted transition-colors duration-150 hover:border-rule hover:text-bone"
      >
        Retry the review status
      </button>
    </form>
  );
}

/** The feedback record chosen as context, shown as the claim it is. */
function FeedbackContext({ entry }: { entry: UserFeedbackEntry }) {
  const prefilled = prefilledGroundTruthLabel(entry.claimed_label);

  return (
    <div className="mt-4 rounded-md border border-line px-4 py-3 text-[12px] leading-relaxed text-muted">
      <p className="text-bone">
        {`Context: feedback from ${entry.user_email} — ${FEEDBACK_ASSESSMENT_LABELS[entry.assessment] ?? entry.assessment}, `}
        {entry.claimed_label === null
          ? "no claimed label"
          : `claims ${FEEDBACK_CLAIMED_LABEL_LABELS[entry.claimed_label] ?? entry.claimed_label}`}
        .
      </p>
      <p className="mt-1">
        {prefilled === ""
          ? "The claim does not name a Ground Truth label exactly, so the label is left for you to choose."
          : "The claim prefills the label only. It is unverified; check it before saving."}
      </p>
    </div>
  );
}

function PromotionSection({
  analysisId,
  candidates,
  media,
  governanceResult,
  groundTruthResult,
  reviewResult,
  feedbackResult,
  mode,
  feedbackUserId,
  outcome,
}: {
  analysisId: string;
  candidates: string[];
  media: string | null;
  governanceResult: DatasetGovernanceResult | null;
  groundTruthResult: GroundTruthResult | null;
  reviewResult: ReviewResult;
  feedbackResult: UserFeedbackResult;
  mode: PromotionMode | null;
  feedbackUserId: string | null;
  outcome: ParsedOutcome | null;
}) {
  const governance = governanceResult !== null && governanceResult.ok ? governanceResult.governance : null;
  const feedback =
    mode === "feedback" && feedbackResult.ok
      ? (feedbackResult.entries.find((entry) => entry.user_id === feedbackUserId) ?? null)
      : null;

  let body: React.ReactNode;
  if (candidates.length === 0) {
    body = <p className="mt-4 text-[13px] text-muted">{NO_SHA256_FOR_GROUND_TRUTH}</p>;
  } else if (mode === null || (outcome !== null && isReviewOnlyFailure(outcome))) {
    // Closed — or open on a review-only failure, where the two records are saved and the retry
    // above is the one thing left to do; a second copy of the whole form would invite re-sending it.
    body = (
      <div className="mt-4 flex flex-wrap items-center gap-3">
        <Link
          href={promotionPath(analysisId, { promote: "manual" })}
          className="rounded-md border border-line px-3 py-1.5 text-[12px] font-medium text-muted transition-colors duration-150 hover:border-rule hover:text-bone"
        >
          Promote to Ground Truth
        </Link>
        <span className="text-[12px] text-muted">
          or use one specific user feedback record above as context.
        </span>
      </div>
    );
  } else if (media === null) {
    // Several media files and none chosen yet: the form is seeded from one file's records, so
    // the file comes first. Each link reopens this panel on that file — nothing is preselected.
    const panel: Record<string, string> =
      mode === "feedback" && feedbackUserId !== null
        ? { promote: "feedback", feedback: feedbackUserId }
        : { promote: mode };
    body = (
      <div className="mt-4">
        <p className="text-[12px] text-muted">
          This analysis has more than one media file. Choose the one these records are about:
        </p>
        <ul className="mt-2 space-y-2">
          {candidates.map((sha256) => (
            <li key={sha256}>
              <Link
                href={promotionPath(analysisId, { ...panel, media: sha256 })}
                className="text-[12px] text-bone underline-offset-2 hover:underline"
              >
                <AdminValue className="break-all">{sha256}</AdminValue>
              </Link>
            </li>
          ))}
        </ul>
      </div>
    );
  } else if (governanceResult === null || !governanceResult.ok) {
    body = (
      <div className="mt-4">
        <AdminAlert tone="error">
          {governanceResult === null ? "Dataset governance could not be read." : governanceResult.error}
        </AdminAlert>
      </div>
    );
  } else if (groundTruthResult === null || !groundTruthResult.ok) {
    body = (
      <div className="mt-4">
        <AdminAlert tone="error">
          {groundTruthResult === null ? "Ground Truth could not be read." : groundTruthResult.error}
        </AdminAlert>
      </div>
    );
  } else if (!reviewResult.ok) {
    body = (
      <div className="mt-4">
        <AdminAlert tone="error">{reviewResult.error}</AdminAlert>
      </div>
    );
  } else if (mode === "feedback" && feedback === null) {
    body = (
      <div className="mt-4">
        <AdminAlert tone="error">
          That feedback record is not on this analysis. Choose one from the list below.
        </AdminAlert>
      </div>
    );
  } else {
    body = (
      <>
        {feedback !== null && <FeedbackContext entry={feedback} />}
        <PromotionForm
          analysisId={analysisId}
          candidates={[media]}
          governance={governance}
          groundTruth={groundTruthResult.groundTruth}
          review={reviewResult.review}
          feedback={feedback}
        />
      </>
    );
  }

  return (
    <div id="promotion">
      <AdminSection
        title="Dataset governance & Ground Truth promotion"
        description="Adds this case's media to the verified dataset in three separate saves, in order: dataset governance, then Ground Truth, then the review status. Each runs only if the one before it succeeded. User feedback may be used as context and never as provenance."
      >
        {media !== null && governanceResult !== null && governanceResult.ok && (
          <p className="mt-4 text-[12px] text-muted">
            {governance === null
              ? `Dataset governance of ${media.slice(0, 12)}…: not recorded.`
              : `Dataset governance of ${media.slice(0, 12)}…: lineage ${governance.source_lineage_id}, ${DATASET_SPLIT_LABELS[governance.dataset_split] ?? governance.dataset_split} split.`}
          </p>
        )}
        {outcome !== null && (
          <div className="mt-4">
            <PromotionOutcome outcome={outcome} />
            {isReviewOnlyFailure(outcome) && reviewResult.ok && (
              <ReviewRetryForm
                analysisId={analysisId}
                feedbackUserId={feedbackUserId}
                media={media}
                review={reviewResult.review}
              />
            )}
          </div>
        )}
        {body}
      </AdminSection>
    </div>
  );
}

/* ------------------------------------------------------------------ *
 * Page
 * ------------------------------------------------------------------ */

/** One query-string value, or null. A repeated parameter is not an outcome. */
function singleParam(value: string | string[] | undefined): string | null {
  return typeof value === "string" ? value : null;
}

export default async function AdminAnalysisReview({
  params,
  searchParams,
}: {
  params: Promise<{ id: string }>;
  searchParams: Promise<Record<string, string | string[] | undefined>>;
}) {
  const { id } = await params;
  const query = await searchParams;

  // Ground Truth is keyed by the original's hash, which only the analysis knows, so its read is
  // chained onto the analysis read — and runs alongside the session and review reads rather than
  // after them. No hash, or no analysis, means no read at all.
  const analysisRead = fetchAnalysis(id);
  const groundTruthRead = analysisRead.then((result) =>
    result.ok && result.analysis.original_sha256 !== null
      ? fetchGroundTruth(result.analysis.original_sha256)
      : null,
  );

  // The promotion form (R14-T5) is seeded from the governance and Ground Truth of the media it
  // is opened on, which is one of the analysis's media hashes — not necessarily the original the
  // Ground Truth card above is keyed by. Only the original's Ground Truth read is shared.
  const promotionMediaRead = analysisRead.then((result) =>
    result.ok
      ? selectedPromotionMedia(
          promotionCandidates(result.analysis.media_sha256s),
          singleParam(query.media),
        )
      : null,
  );
  const governanceRead = promotionMediaRead.then((sha256) =>
    sha256 === null ? null : fetchDatasetGovernance(sha256),
  );
  const promotionGroundTruthRead = Promise.all([analysisRead, promotionMediaRead]).then(
    ([result, sha256]) =>
      sha256 === null
        ? null
        : result.ok && sha256 === result.analysis.original_sha256
          ? groundTruthRead
          : fetchGroundTruth(sha256),
  );

  const [
    user,
    analysisResult,
    reviewResult,
    groundTruthResult,
    feedbackResult,
    promotionMedia,
    governanceResult,
    promotionGroundTruthResult,
  ] = await Promise.all([
    fetchSession(),
    analysisRead,
    fetchReview(id),
    groundTruthRead,
    fetchUserFeedback(id),
    promotionMediaRead,
    governanceRead,
    promotionGroundTruthRead,
  ]);

  // The session the API would not accept. `admin/layout.tsx` has already turned away a reader
  // with no session and one whose role is not administrator, so what is left for this is the
  // narrow case of a session that expired between the guard and these reads. It is not the
  // access control: the API refused both of them before this line.
  if (
    user === null ||
    (!analysisResult.ok && analysisResult.unauthenticated) ||
    (!reviewResult.ok && reviewResult.unauthenticated) ||
    (groundTruthResult !== null && !groundTruthResult.ok && groundTruthResult.unauthenticated) ||
    (!feedbackResult.ok && feedbackResult.unauthenticated) ||
    (governanceResult !== null && !governanceResult.ok && governanceResult.unauthenticated) ||
    (promotionGroundTruthResult !== null &&
      !promotionGroundTruthResult.ok &&
      promotionGroundTruthResult.unauthenticated)
  ) {
    redirect(LOGIN_PATH);
  }

  // An id that names no analysis. Either read is enough to establish it, and both are checked
  // because a 404 from one and a 200 from the other would mean the two endpoints disagree about
  // whether the record exists — in which case the honest answer is still "not found".
  if (
    (!analysisResult.ok && analysisResult.missing) ||
    (!reviewResult.ok && reviewResult.missing)
  ) {
    notFound();
  }

  const error = singleParam(query.error);
  const saved = singleParam(query.saved);
  const groundTruthError = singleParam(query.gt_error);
  const groundTruthSaved = singleParam(query.gt_saved) !== null;

  // The promotion panel: closed, manual, or opened from one specific feedback record.
  const promoteParam = singleParam(query.promote);
  const promotionMode: PromotionMode | null =
    promoteParam === "manual" || promoteParam === "feedback" ? promoteParam : null;
  const promotionFeedbackUserId = singleParam(query.feedback);
  const promotionOutcome = parseOutcome((name) => singleParam(query[name]));

  return (
    <>
      <AdminPageHeader
        title="Case review"
        description={
          <>
            <p>
              One analysis, and what has been said about it. The automated assessment and the
              provenance reading below it are the forensic record — what the detectors concluded
              and what the file itself carries — and neither can be edited from anywhere in this
              application. The human review is a separate record in a separate table: what a
              reviewer noted, revisable at any time, and never a second answer about the media.
            </p>
            <p>
              <AdminValue className="break-all">{id}</AdminValue>
            </p>
          </>
        }
        actions={
          // A reload of this page. The review is state another administrator can change, and
          // there is no JavaScript here to notice when they do. The links back to the queue and
          // the account list are the sidebar now — this screen is not in it, because there is no
          // reviews page to land on, only the review of a particular analysis.
          <Link
            href={adminAnalysisPath(id)}
            className="rounded-md border border-line px-3 py-1.5 text-[12px] font-medium text-muted transition-colors duration-150 hover:border-rule hover:text-bone"
          >
            Refresh
          </Link>
        }
      />

      {/* The outcome of the last save, read out of the query string because the form is a plain
          form and the answer arrives as a redirect. The error text is the API's own
          client-facing sentence, rendered as text. */}
      {error && (
        <div className="mt-5">
          <AdminAlert tone="error">{error}</AdminAlert>
        </div>
      )}
      {saved !== null && !error && (
        <div className="mt-5">
          <AdminAlert tone="success">
            Review saved. The forensic result was not changed.
          </AdminAlert>
        </div>
      )}

      {/* The hierarchy the report is built on, at this surface's density (R9-T8): the automated
          assessment with its coverage, then authenticity and provenance as a separate question,
          then what a person made of the case. The three are three cards and are never merged —
          an operator has to be able to say which of the statements in front of them came from a
          detector, which from the file itself, and which from a colleague. */}
      <div className="mt-5 space-y-5">
        {!analysisResult.ok ? (
          <AdminAlert tone="error">{analysisResult.error}</AdminAlert>
        ) : (
          <>
            <ForensicResult analysis={analysisResult.analysis} />
            <ProvenanceRecord signal={analysisResult.analysis.provenance} />
          </>
        )}

        {!reviewResult.ok ? (
          <AdminAlert tone="error">{reviewResult.error}</AdminAlert>
        ) : (
          <HumanReview review={reviewResult.review} />
        )}

        {/* Last, and its own plate: what the media is, after everything that was concluded about
            it. Only drawn when the analysis was read — without it there is no knowing whether
            a hash exists, and the no-hash sentence would be a claim this page cannot make. */}
        {analysisResult.ok && (
          <GroundTruthSection
            analysisId={id}
            sha256={analysisResult.analysis.original_sha256}
            result={groundTruthResult}
            saved={groundTruthSaved}
            error={groundTruthError}
          />
        )}

        {/* After Ground Truth and apart from it: what the submitter claimed. Read-only. */}
        {analysisResult.ok && (
          <UserFeedbackSection
            result={feedbackResult}
            promotionHref={
              analysisResult.analysis.media_sha256s.length === 0
                ? null
                : (entry) => promotionPath(id, { promote: "feedback", feedback: entry.user_id })
            }
          />
        )}

        {/* Last: the one workflow on this page that writes more than one record — each through
            its own endpoint, in order (R14-T5). */}
        {analysisResult.ok && (
          <PromotionSection
            analysisId={id}
            candidates={promotionCandidates(analysisResult.analysis.media_sha256s)}
            media={promotionMedia}
            governanceResult={governanceResult}
            groundTruthResult={promotionGroundTruthResult}
            reviewResult={reviewResult}
            feedbackResult={feedbackResult}
            mode={promotionMode}
            feedbackUserId={promotionFeedbackUserId}
            outcome={promotionOutcome}
          />
        )}
      </div>
    </>
  );
}
