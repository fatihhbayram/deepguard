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
import { CanonicalLabel, CanonicalOption, Copy, MessageOption, T } from "../../../i18n/client";
import { LOGIN_PATH, adminAnalysisPath } from "../../../session";
import { AdminAlert } from "../../components/AdminAlert";
import { AdminPageHeader } from "../../components/AdminPageHeader";
import { AdminSection } from "../../components/AdminSection";
import { AdminStatusBadge } from "../../components/AdminStatusBadge";
import { AdminValue } from "../../components/AdminValue";
import {
  ANALYST_ASSESSMENT_AGREES,
  ANALYST_ASSESSMENT_DISAGREES,
  ANALYST_ASSESSMENT_NONE,
  ANALYST_ASSESSMENT_UNDETERMINED,
  AnalysisReview,
  MAX_REVIEW_NOTE_LENGTH,
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
// The sentence the Ground Truth card shows in place of its form when the analysis carries no
// hash of its original (`admin.case.gt.noSha`): Ground Truth is keyed by those bytes; without
// them there is nothing to attach it to, and the card says so rather than disappearing.
//
// What the risk column says when no decision was ever taken (`admin.case.noDecision`) is distinct
// from `UNKNOWN`, which is a decision, and phrased as a statement about the record rather than
// about the media — the same wording the report uses for the same reason.
//
// Every label table this page draws from (`REVIEW_STATUS_LABELS`, `GROUND_TRUTH_*_LABELS`, the
// feedback words) is named in the reader's language by a `CANONICAL` domain in `../../../i18n/core`;
// the tables here still say which values a select offers, and the API's own spelling is what
// every option posts (R16 admin localization).

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
function Fact({ label, children }: { label: React.ReactNode; children: React.ReactNode }) {
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
      title={<T k="admin.case.automated.title" />}
      description={<T k="admin.case.automated.description" />}
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
        <Fact label={<T k={legacyLevel ? "admin.queue.recordedRiskLevel" : "admin.queue.decision"} />}>
          {/* The verdict's own title, in the words the report prints it in and through the same
              translation table (`REPORT_COPY`), so the two screens cannot drift apart. */}
          {level === null ? (
            <T k="admin.case.noDecision" />
          ) : (
            <Copy text={classificationLabel(level, analysis.risk_rules_version)} />
          )}
        </Fact>
        {coverage !== null && (
          <Fact label={<T k="admin.case.coverage" />}>
            {/* Three API fields interpolated, including the word. `complete` is never reached
                here by comparing `usable` against `total`: that comparison is the coverage model
                and it lives in one place, on the other side of the API (R9-T4). */}
            <AdminValue>
              {`${coverage.usable}/${coverage.total} `}
              <CanonicalLabel domain="coverageStatus" value={coverage.status} />
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
          <Copy text={wording.clarification} />
        </p>
      )}

      <dl className="mt-5 grid grid-cols-1 gap-5 border-t border-hair pt-5 sm:grid-cols-2">
        <Fact label={<T k="admin.case.analysisStatus" />}>
          <CanonicalLabel domain="analysisStatus" value={analysis.status} />
        </Fact>
        <Fact label={<T k="admin.case.ruleset" />}>
          <AdminValue className="break-all">{analysis.risk_rules_version}</AdminValue>
        </Fact>
        <Fact label={<T k="admin.case.ruleFired" />}>
          <AdminValue className="break-all">{analysis.risk_rule_id}</AdminValue>
        </Fact>
        <Fact label={<T k="admin.case.calibration" />}>
          <AdminValue className="break-all">{analysis.risk_calibration_id}</AdminValue>
        </Fact>
        <Fact label={<T k="admin.case.submitted" />}>
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
      title={<T k="admin.case.provenance.title" />}
      description={<T k="admin.case.provenance.description" />}
    >
      {signal === null ? (
        <p className="mt-4 max-w-[74ch] text-[13px] leading-relaxed text-muted">
          <T k="admin.case.provenance.none" />
        </p>
      ) : (
        <>
          {wording !== null && (
            <div className="mt-4">
              {/* The pair, as the pair it is. Neither axis is ever printed without the other. */}
              {/* The two axes as the report names them; `lang="en"` for a raw one this build
                  does not know, which is drawn as the API spelled it under `uppercase`. */}
              <p lang="en" className="font-mono text-[11px] tracking-[0.08em] text-muted uppercase">
                {signal.provenance_status === null ? null : (
                  <CanonicalLabel domain="provenanceStatus" value={signal.provenance_status} />
                )}{" "}
                ·{" "}
                {signal.provenance_availability === null ? null : (
                  <CanonicalLabel
                    domain="provenanceAvailability"
                    value={signal.provenance_availability}
                  />
                )}
              </p>
              <p className="mt-1.5 text-[15px] font-semibold text-bone">
                <Copy text={wording.title} />
              </p>
              <p className="mt-1.5 max-w-[74ch] text-[13px] leading-relaxed text-muted">
                <Copy text={wording.meaning} />
              </p>
              <p className="mt-2 max-w-[74ch] text-[13px] leading-relaxed text-muted">
                <Copy text={wording.clarification} />
              </p>
            </div>
          )}

          <dl
            className={`grid grid-cols-1 gap-5 sm:grid-cols-2 ${
              wording === null ? "mt-4" : "mt-5 border-t border-hair pt-5"
            }`}
          >
            <Fact label={<T k="admin.case.provenance.readingStatus" />}>
              <AdminValue className="break-all">
                <CanonicalLabel domain="signalStatus" value={signal.status} />
              </AdminValue>
            </Fact>
            <Fact label={<T k="admin.case.provenance.manifestPresent" />}>
              <T
                k={
                  signal.manifest_exists === null
                    ? "admin.case.provenance.manifestUnknown"
                    : signal.manifest_exists
                      ? "admin.common.yes"
                      : "admin.common.no"
                }
              />
            </Fact>
            <Fact label={<T k="admin.case.provenance.validationState" />}>
              <AdminValue className="break-all">{signal.validation_state}</AdminValue>
            </Fact>
            <Fact label={<T k="admin.case.provenance.claimGenerator" />}>
              <AdminValue className="break-all">{signal.claim_generator}</AdminValue>
            </Fact>
            <Fact label={<T k="admin.case.provenance.signatureIssuer" />}>
              <AdminValue className="break-all">{signal.signature_issuer}</AdminValue>
            </Fact>
            <Fact label={<T k="admin.case.provenance.remoteManifest" />}>
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
      <CanonicalLabel domain="reviewStatus" value={status} />
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
      <CanonicalLabel domain="analystAssessment" value={assessment ?? ANALYST_ASSESSMENT_NONE} />
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
      <p className="text-[13px] text-muted">
        <T k="admin.case.review.noNote" />
      </p>
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
          <T k="admin.case.review.nobody" />
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
          <span className="text-[13px] text-muted">
            <T k="admin.case.review.title" />
          </span>
          <StatusBadge status={review.status} />
        </span>
        <span className="flex flex-wrap items-center gap-2">
          <span className="text-[13px] text-muted">
            <T k="admin.case.review.assessment" />
          </span>
          <AssessmentBadge assessment={review.analyst_assessment} />
        </span>
      </div>

      {review.analyst_assessment === ANALYST_ASSESSMENT_DISAGREES && (
        <p className="mt-3 max-w-[74ch] text-[13px] leading-relaxed text-muted">
          <T k="admin.case.review.disagrees" />
        </p>
      )}

      {review.analyst_assessment === null && (
        <p className="mt-3 max-w-[74ch] text-[13px] leading-relaxed text-muted">
          <T k="admin.case.review.noAssessment" />
        </p>
      )}

      <dl className="mt-4 grid grid-cols-1 gap-5 sm:grid-cols-2">
        <Fact label={<T k="admin.case.review.reviewer" />}>
          {/* The address as it read when the review was last written, never looked up now. A
              rename does not change who signed a past review — see `admin/reviews.ts`. */}
          {review.reviewer_email_snapshot ?? (
            <span className="text-muted">
              <T k="admin.common.notRecorded" />
            </span>
          )}
          <div className="mt-1">
            <AdminValue className="break-all">{review.reviewer_id}</AdminValue>
          </div>
        </Fact>
        <Fact label={<T k="admin.case.lastChanged" />}>
          <AdminValue className="break-all">{review.updated_at}</AdminValue>
          <div className="mt-1">
            <AdminValue className="break-all">
              {review.created_at === null ? null : (
                <T k="admin.case.firstWritten" values={{ at: review.created_at }} />
              )}
            </AdminValue>
          </div>
        </Fact>
      </dl>

      <div className="mt-5">
        <dt className="text-[11px] font-medium tracking-[0.08em] text-muted uppercase">
          <T k="admin.case.review.note" />
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
          <T k="admin.case.review.status" />
        </label>
        <select
          id="review-status"
          name="status"
          defaultValue={selected}
          className="rounded-md border border-line bg-ink px-2.5 py-1.5 text-[12px] text-bone transition-colors duration-150 hover:border-rule"
        >
          <CanonicalOption domain="reviewStatus" value={REVIEW_STATUS_REVIEWED} />
          <CanonicalOption domain="reviewStatus" value={REVIEW_STATUS_NEEDS_FOLLOW_UP} />
        </select>
      </div>

      <div className="mt-4 flex flex-wrap items-center gap-2">
        <label className="text-[13px] text-muted" htmlFor="review-analyst-assessment">
          <T k="admin.case.review.assessment" />
        </label>
        <select
          id="review-analyst-assessment"
          name="analyst_assessment"
          defaultValue={selectedAssessment}
          className="rounded-md border border-line bg-ink px-2.5 py-1.5 text-[12px] text-bone transition-colors duration-150 hover:border-rule"
        >
          {/* First, so that "no opinion" is what the control reads as before anybody touches
              it, and so a legacy review round-trips unchanged. */}
          <CanonicalOption domain="analystAssessment" value={ANALYST_ASSESSMENT_NONE} />
          <CanonicalOption domain="analystAssessment" value={ANALYST_ASSESSMENT_AGREES} />
          <CanonicalOption domain="analystAssessment" value={ANALYST_ASSESSMENT_DISAGREES} />
          <CanonicalOption domain="analystAssessment" value={ANALYST_ASSESSMENT_UNDETERMINED} />
        </select>
      </div>

      <p className="mt-2 max-w-[74ch] text-[12px] leading-relaxed text-muted">
        <T k="admin.case.review.opinion" />
      </p>

      <div className="mt-4">
        <label className="text-[13px] text-muted" htmlFor="review-note">
          <T k="admin.case.review.note" />
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
          <T k="admin.case.review.noteHelp" values={{ count: MAX_REVIEW_NOTE_LENGTH }} />
        </p>
      </div>

      <button
        type="submit"
        className="mt-4 rounded-md border border-line px-3 py-1.5 text-[12px] font-medium text-muted transition-colors duration-150 hover:border-rule hover:text-bone"
      >
        <T k="admin.case.review.save" />
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
      title={<T k="admin.case.review.title" />}
      description={<T k="admin.case.review.description" />}
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
        <T k="admin.case.gt.none" />
      </p>
    );
  }

  return (
    <div className="mt-4">
      <dl className="grid grid-cols-1 gap-5 sm:grid-cols-2">
        <Fact label={<T k="admin.case.gt.label" />}>
          {/* Unknown values print as the API spelled them, as the review badges do. */}
          <CanonicalLabel domain="groundTruthLabel" value={groundTruth.label} />
        </Fact>
        <Fact label={<T k="admin.case.gt.sourceClass" />}>
          <CanonicalLabel domain="groundTruthSourceClass" value={groundTruth.source_class} />
        </Fact>
        <Fact label={<T k="admin.case.gt.recordedBy" />}>
          {groundTruth.actor_email_snapshot ?? (
            <span className="text-muted">
              <T k="admin.common.notRecorded" />
            </span>
          )}
          <div className="mt-1">
            <AdminValue className="break-all">{groundTruth.actor_id}</AdminValue>
          </div>
        </Fact>
        <Fact label={<T k="admin.case.lastChanged" />}>
          <AdminValue className="break-all">{groundTruth.updated_at}</AdminValue>
          <div className="mt-1">
            <AdminValue className="break-all">
              <T k="admin.case.firstWritten" values={{ at: groundTruth.created_at }} />
            </AdminValue>
          </div>
        </Fact>
      </dl>

      <div className="mt-5">
        <dt className="text-[11px] font-medium tracking-[0.08em] text-muted uppercase">
          <T k="admin.case.gt.notes" />
        </dt>
        <div className="mt-2">
          {groundTruth.notes === null || groundTruth.notes === "" ? (
            <p className="text-[13px] text-muted">
              <T k="admin.case.gt.noNotes" />
            </p>
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
          <T k="admin.case.gt.label" />
        </label>
        <select
          id="ground-truth-label"
          name="label"
          required
          defaultValue={groundTruth?.label ?? ""}
          className={selectClass}
        >
          <MessageOption value="" k="admin.common.choose" disabled />
          {Object.keys(GROUND_TRUTH_LABEL_LABELS).map((value) => (
            <CanonicalOption key={value} domain="groundTruthLabel" value={value} />
          ))}
        </select>
      </div>

      <div className="mt-4 flex flex-wrap items-center gap-2">
        <label className="text-[13px] text-muted" htmlFor="ground-truth-source-class">
          <T k="admin.case.gt.sourceClass" />
        </label>
        <select
          id="ground-truth-source-class"
          name="source_class"
          required
          defaultValue={groundTruth?.source_class ?? ""}
          className={selectClass}
        >
          <MessageOption value="" k="admin.common.choose" disabled />
          {Object.keys(GROUND_TRUTH_SOURCE_CLASS_LABELS).map((value) => (
            <CanonicalOption key={value} domain="groundTruthSourceClass" value={value} />
          ))}
        </select>
      </div>

      <div className="mt-4">
        <label className="text-[13px] text-muted" htmlFor="ground-truth-notes">
          <T k="admin.case.gt.notes" />{" "}
          <span className="text-muted">
            <T k="admin.common.optional" />
          </span>
        </label>
        <textarea
          id="ground-truth-notes"
          name="notes"
          rows={3}
          defaultValue={groundTruth?.notes ?? ""}
          className="mt-2 block w-full rounded-md border border-line bg-ink px-3 py-2 text-[13px] leading-relaxed text-bone transition-colors duration-150 hover:border-rule focus:border-rule focus:outline-none"
        />
        <p className="mt-2 text-[12px] text-muted">
          <T k="admin.case.gt.notesHelp" />
        </p>
      </div>

      <button
        type="submit"
        className="mt-4 rounded-md border border-line px-3 py-1.5 text-[12px] font-medium text-muted transition-colors duration-150 hover:border-rule hover:text-bone"
      >
        <T k="admin.case.gt.save" />
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
      title={<T k="admin.case.gt.title" />}
      description={<T k="admin.case.gt.description" />}
    >
      {error && (
        <div className="mt-4">
          <AdminAlert tone="error">
            <CanonicalLabel domain="webMessage" value={error} />
          </AdminAlert>
        </div>
      )}
      {saved && !error && (
        <div className="mt-4">
          <AdminAlert tone="success">
            <T k="admin.case.gt.saved" />
          </AdminAlert>
        </div>
      )}

      {sha256 === null ? (
        <p className="mt-4 max-w-[74ch] text-[13px] leading-relaxed text-muted">
          <T k="admin.case.gt.noSha" />
        </p>
      ) : result === null || !result.ok ? (
        <div className="mt-4">
          <AdminAlert tone="error">
            <CanonicalLabel
              domain="webMessage"
              value={result === null ? "Ground Truth could not be read." : result.error}
            />
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
        <Fact label={<T k="admin.case.feedback.assessment" />}>
          <CanonicalLabel domain="feedbackAssessment" value={entry.assessment} />
        </Fact>
        <Fact label={<T k="admin.case.feedback.claim" />}>
          {entry.claimed_label === null ? (
            <span className="text-muted">
              <T k="admin.case.feedback.noClaim" />
            </span>
          ) : (
            <T
              k="admin.case.feedback.claims"
              values={{
                label: <CanonicalLabel domain="feedbackClaimedLabel" value={entry.claimed_label} />,
              }}
            />
          )}
        </Fact>
        <Fact label={<T k="admin.case.feedback.submittedBy" />}>
          {entry.user_email}
          <div className="mt-1">
            <AdminValue className="break-all">{entry.user_id}</AdminValue>
          </div>
        </Fact>
        <Fact label={<T k="admin.case.lastChanged" />}>
          <AdminValue className="break-all">{entry.updated_at}</AdminValue>
          <div className="mt-1">
            <AdminValue className="break-all">
              <T k="admin.case.firstWritten" values={{ at: entry.created_at }} />
            </AdminValue>
          </div>
        </Fact>
      </dl>
      <div className="mt-4">
        {entry.notes === null ? (
          <p className="text-[13px] text-muted">
            <T k="admin.case.feedback.noNotes" />
          </p>
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
          <T k="admin.case.feedback.useAsContext" />
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
      title={<T k="admin.case.feedback.title" />}
      description={<T k="admin.case.feedback.description" />}
    >
      {!result.ok ? (
        <div className="mt-4">
          <AdminAlert tone="error">
            <CanonicalLabel domain="webMessage" value={result.error} />
          </AdminAlert>
        </div>
      ) : result.entries.length === 0 ? (
        <p className="mt-4 text-[13px] text-muted">
          <T k="admin.case.feedback.empty" />
        </p>
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

const PROMOTION_STEP_LABELS = {
  governance: "admin.case.promotion.step.governance",
  ground_truth: "admin.case.promotion.step.groundTruth",
  review: "admin.case.promotion.step.review",
} as const satisfies Record<PromotionStep, string>;

function StepList({ steps }: { steps: PromotionStep[] }) {
  if (steps.length === 0) {
    return <T k="admin.case.promotion.nothing" />;
  }
  return steps.map((step, index) => (
    <span key={step}>
      {index > 0 && ", "}
      <T k={PROMOTION_STEP_LABELS[step]} />
    </span>
  ));
}

/**
 * Why a step failed: the route handler's sentence, named in the reader's language when it is one
 * of this application's own and drawn as it arrived when it is the API's. It is closed with a full
 * stop when it does not carry one, as the English outcome always was.
 */
function FailureReason({ error }: { error: string | null }) {
  if (error === null) {
    return <T k="admin.case.promotion.noReason" />;
  }
  return (
    <>
      <CanonicalLabel domain="webMessage" value={error} />
      {error.endsWith(".") ? "" : "."}
    </>
  );
}

/**
 * What the last promotion did, step by step. A failure after a save is called a partial success
 * and says so, because the saved steps are saved: three endpoints, three transactions.
 */
function PromotionOutcome({ outcome }: { outcome: ParsedOutcome }) {
  if (outcome.failed === null && outcome.error === null) {
    return (
      <AdminAlert tone="success">
        <T k="admin.case.promotion.saved" values={{ steps: <StepList steps={outcome.saved} /> }} />
      </AdminAlert>
    );
  }

  if (outcome.failed === null) {
    return (
      <AdminAlert tone="error">
        <T k="admin.case.promotion.nothingSaved" />{" "}
        {outcome.error === null ? null : <CanonicalLabel domain="webMessage" value={outcome.error} />}
      </AdminAlert>
    );
  }

  // Every step after the one that failed: the sequence stops there.
  const notAttempted = PROMOTION_STEPS.slice(PROMOTION_STEPS.indexOf(outcome.failed) + 1);

  return (
    <AdminAlert tone="error">
      <T k={outcome.saved.length > 0 ? "admin.case.promotion.partial" : "admin.case.promotion.nothingSaved"} />{" "}
      <T
        k="admin.case.promotion.failed"
        values={{
          steps: <StepList steps={outcome.saved} />,
          step: <T k={PROMOTION_STEP_LABELS[outcome.failed]} />,
          reason: <FailureReason error={outcome.error} />,
        }}
      />
      {notAttempted.length > 0 && (
        <>
          {" "}
          <T k="admin.case.promotion.notAttempted" values={{ steps: <StepList steps={notAttempted} /> }} />
        </>
      )}{" "}
      <T k={outcome.failed === "review" ? "admin.case.promotion.retryBelow" : "admin.case.promotion.correctForm"} />
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
  label: React.ReactNode;
  defaultValue: string;
  required?: boolean;
  readOnly?: boolean;
  hint?: React.ReactNode;
}) {
  return (
    <label className="block text-[12px] text-muted">
      {label}
      {!required && (
        <span>
          {" "}
          <T k="admin.common.optional" />
        </span>
      )}
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
  label: React.ReactNode;
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
        <MessageOption value={BOOLEAN_NOT_RECORDED} k="admin.common.notRecordedOption" />
        <MessageOption value="true" k="admin.common.yesOption" />
        <MessageOption value="false" k="admin.common.noOption" />
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
        <T
          k="admin.case.promotion.leaveReviewAs"
          values={{ current: <CanonicalLabel domain="reviewStatus" value={review.status} /> }}
        />
        <select
          name="review_status"
          required
          defaultValue={defaultReviewTarget(review.status)}
          className={inputClass}
        >
          <MessageOption value="" k="admin.common.choose" disabled />
          {REVIEW_TARGETS.map((status) => (
            <CanonicalOption key={status} domain="reviewStatus" value={status} />
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
        <legend className={legendClass}>
          <T k="admin.case.promotion.media" />
        </legend>
        <p className="mt-2 text-[12px] text-muted">
          <T k="admin.case.promotion.mediaHelp" />
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
        <legend className={legendClass}>
          1 · <T k="admin.case.promotion.governance" />
        </legend>
        <p className="mt-2 text-[12px] text-muted">
          <T k="admin.case.promotion.governanceHelp" />
        </p>
        <div className="mt-3 grid grid-cols-1 gap-4 sm:grid-cols-2">
          <TextField
            name="source_lineage_id"
            label={<T k="admin.case.promotion.field.sourceLineage" />}
            required
            readOnly={fixed}
            defaultValue={governance?.source_lineage_id ?? ""}
            hint={fixed ? <T k="admin.case.promotion.lineageKept" /> : undefined}
          />
          <label className="block text-[12px] text-muted">
            <T k="admin.case.promotion.field.datasetSplit" />
            {fixed ? (
              <>
                <input type="hidden" name="dataset_split" value={governance.dataset_split} />
                <span className={`${inputClass} text-muted`}>
                  <CanonicalLabel domain="datasetSplit" value={governance.dataset_split} />
                </span>
              </>
            ) : (
              <select name="dataset_split" required defaultValue="" className={inputClass}>
                <MessageOption value="" k="admin.common.choose" disabled />
                {Object.keys(DATASET_SPLIT_LABELS).map((value) => (
            <CanonicalOption key={value} domain="datasetSplit" value={value} />
          ))}
              </select>
            )}
          </label>
          <TextField name="recording_identity" label={<T k="admin.case.promotion.field.recordingIdentity" />} defaultValue={governance?.recording_identity ?? ""} />
          <TextField name="derived_from_sha256" label={<T k="admin.case.promotion.field.derivedFrom" />} defaultValue={governance?.derived_from_sha256 ?? ""} />
          <TextField name="generation_pipeline" label={<T k="admin.case.promotion.field.generationPipeline" />} defaultValue={governance?.generation_pipeline ?? ""} />
          <TextField name="license" label={<T k="admin.case.promotion.field.license" />} defaultValue={governance?.license ?? ""} />
          <TextField name="permission_status" label={<T k="admin.case.promotion.field.permissionStatus" />} defaultValue={governance?.permission_status ?? ""} />
          <BooleanField name="redistributable" label={<T k="admin.case.promotion.field.redistributable" />} value={governance?.redistributable ?? null} />
          <BooleanField name="private" label={<T k="admin.case.promotion.field.private" />} value={governance?.private ?? null} />
          <TextField name="stratum_primary" label={<T k="admin.case.promotion.field.stratumPrimary" />} defaultValue={governance?.stratum_primary ?? ""} />
          <TextField name="source" label={<T k="admin.case.promotion.field.source" />} defaultValue={governance?.source ?? ""} />
          <TextField name="acquisition_type" label={<T k="admin.case.promotion.field.acquisitionType" />} defaultValue={governance?.acquisition_type ?? ""} />
          <TextField name="benchmark_family" label={<T k="admin.case.promotion.field.benchmarkFamily" />} defaultValue={governance?.benchmark_family ?? ""} />
        </div>
        <label className="mt-4 block text-[12px] text-muted">
          <T k="admin.case.promotion.field.transformations" />{" "}
          <span>
            <T k="admin.case.promotion.transformationsHelp" />
          </span>
          <textarea
            name="transformations"
            rows={2}
            defaultValue={(governance?.transformations ?? []).join("\n")}
            className={inputClass}
          />
        </label>
      </fieldset>

      <fieldset className={fieldsetClass}>
        <legend className={legendClass}>
          2 · <T k="admin.case.promotion.step.groundTruth" />
        </legend>
        {groundTruth !== null && (
          <p className="mt-2 text-[12px] text-muted">
            <T
              k="admin.case.promotion.currentlyRecorded"
              values={{
                label: <CanonicalLabel domain="groundTruthLabel" value={groundTruth.label} />,
                source: (
                  <CanonicalLabel domain="groundTruthSourceClass" value={groundTruth.source_class} />
                ),
              }}
            />
          </p>
        )}
        <div className="mt-3 grid grid-cols-1 gap-4 sm:grid-cols-2">
          <label className="block text-[12px] text-muted">
            <T k="admin.case.gt.label" />
            <select
              name="label"
              required
              defaultValue={feedback === null ? "" : prefilledGroundTruthLabel(feedback.claimed_label)}
              className={inputClass}
            >
              <MessageOption value="" k="admin.common.choose" disabled />
              {Object.keys(GROUND_TRUTH_LABEL_LABELS).map((value) => (
            <CanonicalOption key={value} domain="groundTruthLabel" value={value} />
          ))}
            </select>
          </label>
          <label className="block text-[12px] text-muted">
            <T k="admin.case.gt.sourceClass" />
            {/* Never preselected, from feedback or otherwise. */}
            <select name="source_class" required defaultValue="" className={inputClass}>
              <MessageOption value="" k="admin.common.choose" disabled />
              {Object.keys(GROUND_TRUTH_SOURCE_CLASS_LABELS).map((value) => (
            <CanonicalOption key={value} domain="groundTruthSourceClass" value={value} />
          ))}
            </select>
          </label>
        </div>
        <p className="mt-3 max-w-[74ch] text-[12px] leading-relaxed text-muted">
          <T
            k="admin.case.promotion.claimNotProvenance"
            values={{
              owner: (
                <em>
                  <CanonicalLabel domain="groundTruthSourceClass" value="OWNER_KNOWN" />
                </em>
              ),
              external: (
                <em>
                  <CanonicalLabel domain="groundTruthSourceClass" value="EXTERNAL_VERIFIED" />
                </em>
              ),
              unknown: (
                <em>
                  <CanonicalLabel domain="groundTruthSourceClass" value="UNKNOWN" />
                </em>
              ),
            }}
          />
        </p>
        <label className="mt-3 flex items-start gap-2 text-[12px] text-bone">
          <input type="checkbox" name="provenance_attested" value="1" className="mt-0.5" />
          <span>
            <T k="admin.case.promotion.attest" />
          </span>
        </label>
        <label className="mt-3 block text-[12px] text-muted">
          <T k="admin.case.gt.notes" />{" "}
          <span>
            <T k="admin.case.promotion.notesHelp" />
          </span>
          <textarea name="notes" rows={3} defaultValue="" className={inputClass} />
        </label>
      </fieldset>

      <fieldset className={fieldsetClass}>
        <legend className={legendClass}>
          3 · <T k="admin.case.promotion.reviewLegend" />
        </legend>
        <p className="mt-2 text-[12px] text-muted">
          <T k="admin.case.promotion.reviewHelp" />
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
          <T k="admin.case.promotion.submit" />
        </button>
        <Link
          href={adminAnalysisPath(analysisId)}
          className="text-[12px] text-muted underline-offset-2 hover:text-bone hover:underline"
        >
          <T k="admin.common.cancel" />
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
        <T k="admin.case.promotion.retry" />
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
        <T
          k={entry.claimed_label === null ? "admin.case.context.noClaim" : "admin.case.context.claims"}
          values={{
            email: entry.user_email,
            assessment: <CanonicalLabel domain="feedbackAssessment" value={entry.assessment} />,
            label:
              entry.claimed_label === null ? null : (
                <CanonicalLabel domain="feedbackClaimedLabel" value={entry.claimed_label} />
              ),
          }}
        />
      </p>
      <p className="mt-1">
        <T k={prefilled === "" ? "admin.case.context.notPrefilled" : "admin.case.context.prefilled"} />
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
    body = (
      <p className="mt-4 text-[13px] text-muted">
        <T k="admin.case.gt.noSha" />
      </p>
    );
  } else if (mode === null || (outcome !== null && isReviewOnlyFailure(outcome))) {
    // Closed — or open on a review-only failure, where the two records are saved and the retry
    // above is the one thing left to do; a second copy of the whole form would invite re-sending it.
    body = (
      <div className="mt-4 flex flex-wrap items-center gap-3">
        <Link
          href={promotionPath(analysisId, { promote: "manual" })}
          className="rounded-md border border-line px-3 py-1.5 text-[12px] font-medium text-muted transition-colors duration-150 hover:border-rule hover:text-bone"
        >
          <T k="admin.case.promotion.promote" />
        </Link>
        <span className="text-[12px] text-muted">
          <T k="admin.case.promotion.orFeedback" />
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
          <T k="admin.case.promotion.chooseMedia" />
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
          <CanonicalLabel
            domain="webMessage"
            value={governanceResult === null ? "Dataset governance could not be read." : governanceResult.error}
          />
        </AdminAlert>
      </div>
    );
  } else if (groundTruthResult === null || !groundTruthResult.ok) {
    body = (
      <div className="mt-4">
        <AdminAlert tone="error">
          <CanonicalLabel
            domain="webMessage"
            value={groundTruthResult === null ? "Ground Truth could not be read." : groundTruthResult.error}
          />
        </AdminAlert>
      </div>
    );
  } else if (!reviewResult.ok) {
    body = (
      <div className="mt-4">
        <AdminAlert tone="error">
            <CanonicalLabel domain="webMessage" value={reviewResult.error} />
          </AdminAlert>
      </div>
    );
  } else if (mode === "feedback" && feedback === null) {
    body = (
      <div className="mt-4">
        <AdminAlert tone="error">
          <T k="admin.case.promotion.feedbackMissing" />
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
        title={<T k="admin.case.promotion.title" />}
        description={<T k="admin.case.promotion.description" />}
      >
        {media !== null && governanceResult !== null && governanceResult.ok && (
          <p className="mt-4 text-[12px] text-muted">
            {governance === null
              ? <T k="admin.case.promotion.governanceNone" values={{ media: `${media.slice(0, 12)}…` }} />
              : (
                <T
                  k="admin.case.promotion.governanceRecorded"
                  values={{
                    media: `${media.slice(0, 12)}…`,
                    lineage: governance.source_lineage_id,
                    split: <CanonicalLabel domain="datasetSplit" value={governance.dataset_split} />,
                  }}
                />
              )}
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
        title={<T k="admin.case.title" />}
        description={
          <>
            <p>
              <T k="admin.case.description" />
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
            <T k="admin.common.refresh" />
          </Link>
        }
      />

      {/* The outcome of the last save, read out of the query string because the form is a plain
          form and the answer arrives as a redirect. The error text is the API's own
          client-facing sentence, rendered as text. */}
      {error && (
        <div className="mt-5">
          <AdminAlert tone="error">
            <CanonicalLabel domain="webMessage" value={error} />
          </AdminAlert>
        </div>
      )}
      {saved !== null && !error && (
        <div className="mt-5">
          <AdminAlert tone="success">
            <T k="admin.case.reviewSaved" />
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
          <AdminAlert tone="error">
            <CanonicalLabel domain="webMessage" value={analysisResult.error} />
          </AdminAlert>
        ) : (
          <>
            <ForensicResult analysis={analysisResult.analysis} />
            <ProvenanceRecord signal={analysisResult.analysis.provenance} />
          </>
        )}

        {!reviewResult.ok ? (
          <AdminAlert tone="error">
            <CanonicalLabel domain="webMessage" value={reviewResult.error} />
          </AdminAlert>
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
