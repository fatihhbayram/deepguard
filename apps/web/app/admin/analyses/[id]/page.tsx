/**
 * The human review of one analysis, shown beside the forensic result it is about (R8-T7).
 *
 * **The whole design of this page is the boundary down the middle of it.** The top half is the
 * detector result — the risk classification, the ruleset that produced it, the calibration its
 * thresholds were measured under and the rule that fired — and it has no control of any kind
 * on it. The bottom half is what a person said, and it is the only part of this screen with a
 * form. The two are drawn as separate cards with separate headings and separate explanatory
 * prose, because an operator reading a case has to be able to say which of the two statements
 * in front of them came from evidence and which came from a colleague.
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
 * **Nothing is recomputed.** The risk classification is displayed exactly as the worker
 * committed it, with no detector score compared against a threshold here, and the review is
 * displayed exactly as the API returned it. A screen that re-derived either could contradict
 * the record it exists to show.
 *
 * A Server Component with no client-side code at all. The controls are plain HTML forms, the
 * same as the account controls, so the page works with JavaScript disabled and the outcome of
 * a save arrives as a redirect carrying it in the query string.
 */

import Link from "next/link";
import { notFound, redirect } from "next/navigation";

import {
  ABSENT,
  RISK_LABELS,
  UNSUPPORTED,
  fetchAnalysis,
  fetchSession,
  isSupportedRiskLevel,
} from "../../../analysis";
import { ADMIN_JOBS_PATH, ADMIN_PATH, LOGIN_PATH } from "../../../session";
import {
  AnalysisReview,
  MAX_REVIEW_NOTE_LENGTH,
  REVIEW_STATUS_LABELS,
  REVIEW_STATUS_NEEDS_FOLLOW_UP,
  REVIEW_STATUS_REVIEWED,
  REVIEW_STATUS_UNREVIEWED,
  fetchReview,
} from "../../reviews";

// What the risk column says when no decision was ever taken. Distinct from `UNKNOWN`, which is
// a decision, and phrased as a statement about the record rather than about the media — the
// same wording the report uses for the same reason.
const NO_DECISION = "No decision recorded";

/* ------------------------------------------------------------------ *
 * Primitives
 * ------------------------------------------------------------------ */

/*
 * `Legend`, `Heading` and `Alert` restated a sixth time. `app/admin/jobs/page.tsx` noted the
 * third copy as the point the Rule of Three says to extract; `analytics/page.tsx` carried the
 * count to four and `audit/page.tsx` to five. It is six. The extraction is long owed and is
 * still a change to five other pages that this task is not about — carried forward again so
 * the number stays in front of whoever picks it up rather than resetting quietly at each new
 * screen.
 */

/** The small accented label above a section heading. */
function Legend({ children }: { children: React.ReactNode }) {
  return (
    <p className="text-[11px] font-medium tracking-[0.16em] text-accent uppercase">
      {children}
    </p>
  );
}

/** A section heading. Tight, semibold, at the scale the rest of the application sets it. */
function Heading({ children }: { children: React.ReactNode }) {
  return (
    <h2 className="mt-2.5 text-2xl font-semibold tracking-[-0.02em] text-bone sm:text-[28px]">
      {children}
    </h2>
  );
}

/** The outcome of the last save, or the reason there is nothing on the screen. */
function Alert({
  tone,
  children,
}: {
  tone: "error" | "success";
  children: React.ReactNode;
}) {
  const styles =
    tone === "error"
      ? "border-rose-500/40 bg-rose-500/10 text-rose-200"
      : "border-emerald-500/40 bg-emerald-500/10 text-emerald-200";
  const dot = tone === "error" ? "bg-rose-400" : "bg-emerald-400";

  return (
    <p
      role="status"
      className={`flex items-start gap-3 rounded-md border px-4 py-3 text-[13px] leading-relaxed ${styles}`}
    >
      <span aria-hidden className={`mt-1.5 size-1.5 shrink-0 rounded-full ${dot}`} />
      <span>{children}</span>
    </p>
  );
}

/** A machine value, or an em dash where the record holds nothing. */
function Value({ children }: { children: string | null }) {
  return children === null ? (
    <span className="text-muted">{ABSENT}</span>
  ) : (
    <span className="font-mono text-[11px] break-all text-muted">{children}</span>
  );
}

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
  };
}) {
  const level = analysis.risk_level;

  return (
    <section className="rounded-lg border border-line bg-ink-2 p-5 sm:p-6">
      <h3 className="text-[15px] font-semibold text-bone">Forensic result</h3>
      <p className="mt-2 max-w-[74ch] text-[13px] leading-relaxed text-muted">
        Committed by the detection pipeline under the ruleset named below, and immutable. There
        is no control on this card and no route in this application that can alter any value on
        it — the review beneath is a separate record and does not change what the detectors
        concluded.
      </p>

      <dl className="mt-5 grid grid-cols-1 gap-5 sm:grid-cols-2">
        <Fact label="Risk classification">
          {level === null
            ? NO_DECISION
            : isSupportedRiskLevel(level)
              ? RISK_LABELS[level]
              : UNSUPPORTED}
        </Fact>
        <Fact label="Analysis status">{analysis.status}</Fact>
        <Fact label="Ruleset">
          <Value>{analysis.risk_rules_version}</Value>
        </Fact>
        <Fact label="Rule fired">
          <Value>{analysis.risk_rule_id}</Value>
        </Fact>
        <Fact label="Calibration">
          <Value>{analysis.risk_calibration_id}</Value>
        </Fact>
        <Fact label="Submitted">
          {/* The API's own timestamp, shown as stored rather than reformatted into a local
              rendering the record does not hold — the convention every screen in this surface
              follows. */}
          <Value>{analysis.created_at}</Value>
        </Fact>
      </dl>
    </section>
  );
}

/* ------------------------------------------------------------------ *
 * The review half
 * ------------------------------------------------------------------ */

/** The workflow state as a chip, including the state that is the absence of a record. */
function StatusBadge({ status }: { status: string }) {
  const styles =
    status === REVIEW_STATUS_NEEDS_FOLLOW_UP
      ? "bg-amber-500/10 text-amber-200"
      : status === REVIEW_STATUS_REVIEWED
        ? "bg-emerald-500/10 text-emerald-200"
        : "bg-chip text-muted";

  return (
    <span
      className={`inline-flex items-center rounded-sm px-2 py-1 font-mono text-[11px] tracking-[0.08em] uppercase ${styles}`}
    >
      {/* Unknown statuses print as the API spelled them rather than as a blank chip: the day
          the taxonomy grows, this is what says so. */}
      {REVIEW_STATUS_LABELS[status] ?? status}
    </span>
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
      <div className="mt-5">
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
    <div className="mt-5">
      <StatusBadge status={review.status} />

      <dl className="mt-4 grid grid-cols-1 gap-5 sm:grid-cols-2">
        <Fact label="Reviewer">
          {/* The address as it read when the review was last written, never looked up now. A
              rename does not change who signed a past review — see `admin/reviews.ts`. */}
          {review.reviewer_email_snapshot ?? (
            <span className="text-muted">not recorded</span>
          )}
          <div className="mt-1">
            <Value>{review.reviewer_id}</Value>
          </div>
        </Fact>
        <Fact label="Last changed">
          <Value>{review.updated_at}</Value>
          <div className="mt-1">
            <Value>
              {review.created_at === null ? null : `first written ${review.created_at}`}
            </Value>
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
 * The select offers the two operational statuses and nothing else. `UNREVIEWED` is absent
 * because it is not a value — it is the absence of a review row, and there is no route that
 * un-reviews an analysis. Nothing that reads as a verdict is here either: whether the media is
 * genuine is the card above, decided under a named ruleset, and this control does not get a
 * second opinion on it.
 *
 * An unreviewed analysis opens with `REVIEWED` preselected, which is the ordinary first answer
 * — but the button says "Save review" rather than anything that implies agreement, and nothing
 * is written until somebody presses it.
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
    <section className="rounded-lg border border-accent/30 bg-accent/[0.03] p-5 sm:p-6">
      <h3 className="text-[15px] font-semibold text-bone">Human review</h3>
      <p className="mt-2 max-w-[74ch] text-[13px] leading-relaxed text-muted">
        An operational record of whether somebody has looked at this case — not a second opinion
        on the media. Saving a review leaves the forensic result above exactly as it was, and
        every change is recorded in the audit log with who made it and when.
      </p>

      <ReviewRecord review={review} />
      <ReviewForm review={review} />
    </section>
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

  const [user, analysisResult, reviewResult] = await Promise.all([
    fetchSession(),
    fetchAnalysis(id),
    fetchReview(id),
  ]);

  // The session the API would not accept. `admin/layout.tsx` has already turned away a reader
  // with no session and one whose role is not administrator, so what is left for this is the
  // narrow case of a session that expired between the guard and these reads. It is not the
  // access control: the API refused both of them before this line.
  if (
    user === null ||
    (!analysisResult.ok && analysisResult.unauthenticated) ||
    (!reviewResult.ok && reviewResult.unauthenticated)
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

  return (
    <main className="mx-auto w-full max-w-[1280px] flex-1 px-4 py-14 sm:px-8">
      <Legend>Administration</Legend>
      <Heading>Case review</Heading>
      <p className="mt-3 max-w-[74ch] text-[15px] leading-relaxed text-muted">
        One analysis, and what has been said about it. The forensic result and the human review
        are two separate records kept in two separate tables: the first is what the detectors
        concluded and cannot be edited from anywhere in this application, and the second is what
        a reviewer noted and can be revised at any time.
      </p>
      <p className="mt-3">
        <Value>{id}</Value>
      </p>

      {/* The outcome of the last save, read out of the query string because the form is a plain
          form and the answer arrives as a redirect. The error text is the API's own
          client-facing sentence, rendered as text. */}
      {error && (
        <div className="mt-6">
          <Alert tone="error">{error}</Alert>
        </div>
      )}
      {saved !== null && !error && (
        <div className="mt-6">
          <Alert tone="success">
            Review saved. The forensic result was not changed.
          </Alert>
        </div>
      )}

      <div className="mt-6 space-y-6">
        {!analysisResult.ok ? (
          <Alert tone="error">{analysisResult.error}</Alert>
        ) : (
          <ForensicResult analysis={analysisResult.analysis} />
        )}

        {!reviewResult.ok ? (
          <Alert tone="error">{reviewResult.error}</Alert>
        ) : (
          <HumanReview review={reviewResult.review} />
        )}
      </div>

      <div className="mt-8 flex flex-wrap items-center gap-5">
        <Link href={ADMIN_JOBS_PATH} className="text-sm text-muted underline hover:text-bone">
          ← Detection jobs
        </Link>
        <Link href={ADMIN_PATH} className="text-sm text-muted underline hover:text-bone">
          Accounts
        </Link>
      </div>
    </main>
  );
}
