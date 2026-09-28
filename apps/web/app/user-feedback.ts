/**
 * User feedback (R13-T1): what the owner of an analysis says about its result.
 *
 * A claim and nothing more. It is not Ground Truth — that is recorded by an administrator against
 * the bytes, with a stated source — and it is not a Human Review. Nothing on the report reads it
 * back into the assessment, and the API writes it to its own table only.
 *
 * Only the analysis's owner may write or read it. For anybody else — another user, an
 * administrator looking at someone else's report, an analysis with no owner — the API answers
 * 404, and the report shows no feedback control at all.
 *
 * The vocabularies are restated from `apps/api/app/db/models.py` (`FEEDBACK_ASSESSMENTS`,
 * `FEEDBACK_CLAIMED_LABELS`), which is where they are enforced.
 */

import { apiUrl } from "./analysis";
import { requestIdHeaders } from "./observability";
import { sessionHeaders } from "./session";

export function feedbackUrl(analysisId: string): string {
  return `/api/v1/analyses/${encodeURIComponent(analysisId)}/feedback`;
}

export const FEEDBACK_TIMEOUT_MS = 5000;

export const MAX_FEEDBACK_NOTES_LENGTH = 1000;

export const FEEDBACK_ASSESSMENT_LABELS: Record<string, string> = {
  AGREE: "Agree",
  DISAGREE: "Disagree",
  UNSURE: "Unsure",
};

export const FEEDBACK_CLAIMED_LABEL_LABELS: Record<string, string> = {
  GENUINE: "Genuine",
  AI_GENERATED: "AI-generated",
  FACE_SWAP: "Face swap",
  OTHER: "Other",
};

/** The caller's own feedback. `assessment` is null when nothing has been submitted. */
export type Feedback = {
  assessment: string | null;
  claimed_label: string | null;
  notes: string | null;
  updated_at: string | null;
};

/**
 * `unavailable` is the API's 404: this session may not give feedback on this analysis. It is a
 * normal state, not an error, and the report answers it by drawing nothing.
 */
export type FeedbackResult =
  | { ok: true; feedback: Feedback }
  | { ok: false; unavailable: boolean };

function nullableString(value: unknown): value is string | null {
  return value === null || typeof value === "string";
}

export function parseFeedback(payload: unknown): Feedback | null {
  if (typeof payload !== "object" || payload === null) {
    return null;
  }

  const { assessment, claimed_label, notes, updated_at } = payload as Record<string, unknown>;

  if (
    !nullableString(assessment) ||
    !nullableString(claimed_label) ||
    !nullableString(notes) ||
    !nullableString(updated_at)
  ) {
    return null;
  }

  return { assessment, claimed_label, notes, updated_at };
}

export async function fetchFeedback(analysisId: string): Promise<FeedbackResult> {
  try {
    const response = await fetch(`${apiUrl()}${feedbackUrl(analysisId)}`, {
      cache: "no-store",
      headers: { ...(await sessionHeaders()), ...(await requestIdHeaders()) },
      signal: AbortSignal.timeout(FEEDBACK_TIMEOUT_MS),
    });

    if (response.status === 404) {
      return { ok: false, unavailable: true };
    }

    if (!response.ok) {
      return { ok: false, unavailable: false };
    }

    const feedback = parseFeedback(await response.json().catch(() => null));

    return feedback === null ? { ok: false, unavailable: false } : { ok: true, feedback };
  } catch {
    return { ok: false, unavailable: false };
  }
}
