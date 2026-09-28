/**
 * The administrative read of user feedback on one analysis (R13-T1).
 *
 * Read-only: the API has no administrative write route for feedback, and this module has no
 * mutation. What it reads is what analysis owners *claimed* — kept apart from `reviews.ts` (what
 * an analyst concluded) and `ground-truth.ts` (what the media is), and never rendered as either.
 */

import { apiUrl } from "../analysis";
import { requestIdHeaders } from "../observability";
import { sessionHeaders } from "../session";

export function adminFeedbackUrl(analysisId: string): string {
  return `/api/v1/admin/analyses/${encodeURIComponent(analysisId)}/feedback`;
}

const USER_FEEDBACK_TIMEOUT_MS = 5000;

/** One entry as `FeedbackEntry` in `app/api/user_feedback.py` reports it. */
export type UserFeedbackEntry = {
  user_id: string;
  user_email: string;
  assessment: string;
  claimed_label: string | null;
  notes: string | null;
  created_at: string;
  updated_at: string;
};

export type UserFeedbackResult =
  | { ok: true; entries: UserFeedbackEntry[] }
  | { ok: false; unauthenticated: boolean; error: string };

function parseEntry(payload: unknown): UserFeedbackEntry | null {
  if (typeof payload !== "object" || payload === null) {
    return null;
  }

  const { user_id, user_email, assessment, claimed_label, notes, created_at, updated_at } =
    payload as Record<string, unknown>;

  if (
    typeof user_id !== "string" ||
    typeof user_email !== "string" ||
    typeof assessment !== "string" ||
    (claimed_label !== null && typeof claimed_label !== "string") ||
    (notes !== null && typeof notes !== "string") ||
    typeof created_at !== "string" ||
    typeof updated_at !== "string"
  ) {
    return null;
  }

  return { user_id, user_email, assessment, claimed_label, notes, created_at, updated_at };
}

export async function fetchUserFeedback(analysisId: string): Promise<UserFeedbackResult> {
  const unavailable = {
    ok: false as const,
    unauthenticated: false,
    error: "User feedback is temporarily unavailable.",
  };

  try {
    const response = await fetch(`${apiUrl()}${adminFeedbackUrl(analysisId)}`, {
      cache: "no-store",
      headers: { ...(await sessionHeaders()), ...(await requestIdHeaders()) },
      signal: AbortSignal.timeout(USER_FEEDBACK_TIMEOUT_MS),
    });

    if (response.status === 401) {
      return { ok: false, unauthenticated: true, error: "Sign in to read user feedback." };
    }

    if (response.status === 403) {
      return {
        ok: false,
        unauthenticated: false,
        error: "This account is not an administrator.",
      };
    }

    if (!response.ok) {
      return unavailable;
    }

    const payload = await response.json().catch(() => null);
    if (!Array.isArray(payload)) {
      return unavailable;
    }

    const entries = payload.map(parseEntry);
    if (entries.some((entry) => entry === null)) {
      return { ok: false, unauthenticated: false, error: "User feedback could not be read." };
    }

    return { ok: true, entries: entries as UserFeedbackEntry[] };
  } catch {
    return unavailable;
  }
}
