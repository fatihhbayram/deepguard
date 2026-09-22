/**
 * The administrative surface's half of the Ground Truth API: reading what a set of bytes
 * actually is (R12-T3).
 *
 * The same split `reviews.ts` makes, for the same reason — one record, one reader, one module.
 * It is a separate module from `reviews.ts` and must stay one: a review is what an analyst made
 * of a report, Ground Truth is what the media is, and `apps/api/app/ground_truth.py` states why
 * neither may stand in for the other. Two modules against two endpoints is what keeps them from
 * being merged into one object a renderer could present as a single answer.
 *
 * **Keyed by the bytes, not by the analysis.** The address is the original's SHA-256, because
 * the same file uploaded under two analyses is the same file, and what it is does not depend on
 * which upload somebody is looking at.
 *
 * **Nothing here reads a verdict, a review or a provenance state, and nothing here may.** Ground
 * Truth that was filled in from any of them would score the detectors as correct by
 * construction. The form that writes it starts from the stored record or from nothing.
 *
 * **`manipulation_family` is deliberately absent from the type below.** The API derives it from
 * the label on every read and refuses it on every write; it is not a value an administrator
 * states, so this module does not carry it and no control can be built from it.
 *
 * **This module writes nothing.** The mutation is a plain HTML form posting to
 * `/admin/update-ground-truth`, for the reasons `reviews.ts` gives.
 *
 * **The notes are a string and are only ever rendered as text** — the same invariant, and the
 * same reason, as the review note.
 */

import { apiUrl } from "../analysis";
import { requestIdHeaders } from "../observability";
import { sessionHeaders } from "../session";

/**
 * The API path for the Ground Truth of one set of bytes, in one place.
 *
 * Exported because the route handler that forwards the mutation needs the same address and must
 * not spell it a second time.
 */
export function groundTruthUrl(sha256: string): string {
  return `/api/v1/admin/ground-truth/${encodeURIComponent(sha256)}`;
}

// One indexed read of at most one row, held to the same short bound as the review.
export const GROUND_TRUTH_TIMEOUT_MS = 5000;

// The API's detail for known bytes nobody has labelled, as opposed to `media not found` for a
// hash no upload carries. Both are 404s and only this one means "show an empty form".
const NOT_RECORDED_DETAIL = "ground truth not recorded";

/**
 * How the label is known, as the API spells it.
 *
 * Must match `SourceClass` in `apps/api/app/ground_truth.py`, which is where the value is
 * validated. Restated because there is no way to derive one from the other; the order is the
 * order the select offers them in.
 */
export const GROUND_TRUTH_SOURCE_CLASS_LABELS: Record<string, string> = {
  OWNER_KNOWN: "Known to the owner",
  CONTROLLED_TEST: "Controlled test",
  EXTERNAL_VERIFIED: "Externally verified",
  UNKNOWN: "Unknown",
};

/**
 * What the media is, as the API spells it.
 *
 * Must match `GroundTruthLabel` in `apps/api/app/ground_truth.py`. `UNKNOWN` is a statement
 * somebody makes, not a missing value — which is why it is an option and not a default.
 */
export const GROUND_TRUTH_LABEL_LABELS: Record<string, string> = {
  GENUINE: "Genuine",
  AI_GENERATED: "AI-generated",
  FACE_SWAP: "Face swap",
  AUDIO_MANIPULATION: "Audio manipulation",
  OTHER_MANIPULATION: "Other manipulation",
  UNKNOWN: "Unknown",
};

/**
 * One Ground Truth record as `/api/v1/admin/ground-truth/{sha256}` reports it.
 *
 * Mirrors `GroundTruthState` in `app/api/admin_ground_truth.py`, less the derived family (see
 * above). `source_class` and `label` are plain strings rather than unions, for the reason
 * `AnalysisReview.status` is: a value this file was not written to expect should print as the
 * API spelled it, not fail the whole read.
 */
export type GroundTruth = {
  media_sha256: string;
  source_class: string;
  label: string;
  notes: string | null;
  actor_id: string;
  actor_email_snapshot: string | null;
  created_at: string;
  updated_at: string;
};

/**
 * What a read produced.
 *
 * `null` in the success case means nobody has recorded Ground Truth for these bytes. That is a
 * successful read — the API says so with a 404 carrying a specific detail — and the page answers
 * it with an empty form, not with an error.
 */
export type GroundTruthResult =
  | { ok: true; groundTruth: GroundTruth | null }
  | { ok: false; unauthenticated: boolean; error: string };

/** One record out of the payload, or null for anything that is not one. */
export function parseGroundTruth(payload: unknown): GroundTruth | null {
  if (typeof payload !== "object" || payload === null) {
    return null;
  }

  const {
    media_sha256,
    source_class,
    label,
    notes,
    actor_id,
    actor_email_snapshot,
    created_at,
    updated_at,
  } = payload as Record<string, unknown>;

  if (
    typeof media_sha256 !== "string" ||
    typeof source_class !== "string" ||
    typeof label !== "string" ||
    (notes !== null && typeof notes !== "string") ||
    typeof actor_id !== "string" ||
    (actor_email_snapshot !== null && typeof actor_email_snapshot !== "string") ||
    typeof created_at !== "string" ||
    typeof updated_at !== "string"
  ) {
    return null;
  }

  return {
    media_sha256,
    source_class,
    label,
    notes,
    actor_id,
    actor_email_snapshot,
    created_at,
    updated_at,
  };
}

/** The API's `detail`, when the body has one. */
async function detailOf(response: Response): Promise<string | null> {
  const payload = await response.json().catch(() => null);
  const detail =
    typeof payload === "object" && payload !== null
      ? (payload as Record<string, unknown>).detail
      : null;

  return typeof detail === "string" ? detail : null;
}

/**
 * The Ground Truth recorded for one set of bytes, or the fact that none has been.
 *
 * A 403 is reported rather than treated as a sign-in problem, for the reason `fetchReview`
 * gives. A 404 for `media not found` is an error here, not an empty record: the analysis names
 * bytes the API does not know, and offering a form that could only be refused would hide that.
 */
export async function fetchGroundTruth(sha256: string): Promise<GroundTruthResult> {
  const unavailable = {
    ok: false as const,
    unauthenticated: false,
    error: "Ground Truth is temporarily unavailable.",
  };

  try {
    const response = await fetch(`${apiUrl()}${groundTruthUrl(sha256)}`, {
      cache: "no-store",
      headers: { ...(await sessionHeaders()), ...(await requestIdHeaders()) },
      signal: AbortSignal.timeout(GROUND_TRUTH_TIMEOUT_MS),
    });

    if (response.status === 401) {
      return {
        ok: false,
        unauthenticated: true,
        error: "Sign in to read Ground Truth.",
      };
    }

    if (response.status === 403) {
      return {
        ok: false,
        unauthenticated: false,
        error: "This account is not an administrator.",
      };
    }

    if (response.status === 404) {
      if ((await detailOf(response)) === NOT_RECORDED_DETAIL) {
        return { ok: true, groundTruth: null };
      }

      return {
        ok: false,
        unauthenticated: false,
        error: "The API has no media with this analysis's SHA-256.",
      };
    }

    if (!response.ok) {
      return unavailable;
    }

    const groundTruth = parseGroundTruth(await response.json().catch(() => null));
    if (groundTruth === null) {
      return {
        ok: false,
        unauthenticated: false,
        error: "The Ground Truth record could not be read.",
      };
    }

    return { ok: true, groundTruth };
  } catch {
    // A timeout or an unreachable API — the same one sentence `fetchReview` gives for both.
    return unavailable;
  }
}
