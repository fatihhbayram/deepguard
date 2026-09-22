/**
 * The Ground Truth form's half of the Ground Truth API: carrying a source class, a label and
 * notes to the API's PUT (R12-T3).
 *
 * The same shape `/admin/update-review` has, for the same reasons stated there — no CORS on the
 * API, a browser form can only POST, and the outcome travels back in the query string. The
 * outcome keys are `gt_saved` and `gt_error` rather than the review's `saved` and `error`, so
 * the page can never report a Ground Truth save as a review save or the other way round.
 *
 * **This handler is not behind `admin/layout.tsx`.** The API decides who may write here:
 * `require_admin` refuses the forwarded PUT with a 403, and the refusal is relayed below.
 *
 * **Nothing in this file decides what Ground Truth may say.** The vocabularies are the API's, in
 * `app/ground_truth.py`, and anything outside them is a 422 from there. The only check here is
 * that the two required choices were made at all — the form opens with neither chosen, so an
 * empty value is somebody pressing save before choosing, and answering that beside the form is
 * kinder than relaying a validation error about a missing enum.
 *
 * **This route cannot touch a forensic value or a review.** It forwards three fields to an
 * endpoint whose request model forbids extras — `manipulation_family` included — and that writes
 * only the `ground_truth` row and its audit event.
 */

import { NextResponse } from "next/server";

import { apiUrl } from "../../analysis";
import { logError, logInfo, requestIdHeaders } from "../../observability";
import {
  LOGIN_PATH,
  adminAnalysisPath,
  forwardedOrigin,
  isSameOrigin,
  sessionHeaders,
} from "../../session";
import { groundTruthUrl } from "../ground-truth";

// One upsert of a single row behind a row lock — the same bound as the review.
const UPDATE_TIMEOUT_MS = 5000;

// Enough of the API's message to be useful, and bounded. Rendered as text by React.
const MAX_ERROR_LENGTH = 200;

/** Back to the analysis's screen with the outcome attached — 303, relative, as in update-review. */
function back(analysisId: string, params: Record<string, string>): NextResponse {
  const query = new URLSearchParams(params);

  return new NextResponse(null, {
    status: 303,
    headers: { Location: `${adminAnalysisPath(analysisId)}?${query}` },
  });
}

/** What the API said went wrong, or a generic statement when it said nothing usable. */
async function failureText(response: Response): Promise<string> {
  const payload = await response.json().catch(() => null);
  const detail =
    typeof payload === "object" && payload !== null
      ? (payload as Record<string, unknown>).detail
      : null;

  return typeof detail === "string" && detail.length > 0
    ? detail.slice(0, MAX_ERROR_LENGTH)
    : `The Ground Truth was refused (HTTP ${response.status}).`;
}

export async function POST(request: Request): Promise<NextResponse> {
  // Refused before the body is read, as in update-review.
  if (!isSameOrigin(request)) {
    return new NextResponse(null, { status: 403 });
  }

  let form: FormData;
  try {
    form = await request.formData();
  } catch {
    return new NextResponse(null, { status: 400 });
  }

  const analysisId = (form.get("analysis_id") ?? "").toString().trim();
  const sha256 = (form.get("sha256") ?? "").toString().trim();
  const sourceClass = form.get("source_class");
  const label = form.get("label");
  const notes = form.get("notes");

  // No analysis means nowhere to redirect back to; a form without it did not come from the page.
  if (!analysisId) {
    return new NextResponse(null, { status: 400 });
  }

  // The page renders no form for an analysis without a hash, so a post without one did not come
  // from it either — but there is somewhere to send the answer, so it is sent there.
  if (!sha256) {
    return back(analysisId, {
      gt_error:
        "Ground Truth cannot be recorded because this analysis has no original media SHA-256 identity.",
    });
  }

  if (
    typeof sourceClass !== "string" ||
    sourceClass === "" ||
    typeof label !== "string" ||
    label === ""
  ) {
    return back(analysisId, {
      gt_error: "Choose how the label is known and what the media is before saving.",
    });
  }

  const statement = {
    source_class: sourceClass,
    label,
    // Optional. An empty textarea is "no notes", which the API stores as null — the same value
    // a record created without notes has, so re-saving an unchanged record stays a no-op rather
    // than writing an audit event for a change from null to "". Non-empty notes are forwarded
    // exactly as typed; the API is what refuses control characters.
    notes: typeof notes === "string" && notes !== "" ? notes : null,
  };

  const forwarded = {
    ...(await sessionHeaders()),
    ...forwardedOrigin(request),
    ...(await requestIdHeaders()),
    "content-type": "application/json",
  };

  let response: Response;
  try {
    response = await fetch(`${apiUrl()}${groundTruthUrl(sha256)}`, {
      method: "PUT",
      headers: forwarded,
      body: JSON.stringify(statement),
      signal: AbortSignal.timeout(UPDATE_TIMEOUT_MS),
    });
  } catch (error) {
    const reason = error instanceof Error ? error.message : String(error);
    await logError("The API could not be reached for Ground Truth.", { reason });

    return back(analysisId, { gt_error: "The API could not be reached." });
  }

  if (response.status === 401) {
    return new NextResponse(null, { status: 303, headers: { Location: LOGIN_PATH } });
  }

  if (!response.ok) {
    // Includes the 403 a non-administrator gets, the 404 for bytes the API does not know, and
    // the 422 for a value outside the vocabulary or notes that are not plain text.
    return back(analysisId, { gt_error: await failureText(response) });
  }

  // The hash and nothing else: the notes are free prose, and the API has already logged the
  // label and source against the administrator who stored them.
  await logInfo("Forwarded Ground Truth to the API.", {
    analysis_id: analysisId,
    media_sha256: sha256,
  });

  return back(analysisId, { gt_saved: "1" });
}
