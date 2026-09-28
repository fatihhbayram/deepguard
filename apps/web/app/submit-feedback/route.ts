/**
 * The report's feedback form, forwarded to `POST /api/v1/analyses/{id}/feedback` (R13-T1).
 *
 * The shape `/admin/update-review` has, for the reasons stated there: no CORS on the API, a plain
 * HTML form, and the outcome carried back in the query string (`fb_saved` / `fb_error`, so it is
 * never confused with anything else on the page).
 *
 * **The API decides who may write.** It accepts only the analysis's owner; everybody else gets a
 * 404, relayed below as an error. Nothing here names a verdict, a review or Ground Truth, and the
 * endpoint behind it writes only the feedback row.
 */

import { NextResponse } from "next/server";

import { apiUrl } from "../analysis";
import { logError, logInfo, requestIdHeaders } from "../observability";
import { LOGIN_PATH, forwardedOrigin, isSameOrigin, sessionHeaders } from "../session";
import { FEEDBACK_TIMEOUT_MS, feedbackUrl } from "../user-feedback";

const MAX_ERROR_LENGTH = 200;

function back(analysisId: string, params: Record<string, string>): NextResponse {
  const query = new URLSearchParams(params);

  return new NextResponse(null, {
    status: 303,
    headers: {
      Location: `/app/report/${encodeURIComponent(analysisId)}?${query}#feedback`,
    },
  });
}

async function failureText(response: Response): Promise<string> {
  if (response.status === 404) {
    return "Feedback cannot be given on this analysis.";
  }

  const payload = await response.json().catch(() => null);
  const detail =
    typeof payload === "object" && payload !== null
      ? (payload as Record<string, unknown>).detail
      : null;

  // A 422's detail is a list of validation errors; only a plain sentence is relayed.
  return typeof detail === "string" && detail.length > 0
    ? detail.slice(0, MAX_ERROR_LENGTH)
    : `The feedback was refused (HTTP ${response.status}).`;
}

export async function POST(request: Request): Promise<NextResponse> {
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
  const assessment = form.get("assessment");
  const claimedLabel = form.get("claimed_label");
  const notes = form.get("notes");

  if (!analysisId) {
    return new NextResponse(null, { status: 400 });
  }

  if (typeof assessment !== "string" || assessment === "") {
    return back(analysisId, { fb_error: "Choose Agree, Disagree or Unsure before sending." });
  }

  // Empty controls are "not stated": null, the value the API stores for them, so re-sending an
  // unchanged form stays a no-op.
  const statement = {
    assessment,
    claimed_label: typeof claimedLabel === "string" && claimedLabel !== "" ? claimedLabel : null,
    notes: typeof notes === "string" && notes !== "" ? notes : null,
  };

  let response: Response;
  try {
    response = await fetch(`${apiUrl()}${feedbackUrl(analysisId)}`, {
      method: "POST",
      headers: {
        ...(await sessionHeaders()),
        ...forwardedOrigin(request),
        ...(await requestIdHeaders()),
        "content-type": "application/json",
      },
      body: JSON.stringify(statement),
      signal: AbortSignal.timeout(FEEDBACK_TIMEOUT_MS),
    });
  } catch (error) {
    const reason = error instanceof Error ? error.message : String(error);
    await logError("The API could not be reached for feedback.", { reason });

    return back(analysisId, { fb_error: "The API could not be reached." });
  }

  if (response.status === 401) {
    return new NextResponse(null, { status: 303, headers: { Location: LOGIN_PATH } });
  }

  if (!response.ok) {
    return back(analysisId, { fb_error: await failureText(response) });
  }

  await logInfo("Forwarded feedback to the API.", { analysis_id: analysisId });

  return back(analysisId, { fb_saved: "1" });
}
