/**
 * The promotion form's half of three APIs: dataset governance, Ground Truth, and the review
 * (R14-T5).
 *
 * **Three unchanged endpoints, called in order, never merged.** This handler forwards a governance
 * statement to `PUT /api/v1/admin/dataset-governance/{sha256}`, then — only if that succeeded — a
 * Ground Truth statement to `PUT /api/v1/admin/ground-truth/{sha256}`, then — only if that
 * succeeded — the review status to `PUT /api/v1/admin/analyses/{id}/review`. Each API call is the
 * same one the rest of the surface makes; no endpoint takes a hash and an analysis id together,
 * and none was added. The order and the stop-on-failure rule are `runPromotion`'s, in
 * `../promotion.ts`, where they are tested without a network.
 *
 * It is ordered best effort, not a transaction. Three endpoints commit three transactions, so a
 * failure after the first leaves what was already saved saved, and the redirect says exactly which
 * steps were saved, which failed and which never ran. Every step is idempotent at the API — a
 * statement equal to the stored one writes nothing — so submitting the same form again is the
 * retry, and `retry=review` re-runs the review step alone.
 *
 * **Nothing here is taken from the user's feedback but the prefill the analyst already saw.** The
 * feedback's owner id travels only to reopen the same panel afterwards and into the log line; it
 * is not sent to any API.
 *
 * **This handler is not behind `admin/layout.tsx`**, for the reason `update-review` gives: the API
 * refuses a non-administrator with a 403 on the first PUT, and the refusal is relayed.
 */

import { NextResponse } from "next/server";

import { apiUrl, fetchAnalysis } from "../../analysis";
import { logError, logInfo, requestIdHeaders } from "../../observability";
import {
  LOGIN_PATH,
  adminAnalysisPath,
  forwardedOrigin,
  isSameOrigin,
  sessionHeaders,
} from "../../session";
import { datasetGovernanceUrl } from "../dataset-governance";
import { groundTruthUrl } from "../ground-truth";
import {
  PromotionStep,
  StepResult,
  governanceStatement,
  outcomeParams,
  promotionCandidates,
  provenanceProblem,
  refusalParams,
  runPromotion,
  setReviewStatus,
} from "../promotion";
import { fetchReview, reviewUrl } from "../reviews";

// Each step is one upsert behind a lock — the same bound the single-record forms use.
const STEP_TIMEOUT_MS = 5000;

const MAX_ERROR_LENGTH = 200;

/** A 401 anywhere in the sequence ends it at the sign-in page, as the other forms do. */
class SignedOut extends Error {}

/** Back to the analysis, with the panel reopened as it was and the outcome attached. */
function back(
  analysisId: string,
  panel: Record<string, string>,
  params: Record<string, string>,
): NextResponse {
  const query = new URLSearchParams({ ...panel, ...params });

  return new NextResponse(null, {
    status: 303,
    headers: { Location: `${adminAnalysisPath(analysisId)}?${query}#promotion` },
  });
}

async function failureText(response: Response, what: string): Promise<string> {
  const payload = await response.json().catch(() => null);
  const detail =
    typeof payload === "object" && payload !== null
      ? (payload as Record<string, unknown>).detail
      : null;

  return typeof detail === "string" && detail.length > 0
    ? `${what}: ${detail}`.slice(0, MAX_ERROR_LENGTH)
    : `${what} was refused (HTTP ${response.status}).`;
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

  const field = (name: string): string | null => {
    const value = form.get(name);
    return typeof value === "string" ? value : null;
  };

  const analysisId = (field("analysis_id") ?? "").trim();
  if (!analysisId) {
    return new NextResponse(null, { status: 400 });
  }

  // Reopen the panel the analyst was in: manual, or the one feedback record they chose.
  const feedbackUserId = field("feedback_user_id") ?? "";
  const panel: Record<string, string> =
    feedbackUserId !== ""
      ? { promote: "feedback", feedback: feedbackUserId }
      : { promote: "manual" };
  // And on the media it was about, so the form is seeded from that file's records again.
  const media = field("sha256") ?? field("media") ?? "";
  if (media !== "") {
    panel.media = media;
  }

  const reviewOnly = field("retry") === "review";
  const reviewTarget = field("review_status") ?? "";
  const expectedReviewStatus = field("expected_review_status") ?? "";

  const forwarded = {
    ...(await sessionHeaders()),
    ...forwardedOrigin(request),
    ...(await requestIdHeaders()),
    "content-type": "application/json",
  };

  const put = async (path: string, body: unknown, what: string): Promise<StepResult> => {
    let response: Response;
    try {
      response = await fetch(`${apiUrl()}${path}`, {
        method: "PUT",
        headers: forwarded,
        body: JSON.stringify(body),
        signal: AbortSignal.timeout(STEP_TIMEOUT_MS),
      });
    } catch (error) {
      const reason = error instanceof Error ? error.message : String(error);
      await logError("The API could not be reached during a promotion.", { step: what, reason });
      return { ok: false, error: `${what}: the API could not be reached.` };
    }

    if (response.status === 401) {
      throw new SignedOut();
    }

    return response.ok ? { ok: true } : { ok: false, error: await failureText(response, what) };
  };

  const reviewStep = () =>
    setReviewStatus(reviewTarget, expectedReviewStatus, {
      readReview: async () => {
        const result = await fetchReview(analysisId);
        if (!result.ok && result.unauthenticated) {
          throw new SignedOut();
        }
        return result.ok ? result : { ok: false, error: `Review: ${result.error}` };
      },
      putReview: (body) => put(reviewUrl(analysisId), body, "Review"),
    });

  const refused = async (): Promise<StepResult> => ({
    ok: false,
    error: "This step was not requested.",
  });

  let steps: Record<PromotionStep, () => Promise<StepResult>>;
  let sha256 = "";

  if (reviewOnly) {
    steps = { governance: refused, ground_truth: refused, review: reviewStep };
  } else {
    sha256 = (field("sha256") ?? "").trim();
    const label = field("label") ?? "";
    const sourceClass = field("source_class") ?? "";
    const notes = field("notes") ?? "";
    const attested = field("provenance_attested") === "1";

    // Every check that needs no request, before the first one is made: nothing is saved by a
    // form that was going to be refused anyway.
    if (sha256 === "") {
      return back(analysisId, panel, refusalParams("Choose which media file this record is about."));
    }

    const analysis = await fetchAnalysis(analysisId);
    if (!analysis.ok) {
      if (analysis.unauthenticated) {
        return new NextResponse(null, { status: 303, headers: { Location: LOGIN_PATH } });
      }
      return back(analysisId, panel, refusalParams(analysis.error));
    }
    if (!promotionCandidates(analysis.analysis.media_sha256s).includes(sha256)) {
      return back(
        analysisId,
        panel,
        refusalParams("The chosen SHA-256 is not a media file of this analysis."),
      );
    }

    const governance = governanceStatement(field);
    if (!governance.ok) {
      return back(analysisId, panel, refusalParams(governance.problem));
    }

    if (label === "") {
      return back(analysisId, panel, refusalParams("Choose what the media is."));
    }

    const provenance = provenanceProblem(sourceClass, attested, notes);
    if (provenance !== null) {
      return back(analysisId, panel, refusalParams(provenance));
    }

    if (reviewTarget === "") {
      return back(
        analysisId,
        panel,
        refusalParams("Choose the review status to leave this case in."),
      );
    }

    steps = {
      governance: () =>
        put(datasetGovernanceUrl(sha256), governance.statement, "Dataset governance"),
      ground_truth: () =>
        put(
          groundTruthUrl(sha256),
          // Empty notes are no notes, as in update-ground-truth, so an unchanged re-send is a no-op.
          { source_class: sourceClass, label, notes: notes !== "" ? notes : null },
          "Ground Truth",
        ),
      review: reviewStep,
    };
  }

  let outcome;
  try {
    outcome = await runPromotion(steps, reviewOnly ? "review" : "governance");
  } catch (error) {
    if (error instanceof SignedOut) {
      return new NextResponse(null, { status: 303, headers: { Location: LOGIN_PATH } });
    }
    throw error;
  }

  // Ids and step names only: the notes are free prose and the API has logged each write.
  await logInfo("Forwarded a Ground Truth promotion to the API.", {
    analysis_id: analysisId,
    media_sha256: sha256 || null,
    feedback_user_id: feedbackUserId || null,
    saved: outcome.saved.join(","),
    failed: outcome.failed,
  });

  return back(analysisId, panel, outcomeParams(outcome));
}
