/**
 * The administrative surface's half of the dataset governance API: reading where a set of bytes
 * came from and which evaluation split its lineage is in (R14-T5, over the R12-T5/T5A API).
 *
 * A module of its own for the reason `ground-truth.ts` is one: governance, Ground Truth and the
 * review are three statements with three endpoints, and one module per endpoint is what keeps a
 * renderer from merging them into one object. Keyed by the original's SHA-256, as Ground Truth is.
 *
 * **This module writes nothing.** The write is the promotion form's, through
 * `/admin/promote-ground-truth`, which forwards the statement to the unchanged PUT.
 *
 * **The PUT is the whole statement.** An absent optional field means "none", not "leave it as it
 * is" (`set_dataset_governance`). So a form that revises governance must carry every field, seeded
 * from the stored record — which is why the type below mirrors the response field for field.
 */

import { apiUrl } from "../analysis";
import { requestIdHeaders } from "../observability";
import { sessionHeaders } from "../session";

/** The API path for the governance of one set of bytes, in one place. */
export function datasetGovernanceUrl(sha256: string): string {
  return `/api/v1/admin/dataset-governance/${encodeURIComponent(sha256)}`;
}

const DATASET_GOVERNANCE_TIMEOUT_MS = 5000;

// Known bytes nobody has governed, as opposed to `media not found`. Both are 404s.
const NOT_RECORDED_DETAIL = "dataset governance not recorded";

/**
 * The splits, as the API spells them. Must match `DatasetSplit` in `app/dataset_governance.py`,
 * which is where the value is validated. The order is the order the select offers them in.
 */
export const DATASET_SPLIT_LABELS: Record<string, string> = {
  CALIBRATION: "Calibration",
  VALIDATION: "Validation",
  TEST: "Test",
  HOLDOUT: "Holdout",
};

/** One record as `DatasetGovernanceState` in `app/api/admin_dataset_governance.py` reports it. */
export type DatasetGovernance = {
  media_sha256: string;
  source_lineage_id: string;
  dataset_split: string;
  recording_identity: string | null;
  transformations: string[] | null;
  derived_from_sha256: string | null;
  generation_pipeline: string | null;
  license: string | null;
  permission_status: string | null;
  redistributable: boolean | null;
  private: boolean | null;
  stratum_primary: string | null;
  source: string | null;
  acquisition_type: string | null;
  benchmark_family: string | null;
  actor_id: string;
  actor_email_snapshot: string | null;
  created_at: string;
  updated_at: string;
};

/** `null` in the success case means nobody has governed these bytes — an empty form, not an error. */
export type DatasetGovernanceResult =
  | { ok: true; governance: DatasetGovernance | null }
  | { ok: false; unauthenticated: boolean; error: string };

const NULLABLE_TEXT_FIELDS = [
  "recording_identity",
  "derived_from_sha256",
  "generation_pipeline",
  "license",
  "permission_status",
  "stratum_primary",
  "source",
  "acquisition_type",
  "benchmark_family",
  "actor_email_snapshot",
] as const;

const TEXT_FIELDS = [
  "media_sha256",
  "source_lineage_id",
  "dataset_split",
  "actor_id",
  "created_at",
  "updated_at",
] as const;

/** One record out of the payload, or null for anything that is not one. */
export function parseDatasetGovernance(payload: unknown): DatasetGovernance | null {
  if (typeof payload !== "object" || payload === null) {
    return null;
  }

  const record = payload as Record<string, unknown>;

  if (TEXT_FIELDS.some((name) => typeof record[name] !== "string")) {
    return null;
  }
  if (NULLABLE_TEXT_FIELDS.some((name) => record[name] !== null && typeof record[name] !== "string")) {
    return null;
  }
  for (const name of ["redistributable", "private"]) {
    if (record[name] !== null && typeof record[name] !== "boolean") {
      return null;
    }
  }

  const transformations = record.transformations;
  if (
    transformations !== null &&
    !(Array.isArray(transformations) && transformations.every((step) => typeof step === "string"))
  ) {
    return null;
  }

  return {
    media_sha256: record.media_sha256 as string,
    source_lineage_id: record.source_lineage_id as string,
    dataset_split: record.dataset_split as string,
    recording_identity: record.recording_identity as string | null,
    transformations: transformations as string[] | null,
    derived_from_sha256: record.derived_from_sha256 as string | null,
    generation_pipeline: record.generation_pipeline as string | null,
    license: record.license as string | null,
    permission_status: record.permission_status as string | null,
    redistributable: record.redistributable as boolean | null,
    private: record.private as boolean | null,
    stratum_primary: record.stratum_primary as string | null,
    source: record.source as string | null,
    acquisition_type: record.acquisition_type as string | null,
    benchmark_family: record.benchmark_family as string | null,
    actor_id: record.actor_id as string,
    actor_email_snapshot: record.actor_email_snapshot as string | null,
    created_at: record.created_at as string,
    updated_at: record.updated_at as string,
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

/** The governance recorded for one set of bytes, or the fact that none has been. */
export async function fetchDatasetGovernance(sha256: string): Promise<DatasetGovernanceResult> {
  const unavailable = {
    ok: false as const,
    unauthenticated: false,
    error: "Dataset governance is temporarily unavailable.",
  };

  try {
    const response = await fetch(`${apiUrl()}${datasetGovernanceUrl(sha256)}`, {
      cache: "no-store",
      headers: { ...(await sessionHeaders()), ...(await requestIdHeaders()) },
      signal: AbortSignal.timeout(DATASET_GOVERNANCE_TIMEOUT_MS),
    });

    if (response.status === 401) {
      return { ok: false, unauthenticated: true, error: "Sign in to read dataset governance." };
    }

    if (response.status === 403) {
      return { ok: false, unauthenticated: false, error: "This account is not an administrator." };
    }

    if (response.status === 404) {
      if ((await detailOf(response)) === NOT_RECORDED_DETAIL) {
        return { ok: true, governance: null };
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

    const governance = parseDatasetGovernance(await response.json().catch(() => null));
    if (governance === null) {
      return {
        ok: false,
        unauthenticated: false,
        error: "The dataset governance record could not be read.",
      };
    }

    return { ok: true, governance };
  } catch {
    return unavailable;
  }
}
