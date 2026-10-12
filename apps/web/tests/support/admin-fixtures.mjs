// The administrative API, as the console's fetchers read it (R16 admin localization).
//
// One payload per endpoint, each in the shape the API sends and the module that reads it accepts,
// and `stubApi` to answer the console's requests with them through a replaced `fetch`. Every
// value a reader can see is chosen to be recognisable — an id, an address, a hash — so a test can
// tell a value from the record apart from a word of the interface.

import { FIXTURES } from "./report-fixtures.mjs";

export const ADMIN = { id: "a0000000-0000-4000-8000-000000000001", email: "root@example.com", role: "ADMIN" };
export const OTHER = "b0000000-0000-4000-8000-000000000002";

export const ANALYSIS = FIXTURES.raporA;
export const SHA = ANALYSIS.original_sha256;

export const ACCOUNTS = [
  { id: ADMIN.id, email: ADMIN.email, role: "ADMIN", is_active: true, created_at: "2026-10-01T09:00:00Z" },
  { id: OTHER, email: "analyst@example.com", role: "USER", is_active: false, created_at: "2026-10-02T09:00:00Z" },
];

export const JOBS = [
  ["queued", false, null],
  ["processing", true, null],
  ["completed", false, null],
  ["failed", false, "worker: nvidia_video TIMEOUT after 120s"],
].map(([status, stale, error], index) => ({
  id: `c000000${index}-0000-4000-8000-00000000000${index}`,
  analysis_id: `d000000${index}-0000-4000-8000-00000000000${index}`,
  status,
  request_id: index === 0 ? null : `req-000${index}`,
  lease_expires_at: status === "processing" ? "2026-10-10T10:05:00Z" : null,
  error_message: error,
  created_at: "2026-10-10T10:00:00Z",
  updated_at: "2026-10-10T10:01:00Z",
  is_stale: stale,
}));

export const QUEUE = {
  total: 3,
  limit: 50,
  offset: 0,
  items: [
    ["UNREVIEWED", null, "MANIPULATION_DETECTED", null, null],
    ["REVIEWED", "DISAGREES_WITH_AUTOMATED_ASSESSMENT", null, "HIGH", null],
    ["NEEDS_FOLLOW_UP", "UNDETERMINED", null, null, "r7-v4.0.0/BOGUS"],
  ].map(([review, assessment, decision, level, unrecognised], index) => ({
    analysis_id: `e000000${index}-0000-4000-8000-00000000000${index}`,
    created_at: "2026-10-09T08:00:00Z",
    status: "completed",
    media_sha256s: [`${String(index).repeat(8)}${SHA.slice(8)}`],
    risk_rules_version: decision === null ? "r7-v4.0.0" : "r9-v5.0.0",
    decision,
    recorded_risk_level: level,
    unrecognised_risk_state: unrecognised,
    review_status: review,
    analyst_assessment: assessment,
    feedback_counts: { AGREE: 0, DISAGREE: index, UNSURE: 1 },
    has_disagree_feedback: index > 0,
    signal_error_counts: { FAILED: index, TIMEOUT: 0 },
    has_signal_errors: index > 0,
  })),
};

// On the wire a bucket's assessment counts sit beside its two named fields (`analytics.ts`).
const bucket = (assessments) => ({ total_feedback: 3, disagreement_rate: 0.333, ...assessments });

export const ANALYTICS = {
  window: "7d",
  analyses_total: 12,
  analyses_by_status: { queued: 1, completed: 10, failed: 1 },
  jobs_by_status: { queued: 1, processing: 0, completed: 10, failed: 1 },
  decisions: { MANIPULATION_DETECTED: 2, NO_CALIBRATED_MANIPULATION_SIGNAL: 5, INCONCLUSIVE: 1, UNDECIDED: 1 },
  recorded_risk_levels: { LOW: 1, MEDIUM: 0, HIGH: 1, UNKNOWN: 1 },
  unrecognised: { "r7-v4.0.0/BOGUS": 1 },
  acquisition: { upload: 8, url: 3, unrecorded: 1 },
  detectors_health: [
    { provider: "nvidia", provider_version: "fn-2026.09", status_counts: { SUCCESS: 9, FAILED: 1, TIMEOUT: 0, MISSING: 1 }, failure_rate: 0.1 },
    { provider: "svd", provider_version: null, status_counts: { SUCCESS: 10 }, failure_rate: 0 },
  ],
  feedback_total: 3,
  feedback_by_assessment: { AGREE: 1, DISAGREE: 1, UNSURE: 1 },
  feedback_by_claimed_label: { GENUINE: 1, AI_GENERATED: 1, FACE_SWAP: 0, OTHER: 0 },
  feedback_without_claimed_label: 1,
  feedback_by_decision: { MANIPULATION_DETECTED: bucket({ AGREE: 1, DISAGREE: 1, UNSURE: 1 }) },
  feedback_by_recorded_risk_level: { HIGH: bucket({ AGREE: 2, DISAGREE: 1 }) },
  feedback_by_unrecognised_risk_state: { "r7-v4.0.0/BOGUS": bucket({ UNSURE: 3 }) },
};

/** Every action the API writes (`app/db/models.py`), and one it does not. */
export const AUDIT_ACTIONS = [
  "USER_CREATED",
  "USER_UPDATED",
  "USER_PASSWORD_RESET",
  "API_KEY_CREATED",
  "API_KEY_REVOKED",
  "REVIEW_CREATED",
  "GROUND_TRUTH_CREATED",
  "GROUND_TRUTH_UPDATED",
  "DATASET_GOVERNANCE_CREATED",
  "DATASET_GOVERNANCE_UPDATED",
];
export const UNKNOWN_ACTION = "ACCOUNT_MERGED";

const TARGETS = {
  USER_CREATED: "USER",
  USER_UPDATED: "USER",
  USER_PASSWORD_RESET: "USER",
  API_KEY_CREATED: "API_KEY",
  API_KEY_REVOKED: "API_KEY",
  REVIEW_CREATED: "ANALYSIS",
  GROUND_TRUTH_CREATED: "GROUND_TRUTH",
  GROUND_TRUTH_UPDATED: "GROUND_TRUTH",
  DATASET_GOVERNANCE_CREATED: "DATASET_GOVERNANCE",
  DATASET_GOVERNANCE_UPDATED: "DATASET_GOVERNANCE",
  [UNKNOWN_ACTION]: "WORKSPACE",
};

/** One readable event per action — every one an `{old, new}` diff, which the parser accepts. */
export function auditEvent(action, index, changes = { is_active: { old: true, new: false } }) {
  return {
    id: `f000000${index % 10}-0000-4000-8000-0000000000${String(index).padStart(2, "0")}`,
    actor_id: ADMIN.id,
    actor_email_snapshot: index === 0 ? null : ADMIN.email,
    action,
    target_type: TARGETS[action] ?? "USER",
    target_id: OTHER,
    target_email_snapshot: "analyst@example.com",
    changes,
    request_id: index === 0 ? null : `req-a${index}`,
    created_at: "2026-10-11T12:00:00Z",
  };
}

export const AUDIT = [...AUDIT_ACTIONS, UNKNOWN_ACTION].map((action, index) =>
  auditEvent(action, index, index === 1 ? { role: { old: "USER", new: "ADMIN" }, is_active: { old: false, new: true } } : undefined),
);

/*
 * The event `app/api/admin_analyses.py` writes when a review moves (`_review_changes`, R14):
 * `note_changed` is a bare boolean, not an `{old, new}` pair. See `admin-audit-runtime.test.mjs`.
 */
export const REVIEW_UPDATED_EVENT = auditEvent("REVIEW_UPDATED", 20, {
  note_changed: false,
  status: { old: "UNREVIEWED", new: "REVIEWED" },
});

export const API_KEYS = [
  { id: "k0000000-0000-4000-8000-000000000001", name: "Acme production", is_active: true, created_at: "2026-10-03T09:00:00Z", last_used_at: null },
  { id: "k0000000-0000-4000-8000-000000000002", name: "Old pilot", is_active: false, created_at: "2026-09-03T09:00:00Z", last_used_at: "2026-09-04T09:00:00Z" },
];

export const REVIEW = {
  analysis_id: ANALYSIS.id,
  status: "REVIEWED",
  analyst_assessment: "DISAGREES_WITH_AUTOMATED_ASSESSMENT",
  note: "Checked against the uploader's original.",
  reviewer_id: ADMIN.id,
  reviewer_email_snapshot: ADMIN.email,
  created_at: "2026-10-05T09:00:00Z",
  updated_at: "2026-10-06T09:00:00Z",
};

export const LEGACY_REVIEW = { ...REVIEW, analyst_assessment: null };

export const UNREVIEWED = {
  analysis_id: ANALYSIS.id,
  status: "UNREVIEWED",
  analyst_assessment: null,
  note: "",
  reviewer_id: null,
  reviewer_email_snapshot: null,
  created_at: null,
  updated_at: null,
};

export const GROUND_TRUTH = {
  media_sha256: SHA,
  source_class: "OWNER_KNOWN",
  label: "GENUINE",
  notes: null,
  actor_id: ADMIN.id,
  actor_email_snapshot: ADMIN.email,
  created_at: "2026-10-05T10:00:00Z",
  updated_at: "2026-10-05T10:00:00Z",
};

export const GOVERNANCE = {
  media_sha256: SHA,
  source_lineage_id: "lineage-r16p-008",
  dataset_split: "HOLDOUT",
  recording_identity: null,
  transformations: ["transcode:h264"],
  derived_from_sha256: null,
  generation_pipeline: null,
  license: "owner-consent",
  permission_status: "granted",
  redistributable: false,
  private: true,
  stratum_primary: null,
  source: null,
  acquisition_type: null,
  benchmark_family: null,
  actor_id: ADMIN.id,
  actor_email_snapshot: ADMIN.email,
  created_at: "2026-10-05T10:00:00Z",
  updated_at: "2026-10-05T10:00:00Z",
};

export const FEEDBACK = [
  { user_id: OTHER, user_email: "analyst@example.com", assessment: "DISAGREE", claimed_label: "GENUINE", notes: null, created_at: "2026-10-04T09:00:00Z", updated_at: "2026-10-04T09:00:00Z" },
  { user_id: "b0000000-0000-4000-8000-000000000003", user_email: "owner@example.com", assessment: "UNSURE", claimed_label: null, notes: "Not sure.", created_at: "2026-10-04T10:00:00Z", updated_at: "2026-10-04T10:00:00Z" },
];

/**
 * Answer the console's requests. `overrides` maps a path (or a path prefix ending in `*`) to a
 * payload, or to `{ status, body }` for a refusal. Anything not named is answered from the
 * fixtures above; anything unknown is a 404.
 */
export function stubApi(overrides = {}) {
  const defaults = {
    "/api/v1/auth/me": ADMIN,
    "/api/v1/admin/users": ACCOUNTS,
    [`/api/v1/admin/users/${OTHER}`]: ACCOUNTS[1],
    [`/api/v1/admin/users/${ADMIN.id}`]: ACCOUNTS[0],
    "/api/v1/admin/jobs": JOBS,
    "/api/v1/admin/analyses/queue": QUEUE,
    "/api/v1/admin/analytics": ANALYTICS,
    "/api/v1/admin/audit": AUDIT,
    "/api/v1/admin/api-keys": API_KEYS,
    [`/api/v1/analyses/${ANALYSIS.id}`]: ANALYSIS,
    [`/api/v1/admin/analyses/${ANALYSIS.id}/review`]: REVIEW,
    [`/api/v1/admin/analyses/${ANALYSIS.id}/feedback`]: FEEDBACK,
    [`/api/v1/admin/ground-truth/${SHA}`]: GROUND_TRUTH,
    [`/api/v1/admin/dataset-governance/${SHA}`]: GOVERNANCE,
  };
  const table = { ...defaults, ...overrides };
  globalThis.fetch = async (url) => {
    const path = new URL(String(url)).pathname;
    const key = Object.hasOwn(table, path)
      ? path
      : Object.keys(table).find((name) => name.endsWith("*") && path.startsWith(name.slice(0, -1)));
    if (key === undefined) return new Response("{}", { status: 404 });
    const answer = table[key];
    if (answer && typeof answer === "object" && "status" in answer && "body" in answer && Object.keys(answer).length === 2) {
      return Response.json(answer.body, { status: answer.status });
    }
    return Response.json(answer);
  };
}
