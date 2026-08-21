// This page renders on the server, so it prefers the Docker-internal API URL and
// falls back to the public one used by the browser.
const API_URL =
  process.env.API_INTERNAL_URL ?? process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

const HEALTH_TIMEOUT_MS = 3000;
const ANALYSES_TIMEOUT_MS = 5000;

type HealthResponse = {
  status: string;
  database: string;
};

type HealthResult =
  | { reachable: true; httpOk: boolean; health: HealthResponse }
  | { reachable: false; error: string };

// Mirrors the API's AnalysisSummary. The MIME string is the one the client declared —
// ffprobe proves the bytes are video but never confirms this value — so the field keeps
// the name that says so.
// One clip NVIDIA scored inside the video. `clip_index` is the provider's frame index for
// the clip's middle frame and `logit` its raw model output — there is no time range and no
// probability here, because NVIDIA reports neither per clip.
type SegmentEvidence = {
  clip_index: number;
  logit: number;
};

// The NVIDIA synthetic-video signal the API joined onto the analysis. `score` is the
// provider's own probability, on the provider's own scale: it is displayed as returned
// and never turned into a verdict or a risk band. It is null for every status other than
// SUCCESS, because a detector that did not answer produced no number.
type SyntheticVideoSignal = {
  provider: string;
  signal_type: string;
  status: string;
  score: number | null;
  provider_version: string | null;
  logit: number | null;
  total_clips: number | null;
  // The strongest few clips behind the aggregate, highest logit first. Empty whenever the
  // detection produced none.
  segments: SegmentEvidence[];
};

// The C2PA provenance signal the API joined onto the analysis. These are facts about what
// the file itself claims: whether it carries Content Credentials at all, and what the C2PA
// SDK made of the signature over them. `validation_state` is the SDK's own word, shown as
// it was returned. None of it says whether the media is real — most media carries no
// credentials, and their absence is a missing signal, not a finding.
type ProvenanceSignal = {
  provider: string;
  signal_type: string;
  status: string;
  provider_version: string | null;
  // Null when the reading itself failed: whether credentials exist is unknown then, which
  // is not the same as knowing there are none.
  manifest_exists: boolean | null;
  validation_state: string | null;
  claim_generator: string | null;
  signature_issuer: string | null;
};

type AnalysisSummary = {
  id: string;
  status: string;
  created_at: string;
  original_filename: string | null;
  declared_content_type: string;
  was_normalized: boolean;
  // Null when the analysis carries no such signal at all — a different fact from a
  // detector that ran and failed.
  synthetic_video: SyntheticVideoSignal | null;
  // Null for an analysis processed before provenance was read at all, which is again not
  // the same as a reading that found no credentials.
  provenance: ProvenanceSignal | null;
};

const SIGNAL_STATUS_SUCCESS = "SUCCESS";

// Shown where a detector produced no figure. Never 0%, which would read as an answer.
const UNAVAILABLE = "N/A";
// Shown where there is no detector result to speak of.
const ABSENT = "—";

// The two provenance outcomes that are not a C2PA state: the file was read and carries no
// credentials, and the file could not be read. Kept apart on purpose — "we looked and
// found nothing" and "we could not look" are different facts, and neither means the media
// is fake.
const NO_PROVENANCE = "No provenance";
const EXTRACTION_FAILED = "Extraction failed";

type AnalysesResult =
  | { ok: true; analyses: AnalysisSummary[] }
  | { ok: false; error: string };

function parseHealth(payload: unknown): HealthResponse | null {
  if (typeof payload !== "object" || payload === null) {
    return null;
  }

  const { status, database } = payload as Record<string, unknown>;
  if (typeof status !== "string" || typeof database !== "string") {
    return null;
  }

  return { status, database };
}

/** A number the API may legitimately have left out, or `undefined` for anything else. */
function parseOptionalNumber(value: unknown): number | null | undefined {
  if (value === null) {
    return null;
  }

  return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}

/** A string the API may legitimately have left out, or `undefined` for anything else. */
function parseOptionalString(value: unknown): string | null | undefined {
  if (value === null) {
    return null;
  }

  return typeof value === "string" ? value : undefined;
}

/** A boolean the API may legitimately have left out, or `undefined` for anything else. */
function parseOptionalBoolean(value: unknown): boolean | null | undefined {
  if (value === null) {
    return null;
  }

  return typeof value === "boolean" ? value : undefined;
}

/**
 * The clip evidence on one signal, or `undefined` if it is not a list of real clips.
 *
 * A malformed entry invalidates the whole list rather than being skipped: silently
 * dropping one clip would misrepresent what the detector reported, and evidence that
 * cannot be trusted is not shown at all.
 */
function parseSegments(payload: unknown): SegmentEvidence[] | undefined {
  if (!Array.isArray(payload)) {
    return undefined;
  }

  const segments: SegmentEvidence[] = [];
  for (const entry of payload) {
    if (typeof entry !== "object" || entry === null) {
      return undefined;
    }

    const { clip_index, logit } = entry as Record<string, unknown>;
    if (
      typeof clip_index !== "number" ||
      !Number.isFinite(clip_index) ||
      typeof logit !== "number" ||
      !Number.isFinite(logit)
    ) {
      return undefined;
    }

    segments.push({ clip_index, logit });
  }

  return segments;
}

/**
 * The signal on one analysis: `null` when the API reported none, `undefined` when the
 * payload was not a signal at all. The two are kept apart because an absent signal is a
 * real state to render, while a malformed one means the response cannot be trusted.
 */
function parseSignal(payload: unknown): SyntheticVideoSignal | null | undefined {
  if (payload === null) {
    return null;
  }

  if (typeof payload !== "object") {
    return undefined;
  }

  const {
    provider,
    signal_type,
    status,
    score,
    provider_version,
    logit,
    total_clips,
    segments,
  } = payload as Record<string, unknown>;

  const parsedScore = parseOptionalNumber(score);
  const parsedLogit = parseOptionalNumber(logit);
  const parsedClips = parseOptionalNumber(total_clips);
  const parsedSegments = parseSegments(segments);

  if (
    typeof provider !== "string" ||
    typeof signal_type !== "string" ||
    typeof status !== "string" ||
    parsedScore === undefined ||
    parsedLogit === undefined ||
    parsedClips === undefined ||
    parsedSegments === undefined ||
    !(typeof provider_version === "string" || provider_version === null)
  ) {
    return undefined;
  }

  return {
    provider,
    signal_type,
    status,
    score: parsedScore,
    provider_version,
    logit: parsedLogit,
    total_clips: parsedClips,
    segments: parsedSegments,
  };
}

/**
 * The provenance signal on one analysis, with the same three-way result as `parseSignal`:
 * `null` for an analysis that carries none, `undefined` for a payload that is not one.
 */
function parseProvenance(payload: unknown): ProvenanceSignal | null | undefined {
  if (payload === null) {
    return null;
  }

  if (typeof payload !== "object") {
    return undefined;
  }

  const {
    provider,
    signal_type,
    status,
    provider_version,
    manifest_exists,
    validation_state,
    claim_generator,
    signature_issuer,
  } = payload as Record<string, unknown>;

  const parsedVersion = parseOptionalString(provider_version);
  const parsedExists = parseOptionalBoolean(manifest_exists);
  const parsedState = parseOptionalString(validation_state);
  const parsedGenerator = parseOptionalString(claim_generator);
  const parsedIssuer = parseOptionalString(signature_issuer);

  if (
    typeof provider !== "string" ||
    typeof signal_type !== "string" ||
    typeof status !== "string" ||
    parsedVersion === undefined ||
    parsedExists === undefined ||
    parsedState === undefined ||
    parsedGenerator === undefined ||
    parsedIssuer === undefined
  ) {
    return undefined;
  }

  return {
    provider,
    signal_type,
    status,
    provider_version: parsedVersion,
    manifest_exists: parsedExists,
    validation_state: parsedState,
    claim_generator: parsedGenerator,
    signature_issuer: parsedIssuer,
  };
}

function parseAnalysis(payload: unknown): AnalysisSummary | null {
  if (typeof payload !== "object" || payload === null) {
    return null;
  }

  const {
    id,
    status,
    created_at,
    original_filename,
    declared_content_type,
    was_normalized,
    synthetic_video,
    provenance,
  } = payload as Record<string, unknown>;

  const signal = parseSignal(synthetic_video);
  const provenanceSignal = parseProvenance(provenance);

  if (
    typeof id !== "string" ||
    typeof status !== "string" ||
    typeof created_at !== "string" ||
    typeof declared_content_type !== "string" ||
    typeof was_normalized !== "boolean" ||
    signal === undefined ||
    provenanceSignal === undefined ||
    !(typeof original_filename === "string" || original_filename === null)
  ) {
    return null;
  }

  return {
    id,
    status,
    created_at,
    original_filename,
    declared_content_type,
    was_normalized,
    synthetic_video: signal,
    provenance: provenanceSignal,
  };
}

async function fetchHealth(): Promise<HealthResult> {
  try {
    const response = await fetch(`${API_URL}/health`, {
      cache: "no-store",
      signal: AbortSignal.timeout(HEALTH_TIMEOUT_MS),
    });

    // The API answers 503 with a meaningful degraded body, so the payload is
    // still worth reading when the status code is not ok.
    const payload = await response.json().catch(() => null);
    const health = parseHealth(payload);
    if (!health) {
      return {
        reachable: false,
        error: `Unexpected response body (HTTP ${response.status})`,
      };
    }

    return { reachable: true, httpOk: response.ok, health };
  } catch (error) {
    if (error instanceof Error && error.name === "TimeoutError") {
      return { reachable: false, error: `No response within ${HEALTH_TIMEOUT_MS}ms` };
    }

    return {
      reachable: false,
      error: error instanceof Error ? error.message : "Unknown error",
    };
  }
}

async function fetchAnalyses(): Promise<AnalysesResult> {
  try {
    const response = await fetch(`${API_URL}/api/v1/analyses`, {
      cache: "no-store",
      signal: AbortSignal.timeout(ANALYSES_TIMEOUT_MS),
    });

    // The API's own failure detail is server-side context, not something to surface
    // here, so an unsuccessful status becomes one generic message.
    if (!response.ok) {
      return { ok: false, error: "The analysis list is temporarily unavailable." };
    }

    const payload = await response.json().catch(() => null);
    if (!Array.isArray(payload)) {
      return { ok: false, error: "The analysis list could not be read." };
    }

    const analyses = payload.map(parseAnalysis);
    if (analyses.some((analysis) => analysis === null)) {
      return { ok: false, error: "The analysis list could not be read." };
    }

    return { ok: true, analyses: analyses as AnalysisSummary[] };
  } catch {
    // Timeouts and connection errors are the same thing to a reader of this page: the
    // list is not available right now. The underlying message could leak internal
    // hostnames, so it is not shown.
    return { ok: false, error: "The analysis list is temporarily unavailable." };
  }
}

function StatusRow({ label, ok, detail }: { label: string; ok: boolean; detail: string }) {
  return (
    <div className="flex items-center justify-between gap-6 border-b border-black/10 py-3 last:border-b-0 dark:border-white/15">
      <span className="font-medium">{label}</span>
      <span className="flex items-center gap-2 font-mono text-sm">
        <span
          aria-hidden
          className={`inline-block size-2.5 rounded-full ${ok ? "bg-green-500" : "bg-red-500"}`}
        />
        {detail}
      </span>
    </div>
  );
}

/**
 * The provider's probability as text, or why there is none.
 *
 * The figure is NVIDIA's, so it is shown as NVIDIA reported it, only rendered as a
 * percentage; it is never bucketed into a risk level or read as "fake" or "real" — no
 * part of the product owns that judgement yet. A detector that failed, timed out or
 * returned nothing has no probability, and saying "0%" would invent one.
 */
function probabilityText(signal: SyntheticVideoSignal | null): string {
  if (signal === null) {
    return ABSENT;
  }

  if (signal.status !== SIGNAL_STATUS_SUCCESS || signal.score === null) {
    return UNAVAILABLE;
  }

  return `${(signal.score * 100).toFixed(2)}%`;
}

/** The exact stored figure, so rounding to two decimals never hides the real number. */
function probabilityTitle(signal: SyntheticVideoSignal | null): string | undefined {
  if (signal === null || signal.score === null) {
    return undefined;
  }

  return `NVIDIA score: ${signal.score}`;
}

/**
 * The strongest clips the detector scored inside one video, highest logit first.
 *
 * Deliberately a plain list. The figure shown is NVIDIA's raw per-clip logit, on the
 * model's own unbounded scale — it is not a percentage and is not comparable with the
 * aggregate probability beside it, so it is neither rescaled nor bucketed. The frame index
 * is shown as the frame index it is: NVIDIA reports no timestamps for a clip, and turning
 * one into a time would be inventing a figure the provider never gave.
 */
function ClipEvidence({ signal }: { signal: SyntheticVideoSignal | null }) {
  if (signal === null) {
    return <>{ABSENT}</>;
  }

  if (signal.segments.length === 0) {
    return <>{UNAVAILABLE}</>;
  }

  return (
    <ul className="space-y-0.5">
      {signal.segments.map((segment) => (
        <li key={segment.clip_index} title={`NVIDIA clip logit: ${segment.logit}`}>
          frame {segment.clip_index} · {segment.logit.toFixed(2)}
        </li>
      ))}
    </ul>
  );
}

/**
 * What the C2PA reading found, in C2PA's own words where it has any.
 *
 * Four outcomes, deliberately never merged into fewer:
 * no signal at all, a reading that failed, a reading that found no credentials, and a
 * reading that found some — where the C2PA validation state is shown verbatim rather than
 * being translated into "authentic" or "tampered". A file with no Content Credentials is
 * the ordinary case, not a suspicious one, and an invalid signature means the credentials
 * do not verify — not that the video is fake. No part of this product owns that judgement.
 */
function provenanceText(signal: ProvenanceSignal | null): string {
  if (signal === null) {
    return ABSENT;
  }

  if (signal.status !== SIGNAL_STATUS_SUCCESS) {
    return EXTRACTION_FAILED;
  }

  if (signal.manifest_exists === false) {
    return NO_PROVENANCE;
  }

  // A manifest the reading could not describe, which the API reports rather than guesses.
  return signal.validation_state ?? UNAVAILABLE;
}

/** Who the manifest says made the media, or who signed for it. Absent unless named. */
function provenanceSource(signal: ProvenanceSignal | null): string | null {
  if (signal === null || signal.manifest_exists !== true) {
    return null;
  }

  return signal.claim_generator ?? signal.signature_issuer;
}

function Provenance({ signal }: { signal: ProvenanceSignal | null }) {
  const source = provenanceSource(signal);

  return (
    <>
      <div title={signal?.provider_version ? `C2PA SDK: ${signal.provider_version}` : undefined}>
        {provenanceText(signal)}
      </div>
      {source && <div className="opacity-60">{source}</div>}
    </>
  );
}

function AnalysisTable({ analyses }: { analyses: AnalysisSummary[] }) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-left text-sm">
        <thead className="border-b border-black/10 text-xs uppercase opacity-70 dark:border-white/15">
          <tr>
            <th className="py-2 pr-4 font-medium">ID</th>
            <th className="py-2 pr-4 font-medium">File</th>
            <th className="py-2 pr-4 font-medium">Type</th>
            <th className="py-2 pr-4 font-medium">Status</th>
            <th className="py-2 pr-4 font-medium">Normalized</th>
            <th className="py-2 pr-4 font-medium">NVIDIA SVD</th>
            <th className="py-2 pr-4 font-medium">Synthetic probability</th>
            <th className="py-2 pr-4 font-medium">Clips</th>
            <th className="py-2 pr-4 font-medium">Strongest clips (logit)</th>
            <th className="py-2 pr-4 font-medium">Provenance (C2PA)</th>
            <th className="py-2 font-medium">Created</th>
          </tr>
        </thead>
        <tbody>
          {analyses.map((analysis) => (
            <tr
              key={analysis.id}
              className="border-b border-black/5 last:border-b-0 dark:border-white/10"
            >
              {/* The full id stays in the title so it remains available without a detail page. */}
              <td className="py-2 pr-4 font-mono text-xs" title={analysis.id}>
                {analysis.id.slice(0, 8)}
              </td>
              <td className="py-2 pr-4">{analysis.original_filename ?? "—"}</td>
              <td className="py-2 pr-4 font-mono text-xs">{analysis.declared_content_type}</td>
              <td className="py-2 pr-4">{analysis.status}</td>
              <td className="py-2 pr-4">{analysis.was_normalized ? "yes" : "no"}</td>
              {/* The detector's own state, verbatim: SUCCESS, FAILED or TIMEOUT are three
                  different forensic facts, and an analysis may carry no signal at all. */}
              <td
                className="py-2 pr-4 font-mono text-xs"
                title={
                  analysis.synthetic_video?.provider_version
                    ? `Provider version: ${analysis.synthetic_video.provider_version}`
                    : undefined
                }
              >
                {analysis.synthetic_video?.status ?? "no signal"}
              </td>
              <td
                className="py-2 pr-4 font-mono text-xs"
                title={probabilityTitle(analysis.synthetic_video)}
              >
                {probabilityText(analysis.synthetic_video)}
              </td>
              <td className="py-2 pr-4 font-mono text-xs">
                {analysis.synthetic_video?.total_clips ?? ABSENT}
              </td>
              <td className="py-2 pr-4 align-top font-mono text-xs whitespace-nowrap">
                <ClipEvidence signal={analysis.synthetic_video} />
              </td>
              {/* What the file itself claims, read from the forensic original. A missing
                  or invalid manifest is never rendered as a verdict about the media. */}
              <td className="py-2 pr-4 align-top font-mono text-xs">
                <Provenance signal={analysis.provenance} />
              </td>
              <td className="py-2 font-mono text-xs">{analysis.created_at}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default async function Home() {
  const [result, analysesResult] = await Promise.all([fetchHealth(), fetchAnalyses()]);

  const apiOk = result.reachable && result.httpOk && result.health.status === "ok";
  const dbOk = result.reachable && result.health.database === "ok";
  const systemOk = apiOk && dbOk;

  return (
    <main className="mx-auto flex w-full max-w-4xl flex-1 flex-col justify-center gap-6 p-8">
      <div>
        <h1 className="text-2xl font-semibold">DeepGuard</h1>
        <p className="text-sm opacity-70">Web → API → DB connectivity check</p>
      </div>

      <div className="rounded-lg border border-black/10 p-6 dark:border-white/15">
        <StatusRow
          label="Web"
          ok
          detail="running"
        />
        <StatusRow
          label="API"
          ok={apiOk}
          detail={result.reachable ? result.health.status : "unreachable"}
        />
        <StatusRow
          label="Database"
          ok={dbOk}
          detail={result.reachable ? result.health.database : "unknown"}
        />
      </div>

      <p className="text-sm">
        System status:{" "}
        <strong className={systemOk ? "text-green-600" : "text-red-600"}>
          {systemOk ? "OPERATIONAL" : "DEGRADED"}
        </strong>
      </p>

      {!result.reachable && (
        <p className="font-mono text-xs opacity-70">
          {API_URL} — {result.error}
        </p>
      )}

      <section className="rounded-lg border border-black/10 p-6 dark:border-white/15">
        <h2 className="text-lg font-semibold">Recent analyses</h2>
        <p className="mt-1 text-sm opacity-70">
          Synthetic probability is NVIDIA&apos;s own score for its synthetic-video detector,
          shown as returned. It is not a verdict.
        </p>
        <p className="mt-1 text-sm opacity-70">
          Provenance is what the file itself carries: C2PA Content Credentials, read from
          the forensic original and shown in C2PA&apos;s own words. Most media carries none,
          so <span className="font-mono">{NO_PROVENANCE}</span> is the ordinary case and not
          a finding — and an invalid manifest means the credentials do not verify, not that
          the media is fake.
        </p>
        <p className="mt-1 text-sm opacity-70">
          Strongest clips are the highest-scoring of the clips NVIDIA examined, identified by
          frame index because the detector reports no timestamps. The figure is its raw model
          logit, not a probability and not comparable with the percentage beside it.
        </p>

        {!analysesResult.ok ? (
          <p className="mt-4 text-sm text-red-600">{analysesResult.error}</p>
        ) : analysesResult.analyses.length === 0 ? (
          <p className="mt-4 text-sm opacity-70">No analyses yet.</p>
        ) : (
          <div className="mt-4">
            <AnalysisTable analyses={analysesResult.analyses} />
          </div>
        )}
      </section>
    </main>
  );
}
