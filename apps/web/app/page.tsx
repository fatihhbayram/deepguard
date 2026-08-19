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
type AnalysisSummary = {
  id: string;
  status: string;
  created_at: string;
  original_filename: string | null;
  declared_content_type: string;
  was_normalized: boolean;
};

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

function parseAnalysis(payload: unknown): AnalysisSummary | null {
  if (typeof payload !== "object" || payload === null) {
    return null;
  }

  const { id, status, created_at, original_filename, declared_content_type, was_normalized } =
    payload as Record<string, unknown>;

  if (
    typeof id !== "string" ||
    typeof status !== "string" ||
    typeof created_at !== "string" ||
    typeof declared_content_type !== "string" ||
    typeof was_normalized !== "boolean" ||
    !(typeof original_filename === "string" || original_filename === null)
  ) {
    return null;
  }

  return { id, status, created_at, original_filename, declared_content_type, was_normalized };
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
