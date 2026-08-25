import Link from "next/link";

import {
  ABSENT,
  ANALYSIS_STATUS_COMPLETED,
  ANALYSIS_STATUS_FAILED,
  API_URL,
  AUDIO_UNAVAILABLE,
  ActiveSpeakerSignal,
  AnalysisSummary,
  AudioAuthenticitySignal,
  EXTRACTION_FAILED,
  HEALTH_TIMEOUT_MS,
  MediaFacts,
  NO_AUDIO_WINDOWS,
  NO_PROVENANCE,
  NO_SPEAKING_FACES,
  PENDING,
  ProvenanceSignal,
  REMOTE_PROVENANCE,
  RISK_LABELS,
  RISK_STYLES,
  RISK_UNSUPPORTED_STYLE,
  SIGNAL_STATUS_SUCCESS,
  SPEAKER_UNAVAILABLE,
  SyntheticVideoSignal,
  UNAVAILABLE,
  UNMATCHED_VOICE,
  UNSUPPORTED,
  fetchAnalyses,
  isSupportedRiskLevel,
} from "./analysis";


type HealthResponse = {
  status: string;
  database: string;
};

type HealthResult =
  | { reachable: true; httpOk: boolean; health: HealthResponse }
  | { reachable: false; error: string };
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
 * When NVIDIA saw a face speaking, as a compact summary of the stored timeline.
 *
 * Four outcomes, deliberately never merged into fewer: no signal at all, a chain that did
 * not produce a timeline, a detection that ran and saw nobody speaking, and a real
 * timeline. None of them is a finding about the media — a video with no speaking face in
 * it is the ordinary case for most footage, and a detector that failed said nothing at all.
 *
 * The segments are the ones that were persisted, shown in the order the provider observed
 * them. Where the stored timeline stops short of what the detection found, the count says
 * so rather than letting a partial timeline read as the whole one.
 */
function ActiveSpeaker({ signal }: { signal: ActiveSpeakerSignal | null }) {
  if (signal === null) {
    return <>{ABSENT}</>;
  }

  if (signal.status !== SIGNAL_STATUS_SUCCESS) {
    return <span title={`Active speaker: ${signal.status}`}>{SPEAKER_UNAVAILABLE}</span>;
  }

  if (signal.segments.length === 0) {
    return <>{NO_SPEAKING_FACES}</>;
  }

  // The stored count is what is listed; the detection's own total is named beside it
  // whenever the two differ, so the cut is visible rather than silent.
  const total = signal.total_speaking_segments;
  const shown = signal.segments.length;
  const truncated = total !== null && total > shown;
  const count = truncated ? `${shown} of ${total}` : `${shown}`;

  return (
    <details>
      <summary className="cursor-pointer">
        {count} segment{shown === 1 && !truncated ? "" : "s"}
      </summary>
      <ul className="mt-1 space-y-0.5">
        {signal.segments.map((segment) => (
          <li key={`${segment.start_time}-${segment.face_id}`}>
            {segment.start_time.toFixed(2)}s–{segment.end_time.toFixed(2)}s · Face{" "}
            {segment.face_id} · {segment.speaker_label ?? UNMATCHED_VOICE}
          </li>
        ))}
      </ul>
    </details>
  );
}

/**
 * What the local checkpoint emitted for each window of audio it was given.
 *
 * Four outcomes, deliberately never merged into fewer: no signal at all, a reading that did
 * not happen, a reading that ran and stored no windows, and a reading with windows. The
 * third is worded about the evidence rather than the media — no window is not proof that
 * the file carries no audio, and this page must not assert one from the other.
 *
 * The figures are shown as the model emitted them, in graph order and rounded only for
 * width. Nothing here averages them, ranks them, turns them into a percentage or a
 * verdict, or compares one window against another: the checkpoint ships no threshold and no
 * calibration, so there is nothing to compare against. Consecutive windows of genuine
 * speech routinely disagree, and that is shown rather than smoothed away.
 *
 * The times are the bounds of the windows DeepGuard cut and fed to the model. They are not
 * segments the model found anything in.
 */
function AudioEvidence({ signal }: { signal: AudioAuthenticitySignal | null }) {
  if (signal === null) {
    return <>{ABSENT}</>;
  }

  if (signal.status !== SIGNAL_STATUS_SUCCESS) {
    return <span title={`Audio authenticity: ${signal.status}`}>{AUDIO_UNAVAILABLE}</span>;
  }

  if (signal.windows.length === 0) {
    return <>{NO_AUDIO_WINDOWS}</>;
  }

  // The stored count is what is listed; the sweep's own total is named beside it whenever
  // the two differ, so the cut is visible rather than silent.
  const total = signal.total_audio_windows;
  const shown = signal.windows.length;
  const truncated = total !== null && total > shown;
  const count = truncated ? `${shown} of ${total}` : `${shown}`;

  return (
    <details>
      <summary className="cursor-pointer" title={audioModelTitle(signal)}>
        {count} audio window{shown === 1 && !truncated ? "" : "s"}
      </summary>
      <ul className="mt-1 space-y-0.5">
        {signal.windows.map((window) => (
          <li key={window.clip_index}>
            {window.start_time.toFixed(2)}s–{window.end_time.toFixed(2)}s · Raw logit[0]:{" "}
            {window.logit.toFixed(2)} · Bona fide logit: {window.bona_fide_logit.toFixed(2)}
          </li>
        ))}
      </ul>
    </details>
  );
}

/** Which checkpoint produced these figures. A different revision is a different reading. */
function audioModelTitle(signal: AudioAuthenticitySignal): string | undefined {
  return signal.provider_version ? `Checkpoint: ${signal.provider_version}` : undefined;
}

/**
 * What the C2PA reading found, in C2PA's own words where it has any.
 *
 * Five outcomes, deliberately never merged into fewer:
 * no signal at all, a reading that failed, a reading that found no credentials, a reading
 * that found a manifest kept outside the file, and a reading that found one inside it —
 * where the C2PA validation state is shown verbatim rather than being translated into
 * "authentic" or "tampered". A file with no Content Credentials is the ordinary case, not
 * a suspicious one, and an invalid signature means the credentials do not verify — not
 * that the video is fake. No part of this product owns that judgement.
 *
 * The remote case is separated from the absent one because they are different facts. A
 * file that names a manifest stored elsewhere did claim provenance; the manifest simply is
 * not in these bytes, and DeepGuard deliberately did not go and get it — fetching a URL an
 * uploaded file supplies would let that file steer a request out of the worker. Reporting
 * it as "No provenance" would credit the file with a claim it never made.
 */
function provenanceText(signal: ProvenanceSignal | null): string {
  if (signal === null) {
    return ABSENT;
  }

  if (signal.status !== SIGNAL_STATUS_SUCCESS) {
    return EXTRACTION_FAILED;
  }

  if (signal.manifest_exists === false) {
    return signal.remote_manifest_url === null ? NO_PROVENANCE : REMOTE_PROVENANCE;
  }

  // A manifest the reading could not describe, which the API reports rather than guesses.
  return signal.validation_state ?? UNAVAILABLE;
}

/**
 * The hover detail behind one provenance cell: which SDK read the file, and — for a remote
 * manifest — the URL it named. The URL is shown as text and never as a link: it comes out
 * of an uploaded file, and offering it as something to click would hand the reader the
 * fetch the worker refused to make.
 */
function provenanceTitle(signal: ProvenanceSignal | null): string | undefined {
  if (signal === null) {
    return undefined;
  }

  const parts = [
    signal.provider_version ? `C2PA SDK: ${signal.provider_version}` : null,
    signal.remote_manifest_url ? `Manifest URL (not fetched): ${signal.remote_manifest_url}` : null,
  ].filter((part) => part !== null);

  return parts.length > 0 ? parts.join(" · ") : undefined;
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
      <div title={provenanceTitle(signal)}>{provenanceText(signal)}</div>
      {source && <div className="opacity-60">{source}</div>}
    </>
  );
}

/**
 * Whether an analysis has run its course, and so whether a decision is still owed.
 *
 * Only `completed` and `failed` are ends. Everything before them — `queued`, and whatever
 * the pipeline calls the states in between — is an analysis whose risk decision has simply
 * not been taken yet, which is why the absence is named against the status rather than
 * being read as a verdict of its own.
 */
function isDecided(status: string): boolean {
  return status === ANALYSIS_STATUS_COMPLETED || status === ANALYSIS_STATUS_FAILED;
}

/** The head of the calibration hash. The full value stays on hover, never truncated away. */
function shortCalibration(id: string): string {
  return `${id.slice(0, 8)}…`;
}

/**
 * The risk DeepGuard classified one analysis at, with the trace that makes it explainable.
 *
 * This is a **risk** classification, not a Fake/Real determination, and the column is
 * worded throughout so it cannot be read as one. `High risk` says the calibrated evidence
 * crossed a threshold measured for that purpose; it does not say the media is a deepfake.
 * `Medium risk` is the indeterminate band — evidence that settles nothing — and emphatically
 * not "probably synthetic". No level here rules anything out either: an analysis that is
 * not HIGH has not been cleared of face manipulation, it has only failed to trip a rule
 * that looks at one calibrated signal.
 *
 * Five states, deliberately never merged into fewer:
 *
 * - a level the engine concluded (`HIGH`, `MEDIUM`, `UNKNOWN`);
 * - no decision on an analysis still working, which is `Pending`;
 * - no decision on an analysis that finished, which is nothing at all — everything stored
 *   before the engine existed;
 * - `UNKNOWN`, which belongs to the first group and not the last two: the engine ran, a
 *   rule fired, and the answer is that the evidence supports no classification;
 * - a non-null level outside the allowlist, which is `Unsupported` — the row holds a state
 *   this build has no calibrated meaning for, and saying so is the whole of what is known.
 *
 * The decision is displayed exactly as the API read it off the row. Nothing here derives a
 * level, and in particular nothing looks at the detector scores in the same row to do it:
 * the decision was taken once, under a named ruleset, and re-deriving it in a browser would
 * let the page contradict the record.
 *
 * What the page will not do is repeat a value it cannot vouch for. Presenting an unknown
 * string in the risk column would let whatever is in the database — a level from a later
 * ruleset, a `LOW` this ruleset disabled, a hand-written `FAKE` — appear as an official
 * DeepGuard classification, and the dashboard has no basis for any of those. So the badge is
 * reserved for the allowlist and everything else is named as unsupported.
 */
function Risk({ analysis }: { analysis: AnalysisSummary }) {
  const level = analysis.risk_level;

  if (level === null) {
    return isDecided(analysis.status) ? (
      <>{ABSENT}</>
    ) : (
      <span title={`Analysis ${analysis.status}: no risk decision has been taken yet.`}>
        {PENDING}
      </span>
    );
  }

  // The stored value is only ever a lookup key, never something to echo. A level that is
  // not on the allowlist gets the unsupported state and neutral styling; the raw string
  // stays available for diagnosis on hover, where it reads as the datum it is rather than
  // as a risk class this product recognizes.
  if (!isSupportedRiskLevel(level)) {
    return (
      <div>
        <div
          className={`inline-block rounded px-1.5 py-0.5 ${RISK_UNSUPPORTED_STYLE}`}
          title={`Stored risk state ${level} is not a supported DeepGuard risk classification${
            analysis.risk_rules_version ? ` (ruleset ${analysis.risk_rules_version})` : ""
          }.`}
        >
          {UNSUPPORTED}
        </div>
      </div>
    );
  }

  return (
    <div>
      <div className={`inline-block rounded px-1.5 py-0.5 ${RISK_STYLES[level]}`}>
        {RISK_LABELS[level]}
      </div>
      {analysis.risk_rules_version && (
        <div className="opacity-60">{analysis.risk_rules_version}</div>
      )}
      {/* The rest of the trace, one <details> away. A level is only explainable alongside
          the rule that produced it and the measurement that rule was calibrated on, so
          all three stay reachable without a detail page or any client-side state. */}
      <details className="mt-0.5">
        <summary className="cursor-pointer opacity-60">Trace</summary>
        <ul className="mt-1 space-y-0.5">
          <li>Rule: {analysis.risk_rule_id ?? ABSENT}</li>
          <li>Ruleset: {analysis.risk_rules_version ?? ABSENT}</li>
          <li title={analysis.risk_calibration_id ?? undefined}>
            Calibration:{" "}
            {analysis.risk_calibration_id
              ? shortCalibration(analysis.risk_calibration_id)
              : ABSENT}
          </li>
        </ul>
      </details>
    </div>
  );
}

/** The provider's frame rate as text, without inventing precision it does not have. */
function frameRateText(rate: number): string {
  return Number.isInteger(rate) ? `${rate}` : rate.toFixed(2);
}

/**
 * The container and codec evidence ffprobe read out of the original.
 *
 * A compact summary, not the whole record: codec, resolution and frame rate on one line
 * and the container beneath, with the rest on hover. Every figure is ffprobe's, shown as
 * ffprobe reported it — `format_name` in particular is the demuxer family, one string
 * covering MOV and MP4 alike, and it is not narrowed to a container name the stored
 * evidence does not establish.
 */
function Media({ media }: { media: MediaFacts }) {
  const detail = [
    `${media.duration.toFixed(2)}s`,
    media.pix_fmt,
    media.constant_frame_rate ? "constant frame rate" : "variable frame rate",
  ]
    .filter((part) => part !== null)
    .join(" · ");

  return (
    <div title={detail}>
      <div>
        {media.codec_name} · {media.width}×{media.height} · {frameRateText(media.frame_rate)} fps
      </div>
      <div className="opacity-60">{media.format_name}</div>
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
            <th className="py-2 pr-4 font-medium">Declared type</th>
            <th className="py-2 pr-4 font-medium">Media (ffprobe)</th>
            <th className="py-2 pr-4 font-medium">Status</th>
            <th className="py-2 pr-4 font-medium">Risk</th>
            <th className="py-2 pr-4 font-medium">Normalized</th>
            <th className="py-2 pr-4 font-medium">NVIDIA SVD</th>
            <th className="py-2 pr-4 font-medium">Synthetic probability</th>
            <th className="py-2 pr-4 font-medium">Clips</th>
            <th className="py-2 pr-4 font-medium">Strongest clips (logit)</th>
            <th className="py-2 pr-4 font-medium">Active speaker</th>
            <th className="py-2 pr-4 font-medium">Audio</th>
            <th className="py-2 pr-4 font-medium">Provenance (C2PA)</th>
            <th className="py-2 pr-4 font-medium">Created</th>
            <th className="py-2 font-medium">Report</th>
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
              {/* What the bytes actually are, as opposed to what the client declared them
                  to be. ffprobe established these before any detector ran. */}
              <td className="py-2 pr-4 align-top font-mono text-xs whitespace-nowrap">
                <Media media={analysis.media} />
              </td>
              <td className="py-2 pr-4">{analysis.status}</td>
              {/* DeepGuard's own classification of the calibrated evidence — a risk level
                  and the ruleset that produced it, never a Fake/Real verdict. Read from
                  the analysis row as the engine committed it, never recomputed here. */}
              <td className="py-2 pr-4 align-top font-mono text-xs whitespace-nowrap">
                <Risk analysis={analysis} />
              </td>
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
              {/* When a tracked face was seen speaking. An absent, failed or empty
                  timeline is never rendered as a finding about the media. */}
              <td className="py-2 pr-4 align-top font-mono text-xs whitespace-nowrap">
                <ActiveSpeaker signal={analysis.active_speaker} />
              </td>
              {/* The raw figures the local checkpoint emitted per window of audio, with the
                  preprocessing bounds of the window each came from. Never aggregated, and
                  never rendered as a verdict about the audio. */}
              <td className="py-2 pr-4 align-top font-mono text-xs whitespace-nowrap">
                <AudioEvidence signal={analysis.audio_authenticity} />
              </td>
              {/* What the file itself claims, read from the forensic original. A missing
                  or invalid manifest is never rendered as a verdict about the media. */}
              <td className="py-2 pr-4 align-top font-mono text-xs">
                <Provenance signal={analysis.provenance} />
              </td>
              <td className="py-2 pr-4 font-mono text-xs">{analysis.created_at}</td>
              {/* A separate link, deliberately not the risk badge: a badge that navigated
                  would make the classification look like a control rather than a finding. */}
              <td className="py-2 text-xs whitespace-nowrap">
                <Link href={`/report/${analysis.id}`} className="underline">
                  View report
                </Link>
              </td>
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
          Risk is a deterministic DeepGuard classification based on calibrated forensic
          evidence. It is not a Fake/Real determination.{" "}
          <span className="font-mono">{RISK_LABELS.MEDIUM}</span> is the indeterminate band
          — evidence that settles nothing either way — and the absence of{" "}
          <span className="font-mono">{RISK_LABELS.HIGH}</span> does not rule out face
          manipulation. <span className="font-mono">{RISK_LABELS.UNKNOWN}</span> means the
          engine ran and could not classify, which is not the same as{" "}
          <span className="font-mono">{PENDING}</span>, where no decision has been taken
          yet, or <span className="font-mono">{ABSENT}</span>, where an analysis finished
          before there was an engine to take one. Each level is shown with the ruleset that
          produced it, since the same word means something different under a different one.{" "}
          <span className="font-mono">{UNSUPPORTED}</span> means the stored state is not one
          this build classifies under, so it is reported as unsupported rather than shown as
          a risk class DeepGuard has no calibrated meaning for.
        </p>
        <p className="mt-1 text-sm opacity-70">
          Synthetic probability is NVIDIA&apos;s own score for its synthetic-video detector,
          shown as returned. It is not a verdict.
        </p>
        <p className="mt-1 text-sm opacity-70">
          Provenance is what the file itself carries: C2PA Content Credentials, read from
          the forensic original and shown in C2PA&apos;s own words. Most media carries none,
          so <span className="font-mono">{NO_PROVENANCE}</span> is the ordinary case and not
          a finding — and an invalid manifest means the credentials do not verify, not that
          the media is fake. <span className="font-mono">{REMOTE_PROVENANCE}</span> means the
          file named a manifest stored somewhere else; that URL was recorded and deliberately
          never visited, so nothing is known about what it holds.
        </p>
        <p className="mt-1 text-sm opacity-70">
          Media is what ffprobe read out of the original before any detector ran, shown as
          ffprobe reported it. The container is its demuxer family — one name covers MOV and
          MP4 alike — and it is not narrowed to a container the stored evidence cannot prove.
          The declared type beside it is only what the client claimed.
        </p>
        <p className="mt-1 text-sm opacity-70">
          Active speaker is when NVIDIA saw a tracked face speaking, in seconds from the start
          of the analysed video, with the face it tracked and the diarized voice matched to
          it. It is a record of what was observed, not a finding:{" "}
          <span className="font-mono">{NO_SPEAKING_FACES}</span> means the detector ran and
          saw nobody speaking, which is the ordinary case for most footage, and{" "}
          <span className="font-mono">{SPEAKER_UNAVAILABLE}</span> means it did not get to
          look at all. Neither says the video is fake.
        </p>
        <p className="mt-1 text-sm opacity-70">
          Audio is the two raw logits a local anti-spoofing checkpoint emitted for each
          window of audio it was given, shown as emitted. The times are the bounds of those
          windows — DeepGuard cut the audio into fixed 4.04s pieces because that is all the
          model accepts — and not stretches the model found anything in. The model publishes
          no threshold and no calibration, so neither figure is a probability, a confidence
          or a verdict, and consecutive windows of genuine speech routinely disagree.{" "}
          <span className="font-mono">{NO_AUDIO_WINDOWS}</span> means the reading ran and
          stored none, which is not proof the file carries no audio, and{" "}
          <span className="font-mono">{AUDIO_UNAVAILABLE}</span> means it did not get to run.
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
