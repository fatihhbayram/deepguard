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
    <div className="flex items-center justify-between gap-6 py-2">
      <span className="text-sm text-muted">{label}</span>
      <span className="flex items-center gap-2 font-mono text-xs">
        <span
          aria-hidden
          className={`inline-block size-2 rounded-full ${ok ? "bg-emerald-500" : "bg-rose-500"}`}
        />
        {detail}
      </span>
    </div>
  );
}

/**
 * The one drawn mark on this page.
 *
 * There is no icon library in this project and adding one is outside this task, so the
 * disclosure chevron is authored here and reused by every `<details>` that gets a control
 * affordance — one path, one stroke weight, one size. Nothing else on the page needs a
 * glyph, which is why nothing else has one.
 */
function Chevron({ className = "" }: { className?: string }) {
  return (
    <svg
      aria-hidden
      viewBox="0 0 16 16"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.5"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={`size-3.5 shrink-0 transition-transform duration-200 group-open:rotate-180 ${className}`}
    >
      <path d="M4 6.5 8 10.5 12 6.5" />
    </svg>
  );
}

/**
 * System health as a header control rather than a block of the page.
 *
 * A native `<details>`, deliberately: the detail behind the summary has to be reachable by
 * click, tap and keyboard, and this page ships no JavaScript to open a scripted popover
 * with. The panel is absolutely positioned and the header clips nothing, so it escapes the
 * bar instead of being cut off by it.
 *
 * What it deliberately does not show: the API's base URL, and the reason the health probe
 * failed. `fetchHealth` still distinguishes a timeout from a refused connection from an
 * unparseable body, but that text is a transport error carrying an internal hostname, a
 * port, or a socket message, and this page is served to anyone who can reach it. The three
 * rows below say which component is not answering, which is the whole of what a reader
 * needs; `unreachable` and `unknown` are the generic states that stand in for the detail.
 * Do not render `result.error` or `API_URL` here.
 */
function HealthControl({
  result,
  apiOk,
  dbOk,
  systemOk,
}: {
  result: HealthResult;
  apiOk: boolean;
  dbOk: boolean;
  systemOk: boolean;
}) {
  return (
    <details className="group relative">
      <summary className="flex cursor-pointer list-none items-center gap-2 rounded-full border border-line px-3 py-1.5 text-xs font-medium tracking-wide transition-colors duration-150 select-none hover:bg-surface [&::-webkit-details-marker]:hidden">
        <span className="sr-only">System status: </span>
        <span
          aria-hidden
          className={`inline-block size-2 rounded-full ${systemOk ? "bg-emerald-500" : "bg-rose-500"}`}
        />
        {systemOk ? "OPERATIONAL" : "DEGRADED"}
        <Chevron className="text-faint" />
      </summary>

      <div className="absolute right-0 z-30 mt-2 w-80 rounded-xl border border-line bg-background p-4 shadow-lg shadow-slate-950/10">
        <p className="text-xs text-faint">Web → API → DB connectivity check</p>

        <div className="mt-3 divide-y divide-line">
          <StatusRow label="Web" ok detail="running" />
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
      </div>
    </details>
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
      <summary className="cursor-pointer text-muted transition-colors duration-150 select-none hover:text-foreground">
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
      <summary
        className="cursor-pointer text-muted transition-colors duration-150 select-none hover:text-foreground"
        title={audioModelTitle(signal)}
      >
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
      {source && <div className="text-faint">{source}</div>}
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
        <div className="text-faint">{analysis.risk_rules_version}</div>
      )}
      {/* The rest of the trace, one <details> away. A level is only explainable alongside
          the rule that produced it and the measurement that rule was calibrated on, so
          all three stay reachable without a detail page or any client-side state. */}
      <details className="mt-0.5">
        <summary className="cursor-pointer text-faint transition-colors duration-150 select-none hover:text-foreground">
          Trace
        </summary>
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
      <div className="text-faint">{media.format_name}</div>
    </div>
  );
}

/** One column heading. Sentence case: several of these are phrases that caps make worse. */
function Th({ children, className = "" }: { children: React.ReactNode; className?: string }) {
  return (
    <th
      scope="col"
      className={`border-b border-line px-3 py-2.5 text-xs font-medium whitespace-nowrap text-muted ${className}`}
    >
      {children}
    </th>
  );
}

/**
 * The evidence table.
 *
 * Sixteen columns of forensic record, so the scroll container owns both axes: it is the
 * vertical scrollport that makes `sticky` work on the header, and the horizontal one that
 * keeps the table usable below its own minimum width instead of crushing the columns. The
 * table carries `tabular-nums` throughout — logits, window bounds and percentages only
 * compare down a column when the digits are the same width.
 *
 * Two weights of cell, not one. Identity and outcome — id, file, status, risk, report —
 * read at full contrast; the measurement behind them is the same evidence at a lower
 * emphasis. Nothing is hidden by the distinction: every figure that was on the page before
 * is still on it, in the same words.
 */
function AnalysisTable({ analyses }: { analyses: AnalysisSummary[] }) {
  return (
    <div className="max-h-[75vh] overflow-auto rounded-lg border border-line">
      <table className="w-full min-w-[1400px] border-collapse text-left text-sm tabular-nums">
        <thead className="sticky top-0 z-10 bg-surface">
          <tr>
            <Th>ID</Th>
            <Th>File</Th>
            <Th>Declared type</Th>
            <Th>Media (ffprobe)</Th>
            <Th>Status</Th>
            <Th>Risk</Th>
            <Th>Normalized</Th>
            <Th>NVIDIA SVD</Th>
            <Th>Synthetic probability</Th>
            <Th>Clips</Th>
            <Th>Strongest clips (logit)</Th>
            <Th>Active speaker</Th>
            <Th>Audio</Th>
            <Th>Provenance (C2PA)</Th>
            <Th>Created</Th>
            <Th>Report</Th>
          </tr>
        </thead>
        <tbody className="divide-y divide-line">
          {analyses.map((analysis) => (
            <tr
              key={analysis.id}
              className="transition-colors duration-150 hover:bg-surface"
            >
              {/* The full id stays in the title so it remains available without a detail page. */}
              <td className="px-3 py-3 align-top font-mono text-xs" title={analysis.id}>
                {analysis.id.slice(0, 8)}
              </td>
              <td
                className="max-w-64 truncate px-3 py-3 align-top"
                title={analysis.original_filename ?? undefined}
              >
                {analysis.original_filename ?? "—"}
              </td>
              <td className="px-3 py-3 align-top font-mono text-xs text-muted">
                {analysis.declared_content_type}
              </td>
              {/* What the bytes actually are, as opposed to what the client declared them
                  to be. ffprobe established these before any detector ran. */}
              <td className="px-3 py-3 align-top font-mono text-xs whitespace-nowrap text-muted">
                <Media media={analysis.media} />
              </td>
              <td className="px-3 py-3 align-top font-medium whitespace-nowrap">
                {analysis.status}
              </td>
              {/* DeepGuard's own classification of the calibrated evidence — a risk level
                  and the ruleset that produced it, never a Fake/Real verdict. Read from
                  the analysis row as the engine committed it, never recomputed here. */}
              <td className="px-3 py-3 align-top font-mono text-xs whitespace-nowrap">
                <Risk analysis={analysis} />
              </td>
              <td className="px-3 py-3 align-top text-muted">
                {analysis.was_normalized ? "yes" : "no"}
              </td>
              {/* The detector's own state, verbatim: SUCCESS, FAILED or TIMEOUT are three
                  different forensic facts, and an analysis may carry no signal at all. */}
              <td
                className="px-3 py-3 align-top font-mono text-xs text-muted"
                title={
                  analysis.synthetic_video?.provider_version
                    ? `Provider version: ${analysis.synthetic_video.provider_version}`
                    : undefined
                }
              >
                {analysis.synthetic_video?.status ?? "no signal"}
              </td>
              <td
                className="px-3 py-3 align-top font-mono text-xs text-muted"
                title={probabilityTitle(analysis.synthetic_video)}
              >
                {probabilityText(analysis.synthetic_video)}
              </td>
              <td className="px-3 py-3 align-top font-mono text-xs text-muted">
                {analysis.synthetic_video?.total_clips ?? ABSENT}
              </td>
              <td className="px-3 py-3 align-top font-mono text-xs whitespace-nowrap text-muted">
                <ClipEvidence signal={analysis.synthetic_video} />
              </td>
              {/* When a tracked face was seen speaking. An absent, failed or empty
                  timeline is never rendered as a finding about the media. */}
              <td className="px-3 py-3 align-top font-mono text-xs whitespace-nowrap text-muted">
                <ActiveSpeaker signal={analysis.active_speaker} />
              </td>
              {/* The raw figures the local checkpoint emitted per window of audio, with the
                  preprocessing bounds of the window each came from. Never aggregated, and
                  never rendered as a verdict about the audio. */}
              <td className="px-3 py-3 align-top font-mono text-xs whitespace-nowrap text-muted">
                <AudioEvidence signal={analysis.audio_authenticity} />
              </td>
              {/* What the file itself claims, read from the forensic original. A missing
                  or invalid manifest is never rendered as a verdict about the media. */}
              <td className="px-3 py-3 align-top font-mono text-xs text-muted">
                <Provenance signal={analysis.provenance} />
              </td>
              <td className="px-3 py-3 align-top font-mono text-xs whitespace-nowrap text-muted">
                {analysis.created_at}
              </td>
              {/* A separate link, deliberately not the risk badge: a badge that navigated
                  would make the classification look like a control rather than a finding. */}
              <td className="px-3 py-3 align-top text-xs whitespace-nowrap">
                <Link
                  href={`/report/${analysis.id}`}
                  className="font-medium underline decoration-line transition-colors duration-150 hover:decoration-current"
                >
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

/**
 * The one control on this dashboard: submit a local file, or a URL, for analysis.
 *
 * A plain HTML form posting to `/submit`, which forwards to the API. No client component and
 * no JavaScript: the rest of this page is server-rendered, the API serves no CORS headers
 * for a browser to post across, and a form is what works without either. The trade is that
 * the outcome arrives as a redirect rather than as an in-place update, which is why the
 * result of the last submission is read out of the query string here.
 *
 * Two inputs and one button, deliberately. There is no queue view, no progress, no history
 * of submissions and no drag-and-drop: an accepted submission becomes a row in the table
 * below, which is where the state of an analysis already lives.
 *
 * A URL submission waits for the download, so the request can take as long as fetching the
 * media takes. That is stated on the form rather than hidden, because the page gives no
 * other sign that anything is happening.
 */
function SubmissionPanel({
  submitted,
  error,
}: {
  submitted: string | null;
  error: string | null;
}) {
  return (
    <section className="rounded-xl border border-line bg-surface p-6 sm:p-8">
      <h2 className="text-lg font-semibold tracking-tight">Submit media</h2>
      <p className="mt-2 max-w-[68ch] text-sm text-muted">
        An MP4 or MOV file, or a link to one. A URL is downloaded by the API first, so the
        page waits for the download before the analysis is queued; both then go through the
        same pipeline and appear in the table below. Live streams cannot be analysed.
      </p>

      {/* One form, two lanes. The divider is presentation only: the route accepts either
          field, so splitting them into separate forms — or into a scripted toggle this
          page has no JavaScript for — would change the submission, not just the look. */}
      <form
        action="/submit"
        method="post"
        encType="multipart/form-data"
        className="mt-6"
      >
        <div className="grid items-stretch gap-5 sm:grid-cols-[1fr_auto_1fr]">
          <label className="flex flex-col gap-2">
            <span className="text-sm font-medium">Upload file</span>
            <input
              type="file"
              name="file"
              accept="video/mp4,video/quicktime"
              className="w-full cursor-pointer rounded-lg border border-line bg-background px-3 py-2.5 text-sm text-muted transition-colors duration-150 file:mr-3 file:cursor-pointer file:rounded-md file:border-0 file:bg-foreground file:px-3 file:py-1.5 file:text-xs file:font-medium file:text-background hover:border-faint"
            />
          </label>

          {/* `sm:pt-7` clears the label line above the inputs, so the rule spans the two
              controls it separates rather than starting level with their labels. */}
          <div className="flex items-center gap-3 sm:flex-col sm:self-stretch sm:pt-7">
            <span aria-hidden className="h-px flex-1 bg-line sm:h-auto sm:w-px" />
            <span className="text-xs text-faint">or</span>
            <span aria-hidden className="h-px flex-1 bg-line sm:h-auto sm:w-px" />
          </div>

          <label className="flex flex-col gap-2">
            <span className="text-sm font-medium">From URL</span>
            <input
              type="url"
              name="url"
              placeholder="https://example.com/clip.mp4"
              className="w-full rounded-lg border border-line bg-background px-3 py-2.5 font-mono text-sm transition-colors duration-150 placeholder:text-faint hover:border-faint"
            />
          </label>
        </div>

        <div className="mt-6 flex flex-col gap-4 border-t border-line pt-5 sm:flex-row sm:items-center sm:justify-between">
          <p className="text-xs text-faint">If both are filled in, the URL is used.</p>
          <button
            type="submit"
            className="shrink-0 rounded-lg bg-foreground px-5 py-2.5 text-sm font-medium text-background transition-[opacity,transform] duration-150 hover:opacity-90 active:translate-y-px"
          >
            Analyse
          </button>
        </div>
      </form>

      {/* The API's own refusal, shown as text. It is DeepGuard's client-facing wording —
          extractor, socket and storage detail stay in the server log — and it says which
          rule the submission broke rather than what went wrong inside. */}
      {error && (
        <div className="mt-6">
          <Alert tone="error">{error}</Alert>
        </div>
      )}
      {submitted !== null && !error && (
        <div className="mt-6">
          <Alert tone="success">
            Queued for analysis
            {submitted ? <span className="font-mono"> · {submitted.slice(0, 8)}</span> : null}.
          </Alert>
        </div>
      )}
    </section>
  );
}

/**
 * The outcome of the last submission, and the only place on this page that speaks in a
 * state colour rather than an evidence one. A tinted field and a hairline, deliberately:
 * the risk column owns the loud colour on this page and an alert that shouted over it
 * would compete with the finding it sits above.
 */
function Alert({
  tone,
  children,
}: {
  tone: "error" | "success";
  children: React.ReactNode;
}) {
  const styles =
    tone === "error"
      ? { field: "border-rose-500/30 bg-rose-500/10 text-rose-700 dark:text-rose-300", dot: "bg-rose-500" }
      : {
          field: "border-emerald-500/30 bg-emerald-500/10 text-emerald-700 dark:text-emerald-300",
          dot: "bg-emerald-500",
        };

  return (
    <p
      role="status"
      className={`flex items-start gap-2.5 rounded-lg border px-4 py-3 text-sm ${styles.field}`}
    >
      <span aria-hidden className={`mt-1.5 size-2 shrink-0 rounded-full ${styles.dot}`} />
      <span>{children}</span>
    </p>
  );
}

/** One query-string value, or null. A repeated parameter is not a submission outcome. */
function singleParam(value: string | string[] | undefined): string | null {
  return typeof value === "string" ? value : null;
}

/** One entry of the methodology disclosure: what a column is, and what it is not. */
function Note({ term, children }: { term: string; children: React.ReactNode }) {
  return (
    <div className="mb-6 break-inside-avoid">
      <dt className="text-sm font-medium">{term}</dt>
      <dd className="mt-1 max-w-[65ch] text-sm text-muted">{children}</dd>
    </div>
  );
}

/**
 * The forensic methodology, one disclosure away rather than seven paragraphs deep.
 *
 * Every word here is the wording the page carried before, unchanged: these are the
 * disclaimers that keep a risk level from being read as a Fake/Real verdict, and they are
 * not the sort of copy a visual pass gets to shorten. What changed is that they are now
 * titled and grouped, so a reader looking up one column is not made to read the other six.
 *
 * A native `<details>`, closed by default and reachable by click, tap and keyboard. Not a
 * tooltip and not a separate page: forensic semantics a reader cannot get to on a phone
 * are semantics the product did not actually publish.
 */
function Methodology() {
  return (
    <details className="group rounded-lg border border-line">
      <summary className="flex cursor-pointer list-none items-center justify-between gap-4 px-4 py-3 text-sm font-medium transition-colors duration-150 select-none hover:bg-surface [&::-webkit-details-marker]:hidden">
        How to interpret these results
        <Chevron className="text-faint" />
      </summary>

      {/* Column flow rather than a grid: these notes differ wildly in length, and grid rows
          size to the tallest cell — which left a short note like "Synthetic probability"
          sitting above a block of dead space as tall as the Risk note beside it. */}
      <dl className="border-t border-line px-4 pt-6 pb-0 sm:columns-2 sm:gap-10">
        <Note term="Risk">
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
        </Note>

        <Note term="Synthetic probability">
          Synthetic probability is NVIDIA&apos;s own score for its synthetic-video detector,
          shown as returned. It is not a verdict.
        </Note>

        <Note term="Provenance (C2PA)">
          Provenance is what the file itself carries: C2PA Content Credentials, read from
          the forensic original and shown in C2PA&apos;s own words. Most media carries none,
          so <span className="font-mono">{NO_PROVENANCE}</span> is the ordinary case and not
          a finding — and an invalid manifest means the credentials do not verify, not that
          the media is fake. <span className="font-mono">{REMOTE_PROVENANCE}</span> means the
          file named a manifest stored somewhere else; that URL was recorded and deliberately
          never visited, so nothing is known about what it holds.
        </Note>

        <Note term="Media (ffprobe)">
          Media is what ffprobe read out of the original before any detector ran, shown as
          ffprobe reported it. The container is its demuxer family — one name covers MOV and
          MP4 alike — and it is not narrowed to a container the stored evidence cannot prove.
          The declared type beside it is only what the client claimed.
        </Note>

        <Note term="Active speaker">
          Active speaker is when NVIDIA saw a tracked face speaking, in seconds from the start
          of the analysed video, with the face it tracked and the diarized voice matched to
          it. It is a record of what was observed, not a finding:{" "}
          <span className="font-mono">{NO_SPEAKING_FACES}</span> means the detector ran and
          saw nobody speaking, which is the ordinary case for most footage, and{" "}
          <span className="font-mono">{SPEAKER_UNAVAILABLE}</span> means it did not get to
          look at all. Neither says the video is fake.
        </Note>

        <Note term="Audio">
          Audio is the two raw logits a local anti-spoofing checkpoint emitted for each
          window of audio it was given, shown as emitted. The times are the bounds of those
          windows — DeepGuard cut the audio into fixed 4.04s pieces because that is all the
          model accepts — and not stretches the model found anything in. The model publishes
          no threshold and no calibration, so neither figure is a probability, a confidence
          or a verdict, and consecutive windows of genuine speech routinely disagree.{" "}
          <span className="font-mono">{NO_AUDIO_WINDOWS}</span> means the reading ran and
          stored none, which is not proof the file carries no audio, and{" "}
          <span className="font-mono">{AUDIO_UNAVAILABLE}</span> means it did not get to run.
        </Note>

        <Note term="Strongest clips (logit)">
          Strongest clips are the highest-scoring of the clips NVIDIA examined, identified by
          frame index because the detector reports no timestamps. The figure is its raw model
          logit, not a probability and not comparable with the percentage beside it.
        </Note>
      </dl>
    </details>
  );
}

export default async function Home({
  searchParams,
}: {
  searchParams: Promise<Record<string, string | string[] | undefined>>;
}) {
  const params = await searchParams;
  const [result, analysesResult] = await Promise.all([fetchHealth(), fetchAnalyses()]);

  const apiOk = result.reachable && result.httpOk && result.health.status === "ok";
  const dbOk = result.reachable && result.health.database === "ok";
  const systemOk = apiOk && dbOk;

  return (
    <>
      {/* The bar carries the product name and the one piece of ambient state worth showing
          at all times. It clips nothing, so the health panel opens over the page below. */}
      <header className="sticky top-0 z-20 border-b border-line bg-surface">
        <div className="mx-auto flex h-16 w-full max-w-[1600px] items-center justify-between gap-4 px-6 sm:px-8">
          <div className="flex items-center gap-3">
            <h1 className="text-base font-semibold tracking-tight">DeepGuard</h1>
            {/* Dropped rather than wrapped below `sm`: a two-line bar reads as a defect,
                and the name alone is enough once the viewport is that narrow. */}
            <span aria-hidden className="hidden h-4 w-px bg-line sm:block" />
            <span className="hidden text-sm whitespace-nowrap text-muted sm:inline">
              Media forensics
            </span>
          </div>

          <HealthControl
            result={result}
            apiOk={apiOk}
            dbOk={dbOk}
            systemOk={systemOk}
          />
        </div>
      </header>

      <main className="mx-auto grid w-full max-w-[1600px] flex-1 gap-10 px-6 py-8 sm:px-8">
        <SubmissionPanel
          submitted={singleParam(params.submitted)}
          error={singleParam(params.error)}
        />

        {/* `min-w-0` is load-bearing: a grid item defaults to `min-width: auto`, which sizes
            this column to the table's 1400px minimum and pushes the overflow onto the page
            instead of leaving it inside the table's own scroll container. */}
        <section className="min-w-0">
          <h2 className="text-lg font-semibold tracking-tight">Recent analyses</h2>
          <div className="mt-4">
            <Methodology />
          </div>

          {!analysesResult.ok ? (
            <div className="mt-4">
              <Alert tone="error">{analysesResult.error}</Alert>
            </div>
          ) : analysesResult.analyses.length === 0 ? (
            <div className="mt-4 rounded-lg border border-dashed border-line px-6 py-12 text-center">
              <p className="text-sm font-medium">No analyses yet</p>
              <p className="mx-auto mt-1 max-w-[48ch] text-sm text-muted">
                Submit a file or a URL above. Each accepted submission becomes a row here.
              </p>
            </div>
          ) : (
            <div className="mt-4">
              <AnalysisTable analyses={analysesResult.analyses} />
            </div>
          )}
        </section>
      </main>
    </>
  );
}
