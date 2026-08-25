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
  // Where the file says its manifest lives when it is not inside the file. Present
  // alongside `manifest_exists: false`, and the two together are a different fact from a
  // file that claims no provenance at all. The URL was recorded and never visited.
  remote_manifest_url: string | null;
};

// One stretch of video in which NVIDIA saw a tracked face speaking. The times are seconds
// from the start of the analysed video, `face_id` is NVIDIA's own identifier for the face
// it tracked, and `speaker_label` is the diarized voice matched to it — null when NVIDIA
// matched none, which is an observation about the segment rather than missing data.
type SpeakingSegment = {
  start_time: number;
  end_time: number;
  face_id: number;
  speaker_label: string | null;
};

// The NVIDIA active-speaker signal the API joined onto the analysis. It carries no score:
// this detector reports a timeline, not a figure on a scale. A successful signal with no
// segments means the detector looked and saw nobody speaking — a real result, and not a
// finding about the media.
type ActiveSpeakerSignal = {
  provider: string;
  signal_type: string;
  status: string;
  provider_version: string | null;
  // How many speaking runs the detection found, and whether the stored timeline stops short
  // of them. Both null unless the detection succeeded and recorded them.
  total_speaking_segments: number | null;
  segments_truncated: boolean | null;
  // The persisted timeline, chronological. Empty for a detection that produced none.
  segments: SpeakingSegment[];
};

// One window of audio the local checkpoint was given, and the two raw figures it emitted.
// `clip_index` is the window's place in the sequence DeepGuard cut. The times are that
// window's sample bounds in seconds — preprocessing bounds, not timestamps the model
// produced: AASIST publishes no mapping from its fixed window to time and reports no
// segments. Both logits are raw model output in graph order; upstream reads the second as
// its bona fide column. Neither is a probability or a confidence.
type AudioWindow = {
  clip_index: number;
  start_time: number;
  end_time: number;
  logit: number;
  bona_fide_logit: number;
};

// The local audio-authenticity signal the API joined onto the analysis. It carries no score
// and there is nothing here that could stand in for one: the checkpoint ships no threshold,
// no calibration and no classes, so the windows are the whole of the evidence and they are
// neither averaged nor ranked.
type AudioAuthenticitySignal = {
  provider: string;
  signal_type: string;
  status: string;
  provider_version: string | null;
  // How many windows the sweep produced against how many were stored, and whether the
  // stored set stops short. All null unless the reading succeeded and recorded them.
  total_audio_windows: number | null;
  persisted_audio_windows: number | null;
  windows_truncated: boolean | null;
  // The persisted windows, in the order the audio was cut. Empty for a reading that
  // produced none.
  windows: AudioWindow[];
};

// What ffprobe established about the forensic original, as the database kept it. These
// are facts about the bytes, unlike `declared_content_type`, which is only what the client
// said about them. `format_name` is ffprobe's own name for the demuxer family — one string
// covers MOV and MP4 alike — so it is shown as ffprobe worded it rather than narrowed to a
// container this page cannot prove.
type MediaFacts = {
  format_name: string;
  codec_name: string;
  width: number;
  height: number;
  duration: number;
  frame_rate: number;
  pix_fmt: string | null;
  constant_frame_rate: boolean;
};

type AnalysisSummary = {
  id: string;
  status: string;
  created_at: string;
  original_filename: string | null;
  declared_content_type: string;
  was_normalized: boolean;
  // Never null: an analysis and its media are written in one transaction.
  media: MediaFacts;
  // Null when the analysis carries no such signal at all — a different fact from a
  // detector that ran and failed.
  synthetic_video: SyntheticVideoSignal | null;
  // Null for an analysis processed before provenance was read at all, which is again not
  // the same as a reading that found no credentials.
  provenance: ProvenanceSignal | null;
  // Null when the analysis carries no active-speaker signal at all, which is not the same
  // as a detector that ran and saw no speaking face.
  active_speaker: ActiveSpeakerSignal | null;
  // Null when the analysis carries no audio-authenticity signal at all — everything stored
  // before the local checkpoint was wired in. Not the same as a reading that ran and stored
  // no windows, and not the same as one that could not run.
  audio_authenticity: AudioAuthenticitySignal | null;
};

const SIGNAL_STATUS_SUCCESS = "SUCCESS";

// Shown where a detector produced no figure. Never 0%, which would read as an answer.
const UNAVAILABLE = "N/A";
// Shown where there is no detector result to speak of.
const ABSENT = "—";

// The three provenance outcomes that are not a C2PA state: the file was read and carries
// no credentials, the file names a manifest kept somewhere else, and the file could not be
// read. Kept apart on purpose — "we looked and found nothing", "provenance was claimed but
// is not in these bytes" and "we could not look" are different facts, and none of them
// means the media is fake.
// The two active-speaker outcomes that are not a timeline: the chain did not produce one,
// and it produced one that is empty. Kept apart on purpose — "we could not look" and "we
// looked and nobody was speaking" are different facts, and neither says anything about
// whether the media is genuine.
const SPEAKER_UNAVAILABLE = "Unavailable";
const NO_SPEAKING_FACES = "No speaking faces detected";
// Shown where NVIDIA saw a face speaking but matched it to no diarized voice.
const UNMATCHED_VOICE = "no matched voice";

// The two audio-authenticity outcomes that are not a set of windows: the reading did not
// happen, and it happened and stored none. Kept apart for the same reason as the pair
// above, and phrased to describe the *evidence* rather than the media — nothing here has
// established that a file carries no audio, only that no windows were persisted for it.
const AUDIO_UNAVAILABLE = "Unavailable";
const NO_AUDIO_WINDOWS = "No audio evidence windows";

const NO_PROVENANCE = "No provenance";
const REMOTE_PROVENANCE = "Remote provenance (not fetched)";
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
 * The speaking timeline on one signal, or `undefined` if it is not a list of real segments.
 *
 * A malformed entry invalidates the whole list for the same reason as the clip evidence:
 * a timeline with one range quietly dropped out of it reads as silence that was never
 * observed, and evidence that cannot be trusted is not shown at all.
 */
function parseSpeakingSegments(payload: unknown): SpeakingSegment[] | undefined {
  if (!Array.isArray(payload)) {
    return undefined;
  }

  const segments: SpeakingSegment[] = [];
  for (const entry of payload) {
    if (typeof entry !== "object" || entry === null) {
      return undefined;
    }

    const { start_time, end_time, face_id, speaker_label } = entry as Record<string, unknown>;
    const parsedLabel = parseOptionalString(speaker_label);
    if (
      typeof start_time !== "number" ||
      !Number.isFinite(start_time) ||
      typeof end_time !== "number" ||
      !Number.isFinite(end_time) ||
      typeof face_id !== "number" ||
      !Number.isFinite(face_id) ||
      parsedLabel === undefined
    ) {
      return undefined;
    }

    segments.push({ start_time, end_time, face_id, speaker_label: parsedLabel });
  }

  return segments;
}

/**
 * The active-speaker signal on one analysis, with the same three-way result as
 * `parseSignal`: `null` for an analysis that carries none, `undefined` for a payload that
 * is not one.
 */
function parseActiveSpeaker(payload: unknown): ActiveSpeakerSignal | null | undefined {
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
    total_speaking_segments,
    segments_truncated,
    segments,
  } = payload as Record<string, unknown>;

  const parsedVersion = parseOptionalString(provider_version);
  const parsedTotal = parseOptionalNumber(total_speaking_segments);
  const parsedTruncated = parseOptionalBoolean(segments_truncated);
  const parsedSegments = parseSpeakingSegments(segments);

  if (
    typeof provider !== "string" ||
    typeof signal_type !== "string" ||
    typeof status !== "string" ||
    parsedVersion === undefined ||
    parsedTotal === undefined ||
    parsedTruncated === undefined ||
    parsedSegments === undefined
  ) {
    return undefined;
  }

  return {
    provider,
    signal_type,
    status,
    provider_version: parsedVersion,
    total_speaking_segments: parsedTotal,
    segments_truncated: parsedTruncated,
    segments: parsedSegments,
  };
}

/**
 * The audio windows on one signal, or `undefined` if they are not a list of real windows.
 *
 * A malformed entry invalidates the whole list, as with the other two kinds of evidence.
 * These windows are a contiguous sweep of the audio, so one quietly dropped out of the
 * middle would read as a gap in the recording that was never there.
 */
function parseAudioWindows(payload: unknown): AudioWindow[] | undefined {
  if (!Array.isArray(payload)) {
    return undefined;
  }

  const windows: AudioWindow[] = [];
  for (const entry of payload) {
    if (typeof entry !== "object" || entry === null) {
      return undefined;
    }

    const { clip_index, start_time, end_time, logit, bona_fide_logit } = entry as Record<
      string,
      unknown
    >;
    if (
      typeof clip_index !== "number" ||
      !Number.isFinite(clip_index) ||
      typeof start_time !== "number" ||
      !Number.isFinite(start_time) ||
      typeof end_time !== "number" ||
      !Number.isFinite(end_time) ||
      typeof logit !== "number" ||
      !Number.isFinite(logit) ||
      typeof bona_fide_logit !== "number" ||
      !Number.isFinite(bona_fide_logit)
    ) {
      return undefined;
    }

    windows.push({ clip_index, start_time, end_time, logit, bona_fide_logit });
  }

  return windows;
}

/**
 * The audio-authenticity signal on one analysis, with the same three-way result as
 * `parseSignal`: `null` for an analysis that carries none, `undefined` for a payload that
 * is not one.
 */
function parseAudioAuthenticity(
  payload: unknown,
): AudioAuthenticitySignal | null | undefined {
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
    total_audio_windows,
    persisted_audio_windows,
    windows_truncated,
    windows,
  } = payload as Record<string, unknown>;

  const parsedVersion = parseOptionalString(provider_version);
  const parsedTotal = parseOptionalNumber(total_audio_windows);
  const parsedPersisted = parseOptionalNumber(persisted_audio_windows);
  const parsedTruncated = parseOptionalBoolean(windows_truncated);
  const parsedWindows = parseAudioWindows(windows);

  if (
    typeof provider !== "string" ||
    typeof signal_type !== "string" ||
    typeof status !== "string" ||
    parsedVersion === undefined ||
    parsedTotal === undefined ||
    parsedPersisted === undefined ||
    parsedTruncated === undefined ||
    parsedWindows === undefined
  ) {
    return undefined;
  }

  return {
    provider,
    signal_type,
    status,
    provider_version: parsedVersion,
    total_audio_windows: parsedTotal,
    persisted_audio_windows: parsedPersisted,
    windows_truncated: parsedTruncated,
    windows: parsedWindows,
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
    remote_manifest_url,
  } = payload as Record<string, unknown>;

  const parsedVersion = parseOptionalString(provider_version);
  const parsedExists = parseOptionalBoolean(manifest_exists);
  const parsedState = parseOptionalString(validation_state);
  const parsedGenerator = parseOptionalString(claim_generator);
  const parsedIssuer = parseOptionalString(signature_issuer);
  const parsedRemoteUrl = parseOptionalString(remote_manifest_url);

  if (
    typeof provider !== "string" ||
    typeof signal_type !== "string" ||
    typeof status !== "string" ||
    parsedVersion === undefined ||
    parsedExists === undefined ||
    parsedState === undefined ||
    parsedGenerator === undefined ||
    parsedIssuer === undefined ||
    parsedRemoteUrl === undefined
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
    remote_manifest_url: parsedRemoteUrl,
  };
}

/**
 * The probed media facts on one analysis, or `undefined` if the payload is not a set of
 * them. There is no `null` case: every listed analysis has media, so an absent or
 * malformed object means the response cannot be trusted rather than a state to render.
 */
function parseMedia(payload: unknown): MediaFacts | undefined {
  if (typeof payload !== "object" || payload === null) {
    return undefined;
  }

  const {
    format_name,
    codec_name,
    width,
    height,
    duration,
    frame_rate,
    pix_fmt,
    constant_frame_rate,
  } = payload as Record<string, unknown>;

  const parsedPixFmt = parseOptionalString(pix_fmt);

  if (
    typeof format_name !== "string" ||
    typeof codec_name !== "string" ||
    typeof width !== "number" ||
    !Number.isFinite(width) ||
    typeof height !== "number" ||
    !Number.isFinite(height) ||
    typeof duration !== "number" ||
    !Number.isFinite(duration) ||
    typeof frame_rate !== "number" ||
    !Number.isFinite(frame_rate) ||
    typeof constant_frame_rate !== "boolean" ||
    parsedPixFmt === undefined
  ) {
    return undefined;
  }

  return {
    format_name,
    codec_name,
    width,
    height,
    duration,
    frame_rate,
    pix_fmt: parsedPixFmt,
    constant_frame_rate,
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
    media,
    synthetic_video,
    provenance,
    active_speaker,
    audio_authenticity,
  } = payload as Record<string, unknown>;

  const signal = parseSignal(synthetic_video);
  const provenanceSignal = parseProvenance(provenance);
  const activeSpeaker = parseActiveSpeaker(active_speaker);
  const audioAuthenticity = parseAudioAuthenticity(audio_authenticity);
  const mediaFacts = parseMedia(media);

  if (
    typeof id !== "string" ||
    typeof status !== "string" ||
    typeof created_at !== "string" ||
    typeof declared_content_type !== "string" ||
    typeof was_normalized !== "boolean" ||
    signal === undefined ||
    provenanceSignal === undefined ||
    activeSpeaker === undefined ||
    audioAuthenticity === undefined ||
    mediaFacts === undefined ||
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
    media: mediaFacts,
    synthetic_video: signal,
    provenance: provenanceSignal,
    active_speaker: activeSpeaker,
    audio_authenticity: audioAuthenticity,
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
            <th className="py-2 pr-4 font-medium">Normalized</th>
            <th className="py-2 pr-4 font-medium">NVIDIA SVD</th>
            <th className="py-2 pr-4 font-medium">Synthetic probability</th>
            <th className="py-2 pr-4 font-medium">Clips</th>
            <th className="py-2 pr-4 font-medium">Strongest clips (logit)</th>
            <th className="py-2 pr-4 font-medium">Active speaker</th>
            <th className="py-2 pr-4 font-medium">Audio</th>
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
              {/* What the bytes actually are, as opposed to what the client declared them
                  to be. ffprobe established these before any detector ran. */}
              <td className="py-2 pr-4 align-top font-mono text-xs whitespace-nowrap">
                <Media media={analysis.media} />
              </td>
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
