/**
 * The analysis record as the web app receives it, and the vocabulary for talking about it.
 *
 * Shared by the dashboard listing and the report route. Extracted when the report became the
 * second reader: both render the same persisted record, and two copies of these parsers would
 * be two chances for one of them to accept a payload the other rejects — a report disagreeing
 * with the dashboard about the same analysis is exactly the failure this file prevents.
 *
 * Nothing here derives a figure. Every function either passes a stored value through or
 * refuses the payload; in particular nothing turns a detector score into a classification.
 *
 * Every read below is authenticated, since R1-T2. The API decides which analyses a session
 * may see and answers 401 to one it does not recognise and 404 to an analysis outside its
 * reach; what these functions do with that is carry the distinction to the page, which
 * sends the reader to sign in. Nothing here filters by owner — a page that hid another
 * account's analysis after receiving it would be a security boundary drawn in a browser.
 *
 * Every call also carries this browser request's id (R1-T4), beside the session cookie and
 * for a comparable reason: the cookie says who is asking, and the id says which asking this
 * is. One page render makes three of these calls and they all report the same id, so the
 * API's log lines for them group with this server's — see `./observability`.
 */

import { requestIdHeaders } from "./observability";
import { SESSION_TIMEOUT_MS, SessionUser, parseSessionUser, sessionHeaders } from "./session";

/**
 * Where this server sends its API calls, resolved on every call rather than once (R1-T5).
 *
 * It used to be a module-level constant with `http://localhost:8000` behind it. Both halves
 * of that were wrong for a deployment. The constant is evaluated when the module is first
 * loaded, which during `next build` is while the page is being prerendered — inside the
 * image, where `API_INTERNAL_URL` is not set — so a value read once could be baked from the
 * build environment instead of the running one. Reading it per call is what makes
 * `docker compose up` with a changed `.env` actually change where this server points, with
 * no rebuild. It costs two property lookups on a path that is about to make a network call.
 *
 * The localhost fallback is gone for the same reason the backend's credential defaults are
 * (see `apps/api/app/config.py`): a deployment that forgot to configure this would come up
 * looking healthy and quietly call *itself* on port 8000, and "connection refused" from a
 * hostname nobody configured is a much harder thing to diagnose than a stated refusal.
 *
 * `API_INTERNAL_URL` wins because both readers of this run on the server, where the API is
 * reachable over the Docker network under a name the browser cannot resolve.
 * `NEXT_PUBLIC_API_URL` is the fallback for running this application outside Compose, where
 * the two URLs are the same thing.
 */
export function apiUrl(): string {
  const configured = process.env.API_INTERNAL_URL ?? process.env.NEXT_PUBLIC_API_URL ?? "";

  if (!configured.trim()) {
    throw new Error(
      "Neither API_INTERNAL_URL nor NEXT_PUBLIC_API_URL is set. This server has no API to " +
        "call: set one of them in the environment or in the repository's .env file.",
    );
  }

  return configured.trim();
}

export const HEALTH_TIMEOUT_MS = 3000;
export const ANALYSES_TIMEOUT_MS = 5000;
// Mirrors the API's AnalysisSummary. The MIME string is the one the client declared —
// ffprobe proves the bytes are video but never confirms this value — so the field keeps
// the name that says so.
// One clip NVIDIA scored inside the video. `clip_index` is the provider's frame index for
// the clip's middle frame and `logit` its raw model output — there is no time range and no
// probability here, because NVIDIA reports neither per clip.
export type SegmentEvidence = {
  clip_index: number;
  logit: number;
};

// The NVIDIA synthetic-video signal the API joined onto the analysis. `score` is the
// provider's own probability, on the provider's own scale: it is displayed as returned
// and never turned into a verdict or a risk band. It is null for every status other than
// SUCCESS, because a detector that did not answer produced no number.
export type SyntheticVideoSignal = {
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
export type ProvenanceSignal = {
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
export type SpeakingSegment = {
  start_time: number;
  end_time: number;
  face_id: number;
  speaker_label: string | null;
};

// The NVIDIA active-speaker signal the API joined onto the analysis. It carries no score:
// this detector reports a timeline, not a figure on a scale. A successful signal with no
// segments means the detector looked and saw nobody speaking — a real result, and not a
// finding about the media.
export type ActiveSpeakerSignal = {
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
export type AudioWindow = {
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
export type AudioAuthenticitySignal = {
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
export type MediaFacts = {
  format_name: string;
  codec_name: string;
  width: number;
  height: number;
  duration: number;
  frame_rate: number;
  pix_fmt: string | null;
  constant_frame_rate: boolean;
};

export type AnalysisSummary = {
  id: string;
  status: string;
  created_at: string;
  // What the risk engine concluded, as the API read it off the analysis row. `HIGH`,
  // `MEDIUM` or `UNKNOWN` — never `LOW`, which ruleset v1 measures but does not emit.
  //
  // Null is not `UNKNOWN`. Null means no decision exists for this analysis; `UNKNOWN` means
  // one was taken and it is that the evidence supports no classification. This page keeps
  // the two apart everywhere, because collapsing them would turn "we have not looked yet"
  // and "we looked and cannot say" into the same sentence.
  risk_level: string | null;
  // The trace behind the level: the immutable ruleset in force, the single rule that fired
  // and the calibration its thresholds were measured under. Null exactly when the level is.
  // A level without them is unreadable once the rules move on, so they are shown with it.
  risk_rules_version: string | null;
  risk_rule_id: string | null;
  risk_calibration_id: string | null;
  original_filename: string | null;
  declared_content_type: string;
  // The hash of the analysed original, and its size on disk. Read by the report, which has
  // to print a hash a reader can check the source file against. Nullable because the column
  // is: an analysis stored before hashing was wired in carries neither.
  original_sha256: string | null;
  size_bytes: number | null;
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
export const SIGNAL_STATUS_SUCCESS = "SUCCESS";

// The two states an analysis ends in. Anything else is still on its way through the
// pipeline, which is what separates a decision that has not been taken yet from one that
// never will be.
export const ANALYSIS_STATUS_COMPLETED = "completed";
export const ANALYSIS_STATUS_FAILED = "failed";

// Shown where a detector produced no figure. Never 0%, which would read as an answer.
export const UNAVAILABLE = "N/A";
// Shown where there is no detector result to speak of.
export const ABSENT = "—";
// Shown where the analysis has not finished, so its decision is still owed.
export const PENDING = "Pending";

// The complete vocabulary of risk states this dashboard is entitled to present as a
// DeepGuard classification. It is an allowlist, not a default: a value is rendered as an
// official risk class because it appears here, never because it arrived from the database.
//
// There is deliberately no LOW. P7-T2 measured a low-risk threshold and ruleset v1 does not
// activate it, so no analysis carries that level — and a badge for it would advertise a
// reassurance the engine is not able to give.
export const SUPPORTED_RISK_LEVELS = ["HIGH", "MEDIUM", "UNKNOWN"] as const;

export type SupportedRiskLevel = (typeof SUPPORTED_RISK_LEVELS)[number];

// The supported levels in the words the dashboard says them in.
//
// Every word here is about *risk*, which is what this column reports: a deterministic
// classification of calibrated forensic evidence. None of them is a claim about whether
// the media is genuine, and none of them may become one.
//
// Keyed by the union rather than by `string`, so this table and the allowlist above cannot
// drift apart: adding a level to one without the other fails `tsc --noEmit`, and a lookup
// on an arbitrary database string does not type-check at all.
export const RISK_LABELS: Record<SupportedRiskLevel, string> = {
  HIGH: "High risk",
  MEDIUM: "Medium risk",
  UNKNOWN: "Unknown",
};

// Colour is supportive only: the label carries the meaning and stays legible without it.
export const RISK_STYLES: Record<SupportedRiskLevel, string> = {
  HIGH: "bg-rose-500/10 text-rose-700 dark:text-rose-300",
  MEDIUM: "bg-amber-500/10 text-amber-700 dark:text-amber-300",
  UNKNOWN: "bg-slate-500/10 text-slate-700 dark:text-slate-300",
};

// What a non-null level outside the allowlist is shown as. Deliberately not the stored
// string: rendering `LOW`, `CRITICAL`, `FAKE` or `REAL` in the risk column would let a value
// this build has no calibrated meaning for read as an official DeepGuard classification —
// and in the case of a `FAKE`/`REAL` row, as exactly the certainty semantics the product
// refuses to claim. The state reported is that the value is unsupported, which is the only
// thing that is actually known about it.
export const UNSUPPORTED = "Unsupported";

// Neutral styling for that state: the same muted treatment as any other non-answer on this
// page, carrying no band colour that would imply a severity was read out of the value.
export const RISK_UNSUPPORTED_STYLE = "bg-slate-500/10 text-slate-700 dark:text-slate-300";

/**
 * Whether a stored level is one this build may present as a DeepGuard risk class.
 *
 * Exact membership of the allowlist, which is what makes the guard total: every other
 * non-null string — a level from a later ruleset, a hand-edited row, `LOW` from a build that
 * activated it — lands in the unsupported state instead of borrowing a supported band's
 * label and colour. The predicate narrows to the union, so the label and style lookups after
 * it are the only place a level is indexed and TypeScript proves the key is in range.
 */
export function isSupportedRiskLevel(level: string): level is SupportedRiskLevel {
  return (SUPPORTED_RISK_LEVELS as readonly string[]).includes(level);
}

// Compile-time proof that the allowlist excludes the states this column must never present
// as official, checked by `tsc --noEmit` — the frontend's existing verification command.
// Activating LOW, or widening the union to `string`, breaks the build here rather than
// silently shipping a reassurance the engine cannot give.
export type Excluded<L extends string> = L extends SupportedRiskLevel ? never : L;
export type ExcludedRiskStates = Excluded<"LOW" | "CRITICAL" | "FAKE" | "REAL">;
// Each must survive `Excluded` unchanged, which holds only while none is assignable to
// `SupportedRiskLevel`. If any became supported it would collapse to `never` and this
// assignment would stop compiling.
export const _EXCLUDED_RISK_STATES: ExcludedRiskStates[] = ["LOW", "CRITICAL", "FAKE", "REAL"];
void _EXCLUDED_RISK_STATES;

// The three provenance outcomes that are not a C2PA state: the file was read and carries
// no credentials, the file names a manifest kept somewhere else, and the file could not be
// read. Kept apart on purpose — "we looked and found nothing", "provenance was claimed but
// is not in these bytes" and "we could not look" are different facts, and none of them
// means the media is fake.
// The two active-speaker outcomes that are not a timeline: the chain did not produce one,
// and it produced one that is empty. Kept apart on purpose — "we could not look" and "we
// looked and nobody was speaking" are different facts, and neither says anything about
// whether the media is genuine.
export const SPEAKER_UNAVAILABLE = "Unavailable";
export const NO_SPEAKING_FACES = "No speaking faces detected";
// Shown where NVIDIA saw a face speaking but matched it to no diarized voice.
export const UNMATCHED_VOICE = "no matched voice";

// The two audio-authenticity outcomes that are not a set of windows: the reading did not
// happen, and it happened and stored none. Kept apart for the same reason as the pair
// above, and phrased to describe the *evidence* rather than the media — nothing here has
// established that a file carries no audio, only that no windows were persisted for it.
export const AUDIO_UNAVAILABLE = "Unavailable";
export const NO_AUDIO_WINDOWS = "No audio evidence windows";

export const NO_PROVENANCE = "No provenance";
export const REMOTE_PROVENANCE = "Remote provenance (not fetched)";
export const EXTRACTION_FAILED = "Extraction failed";
// The listing, or why it could not be shown. `unauthenticated` is the API refusing the
// session rather than failing, and it is kept apart from every other failure because it is
// the one the page answers by sending the reader to sign in — telling somebody the list is
// "temporarily unavailable" when they are simply signed out would strand them on a page
// that will never fill in.
export type AnalysesResult =
  | { ok: true; analyses: AnalysisSummary[] }
  | { ok: false; unauthenticated: boolean; error: string };

/** A number the API may legitimately have left out, or `undefined` for anything else. */
export function parseOptionalNumber(value: unknown): number | null | undefined {
  if (value === null) {
    return null;
  }

  return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}

/** A string the API may legitimately have left out, or `undefined` for anything else. */
export function parseOptionalString(value: unknown): string | null | undefined {
  if (value === null) {
    return null;
  }

  return typeof value === "string" ? value : undefined;
}

/** A boolean the API may legitimately have left out, or `undefined` for anything else. */
export function parseOptionalBoolean(value: unknown): boolean | null | undefined {
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
export function parseSegments(payload: unknown): SegmentEvidence[] | undefined {
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
export function parseSignal(payload: unknown): SyntheticVideoSignal | null | undefined {
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
export function parseSpeakingSegments(payload: unknown): SpeakingSegment[] | undefined {
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
export function parseActiveSpeaker(payload: unknown): ActiveSpeakerSignal | null | undefined {
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
export function parseAudioWindows(payload: unknown): AudioWindow[] | undefined {
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
export function parseAudioAuthenticity(
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
export function parseProvenance(payload: unknown): ProvenanceSignal | null | undefined {
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
export function parseMedia(payload: unknown): MediaFacts | undefined {
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

export function parseAnalysis(payload: unknown): AnalysisSummary | null {
  if (typeof payload !== "object" || payload === null) {
    return null;
  }

  const {
    id,
    status,
    created_at,
    risk_level,
    risk_rules_version,
    risk_rule_id,
    risk_calibration_id,
    original_filename,
    declared_content_type,
    original_sha256,
    size_bytes,
    was_normalized,
    media,
    synthetic_video,
    provenance,
    active_speaker,
    audio_authenticity,
  } = payload as Record<string, unknown>;

  // Each parsed on its own three-way rule: a real value, a legitimate null, or `undefined`
  // for a payload that is neither. Null is preserved rather than defaulted — the whole
  // point of the column is that "no decision" is a state of its own.
  const parsedSha256 = parseOptionalString(original_sha256);
  const parsedSize = parseOptionalNumber(size_bytes);
  const riskLevel = parseOptionalString(risk_level);
  const riskRulesVersion = parseOptionalString(risk_rules_version);
  const riskRuleId = parseOptionalString(risk_rule_id);
  const riskCalibrationId = parseOptionalString(risk_calibration_id);

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
    parsedSha256 === undefined ||
    parsedSize === undefined ||
    typeof was_normalized !== "boolean" ||
    riskLevel === undefined ||
    riskRulesVersion === undefined ||
    riskRuleId === undefined ||
    riskCalibrationId === undefined ||
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
    risk_level: riskLevel,
    risk_rules_version: riskRulesVersion,
    risk_rule_id: riskRuleId,
    risk_calibration_id: riskCalibrationId,
    original_filename,
    declared_content_type,
    original_sha256: parsedSha256,
    size_bytes: parsedSize,
    was_normalized,
    media: mediaFacts,
    synthetic_video: signal,
    provenance: provenanceSignal,
    active_speaker: activeSpeaker,
    audio_authenticity: audioAuthenticity,
  };
}
/**
 * Who the browser's session authenticates, or null when it authenticates nobody.
 *
 * The one question this server asks the API about identity, and it asks the API rather than
 * reading the cookie because the cookie is opaque: it carries no account, no role and no
 * expiry, by design (see `app/web_auth.py`). A missing cookie, an expired session and a
 * revoked one all arrive here as the same null, which is all a caller needs — the answer to
 * every one of them is to sign in.
 *
 * A 401 and an unreachable API are deliberately the same null too. Both mean this render
 * cannot establish who the reader is, and a page that showed the dashboard chrome for an
 * identity it could not confirm would be guessing.
 */
export async function fetchSession(): Promise<SessionUser | null> {
  try {
    const response = await fetch(`${apiUrl()}/api/v1/auth/me`, {
      cache: "no-store",
      headers: { ...(await sessionHeaders()), ...(await requestIdHeaders()) },
      signal: AbortSignal.timeout(SESSION_TIMEOUT_MS),
    });

    if (!response.ok) {
      return null;
    }

    return parseSessionUser(await response.json().catch(() => null));
  } catch {
    return null;
  }
}

export async function fetchAnalyses(): Promise<AnalysesResult> {
  try {
    const response = await fetch(`${apiUrl()}/api/v1/analyses`, {
      cache: "no-store",
      headers: { ...(await sessionHeaders()), ...(await requestIdHeaders()) },
      signal: AbortSignal.timeout(ANALYSES_TIMEOUT_MS),
    });

    // The session the API would not accept. Reported as its own state rather than as a
    // failure, because the page can act on it — everything else it can only report.
    if (response.status === 401) {
      return {
        ok: false,
        unauthenticated: true,
        error: "Sign in to see your analyses.",
      };
    }

    // The API's own failure detail is server-side context, not something to surface
    // here, so an unsuccessful status becomes one generic message.
    if (!response.ok) {
      return {
        ok: false,
        unauthenticated: false,
        error: "The analysis list is temporarily unavailable.",
      };
    }

    const payload = await response.json().catch(() => null);
    if (!Array.isArray(payload)) {
      return {
        ok: false,
        unauthenticated: false,
        error: "The analysis list could not be read.",
      };
    }

    const analyses = payload.map(parseAnalysis);
    if (analyses.some((analysis) => analysis === null)) {
      return {
        ok: false,
        unauthenticated: false,
        error: "The analysis list could not be read.",
      };
    }

    return { ok: true, analyses: analyses as AnalysisSummary[] };
  } catch {
    // Timeouts and connection errors are the same thing to a reader of this page: the
    // list is not available right now. The underlying message could leak internal
    // hostnames, so it is not shown.
    return {
      ok: false,
      unauthenticated: false,
      error: "The analysis list is temporarily unavailable.",
    };
  }
}

/**
 * One analysis, or why it could not be shown.
 *
 * `missing` is a 404 and not an error, and since R1-T2 it covers two things the API refuses
 * to tell apart: no analysis has this id, and this session may not see the one that does.
 * That is the API's decision and this type carries it as it stands — a report that
 * distinguished the two would republish the fact the 404 exists to withhold.
 *
 * `unauthenticated` is separate again: the session was not accepted at all, which the page
 * answers by sending the reader to sign in rather than by reporting a missing record.
 */
export type AnalysisResult =
  | { ok: true; analysis: AnalysisSummary }
  | { ok: false; missing: boolean; unauthenticated: boolean; error: string };

/**
 * Read one analysis from the API by id, for the report route.
 *
 * The report asks for its own analysis rather than filtering the listing: a report built
 * from the listing would quietly stop working once its analysis fell out of the most recent
 * fifty, and would fetch every other analysis to render one.
 *
 * A 404 is reported as `missing` rather than as a failure. "No analysis has this id" is a
 * fact worth stating plainly on the page; "the API is unreachable" is not the same thing and
 * must not be shown as though the analysis does not exist.
 */
export async function fetchAnalysis(id: string): Promise<AnalysisResult> {
  try {
    const response = await fetch(`${apiUrl()}/api/v1/analyses/${encodeURIComponent(id)}`, {
      cache: "no-store",
      headers: { ...(await sessionHeaders()), ...(await requestIdHeaders()) },
      signal: AbortSignal.timeout(ANALYSES_TIMEOUT_MS),
    });

    if (response.status === 401) {
      return {
        ok: false,
        missing: false,
        unauthenticated: true,
        error: "Sign in to see this report.",
      };
    }

    if (response.status === 404) {
      return {
        ok: false,
        missing: true,
        unauthenticated: false,
        error: "No analysis was found with this id.",
      };
    }

    if (!response.ok) {
      return {
        ok: false,
        missing: false,
        unauthenticated: false,
        error: "This analysis is temporarily unavailable.",
      };
    }

    const analysis = parseAnalysis(await response.json().catch(() => null));
    if (analysis === null) {
      return {
        ok: false,
        missing: false,
        unauthenticated: false,
        error: "This analysis could not be read.",
      };
    }

    return { ok: true, analysis };
  } catch {
    // A timeout or a connection error is not evidence that the analysis is absent, so the
    // message says the record is unavailable rather than that it does not exist. The
    // underlying error could name internal hosts and is not surfaced.
    return {
      ok: false,
      missing: false,
      unauthenticated: false,
      error: "This analysis is temporarily unavailable.",
    };
  }
}
