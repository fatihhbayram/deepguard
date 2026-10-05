// Analysis payloads for rendering the report in node (R16-T3), in the API's own shape.
//
// Synthetic, and deliberately distinctive: every hash, id, version and figure is a value that
// appears nowhere in the report's own wording, so finding it in a render — and counting it — says
// something about the evidence and nothing about the copy around it. Each fixture is one branch
// of the report: a v5 decision mid-enrichment, each legacy ruleset, no decision at all, and a
// payload full of values this build has no name for.

const SHA = "5f0c3a9e1b7d2c4e6a8f0b1d3c5e7a9b2d4f6a8c0e1b3d5f7a9c2e4b6d8f0a1c";
const CALIBRATION = "c41b7a03e9d2f58c6b1a4e7d0f3c9b2a8e5d1f7c4b0a6e3d9f2c8b5a1e7d4f0c";

const SVD_VERSION = "nvcf-fn-6d1e2f3a-91b4-4c7d-8e2f-a0b1c2d3e4f5";
const FACE_VERSION = "facetorch/efficientnet-b7@0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6a7b8c9d";
const LIP_VERSION = "lipforensics@ab12cd34ef56+9e8d7c6b5a4f3e2d1c0b9a8f7e6d5c4b3a2f1e0d9c";
const SPEAKER_VERSION = "nvcf-fn-7a8b9c0d-1e2f-4a5b-9c6d-e7f8a9b0c1d2";
const AUDIO_VERSION = "aasist@5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c2d3e4f";

function contribution(signal, provider, provider_version, score, threshold, condition, role, decisional, unavailable_reason = null) {
  return { signal, provider, provider_version, score, threshold, condition, unavailable_reason, role, decisional };
}

function base(overrides) {
  return {
    id: "8c2e4a6f-1b3d-4e5f-9a7c-0d2e4f6a8b1c",
    status: "completed",
    created_at: "2026-09-21T10:28:49.652925Z",
    decision_state: "DECIDED",
    aggregate_enrichment_state: "ENRICHMENT_COMPLETE",
    per_component_state: [
      { provider: "aasist", signal_type: "audio_authenticity", state: "completed" },
      { provider: "effort", signal_type: "face_forgery", state: "completed" },
      { provider: "lipforensics", signal_type: "lip_forensics", state: "completed" },
      { provider: "nvidia", signal_type: "active_speaker", state: "completed" },
    ],
    risk_level: null,
    risk_rules_version: null,
    risk_rule_id: null,
    risk_calibration_id: null,
    risk_trace: null,
    original_filename: "fixture_clip_r16t3.mp4",
    declared_content_type: "video/mp4",
    size_bytes: 4210384,
    original_sha256: SHA,
    media_sha256s: [SHA],
    was_normalized: false,
    was_assembled: false,
    acquisition_method: "upload",
    source_host: null,
    media: {
      format_name: "mov,mp4,m4a,3gp,3g2,mj2",
      codec_name: "h264",
      original_width: 1918,
      original_height: 1078,
      display_rotation: 0,
      analyzed_width: 1918,
      analyzed_height: 1078,
      duration: 10.25,
      frame_rate: 23.976,
      pix_fmt: "yuv420p10le",
      constant_frame_rate: true,
    },
    synthetic_video: {
      provider: "nvidia",
      signal_type: "synthetic_video",
      status: "SUCCESS",
      score: 0.9826627969741821,
      provider_version: SVD_VERSION,
      logit: 4.037409782409668,
      total_clips: 241,
      segments: [
        { clip_index: 8611, logit: 6.177785873413086 },
        { clip_index: 9023, logit: 6.039676666259766 },
      ],
    },
    provenance: {
      provider: "c2pa",
      signal_type: "provenance",
      status: "SUCCESS",
      provider_version: "0.90.14-r16t3",
      manifest_exists: true,
      validation_state: "Valid",
      claim_generator: "Fixture C2PA Generator 7.3",
      signature_issuer: "Fixture Signing Authority Ltd",
      remote_manifest_url: null,
      provenance_status: "PROVENANCE_PRESENT",
      provenance_availability: "AVAILABLE",
    },
    active_speaker: {
      provider: "nvidia",
      signal_type: "active_speaker",
      status: "SUCCESS",
      provider_version: SPEAKER_VERSION,
      total_speaking_segments: 2,
      segments_truncated: false,
      segments: [
        { start_time: 0.0, end_time: 5.458333333333333, face_id: 4242, speaker_label: "SPEAKER_17" },
        { start_time: 5.5, end_time: 5.833333333333333, face_id: 4243, speaker_label: null },
      ],
    },
    audio_authenticity: {
      provider: "aasist",
      signal_type: "audio_authenticity",
      status: "SUCCESS",
      provider_version: AUDIO_VERSION,
      total_audio_windows: 3,
      persisted_audio_windows: 3,
      windows_truncated: false,
      windows: [
        { clip_index: 0, start_time: 0.0, end_time: 4.0375, logit: 0.5217356085777283, bona_fide_logit: -1.087714433670044 },
        { clip_index: 1, start_time: 4.0375, end_time: 8.075, logit: 1.8969013690948486, bona_fide_logit: -2.393179416656494 },
      ],
    },
    face_manipulation: {
      provider: "efficientnet-b7",
      signal_type: "face_manipulation",
      status: "SUCCESS",
      score: 0.020748294579486053,
      provider_version: FACE_VERSION,
      frames_requested: 8,
      frames_decoded: 8,
      frames_scored: 6,
    },
    lip_forensics: {
      provider: "lipforensics",
      signal_type: "lip_forensics",
      status: "SUCCESS",
      score: 8.514249429936172e-7,
      provider_version: LIP_VERSION,
      windows_requested: 4,
      windows_read: 4,
      windows_scored: 3,
    },
    ...overrides,
  };
}

const SVD_REACHED = contribution("synthetic_video", "nvidia", SVD_VERSION, 0.9826627969741821, 0.9550971388816833, "threshold_reached", "decisive", true);
const FACE_NOT_REACHED = contribution("face_manipulation", "efficientnet-b7", FACE_VERSION, 0.020748294579486053, 0.9867589175701141, "threshold_not_reached", "considered", true);
const LIP_EVIDENCE = contribution("lip_forensics", "lipforensics", LIP_VERSION, 8.514249429936172e-7, 0.22962537594139576, "threshold_not_reached", "considered", false);

function trace(risk_level, rules_version, rule_id, contributions, decision_coverage = null, interpreted = true) {
  return {
    risk_level,
    rule_id,
    rules_version,
    calibration_id: CALIBRATION,
    rule_summary:
      "Fixture rule summary R16T3: the API's own sentence about rule " + rule_id + ", carried verbatim.",
    contributions,
    interpreted,
    decision_coverage,
  };
}

function decided(level, rulesVersion, ruleId, contributions, coverage = null) {
  return {
    risk_level: level,
    risk_rules_version: rulesVersion,
    risk_rule_id: ruleId,
    risk_calibration_id: CALIBRATION,
    risk_trace: trace(level, rulesVersion, ruleId, contributions, coverage),
  };
}

export const FIXTURES = {
  // A v5 decision whose supplementary evidence is still running, on a normalized URL acquisition.
  v5Processing: base({
    ...decided("MANIPULATION_DETECTED", "r9-v5.0.0", "R9-101", [SVD_REACHED, FACE_NOT_REACHED, LIP_EVIDENCE], {
      usable: 2,
      total: 2,
      is_complete: true,
      status: "complete",
    }),
    aggregate_enrichment_state: "ENRICHMENT_PROCESSING",
    per_component_state: [
      { provider: "aasist", signal_type: "audio_authenticity", state: "completed" },
      { provider: "effort", signal_type: "face_forgery", state: "completed" },
      { provider: "lipforensics", signal_type: "lip_forensics", state: "processing" },
      { provider: "nvidia", signal_type: "active_speaker", state: "queued" },
    ],
    lip_forensics: null,
    active_speaker: null,
    was_normalized: true,
    acquisition_method: "url",
    source_host: "media.fixture-host.example",
    media: { ...base({}).media, display_rotation: 90, analyzed_width: 1078, analyzed_height: 1918 },
  }),
  // v5, inconclusive, on an assembled acquisition whose provenance reading found no manifest.
  v5Inconclusive: base({
    ...decided(
      "INCONCLUSIVE",
      "r9-v5.0.0",
      "R9-301",
      [
        contribution("synthetic_video", "nvidia", SVD_VERSION, null, null, "unavailable", "considered", true, "detector_did_not_report"),
        FACE_NOT_REACHED,
        LIP_EVIDENCE,
      ],
      { usable: 1, total: 2, is_complete: false, status: "partial" },
    ),
    aggregate_enrichment_state: "ENRICHMENT_PARTIAL",
    was_assembled: true,
    acquisition_method: "url",
    source_host: "stream.fixture-host.example",
    synthetic_video: { ...base({}).synthetic_video, status: "TIMEOUT", score: null, logit: null, total_clips: null, segments: [] },
    provenance: {
      ...base({}).provenance,
      manifest_exists: false,
      validation_state: null,
      claim_generator: null,
      signature_issuer: null,
      provenance_status: "UNVERIFIED",
    },
    audio_authenticity: { ...base({}).audio_authenticity, windows: [], persisted_audio_windows: 0, total_audio_windows: 0 },
  }),
  // A quick scan, decided under v5 with nothing reaching its threshold.
  v5QuickScan: base({
    ...decided("NO_CALIBRATED_MANIPULATION_SIGNAL", "r9-v5.0.0", "R9-200", [
      contribution("synthetic_video", "nvidia", SVD_VERSION, 0.1234567891234, 0.9550971388816833, "threshold_not_reached", "considered", true),
      FACE_NOT_REACHED,
    ], { usable: 2, total: 2, is_complete: true, status: "complete" }),
    aggregate_enrichment_state: "ENRICHMENT_NOT_REQUESTED",
    per_component_state: [],
    lip_forensics: null,
    active_speaker: null,
    audio_authenticity: null,
    provenance: { ...base({}).provenance, status: "FAILED", manifest_exists: null, provenance_status: "UNVERIFIED", provenance_availability: "UNAVAILABLE" },
  }),
  v4High: base({
    ...decided("HIGH", "r7-v4.0.0", "R100", [SVD_REACHED, FACE_NOT_REACHED, LIP_EVIDENCE]),
    aggregate_enrichment_state: "ENRICHMENT_FAILED",
    per_component_state: [
      { provider: "aasist", signal_type: "audio_authenticity", state: "failed" },
      { provider: "lipforensics", signal_type: "lip_forensics", state: "abstained" },
    ],
  }),
  v3Partial: base({
    ...decided("MEDIUM", "r5-v3.0.0", "R201", [SVD_REACHED, FACE_NOT_REACHED, { ...LIP_EVIDENCE, decisional: true }]),
    aggregate_enrichment_state: "LEGACY_SINGLE_STAGE",
    face_manipulation: { ...base({}).face_manipulation, status: "FAILED", score: null, frames_scored: 0 },
  }),
  v2Below: base({
    ...decided("UNKNOWN", "r4-v2.0.0", "R200", [SVD_REACHED, FACE_NOT_REACHED]),
    aggregate_enrichment_state: null,
  }),
  v1High: base({
    ...decided("HIGH", "p7-v1.0.0", "R100", [SVD_REACHED]),
    aggregate_enrichment_state: null,
    acquisition_method: null,
    was_assembled: true,
  }),
  noDecision: base({ aggregate_enrichment_state: "ENRICHMENT_NOT_APPLICABLE", decision_state: "DECISION_PENDING" }),
  // Every value here is one this build has no name for, and each must be drawn exactly as sent.
  unknownValues: base({
    risk_level: "MYSTERY_LEVEL",
    risk_rules_version: "x9-v9.9.9",
    risk_rule_id: "RX-404",
    risk_calibration_id: CALIBRATION,
    risk_trace: {
      ...trace(
        "MYSTERY_LEVEL",
        "x9-v9.9.9",
        "RX-404",
        [contribution("mystery_signal", "mystery-provider", "mystery-version-1", 0.31337, 0.73313, "mystery_condition", "mystery_role", null, "mystery_reason")],
        { usable: 1, total: 3, is_complete: false, status: "mystery_coverage" },
        false,
      ),
    },
    decision_state: "MYSTERY_DECISION_STATE",
    aggregate_enrichment_state: "ENRICHMENT_MYSTERY",
    per_component_state: [{ provider: "mystery-provider", signal_type: "mystery_component", state: "mystery_state" }],
    synthetic_video: { ...base({}).synthetic_video, status: "MYSTERY_STATUS" },
    provenance: { ...base({}).provenance, provenance_status: "MYSTERY_PROVENANCE" },
  }),
};

// The values the report prints from the record, per fixture: what must be byte-for-byte the same
// in every language. Gathered from the payload rather than listed by hand, so a fixture edit
// cannot leave one out.
export function canonicalValues(payload) {
  const values = new Set();
  const take = (value) => {
    if (typeof value === "string" && value.length > 0) values.add(value);
    if (typeof value === "number") values.add(String(value));
  };

  for (const key of [
    "id", "status", "created_at", "original_filename", "declared_content_type", "original_sha256",
    "risk_rule_id", "risk_rules_version", "risk_calibration_id", "decision_state",
    "aggregate_enrichment_state", "source_host", "size_bytes",
  ]) {
    take(payload[key]);
  }
  for (const key of ["format_name", "codec_name", "pix_fmt"]) take(payload.media[key]);

  for (const component of payload.per_component_state) take(component.state);

  const trace = payload.risk_trace;
  if (trace) {
    take(trace.rule_summary);
    if (trace.decision_coverage) take(trace.decision_coverage.status);
    for (const entry of trace.contributions) {
      for (const key of ["provider", "provider_version", "score", "threshold", "condition"]) take(entry[key]);
    }
  }

  for (const key of ["synthetic_video", "face_manipulation", "lip_forensics", "active_speaker", "audio_authenticity", "provenance"]) {
    const signal = payload[key];
    if (!signal) continue;
    for (const field of ["provider", "status", "provider_version", "score", "logit", "total_clips"]) take(signal[field]);
  }
  for (const segment of payload.synthetic_video?.segments ?? []) {
    take(segment.clip_index);
    take(segment.logit);
  }
  for (const segment of payload.active_speaker?.segments ?? []) {
    take(segment.face_id);
    take(segment.speaker_label);
  }
  for (const window of payload.audio_authenticity?.windows ?? []) {
    take(window.logit.toFixed(4));
    take(window.bona_fide_logit.toFixed(4));
  }
  const provenance = payload.provenance;
  if (provenance) {
    for (const field of ["validation_state", "claim_generator", "signature_issuer", "provenance_status", "provenance_availability"]) {
      take(provenance[field]);
    }
  }
  return values;
}
