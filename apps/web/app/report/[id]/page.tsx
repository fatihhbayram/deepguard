import Link from "next/link";
import { notFound, redirect } from "next/navigation";

import { LOGIN_PATH } from "../../session";
import {
  ActiveSpeakerSignal,
  AnalysisSummary,
  AudioAuthenticitySignal,
  FaceManipulationSignal,
  LipForensicsSignal,
  MediaFacts,
  ProvenanceSignal,
  FACE_T_HIGH_DISPLAY,
  RISK_LABELS,
  RULES_VERSION_V2,
  RiskRationale,
  SyntheticVideoSignal,
  UNSUPPORTED,
  fetchAnalysis,
  isSupportedRiskLevel,
  riskRationale,
} from "../../analysis";

import { PrintButton } from "./print-button";

/**
 * A printable forensic evidence report for one analysis.
 *
 * This page is a **rendering of persisted DeepGuard evidence** and nothing more. It is not a
 * certificate, not a proof of authenticity, and not a verdict on whether the media is genuine
 * — those words are absent from this file deliberately, because a document that looks
 * official is read as one. Nothing here is cryptographically signed: the SHA-256 shown is the
 * hash of the *analysed media*, never of this report, and the page says so where it is shown.
 *
 * Nothing is recomputed. The risk classification is displayed exactly as the worker committed
 * it under a named ruleset; no detector score is compared against a threshold here, and no
 * signal is summarised into another. A report that re-derived its own conclusion could
 * contradict the record it exists to document.
 *
 * A Server Component. The only client-side code on the page is the print button, which is
 * isolated in its own module so the report itself stays server-rendered and prints correctly
 * with JavaScript disabled.
 */

// What the risk column says when no decision was ever taken. Distinct from `UNKNOWN`, which
// is a decision, and phrased as a statement about the record rather than about the media.
const NO_DECISION = "No risk decision";

// Shown where a value the report would otherwise print is simply not in the record.
const ABSENT = "—";

/** Nothing on this page renders a stored level it has no calibrated meaning for. */
function riskLabel(level: string | null): string {
  if (level === null) {
    return NO_DECISION;
  }

  return isSupportedRiskLevel(level) ? RISK_LABELS[level] : UNSUPPORTED;
}

/**
 * A muted band for every state.
 *
 * Colour is supplemental here and carries no meaning of its own: the report is printed, often
 * in greyscale, so every state has to be legible from its words alone. HIGH is given a tinted
 * border rather than a filled badge for the same reason — it must not read as a stamp.
 */
function riskAccent(level: string | null): string {
  if (level !== null && isSupportedRiskLevel(level) && level === "HIGH") {
    return "border-rose-400 dark:border-rose-500";
  }

  return "border-black/20 dark:border-white/25";
}

function Field({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="break-inside-avoid">
      <dt className="text-xs uppercase tracking-wide opacity-60">{label}</dt>
      <dd className="mt-0.5 font-mono text-xs break-all">{value}</dd>
    </div>
  );
}

function Section({
  title,
  subtitle,
  children,
}: {
  title: string;
  subtitle?: string;
  children: React.ReactNode;
}) {
  return (
    <section className="mt-6 break-inside-avoid rounded border border-black/15 p-4 dark:border-white/20">
      <h2 className="text-base font-semibold">{title}</h2>
      {subtitle && <p className="mt-0.5 text-xs opacity-70">{subtitle}</p>}
      <div className="mt-3">{children}</div>
    </section>
  );
}

/**
 * The state a signal is in, said in the provider's own word.
 *
 * `SUCCESS`, `FAILED` and `TIMEOUT` are three different forensic facts, and an analysis that
 * carries no such signal at all is a fourth. They are never collapsed: "the detector ran and
 * failed" and "this detector never ran" mean different things about what is known.
 */
function SignalState({ status }: { status: string }) {
  return <span className="font-mono text-xs">{status}</span>;
}

function NoSignal({ what }: { what: string }) {
  return (
    <p className="text-xs opacity-70">
      No {what} signal is stored for this analysis. That is not a failed reading — nothing
      recorded one, so there is no evidence from this source either way.
    </p>
  );
}

function frameRateText(rate: number): string {
  return Number.isInteger(rate) ? `${rate}` : rate.toFixed(2);
}

function MediaSection({
  analysis,
  media,
}: {
  analysis: AnalysisSummary;
  media: MediaFacts;
}) {
  return (
    <Section
      title="Analysed media"
      subtitle="What ffprobe established about the forensic original, as the database kept it."
    >
      <dl className="grid grid-cols-1 gap-3 sm:grid-cols-2">
        <Field label="Original filename" value={analysis.original_filename ?? ABSENT} />
        <Field label="Declared content type" value={analysis.declared_content_type} />
        <Field label="Container (ffprobe)" value={media.format_name} />
        <Field label="Video codec" value={media.codec_name} />
        <Field label="Resolution" value={`${media.width} × ${media.height}`} />
        <Field label="Frame rate" value={`${frameRateText(media.frame_rate)} fps`} />
        <Field label="Duration" value={`${media.duration.toFixed(2)} s`} />
        <Field label="Pixel format" value={media.pix_fmt ?? ABSENT} />
        <Field
          label="Constant frame rate"
          value={media.constant_frame_rate ? "yes" : "no"}
        />
        <Field label="Normalized for detection" value={analysis.was_normalized ? "yes" : "no"} />
      </dl>
      <div className="mt-4 break-inside-avoid">
        <dt className="text-xs uppercase tracking-wide opacity-60">
          SHA-256 of the analysed media
        </dt>
        {/* Printed in full, never abbreviated: an abbreviated hash cannot be checked, and
            checking it against the source file is the whole reason it is here. */}
        <dd className="mt-0.5 font-mono text-xs break-all">
          {analysis.original_sha256 ?? ABSENT}
        </dd>
        <p className="mt-1 text-xs opacity-70">
          This is the hash of the media that was analysed. It is not a hash or a signature of
          this report.
        </p>
      </div>
    </Section>
  );
}

/**
 * How each detector contributed, in the words the rationale table gives.
 *
 * The label is fixed per role and the detail comes from the stored rule, so a reader can see
 * at a glance which source produced the level and which merely did not object. That
 * distinction is the whole point of a multi-source ruleset: a detector that scored below its
 * threshold is not a second opinion agreeing that the media is fine, and this row must never
 * let it read as one.
 */
const CONTRIBUTION_LABELS: Record<RiskRationale["syntheticVideo"]["role"], string> = {
  decided: "Reached its threshold — this is what decided",
  below: "Below its threshold — did not contribute",
  unreadable: "Answered, but unreadable",
  unavailable: "No usable reading",
  unread: "Not read by this ruleset",
  unclear: "Not named by the rule — see this detector's own panel below",
};

/**
 * What this detector's own signal row says about itself, for the one rule that cannot say.
 *
 * `R201` means exactly one of the two detectors produced a usable reading, and the rule id
 * alone does not record which — so the rationale table marks both `unclear` rather than
 * guessing. Reporting each detector's *stored status* closes that gap without re-deriving
 * anything: presence and status are persisted facts about the signal row, read straight out
 * of the record, and no score is compared against any threshold here. A successful reading is
 * still only reported as a reading — whether it was the one a rule could be applied to also
 * depends on the build that produced it, which its own section below states in full.
 */
function storedStatusNote(status: string | null | undefined): string {
  if (status === undefined || status === null) {
    return "No signal from this detector is recorded for this analysis.";
  }

  if (status !== "SUCCESS") {
    return `This detector returned no reading — its signal is stored as ${status}.`;
  }

  return "This detector returned a reading; its section below gives the figures and the build that produced them.";
}

function Contribution({
  detector,
  contribution,
  status,
}: {
  detector: string;
  contribution: RiskRationale["syntheticVideo"];
  status?: string | null;
}) {
  return (
    <div className="border-t border-black/10 pt-2 dark:border-white/15">
      <p className="text-sm font-medium">{detector}</p>
      <p className="mt-0.5 text-xs font-medium opacity-90">
        {CONTRIBUTION_LABELS[contribution.role]}
      </p>
      <p className="mt-1 text-xs opacity-70">
        {contribution.role === "unclear"
          ? storedStatusNote(status)
          : contribution.detail}
      </p>
    </div>
  );
}

/**
 * The persisted product-level risk classification, with the trace that makes it explainable.
 *
 * Read from the analysis row exactly as stored. Nothing on this page looks at the detector
 * scores below to decide what to print here: the rationale is derived from `risk_rule_id` and
 * `risk_rules_version` — the two strings the worker committed alongside the level — so the
 * explanation is always the rule that actually fired rather than a fresh conclusion this page
 * reached about the same evidence.
 *
 * A row whose ruleset this build does not recognise gets the trace with no explanation beside
 * it, which is the honest rendering: a rule id only means something inside the version that
 * defined it, and borrowing another version's wording would describe a decision nobody took.
 */
function RiskSection({ analysis }: { analysis: AnalysisSummary }) {
  const level = analysis.risk_level;
  const rationale = riskRationale(
    analysis.risk_rules_version,
    analysis.risk_rule_id,
  );

  return (
    <section
      className={`mt-6 break-inside-avoid rounded border-2 p-4 ${riskAccent(level)}`}
    >
      <h2 className="text-base font-semibold">InspectRoot risk classification</h2>

      <p className="mt-2 text-2xl font-semibold">{riskLabel(level)}</p>

      {level === null ? (
        <p className="mt-1 text-xs opacity-70">
          No risk decision is stored for this analysis. That is not the same as{" "}
          <span className="font-mono">Unknown</span>: nothing classified this analysis, so
          there is no decision to report — it was analysed before the risk engine existed, or
          it has not finished.
        </p>
      ) : !isSupportedRiskLevel(level) ? (
        <p className="mt-1 text-xs opacity-70">
          The stored risk state{" "}
          <span className="font-mono">{level}</span> is not a risk class this build
          classifies under, so it is reported as unsupported rather than presented as an
          InspectRoot classification.
        </p>
      ) : level === "UNKNOWN" ? (
        <p className="mt-1 text-xs opacity-70">
          The risk engine ran and a rule fired. Its conclusion is that the evidence does not
          support a classification — an answer, not a missing one.
        </p>
      ) : null}

      {rationale !== null && (
        <div className="mt-4">
          <h3 className="text-sm font-semibold">Why this classification</h3>
          <p className="mt-1 text-sm">{rationale.summary}</p>

          <h4 className="mt-3 text-xs font-semibold uppercase tracking-wide opacity-70">
            How each detector contributed
          </h4>
          <div className="mt-2 space-y-2">
            <Contribution
              detector="NVIDIA synthetic-video detector"
              contribution={rationale.syntheticVideo}
              status={analysis.synthetic_video?.status ?? null}
            />
            <Contribution
              detector="EfficientNet-B7 face-manipulation classifier"
              contribution={rationale.faceManipulation}
              status={analysis.face_manipulation?.status ?? null}
            />
          </div>
          <p className="mt-2 text-xs opacity-70">
            The mouth-dynamics model is not listed here because no ruleset reads it. Its score is
            recorded below as independent evidence and took no part in this classification.
          </p>

          <h4 className="mt-3 text-xs font-semibold uppercase tracking-wide opacity-70">
            What this covers
          </h4>
          <p className="mt-1 text-xs opacity-80">{rationale.coverage}</p>
        </div>
      )}

      <dl className="mt-4 grid grid-cols-1 gap-3 sm:grid-cols-3">
        <Field label="Rule fired" value={analysis.risk_rule_id ?? ABSENT} />
        <Field label="Ruleset version" value={analysis.risk_rules_version ?? ABSENT} />
        <Field label="Calibration ID" value={analysis.risk_calibration_id ?? ABSENT} />
      </dl>

      {analysis.risk_rule_id !== null && rationale === null && (
        <p className="mt-2 text-xs opacity-70">
          This build has no description for rule{" "}
          <span className="font-mono">{analysis.risk_rule_id}</span> under ruleset{" "}
          <span className="font-mono">{analysis.risk_rules_version ?? ABSENT}</span>, so the
          trace above is shown without one. The decision itself is reproduced exactly as it
          was stored.
        </p>
      )}

      <p className="mt-4 text-xs opacity-80">
        Risk is a deterministic InspectRoot classification based on calibrated forensic
        evidence. It is not a Fake/Real determination. Each detector was compared only against
        the threshold measured for it; the scores were never averaged, weighted or combined
        into a single number. This classification was recorded when the analysis ran and is
        reproduced here unchanged; it is not recalculated by this report.
      </p>
    </section>
  );
}

/**
 * The scope limit, stated where it cannot be missed.
 *
 * Not a footnote and not a tooltip. The measured detection rates behind this wording are the
 * single most important thing a reader of this report needs to know, and burying them would
 * let the document imply a coverage it does not have.
 */
function ScopeDisclosure({ analysis }: { analysis: AnalysisSummary }) {
  // Scope is a property of the ruleset the decision was taken under, not of this build. A
  // decision taken under p7-v1.0.0 really did read one detector and really was not validated
  // for face swaps, and saying otherwise on an old report would overstate what was checked.
  const multiSource = analysis.risk_rules_version === RULES_VERSION_V2;

  return (
    <section className="mt-4 break-inside-avoid rounded border-2 border-amber-500/60 p-4">
      <h2 className="text-sm font-semibold uppercase tracking-wide">
        Scope of this risk model
      </h2>
      {multiSource ? (
        <>
          <p className="mt-2 text-sm font-medium">
            This risk model is validated for generated video and for face swaps, by two
            separate detectors with separate thresholds. Absence of HIGH risk does not mean
            the media is genuine.
          </p>
          <p className="mt-2 text-xs opacity-80">
            Both thresholds were set to almost never flag legitimate footage: neither detector
            flagged any of the 54 genuine clips in the calibration corpus. That choice is paid
            for in detection rate. At these operating points the synthetic-video detector
            flagged 54.6% of generated video and the face classifier flagged 44% of face
            swaps, so a great deal of manipulated media is correctly not flagged as HIGH.
          </p>
          <p className="mt-2 text-xs opacity-80">
            The two detectors cover different things and are read independently. Neither one
            scoring low is evidence against the other: in the calibration study the two never
            agreed on a single clip, and each was blind to the manipulation family the other
            was calibrated for.
          </p>
        </>
      ) : (
        <p className="mt-2 text-sm font-medium">
          This decision was taken under a single-detector ruleset that is validated for
          generated video and is not validated for face-swap detection. Absence of HIGH risk
          does not rule out face manipulation.
        </p>
      )}
      <p className="mt-2 text-xs opacity-80">
        Risk is a deterministic InspectRoot classification based on calibrated forensic
        evidence. It is not a Fake/Real determination.
      </p>
    </section>
  );
}

function SyntheticVideoSection({ signal }: { signal: SyntheticVideoSignal | null }) {
  return (
    <Section
      title="NVIDIA synthetic-video detector"
      subtitle="Direct-risk evidence. The figures below are NVIDIA's own output on NVIDIA's own scale."
    >
      {signal === null ? (
        <NoSignal what="synthetic-video" />
      ) : (
        <>
          <dl className="grid grid-cols-1 gap-3 sm:grid-cols-2">
            <Field label="Provider" value={signal.provider} />
            <Field label="Signal type" value={signal.signal_type} />
            <Field label="Status" value={<SignalState status={signal.status} />} />
            <Field label="Provider version (NVCF function ID)" value={signal.provider_version ?? ABSENT} />
            <Field
              label="NVIDIA synthetic probability"
              value={signal.score === null ? ABSENT : signal.score.toString()}
            />
            <Field
              label="NVIDIA aggregate logit"
              value={signal.logit === null ? ABSENT : signal.logit.toString()}
            />
            <Field
              label="Clips aggregated by NVIDIA"
              value={signal.total_clips === null ? ABSENT : signal.total_clips.toString()}
            />
          </dl>

          <p className="mt-3 text-xs opacity-70">
            The probability is the provider&apos;s score for its own detector, shown as
            returned. It is NVIDIA evidence, not an InspectRoot confidence, and it is not the
            risk classification above.
          </p>

          <h3 className="mt-4 text-sm font-medium">
            Persisted strongest clips{" "}
            <span className="font-normal opacity-60">(highest logit first)</span>
          </h3>
          {signal.segments.length === 0 ? (
            <p className="mt-1 text-xs opacity-70">
              No clip evidence is stored for this signal.
            </p>
          ) : (
            <table className="mt-2 w-full table-fixed text-left text-xs">
              <thead>
                <tr className="border-b border-black/15 dark:border-white/20">
                  <th className="py-1 font-medium">Frame index</th>
                  <th className="py-1 font-medium">Raw logit</th>
                </tr>
              </thead>
              <tbody>
                {signal.segments.map((segment) => (
                  <tr key={segment.clip_index} className="border-b border-black/5 last:border-b-0 dark:border-white/10">
                    <td className="py-1 font-mono">{segment.clip_index}</td>
                    <td className="py-1 font-mono">{segment.logit}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
          <p className="mt-2 text-xs opacity-70">
            The frame index is NVIDIA&apos;s own index for the clip&apos;s middle frame. These
            are the strongest clips the provider reported, not a claim that manipulation
            occurs at those frames.
          </p>
        </>
      )}
    </Section>
  );
}

function ProvenanceSection({ signal }: { signal: ProvenanceSignal | null }) {
  return (
    <Section
      title="C2PA provenance"
      subtitle="What the file itself claims about its origin, read from the forensic original."
    >
      {signal === null ? (
        <NoSignal what="provenance" />
      ) : (
        <>
          <dl className="grid grid-cols-1 gap-3 sm:grid-cols-2">
            <Field label="Provider" value={signal.provider} />
            <Field label="Status" value={<SignalState status={signal.status} />} />
            <Field label="C2PA SDK version" value={signal.provider_version ?? ABSENT} />
            <Field
              label="Manifest present in the file"
              value={
                signal.manifest_exists === null
                  ? "unknown — the reading failed"
                  : signal.manifest_exists
                    ? "yes"
                    : "no"
              }
            />
            <Field label="Validation state" value={signal.validation_state ?? ABSENT} />
            <Field label="Claim generator" value={signal.claim_generator ?? ABSENT} />
            <Field label="Signature issuer" value={signal.signature_issuer ?? ABSENT} />
            <Field
              label="Remote manifest URL"
              value={signal.remote_manifest_url ?? ABSENT}
            />
          </dl>
          <p className="mt-3 text-xs opacity-70">
            Provenance answers who signed these bytes, which is a different question from
            whether the media was manipulated. The absence of Content Credentials is not
            evidence of manipulation — most media carries none — and their presence is not
            evidence of authenticity. Any remote manifest URL was recorded and never fetched.
          </p>
        </>
      )}
    </Section>
  );
}

function ActiveSpeakerSection({ signal }: { signal: ActiveSpeakerSignal | null }) {
  return (
    <Section
      title="Active speaker (cross-modal speaking evidence)"
      subtitle="Where a tracked face was observed speaking. This is not a deepfake detector."
    >
      {signal === null ? (
        <NoSignal what="active-speaker" />
      ) : (
        <>
          <dl className="grid grid-cols-1 gap-3 sm:grid-cols-2">
            <Field label="Provider" value={signal.provider} />
            <Field label="Status" value={<SignalState status={signal.status} />} />
            <Field label="Provider version" value={signal.provider_version ?? ABSENT} />
            <Field
              label="Speaking segments found"
              value={
                signal.total_speaking_segments === null
                  ? ABSENT
                  : signal.total_speaking_segments.toString()
              }
            />
            <Field
              label="Stored timeline truncated"
              value={
                signal.segments_truncated === null
                  ? ABSENT
                  : signal.segments_truncated
                    ? "yes"
                    : "no"
              }
            />
          </dl>

          {signal.segments.length === 0 ? (
            <p className="mt-3 text-xs opacity-70">
              {signal.status === "SUCCESS"
                ? "The detector ran and recorded no speaking segments. That is an observation about this media, not a missing reading."
                : "No speaking timeline is stored for this signal."}
            </p>
          ) : (
            <table className="mt-3 w-full table-fixed text-left text-xs">
              <thead>
                <tr className="border-b border-black/15 dark:border-white/20">
                  <th className="py-1 font-medium">Start</th>
                  <th className="py-1 font-medium">End</th>
                  <th className="py-1 font-medium">Face ID</th>
                  <th className="py-1 font-medium">Diarized speaker</th>
                </tr>
              </thead>
              <tbody>
                {signal.segments.map((segment, index) => (
                  <tr
                    key={`${segment.start_time}-${segment.face_id}-${index}`}
                    className="border-b border-black/5 last:border-b-0 dark:border-white/10"
                  >
                    <td className="py-1 font-mono">{segment.start_time.toFixed(2)}s</td>
                    <td className="py-1 font-mono">{segment.end_time.toFixed(2)}s</td>
                    <td className="py-1 font-mono">{segment.face_id}</td>
                    <td className="py-1 font-mono">
                      {segment.speaker_label ?? "no matched voice"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}

          <p className="mt-2 text-xs opacity-70">
            The face ID is the provider&apos;s own identifier for a tracked face; the speaker
            label is the diarized voice matched to it. This timeline says who was speaking
            when. It makes no claim about whether the media is genuine.
          </p>
        </>
      )}
    </Section>
  );
}

function AudioSection({ signal }: { signal: AudioAuthenticitySignal | null }) {
  return (
    <Section
      title="AASIST audio evidence"
      subtitle="Raw model output per preprocessing window. No threshold, no calibration, no classes."
    >
      {signal === null ? (
        <NoSignal what="audio-authenticity" />
      ) : (
        <>
          <dl className="grid grid-cols-1 gap-3 sm:grid-cols-2">
            <Field label="Provider" value={signal.provider} />
            <Field label="Status" value={<SignalState status={signal.status} />} />
            <Field label="Checkpoint" value={signal.provider_version ?? ABSENT} />
            <Field
              label="Windows produced"
              value={
                signal.total_audio_windows === null
                  ? ABSENT
                  : signal.total_audio_windows.toString()
              }
            />
            <Field
              label="Windows stored"
              value={
                signal.persisted_audio_windows === null
                  ? ABSENT
                  : signal.persisted_audio_windows.toString()
              }
            />
            <Field
              label="Stored windows truncated"
              value={
                signal.windows_truncated === null
                  ? ABSENT
                  : signal.windows_truncated
                    ? "yes"
                    : "no"
              }
            />
          </dl>

          {signal.windows.length === 0 ? (
            <p className="mt-3 text-xs opacity-70">
              {signal.status === "SUCCESS"
                ? "The reading succeeded and stored no windows."
                : "No audio evidence windows are stored for this signal."}
            </p>
          ) : (
            <table className="mt-3 w-full table-fixed text-left text-xs">
              <thead>
                <tr className="border-b border-black/15 dark:border-white/20">
                  <th className="py-1 font-medium">Window</th>
                  <th className="py-1 font-medium">Start</th>
                  <th className="py-1 font-medium">End</th>
                  <th className="py-1 font-medium">Raw logit[0]</th>
                  <th className="py-1 font-medium">Bona fide logit</th>
                </tr>
              </thead>
              <tbody>
                {signal.windows.map((window) => (
                  <tr
                    key={window.clip_index}
                    className="border-b border-black/5 last:border-b-0 dark:border-white/10"
                  >
                    <td className="py-1 font-mono">{window.clip_index}</td>
                    <td className="py-1 font-mono">{window.start_time.toFixed(2)}s</td>
                    <td className="py-1 font-mono">{window.end_time.toFixed(2)}s</td>
                    <td className="py-1 font-mono">{window.logit.toFixed(4)}</td>
                    <td className="py-1 font-mono">{window.bona_fide_logit.toFixed(4)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}

          <p className="mt-3 text-xs opacity-80">
            These figures are raw model output. They are <strong>not probabilities</strong>,{" "}
            <strong>not confidence values</strong>, and <strong>not Fake/Real decisions</strong>
            . The checkpoint publishes no threshold, no calibration and no classes, so no
            classification is derived from them and none should be read into them.
          </p>
          <p className="mt-2 text-xs opacity-80">
            The time bounds are <strong>InspectRoot preprocessing windows</strong> — where the
            audio was cut before being given to the model. They are not model-detected
            manipulation timestamps, and the model reports no timeline of its own.
          </p>
        </>
      )}
    </Section>
  );
}

function FaceManipulationSection({
  signal,
  analysis,
}: {
  signal: FaceManipulationSignal | null;
  analysis: AnalysisSummary;
}) {
  // Whether this detector was eligible to decide is a property of the ruleset the decision
  // was taken under. R4-T2 promoted it from independent evidence to a calibrated decider, and
  // a report on an older decision must keep saying what was true of that decision.
  const decides = analysis.risk_rules_version === RULES_VERSION_V2;

  return (
    <Section
      title="EfficientNet-B7 face manipulation detector"
      subtitle={
        decides
          ? "Calibrated evidence. The score below is the model's own output, banded against a threshold measured for it in R4-T1."
          : "Independent evidence. The score below is the model's own output and is not part of the risk classification."
      }
    >
      {signal === null ? (
        <NoSignal what="face-manipulation" />
      ) : (
        <>
          <dl className="grid grid-cols-1 gap-3 sm:grid-cols-2">
            <Field label="Provider" value={signal.provider} />
            <Field label="Signal type" value={signal.signal_type} />
            <Field label="Status" value={<SignalState status={signal.status} />} />
            <Field label="Checkpoint" value={signal.provider_version ?? ABSENT} />
            <Field
              label="Model score"
              value={signal.score === null ? ABSENT : signal.score.toString()}
            />
            <Field
              label="Frames sampled"
              value={
                signal.frames_requested === null
                  ? ABSENT
                  : signal.frames_requested.toString()
              }
            />
            <Field
              label="Frames decoded"
              value={
                signal.frames_decoded === null ? ABSENT : signal.frames_decoded.toString()
              }
            />
            <Field
              label="Frames with a detected face"
              value={
                signal.frames_scored === null ? ABSENT : signal.frames_scored.toString()
              }
            />
          </dl>

          {signal.status !== "SUCCESS" && (
            <p className="mt-3 text-xs opacity-70">
              This reading did not produce a score. A clip in which no face was found is the
              ordinary case, and it means the classifier was never asked — it is not a finding
              that the media is genuine.
            </p>
          )}

          <p className="mt-3 text-xs opacity-80">
            The score is the mean of the model&apos;s per-frame output over the frames above,
            shown exactly as the model produced it. It is{" "}
            <strong>not a probability that this media is manipulated</strong> and{" "}
            <strong>not a Fake/Real decision</strong>.
          </p>
          {decides ? (
            <p className="mt-2 text-xs opacity-80">
              Under this ruleset the score is compared against{" "}
              <span className="font-mono">{FACE_T_HIGH_DISPLAY}</span> — the threshold measured
              for this detector in R4-T1, and for this detector only. It is never averaged or
              combined with the synthetic-video score above; the two are separate questions
              with separate answers, and the risk classification names which of them decided.
            </p>
          ) : (
            <p className="mt-2 text-xs opacity-80">
              It is <strong>not calibrated</strong> under this ruleset, no threshold is applied
              to it, and it does not contribute to the risk classification above and cannot
              change it. This signal is recorded as an independent forensic fact only.
            </p>
          )}
        </>
      )}
    </Section>
  );
}

function LipForensicsSection({ signal }: { signal: LipForensicsSignal | null }) {
  // No ruleset-dependent branch, unlike the section above. That one has to say what was true
  // of the decision being reported because R4-T2 promoted its detector from independent
  // evidence to a calibrated decider; this detector has never been read by any ruleset a
  // stored decision can name, so there is one thing to say and it is true of every report.
  return (
    <Section
      title="LipForensics mouth-dynamics detector"
      subtitle="Independent evidence. The score below is the model's own output and is not part of the risk classification."
    >
      {signal === null ? (
        <NoSignal what="mouth-dynamics" />
      ) : (
        <>
          <dl className="grid grid-cols-1 gap-3 sm:grid-cols-2">
            <Field label="Provider" value={signal.provider} />
            <Field label="Signal type" value={signal.signal_type} />
            <Field label="Status" value={<SignalState status={signal.status} />} />
            <Field label="Model" value={signal.provider_version ?? ABSENT} />
            <Field
              label="Model score"
              value={signal.score === null ? ABSENT : signal.score.toString()}
            />
            <Field
              label="Runs sampled"
              value={
                signal.windows_requested === null
                  ? ABSENT
                  : signal.windows_requested.toString()
              }
            />
            <Field
              label="Runs decoded"
              value={signal.windows_read === null ? ABSENT : signal.windows_read.toString()}
            />
            <Field
              label="Runs with a tracked face"
              value={
                signal.windows_scored === null ? ABSENT : signal.windows_scored.toString()
              }
            />
          </dl>

          {signal.status !== "SUCCESS" && (
            <p className="mt-3 text-xs opacity-70">
              This reading did not produce a score. A clip in which no run held a trackable
              face throughout is the ordinary case, and it means the model was never asked —
              it is not a finding that the media is genuine.
            </p>
          )}

          <p className="mt-3 text-xs opacity-80">
            The score is the model&apos;s output for the runs above — each a stretch of 25
            consecutive frames, scored on how the mouth moves across them — shown exactly as
            the model produced it. It is{" "}
            <strong>not a probability that this media is manipulated</strong> and{" "}
            <strong>not a Fake/Real decision</strong>.
          </p>
          <p className="mt-2 text-xs opacity-80">
            Despite the model&apos;s name, this is{" "}
            <strong>not a measure of audio/video lip synchronisation</strong>. The model is
            given no audio at all: it reads the movement of the mouth in the picture and
            nothing else, and what it was trained to separate is forged facial motion from
            genuine facial motion.
          </p>
          <p className="mt-2 text-xs opacity-80">
            It is <strong>not calibrated</strong>, no threshold is applied to it, and it does
            not contribute to the risk classification above and cannot change it. This signal
            is recorded as an independent forensic fact only.
          </p>
          <p className="mt-2 text-xs opacity-80">
            It is also <strong>not a second reading of the face-manipulation score above</strong>
            . That model judges the appearance of a face crop; this one judges movement over
            time. The two figures are on different scales and are never averaged, compared or
            reconciled — agreement between them would not strengthen a finding, and
            disagreement does not weaken one.
          </p>
        </>
      )}
    </Section>
  );
}

export default async function Report({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  const result = await fetchAnalysis(id);

  // No usable session. The reader is sent to sign in rather than shown a report page that
  // cannot fill in — and deliberately before the 404 below, so a signed-out reader is never
  // told anything at all about whether this id exists.
  if (!result.ok && result.unauthenticated) {
    redirect(LOGIN_PATH);
  }

  // A well-formed id the API answered 404 for names no analysis, and the route says so with
  // the status code rather than with a 200 that only reads as an error. `notFound()` works by
  // throwing, so it is called out here where nothing catches: putting it inside `fetchAnalysis`
  // would hand its control-flow exception straight to that function's own `catch`, which would
  // turn a deliberate 404 into "temporarily unavailable".
  //
  // Only an explicit 404 qualifies. A 5xx, an unreachable API and a response that could not be
  // parsed are all states where the analysis may well exist and something else went wrong;
  // reporting those as missing would tell the reader a record is gone when nothing established
  // that. They keep the error page, which does not claim to know either way.
  //
  // Since R1-T2 the API also answers 404 for an analysis this session may not see, and that
  // arrives here as the same `missing`. This page must not tell the two apart or it would
  // hand back the very fact the API withheld: that the id is real, and belongs to somebody
  // else. "Not Found" is the whole of what a reader without access is told.
  if (!result.ok && result.missing) {
    notFound();
  }

  if (!result.ok) {
    return (
      <main className="mx-auto w-full max-w-3xl p-8">
        <Link href="/" className="text-sm underline print:hidden">
          ← Back to dashboard
        </Link>
        <h1 className="mt-6 text-xl font-semibold">Report unavailable</h1>
        <p className="mt-2 text-sm opacity-70">{result.error}</p>
        <p className="mt-1 font-mono text-xs break-all opacity-60">{id}</p>
      </main>
    );
  }

  const analysis = result.analysis;

  return (
    <main className="mx-auto w-full max-w-3xl p-8 print:max-w-none print:p-0">
      {/* Page setup for printing. Plain CSS because @page has no Tailwind equivalent, and
          the report must print correctly with JavaScript disabled. */}
      <style>{`
        @page { size: A4; margin: 14mm; }
        @media print {
          html, body { background: #fff !important; color: #000 !important; }
        }
      `}</style>

      <div className="flex items-start justify-between gap-4">
        <div>
          <p className="text-sm font-semibold tracking-wide">InspectRoot</p>
          <h1 className="text-xl font-semibold">Forensic Evidence Report</h1>
        </div>
        {/* Screen-only controls. Hidden in print so the document carries no dead UI. */}
        <div className="flex items-center gap-4 print:hidden">
          <Link href="/" className="text-sm underline">
            ← Dashboard
          </Link>
          <PrintButton />
        </div>
      </div>

      <dl className="mt-4 grid grid-cols-1 gap-3 sm:grid-cols-2">
        <Field label="Analysis ID" value={analysis.id} />
        <Field label="Analysis status" value={analysis.status} />
        <Field label="Analysis timestamp (UTC)" value={analysis.created_at} />
        <Field
          label="Size on disk"
          value={analysis.size_bytes === null ? ABSENT : `${analysis.size_bytes} bytes`}
        />
      </dl>

      <ScopeDisclosure analysis={analysis} />

      <RiskSection analysis={analysis} />

      <MediaSection analysis={analysis} media={analysis.media} />

      <h2 className="mt-8 text-sm font-semibold uppercase tracking-wide">
        Independent forensic evidence
      </h2>
      <p className="mt-1 text-xs opacity-70">
        Each source is recorded separately and none of them is combined into the other.
        {analysis.risk_rules_version === RULES_VERSION_V2
          ? " Two of them are calibrated and can reach the risk classification above — the synthetic-video detector and the face-manipulation classifier — each against a threshold measured for it alone, and never by pooling their scores. Provenance, speaking evidence, mouth-dynamics evidence and audio evidence have no calibrated threshold and cannot change that classification."
          : " Only the synthetic-video detector contributes to the risk classification above; provenance, speaking evidence, face-manipulation evidence, mouth-dynamics evidence and audio evidence are recorded as independent forensic facts and cannot change that classification."}
      </p>

      <SyntheticVideoSection signal={analysis.synthetic_video} />
      <ProvenanceSection signal={analysis.provenance} />
      <ActiveSpeakerSection signal={analysis.active_speaker} />
      <FaceManipulationSection signal={analysis.face_manipulation} analysis={analysis} />
      <LipForensicsSection signal={analysis.lip_forensics} />
      <AudioSection signal={analysis.audio_authenticity} />

      <footer className="mt-8 break-inside-avoid border-t border-black/15 pt-4 text-xs opacity-70 dark:border-white/20">
        <p>
          This report is a rendering of forensic evidence persisted by InspectRoot for the
          analysis named above. It is not cryptographically signed, and reproducing it does
          not establish that its contents are unaltered. The SHA-256 shown is the hash of the
          analysed media, not of this report.
        </p>
        <p className="mt-2">
          Nothing in this document states that the analysed media is genuine or manipulated.
        </p>
      </footer>
    </main>
  );
}
