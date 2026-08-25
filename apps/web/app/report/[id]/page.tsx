import Link from "next/link";
import { notFound } from "next/navigation";

import {
  ActiveSpeakerSignal,
  AnalysisSummary,
  AudioAuthenticitySignal,
  MediaFacts,
  ProvenanceSignal,
  RISK_LABELS,
  SyntheticVideoSignal,
  UNSUPPORTED,
  fetchAnalysis,
  isSupportedRiskLevel,
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
 * The persisted product-level risk classification, with the trace that makes it explainable.
 *
 * Read from the analysis row exactly as stored. Nothing on this page looks at the NVIDIA
 * score below to decide what to print here.
 */
function RiskSection({ analysis }: { analysis: AnalysisSummary }) {
  const level = analysis.risk_level;

  return (
    <section
      className={`mt-6 break-inside-avoid rounded border-2 p-4 ${riskAccent(level)}`}
    >
      <h2 className="text-base font-semibold">DeepGuard risk classification</h2>

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
          classifies under, so it is reported as unsupported rather than presented as a
          DeepGuard classification.
        </p>
      ) : level === "UNKNOWN" ? (
        <p className="mt-1 text-xs opacity-70">
          The risk engine ran and a rule fired. Its conclusion is that the evidence does not
          support a classification — an answer, not a missing one.
        </p>
      ) : null}

      <dl className="mt-4 grid grid-cols-1 gap-3 sm:grid-cols-3">
        <Field label="Rule fired" value={analysis.risk_rule_id ?? ABSENT} />
        <Field label="Ruleset version" value={analysis.risk_rules_version ?? ABSENT} />
        <Field label="Calibration ID" value={analysis.risk_calibration_id ?? ABSENT} />
      </dl>

      <p className="mt-4 text-xs opacity-80">
        Risk is a deterministic DeepGuard classification based on calibrated forensic
        evidence. It is not a Fake/Real determination. This classification was recorded when
        the analysis ran and is reproduced here unchanged; it is not recalculated by this
        report.
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
function ScopeDisclosure() {
  return (
    <section className="mt-4 break-inside-avoid rounded border-2 border-amber-500/60 p-4">
      <h2 className="text-sm font-semibold uppercase tracking-wide">
        Scope of this risk model
      </h2>
      <p className="mt-2 text-sm font-medium">
        This risk model is validated for generated video and is not validated for face-swap
        detection. Absence of HIGH risk does not rule out face manipulation.
      </p>
      <p className="mt-2 text-xs opacity-80">
        Risk is a deterministic DeepGuard classification based on calibrated forensic
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
            returned. It is NVIDIA evidence, not a DeepGuard confidence, and it is not the
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
            The time bounds are <strong>DeepGuard preprocessing windows</strong> — where the
            audio was cut before being given to the model. They are not model-detected
            manipulation timestamps, and the model reports no timeline of its own.
          </p>
        </>
      )}
    </Section>
  );
}

export default async function Report({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  const result = await fetchAnalysis(id);

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
          <p className="text-sm font-semibold tracking-wide">DeepGuard</p>
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

      <ScopeDisclosure />

      <RiskSection analysis={analysis} />

      <MediaSection analysis={analysis} media={analysis.media} />

      <h2 className="mt-8 text-sm font-semibold uppercase tracking-wide">
        Independent forensic evidence
      </h2>
      <p className="mt-1 text-xs opacity-70">
        Each source is recorded separately and none of them is combined into the other. Only
        the synthetic-video detector contributes to the risk classification above; provenance,
        speaking evidence and audio evidence are recorded as independent forensic facts and
        cannot change that classification.
      </p>

      <SyntheticVideoSection signal={analysis.synthetic_video} />
      <ProvenanceSection signal={analysis.provenance} />
      <ActiveSpeakerSection signal={analysis.active_speaker} />
      <AudioSection signal={analysis.audio_authenticity} />

      <footer className="mt-8 break-inside-avoid border-t border-black/15 pt-4 text-xs opacity-70 dark:border-white/20">
        <p>
          This report is a rendering of forensic evidence persisted by DeepGuard for the
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
