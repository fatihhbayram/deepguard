"""The real-world degradations, as ffmpeg filter graphs and encoder settings.

A cross-dataset robustness measurement needs media that has been through what real media goes
through, and almost none of the publicly mirrored research corpora have. What they carry is one
encode of one upload. What arrives at a product is that upload after a messaging app halved its
resolution, after a platform re-encoded it at a bitrate chosen for bandwidth rather than
fidelity, in a room with one lamp, held in a hand.

So the degradations are constructed, deterministically, from clips whose ground truth is known.
This is a weaker instrument than collecting the real thing and it is labelled as such
everywhere it is reported: a simulated platform transcode is an approximation of one, and a
`camera_motion` derivative is a crop that moves, not a person walking. It is nevertheless the
only way to ask a detector the question at all across hundreds of lineages, and the failure it
is looking for — a mouth-dynamics score that climbs when frames are re-timed or smeared — does
not need the simulation to be perfect to show up.

**Every derivative keeps its base's ground truth and its base's lineage.** A transcode of
genuine media is genuine; a transcode of a face swap is that face swap. Nothing here can turn
one label into another, and `build_corpus` copies the lineage id across unchanged so the split
cannot separate a derivative from what it came from.

Encoding is pinned: libx264, fixed CRF and preset per derivation, `-an` never used (the audio
is kept, because the detectors' preparation path probes it and a stream that vanishes changes
what is measured for reasons unrelated to the picture). Two runs of the same derivation over
the same base produce the same bytes on the same ffmpeg build, and the build is recorded.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Derivation:
    """One degradation: what it is called, what it does, and what it stands in for.

    `stratum` is the stratum the *derivative* belongs to, which is never the base's stratum —
    that is the entire point of making it. `rationale` is carried into the corpus artifact so a
    reader of a per-stratum table is told what the stratum actually contains rather than
    guessing from its name.
    """

    derivation_id: str
    stratum: str
    video_filter: str
    crf: int
    preset: str
    audio_bitrate: str
    extra_output_args: tuple[str, ...]
    rationale: str


# The eight degradations, chosen to cover the genuine strata the plan names that no mirrored
# research corpus supplies on its own. Values are the ones consumer platforms and consumer
# hardware actually land on, not extremes: an implausible degradation that breaks a detector
# proves nothing about media anyone would upload.
DERIVATIONS = (
    Derivation(
        "phone_reencode",
        "genuine_phone_reencode",
        "scale=-2:720",
        26,
        "medium",
        "128k",
        (),
        "A share-sheet re-encode: 720p, moderate CRF, the shape a clip takes when it leaves "
        "one phone and arrives on another.",
    ),
    Derivation(
        "social_transcode_reels",
        "genuine_social_transcode",
        "scale=-2:1080,fps=30",
        28,
        "veryfast",
        "128k",
        (),
        "A vertical-feed transcode: 1080-tall, re-timed to a constant 30 fps, encoded fast at "
        "the bitrate a feed picks rather than the one the source deserved.",
    ),
    Derivation(
        "social_transcode_shorts",
        "genuine_social_transcode",
        "scale=-2:540,fps=30",
        32,
        "veryfast",
        "96k",
        (),
        "The second rung of the same ladder, which is what most viewers of a clip actually "
        "see: 540-tall at a visibly lossy CRF.",
    ),
    Derivation(
        "heavy_compression",
        "genuine_heavy_compression",
        "scale=-2:480",
        38,
        "veryfast",
        "64k",
        (),
        "The bottom rung, and the closest constructed analogue of the two real-world "
        "regression lineages, both of which arrived at 480 lines or fewer.",
    ),
    Derivation(
        "low_light",
        "genuine_low_light",
        "eq=brightness=-0.16:contrast=1.06:gamma=0.72,noise=alls=6:allf=t",
        30,
        "medium",
        "128k",
        (),
        "One lamp and a sensor pushed past it: the picture darkened non-linearly and given "
        "the temporal luminance noise a phone adds when it raises gain.",
    ),
    Derivation(
        "camera_motion",
        "genuine_camera_motion",
        "crop=iw*0.82:ih*0.82:(iw-ow)/2+sin(t*5.1)*iw*0.045:(ih-oh)/2+cos(t*3.7)*ih*0.045",
        28,
        "medium",
        "128k",
        (),
        "A handheld frame: the picture cropped to 82% and the crop window driven around it by "
        "two incommensurate sinusoids, so the motion never repeats within a clip.",
    ),
    Derivation(
        "low_framerate",
        "genuine_low_framerate",
        "fps=15",
        28,
        "medium",
        "96k",
        (),
        "Half the frames, which is the case a temporal detector should be asked about "
        "explicitly: LipForensics reads runs of 25 consecutive frames and a decimated clip "
        "gives it twice the mouth movement per run.",
    ),
    Derivation(
        "small_face",
        "genuine_small_face",
        "scale=iw*0.45:ih*0.45,pad=iw/0.45:ih/0.45:(ow-iw)/2:(oh-ih)/2:black",
        28,
        "medium",
        "128k",
        (),
        "The subject further away: the picture shrunk into a frame of its original size, so "
        "the face occupies about a fifth of the pixels it did without changing the geometry.",
    ),
)

DERIVATIONS_BY_ID = {derivation.derivation_id: derivation for derivation in DERIVATIONS}

# The subset applied to manipulated media. A manipulated clip is in this corpus to answer
# "does the detector still find it after the internet has been through it", and the platform
# ladder is what the internet is; the acquisition-condition derivations (light, motion, face
# size) are questions about genuine capture and are not asked of a face swap.
MANIPULATED_DERIVATIONS = (
    "social_transcode_reels",
    "social_transcode_shorts",
    "heavy_compression",
    "low_framerate",
)


class DeriveError(Exception):
    """A derivative could not be produced, so it is absent rather than approximate."""


def ffmpeg_version() -> str:
    """The encoder build every derivative in this corpus was produced by."""
    completed = subprocess.run(
        ["ffmpeg", "-version"], capture_output=True, text=True, check=False
    )
    return completed.stdout.splitlines()[0] if completed.stdout else "unknown"


def derive(source: Path, destination: Path, derivation: Derivation) -> None:
    """Produce one derivative, or raise and leave nothing behind.

    `-map_metadata -1` is not passed: container metadata is part of what a detector's
    preparation path probes, and stripping it would make every derivative in this corpus
    look like a class of file that no camera and no platform produces.

    A partially written output is removed. A truncated mp4 that decodes to three frames would
    be scored, and a score over three frames is not a measurement of anything.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-nostdin",
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(source),
        "-vf",
        derivation.video_filter,
        "-c:v",
        "libx264",
        "-crf",
        str(derivation.crf),
        "-preset",
        derivation.preset,
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        derivation.audio_bitrate,
        "-movflags",
        "+faststart",
        *derivation.extra_output_args,
        str(destination),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0 or not destination.is_file():
        destination.unlink(missing_ok=True)
        raise DeriveError(
            f"{derivation.derivation_id} of {source.name} failed: "
            f"{completed.stderr.strip()[:400]}"
        )
    if destination.stat().st_size == 0:
        destination.unlink()
        raise DeriveError(f"{derivation.derivation_id} of {source.name} produced no bytes")
