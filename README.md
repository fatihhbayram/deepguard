# DeepGuard

DeepGuard is an evidence-oriented content authenticity and forensics platform. It is designed to
combine independent synthetic-media, provenance, metadata and speaker signals into a reviewable
body of evidence, rather than returning a single verdict on whether media is fake.

## Current status

The upload pipeline exists end to end and analysis runs asynchronously behind it. A video can
be submitted, validated, stored, described, queued, analysed in a background worker by NVIDIA's
synthetic video detector, by NVIDIA's Active Speaker Detection NIM and by the C2PA provenance
reader, and persisted, and the resulting analysis is listed in the dashboard. What runs today:

- `POST /api/v1/analyses` accepts a video upload;
- the upload is admitted only by declared MIME type (`video/mp4`, `video/quicktime`) and a
  100 MiB size ceiling, both enforced while the stream is read in bounded chunks;
- the original bytes are hashed with SHA-256 in that same single read pass;
- the original is stored in MinIO under a content-addressed key, byte for byte;
- `ffprobe` proves the bytes really are video and yields container, codec, dimensions,
  duration, frame rate and pixel format;
- media that is not already in the provider-compatible shape (MP4 container, H.264,
  `yuv420p`, constant frame rate) is marked as needing a separate MP4/H.264 derivative,
  which the worker transcodes with `ffmpeg` — not the request. A 4K source can take
  minutes, and the deadline that used to bound it on the request path rejected perfectly
  good media before anything had been analysed;
- original and derivative are kept as distinct artifacts, each with its own SHA-256 and its
  own storage key — the original is never rewritten;
- the analysis, its media facts and a `queued` job are written to PostgreSQL in one
  transaction, and the upload answers `202 Accepted` — an analysis committed without its job
  would be an upload accepted and then silently forgotten;
- neither transcoding nor detection runs on the request. Both are unbounded in a way
  validating, hashing and probing the uploaded bytes are not, and no client is made to
  hold a connection open through them;
- a separate worker process claims queued jobs with `SELECT ... FOR UPDATE SKIP LOCKED`,
  marks each `processing` and commits before doing any work, so more than one worker is safe
  and no transaction is held open across inference;
- the worker fetches the forensic original from MinIO once, transcodes it if a derivative
  is owed, stores that derivative under its own content-addressed key, sends the resulting
  artifact to NVIDIA's Synthetic Video Detector and then to NVIDIA's Active Speaker Detection
  NIM over gRPC, and deletes every local copy either way;
- the derivative's key and hash are written to the database only once the object exists —
  a queued analysis that still owes one carries null in both columns rather than a key
  naming nothing;
- media that cannot be transcoded is recorded as a failed NVIDIA signal, not a failed job:
  the provider was never reachable for it, and the provenance already read off the same
  file is kept;
- the worker separately reads the C2PA Content Credentials embedded in the *forensic
  original* — never the derivative, because normalization re-encodes the video and strips
  any manifest with it, which would report every normalized upload as unsigned;
- what C2PA reports is passed through in C2PA's own words: whether a manifest exists at all,
  the SDK's validation state (`Trusted`, `Valid`, `Invalid`), its failure codes, the claim
  generator and the signature issuer. Remote manifest fetching and OCSP revocation lookups
  are switched off, so an uploaded file can never steer a request out of the worker; a
  remote manifest URL is recorded as evidence and not visited;
- absent Content Credentials are a successful reading, not a failure and not a finding. Most
  media carries none, and neither their absence nor an invalid signature is treated as
  evidence that the media is fake;
- NVIDIA's probability is recorded as returned, on NVIDIA's own scale, as one signal among
  the several a full analysis will eventually carry;
- provenance is persisted as its own independent signal beside it, with no score and no risk
  level: it is the state of a signature, not a figure on a scale;
- the clips NVIDIA scored inside the video are kept as evidence in their own right, capped at
  the strongest twenty per detection, each with the provider's clip index and raw logit;
- the same prepared artifact is also sent to NVIDIA's Active Speaker Detection NIM, which
  answers per frame which tracked face is speaking. That NIM does not diarize — it takes
  speaker turns as an *input* — so the turns are produced locally by
  `pyannote/speaker-diarization-community-1` running on CPU in the worker;
- audio is extracted once, as a mono 16 kHz PCM WAV, from the same artifact NVIDIA is given,
  and used twice: pyannote diarizes it and the identical file is streamed to NVIDIA as the
  separate audio track it wants beside the video. Extracting from the analysed artifact rather
  than the original is what keeps the two timelines the same one;
- pyannote's speaker labels are strings, so the worker assigns each voice the integer NVIDIA's
  proto requires by order of first appearance. Nothing is parsed out of the label. The integer
  is a wire-format detail and is never stored as the speaker's identity — the label is;
- NVIDIA's per-frame verdicts are collapsed into unbroken speaking runs, keyed by face *and*
  by the voice matched to it, and closed on any gap in the frame numbering so no run asserts
  speech across frames the provider said nothing about. Runs become time ranges against the
  frame rate the artifact was analysed at;
- speaking segments are persisted in chronological order, capped at the first fifty, with the
  detection's own total and a truncation flag beside them — a timeline is only readable as a
  contiguous run from the start, so the cap keeps a prefix rather than a selection;
- the active-speaker signal carries no score. It produces a timeline, not a figure on a scale,
  and NVIDIA reports no speaking confidence: `is_speaking` is already thresholded provider-side,
  and the confidence it does report scores the face *detection*, not the speech;
- the whole active-speaker chain — audio extraction, diarization, NVIDIA — records every one of
  its failures as a `FAILED` signal and never fails the job. Media with no audio, a missing
  Hugging Face token or an unreachable second NVIDIA function costs that one signal and leaves
  the provenance and synthetic-video evidence untouched;
- a detector that fails, times out or is misconfigured does not fail the job: the failure is
  itself persisted as a signal, so the gap in the evidence is visible rather than silent. A
  fault on this side — object storage unreachable, an unreadable artifact — fails the job and
  the analysis instead, with the failure kind on the job row;
- one evidence source breaking does not cost the analysis the other: an analysis can carry a
  failed detection beside a successful provenance reading, or the reverse, and the job still
  completes;
- all three signals, the clip evidence, the speaking timeline and the `completed` statuses are
  written in one final transaction;
- `GET /api/v1/analyses` returns the most recent analyses with all three signals, the
  detection's strongest clips, the speaking timeline, and the container/codec facts ffprobe
  established about the original;
- the Next.js dashboard renders that list beside the connectivity status.

No risk classification or reporting functionality is implemented yet. Signals carry a
provider's own figures but no risk level; deciding what a score or a validation state means
is a later phase.

## Architecture

```text
Next.js (web)
   ↓ REST
FastAPI (api) ──→ PostgreSQL ←── worker (api-worker)
   ↓                  ↑              ↓
 MinIO ───────────────┘         NVIDIA SVD + NVIDIA ASD + C2PA
                                     ↑
                                pyannote diarization (local, CPU)
```

The API and the worker are the same image with different commands. They never call each
other: the API writes a job, the worker reads one, and the jobs table is the whole of their
conversation. Adding a second worker needs no coordination beyond that — `SKIP LOCKED` is
what keeps two of them off the same row.

The browser reaches the API over the published host port. Server-side rendering in the web
container reaches it over the Docker network at `http://api:8000`. Both API and worker reach
MinIO over the Docker network at `minio:9000`, never the published host port. NVIDIA is
reached from the worker over TLS gRPC at its hosted endpoint, addressed by function ID.

Uploads return once the media is staged and the work is queued: storage, probing and
persistence run on the request, transcoding and detection do not. If storage or probing
fails, no analysis row and no job are written. Local temp files are always
removed — the worker reads the canonical object back out of MinIO rather than depending on a
file the request left behind. Stored MinIO objects are not deleted on failure, because their
keys are content-addressed and may already be referenced by an earlier analysis of identical
bytes.

The worker takes each job in three steps, never one transaction: claim it and commit, then
download, read provenance, transcode, diarize and detect with nothing open, then write the
derivative's identity and the evidence and close the job out. Holding the claim across that
middle step would pin a connection and a row lock for however long ffmpeg, pyannote and NVIDIA
take. The evidence sources read different artifacts — C2PA the forensic original, both NVIDIA
NIMs the provider-compatible one — and none waits on another's outcome. The two NVIDIA calls
run in separate event loops for the same reason: one being slow, cancelled or broken must not
reach into the other.

Diarization is the only model that runs on this machine. It is CPU-only by construction — the
PyTorch stack is pinned to the CPU wheel index — and the model is fetched from Hugging Face on
first use and cached inside the container, so the first job after a rebuild is slower than the
ones after it and needs network access to `huggingface.co`.

A job is claimed once. Retrying a failed job and recovering one whose worker died mid-flight
are not implemented — a job left `processing` stays there.

## Repository structure

```text
apps/api/                FastAPI service and the worker that shares its image
  app/main.py            /health endpoint, router wiring
  app/api/analyses.py    upload pipeline and analysis listing
  app/worker.py          job claiming and the background processing loop
  app/detection.py       each evidence source's answer, turned into a signal
  app/c2pa_extractor.py  C2PA Content Credentials read from a local file
  app/media.py           ffprobe validation and metadata extraction
  app/normalization.py   provider-compatible MP4/H.264 derivatives
  app/nvidia_video.py    NVIDIA Synthetic Video Detector gRPC client
  app/nvidia_active_speaker.py       NVIDIA Active Speaker Detection gRPC client
  app/nvidia_active_speaker_proto/   vendored NVIDIA protos and generated stubs
  app/speaker_diarization.py         WAV extraction and local pyannote diarization
  app/storage.py         MinIO object storage
  app/db/                SQLAlchemy models, engine and session
  alembic/               database migrations
  tests/                 pytest suite
apps/web/                Next.js application
  app/page.tsx           connectivity status and recent analyses
docs/planning/           product, API and data-model scope
docker-compose.yml       service definitions
```

## Prerequisites

- Docker with the Compose plugin — sufficient on its own to run the whole stack.
- Node.js 22 — only for running frontend checks outside the container.
- Python 3.12 — only for running backend tests outside the container.

## Setup

```bash
cp .env.example .env
docker compose up --build -d
```

The defaults in `.env.example` are development credentials and work as-is. Change
`POSTGRES_PASSWORD` and `MINIO_ROOT_PASSWORD` before deploying anywhere shared.

Apply the database schema once the stack is up. Migrations are run explicitly, not on
container start:

```bash
docker compose exec api alembic upgrade head
```

The backend image is about **2.1 GB**. Almost all of it is the CPU-only PyTorch stack that
local speaker diarization needs, and the API and the worker share one image, so the API
carries it without ever using it. This is an accepted MVP trade-off, not an oversight:
splitting the images would buy back disk at the cost of a second build and a second thing to
keep in step, and nothing yet needs that. See *Notes on changing dependencies*.

The first analysis after a build also downloads the diarization model from Hugging Face and
caches it inside the container, so it is slower than the ones after it and needs network
access to `huggingface.co`.

## Services and default ports

| Service | URL | Purpose |
| --- | --- | --- |
| Web | http://localhost:3000 | Next.js application |
| API | http://localhost:8000 | FastAPI service |
| PostgreSQL | localhost:5432 | Database |
| MinIO API | http://localhost:9002 | Object storage |
| MinIO console | http://localhost:9003 | Object storage UI |

Every port is published on `127.0.0.1` only, and each one can be overridden in `.env`.

## Verifying the stack

Check the containers. Five run; the four that serve something report healthy:

```bash
docker compose ps
```

`api-worker` has no healthcheck and publishes no port — nothing calls it, so there is
nothing to probe. That it is `Up` is what there is to check; what it is doing is in its log.

Query the API directly:

```bash
curl http://localhost:8000/health
```

A healthy stack returns `{"status":"ok","database":"ok"}`. If the database is unreachable, the
endpoint returns HTTP 503 with `{"status":"degraded","database":"unavailable"}`.

Then open http://localhost:3000, which shows the same information as a status page and reports
`OPERATIONAL` when the API and database are both reachable. The page also lists the most
recent analyses, or `No analyses yet.` on a fresh database.

## Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/health` | API and database reachability |
| POST | `/api/v1/analyses` | Upload a video and create an analysis |
| GET | `/api/v1/analyses` | List the most recent analyses |

Upload a video and see it appear in the dashboard:

```bash
curl -F 'file=@sample.mp4;type=video/mp4' http://localhost:8000/api/v1/analyses
```

The answer is `202 Accepted`, carrying the analysis id, the original's SHA-256 and storage
key, the ffprobe metadata, `was_normalized`, and the derivative's storage key and SHA-256.
When the original is already provider-compatible, no derivative is produced:
`was_normalized` is `false`, `derivative_storage_key` is the original's key and
`derivative_sha256` is `null`. `status` is `queued`, because nothing has looked at the video
yet. The queued work is a row of its own:

```sql
SELECT status, error_message, created_at, updated_at FROM analysis_jobs;
```

Neither signal is in the upload response — they are evidence attached to the analysis,
returned by `GET /api/v1/analyses` and shown on the dashboard. They can also be read directly:

```sql
SELECT provider, signal_type, status, score, metadata FROM analysis_signals;

SELECT clip_index, logit FROM analysis_segments ORDER BY logit DESC;

SELECT start_time, end_time, face_id, speaker_label FROM analysis_segments
WHERE start_time IS NOT NULL ORDER BY start_time;
```

`analysis_segments` carries two shapes of evidence, and which one a row is comes from its
signal's `signal_type` rather than from a column of its own. A clip row is one clip the
synthetic-video detector scored: NVIDIA reports a frame index and a raw logit per clip and no
timestamps, so those two figures are what is stored — the logit is not a probability and is not
comparable with the signal's `score`. A speaking row is one unbroken stretch in which the
Active Speaker NIM saw a tracked face speaking, in seconds from the start of the analysed
video, with the face NVIDIA tracked and pyannote's label for the voice it was matched to. A
face matched to no voice keeps its row with a null label — that is an observation, not a gap.
Provenance produces no segments: a signature has no timeline.

Each analysis carries one row per evidence source — `nvidia`/`synthetic_video`,
`nvidia`/`active_speaker` and `c2pa`/`provenance` — and the dashboard shows them in separate
columns. The two NVIDIA signals are separate deployments and each records the function ID that
answered it, so one `provider` carries two different `provider_version` values. The
active-speaker column keeps four outcomes apart: `—` (no signal), `Unavailable` (the chain did
not get to look), `No speaking faces detected` (it looked and saw nobody) and the timeline
itself, shown as `N segments` or `N of M segments` where the stored prefix stops short of what
the detection found. Active Speaker Detection is not a deepfake detector and none of these is a
verdict about the media. The provenance column
keeps five outcomes apart: `No provenance` (the file was read and carries no credentials),
`Remote provenance (not fetched)` (the file names a manifest kept elsewhere, and that URL was
recorded and deliberately never visited), the C2PA validation state verbatim,
`Extraction failed` (the file could not be read), and `—` (no reading was ever made). None of
them is a verdict about the media.

The dashboard also shows what ffprobe read out of the original — codec, resolution, frame rate
and container — beside the MIME type the client declared. The container is reported as
ffprobe's demuxer family, one name covering MOV and MP4 alike, because only `major_brand`
separates them and no column holds it.

Detection needs `NVIDIA_API_KEY` and `NVIDIA_SVD_FUNCTION_ID` in `.env`, read by the worker
rather than the API. Missing or wrong credentials never fail an upload and never fail a job:
the analysis completes carrying a `FAILED` signal instead of a score.

Active speaker needs two more, both worker-only: `NVIDIA_ASD_FUNCTION_ID`, a separate NVCF
deployment from the synthetic-video detector, and `HUGGINGFACE_TOKEN`, belonging to an account
that has accepted the conditions of the gated
[`pyannote/speaker-diarization-community-1`](https://huggingface.co/pyannote/speaker-diarization-community-1)
model. A read-only token is enough. Without either one the active-speaker signal is recorded as
`FAILED` and the other two evidence sources are unaffected.

Watch a job being processed:

```bash
docker compose logs -f api-worker
```

Rejections are bounded and generic: `415` for an unsupported MIME type, `413` above the
100 MiB limit, `422` when the bytes are not usable video, `503` when object storage or the
media processor is unavailable. Neither a provider failure nor a failed transcode is one of
them — both happen after the response and are recorded as evidence instead.

## Tests and quality checks

Most backend tests use a stub session, so no running database is required:

```bash
cd apps/api
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
pytest
```

The tests marked `integration` verify persistence against the real PostgreSQL from
Compose and skip when it is unreachable. Run them with the stack up and the migrations
applied:

```bash
pytest -m integration
```

Frontend checks:

```bash
cd apps/web
npm install
npm run typecheck
npm run lint
```

## Notes on changing dependencies

Dependencies are installed at image build time. After editing `apps/api/requirements.txt` or
`apps/web/package.json`, rebuild the affected service — restarting it is not enough:

```bash
docker compose build api
docker compose up -d api api-worker
```

The API and the worker share one image, so a change to either rebuilds both. That image is
about 2.1 GB, almost all of it the CPU-only PyTorch stack `pyannote.audio` requires. `torch`,
`torchaudio` and `torchcodec` are pinned to `+cpu` local versions against
`https://download.pytorch.org/whl/cpu`, so the far larger CUDA wheels cannot resolve in. The
API process never diarizes and imports `pyannote.audio` lazily, but it ships the same bytes.
