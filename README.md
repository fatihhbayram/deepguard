# DeepGuard

DeepGuard is an evidence-oriented content authenticity and forensics platform. It is designed to
combine independent synthetic-media, provenance, metadata and speaker signals into a reviewable
body of evidence, rather than returning a single verdict on whether media is fake.

## Current status

The upload pipeline exists end to end and detection runs asynchronously behind it. A video can
be submitted, validated, stored, described, queued, analysed by NVIDIA's synthetic video
detector in a background worker and persisted, and the resulting analysis is listed in the
dashboard. What runs today:

- `POST /api/v1/analyses` accepts a video upload;
- the upload is admitted only by declared MIME type (`video/mp4`, `video/quicktime`) and a
  100 MiB size ceiling, both enforced while the stream is read in bounded chunks;
- the original bytes are hashed with SHA-256 in that same single read pass;
- the original is stored in MinIO under a content-addressed key, byte for byte;
- `ffprobe` proves the bytes really are video and yields container, codec, dimensions,
  duration, frame rate and pixel format;
- media that is not already in the provider-compatible shape (MP4 container, H.264,
  `yuv420p`, constant frame rate) gets a separate MP4/H.264 derivative via `ffmpeg`;
- original and derivative are kept as distinct artifacts, each with its own SHA-256 and its
  own storage key — the original is never rewritten;
- the analysis, its media facts and a `queued` job are written to PostgreSQL in one
  transaction, and the upload answers `202 Accepted` — an analysis committed without its job
  would be an upload accepted and then silently forgotten;
- detection does not run on the request. NVIDIA can take minutes on a video, and no client is
  made to hold a connection open through them;
- a separate worker process claims queued jobs with `SELECT ... FOR UPDATE SKIP LOCKED`,
  marks each `processing` and commits before doing any work, so more than one worker is safe
  and no transaction is held open across inference;
- the worker fetches the provider-compatible object from MinIO, sends it to NVIDIA's
  Synthetic Video Detector over gRPC, and deletes the local copy either way;
- NVIDIA's probability is recorded as returned, on NVIDIA's own scale, as one signal among
  the several a full analysis will eventually carry;
- the clips NVIDIA scored inside the video are kept as evidence in their own right, capped at
  the strongest twenty per detection, each with the provider's clip index and raw logit;
- a detector that fails, times out or is misconfigured does not fail the job: the failure is
  itself persisted as a signal, so the gap in the evidence is visible rather than silent. A
  fault on this side — object storage unreachable, an unreadable artifact — fails the job and
  the analysis instead, with the failure kind on the job row;
- the signal, its clip evidence and the `completed` statuses are written in one final
  transaction;
- `GET /api/v1/analyses` returns the most recent analyses with their signal and its strongest
  clips;
- the Next.js dashboard renders that list beside the connectivity status.

No provenance, risk classification or reporting functionality is implemented yet. Signals
carry a provider score but no risk level; deciding what a score means is a later phase.

## Architecture

```text
Next.js (web)
   ↓ REST
FastAPI (api) ──→ PostgreSQL ←── worker (api-worker)
   ↓                  ↑              ↓
 MinIO ───────────────┘         NVIDIA SVD
```

The API and the worker are the same image with different commands. They never call each
other: the API writes a job, the worker reads one, and the jobs table is the whole of their
conversation. Adding a second worker needs no coordination beyond that — `SKIP LOCKED` is
what keeps two of them off the same row.

The browser reaches the API over the published host port. Server-side rendering in the web
container reaches it over the Docker network at `http://api:8000`. Both API and worker reach
MinIO over the Docker network at `minio:9000`, never the published host port. NVIDIA is
reached from the worker over TLS gRPC at its hosted endpoint, addressed by function ID.

Uploads return once the media is staged and the work is queued: storage, probing,
normalization and persistence run on the request, detection does not. If storage, probing or
normalization fails, no analysis row and no job are written. Local temp files are always
removed — the worker reads the canonical object back out of MinIO rather than depending on a
file the request left behind. Stored MinIO objects are not deleted on failure, because their
keys are content-addressed and may already be referenced by an earlier analysis of identical
bytes.

The worker takes each job in three steps, never one transaction: claim it and commit, then
download and detect with nothing open, then write the evidence and close the job out.
Holding the claim across inference would pin a connection and a row lock for however long
NVIDIA takes.

A job is claimed once. Retrying a failed job and recovering one whose worker died mid-flight
are not implemented — a job left `processing` stays there.

## Repository structure

```text
apps/api/              FastAPI service and the worker that shares its image
  app/main.py          /health endpoint, router wiring
  app/api/analyses.py  upload pipeline and analysis listing
  app/worker.py        job claiming and the background processing loop
  app/detection.py     one detector's answer, turned into evidence
  app/media.py         ffprobe validation and metadata extraction
  app/normalization.py provider-compatible MP4/H.264 derivatives
  app/nvidia_video.py  NVIDIA Synthetic Video Detector gRPC client
  app/storage.py       MinIO object storage
  app/db/              SQLAlchemy models, engine and session
  alembic/             database migrations
  tests/               pytest suite
apps/web/              Next.js application
  app/page.tsx         connectivity status and recent analyses
docs/planning/         product, API and data-model scope
docker-compose.yml     service definitions
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

The NVIDIA signal is not in the upload response — it is evidence attached to the analysis,
returned by `GET /api/v1/analyses` and shown on the dashboard. It can also be read directly:

```sql
SELECT provider, signal_type, status, score, metadata FROM analysis_signals;

SELECT clip_index, logit FROM analysis_segments ORDER BY logit DESC;
```

A segment is one clip the detector scored. NVIDIA reports a frame index and a raw logit per
clip and no timestamps, so those two figures are what is stored — the logit is not a
probability and is not comparable with the signal's `score`.

Detection needs `NVIDIA_API_KEY` and `NVIDIA_SVD_FUNCTION_ID` in `.env`, read by the worker
rather than the API. Missing or wrong credentials never fail an upload and never fail a job:
the analysis completes carrying a `FAILED` signal instead of a score.

Watch a job being processed:

```bash
docker compose logs -f api-worker
```

Rejections are bounded and generic: `415` for an unsupported MIME type, `413` above the
100 MiB limit, `422` when the bytes are not usable video, `503` when object storage or the
media processor is unavailable. A provider failure is never one of them.

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

The API and the worker share one image, so a change to either rebuilds both.
