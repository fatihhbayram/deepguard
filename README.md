# DeepGuard

DeepGuard is an evidence-oriented content authenticity and forensics platform. It is designed to
combine independent synthetic-media, provenance, metadata and speaker signals into a reviewable
body of evidence, rather than returning a single verdict on whether media is fake.

## Current status

The upload pipeline exists end to end and detection has moved off it. A video can be
submitted, validated, stored, described and queued for analysis, and the resulting analysis is
listed in the dashboard. What runs today:

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
- `GET /api/v1/analyses` returns the most recent analyses with their signal and its strongest
  clips;
- the Next.js dashboard renders that list beside the connectivity status.

No provenance, risk classification or reporting functionality is implemented yet. Signals
carry a provider score but no risk level; deciding what a score means is a later phase.

## Architecture

```text
Next.js (web)
   ↓ REST
FastAPI (api)
   ↓
PostgreSQL   MinIO   NVIDIA SVD
```

The browser reaches the API over the published host port. Server-side rendering in the web
container reaches it over the Docker network at `http://api:8000`. The API reaches MinIO over
the Docker network at `minio:9000`, never the published host port. NVIDIA is reached over TLS
gRPC at its hosted endpoint, addressed by function ID.

Uploads return once the media is staged and the work is queued: storage, probing,
normalization and persistence run on the request, detection does not. If storage, probing or
normalization fails, no analysis row and no job are written. Local temp files are always
removed — the runner reads the canonical object back out of MinIO rather than depending on a
file the request left behind. Stored MinIO objects are not deleted on failure, because their
keys are content-addressed and may already be referenced by an earlier analysis of identical
bytes.

Nothing runs the queued jobs yet. The detector code and its evidence handling are in place
and tested; the runner that claims a job, calls NVIDIA and records the result is the next
task, so an upload today stays `queued` and carries no signal.

## Repository structure

```text
apps/api/              FastAPI service
  app/main.py          /health endpoint, router wiring
  app/api/analyses.py  upload pipeline and analysis listing
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

Check that all four containers report healthy:

```bash
docker compose ps
```

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

Detection needs `NVIDIA_API_KEY` and `NVIDIA_SVD_FUNCTION_ID` in `.env`. Missing or wrong
credentials never fail an upload: they are a detector's problem, and the detector no longer
runs on the request at all.

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
docker compose up -d api
```
