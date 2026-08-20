# DeepGuard

DeepGuard is an evidence-oriented content authenticity and forensics platform. It is designed to
combine independent synthetic-media, provenance, metadata and speaker signals into a reviewable
body of evidence, rather than returning a single verdict on whether media is fake.

## Current status

The upload pipeline exists end to end and produces its first real forensic signal. A video can
be submitted, validated, stored, described, analysed by NVIDIA's synthetic video detector and
persisted, and the resulting analysis is listed in the dashboard. What runs today:

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
- whichever artifact is provider-compatible — the derivative when one was made, the original
  otherwise — is sent to NVIDIA's Synthetic Video Detector over gRPC while it is still on
  local disk;
- NVIDIA's probability is recorded as returned, on NVIDIA's own scale, as one signal among
  the several a full analysis will eventually carry;
- a detector that fails, times out or is misconfigured does not fail the upload: the failure
  is itself persisted as a signal, so the gap in the evidence is visible rather than silent —
  a fault on this server's own side is not treated that way and still surfaces as an error;
- the clips NVIDIA scored inside the video are kept as evidence in their own right, capped at
  the strongest twenty per detection, each with the provider's frame index and raw logit;
- the completed analysis, its media facts, that signal and its clip evidence are written to
  PostgreSQL in one transaction, after detection has finished — no transaction is held open
  across the call;
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

Uploads are handled synchronously: the request returns only after storage, probing,
normalization, detection and persistence have all run. If storage, probing or normalization
fails, no analysis row is written. Detection is different — it is evidence gathering, not
admission control, so its failure is recorded rather than fatal. Local temp files are always
removed; stored MinIO objects are not deleted on
failure, because their keys are content-addressed and may already be referenced by an
earlier analysis of identical bytes.

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

The response carries the analysis id, the original's SHA-256 and storage key, the ffprobe
metadata, `was_normalized`, and the derivative's storage key and SHA-256. When the original
is already provider-compatible, no derivative is produced: `was_normalized` is `false`,
`derivative_storage_key` is the original's key and `derivative_sha256` is `null`.

The NVIDIA signal is not in that response — it is evidence attached to the analysis, returned
by `GET /api/v1/analyses` and shown on the dashboard. It can also be read directly:

```sql
SELECT provider, signal_type, status, score, metadata FROM analysis_signals;

SELECT clip_index, logit FROM analysis_segments ORDER BY logit DESC;
```

A segment is one clip the detector scored. NVIDIA reports a frame index and a raw logit per
clip and no timestamps, so those two figures are what is stored — the logit is not a
probability and is not comparable with the signal's `score`.

Detection needs `NVIDIA_API_KEY` and `NVIDIA_SVD_FUNCTION_ID` in `.env`. Without them the
upload still succeeds and the analysis records a `FAILED` signal instead of a score.

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
