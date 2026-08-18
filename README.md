# DeepGuard

DeepGuard is an evidence-oriented content authenticity and forensics platform. It is designed to
combine independent synthetic-media, provenance, metadata and speaker signals into a reviewable
body of evidence, rather than returning a single verdict on whether media is fake.

## Current status

The repository is at the foundation stage. What exists today is the running skeleton of the stack:

- a Next.js page that reports web → API → database connectivity;
- a FastAPI service exposing a single `/health` endpoint;
- a PostgreSQL database;
- a MinIO instance, provisioned in Compose but not yet used by the API.

No analysis, upload, authentication or detector functionality is implemented yet.

## Architecture

```text
Next.js (web)
   ↓ REST
FastAPI (api)
   ↓
PostgreSQL
```

The browser reaches the API over the published host port. Server-side rendering in the web
container reaches it over the Docker network at `http://api:8000`.

## Repository structure

```text
apps/api/          FastAPI service
  app/main.py      /health endpoint
  app/db/          SQLAlchemy engine and session
  alembic/         migration setup (no migrations yet)
  tests/           pytest suite
apps/web/          Next.js application
  app/page.tsx     connectivity status page
docs/planning/     product, API and data-model scope
docker-compose.yml service definitions
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
`OPERATIONAL` when the API and database are both reachable.

## Tests and quality checks

Backend tests use a stub session, so no running database is required:

```bash
cd apps/api
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
pytest
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
