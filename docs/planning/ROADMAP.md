# DeepGuard MVP Roadmap

## Product goal

Build a B2B Media Authenticity & Forensics Infrastructure that produces evidence-oriented analysis rather than a simplistic "real/fake" answer.


Primary flow:

```text
UPLOAD
   ↓
VALIDATE
   ↓
STORE ORIGINAL
   ↓
FFPROBE
   ↓
NORMALIZE
   ↓
ANALYZE
   ↓
NORMALIZE SIGNALS
   ↓
RISK ASSESSMENT
   ↓
REPORT
```

---

# P0 — Foundation

## Objective

Create the smallest runnable technical foundation.

## Scope

- monorepo
- Next.js app
- FastAPI app
- PostgreSQL
- MinIO
- Docker Compose
- minimal configuration
- health endpoints
- web → API → DB validation

## Exit criteria

- local stack starts reliably;
- frontend can reach API;
- API can reach PostgreSQL;
- MinIO is reachable;
- baseline lint/type/test commands exist;
- no detector logic yet.

## Explicitly out of scope

- NVIDIA
- C2PA
- jobs
- reports
- auth complexity
- public API keys

---

# P1 — Upload Pipeline

## Objective

Accept a video safely and create a persistent analysis record.

## Flow

```text
upload
  ↓
validate MIME / size
  ↓
SHA-256
  ↓
store original
  ↓
ffprobe
  ↓
provider-compatible normalization
  ↓
persist metadata
```

## Exit criteria

- supported video upload works;
- invalid media is rejected;
- SHA-256 stored;
- original stored in MinIO/S3;
- basic ffprobe metadata stored;
- provider-compatible derivative generated (if needed) and stored;
- uploaded analysis visible in dashboard.

---

# P2 — NVIDIA Synthetic Video

## Objective

Produce the first real forensic signal.

## Flow

```text
stored video
  ↓
normalize if required
  ↓
NVIDIA Synthetic Video Detector
  ↓
normalize result
  ↓
persist signals/segments
  ↓
display timeline/evidence
```

## Exit criteria

- NVIDIA client works;
- timeout/error states handled;
- result stored;
- suspicious segment/timeline data displayed;
- provider version captured;
- provider failure does not corrupt the analysis record.

At the end of P2 the first meaningful product demo exists.

---

# P3 — Async Job Processing

## Objective

Move video analysis out of synchronous request execution.

## State model

```text
queued
processing
completed
failed
```

## Approach

Use PostgreSQL-backed jobs and a Python worker first.

## Exit criteria

- create-analysis returns quickly;
- worker processes jobs;
- duplicate job execution is prevented or safely handled;
- failed jobs are observable;
- no Redis/Celery.

---

# P4 — C2PA + Metadata

## Objective

Add provenance and container/codec evidence.

## Signals

- C2PA manifest present/not present
- signature validity
- provenance information
- codec
- container
- fps
- resolution
- relevant timestamps/metadata

## Important semantic rule

No C2PA credentials does **not** imply fake media.

## Exit criteria

- C2PA result stored independently;
- relevant metadata visible;
- report clearly distinguishes `NOT FOUND` from `INVALID`.

---

# P5 — NVIDIA Active Speaker

## Objective

Add speaker/face temporal evidence.

## Scope

- active speaker integration;
- speaker segments;
- confidence;
- timeline visualization.

## Important semantic rule

Active Speaker Detection is not itself a deepfake detector.

Treat it as a cross-modal forensic signal.

---

# P6 — Audio / Multimodal Detector

## Objective

Add a separate audio or multimodal synthetic-media detector.

## Scope

- one provider only initially;
- independent signal persistence;
- graceful provider failure;
- timeline data where available.

## Exit criteria

Video and audio/multimodal signals can coexist without forcing a fake combined percentage.

---

# P7 — Risk Engine

## Objective

Turn independent evidence into a deterministic product-level risk classification.

## Output

```text
LOW
MEDIUM
HIGH
UNKNOWN
```

## Rules

- deterministic;
- explainable;
- evidence-first;
- version the risk rules;
- never imply certainty.

## ChatGPT involvement

Architecture review is required before implementation because this phase changes interpretation semantics.

---

# P8 — Reporting

## Objective

Produce a useful forensic report.

## First target

HTML report.

## Then

PDF export if needed.

## Report contents

- file hash
- analysis timestamp
- media metadata
- provider/model versions
- independent signals
- suspicious segments
- C2PA/provenance
- risk classification
- limitations/disclaimer

---

# P9 — Public API

## Objective

Expose the stabilized analysis workflow to B2B customers.

## Scope

- API keys
- usage limits
- REST endpoints
- JSON analysis/report output

## Defer

- SDKs
- webhooks
- GraphQL
- billing
- organization hierarchy

until real customer demand exists.

---

# P10 — URL Ingestion

## Objective

Expand media ingestion beyond direct file uploads to support public media URLs.

## Scope

- YouTube ingestion
- TikTok / Instagram / Social media URLs
- Publicly accessible media URLs
- Automatic downloading and temporary storage

## Constraints

- P1 data model must not be speculatively complicated to support URL ingestion.
- Implemented strictly when URL ingestion becomes the immediate priority.
