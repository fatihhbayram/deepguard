# MVP Data Model Direction

Avoid designing 20–30 tables before implementation.

Initial target: approximately 7 tables.

## Candidate tables

```text
users
analyses
media_files
analysis_jobs
analysis_signals
analysis_segments
reports
```

`api_keys` can be added in P9.

## Analysis

Stores:

- owner
- status
- timestamps
- overall deterministic risk level
- risk-engine version

## Media file

Stores:

- analysis id
- original object key
- normalized object key if created
- SHA-256
- MIME
- size
- duration
- container
- codec
- fps
- resolution

## Analysis job

Stores the detection work an analysis is owed:

```text
analysis_id unique
status queued | processing | completed | failed
error_message nullable
created_at
updated_at
```

One job per analysis, enforced by the unique constraint: a retry re-runs the row rather than
adding a second one, so "has this analysis been detected?" has a single answer.

## Analysis signal

Stores normalized provider evidence:

```text
provider
signal_type
score nullable
logit nullable
risk_level
provider_version
status
metadata JSON
```

## Analysis segment

Stores timeline evidence:

```text
signal_id
frame_index nullable
start_time nullable
end_time nullable
score nullable
logit nullable
risk_level nullable
metadata JSON
```

Do not over-normalize JSON payloads from providers before product requirements prove that relational fields are needed.
