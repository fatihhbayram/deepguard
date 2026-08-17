# Product Scope

## Positioning

DeepGuard is not:

> "Upload a video and we tell you with certainty whether it is fake."

DeepGuard is:

> An evidence-oriented AI Content Authenticity & Forensics platform that combines independent synthetic-media, provenance, metadata and speaker signals.

## Initial customer direction

B2B-first.

Potential later markets:

- media/news verification
- trust & safety
- marketplace fraud review
- insurance evidence screening
- corporate communications
- moderation workflows

High-impact domains must use DeepGuard as a review/risk signal, not as the sole automatic decision-maker.

## MVP output

For each analysis:

```text
Video Synthetic      HIGH / MEDIUM / LOW / UNKNOWN
Audio Synthetic      HIGH / MEDIUM / LOW / UNKNOWN
Speaker Consistency  NORMAL / SUSPICIOUS / UNKNOWN
C2PA Provenance      VALID / INVALID / NOT FOUND / UNKNOWN
Metadata             NORMAL / SUSPICIOUS / UNKNOWN

Overall Risk         HIGH / MEDIUM / LOW / UNKNOWN
```

## MVP screens

```text
/login
/dashboard
/analyze
/analyses/:id
/settings
```

Most important screen:

`/analyses/:id`

It should prioritize:

1. media preview
2. overall risk classification
3. independent signals
4. suspicious timeline segments
5. evidence/details
6. detector/provider version
