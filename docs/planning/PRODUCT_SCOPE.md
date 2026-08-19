# Product Scope

## Positioning

DeepGuard is not a generic consumer "fake/real scanner".

DeepGuard is:

> A B2B Media Authenticity & Forensics Infrastructure. It is an evidence-oriented platform that orchestrates independent synthetic-media, provenance, metadata, and speaker signals.

DeepGuard does not average detector scores into a single "87% fake" probability. Instead, it aggregates multiple independent forensic signals and provides explainable evidence, suspicious timelines, and deterministic risk levels.

## Differentiation / Moat

1. Multi-signal evidence orchestration
2. Provenance-first approach (especially C2PA)
3. Explainable suspicious timeline / segment evidence
4. API-first B2B integration
5. Enterprise privacy / future on-prem deployment readiness
6. Provider-independent architecture (without premature abstraction)

## Initial customer direction

B2B-first.

Potential high-value verticals:

- fraud investigation
- marketplace / classified listing fraud
- insurance fraud
- identity / onboarding review
- enterprise impersonation
- trust & safety / moderation
- media authenticity workflows

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
