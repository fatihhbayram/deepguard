"""Offline detector benchmark framework (R2-T1).

This package is deliberately *not* part of the running product. It is a CLI-only
evaluation harness: it opens no database connection, imports nothing from
`apps/api/app`, and never touches the analysis-job worker. A benchmark run reads a
local manifest, calls a model, and writes files. That separation is the point —
R2's rule is that a detector is measured before it is allowed anywhere near the Risk
Engine, and a harness that shared runtime state with the product could not produce
an honest measurement of a model the product has not adopted yet.

Modules:

- `dataset` — parses a ground-truth manifest into validated clips.
- `metrics` — turns predictions and ground truth into accuracy/precision/recall/FPR/FNR.
- `cli`     — the entrypoint that runs a model over a dataset and writes the artifacts.

Standard library only. No numpy, no pandas, no scikit-learn: the whole metric surface
here is nine integer counters and five divisions, and a dependency that large would be
carried by the API image for arithmetic that fits on one screen.
"""
