"""Replay the production risk rules over benchmark runs, changing nothing.

    PYTHONPATH=scripts python3 scripts/eval/replay.py \
        --corpus ../deepguard-corpus/r7t5/corpus.json \
        --lip-run ../deepguard-corpus/runs/r7t5-lip/results.json \
        --face-run ../deepguard-corpus/runs/r7t5-face/results.json \
        --svd-run ../deepguard-corpus/runs/r7t5-svd/results.json \
        --output ../deepguard-corpus/r7t5/replay.json

**The rules are imported, not reimplemented.** `apps/api/app/risk_engine.py` is loaded from
source by path and its `evaluate` is called with its own evidence types. A reimplementation
would be a second copy of the thing under observation, and the first time the two drifted this
report would be describing a rule set that never shipped. The module imports only `math` and
`dataclasses`, so loading it pulls in no FastAPI app, no session, no worker and no configuration
— which is what makes importing it from an offline script legitimate rather than a shortcut.

**Nothing is written back.** No database, no storage, no file under `apps/`. The thresholds, the
rule ids and the calibration id come out of the production module and are recorded in the output
exactly as found, so a reader can confirm which rule set produced these observations.

**Evidence is assembled to match what production would have persisted**, field for field: the
provider and signal type the worker writes, the `provider_version` string `app.detection` builds
from the model's own identity, the status, the score, and the degeneracy count. The version
strings are *derived from each run's observed provenance and then compared* against the
production constants rather than copied from them — if a run was produced by a different
deployment or checkpoint, the honest replay is one where the rules find that signal ineligible,
and `eligibility` records why.

A clip the harness excluded (an abstention, a decode failure) contributes a `FAILED` signal, not
a missing one and not a zero. That distinction is the difference between `R201` and a fabricated
negative, and it is the whole reason the counted wrappers exist.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.corpus import SPLIT_EVALUATION, CorpusItem, read_corpus

# Where the rules live. Loaded by path so this script needs neither the api package on
# PYTHONPATH nor its dependencies installed.
RISK_ENGINE_PATH = (
    Path(__file__).resolve().parents[2] / "apps" / "api" / "app" / "risk_engine.py"
)


def load_risk_engine(path: Path = RISK_ENGINE_PATH) -> ModuleType:
    """Load the production rules from source, read-only, as their own module."""
    spec = importlib.util.spec_from_file_location("deepguard_risk_engine_readonly", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load the risk engine from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_run(path: Path | None) -> dict:
    """One benchmark `results.json`, or an empty stand-in when a detector was not run.

    A detector that was not run is silent, which is a state the rules already handle. It is not
    the same as a detector that ran and abstained, and the output keeps them apart.
    """
    if path is None:
        return {"clips": {}, "provenance": {}, "ran": False}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "clips": {clip["clip_id"]: clip for clip in payload["clips"]},
        "provenance": payload["run"].get("model_provenance") or {},
        "ran": True,
        "results_path": str(path),
    }


def lip_provider_version(provenance: dict) -> str | None:
    """Rebuild the `provider_version` string `app.detection` writes for this detector."""
    upstream = provenance.get("upstream") or {}
    classifier = provenance.get("classifier") or {}
    repository, revision = upstream.get("repository"), upstream.get("revision")
    digest = classifier.get("sha256")
    if not (repository and revision and digest):
        return None
    return f"{repository}@{revision}+{digest}"


def face_provider_version(provenance: dict) -> str | None:
    """The same, for the face classifier: model repository at the revision that was loaded."""
    classifier = provenance.get("classifier") or {}
    artifact, revision = classifier.get("artifact"), classifier.get("revision")
    if not (artifact and revision):
        return None
    return f"{artifact.rsplit('/', 1)[0]}@{revision}"


def svd_provider_version(provenance: dict) -> str | None:
    """The NVCF deployment that actually answered, which is the only version handle there is."""
    observed = provenance.get("observed_function_ids") or []
    if len(observed) == 1:
        return observed[0]
    return None


def _status(record: dict | None, engine: ModuleType) -> str | None:
    """`SUCCESS` only for a clip the detector actually scored; `FAILED` for an abstention."""
    if record is None:
        return None
    return engine.SIGNAL_STATUS_SUCCESS if record.get("status") == "ok" else "FAILED"


def build_evidence(
    clip_id: str,
    engine: ModuleType,
    svd_run: dict,
    face_run: dict,
    lip_run: dict,
    versions: dict[str, str | None],
):
    """The three persisted signals for one clip, as the database would hold them."""
    svd_record = svd_run["clips"].get(clip_id)
    face_record = face_run["clips"].get(clip_id)
    lip_record = lip_run["clips"].get(clip_id)

    svd = (
        engine.SvdEvidence(
            provider=engine.SVD_PROVIDER,
            signal_type=engine.SVD_SIGNAL_TYPE,
            status=_status(svd_record, engine),
            provider_version=versions["svd"],
            score=svd_record.get("score"),
            total_clips=(svd_run["provenance"].get("total_clips_by_clip_id") or {}).get(
                clip_id
            ),
        )
        if svd_record is not None
        else None
    )
    face = (
        engine.FaceEvidence(
            provider=engine.FACE_PROVIDER,
            signal_type=engine.FACE_SIGNAL_TYPE,
            status=_status(face_record, engine),
            provider_version=versions["face"],
            score=face_record.get("score"),
            frames_scored=(face_run["provenance"].get("frames_scored_by_clip_id") or {}).get(
                clip_id
            ),
        )
        if face_record is not None
        else None
    )
    lip = (
        engine.LipEvidence(
            provider=engine.LIP_PROVIDER,
            signal_type=engine.LIP_SIGNAL_TYPE,
            status=_status(lip_record, engine),
            provider_version=versions["lip"],
            score=lip_record.get("score"),
            windows_scored=(lip_run["provenance"].get("windows_scored_by_clip_id") or {}).get(
                clip_id
            ),
        )
        if lip_record is not None
        else None
    )
    return svd, face, lip


def responsible_detectors(engine: ModuleType, svd, face, lip) -> list[str]:
    """Which detectors reached their own threshold, for attributing a HIGH to its source.

    Recomputed from the same eligibility and usability predicates the rules use, not guessed
    from the rule id: `R102` names no detector, and a genuine HIGH that nobody can attribute is
    a finding nobody can act on.
    """
    flagged = []
    if svd is not None and engine.is_eligible_svd(svd) and engine.is_usable_svd(svd):
        if svd.score >= engine.SVD_T_HIGH:
            flagged.append(engine.SVD_PROVIDER)
    if face is not None and engine.is_eligible_face(face) and engine.is_usable_face(face):
        if face.score >= engine.FACE_T_HIGH:
            flagged.append(engine.FACE_PROVIDER)
    if lip is not None and engine.is_eligible_lip(lip) and engine.is_usable_lip(lip):
        if lip.score >= engine.LIP_T_HIGH:
            flagged.append(engine.LIP_PROVIDER)
    return flagged


def replay(
    items: list[CorpusItem],
    engine: ModuleType,
    svd_run: dict,
    face_run: dict,
    lip_run: dict,
) -> dict:
    """Classify every clip in the corpus under the current rules and record the trace."""
    versions = {
        "svd": svd_provider_version(svd_run["provenance"]),
        "face": face_provider_version(face_run["provenance"]),
        "lip": lip_provider_version(lip_run["provenance"]),
    }
    expected = {
        "svd": engine.SVD_PROVIDER_VERSION,
        "face": engine.FACE_PROVIDER_VERSION,
        "lip": engine.LIP_PROVIDER_VERSION,
    }
    eligibility = {
        name: {
            "observed_provider_version": versions[name],
            "production_provider_version": expected[name],
            "matches_calibrated_deployment": versions[name] == expected[name],
        }
        for name in ("svd", "face", "lip")
    }

    decisions = []
    for item in items:
        svd, face, lip = build_evidence(
            item.clip_id, engine, svd_run, face_run, lip_run, versions
        )
        decision = engine.evaluate(svd=svd, face=face, lip=lip)
        decisions.append(
            {
                "clip_id": item.clip_id,
                "split": item.split,
                "label": item.label,
                "family": item.family,
                "stratum": item.stratum_primary,
                "source_lineage_id": item.source_lineage_id,
                "derivation": item.derivation,
                "private": item.private,
                "risk_level": decision.risk_level,
                "rule_id": decision.rule_id,
                "rules_version": decision.rules_version,
                "calibration_id": decision.calibration_id,
                "responsible_detectors": responsible_detectors(engine, svd, face, lip),
                "scores": {
                    "svd": svd.score if svd else None,
                    "face": face.score if face else None,
                    "lip": lip.score if lip else None,
                },
                "counts": {
                    "svd_total_clips": svd.total_clips if svd else None,
                    "face_frames_scored": face.frames_scored if face else None,
                    "lip_windows_scored": lip.windows_scored if lip else None,
                },
                "statuses": {
                    "svd": svd.status if svd else None,
                    "face": face.status if face else None,
                    "lip": lip.status if lip else None,
                },
            }
        )

    return {
        "schema_version": "r7-t5-replay-1",
        "replayed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rules": {
            "rules_version": engine.RULES_VERSION,
            "calibration_id": engine.CALIBRATION_ID,
            "svd_t_high": engine.SVD_T_HIGH,
            "face_t_high": engine.FACE_T_HIGH,
            "lip_t_high": engine.LIP_T_HIGH,
            "source": str(RISK_ENGINE_PATH),
            "modified_by_this_task": False,
        },
        "runs": {
            "svd": {"ran": svd_run["ran"], "path": svd_run.get("results_path")},
            "face": {"ran": face_run["ran"], "path": face_run.get("results_path")},
            "lip": {"ran": lip_run["ran"], "path": lip_run.get("results_path")},
        },
        "eligibility": eligibility,
        "rule_counts": dict(sorted(Counter(d["rule_id"] for d in decisions).items())),
        "level_counts": dict(sorted(Counter(d["risk_level"] for d in decisions).items())),
        "decisions": decisions,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="replay",
        description="Replay the current v3 risk rules offline over a scored corpus.",
    )
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--lip-run", type=Path)
    parser.add_argument("--face-run", type=Path)
    parser.add_argument("--svd-run", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    engine = load_risk_engine()
    items = read_corpus(args.corpus)
    report = replay(
        items,
        engine,
        read_run(args.svd_run),
        read_run(args.face_run),
        read_run(args.lip_run),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"rules {report['rules']['rules_version']} "
          f"calibration {report['rules']['calibration_id'][:12]}…")
    for name, entry in report["eligibility"].items():
        state = "calibrated" if entry["matches_calibrated_deployment"] else "NOT CALIBRATED"
        print(f"  {name}: {state}")
    print(f"levels: {report['level_counts']}")
    print(f"rules fired: {report['rule_counts']}")
    evaluation = [d for d in report["decisions"] if d["split"] == SPLIT_EVALUATION]
    genuine_high = [
        d for d in evaluation if d["label"] == "real" and d["risk_level"] == "HIGH"
    ]
    print(f"evaluation split: {len(evaluation)} clips, {len(genuine_high)} genuine HIGH")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
