"""R13-T2/T3: the operational summary never counts a legacy level in the list of verdicts.

`analyses.risk_level` is one column written in two vocabularies — the `r9-v5.0.0` verdicts and
the `HIGH`/`MEDIUM`/`UNKNOWN` levels of every earlier ruleset — and which one a row holds is
named by the `risk_rules_version` stored beside it. R13-T2 split the summary into two lists in
the browser, by guessing each key's vocabulary from its spelling alone. R13-T3 moved the split
to the API, which groups on both columns and returns three maps already placed
(`test_admin_analytics.py` holds that bucketing against real rows). This file holds the web half
of the contract:

* the page draws the API's three maps as they arrive, and the old single map is gone;
* `analytics.ts` no longer contains any split of its own — no allowlist, no `splitRiskDistribution`;
* the three maps are parsed through unchanged, and a payload missing any of them is refused;
* the detail cards label a legacy level as the recorded risk level, chosen by ruleset.

The parse is executed rather than pattern-matched, for the reason `test_risk_trace.py` executes
its resolvers: the failure guarded against is a map read from the wrong field, and a source-text
assertion would pass just as happily with two fields swapped.

Frontend only. Nothing here touches the API, the schema or the risk engine.
"""

import json
import subprocess
from functools import lru_cache
from pathlib import Path
from shutil import which

import pytest

WEB_ROOT = Path(__file__).resolve().parents[2] / "web"
WEB_ANALYTICS = WEB_ROOT / "app" / "admin" / "analytics.ts"
WEB_ANALYTICS_PAGE = WEB_ROOT / "app" / "admin" / "analytics" / "page.tsx"
WEB_ADMIN_ANALYSIS = WEB_ROOT / "app" / "admin" / "analyses" / "[id]" / "page.tsx"
WEB_REPORT = WEB_ROOT / "app" / "app" / "report" / "[id]" / "page.tsx"

NODE = "node"
TYPESCRIPT = WEB_ROOT / "node_modules" / "typescript"

requires_web = pytest.mark.skipif(
    not WEB_ROOT.exists(), reason="the web application is not present"
)

requires_node = pytest.mark.skipif(
    not (WEB_ANALYTICS.exists() and TYPESCRIPT.exists() and which(NODE) is not None),
    reason="node and the web application's TypeScript are needed to run the split",
)

# A payload as `/api/v1/admin/analytics` sends it, with every risk bucket holding a distinct
# count so a map read from the wrong field cannot pass by coincidence.
PAYLOAD = {
    "window": "7d",
    "analyses_total": 31,
    "analyses_by_status": {"queued": 0, "completed": 30, "failed": 1},
    "jobs_by_status": {"queued": 0, "processing": 0, "completed": 30, "failed": 1},
    "decisions": {
        "MANIPULATION_DETECTED": 4,
        "NO_CALIBRATED_MANIPULATION_SIGNAL": 7,
        "INCONCLUSIVE": 2,
        "UNDECIDED": 3,
    },
    "recorded_risk_levels": {"HIGH": 9, "MEDIUM": 6, "UNKNOWN": 1},
    "unrecognised": {"r9-v5.0.0/MEDIUM": 5, "r7-v4.0.0/MANIPULATION_DETECTED": 8},
    "acquisition": {"upload": 30, "url": 1, "unrecorded": 0},
    "detectors": {},
}

RISK_FIELDS = ("decisions", "recorded_risk_levels", "unrecognised")

CASES = {
    "full": PAYLOAD,
    # The pre-R13-T3 contract: one mixed map and none of the three. Refused, not half-drawn.
    "legacy_contract": {
        **{k: v for k, v in PAYLOAD.items() if k not in RISK_FIELDS},
        "risk_distribution": {"HIGH": 9, "MANIPULATION_DETECTED": 4},
    },
    **{
        f"missing_{field}": {k: v for k, v in PAYLOAD.items() if k != field}
        for field in RISK_FIELDS
    },
}


@lru_cache(maxsize=1)
def _parsed() -> dict[str, dict | None]:
    """Run the real `parseAnalytics` over every case, in one node process.

    `analytics.ts` is transpiled and executed with every import stubbed: nothing reachable from
    the parse reads a session, a header or the network.
    """
    driver = """
        const fs = require("fs");
        const ts = require(process.argv[1]);
        const compiled = ts.transpileModule(
            fs.readFileSync(process.argv[2], "utf8"),
            { compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2020 } }
        ).outputText;
        const stub = new Proxy({}, { get: () => () => undefined });
        const module_ = { exports: {} };
        new Function("exports", "module", "require", compiled)(
            module_.exports, module_, () => stub
        );
        const cases = JSON.parse(process.argv[3]);
        const out = {};
        for (const [name, payload] of Object.entries(cases)) {
            out[name] = module_.exports.parseAnalytics(payload);
        }
        console.log(JSON.stringify(out));
    """

    result = subprocess.run(
        [NODE, "-e", driver, "--", str(TYPESCRIPT), str(WEB_ANALYTICS), json.dumps(CASES)],
        capture_output=True,
        text=True,
        timeout=120,
    )

    if result.returncode != 0:
        raise RuntimeError(f"node exited {result.returncode}:\n{result.stderr.strip()}")

    return json.loads(result.stdout)


@requires_node
def test_the_three_risk_maps_are_parsed_through_unchanged():
    """A partition the API made, carried as it arrived: no key moved, dropped or added."""
    parsed = _parsed()["full"]

    assert parsed["decisions"] == PAYLOAD["decisions"]
    assert parsed["recordedRiskLevels"] == PAYLOAD["recorded_risk_levels"]
    assert parsed["unrecognised"] == PAYLOAD["unrecognised"]
    assert "risk_distribution" not in parsed


@pytest.mark.parametrize(
    "case", ["legacy_contract", *(f"missing_{field}" for field in RISK_FIELDS)]
)
@requires_node
def test_a_payload_without_all_three_maps_is_refused(case):
    assert _parsed()[case] is None


@requires_web
def test_the_summary_page_draws_the_api_maps_directly():
    source = WEB_ANALYTICS_PAGE.read_text(encoding="utf-8")

    assert "counts={analytics.decisions}" in source
    assert "counts={analytics.recordedRiskLevels}" in source
    assert "counts={analytics.unrecognised}" in source
    assert "risk_distribution" not in source
    assert "splitRiskDistribution" not in source
    assert 'title="Recorded risk level"' in source


@requires_web
def test_the_web_no_longer_guesses_a_vocabulary_from_a_key():
    """The split lives in the API now; no second copy of it may survive in the browser."""
    source = WEB_ANALYTICS.read_text(encoding="utf-8")

    assert "splitRiskDistribution" not in source
    assert "risk_distribution" not in source
    for name in ("isV5Verdict", "isSupportedRiskLevel", "V5_VERDICTS", "SUPPORTED_RISK_LEVELS"):
        assert name not in source, name


@requires_web
def test_the_admin_card_labels_a_legacy_level_by_ruleset():
    source = WEB_ADMIN_ANALYSIS.read_text(encoding="utf-8")

    assert (
        "const legacyLevel = level !== null && analysis.risk_rules_version !== RULES_VERSION_V5;"
        in source
    )
    assert '<Fact label={legacyLevel ? "Recorded risk level" : "Decision"}>' in source


@requires_web
def test_the_report_names_a_legacy_level_as_the_recorded_risk_level():
    section = WEB_REPORT.read_text(encoding="utf-8").split(
        "function RiskSection(", 1
    )[1].split("\nfunction ", 1)[0]

    assert "const legacyVocabulary = rulesVersion !== RULES_VERSION_V5;" in section
    assert 'legacyVocabulary && level !== null\n          ? "Recorded risk level"' in section
