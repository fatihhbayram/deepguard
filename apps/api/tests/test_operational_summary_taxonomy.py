"""R13-T2: the operational summary never counts a legacy level in the list of verdicts.

`analyses.risk_level` is one column written in two vocabularies — the `r9-v5.0.0` verdicts and
the `HIGH`/`MEDIUM`/`UNKNOWN` levels of every earlier ruleset — and `/api/v1/admin/analytics`
groups on that column alone. The admin summary used to draw the result as a single list, so a
`MEDIUM` sat among the verdicts as if it answered the same question. The page now draws it
through `splitRiskDistribution` as two lists, and this file holds the split to four things:

* a verdict is counted only in the decisions list, and a level only in the recorded-risk list;
* nothing is translated: a `HIGH` never raises `MANIPULATION_DETECTED`, a `MEDIUM` never
  raises `INCONCLUSIVE`;
* every count the API sent comes out exactly once, unchanged — an unrecognised value included;
* the detail cards label a legacy level as the recorded risk level, chosen by ruleset.

The split is executed rather than pattern-matched, for the reason `test_risk_trace.py` executes
its resolvers: the failure guarded against is a key on the wrong branch, and a source-text
assertion would pass just as happily with the branches swapped.

Frontend only. Nothing here touches the API, the schema or the risk engine.
"""

import json
import subprocess
from functools import lru_cache
from pathlib import Path
from shutil import which

import pytest

WEB_ROOT = Path(__file__).resolve().parents[2] / "web"
WEB_ANALYSIS = WEB_ROOT / "app" / "analysis.ts"
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

# Written out rather than imported from the module under test, so a verdict dropped from
# `V5_VERDICTS` fails here instead of being agreed with.
VERDICTS = ("MANIPULATION_DETECTED", "NO_CALIBRATED_MANIPULATION_SIGNAL", "INCONCLUSIVE")
DECISION_KEYS = {*VERDICTS, "UNDECIDED"}
LEGACY_LEVELS = {"HIGH", "MEDIUM", "UNKNOWN"}

# Each case is a `risk_distribution` as the API would send it.
CASES = {
    # The packet's QA case. The column holds one value per row, so "a v5 decision beside a
    # `MEDIUM`" is two rows in the week: one `r9-v5.0.0` row that wrote `MANIPULATION_DETECTED`
    # and one earlier row that wrote `MEDIUM`, with the API's zero floors for the rest.
    "qa": {
        "MANIPULATION_DETECTED": 1,
        "MEDIUM": 1,
        "HIGH": 0,
        "UNKNOWN": 0,
        "UNDECIDED": 0,
    },
    # Legacy levels only: no verdict may appear from them.
    "legacy_only": {"HIGH": 5, "MEDIUM": 3, "UNKNOWN": 2, "UNDECIDED": 0},
    # A quiet week, and a payload with nothing in it at all.
    "empty": {},
    # Every vocabulary at once, plus two values no build wrote.
    "mixed": {
        "MANIPULATION_DETECTED": 4,
        "NO_CALIBRATED_MANIPULATION_SIGNAL": 7,
        "INCONCLUSIVE": 2,
        "HIGH": 9,
        "MEDIUM": 6,
        "UNKNOWN": 1,
        "UNDECIDED": 3,
        "FAKE": 1,
        "LOW": 8,
    },
}


@lru_cache(maxsize=1)
def _split() -> dict[str, dict[str, dict[str, int]]]:
    """Run the real `splitRiskDistribution` over every case, in one node process.

    `analytics.ts` is transpiled and executed with its `../analysis` import resolved to the
    real, also-transpiled `analysis.ts`, so the allowlists under test are the ones the page
    uses rather than copies. Every other import is stubbed: nothing reachable from the split
    reads a session, a header or the network.
    """
    driver = """
        const fs = require("fs");
        const ts = require(process.argv[1]);
        const load = (path, resolve) => {
            const compiled = ts.transpileModule(
                fs.readFileSync(path, "utf8"),
                { compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2020 } }
            ).outputText;
            const module_ = { exports: {} };
            new Function("exports", "module", "require", compiled)(
                module_.exports, module_, resolve
            );
            return module_.exports;
        };
        const stub = new Proxy({}, { get: () => () => undefined });
        const analysis = load(process.argv[2], () => stub);
        const analytics = load(
            process.argv[3], (name) => (name === "../analysis" ? analysis : stub)
        );
        const cases = JSON.parse(process.argv[4]);
        const out = {};
        for (const [name, counts] of Object.entries(cases)) {
            out[name] = analytics.splitRiskDistribution(counts);
        }
        console.log(JSON.stringify(out));
    """

    result = subprocess.run(
        [
            NODE,
            "-e",
            driver,
            "--",
            str(TYPESCRIPT),
            str(WEB_ANALYSIS),
            str(WEB_ANALYTICS),
            json.dumps(CASES),
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )

    if result.returncode != 0:
        raise RuntimeError(f"node exited {result.returncode}:\n{result.stderr.strip()}")

    return json.loads(result.stdout)


@requires_node
def test_the_qa_case_raises_only_the_verdict_in_the_decisions_list():
    """`MANIPULATION_DETECTED` +1 among the decisions; `MEDIUM` nowhere in that list."""
    split = _split()["qa"]

    assert split["decisions"]["MANIPULATION_DETECTED"] == 1
    assert "MEDIUM" not in split["decisions"]
    # The level is still counted — once, and only in its own list.
    assert split["recordedRiskLevels"]["MEDIUM"] == 1
    assert "MANIPULATION_DETECTED" not in split["recordedRiskLevels"]
    assert split["unrecognised"] == {}


@requires_node
def test_a_legacy_level_is_never_counted_as_a_verdict():
    """No `HIGH` becomes a detection and no `MEDIUM` becomes an `INCONCLUSIVE` (R9-T1 inv. 4)."""
    split = _split()["legacy_only"]

    for verdict in VERDICTS:
        assert split["decisions"][verdict] == 0, verdict
    assert split["recordedRiskLevels"] == {"HIGH": 5, "MEDIUM": 3, "UNKNOWN": 2}


@pytest.mark.parametrize("case", sorted(CASES))
@requires_node
def test_the_lists_hold_only_their_own_vocabulary(case):
    split = _split()[case]

    assert set(split["decisions"]) == DECISION_KEYS
    assert set(split["recordedRiskLevels"]) == LEGACY_LEVELS
    assert not set(split["unrecognised"]) & (DECISION_KEYS | LEGACY_LEVELS)


@pytest.mark.parametrize("case", sorted(CASES))
@requires_node
def test_every_count_the_api_sent_comes_out_once_and_unchanged(case):
    """A partition, not a transformation: no count is dropped, merged, summed or invented."""
    split = _split()[case]
    lists = (split["decisions"], split["recordedRiskLevels"], split["unrecognised"])

    for key, value in CASES[case].items():
        holders = [counts for counts in lists if key in counts]
        assert len(holders) == 1, key
        assert holders[0][key] == value, key

    # Anything the API did not send is a zero floor, never a figure.
    for counts in lists:
        for key, value in counts.items():
            if key not in CASES[case]:
                assert value == 0, key


@requires_node
def test_an_unrecognised_value_is_kept_under_its_own_name():
    split = _split()["mixed"]

    assert split["unrecognised"] == {"FAKE": 1, "LOW": 8}
    assert split["decisions"]["UNDECIDED"] == 3


@requires_web
def test_the_summary_page_draws_the_split_and_never_the_raw_distribution():
    source = WEB_ANALYTICS_PAGE.read_text(encoding="utf-8")

    assert "splitRiskDistribution(analytics.risk_distribution)" in source
    assert "counts={risk.decisions}" in source
    assert "counts={risk.recordedRiskLevels}" in source
    assert "counts={risk.unrecognised}" in source
    # The mixed list is gone, not merely joined by two more.
    assert "counts={analytics.risk_distribution}" not in source
    assert 'title="Recorded risk level"' in source


@requires_web
def test_the_split_reads_the_shared_allowlists_and_maps_nothing():
    """No second copy of either vocabulary, and no lookup from one into the other."""
    source = WEB_ANALYTICS.read_text(encoding="utf-8")
    body = source.split("export function splitRiskDistribution(", 1)[1].split("\n}\n", 1)[0]

    assert "isV5Verdict(key)" in body
    assert "isSupportedRiskLevel(key)" in body
    for verdict in VERDICTS:
        assert verdict not in body, verdict
    for level in LEGACY_LEVELS:
        assert f'"{level}"' not in body, level


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
