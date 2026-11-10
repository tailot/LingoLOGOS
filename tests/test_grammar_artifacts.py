"""Staleness gate for the committed grammar artifacts under `artifacts/`.

That folder carries the deliverables of the Python grammar track: the induced grammar in JSON and
YAML, the Grammatical Framework module, the trained weights and the verification report. No library
module reads them at runtime, so before this file nothing in the suite noticed when they drifted
away from the code that writes them, while `AGENTS.md` makes their validity part of the protocol and
gate 3 of the ASIC suite reads `grammar.json` to compare the automaton alphabet of the chip with the
terminal set of the grammar. This file runs the synthesis into a temporary folder and compares the
formal outputs with the committed copies.

Floating point results are deliberately not compared byte by byte: two synthesis runs with the
same seed already differed in the eighth significant digit of the reported losses on one host, so
`grammar_weights.pt` and the numeric metrics are only checked through the gates they record, while
the exported grammar is exact and is therefore compared exactly.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
COMMITTED_ARTIFACTS = REPO_ROOT / "artifacts"
SYNTHESIS_SCRIPT = REPO_ROOT / "src" / "neuro_symbolic" / "synthesize.py"
SYNTHESIS_TIMEOUT_SECONDS = 600.0

GRAMMAR_TERMINALS = ["(", ")", "[", "]"]


def _normalized_text(path: Path) -> str:
    """Return the file text with line endings normalized, so a checkout cannot change the verdict."""
    return path.read_text(encoding="utf-8").replace("\r\n", "\n")


def _assert_report_is_sound(report: dict, label: str) -> None:
    """Assert the four synthesis gates, the UNSAT certificate and the coverage of the corpus."""
    gates = report["gates"]
    for name in sorted(gates):
        assert gates[name] is True, f"{label} reports gate {name} as false"
    assert report["soundness"]["status"] == "UNSAT", f"{label} does not end with an UNSAT certificate"
    assert report["soundness"]["sound"] is True, f"{label} marks the certificate as not sound"
    assert report["language"]["corpus_coverage"] == 1.0, f"{label} does not cover the whole corpus"
    assert report["grammar"]["valid_pcfg"] is True, f"{label} does not hold a valid probabilistic grammar"


@pytest.fixture(scope="session")
def regenerated_artifacts(tmp_path_factory):
    """Run the synthesis once per session into a temporary folder outside the checkout."""
    out_dir = tmp_path_factory.mktemp("grammar_artifacts")
    command = [sys.executable, str(SYNTHESIS_SCRIPT), "--artifacts-dir", str(out_dir)]
    try:
        completed = subprocess.run(
            command,
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=SYNTHESIS_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        pytest.fail(f"synthesis did not finish in {SYNTHESIS_TIMEOUT_SECONDS} seconds")
    assert completed.returncode == 0, "synthesis failed: " + completed.stderr[-500:]
    return out_dir


def test_committed_artifacts_are_all_present() -> None:
    """The five deliverables of the grammar track must exist in the checkout."""
    expected = (
        "grammar.json",
        "grammar.yaml",
        "DynamicGrammarIta.gf",
        "grammar_weights.pt",
        "metrics.json",
    )
    missing = [name for name in expected if not (COMMITTED_ARTIFACTS / name).is_file()]
    assert not missing, "missing committed artifacts: " + ", ".join(missing)


def test_committed_report_records_a_verified_synthesis() -> None:
    """The committed report must state that the grammar is valid, covers the corpus and is sound."""
    report = json.loads((COMMITTED_ARTIFACTS / "metrics.json").read_text(encoding="utf-8"))
    _assert_report_is_sound(report, "the committed metrics report")


def test_committed_grammar_carries_the_bracket_alphabet() -> None:
    """The grammar JSON is the source gate 3 of the ASIC suite reads for the automaton alphabet."""
    grammar = json.loads((COMMITTED_ARTIFACTS / "grammar.json").read_text(encoding="utf-8"))
    metadata = grammar["grammar_metadata"]
    assert metadata["terminals"] == GRAMMAR_TERMINALS, "the grammar terminal set is not the bracket alphabet"
    assert grammar["production_rules"], "the committed grammar carries no production rule"


def test_regenerated_report_passes_the_same_gates(regenerated_artifacts: Path) -> None:
    """A synthesis run from the current code must end with the same verdict as the committed report."""
    report = json.loads((regenerated_artifacts / "metrics.json").read_text(encoding="utf-8"))
    _assert_report_is_sound(report, "the regenerated metrics report")


def test_regenerated_grammar_matches_the_committed_grammar(regenerated_artifacts: Path) -> None:
    """The formal outputs of the current code must equal the committed ones."""
    json_committed = json.loads((COMMITTED_ARTIFACTS / "grammar.json").read_text(encoding="utf-8"))
    json_regenerated = json.loads((regenerated_artifacts / "grammar.json").read_text(encoding="utf-8"))
    assert json_regenerated == json_committed, "artifacts/grammar.json is stale with respect to the code"

    for name in ("grammar.yaml", "DynamicGrammarIta.gf"):
        committed = _normalized_text(COMMITTED_ARTIFACTS / name)
        regenerated = _normalized_text(regenerated_artifacts / name)
        assert committed == regenerated, f"artifacts/{name} is stale with respect to the code"
