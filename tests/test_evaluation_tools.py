"""Unit tests for the gate arithmetic of the checkpoint evaluator.

The evaluator lives in datasets/multilingual/tools and its heavy parts need a trained checkpoint, so the
tests here cover only the pure arithmetic that turns decoded rows into gate numbers: the
accuracy bookkeeping, the legality tally, the numeric audit and the registry audit.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS = next(
    (p for p in sorted((REPO_ROOT / "datasets").glob("*/tools")) if (p / "evaluate_model.py").is_file()),
    None,
)
if TOOLS is None:
    raise LookupError("no corpus tools folder carrying evaluate_model.py under datasets")
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import evaluate_model as gates  # noqa: E402

from neuro_symbolic.masking import parse_algebra  # noqa: E402

REGISTRY: Dict[str, Dict[str, Any]] = {
    "IF": {"arity": 2},
    "AT": {"arity": 1},
    "SCHEDULE": {"arity": 1},
    "WEEKDAY_ORD": {"arity": 2},
}

SIMPLE = ["(", "IF", "[", "flag", "on", "]", "[", "pct", "20", "]", ")"]
SAME_SHAPE = ["(", "IF", "[", "flag", "on", "]", "[", "pct", "20", "]", ")"]
OTHER_NUMBER = ["(", "IF", "[", "flag", "on", "]", "[", "pct", "30", "]", ")"]
OTHER_OPERATOR = ["(", "AT", "[", "flag", "on", "]", ")"]


def scored(
    identifier: str,
    generated: Sequence[str],
    reference: Sequence[str],
    language: str = "de",
    domain: str = "agenda",
    logic_family: str = "conditional",
    exact: bool = False,
    exact_shape: bool = False,
    legal: bool = True,
    steps: int = 11,
    ended_at_end: bool = True,
) -> gates.Scored:
    """Build one decoded row with the verdicts the evaluator would have computed."""
    return gates.Scored(
        identifier=identifier,
        language=language,
        domain=domain,
        logic_family=logic_family,
        generated=list(generated),
        reference=list(reference),
        steps=steps,
        ended_at_end=ended_at_end,
        mean_probability=0.5,
        exact=exact,
        exact_shape=exact_shape,
        legal=legal,
        probabilities=[0.5, 0.5],
    )


def test_accuracy_gate_counts_trees_shapes_and_truncation() -> None:
    rows = [
        scored("a", SIMPLE, SIMPLE, exact=True, exact_shape=True),
        scored("b", SAME_SHAPE, SIMPLE, exact=False, exact_shape=True),
        scored("c", OTHER_OPERATOR, SIMPLE, exact=False, exact_shape=False, ended_at_end=False, steps=4),
    ]
    gate = gates.accuracy_gate(rows)
    assert gate["examples"] == 3
    assert gate["exact_trees"] == 1
    assert gate["exact_tree_accuracy"] == pytest.approx(1 / 3)
    assert gate["exact_skeleton_accuracy"] == pytest.approx(2 / 3)
    assert gate["truncated"] == 1
    assert gate["ended_at_end_marker"] == 2
    assert gate["steps"]["minimum"] == 4
    assert gate["steps"]["maximum"] == 11
    assert gate["steps"]["median"] == 11


def test_accuracy_gate_on_an_empty_split_is_zero_and_not_a_crash() -> None:
    gate = gates.accuracy_gate([])
    assert gate["examples"] == 0
    assert gate["exact_tree_accuracy"] == 0.0
    assert gate["steps"]["maximum"] == 0


def test_accuracy_breakdown_groups_by_row_field() -> None:
    rows = [
        scored("a", SIMPLE, SIMPLE, language="de", domain="agenda", exact=True),
        scored("b", SIMPLE, SIMPLE, language="de", domain="travel", exact=False),
        scored("c", SIMPLE, SIMPLE, language="es", domain="travel", exact=True),
        scored("d", SIMPLE, SIMPLE, language="es", domain="travel", exact=True),
    ]
    by_language = gates.accuracy_breakdown(rows, "language")
    assert by_language["de"]["examples"] == 2
    assert by_language["de"]["accuracy"] == pytest.approx(0.5)
    assert by_language["es"]["accuracy"] == pytest.approx(1.0)
    by_domain = gates.accuracy_breakdown(rows, "domain")
    assert by_domain["travel"]["exact_trees"] == 2
    assert list(by_language) == sorted(by_language)


def test_legality_gate_counts_and_reports_illegal_streams() -> None:
    rows = [
        scored("ok", SIMPLE, SIMPLE, legal=True),
        scored("bad", OTHER_OPERATOR, SIMPLE, legal=False),
    ]
    gate = gates.legality_gate(rows)
    assert gate["examples"] == 2
    assert gate["legal_streams"] == 1
    assert gate["legality"] == pytest.approx(0.5)
    assert len(gate["failures"]) == 1
    assert gate["failures"][0].startswith("bad")


def test_numeric_leaves_reads_unit_values_only() -> None:
    tree = parse_algebra("( IF [ pct 20 ] [ clock 07 00 ] )")
    assert gates.numeric_leaves(tree) == {20.0}
    assert gates.numeric_leaves(parse_algebra("( AT [ flag on ] )")) == set()


def test_formal_gate_flags_a_number_the_reference_tree_does_not_have() -> None:
    rows = [scored("keep", SIMPLE, SIMPLE), scored("wrong_number", OTHER_NUMBER, SIMPLE)]
    report = gates.formal_gate(rows, REGISTRY, use_z3=False)
    assert report["trees_audited"] == 2
    assert report["registry_rejections"] == 0
    assert report["numbers_checked"] == 2
    assert report["invented_numbers"] == 1
    assert report["invented_number_examples"][0].startswith("wrong_number")
    assert report["invented_number_rate"] == pytest.approx(0.5)
    assert report["z3"] == "skipped"


def test_formal_gate_counts_registry_rejections_without_auditing_them() -> None:
    rows = [scored("unknown_operator", OTHER_OPERATOR, SIMPLE, legal=True)]
    report = gates.formal_gate(rows, {"IF": {"arity": 2}}, use_z3=False)
    assert report["trees_audited"] == 0
    assert report["registry_rejections"] == 1
    assert "unknown operator AT" in report["registry_rejection_examples"][0]
    assert report["numbers_checked"] == 0


def test_formal_gate_skips_illegal_streams_before_the_audit() -> None:
    rows = [scored("broken_stream", ["(", "IF"], ["(", "IF", "[", "flag", "on", "]", ")"], legal=False)]
    report = gates.formal_gate(rows, REGISTRY, use_z3=False)
    assert report["trees_audited"] == 0
    assert report["invented_numbers"] == 0


def test_missing_checkpoint_is_an_error_and_not_a_traceback() -> None:
    missing = Path(__file__).resolve().parent / "no_such_checkpoint.pt"
    with pytest.raises(gates.EvaluationError):
        gates.main(["--checkpoint", str(missing)])
