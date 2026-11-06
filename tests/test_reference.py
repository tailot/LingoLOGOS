"""Tests of the formal references (gold Dyck-2 CNF, negative controls, gold rule F1)."""

import pytest
from neuro_symbolic.reference import (
    GOLD_NON_TERMINALS,
    GOLD_RULE_TRIPLES,
    generate_negative_controls,
    gold_dyck2_grammar,
    gold_rule_f1,
    is_not_in_dyck,
)
from neuro_symbolic.verification import is_well_bracketed


def test_gold_grammar_is_consistent_with_triples():
    grammar = gold_dyck2_grammar()
    assert grammar.start_symbol == "S"
    assert len(grammar.rules) == len(GOLD_RULE_TRIPLES)
    assert set(grammar.non_terminals) == set(GOLD_NON_TERMINALS)


def test_gold_grammar_language_matches_dyck2_membership():
    grammar = gold_dyck2_grammar()
    for word in ["()", "[]", "(())", "(()[])", "[(())[]]"]:
        assert grammar.derive(list(word)) is not None, f"gold should derive {word}"
    for word in ["(]", "([", "][", "())", ")(", "((", "]]"]:
        assert grammar.derive(list(word)) is None, f"gold should not derive {word}"


def test_negative_controls_are_really_outside_dyck2():
    corpus = ["()", "[]", "(())", "(()[])[]"]
    negatives = generate_negative_controls(corpus, max_controls=500, seed=11)
    assert len(negatives) > 30
    for word in negatives:
        assert is_not_in_dyck(word) is True, word
        assert is_well_bracketed(list(word)) is False, word


def test_gold_rule_f1_is_one_on_reference_grammar():
    precision, recall, f1, details = gold_rule_f1(gold_dyck2_grammar())
    assert (precision, recall, f1) == pytest.approx((1.0, 1.0, 1.0))
    assert details["matched"] == len(GOLD_RULE_TRIPLES)


def test_gold_rule_f1_penalizes_spurious_rules():
    """A grammar with one extra rule must not report F1 = 1."""
    from neuro_symbolic.grammar import ProductionRule, Symbol

    grammar = gold_dyck2_grammar()
    extra_lhs = Symbol(name="CLS_R")
    extra_rhs = [Symbol(name="(", is_terminal=True)]
    grammar.rules.append(ProductionRule(lhs=extra_lhs, rhs=extra_rhs, probability=0.1))
    for rule in grammar.rules:
        if rule.lhs.name == "CLS_R":
            rule.probability = 0.45 if rule.rhs[0].name != "(" else 0.1
    precision, recall, f1, _ = gold_rule_f1(grammar)
    assert f1 < 1.0


def test_gold_rule_f1_alignes_renamed_non_terminals():
    """
    Regression: the metric must search for the induced -> gold alignment. An isomorphic copy
    of the gold CNF with renamed non-terminals must score F1 = 1.0; the version with the
    wrong permutations (only over the induced names) returned 0.0.
    """
    from neuro_symbolic.grammar import PCFGrammar, ProductionRule, Symbol
    from neuro_symbolic.reference import GOLD_NON_TERMINALS, GOLD_RULE_TRIPLES, GOLD_START

    rename = {gold: f"X{index}" for index, gold in enumerate(GOLD_NON_TERMINALS)}
    rules = [
        ProductionRule(
            lhs=Symbol(name=rename[lhs]),
            rhs=[
                Symbol(name=name, is_terminal=(name not in rename))
                for name in rhs_names
            ],
            probability=1.0 / 3,
        )
        for lhs, rhs_names in GOLD_RULE_TRIPLES
    ]
    renamed_grammar = PCFGrammar(
        rules=rules,
        non_terminals=[rename[name] for name in GOLD_NON_TERMINALS],
        terminals=["(", ")", "[", "]"],
        start_symbol=rename[GOLD_START],
    )
    precision, recall, f1, details = gold_rule_f1(renamed_grammar)
    assert (precision, recall, f1) == pytest.approx((1.0, 1.0, 1.0))
    assert details["matched"] == len(GOLD_RULE_TRIPLES)
    assert details["spurious"] == [] and details["missing"] == []
