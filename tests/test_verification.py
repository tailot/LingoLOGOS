import pytest
from src.neuro_symbolic.grammar import PCFGrammar, Symbol, ProductionRule
from src.neuro_symbolic.verification import GrammarVerifier


def test_lark_syntax_verification():
    s = Symbol("S")
    nt1 = Symbol("NT_1")
    lp = Symbol("(", is_terminal=True)
    rp = Symbol(")", is_terminal=True)

    rules = [
        ProductionRule(lhs=s, rhs=[nt1, nt1], probability=0.9),
        ProductionRule(lhs=nt1, rhs=[lp], probability=0.5),
        ProductionRule(lhs=nt1, rhs=[rp], probability=0.5),
    ]

    grammar = PCFGrammar(
        non_terminals=["S", "NT_1"],
        terminals=["(", ")"],
        start_symbol="S",
        rules=rules,
    )

    verifier = GrammarVerifier(grammar)
    ok, err = verifier.verify_lark_syntax()
    assert ok, f"Lark compilation failed: {err}"


def test_smt_soundness_unsat():
    # AST bilanciato: S -> NT_1('(') NT_2(')')
    ast_balanced = {
        "type": "S",
        "children": [
            {"type": "NT_1", "token": "("},
            {"type": "NT_2", "token": ")"},
        ],
    }

    sound, status, model = GrammarVerifier.verify_smt_soundness(ast_balanced, expected_balanced=True)
    assert sound is True
    assert status == "UNSAT"


def test_smt_soundness_counterexample_sat():
    # AST sbilanciato: S -> '(' '('
    ast_unbalanced = {
        "type": "S",
        "children": [
            {"type": "NT_1", "token": "("},
            {"type": "NT_2", "token": "("},
        ],
    }

    sound, status, model = GrammarVerifier.verify_smt_soundness(ast_unbalanced, expected_balanced=True)
    # P and not G deve risultare SAT con controesempio perché G non è soddisfatto
    assert sound is False
    assert status == "SAT"
    assert model is not None
