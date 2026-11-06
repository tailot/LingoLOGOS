"""
Tests of the formal verification.

Two tests in particular are regression tests for bugs that were fixed:
  * `test_single_string_check_is_not_vacuous`: the old SMT check put the prefix
    invariant AMONG THE PREMISES, so for strings like `())` or `)(` P was inconsistent
    and P ^ !G came out trivially UNSAT (a false "soundness 100%").
  * `test_overgenerating_grammar_is_refuted`: a grammar that generates `(]` must be
    REFUTED (SAT with a witness), not declared sound.
"""

import pytest
from neuro_symbolic.grammar import PCFGrammar, ProductionRule, Symbol
from neuro_symbolic.reference import gold_dyck2_grammar, generate_negative_controls
from neuro_symbolic.verification import (
    GrammarVerifier,
    counter_signature,
    is_well_bracketed,
)


def build_grammar(nt, terminals, triples):
    """Builds a CNF PCFG out of (lhs, rhs, probability)."""
    nts = {name: Symbol(name=name) for name in nt}
    ts = {name: Symbol(name=name, is_terminal=True) for name in terminals}
    rules = [
        ProductionRule(
            lhs=nts[lhs],
            rhs=[ts[name] if name in ts else nts[name] for name in rhs],
            probability=prob,
        )
        for lhs, rhs, prob in triples
    ]
    return PCFGrammar(list(nt), list(terminals), "S", rules)


# --------------------------------------------------------------------------- #
# Soundness certificate on the gold grammar
# --------------------------------------------------------------------------- #
def test_gold_grammar_certificate_is_unsat():
    grammar = gold_dyck2_grammar()
    report = GrammarVerifier(grammar).verify_language_soundness(search_bound=5)
    assert report.status == "UNSAT", str(report)
    assert report.sound is True
    assert report.counterexample is None
    # The certificate must give the start symbol null deltas and non-negative minima
    d_r, d_s, m_r, m_s = report.certificate["S"]
    assert (d_r, d_s) == (0, 0)
    assert m_r >= 0 and m_s >= 0


def test_gold_grammar_passes_all_structural_checks():
    grammar = gold_dyck2_grammar()
    verifier = GrammarVerifier(grammar)

    ok, err = verifier.verify_lark_compilation()
    assert ok, err

    valid, masses = verifier.verify_pcfg_distribution()
    assert valid, masses

    corpus = ["()", "[]", "(())", "()[]", "[()]", "(()[])[]", "[(())]", "[[]]"]
    coverage, failed = verifier.verify_corpus_coverage(corpus)
    assert coverage == 1.0, failed


def test_gold_grammar_rejects_all_negative_controls():
    corpus = ["()", "[]", "(())", "()[]", "[()]", "(()[])[]"]
    negatives = generate_negative_controls(corpus, max_controls=100, seed=3)
    assert len(negatives) > 20
    rejection, accepted_wrong = GrammarVerifier(gold_dyck2_grammar()).verify_negative_controls(negatives)
    assert rejection == 1.0, f"wrongly accepted: {accepted_wrong[:10]}"


# --------------------------------------------------------------------------- #
# NEGATIVE TEST: a deliberately overgenerating grammar must be REFUTED
# --------------------------------------------------------------------------- #
def overgenerating_grammar():
    """Same structure as the gold grammar but NT_2 emits both `)` and `]`: it generates `(]`."""
    return build_grammar(
        ["S", "NT_1", "NT_2", "NT_3"],
        ["(", ")", "[", "]"],
        [
            ("S", ["NT_1", "NT_2"], 0.55),
            ("S", ["NT_3", "S"], 0.28),
            ("S", ["S", "S"], 0.04),
            ("S", ["NT_3", "NT_2"], 0.03),
            ("NT_1", ["("], 1.0),
            ("NT_2", [")"], 0.58),
            ("NT_2", ["]"], 0.42),
            ("NT_3", ["["], 1.0),
        ],
    )


def test_overgenerating_grammar_is_refuted():
    verifier = GrammarVerifier(overgenerating_grammar())
    report = verifier.verify_language_soundness(search_bound=5)
    assert report.status == "SAT", str(report)
    assert report.sound is False
    assert report.counterexample is not None
    # The witness must be a string genuinely outside Dyck-2
    assert not is_well_bracketed(list(report.counterexample))
    # ... and it must really be derivable: the refutation is constructive
    assert verifier.grammar.derive(list(report.counterexample)) is not None


def test_overgenerating_grammar_rejected_by_negative_controls():
    negatives = ["(]", "([", "][", "())", "((", "]]"]
    rejection, accepted_wrong = GrammarVerifier(overgenerating_grammar()).verify_negative_controls(negatives)
    assert rejection < 1.0
    assert "(]" in accepted_wrong


# --------------------------------------------------------------------------- #
# Regression test: the check on a single string must not be vacuous
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("word", ["()", "[]", "(())", "[]()", "[()]"])
def test_single_string_check_valid_on_well_formed(word):
    sound, status, model = GrammarVerifier.check_word_balance_z3(list(word))
    assert sound is True, status
    assert status == "VALID"


@pytest.mark.parametrize("word", ["())", ")(", "][", "((", "[(]", "(()[]"])
def test_single_string_check_is_not_vacuous(word):
    """The old encoding answered 'UNSAT / sound' even on `())` and `)(`."""
    sound, status, model = GrammarVerifier.check_word_balance_z3(list(word))
    assert sound is False, f"expected INVALID for {word!r}, got {status}"
    assert status == "INVALID"
    assert model is not None


def test_counter_signature_matches_bracket_semantics():
    assert counter_signature(list("(())")) == (0, 0, 0, 0)
    assert counter_signature(list("())")) == (-1, 0, -1, 0)
    # `][` closes before it opens: the minimum of the square counter is -1, the final one is 0
    assert counter_signature(list("][")) == (0, 0, 0, -1)
    assert counter_signature(list("]]")) == (0, -2, 0, -2)
    assert is_well_bracketed(list("()[]")) is True
    assert is_well_bracketed(list("())(")) is False


# --------------------------------------------------------------------------- #
# Distribution invariant: an unnormalized grammar must fail the check
# --------------------------------------------------------------------------- #
def test_distribution_check_flags_unnormalized_grammar():
    bad = build_grammar(
        ["S"],
        ["(", ")"],
        [("S", ["(", ")"], 1.0), ("S", ["("], 0.92)],
    )
    valid, masses = GrammarVerifier(bad).verify_pcfg_distribution()
    assert valid is False
    assert masses["S"] == pytest.approx(1.92, abs=1e-6)


def test_certificate_unsound_when_nt_generates_incompatible_signatures():
    """
    A non-terminal that generates both `()` (delta 0) and `((` (delta 2) cannot have a
    functional signature: the certificate fails and the grammar must be refuted with a witness.
    """
    grammar = build_grammar(
        ["S", "A"],
        ["(", ")"],
        [
            ("S", ["A", "A"], 1.0),
            ("A", [")"], 0.5),
            ("A", ["("], 0.5),
        ],
    )
    report = GrammarVerifier(grammar).verify_language_soundness(search_bound=4)
    assert report.status in ("SAT", "INCONCLUSIVE")
    assert report.sound is False
