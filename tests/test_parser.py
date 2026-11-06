import math
import pytest
import torch
from neuro_symbolic.parser import DifferentiableCYKParser


def test_cyk_parser_shapes():
    num_nt = 4
    vocab_size = 4
    parser = DifferentiableCYKParser(num_nt, vocab_size)

    # Batch of 2 sequences of length 4
    x = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]], dtype=torch.long)
    log_probs = parser(x)

    assert log_probs.shape == (2,)
    assert not torch.isnan(log_probs).any()
    assert not torch.isinf(log_probs).any()


def test_cyk_differentiability():
    num_nt = 3
    vocab_size = 3
    parser = DifferentiableCYKParser(num_nt, vocab_size)

    x = torch.tensor([[0, 1]], dtype=torch.long)
    log_prob = parser(x)
    loss = -log_prob.mean() + 0.1 * parser.rule_entropy()
    loss.backward()

    assert parser.rule_logits.grad is not None
    assert parser.term_logits.grad is not None
    assert not torch.isnan(parser.rule_logits.grad).any()
    assert not torch.isnan(parser.term_logits.grad).any()
    # The family gate must receive gradient, otherwise the joint
    # normalization would never be learned
    assert parser.family_logits.grad is not None
    assert not torch.isnan(parser.family_logits.grad).any()


def test_pcfg_mass_invariant_per_lhs():
    """
    Distribution invariant: for every non-terminal A,
        sum_{B,C} P(A -> B C) + sum_a P(A -> a) = 1.
    Historically the model normalized the two production families separately,
    getting mass 2 for every LHS: the measured perplexity was therefore an artifact
    (the marginal string probabilities were inflated).
    """
    parser = DifferentiableCYKParser(num_non_terminals=5, vocab_size=4)
    with torch.no_grad():
        parser.rule_logits.normal_(0.0, 1.0)
        parser.term_logits.normal_(0.0, 1.0)
        parser.family_logits.normal_(0.0, 1.0)

    masses = parser.production_mass_per_lhs().tolist()
    assert len(masses) == 5
    for mass in masses:
        assert abs(mass - 1.0) < 1e-5, f"mass per LHS = {mass}, expected 1.0"


def test_log_prob_is_a_real_distribution_over_strings():
    """
    With the degenerate grammar in which S produces a single terminal with probability 1,
    the inside probability of the corresponding string must be 1 (log 0), not 2.
    """
    parser = DifferentiableCYKParser(num_non_terminals=1, vocab_size=2)
    with torch.no_grad():
        parser.rule_logits.fill_(-20.0)
        parser.term_logits.fill_(-20.0)
        parser.term_logits[0, 0] = 20.0
        parser.family_logits.data.fill_(-20.0)  # terminal family active

    x = torch.tensor([[0]], dtype=torch.long)
    log_prob = parser(x).item()
    assert abs(log_prob) < 1e-4, f"log P(0) = {log_prob}, expected ~0"

    # An impossible token must pay exactly the cost of the terminal softmax (-40 nats)
    x_bad = torch.tensor([[1]], dtype=torch.long)
    bad = parser(x_bad).item()
    assert bad == pytest.approx(-40.0, abs=1e-3), f"log P(1) = {bad}"


def test_entropy_penalty_is_bounded_and_non_negative():
    parser = DifferentiableCYKParser(num_non_terminals=4, vocab_size=4)
    entropy = parser.rule_entropy().item()
    assert entropy >= 0.0
    # Upper bound: for every LHS h <= ln2 (gate) + ln|R| + ln|V|
    upper_bound = 4 * (math.log(2) + math.log(16) + math.log(4))
    assert entropy <= upper_bound + 1e-6

    with torch.no_grad():
        parser.rule_logits.fill_(-20.0)
        parser.term_logits.fill_(-20.0)
        parser.family_logits.data.copy_(torch.tensor([8.0, -20.0, -20.0, -20.0]))
        parser.rule_logits[0, 1, 1] = 20.0
        parser.term_logits[1, 0] = 20.0
        parser.term_logits[2, 0] = 20.0
        parser.term_logits[3, 0] = 20.0
    assert parser.rule_entropy().item() < 0.1


def test_cyk_viterbi_parse():
    num_nt = 2
    vocab_size = 2
    parser = DifferentiableCYKParser(num_nt, vocab_size)

    # Set deterministic weights: S -> NT_1 NT_1, NT_1 -> 0
    with torch.no_grad():
        parser.rule_logits.fill_(-10.0)
        parser.term_logits.fill_(-10.0)
        parser.family_logits.data.copy_(torch.tensor([4.0, -4.0]))
        # S (0) -> NT_1 (1) NT_1 (1)
        parser.rule_logits[0, 1, 1] = 5.0
        # NT_1 (1) -> '0' (0)
        parser.term_logits[1, 0] = 5.0

    ast, score = parser.parse_viterbi([0, 0])
    assert ast is not None
    assert ast["type"] == "S"
    assert len(ast["children"]) == 2
    assert ast["children"][0]["type"] == "NT_1"
    assert ast["children"][0]["token"] == "0"


def test_viterbi_matches_inside_for_unambiguous_grammar():
    """
    On a degenerate grammar (only one possible derivation) Viterbi and inside must
    coincide: this is the cross-check guaranteeing that the two CYK variants use the same
    span indexing.
    """
    parser = DifferentiableCYKParser(num_non_terminals=2, vocab_size=2)
    with torch.no_grad():
        parser.rule_logits.fill_(-20.0)
        parser.term_logits.fill_(-20.0)
        parser.family_logits.data.copy_(torch.tensor([4.0, -4.0]))
        parser.rule_logits[0, 1, 1] = 5.0
        parser.term_logits[1, 0] = 3.0
        parser.term_logits[1, 1] = 2.0

    x = torch.tensor([[0, 1]], dtype=torch.long)
    inside = parser(x).item()
    _, viterbi_score = parser.parse_viterbi([0, 1])
    assert abs(inside - viterbi_score) < 1e-4
