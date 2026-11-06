import pytest
import torch
from src.neuro_symbolic.parser import DifferentiableCYKParser


def test_cyk_parser_shapes():
    num_nt = 4
    vocab_size = 4
    parser = DifferentiableCYKParser(num_nt, vocab_size)

    # Batch di 2 sequenze di lunghezza 4
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


def test_cyk_viterbi_parse():
    num_nt = 2
    vocab_size = 2
    parser = DifferentiableCYKParser(num_nt, vocab_size)

    # Impostiamo pesi deterministici: S -> NT_1 NT_1, NT_1 -> 0
    with torch.no_grad():
        parser.rule_logits.fill_(-10.0)
        parser.term_logits.fill_(-10.0)
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
