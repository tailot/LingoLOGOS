import pytest
import json
import yaml
import torch
from neuro_symbolic.parser import DifferentiableCYKParser
from neuro_symbolic.grammar import PCFGrammar, Symbol, ProductionRule


def test_grammar_serialization(tmp_path):
    s = Symbol(name="S", is_terminal=False)
    nt1 = Symbol(name="NT_1", is_terminal=False)
    a = Symbol(name="a", is_terminal=True)

    rule1 = ProductionRule(lhs=s, rhs=[nt1, nt1], probability=0.85)
    rule2 = ProductionRule(lhs=nt1, rhs=[a], probability=0.99)

    grammar = PCFGrammar(
        non_terminals=["S", "NT_1"],
        terminals=["a"],
        start_symbol="S",
        rules=[rule1, rule2],
    )

    json_file = tmp_path / "test_grammar.json"
    yaml_file = tmp_path / "test_grammar.yaml"
    gf_file = tmp_path / "test_grammar.gf"

    grammar.to_json(str(json_file))
    grammar.to_yaml(str(yaml_file))
    gf_str = grammar.to_gf("TestGrammar", str(gf_file))

    # Check the deserialization
    with open(json_file) as f:
        data_json = json.load(f)
    assert data_json["grammar_metadata"]["start_symbol"] == "S"
    assert len(data_json["production_rules"]) == 2

    with open(yaml_file) as f:
        data_yaml = yaml.safe_load(f)
    assert len(data_yaml["production_rules"]) == 2

    assert "abstract TestGrammar" in gf_str
    assert "concrete TestGrammarEng of TestGrammar" in gf_str


def test_grammar_pruning():
    parser = DifferentiableCYKParser(num_non_terminals=2, vocab_size=2)
    with torch.no_grad():
        parser.rule_logits.fill_(-20.0)
        parser.term_logits.fill_(-20.0)
        parser.rule_logits[0, 1, 1] = 10.0
        parser.term_logits[1, 0] = 10.0

    grammar = PCFGrammar.from_model(
        parser,
        nt_labels=["S", "NT_1"],
        vocab_labels=["(", ")"],
        start_symbol="S",
        prune_threshold=0.01,
    )

    assert len(grammar.rules) >= 2
    rule_strs = [r.to_string() for r in grammar.rules]
    assert "S -> NT_1 NT_1" in rule_strs
    assert "NT_1 -> '('" in rule_strs


def test_pruning_renormalizes_mass_per_lhs():
    """
    After the tau pruning the probability sum for every non-terminal must stay 1
    (the PCFG distribution invariant). Without renormalization the exported grammar
    would not be a valid PCFG and the perplexity would not be comparable.
    """
    parser = DifferentiableCYKParser(num_non_terminals=3, vocab_size=4)
    with torch.no_grad():
        parser.rule_logits.normal_(0.0, 0.5)
        parser.term_logits.normal_(0.0, 0.5)
        # S is structural, NT_1 and NT_2 produce terminals
        parser.family_logits.data.copy_(torch.tensor([4.0, -4.0, -4.0]))

    grammar = PCFGrammar.from_model(
        parser,
        nt_labels=["S", "NT_1", "NT_2"],
        vocab_labels=["(", ")", "[", "]"],
        start_symbol="S",
        prune_threshold=0.05,
        renormalize=True,
    )

    valid, masses = grammar.is_valid_pcfg(tolerance=1e-6)
    assert valid, f"masses per LHS not normalized: {masses}"
    assert len(grammar.rules) > 0


def test_pruning_keeps_raw_probability_for_audit():
    """The pre-pruning probability must stay traceable to inspect the pruning."""
    parser = DifferentiableCYKParser(num_non_terminals=2, vocab_size=2)
    with torch.no_grad():
        parser.rule_logits.fill_(0.0)
        parser.term_logits.fill_(-20.0)
        parser.rule_logits[0, 1, 1] = 2.0
        parser.family_logits.data.copy_(torch.tensor([4.0, -4.0]))

    raw_grammar = PCFGrammar.from_model(
        parser,
        nt_labels=["S", "NT_1"],
        vocab_labels=["(", ")"],
        start_symbol="S",
        prune_threshold=0.0,
        temperature=0.75,
        renormalize=False,
    )
    grammar = PCFGrammar.from_model(
        parser,
        nt_labels=["S", "NT_1"],
        vocab_labels=["(", ")"],
        start_symbol="S",
        prune_threshold=0.05,
        temperature=0.75,
        renormalize=True,
    )

    assert grammar.rules and raw_grammar.rules
    raw_lookup = {
        (r.lhs.name, tuple(str(s) for s in r.rhs)): r.probability for r in raw_grammar.rules
    }
    for rule in grammar.rules:
        key = (rule.lhs.name, tuple(str(s) for s in rule.rhs))
        assert rule.raw_probability == pytest.approx(raw_lookup[key], rel=1e-6)
        # The post-pruning renormalization can only increase the probability
        assert rule.probability >= rule.raw_probability - 1e-9


def test_derive_returns_tree_or_none():
    """Discrete CYK: the balanced string is derivable, the malformed one is not."""
    s = Symbol("S")
    opn = Symbol("OPN")
    cls = Symbol("CLS")
    lp = Symbol("(", is_terminal=True)
    rp = Symbol(")", is_terminal=True)

    grammar = PCFGrammar(
        non_terminals=["S", "OPN", "CLS"],
        terminals=["(", ")"],
        start_symbol="S",
        rules=[
            ProductionRule(lhs=s, rhs=[s, s], probability=0.25),
            ProductionRule(lhs=s, rhs=[opn, cls], probability=0.75),
            ProductionRule(lhs=cls, rhs=[s, cls], probability=0.5),
            ProductionRule(lhs=opn, rhs=[lp], probability=1.0),
            ProductionRule(lhs=cls, rhs=[rp], probability=1.0),
        ],
    )

    tree = grammar.derive(list("(())"))
    assert tree is not None and tree["type"] == "S"
    assert grammar.derive(list("((")) is None
    assert grammar.derive(list(")(")) is None
