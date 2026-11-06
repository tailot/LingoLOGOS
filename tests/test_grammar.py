import pytest
import os
import json
import yaml
import torch
from src.neuro_symbolic.parser import DifferentiableCYKParser
from src.neuro_symbolic.grammar import PCFGrammar, Symbol, ProductionRule


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

    json_str = grammar.to_json(str(json_file))
    yaml_str = grammar.to_yaml(str(yaml_file))
    gf_str = grammar.to_gf("TestGrammar", str(gf_file))

    # Verifica deserializzazione
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
