import pytest
import torch
from neuro_symbolic.parser import DifferentiableCYKParser
from neuro_symbolic.trainer import DyckDataset, GrammarTrainer


def test_dyck_dataset_generation():
    ds = DyckDataset.generate_dyck2(num_samples=25, max_depth=3, max_length=12, seed=42)
    assert len(ds) == 25
    for s in ds.samples:
        assert len(s) % 2 == 0
        assert all(0 <= tok < 4 for tok in s)


def test_dataset_split_is_disjoint():
    """
    The train/test split must be over SENTENCES: a test that reuses train sentences
    measures memorization, not generalization.
    """
    ds = DyckDataset.generate_dyck2(num_samples=60, max_depth=3, max_length=12, seed=42)
    train, test = ds.split(test_fraction=0.25, seed=42)
    assert len(train) > 0 and len(test) > 0
    assert len(train) + len(test) == len(ds)
    train_strings, test_strings = set(train.as_strings()), set(test.as_strings())
    assert train_strings.isdisjoint(test_strings)
    assert len(train_strings) == len(train)


def test_as_strings_roundtrip():
    ds = DyckDataset.generate_dyck2(num_samples=8, max_depth=2, max_length=6, seed=7)
    strings = ds.as_strings()
    assert len(strings) == len(ds)
    assert all(set(s) <= set("()[]") for s in strings)


def test_grammar_trainer_step():
    parser = DifferentiableCYKParser(num_non_terminals=4, vocab_size=4)
    trainer = GrammarTrainer(
        parser=parser,
        learning_rate=3.5e-3,
        temperature_tau=0.75,
        entropy_penalty=0.05,
    )
    dataset = DyckDataset.generate_dyck2(num_samples=10, max_depth=2, max_length=6, seed=42)

    stats = trainer.train_epoch(dataset, batch_size=4)
    assert "avg_loss" in stats
    assert "avg_entropy" in stats
    assert "grad_norm" in stats

    eval_res = trainer.evaluate(dataset)
    assert {"mean_nll_per_symbol", "ppl_per_symbol", "model_coverage", "rule_sparsity"} <= set(eval_res)
    assert eval_res["model_coverage"] >= 0.90
    # An untrained parser can have cross-entropy even above log|V| = ln 4:
    # the constraint is that the metric be finite and non-degenerate (no inf / -1e8)
    assert eval_res["mean_nll_per_symbol"] < 50.0
    assert eval_res["ppl_per_symbol"] < 1e20


def test_training_reduces_nll_per_symbol():
    """The objective maximizes P(S =>* X): the cost per symbol must go down with training."""
    torch.manual_seed(0)
    parser = DifferentiableCYKParser(num_non_terminals=5, vocab_size=4)
    trainer = GrammarTrainer(
        parser=parser,
        learning_rate=5.0e-3,
        temperature_tau=0.75,
        entropy_penalty=0.5,
    )
    dataset = DyckDataset.generate_dyck2(num_samples=24, max_depth=3, max_length=10, seed=42)

    first = trainer.evaluate(dataset)["mean_nll_per_symbol"]
    for _ in range(25):
        trainer.train_epoch(dataset, batch_size=8)
    later = trainer.evaluate(dataset)["mean_nll_per_symbol"]
    assert later < first - 0.1, f"nll per symbol not improved: {first:.3f} -> {later:.3f}"


def test_learned_grammar_respects_distribution_invariant():
    torch.manual_seed(0)
    parser = DifferentiableCYKParser(num_non_terminals=5, vocab_size=4)
    trainer = GrammarTrainer(parser=parser, entropy_penalty=0.5)
    dataset = DyckDataset.generate_dyck2(num_samples=12, max_depth=2, max_length=8, seed=42)
    trainer.train_epoch(dataset, batch_size=8)

    eval_res = trainer.evaluate(dataset)
    assert eval_res["max_lhs_mass_error"] < 1e-5
