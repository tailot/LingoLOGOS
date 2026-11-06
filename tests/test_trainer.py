import pytest
import torch
from src.neuro_symbolic.parser import DifferentiableCYKParser
from src.neuro_symbolic.trainer import DyckDataset, GrammarTrainer


def test_dyck_dataset_generation():
    ds = DyckDataset.generate_dyck2(num_samples=25, max_depth=3, max_length=12, seed=42)
    assert len(ds) == 25
    for s in ds.samples:
        assert len(s) % 2 == 0
        assert all(0 <= tok < 4 for tok in s)


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
    assert "perplexity" in eval_res
    assert "parsing_f1" in eval_res
    assert eval_res["parsing_f1"] >= 0.90
