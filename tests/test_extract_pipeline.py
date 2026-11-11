"""Unit and integration tests for dataset extraction pipeline in extract/."""

import json
from pathlib import Path

import pytest

from extract.d1_gf_ast import D1Generator, LANGUAGES
from extract.d2_smt_consistency import D2Generator
from extract.d3_command_execution import D3Generator, DOMAINS
from extract.d4_syntactic_robustness import D4Generator
from extract.d5_logit_masking import D5Generator
from extract.pipeline import export_dataset


def test_d1_multilingual_distribution_and_depth():
    gen = D1Generator(seed=42, max_depth=6)
    samples = [gen.generate_sample(i) for i in range(100)]

    seen_langs = set(s.language for s in samples)
    assert seen_langs == set(LANGUAGES)

    for s in samples:
        assert s.ast_depth <= 6
        assert len(s.tokens) > 0
        assert s.ast_tree.startswith("PhrUtt MarkP")
        d = s.to_dict()
        assert d["dataset_id"] == "D1_GF_RGL"


def test_d2_smt_50_50_balance_and_z3():
    gen = D2Generator(seed=42, validate_with_z3=True)
    samples = [gen.generate_sample(i) for i in range(100)]

    sat_count = sum(1 for s in samples if s.expected_satisfiability_result == "SAT")
    unsat_count = sum(1 for s in samples if s.expected_satisfiability_result == "UNSAT")

    assert sat_count == 50
    assert unsat_count == 50

    for s in samples:
        assert s.verification_passed is True
        assert "(check-sat)" in s.smt_assertions_block


def test_d3_10_domains_partition_and_agents():
    gen = D3Generator(seed=42)
    samples = [gen.generate_sample(i) for i in range(100)]

    domains_seen = set(s.domain for s in samples)
    assert len(domains_seen) == 10
    assert domains_seen == set(DOMAINS)

    agents_seen = set(s.target_agent for s in samples)
    assert agents_seen == {"agente_alfa", "agente_beta", "arbitro_logico"}

    for s in samples:
        assert "routing_header" in s.execution_payload
        assert "volatile_context" in s.execution_payload


def test_d4_syntactic_robustness_dyck2_and_noise_tags():
    gen = D4Generator(seed=42)
    samples = [gen.generate_sample(i) for i in range(100)]

    dyck2_count = sum(1 for s in samples if s.is_dyck2)
    assert dyck2_count == 50

    tags = set(s.error_type_tag for s in samples)
    assert "OOV_TOKEN" in tags
    assert "DELIMITER_IMBALANCE" in tags

    for s in samples:
        assert len(s.corrupted_sequence) > 0
        assert len(s.corrected_sequence) > 0


def test_d5_logit_masking_bitset():
    gen = D5Generator(seed=42)
    samples = [gen.generate_sample(i) for i in range(100)]

    for s in samples:
        assert len(s.active_mask_bitset) == gen.vocab_size
        assert "1" in s.active_mask_bitset
        assert len(s.next_valid_tokens) > 0
        assert len(s.next_valid_token_ids) == len(s.next_valid_tokens)


def test_deterministic_reproducibility():
    gen1 = D1Generator(seed=42)
    gen2 = D1Generator(seed=42)
    s1 = gen1.generate_sample(7)
    s2 = gen2.generate_sample(7)
    assert s1.to_dict() == s2.to_dict()


def test_pipeline_export(tmp_path: Path):
    out_file = export_dataset(
        dataset_name="D1",
        num_samples=20,
        seed=42,
        output_dir=tmp_path,
        fmt="jsonl",
    )
    assert out_file.exists()
    lines = out_file.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 20
    first_record = json.loads(lines[0])
    assert first_record["dataset_id"] == "D1_GF_RGL"
