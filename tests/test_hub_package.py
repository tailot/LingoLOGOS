"""Tests for the safetensors release format and the hub packaging tool.

Three layers are covered: the safetensors serialization of the model, the JSON
specification of the algebra mask, and the packaging tool in
``datasets/multilingual/tools`` that assembles the release around a gate report. No test here
reaches the network: uploading is exercised only through the package builder.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS = next(
    (p for p in sorted((REPO_ROOT / "datasets").glob("*/tools")) if (p / "publish_to_hub.py").is_file()),
    None,
)
if TOOLS is None:
    raise LookupError("no corpus tools folder carrying publish_to_hub.py under datasets")
for entry in (str(REPO_ROOT), str(REPO_ROOT / "src"), str(TOOLS)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from neuro_symbolic.masking import AlgebraMask, AlgebraMaskError  # noqa: E402
from neuro_symbolic.model import (  # noqa: E402
    FORMAT_NAME,
    EncoderDecoderModel,
    ModelConfig,
    ModelError,
    apply_initialization,
    load_model,
    load_model_safetensors,
    read_model_safetensors_header,
    save_model,
    save_model_safetensors,
)
from neuro_symbolic.tokenizer import Tokenizer  # noqa: E402

import publish_to_hub as hub  # noqa: E402

ARITIES = {"IF": 2, "AT": 1, "SCHEDULE": 1, "WEEKDAY_ORD": 2}
LEAF_PAYLOADS = [["flag"], ["pct", "50"], ["day", "mon"], ["ord", "3"]]


def tiny_config(vocabulary_size: int) -> ModelConfig:
    return ModelConfig(
        vocabulary_size=vocabulary_size,
        d_model=32,
        encoder_layers=2,
        decoder_layers=2,
        attention_heads=4,
        feed_forward_size=64,
        max_encoder_positions=16,
        max_decoder_positions=16,
    )


def tiny_checkpoint(path: Path, vocabulary_size: int, tokenizer_path: str) -> None:
    config = tiny_config(vocabulary_size)
    model = EncoderDecoderModel(config, padding_id=0)
    apply_initialization(model, seed=5)
    save_model(
        model,
        str(path),
        metadata={
            "external_weights": "none",
            "tokenizer": tokenizer_path,
            "vocabulary_size": vocabulary_size,
            "trained_at_epoch": 2,
            "best_validation_loss": 0.25,
            "parameters": {"total": 0, "trainable": 0},
            "settings": {"epochs": 2, "lr": 0.001, "batch_size": 8, "clip": 1.0, "weight_decay": 0.01, "amp": "off"},
            "hardware": {"name": "test device"},
        },
    )


def gate_report(path: Path, verdict: str = "PASS") -> str:
    report = {
        "final_verdict": verdict,
        "split": "test",
        "examples": 40,
        "decoding": {"mode": "greedy", "seed": 7},
        "gates": {
            "E1": {
                "exact_tree_accuracy": 0.75,
                "exact_trees": 30,
                "exact_skeleton_accuracy": 1.0,
                "truncated": 0,
                "ended_at_end_marker": 40,
                "mean_selected_probability": 0.98,
                "steps": {"mean": 21.5, "minimum": 9, "maximum": 33, "median": 20},
                "by_language": {"de": {"accuracy": 0.7, "exact_trees": 14, "examples": 20}},
                "by_domain": {"agenda": {"accuracy": 0.75, "exact_trees": 30, "examples": 40}},
                "by_logic_family": {
                    "threshold": {"accuracy": 0.5, "exact_trees": 10, "examples": 20},
                    "recurrence": {"accuracy": 1.0, "exact_trees": 20, "examples": 20},
                },
            },
            "E2": {"legality": 1.0, "legal_streams": 40, "examples": 40, "failures": []},
            "E3": {
                "trees_audited": 40,
                "numbers_checked": 61,
                "registry_rejections": 0,
                "invented_numbers": 1,
                "invented_number_rate": 0.0164,
                "z3": {
                    "forms": 40,
                    "contingent": 40,
                    "unsatisfiable": 0,
                    "valid": 0,
                    "build_errors": 0,
                    "seconds": 0.2,
                },
            },
            "E5": {
                "legality": 0.0,
                "legal_streams": 0,
                "examples": 40,
                "ended_at_end_marker": 0,
                "rows_outside_the_algebra_alphabet": 0,
            },
        },
        "verdicts": {
            "E1": {"passed": True, "exact_tree_accuracy": 0.75, "threshold": 0.7},
            "E2": {"passed": True, "legal_streams": 40, "examples": 40},
            "E3": {"passed": True, "structural": True, "max_invented_number_rate": 0.02, "max_noncontingent_rate": 0.02},
            "E5": {"passed": True, "legality_with_mask": 1.0, "legality_without_mask": 0.0},
        },
    }
    path.write_text(json.dumps(report), encoding="utf-8")
    return str(path)


# ---------------------------------------------------------------------------
# safetensors serialization
# ---------------------------------------------------------------------------
def test_safetensors_round_trip_restores_the_tied_projection(tmp_path: Path) -> None:
    model = EncoderDecoderModel(tiny_config(60), padding_id=0)
    apply_initialization(model, seed=7)
    report = save_model_safetensors(model, str(tmp_path / "model.safetensors"), {"note": "test"})
    restored = load_model_safetensors(str(tmp_path / "model.safetensors"))

    before = model.state_dict()
    after = restored.state_dict()
    assert list(before) == list(after)
    assert all(torch.equal(before[name], after[name]) for name in before)
    assert restored.decoder_embeddings.tokens.weight.data_ptr() == restored.output_projection.weight.data_ptr()
    assert restored.config.to_dict() == model.config.to_dict()
    assert report["external_weights"] == "none"
    assert report["shared_tensors_stored_once"] == 1
    assert report["tensors"] == report["declared_names"] - 1


def test_safetensors_report_counts_the_shared_tensor_once(tmp_path: Path) -> None:
    model = EncoderDecoderModel(tiny_config(60), padding_id=0)
    apply_initialization(model, seed=11)
    report = save_model_safetensors(model, str(tmp_path / "weights.safetensors"))
    header = read_model_safetensors_header(str(tmp_path / "weights.safetensors"))

    assert header["format"] == FORMAT_NAME
    assert header["padding_id"] == 0
    assert header["config"]["vocabulary_size"] == 60
    assert header["metadata"] == {}
    total = sum(int(tensor.numel()) for name, tensor in dict(model.named_parameters()).items())
    assert report["parameters"] == total


def test_safetensors_reader_refuses_a_foreign_header(tmp_path: Path) -> None:
    from safetensors.torch import save_file

    path = tmp_path / "other.safetensors"
    save_file({"weight": torch.zeros(3)}, str(path), metadata={"format": "something-else"})
    with pytest.raises(ModelError):
        read_model_safetensors_header(str(path))
    with pytest.raises(ModelError):
        load_model_safetensors(str(path))


# ---------------------------------------------------------------------------
# mask specification
# ---------------------------------------------------------------------------
def test_mask_spec_round_trips_through_json() -> None:
    mask = AlgebraMask.from_leaf_payloads(ARITIES, LEAF_PAYLOADS)
    spec = json.loads(json.dumps(mask.to_spec()))
    rebuilt = AlgebraMask.from_spec(spec)

    assert rebuilt.to_spec() == mask.to_spec()
    assert rebuilt.describe() == mask.describe()
    state = rebuilt.initial_state()
    assert rebuilt.allowed_symbols(state) == mask.allowed_symbols(state)
    opened = rebuilt.step(state, "(")
    reference_opened = mask.step(mask.initial_state(), "(")
    assert rebuilt.allowed_symbols(opened) == mask.allowed_symbols(reference_opened)
    filled = rebuilt.step(opened, "IF")
    assert rebuilt.allowed_symbols(filled) == mask.allowed_symbols(mask.step(reference_opened, "IF"))


def test_mask_spec_survives_child_kind_refinement() -> None:
    observed: Counter = Counter()
    AlgebraMask.observe_child_kinds(
        {"op": "IF", "args": [{"leaf": ["flag"]}, {"op": "AT", "args": [{"leaf": ["pct"]}]}]},
        observed,
    )
    mask = AlgebraMask.from_leaf_payloads(ARITIES, LEAF_PAYLOADS, AlgebraMask.child_kinds_from_counts(observed))
    rebuilt = AlgebraMask.from_spec(json.loads(json.dumps(mask.to_spec())))
    assert rebuilt.child_kinds == mask.child_kinds


def test_mask_spec_refuses_incomplete_or_malformed_input() -> None:
    with pytest.raises(AlgebraMaskError):
        AlgebraMask.from_spec({"arities": ARITIES})
    with pytest.raises(AlgebraMaskError):
        AlgebraMask.from_spec({"arities": ARITIES, "leaf_forms": {}, "child_kinds": {"IF zero": ["("]}})
    with pytest.raises(AlgebraMaskError):
        AlgebraMask.from_spec("not a mapping")


LEARNED_TOKENIZER = TOOLS.parent / "checkpoints" / "tokenizer" / "tokenizer.json"


def learned_tokenizer() -> Tokenizer:
    """The learned vocabulary is a local artifact, skip when it is not present."""
    if not LEARNED_TOKENIZER.is_file():
        pytest.skip("no learned vocabulary at " + str(LEARNED_TOKENIZER))
    return Tokenizer.load(str(LEARNED_TOKENIZER))


# ---------------------------------------------------------------------------
# gate verdict handling
# ---------------------------------------------------------------------------
def test_gate_digest_projects_the_numbers_the_card_prints(tmp_path: Path) -> None:
    report = json.loads(Path(gate_report(tmp_path / "report.json")).read_text(encoding="utf-8"))
    digest = hub.gate_digest(report)

    assert digest["final_verdict"] == "PASS"
    assert digest["E1"]["exact_tree_accuracy"] == 0.75
    assert digest["E2"]["failures"] == 0
    assert digest["E3"]["z3"]["contingent"] == 40
    assert digest["E5"]["legality_without_mask"] == 0.0
    assert digest["E5"]["legality_with_mask"] == 1.0


def test_require_gate_pass_refuses_a_failing_verdict(tmp_path: Path) -> None:
    failing = {"final_verdict": "FAIL"}
    with pytest.raises(hub.PublishError):
        hub.require_gate_pass(failing, False)
    assert hub.require_gate_pass(failing, True) == "FAIL"
    assert hub.require_gate_pass({"final_verdict": "PASS"}, False) == "PASS"


# ---------------------------------------------------------------------------
# package assembly
# ---------------------------------------------------------------------------
def package_arguments(tmp_path: Path, tokenizer_path: str, report_path: str, *extra: str):
    return hub.command_line(
        [
            "--checkpoint",
            str(tmp_path / "model.pt"),
            "--report",
            report_path,
            "--tokenizer",
            tokenizer_path,
            "--out-dir",
            str(tmp_path / "hub"),
            "--repo",
            "someone/somewhere",
            *extra,
        ]
    )


def test_package_builder_writes_every_declared_artifact(tmp_path: Path) -> None:
    tokenizer = learned_tokenizer()
    tiny_checkpoint(tmp_path / "model.pt", int(tokenizer.vocabulary_size), str(LEARNED_TOKENIZER))
    arguments = package_arguments(tmp_path, "", gate_report(tmp_path / "report.json"))
    (tmp_path / "hub").mkdir(parents=True, exist_ok=True)
    (tmp_path / "hub" / "stale_from_an_earlier_run.json").write_text("{}", encoding="utf-8")

    manifest = hub.build_package(arguments)

    assert manifest["final_verdict"] == "PASS"
    assert manifest["external_weights"] == "none"
    assert manifest["round_trip"]["maximum_logit_difference"] == 0.0
    names = sorted(row["path"] for row in manifest["files"])
    assert names == sorted(name for name in hub.PACKAGE_FILES if name != "metadata.json")
    for row in manifest["files"]:
        assert hub.sha256_of(tmp_path / "hub" / row["path"]) == row["sha256"]
    assert all("stale" not in row["path"] for row in manifest["files"])

    card = (tmp_path / "hub" / "README.md").read_text(encoding="utf-8")
    assert "0.7500" in card
    assert "E5 ablation" in card
    assert "license: other" in card

    written = json.loads((tmp_path / "hub" / "metadata.json").read_text(encoding="utf-8"))
    assert written["weights_format"] == FORMAT_NAME
    assert written["gates"]["E1"]["exact_tree_accuracy"] == 0.75
    assert all(len(row["sha256"]) == 64 for row in written["files"])

    spec = json.loads((tmp_path / "hub" / "mask_spec.json").read_text(encoding="utf-8"))
    rebuilt = AlgebraMask.from_spec(spec["algebra"])
    assert rebuilt.describe()["operators"] == spec["token_view"]["operators"]


def test_package_builder_refuses_an_ungated_checkpoint(tmp_path: Path) -> None:
    tokenizer = learned_tokenizer()
    tiny_checkpoint(
        tmp_path / "model.pt",
        int(tokenizer.vocabulary_size),
        str(LEARNED_TOKENIZER),
    )
    arguments = package_arguments(tmp_path, "", gate_report(tmp_path / "report.json", "FAIL"))
    with pytest.raises(hub.PublishError):
        hub.build_package(arguments)


def test_package_builder_refuses_a_vocabulary_mismatch(tmp_path: Path) -> None:
    tokenizer = learned_tokenizer()
    tiny_checkpoint(
        tmp_path / "model.pt",
        int(tokenizer.vocabulary_size) - 1,
        str(LEARNED_TOKENIZER),
    )
    arguments = package_arguments(tmp_path, "", gate_report(tmp_path / "report.json"))
    with pytest.raises(hub.PublishError):
        hub.build_package(arguments)
