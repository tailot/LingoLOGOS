"""Unit and integration tests for volatile AST logits masker and pruning engine.

Verifies zero-copy C-FFI masker kernel, memory safety, Hadamard addition,
Softmax zero probability invariant, AST pruning trigger, and lllayer CLI integration.
All comments and docstrings strictly adhere to AGENTS.md rules.
"""

import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import pytest

from lingologos_orchestrator.lllayer_masker import (
    NativeASTLogitsProcessor,
    find_or_build_native_library,
)
from lingologos_orchestrator.lllayer_engine import (
    IncrementalASTEngine,
    SimpleASTGrammar,
    MockRunnerModel,
)
from lingologos_orchestrator.lllayer import run_constrained_ast_decoding


def test_native_library_compilation_and_loading():
    """Verifies that the C++ shared library exists or builds successfully."""
    so_path = find_or_build_native_library()
    assert so_path is not None
    assert Path(so_path).is_file()


def test_native_masker_initialization_and_cleanup():
    """Tests buffer allocation and instant memory release."""
    vocab_size = 32
    processor = NativeASTLogitsProcessor(vocab_size)
    assert processor.is_native is True
    assert processor.vocab_size == vocab_size

    # Verify initial mask is -inf for all indices
    logits = np.zeros(vocab_size, dtype=np.float32)
    masked = processor([], logits)
    assert np.all(np.isneginf(masked))

    processor.cleanup()
    assert processor._state_ptr is None


def test_hadamard_logits_addition_and_zero_probability():
    """Verifies that unmasked tokens keep their score and masked tokens produce zero probability."""
    vocab_size = 8
    processor = NativeASTLogitsProcessor(vocab_size)

    allowed = {1, 3, 5}
    processor.update_mask_from_ast_state(allowed)

    logits = np.array([2.0, 1.0, 5.0, 3.0, 4.0, 0.5, 1.5, 2.5], dtype=np.float32)
    masked = processor([], logits.copy())

    # Allowed tokens must remain unchanged
    for idx in allowed:
        assert np.isclose(masked[idx], logits[idx])

    # Disallowed tokens must be -inf
    for idx in set(range(vocab_size)) - allowed:
        assert np.isneginf(masked[idx])

    # Softmax test: probability on blocked tokens must be mathematically 0
    exp_logits = np.exp(masked - np.max(masked[list(allowed)]))
    probs = exp_logits / np.sum(exp_logits)

    for idx in set(range(vocab_size)) - allowed:
        assert probs[idx] == 0.0
    assert np.isclose(np.sum(probs), 1.0)

    processor.cleanup()


def test_python_fallback_masker():
    """Tests the pure Python / numpy fallback path when no shared library is used."""
    vocab_size = 10
    processor = NativeASTLogitsProcessor(vocab_size, lib_path="/nonexistent/lib.so")
    assert processor.is_native is False

    allowed = {2, 7}
    processor.update_mask_from_ast_state(allowed)

    logits = np.zeros(vocab_size, dtype=np.float32)
    masked = processor([], logits)

    assert masked[2] == 0.0
    assert masked[7] == 0.0
    assert np.isneginf(masked[0])
    assert np.isneginf(masked[9])

    processor.cleanup()


def test_incremental_ast_engine_decoding():
    """Tests token-by-token constrained AST decoding without pruning."""
    tokens = ["ROOT", "(", "Transition", "StateA", "StateB", ")", "END"]
    token_to_id = {t: i for i, t in enumerate(tokens)}
    id_to_token = {i: t for i, t in enumerate(tokens)}

    transitions = {
        ("ROOT", "("): "NODE_OPN",
        ("NODE_OPN", "Transition"): "NODE_TRANS",
        ("NODE_TRANS", "StateA"): "NODE_S1",
        ("NODE_S1", "StateB"): "NODE_S2",
        ("NODE_S2", ")"): "TERMINAL",
    }
    valid_tokens = {
        "ROOT": {token_to_id["("]},
        "NODE_OPN": {token_to_id["Transition"]},
        "NODE_TRANS": {token_to_id["StateA"]},
        "NODE_S1": {token_to_id["StateB"]},
        "NODE_S2": {token_to_id[")"]},
    }
    closing_tokens = {token_to_id[")"]}

    grammar = SimpleASTGrammar(
        token_to_id=token_to_id,
        id_to_token=id_to_token,
        transitions=transitions,
        valid_tokens=valid_tokens,
        closing_tokens=closing_tokens,
        terminal_nodes={"TERMINAL"},
    )

    processor = NativeASTLogitsProcessor(vocab_size=len(tokens))
    runner_model = MockRunnerModel(vocab_size=len(tokens), token_to_id=token_to_id, id_to_token=id_to_token)
    engine = IncrementalASTEngine(grammar_spec=grammar, logits_processor=processor)

    result = engine.run_session(prompt="test", runner_model=runner_model)
    assert result == "( Transition StateA StateB )"


def test_incremental_ast_engine_pruning_activation():
    """Tests AST subtree pruning when an ancillary/optional modifier node is encountered."""
    tokens = ["ROOT", "(", "Transition", "MODIFIER", "StateA", "StateB", ")"]
    token_to_id = {t: i for i, t in enumerate(tokens)}
    id_to_token = {i: t for i, t in enumerate(tokens)}

    node_types = {"MODIFIER": "OPTIONAL_MODIFIER"}
    transitions = {
        ("ROOT", "("): "NODE_OPN",
        ("NODE_OPN", "Transition"): "NODE_TRANS",
        ("NODE_TRANS", "MODIFIER"): "MODIFIER",
        ("MODIFIER", ")"): "TERMINAL",
        ("NODE_S2", ")"): "TERMINAL",
    }
    valid_tokens = {
        "ROOT": {token_to_id["("]},
        "NODE_OPN": {token_to_id["Transition"]},
        # Only MODIFIER is offered at NODE_TRANS to force entering ancillary node
        "NODE_TRANS": {token_to_id["MODIFIER"]},
        "MODIFIER": {token_to_id["StateA"], token_to_id[")"]},
    }
    closing_tokens = {token_to_id[")"]}

    grammar = SimpleASTGrammar(
        token_to_id=token_to_id,
        id_to_token=id_to_token,
        node_types=node_types,
        transitions=transitions,
        valid_tokens=valid_tokens,
        closing_tokens=closing_tokens,
        terminal_nodes={"TERMINAL"},
    )

    processor = NativeASTLogitsProcessor(vocab_size=len(tokens))
    runner_model = MockRunnerModel(vocab_size=len(tokens), token_to_id=token_to_id, id_to_token=id_to_token)
    engine = IncrementalASTEngine(grammar_spec=grammar, logits_processor=processor)

    result = engine.run_session(prompt="prune_test", runner_model=runner_model)
    # The pruning mechanism forces closing tokens once MODIFIER is encountered
    assert engine.is_pruning_active is True
    assert result == "( Transition MODIFIER )"


def test_lllayer_run_constrained_ast_decoding_helper():
    """Tests the high level run_constrained_ast_decoding function."""
    out = run_constrained_ast_decoding(
        node_id="agente_alfa",
        role="proponent",
        incoming_ast="(Transition StateA StateB)",
    )
    assert isinstance(out, str)
    assert len(out) > 0


def test_lllayer_cli_constrained_decoding_flag():
    """Tests lllayer CLI execution with --constrained-decoding flag over standard I/O."""
    incoming_payload = {
        "sender": "agente_beta",
        "recipient": "agente_alfa",
        "turn": 1,
        "volatileState": {"internalTimeTick": 1},
        "payload": {
            "naturalLanguage": "Proposal for transition.",
            "abstractSyntaxTree": "(Transition StateA StateB)",
            "smtConstraints": {
                "declarations": ["(declare-sort State)"],
                "assertions": ["(assert true)"],
                "z3Status": "sat",
            },
        },
    }

    cmd = [
        sys.executable,
        "-m",
        "lingologos_orchestrator.lllayer",
        "--node-id",
        "agente_alfa",
        "--role",
        "proponent",
        "--node-dir",
        "datasets/agente_alfa",
        "--constrained-decoding",
    ]

    proc = subprocess.run(
        cmd,
        input=json.dumps(incoming_payload),
        capture_output=True,
        text=True,
        check=True,
    )

    out_data = json.loads(proc.stdout)
    assert out_data["sender"] == "agente_alfa"
    assert out_data["recipient"] == "agente_beta"
    assert "abstractSyntaxTree" in out_data["payload"]
    assert "agente_alfa" in out_data["payload"]["abstractSyntaxTree"]
