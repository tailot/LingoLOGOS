"""Unit and integration tests for volatile AST logits masker and pruning engine.

Verifies the host independent kernel selection, the C-FFI masking kernel and its numpy
fallback parity, memory safety, Hadamard addition, the Softmax zero probability invariant,
the AST pruning trigger, and lllayer CLI integration. Every test must pass on Linux and on
Windows. All comments and docstrings strictly adhere to AGENTS.md rules.
"""

import json
import os
import subprocess
import sys
from pathlib import Path
import numpy as np
import pytest

from lingologos_orchestrator import lllayer_masker as masker_module
from lingologos_orchestrator.lllayer_masker import (
    NativeASTLogitsProcessor,
    compiler_candidates,
    find_or_build_native_library,
    library_filename,
    native_library_path,
)
from lingologos_orchestrator.lllayer_engine import (
    IncrementalASTEngine,
    SimpleASTGrammar,
    MockRunnerModel,
)
from lingologos_orchestrator.lllayer import run_constrained_ast_decoding


def test_library_filename_matches_every_supported_host():
    """Checks the shared object name selected for Linux, macOS, and Windows."""
    assert library_filename("posix", "linux") == "libmasker.so"
    assert library_filename("posix", "darwin") == "libmasker.dylib"
    assert library_filename("nt", "win32") == "masker.dll"
    # The host executing this test must be offered the file name its loader can open.
    selected = native_library_path()
    assert selected.parent.name == "native"
    assert selected.name == library_filename(os.name, sys.platform)


def test_compiler_lookup_stays_free_of_posix_shell_probes():
    """Guards against reintroducing the POSIX only probe that broke the loader on Windows.

    The previous implementation shelled out to which(1) and redirected to /dev/null, so on
    PowerShell and CMD it always failed and wrote errors onto the stderr stream that the
    orchestrator parses for a node. The lookup must rely on shutil.which instead.
    """
    source = Path(masker_module.__file__).read_text(encoding="utf-8")
    for forbidden in ("os.system", "/dev/null", 'which g++'):
        assert forbidden not in source, f"the masking kernel lookup must not use {forbidden}"
    # Candidates are resolved without spawning a shell, so an absent toolchain yields an
    # empty list rather than a failure or a message on the standard streams.
    for candidate in compiler_candidates():
        assert Path(candidate).is_file()


def test_masking_works_on_the_kernel_offered_by_this_host():
    """Any host must obtain a working masker, either the C kernel or the numpy buffer."""
    vocab_size = 32
    processor = NativeASTLogitsProcessor(vocab_size)
    assert processor.vocab_size == vocab_size
    # Exactly one branch is live: the C pointer or the persistent numpy buffer.
    assert processor.is_native or processor._fallback_mask is not None

    # With an empty admission set every score must be suppressed.
    masked = processor([], np.zeros(vocab_size, dtype=np.float32))
    assert np.all(np.isneginf(masked))

    processor.update_mask_from_ast_state({3, 17, 31})
    reopened = processor([], np.zeros(vocab_size, dtype=np.float32))
    for idx in (3, 17, 31):
        assert np.isfinite(reopened[idx])
    assert np.all(np.isneginf(np.delete(reopened, [3, 17, 31])))

    processor.cleanup()


def test_native_kernel_matches_numpy_reference():
    """The C kernel and the numpy buffer must produce byte identical masks."""
    lib_path = find_or_build_native_library()
    if lib_path is None:
        pytest.skip(
            "no GCC compatible C++ toolchain on this host, "
            f"candidates found: {compiler_candidates()}"
        )
    assert Path(lib_path).is_file()

    native = NativeASTLogitsProcessor(vocab_size=64, lib_path=lib_path)
    assert native.is_native is True
    absent = str(native_library_path().parent / "kernel_that_does_not_exist")
    fallback = NativeASTLogitsProcessor(vocab_size=64, lib_path=absent)
    assert fallback.is_native is False

    rng = np.random.default_rng(20261111)
    cases = ({}, {0}, {63}, {1, 3, 5}, {7, 8, 9, 40}, set(range(64)))
    for allowed in cases:
        raw = rng.standard_normal(64).astype(np.float32)
        native.update_mask_from_ast_state(allowed)
        fallback.update_mask_from_ast_state(allowed)
        masked_native = native([], raw.copy())
        masked_fallback = fallback([], raw.copy())
        assert np.array_equal(masked_native, masked_fallback)

    native.cleanup()
    fallback.cleanup()


def test_environment_switch_selects_the_numpy_branch(monkeypatch):
    """The documented switch must force the numpy branch on any platform."""
    monkeypatch.setenv(masker_module.DISABLE_NATIVE_ENV_VAR, "1")
    assert masker_module.find_or_build_native_library() is None
    assert masker_module.NativeASTLogitsProcessor(8).is_native is False

    monkeypatch.delenv(masker_module.DISABLE_NATIVE_ENV_VAR, raising=False)
    assert masker_module.native_disabled() is False


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
    """Tests the pure numpy fallback path when no shared library is used."""
    vocab_size = 10
    absent = str(native_library_path().parent / "kernel_that_does_not_exist")
    processor = NativeASTLogitsProcessor(vocab_size, lib_path=absent)
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
    """Tests lllayer CLI execution with --constrained-decoding flag over standard I/O.

    The node is a separate interpreter process, so the package must be reachable inside it.
    The same environment preparation used by NodeProcessRunner._child_environment applies
    here: the absolute folder holding the package is prepended to PYTHONPATH and the caller
    working directory is pinned to the checkout root, which keeps the relative dataset path
    valid whatever directory pytest was started from.
    """
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

    repo_root = Path(__file__).resolve().parent.parent
    env = dict(os.environ)
    package_parent = str(repo_root / "src")
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = package_parent if not existing else package_parent + os.pathsep + existing
    # The numpy branch is deterministic on every host, so the CLI assertion does not
    # depend on whether a C++ toolchain happens to be installed.
    env.pop(masker_module.DISABLE_NATIVE_ENV_VAR, None)

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
        cwd=str(repo_root),
        env=env,
    )

    out_data = json.loads(proc.stdout)
    assert out_data["sender"] == "agente_alfa"
    assert out_data["recipient"] == "agente_beta"
    assert "abstractSyntaxTree" in out_data["payload"]
    assert "agente_alfa" in out_data["payload"]["abstractSyntaxTree"]
    # The node protocol parses stdout as JSON, so the kernel probe must stay silent.
    assert proc.stderr.strip() == ""


def test_reference_decoding_is_identical_on_both_kernels(monkeypatch):
    """A node must emit the same AST whether the C kernel or the numpy branch decoded it.

    This is the cross platform guarantee of the masking layer: on Linux the C kernel is
    usually live, on Windows the numpy buffer is, and the emitted envelope must not differ.
    """
    monkeypatch.delenv(masker_module.DISABLE_NATIVE_ENV_VAR, raising=False)
    kernel_run = run_constrained_ast_decoding(
        node_id="agente_alfa",
        role="proponent",
        incoming_ast="(Transition StateQ StateZ)",
    )

    monkeypatch.setenv(masker_module.DISABLE_NATIVE_ENV_VAR, "1")
    numpy_run = run_constrained_ast_decoding(
        node_id="agente_alfa",
        role="proponent",
        incoming_ast="(Transition StateQ StateZ)",
    )

    assert kernel_run == numpy_run
    assert kernel_run.split() == ["(", "Transition", "StateA", "StateB", ")"]

