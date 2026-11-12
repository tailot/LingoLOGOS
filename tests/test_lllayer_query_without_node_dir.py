"""The query mode needs no node folder: the grammar source carries everything it reads."""

import json
from pathlib import Path

from tests.test_lllayer_text_routing import BASE_ENVELOPE, REPO_ROOT, run_lllayer


def test_query_mode_runs_without_the_node_folder():
    """A release package is a complete grammar source, so no node folder is involved."""
    if not (REPO_ROOT / "datasets" / "multilingual" / "checkpoints" / "hub" / "model.safetensors").is_file():
        import pytest

        pytest.skip("no release package with weights on this host")
    process = run_lllayer(
        [
            "--node-id", "agente_alfa",
            "--model-package", "datasets/multilingual/checkpoints/hub",
            "--data", "imposta un ora di 10 minuti",
            "--format", "tree",
        ]
    )
    assert process.returncode == 0, process.stderr
    assert process.stdout.startswith("(")


def test_generic_query_mode_needs_no_folder_at_all():
    """The built-in alphabet lives in the wrapper, so nothing on disk is required."""
    process = run_lllayer(
        [
            "--node-id", "agente_alfa",
            "--data", "notifica l utente",
            "--format", "tree",
        ]
    )
    assert process.returncode == 0, process.stderr
    assert process.stdout.strip() == "( Transition StateA StateB )"


def test_topology_query_mode_needs_no_folder_at_all():
    """The declared topology is the second source that stands on its own."""
    process = run_lllayer(
        [
            "--node-id", "agente_alfa",
            "--topology-config", "config/topology.json",
            "--data", "passa da agente alfa ad arbitro logico",
            "--format", "json",
        ]
    )
    assert process.returncode == 0, process.stderr
    report = json.loads(process.stdout)
    assert report["grammar"] == "topology"
    assert report["tree"] == "( Transition agente_alfa arbitro_logico )"


def test_query_envelope_still_names_the_node_without_a_folder():
    """The envelope protocol keeps its identity fields even with no node folder."""
    process = run_lllayer(
        [
            "--node-id", "arbitro_logico",
            "--topology-config", "config/topology.json",
            "--data", "passa da agente alfa ad arbitro logico",
            "--format", "envelope",
        ],
        stdin_text=json.dumps(BASE_ENVELOPE),
    )
    assert process.returncode == 0, process.stderr
    reply = json.loads(process.stdout)
    assert reply["sender"] == "arbitro_logico"
    assert reply["payload"]["abstractSyntaxTree"] == "( Transition agente_alfa arbitro_logico )"


def test_turn_mode_still_demands_the_node_folder():
    """Turn mode reads seed_payload.json, so the folder stays mandatory there."""
    process = run_lllayer(
        ["--node-id", "agente_alfa", "--role", "proponent"],
        stdin_text=json.dumps(BASE_ENVELOPE),
    )
    assert process.returncode == 2
    assert "--node-dir" in process.stderr


def test_a_data_file_with_a_byte_order_mark_reads_as_the_same_sentence(tmp_path):
    """Windows text editors prepend a mark that must not change the decoded tree."""
    marked = tmp_path / "query.txt"
    marked.write_bytes(b"\xef\xbb\xbfimposta un ora di 10 minuti")
    plain = tmp_path / "plain.txt"
    plain.write_text("imposta un ora di 10 minuti", encoding="utf-8")
    arguments = ["--node-id", "agente_alfa"]
    if not (REPO_ROOT / "datasets" / "multilingual" / "checkpoints" / "hub" / "model.safetensors").is_file():
        import pytest

        pytest.skip("no release package with weights on this host")
    arguments += ["--model-package", "datasets/multilingual/checkpoints/hub"]
    first = run_lllayer(arguments + ["--data-file", str(marked), "--format", "tree"])
    second = run_lllayer(arguments + ["--data-file", str(plain), "--format", "tree"])
    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr
    assert first.stdout == second.stdout
