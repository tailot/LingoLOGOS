"""Text routing tests: lllayer turns one sentence into one validated syntax tree.

Three grammar sources are covered: the built-in AST alphabet of the node wrapper,
the node names and routing matrix of a declared topology, and the trained model
plus bracket algebra automaton of a release package. The command line contract of
``--data`` is checked in a separate interpreter process, because a node is never
imported as a module by the orchestrator.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from lingologos_orchestrator.lllayer_text import (
    DatasetGrammarRouter,
    TextRoutingError,
    GRAMMAR_DATASET,
    GRAMMAR_GENERIC,
    GRAMMAR_TOPOLOGY,
    first_mention,
    fold_text,
    load_topology_binding,
    mentioned_nodes,
    resolve_model_package,
    route_text,
    skeleton_of_parentheses,
    validate_parentheses,
)
from lingologos_orchestrator.schema import Envelope

REPO_ROOT = Path(__file__).resolve().parent.parent
TOPOLOGY_FILE = REPO_ROOT / "config" / "topology.json"
HUB_PACKAGE = REPO_ROOT / "datasets" / "multilingual" / "checkpoints" / "hub"
CORPUS_SENTENCE_FILE = REPO_ROOT / "datasets" / "multilingual" / "data" / "it" / "test-00000.jsonl"


def run_lllayer(arguments, stdin_text=None):
    """Launch lllayer as a separate process the way NodeProcessRunner prepares a child."""
    environment = dict(os.environ)
    package_parent = str(REPO_ROOT / "src")
    existing = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = package_parent if not existing else package_parent + os.pathsep + existing
    process = subprocess.run(
        [sys.executable, "-m", "lingologos_orchestrator.lllayer"] + arguments,
        input=stdin_text,
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        env=environment,
    )
    return process


BASE_ENVELOPE = {
    "sender": "agente_beta",
    "recipient": "agente_alfa",
    "turn": 1,
    "volatileState": {"internalTimeTick": 3},
    "payload": {
        "naturalLanguage": "Proposal for transition.",
        "abstractSyntaxTree": "(Transition StateA StateB)",
        "smtConstraints": {"declarations": [], "assertions": ["(assert true)"], "z3Status": "sat"},
    },
}


# ---------------------------------------------------------------------------
# Evidence helpers
# ---------------------------------------------------------------------------
def test_folding_matches_accents_and_underscores():
    """Evidence matching is ASCII insensitive so node ids and sentences meet."""
    assert fold_text("caf\u00e9 umidit\u00e0") == "cafe umidita"
    assert fold_text("Agente_Alf!") == "agente alf"
    assert fold_text("Se la temperatura supera 24 gradi") == "se la temperatura supera 24 gradi"
    assert first_mention(["urgente", "urgent"], "allarme urgente in corso") == 8
    assert first_mention(["urgente", "urgent"], "nessuna parola chiave") == -1


# ---------------------------------------------------------------------------
# Generic route: the built-in AST alphabet
# ---------------------------------------------------------------------------
def test_generic_route_returns_one_balanced_tree():
    """The generic grammar needs no corpus and no topology file."""
    result = route_text("notifica l utente in caso di allarme", grammar=GRAMMAR_GENERIC)
    assert result.grammar == GRAMMAR_GENERIC
    assert result.tree == "( Transition StateA StateB )"
    assert result.skeleton == "()"
    assert result.valid is True
    validate_parentheses(result.tree)


def test_generic_route_fires_the_modifier_branch_from_the_text():
    """A modifier word in the sentence selects the optional branch the engine prunes."""
    result = route_text("in caso di allarme urgente notifica l utente", grammar=GRAMMAR_GENERIC)
    assert result.tree == "( Transition MODIFIER )"
    assert result.evidence["modifierMentioned"] is True


def test_generic_route_is_the_session_of_the_constrained_decoding_flag():
    """The turn path and the query path share one grammar."""
    from lingologos_orchestrator.lllayer import run_constrained_ast_decoding

    assert run_constrained_ast_decoding("agente_alfa", "proponent", "(Transition StateA StateB)") == (
        "( Transition StateA StateB )"
    )


# ---------------------------------------------------------------------------
# Topology route: declared nodes and routing matrix
# ---------------------------------------------------------------------------
def test_topology_binding_reads_nodes_and_edges():
    """The binding comes from the declared topology, never from hardcoded names."""
    binding = load_topology_binding(TOPOLOGY_FILE)
    assert binding.session_id == "lingologos-multiagent-01"
    assert len(binding.node_ids) >= 2
    assert any(binding.adjacency.get(node) for node in binding.node_ids)
    assert binding.routes()


def test_topology_route_names_only_declared_and_mentioned_nodes():
    """A sentence naming two connected nodes yields the routing tree between them."""
    binding = load_topology_binding(TOPOLOGY_FILE)
    source = next(node for node in binding.node_ids if binding.adjacency.get(node))
    target = binding.adjacency[source][0]
    sentence = "passa da " + source.replace("_", " ") + " a " + target.replace("_", " ")
    result = route_text(sentence, grammar=GRAMMAR_TOPOLOGY, topology_config=TOPOLOGY_FILE)
    assert result.grammar == GRAMMAR_TOPOLOGY
    assert result.valid is True
    assert result.skeleton == "()"
    words = result.tree.split()
    assert words[0] == "(" and words[1] == "Transition"
    assert words[2] == source and words[3] == target
    assert result.evidence["route"] == {"source": source, "target": target, "declared": True}


def test_topology_route_refuses_a_route_that_is_not_declared(tmp_path):
    """An edge missing from the routing matrix never enters the mask."""
    config = json.loads(TOPOLOGY_FILE.read_text(encoding="utf-8"))
    source = config["instances"][0]["id"]
    unreachable = next(node["id"] for node in config["instances"] if node["id"] != source)
    rules = [rule for rule in config["routingMatrix"] if rule["from"] != source]
    for rule in rules:
        rule["to"] = [target for target in rule["to"] if target != unreachable]
    rules = [rule for rule in rules if rule["to"]]
    rules = [rule for rule in rules if rule["from"] != unreachable and source not in rule["to"]]
    config["routingMatrix"] = rules
    restricted = tmp_path / "topology_restricted.json"
    restricted.write_text(json.dumps(config), encoding="utf-8")
    binding = load_topology_binding(restricted)
    assert not binding.route_exists(source, unreachable)
    assert not binding.route_exists(unreachable, source)
    sentence = "passa da " + source.replace("_", " ") + " a " + unreachable.replace("_", " ")
    with pytest.raises(TextRoutingError) as refused:
        route_text(sentence, grammar=GRAMMAR_TOPOLOGY, topology_config=restricted)
    assert "no declared route" in str(refused.value)


def test_topology_route_refuses_a_sentence_naming_one_node():
    """Refusing beats inventing an endpoint the sentence never mentions."""
    with pytest.raises(TextRoutingError) as refused:
        route_text(
            "riorganizza la sessione",
            grammar=GRAMMAR_TOPOLOGY,
            topology_config=TOPOLOGY_FILE,
        )
    assert "naming a source and a destination is required" in str(refused.value)


def test_topology_mentions_are_scanned_in_order():
    """Mention order decides which endpoint is the source."""
    binding = load_topology_binding(TOPOLOGY_FILE)
    mentions = mentioned_nodes("prima arbitro logico poi agente alfa", binding)
    assert [node for node, _ in mentions] == ["arbitro_logico", "agente_alfa"]
    assert mentions[0][1] < mentions[1][1]


# ---------------------------------------------------------------------------
# Dataset route: trained weights plus the algebra automaton
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def dataset_router():
    """One release package load for the whole module."""
    if not (HUB_PACKAGE / "model.safetensors").is_file():
        pytest.skip("no release package with weights on this host")
    if not (HUB_PACKAGE / "mask_spec.json").is_file():
        pytest.skip("no mask specification in the release package")
    return DatasetGrammarRouter(resolve_model_package(HUB_PACKAGE))


def one_corpus_sentence():
    if not CORPUS_SENTENCE_FILE.is_file():
        pytest.skip("the multilingual corpus is not present on this host")
    line = CORPUS_SENTENCE_FILE.read_text(encoding="utf-8").strip().splitlines()[0]
    return json.loads(line)


def test_dataset_route_decodes_a_sentence_into_a_legal_tree(dataset_router):
    """The mask makes structural hallucination impossible, whatever the model proposes."""
    row = one_corpus_sentence()
    result = dataset_router.route(row["phrase"])
    assert result.grammar == GRAMMAR_DATASET
    assert result.valid is True
    assert result.tree.split()[0] == "("
    assert dataset_router.token_mask.mask.validate_stream(result.symbols)
    assert result.skeleton == "".join(
        symbol for symbol in result.symbols if symbol in "([])"
    )
    assert result.model["operators"] == len(dataset_router.mask_spec["arities"])


def test_dataset_route_refuses_an_empty_sentence(dataset_router):
    with pytest.raises(TextRoutingError):
        dataset_router.route("   ")


def test_dataset_route_is_reachable_through_the_auto_selection(dataset_router):
    """A named package is enough for auto mode to pick the dataset grammar."""
    row = one_corpus_sentence()
    result = route_text(row["phrase"], model_package=HUB_PACKAGE)
    assert result.grammar == GRAMMAR_DATASET
    assert result.tree.startswith("( ")


def test_resolve_model_package_reports_what_is_missing(tmp_path):
    empty = tmp_path / "empty_package"
    empty.mkdir()
    with pytest.raises(TextRoutingError) as refused:
        resolve_model_package(empty)
    assert "mask_spec.json" in str(refused.value)


# ---------------------------------------------------------------------------
# Command line contract
# ---------------------------------------------------------------------------
def test_cli_query_mode_prints_the_tree_of_the_generic_grammar():
    process = run_lllayer(
        [
            "--node-id", "agente_alfa",
            "--node-dir", "datasets/agente_alfa",
            "--data", "notifica l utente",
            "--format", "tree",
        ]
    )
    assert process.returncode == 0, process.stderr
    assert process.stdout.strip() == "( Transition StateA StateB )"


def test_cli_query_mode_answers_an_incoming_envelope():
    """A piped envelope still gets an envelope back, so the protocol holds."""
    process = run_lllayer(
        [
            "--node-id", "agente_alfa",
            "--role", "proponent",
            "--node-dir", "datasets/agente_alfa",
            "--data", "manda urgente da agente beta ad agente alfa",
            "--topology-config", "config/topology.json",
        ],
        stdin_text=json.dumps(BASE_ENVELOPE),
    )
    assert process.returncode == 0, process.stderr
    envelope = Envelope.from_dict(json.loads(process.stdout))
    assert envelope.sender == "agente_alfa"
    assert envelope.recipient == "agente_beta"
    assert envelope.turn == 1
    assert envelope.payload.abstractSyntaxTree == "( Transition agente_beta agente_alfa MODIFIER )"
    assert envelope.topologyCtrl is not None
    assert json.loads(process.stdout)["volatileState"]["grammarSource"] == "topology"


def test_cli_query_mode_reports_the_dataset_grammar():
    if not (HUB_PACKAGE / "model.safetensors").is_file():
        pytest.skip("no release package with weights on this host")
    process = run_lllayer(
        [
            "--node-id", "agente_alfa",
            "--node-dir", "datasets/agente_alfa",
            "--dataset-root", "datasets/multilingual",
            "--data", "Spegni il condizionatore quando la temperatura supera 24 gradi.",
            "--format", "json",
        ]
    )
    assert process.returncode == 0, process.stderr
    report = json.loads(process.stdout)
    assert report["grammar"] == "dataset"
    assert report["valid"] is True
    assert report["model"]["vocabularySize"] > 1000
    validate_parentheses(report["tree"])


def test_cli_show_topology_lists_nodes_and_routes():
    process = run_lllayer(
        [
            "--node-id", "agente_alfa",
            "--node-dir", "datasets/agente_alfa",
            "--topology-config", "config/topology.json",
            "--show-topology",
        ]
    )
    assert process.returncode == 0, process.stderr
    declaration = json.loads(process.stdout)
    assert declaration["sessionId"] == "lingologos-multiagent-01"
    assert len(declaration["nodes"]) >= 2
    assert "routingMatrix" in declaration
    assert "topologyCtrl" in declaration


def test_cli_query_mode_refuses_with_a_clear_line_on_stderr():
    process = run_lllayer(
        [
            "--node-id", "agente_alfa",
            "--node-dir", "datasets/agente_alfa",
            "--data", "riorganizza la sessione",
            "--grammar", "topology",
            "--topology-config", "config/topology.json",
        ]
    )
    assert process.returncode == 2
    assert process.stdout == ""
    assert "naming a source and a destination is required" in process.stderr


def test_cli_envelope_mode_is_unchanged_by_the_query_flags():
    """Without --data the node still answers the turn protocol exactly as before."""
    process = run_lllayer(
        [
            "--node-id", "agente_alfa",
            "--role", "proponent",
            "--node-dir", "datasets/agente_alfa",
            "--constrained-decoding",
        ],
        stdin_text=json.dumps(BASE_ENVELOPE),
    )
    assert process.returncode == 0, process.stderr
    reply = json.loads(process.stdout)
    assert reply["sender"] == "agente_alfa"
    assert "agente_alfa" in reply["payload"]["abstractSyntaxTree"]
    assert skeleton_of_parentheses(reply["payload"]["abstractSyntaxTree"]).startswith("(")
