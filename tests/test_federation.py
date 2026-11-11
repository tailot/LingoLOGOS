"""Unit and integration tests for LingoLOGOS federated infrastructure and scalability.

Tests:
  1. Federated schema parsing and roundtrip with topologies/federated_mesh.json.
  2. resolve_node_transport with local, git, and http.
  3. NodeResolver specs resolution.
  4. EventBus topic publish/subscribe, wildcards, and history.
  5. DynamicNodeSpawner generative sub-agent metadata generation.
  6. SMTSolverPool concurrent SMT verification.
  7. ModelConsolidator graph contraction to triad and unified models.
  8. Federated orchestrator execution and CLI --consolidate.
  9. Bearer token wiring of HTTP nodes and git clone performed by the orchestrator.

Adheres to English-only, ASCII-only rules and avoids apostrophes.
"""

import os
import sys
import json
import pathlib
import subprocess
import pytest
from unittest.mock import patch, MagicMock

from lingologos_orchestrator import cli
from lingologos_orchestrator.schema import (
    FederatedTopologyConfig,
    FederatedNodeConfig,
    FederatedRoutingRule,
    NodeSpecs,
    Envelope,
    VolatileState,
    Payload,
    SMTConstraints,
    SchemaValidationError,
    StaticValidationError,
)
from lingologos_orchestrator.resolver import resolve_node_transport, NodeResolver
from lingologos_orchestrator.node_runner import HttpNodeRunner, NodeProcessRunner
from lingologos_orchestrator.event_bus import EventBus, DynamicNodeSpawner
from lingologos_orchestrator.solver_pool import SMTSolverPool, SolverTask
from lingologos_orchestrator.consolidation import ModelConsolidator
from lingologos_orchestrator.orchestrator import MultiInstanceOrchestrator

REFERENCE_SPEC_DIR = pathlib.Path("datasets/agente_alfa")


def _write_spec_files(target_dir: pathlib.Path, node_id: str) -> None:
    """Copies the reference node specification and relabels the seed sender for node_id."""
    target_dir.mkdir(parents=True, exist_ok=True)
    for name in ("grammar.gf", "ontology.smt2"):
        (target_dir / name).write_text(
            (REFERENCE_SPEC_DIR / name).read_text(encoding="utf-8"), encoding="utf-8"
        )
    seed = json.loads((REFERENCE_SPEC_DIR / "seed_payload.json").read_text(encoding="utf-8"))
    seed["sender"] = node_id
    (target_dir / "seed_payload.json").write_text(json.dumps(seed), encoding="utf-8")


def _envelope(sender: str, recipient: str, turn: int, text: str) -> Envelope:
    """Builds a minimal envelope for transport tests."""
    return Envelope(
        sender=sender,
        recipient=recipient,
        turn=turn,
        volatileState=VolatileState(internalTimeTick=turn),
        payload=Payload(
            naturalLanguage=text,
            abstractSyntaxTree="QueryAST",
            smtConstraints=SMTConstraints(),
        ),
    )


def test_federated_mesh_schema():
    """Validates that topologies/federated_mesh.json matches FederatedTopologyConfig."""
    mesh_path = pathlib.Path("topologies/federated_mesh.json")
    assert mesh_path.is_file()

    with open(mesh_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    fed = FederatedTopologyConfig.from_dict(data)
    assert fed.federation_name == "lingologos_universal_federation"
    assert fed.version == "2.0.0"
    assert len(fed.nodes) == 3
    assert len(fed.routing_rules) == 4

    # The sample topology is runnable: every node id matches a dataset folder of the repository,
    # so the documented CLI commands execute it end to end without inventing data.
    node_ids = sorted(n.node_id for n in fed.nodes)
    assert node_ids == ["agente_alfa", "agente_beta", "arbitro_logico"]
    assert {n.role for n in fed.nodes} == {"proponent", "opponent", "validator"}
    assert [n.transport for n in fed.nodes] == ["local", "git", "local"]
    for node in fed.nodes:
        if node.transport != "git":
            specs_dir = pathlib.Path(node.local_cache_path)
            for spec in (node.specs.grammar, node.specs.ontology, node.specs.seed):
                assert (specs_dir / spec).is_file()

    # Test conversion to legacy TopologyConfig
    legacy_topo = fed.to_legacy_topology(max_turns=10)
    assert legacy_topo.sessionId == "lingologos_universal_federation"
    assert len(legacy_topo.instances) == 3
    assert legacy_topo.is_route_allowed("agente_alfa", "agente_beta")
    assert legacy_topo.is_route_allowed("arbitro_logico", "agente_alfa")


def test_resolve_node_transport_local(tmp_path):
    """Local transport resolves to local Path."""
    conf = {"transport": "local", "node_id": "test_local"}
    res = resolve_node_transport(conf)
    assert isinstance(res, pathlib.Path)


def test_resolve_node_transport_http():
    """HTTP transport resolves to endpoint URL string."""
    url = "http://127.0.0.1:8080/v1/evaluate"
    conf = {"transport": "http", "node_id": "test_http", "endpoint_url": url}
    res = resolve_node_transport(conf)
    assert res == url


@patch("subprocess.run")
def test_resolve_node_transport_git(mock_subproc, tmp_path):
    """Git transport clones repository if absent or pulls if present."""
    cache_dir = tmp_path / "cache" / "test_repo"
    conf = {
        "transport": "git",
        "remote_source": "http://127.0.0.1:9418/repo.git",
        "branch": "main",
        "local_cache_path": str(cache_dir),
    }

    # Case 1: absent cache path -> clone
    res = resolve_node_transport(conf)
    assert res == cache_dir
    mock_subproc.assert_called_with(
        ["git", "clone", "--branch", "main", "--depth", "1", "http://127.0.0.1:9418/repo.git", str(cache_dir)],
        check=True,
        capture_output=True,
        text=True,
    )

    # Case 2: present cache path -> pull
    cache_dir.mkdir(parents=True)
    res2 = resolve_node_transport(conf)
    assert res2 == cache_dir
    mock_subproc.assert_called_with(
        ["git", "-C", str(cache_dir), "pull", "origin", "main"],
        check=True,
        capture_output=True,
        text=True,
    )


def test_node_resolver(tmp_path):
    """NodeResolver correctly structures specification paths."""
    resolver = NodeResolver(workspace_dir=tmp_path)
    conf = {
        "node_id": "core_node",
        "transport": "local",
        "specs": {
            "grammar": "grammars/g.gf",
            "ontology": "ontologies/o.smt2",
            "seed": "seeds/s.json",
        },
    }
    resolved = resolver.resolve(conf)
    assert resolved["transport"] == "local"
    assert resolved["root_path"] == tmp_path.resolve()
    # Compare Path objects: forward slashes in the assertion break on Windows separators.
    assert resolved["grammar_path"] == (tmp_path / "grammars" / "g.gf").resolve()
    assert resolved["ontology_path"] == (tmp_path / "ontologies" / "o.smt2").resolve()
    # node_dir is the folder holding the three specification files, handed to the node process.
    assert resolved["node_dir"] == (tmp_path / "ontologies").resolve()


def test_event_bus():
    """EventBus delivers messages to exact and wildcard topic subscribers."""
    bus = EventBus()
    received_a = []
    received_wildcard = []

    bus.subscribe("channel/a/b", lambda env: received_a.append(env))
    bus.subscribe("*", lambda env: received_wildcard.append(env))

    env1 = Envelope(
        sender="a",
        recipient="b",
        turn=1,
        volatileState=VolatileState(internalTimeTick=1),
        payload=Payload(
            naturalLanguage="Hello B",
            abstractSyntaxTree="AST (A B)",
            smtConstraints=SMTConstraints(),
        ),
    )
    bus.publish("channel/a/b", env1)

    assert len(received_a) == 1
    assert len(received_wildcard) == 1
    assert bus.get_history() == [env1]

    # Non-matching topic
    bus.publish("channel/other", env1)
    assert len(received_a) == 1
    assert len(received_wildcard) == 2


def test_dynamic_node_spawner():
    """DynamicNodeSpawner generates valid sub-agent specifications."""
    spawner = DynamicNodeSpawner(base_role="analyzer", parent_id="root_agent")
    sub1 = spawner.spawn_sub_node("Evaluate complex sub-formula")
    sub2 = spawner.spawn_sub_node("Evaluate second sub-formula")

    assert sub1["role"] == "analyzer_subagent"
    assert sub1["parent_id"] == "root_agent"
    assert "sub_node_root_agent_1" in sub1["node_id"]
    assert "sub_node_root_agent_2" in sub2["node_id"]


def test_smt_solver_pool():
    """SMTSolverPool evaluates multiple tasks concurrently."""
    pool = SMTSolverPool(max_workers=2)

    sat_task = SolverTask(
        task_id="t_sat",
        constraints=SMTConstraints(declarations=["(declare-const x Int)"], assertions=["(assert (> x 0))"]),
    )
    unsat_task = SolverTask(
        task_id="t_unsat",
        constraints=SMTConstraints(declarations=["(declare-const x Int)"], assertions=["(assert (> x 0))", "(assert (< x 0))"]),
    )

    results = pool.verify_batch([sat_task, unsat_task])
    pool.shutdown()

    assert "t_sat" in results
    assert "t_unsat" in results
    assert results["t_sat"].is_sat is True
    assert results["t_unsat"].is_unsat is True


def test_model_consolidator():
    """ModelConsolidator contracts N-node topology into triad and unified models."""
    consolidator = ModelConsolidator()
    nodes = [
        {"id": "agent_alpha", "role": "proponent"},
        {"id": "agent_alpha_helper", "role": "proponent"},
        {"id": "agent_beta", "role": "opponent"},
        {"id": "judge_1", "role": "validator"},
        {"id": "judge_2", "role": "validator"},
    ]
    edges = [
        ("agent_alpha", "agent_beta"),
        ("agent_alpha_helper", "agent_beta"),
        ("agent_beta", "judge_1"),
        ("judge_1", "agent_alpha"),
    ]

    # Consolidate to Triad
    triad = consolidator.consolidate_to_triad(nodes, edges)
    assert triad.mode == "triad"
    assert len(triad.macro_nodes) == 3
    assert triad.node_mapping["agent_alpha"] == "macro_proponent"
    assert triad.node_mapping["agent_alpha_helper"] == "macro_proponent"
    assert triad.node_mapping["agent_beta"] == "macro_opponent"
    assert triad.node_mapping["judge_1"] == "macro_validator"
    assert triad.node_mapping["judge_2"] == "macro_validator"

    # Consolidate to Unified
    unified = consolidator.consolidate_to_unified(nodes)
    assert unified.mode == "unified"
    assert len(unified.macro_nodes) == 1
    assert unified.node_mapping["agent_alpha"] == "unified_macro_engine"


def test_federated_orchestrator_initialization(tmp_path):
    """MultiInstanceOrchestrator initializes correctly from federated mesh configuration."""
    fed_path = tmp_path / "fed.json"
    fed_data = {
        "federation_name": "test_fed",
        "version": "2.0.0",
        "description": "Test federation",
        "nodes": [
            {
                "node_id": "node_http",
                "role": "Remote Validator",
                "transport": "http",
                "endpoint_url": "http://127.0.0.1:8080/eval",
                "outbound_channels": [],
            }
        ],
        "routing_rules": [],
    }
    with open(fed_path, "w", encoding="utf-8") as f:
        json.dump(fed_data, f)

    orch = MultiInstanceOrchestrator.from_config_file(
        config_path=fed_path,
        datasets_dir=tmp_path / "datasets",
        workspace_dir=tmp_path,
    )
    assert orch.federated_config is not None
    assert orch.validate_static() is True


@patch("httpx.Client.post")
def test_http_node_runner_local(mock_post):
    """HttpNodeRunner interacts strictly with local endpoint."""
    fake_response = MagicMock()
    fake_response.json.return_value = {
        "sender": "node_http",
        "recipient": "node_core",
        "turn": 1,
        "volatileState": {"internalTimeTick": 1},
        "payload": {
            "naturalLanguage": "Local HTTP response",
            "abstractSyntaxTree": "ResponseAST",
            "smtConstraints": {"declarations": [], "assertions": ["(assert true)"], "z3Status": "sat"},
        },
    }
    mock_post.return_value = fake_response

    runner = HttpNodeRunner(
        node_id="node_http",
        endpoint_url="http://127.0.0.1:8080/evaluate",
        role="validator",
    )
    in_env = Envelope(
        sender="node_core",
        recipient="node_http",
        turn=1,
        volatileState=VolatileState(internalTimeTick=1),
        payload=Payload(
            naturalLanguage="Local query",
            abstractSyntaxTree="QueryAST",
            smtConstraints=SMTConstraints(),
        ),
    )
    out_env = runner.execute_turn(in_env)
    assert out_env.sender == "node_http"
    assert out_env.payload.naturalLanguage == "Local HTTP response"
    mock_post.assert_called_once()
    assert mock_post.call_args[0][0] == "http://127.0.0.1:8080/evaluate"


# ---------------------------------------------------------------------------
# 9. Transport wiring executed by the orchestrator itself
# ---------------------------------------------------------------------------

def test_local_transport_resolves_cache_path_on_workspace(tmp_path):
    """A local node reads its specifications from local_cache_path anchored on the workspace."""
    res = resolve_node_transport(
        {"transport": "local", "node_id": "n", "local_cache_path": "datasets/n"},
        workspace_dir=tmp_path,
    )
    assert res == (tmp_path / "datasets" / "n").resolve()
    resolver = NodeResolver(workspace_dir=tmp_path)
    resolved = resolver.resolve({"node_id": "n", "transport": "local", "local_cache_path": "datasets/n"})
    assert resolved["node_dir"] == (tmp_path / "datasets" / "n").resolve()


@patch("subprocess.run")
def test_git_cache_path_is_anchored_on_workspace(mock_subproc, tmp_path):
    """The git cache of a node is a workspace path, not a path relative to the current folder."""
    res = resolve_node_transport(
        {
            "transport": "git",
            "node_id": "x",
            "remote_source": "http://127.0.0.1:9418/r.git",
            "local_cache_path": "cache/nodes/x",
        },
        workspace_dir=tmp_path,
    )
    assert res == (tmp_path / "cache" / "nodes" / "x").resolve()
    clone_cmd = mock_subproc.call_args[0][0]
    assert clone_cmd[-1] == str(tmp_path / "cache" / "nodes" / "x")


def test_federated_node_auth_token_env_roundtrip():
    """A node declares the environment variable carrying its bearer token, never the token."""
    node = FederatedNodeConfig.from_dict(
        {
            "node_id": "api_node",
            "role": "validator",
            "transport": "http",
            "endpoint_url": "http://127.0.0.1:8099/api",
            "auth_method": "bearer_token",
            "auth_token_env": "LINGOLOGOS_HTTP_TOKEN",
        }
    )
    assert node.auth_token_env == "LINGOLOGOS_HTTP_TOKEN"
    assert node.to_dict()["auth_token_env"] == "LINGOLOGOS_HTTP_TOKEN"


def test_federated_node_auth_token_env_rejects_invalid():
    """Empty token variables and token variables without bearer auth are rejected."""
    with pytest.raises(SchemaValidationError, match="auth_token_env"):
        FederatedNodeConfig.from_dict(
            {
                "node_id": "api_node",
                "role": "validator",
                "transport": "http",
                "endpoint_url": "http://127.0.0.1:8099/api",
                "auth_method": "bearer_token",
                "auth_token_env": "   ",
            }
        )
    with pytest.raises(SchemaValidationError, match="requires auth_method"):
        FederatedNodeConfig.from_dict(
            {
                "node_id": "api_node",
                "role": "validator",
                "transport": "http",
                "endpoint_url": "http://127.0.0.1:8099/api",
                "auth_token_env": "LINGOLOGOS_HTTP_TOKEN",
            }
        )


@patch("httpx.Client.post")
def test_http_node_runner_sends_bearer_token(mock_post):
    """The bearer token reaches the remote endpoint in the Authorization header."""
    fake_response = MagicMock()
    fake_response.json.return_value = {
        "sender": "node_http",
        "recipient": "node_core",
        "turn": 1,
        "volatileState": {"internalTimeTick": 1},
        "payload": {
            "naturalLanguage": "Authenticated response",
            "abstractSyntaxTree": "ResponseAST",
            "smtConstraints": {"declarations": [], "assertions": ["(assert true)"], "z3Status": "sat"},
        },
    }
    mock_post.return_value = fake_response

    runner = HttpNodeRunner(
        node_id="node_http",
        endpoint_url="http://127.0.0.1:8099/evaluate",
        role="validator",
        auth_method="bearer_token",
        auth_token="secret-token-42",
    )
    out_env = runner.execute_turn(_envelope("node_core", "node_http", 1, "Signed query"))

    assert out_env.payload.naturalLanguage == "Authenticated response"
    headers = mock_post.call_args.kwargs["headers"]
    assert headers["Authorization"] == "Bearer secret-token-42"


@patch("httpx.Client.post")
def test_http_node_runner_without_token_sends_no_authorization(mock_post):
    """A node without bearer authentication never emits an Authorization header."""
    fake_response = MagicMock()
    fake_response.json.return_value = {
        "sender": "node_http",
        "recipient": "node_core",
        "turn": 1,
        "volatileState": {"internalTimeTick": 1},
        "payload": {
            "naturalLanguage": "Anonymous response",
            "abstractSyntaxTree": "ResponseAST",
            "smtConstraints": {"declarations": [], "assertions": ["(assert true)"], "z3Status": "sat"},
        },
    }
    mock_post.return_value = fake_response

    runner = HttpNodeRunner(
        node_id="node_http",
        endpoint_url="http://127.0.0.1:8099/evaluate",
        role="validator",
    )
    runner.execute_turn(_envelope("node_core", "node_http", 1, "Unsigned query"))

    headers = mock_post.call_args.kwargs["headers"]
    assert "Authorization" not in headers


def test_orchestrator_reads_http_token_from_environment(tmp_path, monkeypatch):
    """Phase 02 injects into the HTTP runner the token named by auth_token_env."""
    fed_path = tmp_path / "fed_http.json"
    fed_path.write_text(
        json.dumps(
            {
                "federation_name": "http_federation",
                "version": "2.0.0",
                "description": "Bearer token wiring federation",
                "nodes": [
                    {
                        "node_id": "node_http",
                        "role": "validator",
                        "transport": "http",
                        "endpoint_url": "http://127.0.0.1:8099/evaluate",
                        "auth_method": "bearer_token",
                        "auth_token_env": "LINGOLOGOS_TEST_HTTP_TOKEN",
                    }
                ],
                "routing_rules": [],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("LINGOLOGOS_TEST_HTTP_TOKEN", "secret-token-42")

    orch = MultiInstanceOrchestrator.from_config_file(
        config_path=fed_path,
        datasets_dir=tmp_path / "datasets",
        runtime_state_dir=tmp_path / "runtime",
        workspace_dir=tmp_path,
    )
    orch.validate_static()
    orch.initialize_processes()

    runner = orch.runners["node_http"]
    assert isinstance(runner, HttpNodeRunner)
    assert runner.auth_method == "bearer_token"
    assert runner.auth_token == "secret-token-42"


def test_orchestrator_executes_git_node_after_cloning_source(tmp_path):
    """Phase 01 synchronizes the git source of a node and Phase 03 executes the cloned specs."""
    repo_dir = tmp_path / "spec_repo"
    _write_spec_files(repo_dir / "specs" / "n_git", "n_git")
    for cmd in (
        ["git", "init", "-q", "-b", "main", str(repo_dir)],
        ["git", "-C", str(repo_dir), "add", "-A"],
        [
            "git", "-C", str(repo_dir),
            "-c", "user.name=LingoLOGOS Tests", "-c", "user.email=tests@lingologos.invalid",
            "commit", "-q", "-m", "Add remote node specification",
        ],
    ):
        subprocess.run(cmd, check=True, capture_output=True, text=True)

    _write_spec_files(tmp_path / "local_specs", "n_local")

    fed_path = tmp_path / "fed_git.json"
    fed_path.write_text(
        json.dumps(
            {
                "federation_name": "git_federation",
                "version": "2.0.0",
                "description": "Git transport end to end",
                "nodes": [
                    {
                        "node_id": "n_local",
                        "role": "proponent",
                        "transport": "local",
                        "local_cache_path": "local_specs",
                        "outbound_channels": ["n_git"],
                    },
                    {
                        "node_id": "n_git",
                        "role": "validator",
                        "transport": "git",
                        "remote_source": str(repo_dir).replace("\\", "/"),
                        "branch": "main",
                        "local_cache_path": "cache_repo/n_git",
                        "specs": {
                            "grammar": "specs/n_git/grammar.gf",
                            "ontology": "specs/n_git/ontology.smt2",
                            "seed": "specs/n_git/seed_payload.json",
                        },
                        "outbound_channels": ["n_local"],
                    },
                ],
                "routing_rules": [
                    {"source": "n_local", "target": "n_git", "condition": "REQUEST_VALIDATION"},
                    {"source": "n_git", "target": "n_local", "condition": "VERDICT_RETURNED"},
                ],
            }
        ),
        encoding="utf-8",
    )

    orch = MultiInstanceOrchestrator.from_config_file(
        config_path=fed_path,
        datasets_dir=tmp_path / "datasets",
        runtime_state_dir=tmp_path / "runtime",
        workspace_dir=tmp_path,
    )
    assert orch.validate_static() is True

    cloned_spec = (tmp_path / "cache_repo" / "n_git" / "specs" / "n_git").resolve()
    assert (cloned_spec / "ontology.smt2").is_file()
    assert orch.node_dataset_dirs["n_git"] == cloned_spec

    result = orch.run_session()
    assert result.success is True
    assert result.total_turns >= 2
    assert result.termination_reason == "SAT_CONVERGENCE"
    assert orch.runners["n_git"].node_dir == cloned_spec
    assert orch.solver_pool.is_shutdown() is True


def test_orchestrator_rejects_invalid_ontology_through_solver_pool(tmp_path):
    """The pooled Z3 workers report the ontology syntax error during Phase 01."""
    node_dir = tmp_path / "broken_specs"
    _write_spec_files(node_dir, "n_broken")
    (node_dir / "ontology.smt2").write_text("(assert (= 1 2", encoding="utf-8")

    fed_path = tmp_path / "fed_broken.json"
    fed_path.write_text(
        json.dumps(
            {
                "federation_name": "broken_federation",
                "version": "2.0.0",
                "description": "Ontology with broken syntax",
                "nodes": [
                    {
                        "node_id": "n_broken",
                        "role": "proponent",
                        "transport": "local",
                        "local_cache_path": "broken_specs",
                    }
                ],
                "routing_rules": [],
            }
        ),
        encoding="utf-8",
    )
    orch = MultiInstanceOrchestrator.from_config_file(
        config_path=fed_path,
        datasets_dir=tmp_path / "datasets",
        runtime_state_dir=tmp_path / "runtime",
        workspace_dir=tmp_path,
    )
    with pytest.raises(StaticValidationError, match="Syntax error in ontology.smt2"):
        orch.validate_static()


def test_session_streams_to_event_bus_and_releases_solver_pool(tmp_path):
    """A session publishes one envelope per turn and releases the pooled Z3 workers."""
    orch = MultiInstanceOrchestrator.from_config_file(
        config_path=pathlib.Path("config/topology.json"),
        datasets_dir=pathlib.Path("datasets"),
        runtime_state_dir=tmp_path / "runtime",
        workspace_dir=pathlib.Path("."),
    )

    received: list = []
    orch.event_bus.subscribe("*", received.append)

    batch_sizes: list = []
    original_batch = orch.solver_pool.verify_batch

    def counting_batch(tasks):
        batch_sizes.append(len(tasks))
        return original_batch(tasks)

    orch.solver_pool.verify_batch = counting_batch

    result = orch.run_session()

    assert batch_sizes == [len(orch.config.instances)]
    assert result.total_turns > 0
    assert len(orch.event_bus.get_history()) == result.total_turns
    assert len(received) == result.total_turns
    assert orch.solver_pool.is_shutdown() is True


def test_consolidation_representatives_and_degenerate_triad():
    """Contraction records the representative node of each macro node and drops empty categories."""
    consolidator = ModelConsolidator()
    nodes = [
        {"id": "n1", "role": "proponent"},
        {"id": "n2", "role": "critic"},
        {"id": "n3", "role": "arbiter"},
    ]

    triad = consolidator.consolidate_to_triad(nodes, [])
    assert triad.representatives() == {
        "macro_proponent": "n1",
        "macro_opponent": "n2",
        "macro_validator": "n3",
    }
    topo = triad.to_topology_config(session_id="s-triad", max_turns=4)
    assert [i.executable for i in topo.instances] == ["lllayer"] * 3
    assert len(topo.routingMatrix) == 3

    dyad = consolidator.consolidate_to_triad(
        [{"id": "n1", "role": "proponent"}, {"id": "n2", "role": "validator"}], []
    )
    assert len(dyad.macro_nodes) == 2
    assert dyad.representatives() == {"macro_proponent": "n1", "macro_validator": "n2"}
    assert {(r["from"], tuple(r["to"])) for r in dyad.macro_routing} == {
        ("macro_proponent", ("macro_validator",)),
        ("macro_validator", ("macro_proponent",)),
    }

    unified = consolidator.consolidate_to_unified(nodes)
    assert unified.representatives() == {"unified_macro_engine": "n1"}
    assert unified.macro_routing == []
    unified_topo = unified.to_topology_config(session_id="s-unified", max_turns=4)
    assert len(unified_topo.instances) == 1
    assert unified_topo.routingMatrix == []


def test_consolidated_session_runs_on_representative_datasets(tmp_path):
    """Macro nodes of a consolidated topology run on the dataset of the node they represent."""
    orch = MultiInstanceOrchestrator.from_config_file(
        config_path=pathlib.Path("config/topology.json"),
        datasets_dir=pathlib.Path("datasets"),
        runtime_state_dir=tmp_path / "runtime",
        workspace_dir=pathlib.Path("."),
    )
    model = ModelConsolidator().consolidate_topology(orch.config, mode="triad")
    orch.set_representative_nodes(model.representatives())
    orch.config = model.to_topology_config(session_id="consolidated-session", max_turns=orch.config.maxTurns)

    assert orch.validate_static() is True
    expected = {
        "macro_proponent": "agente_alfa",
        "macro_opponent": "agente_beta",
        "macro_validator": "arbitro_logico",
    }
    for macro_id, physical_id in expected.items():
        assert orch.node_dataset_dirs[macro_id] == (pathlib.Path("datasets") / physical_id).resolve()
    orch.initialize_processes()
    assert orch.runners["macro_opponent"].node_dir == (pathlib.Path("datasets") / "agente_beta").resolve()


def test_cli_consolidate_triad_executes_session(tmp_path, monkeypatch, capsys):
    """The documented --consolidate triad command runs the consolidated session end to end."""
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "lingologos-orchestrator",
            "--config", "config/topology.json",
            "--datasets", "datasets/",
            "--runtime-state", str(tmp_path / "runtime"),
            "--consolidate", "triad",
        ],
    )
    exit_code = cli.main()
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "consolidated" in output
    assert "Termination reason: SAT_CONVERGENCE" in output
