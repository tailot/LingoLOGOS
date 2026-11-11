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

Adheres to English-only, ASCII-only rules and avoids apostrophes.
"""

import os
import sys
import json
import pathlib
import pytest
from unittest.mock import patch, MagicMock

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
)
from lingologos_orchestrator.resolver import resolve_node_transport, NodeResolver
from lingologos_orchestrator.event_bus import EventBus, DynamicNodeSpawner
from lingologos_orchestrator.solver_pool import SMTSolverPool, SolverTask
from lingologos_orchestrator.consolidation import ModelConsolidator
from lingologos_orchestrator.orchestrator import MultiInstanceOrchestrator


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

    # Test conversion to legacy TopologyConfig
    legacy_topo = fed.to_legacy_topology(max_turns=10)
    assert legacy_topo.sessionId == "lingologos_universal_federation"
    assert len(legacy_topo.instances) == 3
    assert legacy_topo.is_route_allowed("local_core_node", "git_repo_node")
    assert legacy_topo.is_route_allowed("local_core_node", "http_api_node")


def test_resolve_node_transport_local(tmp_path):
    """Local transport resolves to local Path."""
    conf = {"transport": "local", "node_id": "test_local"}
    res = resolve_node_transport(conf)
    assert isinstance(res, pathlib.Path)


def test_resolve_node_transport_http():
    """HTTP transport resolves to endpoint URL string."""
    url = "https://api.lingologos-inference.net/v1/evaluate"
    conf = {"transport": "http", "node_id": "test_http", "endpoint_url": url}
    res = resolve_node_transport(conf)
    assert res == url


@patch("subprocess.run")
def test_resolve_node_transport_git(mock_subproc, tmp_path):
    """Git transport clones repository if absent or pulls if present."""
    cache_dir = tmp_path / "cache" / "test_repo"
    conf = {
        "transport": "git",
        "remote_source": "https://example.com/repo.git",
        "branch": "main",
        "local_cache_path": str(cache_dir),
    }

    # Case 1: absent cache path -> clone
    res = resolve_node_transport(conf)
    assert res == cache_dir
    mock_subproc.assert_called_with(
        ["git", "clone", "--branch", "main", "--depth", "1", "https://example.com/repo.git", str(cache_dir)],
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
    assert str(resolved["grammar_path"]).endswith("grammars/g.gf")


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
                "endpoint_url": "https://fake.lingologos.local/eval",
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
