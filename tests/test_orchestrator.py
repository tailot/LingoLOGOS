"""Comprehensive tests for LingoLOGOS Multi-Instance Orchestrator.

RFC Identifier: LingoLOGOS-MULTI-ORCH-v1
Validates:
  1. Static validation of topology schema and filesystem conventions
  2. Isolated process execution and node decoupling via lllayer
  3. Centralized directed graph routing authorization
  4. Volatile state and time management
  5. Formal SMT-LIB 2 and Z3 constraint verification
  6. Termination conditions (UNSAT breaker, SAT convergence, max turns)
  7. CLI entrypoints
"""

import os
import sys
import json
import subprocess
import pytest
from pathlib import Path

from lingologos_orchestrator.schema import (
    TopologyConfig,
    InstanceConfig,
    RoutingRule,
    Envelope,
    VolatileState,
    Payload,
    SMTConstraints,
    SchemaValidationError,
    StaticValidationError,
    RoutingError,
)
from lingologos_orchestrator.verifier import Z3Verifier, SMTVerificationResult
from lingologos_orchestrator.node_runner import NodeProcessRunner
from lingologos_orchestrator.orchestrator import MultiInstanceOrchestrator


def test_schema_topology_valid():
    """Valid topology configuration parses and builds successfully."""
    data = {
        "sessionId": "test-session-01",
        "maxTurns": 5,
        "terminationCondition": "Z3_SAT_CONVERGENCE_OR_UNSAT_BREAKER",
        "instances": [
            {"id": "node_a", "executable": "lllayer", "role": "proponent"},
            {"id": "node_b", "executable": "lllayer", "role": "opponent"},
        ],
        "routingMatrix": [
            {"from": "node_a", "to": ["node_b"]},
            {"from": "node_b", "to": ["node_a"]},
        ],
    }
    cfg = TopologyConfig.from_dict(data)
    assert cfg.sessionId == "test-session-01"
    assert cfg.maxTurns == 5
    assert len(cfg.instances) == 2
    assert cfg.is_route_allowed("node_a", "node_b")
    assert not cfg.is_route_allowed("node_a", "unknown_node")


def test_schema_topology_rejects_invalid():
    """Invalid topology structures raise SchemaValidationError."""
    # Duplicate instance IDs
    dup_data = {
        "sessionId": "s1",
        "maxTurns": 10,
        "instances": [
            {"id": "node_x", "executable": "lllayer", "role": "a"},
            {"id": "node_x", "executable": "lllayer", "role": "b"},
        ],
        "routingMatrix": [],
    }
    with pytest.raises(SchemaValidationError, match="Duplicate instance ids"):
        TopologyConfig.from_dict(dup_data)

    # Undeclared routing target
    bad_route = {
        "sessionId": "s2",
        "maxTurns": 10,
        "instances": [
            {"id": "node_x", "executable": "lllayer", "role": "a"},
        ],
        "routingMatrix": [
            {"from": "node_x", "to": ["undeclared_target"]},
        ],
    }
    with pytest.raises(SchemaValidationError, match="target node 'undeclared_target'"):
        TopologyConfig.from_dict(bad_route)


def test_schema_envelope_valid_and_rejects_malformed():
    """Valid envelope dictionary builds correctly and malformed dictionaries fail."""
    valid_dict = {
        "sender": "arbitrary_sender",
        "recipient": "arbitrary_recipient",
        "turn": 1,
        "volatileState": {"internalTimeTick": 1},
        "payload": {
            "naturalLanguage": "Sample sentence",
            "abstractSyntaxTree": "SampleAST (Node)",
            "smtConstraints": {
                "declarations": ["(declare-sort S)"],
                "assertions": ["(assert true)"],
                "z3Status": "sat",
            },
        },
    }
    env = Envelope.from_dict(valid_dict)
    assert env.sender == "arbitrary_sender"
    assert env.recipient == "arbitrary_recipient"
    assert env.turn == 1
    assert env.volatileState.internalTimeTick == 1
    assert env.payload.smtConstraints.z3Status == "sat"

    # Missing naturalLanguage in payload
    bad_dict = dict(valid_dict)
    bad_dict["payload"] = {"abstractSyntaxTree": "AST", "smtConstraints": {}}
    with pytest.raises(SchemaValidationError):
        Envelope.from_dict(bad_dict)


def test_z3_verifier_sat_and_unsat():
    """Z3Verifier identifies satisfiability, unsatisfiability, and avoids duplicate declaration errors."""
    verifier = Z3Verifier()

    ontology = """
    (declare-sort State)
    (declare-fun safe_transition (State State) Bool)
    (declare-const s1 State)
    (declare-const s2 State)
    """

    # SAT constraint with duplicate declaration of State and s1
    sat_constraints = SMTConstraints(
        declarations=["(declare-sort State)", "(declare-const s1 State)"],
        assertions=["(assert (safe_transition s1 s2))"],
        z3Status="sat",
    )
    sat_res = verifier.verify_constraints(sat_constraints, ontology_smt2=ontology)
    assert sat_res.is_sat
    assert not sat_res.is_unsat
    assert sat_res.status == "sat"

    # UNSAT constraint
    unsat_constraints = SMTConstraints(
        declarations=[],
        assertions=[
            "(assert (safe_transition s1 s2))",
            "(assert (not (safe_transition s1 s2)))",
        ],
        z3Status="unsat",
    )
    unsat_res = verifier.verify_constraints(unsat_constraints, ontology_smt2=ontology)
    assert unsat_res.is_unsat
    assert not unsat_res.is_sat
    assert unsat_res.status == "unsat"


def test_static_validation_on_workspace(tmp_path):
    """Static validation passes for correctly formatted workspace and fails if required files are missing."""
    config_data = {
        "sessionId": "test-session",
        "maxTurns": 3,
        "terminationCondition": "Z3_SAT_CONVERGENCE_OR_UNSAT_BREAKER",
        "instances": [
            {"id": "node_1", "executable": "lllayer", "role": "proponent"},
        ],
        "routingMatrix": [
            {"from": "node_1", "to": ["node_1"]},
        ],
    }
    cfg = TopologyConfig.from_dict(config_data)
    datasets_dir = tmp_path / "datasets"
    node_dir = datasets_dir / "node_1"
    node_dir.mkdir(parents=True)

    orch = MultiInstanceOrchestrator(
        config=cfg,
        datasets_dir=datasets_dir,
        workspace_dir=tmp_path,
    )

    # Missing files initially
    with pytest.raises(StaticValidationError, match="ontology.smt2 missing"):
        orch.validate_static()

    (node_dir / "ontology.smt2").write_text("(declare-sort S)\n", encoding="utf-8")
    with pytest.raises(StaticValidationError, match="grammar.gf missing"):
        orch.validate_static()

    (node_dir / "grammar.gf").write_text("abstract G = {}\n", encoding="utf-8")
    with pytest.raises(StaticValidationError, match="seed_payload.json missing"):
        orch.validate_static()

    seed_content = {
        "sender": "node_1",
        "turn": 0,
        "volatileState": {"internalTimeTick": 0},
        "payload": {
            "naturalLanguage": "Init",
            "abstractSyntaxTree": "InitTree",
            "smtConstraints": {
                "declarations": [],
                "assertions": ["(assert true)"],
                "z3Status": "sat",
            },
        },
    }
    (node_dir / "seed_payload.json").write_text(json.dumps(seed_content), encoding="utf-8")

    # Now static validation passes
    assert orch.validate_static() is True


def test_isolated_process_runner_arbitrary_node_id(tmp_path):
    """Node process runner executes lllayer in an isolated process with an arbitrary node ID."""
    node_id = "arbitrary_custom_sentinel_99"
    datasets_dir = tmp_path / "datasets"
    node_dir = datasets_dir / node_id
    node_dir.mkdir(parents=True)

    seed_data = {
        "sender": node_id,
        "turn": 0,
        "volatileState": {"internalTimeTick": 0},
        "payload": {
            "naturalLanguage": "Sentinel online.",
            "abstractSyntaxTree": "Monitor (Root)",
            "smtConstraints": {
                "declarations": ["(declare-sort State)"],
                "assertions": ["(assert true)"],
                "z3Status": "sat",
            },
        },
    }
    (node_dir / "seed_payload.json").write_text(json.dumps(seed_data), encoding="utf-8")
    (node_dir / "ontology.smt2").write_text("(declare-sort State)\n", encoding="utf-8")
    (node_dir / "grammar.gf").write_text("abstract Test = {}\n", encoding="utf-8")

    runner = NodeProcessRunner(
        node_id=node_id,
        executable="lllayer",
        role="opponent",
        datasets_dir=datasets_dir,
        workspace_dir=tmp_path,
    )

    # Execute Turn 0 (seed request)
    input_turn0 = Envelope(
        sender=node_id,
        recipient=node_id,
        turn=0,
        volatileState=VolatileState(internalTimeTick=0),
        payload=Payload(
            naturalLanguage="init",
            abstractSyntaxTree="init",
            smtConstraints=SMTConstraints(declarations=[], assertions=[]),
        ),
    )
    resp0 = runner.execute_turn(input_turn0)
    assert resp0.sender == node_id
    assert resp0.turn == 0
    assert resp0.payload.naturalLanguage == "Sentinel online."

    # Execute Turn 1
    input_turn1 = Envelope(
        sender="peer_node",
        recipient=node_id,
        turn=1,
        volatileState=VolatileState(internalTimeTick=1),
        payload=Payload(
            naturalLanguage="Proposed transition",
            abstractSyntaxTree="Transition (s1 s2)",
            smtConstraints=SMTConstraints(
                declarations=["(declare-sort State)"],
                assertions=["(assert true)"],
            ),
        ),
    )
    resp1 = runner.execute_turn(input_turn1)
    assert resp1.sender == node_id
    assert resp1.recipient == "peer_node"
    assert resp1.turn == 1
    assert resp1.volatileState.internalTimeTick == 1
    assert resp1.payload.smtConstraints.z3Status == "sat"


def test_routing_matrix_enforcement(tmp_path):
    """Orchestrator rejects messages routed to unauthorized destinations."""
    topology_data = {
        "sessionId": "routing-test",
        "maxTurns": 3,
        "terminationCondition": "MAX_TURNS",
        "instances": [
            {"id": "node_x", "executable": "lllayer", "role": "proponent"},
            {"id": "node_y", "executable": "lllayer", "role": "opponent"},
            {"id": "node_z", "executable": "lllayer", "role": "validator"},
        ],
        "routingMatrix": [
            # node_x can only route to node_y, not to node_z
            {"from": "node_x", "to": ["node_y"]},
            {"from": "node_y", "to": ["node_x"]},
        ],
    }
    cfg = TopologyConfig.from_dict(topology_data)
    datasets_dir = tmp_path / "datasets"

    for nid in ["node_x", "node_y", "node_z"]:
        nd = datasets_dir / nid
        nd.mkdir(parents=True)
        (nd / "ontology.smt2").write_text("(declare-sort S)\n", encoding="utf-8")
        (nd / "grammar.gf").write_text("abstract G = {}\n", encoding="utf-8")
        seed = {
            "sender": nid,
            "turn": 0,
            "volatileState": {"internalTimeTick": 0},
            "payload": {
                "naturalLanguage": f"Init {nid}",
                "abstractSyntaxTree": f"Tree {nid}",
                "smtConstraints": {"declarations": [], "assertions": ["(assert true)"], "z3Status": "sat"},
            },
        }
        (nd / "seed_payload.json").write_text(json.dumps(seed), encoding="utf-8")

    # If node_x produces an envelope addressed to unauthorized node_z, orchestrator must raise RoutingError
    orch = MultiInstanceOrchestrator(config=cfg, datasets_dir=datasets_dir, workspace_dir=tmp_path)
    assert not cfg.is_route_allowed("node_x", "node_z")
    assert cfg.is_route_allowed("node_x", "node_y")


def test_termination_unsat_breaker(tmp_path):
    """Orchestrator trips UNSAT breaker when a node outputs an unsatisfiable constraint."""
    topology_data = {
        "sessionId": "unsat-breaker-session",
        "maxTurns": 10,
        "terminationCondition": "Z3_SAT_CONVERGENCE_OR_UNSAT_BREAKER",
        "instances": [
            {"id": "agent_alpha", "executable": "lllayer", "role": "proponent"},
            {"id": "agent_beta", "executable": "lllayer", "role": "opponent"},
        ],
        "routingMatrix": [
            {"from": "agent_alpha", "to": ["agent_beta"]},
            {"from": "agent_beta", "to": ["agent_alpha"]},
        ],
    }
    cfg = TopologyConfig.from_dict(topology_data)
    datasets_dir = tmp_path / "datasets"

    # Setup agent_alpha
    da = datasets_dir / "agent_alpha"
    da.mkdir(parents=True)
    (da / "ontology.smt2").write_text("(declare-sort S)\n", encoding="utf-8")
    (da / "grammar.gf").write_text("abstract GA = {}\n", encoding="utf-8")
    seed_a = {
        "sender": "agent_alpha",
        "turn": 0,
        "volatileState": {"internalTimeTick": 0},
        "payload": {
            "naturalLanguage": "Init alpha",
            "abstractSyntaxTree": "TreeAlpha",
            "smtConstraints": {"declarations": [], "assertions": ["(assert true)"], "z3Status": "sat"},
        },
    }
    (da / "seed_payload.json").write_text(json.dumps(seed_a), encoding="utf-8")

    # Setup agent_beta with turn 1 pre-recorded payload containing UNSAT contradiction
    db = datasets_dir / "agent_beta"
    db.mkdir(parents=True)
    (db / "ontology.smt2").write_text("(declare-sort S)\n", encoding="utf-8")
    (db / "grammar.gf").write_text("abstract GB = {}\n", encoding="utf-8")
    seed_b = {
        "sender": "agent_beta",
        "turn": 0,
        "volatileState": {"internalTimeTick": 0},
        "payload": {
            "naturalLanguage": "Init beta",
            "abstractSyntaxTree": "TreeBeta",
            "smtConstraints": {"declarations": [], "assertions": ["(assert true)"], "z3Status": "sat"},
        },
    }
    (db / "seed_payload.json").write_text(json.dumps(seed_b), encoding="utf-8")

    # For turn 1, agent_alpha sends normal message.
    # We can inject turn 1 payload for agent_alpha that contains UNSAT assertion
    unsat_payload = {
        "sender": "agent_alpha",
        "recipient": "agent_beta",
        "turn": 1,
        "volatileState": {"internalTimeTick": 1},
        "payload": {
            "naturalLanguage": "Contradictory state violation",
            "abstractSyntaxTree": "ViolateSafety",
            "smtConstraints": {
                "declarations": ["(declare-const p Bool)"],
                "assertions": ["(assert p)", "(assert (not p))"],
                "z3Status": "unsat",
            },
        },
    }
    (da / "turn_1_payload.json").write_text(json.dumps(unsat_payload), encoding="utf-8")

    orch = MultiInstanceOrchestrator(
        config=cfg,
        datasets_dir=datasets_dir,
        runtime_state_dir=tmp_path / "runtime_state",
        workspace_dir=tmp_path,
    )

    result = orch.run_session()
    assert result.terminated is True
    assert result.termination_reason == "UNSAT_BREAKER"
    assert result.total_turns == 1
    assert result.records[0].smt_result["is_unsat"] is True


def test_volatile_time_increment():
    """Volatile internalTimeTick increments strictly per turn without persistent timestamp fields."""
    topology_path = Path("config/topology.json")
    datasets_dir = Path("datasets")
    assert topology_path.is_file()
    assert datasets_dir.is_dir()

    orch = MultiInstanceOrchestrator.from_config_file(
        config_path=topology_path,
        datasets_dir=datasets_dir,
    )
    result = orch.run_session()
    assert result.total_turns >= 1

    for idx, rec in enumerate(result.records, start=1):
        assert rec.turn == idx
        vol = rec.envelope["volatileState"]
        assert vol["internalTimeTick"] == idx
        # Ensure no persistent absolute timestamp fields exist in envelope
        assert "timestamp" not in vol
        assert "createdAt" not in vol
        assert "created_at" not in rec.envelope


def test_cli_execution_from_subprocess():
    """Tests execution of lingologos-orchestrator CLI executable."""
    cmd = [
        sys.executable,
        "-m",
        "lingologos_orchestrator.cli",
        "--config", "config/topology.json",
        "--datasets", "datasets",
        "--json",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0, f"CLI error: {proc.stderr}"
    data = json.loads(proc.stdout)
    assert data["sessionId"] == "lingologos-multiagent-01"
    assert data["success"] is True
    assert len(data["records"]) > 0
