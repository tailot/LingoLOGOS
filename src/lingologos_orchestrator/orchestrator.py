"""Core Multi-Instance Orchestrator for LingoLOGOS (lllayer).

RFC Identifier: LingoLOGOS-MULTI-ORCH-v1
Implements the 4-phase lifecycle:
  Phase 01: Static Validation
  Phase 02: Process Initialization
  Phase 03: Turn Loop with Centralized Directed-Graph Routing & Z3 SMT Verification
  Phase 04: Termination (SAT Convergence or UNSAT Breaker)
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from .schema import (
    TopologyConfig,
    InstanceConfig,
    Envelope,
    VolatileState,
    SMTConstraints,
    Payload,
    OrchestratorError,
    StaticValidationError,
    RoutingError,
    SchemaValidationError,
)
from .verifier import Z3Verifier, SMTVerificationResult
from .node_runner import NodeProcessRunner
from .topology_ctrl import TopologyCtrlRegister


@dataclass
class TurnRecord:
    """Record of a single turn execution in the orchestrator."""
    turn: int
    sender: str
    recipient: str
    envelope: Dict[str, Any]
    smt_result: Dict[str, Any]
    routing_authorized: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "turn": self.turn,
            "sender": self.sender,
            "recipient": self.recipient,
            "envelope": self.envelope,
            "smt_result": self.smt_result,
            "routing_authorized": self.routing_authorized,
        }


@dataclass
class ExecutionResult:
    """Result of an orchestrator session execution."""
    sessionId: str
    total_turns: int
    terminated: bool
    termination_reason: str
    records: List[TurnRecord] = field(default_factory=list)
    initial_states: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    success: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sessionId": self.sessionId,
            "total_turns": self.total_turns,
            "terminated": self.terminated,
            "termination_reason": self.termination_reason,
            "success": self.success,
            "initial_states": self.initial_states,
            "records": [r.to_dict() for r in self.records],
        }


class MultiInstanceOrchestrator:
    """Centralized message broker and formal orchestrator for distributed LingoLOGOS nodes."""

    def __init__(
        self,
        config: TopologyConfig,
        datasets_dir: Path,
        runtime_state_dir: Optional[Path] = None,
        workspace_dir: Optional[Path] = None,
    ):
        self.config = config
        self.datasets_dir = Path(datasets_dir).resolve()
        self.workspace_dir = Path(workspace_dir).resolve() if workspace_dir else Path.cwd()
        self.runtime_state_dir = (
            Path(runtime_state_dir).resolve()
            if runtime_state_dir
            else self.workspace_dir / "runtime_state"
        )
        self.verifier = Z3Verifier()
        self.runners: Dict[str, NodeProcessRunner] = {}
        self.ontologies: Dict[str, str] = {}
        self.grammars: Dict[str, str] = {}
        self.initial_envelopes: Dict[str, Envelope] = {}
        self.topology_ctrl_by_node: Dict[str, TopologyCtrlRegister] = {}

    @classmethod
    def from_config_file(
        cls,
        config_path: Path,
        datasets_dir: Path,
        runtime_state_dir: Optional[Path] = None,
        workspace_dir: Optional[Path] = None,
    ) -> "MultiInstanceOrchestrator":
        """Builds an orchestrator from a topology JSON file."""
        cfg_file = Path(config_path).resolve()
        if not cfg_file.is_file():
            raise FileNotFoundError(f"Topology configuration file not found: {cfg_file}")
        with open(cfg_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        config = TopologyConfig.from_dict(data)
        return cls(
            config=config,
            datasets_dir=datasets_dir,
            runtime_state_dir=runtime_state_dir,
            workspace_dir=workspace_dir,
        )

    # -------------------------------------------------------------------------
    # Phase 01: Static Validation
    # -------------------------------------------------------------------------
    def validate_static(self) -> bool:
        """Validates schema, filesystem conventions, and SMT syntax for all nodes.

        Filesystem convention:
          /datasets/{node_id}/
            ├── ontology.smt2
            ├── grammar.gf
            └── seed_payload.json
        """
        if not self.datasets_dir.is_dir():
            raise StaticValidationError(f"Datasets directory not found: {self.datasets_dir}")

        for instance in self.config.instances:
            node_id = instance.id
            node_dir = self.datasets_dir / node_id

            if not node_dir.is_dir():
                raise StaticValidationError(
                    f"Dataset directory missing for node '{node_id}': expected {node_dir}"
                )

            ontology_file = node_dir / "ontology.smt2"
            if not ontology_file.is_file():
                raise StaticValidationError(
                    f"ontology.smt2 missing for node '{node_id}': expected {ontology_file}"
                )

            grammar_file = node_dir / "grammar.gf"
            if not grammar_file.is_file():
                raise StaticValidationError(
                    f"grammar.gf missing for node '{node_id}': expected {grammar_file}"
                )

            seed_file = node_dir / "seed_payload.json"
            if not seed_file.is_file():
                raise StaticValidationError(
                    f"seed_payload.json missing for node '{node_id}': expected {seed_file}"
                )

            # Read and validate seed payload schema
            try:
                with open(seed_file, "r", encoding="utf-8") as f:
                    seed_json = json.load(f)
                seed_envelope = Envelope.from_dict(seed_json)
                if seed_envelope.sender != node_id:
                    raise StaticValidationError(
                        f"Seed payload sender '{seed_envelope.sender}' does not match node id '{node_id}'."
                    )
                self.initial_envelopes[node_id] = seed_envelope
            except Exception as exc:
                raise StaticValidationError(
                    f"Invalid seed_payload.json for node '{node_id}': {exc}"
                ) from exc

            # Read ontology and grammar
            with open(ontology_file, "r", encoding="utf-8") as f:
                self.ontologies[node_id] = f.read()

            with open(grammar_file, "r", encoding="utf-8") as f:
                self.grammars[node_id] = f.read()

            # Verify ontology SMT syntax using Z3
            res = self.verifier.verify_constraints(
                constraints=SMTConstraints(declarations=[], assertions=[]),
                ontology_smt2=self.ontologies[node_id],
            )
            if res.status == "error":
                raise StaticValidationError(
                    f"Syntax error in ontology.smt2 for node '{node_id}': {res.error_message}"
                )

        return True

    # -------------------------------------------------------------------------
    # Phase 02: Process Initialization & Topology Negotiation
    # -------------------------------------------------------------------------
    def negotiate_topology(self) -> Dict[str, TopologyCtrlRegister]:
        """Negotiates dynamic board topology for all instances.

        Single-Chip / Standalone Mode (1 instance):
          TOPOLOGY_CTRL.STANDALONE = 1
          DATASET_ID = 0 (primary local dataset)

        Multi-Chip / Distributed Mode (N > 1 instances):
          TOPOLOGY_CTRL.STANDALONE = 0
          TOTAL_NODES = N
          DATASET_ID = assigned shard index (0..N-1)
          ROUTING_MASK = computed bitmask of reachable neighbors
        """
        total = len(self.config.instances)
        id_to_hw: Dict[str, int] = {
            inst.id: idx for idx, inst in enumerate(self.config.instances)
        }

        for idx, instance in enumerate(self.config.instances):
            hw_id = id_to_hw[instance.id]
            peers = total - 1

            # Compute routing mask
            routing_mask = 0
            for rule in self.config.routingMatrix:
                if rule.from_node == instance.id:
                    for target in rule.to_nodes:
                        if target in id_to_hw:
                            target_hw = id_to_hw[target]
                            routing_mask |= (1 << target_hw)

            reg = TopologyCtrlRegister.from_handshake(
                hardware_node_id=hw_id,
                discovered_peers_count=peers,
                dataset_id=idx,
                routing_mask=routing_mask,
            )
            self.topology_ctrl_by_node[instance.id] = reg

        return self.topology_ctrl_by_node

    def initialize_processes(self) -> None:
        """Initializes runners for isolated node processes and sets up volatile runtime state."""
        self.runtime_state_dir.mkdir(parents=True, exist_ok=True)
        self.negotiate_topology()

        for instance in self.config.instances:
            runner = NodeProcessRunner(
                node_id=instance.id,
                executable=instance.executable,
                role=instance.role,
                datasets_dir=self.datasets_dir,
                workspace_dir=self.workspace_dir,
            )
            reg = self.topology_ctrl_by_node.get(instance.id)
            if reg:
                runner.hardware_node_id = reg.node_id
                runner.probe_peers = reg.total_nodes - 1 if reg.total_nodes > 1 else 0
                runner.dataset_id = reg.dataset_id
                runner.topology_ctrl_reg = f"0x{reg.pack():08X}"
            self.runners[instance.id] = runner

    # -------------------------------------------------------------------------
    # Phase 03 & 04: Turn Loop & Centralized Routing & Termination
    # -------------------------------------------------------------------------
    def run_session(self) -> ExecutionResult:
        """Executes the complete session lifecycle across all turns."""
        # Phase 01: Static Validation
        self.validate_static()

        # Phase 02: Process Initialization
        self.initialize_processes()

        records: List[TurnRecord] = []
        initial_states_dict: Dict[str, Dict[str, Any]] = {
            nid: env.to_dict() for nid, env in self.initial_envelopes.items()
        }

        # Build ordered list of directed edges to route messages along
        edges: List[Tuple[str, str]] = []
        for rule in self.config.routingMatrix:
            for target in rule.to_nodes:
                edges.append((rule.from_node, target))

        if not edges:
            return ExecutionResult(
                sessionId=self.config.sessionId,
                total_turns=0,
                terminated=True,
                termination_reason="EMPTY_ROUTING_MATRIX",
                records=[],
                initial_states=initial_states_dict,
                success=True,
            )

        current_payload_by_node: Dict[str, Envelope] = dict(self.initial_envelopes)
        termination_reason = "MAX_TURNS_REACHED"
        terminated = False
        consecutive_sat_count = 0

        # Phase 03: Turn Loop (k = 1 ... maxTurns)
        for k in range(1, self.config.maxTurns + 1):
            edge_idx = (k - 1) % len(edges)
            sender_id, recipient_id = edges[edge_idx]

            # Enforce directed graph routing verification
            if not self.config.is_route_allowed(sender_id, recipient_id):
                raise RoutingError(
                    f"Route from '{sender_id}' to '{recipient_id}' is not permitted by routingMatrix."
                )

            # Volatile state with internal time tick (no persistent timestamps)
            volatile_state = VolatileState(internalTimeTick=k)

            # Determine incoming message to pass to sender
            prior_incoming = current_payload_by_node.get(recipient_id) or current_payload_by_node[sender_id]
            input_envelope = Envelope(
                sender=prior_incoming.sender,
                recipient=sender_id,
                turn=k,
                volatileState=volatile_state,
                payload=prior_incoming.payload,
            )

            # Execute isolated process via CLI wrapper
            runner = self.runners[sender_id]
            output_envelope = runner.execute_turn(input_envelope)

            # Verify output recipient authorization
            actual_recipient = output_envelope.recipient or recipient_id
            if not self.config.is_route_allowed(sender_id, actual_recipient):
                raise RoutingError(
                    f"Node '{sender_id}' attempted to send to unauthorized recipient '{actual_recipient}'."
                )

            # SMT satisfiability check using Z3 solver
            ontology = self.ontologies.get(sender_id, "")
            smt_result = self.verifier.verify_constraints(
                output_envelope.payload.smtConstraints,
                ontology_smt2=ontology,
            )

            record = TurnRecord(
                turn=k,
                sender=sender_id,
                recipient=actual_recipient,
                envelope=output_envelope.to_dict(),
                smt_result=smt_result.to_dict(),
                routing_authorized=True,
            )
            records.append(record)
            current_payload_by_node[sender_id] = output_envelope

            # Phase 04: Termination Conditions Check
            cond = self.config.terminationCondition

            # 1. UNSAT Breaker
            if smt_result.is_unsat and ("UNSAT_BREAKER" in cond or cond == "Z3_SAT_CONVERGENCE_OR_UNSAT_BREAKER"):
                terminated = True
                termination_reason = "UNSAT_BREAKER"
                break

            # 2. SAT Convergence Check
            if smt_result.is_sat:
                consecutive_sat_count += 1
                # If all edges or full round of instances have maintained SAT convergence
                if consecutive_sat_count >= len(edges) and (
                    "SAT_CONVERGENCE" in cond or cond == "Z3_SAT_CONVERGENCE_OR_UNSAT_BREAKER"
                ):
                    terminated = True
                    termination_reason = "SAT_CONVERGENCE"
                    break
            else:
                consecutive_sat_count = 0

        if not terminated:
            terminated = True
            termination_reason = "MAX_TURNS_REACHED"

        result = ExecutionResult(
            sessionId=self.config.sessionId,
            total_turns=len(records),
            terminated=terminated,
            termination_reason=termination_reason,
            records=records,
            initial_states=initial_states_dict,
            success=True,
        )

        # Write volatile session transcript to runtime_state
        self._write_session_log(result)
        return result

    def _write_session_log(self, result: ExecutionResult) -> None:
        """Writes volatile session transcript to runtime_state directory."""
        log_file = self.runtime_state_dir / f"{self.config.sessionId}_transcript.json"
        try:
            with open(log_file, "w", encoding="utf-8") as f:
                json.dump(result.to_dict(), f, indent=2)
        except Exception:
            pass
