"""LingoLOGOS Multi-Instance Orchestrator (lllayer).

RFC Identifier: LingoLOGOS-MULTI-ORCH-v1
Scope: Distributed Neuro-Symbolic Architecture
Key Components: LingoLOGOS, lllayer, Grammatical Framework (GF), Z3 SMT Solver
"""

from .schema import (
    TopologyConfig,
    InstanceConfig,
    RoutingRule,
    VolatileState,
    SMTConstraints,
    Payload,
    Envelope,
    NodeSpecs,
    FederatedNodeConfig,
    FederatedRoutingRule,
    FederatedTopologyConfig,
    OrchestratorError,
    SchemaValidationError,
    RoutingError,
    StaticValidationError,
    SMTVerificationError,
)
from .topology_ctrl import TopologyCtrlRegister
from .verifier import Z3Verifier, SMTVerificationResult
from .node_runner import NodeProcessRunner, HttpNodeRunner, ProcessExecutionError
from .resolver import NodeResolver, resolve_node_transport
from .event_bus import EventBus, DynamicNodeSpawner
from .solver_pool import SMTSolverPool, SolverTask
from .consolidation import ModelConsolidator, ConsolidatedModel
from .orchestrator import MultiInstanceOrchestrator, ExecutionResult, TurnRecord
from .lllayer_masker import NativeASTLogitsProcessor
from .lllayer_engine import (
    IncrementalASTEngine,
    IncrementalGrammarSpec,
    SimpleASTGrammar,
    MockRunnerModel,
)

__version__ = "0.2.0"

__all__ = [
    "TopologyConfig",
    "InstanceConfig",
    "RoutingRule",
    "VolatileState",
    "SMTConstraints",
    "Payload",
    "Envelope",
    "NodeSpecs",
    "FederatedNodeConfig",
    "FederatedRoutingRule",
    "FederatedTopologyConfig",
    "TopologyCtrlRegister",
    "OrchestratorError",
    "SchemaValidationError",
    "RoutingError",
    "StaticValidationError",
    "SMTVerificationError",
    "Z3Verifier",
    "SMTVerificationResult",
    "NodeProcessRunner",
    "HttpNodeRunner",
    "ProcessExecutionError",
    "NodeResolver",
    "resolve_node_transport",
    "EventBus",
    "DynamicNodeSpawner",
    "SMTSolverPool",
    "SolverTask",
    "ModelConsolidator",
    "ConsolidatedModel",
    "MultiInstanceOrchestrator",
    "ExecutionResult",
    "TurnRecord",
    "NativeASTLogitsProcessor",
    "IncrementalASTEngine",
    "IncrementalGrammarSpec",
    "SimpleASTGrammar",
    "MockRunnerModel",
]
