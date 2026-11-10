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
    OrchestratorError,
    SchemaValidationError,
    RoutingError,
    StaticValidationError,
    SMTVerificationError,
)
from .verifier import Z3Verifier, SMTVerificationResult
from .node_runner import NodeProcessRunner, ProcessExecutionError
from .orchestrator import MultiInstanceOrchestrator, ExecutionResult, TurnRecord

__version__ = "0.1.0"

__all__ = [
    "TopologyConfig",
    "InstanceConfig",
    "RoutingRule",
    "VolatileState",
    "SMTConstraints",
    "Payload",
    "Envelope",
    "OrchestratorError",
    "SchemaValidationError",
    "RoutingError",
    "StaticValidationError",
    "SMTVerificationError",
    "Z3Verifier",
    "SMTVerificationResult",
    "NodeProcessRunner",
    "ProcessExecutionError",
    "MultiInstanceOrchestrator",
    "ExecutionResult",
    "TurnRecord",
]
