"""Schema definitions and validation for LingoLOGOS Multi-Instance Orchestrator.

RFC identifier: LingoLOGOS-MULTI-ORCH-v1
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional


class OrchestratorError(Exception):
    """Base exception for orchestrator errors."""
    pass


class SchemaValidationError(OrchestratorError):
    """Raised when JSON schema validation fails."""
    pass


class RoutingError(OrchestratorError):
    """Raised when a message routing rule is violated."""
    pass


class StaticValidationError(OrchestratorError):
    """Raised during the static verification phase."""
    pass


class SMTVerificationError(OrchestratorError):
    """Raised when SMT verification fails unexpectedly."""
    pass


@dataclass
class VolatileState:
    """Volatile internal state and contextual parameters passed across turns."""
    internalTimeTick: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VolatileState":
        if not isinstance(data, dict):
            raise SchemaValidationError("volatileState must be a dictionary.")
        tick = data.get("internalTimeTick", 0)
        if not isinstance(tick, int):
            raise SchemaValidationError("internalTimeTick must be an integer.")
        extra = {k: v for k, v in data.items() if k != "internalTimeTick"}
        return cls(internalTimeTick=tick, extra=extra)

    def to_dict(self) -> Dict[str, Any]:
        res = {"internalTimeTick": self.internalTimeTick}
        res.update(self.extra)
        return res


@dataclass
class SMTConstraints:
    """SMT-LIB 2 declarations and assertions with expected or verified status."""
    declarations: List[str] = field(default_factory=list)
    assertions: List[str] = field(default_factory=list)
    z3Status: str = "sat"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SMTConstraints":
        if not isinstance(data, dict):
            raise SchemaValidationError("smtConstraints must be a dictionary.")
        decls = data.get("declarations", [])
        asserts = data.get("assertions", [])
        status = data.get("z3Status", "sat")

        if not isinstance(decls, list) or not all(isinstance(x, str) for x in decls):
            raise SchemaValidationError("smtConstraints.declarations must be a list of strings.")
        if not isinstance(asserts, list) or not all(isinstance(x, str) for x in asserts):
            raise SchemaValidationError("smtConstraints.assertions must be a list of strings.")
        if not isinstance(status, str):
            raise SchemaValidationError("smtConstraints.z3Status must be a string.")

        return cls(declarations=list(decls), assertions=list(asserts), z3Status=status.lower())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "declarations": list(self.declarations),
            "assertions": list(self.assertions),
            "z3Status": self.z3Status,
        }


@dataclass
class Payload:
    """Neuro-symbolic content combining natural language, AST, and SMT constraints."""
    naturalLanguage: str
    abstractSyntaxTree: str
    smtConstraints: SMTConstraints

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Payload":
        if not isinstance(data, dict):
            raise SchemaValidationError("payload must be a dictionary.")
        nl = data.get("naturalLanguage")
        ast = data.get("abstractSyntaxTree")
        smt_data = data.get("smtConstraints")

        if not isinstance(nl, str):
            raise SchemaValidationError("payload.naturalLanguage must be a string.")
        if not isinstance(ast, str):
            raise SchemaValidationError("payload.abstractSyntaxTree must be a string.")
        if not isinstance(smt_data, dict):
            raise SchemaValidationError("payload.smtConstraints must be a dictionary.")

        return cls(
            naturalLanguage=nl,
            abstractSyntaxTree=ast,
            smtConstraints=SMTConstraints.from_dict(smt_data),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "naturalLanguage": self.naturalLanguage,
            "abstractSyntaxTree": self.abstractSyntaxTree,
            "smtConstraints": self.smtConstraints.to_dict(),
        }


@dataclass
class Envelope:
    """Neuro-symbolic communication envelope exchanged through the orchestrator."""
    sender: str
    turn: int
    volatileState: VolatileState
    payload: Payload
    recipient: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Envelope":
        if not isinstance(data, dict):
            raise SchemaValidationError("Envelope data must be a dictionary.")
        sender = data.get("sender")
        recipient = data.get("recipient")
        turn = data.get("turn")
        vs_data = data.get("volatileState")
        payload_data = data.get("payload")

        if not isinstance(sender, str) or not sender.strip():
            raise SchemaValidationError("sender must be a non-empty string.")
        if recipient is not None and (not isinstance(recipient, str) or not recipient.strip()):
            raise SchemaValidationError("recipient must be a non-empty string when provided.")
        if not isinstance(turn, int) or turn < 0:
            raise SchemaValidationError("turn must be a non-negative integer.")
        if not isinstance(vs_data, dict):
            raise SchemaValidationError("volatileState must be a dictionary.")
        if not isinstance(payload_data, dict):
            raise SchemaValidationError("payload must be a dictionary.")

        return cls(
            sender=sender,
            recipient=recipient,
            turn=turn,
            volatileState=VolatileState.from_dict(vs_data),
            payload=Payload.from_dict(payload_data),
        )

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "sender": self.sender,
            "turn": self.turn,
            "volatileState": self.volatileState.to_dict(),
            "payload": self.payload.to_dict(),
        }
        if self.recipient is not None:
            data["recipient"] = self.recipient
        return data


@dataclass
class InstanceConfig:
    """Configuration for an isolated node instance."""
    id: str
    executable: str
    role: str

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InstanceConfig":
        if not isinstance(data, dict):
            raise SchemaValidationError("Instance config must be a dictionary.")
        node_id = data.get("id")
        executable = data.get("executable")
        role = data.get("role", "node")

        if not isinstance(node_id, str) or not node_id.strip():
            raise SchemaValidationError("instance id must be a non-empty string.")
        if not isinstance(executable, str) or not executable.strip():
            raise SchemaValidationError("instance executable must be a non-empty string.")
        if not isinstance(role, str) or not role.strip():
            raise SchemaValidationError("instance role must be a non-empty string.")

        return cls(id=node_id, executable=executable, role=role)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "executable": self.executable,
            "role": self.role,
        }


@dataclass
class RoutingRule:
    """Directed edge routing rule in the communication topology."""
    from_node: str
    to_nodes: List[str]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RoutingRule":
        if not isinstance(data, dict):
            raise SchemaValidationError("Routing rule must be a dictionary.")
        from_node = data.get("from")
        to_nodes = data.get("to")

        if not isinstance(from_node, str) or not from_node.strip():
            raise SchemaValidationError("routing rule 'from' must be a non-empty string.")
        if not isinstance(to_nodes, list) or not all(isinstance(x, str) and x.strip() for x in to_nodes):
            raise SchemaValidationError("routing rule 'to' must be a list of non-empty strings.")

        return cls(from_node=from_node, to_nodes=list(to_nodes))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "from": self.from_node,
            "to": list(self.to_nodes),
        }


@dataclass
class TopologyConfig:
    """Topology graph and session configuration."""
    sessionId: str
    maxTurns: int
    terminationCondition: str
    instances: List[InstanceConfig]
    routingMatrix: List[RoutingRule]
    schema_url: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TopologyConfig":
        if not isinstance(data, dict):
            raise SchemaValidationError("Topology configuration must be a dictionary.")

        session_id = data.get("sessionId")
        max_turns = data.get("maxTurns")
        term_cond = data.get("terminationCondition", "Z3_SAT_CONVERGENCE_OR_UNSAT_BREAKER")
        instances_raw = data.get("instances")
        routing_raw = data.get("routingMatrix")
        schema_url = data.get("$schema")

        if not isinstance(session_id, str) or not session_id.strip():
            raise SchemaValidationError("sessionId must be a non-empty string.")
        if not isinstance(max_turns, int) or max_turns <= 0:
            raise SchemaValidationError("maxTurns must be a positive integer.")
        if not isinstance(term_cond, str) or not term_cond.strip():
            raise SchemaValidationError("terminationCondition must be a non-empty string.")
        if not isinstance(instances_raw, list) or len(instances_raw) == 0:
            raise SchemaValidationError("instances must be a non-empty list.")
        if not isinstance(routing_raw, list):
            raise SchemaValidationError("routingMatrix must be a list.")

        instances = [InstanceConfig.from_dict(i) for i in instances_raw]
        instance_ids = [inst.id for inst in instances]
        if len(instance_ids) != len(set(instance_ids)):
            raise SchemaValidationError("Duplicate instance ids are not permitted.")

        routing = [RoutingRule.from_dict(r) for r in routing_raw]

        # Verify that routing rules refer to declared instance ids
        known_ids = set(instance_ids)
        for rule in routing:
            if rule.from_node not in known_ids:
                raise SchemaValidationError(
                    f"Routing rule 'from' node '{rule.from_node}' is not in declared instances."
                )
            for target in rule.to_nodes:
                if target not in known_ids:
                    raise SchemaValidationError(
                        f"Routing rule target node '{target}' is not in declared instances."
                    )

        return cls(
            sessionId=session_id,
            maxTurns=max_turns,
            terminationCondition=term_cond,
            instances=instances,
            routingMatrix=routing,
            schema_url=schema_url,
        )

    def to_dict(self) -> Dict[str, Any]:
        res: Dict[str, Any] = {}
        if self.schema_url:
            res["$schema"] = self.schema_url
        res.update({
            "sessionId": self.sessionId,
            "maxTurns": self.maxTurns,
            "terminationCondition": self.terminationCondition,
            "instances": [inst.to_dict() for inst in self.instances],
            "routingMatrix": [r.to_dict() for r in self.routingMatrix],
        })
        return res

    def get_adjacency_map(self) -> Dict[str, List[str]]:
        """Returns adjacency dictionary mapping from_node to list of allowed to_nodes."""
        adj: Dict[str, List[str]] = {inst.id: [] for inst in self.instances}
        for rule in self.routingMatrix:
            adj.setdefault(rule.from_node, [])
            for target in rule.to_nodes:
                if target not in adj[rule.from_node]:
                    adj[rule.from_node].append(target)
        return adj

    def is_route_allowed(self, from_node: str, to_node: str) -> bool:
        """Checks if a directed edge exists from from_node to to_node."""
        adj = self.get_adjacency_map()
        return to_node in adj.get(from_node, [])
