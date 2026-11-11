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
    topologyCtrl: Optional[Dict[str, Any]] = None
    datasetId: Optional[int] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Envelope":
        if not isinstance(data, dict):
            raise SchemaValidationError("Envelope data must be a dictionary.")
        sender = data.get("sender")
        recipient = data.get("recipient")
        turn = data.get("turn")
        vs_data = data.get("volatileState")
        payload_data = data.get("payload")
        topology_ctrl = data.get("topologyCtrl")
        dataset_id = data.get("datasetId")

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
            topologyCtrl=topology_ctrl if isinstance(topology_ctrl, dict) else None,
            datasetId=dataset_id if isinstance(dataset_id, int) else None,
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
        if self.topologyCtrl is not None:
            data["topologyCtrl"] = self.topologyCtrl
        if self.datasetId is not None:
            data["datasetId"] = self.datasetId
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


@dataclass
class NodeSpecs:
    """Formal specifications pointing to grammar, ontology, and seed payload."""
    grammar: str
    ontology: str
    seed: str

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NodeSpecs":
        if not isinstance(data, dict):
            raise SchemaValidationError("NodeSpecs must be a dictionary.")
        grammar = data.get("grammar")
        ontology = data.get("ontology")
        seed = data.get("seed")

        if not isinstance(grammar, str) or not grammar.strip():
            raise SchemaValidationError("specs.grammar must be a non-empty string.")
        if not isinstance(ontology, str) or not ontology.strip():
            raise SchemaValidationError("specs.ontology must be a non-empty string.")
        if not isinstance(seed, str) or not seed.strip():
            raise SchemaValidationError("specs.seed must be a non-empty string.")

        return cls(grammar=grammar, ontology=ontology, seed=seed)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "grammar": self.grammar,
            "ontology": self.ontology,
            "seed": self.seed,
        }


@dataclass
class FederatedNodeConfig:
    """Configuration for a node in a federated topology mesh."""
    node_id: str
    role: str
    transport: str = "local"
    specs: Optional[NodeSpecs] = None
    remote_source: Optional[str] = None
    branch: Optional[str] = None
    local_cache_path: Optional[str] = None
    endpoint_url: Optional[str] = None
    auth_method: Optional[str] = None
    auth_token_env: Optional[str] = None
    outbound_channels: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FederatedNodeConfig":
        if not isinstance(data, dict):
            raise SchemaValidationError("Federated node config must be a dictionary.")
        node_id = data.get("node_id")
        role = data.get("role", "node")
        transport = data.get("transport", "local").lower()

        if not isinstance(node_id, str) or not node_id.strip():
            raise SchemaValidationError("node_id must be a non-empty string.")
        if not isinstance(role, str) or not role.strip():
            raise SchemaValidationError("role must be a non-empty string.")
        if transport not in ("local", "git", "http"):
            raise SchemaValidationError(f"Unsupported transport: {transport}. Must be local, git, or http.")

        specs = None
        if "specs" in data and isinstance(data["specs"], dict):
            specs = NodeSpecs.from_dict(data["specs"])

        remote_source = data.get("remote_source")
        branch = data.get("branch")
        local_cache_path = data.get("local_cache_path")
        endpoint_url = data.get("endpoint_url")
        auth_method = data.get("auth_method")
        auth_token_env = data.get("auth_token_env")
        outbound = data.get("outbound_channels", [])

        if not isinstance(outbound, list) or not all(isinstance(x, str) for x in outbound):
            raise SchemaValidationError("outbound_channels must be a list of strings.")

        if auth_token_env is not None and (not isinstance(auth_token_env, str) or not auth_token_env.strip()):
            raise SchemaValidationError("auth_token_env must be a non-empty environment variable name.")
        if auth_token_env is not None and auth_method != "bearer_token":
            raise SchemaValidationError("auth_token_env requires auth_method set to bearer_token.")

        if transport == "git":
            if not remote_source or not isinstance(remote_source, str):
                raise SchemaValidationError("remote_source is required for git transport.")
        elif transport == "http":
            if not endpoint_url or not isinstance(endpoint_url, str):
                raise SchemaValidationError("endpoint_url is required for http transport.")

        return cls(
            node_id=node_id,
            role=role,
            transport=transport,
            specs=specs,
            remote_source=remote_source,
            branch=branch,
            local_cache_path=local_cache_path,
            endpoint_url=endpoint_url,
            auth_method=auth_method,
            auth_token_env=auth_token_env,
            outbound_channels=list(outbound),
        )

    def to_dict(self) -> Dict[str, Any]:
        res: Dict[str, Any] = {
            "node_id": self.node_id,
            "role": self.role,
            "transport": self.transport,
            "outbound_channels": list(self.outbound_channels),
        }
        if self.specs is not None:
            res["specs"] = self.specs.to_dict()
        if self.remote_source is not None:
            res["remote_source"] = self.remote_source
        if self.branch is not None:
            res["branch"] = self.branch
        if self.local_cache_path is not None:
            res["local_cache_path"] = self.local_cache_path
        if self.endpoint_url is not None:
            res["endpoint_url"] = self.endpoint_url
        if self.auth_method is not None:
            res["auth_method"] = self.auth_method
        if self.auth_token_env is not None:
            res["auth_token_env"] = self.auth_token_env
        return res


@dataclass
class FederatedRoutingRule:
    """Routing rule between federated nodes with trigger conditions."""
    source: str
    target: str
    condition: str

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FederatedRoutingRule":
        if not isinstance(data, dict):
            raise SchemaValidationError("Federated routing rule must be a dictionary.")
        source = data.get("source")
        target = data.get("target")
        condition = data.get("condition", "DEFAULT")

        if not isinstance(source, str) or not source.strip():
            raise SchemaValidationError("source must be a non-empty string.")
        if not isinstance(target, str) or not target.strip():
            raise SchemaValidationError("target must be a non-empty string.")
        if not isinstance(condition, str):
            raise SchemaValidationError("condition must be a string.")

        return cls(source=source, target=target, condition=condition)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "condition": self.condition,
        }


@dataclass
class FederatedTopologyConfig:
    """Federated multi-node mesh topology specification."""
    federation_name: str
    version: str
    description: str
    nodes: List[FederatedNodeConfig]
    routing_rules: List[FederatedRoutingRule]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FederatedTopologyConfig":
        if not isinstance(data, dict):
            raise SchemaValidationError("Federated topology config must be a dictionary.")

        fed_name = data.get("federation_name")
        version = data.get("version", "2.0.0")
        desc = data.get("description", "")
        nodes_raw = data.get("nodes")
        rules_raw = data.get("routing_rules", [])

        if not isinstance(fed_name, str) or not fed_name.strip():
            raise SchemaValidationError("federation_name must be a non-empty string.")
        if not isinstance(nodes_raw, list) or len(nodes_raw) == 0:
            raise SchemaValidationError("nodes must be a non-empty list.")
        if not isinstance(rules_raw, list):
            raise SchemaValidationError("routing_rules must be a list.")

        nodes = [FederatedNodeConfig.from_dict(n) for n in nodes_raw]
        node_ids = [n.node_id for n in nodes]
        if len(node_ids) != len(set(node_ids)):
            raise SchemaValidationError("Duplicate node_id values are not permitted in federation.")

        rules = [FederatedRoutingRule.from_dict(r) for r in rules_raw]
        known_ids = set(node_ids)
        for r in rules:
            if r.source not in known_ids:
                raise SchemaValidationError(f"Rule source '{r.source}' not declared in nodes.")
            if r.target not in known_ids:
                raise SchemaValidationError(f"Rule target '{r.target}' not declared in nodes.")

        return cls(
            federation_name=fed_name,
            version=version,
            description=desc,
            nodes=nodes,
            routing_rules=rules,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "federation_name": self.federation_name,
            "version": self.version,
            "description": self.description,
            "nodes": [n.to_dict() for n in self.nodes],
            "routing_rules": [r.to_dict() for r in self.routing_rules],
        }

    def to_legacy_topology(self, max_turns: int = 15) -> TopologyConfig:
        """Converts federated configuration to TopologyConfig for orchestrator execution."""
        instances = [
            InstanceConfig(id=n.node_id, executable="lllayer", role=n.role)
            for n in self.nodes
        ]
        adjacency: Dict[str, List[str]] = {n.node_id: [] for n in self.nodes}
        for n in self.nodes:
            for out in n.outbound_channels:
                if out in adjacency and out not in adjacency[n.node_id]:
                    adjacency[n.node_id].append(out)
        for rule in self.routing_rules:
            if rule.target not in adjacency[rule.source]:
                adjacency[rule.source].append(rule.target)

        routing_matrix = [
            RoutingRule(from_node=src, to_nodes=targets)
            for src, targets in adjacency.items()
            if targets
        ]

        return TopologyConfig(
            sessionId=self.federation_name,
            maxTurns=max_turns,
            terminationCondition="Z3_SAT_CONVERGENCE_OR_UNSAT_BREAKER",
            instances=instances,
            routingMatrix=routing_matrix,
        )
