"""Generic, dataset driven topology graph: the model behind the node canvas.

Nothing in this module fixes what a node *is*. A node is an identifier plus a kind
string plus a port stack plus attributes, and all four come from the input document,
from the shape of a mixed dataset, or from the user. That is what makes the canvas
agnostic: an arbitrary kind such as ``neural_encoder`` or ``symbolic_generator``
appears with its own ports without one line of the workbench being edited, and the
kind registry only ever adds presentation hints, never permission.

Three input dialects are understood, and the dialect is detected from the document
shape, not from the file name:
    1. Orchestrator topology (``config/topology.json``): ``instances`` plus
       ``routingMatrix``.
    2. Federated mesh (``topologies/federated_mesh.json``): ``nodes`` with
       ``outbound_channels`` plus ``routing_rules``.
    3. Workbench native spec: ``nodes`` with explicit ``ports`` and ``attributes``,
       which is what the canvas saves and what a hand written spec may carry.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

DIRECTION_IN = "in"
DIRECTION_OUT = "out"

ACCENT_CYCLE = ("accent", "gf", "smt", "data", "pass", "warn", "link")


@dataclass
class PortSpec:
    """One dynamic port of a node, derived from a vector, a channel or a payload key."""

    name: str
    direction: str = DIRECTION_IN
    width: int = 1
    dtype: str = "scalar"
    origin: str = "declared"

    @property
    def key(self) -> str:
        return f"{self.direction}:{self.name}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "direction": self.direction,
            "width": self.width,
            "dtype": self.dtype,
            "origin": self.origin,
        }


@dataclass
class NodeSpec:
    """A generic node: identity, kind, port stack, and free form attributes."""

    node_id: str
    kind: str = "generic"
    label: str = ""
    ports: List[PortSpec] = field(default_factory=list)
    attributes: Dict[str, str] = field(default_factory=dict)
    position: Optional[Tuple[float, float]] = None

    def __post_init__(self) -> None:
        if not self.label:
            self.label = self.node_id

    def port(self, direction: str, name: str) -> Optional[PortSpec]:
        for candidate in self.ports:
            if candidate.direction == direction and candidate.name == name:
                return candidate
        return None

    def ensure_port(self, direction: str, name: str, **kwargs: Any) -> PortSpec:
        existing = self.port(direction, name)
        if existing is not None:
            return existing
        created = PortSpec(name=name, direction=direction, **kwargs)
        self.ports.append(created)
        return created

    @property
    def inputs(self) -> List[PortSpec]:
        return [p for p in self.ports if p.direction == DIRECTION_IN]

    @property
    def outputs(self) -> List[PortSpec]:
        return [p for p in self.ports if p.direction == DIRECTION_OUT]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.node_id,
            "kind": self.kind,
            "label": self.label,
            "attributes": dict(self.attributes),
            "ports": [p.to_dict() for p in self.ports],
            "position": list(self.position) if self.position else None,
        }


@dataclass
class EdgeSpec:
    """A directed connection between two ports of two nodes."""

    source: str
    target: str
    source_port: str = ""
    target_port: str = ""
    label: str = ""
    kind: str = "channel"

    def key(self) -> str:
        return f"{self.source}[{self.source_port}]->{self.target}[{self.target_port}]"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "from": self.source,
            "to": self.target,
            "from_port": self.source_port,
            "to_port": self.target_port,
            "label": self.label,
            "kind": self.kind,
        }


@dataclass
class KindSpec:
    """Presentation hints for a node kind: title, accent, defaults, known attribute keys."""

    kind: str
    title: str = ""
    accent: str = "accent"
    default_in_ports: Tuple[str, ...] = ()
    default_out_ports: Tuple[str, ...] = ()
    attribute_keys: Tuple[str, ...] = ()
    builtin: bool = False

    def __post_init__(self) -> None:
        if not self.title:
            self.title = self.kind.replace("_", " ").title()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "title": self.title,
            "accent": self.accent,
            "default_in_ports": list(self.default_in_ports),
            "default_out_ports": list(self.default_out_ports),
            "attribute_keys": list(self.attribute_keys),
        }


class KindRegistry:
    """Kind catalogue: seeded from code, extended by any JSON document, never closed.

    Unknown kinds are synthesized on demand from the kind name, so a dataset or a
    user that introduces a kind the workbench has never seen still gets a stable
    accent colour and a readable title.
    """

    def __init__(self) -> None:
        self._kinds: Dict[str, KindSpec] = {}
        for spec in (
            KindSpec("generic", "Generic Node", "accent", ("in",), ("out",), ("note",), True),
            KindSpec(
                "instance",
                "Executable Instance",
                "accent",
                ("rx",),
                ("tx",),
                ("executable", "role", "terminationCondition"),
                True,
            ),
            KindSpec(
                "federated_node",
                "Federated Node",
                "gf",
                ("rx",),
                ("tx",),
                ("transport", "grammar", "ontology", "seed", "local_cache_path"),
                True,
            ),
            KindSpec("transform", "Data Transform", "data", ("raw",), ("cooked",), ("script",), True),
            KindSpec("filter", "Filter Node", "warn", ("raw",), ("kept",), ("predicate",), True),
            KindSpec("generator", "Symbolic Generator", "smt", ("seed",), ("formula",), ("smt",), True),
            KindSpec("encoder", "Neural Encoder", "pass", ("tokens",), ("embedding",), ("model",), True),
            KindSpec("validator", "Formal Validator", "smt", ("candidate",), ("verdict",), ("smt",), True),
            KindSpec("router", "Topology Router", "link", ("tx",), ("link",), ("routing_mask",), True),
            KindSpec("grammar", "GF Grammar", "gf", (), ("rules",), ("grammar",), True),
            KindSpec("dataset", "Dataset Payload", "data", (), ("records",), ("path", "format"), True),
            KindSpec("hdl", "Hardware Entity", "warn", ("pins",), ("pins",), ("path",), True),
            KindSpec("python", "Python Script", "accent", ("argv",), ("stdout",), ("script",), True),
        ):
            self._kinds[spec.kind] = spec

    def register(self, spec: KindSpec) -> KindSpec:
        self._kinds[spec.kind] = spec
        return spec

    def known(self) -> List[str]:
        return sorted(self._kinds)

    def get(self, kind: str) -> KindSpec:
        """Return the spec for a kind, synthesizing one when it is new to the workbench."""
        clean = (kind or "generic").strip() or "generic"
        spec = self._kinds.get(clean)
        if spec is not None:
            return spec
        digest = int(hashlib.sha256(clean.encode("utf-8")).hexdigest(), 16)
        return KindSpec(
            kind=clean,
            title=clean.replace("_", " ").replace("-", " ").title(),
            accent=ACCENT_CYCLE[digest % len(ACCENT_CYCLE)],
            default_in_ports=("in",),
            default_out_ports=("out",),
        )

    def load_from_document(self, document: Dict[str, Any]) -> List[str]:
        """Merge a ``kinds`` block of a workbench spec into the registry."""
        added: List[str] = []
        block = document.get("kinds")
        if not isinstance(block, dict):
            return added
        for name, body in block.items():
            if not isinstance(body, dict):
                continue
            self.register(
                KindSpec(
                    kind=str(name),
                    title=str(body.get("title", "")),
                    accent=str(body.get("accent", "accent")),
                    default_in_ports=tuple(body.get("in_ports", ())),
                    default_out_ports=tuple(body.get("out_ports", ())),
                    attribute_keys=tuple(body.get("attribute_keys", ())),
                )
            )
            added.append(str(name))
        return added


@dataclass
class Issue:
    """One validation finding: an error, a warning or a note about the graph."""

    severity: str
    code: str
    message: str
    subjects: List[str] = field(default_factory=list)


@dataclass
class TopologyGraph:
    """Nodes, edges and their provenance; the single source of truth of the canvas."""

    nodes: Dict[str, NodeSpec] = field(default_factory=dict)
    edges: List[EdgeSpec] = field(default_factory=list)
    name: str = "untitled"
    source: Optional[Path] = None
    dialect: str = "native"
    notes: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # --- mutation --------------------------------------------------------------
    def add_node(self, spec: NodeSpec) -> NodeSpec:
        self.nodes[spec.node_id] = spec
        return spec

    def remove_node(self, node_id: str) -> None:
        self.nodes.pop(node_id, None)
        self.edges = [e for e in self.edges if e.source != node_id and e.target != node_id]

    def connect(self, edge: EdgeSpec) -> Optional[EdgeSpec]:
        """Add an edge, refusing a duplicate on the same port pair. Returns None if dropped."""
        if edge.key() in {e.key() for e in self.edges}:
            return None
        self.edges.append(edge)
        return edge

    def remove_edge(self, edge: EdgeSpec) -> None:
        self.edges = [e for e in self.edges if e.key() != edge.key()]

    def out_edges(self, node_id: str) -> List[EdgeSpec]:
        return [e for e in self.edges if e.source == node_id]

    def in_edges(self, node_id: str) -> List[EdgeSpec]:
        return [e for e in self.edges if e.target == node_id]

    def degree(self, node_id: str) -> int:
        return len(self.out_edges(node_id)) + len(self.in_edges(node_id))

    def neighbours(self, node_id: str) -> List[str]:
        seen: List[str] = []
        for edge in self.out_edges(node_id):
            if edge.target not in seen:
                seen.append(edge.target)
        return seen

    # --- serialisation ---------------------------------------------------------
    def to_workbench_dict(self) -> Dict[str, Any]:
        return {
            "spec": "lingologos-workbench-topology-v1",
            "name": self.name,
            "dialect": self.dialect,
            "metadata": self.metadata,
            "nodes": [self.nodes[k].to_dict() for k in sorted(self.nodes)],
            "edges": [e.to_dict() for e in self.edges],
            "notes": list(self.notes),
        }

    def stats(self) -> Dict[str, Any]:
        kinds: Dict[str, int] = {}
        for node in self.nodes.values():
            kinds[node.kind] = kinds.get(node.kind, 0) + 1
        layers = dag_layers(self)
        return {
            "nodes": len(self.nodes),
            "edges": len(self.edges),
            "ports": sum(len(n.ports) for n in self.nodes.values()),
            "kinds": kinds,
            "layers": len(layers),
            "acyclic": bool(layers) or not self.nodes,
        }


# ---------------------------------------------------------------------------
# Port derivation: the port stack of a node is inferred from the data it carries.
# ---------------------------------------------------------------------------
def _dtype_of(value: Any) -> Tuple[str, int]:
    if isinstance(value, bool):
        return "bool", 1
    if isinstance(value, (int, float)):
        return "scalar", 1
    if isinstance(value, str):
        return "text", 1
    if isinstance(value, list):
        inner = {_dtype_of(item)[0] for item in value} or {"scalar"}
        return f"vector[{len(value)}]", len(value) or 1
    if isinstance(value, dict):
        return "record", len(value)
    return "scalar", 1


def derive_ports_from_mapping(mapping: Dict[str, Any], direction: str = DIRECTION_OUT) -> List[PortSpec]:
    """Turn one payload mapping into one port per key, with width from vector length.

    This is how a mixed dataset defines the I/O stack of a node without any fixed
    role: the keys of the payload are the ports, and the depth of a vector is the
    width of the port.
    """
    ports: List[PortSpec] = []
    for key, value in mapping.items():
        dtype, width = _dtype_of(value)
        ports.append(PortSpec(name=str(key), direction=direction, width=width, dtype=dtype, origin="payload"))
    return ports


YAML_SUFFIXES = (".yaml", ".yml")


def _yaml_module(path: Path):
    """PyYAML, imported only when a YAML document is really read or written.

    The package is optional for the workbench: a JSON only session never needs it, and a
    missing install turns into a readable error on the one document that asked for it.
    """
    try:
        import yaml
    except ImportError as exc:
        raise ValueError(
            f"{path.name} is a YAML document and PyYAML is not installed: "
            "install it with pip install PyYAML, or keep the topology as JSON"
        ) from exc
    return yaml


def _read_document(path: Path) -> Dict[str, Any]:
    """Read a topology document, JSON or YAML according to its suffix."""
    is_yaml = path.suffix.lower() in YAML_SUFFIXES
    with open(path, "r", encoding="utf-8") as handle:
        if is_yaml:
            yaml = _yaml_module(path)
            try:
                data = yaml.safe_load(handle.read())
            except yaml.YAMLError as exc:
                raise ValueError(f"{path.name} is not valid YAML: {exc}") from exc
        else:
            data = json.load(handle)
    if not isinstance(data, dict):
        kind = "mapping" if is_yaml else "JSON object"
        raise ValueError(f"{path} must hold a {kind} at the top level")
    return data


def _read_json(path: Path) -> Dict[str, Any]:
    """Read a JSON object; kept for the seed payloads, which are JSON by contract."""
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must hold a JSON object at the top level")
    return data


def _seed_mapping(node_id: str, datasets_dir: Optional[Path]) -> Dict[str, Any]:
    """Look for the seed payload of a node so its payload keys can become ports."""
    if datasets_dir is None:
        return {}
    payload_path = datasets_dir / node_id / "seed_payload.json"
    if not payload_path.is_file():
        return {}
    try:
        document = _read_json(payload_path)
    except (OSError, ValueError, json.JSONDecodeError):
        return {}
    inner = document.get("payload")
    return inner if isinstance(inner, dict) else {}


# ---------------------------------------------------------------------------
# Dialect loaders.
# ---------------------------------------------------------------------------
def _load_orchestrator_dialect(document: Dict[str, Any], source: Optional[Path], registry: KindRegistry, datasets_dir: Optional[Path]) -> TopologyGraph:
    graph = TopologyGraph(
        name=str(document.get("sessionId", "session")),
        source=source,
        dialect="orchestrator",
    )
    graph.metadata = {
        "maxTurns": document.get("maxTurns"),
        "terminationCondition": document.get("terminationCondition"),
        "schema": document.get("$schema"),
    }
    for instance in document.get("instances", []):
        if not isinstance(instance, dict):
            continue
        node_id = str(instance.get("id", ""))
        if not node_id:
            continue
        node = NodeSpec(node_id=node_id, kind="instance", label=node_id)
        node.attributes = {k: str(v) for k, v in instance.items() if k != "id"}
        if document.get("terminationCondition"):
            node.attributes.setdefault("terminationCondition", str(document["terminationCondition"]))
        payload = _seed_mapping(node_id, datasets_dir)
        if payload:
            graph.notes.append(f"ports of {node_id} derived from payload keys of its seed file")
            node.ports.extend(derive_ports_from_mapping(payload, DIRECTION_OUT))
            for key in payload:
                node.ensure_port(DIRECTION_IN, key, origin="payload")
        else:
            registry.get(node.kind)
            node.ensure_port(DIRECTION_IN, "rx", origin="kind")
            node.ensure_port(DIRECTION_OUT, "tx", origin="kind")
        graph.add_node(node)

    for rule in document.get("routingMatrix", []):
        if not isinstance(rule, dict):
            continue
        source_id = str(rule.get("from", ""))
        for target in rule.get("to", []) or []:
            target_id = str(target)
            if source_id in graph.nodes and target_id in graph.nodes:
                out_port = graph.nodes[source_id].outputs[0].name if graph.nodes[source_id].outputs else "tx"
                in_port = graph.nodes[target_id].inputs[0].name if graph.nodes[target_id].inputs else "rx"
                graph.connect(
                    EdgeSpec(
                        source=source_id,
                        target=target_id,
                        source_port=out_port,
                        target_port=in_port,
                        label="routingMatrix",
                        kind="routing",
                    )
                )
    return graph


def _load_federated_dialect(document: Dict[str, Any], source: Optional[Path], registry: KindRegistry, datasets_dir: Optional[Path]) -> TopologyGraph:
    graph = TopologyGraph(
        name=str(document.get("federation_name", "federation")),
        source=source,
        dialect="federated",
    )
    graph.metadata = {
        "version": document.get("version"),
        "description": document.get("description"),
    }
    for node_body in document.get("nodes", []):
        if not isinstance(node_body, dict):
            continue
        node_id = str(node_body.get("node_id", ""))
        if not node_id:
            continue
        node = NodeSpec(node_id=node_id, kind=str(node_body.get("kind", "federated_node")))
        for key, value in node_body.items():
            if key in ("node_id", "outbound_channels", "specs"):
                continue
            if isinstance(value, (str, int, float, bool)):
                node.attributes[key] = str(value)
        specs = node_body.get("specs")
        if isinstance(specs, dict):
            for key, value in specs.items():
                node.attributes[str(key)] = str(value)
        channels = [str(c) for c in (node_body.get("outbound_channels") or [])]
        for channel in channels:
            node.ensure_port(DIRECTION_OUT, f"tx:{channel}", origin="channel")
        if channels:
            node.ensure_port(DIRECTION_IN, "rx", origin="channel")
        payload = _seed_mapping(node_id, datasets_dir)
        if payload:
            node.ports.extend(derive_ports_from_mapping(payload, DIRECTION_OUT))
        graph.add_node(node)

    for node_id in list(graph.nodes):
        graph.nodes[node_id].ensure_port(DIRECTION_IN, "rx", origin="channel")

    for rule in document.get("routing_rules", []):
        if not isinstance(rule, dict):
            continue
        src, dst = str(rule.get("source", "")), str(rule.get("target", ""))
        if src in graph.nodes and dst in graph.nodes:
            out_name = next(
                (p.name for p in graph.nodes[src].outputs if p.name.endswith(f":{dst}")),
                graph.nodes[src].outputs[0].name if graph.nodes[src].outputs else "tx",
            )
            graph.connect(
                EdgeSpec(
                    source=src,
                    target=dst,
                    source_port=out_name,
                    target_port="rx",
                    label=str(rule.get("condition", "")),
                    kind="routing_rule",
                )
            )
    return graph


def _load_native_dialect(document: Dict[str, Any], source: Optional[Path], registry: KindRegistry) -> TopologyGraph:
    graph = TopologyGraph(
        name=str(document.get("name", "graph")),
        source=source,
        dialect="native",
    )
    registry.load_from_document(document)
    metadata = document.get("metadata")
    if isinstance(metadata, dict):
        graph.metadata = dict(metadata)
    for body in document.get("nodes", []):
        if not isinstance(body, dict):
            continue
        node_id = str(body.get("id", body.get("node_id", "")))
        if not node_id:
            continue
        kind = str(body.get("kind", "generic"))
        node = NodeSpec(
            node_id=node_id,
            kind=kind,
            label=str(body.get("label", node_id)),
            attributes={str(k): str(v) for k, v in (body.get("attributes") or {}).items()},
        )
        position = body.get("position")
        if isinstance(position, (list, tuple)) and len(position) == 2:
            try:
                node.position = (float(position[0]), float(position[1]))
            except (TypeError, ValueError):
                node.position = None
        ports = body.get("ports")
        if isinstance(ports, list) and ports:
            for port_body in ports:
                if not isinstance(port_body, dict):
                    continue
                direction = str(port_body.get("direction", DIRECTION_IN))
                if direction not in (DIRECTION_IN, DIRECTION_OUT):
                    direction = DIRECTION_IN
                try:
                    width = int(port_body.get("width", 1))
                except (TypeError, ValueError):
                    width = 1
                node.ports.append(
                    PortSpec(
                        name=str(port_body.get("name", "port")),
                        direction=direction,
                        width=max(1, width),
                        dtype=str(port_body.get("dtype", "scalar")),
                        origin=str(port_body.get("origin", "declared")),
                    )
                )
        else:
            spec = registry.get(kind)
            for name in spec.default_in_ports:
                node.ensure_port(DIRECTION_IN, name, origin="kind")
            for name in spec.default_out_ports:
                node.ensure_port(DIRECTION_OUT, name, origin="kind")
        graph.add_node(node)

    for body in document.get("edges", document.get("channels", [])):
        if not isinstance(body, dict):
            continue
        src = str(body.get("from", body.get("source", "")))
        dst = str(body.get("to", body.get("target", "")))
        if src and dst:
            graph.connect(
                EdgeSpec(
                    source=src,
                    target=dst,
                    source_port=str(body.get("from_port", body.get("source_port", ""))),
                    target_port=str(body.get("to_port", body.get("target_port", ""))),
                    label=str(body.get("label", "")),
                    kind=str(body.get("kind", "channel")),
                )
            )
    return graph


def classify_document(document: Dict[str, Any]) -> str:
    """Detect the input dialect from the shape of the document."""
    if "instances" in document or "routingMatrix" in document:
        return "orchestrator"
    nodes = document.get("nodes")
    if isinstance(nodes, list) and nodes and isinstance(nodes[0], dict):
        if "node_id" in nodes[0] or "routing_rules" in document:
            return "federated"
        return "native"
    if "edges" in document or "spec" in document:
        return "native"
    return "unknown"


def load_graph_file(path: Path, registry: Optional[KindRegistry] = None, datasets_dir: Optional[Path] = None) -> TopologyGraph:
    """Load any understood topology document into a graph.

    :raises ValueError: when the document shape matches no known dialect.
    """
    kinds = registry or KindRegistry()
    document = _read_document(Path(path))
    dialect = classify_document(document)
    if dialect == "orchestrator":
        return _load_orchestrator_dialect(document, Path(path), kinds, datasets_dir)
    if dialect == "federated":
        return _load_federated_dialect(document, Path(path), kinds, datasets_dir)
    if dialect == "native":
        return _load_native_dialect(document, Path(path), kinds)
    raise ValueError(f"{path} does not match any known topology dialect")


def graph_from_board(board) -> TopologyGraph:
    """Project the hardware board onto the same graph model, so links can be validated.

    The board wrapper is a graph of dies exactly like a topology is a graph of
    agents, so the DAG checker, the cycle detector and the layout of the canvas all
    apply to hardware without a second implementation.
    """
    graph = TopologyGraph(name=board.name, source=board.source, dialect="board")
    graph.metadata = dict(board.parameters)
    for die in board.dies:
        node = NodeSpec(node_id=die.name, kind="hdl", label=f"die {die.index}: {die.name}")
        node.attributes = {
            "node_id": str(die.node_id),
            "total_nodes": str(die.total_nodes),
            "routing_mask": f"0b{die.routing_mask:0{die.mask_width}b}",
            "standalone": "true" if die.standalone else "false",
        }
        for pin in die.pins:
            node.ensure_port(
                DIRECTION_IN if pin.direction == "input" else DIRECTION_OUT,
                pin.name,
                width=pin.width,
                dtype=pin.dtype,
                origin="board_ports",
            )
        graph.add_node(node)
    for link in board.links:
        graph.connect(
            EdgeSpec(
                source=link.source_name,
                target=link.target_name,
                source_port=f"link_tx[{link.target_index}]",
                target_port=f"link_rx[{link.source_index}]",
                label=f"link {link.index}",
                kind="link",
            )
        )
    return graph


# ---------------------------------------------------------------------------
# Validation: DAG check, cycles, isolated nodes, dangling ports.
# ---------------------------------------------------------------------------
def dag_layers(graph: TopologyGraph) -> List[List[str]]:
    """Return the layering by Kahn topological peeling, or [] when a cycle exists."""
    indegree = {node_id: 0 for node_id in graph.nodes}
    for edge in graph.edges:
        if edge.target in indegree and edge.source != edge.target:
            indegree[edge.target] += 1
    layers: List[List[str]] = []
    remaining = dict(indegree)
    while remaining:
        current = sorted(n for n, d in remaining.items() if d == 0)
        if not current:
            return []
        layers.append(current)
        for node_id in current:
            remaining.pop(node_id, None)
            for target in graph.neighbours(node_id):
                if target in remaining:
                    remaining[target] -= 1
    return layers


def find_cycles(graph: TopologyGraph) -> List[List[str]]:
    """Return the elementary node cycles found by an iterative depth first walk."""
    adjacency = {node_id: graph.neighbours(node_id) for node_id in graph.nodes}
    cycles: List[List[str]] = []
    seen_signatures = set()
    state = {node_id: 0 for node_id in graph.nodes}  # 0 new, 1 on stack, 2 done
    for root in sorted(graph.nodes):
        if state[root] != 0:
            continue
        stack: List[Tuple[str, int]] = [(root, 0)]
        path: List[str] = []
        while stack:
            node_id, cursor = stack[-1]
            if cursor == 0:
                state[node_id] = 1
                path.append(node_id)
            children = adjacency.get(node_id, [])
            if cursor < len(children):
                stack[-1] = (node_id, cursor + 1)
                child = children[cursor]
                if state.get(child, 2) == 1:
                    start = path.index(child) if child in path else 0
                    cycle = path[start:] + [child]
                    signature = ">=<".join(sorted(set(cycle)))
                    if signature not in seen_signatures:
                        seen_signatures.add(signature)
                        cycles.append(cycle)
                elif state.get(child, 0) == 0:
                    stack.append((child, 0))
            else:
                stack.pop()
                state[node_id] = 2
                if path:
                    path.pop()
    return cycles


def validate_graph(graph: TopologyGraph) -> List[Issue]:
    """Recursive graph validation used by the canvas badge and the issues panel."""
    issues: List[Issue] = []
    if not graph.nodes:
        issues.append(Issue("info", "EMPTY_GRAPH", "the graph has no node yet"))
        return issues

    for edge in graph.edges:
        if edge.source not in graph.nodes:
            issues.append(Issue("error", "UNKNOWN_ENDPOINT", f"edge from unknown node {edge.source}", [edge.source]))
        if edge.target not in graph.nodes:
            issues.append(Issue("error", "UNKNOWN_ENDPOINT", f"edge to unknown node {edge.target}", [edge.target]))
        if edge.source == edge.target:
            issues.append(Issue("error", "SELF_LOOP", f"{edge.source} connects to itself", [edge.source]))
        if edge.source in graph.nodes and edge.source_port:
            if graph.nodes[edge.source].port(DIRECTION_OUT, edge.source_port) is None:
                issues.append(
                    Issue(
                        "warn",
                        "UNKNOWN_PORT",
                        f"no out port {edge.source_port} on {edge.source}",
                        [edge.source],
                    )
                )
        if edge.target in graph.nodes and edge.target_port:
            if graph.nodes[edge.target].port(DIRECTION_IN, edge.target_port) is None:
                issues.append(
                    Issue(
                        "warn",
                        "UNKNOWN_PORT",
                        f"no in port {edge.target_port} on {edge.target}",
                        [edge.target],
                    )
                )

    signatures: Dict[str, int] = {}
    for edge in graph.edges:
        pair = f"{edge.source}->{edge.target}"
        signatures[pair] = signatures.get(pair, 0) + 1
    for pair, count in sorted(signatures.items()):
        if count > 1:
            issues.append(Issue("warn", "DUPLICATE_EDGE", f"{count} edges between {pair}", pair.split("->")))

    for cycle in find_cycles(graph):
        issues.append(
            Issue("error", "CYCLE", "cycle detected: " + " -> ".join(cycle), cycle)
        )

    for node_id in sorted(graph.nodes):
        if graph.degree(node_id) == 0:
            issues.append(Issue("warn", "ISOLATED_NODE", f"{node_id} has no channel", [node_id]))
        node = graph.nodes[node_id]
        unconnected_out = [
            p.name
            for p in node.outputs
            if not any(e.source == node_id and e.source_port == p.name for e in graph.edges)
        ]
        unconnected_in = [
            p.name
            for p in node.inputs
            if not any(e.target == node_id and e.target_port == p.name for e in graph.edges)
        ]
        if unconnected_out or unconnected_in:
            issues.append(
                Issue(
                    "info",
                    "DANGLING_PORT",
                    f"{node_id}: dangling out {unconnected_out} in {unconnected_in}",
                    [node_id],
                )
            )
    return issues


def auto_layout(
    graph: TopologyGraph,
    origin_x: float = 60.0,
    origin_y: float = 60.0,
    column_step: float = 300.0,
    row_step: float = 150.0,
    heights: Optional[Dict[str, float]] = None,
) -> Dict[str, Tuple[float, float]]:
    """Layered layout positions for every node; cyclic graphs fall back to a ring.

    ``heights`` maps a node id to the height its card occupies on the canvas: a node with
    a long port stack then gets pushed further down its own column instead of overlapping
    the next one. Without it every row uses ``row_step``.

    The computed positions are written back on the node specs as well, so a graph that
    was laid out once keeps that layout when the canvas saves it and reads it again.
    """
    footprint = heights or {}
    layers = dag_layers(graph)
    positions: Dict[str, Tuple[float, float]] = {}
    if layers:
        for column, layer in enumerate(layers):
            y = origin_y
            for node_id in layer:
                positions[node_id] = (origin_x + column * column_step, y)
                y += max(row_step, float(footprint.get(node_id, 0.0)) + 30.0)
    else:
        ring = sorted(graph.nodes)
        radius = max(180.0, 26.0 * len(ring))
        for index, node_id in enumerate(ring):
            angle = 2 * math.pi * index / max(1, len(ring))
            positions[node_id] = (
                origin_x + radius + radius * math.cos(angle),
                origin_y + radius + radius * math.sin(angle),
            )
    for node_id, position in positions.items():
        node = graph.nodes.get(node_id)
        if node is not None:
            node.position = position
    return positions


def save_graph(graph: TopologyGraph, path: Optional[Path] = None, registry: Optional[KindRegistry] = None) -> Path:
    """Write the graph in the workbench native dialect, kinds included.

    The kind block records only the kinds the graph actually uses and that the registry
    did not ship with, so a new kind introduced on the canvas survives a reload and can
    also be edited by hand in the JSON file without touching the interface.
    """
    target = Path(path) if path is not None else graph.source
    if target is None:
        raise ValueError("no destination given and this graph was never loaded from a file")
    body = graph.to_workbench_dict()
    if registry is not None:
        declared = []
        for kind in sorted({node.kind for node in graph.nodes.values()}):
            spec = registry.get(kind)
            if not spec.builtin:
                declared.append(spec.to_dict())
        if declared:
            body["kinds"] = declared
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.suffix.lower() in YAML_SUFFIXES:
        yaml = _yaml_module(target)
        text = yaml.safe_dump(body, sort_keys=False, allow_unicode=True, default_flow_style=False)
        target.write_text(text, encoding="utf-8")
    else:
        target.write_text(json.dumps(body, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    graph.source = target
    return target


def graph_from_document(document: Dict[str, Any], source: Optional[Path] = None, registry: Optional[KindRegistry] = None, datasets_dir: Optional[Path] = None) -> TopologyGraph:
    """Build a graph from an already parsed document, dispatching on its dialect."""
    handle = registry or KindRegistry()
    dialect = classify_document(document)
    if dialect == "orchestrator":
        return _load_orchestrator_dialect(document, source, handle, datasets_dir)
    if dialect == "federated":
        return _load_federated_dialect(document, source, handle, datasets_dir)
    if dialect == "native":
        return _load_native_dialect(document, source, handle)
    graph = _load_native_dialect(document, source, handle)
    graph.notes.append(f"document shape unrecognised, read as native with {len(graph.nodes)} nodes")
    return graph


__all__ = [
    "DIRECTION_IN",
    "DIRECTION_OUT",
    "EdgeSpec",
    "Issue",
    "KindRegistry",
    "KindSpec",
    "NodeSpec",
    "PortSpec",
    "TopologyGraph",
    "auto_layout",
    "classify_document",
    "dag_layers",
    "derive_ports_from_mapping",
    "find_cycles",
    "graph_from_board",
    "graph_from_document",
    "load_graph_file",
    "save_graph",
    "validate_graph",
]
