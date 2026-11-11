"""Model Consolidation Engine for LingoLOGOS.

Performs graph contraction and clustering over distributed N-node topologies,
reducing the multi-agent mesh into:
  1. 2 or 3 macro-nodes (Proponent, Opponent, Validator).
  2. A single unified execution model optimized for local execution.

Adheres to English-only, ASCII-only rules and avoids apostrophes.
"""

from typing import Dict, List, Any, Set, Tuple
from dataclasses import dataclass

from .schema import (
    TopologyConfig,
    InstanceConfig,
    RoutingRule,
    FederatedTopologyConfig,
    FederatedNodeConfig,
    FederatedRoutingRule,
)


@dataclass
class ConsolidatedModel:
    """Represents a contracted topology configuration with clustered macro-nodes."""
    mode: str  # "triad" (3 macro nodes), "dyad" (2 macro nodes), or "unified" (1 macro node)
    node_mapping: Dict[str, str]  # maps original node_id -> macro_node_id
    macro_nodes: List[Dict[str, Any]]
    macro_routing: List[Dict[str, Any]]

    def representatives(self) -> Dict[str, str]:
        """Maps each macro node id to the original node whose dataset it executes on."""
        return {
            m["id"]: m["representative"]
            for m in self.macro_nodes
            if m.get("representative")
        }

    def to_topology_config(self, session_id: str, max_turns: int = 15) -> TopologyConfig:
        """Converts consolidated model to a runnable TopologyConfig."""
        instances = [
            InstanceConfig(
                id=m["id"],
                executable=m.get("executable", "lllayer"),
                role=m.get("role", "node"),
            )
            for m in self.macro_nodes
        ]
        routing_rules = [
            RoutingRule(from_node=r["from"], to_nodes=r["to"])
            for r in self.macro_routing
        ]
        return TopologyConfig(
            sessionId=session_id,
            maxTurns=max_turns,
            terminationCondition="Z3_SAT_CONVERGENCE_OR_UNSAT_BREAKER",
            instances=instances,
            routingMatrix=routing_rules,
        )


class ModelConsolidator:
    """Engine for contracting N-node topologies into macro-nodes or a unified model."""

    @staticmethod
    def _classify_role(role_name: str) -> str:
        """Categorizes arbitrary role strings into primary triad roles."""
        role_lower = role_name.lower()
        if any(k in role_lower for k in ["valid", "arbit", "verif", "check", "eval"]):
            return "validator"
        if any(k in role_lower for k in ["oppon", "critic", "contest", "adversary"]):
            return "opponent"
        return "proponent"

    def consolidate_to_triad(
        self,
        nodes: List[Dict[str, Any]],
        edges: List[Tuple[str, str]],
    ) -> ConsolidatedModel:
        """Consolidates an arbitrary set of N nodes into 3 macro nodes (Proponent, Opponent, Validator)."""
        mapping: Dict[str, str] = {}
        triad_nodes = {
            "macro_proponent": {"id": "macro_proponent", "role": "proponent", "executable": "lllayer", "members": []},
            "macro_opponent": {"id": "macro_opponent", "role": "opponent", "executable": "lllayer", "members": []},
            "macro_validator": {"id": "macro_validator", "role": "validator", "executable": "lllayer", "members": []},
        }

        for n in nodes:
            nid = n.get("node_id") or n.get("id")
            role = n.get("role", "node")
            cat = self._classify_role(role)
            target_macro = f"macro_{cat}"
            mapping[nid] = target_macro
            triad_nodes[target_macro]["members"].append(nid)

        # A macro node with no member has no dataset and no role to play, so it is dropped and the
        # contraction degenerates to a dyad. Each surviving macro node runs on the dataset of its
        # first member, which is the representative recorded below.
        macro_nodes = [m for m in triad_nodes.values() if m["members"]]
        for m in macro_nodes:
            m["representative"] = m["members"][0]
        surviving_ids = [m["id"] for m in macro_nodes]

        # Consolidate directed edges
        macro_edges: Dict[str, Set[str]] = {m: set() for m in surviving_ids}
        for src, dst in edges:
            m_src = mapping.get(src)
            m_dst = mapping.get(dst)
            if m_src in macro_edges and m_dst in macro_edges and m_src != m_dst:
                macro_edges[m_src].add(m_dst)

        # If no inter-macro edges existed, ensure the standard cyclical verification pattern
        if surviving_ids and not any(macro_edges.values()):
            if len(surviving_ids) > 1:
                for idx, src in enumerate(surviving_ids):
                    macro_edges[src].add(surviving_ids[(idx + 1) % len(surviving_ids)])

        macro_routing = [
            {"from": src, "to": sorted(list(targets))}
            for src, targets in macro_edges.items()
            if targets
        ]

        return ConsolidatedModel(
            mode="triad",
            node_mapping=mapping,
            macro_nodes=macro_nodes,
            macro_routing=macro_routing,
        )

    def consolidate_to_unified(
        self,
        nodes: List[Dict[str, Any]],
    ) -> ConsolidatedModel:
        """Consolidates all N nodes into a single unified execution model."""
        mapping: Dict[str, str] = {}
        all_ids = []
        for n in nodes:
            nid = n.get("node_id") or n.get("id")
            mapping[nid] = "unified_macro_engine"
            all_ids.append(nid)

        macro_node = {
            "id": "unified_macro_engine",
            "role": "unified_engine",
            "executable": "lllayer",
            "members": all_ids,
            "representative": all_ids[0] if all_ids else None,
        }

        # A single engine has no peer to talk to: the routing matrix is left empty on purpose, and a
        # session over it ends with EMPTY_ROUTING_MATRIX. Unified mode is therefore the model
        # contraction used for offline analysis, while triad mode is the runnable consolidation.
        return ConsolidatedModel(
            mode="unified",
            node_mapping=mapping,
            macro_nodes=[macro_node],
            macro_routing=[],
        )

    def consolidate_federation(
        self,
        federation: FederatedTopologyConfig,
        mode: str = "triad",
    ) -> ConsolidatedModel:
        """Consolidates a FederatedTopologyConfig instance."""
        node_dicts = [
            {"node_id": n.node_id, "role": n.role}
            for n in federation.nodes
        ]
        edges: List[Tuple[str, str]] = []
        for n in federation.nodes:
            for out in n.outbound_channels:
                edges.append((n.node_id, out))
        for r in federation.routing_rules:
            edges.append((r.source, r.target))

        if mode == "unified":
            return self.consolidate_to_unified(node_dicts)
        return self.consolidate_to_triad(node_dicts, edges)

    def consolidate_topology(
        self,
        topology: TopologyConfig,
        mode: str = "triad",
    ) -> ConsolidatedModel:
        """Consolidates a legacy TopologyConfig instance."""
        node_dicts = [
            {"id": i.id, "role": i.role}
            for i in topology.instances
        ]
        edges: List[Tuple[str, str]] = []
        for r in topology.routingMatrix:
            for t in r.to_nodes:
                edges.append((r.from_node, t))

        if mode == "unified":
            return self.consolidate_to_unified(node_dicts)
        return self.consolidate_to_triad(node_dicts, edges)
