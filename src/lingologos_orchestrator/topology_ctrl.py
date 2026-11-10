"""CSR Register TOPOLOGY_CTRL definitions and bitfield helpers for LingoLOGOS.

RFC Identifier: LingoLOGOS-MULTI-ORCH-v1
Edge-NSAC Dynamic Board Topology Control:
  Bit 0: STANDALONE (1 = internal loopback, 0 = external mesh links)
  Bits 4:1: NODE_ID (read from physical strapping pins HARDWARE_NODE_ID[3:0])
  Bits 8:5: TOTAL_NODES (total chips detected on board)
  Bits 15:9: DATASET_ID (logical dataset shard assigned to this chip)
  Bits 31:16: ROUTING_MASK (forwarding mask for adjacent nodes)
"""

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class TopologyCtrlRegister:
    """Represents the 32-bit TOPOLOGY_CTRL CSR register."""

    standalone: int = 1
    node_id: int = 0
    total_nodes: int = 1
    dataset_id: int = 0
    routing_mask: int = 0

    def __post_init__(self) -> None:
        self.standalone = int(bool(self.standalone))
        self.node_id = self.node_id & 0x0F
        self.total_nodes = self.total_nodes & 0x0F
        self.dataset_id = self.dataset_id & 0x7F
        self.routing_mask = self.routing_mask & 0xFFFF

    def pack(self) -> int:
        """Packs bitfields into a single 32-bit unsigned integer."""
        raw = 0
        raw |= (self.standalone & 0x01)
        raw |= (self.node_id & 0x0F) << 1
        raw |= (self.total_nodes & 0x0F) << 5
        raw |= (self.dataset_id & 0x7F) << 9
        raw |= (self.routing_mask & 0xFFFF) << 16
        return raw & 0xFFFFFFFF

    @classmethod
    def unpack(cls, raw: int) -> "TopologyCtrlRegister":
        """Unpacks a 32-bit unsigned integer into TopologyCtrlRegister."""
        raw = raw & 0xFFFFFFFF
        standalone = raw & 0x01
        node_id = (raw >> 1) & 0x0F
        total_nodes = (raw >> 5) & 0x0F
        dataset_id = (raw >> 9) & 0x7F
        routing_mask = (raw >> 16) & 0xFFFF
        return cls(
            standalone=standalone,
            node_id=node_id,
            total_nodes=total_nodes,
            dataset_id=dataset_id,
            routing_mask=routing_mask,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Returns dictionary representation."""
        return {
            "standalone": self.standalone,
            "node_id": self.node_id,
            "total_nodes": self.total_nodes,
            "dataset_id": self.dataset_id,
            "routing_mask": self.routing_mask,
            "raw_hex": f"0x{self.pack():08X}",
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TopologyCtrlRegister":
        """Builds TopologyCtrlRegister from dictionary."""
        if "raw_hex" in data and len(data) == 1:
            return cls.unpack(int(data["raw_hex"], 16))
        return cls(
            standalone=int(data.get("standalone", 1)),
            node_id=int(data.get("node_id", 0)),
            total_nodes=int(data.get("total_nodes", 1)),
            dataset_id=int(data.get("dataset_id", 0)),
            routing_mask=int(data.get("routing_mask", 0)),
        )

    @classmethod
    def from_handshake(
        cls,
        hardware_node_id: int,
        discovered_peers_count: int,
        dataset_id: int = 0,
        routing_mask: int = 0,
    ) -> "TopologyCtrlRegister":
        """Creates configuration based on hardware strapping and network probe.

        If discovered_peers_count == 0 (probe timeout), standalone mode is enabled.
        If discovered_peers_count > 0, multi-chip interconnected mesh mode is enabled.
        """
        if discovered_peers_count == 0:
            return cls(
                standalone=1,
                node_id=hardware_node_id,
                total_nodes=1,
                dataset_id=dataset_id,
                routing_mask=routing_mask,
            )
        return cls(
            standalone=0,
            node_id=hardware_node_id,
            total_nodes=1 + discovered_peers_count,
            dataset_id=dataset_id,
            routing_mask=routing_mask,
        )
