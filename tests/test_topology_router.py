"""Unit and integration tests for Edge-NSAC Dynamic Board Topology.

Covers:
  1. SystemVerilog router reference behavior (standalone loopback vs mesh routing).
  2. TOPOLOGY_CTRL register packing, unpacking, and field definitions.
  3. Software auto-discovery and handshake logic in lllayer.
  4. MultiInstanceOrchestrator topology negotiation and dataset partitioning.
"""

import json
import pytest
from pathlib import Path

from lingologos_orchestrator.topology_ctrl import TopologyCtrlRegister
from lingologos_orchestrator.lllayer import discover_board_topology
from lingologos_orchestrator.schema import TopologyConfig, Envelope, VolatileState, Payload, SMTConstraints
from lingologos_orchestrator.orchestrator import MultiInstanceOrchestrator


def simulate_topology_router(
    standalone_mode: bool,
    current_node_id: int,
    tx_valid_local: bool,
    tx_dest_node: int,
    tx_data_local: int,
    rx_ready_local: bool,
    rx_valid_ext: bool,
    rx_data_ext: int,
):
    """Python behavioral reference of lingologos_topology_router.sv."""
    if standalone_mode:
        rx_valid_local = tx_valid_local
        rx_data_local = tx_data_local
        tx_ready_local = rx_ready_local
        tx_valid_ext = False
        tx_data_ext = 0
    else:
        if tx_dest_node == current_node_id:
            rx_valid_local = tx_valid_local
            rx_data_local = tx_data_local
            tx_ready_local = rx_ready_local
            tx_valid_ext = False
            tx_data_ext = 0
        else:
            tx_valid_ext = tx_valid_local
            tx_data_ext = tx_data_local
            tx_ready_local = True
            rx_valid_local = rx_valid_ext
            rx_data_local = rx_data_ext

    return (
        tx_ready_local,
        rx_valid_local,
        rx_data_local,
        tx_valid_ext,
        tx_data_ext,
    )


def test_topology_router_standalone_loopback():
    """In standalone mode, local tx is looped back to local rx and ext tx is silenced."""
    tx_ready, rx_valid, rx_data, tx_val_ext, tx_dat_ext = simulate_topology_router(
        standalone_mode=True,
        current_node_id=1,
        tx_valid_local=True,
        tx_dest_node=5,  # even if dest differs, standalone forces internal loopback
        tx_data_local=0xDEADBEEFCAFE0001,
        rx_ready_local=True,
        rx_valid_ext=True,
        rx_data_ext=0x1111222233334444,
    )
    assert rx_valid is True
    assert rx_data == 0xDEADBEEFCAFE0001
    assert tx_ready is True
    assert tx_val_ext is False
    assert tx_dat_ext == 0


def test_topology_router_mesh_local_forwarding():
    """In mesh mode, packets addressed to current_node_id stay on local rx."""
    tx_ready, rx_valid, rx_data, tx_val_ext, tx_dat_ext = simulate_topology_router(
        standalone_mode=False,
        current_node_id=3,
        tx_valid_local=True,
        tx_dest_node=3,  # matches current node
        tx_data_local=0xAAAAAAAA55555555,
        rx_ready_local=True,
        rx_valid_ext=False,
        rx_data_ext=0,
    )
    assert rx_valid is True
    assert rx_data == 0xAAAAAAAA55555555
    assert tx_ready is True
    assert tx_val_ext is False
    assert tx_dat_ext == 0


def test_topology_router_mesh_ext_forwarding():
    """In mesh mode, packets addressed to foreign node IDs are routed to ext tx."""
    tx_ready, rx_valid, rx_data, tx_val_ext, tx_dat_ext = simulate_topology_router(
        standalone_mode=False,
        current_node_id=2,
        tx_valid_local=True,
        tx_dest_node=7,  # foreign destination
        tx_data_local=0x1234567890ABCDEF,
        rx_ready_local=True,
        rx_valid_ext=True,
        rx_data_ext=0xFEDCBA0987654321,
    )
    assert tx_val_ext is True
    assert tx_dat_ext == 0x1234567890ABCDEF
    assert tx_ready is True
    assert rx_valid is True
    assert rx_data == 0xFEDCBA0987654321


def test_topology_ctrl_register_packing_and_unpacking():
    """TOPOLOGY_CTRL packs and unpacks bitfields with exact width adherence."""
    reg = TopologyCtrlRegister(
        standalone=1,
        node_id=0b1011,      # 11
        total_nodes=0b0100,  # 4
        dataset_id=0b1010101,# 85
        routing_mask=0xABCD, # 43981
    )
    raw = reg.pack()
    assert isinstance(raw, int)

    unpacked = TopologyCtrlRegister.unpack(raw)
    assert unpacked.standalone == 1
    assert unpacked.node_id == 11
    assert unpacked.total_nodes == 4
    assert unpacked.dataset_id == 85
    assert unpacked.routing_mask == 0xABCD

    # Check to_dict and from_dict
    d = unpacked.to_dict()
    assert d["standalone"] == 1
    assert d["node_id"] == 11
    rebuilt = TopologyCtrlRegister.from_dict(d)
    assert rebuilt == unpacked


def test_auto_discovery_handshake():
    """Auto-discovery negotiates standalone on timeout and mesh on discovered peers."""
    # Probe timeout (0 discovered peers)
    sa = discover_board_topology(hardware_node_id=2, discovered_peers_count=0, dataset_id=0)
    assert sa.standalone == 1
    assert sa.node_id == 2
    assert sa.total_nodes == 1
    assert sa.dataset_id == 0

    # Multi-chip response (e.g. 2 peers discovered -> 3 total nodes)
    mc = discover_board_topology(hardware_node_id=1, discovered_peers_count=2, dataset_id=1, routing_mask=0x06)
    assert mc.standalone == 0
    assert mc.node_id == 1
    assert mc.total_nodes == 3
    assert mc.dataset_id == 1
    assert mc.routing_mask == 0x06


def test_orchestrator_negotiates_topology_and_partitions_datasets():
    """MultiInstanceOrchestrator assigns hardware IDs, dataset shards, and routing masks."""
    topology_path = Path("config/topology.json")
    datasets_dir = Path("datasets")
    assert topology_path.is_file()

    orch = MultiInstanceOrchestrator.from_config_file(
        config_path=topology_path,
        datasets_dir=datasets_dir,
    )
    regs = orch.negotiate_topology()
    assert len(regs) == 3

    # For multi-chip topology with 3 instances, standalone must be 0
    for node_id, reg in regs.items():
        assert reg.standalone == 0
        assert reg.total_nodes == 3
        assert reg.node_id in (0, 1, 2)
        assert reg.dataset_id in (0, 1, 2)
        assert reg.routing_mask > 0

    # Execute session and verify envelopes preserve datasetId and topologyCtrl
    res = orch.run_session()
    assert res.success is True
    assert len(res.records) > 0
    for rec in res.records:
        env = rec.envelope
        assert "topologyCtrl" in env
        assert "datasetId" in env
        assert env["topologyCtrl"]["standalone"] == 0
        assert env["topologyCtrl"]["total_nodes"] == 3
