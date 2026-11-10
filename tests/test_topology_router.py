"""Unit and integration tests for Edge-NSAC Dynamic Board Topology.

Covers:
  1. SystemVerilog router reference behavior (standalone loopback vs mesh routing).
  2. TOPOLOGY_CTRL register packing, unpacking, and field definitions.
  3. Software auto-discovery and handshake logic in lllayer.
  4. MultiInstanceOrchestrator topology negotiation and dataset partitioning.
  5. Cross layer gates: the register fields, the board the config declares and the
     SystemVerilog switch under asic_chip, so the three of them cannot drift apart in
     silence.
"""

import json
import re
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
    """Python behavioral reference of the addressing rule of the switch interface.

    This model is the one the interface was specified with: it decides in one step where a
    frame goes and it hands the frame over without waiting. The registered switch in
    asic_chip/rtl/core/lingologos_topology_router.sv is stricter, it consults the routing
    mask, it refuses a destination that no channel of the die names, it takes two clock
    cycles, and it pushes back on the port of the core. Those properties are checked by
    the SystemVerilog bench asic_chip/sim/topology_testbench.sv, run through the
    sim-topology target of asic_chip/Makefile.
    """
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


# ---------------------------------------------------------------------------
# Cross layer gates.
#
# The register, the board the config declares and the switch written in
# SystemVerilog live in three different places. These tests read all three of
# them and compare what they say, because a number that appears in only one of
# them is a number that nobody checks.
# ---------------------------------------------------------------------------

TOPOLOGY_CONFIG = Path("config/topology.json")
RTL_SWITCH = Path("asic_chip/rtl/core/lingologos_topology_router.sv")
RTL_TESTBENCH = Path("asic_chip/sim/topology_testbench.sv")
ASIC_MAKEFILE = Path("asic_chip/Makefile")
ASIC_TOP = Path("asic_chip/rtl/top/lingologos_top.sv")
OPENLANE_CONFIGS = (
    Path("asic_chip/openlane/config.json"),
    Path("asic_chip/openlane/config.bringup.json"),
)


def _rtl_parameter(text, name):
    match = re.search(r"\b" + name + r"\s*=\s*([0-9]+)", text)
    assert match is not None, "the switch does not parameterize " + name
    return int(match.group(1))


def test_switch_is_sized_by_the_register_it_is_wired_to():
    """Every width of the switch is a width of TOPOLOGY_CTRL, and no field is dropped."""
    assert RTL_SWITCH.is_file()
    text = RTL_SWITCH.read_text(encoding="utf-8")
    node_width = _rtl_parameter(text, "NODE_ID_WIDTH")
    assert node_width == 4
    assert _rtl_parameter(text, "MAX_NODES") == 1 << node_width
    assert _rtl_parameter(text, "DATA_WIDTH") == 64

    probe = TopologyCtrlRegister(standalone=1, node_id=0x0F, total_nodes=0x0F,
                                 dataset_id=0x7F, routing_mask=0xFFFF)
    packed = probe.pack()
    assert packed == 0xFFFFFFFF
    # The mask field of the register spans one bit per node id, and the switch has one
    # channel per node id, so a bit of the mask and an index of a channel mean the same.
    assert ((packed >> 16) & 0xFFFF).bit_length() <= _rtl_parameter(text, "MAX_NODES")
    assert ((packed >> 1) & 0x0F) == (1 << node_width) - 1

    port_match = re.search(
        r"module\s+lingologos_topology_router\b[\s\S]*?\)\s*\(([\s\S]*?)\n\);",
        text,
    )
    assert port_match is not None, "the port list of the switch was not found"
    head = port_match.group(1)
    for port in ("standalone", "node_id", "total_nodes", "routing_mask", "routing_error",
                 "tx_valid", "tx_ready", "tx_dest", "tx_data", "rx_valid", "rx_ready",
                 "rx_data", "link_tx_valid", "link_tx_ready", "link_rx_valid"):
        assert re.search(r"\b" + port + r"\b", head) is not None, "no port " + port
    # DATASET_ID names the shard of the corpus a die works on, which is a matter for the
    # software on that die. The switch is not told about it and does not use it.
    assert "DATASET" not in head.upper()


def test_every_link_the_config_declares_is_a_direct_channel_of_the_switch():
    """The single hop policy of the switch covers the board this config declares."""
    declared = json.loads(TOPOLOGY_CONFIG.read_text(encoding="utf-8"))
    names = [instance["id"] for instance in declared["instances"]]
    hw_of = {name: index for index, name in enumerate(names)}
    matrix = {rule["from"]: set(rule["to"]) for rule in declared["routingMatrix"]}

    orch = MultiInstanceOrchestrator.from_config_file(
        config_path=TOPOLOGY_CONFIG,
        datasets_dir=Path("datasets"),
    )
    regs = orch.negotiate_topology()
    assert set(regs) == set(names)

    for name in names:
        reg = regs[name]
        expected = 0
        for peer in matrix[name]:
            expected |= 1 << hw_of[peer]
        assert reg.routing_mask == expected

        for bit in hw_of.values():
            if (reg.routing_mask >> bit) & 1:
                assert names[bit] in matrix[name]
                assert names[bit] != name

        # A die of this board has no path table and no relay: it hands a frame to the
        # channel of the destination or it refuses the frame. That is only enough when
        # every other die of the board is wired to it directly, which is what the mask
        # says and what this check locks.
        for other in names:
            if other == name:
                continue
            assert (reg.routing_mask >> hw_of[other]) & 1, (
                name + " is not wired to " + other
                + ", so the switch of that die would have to relay a frame"
            )


def test_the_switch_is_linted_and_simulated_but_not_synthesized():
    """The Makefile builds the switch, the place and route flow of this repo does not."""
    assert RTL_TESTBENCH.is_file()
    make = ASIC_MAKEFILE.read_text(encoding="utf-8")
    assert RTL_SWITCH.as_posix().split("asic_chip/")[-1] in make, "the Makefile never reads the switch source"
    assert RTL_TESTBENCH.as_posix().split("asic_chip/")[-1] in make, "the Makefile never reads the bench source"
    for target in ("lint-topology:", "sim-topology:"):
        assert target in make, "the Makefile declares no " + target + " target"

    # Not instantiated in the top level and not listed in the source list of OpenLane:
    # the switch is simulated, it has never been placed and routed. A commit that moves
    # it into either list has to update the wording of the verification report too.
    assert "lingologos_topology_router" not in ASIC_TOP.read_text(encoding="utf-8")
    for path in OPENLANE_CONFIGS:
        assert path.is_file()
        listed = json.loads(path.read_text(encoding="utf-8")).get("VERILOG_FILES", [])
        assert not any("topology_router" in entry for entry in listed)
