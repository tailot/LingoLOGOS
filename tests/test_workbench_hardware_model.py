"""Hardware model of the workbench: the board comes out of the files, not out of a constant.

Covers:
  1. Verilog literal, range and comment helpers the board reader is built on.
  2. A generated N die wrapper: the reader adapts to 2, 4 and 8 dies without a code change.
  3. The exported board of this repository: masks, links, pins and parameters as read.
  4. The board derived from the orchestrator topology document instead of from Verilog.
  5. Fault injection and miswiring sites, masks, tasks and verdicts of a testbench.
  6. The role index the board viewer queries.
"""

from pathlib import Path

import pytest

from gui.model.hardware import (
    ArtifactIndex,
    BoardDescriptor,
    board_from_topology_json,
    find_instances,
    load_board,
    owner_channel_width,
    parse_board_verilog,
    parse_fault_plan,
    parse_verilog_literal,
    read_parameters,
    resolve_range,
    strip_comments,
)
from gui.paths import Artifact

REPO_ROOT = Path(__file__).resolve().parents[1]
BOARD_WRAPPER = (
    REPO_ROOT / "export_artifacts" / "visualization_board" / "system_board" / "board_complete.v"
)
TOPOLOGY_DOCUMENT = REPO_ROOT / "config" / "topology.json"
needs_board = pytest.mark.skipif(
    not BOARD_WRAPPER.is_file(),
    reason="the generated board wrapper is absent: run the ASIC export to produce it",
)


def die_names(count):
    return ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta"][:count]


def board_text(count, max_nodes=8):
    """Write a board wrapper for exactly ``count`` dies, fully meshed pairwise.

    The text is deliberately ordinary Verilog of the shape the export emits: a module
    header, parameters, per die channel port arrays, one switch plus one core instance
    per die, and one link comment per directed pair. Nothing in the reader is told the
    die count: it has to discover it here.
    """
    names = die_names(count)
    lines = [
        "module generated_board #(",
        "  parameter NODE_ID_WIDTH = 4,",
        "  parameter DATA_WIDTH = 64,",
        f"  parameter MAX_NODES = {max_nodes}",
        ") (",
        "  input wire clk,",
        "  input wire rst_n,",
    ]
    for name in names:
        lines.append(f"  input  wire [DATA_WIDTH-1:0] {name}_link_rx_data [0],")
        lines.append(f"  output wire [DATA_WIDTH-1:0] {name}_link_tx_data [0],")
    lines.append(");")
    index = 0
    for src in range(count):
        for dst in range(count):
            if src == dst:
                continue
            lines.append(
                f"  // link {index}: node {src} ({names[src]}) to node {dst} ({names[dst]})"
            )
            index += 1
    for position, name in enumerate(names):
        mask = ((1 << max_nodes) - 1) ^ (1 << position)
        lines.append(f"  lingologos_topology_switch #(")
        lines.append(f"    .NODE_ID_WIDTH(NODE_ID_WIDTH)")
        lines.append(f"  ) u_switch_{name} (")
        lines.append(f"    .node_id(4'd{position}),")
        lines.append(f"    .total_nodes(4'd{count}),")
        lines.append(f"    .routing_mask(8'b{mask:08b}),")
        lines.append(f"    .standalone(1'b0)")
        lines.append(f"  );")
        lines.append(f"  lingologos_core u_chip_{name} (")
        lines.append(f"    .clk(clk)")
        lines.append(f"  );")
    lines.append(f"  assign {names[0]}_link_tx_data [7] = 8'd0;")
    lines.append("endmodule")
    return "\n".join(lines) + "\n"


def write(tmp_path, text, name="generated_board.v"):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_verilog_literals_are_read_with_their_width():
    assert parse_verilog_literal("4'b0110") == (4, 6)
    assert parse_verilog_literal("8'hFF") == (8, 255)
    assert parse_verilog_literal("2'o17") == (2, 15)
    assert parse_verilog_literal("4'd12,") == (4, 12)
    assert parse_verilog_literal("1'bz") == (1, None)
    assert parse_verilog_literal("17") == (None, 17)
    assert parse_verilog_literal("not a literal") == (None, None)


def test_declared_ranges_resolve_through_the_parameter_table():
    assert resolve_range("[3:0]", {}) == (4, "[3:0]")
    assert resolve_range("[NODE_ID_WIDTH-1:0]", {"NODE_ID_WIDTH": 4}) == (4, "[NODE_ID_WIDTH-1:0]")
    assert resolve_range("[DATA_WIDTH-1:0]", {"DATA_WIDTH": 64}) == (64, "[DATA_WIDTH-1:0]")
    assert resolve_range(None, {}) == (1, "scalar")
    width, note = resolve_range("[MYSTERY-1:0]", {})
    assert width == 1 and "unresolved" in note


def test_comment_stripping_keeps_line_numbers_and_removes_mask_words():
    source = "a // MASK = 4'b1111\nb /* MASK = 4'b0001\nc */ d\n"
    cleaned = strip_comments(source)
    assert len(cleaned.splitlines()) == len(source.splitlines()) == 3
    assert "MASK" not in cleaned
    assert cleaned.splitlines()[0].strip() == "a"
    assert cleaned.splitlines()[1].strip() == "b"
    assert cleaned.splitlines()[2].split() == ["d"]


def test_channel_width_helper_uses_the_declared_bus_width():
    parameters = {"DATA_WIDTH": 64, "NODE_ID_WIDTH": 4}
    assert owner_channel_width("link_tx_data", parameters) >= 1
    assert "DATA_WIDTH" in parameters


def test_parameters_and_instances_are_discovered_without_hardcoded_names():
    text = board_text(3)
    parameters = read_parameters(text)
    assert parameters["NODE_ID_WIDTH"] == 4 and parameters["MAX_NODES"] == 8
    instances = find_instances(strip_comments(text))
    named = {(module, instance) for module, instance, _, _ in instances}
    assert ("lingologos_topology_switch", "u_switch_alpha") in named
    assert ("lingologos_core", "u_chip_beta") in named


@pytest.mark.parametrize("count", [2, 4, 8])
def test_the_board_reader_adapts_to_any_die_count(tmp_path, count):
    board = parse_board_verilog(write(tmp_path, board_text(count, max_nodes=count)))
    assert board.die_count == count
    assert [die.name for die in board.dies] == die_names(count)
    assert len(board.links) == count * (count - 1)
    matrix = board.adjacency_matrix()
    assert len(matrix) == count and all(len(row) == count for row in matrix)
    for position, die in enumerate(board.dies):
        assert die.mask_bits()[:count] == [0 if bit == position else 1 for bit in range(count)]
        assert die.enabled_slots() == [bit for bit in range(count) if bit != position]
        assert die.is_enabled(position) is False
        assert matrix[position] == [0 if bit == position else 1 for bit in range(count)]
    pairs = board.channel_pairs()
    assert len(pairs) == count * (count - 1) // 2
    for forward, reverse in pairs:
        assert forward.source_index == reverse.target_index
        assert forward.target_index == reverse.source_index
    assert board.max_nodes == count
    assert board.warnings == []


def test_channel_ports_are_grouped_under_the_die_that_owns_them(tmp_path):
    board = parse_board_verilog(write(tmp_path, board_text(4)))
    alpha = board.die_by_name("alpha")
    assert alpha is not None
    assert alpha.chip_instance == "u_chip_alpha"
    assert alpha.switch_instance == "u_switch_alpha"
    assert alpha.pins, "the channel stack of a die must be read from the port list"
    groups = alpha.pins_grouped()
    assert any("rx" in key or "tx" in key for key in groups)
    assert board.die_by_name("nobody") is None


def test_tied_off_channel_slots_are_reported_as_not_driven(tmp_path):
    text = board_text(3).replace(
        "  assign alpha_link_tx_data [7] = 8'd0;",
        "  assign beta_link_tx_valid [7] = 1'b0;",
    )
    board = parse_board_verilog(write(tmp_path, text))
    beta = board.die_by_name("beta")
    assert beta is not None
    assert 7 in beta.tied_slots


@needs_board
def test_the_exported_board_of_this_repository_is_read_as_it_is_written():
    board = load_board(BOARD_WRAPPER)
    assert board.die_count == 3
    assert {die.routing_mask for die in board.dies} == {6, 5, 3}
    assert len(board.links) == 6
    assert board.adjacency_matrix() == [[0, 1, 1], [1, 0, 1], [1, 1, 0]]
    assert board.parameters["NODE_ID_WIDTH"] == 4
    assert board.parameters["DATA_WIDTH"] == 64
    assert board.parameters["MAX_NODES"] == 16
    assert board.max_nodes == 16
    assert len(board.channel_pairs()) == 3
    summary = board.summary()
    assert summary["dies"] == 3 and summary["links"] == 6
    assert summary["parameters"]["MAX_NODES"] == 16


def test_a_board_also_comes_from_the_topology_document():
    board = load_board(TOPOLOGY_DOCUMENT)
    assert isinstance(board, BoardDescriptor)
    assert board.die_count == 3
    assert board.provenance != "verilog-board"
    assert board.links


@needs_board
def test_json_and_verilog_describe_the_same_three_dies():
    from_verilog = load_board(BOARD_WRAPPER)
    from_json = load_board(TOPOLOGY_DOCUMENT)
    assert from_verilog.die_count == from_json.die_count


FAULT_TEXT = """
module topology_fault_tb;
  localparam int DIES = 3;
  localparam logic [3:0] M0_MESH = 4'b0110;
  localparam logic [3:0] M1_MESH = 4'b0101;

  logic miswire_valid;
  logic miswire_dest;

  task automatic inject_fault(input int slot);
    miswire_valid = 1'b1;
  endtask : inject_fault

  lingologos_topology_switch u_switch_alpha (
    .routing_mask(M0_MESH)
  );
  lingologos_core die0 (
    .clk(clk)
  );

  initial begin
    miswire_valid = 1'b1;
    miswire_dest = 4'd9;
    $display("FINAL VERDICT: PASS");
  end
endmodule
"""


def test_a_testbench_yields_its_masks_faults_tasks_and_verdicts(tmp_path):
    path = write(tmp_path, FAULT_TEXT, "fault_tb.sv")
    plan = parse_fault_plan(path)
    assert plan.masks["M0_MESH"] == 6 and plan.masks["M1_MESH"] == 5
    assert plan.mask_widths["M0_MESH"] == 4
    assert plan.parameters["DIES"] == 3
    assert plan.die_count == 3
    assert sorted(plan.fault_signals) == ["miswire_dest", "miswire_valid"]
    assert len(plan.fault_signals["miswire_valid"]) == 2
    for site in plan.fault_signals["miswire_dest"]:
        assert site.line > 0 and site.value and site.signal == "miswire_dest"
    assert any(name == "inject_fault" for name, _ in plan.tasks)
    assert plan.verdicts and plan.verdicts[0][1] == "FINAL VERDICT: PASS"
    assert any(instance == "die0" for instance, _ in plan.die_instances)
    summary = plan.summary()
    assert summary["verdicts"] == len(plan.verdicts)
    assert summary["fault_sites"]["miswire_valid"] == 2


def test_the_real_topology_testbench_parses_without_losing_its_shape():
    path = REPO_ROOT / "asic_chip" / "sim" / "topology_testbench.sv"
    if not path.is_file():
        pytest.skip("the SystemVerilog testbench is not present in this checkout")
    plan = parse_fault_plan(path)
    assert plan.parameters, "the testbench declares localparams the viewer must show"
    assert isinstance(plan.masks, dict) and isinstance(plan.tasks, list)
    assert isinstance(plan.fault_signals, dict) and isinstance(plan.verdicts, list)


def test_the_role_index_answers_the_queries_of_the_board_viewer(tmp_path):
    first = write(tmp_path, "module a; endmodule\n", "older_board.v")
    second = write(tmp_path, "module b; endmodule\n", "board_complete.v")
    scan = {
        "board": [
            Artifact(first, "board", ".v", first.stat().st_size, "2024-01-01T00:00:00Z"),
            Artifact(second, "board", ".v", second.stat().st_size, "2024-01-02T00:00:00Z"),
        ],
        "topology": [],
    }
    index = ArtifactIndex.build(scan)
    assert index.counts() == {"board": 2, "topology": 0}
    assert index.first("board") == first
    assert index.paths_of("board") == [first, second]
    assert index.newest("board", hint="complete") == second
    assert index.newest("board", hint="nothing") is None
    assert index.first("absent") is None
    assert index.missing_roles(["board", "topology", "gds"]) == ["topology", "gds"]
