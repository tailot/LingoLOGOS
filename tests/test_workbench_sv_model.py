"""SystemVerilog analyzer of the workbench: what a testbench author cares to see.

Covers:
  1. Localparams with their literal width and numeric value, and the mask subset of them.
  2. Modules, instances, tasks and functions with the line each one sits on.
  3. Fault signal names and the verdict lines a simulation prints.
  4. The highlighting contract: rule order, keyword pattern, symbol and range patterns.
  5. The real testbench of this repository, parsed without losing any of its shape.
"""

from pathlib import Path

import pytest

from gui.model.svdoc import SvAnalysis, SvSymbol, load_sv

REPO_ROOT = Path(__file__).resolve().parents[1]
SIM_DIR = REPO_ROOT / "asic_chip" / "sim"

SOURCE = """// fabric testbench: masks and fault injection
module fabric_tb;
  /* block comment mentioning GAMMA_MESH only in prose */
  localparam int DIES = 4;
  localparam logic [3:0] ALPHA_MESH = 4'b1011;
  localparam logic [3:0] BETA_MASK = 4'b0101;
  localparam int DATA_WIDTH = 64;

  logic inject_error;
  logic miswire_route;

  lingologos_topology_switch u_switch_alpha (
    .node_id(4'd0),
    .routing_mask(ALPHA_MESH)
  );

  task automatic drive_channel(input int slot);
    inject_error = 1'b1;
  endtask : drive_channel

  function int odd_of(input int value);
    return value % 2;
  endfunction : odd_of

  initial begin
    assign sel = GAMMA_MESH;
    if (BETA_MASK !== 4'b0101) begin
      $display("FINAL VERDICT: FAIL");
    end else begin
      $display("FINAL VERDICT: PASS");
    end
  end
endmodule
"""


def analysis():
    return SvAnalysis.parse(SOURCE)


def test_localparams_are_collected_with_their_line_and_value():
    parsed = analysis()
    assert [symbol.name for symbol in parsed.symbols] == [
        "DIES",
        "ALPHA_MESH",
        "BETA_MASK",
        "DATA_WIDTH",
    ]
    dies = parsed.symbol_named("DIES")
    assert dies.kind == "localparam"
    assert dies.note == "4"
    assert dies.line == SOURCE.splitlines().index("  localparam int DIES = 4;") + 1
    assert parsed.symbol_named("DATA_WIDTH").note == "64"


def test_mesh_and_mask_parameters_are_marked_as_masks_with_their_width():
    parsed = analysis()
    alpha = parsed.symbol_named("ALPHA_MESH")
    assert alpha.kind == "mask"
    assert alpha.width == 4
    assert alpha.note == "11"
    assert parsed.symbol_named("BETA_MASK").kind == "mask"
    assert sorted(symbol.name for symbol in parsed.mesh_masks()) == ["ALPHA_MESH", "GAMMA_MESH"]
    assert sorted(symbol.name for symbol in parsed.masks if symbol.kind == "mask") == [
        "ALPHA_MESH",
        "BETA_MASK",
    ]


def test_a_mask_that_is_only_referenced_is_still_surfaced():
    parsed = analysis()
    referenced = [symbol for symbol in parsed.masks if symbol.kind == "mask-ref"]
    assert [symbol.name for symbol in referenced] == ["GAMMA_MESH"]
    assert referenced[0].line > 0
    assert "GAMMA_MESH" not in [symbol.name for symbol in parsed.symbols]


def test_a_symbol_row_is_the_four_columns_the_panel_table_shows():
    row = analysis().symbol_named("ALPHA_MESH").as_row()
    assert row[0] == "mask" and row[1] == "ALPHA_MESH" and row[2] == "4'b1011"
    assert row[3].isdigit() and int(row[3]) > 1


def test_modules_instances_tasks_and_functions_carry_their_origin():
    parsed = analysis()
    assert [symbol.name for symbol in parsed.modules] == ["fabric_tb"]
    assert len(parsed.instances) == 1
    instance = parsed.instances[0]
    assert instance.name == "u_switch_alpha"
    assert instance.value == "lingologos_topology_switch"
    assert "node_id=4'd0" in instance.note
    assert instance.line > 0

    assert [symbol.name for symbol in parsed.tasks] == ["drive_channel"]
    assert parsed.tasks[0].note == "automatic"
    assert parsed.tasks[0].value == "input int slot"

    assert [symbol.name for symbol in parsed.functions] == ["odd_of"]
    assert parsed.functions[0].value == "input int value"
    assert parsed.functions[0].note == "int"


def test_fault_signals_and_verdict_lines_are_found_in_the_text_not_in_comments():
    parsed = analysis()
    assert sorted(symbol.name for symbol in parsed.faults) == ["inject_error", "miswire_route"]
    assert parsed.faults[0].kind == "fault"
    verdicts = parsed.verdicts
    assert len(verdicts) == 2
    assert [text for _, text in verdicts] == ["FINAL VERDICT: FAIL", "FINAL VERDICT: PASS"]
    assert all(line > 0 for line, _ in verdicts)
    lines = SOURCE.splitlines()
    assert lines[verdicts[0][0] - 1].strip().startswith("$display")


def test_the_summary_counts_what_the_panel_displays():
    parsed = analysis()
    summary = parsed.summary()
    assert summary["path"] == "<memory>"
    assert summary["lines"] == SOURCE.count("\n") + 1
    assert summary["modules"] == len(parsed.modules)
    assert summary["instances"] == len(parsed.instances)
    assert summary["tasks"] == len(parsed.tasks)
    assert summary["functions"] == len(parsed.functions)
    assert summary["localparams"] == len(parsed.symbols)
    assert summary["masks"] == len(parsed.masks)
    assert summary["fault_signals"] == len(parsed.faults)
    assert summary["verdict_lines"] == len(parsed.verdicts)
    assert parsed.parameter_map()["DIES"] == 4
    assert parsed.parameter_map()["DATA_WIDTH"] == 64


def test_an_unparsable_fragment_never_raises_and_reports_nothing():
    odd = SvAnalysis.parse("}{ ][ ; not verilog at all\n\n")
    assert odd.modules == [] and odd.symbols == [] and odd.instances == []
    assert odd.summary()["lines"] == 3


def test_the_highlighting_rules_are_ordered_and_cover_the_source():
    parsed = analysis()
    rules = parsed.rule_table()
    assert rules, "the source view needs at least one rule to colour anything"
    names = [name for _, name in rules]
    assert names == list(dict.fromkeys(names)), "a rule name is used once, so order is meaningful"
    for pattern, _ in rules:
        assert pattern.pattern
    for name in ("comment", "string", "keyword", "mask", "verdict", "decl"):
        assert name in names
    assert names.index("comment") < names.index("keyword"), "a keyword inside a comment stays a comment"

    assert parsed.keyword_pattern().search("localparam int DIES") is not None
    assert parsed.local_symbol_pattern().search("localparam logic [3:0] ALPHA_MESH = 4'b1011;")
    whole_line = parsed.local_range_pattern()
    assert whole_line.match("  localparam int DIES = 4;") is not None
    assert whole_line.match("  assign sel = GAMMA_MESH;") is None


def test_loading_a_file_keeps_its_path_and_its_text(tmp_path):
    path = tmp_path / "fabric_tb.sv"
    path.write_text(SOURCE, encoding="utf-8")
    parsed = load_sv(path)
    assert parsed.source == path
    assert parsed.text == SOURCE
    assert parsed.summary()["path"] == str(path)
    assert parsed.summary()["verdict_lines"] == 2


@pytest.mark.skipif(not SIM_DIR.is_dir(), reason="no asic_chip sim directory in this checkout")
def test_the_real_testbench_of_the_repository_is_understood():
    sources = sorted(SIM_DIR.glob("*.sv"))
    assert sources
    for path in sources:
        parsed = load_sv(path)
        assert parsed.modules, f"{path.name} declares no module"
        assert isinstance(parsed.tasks, list) and isinstance(parsed.masks, list)
        assert isinstance(parsed.faults, list) and isinstance(parsed.verdicts, list)
        summary = parsed.summary()
        assert summary["lines"] > 10
        assert all(isinstance(symbol, SvSymbol) for symbol in parsed.symbols)
