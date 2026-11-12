"""Topology model of the workbench: dialect detection, ports, cycles, layouts.

Covers:
  1. Dialect detection for the orchestrator, the federated mesh and the workbench format.
  2. Ports and attributes derived for every node of the two real repository documents.
  3. Recursive DAG validation and cycle reporting, including the known cyclic config.
  4. Layered automatic layout and the save plus load round trip of a canvas graph.
  5. Kind extensibility: a new kind arrives from data, never from an edit of the UI code.
  6. The board projection, so the hardware viewer and the canvas share one graph model.
"""

from pathlib import Path

import pytest

from gui.model.hardware import load_board
from gui.model.topology import (
    EdgeSpec,
    KindRegistry,
    KindSpec,
    NodeSpec,
    PortSpec,
    TopologyGraph,
    auto_layout,
    classify_document,
    dag_layers,
    find_cycles,
    graph_from_board,
    graph_from_document,
    load_graph_file,
    save_graph,
    validate_graph,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
ORCHESTRATOR_DOCUMENT = REPO_ROOT / "config" / "topology.json"
FEDERATED_DOCUMENT = REPO_ROOT / "topologies" / "federated_mesh.json"
BOARD_WRAPPER = (
    REPO_ROOT / "export_artifacts" / "visualization_board" / "system_board" / "board_complete.v"
)
needs_board = pytest.mark.skipif(
    not BOARD_WRAPPER.is_file(),
    reason="the generated board wrapper is absent: run the ASIC export to produce it",
)


def read(path):
    import json

    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def errors_of(graph):
    """The findings that make a graph unusable, ignoring advisory notes."""
    return [issue for issue in validate_graph(graph) if issue.severity == "error"]


def port_pair():
    return [PortSpec(name="in"), PortSpec(name="out", direction="out")]


def test_dialect_is_detected_from_the_shape_of_the_document():
    assert classify_document(read(ORCHESTRATOR_DOCUMENT)) == "orchestrator"
    assert classify_document(read(FEDERATED_DOCUMENT)) == "federated"
    assert classify_document({"nodes": [{"id": "a", "kind": "generic"}], "edges": []}) == "native"
    assert classify_document({"nothing": True}) == "unknown"


def test_unknown_shape_is_refused_with_an_english_message(tmp_path):
    path = tmp_path / "mystery.json"
    path.write_text('{"totally": "unknown"}', encoding="utf-8")
    with pytest.raises(ValueError) as captured:
        load_graph_file(path)
    assert "dialect" in str(captured.value)


def test_orchestrator_config_becomes_a_graph_of_named_instances():
    graph = load_graph_file(ORCHESTRATOR_DOCUMENT)
    assert graph.dialect == "orchestrator"
    assert set(graph.nodes) == {"agente_alfa", "agente_beta", "arbitro_logico"}
    assert len(graph.edges) == 6
    for spec in graph.nodes.values():
        assert spec.kind and spec.label
        assert spec.ports, f"node {spec.node_id} has no port to connect to"
    stats = graph.stats()
    assert stats["nodes"] == 3 and stats["edges"] == 6
    assert stats["ports"] >= 6


def test_federated_mesh_becomes_a_graph_of_federated_nodes():
    graph = load_graph_file(FEDERATED_DOCUMENT)
    assert graph.dialect == "federated"
    assert len(graph.nodes) == 3 and len(graph.edges) == 4
    assert all("federated" in spec.kind or spec.kind in graph.nodes for spec in graph.nodes.values())


def test_the_known_configured_topology_is_cyclic_and_says_so():
    graph = load_graph_file(ORCHESTRATOR_DOCUMENT)
    cycles = find_cycles(graph)
    assert cycles, "config/topology.json declares a routing loop and must be reported as cyclic"
    assert all(len(cycle) >= 2 for cycle in cycles)
    codes = {issue.code for issue in validate_graph(graph)}
    assert "CYCLE" in codes


def test_validation_reports_the_loop_as_an_error_and_the_rest_as_advice():
    issues = validate_graph(load_graph_file(ORCHESTRATOR_DOCUMENT))
    assert issues
    severities = {issue.severity for issue in issues}
    assert severities <= {"error", "warning", "notice"}
    assert "error" in severities
    for issue in issues:
        assert issue.message and issue.code
        assert isinstance(issue.subjects, list)


def test_an_empty_graph_is_flagged_and_a_clean_chain_is_not():
    empty = TopologyGraph(name="empty")
    assert {issue.code for issue in validate_graph(empty)} == {"EMPTY_GRAPH"}
    chain = TopologyGraph(name="chain")
    for name in ("first", "second", "third"):
        chain.add_node(NodeSpec(node_id=name, kind="generic", label=name, ports=port_pair()))
    chain.connect(EdgeSpec(source="first", target="second", source_port="out", target_port="in"))
    chain.connect(EdgeSpec(source="second", target="third", source_port="out", target_port="in"))
    assert errors_of(chain) == []
    assert dag_layers(chain) == [["first"], ["second"], ["third"]]
    assert chain.stats()["acyclic"] is True


def test_dangling_port_endpoints_are_reported():
    graph = TopologyGraph(name="dangling")
    graph.add_node(NodeSpec(node_id="solo", kind="generic", label="solo", ports=port_pair()))
    graph.connect(EdgeSpec(source="solo", target="ghost", source_port="out", target_port="in"))
    codes = {issue.code for issue in validate_graph(graph)}
    assert "UNKNOWN_ENDPOINT" in codes


def test_auto_layout_separates_layers_and_survives_a_cycle(tmp_path):
    chain = TopologyGraph(name="chain")
    for name in ("a", "b", "c"):
        chain.add_node(NodeSpec(node_id=name, kind="generic", label=name, ports=port_pair()))
    chain.connect(EdgeSpec(source="a", target="b", source_port="out", target_port="in"))
    chain.connect(EdgeSpec(source="b", target="c", source_port="out", target_port="in"))
    placed = auto_layout(chain)
    assert len(placed) == 3
    assert placed["a"][0] < placed["b"][0] < placed["c"][0]
    looped = load_graph_file(ORCHESTRATOR_DOCUMENT)
    loop_positions = auto_layout(looped)
    assert len(loop_positions) == len(looped.nodes)
    assert len({pair for pair in loop_positions.values()}) == len(loop_positions)


def test_saved_graph_returns_its_positions_kinds_and_edges(tmp_path):
    graph = load_graph_file(ORCHESTRATOR_DOCUMENT)
    auto_layout(graph)
    path = save_graph(graph, tmp_path / "canvas_copy.json")
    assert path.is_file()
    again = load_graph_file(path)
    assert again.dialect == "native"
    assert set(again.nodes) == set(graph.nodes)
    assert len(again.edges) == len(graph.edges)
    for node_id, spec in again.nodes.items():
        assert spec.position is not None
        assert spec.position == pytest.approx(tuple(graph.nodes[node_id].position))
        assert spec.kind == graph.nodes[node_id].kind


def test_a_new_kind_arrives_from_data_and_not_from_the_canvas_code():
    registry = KindRegistry()
    assert "generic" in registry.known()
    synthesized = registry.get("optical_switch")
    assert synthesized.title == "Optical Switch"
    assert synthesized.default_in_ports == ("in",) and synthesized.default_out_ports == ("out",)
    added = registry.load_from_document(
        {
            "kinds": {
                "optical_switch": {
                    "title": "Optical Switch",
                    "accent": "link",
                    "in_ports": ["beam_in"],
                    "out_ports": ["beam_out"],
                    "attribute_keys": ["wavelength"],
                }
            }
        }
    )
    assert added == ["optical_switch"]
    merged = registry.get("optical_switch")
    assert merged.default_in_ports == ("beam_in",)
    assert merged.attribute_keys == ("wavelength",)
    document = {
        "spec": "lingologos-workbench-topology-v1",
        "name": "photonics",
        "kinds": {
            "optical_switch": {
                "title": "Optical Switch",
                "in_ports": ["beam_in"],
                "out_ports": ["beam_out"],
            }
        },
        "nodes": [
            {"id": "laser", "kind": "optical_switch", "label": "Laser"},
            {"id": "ring", "kind": "optical_switch", "label": "Ring"},
        ],
        "edges": [
            {"from": "laser", "to": "ring", "from_port": "beam_out", "to_port": "beam_in"}
        ],
    }
    graph = graph_from_document(document, registry=registry)
    assert set(graph.nodes) == {"laser", "ring"}
    assert graph.nodes["laser"].kind == "optical_switch"
    assert "beam_out" in [port.name for port in graph.nodes["laser"].ports]
    assert errors_of(graph) == []


def test_registry_spec_registration_overrides_a_builtin_without_touching_the_view():
    registry = KindRegistry()
    registry.register(KindSpec("generic", "Custom Generic", "warn", ("left",), ("right",)))
    spec = registry.get("generic")
    assert spec.title == "Custom Generic"
    assert spec.default_in_ports == ("left",)


@needs_board
def test_board_projection_shares_the_graph_model_and_validates_cleanly():
    board = load_board(BOARD_WRAPPER)
    graph = graph_from_board(board)
    assert len(graph.nodes) == board.die_count == 3
    assert len(graph.edges) == len(board.links) == 6
    assert all(edge.source in graph.nodes and edge.target in graph.nodes for edge in graph.edges)
    codes = {issue.code for issue in validate_graph(graph)}
    assert "UNKNOWN_ENDPOINT" not in codes and "SELF_LOOP" not in codes
