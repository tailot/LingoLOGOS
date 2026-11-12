"""Liquid panels: cards that fold, a canvas and a schematic that zoom and pan.

These tests build the real Qt widgets on the offscreen platform, so no window is shown.
They cover the three things the person asked for when the workbench felt cramped:

  1. Every ``widgets.Card`` is an accordion: closing it hides its body and hands the room
     back to the cards that stayed open, and ``share_column`` redistributes the column.
  2. A node card on the canvas keeps its title inside the header band, reserves a band for
     the port labels, grows with the number of ports and never outgrows the layout step.
  3. The board schematic keeps its current fit-to-view as the default, then lets the person
     zoom in and out and walk between blocks, and its canvas grows when there are many
     blocks so every one of them still has readable space.
"""

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

try:
    from PyQt6.QtCore import QPointF, QRectF
    from PyQt6.QtWidgets import QApplication, QVBoxLayout, QWidget
except ImportError as missing:  # no Qt runtime on this host: these widgets cannot be built
    pytest.skip(f"Qt is not loadable here: {missing}", allow_module_level=True)

from gui import theme, widgets
from gui.model.hardware import BoardDescriptor, BoardDie, BoardLink, PinSpec
from gui.model.topology import EdgeSpec, NodeSpec, PortSpec, TopologyGraph, auto_layout
from gui.paths import open_workspace
from gui.views import canvas
from gui.views.canvas import (
    CARD_STEP_X,
    HEADER_HEIGHT,
    LABEL_BAND,
    NODE_MAX_WIDTH,
    NODE_WIDTH,
    PORT_ROW,
    NodeItem,
    TopologyScene,
    TopologyView,
)
from gui.views.hardware import (
    HEADER_BAND,
    MAX_DIE_W,
    MAX_ZOOM,
    MIN_DIE_H,
    MIN_DIE_W,
    MIN_ZOOM,
    VIRTUAL_H,
    VIRTUAL_W,
    HardwarePanel,
    SchematicView,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def qapp() -> QApplication:
    app = QApplication.instance() or QApplication([])
    app.setApplicationName("LingoLOGOS layout test session")
    app.setStyleSheet(theme.stylesheet())
    return app


@pytest.fixture(scope="session")
def workspace():
    return open_workspace(REPO_ROOT)


def tall_spec(label: str, ports: int) -> NodeSpec:
    """A node spec with long-ish port names, the shape that made the old fixed cards cramped."""
    spec = NodeSpec(node_id=label, kind="federated_node")
    spec.label = label
    spec.attributes = {"source": "datasets/agent_beta.jsonl"}
    for index in range(ports):
        spec.ports.append(PortSpec(name=f"natural_language_channel_{index}", direction="in"))
    for index in range(ports):
        spec.ports.append(PortSpec(name=f"tx:arbiter_channel_{index}", direction="out"))
    return spec


def make_board(count: int) -> BoardDescriptor:
    """A board description with ``count`` dies chained one into the next."""
    dies = [
        BoardDie(
            index=index,
            name=f"die_{index}",
            node_id=index,
            total_nodes=count,
            routing_mask=0b1011,
            mask_width=4,
            pins=[PinSpec(name=f"chan_{index}", direction="output", width=8, dtype="logic", die_name=f"die_{index}", slot=index)],
            chip_module="asic_chip.die_core",
            switch_module="asic_chip.switch_fabric",
            channel_slots=4,
            parameters={"N": 4},
            line=10 + index,
        )
        for index in range(count)
    ]
    links = [
        BoardLink(
            index=index,
            source_index=index,
            target_index=index + 1,
            source_name=f"die_{index}",
            target_name=f"die_{index + 1}",
            signal_count=40,
            slot_source=index,
            slot_target=index,
            line=100 + index,
        )
        for index in range(count - 1)
    ]
    return BoardDescriptor(name="probe", parameters={}, dies=dies, links=links, top_ports=[])


# --------------------------------------------------------------------------- accordion cards
def test_a_card_folds_to_its_title_only(qapp):
    card = widgets.Card("Ports")
    body = QWidget()
    body.setMinimumHeight(220)
    card.body.addWidget(body)
    qapp.processEvents()

    assert card.expanded is True
    open_height = card.sizeHint().height()
    card.set_expanded(False)
    qapp.processEvents()

    assert card.expanded is False
    assert not card._shell.isVisibleTo(card), "a closed card hides its whole body"
    assert card.property("folded") is True
    assert card.sizeHint().height() <= widgets.Card.CLOSED_HEIGHT + 4
    assert card.sizeHint().height() < open_height - 100, "the fold releases the body height"
    card.set_expanded(True)
    assert card._shell.isVisibleTo(card)


def test_folding_a_card_hands_its_room_to_the_others(qapp):
    holder = QWidget()
    layout = QVBoxLayout(holder)
    first = widgets.Card("First")
    second = widgets.Card("Second")
    for card in (first, second):
        body = QWidget()
        body.setMinimumHeight(180)
        card.body.addWidget(body)
    widgets.share_column(layout, first, 5)
    widgets.share_column(layout, second, 3)
    holder.resize(360, 600)
    holder.show()
    qapp.processEvents()

    assert layout.stretch(0) == 5 and layout.stretch(1) == 3
    first.set_expanded(False)
    qapp.processEvents()
    assert layout.stretch(0) == 0, "a closed card stops asking for column height"
    assert layout.stretch(1) == 3
    assert second.height() > 400
    first.set_expanded(True)
    assert layout.stretch(0) == 5

    # the click on the title bar is a fold, too
    first.fold.click()
    assert first.expanded is False
    holder.hide()


# --------------------------------------------------------------------------- node card
def test_the_title_sits_inside_the_header_band(qapp):
    scene = TopologyScene()
    item = NodeItem(tall_spec("arbitro_logico_principale", 2), scene.registry)
    scene.addItem(item)
    qapp.processEvents()

    bounds = item.boundingRect()
    assert bounds.top() == 0.0
    assert bounds.height() >= item.content_height(), "the card is as tall as its own content"
    assert bounds.width() <= NODE_MAX_WIDTH
    # a header band really exists: the label band and the first port start below it
    first_port = item.port_anchor(item.spec.inputs[0])
    assert first_port.y() > HEADER_HEIGHT + LABEL_BAND
    assert NODE_WIDTH <= item.content_width() <= NODE_MAX_WIDTH


def test_the_card_grows_with_its_ports_but_not_its_width(qapp):
    scene = TopologyScene()
    few = NodeItem(tall_spec("small", 1), scene.registry)
    many = NodeItem(tall_spec("a very long node label that cannot fit in a card", 12), scene.registry)
    scene.addItem(few)
    scene.addItem(many)
    qapp.processEvents()

    assert many.content_height() > few.content_height() + 10 * PORT_ROW
    assert many.boundingRect().width() <= NODE_MAX_WIDTH
    assert CARD_STEP_X > NODE_MAX_WIDTH, "cards never touch each other sideways"
    assert many.content_height() > 0
    last_port = many.port_anchor(many.spec.outputs[-1])
    assert last_port.y() < many.boundingRect().bottom(), "no port is pushed out of the card"


def test_layout_steps_clear_cards_of_different_heights(qapp):
    graph = TopologyGraph()
    for name in ("root", "left", "right"):
        graph.nodes[name] = tall_spec(name, 8)
    graph.connect(EdgeSpec(source="root", target="left", source_port="tx:0", target_port="rx:0"))
    graph.connect(EdgeSpec(source="root", target="right", source_port="tx:0", target_port="rx:0"))

    flat = auto_layout(graph)
    tall = auto_layout(graph, heights={"left": 320.0})

    assert flat["root"] == tall["root"], "the first layer does not move"
    assert flat["left"][0] == flat["right"][0], "the two children share one column"
    assert flat["right"][1] > flat["left"][1]
    assert (tall["right"][1] - tall["left"][1]) > (flat["right"][1] - flat["left"][1]) + 100.0, (
        "a taller card pushes the card under it down"
    )


# --------------------------------------------------------------------------- canvas zoom
def test_the_canvas_still_fits_by_default_and_then_zooms(qapp):
    scene = TopologyScene()
    for column, name in enumerate(("a", "b", "c", "d")):
        item = NodeItem(tall_spec(name, 3), scene.registry)
        scene.addItem(item)
        item.setPos(column * CARD_STEP_X, 0.0)
    scene.setSceneRect(scene.itemsBoundingRect().adjusted(-20, -20, 20, 20))
    view = TopologyView(scene)
    view.resize(520, 380)
    view.show()
    view.fit()
    qapp.processEvents()

    fit_scale = view.zoom_level()
    assert fit_scale > 0.0
    # the default is the old fit: the whole scene lands inside the viewport
    box = scene.itemsBoundingRect()
    corner_top_left = view.mapToScene(view.viewport().rect().topLeft())
    corner_bottom_right = view.mapToScene(view.viewport().rect().bottomRight())
    assert corner_top_left.x() <= box.left() + 1.0 and corner_top_left.y() <= box.top() + 1.0
    assert corner_bottom_right.x() >= box.right() - 1.0 and corner_bottom_right.y() >= box.bottom() - 1.0

    observed = []
    view.zoom_changed.connect(observed.append)
    view.zoom_in()
    assert view.zoom_level() > fit_scale
    view.zoom_out()
    view.zoom_out()
    assert view.zoom_level() < fit_scale
    for _ in range(40):
        view.zoom_in()
    assert view.zoom_level() <= TopologyView.MAX_SCALE
    for _ in range(60):
        view.zoom_out()
    assert view.zoom_level() >= TopologyView.MIN_SCALE
    assert observed, "the view reports the zoom it just applied"
    view.fit()
    assert abs(view.zoom_level() - fit_scale) < 1e-9, "Fit returns to the default zoom"
    view.hide()


def test_the_canvas_panel_offers_zoom_buttons_and_a_readout(qapp, workspace):
    panel = canvas.CanvasPanel(workspace)
    panel.resize(900, 620)
    panel.show()
    qapp.processEvents()

    assert panel.btn_zoom_in.text() == "Zoom in"
    assert panel.btn_zoom_out.text() == "Zoom out"
    assert "zoom" in panel.chip_zoom.text()
    before = panel.view.zoom_level()
    panel.btn_zoom_in.click()
    qapp.processEvents()
    assert panel.view.zoom_level() > before
    assert "pct" in panel.chip_zoom.text()
    panel.btn_zoom_out.click()
    panel.btn_fit.click()
    qapp.processEvents()
    assert panel.chip_zoom.text() == "zoom 100 pct"
    panel.hide()


# --------------------------------------------------------------------------- schematic
def test_the_schematic_default_view_is_the_fit_it_always_had(qapp):
    view = SchematicView()
    view.resize(900, 560)
    view.set_board(make_board(4))
    view.show()
    qapp.processEvents()

    assert (view.virtual_size.width(), view.virtual_size.height()) == (VIRTUAL_W, VIRTUAL_H)
    assert view.zoom == 1.0

    usable = QRectF(view.rect()).adjusted(8, HEADER_BAND, -8, -8)
    expected_scale = min(usable.width() / VIRTUAL_W, usable.height() / VIRTUAL_H)
    middle = QPointF(usable.center().x(), usable.center().y())
    drawn = view._transform.map(QPointF(VIRTUAL_W / 2.0, VIRTUAL_H / 2.0))
    assert abs(drawn.x() - middle.x()) < 0.75 and abs(drawn.y() - middle.y()) < 0.75
    assert abs(view._transform.m11() - expected_scale) < 1e-9

    near = view._transform.map(QPointF(0.0, 0.0))
    far = view._transform.map(QPointF(VIRTUAL_W, VIRTUAL_H))
    drawn_box = QRectF(near, far)
    left_margin = drawn_box.left() - usable.left()
    right_margin = usable.right() - drawn_box.right()
    top_margin = drawn_box.top() - usable.top()
    bottom_margin = usable.bottom() - drawn_box.bottom()
    assert abs(left_margin - right_margin) < 0.75 and abs(top_margin - bottom_margin) < 0.75, (
        "the drawing sits centred in the band, exactly as it always did"
    )
    view.hide()


def test_the_canvas_grows_when_the_blocks_outnumber_it(qapp):
    view = SchematicView()
    view.resize(900, 560)
    small = make_board(4)
    view.set_board(small)
    qapp.processEvents()
    small_rects = dict(view.die_rects())

    view.set_board(make_board(40))
    qapp.processEvents()
    rects = view.die_rects()
    assert len(rects) == 40
    size = view.virtual_size
    assert size.width() > VIRTUAL_W and size.height() > VIRTUAL_H, "the drawing area grows"
    for rect in rects.values():
        assert rect.width() >= MIN_DIE_W and rect.height() >= MIN_DIE_H
        assert QRectF(0.0, 0.0, size.width(), size.height()).contains(rect)
    # at the default fit every block is still a readable box, not a dot
    scale = min((view.width() - 16) / size.width(), (view.height() - 38) / size.height())
    assert min(rect.width() for rect in rects.values()) * scale >= 60.0
    assert min(rect.height() for rect in rects.values()) * scale >= 40.0
    assert len(small_rects) == 4
    assert all(MIN_DIE_W <= rect.width() <= MAX_DIE_W for rect in small_rects.values())


def test_zoom_and_walk_do_not_move_the_target_under_the_cursor(qapp):
    view = SchematicView()
    view.resize(900, 560)
    view.set_board(make_board(12))
    view.show()
    qapp.processEvents()
    view._rects = view.die_rects()

    target = 7
    centre = view.die_rects()[target].center()
    assert view._pick(QPointF(view._transform.map(centre)))
    assert view.selected_die == target

    point_on_screen = QPointF(view._transform.map(view.die_rects()[target].center()))
    view.set_zoom(3.0, anchor=point_on_screen)
    qapp.processEvents()
    assert view.zoom == 3.0
    moved = QPointF(view._transform.map(view.die_rects()[target].center()))
    assert abs(moved.x() - point_on_screen.x()) < 2.0 and abs(moved.y() - point_on_screen.y()) < 2.0, "the block stays under the cursor"

    view.pan_by(60.0, -40.0)
    qapp.processEvents()
    panned = QPointF(view._transform.map(view.die_rects()[target].center()))
    assert abs(panned.x() - (moved.x() + 60.0)) < 1.0 and abs(panned.y() - (moved.y() - 40.0)) < 1.0
    view._rects = view.die_rects()
    assert view._pick(panned), "the hit test follows the pan"
    assert view.selected_die == target

    view.focus_die(3)
    qapp.processEvents()
    focused = QPointF(view._transform.map(view.die_rects()[3].center()))
    usable = QRectF(view.rect()).adjusted(8, HEADER_BAND, -8, -8)
    assert abs(focused.x() - usable.center().x()) < 1.5 and abs(focused.y() - usable.center().y()) < 1.5
    view.hide()


def test_the_schematic_zoom_is_clamped_and_reported(qapp):
    view = SchematicView()
    view.resize(900, 560)
    view.set_board(make_board(6))
    view.show()
    qapp.processEvents()

    seen = []
    view.zoom_changed.connect(seen.append)
    for _ in range(30):
        view.zoom_in()
    assert view.zoom == MAX_ZOOM
    for _ in range(80):
        view.zoom_out()
    assert view.zoom == MIN_ZOOM
    view.reset_view()
    assert view.zoom == 1.0
    assert view.zoom_percent() == 100
    assert seen and seen[-1] == 1.0
    view.hide()


def test_the_hardware_panel_offers_zoom_buttons_and_centres_a_die(qapp, workspace):
    panel = HardwarePanel(workspace)
    panel.resize(1000, 700)
    panel.show()
    qapp.processEvents()

    view = panel.schematic
    assert panel.btn_zoom_in.text() == "Zoom in"
    assert panel.btn_zoom_fit.text() == "Fit board"
    assert panel.chip_zoom.text() == "zoom 100 pct"

    panel.btn_zoom_in.click()
    qapp.processEvents()
    assert view.zoom > 1.0
    assert "pct" in panel.chip_zoom.text() and panel.chip_zoom.text() != "zoom 100 pct"

    view.set_board(make_board(8))
    qapp.processEvents()
    view.set_zoom(4.0)
    view.select_die(3)
    panel.btn_center_die.click()
    qapp.processEvents()
    assert view.zoom > 1.0
    centred = QPointF(view._transform.map(view.die_rects()[3].center()))
    usable = QRectF(view.rect()).adjusted(8, HEADER_BAND, -8, -8)
    assert abs(centred.x() - usable.center().x()) < 1.5 and abs(centred.y() - usable.center().y()) < 1.5
    panel.btn_zoom_fit.click()
    qapp.processEvents()
    assert view.zoom == 1.0
    assert panel.chip_zoom.text() == "zoom 100 pct"
    panel.hide()
