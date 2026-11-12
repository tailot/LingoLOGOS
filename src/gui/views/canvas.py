"""Generic node based topology canvas.

The canvas is a view over a :class:`gui.model.topology.TopologyGraph`. It does not know
what a node means: it draws the identifier, the kind, the dynamic port stack and the
free form attributes, and it lets the person add, move, wire, rename and delete any of
that. Node kinds are data, not code: the kind combo box is editable, an unknown kind gets
a synthesized presentation entry, and a kind can be registered from the panel so a hand
written spec or a dataset introduces new shapes without the interface being edited.

All rendering is local ``QPainter`` work. There are no icons, fonts, styles or scripts
fetched from anywhere, and links are cubic Beziers with an optional pulse that a stream
event drives.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PyQt6.QtCore import (
    QAbstractAnimation,
    QPointF,
    QRectF,
    Qt,
    QTimer,
    QVariantAnimation,
    pyqtSignal,
)
from PyQt6.QtGui import (
    QBrush,
    QColor,
    QCursor,
    QFont,
    QFontMetrics,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPainterPathStroker,
    QPen,
)
from PyQt6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QGraphicsItem,
    QGraphicsObject,
    QGraphicsPathItem,
    QGraphicsScene,
    QGraphicsView,
    QGraphicsTextItem,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from gui.model.topology import (
    DIRECTION_IN,
    DIRECTION_OUT,
    EdgeSpec,
    Issue,
    KindRegistry,
    KindSpec,
    NodeSpec,
    PortSpec,
    TopologyGraph,
    auto_layout,
    derive_ports_from_mapping,
    find_cycles,
    load_graph_file,
    save_graph,
    validate_graph,
)

from .. import theme, widgets
from ..paths import Workspace, open_workspace

NODE_WIDTH = 208.0
NODE_MAX_WIDTH = 264.0
HEADER_HEIGHT = 26.0
LABEL_BAND = 22.0
PORT_ROW = 19.0
PORT_HIT = 12.0
FOOTER_BAND = 16.0
CARD_STEP_X = 300.0
CARD_STEP_Y = 190.0


def _accent_of(kind: str, registry: KindRegistry) -> str:
    return registry.get(kind).accent


def _fit_text(font: QFont, text: str, width: float) -> str:
    """Shrink ``text`` to one line that fits ``width`` scene units, ellipsis when trimmed."""
    flat = " ".join(str(text).split())
    room = int(max(0.0, float(width)))
    if room <= 0:
        return ""
    metrics = QFontMetrics(font)
    if metrics.horizontalAdvance(flat) <= room:
        return flat
    trimmed = metrics.elidedText(flat, Qt.TextElideMode.ElideRight, room)
    return trimmed if trimmed else flat[:1]


# ---------------------------------------------------------------------------
# Items
# ---------------------------------------------------------------------------
class PortItem(QGraphicsObject):
    """A connectable dot on the side of a node, one per entry of the port stack."""

    def __init__(self, node: "NodeItem", port: PortSpec, parent: Optional[QGraphicsItem] = None) -> None:
        super().__init__(parent)
        self.node = node
        self.port = port
        self._hover = False
        self.setAcceptHoverEvents(True)
        self.setZValue(6)
        self.setToolTip(f"{port.direction} {port.name}")
        self.setPos(self._anchor())

    @property
    def direction(self) -> str:
        return self.port.direction

    @property
    def node_id(self) -> str:
        return self.node.node_id

    def _anchor(self) -> QPointF:
        return self.node.port_anchor(self.port)

    def sync(self) -> None:
        self.prepareGeometryChange()
        self.setPos(self._anchor())

    def boundingRect(self) -> QRectF:  # noqa: N802 - Qt naming
        half = PORT_HIT / 2.0
        return QRectF(-half, -half, PORT_HIT, PORT_HIT)

    def paint(self, painter: QPainter, option, widget=None) -> None:  # noqa: N802 - Qt naming
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        accent = theme.color(_accent_of(self.node.kind, self.node.registry))
        radius = 4.0 if not self._hover else 5.5
        if self.port.origin == "payload":
            painter.setPen(QPen(accent.darker(120), 1.4))
            painter.setBrush(QBrush(accent))
        else:
            painter.setPen(QPen(accent, 1.4))
            painter.setBrush(QBrush(theme.color("window")))
        painter.drawEllipse(QPointF(0, 0), radius, radius)

    def hoverEnterEvent(self, event) -> None:  # noqa: N802 - Qt naming
        self._hover = True
        self.prepareGeometryChange()
        self.update()

    def hoverLeaveEvent(self, event) -> None:  # noqa: N802 - Qt naming
        self._hover = False
        self.prepareGeometryChange()
        self.update()

    def itemChange(self, change, value):  # noqa: N802 - Qt naming
        return value


class NodeItem(QGraphicsObject):
    """The drawing of one NodeSpec: header, label, attributes, and the two port stacks."""

    def __init__(self, spec: NodeSpec, registry: KindRegistry, parent: Optional[QGraphicsItem] = None) -> None:
        super().__init__(parent)
        self.spec = spec
        self.registry = registry
        self._port_items: Dict[str, PortItem] = {}
        self._flag: Optional[QGraphicsTextItem] = None
        self._flag_text = ""
        self._selected = False
        self._width_cache: Optional[float] = None
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges, True)
        self.setAcceptHoverEvents(True)
        self.setZValue(4)
        if spec.position:
            self.setPos(spec.position[0], spec.position[1])
        self.rebuild_ports()

    # --- identity ------------------------------------------------------------
    @property
    def node_id(self) -> str:
        return self.spec.node_id

    @property
    def kind(self) -> str:
        return self.spec.kind

    def ports_of(self, direction: str) -> List[PortItem]:
        return [item for key, item in self._port_items.items() if key.startswith(f"{direction}:")]

    def port_item(self, direction: str, name: str) -> Optional[PortItem]:
        return self._port_items.get(f"{direction}:{name}")

    # --- geometry ------------------------------------------------------------
    def port_row_count(self) -> int:
        return max(1, len(self.spec.inputs), len(self.spec.outputs))

    def footer_height(self) -> float:
        """Reserved band for the attribute counter and the validation badge."""
        if self.spec.attributes or self._flag_text:
            return FOOTER_BAND + 4.0
        return 6.0

    def content_height(self) -> float:
        """Header, label, one row per port and the footer: nothing overlaps."""
        return HEADER_HEIGHT + LABEL_BAND + self.port_row_count() * PORT_ROW + self.footer_height()

    def content_width(self) -> float:
        """Card width: wide enough for the longest title, label or port name it can show."""
        if self._width_cache is None:
            self._width_cache = self._measure_width()
        return self._width_cache

    def _measure_width(self) -> float:
        title_font = theme.ui_font(9.0, True)
        label_font = theme.mono_font(10.0, True)
        port_font = theme.mono_font(8.0)
        widest_port = 0
        for port in self.spec.ports:
            widest_port = max(widest_port, QFontMetrics(port_font).horizontalAdvance(_flat(port.name)))
        needed = 2.0 * (widest_port + 20.0)
        needed = max(needed, QFontMetrics(title_font).horizontalAdvance(_flat(self.registry.get(self.kind).title)) + 20.0)
        needed = max(needed, QFontMetrics(label_font).horizontalAdvance(_flat(self.spec.label)) + 20.0)
        return max(NODE_WIDTH, min(NODE_MAX_WIDTH, needed))

    def boundingRect(self) -> QRectF:  # noqa: N802 - Qt naming
        return QRectF(0, 0, self.content_width(), self.content_height())

    def port_anchor(self, port: PortSpec) -> QPointF:
        stack = port.direction == DIRECTION_OUT and self.spec.outputs or self.spec.inputs
        index = 0
        for position, candidate in enumerate(stack):
            if candidate.name == port.name:
                index = position
                break
        y = HEADER_HEIGHT + LABEL_BAND + index * PORT_ROW + PORT_ROW / 2.0
        x = self.content_width() if port.direction == DIRECTION_OUT else 0.0
        return QPointF(x, y)

    def port_label_band(self, direction: str) -> QRectF:
        """Where a port name is drawn, inside the card and clear of the port dot."""
        width = self.content_width()
        name_width = max(24.0, width / 2.0 - 22.0)
        if direction == DIRECTION_OUT:
            return QRectF(width - 16.0 - name_width, 0.0, name_width, PORT_ROW)
        return QRectF(16.0, 0.0, name_width, PORT_ROW)

    def refresh_text(self) -> None:
        """Drop the measured width and the tool tip after the spec or the registry changed."""
        self._width_cache = None
        self._refresh_tool_tip()
        self.prepareGeometryChange()
        self.update()

    def _refresh_tool_tip(self) -> None:
        kind = self.registry.get(self.kind)
        lines = [f"{self.spec.node_id}  ({kind.title})"]
        if self.spec.label and self.spec.label != self.spec.node_id:
            lines.append(f"label {self.spec.label}")
        lines += [f"in   {port.name}" for port in self.spec.inputs]
        lines += [f"out  {port.name}" for port in self.spec.outputs]
        if self.spec.attributes:
            keys = ", ".join(sorted(self.spec.attributes))
            lines.append(f"attributes {keys}")
        self.setToolTip("\n".join(lines))

    def rebuild_ports(self) -> None:
        scene = self.scene()
        for item in self._port_items.values():
            item.setParentItem(None)
            if scene is not None:
                scene.removeItem(item)
        self._port_items.clear()
        for port in self.spec.ports:
            item = PortItem(self, port, self)
            self._port_items[port.key] = item
        self.prepareGeometryChange()
        self._width_cache = None
        self._refresh_tool_tip()
        self.update()

    def sync_ports(self) -> None:
        for item in self._port_items.values():
            item.sync()

    # --- presentation --------------------------------------------------------
    def set_flag(self, text: str) -> None:
        """Overlay a short badge, used by validation and by the stream to mark activity."""
        if text == self._flag_text:
            return
        self._flag_text = text
        self.prepareGeometryChange()
        self.update()

    def paint(self, painter: QPainter, option, widget=None) -> None:  # noqa: N802 - Qt naming
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        width = self.content_width()
        height = self.content_height()
        accent = QColor(theme.color(_accent_of(self.kind, self.registry)))
        body = QRectF(0.0, 0.0, width, height)

        outline = QPainterPath()
        outline.addRoundedRect(body, 8, 8)
        gradient = QLinearGradient(body.topLeft(), body.bottomLeft())
        gradient.setColorAt(0.0, theme.color("panel_alt"))
        gradient.setColorAt(1.0, theme.color("panel"))
        painter.setPen(QPen(Qt.PenStyle.NoPen))
        painter.setBrush(QBrush(gradient))
        painter.drawPath(outline)

        # The header band is a plain rectangle clipped by the card outline: filling two
        # overlapping sub paths punched the band out instead, because QPainterPath fills
        # with the odd even rule by default.
        painter.setClipPath(outline)
        header = QRectF(body.left(), body.top(), width, HEADER_HEIGHT)
        label = QRectF(body.left(), header.bottom(), width, LABEL_BAND)
        painter.setPen(QPen(Qt.PenStyle.NoPen))
        painter.setBrush(QBrush(accent))
        painter.drawRect(header)
        painter.setPen(QPen(accent.darker(150), 1))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawLine(label.topLeft(), label.topRight())
        painter.drawLine(header.bottomLeft(), header.bottomRight())
        painter.setClipping(False)

        border = accent if self._selected else theme.color("frame")
        painter.setPen(QPen(QColor(border), 2 if self._selected else 1))
        painter.drawPath(outline)

        title_font = theme.ui_font(9.0, True)
        painter.setFont(title_font)
        painter.setPen(QPen(QColor("#0b0f14")))
        title_cell = QRectF(header.left() + 10.0, header.top(), width - 20.0, HEADER_HEIGHT)
        painter.drawText(
            title_cell,
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
            _fit_text(title_font, self.registry.get(self.kind).title, title_cell.width()),
        )

        label_font = theme.mono_font(10.0, True)
        painter.setFont(label_font)
        painter.setPen(QPen(theme.color("text")))
        label_cell = QRectF(body.left() + 10.0, label.top(), width - 20.0, LABEL_BAND)
        painter.drawText(
            label_cell,
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
            _fit_text(label_font, self.spec.label, label_cell.width()),
        )

        port_font = theme.mono_font(8.0)
        painter.setFont(port_font)
        painter.setPen(QPen(theme.color("text_dim")))
        for port in self.spec.inputs:
            band = self.port_label_band(DIRECTION_IN)
            anchor = self.port_anchor(port)
            cell = QRectF(band.left(), anchor.y() - PORT_ROW / 2.0, band.width(), band.height())
            painter.drawText(cell, Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, _fit_text(port_font, port.name, cell.width()))
        for port in self.spec.outputs:
            band = self.port_label_band(DIRECTION_OUT)
            anchor = self.port_anchor(port)
            cell = QRectF(band.left(), anchor.y() - PORT_ROW / 2.0, band.width(), band.height())
            painter.drawText(cell, Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight, _fit_text(port_font, port.name, cell.width()))

        if self.spec.attributes or self._flag_text:
            footer_top = height - self.footer_height()
            if self._flag_text:
                painter.setPen(QPen(theme.color("warn")))
                flag_cell = QRectF(body.left() + 10.0, footer_top, width * 0.55, FOOTER_BAND)
                painter.drawText(flag_cell, Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, _fit_text(label_font, self._flag_text, flag_cell.width()))
            if self.spec.attributes:
                painter.setFont(theme.mono_font(8.0))
                painter.setPen(QPen(theme.color("data")))
                count_cell = QRectF(body.left() + width * 0.45, footer_top, width * 0.55 - 10.0, FOOTER_BAND)
                painter.drawText(count_cell, Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight, f"{len(self.spec.attributes)} attributes")

    def itemChange(self, change, value):  # noqa: N802 - Qt naming
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            self.sync_ports()
            scene = self.scene()
            if isinstance(scene, TopologyScene):
                scene.node_moved(self.node_id, (value.x(), value.y()))
        elif change == QGraphicsItem.GraphicsItemChange.ItemSelectedHasChanged:
            self._selected = bool(value)
            self.update()
        return value

    def contextMenuEvent(self, event) -> None:  # noqa: N802 - Qt naming
        scene = self.scene()
        if isinstance(scene, TopologyScene):
            scene.node_context_menu(self, event.scenePos())
        event.accept()


class LinkItem(QGraphicsObject):
    """Directed cubic Bezier between two ports, with an arrow head and a pulse mode."""

    def __init__(self, edge: EdgeSpec, source: Optional[PortItem], target: Optional[PortItem], scene: "TopologyScene") -> None:
        super().__init__()
        self.edge = edge
        self.source = source
        self.target = target
        self._scene = scene
        self._phase = 0.0
        self._pulse = 0.0
        self._selected = False
        self._invalid = False
        self.setAcceptHoverEvents(True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setZValue(2)
        self._animation = QVariantAnimation(self)
        self._animation.setDuration(1400)
        self._animation.setStartValue(0.0)
        self._animation.setEndValue(1.0)
        self._animation.setLoopCount(-1)
        self._animation.valueChanged.connect(self._on_phase)
        self._animation.start()
        self.refresh()

    def _on_phase(self, value) -> None:
        self._phase = float(value)
        if self._pulse > 0.0:
            self._pulse = max(0.0, self._pulse - 0.02)
        self.update()

    def endpoints(self) -> Tuple[QPointF, QPointF]:
        if self.source is not None and self.target is not None:
            return self.source.scenePos(), self.target.scenePos()
        scene = self._scene
        start = scene.node_anchor(self.edge.source, DIRECTION_OUT, self.edge.source_port)
        end = scene.node_anchor(self.edge.target, DIRECTION_IN, self.edge.target_port)
        return start, end

    def build_path(self) -> QPainterPath:
        start, end = self.endpoints()
        dx = max(40.0, abs(end.x() - start.x()) * 0.45)
        path = QPainterPath(start)
        path.cubicTo(QPointF(start.x() + dx, start.y()), QPointF(end.x() - dx, end.y()), end)
        return path

    def refresh(self) -> None:
        self.prepareGeometryChange()
        self._path = self.build_path()
        stroker = QPainterPathStroker()
        stroker.setWidth(12.0)
        self._bounds = stroker.createStroke(self._path).boundingRect()
        self.update()

    def boundingRect(self) -> QRectF:  # noqa: N802 - Qt naming
        bounds = getattr(self, "_bounds", None)
        if bounds is None or bounds.isEmpty():
            return QRectF(-1, -1, 2, 2)
        return bounds

    def shape(self) -> QPainterPath:  # noqa: N802 - Qt naming
        stroker = QPainterPathStroker()
        stroker.setWidth(14.0)
        return stroker.createStroke(self._path)

    def pulse(self) -> None:
        self._pulse = 1.0
        if self._animation.state() != QAbstractAnimation.State.Running:
            self._animation.start()
        self.update()

    def stop_pulse(self) -> None:
        """Release the flow animation of this edge; used when the panel is torn down."""
        self._pulse = 0.0
        self._animation.stop()
        self.update()

    def mark_invalid(self, flag: bool) -> None:
        self._invalid = bool(flag)
        self.update()

    def paint(self, painter: QPainter, option, widget=None) -> None:  # noqa: N802 - Qt naming
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        base = theme.color("link")
        if self._invalid:
            base = theme.color("fail")
        elif self._selected:
            base = theme.color("link_hot")
        width = 1.6 + 2.2 * self._pulse
        painter.setPen(QPen(base, width, Qt.PenStyle.SolidLine if self._pulse <= 0.0 else Qt.PenStyle.DashLine, Qt.PenCapStyle.RoundCap))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPath(self._path)

        start, end = self.endpoints()
        arrow = QPainterPath()
        angle = _angle_of(start, end)
        head = 9.0
        arrow.moveTo(end)
        arrow.lineTo(end - QPointF(head * _cos(angle - 0.5), head * _sin(angle - 0.5)))
        arrow.lineTo(end - QPointF(head * _cos(angle + 0.5), head * _sin(angle + 0.5)))
        arrow.closeSubpath()
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(base))
        painter.drawPath(arrow)

        if self._pulse > 0.0:
            point = self._path.pointAtPercent(min(1.0, self._phase))
            painter.setBrush(QBrush(theme.color("link_hot")))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(point, 4.5, 4.5)

        label = self.edge.label or self.edge.target_port or self.edge.source_port
        if label:
            mid = self._path.pointAtPercent(0.5)
            painter.setFont(theme.mono_font(7.5))
            painter.setPen(QPen(theme.color("text_dim")))
            painter.drawText(QRectF(mid.x() - 60, mid.y() - 12, 120, 12), Qt.AlignmentFlag.AlignCenter, _elide(label, 22))

    def itemChange(self, change, value):  # noqa: N802 - Qt naming
        if change == QGraphicsItem.GraphicsItemChange.ItemSelectedHasChanged:
            self._selected = bool(value)
            self.update()
        return value

    def hoverEnterEvent(self, event) -> None:  # noqa: N802 - Qt naming
        self.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))

    def contextMenuEvent(self, event) -> None:  # noqa: N802 - Qt naming
        if isinstance(self._scene, TopologyScene):
            self._scene.link_context_menu(self, event.scenePos())
        event.accept()


def _flat(text: str) -> str:
    return " ".join(str(text).split())


def _elide(text: str, width: int) -> str:
    flat = _flat(text)
    return flat if len(flat) <= width else flat[: max(1, width - 1)] + "~"


def _cos(angle: float) -> float:
    return math.cos(angle)


def _sin(angle: float) -> float:
    return math.sin(angle)


def _angle_of(start: QPointF, end: QPointF) -> float:
    return math.atan2(end.y() - start.y(), end.x() - start.x())


# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------
class TopologyScene(QGraphicsScene):
    """Holds the items and keeps them synchronized with the graph model."""

    graph_changed = pyqtSignal()
    node_selected = pyqtSignal(str)

    def __init__(self, registry: Optional[KindRegistry] = None, parent=None) -> None:
        super().__init__(parent)
        self.registry = registry or KindRegistry()
        self.graph = TopologyGraph()
        self._nodes: Dict[str, NodeItem] = {}
        self._links: List[LinkItem] = []
        self._pending: Optional[QGraphicsPathItem] = None
        self._pending_start: Optional[QPointF] = None
        self.setSceneRect(-200, -200, 2400, 1600)

    # --- loading -------------------------------------------------------------
    def set_graph(self, graph: TopologyGraph) -> None:
        self.graph = graph
        self.sync()
        if graph.nodes and not any(spec.position for spec in graph.nodes.values()):
            self.apply_layout()
        self.graph_changed.emit()

    def sync(self) -> None:
        for item in list(self._nodes.values()) + self._links:
            self.removeItem(item)
        self._nodes.clear()
        self._links.clear()
        for node_id, spec in self.graph.nodes.items():
            item = NodeItem(spec, self.registry)
            self._place(item, spec)
            self.addItem(item)
            self._nodes[node_id] = item
        for edge in self.graph.edges:
            self._add_link(edge)
        self._fit_scene_rect()

    def _place(self, item: NodeItem, spec: NodeSpec) -> None:
        """Restore the stored position of a node, or spread new nodes out."""
        if spec.position:
            item.setPos(float(spec.position[0]), float(spec.position[1]))
        else:
            item.setPos(self._free_spot())

    def _add_link(self, edge: EdgeSpec) -> Optional[LinkItem]:
        source_node = self._nodes.get(edge.source)
        target_node = self._nodes.get(edge.target)
        source = source_node.port_item(DIRECTION_OUT, edge.source_port) if source_node else None
        target = target_node.port_item(DIRECTION_IN, edge.target_port) if target_node else None
        item = LinkItem(edge, source, target, self)
        self.addItem(item)
        self._links.append(item)
        return item

    def _fit_scene_rect(self) -> None:
        bounds = self.itemsBoundingRect().adjusted(-160, -160, 320, 320)
        self.setSceneRect(bounds)

    # --- editing -------------------------------------------------------------
    def add_node(self, node_id: str, kind: str, label: str = "", position: Optional[Tuple[float, float]] = None) -> NodeItem:
        spec = self.graph.nodes.get(node_id)
        if spec is None:
            spec = NodeSpec(node_id=node_id, kind=kind, label=label or node_id, position=position)
            spec_kind = self.registry.get(kind)
            for name in spec_kind.default_in_ports:
                spec.ensure_port(DIRECTION_IN, name, origin="kind")
            for name in spec_kind.default_out_ports:
                spec.ensure_port(DIRECTION_OUT, name, origin="kind")
            self.graph.add_node(spec)
        item = NodeItem(spec, self.registry)
        if position is None and not spec.position:
            item.setPos(self._free_spot())
        self.addItem(item)
        self._nodes[node_id] = item
        self._fit_scene_rect()
        self.graph_changed.emit()
        return item

    def _free_spot(self) -> QPointF:
        count = len(self._nodes)
        tallest = max([item.content_height() for item in self._nodes.values()] + [CARD_STEP_Y])
        return QPointF(60.0 + (count % 6) * CARD_STEP_X, 60.0 + (count // 6) * max(CARD_STEP_Y, tallest + 24.0))

    def remove_node(self, node_id: str) -> None:
        item = self._nodes.pop(node_id, None)
        if item is not None:
            self.removeItem(item)
        for link in [link for link in self._links if link.edge.source == node_id or link.edge.target == node_id]:
            self._links.remove(link)
            self.removeItem(link)
        self.graph.remove_node(node_id)
        for link in self._links:
            link.refresh()
        self.graph_changed.emit()

    def remove_link(self, link: LinkItem) -> None:
        if link in self._links:
            self._links.remove(link)
        self.removeItem(link)
        self.graph.remove_edge(link.edge)
        self.graph_changed.emit()

    def connect_ports(self, source: PortItem, target: PortItem) -> Tuple[bool, str]:
        if source.direction != DIRECTION_OUT:
            return False, "the drag must start on an output port"
        if target.direction != DIRECTION_IN:
            return False, "the drag must end on an input port"
        if source.node_id == target.node_id:
            return False, "a node cannot feed itself"
        edge = EdgeSpec(
            source=source.node_id,
            target=target.node_id,
            source_port=source.port.name,
            target_port=target.port.name,
            label=source.port.name,
            kind="channel",
        )
        stored = self.graph.connect(edge)
        if stored is None:
            return False, "that port pair is already wired"
        self._add_link(stored)
        self.graph_changed.emit()
        return True, f"{source.node_id}.{source.port.name} -> {target.node_id}.{target.port.name}"

    def add_port(self, node_id: str, direction: str, name: str) -> bool:
        node = self.graph.nodes.get(node_id)
        if node is None or not name:
            return False
        if node.port(direction, name) is not None:
            return False
        node.ensure_port(direction, name, origin="manual")
        item = self._nodes.get(node_id)
        if item is not None:
            item.rebuild_ports()
        for link in self._links:
            link.refresh()
        self.graph_changed.emit()
        return True

    def remove_port(self, node_id: str, direction: str, name: str) -> None:
        node = self.graph.nodes.get(node_id)
        if node is None:
            return
        node.ports = [p for p in node.ports if not (p.direction == direction and p.name == name)]
        for edge in [e for e in list(self.graph.edges) if (e.source == node_id and e.source_port == name and direction == DIRECTION_OUT) or (e.target == node_id and e.target_port == name and direction == DIRECTION_IN)]:
            self.graph.remove_edge(edge)
        item = self._nodes.get(node_id)
        if item is not None:
            item.rebuild_ports()
        for link in list(self._links):
            if link.edge.key() not in {e.key() for e in self.graph.edges}:
                self._links.remove(link)
                self.removeItem(link)
            else:
                link.refresh()
        self.graph_changed.emit()

    def set_attribute(self, node_id: str, key: str, value: str) -> None:
        node = self.graph.nodes.get(node_id)
        if node is None or not key:
            return
        node.attributes[key] = value
        item = self._nodes.get(node_id)
        if item is not None:
            item.update()
        self.graph_changed.emit()

    def rename_node(self, node_id: str, label: str) -> None:
        node = self.graph.nodes.get(node_id)
        if node is None:
            return
        node.label = label
        item = self._nodes.get(node_id)
        if item is not None:
            item.update()
        self.graph_changed.emit()

    def set_kind(self, node_id: str, kind: str) -> None:
        node = self.graph.nodes.get(node_id)
        if node is None or not kind:
            return
        node.kind = kind
        item = self._nodes.get(node_id)
        if item is not None:
            item.update()
        self.graph_changed.emit()

    def register_kind(self, spec: KindSpec) -> None:
        self.registry.register(spec)
        for item in self._nodes.values():
            item.update()
        self.graph_changed.emit()

    def node_moved(self, node_id: str, position: Tuple[float, float]) -> None:
        node = self.graph.nodes.get(node_id)
        if node is not None:
            node.position = position
        for link in self._links:
            if link.edge.source == node_id or link.edge.target == node_id:
                link.refresh()

    def node_anchor(self, node_id: str, direction: str, port_name: str) -> QPointF:
        item = self._nodes.get(node_id)
        if item is None:
            return QPointF(0, 0)
        node = item.spec
        port = node.port(direction, port_name)
        if port is None:
            stack = node.outputs if direction == DIRECTION_OUT else node.inputs
            if stack:
                port = stack[0]
        if port is None:
            corner = QPointF(item.content_width(), 0) if direction == DIRECTION_OUT else QPointF(0, 0)
            return item.mapToScene(corner)
        return item.mapToScene(item.port_anchor(port))

    # --- pending drag link ---------------------------------------------------
    def begin_pending(self, start: PortItem) -> None:
        self._pending_start = start.scenePos()
        self._pending = QGraphicsPathItem()
        self._pending.setPen(QPen(theme.color("link_hot"), 1.6, Qt.PenStyle.DashLine))
        self._pending.setZValue(8)
        self.addItem(self._pending)
        self.move_pending(self._pending_start)

    def move_pending(self, point: QPointF) -> None:
        if self._pending is None:
            return
        start = self._pending_start or point
        dx = max(40.0, abs(point.x() - start.x()) * 0.45)
        path = QPainterPath(start)
        path.cubicTo(QPointF(start.x() + dx, start.y()), QPointF(point.x() - dx, point.y()), point)
        self._pending.setPath(path)

    def end_pending(self) -> None:
        if self._pending is not None:
            self.removeItem(self._pending)
            self._pending = None
        self._pending_start = None

    # --- validation and layout ----------------------------------------------
    def validate(self) -> List[Issue]:
        issues = validate_graph(self.graph)
        for item in self._nodes.values():
            item.set_flag("")
        for link in self._links:
            link.mark_invalid(False)
        for cycle in find_cycles(self.graph):
            ring = list(cycle) + cycle[:1]
            for position in range(len(cycle)):
                self._mark_link(ring[position], ring[position + 1], True)
        for issue in issues:
            for node_id in issue.subjects:
                item = self._nodes.get(node_id)
                if item is not None:
                    item.set_flag("check " + issue.code)
        return issues

    def _mark_link(self, source_id: str, target_id: str, flag: bool) -> None:
        for link in self._links:
            if link.edge.source == source_id and link.edge.target == target_id:
                link.mark_invalid(flag)

    def apply_layout(self) -> Dict[str, Tuple[float, float]]:
        heights = {node_id: item.content_height() for node_id, item in self._nodes.items()}
        positions = auto_layout(self.graph, heights=heights)
        for node_id, (x, y) in positions.items():
            item = self._nodes.get(node_id)
            if item is not None:
                item.setPos(x, y)
                node = self.graph.nodes.get(node_id)
                if node is not None:
                    node.position = (x, y)
        for link in self._links:
            link.refresh()
        self._fit_scene_rect()
        self.graph_changed.emit()
        return positions

    def pulse_edge(self, source_id: str, target_id: str) -> int:
        hits = 0
        for link in self._links:
            if link.edge.source == source_id and link.edge.target == target_id:
                link.pulse()
                hits += 1
        return hits

    def stop_animations(self) -> int:
        """Stop every running edge flow animation; returns how many were stopped."""
        stopped = 0
        for link in self._links:
            if link._animation.state() == QAbstractAnimation.State.Running:
                stopped += 1
            link.stop_pulse()
        return stopped

    def mark_node_activity(self, node_id: str, text: str = "active") -> None:
        item = self._nodes.get(node_id)
        if item is not None:
            item.set_flag(text)
            QTimer.singleShot(1200, lambda: item.set_flag("") if item.scene() is not None else None)

    def selected_node(self) -> Optional[NodeItem]:
        for item in self.selectedItems():
            if isinstance(item, NodeItem):
                return item
        return None

    def context_menu_for(self, item: Optional[QGraphicsItem], position: QPointF) -> QMenu:
        menu = QMenu()
        if isinstance(item, NodeItem):
            menu.setTitle(item.node_id)
            action = menu.addAction("Delete node")
            action.triggered.connect(lambda: self.remove_node(item.node_id))
        elif isinstance(item, LinkItem):
            menu.setTitle(f"{item.edge.source} -> {item.edge.target}")
            action = menu.addAction("Delete link")
            action.triggered.connect(lambda: self.remove_link(item))
        return menu

    def node_context_menu(self, item: NodeItem, position: QPointF) -> None:
        self.clearSelection()
        item.setSelected(True)
        self.node_selected.emit(item.node_id)
        menu = self.context_menu_for(item, position)
        if menu.isEmpty():
            return
        if self._view():
            menu.exec(self._view().mapToGlobal(position.toPoint()))

    def link_context_menu(self, item: LinkItem, position: QPointF) -> None:
        menu = self.context_menu_for(item, position)
        if menu.isEmpty():
            return
        if self._view():
            menu.exec(self._view().mapToGlobal(position.toPoint()))

    def _view(self):
        views = self.views()
        return views[0] if views else None


# ---------------------------------------------------------------------------
# View
# ---------------------------------------------------------------------------
class TopologyView(QGraphicsView):
    """Interactive view: zoom, pan, rubber band, port drag wiring, file drop.

    Zoom is one number the view reports every time it changes, so the panel can show it and
    the buttons, the keyboard and the wheel all end up in the same place. Fitting the graph
    stays the default way to open a document.
    """

    file_dropped = pyqtSignal(str)
    zoom_changed = pyqtSignal(float)

    MIN_SCALE = 0.1
    MAX_SCALE = 12.0

    def __init__(self, scene: TopologyScene, parent: Optional[QWidget] = None) -> None:
        super().__init__(scene, parent)
        self.topology_scene = scene
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setRenderHint(QPainter.RenderHint.TextAntialiasing)
        self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setAcceptDrops(True)
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.SmartViewportUpdate)
        self._drag_from: Optional[PortItem] = None

    def drawBackground(self, painter: QPainter, rect: QRectF) -> None:  # noqa: N802 - Qt naming
        painter.fillRect(rect, theme.color("window"))
        grid = QColor(theme.color("frame"))
        grid.setAlpha(90)
        painter.setPen(QPen(grid, 0.0))
        step = 40.0
        left = int(rect.left() - (rect.left() % step))
        top = int(rect.top() - (rect.top() % step))
        for x in range(left, int(rect.right()), int(step)):
            painter.drawLine(QPointF(x, rect.top()), QPointF(x, rect.bottom()))
        for y in range(top, int(rect.bottom()), int(step)):
            painter.drawLine(QPointF(rect.left(), y), QPointF(rect.right(), y))

    def wheelEvent(self, event) -> None:  # noqa: N802 - Qt naming
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            self.apply_zoom(1.15 if event.angleDelta().y() > 0 else 1.0 / 1.15)
            event.accept()
            return
        super().wheelEvent(event)

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt naming
        item = self.itemAt(event.position().toPoint())
        if isinstance(item, PortItem) and event.button() == Qt.MouseButton.LeftButton:
            self._drag_from = item
            self.topology_scene.begin_pending(item)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802 - Qt naming
        if self._drag_from is not None:
            self.topology_scene.move_pending(self.mapToScene(event.position().toPoint()))
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802 - Qt naming
        if self._drag_from is not None:
            target = self.itemAt(event.position().toPoint())
            start = self._drag_from
            self._drag_from = None
            self.topology_scene.end_pending()
            if isinstance(target, PortItem):
                self.topology_scene.connect_ports(start, target)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event) -> None:  # noqa: N802 - Qt naming
        if event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            for item in list(self.scene().selectedItems()):
                if isinstance(item, NodeItem):
                    self.topology_scene.remove_node(item.node_id)
                elif isinstance(item, LinkItem):
                    self.topology_scene.remove_link(item)
            event.accept()
            return
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier and event.key() in (Qt.Key.Key_Equal, Qt.Key.Key_Plus):
            self.zoom_in()
            event.accept()
            return
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier and event.key() == Qt.Key.Key_Minus:
            self.zoom_out()
            event.accept()
            return
        super().keyPressEvent(event)

    # --- zoom ----------------------------------------------------------------
    def zoom_level(self) -> float:
        """The scale currently in force, whatever path got us there."""
        return float(self.transform().m11())

    def apply_zoom(self, factor: float) -> None:
        """Scale the scene, clamped to a sane range, and report the number that resulted."""
        current = self.zoom_level()
        if current <= 0.0:
            return
        target = max(self.MIN_SCALE, min(self.MAX_SCALE, current * float(factor)))
        step = target / current
        if abs(step - 1.0) < 1e-9:
            return
        self.scale(step, step)
        self.zoom_changed.emit(self.zoom_level())

    def zoom_in(self) -> None:
        self.apply_zoom(1.15)

    def zoom_out(self) -> None:
        self.apply_zoom(1.0 / 1.15)

    def fit(self) -> None:
        bounds = self.scene().itemsBoundingRect()
        if not bounds.isEmpty():
            self.fitInView(bounds.adjusted(-40, -40, 40, 40), Qt.AspectRatioMode.KeepAspectRatio)
            self.zoom_changed.emit(self.zoom_level())

    # --- drops ---------------------------------------------------------------
    def dragEnterEvent(self, event) -> None:  # noqa: N802 - Qt naming
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:  # noqa: N802 - Qt naming
        for url in event.mimeData().urls():
            self.file_dropped.emit(url.toLocalFile())
        event.acceptProposedAction()


# ---------------------------------------------------------------------------
# Panel
# ---------------------------------------------------------------------------
class CanvasPanel(QWidget):
    """The topology module: canvas, node editor, kind registry, validation report."""

    file_dropped = pyqtSignal(str)

    def __init__(self, workspace: Optional[Workspace] = None, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.workspace = workspace or open_workspace(Path.cwd())
        self.registry = KindRegistry()
        self.scene = TopologyScene(self.registry)
        self.view = TopologyView(self.scene)
        self.console: Optional[widgets.LogConsole] = None
        self._build()
        self.scene.graph_changed.connect(self._refresh_summary)
        self.scene.node_selected.connect(self._show_node)
        self.view.file_dropped.connect(self.file_dropped.emit)

    # --- layout --------------------------------------------------------------
    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.addWidget(self._toolbar())

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self.view)
        splitter.addWidget(self._side())
        splitter.setStretchFactor(0, 5)
        splitter.setStretchFactor(1, 2)
        root.addWidget(splitter, 1)

        self.chip_nodes = widgets.Chip("nodes 0")
        self.chip_edges = widgets.Chip("edges 0")
        self.chip_layers = widgets.Chip("layers -")
        self.chip_kind = widgets.Chip("kinds 0")
        bar = widgets.ToolRow()
        bar.add(self.chip_nodes)
        bar.add(self.chip_edges)
        bar.add(self.chip_layers)
        bar.add(self.chip_kind)
        bar.add_spacer()
        bar.add_label(self.scene.graph.name)
        root.addWidget(bar)

    def _report_zoom(self, value: float) -> None:
        """The toolbar read-out of how large the graph is on screen right now."""
        self.chip_zoom.setText(f"zoom {int(round(value * 100))} pct")
        self.chip_zoom.set_tone("" if abs(value - 1.0) < 0.01 else "warn")

    def _toolbar(self) -> widgets.ToolRow:
        row = widgets.ToolRow()
        self.btn_open = QPushButton("Open topology")
        self.btn_open.clicked.connect(self._open_dialog)
        self.btn_dataset = QPushButton("Add node from dataset")
        self.btn_dataset.clicked.connect(self._dataset_dialog)
        self.btn_save = QPushButton("Save graph")
        self.btn_save.clicked.connect(self._save_dialog)
        self.btn_layout = QPushButton("Auto layout")
        self.btn_layout.clicked.connect(self._auto_layout)
        self.btn_validate = QPushButton("Validate DAG")
        self.btn_validate.clicked.connect(self._validate)
        self.btn_fit = QPushButton("Fit")
        self.btn_fit.clicked.connect(self.view.fit)
        self.btn_zoom_out = QPushButton("Zoom out")
        self.btn_zoom_out.clicked.connect(self.view.zoom_out)
        self.btn_zoom_in = QPushButton("Zoom in")
        self.btn_zoom_in.clicked.connect(self.view.zoom_in)
        for button in (self.btn_open, self.btn_dataset, self.btn_save, self.btn_layout, self.btn_validate, self.btn_fit, self.btn_zoom_out, self.btn_zoom_in):
            row.add(button)
        row.add_spacer()
        self.chip_zoom = widgets.Chip("zoom 100 pct")
        row.add(self.chip_zoom)
        self.view.zoom_changed.connect(self._report_zoom)
        row.add_label("kind")
        self.kind_combo = QComboBox()
        self.kind_combo.setEditable(True)
        self.kind_combo.addItems(sorted(self.registry.known()))
        row.add(self.kind_combo)
        self.btn_add = QPushButton("Add node")
        self.btn_add.setObjectName("Primary")
        self.btn_add.clicked.connect(self._add_node)
        row.add(self.btn_add)
        return row

    def _side(self) -> QWidget:
        holder = QWidget()
        layout = QVBoxLayout(holder)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        node_card = widgets.Card("Node")
        self.node_id_label = QLabel("")
        self.node_id_label.setObjectName("MutedLabel")
        self.node_label_edit = QLineEdit()
        self.node_label_edit.setPlaceholderText("label")
        self.node_label_edit.editingFinished.connect(self._commit_label)
        self.node_kind_combo = QComboBox()
        self.node_kind_combo.setEditable(True)
        self.node_kind_combo.addItems(sorted(self.registry.known()))
        self.node_kind_combo.currentTextChanged.connect(self._commit_kind)
        node_card.add_row(self.node_id_label)
        node_card.add_row(self.node_label_edit, self.node_kind_combo)
        self.attr_table = QTableWidget(0, 2)
        self.attr_table.setHorizontalHeaderLabels(["attribute", "value"])
        self.attr_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.attr_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.attr_table.verticalHeader().setVisible(False)
        self.attr_table.itemChanged.connect(self._commit_attribute)
        table_row = node_card.body.rowCount()
        node_card.add_span(self.attr_table, table_row, 0, 2)
        node_card.set_body_stretch(table_row, 1)
        add_row = widgets.ToolRow()
        self.attr_key = QLineEdit()
        self.attr_key.setPlaceholderText("new attribute")
        self.attr_value = QLineEdit()
        self.attr_value.setPlaceholderText("value")
        add_attr = QPushButton("Add")
        add_attr.clicked.connect(self._add_attribute)
        add_row.add(self.attr_key)
        add_row.add(self.attr_value)
        add_row.add(add_attr)
        node_card.add_span(add_row, node_card.body.rowCount(), 0, 2)
        widgets.share_column(layout, node_card, 5)

        port_card = widgets.Card("Ports")
        self.port_list = QListWidget()
        self.port_list.itemSelectionChanged.connect(self._port_selected)
        list_row = port_card.body.rowCount()
        port_card.add_span(self.port_list, list_row, 0, 3)
        port_card.set_body_stretch(list_row, 1)
        port_row = widgets.ToolRow()
        self.port_name = QLineEdit()
        self.port_name.setPlaceholderText("port name")
        self.port_direction = QComboBox()
        self.port_direction.addItems([DIRECTION_OUT, DIRECTION_IN])
        add_port = QPushButton("Add port")
        add_port.clicked.connect(self._add_port)
        drop_port = QPushButton("Remove")
        drop_port.setObjectName("Danger")
        drop_port.clicked.connect(self._remove_port)
        port_row.add(self.port_name)
        port_row.add(self.port_direction)
        port_row.add(add_port)
        port_row.add(drop_port)
        port_card.add_span(port_row, port_card.body.rowCount(), 0, 3)
        widgets.share_column(layout, port_card, 4)

        kind_card = widgets.Card("Kind registry")
        self.kind_list = QListWidget()
        self.kind_list.setMinimumHeight(80)
        kind_card.add_span(self.kind_list, kind_card.body.rowCount(), 0, 3)
        kind_card.set_body_stretch(0, 1)
        self._fill_kind_list()
        widgets.share_column(layout, kind_card, 3)

        issues_card = widgets.Card("Validation")
        self.issue_list = QListWidget()
        issues_card.add_span(self.issue_list, issues_card.body.rowCount(), 0, 2)
        issues_card.set_body_stretch(0, 1)
        widgets.share_column(layout, issues_card, 4)
        return holder

    # --- console and refresh -------------------------------------------------
    def attach_console(self, console: widgets.LogConsole) -> None:
        self.console = console

    def _say(self, message: str, level: str = "info") -> None:
        if self.console is not None:
            self.console.log(message, level)

    def refresh(self, scan: Any = None) -> None:
        """Pick the newest topology document of the workspace when nothing is loaded."""
        if self.graph_has_content():
            return
        candidate = self.workspace.newest("topology")
        if candidate is not None:
            self.load_path(candidate.path)

    def graph_has_content(self) -> bool:
        return bool(self.scene.graph.nodes)

    # --- loading -------------------------------------------------------------
    def load_path(self, path: Any) -> bool:
        file_path = Path(str(path))
        if not file_path.is_file():
            self._say(f"no such topology document: {file_path}", "error")
            return False
        try:
            graph = load_graph_file(file_path, self.registry, self.workspace.datasets_dir)
        except Exception as exc:  # a bad document must not take the canvas down
            self._say(f"cannot read {file_path.name}: {exc}", "error")
            return False
        self.scene.set_graph(graph)
        self._fill_kind_list()
        self.view.fit()
        notes = "; ".join(graph.notes) if graph.notes else "read cleanly"
        self._say(f"{file_path.name}: dialect {graph.dialect}, {len(graph.nodes)} nodes, {len(graph.edges)} edges ({notes})")
        self._validate()
        return True

    def _open_dialog(self) -> None:
        start = self.workspace.topologies_dir if self.workspace.topologies_dir.is_dir() else self.workspace.config_dir
        found, _ = QFileDialog.getOpenFileName(self, "Open topology", str(start), widgets.filter_for(widgets.OPEN_FAMILIES["topology_document"]))
        if found:
            self.load_path(found)

    def _save_dialog(self) -> None:
        start = self.workspace.topologies_dir if self.workspace.topologies_dir.is_dir() else self.workspace.root
        found, _ = QFileDialog.getSaveFileName(self, "Save topology", str(start / f"{self.scene.graph.name}.json"), widgets.filter_for(widgets.OPEN_FAMILIES["topology_document"]))
        if not found:
            return
        try:
            target = save_graph(self.scene.graph, Path(found), self.registry)
        except (OSError, ValueError) as exc:
            self._say(f"save failed: {exc}", "error")
            return
        self._say(f"saved {self.workspace.relative(target)} in the native dialect with its kind block")

    def _dataset_dialog(self) -> None:
        start = self.workspace.datasets_dir
        found, _ = QFileDialog.getOpenFileName(self, "Open dataset", str(start), widgets.filter_for(widgets.OPEN_FAMILIES["dataset"]))
        if not found:
            return
        self.add_dataset_node(Path(found))

    def add_dataset_node(self, path: Path) -> Optional[NodeItem]:
        """Create a node whose port stack is the payload schema of a dataset file."""
        from gui.model.dataset import load_dataset  # lazy: keeps the panel import light

        try:
            document = load_dataset(path, limit=400)
        except Exception as exc:
            self._say(f"cannot read dataset {path.name}: {exc}", "error")
            return None
        mapping: Dict[str, Any] = {}
        for record in document.records[:200]:
            for key, value in record.payload.items():
                mapping.setdefault(key, value)
        if not mapping:
            self._say(f"{path.name} exposes no payload keys to derive ports from", "warn")
            return None
        node_id = path.parent.name if path.parent.name not in ("", ".") else path.stem
        item = self._node_from_payload(
            node_id,
            "dataset",
            f"{node_id} [{document.format}]",
            mapping,
            {"source": str(self.workspace.relative(path)), "records": str(len(document.records))},
        )
        if item is not None:
            self._say(f"{item.spec.node_id}: {len(item.spec.ports)} ports derived from the payload keys of {path.name}")
        return item

    def add_dataset_record_node(self, record: Any) -> Optional[NodeItem]:
        """Create a node whose port stack is the payload schema of one dataset record."""
        payload = getattr(record, "payload", None)
        if not isinstance(payload, dict) or not payload:
            self._say("that record carries no payload keys to derive ports from", "warn")
            return None
        index = int(getattr(record, "index", 0))
        node_id = self._free_node_id(f"sample_{index:03d}")
        item = self._node_from_payload(
            node_id,
            "dataset",
            f"{node_id} [record]",
            dict(payload),
            {"record": str(index)},
        )
        if item is not None:
            self._say(f"{node_id}: {len(item.spec.ports)} ports derived from the payload of record {index}")
        return item

    def _free_node_id(self, base: str) -> str:
        """First free node id based on a name, so repeated drops never collide."""
        if base not in self.scene.graph.nodes:
            return base
        suffix = 2
        while f"{base}_{suffix}" in self.scene.graph.nodes:
            suffix += 1
        return f"{base}_{suffix}"

    def _node_from_payload(
        self,
        node_id: str,
        kind: str,
        label: str,
        mapping: Dict[str, Any],
        attributes: Dict[str, str],
    ) -> Optional[NodeItem]:
        """Build one node whose ports come from a payload mapping, both directions."""
        if not mapping:
            return None
        item = self.scene.add_node(node_id, kind, label)
        node = item.spec
        for key, value in attributes.items():
            node.attributes[key] = value
        for direction in (DIRECTION_IN, DIRECTION_OUT):
            for port in derive_ports_from_mapping(mapping, direction):
                node.ports = [p for p in node.ports if not (p.direction == direction and p.name == port.name)]
                node.ports.append(port)
        item.rebuild_ports()
        self._show_node(node_id)
        return item

    # --- editing -------------------------------------------------------------
    def _add_node(self) -> None:
        kind = self.kind_combo.currentText().strip() or "generic"
        node_id, ok = QInputDialog.getText(self, "Add node", "identifier:")
        node_id = node_id.strip()
        if not ok or not node_id:
            return
        if node_id in self.scene.graph.nodes:
            self._say(f"a node named {node_id} already exists", "warn")
            return
        self.scene.add_node(node_id, kind)
        self._fill_kind_list()
        self._show_node(node_id)
        self._say(f"added node {node_id} of kind {kind}")

    def _show_node(self, node_id: str) -> None:
        node = self.scene.graph.nodes.get(node_id)
        if node is None:
            return
        self._editing = node_id
        self.node_id_label.setText(f"{node.node_id}  [{node.kind}]")
        self.node_label_edit.blockSignals(True)
        self.node_label_edit.setText(node.label)
        self.node_label_edit.blockSignals(False)
        self.node_kind_combo.blockSignals(True)
        self.node_kind_combo.setCurrentText(node.kind)
        self.node_kind_combo.blockSignals(False)
        self._fill_attributes(node)
        self._fill_ports(node)

    def _fill_attributes(self, node: NodeSpec) -> None:
        self._loading_attributes = True
        self.attr_table.clearContents()
        self.attr_table.setRowCount(len(node.attributes))
        for row, (key, value) in enumerate(sorted(node.attributes.items())):
            self.attr_table.setItem(row, 0, QTableWidgetItem(key))
            self.attr_table.setItem(row, 1, QTableWidgetItem(str(value)))
        self._loading_attributes = False

    def _fill_ports(self, node: NodeSpec) -> None:
        self.port_list.clear()
        for port in node.ports:
            entry = QListWidgetItem(f"{port.direction}  {port.name}  {port.dtype}[{port.width}]  {port.origin}")
            entry.setData(Qt.ItemDataRole.UserRole, (port.direction, port.name))
            self.port_list.addItem(entry)

    def _port_selected(self) -> None:
        items = self.port_list.selectedItems()
        if not items:
            return
        direction, name = items[0].data(Qt.ItemDataRole.UserRole)
        self.port_name.setText(name)
        self.port_direction.setCurrentText(direction)

    def _commit_label(self) -> None:
        node_id = getattr(self, "_editing", "")
        if node_id:
            self.scene.rename_node(node_id, self.node_label_edit.text().strip() or node_id)

    def _commit_kind(self, kind: str) -> None:
        node_id = getattr(self, "_editing", "")
        if node_id and kind:
            self.scene.set_kind(node_id, kind.strip())
            self._fill_kind_list()

    def _commit_attribute(self, item: QTableWidgetItem) -> None:
        if getattr(self, "_loading_attributes", False):
            return
        node_id = getattr(self, "_editing", "")
        if not node_id or item.column() != 1:
            return
        key_item = self.attr_table.item(item.row(), 0)
        if key_item is None:
            return
        self.scene.set_attribute(node_id, key_item.text(), item.text())

    def _add_attribute(self) -> None:
        node_id = getattr(self, "_editing", "")
        key = self.attr_key.text().strip()
        if not node_id or not key:
            return
        self.scene.set_attribute(node_id, key, self.attr_value.text())
        node = self.scene.graph.nodes.get(node_id)
        if node is not None:
            self._fill_attributes(node)
        self.attr_key.clear()
        self.attr_value.clear()

    def _add_port(self) -> None:
        node_id = getattr(self, "_editing", "")
        name = self.port_name.text().strip()
        if not node_id or not name:
            return
        if self.scene.add_port(node_id, self.port_direction.currentText(), name):
            node = self.scene.graph.nodes.get(node_id)
            if node is not None:
                self._fill_ports(node)
            self._say(f"{node_id}: port {self.port_direction.currentText()} {name} added")
        else:
            self._say(f"{node_id}: port {name} already exists on that side", "warn")

    def _remove_port(self) -> None:
        node_id = getattr(self, "_editing", "")
        items = self.port_list.selectedItems()
        if not node_id or not items:
            return
        direction, name = items[0].data(Qt.ItemDataRole.UserRole)
        self.scene.remove_port(node_id, direction, name)
        node = self.scene.graph.nodes.get(node_id)
        if node is not None:
            self._fill_ports(node)
        self._say(f"{node_id}: port {direction} {name} removed with its links")

    def _fill_kind_list(self) -> None:
        counts: Dict[str, int] = {}
        for node in self.scene.graph.nodes.values():
            counts[node.kind] = counts.get(node.kind, 0) + 1
        self.kind_list.clear()
        for kind in sorted(set(counts) | set(self.registry.known())):
            spec = self.registry.get(kind)
            entry = QListWidgetItem(f"{spec.title}  ({kind})  x{counts.get(kind, 0)}  {'builtin' if spec.builtin else 'declared'}")
            entry.setData(Qt.ItemDataRole.UserRole, kind)
            self.kind_list.addItem(entry)
        for combo in (self.kind_combo, self.node_kind_combo):
            current = combo.currentText()
            names = set(self.registry.known())
            if current:
                names.add(current)
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(sorted(names))
            combo.setCurrentText(current)
            combo.blockSignals(False)

    # --- validation ----------------------------------------------------------
    def _validate(self) -> None:
        issues = self.scene.validate()
        self.issue_list.clear()
        for issue in issues:
            entry = QListWidgetItem(f"[{issue.severity}] {issue.code}: {issue.message}")
            entry.setForeground(QBrush(theme.color("fail") if issue.severity == "error" else theme.color("warn")))
            self.issue_list.addItem(entry)
        if not issues:
            entry = QListWidgetItem("no issue: every edge resolves, the graph is acyclic")
            entry.setForeground(QBrush(theme.color("pass")))
            self.issue_list.addItem(entry)
            self._say("validation clean", "pass")
        else:
            errors = sum(1 for issue in issues if issue.severity == "error")
            self._say(f"validation: {errors} errors, {len(issues) - errors} notices", "warn" if errors else "info")

    def _auto_layout(self) -> None:
        positions = self.scene.apply_layout()
        self.view.fit()
        self._say(f"layered layout applied to {len(positions)} nodes")

    def _refresh_summary(self) -> None:
        stats = self.scene.graph.stats()
        self.chip_nodes.setText(f"nodes {stats['nodes']}")
        self.chip_edges.setText(f"edges {stats['edges']}")
        self.chip_layers.setText(f"layers {stats['layers']}")
        self.chip_layers.set_tone("pass" if stats["acyclic"] else "fail")
        self.chip_kind.setText(f"kinds {len(stats['kinds'])}")

    # --- stream hooks --------------------------------------------------------
    def handle_stream_event(self, event: Any) -> None:
        hits = self.scene.pulse_edge(event.sender, event.recipient)
        self.scene.mark_node_activity(event.sender)
        if hits == 0:
            self.scene.mark_node_activity(event.recipient, "seen")

    def project_board(self, board: Any) -> None:
        """Project a parsed board onto this same canvas, so links can be checked."""
        from gui.model.topology import graph_from_board

        self.scene.set_graph(graph_from_board(board))
        self.scene.apply_layout()
        self.view.fit()
        self._say(f"board {board.name} projected: {len(board.dies)} dies, {len(board.links)} links")

    def stop_worker(self) -> None:
        """Release what this module keeps running: pending wiring and edge flow animations."""
        self.scene.end_pending()
        self.scene.stop_animations()


__all__ = ["CanvasPanel", "LinkItem", "NodeItem", "PortItem", "TopologyScene", "TopologyView"]
