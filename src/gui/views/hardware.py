"""Board and hardware viewer: a tree and a schematic built only from files on disk.

Nothing here is hard coded: the die stack, the routing masks, the pin groups and the
inter die links come from reading the exported SystemVerilog board
(``export_artifacts/visualization_board/**/board_complete.v``), and the layout corner
comes from the netlist, LEF, DEF and GDS artifacts the workspace scan found. The drawing
adapts to however many dies the file declares, so two dies, four dies, eight dies or
sixteen dies go through the same code path.

The fault view is also file derived: a SystemVerilog testbench is parsed for its mesh
masks, its miswire assignments and its verdict lines, and those are exactly the values a
simulation uses to cut a route or swap a wire, so the schematic highlights them.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PyQt6.QtCore import QPointF, QRectF, QSizeF, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QFontMetrics,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPainterPathStroker,
    QPen,
    QPolygonF,
    QTransform,
)
from PyQt6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QPushButton,
    QSplitter,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from gui.model.exportdoc import ExportManifest, load_manifest
from gui.model.hardware import BoardDescriptor, BoardDie, FaultPlan, load_board, parse_fault_plan

from .. import theme, widgets
from ..paths import Workspace, open_workspace
from ..workers import (
    CommandWorker,
    build_export_plan,
    build_make_plan,
    export_note,
    probe_verilator,
)

DIE_HUES = ("accent", "gf", "data", "smt", "pass", "warn")
VIRTUAL_W = 1240.0
VIRTUAL_H = 820.0
# The virtual canvas is only a starting size: a board with more dies than it can hold at a
# readable cell size grows it, and the zoom shows the rest. These are the cell bounds.
MIN_DIE_W = 250.0
MIN_DIE_H = 190.0
MAX_DIE_W = 340.0
MAX_DIE_H = 236.0
DIE_GAP = 34.0
DIE_MARGIN = 46.0
HEADER_BAND = 30.0
MIN_ZOOM = 0.25
MAX_ZOOM = 8.0
ZOOM_STEP = 1.2
# Text is kept readable at every zoom: when the fit scale is small the font grows in
# virtual units instead of shrinking on screen, and a line that no longer fits is dropped.
MIN_TEXT_PT = 6.5
ARTIFACT_ROLES = ("board", "netlist", "lef", "def", "gds", "sdc", "report", "rtl", "testbench", "vcd")
REQUIRED_ROLES = ("board", "netlist", "def", "gds")
# A board description reaches the schematic through two documents: the exported Verilog
# wrapper, and the die document config/topology.json that the export itself reads.
BOARD_SUFFIXES = tuple(widgets.HDL_SUFFIXES) + tuple(widgets.TOPOLOGY_SUFFIXES)


# ---------------------------------------------------------------------------
# Schematic
# ---------------------------------------------------------------------------
class SchematicView(QWidget):
    """Paints the die stack and its links, scaled to whatever the file declares.

    The view is a free canvas: it fits the whole board by default, then the person can
    zoom in and out and drag the drawing around to walk between the blocks, however many
    dies the file declares. Zooming is anchored on the cursor, the pan is in screen
    pixels, and hit testing goes through the same transform, so a click always lands on
    the block it is drawn on.
    """

    die_clicked = pyqtSignal(int)
    link_clicked = pyqtSignal(int)
    zoom_changed = pyqtSignal(float)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.board: Optional[BoardDescriptor] = None
        self.plan: Optional[FaultPlan] = None
        self._rects: Dict[int, QRectF] = {}
        self._link_paths: Dict[int, Tuple[QPainterPath, QPointF]] = {}
        self._transform = QTransform()
        self._selected_die: Optional[int] = None
        self._selected_link: Optional[int] = None
        self._active: set = set()
        self._virtual = QSizeF(VIRTUAL_W, VIRTUAL_H)
        self._zoom = 1.0
        self._pan = QPointF(0.0, 0.0)
        self._scale = 1.0
        self._drag_origin: Optional[QPointF] = None
        self._drag_from = QPointF(0.0, 0.0)
        self._drag_moved = False
        self.setMinimumHeight(300)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        self.setCursor(Qt.CursorShape.OpenHandCursor)

    # --- data ----------------------------------------------------------------
    def set_board(self, board: Optional[BoardDescriptor]) -> None:
        self.board = board
        self._selected_die = None
        self._selected_link = None
        self.die_rects()
        self.reset_view()
        self.update()

    def set_fault_plan(self, plan: Optional[FaultPlan]) -> None:
        self.plan = plan
        self.update()

    def select_die(self, index: Optional[int]) -> None:
        self._selected_die = index
        self.update()

    @property
    def selected_die(self) -> Optional[int]:
        return self._selected_die

    @property
    def selected_link(self) -> Optional[int]:
        return self._selected_link

    def pulse_pair(self, first: str, second: str) -> int:
        """Flash the links that carry traffic between two node names."""
        if self.board is None:
            return 0
        hits = 0
        for link in self.board.links:
            if {link.source_name, link.target_name} == {first, second}:
                self._active.add(link.index)
                hits += 1
        if hits:
            self.update()
            QTimer.singleShot(1400, self._clear_active)
        return hits

    def _clear_active(self) -> None:
        self._active.clear()
        self.update()

    # --- geometry ------------------------------------------------------------
    def die_rects(self) -> Dict[int, QRectF]:
        """Adaptive placement: a balanced grid, on a canvas that grows with the die count.

        A two die board keeps the 1240 x 820 canvas it has always had. A board with many
        more dies than that can hold at a readable cell size pushes the canvas out, so
        every block keeps the room its text needs and the zoom, not the layout, decides
        how much of it is on screen.
        """
        if self.board is None or not self.board.dies:
            self._virtual = QSizeF(VIRTUAL_W, VIRTUAL_H)
            return {}
        total = len(self.board.dies)
        columns = max(1, int(math.ceil(math.sqrt(total))))
        rows = max(1, int(math.ceil(total / columns)))
        cell_w = max((VIRTUAL_W - 2 * DIE_MARGIN) / columns, MIN_DIE_W + DIE_GAP)
        cell_h = max((VIRTUAL_H - 2 * DIE_MARGIN) / rows, MIN_DIE_H + DIE_GAP)
        width = min(cell_w - DIE_GAP, MAX_DIE_W)
        height = min(cell_h - DIE_GAP, MAX_DIE_H)
        self._virtual = QSizeF(2 * DIE_MARGIN + columns * cell_w, 2 * DIE_MARGIN + rows * cell_h)
        rects: Dict[int, QRectF] = {}
        for position, die in enumerate(self.board.dies):
            row = position // columns
            column = position % columns
            center_x = DIE_MARGIN + column * cell_w + cell_w / 2.0
            center_y = DIE_MARGIN + row * cell_h + cell_h / 2.0
            rects[die.index] = QRectF(center_x - width / 2.0, center_y - height / 2.0, width, height)
        return rects

    # --- zoom and pan --------------------------------------------------------
    @property
    def zoom(self) -> float:
        """1.0 means the whole board is fitted on screen, the default the view opens with."""
        return self._zoom

    @property
    def virtual_size(self) -> QSizeF:
        return QSizeF(self._virtual)

    def zoom_percent(self) -> int:
        return int(round(self._zoom * 100))

    def _usable(self) -> QRectF:
        return QRectF(self.rect()).adjusted(8, HEADER_BAND, -8, -8)

    def _fit_scale(self) -> float:
        usable = self._usable()
        if self._virtual.width() <= 0 or self._virtual.height() <= 0:
            return 1.0
        return max(0.05, min(usable.width() / self._virtual.width(), usable.height() / self._virtual.height()))

    def _refresh_transform(self) -> QTransform:
        """Compose fit scale, zoom and pan into the one transform drawing and picking use."""
        self._scale = self._fit_scale() * self._zoom
        base = self._usable().center()
        transform = (
            QTransform()
            .translate(base.x() + self._pan.x(), base.y() + self._pan.y())
            .scale(self._scale, self._scale)
            .translate(-self._virtual.width() / 2.0, -self._virtual.height() / 2.0)
        )
        self._transform = transform
        return transform

    def set_zoom(self, value: float, anchor: Optional[QPointF] = None) -> None:
        """Scale the drawing, holding the point under ``anchor`` (the cursor) still."""
        target = max(MIN_ZOOM, min(MAX_ZOOM, float(value)))
        if abs(target - self._zoom) < 1e-6:
            return
        point = QPointF(anchor) if anchor is not None else self._usable().center()
        held = self._to_virtual(point)
        self._zoom = target
        self._refresh_transform()
        base = self._usable().center()
        self._pan = QPointF(
            point.x() - self._scale * (held.x() - self._virtual.width() / 2.0) - base.x(),
            point.y() - self._scale * (held.y() - self._virtual.height() / 2.0) - base.y(),
        )
        self._refresh_transform()
        self.zoom_changed.emit(self._zoom)
        self.update()

    def zoom_in(self) -> None:
        self.set_zoom(self._zoom * ZOOM_STEP)

    def zoom_out(self) -> None:
        self.set_zoom(self._zoom / ZOOM_STEP)

    def reset_view(self) -> None:
        """Back to the default: the whole board fitted, centred, zoom factor one."""
        self._zoom = 1.0
        self._pan = QPointF(0.0, 0.0)
        self._refresh_transform()
        self.zoom_changed.emit(self._zoom)
        self.update()

    def pan_by(self, dx: float, dy: float) -> None:
        self._pan = QPointF(self._pan.x() + float(dx), self._pan.y() + float(dy))
        self._refresh_transform()
        self.update()

    def center_on(self, point: QPointF) -> None:
        """Slide the drawing until a point of the virtual canvas sits in the middle."""
        self._pan = QPointF(
            -self._scale * (point.x() - self._virtual.width() / 2.0),
            -self._scale * (point.y() - self._virtual.height() / 2.0),
        )
        self._refresh_transform()
        self.update()

    def focus_die(self, index: int) -> bool:
        """Bring one die into the middle of the view: how you walk between the blocks."""
        self.die_rects()
        self._refresh_transform()
        rect = self._rects.get(index)
        if rect is None:
            return False
        self.center_on(rect.center())
        return True

    def _edge_anchor(self, rect: QRectF, towards: QPointF) -> QPointF:
        center = rect.center()
        delta = towards - center
        if delta.x() == 0 and delta.y() == 0:
            return center
        scale = 1.0
        if delta.x():
            edge = rect.right() if delta.x() > 0 else rect.left()
            scale = min(scale, abs(edge - center.x()) / abs(delta.x()))
        if delta.y():
            edge = rect.bottom() if delta.y() > 0 else rect.top()
            scale = min(scale, abs(edge - center.y()) / abs(delta.y()))
        return center + QPointF(delta.x() * scale, delta.y() * scale)

    def build_paths(self) -> None:
        self._link_paths.clear()
        if self.board is None:
            return
        for link in self.board.links:
            start_rect = self._rects.get(link.source_index)
            end_rect = self._rects.get(link.target_index)
            if start_rect is None or end_rect is None:
                continue
            start = self._edge_anchor(start_rect, end_rect.center())
            end = self._edge_anchor(end_rect, start_rect.center())
            path = QPainterPath(start)
            if abs(end.x() - start.x()) >= abs(end.y() - start.y()):
                bend = max(30.0, abs(end.x() - start.x()) * 0.35)
                path.cubicTo(QPointF(start.x() + bend, start.y()), QPointF(end.x() - bend, end.y()), end)
            else:
                bend = max(30.0, abs(end.y() - start.y()) * 0.35)
                path.cubicTo(QPointF(start.x(), start.y() + bend), QPointF(end.x(), end.y() - bend), end)
            self._link_paths[link.index] = (path, end)

    # --- painting ------------------------------------------------------------
    def paintEvent(self, event) -> None:  # noqa: N802 - Qt naming
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), theme.color("window"))
        if self.board is None:
            painter.setPen(QPen(theme.color("text_dim")))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "no board loaded: open an exported board Verilog")
            painter.end()
            return

        self._rects = self.die_rects()
        painter.setWorldTransform(self._refresh_transform())
        self.build_paths()
        self._paint_links(painter)
        self._paint_dies(painter)
        painter.setWorldTransform(QTransform())
        self._paint_header(painter)
        painter.end()

    def _readable_font(self, font: QFont) -> QFont:
        """Keep text legible at any zoom: it grows in virtual units instead of shrinking."""
        if self._scale < 1.0:
            needed = MIN_TEXT_PT / max(self._scale, 0.02)
            if needed > font.pointSizeF():
                font.setPointSizeF(needed)
        return font

    def _line_width(self, base: float) -> float:
        """A pen width in virtual units that never falls below a hairline on screen."""
        return max(base, 1.3 / max(self._scale, 0.02))

    def _paint_header(self, painter: QPainter) -> None:
        """The band above the drawing, drawn in screen space so it is always readable."""
        summary = self.board.summary()
        band = QRectF(8.0, 6.0, max(40.0, self.width() - 16.0), 18.0)
        font = theme.mono_font(10.5)
        painter.setFont(font)
        note = ""
        if self.plan is not None and self.plan.masks:
            source = self.plan.source.name if self.plan.source else "testbench"
            note = f"masks forced by {source}"
        note_width = QFontMetrics(font).horizontalAdvance(note) + 14.0 if note else 0.0
        room = max(40.0, band.width() - note_width)
        label = f"{summary['board']}  {summary['dies']} dies  {summary['links']} links  provenance {summary['provenance']}"
        if self._zoom != 1.0:
            label += f"  zoom {self.zoom_percent()} pct"
        painter.setPen(QPen(theme.color("text")))
        painter.drawText(QRectF(band.left(), band.top(), room, band.height()), Qt.AlignmentFlag.AlignLeft, _fit_text(font, label, room))
        if note:
            painter.setPen(QPen(theme.color("warn")))
            painter.drawText(band, Qt.AlignmentFlag.AlignRight, note)

    def _paint_links(self, painter: QPainter) -> None:
        if self.board is None:
            return
        for link in self.board.links:
            entry = self._link_paths.get(link.index)
            if entry is None:
                continue
            path, end = entry
            hot = link.index in self._active
            picked = link.index == self._selected_link
            base = theme.color("link_hot") if hot else (theme.color("data") if picked else theme.color("link"))
            painter.setPen(QPen(base, self._line_width(3.4 if hot else 2.2 if picked else 1.5)))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawPath(path)

            tail = path.pointAtPercent(0.92)
            angle = math.atan2(end.y() - tail.y(), end.x() - tail.x())
            head = 11.0
            arrow = QPolygonF(
                [
                    end,
                    end - QPointF(head * math.cos(angle - 0.42), head * math.sin(angle - 0.42)),
                    end - QPointF(head * math.cos(angle + 0.42), head * math.sin(angle + 0.42)),
                ]
            )
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(base))
            painter.drawPolygon(arrow)

            font = self._readable_font(theme.mono_font(9.0))
            height = QFontMetrics(font).height()
            mid = path.pointAtPercent(0.5)
            cell = QRectF(mid.x() - 90.0, mid.y() - height / 2.0, 180.0, height)
            painter.setFont(font)
            painter.setPen(QPen(base))
            painter.drawText(cell, Qt.AlignmentFlag.AlignCenter, _fit_text(font, f"L{link.index} {link.signal_count} sig", cell.width()))

    def _paint_dies(self, painter: QPainter) -> None:
        for die in self.board.dies:
            rect = self._rects.get(die.index)
            if rect is None:
                continue
            accent = theme.color(DIE_HUES[die.index % len(DIE_HUES)])
            picked = die.index == self._selected_die

            shape = QPainterPath()
            shape.addRoundedRect(rect, 10, 10)
            gradient = QLinearGradient(rect.topLeft(), rect.bottomLeft())
            gradient.setColorAt(0.0, theme.color("panel_alt"))
            gradient.setColorAt(1.0, theme.color("panel"))
            painter.setBrush(QBrush(gradient))
            painter.setPen(QPen(accent, self._line_width(3.0 if picked else 1.4)))
            painter.drawPath(shape)
            self._paint_die_body(painter, die, rect, accent)

    def _paint_die_body(self, painter: QPainter, die: BoardDie, rect: QRectF, accent: QColor) -> None:
        """What a block can show at this zoom, from the die name down, without overlap."""
        pad = 12.0
        inner = max(10.0, rect.width() - 2 * pad)
        used = 16.0

        name_font = self._readable_font(theme.ui_font(12.0, True))
        name_h = QFontMetrics(name_font).height()
        painter.setFont(name_font)
        painter.setPen(QPen(accent))
        painter.drawText(QRectF(rect.left() + pad, rect.top() + 8, inner, name_h), Qt.AlignmentFlag.AlignLeft, _fit_text(name_font, die.name, inner))
        used += name_h + 4.0

        id_font = self._readable_font(theme.mono_font(9.5))
        id_h = QFontMetrics(id_font).height()
        suffix = "  standalone" if die.standalone else ""
        id_text = f"node id {die.node_id} of {die.total_nodes}{suffix}"
        if rect.height() - used > 2 * (id_h + 4.0) + 30.0:
            painter.setFont(id_font)
            painter.setPen(QPen(theme.color("text_dim")))
            painter.drawText(QRectF(rect.left() + pad, rect.top() + used, inner, id_h), Qt.AlignmentFlag.AlignLeft, _fit_text(id_font, id_text, inner))
            used += id_h + 4.0

        band = QRectF(rect.left() + pad, rect.top() + used, inner, max(0.0, rect.height() - used - 12.0))
        used += self._paint_mask(painter, die, band, accent)

        lines: List[str] = []
        if die.chip_instance:
            lines.append(f"chip   {die.chip_module} {die.chip_instance}")
        if die.switch_instance:
            lines.append(f"switch {die.switch_module} {die.switch_instance}")
        lines.append(f"pins {len(die.pins)}  slots {die.channel_slots}  tied {len(die.tied_slots)}")
        row_font = self._readable_font(theme.mono_font(9.0))
        row_h = QFontMetrics(row_font).height()
        space = rect.height() - used - 16.0
        keep = max(0, min(len(lines), int(space // (row_h + 2.0))))
        y = rect.bottom() - 8.0 - keep * (row_h + 2.0)
        painter.setFont(row_font)
        painter.setPen(QPen(theme.color("text")))
        for text in lines[len(lines) - keep :] if keep else []:
            painter.drawText(QRectF(rect.left() + pad, y, inner, row_h), Qt.AlignmentFlag.AlignLeft, _fit_text(row_font, text, inner))
            y += row_h + 2.0

    def _paint_mask(self, painter: QPainter, die: BoardDie, band: QRectF, accent: QColor) -> float:
        """One cell per routing slot: filled means the route is open. Returns the height used."""
        forced = self._forced_mask(die)
        bits = forced if forced is not None else die.mask_bits()
        if not bits:
            return 0.0
        font = self._readable_font(theme.mono_font(8.5))
        label_h = QFontMetrics(font).height()
        gap = 4.0
        room_w = (band.width() + gap) / len(bits) - gap
        box = min(15.0, room_w, max(0.0, band.height() - label_h - 6.0))
        box = max(box, min(10.0 / max(self._scale, 0.02), room_w))
        if box < 6.0 or band.height() < label_h + box + 6.0:
            return 0.0
        top = band.top() + label_h + 2.0
        painter.setFont(font)
        painter.setPen(QPen(theme.color("text_dim")))
        label = "testbench mask" if forced else "routing mask"
        painter.drawText(QRectF(band.left(), band.top(), band.width(), label_h), Qt.AlignmentFlag.AlignLeft, _fit_text(font, label, band.width()))
        for slot, bit in enumerate(bits):
            x = band.left() + slot * (box + gap)
            if x + box > band.right():
                break
            cell = QRectF(x, top, box, box)
            if bit:
                painter.setPen(QPen(accent.darker(130), self._line_width(1.0)))
                painter.setBrush(QBrush(accent))
            else:
                painter.setPen(QPen(theme.color("frame"), self._line_width(1.0)))
                painter.setBrush(QBrush(theme.color("window")))
            painter.drawRect(cell)
            if box >= 9.0:
                painter.setPen(QPen(theme.color("text_dim")))
                painter.drawText(cell, Qt.AlignmentFlag.AlignCenter, str(slot))
        return label_h + 2.0 + box + 6.0

    def _forced_mask(self, die: BoardDie) -> Optional[List[int]]:
        """A testbench mask for this die index overrides what the board file says."""
        if self.plan is None or not self.plan.masks:
            return None
        wanted = {f"M{die.index}_MESH", f"M{die.index}_MASK"}
        key = next((name for name in self.plan.masks if name.upper() in wanted), None)
        if key is None:
            return None
        width = self.plan.mask_widths.get(key, 0)
        value = self.plan.masks[key]
        return [(value >> slot) & 1 for slot in range(width)]

    # --- hit testing and navigation -----------------------------------------
    def _to_virtual(self, point: QPointF) -> QPointF:
        inverted, ok = self._transform.inverted()
        return inverted.map(point) if ok else QPointF(point)

    def _pick(self, point: QPointF) -> bool:
        """Select the die or link under a screen point; the transform makes it zoom proof."""
        position = self._to_virtual(point)
        for index, rect in self._rects.items():
            if rect.contains(position):
                self._selected_die = index
                self._selected_link = None
                self.die_clicked.emit(index)
                self.update()
                return True
        stroker = QPainterPathStroker()
        stroker.setWidth(max(16.0, 12.0 / max(self._scale, 0.02)))
        for index, entry in self._link_paths.items():
            if stroker.createStroke(entry[0]).contains(position):
                self._selected_link = index
                self._selected_die = None
                self.link_clicked.emit(index)
                self.update()
                return True
        return False

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt naming
        point = QPointF(event.position())
        if event.button() == Qt.MouseButton.MiddleButton:
            self._begin_drag(point)
            event.accept()
            return
        if event.button() == Qt.MouseButton.LeftButton:
            if not self._pick(point):
                # not on a block or a link: the drag walks through the board, a click clears
                self._begin_drag(point)
            event.accept()
            return
        super().mousePressEvent(event)

    def _begin_drag(self, point: QPointF) -> None:
        self._drag_origin = point
        self._drag_from = QPointF(self._pan)
        self._drag_moved = False
        self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802 - Qt naming
        if self._drag_origin is None:
            super().mouseMoveEvent(event)
            return
        current = QPointF(event.position())
        delta = current - self._drag_origin
        if not self._drag_moved and (abs(delta.x()) > 2 or abs(delta.y()) > 2):
            self._drag_moved = True
        if self._drag_moved:
            self._pan = QPointF(self._drag_from.x() + delta.x(), self._drag_from.y() + delta.y())
            self._refresh_transform()
            self.update()

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt naming
        """Keep the transform current so a click never uses the maths of an older size."""
        super().resizeEvent(event)
        self._refresh_transform()
        self.update()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802 - Qt naming
        if self._drag_origin is None or event.button() not in (Qt.MouseButton.LeftButton, Qt.MouseButton.MiddleButton):
            super().mouseReleaseEvent(event)
            return
        moved = self._drag_moved
        self._drag_origin = None
        self.setCursor(Qt.CursorShape.OpenHandCursor)
        if not moved and event.button() == Qt.MouseButton.LeftButton:
            self._selected_die = None
            self._selected_link = None
            self.update()

    def wheelEvent(self, event) -> None:  # noqa: N802 - Qt naming
        """Zooming is anchored on the cursor, so the block you point at stays under it."""
        steps = event.angleDelta().y()
        if steps == 0:
            super().wheelEvent(event)
            return
        factor = ZOOM_STEP if steps > 0 else 1.0 / ZOOM_STEP
        self.set_zoom(self._zoom * factor, event.position())
        event.accept()

    def keyPressEvent(self, event) -> None:  # noqa: N802 - Qt naming
        key = event.key()
        if key in (Qt.Key.Key_Plus, Qt.Key.Key_Equal):
            self.zoom_in()
        elif key in (Qt.Key.Key_Minus, Qt.Key.Key_Underscore):
            self.zoom_out()
        elif key == Qt.Key.Key_0:
            self.reset_view()
        elif key == Qt.Key.Key_Left:
            self.pan_by(60, 0)
        elif key == Qt.Key.Key_Right:
            self.pan_by(-60, 0)
        elif key == Qt.Key.Key_Up:
            self.pan_by(0, 60)
        elif key == Qt.Key.Key_Down:
            self.pan_by(0, -60)
        else:
            super().keyPressEvent(event)


def _fit_text(font: QFont, text: str, width: float) -> str:
    """One line that fits ``width`` drawing units, ellipsised by real font metrics."""
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
# Panel
# ---------------------------------------------------------------------------
class HardwarePanel(QWidget):
    """Board module: artifact tree, schematic, derived fault plan, export and runners."""

    board_requested = pyqtSignal(object)
    source_requested = pyqtSignal(str)
    rescan_requested = pyqtSignal()

    def __init__(self, workspace: Optional[Workspace] = None, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.workspace = workspace or open_workspace(Path.cwd())
        self.board: Optional[BoardDescriptor] = None
        self.plan: Optional[FaultPlan] = None
        self.manifest: Optional[ExportManifest] = None
        self.console: Optional[widgets.LogConsole] = None
        self.worker: Optional[CommandWorker] = None
        self.export_worker: Optional[CommandWorker] = None
        self.tool_plans: Dict[str, Any] = {}
        self.export_plan: Any = None
        self.export_dry_plan: Any = None
        self._build()
        self._probe_tools()
        self._probe_export()

    # --- layout --------------------------------------------------------------
    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.addLayout(self._toolbar())

        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["element", "detail"])
        self.tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.tree.currentItemChanged.connect(self._tree_picked)
        splitter.addWidget(self.tree)

        right = QSplitter(Qt.Orientation.Vertical)
        self.schematic = SchematicView()
        self.schematic.die_clicked.connect(self._die_clicked)
        self.schematic.link_clicked.connect(self._describe_link)
        right.addWidget(self._schematic_frame())
        self.artifacts = QTreeWidget()
        self.artifacts.setHeaderLabels(["role", "file", "size", "modified"])
        self.artifacts.header().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.artifacts.itemDoubleClicked.connect(self._artifact_opened)
        right.addWidget(self.artifacts)
        self.fault_card = self._fault_card()
        right.addWidget(self.fault_card)
        self.export_card = self._export_card()
        right.addWidget(self.export_card)
        right.setStretchFactor(0, 5)
        right.setStretchFactor(1, 2)
        right.setStretchFactor(2, 3)
        right.setStretchFactor(3, 3)
        right.setSizes([560, 190, 230, 230])
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 5)
        root.addWidget(splitter, 1)

        self.chip_dies = widgets.Chip("dies -")
        self.chip_links = widgets.Chip("links -")
        self.chip_params = widgets.Chip("params -")
        self.chip_faults = widgets.Chip("faults -")
        self.chip_verdict = widgets.Chip("no verdict")
        bar = widgets.ToolRow()
        for chip in (self.chip_dies, self.chip_links, self.chip_params, self.chip_faults, self.chip_verdict):
            bar.add(chip)
        bar.add_spacer()
        self.status = widgets.Chip("idle")
        bar.add(self.status)
        root.addWidget(bar)

    def _schematic_frame(self) -> QWidget:
        """The drawing with its own navigation strip: zoom in, zoom out, fit, and where you are."""
        frame = QWidget()
        column = QVBoxLayout(frame)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(4)

        row = widgets.ToolRow()
        self.btn_zoom_out = QPushButton("Zoom out")
        self.btn_zoom_out.clicked.connect(self.schematic.zoom_out)
        row.add(self.btn_zoom_out)
        self.btn_zoom_in = QPushButton("Zoom in")
        self.btn_zoom_in.clicked.connect(self.schematic.zoom_in)
        row.add(self.btn_zoom_in)
        self.btn_zoom_fit = QPushButton("Fit board")
        self.btn_zoom_fit.clicked.connect(self.schematic.reset_view)
        row.add(self.btn_zoom_fit)
        self.btn_center_die = QPushButton("Center selected die")
        self.btn_center_die.clicked.connect(self._center_selected_die)
        row.add(self.btn_center_die)
        row.add_spacer()
        self.chip_zoom = widgets.Chip("zoom 100 pct")
        self.chip_canvas = widgets.Chip("canvas -")
        row.add(self.chip_zoom)
        row.add(self.chip_canvas)
        column.addWidget(row)
        column.addWidget(self.schematic, 1)

        self.schematic.zoom_changed.connect(self._report_zoom)
        return frame

    def _report_zoom(self, value: float) -> None:
        """Keep the read-outs honest about how much of the board is on screen right now."""
        self.chip_zoom.setText(f"zoom {int(round(value * 100))} pct")
        self.chip_zoom.set_tone("" if abs(value - 1.0) < 0.01 else "warn")
        size = self.schematic.virtual_size
        self.chip_canvas.setText(f"canvas {int(size.width())} x {int(size.height())}")

    def _center_selected_die(self) -> None:
        """Bring the chosen block to the middle: how you walk between many blocks."""
        index = self.schematic.selected_die
        if index is None:
            self._say("select a die on the drawing or in the element tree first", "info")
            return
        self.schematic.focus_die(index)
        die = self._die_at(index)
        if die is not None:
            self._say(f"centered on die {die.name} at zoom {self.schematic.zoom_percent()} pct")

    def _toolbar(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(6)
        self.picker = widgets.PathPicker(
            self.workspace.sim_dir,
            BOARD_SUFFIXES,
            "board Verilog such as board_complete.v, or a die document such as topology.json",
        )
        self.picker.chosen.connect(self.load_board_path)
        row.addWidget(self.picker, 1)
        self.btn_reload = QPushButton("Reload from disk")
        self.btn_reload.clicked.connect(self._reload)
        row.addWidget(self.btn_reload)
        self.btn_canvas = QPushButton("Project on canvas")
        self.btn_canvas.clicked.connect(self._send_to_canvas)
        row.addWidget(self.btn_canvas)
        return row

    def _fault_card(self) -> widgets.Card:
        card = widgets.Card("Fault plan read from a SystemVerilog testbench")
        row = widgets.ToolRow()
        self.tb_combo = QComboBox()
        self.tb_combo.setMinimumWidth(280)
        row.add(self.tb_combo)
        self.btn_faults = QPushButton("Analyze testbench")
        self.btn_faults.setObjectName("Primary")
        self.btn_faults.clicked.connect(self._analyze_testbench)
        row.add(self.btn_faults)
        self.btn_release = QPushButton("Release masks")
        self.btn_release.clicked.connect(self._release_faults)
        row.add(self.btn_release)
        row.add_spacer()
        self.btn_lint = QPushButton("Lint gate level")
        self.btn_lint.clicked.connect(lambda: self._run_target("lint-topology"))
        row.add(self.btn_lint)
        self.btn_sim = QPushButton("Simulate topology")
        self.btn_sim.clicked.connect(lambda: self._run_target("sim-topology"))
        row.add(self.btn_sim)
        card.add_span(row, 0, 0, 2)

        self.fault_tree = QTreeWidget()
        self.fault_tree.setHeaderLabels(["item", "value", "line"])
        self.fault_tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.fault_tree.setMinimumHeight(110)
        self.fault_tree.currentItemChanged.connect(self._fault_picked)
        card.add_span(self.fault_tree, 1, 0, 2)
        card.set_body_stretch(1, 1)
        return card

    def _export_card(self) -> widgets.Card:
        """Fabrication and board export: run it, then read what it wrote."""
        card = widgets.Card("Fabrication and board artifacts of export_artifacts")
        row = widgets.ToolRow()
        self.btn_export_check = QPushButton("Check export plan")
        self.btn_export_check.clicked.connect(lambda: self._run_export(True))
        row.add(self.btn_export_check)
        self.btn_export = QPushButton("Export artifacts")
        self.btn_export.setObjectName("Primary")
        self.btn_export.clicked.connect(lambda: self._run_export(False))
        row.add(self.btn_export)
        self.btn_manifest = QPushButton("Reload manifest")
        self.btn_manifest.clicked.connect(self._reload_manifest)
        row.add(self.btn_manifest)
        row.add_spacer()
        self.chip_export_run = widgets.Chip("run -")
        self.chip_export_count = widgets.Chip("artifacts -")
        self.chip_export_gap = widgets.Chip("required missing -")
        for chip in (self.chip_export_run, self.chip_export_count, self.chip_export_gap):
            row.add(chip)
        card.add_span(row, 0, 0, 2)

        self.export_tree = QTreeWidget()
        self.export_tree.setHeaderLabels(["artifact", "status", "size", "origin"])
        self.export_tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.export_tree.setMinimumHeight(110)
        self.export_tree.currentItemChanged.connect(self._export_picked)
        card.add_span(self.export_tree, 1, 0, 2)
        card.set_body_stretch(1, 1)
        return card

    # --- console and refresh -------------------------------------------------
    def attach_console(self, console: widgets.LogConsole) -> None:
        self.console = console

    def _say(self, message: str, level: str = "info") -> None:
        if self.console is not None:
            self.console.log(message, level)

    def refresh(self, scan: Any = None) -> None:
        """Rebuild every view of this panel from a workspace scan."""
        source = scan if isinstance(scan, dict) else self.workspace.scan
        self._probe_export()
        self._fill_artifacts(source)
        self._fill_testbenches(source)
        self._fill_manifest()
        if self.board is None:
            candidate = self.workspace.newest("board")
            if candidate is not None:
                self.load_board_path(candidate.path)
            else:
                self._say("the scan found no board Verilog: export the artifacts first", "warn")

    def _fill_artifacts(self, scan: Dict[str, Any]) -> None:
        self.artifacts.clear()
        for role in ARTIFACT_ROLES:
            items = scan.get(role, [])
            if not items:
                continue
            group = QTreeWidgetItem([role, f"{len(items)} files", "", ""])
            self.artifacts.addTopLevelItem(group)
            for artifact in items:
                path = artifact.path
                group.addChild(
                    QTreeWidgetItem(
                        [
                            artifact.role,
                            self.workspace.relative(path),
                            widgets.describe_size(artifact.size_bytes),
                            artifact.modified_utc,
                        ]
                    )
                )
            group.setExpanded(role in ("board", "testbench"))
        missing = [role for role in REQUIRED_ROLES if not scan.get(role)]
        if missing:
            entry = QTreeWidgetItem(["missing", ", ".join(missing), "", ""])
            entry.setForeground(1, QBrush(theme.color("fail")))
            self.artifacts.addTopLevelItem(entry)

    def _fill_testbenches(self, scan: Dict[str, Any]) -> None:
        self.tb_combo.clear()
        for artifact in scan.get("testbench", []):
            self.tb_combo.addItem(self.workspace.relative(artifact.path), str(artifact.path))
        if self.tb_combo.count() == 0:
            self.tb_combo.addItem("no testbench in the scan", "")

    # --- board ---------------------------------------------------------------
    def load_board_path(self, raw: Any) -> bool:
        path = Path(str(raw))
        if not path.is_file():
            self._say(f"no board file at {path}", "error")
            return False
        try:
            board = load_board(path)
        except Exception as exc:  # a malformed export must not take the panel down
            self._say(f"board parse failed for {path.name}: {exc}", "error")
            return False
        if not board.dies:
            self._say(f"{path.name} declares no die instance: nothing to draw", "warn")
            return False
        self.board = board
        self.schematic.set_board(board)
        self._fill_tree(board)
        self._update_chips(board)
        summary = board.summary()
        self._say(f"{self.workspace.relative(path)}: {summary['dies']} dies, {summary['links']} links, {summary['pins']} top ports")
        for warning in board.warnings:
            self._say(f"board notice: {warning}", "warn")
        return True

    def _reload(self) -> None:
        if self.board is not None and self.board.source is not None:
            self.load_board_path(self.board.source)

    def _fill_tree(self, board: BoardDescriptor) -> None:
        self.tree.clear()
        root = QTreeWidgetItem(["board", board.name])
        self.tree.addTopLevelItem(root)
        root.addChild(QTreeWidgetItem(["source", self.workspace.relative(board.source) if board.source else ""]))
        root.addChild(QTreeWidgetItem(["provenance", board.provenance]))

        parameters = QTreeWidgetItem(["parameters", f"{len(board.parameters)}"])
        root.addChild(parameters)
        for name, value in sorted(board.parameters.items()):
            parameters.addChild(QTreeWidgetItem([name, str(value)]))

        dies = QTreeWidgetItem(["dies", f"{len(board.dies)}"])
        root.addChild(dies)
        for die in board.dies:
            entry = QTreeWidgetItem([die.name, f"index {die.index}  node id {die.node_id}  mask {bin(die.routing_mask)}"])
            entry.setData(0, Qt.ItemDataRole.UserRole, ("die", die.index))
            dies.addChild(entry)
            entry.addChild(QTreeWidgetItem(["chip instance", f"{die.chip_module} {die.chip_instance}"]))
            entry.addChild(QTreeWidgetItem(["switch instance", f"{die.switch_module} {die.switch_instance}"]))
            entry.addChild(QTreeWidgetItem(["open routes", ", ".join(str(slot) for slot in die.enabled_slots())]))
            entry.addChild(QTreeWidgetItem(["tied slots", ", ".join(str(slot) for slot in die.tied_slots) or "none"]))
            for group, pins in die.pins_grouped().items():
                holder = QTreeWidgetItem([f"pin group {group}", f"{len(pins)} pins"])
                entry.addChild(holder)
                for pin in pins:
                    holder.addChild(QTreeWidgetItem([pin.name, f"{pin.direction} [{pin.width - 1}:0] {pin.dtype}"]))

        links = QTreeWidgetItem(["links", f"{len(board.links)}"])
        root.addChild(links)
        for link in board.links:
            entry = QTreeWidgetItem([link.label, f"{link.signal_count} signals, line {link.line}"])
            entry.setData(0, Qt.ItemDataRole.UserRole, ("link", link.index))
            links.addChild(entry)
            for signal in link.signals():
                entry.addChild(QTreeWidgetItem([signal, ""]))

        ports = QTreeWidgetItem(["top ports", f"{len(board.top_ports)}"])
        root.addChild(ports)
        for pin in board.top_ports:
            ports.addChild(QTreeWidgetItem([pin.name, f"{pin.direction} [{pin.width - 1}:0] {pin.dtype}"]))

        if board.warnings:
            holder = QTreeWidgetItem(["warnings", f"{len(board.warnings)}"])
            root.addChild(holder)
            for text in board.warnings:
                holder.addChild(QTreeWidgetItem([text, ""]))
        root.setExpanded(True)
        dies.setExpanded(True)

    def _update_chips(self, board: BoardDescriptor) -> None:
        summary = board.summary()
        self.chip_dies.setText(f"dies {summary['dies']}")
        self.chip_links.setText(f"links {summary['links']}")
        self.chip_params.setText(f"parameters {len(summary['parameters'])}")
        self.chip_dies.set_tone("pass" if not summary["warnings"] else "warn")

    # --- selection -----------------------------------------------------------
    def _tree_picked(self, item: Optional[QTreeWidgetItem], _previous: Any) -> None:
        if item is None:
            return
        payload = item.data(0, Qt.ItemDataRole.UserRole)
        if not payload:
            return
        kind, index = payload
        if kind == "die":
            self.schematic.select_die(index)
            if self.schematic.zoom > 1.05:
                # zoomed in, the block can be off screen: walking to it is part of picking it
                self.schematic.focus_die(index)
            die = self._die_at(index)
            if die is not None:
                self._say(f"die {die.name}: mask {bin(die.routing_mask)} opens routes {die.enabled_slots()}")
        elif kind == "link":
            self._describe_link(index)

    def _die_at(self, index: int) -> Optional[BoardDie]:
        if self.board is None:
            return None
        for die in self.board.dies:
            if die.index == index:
                return die
        return None

    def _die_clicked(self, index: int) -> None:
        die = self._die_at(index)
        if die is None:
            return
        self._say(f"selected die {die.name} declared at line {die.line}")

    def _describe_link(self, index: int) -> None:
        if self.board is None:
            return
        for link in self.board.links:
            if link.index == index:
                self._say(f"{link.label}: {link.source_name} to {link.target_name}, {link.signal_count} signals, declared at line {link.line}")
                return

    # --- artifacts -----------------------------------------------------------
    def _artifact_opened(self, item: QTreeWidgetItem, _column: int) -> None:
        role = item.text(0)
        relative = item.text(1)
        if role in ARTIFACT_ROLES and relative and not relative.endswith("files"):
            path = self.workspace.root / relative
            if role == "board":
                self.load_board_path(path)
                return
            if role == "testbench":
                index = self.tb_combo.findData(str(path))
                if index >= 0:
                    self.tb_combo.setCurrentIndex(index)
                self._analyze_testbench()
                return
            self.source_requested.emit(str(path))
            return
        if role == "missing":
            self._say(f"these artifact roles are absent from the workspace: {item.text(1)}", "warn")

    def _send_to_canvas(self) -> None:
        if self.board is None:
            self._say("nothing to project: no board has been read yet", "warn")
            return
        self.board_requested.emit(self.board)

    # --- fault plan ----------------------------------------------------------
    def analyze_testbench_path(self, raw: Any) -> bool:
        """Analyze one specific testbench, the file the source panel handed over."""
        path = Path(str(raw))
        index = self.tb_combo.findData(str(path))
        if index >= 0:
            self.tb_combo.setCurrentIndex(index)
        elif path.is_file():
            self.tb_combo.addItem(self.workspace.relative(path), str(path))
            self.tb_combo.setCurrentIndex(self.tb_combo.count() - 1)
        else:
            self._say(f"no testbench at {path}", "error")
            return False
        self._analyze_testbench()
        return True

    def _analyze_testbench(self) -> None:
        raw = self.tb_combo.currentData()
        if not raw:
            self._say("choose a SystemVerilog testbench from the scan first", "warn")
            return
        path = Path(str(raw))
        try:
            plan = parse_fault_plan(path)
        except Exception as exc:
            self._say(f"testbench parse failed for {path.name}: {exc}", "error")
            return
        self.plan = plan
        self.schematic.set_fault_plan(plan)
        self._fill_fault_tree(plan)
        sites = sum(len(entries) for entries in plan.fault_signals.values())
        self.chip_faults.setText(f"fault sites {sites}")
        self.chip_faults.set_tone("fail" if sites else "pass")
        self.chip_verdict.setText(f"declared verdicts {len(plan.verdicts)}")
        self._say(f"{path.name}: masks {sorted(plan.masks)}, tasks {len(plan.tasks)}, verdict lines {len(plan.verdicts)}")

    def _fill_fault_tree(self, plan: FaultPlan) -> None:
        self.fault_tree.clear()
        parameters = QTreeWidgetItem(["testbench parameters", f"{len(plan.parameters)}", ""])
        self.fault_tree.addTopLevelItem(parameters)
        for name, value in sorted(plan.parameters.items()):
            parameters.addChild(QTreeWidgetItem([name, str(value), ""]))

        masks = QTreeWidgetItem(["mesh masks", f"{len(plan.masks)}", ""])
        self.fault_tree.addTopLevelItem(masks)
        for name, value in sorted(plan.masks.items()):
            width = plan.mask_widths.get(name, 0)
            bits = format(value, f"0{width}b") if width else bin(value)
            entry = QTreeWidgetItem([name, f"{bits}  ({value})  width {width}", ""])
            entry.setData(0, Qt.ItemDataRole.UserRole, ("mask", name))
            masks.addChild(entry)
        masks.setExpanded(True)

        faults = QTreeWidgetItem(["fault and miswire assignments", str(sum(len(v) for v in plan.fault_signals.values())), ""])
        self.fault_tree.addTopLevelItem(faults)
        for signal, sites in sorted(plan.fault_signals.items()):
            for site in sites:
                entry = QTreeWidgetItem([site.signal, f"{site.value}  {site.note}".strip(), str(site.line)])
                entry.setData(0, Qt.ItemDataRole.UserRole, ("fault", site.line))
                faults.addChild(entry)
        faults.setExpanded(True)

        tasks = QTreeWidgetItem(["tasks", f"{len(plan.tasks)}", ""])
        self.fault_tree.addTopLevelItem(tasks)
        for name, line in plan.tasks:
            tasks.addChild(QTreeWidgetItem([name, "", str(line)]))

        verdicts = QTreeWidgetItem(["verdict lines", f"{len(plan.verdicts)}", ""])
        self.fault_tree.addTopLevelItem(verdicts)
        for line, verdict in plan.verdicts:
            entry = QTreeWidgetItem(["verdict line", verdict, str(line)])
            tone = "fail" if "PASS" not in verdict.upper() else "pass"
            entry.setForeground(1, QBrush(theme.color(tone)))
            verdicts.addChild(entry)

        instances = QTreeWidgetItem(["die instances in the testbench", f"{len(plan.die_instances)}", ""])
        self.fault_tree.addTopLevelItem(instances)
        for name, line in plan.die_instances:
            instances.addChild(QTreeWidgetItem([name, "", str(line)]))

    def _fault_picked(self, item: Optional[QTreeWidgetItem], _previous: Any) -> None:
        if item is None:
            return
        payload = item.data(0, Qt.ItemDataRole.UserRole)
        if not payload:
            return
        kind, value = payload
        if kind == "mask":
            index = _mask_die_index(str(value), self.board)
            self.schematic.select_die(index)
            self._say(f"mask {value} drives the routing of die index {index}")
        elif kind == "fault":
            self._say(f"injection site at line {value}: {item.text(0)} = {item.text(1)}", "warn")

    def _release_faults(self) -> None:
        self.plan = None
        self.schematic.set_fault_plan(None)
        self.chip_faults.setText("faults -")
        self.chip_faults.set_tone("")
        self._say("testbench masks released: the schematic shows what the board file declares")

    # --- gate level runners --------------------------------------------------
    def _probe_tools(self) -> None:
        self.verilator = probe_verilator(self.workspace.root)
        for button, target in ((self.btn_lint, "lint-topology"), (self.btn_sim, "sim-topology")):
            plan = build_make_plan(self.workspace.root, target)
            self.tool_plans[target] = plan
            usable = plan.available and self.verilator.available
            button.setEnabled(bool(usable))
            button.setToolTip(self._blocked_reason(plan) if not usable else plan.command_text)
        if not self.verilator.available:
            self._say(self.verilator.reason, "warn")
            self._say(f"command of record: {self.verilator.hint}", "info")

    def _blocked_reason(self, plan: Any) -> str:
        """Say what stops the run, and name the command that would work elsewhere."""
        if not self.verilator.available:
            return f"{self.verilator.reason}\ncommand of record: {self.verilator.hint}"
        return plan.blocked_because()

    def _run_target(self, target: str) -> None:
        plan = self.tool_plans.get(target) or build_make_plan(self.workspace.root, target)
        if not plan.available or not self.verilator.available:
            self._say(self._blocked_reason(plan).replace("\n", " | "), "error")
            return
        if self.worker is not None and self.worker.isRunning():
            self._say("a gate level run is already in progress", "warn")
            return
        self.status.setText("running")
        self.status.set_tone("warn")
        self._say(f"starting {plan.label}: {plan.command_text}")
        worker = CommandWorker(plan)
        worker.line_ready.connect(self._tool_line)
        worker.ended.connect(self._tool_finished)
        self.worker = worker
        worker.start()

    def _tool_line(self, line: str, level: str) -> None:
        if self.console is not None:
            self.console.log(line, level)

    def _tool_finished(self, code: int, summary: str) -> None:
        tone = "pass" if "PASS" in summary else "fail" if "FAIL" in summary else ("pass" if code == 0 else "fail")
        self.status.setText(f"exit {code}")
        self.status.set_tone(tone)
        self.chip_verdict.setText(summary or f"exit {code}")
        self.chip_verdict.set_tone(tone)
        self._say(summary or f"process finished with exit code {code}", "pass" if tone == "pass" else "error")

    # --- artifact export -----------------------------------------------------
    def _probe_export(self) -> None:
        """Work out what an export here can do, and say it in the button tooltip."""
        self.export_plan = build_export_plan(self.workspace.root)
        self.export_dry_plan = build_export_plan(self.workspace.root, dry_run=True)
        note = export_note(self.workspace.root)
        for button, plan in (
            (self.btn_export, self.export_plan),
            (self.btn_export_check, self.export_dry_plan),
        ):
            button.setEnabled(bool(plan.available))
            detail = plan.command_text if plan.available else plan.blocked_because()
            if plan.reason:
                detail = f"{plan.reason}\n{detail}"
            if plan.hint:
                detail = f"{detail}\n{plan.hint}"
            button.setToolTip(detail)
        self.chip_export_run.setText(note)
        self.chip_export_run.set_tone("warn" if note.startswith("no OpenLane") else "pass")

    def _reload_manifest(self) -> None:
        self._fill_manifest(load_manifest(self.workspace.artifacts_dir))
        if self.manifest is not None:
            self._say(self.manifest.headline())

    def _fill_manifest(self, manifest: Optional[ExportManifest] = None) -> None:
        """Show the manifest of the exported tree, exactly as the export wrote it."""
        self.manifest = (
            manifest if manifest is not None else load_manifest(self.workspace.artifacts_dir)
        )
        self.export_tree.clear()
        if self.manifest is None or not self.manifest.entries:
            self.chip_export_count.setText("artifacts 0")
            self.chip_export_count.set_tone("warn")
            self.chip_export_gap.setText("no manifest")
            self.chip_export_gap.set_tone("warn")
            entry = QTreeWidgetItem(["manifest.json", "absent", "", "run the export to get one"])
            entry.setForeground(1, QBrush(theme.color("warn")))
            self.export_tree.addTopLevelItem(entry)
            return
        for group, entries in self.manifest.by_group().items():
            present = sum(1 for item in entries if item.present)
            group_item = QTreeWidgetItem([group, f"{present} of {len(entries)}", "", ""])
            self.export_tree.addTopLevelItem(group_item)
            for item in entries:
                child = QTreeWidgetItem(item.as_row())
                child.setForeground(1, QBrush(theme.color(item.tone)))
                group_item.addChild(child)
            group_item.setExpanded(present < len(entries))
        self.chip_export_run.setText(self.manifest.run_name())
        self.chip_export_run.setToolTip(self.manifest.run_detail())
        self.chip_export_count.setText(
            f"artifacts {self.manifest.present} of {self.manifest.total}"
        )
        self.chip_export_count.set_tone("pass" if self.manifest.present else "warn")
        gaps = self.manifest.missing_required_count
        self.chip_export_gap.setText(f"required missing {gaps}")
        self.chip_export_gap.set_tone("fail" if gaps else "pass")

    def _export_picked(self, item: Optional[QTreeWidgetItem], _previous: Any) -> None:
        """Report the purpose and the digest the manifest recorded for one artifact."""
        if item is None or self.manifest is None:
            return
        label = item.text(0)
        for entry in self.manifest.entries:
            if entry.artifact == label:
                self._say(f"{entry.status} {entry.artifact}: {entry.purpose or 'no purpose recorded'}")
                if entry.sha256:
                    self._say(f"sha256 {entry.sha256}", "info")
                return
        for group, entries in self.manifest.by_group().items():
            if group == label:
                counts = self.manifest.counts()
                joined = ", ".join(f"{status} {count}" for status, count in counts.items())
                self._say(f"{group}: {len(entries)} artifacts | {joined}")
                return

    def request_export(self, dry_run: bool = False) -> None:
        """Ask this module to export, or to report what an export here would produce.

        The plan is built again, because the OpenLane run tree may have appeared or
        disappeared since the panel was opened, and the answer must match the disk.
        """
        self._probe_export()
        self._run_export(bool(dry_run))

    def _run_export(self, dry_run: bool) -> None:
        plan = (self.export_dry_plan if dry_run else self.export_plan) or build_export_plan(
            self.workspace.root, dry_run=dry_run
        )
        if not plan.available:
            hint = f" | what would work: {plan.hint}" if plan.hint else ""
            self._say(f"export blocked: {plan.blocked_because()}{hint}", "error")
            return
        if self.export_worker is not None and self.export_worker.isRunning():
            self._say("an export run is already in progress", "warn")
            return
        self.status.setText("exporting")
        self.status.set_tone("warn")
        self._say(f"starting {plan.label}: {plan.command_text}")
        worker = CommandWorker(plan)
        worker.line_ready.connect(self._export_line)
        worker.ended.connect(self._export_finished)
        self.export_worker = worker
        worker.start()

    def _export_line(self, line: str, level: str) -> None:
        if self.console is not None:
            self.console.log(line, level)

    def _export_finished(self, code: int, summary: str) -> None:
        tone = "pass" if code == 0 else "fail"
        self.status.setText(f"export exit {code}")
        self.status.set_tone(tone)
        self._say(summary or f"export finished with exit code {code}", "pass" if code == 0 else "error")
        self._fill_manifest(load_manifest(self.workspace.artifacts_dir))
        if self.manifest is not None:
            self._say(self.manifest.headline(), "pass" if self.manifest.complete else "warn")
        if code == 0:
            self.rescan_requested.emit()

    # --- stream hook ---------------------------------------------------------
    def ping_channel(self, first: str, second: str) -> int:
        """Flash the schematic link between two names, the pair a stream event mentions."""
        return self.schematic.pulse_pair(str(first), str(second))

    def handle_stream_event(self, event: Any) -> None:
        if self.board is None:
            return
        hits = self.schematic.pulse_pair(event.sender, event.recipient)
        if hits:
            self.schematic.update()

    def stop_worker(self) -> None:
        for worker in (self.worker, self.export_worker):
            if worker is not None and worker.isRunning():
                worker.stop()


def _mask_die_index(name: str, board: Optional[BoardDescriptor]) -> Optional[int]:
    """M0_MESH belongs to die 0, M1_MESH to die 1, and so on."""
    digits = "".join(character for character in name if character.isdigit())
    if not digits:
        return None
    index = int(digits[0])
    if board is None:
        return index
    return index if any(die.index == index for die in board.dies) else None


__all__ = ["HardwarePanel", "SchematicView"]
