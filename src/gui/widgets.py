"""Shared widgets of the workbench shell: log console, panels, path picker, table setup.

Kept in one place so the five modules look and behave the same, and so the verdict
detection of the console (the ``FINAL VERDICT: PASS`` / ``FINAL VERDICT: FAIL`` line
the SystemVerilog testbenches print, and the ``Verdict: PASS`` line the orchestrator
CLI prints) is written once.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Sequence

from PyQt6.QtCore import QObject, Qt, pyqtSignal
from PyQt6.QtGui import QTextCharFormat, QTextCursor
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QTableView,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from . import theme
from .paths import DATASET_SUFFIXES, HDL_SUFFIXES, TOPOLOGY_DOCUMENT_SUFFIXES, TOPOLOGY_SUFFIXES

_VERDICT = re.compile(r"(FINAL VERDICT|Verdict)\s*:\s*(PASS|FAIL)")
_UNSAT = re.compile(r"\b(UNSAT|INCONCLUSIVE|INVALID)\b")
_SAT = re.compile(r"\b(SAT)\b")


class LogConsole(QPlainTextEdit):
    """Timestamped console pane: the single place every tool writes its output."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("LogConsole")
        self.setFont(theme.mono_font())
        self.setReadOnly(True)
        self.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.setMaximumBlockCount(5000)
        self._lines = 0

    def log(self, message: str, level: str = "info") -> None:
        """Append one timestamped line and colour it by level and by verdict content."""
        stamp = datetime.now().strftime("%H:%M:%S")
        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        format_handle = QTextCharFormat()
        format_handle.setForeground(theme.color("text_dim"))
        cursor.insertText(f"{stamp} ", format_handle)

        level_format = QTextCharFormat()
        if level in ("error", "fail"):
            level_format.setForeground(theme.color("fail"))
        elif level in ("warn",):
            level_format.setForeground(theme.color("warn"))
        elif level in ("pass", "ok"):
            level_format.setForeground(theme.color("pass"))
        elif level == "smt":
            level_format.setForeground(theme.color("smt"))
        else:
            level_format.setForeground(theme.color("text"))
        cursor.insertText(message + "\n", level_format)

        self.setTextCursor(cursor)
        self._lines += 1
        if _VERDICT.search(message) or _UNSAT.search(message):
            self.verticalScrollBar().setValue(self.verticalScrollBar().maximum())

    def info(self, message: str) -> None:
        self.log(message, "info")

    def warn(self, message: str) -> None:
        self.log(message, "warn")

    def error(self, message: str) -> None:
        self.log(message, "error")

    def ok(self, message: str) -> None:
        self.log(message, "pass")

    def smt(self, message: str) -> None:
        self.log(message, "smt")

    @property
    def line_count(self) -> int:
        return self._lines

    def clear_log(self) -> None:
        self.clear()
        self._lines = 0


class Card(QGroupBox):
    """Titled panel used by every module for its sub-blocks, and it folds like an accordion.

    The body lives in its own container widget, so closing a card hides the whole body
    (widgets, nested layouts and anything added afterwards) and the card keeps only its
    title bar: the room goes to the cards the person left open. The fold button is the
    native style arrow, there is no icon asset, and a click on the title bar toggles too.
    """

    #: emitted with the new open state
    toggled = pyqtSignal(bool)

    #: height of a closed card, title bar only
    CLOSED_HEIGHT = 30

    def __init__(self, title: str, parent: Optional[QWidget] = None, expanded: bool = True) -> None:
        super().__init__(title, parent)
        self.setObjectName("Card")
        self.setProperty("folded", False)
        self._shell = QWidget(self)
        self.body = QGridLayout(self._shell)
        self.body.setContentsMargins(10, 12, 10, 10)
        self.body.setHorizontalSpacing(8)
        self.body.setVerticalSpacing(6)
        shell_row = QVBoxLayout(self)
        shell_row.setContentsMargins(2, 4, 2, 2)
        shell_row.setSpacing(0)
        shell_row.addWidget(self._shell)
        self.fold = QToolButton(self)
        self.fold.setObjectName("CardFold")
        self.fold.setAutoRaise(True)
        self.fold.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.fold.setCursor(Qt.CursorShape.PointingHandCursor)
        self.fold.setFixedSize(20, 16)
        self.fold.clicked.connect(self.toggle)
        self._open = not bool(expanded)
        self.set_expanded(expanded)

    # --- accordion -----------------------------------------------------------
    @property
    def expanded(self) -> bool:
        return self._open

    def toggle(self) -> None:
        self.set_expanded(not self._open)

    def set_expanded(self, flag: bool) -> None:
        """Open or close the body; a closed card shows its title bar and nothing else."""
        flag = bool(flag)
        if flag == self._open:
            return
        self._open = flag
        self._shell.setVisible(flag)
        self.fold.setArrowType(Qt.ArrowType.DownArrow if flag else Qt.ArrowType.RightArrow)
        self.fold.setToolTip("Collapse this panel" if flag else "Expand this panel")
        self.fold.setAccessibleName(self.fold.toolTip())
        self.setProperty("folded", not flag)
        self.style().unpolish(self)
        self.style().polish(self)
        self.updateGeometry()
        self.update()
        self.toggled.emit(flag)

    def set_body_stretch(self, row: int, factor: int = 1) -> None:
        """Let one body row absorb the vertical space the card is given."""
        self.body.setRowStretch(row, factor)

    # --- content -------------------------------------------------------------
    def add_row(self, *widgets: QWidget, row: Optional[int] = None) -> int:
        target = self.body.rowCount() if row is None else row
        for column, widget in enumerate(widgets):
            self.body.addWidget(widget, target, column)
        return target

    def add_span(self, widget: QWidget, row: int, column: int, span: int = 1) -> None:
        self.body.addWidget(widget, row, column, 1, span)

    # --- geometry ------------------------------------------------------------
    def title_zone(self) -> int:
        """Vertical band owned by the title bar and its fold button."""
        return max(18, self.fontMetrics().height() + 8)

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt naming
        super().resizeEvent(event)
        self.fold.move(self.width() - self.fold.width() - 8, max(2, self.title_zone() - self.fold.height() - 2))

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt naming
        """A click on the title bar, or anywhere on a closed card, folds it open or shut."""
        if event.button() == Qt.MouseButton.LeftButton and (
            not self._open or float(event.position().y()) <= float(self.title_zone())
        ):
            self.toggle()
            event.accept()
            return
        super().mousePressEvent(event)


def share_column(layout: QVBoxLayout, card: Card, weight: int) -> None:
    """Add a card to a column so it shares the height, and give its share back when folded.

    This is what makes a stack of cards liquid instead of a stack of fixed boxes: open cards
    split the room by weight, and a closed card leaves nothing behind but its title bar.
    """
    layout.addWidget(card, weight if card.expanded else 0)
    card.toggled.connect(lambda open_now, item=card, share=weight: layout.setStretchFactor(item, share if open_now else 0))


class Chip(QLabel):
    """Small status pill: kind of a node, SAT/UNSAT verdict, PASS/FAIL result."""

    def __init__(self, text: str, tone: str = "", parent: Optional[QWidget] = None) -> None:
        super().__init__(text, parent)
        self.set_tone(tone)

    def set_tone(self, tone: str) -> None:
        tone_map = {"pass": "ChipPass", "fail": "ChipFail", "warn": "ChipWarn"}
        self.setObjectName(tone_map.get(tone, "Chip"))
        self.style().unpolish(self)
        self.style().polish(self)


class ToolRow(QWidget):
    """Horizontal strip of buttons and labels that opens a panel."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.row = QHBoxLayout(self)
        self.row.setContentsMargins(0, 0, 0, 0)
        self.row.setSpacing(6)

    def add(self, widget: QWidget) -> QWidget:
        self.row.addWidget(widget)
        return widget

    def add_label(self, text: str, muted: bool = True) -> QLabel:
        label = QLabel(text)
        if muted:
            label.setObjectName("MutedLabel")
        self.row.addWidget(label)
        return label

    def add_spacer(self) -> None:
        self.row.addStretch(1)


class PathPicker(QWidget):
    """Path line plus browse button, with a filter built from known suffixes."""

    chosen = pyqtSignal(str)

    def __init__(
        self,
        start: Path,
        suffixes: Sequence[str] = DATASET_SUFFIXES,
        label: str = "File",
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._start = Path(start)
        self._suffixes = tuple(suffixes)
        self.edit = QLineEdit()
        self.edit.setPlaceholderText(label)
        self.browse = QPushButton("Browse")
        self.browse.clicked.connect(self._browse)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.edit, 1)
        layout.addWidget(self.browse)
        self.edit.editingFinished.connect(
            lambda: self.chosen.emit(self.edit.text().strip())
        )

    def set_path(self, path: Optional[Path]) -> None:
        self.edit.setText(str(path) if path else "")

    def path(self) -> Optional[Path]:
        text = self.edit.text().strip()
        if not text:
            return None
        candidate = Path(text).expanduser()
        return candidate if candidate.exists() else None

    def _browse(self) -> None:
        patterns = " ".join(f"*{s}" for s in self._suffixes)
        start = self._start if self._start.exists() else Path.home()
        found, _ = QFileDialog.getOpenFileName(
            self, "Open file", str(start), f"Matching ({patterns});;All files (*)"
        )
        if found:
            self.edit.setText(found)
            self.chosen.emit(found)


def filter_for(suffixes: Sequence[str]) -> str:
    """QFileDialog filter text for a suffix family."""
    patterns = " ".join(f"*{s}" for s in suffixes)
    return f"Matching ({patterns});;All files (*)"


# Re-exported families so panels build their pickers from one source of truth.
OPEN_FAMILIES = {
    "topology": TOPOLOGY_SUFFIXES,
    "topology_document": TOPOLOGY_DOCUMENT_SUFFIXES,
    "dataset": DATASET_SUFFIXES,
    "hdl": HDL_SUFFIXES,
}


class TableSetup(QObject):
    """Common QTableView configuration, applied once per table."""

    @staticmethod
    def apply(view: QTableView, wrap: bool = False) -> QTableView:
        view.setAlternatingRowColors(True)
        view.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        view.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        view.setSortingEnabled(True)
        view.verticalHeader().setVisible(True)
        view.verticalHeader().setDefaultSectionSize(22)
        view.horizontalHeader().setHighlightSections(False)
        view.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        view.horizontalHeader().setStretchLastSection(True)
        if not wrap:
            view.setWordWrap(False)
        return view


def status_label(text: str = "", tone: str = "") -> Chip:
    return Chip(text, tone)


def describe_size(size_bytes: int) -> str:
    """Human readable byte count for artifact tables."""
    units = ("B", "KB", "MB", "GB", "TB")
    value = float(size_bytes)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{size_bytes} B"


__all__: List[str] = [
    "Card",
    "Chip",
    "DATASET_SUFFIXES",
    "HDL_SUFFIXES",
    "LogConsole",
    "OPEN_FAMILIES",
    "PathPicker",
    "TableSetup",
    "TOPOLOGY_SUFFIXES",
    "ToolRow",
    "describe_size",
    "filter_for",
    "status_label",
]
