"""SystemVerilog testbench analyzer: read the source, highlight what matters, run it.

The panel parses a .sv or .v file through the Qt-free analyzer in the model layer, which
extracts localparams, mesh masks, tasks, instances and the verdict lines a simulation
prints. Those symbols drive the highlighter and the symbol tree, so clicking a mask jumps
to the line that declares it.

Run buttons go through the same tool plans the board view uses. On a host where Verilator
cannot be launched directly the buttons stay disabled and say why, together with the
command that works elsewhere; no button pretends to run something it cannot run.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, List, Optional, Tuple

from PyQt6.QtCore import QSize, Qt, pyqtSignal
from PyQt6.QtGui import (
    QBrush,
    QColor,
    QPainter,
    QSyntaxHighlighter,
    QTextCharFormat,
    QTextCursor,
)
from PyQt6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QTextEdit,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from gui.model.svdoc import SvAnalysis, SvSymbol, load_sv

from .. import theme, widgets
from ..paths import HDL_SUFFIXES, Workspace, open_workspace
from ..workers import (
    CommandWorker,
    ParseWorker,
    build_make_plan,
    build_verilator_plan,
    probe_verilator,
    verdict_of_line,
)

HDL_ROLES = ("testbench", "rtl", "board", "netlist")
SYMBOL_GROUPS = (
    ("localparams", "symbols"),
    ("mesh masks", "masks"),
    ("fault assignments", "faults"),
    ("tasks", "tasks"),
    ("functions", "functions"),
    ("modules", "modules"),
    ("instances", "instances"),
)
BOLD_KEYS = ("verdict", "decl", "mask", "param")
MAKE_TARGET_FOR_TB = "sim-tb"
MAX_MARKS = 500


# ---------------------------------------------------------------------------
# Highlighting
# ---------------------------------------------------------------------------
class SvHighlighter(QSyntaxHighlighter):
    """Applies the style rules the analyzer hands out, resolved through the theme."""

    def __init__(self, document: Any) -> None:
        super().__init__(document)
        self._rules: List[Tuple[re.Pattern, QColor, bool]] = []

    def set_source(self, analysis: Optional[SvAnalysis]) -> None:
        """Take the rule table from a fresh analysis and repaint every block."""
        rules: List[Tuple[re.Pattern, QColor, bool]] = []
        if analysis is not None:
            for pattern, style_key in analysis.rule_table():
                rules.append((pattern, theme.style_color(style_key), style_key in BOLD_KEYS))
        self._rules = rules
        self.rehighlight()

    def highlightBlock(self, text: str) -> None:  # noqa: N802 - Qt naming
        for pattern, tint, bold in self._rules:
            for match in pattern.finditer(text):
                format_handle = QTextCharFormat()
                format_handle.setForeground(tint)
                if bold:
                    format_handle.setFontWeight(700)
                self.setFormat(match.start(), match.end() - match.start(), format_handle)


class LineNumberArea(QWidget):
    """Gutter that shows source line numbers next to the code view."""

    def __init__(self, view: "SourceCodeView") -> None:
        super().__init__(view)
        self.view = view

    def sizeHint(self) -> QSize:  # noqa: N802
        return QSize(self.view.line_number_width(), 0)

    def paintEvent(self, event: Any) -> None:  # noqa: N802 - Qt naming
        self.view.paint_line_numbers()


class SourceCodeView(QPlainTextEdit):
    """Read-only code pane with a line gutter, focus-line marker and search marks."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("SourceView")
        self.setReadOnly(True)
        self.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.setFont(theme.mono_font(9.5))
        self.highlighter = SvHighlighter(self.document())
        self._focus_line = 0
        self._marks: List[int] = []
        self._mark_term = ""
        self.gutter = LineNumberArea(self)
        self.blockCountChanged.connect(self._update_gutter_width)
        self.updateRequest.connect(lambda *_: self.gutter.update())
        self.cursorPositionChanged.connect(self._repaint_selections)
        self.verticalScrollBar().valueChanged.connect(lambda *_: self.gutter.update())
        self._update_gutter_width(0)

    # --- gutter --------------------------------------------------------------
    def line_number_width(self) -> int:
        digits = max(3, len(str(max(1, self.blockCount()))))
        return 14 + 9 * digits

    def _update_gutter_width(self, _count: int) -> None:
        self.setViewportMargins(self.line_number_width(), 0, 0, 0)

    def resizeEvent(self, event: Any) -> None:  # noqa: N802 - Qt naming
        super().resizeEvent(event)
        self.gutter.setGeometry(0, 0, self.line_number_width(), self.height())

    def paint_line_numbers(self) -> None:
        """Draw the visible block numbers into the gutter."""
        painter = QPainter(self.gutter)
        area = self.gutter.rect()
        painter.fillRect(area, theme.color("panel"))
        painter.setFont(theme.mono_font(9.0))
        block = self.firstVisibleBlock()
        number = block.blockNumber()
        top = round(self.blockBoundingGeometry(block).translated(self.contentOffset()).top())
        bottom = top + round(self.blockBoundingRect(block).height())
        height = self.fontMetrics().height()
        while block.isValid() and top <= area.bottom():
            if block.isVisible() and bottom >= area.top():
                tint = theme.color("accent") if number + 1 == self._focus_line else theme.color("text_dim")
                painter.setPen(tint)
                painter.drawText(0, top, self.gutter.width() - 8, height, Qt.AlignmentFlag.AlignRight, str(number + 1))
            block = block.next()
            top = bottom
            bottom = top + round(self.blockBoundingRect(block).height())
            number += 1
        painter.end()

    # --- navigation ----------------------------------------------------------
    def set_focus_line(self, line: int) -> None:
        """Mark one source line as the line of interest."""
        self._focus_line = max(0, line)
        self.gutter.update()
        self._repaint_selections()

    def goto_line(self, line: int) -> None:
        """Move the cursor to a source line and center it."""
        if line < 1:
            return
        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.Start)
        cursor.movePosition(QTextCursor.MoveOperation.Down, QTextCursor.MoveMode.MoveAnchor, line - 1)
        cursor.movePosition(QTextCursor.MoveOperation.StartOfBlock)
        self.setTextCursor(cursor)
        self.set_focus_line(line)
        self.centerCursor()

    def line_of_position(self, position: int) -> int:
        return self.document().findBlock(position).blockNumber() + 1

    # --- search --------------------------------------------------------------
    def set_marks(self, needle: str) -> int:
        """Mark every occurrence of a string and report how many were found."""
        self._marks = []
        self._mark_term = needle
        if needle:
            haystack = self.toPlainText().lower()
            probe = needle.lower()
            position = haystack.find(probe)
            while position != -1 and len(self._marks) < MAX_MARKS:
                self._marks.append(position)
                position = haystack.find(probe, position + max(1, len(probe)))
        self._repaint_selections()
        return len(self._marks)

    def jump_to_mark(self, index: int) -> Tuple[int, int]:
        """Select one occurrence and return its one-based ordinal and line."""
        if not self._marks:
            return 0, 0
        ordinal = index % len(self._marks)
        position = self._marks[ordinal]
        cursor = self.textCursor()
        cursor.setPosition(position)
        cursor.movePosition(QTextCursor.MoveOperation.Right, QTextCursor.MoveMode.KeepAnchor, max(1, len(self._mark_term)))
        self.setTextCursor(cursor)
        line = self.line_of_position(position)
        self.set_focus_line(line)
        self.centerCursor()
        return ordinal + 1, line

    def _repaint_selections(self) -> None:
        selections: List[QTextEdit.ExtraSelection] = []
        if self._focus_line:
            cursor = self.textCursor()
            cursor.movePosition(QTextCursor.MoveOperation.StartOfBlock)
            cursor.movePosition(QTextCursor.MoveOperation.EndOfBlock, QTextCursor.MoveMode.KeepAnchor)
            holder = QTextEdit.ExtraSelection()
            holder.cursor = cursor
            holder.format.setBackground(theme.color("accent_soft"))
            selections.append(holder)
        for position in self._marks:
            cursor = self.textCursor()
            cursor.setPosition(position)
            cursor.movePosition(QTextCursor.MoveOperation.Right, QTextCursor.MoveMode.KeepAnchor, max(1, len(self._mark_term)))
            holder = QTextEdit.ExtraSelection()
            holder.cursor = cursor
            holder.format.setBackground(theme.color("warn"))
            holder.format.setForeground(theme.color("window"))
            selections.append(holder)
        self.setExtraSelections(selections)


# ---------------------------------------------------------------------------
# Panel
# ---------------------------------------------------------------------------
class SourcePanel(QWidget):
    """Testbench module: source viewer, symbol index, gate level runner controls."""

    masks_requested = pyqtSignal(str)

    def __init__(self, workspace: Optional[Workspace] = None, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.workspace = workspace or open_workspace(Path.cwd())
        self.analysis: Optional[SvAnalysis] = None
        self.console: Optional[widgets.LogConsole] = None
        self.parser: Optional[ParseWorker] = None
        # Serial of the newest parse request. A result that belongs to an older request is
        # dropped, so the file the user asked for last is the file the panel ends up with.
        self._parse_serial = 0
        self.runner: Optional[CommandWorker] = None
        self.run_plan: Any = None
        self.make_plan: Any = None
        self.verilator: Any = None
        self._mark_index = 0
        self._build()
        self._probe_tools()

    # --- layout --------------------------------------------------------------
    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.addLayout(self._file_row())

        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.symbols = QTreeWidget()
        self.symbols.setHeaderLabels(["symbol", "value", "line"])
        self.symbols.header().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.symbols.itemDoubleClicked.connect(self._symbol_activated)
        self.symbols.currentItemChanged.connect(self._symbol_picked)
        splitter.addWidget(self.symbols)

        right = QWidget()
        layout = QVBoxLayout(right)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(self._find_row())
        self.code = SourceCodeView()
        layout.addWidget(self.code, 1)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 5)
        root.addWidget(splitter, 1)

        root.addWidget(self._run_card())

        self.chip_lines = widgets.Chip("lines -")
        self.chip_params = widgets.Chip("localparams -")
        self.chip_masks = widgets.Chip("masks -")
        self.chip_tasks = widgets.Chip("tasks -")
        self.chip_verdicts = widgets.Chip("verdict lines -")
        self.chip_verdict = widgets.Chip("no verdict")
        bar = widgets.ToolRow()
        for chip in (self.chip_lines, self.chip_params, self.chip_masks, self.chip_tasks, self.chip_verdicts, self.chip_verdict):
            bar.add(chip)
        bar.add_spacer()
        self.status = widgets.Chip("idle")
        bar.add(self.status)
        root.addWidget(bar)

    def _file_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(6)
        self.picker = widgets.PathPicker(self.workspace.sim_dir, HDL_SUFFIXES, "SystemVerilog or Verilog source")
        self.picker.chosen.connect(self.load_path)
        row.addWidget(self.picker, 3)
        self.recent = QComboBox()
        self.recent.setMinimumWidth(260)
        self.recent.currentIndexChanged.connect(self._recent_picked)
        row.addWidget(self.recent, 2)
        self.btn_reload = QPushButton("Reparse")
        self.btn_reload.clicked.connect(self._reload)
        row.addWidget(self.btn_reload)
        return row

    def _find_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(6)
        row.addWidget(QLabel("find"))
        self.find = QLineEdit()
        self.find.setPlaceholderText("mark every occurrence, Enter jumps to the next")
        self.find.textChanged.connect(self._find_changed)
        self.find.returnPressed.connect(self._find_next)
        row.addWidget(self.find, 1)
        self.find_label = QLabel("0 hits")
        self.find_label.setObjectName("MutedLabel")
        row.addWidget(self.find_label)
        self.btn_find = QPushButton("Next")
        self.btn_find.clicked.connect(self._find_next)
        row.addWidget(self.btn_find)
        row.addSpacing(10)
        self.top_combo = QComboBox()
        self.top_combo.setMinimumWidth(220)
        self.top_combo.currentIndexChanged.connect(self._probe_tools)
        row.addWidget(QLabel("top module"))
        row.addWidget(self.top_combo)
        row.addStretch(1)
        return row

    def _run_card(self) -> widgets.Card:
        card = widgets.Card("Gate level run")
        row = widgets.ToolRow()
        self.btn_verilator = QPushButton("Run with Verilator")
        self.btn_verilator.setObjectName("Primary")
        self.btn_verilator.clicked.connect(lambda: self._run(self.run_plan))
        row.add(self.btn_verilator)
        self.btn_make = QPushButton("Run make sim-tb")
        self.btn_make.clicked.connect(lambda: self._run(self.make_plan))
        row.add(self.btn_make)
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setObjectName("Danger")
        self.btn_stop.clicked.connect(self._stop)
        row.add(self.btn_stop)
        row.add_spacer()
        self.btn_masks = QPushButton("Send masks to board view")
        self.btn_masks.clicked.connect(self._send_masks)
        row.add(self.btn_masks)
        card.add_span(row, 0, 0, 2)
        self.plan_label = QLabel("no run plan")
        self.plan_label.setObjectName("MutedLabel")
        self.plan_label.setWordWrap(True)
        card.add_span(self.plan_label, 1, 0, 2)
        self.verdict_note = QLabel("the console colours every line matching FINAL VERDICT: PASS or FINAL VERDICT: FAIL")
        self.verdict_note.setObjectName("MutedLabel")
        card.add_span(self.verdict_note, 2, 0, 2)
        return card

    # --- console and refresh -------------------------------------------------
    def attach_console(self, console: widgets.LogConsole) -> None:
        self.console = console

    def _say(self, message: str, level: str = "info") -> None:
        if self.console is not None:
            self.console.log(message, level)

    def refresh(self, scan: Any = None) -> None:
        """Offer every HDL file the workspace scan found."""
        source = scan if isinstance(scan, dict) else self.workspace.scan
        previous = self.recent.currentData()
        self.recent.blockSignals(True)
        self.recent.clear()
        for role in HDL_ROLES:
            for artifact in source.get(role, []):
                self.recent.addItem(f"{role}: {self.workspace.relative(artifact.path)}", str(artifact.path))
        self.recent.blockSignals(False)
        if self.recent.count() == 0:
            self.recent.addItem("no HDL source in the scan", "")
            return
        if previous:
            index = self.recent.findData(previous)
            if index >= 0:
                self.recent.setCurrentIndex(index)
        if self.analysis is None and self._parse_serial == 0:
            # Only when nothing was ever requested: an explicit load that is still being
            # parsed must not be overtaken by this automatic one.
            candidate = self.workspace.newest("testbench") or self.workspace.newest("rtl")
            if candidate is not None:
                self.load_path(candidate.path)

    def _recent_picked(self, _index: int) -> None:
        raw = self.recent.currentData()
        if raw:
            self.load_path(raw)

    # --- parsing -------------------------------------------------------------
    def load_path(self, raw: Any) -> bool:
        path = Path(str(raw))
        if not path.is_file():
            self._say(f"no source file at {path}", "error")
            return False
        if self.parser is not None and self.parser.isRunning():
            self.parser.wait(3000)
        self.status.setText("parsing")
        self.status.set_tone("warn")
        self._parse_serial += 1
        serial = self._parse_serial
        worker = ParseWorker(load_sv, path)
        worker.parsed.connect(lambda analysis, serial=serial: self._parsed_if_current(serial, analysis))
        worker.failed.connect(lambda message, serial=serial: self._parse_failed(serial, path, message))
        self.parser = worker
        worker.start()
        return True

    def _reload(self) -> None:
        if self.analysis is not None and self.analysis.source is not None:
            self.load_path(self.analysis.source)

    def _parsed_if_current(self, serial: int, analysis: SvAnalysis) -> None:
        """Apply a parse result only when no newer request has been made since."""
        if serial == self._parse_serial:
            self._parsed(analysis)

    def _parse_failed(self, serial: int, path: Path, message: str) -> None:
        if serial == self._parse_serial:
            self._say(f"parse failed for {path.name}: {message}", "error")

    def _parsed(self, analysis: SvAnalysis) -> None:
        self.analysis = analysis
        self.code.setPlainText(analysis.text)
        self.code.highlighter.set_source(analysis)
        self._fill_symbols(analysis)
        self._fill_top_modules(analysis)
        self._update_chips(analysis)
        self.code.goto_line(1)
        self.picker.set_path(str(analysis.source) if analysis.source else "")
        summary = analysis.summary()
        self._say(
            f"{Path(summary['path']).name}: {summary['lines']} lines, {summary['localparams']} localparams, "
            f"{summary['masks']} masks, {summary['tasks']} tasks, {summary['verdict_lines']} verdict lines"
        )
        masks = analysis.mesh_masks()
        if masks:
            self._say("mesh masks: " + ", ".join(f"{mask.name}={mask.value}" for mask in masks))
        for line, text in analysis.verdicts:
            self._say(f"line {line}: {text}", "pass" if "PASS" in text.upper() else "error")
        self.status.setText("parsed")
        self.status.set_tone("pass")
        self._probe_tools()

    def _fill_symbols(self, analysis: SvAnalysis) -> None:
        self.symbols.clear()
        for title, attribute in SYMBOL_GROUPS:
            entries: List[SvSymbol] = getattr(analysis, attribute, [])
            if not entries:
                continue
            group = QTreeWidgetItem([title, f"{len(entries)}", ""])
            self.symbols.addTopLevelItem(group)
            for symbol in entries:
                kind, name, value, line = symbol.as_row()
                entry = QTreeWidgetItem([name, value, str(line)])
                entry.setData(0, Qt.ItemDataRole.UserRole, line)
                entry.setForeground(1, self._tint_for(kind, name))
                group.addChild(entry)
            if title in ("mesh masks", "localparams"):
                group.setExpanded(True)

        if analysis.verdicts:
            group = QTreeWidgetItem(["verdict lines", f"{len(analysis.verdicts)}", ""])
            self.symbols.addTopLevelItem(group)
            for line, text in analysis.verdicts:
                entry = QTreeWidgetItem(["verdict", text, str(line)])
                entry.setData(0, Qt.ItemDataRole.UserRole, line)
                entry.setForeground(1, QBrush(theme.color("pass") if "PASS" in text.upper() else theme.color("fail")))
                group.addChild(entry)
            group.setExpanded(True)

    def _tint_for(self, kind: str, name: str) -> Any:
        """Colour for a symbol row: masks in red, code entities in violet, values in amber."""
        upper = name.upper()
        if "MESH" in upper or "MASK" in upper or kind in ("mask", "fault"):
            return QBrush(theme.color("fail"))
        if kind in ("task", "function", "module", "instance"):
            return QBrush(theme.color("smt"))
        return QBrush(theme.color("data"))

    def _fill_top_modules(self, analysis: SvAnalysis) -> None:
        current = self.top_combo.currentData()
        self.top_combo.blockSignals(True)
        self.top_combo.clear()
        names = [module.name for module in analysis.modules]
        for name in names:
            self.top_combo.addItem(name, name)
        if not names:
            self.top_combo.addItem("(no module declared in this file)", "")
        if current:
            index = self.top_combo.findData(current)
            if index >= 0:
                self.top_combo.setCurrentIndex(index)
        self.top_combo.blockSignals(False)

    def _update_chips(self, analysis: SvAnalysis) -> None:
        summary = analysis.summary()
        self.chip_lines.setText(f"lines {summary['lines']}")
        self.chip_params.setText(f"localparams {summary['localparams']}")
        self.chip_masks.setText(f"masks {summary['masks']}")
        self.chip_tasks.setText(f"tasks {summary['tasks']}")
        self.chip_verdicts.setText(f"verdict lines {summary['verdict_lines']}")
        self.chip_masks.set_tone("fail" if summary["masks"] else "")
        self.chip_verdicts.set_tone("warn" if summary["verdict_lines"] else "")

    # --- symbol navigation ---------------------------------------------------
    def _symbol_activated(self, item: QTreeWidgetItem, _column: int) -> None:
        line = item.data(0, Qt.ItemDataRole.UserRole)
        if line:
            self.code.goto_line(int(line))

    def _symbol_picked(self, item: Optional[QTreeWidgetItem], _previous: Any) -> None:
        if item is None:
            return
        line = item.data(0, Qt.ItemDataRole.UserRole)
        if line:
            self.code.set_focus_line(int(line))

    # --- search --------------------------------------------------------------
    def _find_changed(self, text: str) -> None:
        hits = self.code.set_marks(text.strip())
        self._mark_index = 0
        self.find_label.setText(f"{hits} hits")

    def _find_next(self) -> None:
        needle = self.find.text().strip()
        if not needle:
            return
        ordinal, line = self.code.jump_to_mark(self._mark_index)
        self._mark_index += 1
        if ordinal:
            self._say(f"find {needle}: occurrence {ordinal} at line {line}")

    # --- run plans -----------------------------------------------------------
    def _probe_tools(self) -> None:
        self.verilator = probe_verilator(self.workspace.root)
        source = self.analysis.source if self.analysis is not None else None
        top = self.top_combo.currentData() or ""
        self.run_plan = build_verilator_plan(self.workspace.root, source, top) if source else None
        self.make_plan = build_make_plan(self.workspace.root, MAKE_TARGET_FOR_TB)
        runnable = self.run_plan is not None and self.run_plan.available and self.verilator.available
        self.btn_verilator.setEnabled(bool(runnable))
        self.btn_verilator.setToolTip("run this plan" if runnable else self._why_blocked(self.run_plan))
        make_ok = self.make_plan.available and self.verilator.available
        self.btn_make.setEnabled(bool(make_ok))
        self.btn_make.setToolTip(self.make_plan.command_text if make_ok else self._why_blocked(self.make_plan))
        self.plan_label.setText(f"run plan: {self.run_plan.command_text}" if runnable else self._why_blocked(self.run_plan).replace("\n", " "))

    def _why_blocked(self, plan: Any) -> str:
        """Plain English for why a run button is disabled."""
        if plan is None:
            return "no source loaded yet, so there is nothing to run"
        if not self.verilator.available:
            return f"{self.verilator.reason} Command of record: {self.verilator.hint}. The plan would be: {plan.command_text}"
        return f"{plan.blocked_because()} The plan would be: {plan.command_text}"

    def _run(self, plan: Any) -> None:
        if plan is None or not plan.available or not self.verilator.available:
            self._say(self._why_blocked(plan).replace("\n", " | "), "error")
            return
        if self.runner is not None and self.runner.isRunning():
            self._say("a gate level run is already in progress", "warn")
            return
        self.status.setText("running")
        self.status.set_tone("warn")
        self._say(f"starting {plan.label}: {plan.command_text}")
        runner = CommandWorker(plan)
        runner.line_ready.connect(self._tool_line)
        runner.ended.connect(self._tool_finished)
        self.runner = runner
        runner.start()

    def _tool_line(self, line: str, level: str) -> None:
        if self.console is not None:
            self.console.log(line, level)
        verdict = verdict_of_line(line)
        if verdict:
            self.chip_verdict.setText(verdict)
            self.chip_verdict.set_tone("pass" if "PASS" in verdict.upper() else "fail")

    def _tool_finished(self, code: int, summary: str) -> None:
        tone = "fail" if "FAIL" in summary.upper() else ("pass" if code == 0 else "fail")
        self.status.setText(f"exit {code}")
        self.status.set_tone(tone)
        self.chip_verdict.setText(summary or f"exit {code}")
        self.chip_verdict.set_tone(tone)
        self._say(summary or f"process finished with exit code {code}", "pass" if tone == "pass" else "error")

    def _stop(self) -> None:
        if self.runner is not None and self.runner.isRunning():
            self.runner.stop()
            self._say("stop requested for the running process", "warn")

    def _send_masks(self) -> None:
        if self.analysis is None or self.analysis.source is None:
            self._say("no testbench loaded, so there is nothing to send to the board view", "warn")
            return
        if not self.analysis.masks:
            self._say(f"{self.analysis.source.name} declares no mask parameter: the board view keeps the board file masks", "warn")
            return
        self.masks_requested.emit(str(self.analysis.source))

    # --- stream hook ---------------------------------------------------------
    def handle_stream_event(self, event: Any) -> None:
        """Mirror a verdict seen on the live stream into the verdict chip."""
        verdict = getattr(event, "verdict", None)
        if callable(verdict):
            verdict = verdict()
        if verdict:
            self.chip_verdict.setText(str(verdict))
            self.chip_verdict.set_tone("pass" if "PASS" in str(verdict).upper() else "fail")

    def stop_worker(self) -> None:
        if self.runner is not None and self.runner.isRunning():
            self.runner.stop()


__all__ = ["LineNumberArea", "SourceCodeView", "SourcePanel", "SvHighlighter"]
