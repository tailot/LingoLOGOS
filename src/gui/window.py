"""Workbench shell: five modules over one workspace, sharing one log console.

The shell owns nothing except coordination. It holds the workspace scan, hands a copy of
it to every module on refresh, and routes the handful of cross module requests the
modules announce through signals: a board projected onto the canvas, masks sent from a
testbench to the schematic, a dataset record turned into a node, a transcript topology
opened on the canvas, and every stream event pinged on both graph views.

Every widget here is a native Qt widget or a local style sheet rule. No web engine, no
remote font, no remote icon: the workbench has to work on an air gapped host.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction, QKeySequence
from PyQt6.QtWidgets import (
    QDockWidget,
    QMainWindow,
    QMessageBox,
    QTabWidget,
    QToolBar,
)

import gui
from gui import theme, widgets
from gui.paths import (
    DATASET_SUFFIXES,
    GRAMMAR_SUFFIXES,
    HDL_SUFFIXES,
    LAYOUT_SUFFIXES,
    Artifact,
    Workspace,
    open_workspace,
)
from gui.workers import ScanWorker

from gui.views.canvas import CanvasPanel
from gui.views.dataset import DatasetPanel
from gui.views.hardware import HardwarePanel
from gui.views.lllayer import OrchestratorPanel
from gui.views.sv import SourcePanel

TAB_ORDER = (
    ("canvas", "Topology Canvas"),
    ("lllayer", "Lllayer Stream"),
    ("hardware", "Board and Hardware"),
    ("dataset", "Datasets and Payloads"),
    ("source", "SystemVerilog Source"),
)
DOCK_TITLE = "Execution and Verification Log"
ROLE_ROUTE = {
    "topology": "canvas",
    "board": "hardware",
    "testbench": "source",
    "rtl": "source",
    "transcript": "lllayer",
    "dataset": "dataset",
    "corpus": "dataset",
}
CROSS_SUFFIX_FALLBACK = {
    ".json": "canvas",
    ".yaml": "canvas",
    ".yml": "canvas",
    ".v": "hardware",
    ".sv": "source",
    ".jsonl": "dataset",
    ".txt": "dataset",
}


class WorkbenchWindow(QMainWindow):
    """The window the user sees: one tab per module, one shared console, one scan."""

    def __init__(self, workspace: Optional[Workspace] = None, parent: Optional[Any] = None) -> None:
        super().__init__(parent)
        self.workspace = workspace or open_workspace(Path.cwd())
        self.scan: Dict[str, List[Artifact]] = {}
        self.scan_worker: Optional[ScanWorker] = None
        self.panels: Dict[str, Any] = {}
        self.setObjectName("WorkbenchShell")
        self.setWindowTitle(f"LingoLOGOS Workbench - {self.workspace.root}")
        self.resize(1520, 960)
        self._build_tabs()
        self._build_console()
        self._build_menus()
        self._build_toolbar()
        self._build_status()
        self._wire_panels()
        self.log(f"workbench opened on {self.workspace.root}")
        self._report_profile()
        self.start_scan()

    # --- construction --------------------------------------------------------
    def _build_tabs(self) -> None:
        self.tabs = QTabWidget()
        self.tabs.setObjectName("ModuleTabs")
        self.tabs.setDocumentMode(True)
        self.tabs.setMovable(False)
        self.canvas = CanvasPanel(self.workspace)
        self.orchestrator = OrchestratorPanel(self.workspace)
        self.hardware = HardwarePanel(self.workspace)
        self.dataset = DatasetPanel(self.workspace)
        self.source = SourcePanel(self.workspace)
        self.panels = {
            "canvas": self.canvas,
            "lllayer": self.orchestrator,
            "hardware": self.hardware,
            "dataset": self.dataset,
            "source": self.source,
        }
        for key, title in TAB_ORDER:
            self.tabs.addTab(self.panels[key], title)
        self.setCentralWidget(self.tabs)

    def _build_console(self) -> None:
        self.console = widgets.LogConsole()
        self.console.setMinimumHeight(180)
        self.dock = QDockWidget(DOCK_TITLE, self)
        self.dock.setObjectName("LogDock")
        self.dock.setAllowedAreas(Qt.DockWidgetArea.BottomDockWidgetArea | Qt.DockWidgetArea.TopDockWidgetArea)
        self.dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable
            | QDockWidget.DockWidgetFeature.DockWidgetFloatable
        )
        self.dock.setWidget(self.console)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.dock)
        for panel in self.panels.values():
            panel.attach_console(self.console)

    def _build_menus(self) -> None:
        bar = self.menuBar()
        workspace_menu = bar.addMenu("Workspace")
        self.action_scan = QAction("Rescan workspace", self)
        self.action_scan.setShortcut(QKeySequence("Ctrl+R"))
        self.action_scan.triggered.connect(self.start_scan)
        workspace_menu.addAction(self.action_scan)
        self.action_clear_log = QAction("Clear log console", self)
        self.action_clear_log.triggered.connect(self.console.clear_log)
        workspace_menu.addAction(self.action_clear_log)
        self.action_show_log = QAction("Show log console", self)
        self.action_show_log.triggered.connect(lambda: (self.dock.show(), self.dock.raise_()))
        workspace_menu.addAction(self.action_show_log)
        workspace_menu.addSeparator()
        self.action_export_check = QAction("Check export plan", self)
        self.action_export_check.setShortcut(QKeySequence("Ctrl+Shift+E"))
        self.action_export_check.triggered.connect(lambda _checked=False: self._export_route(True))
        workspace_menu.addAction(self.action_export_check)
        self.action_export = QAction("Export hardware artifacts", self)
        self.action_export.setShortcut(QKeySequence("Ctrl+Shift+S"))
        self.action_export.triggered.connect(lambda _checked=False: self._export_route(False))
        workspace_menu.addAction(self.action_export)
        workspace_menu.addSeparator()
        self.action_quit = QAction("Quit", self)
        self.action_quit.setShortcut(QKeySequence("Ctrl+Q"))
        self.action_quit.triggered.connect(self.close)
        workspace_menu.addAction(self.action_quit)

        modules_menu = bar.addMenu("Modules")
        for index, (key, title) in enumerate(TAB_ORDER, start=1):
            action = QAction(f"{index} - {title}", self)
            action.setShortcut(QKeySequence(f"Alt+{index}"))
            action.triggered.connect(lambda _checked=False, name=key: self._focus(name))
            modules_menu.addAction(action)

        help_menu = bar.addMenu("Help")
        self.action_about = QAction("About this workbench", self)
        self.action_about.triggered.connect(self._show_about)
        help_menu.addAction(self.action_about)
        self.action_roles = QAction("Artifact roles under this workspace", self)
        self.action_roles.triggered.connect(self._show_roles)
        help_menu.addAction(self.action_roles)

    def _build_toolbar(self) -> None:
        bar = QToolBar("Workspace tools")
        bar.setObjectName("WorkspaceTools")
        bar.setMovable(False)
        bar.setFloatable(False)
        bar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        scan = QAction(theme.standard_icon(self.style(), "refresh"), "Rescan workspace", self)
        scan.triggered.connect(self.start_scan)
        bar.addAction(scan)
        clear = QAction(theme.standard_icon(self.style(), "cancel"), "Clear log", self)
        clear.triggered.connect(self.console.clear_log)
        bar.addAction(clear)
        run = QAction(theme.standard_icon(self.style(), "run"), "Replay stream", self)
        run.setToolTip("replay the recorded session transcript in the Lllayer Stream module")
        run.triggered.connect(lambda: (self._focus("lllayer"), self.orchestrator.play_stream()))
        bar.addAction(run)
        self.action_export_tool = QAction(theme.standard_icon(self.style(), "save"), "Export artifacts", self)
        self.action_export_tool.setToolTip("write the fabrication and board artifacts under export_artifacts")
        self.action_export_tool.triggered.connect(lambda _checked=False: self._export_route(False))
        bar.addAction(self.action_export_tool)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, bar)

    def _build_status(self) -> None:
        self.chip_root = widgets.Chip(str(self.workspace.root))
        self.chip_roles = widgets.Chip("no scan yet")
        self.chip_module = widgets.Chip(TAB_ORDER[0][1])
        self.chip_scan = widgets.Chip("scan pending")
        bar = self.statusBar()
        bar.addWidget(self.chip_module)
        bar.addPermanentWidget(self.chip_scan)
        bar.addPermanentWidget(self.chip_roles)
        bar.addPermanentWidget(self.chip_root)
        self.tabs.currentChanged.connect(lambda index: self.chip_module.setText(TAB_ORDER[max(0, index)][1]))
        self._sync_export_actions()

    # --- coordination between modules ----------------------------------------
    def _wire_panels(self) -> None:
        self.hardware.board_requested.connect(self._project_board)
        self.hardware.source_requested.connect(self._open_source)
        self.hardware.rescan_requested.connect(self.start_scan)
        self.source.masks_requested.connect(self._apply_masks)
        self.dataset.record_requested.connect(self._record_to_canvas)
        self.orchestrator.topology_requested.connect(self._open_topology)
        self.orchestrator.edge_pinged.connect(self._ping_channel)
        self.orchestrator.stream_event.connect(self.canvas.handle_stream_event)
        self.orchestrator.stream_event.connect(self.hardware.handle_stream_event)
        self.orchestrator.stream_event.connect(self.source.handle_stream_event)
        self.orchestrator.stream_event.connect(self.dataset.handle_stream_event)
        self.canvas.file_dropped.connect(self._accept_dropped_file)

    def _focus(self, key: str) -> None:
        panel = self.panels.get(key)
        if panel is not None:
            self.tabs.setCurrentWidget(panel)

    def _project_board(self, board: Any) -> None:
        self.canvas.project_board(board)
        self.log(f"the board view projected {board.name} onto the Topology Canvas module", "info")

    def _open_source(self, raw: Any) -> None:
        if self.source.load_path(raw):
            self._focus("source")

    def _apply_masks(self, raw: Any) -> None:
        if self.hardware.analyze_testbench_path(raw):
            self._focus("hardware")
            self.log(f"the fault plan of {Path(str(raw)).name} is now drawn on the board schematic", "info")

    def _record_to_canvas(self, record: Any) -> None:
        item = self.canvas.add_dataset_record_node(record)
        if item is not None:
            self._focus("canvas")

    def _open_topology(self, raw: Any) -> None:
        if self.canvas.load_path(raw):
            self.orchestrator.ensure_nodes(list(self.canvas.scene.graph.nodes))
            self._focus("canvas")

    def _ping_channel(self, first: str, second: str) -> None:
        self.hardware.ping_channel(first, second)

    def _accept_dropped_file(self, raw: str) -> None:
        """Send a dropped file to the module that understands it, role first, suffix second."""
        path = Path(str(raw))
        key = self._module_for(path)
        if key == "canvas":
            self.canvas.load_path(path)
        elif key == "hardware":
            self.hardware.load_board_path(path)
        elif key == "source":
            self.source.load_path(path)
        elif key == "dataset":
            self.dataset.load_path(path)
        elif key == "lllayer":
            self.orchestrator.load_path(path)
        else:
            self.log(f"no module claims a {path.suffix or 'suffix less'} file: {self.workspace.relative(path)}", "warn")
            return
        self._focus(key)

    def _module_for(self, path: Path) -> str:
        """Route by the role the scan gave the file, falling back to its suffix family."""
        target = Path(path)
        for role, artifacts in self.scan.items():
            for artifact in artifacts:
                if Path(artifact.path) == target:
                    return ROLE_ROUTE.get(role, "")
        suffix = target.suffix.lower()
        if suffix in DATASET_SUFFIXES:
            return "dataset"
        if suffix in HDL_SUFFIXES:
            return "source"
        if suffix in LAYOUT_SUFFIXES or suffix in GRAMMAR_SUFFIXES:
            return "hardware"
        return CROSS_SUFFIX_FALLBACK.get(suffix, "")

    # --- artifact export of section 4 ----------------------------------------
    def _export_route(self, dry_run: bool) -> None:
        """The hardware module owns the export; the shell only routes to it and reports."""
        self._focus("hardware")
        self.hardware.request_export(dry_run)
        self._sync_export_actions()

    def _sync_export_actions(self) -> None:
        """Mirror what the hardware module can export right now, in menus and toolbar."""
        for action, plan in (
            (self.action_export, self.hardware.export_plan),
            (self.action_export_check, self.hardware.export_dry_plan),
            (self.action_export_tool, self.hardware.export_plan),
        ):
            if plan is None:
                continue
            action.setEnabled(bool(plan.available))
            detail = plan.command_text if plan.available else plan.blocked_because()
            action.setToolTip(f"{plan.reason} | {detail}" if plan.reason else detail)

    # --- workspace scan ------------------------------------------------------
    def start_scan(self) -> None:
        """Rescan the filesystem in the background; panels never block on it."""
        if self.scan_worker is not None and self.scan_worker.isRunning():
            self.log("a workspace scan is already running", "warn")
            return
        worker = ScanWorker(self.workspace, self)
        worker.scanned.connect(self._apply_scan)
        worker.failed.connect(lambda message: self.log(message, "error"))
        worker.finished.connect(self._scan_finished)
        self.scan_worker = worker
        self.chip_scan.setText("scanning")
        self.chip_scan.set_tone("warn")
        self.statusBar().showMessage(f"scanning {self.workspace.root}")
        worker.start()

    def _scan_finished(self) -> None:
        worker = self.scan_worker
        self.scan_worker = None
        if worker is not None:
            worker.deleteLater()

    def _apply_scan(self, scan: Any) -> None:
        """Hand the fresh scan to every module and report what the tree contains."""
        self.scan = scan if isinstance(scan, dict) else self.workspace.scan
        for panel in self.panels.values():
            panel.refresh(self.scan)
        self._sync_export_actions()
        total = sum(len(items) for items in self.scan.values())
        named = {role: len(items) for role, items in self.scan.items() if items}
        self.chip_roles.setText(f"artifacts {total}")
        self.chip_roles.set_tone("pass" if total else "warn")
        self.chip_scan.setText("scan " + datetime.now().strftime("%H:%M:%S"))
        self.chip_scan.set_tone("")
        self.statusBar().showMessage(f"{total} artifact files in {len(named)} roles under {self.workspace.root}", 12000)
        summary = ", ".join(f"{role} {count}" for role, count in sorted(named.items()))
        self.log(f"workspace scan: {summary or 'nothing found under this root'}")

    # --- console -------------------------------------------------------------
    def _report_profile(self) -> None:
        """Say which workspace profile is in force and why a profile file was only partly used."""
        profile = self.workspace.profile
        if profile is None:
            return
        self.log(profile.describe())
        for problem in profile.problems:
            self.log(f"workspace profile: {problem}", "warn")

    def log(self, message: str, level: str = "info") -> None:
        self.console.log(str(message), level)

    # --- dialogs -------------------------------------------------------------
    def _show_about(self) -> None:
        from PyQt6.QtCore import QT_VERSION_STR, PYQT_VERSION_STR

        self.log("offline by design: no web engine, no remote font, no remote icon", "info")
        QMessageBox.about(
            self,
            "About this workbench",
            "\n".join(
                [
                    "LingoLOGOS Workbench",
                    f"workbench version {gui.__version__}",
                    f"PyQt {PYQT_VERSION_STR}, Qt {QT_VERSION_STR}",
                    "",
                    "The five modules read the workspace filesystem directly: the canvas loads a",
                    "topology document, the board view derives its tree and schematic from the Verilog",
                    "board and the layout artifacts, the stream module replays the transcripts that a",
                    "session wrote, the payload manager edits the training files, and the source",
                    "analyzer parses the testbenches.",
                    "",
                    "Every control, icon and font in this window is native to Qt or shipped with it.",
                    "No web resource is loaded, so the interface works with no network at all.",
                ]
            ),
        )

    def _show_roles(self) -> None:
        if not self.scan:
            self.log("no scan to report yet: press Ctrl+R", "warn")
            return
        for row in self.workspace.role_summary():
            self.log("role " + " | ".join(str(cell) for cell in row))
        self.log(f"root of this workbench: {self.workspace.root}", "info")

    # --- teardown ------------------------------------------------------------
    def closeEvent(self, event) -> None:  # noqa: N802 - Qt naming
        for panel in self.panels.values():
            panel.stop_worker()
        worker = self.scan_worker
        if worker is not None and worker.isRunning():
            worker.requestInterruption()
            worker.wait(2000)
        event.accept()


__all__ = ["DOCK_TITLE", "TAB_ORDER", "WorkbenchWindow"]
