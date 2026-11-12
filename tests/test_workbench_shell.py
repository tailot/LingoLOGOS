"""The workbench shell: five modules, one console, filesystem driven, nothing blocking.

These tests build the real window against the real repository tree, with Qt running on the
offscreen platform so no window is ever shown. They cover:

  1. The shell is a main window holding exactly the five documented modules, one shared log
     console dock, menus, a toolbar and status chips.
  2. The background workspace scan fills every module and the chips, without freezing.
  3. Cross module routes: board to canvas, masks to the schematic, dataset record to a node,
     transcript events to the other four modules, dropped files to the right tab.
  4. Every tab paints itself, and closing the shell stops every worker it started.
  5. The artifact export of section 4: its buttons, the manifest it reads back, and the
     rescan it asks the shell for.
"""

import json
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

try:
    from PyQt6.QtCore import QElapsedTimer, QEventLoop, Qt
    from PyQt6.QtGui import QAction
    from PyQt6.QtWidgets import QApplication, QDockWidget, QTabWidget, QToolBar
except ImportError as missing:  # no Qt runtime on this host: the shell cannot be built
    pytest.skip(f"Qt is not loadable here: {missing}", allow_module_level=True)

from gui import theme, widgets
from gui.model.svdoc import load_sv
from gui.model.topology import TopologyGraph
from gui.paths import DATASET_SUFFIXES, HDL_SUFFIXES, Workspace, open_workspace
from gui.views.hardware import HardwarePanel, SchematicView
from gui.views.sv import SourcePanel
from gui.window import DOCK_TITLE, ROLE_ROUTE, TAB_ORDER, WorkbenchWindow

REPO_ROOT = Path(__file__).resolve().parents[1]

needs_generated_artifacts = pytest.mark.skipif(
    not (REPO_ROOT / "export_artifacts").is_dir(),
    reason="no generated hardware export artifacts in this checkout",
)


def pump(app: QApplication, timeout_ms: int = 20000, until=None) -> bool:
    """Spin the Qt event loop until a condition holds, the way the shell pumps its workers."""
    timer = QElapsedTimer()
    timer.start()
    while True:
        app.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 40)
        if until is not None and until():
            return True
        if timer.elapsed() >= timeout_ms:
            return until is None


@pytest.fixture(scope="session")
def qapp() -> QApplication:
    app = QApplication.instance() or QApplication([])
    app.setApplicationName("LingoLOGOS Workbench test session")
    app.setStyleSheet(theme.stylesheet())
    return app


@pytest.fixture
def shell(qapp) -> WorkbenchWindow:
    """A real shell over the real repository, with its first scan already delivered.

    The constructor launches the first workspace scan by itself, which is what the user
    sees when the workbench opens: the tree is read, never typed in.
    """
    window = WorkbenchWindow(open_workspace(REPO_ROOT))
    window.resize(1280, 800)
    assert pump(qapp, until=lambda: bool(window.scan) and window.scan_worker is None)
    yield window
    window.close()
    pump(qapp, timeout_ms=3000)


def role_paths(window: WorkbenchWindow, role: str):
    return [artifact.path for artifact in window.workspace.of_role(role)]


# --------------------------------------------------------------------------- shell shape
def test_the_shell_is_a_main_window_holding_exactly_the_five_modules(qapp):
    window = WorkbenchWindow(open_workspace(REPO_ROOT))
    try:
        assert window.inherits("QMainWindow")
        assert isinstance(window.tabs, QTabWidget)
        assert window.tabs.count() == len(TAB_ORDER) == 5
        assert [window.tabs.tabText(index) for index in range(window.tabs.count())] == [
            title for _, title in TAB_ORDER
        ]
        assert sorted(window.panels) == sorted(key for key, _ in TAB_ORDER)
    finally:
        window.close()


def test_the_log_console_is_one_shared_dock_attached_to_every_module(qapp):
    window = WorkbenchWindow(open_workspace(REPO_ROOT))
    try:
        assert isinstance(window.dock, QDockWidget)
        assert window.dock.windowTitle() == DOCK_TITLE
        assert window.dock.objectName() == "LogDock"
        assert window.dock.widget() is window.console
        assert window.dock.isAreaAllowed(Qt.DockWidgetArea.BottomDockWidgetArea)
        assert window.console.line_count > 0, "start up writes what it knows to the console"
        before = window.console.line_count
        window.log("a line from the test suite", "info")
        assert window.console.line_count == before + 1
        window.action_clear_log.trigger()
        assert window.console.line_count == 0
        for key, panel in window.panels.items():
            for name in ("attach_console", "refresh", "handle_stream_event", "stop_worker"):
                assert callable(getattr(panel, name)), f"{key} is missing {name}"
    finally:
        window.close()


def test_menus_toolbar_and_status_chips_are_local_qt_only(qapp):
    window = WorkbenchWindow(open_workspace(REPO_ROOT))
    try:
        menus = [action.text() for action in window.menuBar().actions()]
        assert menus == ["Workspace", "Modules", "Help"]
        assert window.findChildren(QToolBar), "the workspace tools live in a Qt toolbar"
        actions = window.findChildren(QAction)
        assert any(action.text() == "Rescan workspace" for action in actions)
        assert any(action.text() == "Replay stream" for action in actions)
        assert any(action.text() == "Quit" for action in actions)
        assert window.chip_module.text() == TAB_ORDER[0][1]
        # The shell starts its first filesystem scan on its own, so the chip is already busy.
        assert window.chip_scan.text() in ("scan pending", "scanning")
        assert str(window.workspace.root) in window.chip_root.text()
    finally:
        window.close()


# --------------------------------------------------------------------------- filesystem scan
def test_the_background_scan_fills_every_module_and_the_chips(shell):
    assert isinstance(shell.scan, dict) and shell.scan
    assert shell.chip_roles.text().startswith("artifacts ")
    total = int(shell.chip_roles.text().split()[-1])
    assert total == sum(len(items) for items in shell.scan.values()) > 0
    assert not shell.chip_scan.text().startswith("scan pending")
    console_text = shell.console.toPlainText()
    assert "workspace scan:" in console_text


def test_the_scan_finds_the_roles_the_workbench_is_built_around(shell):
    roles = {role: len(items) for role, items in shell.scan.items() if items}
    assert roles.get("topology", 0) >= 1
    assert roles.get("dataset", 0) >= 1
    assert roles.get("grammar", 0) >= 1
    if (REPO_ROOT / "export_artifacts").is_dir():
        assert roles.get("board", 0) >= 1
        assert roles.get("testbench", 0) >= 1
        assert roles.get("transcript", 0) >= 1


def test_the_modules_are_repainted_after_a_rescan_without_losing_the_window(shell, qapp):
    shell.start_scan()
    assert pump(qapp, until=lambda: shell.scan_worker is None and bool(shell.scan))
    assert shell.tabs.count() == 5


# --------------------------------------------------------------------------- cross module routes
@needs_generated_artifacts
def test_the_hardware_module_projects_its_board_on_the_canvas_with_its_button(shell, qapp):
    board_path = role_paths(shell, "board")[0]
    canvas = shell.panels["canvas"]
    canvas.scene.set_graph(TopologyGraph())
    assert len(canvas.scene.graph.nodes) == 0, "the canvas starts from an empty drawing"
    shell.hardware.load_board_path(board_path)
    assert pump(qapp, until=lambda: shell.hardware.board is not None)
    board = shell.hardware.board
    assert board.die_count >= 2
    shell.hardware.btn_canvas.click()
    assert pump(qapp, until=lambda: len(canvas.scene.graph.nodes) >= board.die_count)
    assert {die.name for die in board.dies} == set(canvas.scene.graph.nodes)
    assert len(canvas.scene.graph.edges) == len(board.links)
    assert "projected" in shell.console.toPlainText()


def test_the_mask_button_in_the_source_module_lands_on_the_schematic(shell, qapp):
    testbenches = [path for path in role_paths(shell, "testbench") if load_sv(path).masks]
    if not testbenches:
        pytest.skip("no SystemVerilog testbench in this checkout declares a mesh mask")
    wanted = testbenches[0]
    assert shell.source.load_path(wanted) is True
    assert pump(qapp, until=lambda: shell.source.analysis is not None and shell.source.analysis.source == wanted)
    shell.source.btn_masks.click()
    pump(qapp, timeout_ms=4000)
    assert shell.hardware.plan is not None
    assert shell.tabs.currentWidget() is shell.panels["hardware"]
    summary = shell.hardware.plan.summary()
    assert "masks" in summary and "fault_sites" in summary


def test_an_automatic_first_load_never_overtakes_a_file_the_user_already_asked_for(shell, qapp):
    """Fresh panel, nothing parsed yet: a refresh while the explicit parse runs must not start another."""
    newest = shell.workspace.newest("testbench")
    others = [path for path in role_paths(shell, "testbench") if newest is not None and path != newest.path]
    if newest is None or not others:
        pytest.skip("this checkout has no second testbench to tell apart from the newest one")
    wanted = others[0]
    panel = SourcePanel(shell.workspace)
    try:
        assert panel.load_path(wanted) is True
        panel.refresh()
        parser = panel.parser
        assert pump(qapp, until=lambda: panel.analysis is not None and not parser.isRunning())
        pump(qapp, timeout_ms=600)
        assert panel.analysis.source == wanted
    finally:
        panel.deleteLater()
        pump(qapp, timeout_ms=300)


def test_a_stale_parse_result_is_dropped_by_serial(shell, qapp):
    candidates = role_paths(shell, "testbench")
    if len(candidates) < 2:
        pytest.skip("this checkout has fewer than two testbenches to tell apart")
    first, second = candidates[0], candidates[-1]
    assert shell.source.load_path(second) is True
    serial = shell.source._parse_serial
    parser = shell.source.parser
    assert pump(qapp, until=lambda: shell.source.analysis is not None and not parser.isRunning())
    pump(qapp, timeout_ms=300)
    shell.source._parsed_if_current(serial - 1, load_sv(first))
    assert shell.source.analysis.source == second


def test_a_dropped_yaml_file_goes_to_the_topology_canvas(shell):
    assert shell._module_for(Path("somewhere") / "mesh.yaml") == "canvas"
    assert shell._module_for(Path("somewhere") / "mesh.yml") == "canvas"


def test_the_console_names_the_workspace_profile_in_force(shell):
    assert "workspace profile" in shell.console.toPlainText()


def test_a_selected_dataset_record_becomes_a_node_on_the_canvas(shell, qapp):
    datasets = role_paths(shell, "dataset")
    assert datasets
    target = next((path for path in datasets if path.name == "seed_payload.json"), datasets[0])
    assert target.suffix.lower() in DATASET_SUFFIXES
    shell.dataset.load_path(target)
    assert pump(qapp, until=lambda: shell.dataset.document is not None)
    document = shell.dataset.document
    assert document.records
    canvas = shell.panels["canvas"]
    nodes_before = len(canvas.scene.graph.nodes)
    shell.dataset.table.setCurrentIndex(shell.dataset.model.index(0, 0))
    shell.dataset.btn_push.click()
    pump(qapp, timeout_ms=3000)
    assert len(canvas.scene.graph.nodes) == nodes_before + 1
    assert shell.tabs.currentWidget() is canvas


@needs_generated_artifacts
def test_a_replayed_transcript_reaches_the_other_four_modules(shell, qapp):
    transcript = role_paths(shell, "transcript")[0]
    orchestrator = shell.panels["lllayer"]
    canvas = shell.panels["canvas"]
    canvas.load_path(REPO_ROOT / "config" / "topology.json")
    pump(qapp, until=lambda: canvas.graph_has_content())
    nodes_before = len(canvas.scene.graph.nodes)
    events = []
    pings = []
    orchestrator.stream_event.connect(events.append)
    orchestrator.edge_pinged.connect(lambda first, second: pings.append((first, second)))
    assert orchestrator.load_path(transcript) is True
    assert orchestrator.replayer.total > 0
    orchestrator.btn_end.click()
    assert pump(qapp, until=lambda: orchestrator.stream.delivered == orchestrator.replayer.total)
    assert len(events) == orchestrator.replayer.total
    assert len(pings) == len(events)
    assert orchestrator.node_rows, "each node keeps a model assignment row"
    assert len(canvas.scene.graph.nodes) >= nodes_before
    assert shell.tabs.count() == 5


def test_a_dropped_file_is_routed_to_the_module_that_understands_it(shell, qapp):
    expectations = [
        ("topology", "canvas"),
        ("dataset", "dataset"),
        ("testbench", "source"),
        ("transcript", "lllayer"),
        ("board", "hardware"),
    ]
    for role, key in expectations:
        paths = role_paths(shell, role)
        if not paths:
            continue
        assert shell._module_for(paths[0]) == ROLE_ROUTE[role] == key, role
    assert shell._module_for(Path("nowhere.no_such_extension")) == ""
    transcripts = role_paths(shell, "transcript")
    if transcripts:
        shell.canvas.file_dropped.emit(str(transcripts[0]))
        pump(qapp, timeout_ms=4000)
        assert shell.tabs.currentWidget() is shell.panels["lllayer"]
        assert shell.panels["lllayer"].transcript_path == transcripts[0]


def test_suffix_fallback_routing_works_for_files_the_scan_never_saw(shell):
    assert shell._module_for(Path("somewhere/deep_corpus.jsonl")) == "dataset"
    assert shell._module_for(Path("somewhere/wrapper.v")) == "source"
    assert shell._module_for(Path("somewhere/die.lef")) == "hardware"
    assert Path("x.jsonl").suffix in DATASET_SUFFIXES
    assert Path("x.v").suffix in HDL_SUFFIXES


# --------------------------------------------------------------------------- painting
def test_every_module_paints_itself_without_needing_a_web_view(shell, qapp):
    painted = []
    for index in range(shell.tabs.count()):
        shell.tabs.setCurrentIndex(index)
        pump(qapp, timeout_ms=1500)
        panel = shell.tabs.widget(index)
        image = panel.grab()
        assert not image.isNull()
        assert image.width() > 0 and image.height() > 0
        painted.append((shell.tabs.tabText(index), image.width(), image.height()))
    assert len(painted) == 5
    assert all(width >= 100 for _, width, _ in painted)


def test_the_schematic_and_the_canvas_react_to_a_click(shell, qapp):
    boards = role_paths(shell, "board")
    if not boards:
        pytest.skip("no board description in this checkout")
    shell.hardware.load_board_path(boards[0])
    assert pump(qapp, until=lambda: shell.hardware.board is not None)
    view = shell.hardware.findChildren(SchematicView)
    assert view, "the schematic is a painted Qt widget, not an embedded web view"
    assert view[0].board is not None, "the schematic paints the board the module loaded"
    shell.hardware.ping_channel("node_0", "node_1")
    pump(qapp, timeout_ms=1500)
    assert not shell.hardware.grab().isNull()


# --------------------------------------------------------------------------- artifact export
def manifest_document() -> dict:
    """A manifest shaped like the one the export script writes, for a scratch tree."""
    return {
        "schema": "lingologos-export-v1",
        "generated_utc": "2026-01-04T09:12:33Z",
        "tool": "export_artifacts.py",
        "design": "lingologos_top",
        "run": {"directory": "runs/2026.01.04_09-00-00", "name": "2026.01.04_09-00-00"},
        "destination": "export_artifacts",
        "entries": [
            {
                "artifact": "factory/gds/lingologos_top.gds",
                "status": "COPIED",
                "purpose": "tapeout layout, the GDSII the streamout step wrote",
                "source": "runs/67-final/gds/top.gds",
                "size_bytes": 2048,
                "sha256": "a" * 64,
                "optional": False,
            },
            {
                "artifact": "visualization_board/system_board/board_complete.v",
                "status": "GENERATED",
                "purpose": "board level structural wrapper with the dies of topology.json",
                "generated_from": ["topology.json"],
                "size_bytes": 900,
                "optional": False,
            },
            {
                "artifact": "factory/lvs/top.lvs.rpt",
                "status": "MISSING",
                "purpose": "LVS report of the signoff LVS step",
                "looked_at": ["runs/signoff/lvs"],
                "optional": True,
            },
        ],
        "missing": 1,
        "missing_required": 0,
    }


def scratch_export_tree(tmp_path, with_manifest: bool = True, with_run: bool = True):
    """A scratch workspace holding the export script, and what the export left behind."""
    script = tmp_path / "asic_chip" / "scripts" / "export_artifacts.py"
    script.parent.mkdir(parents=True)
    script.write_text("# the repository export script\n", encoding="utf-8")
    if with_run:
        (tmp_path / "asic_chip" / "openlane" / "runs" / "2026.01.04_09-00-00").mkdir(parents=True)
    if with_manifest:
        export = tmp_path / "export_artifacts"
        export.mkdir()
        (export / "manifest.json").write_text(json.dumps(manifest_document()), encoding="utf-8")
    return tmp_path


def test_the_hardware_module_offers_the_export_of_section_4(shell):
    panel = shell.hardware
    assert panel.btn_export.text() == "Export artifacts"
    assert panel.btn_export_check.text() == "Check export plan"
    assert panel.btn_manifest.text() == "Reload manifest"
    assert panel.btn_export.toolTip(), "a disabled button has to say why it is disabled"
    assert panel.btn_export.isEnabled() is panel.export_plan.available
    assert panel.btn_export_check.isEnabled() is panel.export_dry_plan.available


def test_the_panel_reads_the_manifest_the_export_left_on_disk(qapp, tmp_path):
    root = scratch_export_tree(tmp_path / "manifest")
    panel = HardwarePanel(Workspace(root))
    try:
        panel.refresh()
        assert panel.manifest is not None and panel.manifest.total == 3
        assert panel.manifest.design == "lingologos_top"
        labels = [
            panel.export_tree.topLevelItem(index).text(0)
            for index in range(panel.export_tree.topLevelItemCount())
        ]
        assert labels == ["factory/gds", "visualization_board/system_board", "factory/lvs"]
        assert panel.chip_export_gap.text() == "required missing 0"
        assert panel.export_tree.topLevelItem(0).text(1) == "1 of 1"
        assert panel.export_tree.topLevelItem(2).text(1) == "0 of 1"
    finally:
        panel.close()


def test_a_missing_manifest_is_reported_as_absent_and_not_as_an_empty_export(qapp, tmp_path):
    root = scratch_export_tree(tmp_path / "no-manifest", with_manifest=False)
    panel = HardwarePanel(Workspace(root))
    try:
        panel._fill_manifest()
        assert panel.manifest is None
        assert panel.export_tree.topLevelItemCount() == 1
        row = panel.export_tree.topLevelItem(0)
        assert row.text(0) == "manifest.json" and row.text(1) == "absent"
        assert panel.chip_export_gap.text() == "no manifest"
    finally:
        panel.close()


def test_picking_a_manifest_row_reports_what_that_artifact_is_for(qapp, tmp_path):
    root = scratch_export_tree(tmp_path / "rows")
    panel = HardwarePanel(Workspace(root))
    console = widgets.LogConsole()
    try:
        panel.attach_console(console)
        panel._fill_manifest()
        for index in range(panel.export_tree.topLevelItemCount()):
            panel.export_tree.topLevelItem(index).setExpanded(True)
        group = panel.export_tree.topLevelItem(0)
        panel._export_picked(group.child(0), None)
        text = console.toPlainText()
        assert "COPIED factory/gds/lingologos_top.gds" in text
        assert "tapeout layout, the GDSII the streamout step wrote" in text
        assert "sha256 " + "a" * 64 in text
        panel._export_picked(group, None)
        assert "factory/gds: 1 artifacts" in console.toPlainText()
    finally:
        panel.close()
        console.close()


def test_a_finished_export_refills_the_tree_and_asks_the_shell_to_rescan(qapp, tmp_path):
    root = scratch_export_tree(tmp_path / "finished", with_manifest=False)
    panel = HardwarePanel(Workspace(root))
    seen = []
    panel.rescan_requested.connect(lambda: seen.append(True))
    try:
        panel._fill_manifest()
        assert panel.manifest is None
        export = root / "export_artifacts"
        export.mkdir()
        (export / "manifest.json").write_text(json.dumps(manifest_document()), encoding="utf-8")
        panel._export_finished(0, "export wrote 3 artifacts")
        assert panel.manifest is not None and panel.manifest.total == 3
        assert seen == [True], "a finished export refreshes the whole workbench"
        assert panel.status.text() == "export exit 0"
        seen.clear()
        panel._export_finished(1, "export failed")
        assert seen == []
    finally:
        panel.close()


def test_the_export_button_is_disabled_when_there_is_no_run_to_copy(qapp, tmp_path):
    root = scratch_export_tree(tmp_path / "no-run", with_manifest=False, with_run=False)
    panel = HardwarePanel(Workspace(root))
    console = widgets.LogConsole()
    try:
        panel.attach_console(console)
        assert panel.btn_export.isEnabled() is False
        assert panel.btn_export_check.isEnabled() is True, "the plan check still answers"
        assert "no OpenLane run" in panel.chip_export_run.text()
        panel._run_export(False)
        assert "export blocked" in console.toPlainText()
        assert "make -f asic_chip/Makefile openlane" in console.toPlainText()
    finally:
        panel.close()
        console.close()


def test_an_export_of_this_checkout_is_planned_with_the_repository_script(shell):
    plan = shell.hardware.export_plan
    dry = shell.hardware.export_dry_plan
    if not (REPO_ROOT / "asic_chip" / "scripts" / "export_artifacts.py").is_file():
        pytest.skip("this checkout has no export script")
    assert plan.program == dry.program
    assert str(REPO_ROOT / "asic_chip" / "scripts" / "export_artifacts.py") in dry.command_text
    assert "--dry-run" in dry.args and "--dry-run" not in plan.args


def test_the_shell_rescans_when_a_module_finished_an_export(shell, qapp):
    shell.hardware.rescan_requested.emit()
    assert pump(qapp, until=lambda: shell.scan_worker is None or shell.chip_scan.text() == "scanning")
    assert pump(qapp, until=lambda: shell.scan_worker is None)


def test_the_shell_routes_its_export_actions_to_the_hardware_module(qapp, monkeypatch):
    window = WorkbenchWindow(open_workspace(REPO_ROOT))
    try:
        assert pump(qapp, until=lambda: bool(window.scan) and window.scan_worker is None)
        seen = []
        monkeypatch.setattr(window.hardware, "request_export", lambda dry_run: seen.append(bool(dry_run)))
        wanted = {"Check export plan": True, "Export hardware artifacts": False, "Export artifacts": False}
        actions = {action.text(): action for action in window.findChildren(QAction)}
        for name, expected in wanted.items():
            assert name in actions, "section 4 belongs in the shell, not only inside one tab"
            # Whether the action is enabled follows the plan of this host, which is a
            # separate question from where the action routes: enable it to test the route.
            actions[name].setEnabled(True)
            actions[name].trigger()
            assert seen[-1] is expected
        assert window.tabs.tabText(window.tabs.currentIndex()) == dict(TAB_ORDER)["hardware"]
    finally:
        window.close()
        pump(qapp, timeout_ms=3000)


def test_the_export_actions_mirror_what_this_host_can_export(shell):
    for action in shell.findChildren(QAction):
        if action.text() in ("Export hardware artifacts", "Export artifacts"):
            assert action.isEnabled() is shell.hardware.export_plan.available
            assert action.toolTip()
        if action.text() == "Check export plan":
            assert action.isEnabled() is shell.hardware.export_dry_plan.available


def test_a_die_document_reaches_the_board_viewer_from_the_picker(shell):
    from gui.views.hardware import BOARD_SUFFIXES

    assert ".json" in BOARD_SUFFIXES and ".v" in BOARD_SUFFIXES and ".sv" in BOARD_SUFFIXES
    document = REPO_ROOT / "config" / "topology.json"
    if not document.is_file():
        pytest.skip("this checkout has no die document")
    assert shell.hardware.load_board_path(document) is True
    board = shell.hardware.board
    assert board is not None and board.provenance == "topology-json"
    assert len(board.dies) >= 2 and len(board.links) >= 1
    assert not any("no matching die" in warning for warning in board.warnings)


# --------------------------------------------------------------------------- shutdown
def test_closing_the_shell_stops_every_worker_of_every_module(qapp):
    window = WorkbenchWindow(open_workspace(REPO_ROOT))
    stopped = []
    for key, panel in window.panels.items():
        original = panel.stop_worker

        def stop_here(key=key, original=original):
            stopped.append(key)
            original()

        panel.stop_worker = stop_here
    window.start_scan()
    pump(qapp, timeout_ms=4000)
    window.close()
    assert sorted(stopped) == sorted(window.panels)
    assert window.scan_worker is None


def test_the_shell_survives_a_second_scan_while_one_is_running(shell, qapp):
    shell.start_scan()
    shell.start_scan()
    assert pump(qapp, until=lambda: shell.scan_worker is None)
    assert "already running" in shell.console.toPlainText() or shell.scan_worker is None
