"""Lllayer orchestrator panel: the animated stream and the per node model assignment.

Two feeds drive the same view. A recorded session transcript from ``runtime_state`` is
replayed with the spacing its timestamps say, which makes the animation and the canvas
link pulses reproducible on a machine with no model installed. A live session started
from this panel pushes its own stdout and JSON lines through the very same path, and the
transcript the run leaves behind is reloaded so the recorded history matches what ran.

Model names are not invented here. They come from the configuration files when those
declare them; when they do not, as in this repository, the profile list is GUI owned and
lives in ``config/models.json``, which nothing outside the workbench reads.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QBrush
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QSplitter,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from gui.model.stream import (
    KIND_FAULT,
    StreamEvent,
    assignment_path,
    live_command,
    load_assignments,
    model_catalog,
    save_assignments,
    transcript_summary,
)
from gui.model.topology import TopologyGraph, load_graph_file
from gui.workers import SessionRunner, TranscriptReplayer, python_path_entries, verdict_of_line

from .. import theme, widgets
from ..paths import Workspace, open_workspace

TRANSCRIPT_SUFFIXES = (".json",)
TRANSCRIPT_ROLES = ("transcript",)
NO_MODEL = "(no model assigned)"
ROLE_CHOICES = ("llm", "prover", "router", "embedder", "critic", "other")
CONSOLIDATE_CHOICES = (("", "none"), ("triad", "triad"), ("unified", "unified"))
SPEED_CHOICES = ((0.25, "0.25 x"), (0.5, "0.5 x"), (1.0, "1 x"), (2.0, "2 x"), (4.0, "4 x"))
STREAM_COLUMNS = ("turn", "stamp", "channel", "solver", "routing", "payload")
NODE_COLUMNS = ("node", "kind", "assigned model")
_TONE_BRUSH: Dict[str, QBrush] = {}


def _brush(key: str) -> QBrush:
    """Cached themed brush for stream rows, so each row does not allocate one."""
    brush = _TONE_BRUSH.get(key)
    if brush is None:
        brush = QBrush(theme.color(key))
        _TONE_BRUSH[key] = brush
    return brush


def _tone_of(event: StreamEvent) -> str:
    """Theme key for a stream row: faults amber, solver verdicts by their meaning."""
    if event.kind == KIND_FAULT:
        return "warn"
    tone = event.tone()
    return tone if tone in ("pass", "fail", "warn") else "text"


def transcript_summary_for(replayer: TranscriptReplayer) -> Dict[str, Any]:
    """Summarise what the replayer holds, adding the participants of the session."""
    events = replayer.events
    summary = transcript_summary(events)
    participants: List[str] = []
    for event in events:
        for name in (event.sender, event.recipient):
            if name and name not in participants:
                participants.append(name)
    summary["participants"] = participants
    return summary


# ---------------------------------------------------------------------------
# Stream table
# ---------------------------------------------------------------------------
class StreamTable(QTreeWidget):
    """Timestamped, animated stream of envelopes: one row per delivered event."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setHeaderLabels(list(STREAM_COLUMNS))
        self.setRootIsDecorated(False)
        self.setAlternatingRowColors(True)
        self.setSelectionBehavior(QTreeWidget.SelectionBehavior.SelectRows)
        self.setFont(theme.mono_font(9.5))
        self.header().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.header().setSectionResizeMode(5, QHeaderView.ResizeMode.Stretch)
        self.setColumnWidth(0, 56)
        self.setColumnWidth(1, 110)
        self.setColumnWidth(2, 220)
        self.setColumnWidth(3, 80)
        self.setColumnWidth(4, 80)
        self._delivered = 0

    def append_event(self, event: StreamEvent) -> QTreeWidgetItem:
        """Add one row for an event and scroll to it, which is what makes it a stream."""
        routing = event.envelope.get("routing_authorized")
        row = QTreeWidgetItem(
            [
                f"{event.turn:03d}",
                event.stamp_text or f"turn {event.turn:03d}",
                event.channel,
                event.verdict(),
                "" if routing is None else ("authorized" if routing else "BLOCKED"),
                event.preview(200),
            ]
        )
        brush = _brush(_tone_of(event))
        for column in range(len(STREAM_COLUMNS)):
            row.setForeground(column, brush)
        row.setData(0, Qt.ItemDataRole.UserRole, event)
        self.addTopLevelItem(row)
        self._delivered += 1
        self.scrollToBottom()
        return row

    def clear_stream(self) -> None:
        self.clear()
        self._delivered = 0

    @property
    def delivered(self) -> int:
        return self._delivered

    def event_of(self, item: Optional[QTreeWidgetItem]) -> Optional[StreamEvent]:
        if item is None:
            return None
        payload = item.data(0, Qt.ItemDataRole.UserRole)
        return payload if isinstance(payload, StreamEvent) else None


# ---------------------------------------------------------------------------
# Panel
# ---------------------------------------------------------------------------
class OrchestratorPanel(QWidget):
    """The Lllayer module: replay recorded sessions, run live ones, assign models per node."""

    stream_event = pyqtSignal(object)
    edge_pinged = pyqtSignal(str, str)
    topology_requested = pyqtSignal(str)

    def __init__(self, workspace: Optional[Workspace] = None, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.workspace = workspace or open_workspace(Path.cwd())
        self.console: Optional[widgets.LogConsole] = None
        self.transcript_path: Optional[Path] = None
        self.profiles: List[Dict[str, str]] = []
        self.assignments: Dict[str, str] = {}
        self.combos: Dict[str, QComboBox] = {}
        self.node_rows: Dict[str, QTreeWidgetItem] = {}
        self.dirty = False
        self.assignment_file = assignment_path(self.workspace.root)
        self.replayer = TranscriptReplayer(self)
        self.runner = SessionRunner(self.workspace.root, self)
        self._build()
        self._wire()
        self._load_assignment_file()

    # --- layout --------------------------------------------------------------
    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.addLayout(self._source_row())

        splitter = QSplitter(Qt.Orientation.Vertical)
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addLayout(self._transport_row())
        self.stream = StreamTable()
        left_layout.addWidget(self.stream, 1)
        splitter.addWidget(left)

        side = QWidget()
        side_layout = QVBoxLayout(side)
        side_layout.setContentsMargins(0, 0, 0, 0)
        assignment_card = self._assignment_card()
        session_card = self._session_card()
        widgets.share_column(side_layout, assignment_card, 3)
        widgets.share_column(side_layout, session_card, 2)
        splitter.addWidget(side)
        splitter.setStretchFactor(0, 5)
        splitter.setStretchFactor(1, 2)
        root.addWidget(splitter, 1)

        self.chip_state = widgets.Chip("replay idle")
        self.chip_events = widgets.Chip("events -")
        self.chip_turns = widgets.Chip("turns -")
        self.chip_channels = widgets.Chip("channels -")
        self.chip_verdicts = widgets.Chip("solver verdicts -")
        self.chip_nodes = widgets.Chip("nodes -")
        self.chip_session = widgets.Chip("session idle")
        bar = widgets.ToolRow()
        for chip in (self.chip_state, self.chip_events, self.chip_turns, self.chip_channels, self.chip_verdicts, self.chip_nodes, self.chip_session):
            bar.add(chip)
        bar.add_spacer()
        bar.add_label("every event lands in the stream table; solver and fault lines also reach the log console")
        root.addWidget(bar)

    def _source_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(6)
        self.picker = widgets.PathPicker(self.workspace.runtime_state_dir, TRANSCRIPT_SUFFIXES, "Recorded session transcript")
        self.picker.chosen.connect(self.load_path)
        row.addWidget(self.picker, 3)
        self.transcripts = QComboBox()
        self.transcripts.setMinimumWidth(320)
        self.transcripts.currentIndexChanged.connect(self._transcript_picked)
        row.addWidget(self.transcripts, 2)
        self.btn_open_canvas = QPushButton("Open its topology on the canvas")
        self.btn_open_canvas.clicked.connect(self._open_topology)
        row.addWidget(self.btn_open_canvas)
        return row

    def _transport_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(6)
        self.btn_play = QPushButton("Play")
        self.btn_play.setObjectName("Primary")
        self.btn_play.clicked.connect(self._play)
        row.addWidget(self.btn_play)
        self.btn_pause = QPushButton("Pause")
        self.btn_pause.clicked.connect(lambda: self.replayer.pause())
        row.addWidget(self.btn_pause)
        self.btn_step = QPushButton("Step")
        self.btn_step.clicked.connect(lambda: self.replayer.step(1))
        row.addWidget(self.btn_step)
        self.btn_end = QPushButton("Skip to end")
        self.btn_end.clicked.connect(self._skip_to_end)
        row.addWidget(self.btn_end)
        self.btn_restart = QPushButton("Restart")
        self.btn_restart.clicked.connect(self._restart)
        row.addWidget(self.btn_restart)
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setObjectName("Danger")
        self.btn_stop.clicked.connect(lambda: self.replayer.stop())
        row.addWidget(self.btn_stop)
        self.btn_clear = QPushButton("Clear stream")
        self.btn_clear.clicked.connect(self._clear_stream)
        row.addWidget(self.btn_clear)
        row.addSpacing(8)
        row.addWidget(QLabel("speed"))
        self.speed = QComboBox()
        for value, label in SPEED_CHOICES:
            self.speed.addItem(label, value)
        self.speed.setCurrentIndex(2)
        self.speed.currentIndexChanged.connect(self._speed_changed)
        row.addWidget(self.speed)
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setValue(0)
        self.progress.setMaximumWidth(180)
        row.addWidget(self.progress)
        self.progress_label = QLabel("no transcript loaded")
        self.progress_label.setObjectName("MutedLabel")
        row.addWidget(self.progress_label, 1)
        return row

    def _assignment_card(self) -> widgets.Card:
        card = widgets.Card("Model assigned to each node")
        self.node_tree = QTreeWidget()
        self.node_tree.setHeaderLabels(list(NODE_COLUMNS))
        self.node_tree.setRootIsDecorated(False)
        self.node_tree.setAlternatingRowColors(True)
        self.node_tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.node_tree.setColumnWidth(1, 130)
        self.node_tree.setColumnWidth(2, 230)
        self.node_tree.setMinimumHeight(150)
        card.body.addWidget(self.node_tree, 0, 0, 1, 3)
        card.set_body_stretch(0, 3)

        tools = widgets.ToolRow()
        self.btn_reload_assignment = QPushButton("Reload file")
        self.btn_reload_assignment.clicked.connect(self._reload_assignment_file)
        tools.add(self.btn_reload_assignment)
        self.btn_save_assignment = QPushButton("Save assignment")
        self.btn_save_assignment.setObjectName("Primary")
        self.btn_save_assignment.clicked.connect(self._save_assignments)
        tools.add(self.btn_save_assignment)
        self.btn_unassign = QPushButton("Clear row")
        self.btn_unassign.clicked.connect(self._clear_row_assignment)
        tools.add(self.btn_unassign)
        tools.add_spacer()
        self.assignment_status = widgets.Chip("no assignment file")
        tools.add(self.assignment_status)
        card.body.addWidget(tools, 1, 0, 1, 3)

        profile_row = QHBoxLayout()
        profile_row.setSpacing(6)
        self.profile_name = QLineEdit()
        self.profile_name.setPlaceholderText("new model profile, for example Llama-3-8B-Instruct")
        profile_row.addWidget(self.profile_name, 3)
        self.profile_role = QComboBox()
        self.profile_role.addItems(list(ROLE_CHOICES))
        profile_row.addWidget(self.profile_role, 1)
        self.btn_add_profile = QPushButton("Add profile")
        self.btn_add_profile.clicked.connect(self._add_profile)
        profile_row.addWidget(self.btn_add_profile)
        self.btn_remove_profile = QPushButton("Remove selected profile")
        self.btn_remove_profile.clicked.connect(self._remove_profile)
        profile_row.addWidget(self.btn_remove_profile)
        card.body.addLayout(profile_row, 2, 0, 1, 3)

        self.profiles_tree = QTreeWidget()
        self.profiles_tree.setHeaderLabels(["profile", "role", "origin"])
        self.profiles_tree.setRootIsDecorated(False)
        self.profiles_tree.setAlternatingRowColors(True)
        self.profiles_tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.profiles_tree.setColumnWidth(1, 90)
        self.profiles_tree.setColumnWidth(2, 130)
        self.profiles_tree.setMinimumHeight(100)
        card.body.addWidget(self.profiles_tree, 3, 0, 1, 3)
        card.set_body_stretch(3, 2)

        self.catalog_note = QLabel("")
        self.catalog_note.setObjectName("MutedLabel")
        self.catalog_note.setWordWrap(True)
        card.body.addWidget(self.catalog_note, 4, 0, 1, 3)
        return card

    def _session_card(self) -> widgets.Card:
        card = widgets.Card("Live orchestrator session")
        config_row = QHBoxLayout()
        config_row.setSpacing(6)
        config_row.addWidget(QLabel("topology"))
        self.configs = QComboBox()
        self.configs.setMinimumWidth(240)
        self.configs.currentIndexChanged.connect(self._config_picked)
        config_row.addWidget(self.configs, 3)
        config_row.addWidget(QLabel("consolidate"))
        self.consolidate = QComboBox()
        for value, label in CONSOLIDATE_CHOICES:
            self.consolidate.addItem(label, value)
        config_row.addWidget(self.consolidate, 1)
        self.json_output = QCheckBox("json result")
        self.json_output.setChecked(True)
        self.json_output.toggled.connect(lambda _state: self._update_command_note())
        config_row.addWidget(self.json_output)
        card.body.addLayout(config_row, 0, 0, 1, 3)

        tools = widgets.ToolRow()
        self.btn_run = QPushButton("Run session now")
        self.btn_run.setObjectName("Primary")
        self.btn_run.clicked.connect(self._run_session)
        tools.add(self.btn_run)
        self.btn_stop_session = QPushButton("Stop session")
        self.btn_stop_session.setObjectName("Danger")
        self.btn_stop_session.clicked.connect(lambda: self.runner.stop())
        tools.add(self.btn_stop_session)
        tools.add_spacer()
        self.session_state = widgets.Chip("idle")
        tools.add(self.session_state)
        card.body.addWidget(tools, 1, 0, 1, 3)

        self.command_note = QLabel("")
        self.command_note.setObjectName("MutedLabel")
        self.command_note.setWordWrap(True)
        card.body.addWidget(self.command_note, 2, 0, 1, 3)
        return card

    # --- wiring --------------------------------------------------------------
    def _wire(self) -> None:
        self.replayer.event_delivered.connect(self._on_event)
        self.replayer.progressed.connect(self._on_progress)
        self.replayer.state_changed.connect(self._on_replay_state)
        self.runner.event_arrived.connect(self._on_event)
        self.runner.text_arrived.connect(self._on_session_text)
        self.runner.state_changed.connect(self._on_session_state)
        self.runner.run_finished.connect(self._on_session_finished)
        self.stream.itemSelectionChanged.connect(self._stream_selected)
        self.stream.itemDoubleClicked.connect(self._stream_inspected)

    # --- console and refresh -------------------------------------------------
    def attach_console(self, console: widgets.LogConsole) -> None:
        self.console = console

    def _say(self, message: str, level: str = "info") -> None:
        if self.console is not None:
            self.console.log(message, level)

    def refresh(self, scan: Any = None) -> None:
        """Take transcripts and topologies from the workspace scan, then offer them."""
        source = scan if isinstance(scan, dict) else self.workspace.scan
        self._fill_transcripts(source)
        self._fill_configs(source)
        self._refresh_catalog(source)
        if self.transcript_path is None:
            candidate = self.workspace.newest("transcript")
            if candidate is not None:
                self.load_path(candidate.path)

    def _fill_transcripts(self, source: Dict[str, Any]) -> None:
        previous = self.transcripts.currentData()
        self.transcripts.blockSignals(True)
        self.transcripts.clear()
        for role in TRANSCRIPT_ROLES:
            for artifact in source.get(role, []):
                self.transcripts.addItem(self.workspace.relative(artifact.path), str(artifact.path))
        if self.transcripts.count() == 0:
            self.transcripts.addItem("no transcript in runtime_state", "")
        index = self.transcripts.findData(previous) if previous else -1
        if index >= 0:
            self.transcripts.setCurrentIndex(index)
        self.transcripts.blockSignals(False)

    def _fill_configs(self, source: Dict[str, Any]) -> None:
        previous = self.configs.currentData()
        self.configs.blockSignals(True)
        self.configs.clear()
        for artifact in source.get("topology", []):
            self.configs.addItem(self.workspace.relative(artifact.path), str(artifact.path))
        if self.configs.count() == 0:
            self.configs.addItem("no topology configuration in the scan", "")
        index = self.configs.findData(previous) if previous else -1
        if index >= 0:
            self.configs.setCurrentIndex(index)
        self.configs.blockSignals(False)
        self._update_command_note()
        self._nodes_of_selected_config()

    def _transcript_picked(self, _index: int) -> None:
        raw = self.transcripts.currentData()
        if raw:
            self.load_path(raw)

    def _config_picked(self, _index: int) -> None:
        self._update_command_note()
        self._nodes_of_selected_config()

    def _open_topology(self) -> None:
        raw = self.configs.currentData()
        if not raw:
            self._say("no topology configuration is selected, so there is nothing to open on the canvas", "warn")
            return
        self.topology_requested.emit(str(raw))

    # --- transcript loading and replay ---------------------------------------
    def load_path(self, raw: Any) -> bool:
        path = Path(str(raw))
        if not path.is_file():
            self._say(f"no transcript at {path}", "error")
            return False
        try:
            self.replayer.load(path)
        except (OSError, ValueError) as exc:
            self._say(f"cannot read transcript {path.name}: {exc}", "error")
            return False
        self.transcript_path = path
        self.picker.set_path(path)
        summary = transcript_summary_for(self.replayer)
        self._apply_summary(summary, path)
        self._say(f"loaded {path.name}: {summary['events']} events over {summary['turns']} turns on {len(summary['channels'])} channels")
        for channel, count in sorted(summary["channels"].items()):
            self._say(f"channel {channel}: {count} messages")
        for verdict, count in sorted(summary["verdicts"].items()):
            self._say(f"solver verdict {verdict}: {count}", "smt")
        self.ensure_nodes(summary["participants"])
        return True

    def _apply_summary(self, summary: Dict[str, Any], path: Path) -> None:
        total = int(summary["events"])
        self.chip_events.setText(f"events 0 / {total}")
        self.chip_turns.setText(f"turns {summary['turns']}")
        self.chip_channels.setText(f"channels {len(summary['channels'])}")
        verdicts = summary.get("verdicts") or {}
        label = ", ".join(f"{key} {value}" for key, value in sorted(verdicts.items())) if verdicts else "none recorded"
        self.chip_verdicts.setText(f"solver verdicts {label}")
        self.chip_verdicts.set_tone("smt" if verdicts else "")
        self.progress.setRange(0, max(1, total))
        self.progress.setValue(0)
        self.progress_label.setText(f"{path.name}: {total} events ready to replay")

    def play_stream(self) -> None:
        """Replay the loaded transcript from the start, clearing the stream table."""
        self._play()

    def _play(self) -> None:
        if self.replayer.total == 0:
            self._say("the replayer holds no events: load a session transcript first", "warn")
            return
        self.stream.clear_stream()
        self.progress.setValue(0)
        self.replayer.start()

    def _restart(self) -> None:
        self.stream.clear_stream()
        self.replayer.restart()

    def _skip_to_end(self) -> None:
        remaining = self.replayer.total - self.replayer.cursor
        if remaining <= 0:
            self._say("the replay is already at the end of the transcript", "warn")
            return
        self.replayer.step(remaining)

    def _speed_changed(self, _index: int) -> None:
        value = self.speed.currentData()
        if value is not None:
            self.replayer.set_speed(float(value))

    def _clear_stream(self) -> None:
        self.stream.clear_stream()
        self.progress.setValue(0)
        self.progress_label.setText("stream cleared")

    def _on_event(self, event: StreamEvent) -> None:
        self.stream.append_event(event)
        self.stream_event.emit(event)
        self.chip_events.setText(f"events {self.stream.delivered} / {max(1, self.replayer.total)}")
        verdict = event.verdict()
        if verdict:
            self.chip_verdicts.setText(f"solver verdicts {verdict}")
            self.chip_verdicts.set_tone("pass" if verdict.upper() in ("UNSAT", "PASS") else "warn")
        if event.kind == KIND_FAULT or verdict:
            self._say(event.to_line(), "warn" if event.kind == KIND_FAULT else "smt")
        self.edge_pinged.emit(event.sender, event.recipient)

    def _on_progress(self, cursor: int, total: int) -> None:
        self.progress.setRange(0, max(1, total))
        self.progress.setValue(cursor)
        self.progress_label.setText(f"event {cursor} of {total}")

    def _on_replay_state(self, state: str) -> None:
        self.chip_state.setText(f"replay {state}")
        tone = "pass" if state == "finished" else ("warn" if state == "running" else "")
        self.chip_state.set_tone(tone)
        if state == "empty":
            self._say("the replayer holds no events: load a transcript first", "warn")
        if state == "finished":
            self._say("replay finished", "pass")

    def _stream_selected(self) -> None:
        event = self.stream.event_of(self.stream.currentItem())
        if event is not None:
            self.edge_pinged.emit(event.sender, event.recipient)

    def _stream_inspected(self, item: QTreeWidgetItem, _column: int) -> None:
        event = self.stream.event_of(item)
        if event is None:
            return
        keys = ", ".join(sorted(str(key) for key in event.envelope))
        self._say(f"turn {event.turn} {event.channel} kind {event.kind} envelope keys: {keys}")

    # --- nodes and assignment ------------------------------------------------
    def set_graph(self, graph: TopologyGraph) -> None:
        """Take the node ids and kinds of a graph the canvas just loaded."""
        self.ensure_nodes(sorted(graph.nodes))
        for node_id, spec in graph.nodes.items():
            self._set_node_kind(node_id, spec.kind)

    def ensure_nodes(self, node_ids: Iterable[str]) -> int:
        """Make sure every node id has a row with an editable model combo."""
        added = 0
        for node_id in node_ids:
            clean = str(node_id)
            if not clean or clean in self.node_rows:
                continue
            row = QTreeWidgetItem([clean, "", ""])
            self.node_tree.addTopLevelItem(row)
            self.node_rows[clean] = row
            combo = self._make_combo(clean)
            self.node_tree.setItemWidget(row, 2, combo)
            added += 1
        if added:
            self._update_node_chip()
        return added

    def _make_combo(self, node_id: str) -> QComboBox:
        combo = QComboBox()
        combo.setEditable(True)
        combo.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        combo.lineEdit().setPlaceholderText(NO_MODEL)
        self._fill_combo(combo, self.assignments.get(node_id, ""))
        combo.currentTextChanged.connect(lambda text, key=node_id: self._assignment_edited(key, text))
        self.combos[node_id] = combo
        return combo

    def _fill_combo(self, combo: QComboBox, chosen: str) -> None:
        combo.blockSignals(True)
        combo.clear()
        names = [NO_MODEL, *(str(profile.get("name", "")) for profile in self.profiles)]
        for name in dict.fromkeys(item for item in names if item):
            combo.addItem(name)
        combo.setCurrentText(chosen or NO_MODEL)
        combo.blockSignals(False)

    def _refresh_catalog(self, source: Dict[str, Any]) -> None:
        """Read declared model names from the configuration files, never from a fixed list."""
        config_paths: List[Path] = [self.assignment_file]
        for role in ("topology", "board"):
            for artifact in source.get(role, []):
                config_paths.append(artifact.path)
        found = model_catalog(config_paths)
        known = {str(profile.get("name", "")).lower() for profile in self.profiles}
        for entry in found:
            name = str(entry.get("name", ""))
            if name.lower() not in known:
                self.profiles.append({"name": name, "role": str(entry.get("role", "llm")), "origin": str(entry.get("origin", ""))})
                known.add(name.lower())
        self._fill_profiles()
        if found:
            self.catalog_note.setText(f"{len(found)} model names were harvested from the configuration files of this workspace.")
            self._say("model catalog from the configuration files: " + ", ".join(entry["name"] for entry in found))
        else:
            self.catalog_note.setText(
                "No configuration file of this workspace declares a model name, so the profile list is maintained here and stored in config/models.json."
            )
        self._reseed_combos()

    def _fill_profiles(self) -> None:
        self.profiles_tree.clear()
        for profile in self.profiles:
            entry = QTreeWidgetItem(
                [str(profile.get("name", "")), str(profile.get("role", "llm")), str(profile.get("origin", "workbench"))]
            )
            entry.setForeground(0, _brush("data"))
            self.profiles_tree.addTopLevelItem(entry)

    def _reseed_combos(self) -> None:
        for node_id, combo in self.combos.items():
            self._fill_combo(combo, self.assignments.get(node_id, ""))

    def _add_profile(self) -> None:
        name = self.profile_name.text().strip()
        if not name:
            self._say("type a model profile name before adding it", "warn")
            return
        known = {str(profile.get("name", "")).lower() for profile in self.profiles}
        if name.lower() in known:
            self._say(f"profile {name} is already in the list", "warn")
            return
        role = self.profile_role.currentText()
        self.profiles.append({"name": name, "role": role, "origin": "workbench"})
        self.profile_name.clear()
        self._fill_profiles()
        self._reseed_combos()
        self._save_assignments()
        self._say(f"profile {name} added with role {role}")

    def _remove_profile(self) -> None:
        item = self.profiles_tree.currentItem()
        if item is None:
            self._say("select a profile row to remove it", "warn")
            return
        name = item.text(0)
        self.profiles = [profile for profile in self.profiles if str(profile.get("name", "")).lower() != name.lower()]
        self._fill_profiles()
        self._reseed_combos()
        self._save_assignments()
        self._say(f"profile {name} removed from the list", "warn")

    def _assignment_edited(self, node_id: str, text: str) -> None:
        chosen = "" if not text or text == NO_MODEL else text.strip()
        if chosen:
            self.assignments[node_id] = chosen
            known = {str(profile.get("name", "")).lower() for profile in self.profiles}
            if chosen.lower() not in known:
                self.profiles.append({"name": chosen, "role": "llm", "origin": "typed"})
                self._fill_profiles()
                self._reseed_combos()
        else:
            self.assignments.pop(node_id, None)
        self._update_node_chip()
        self._save_assignments()

    def _clear_row_assignment(self) -> None:
        item = self.node_tree.currentItem()
        if item is None:
            self._say("select a node row to clear its assignment", "warn")
            return
        combo = self.combos.get(item.text(0))
        if combo is not None:
            combo.setCurrentText(NO_MODEL)
            self._say(f"assignment cleared for {item.text(0)}", "warn")

    def _set_node_kind(self, node_id: str, kind: str) -> None:
        row = self.node_rows.get(node_id)
        if row is not None:
            row.setText(1, str(kind))

    def _nodes_of_selected_config(self) -> None:
        raw = self.configs.currentData()
        if not raw:
            return
        path = Path(str(raw))
        if not path.is_file():
            return
        try:
            graph = load_graph_file(path)
        except Exception as exc:  # a broken configuration must not take the panel down
            self._say(f"cannot read nodes from {path.name}: {exc}", "error")
            return
        self.set_graph(graph)
        self._say(f"{path.name} declares {len(graph.nodes)} nodes and {len(graph.edges)} links")

    def _update_node_chip(self) -> None:
        self.chip_nodes.setText(f"nodes {len(self.node_rows)} assigned {len(self.assignments)}")
        self.chip_nodes.set_tone("warn" if self.node_rows and not self.assignments else "")

    # --- assignment file -----------------------------------------------------
    def _load_assignment_file(self) -> None:
        document = load_assignments(self.assignment_file)
        self.profiles = [dict(item) for item in document.get("profiles", [])]
        self.assignments = dict(document.get("assignments", {}))
        self._fill_profiles()
        self._reseed_combos()
        self._update_assignment_status()
        self._update_node_chip()

    def _reload_assignment_file(self) -> None:
        self.dirty = False
        self._load_assignment_file()
        self._say(f"reloaded {self.assignment_file.name}")

    def _save_assignments(self) -> None:
        try:
            save_assignments(self.assignment_file, self.profiles, self.assignments)
        except OSError as exc:
            self._say(f"cannot write {self.assignment_file}: {exc}", "error")
            return
        self.dirty = False
        self._update_assignment_status()

    def _update_assignment_status(self) -> None:
        if self.assignment_file.is_file():
            self.assignment_status.setText(f"{self.assignment_file.name}: {len(self.profiles)} profiles / {len(self.assignments)} nodes")
            self.assignment_status.set_tone("pass")
        else:
            self.assignment_status.setText("no assignment file")
            self.assignment_status.set_tone("")

    # --- live session --------------------------------------------------------
    def _extra_tail(self) -> List[str]:
        """Extra CLI arguments of a live run: consolidation plus absolute workspace paths."""
        tail: List[str] = []
        mode = self.consolidate.currentData()
        if mode:
            tail = ["--consolidate", str(mode)]
        return [
            *tail,
            "--workspace",
            str(self.workspace.root),
            "--datasets",
            str(self.workspace.datasets_dir),
            "--runtime-state",
            str(self.workspace.runtime_state_dir),
        ]

    def session_command(self, config: Path) -> Tuple[str, List[str]]:
        """The full command a live run uses, so the note never drifts from reality."""
        program, args = live_command(self.workspace.root, config)
        args = [item for item in args if item != "--json"]
        if self.json_output.isChecked():
            args = [*args, "--json"]
        return program, [*args, *self._extra_tail()]

    def _update_command_note(self) -> None:
        raw = self.configs.currentData()
        if not raw:
            self.command_note.setText("no topology configuration is available, so no live session can be planned")
            return
        if not self.workspace.profile.has_session:
            self.command_note.setText("this workspace profile declares no live session command")
            return
        program, args = self.session_command(Path(str(raw)))
        tail = " ".join([program, *args])
        self.command_note.setText(f"command of record: {tail}   with PYTHONPATH={python_path_entries(self.workspace.root)}")

    def _run_session(self) -> bool:
        raw = self.configs.currentData()
        if not raw:
            self._say("a live session needs a topology configuration: none is selected", "error")
            return False
        if self.runner.running:
            self._say("a live session is already running", "warn")
            return False
        if not self.workspace.profile.has_session:
            self._say("this workspace profile declares no live session command", "error")
            return False
        config = Path(str(raw))
        program, args = self.session_command(config)
        self.stream.clear_stream()
        self._say(f"starting live session: {program} {' '.join(args)}")
        started = self.runner.start(config, extra_args=self._extra_tail(), json_output=self.json_output.isChecked())
        if not started:
            self._say("the live session could not be started", "error")
        return bool(started)

    def _on_session_text(self, line: str, level: str) -> None:
        self._say(line, level)
        verdict = verdict_of_line(line)
        if verdict:
            self.chip_verdicts.setText(f"solver verdicts {verdict}")
            self.chip_verdicts.set_tone("pass" if "PASS" in verdict.upper() else "fail")

    def _on_session_state(self, state: str) -> None:
        self.chip_session.setText(f"session {state}")
        self.session_state.setText(state)
        tone = "pass" if state == "finished" else ("fail" if state == "failed" else ("warn" if state == "running" else ""))
        self.chip_session.set_tone(tone)
        self.session_state.set_tone(tone)

    def _on_session_finished(self, code: int, summary: str) -> None:
        level = "pass" if code == 0 else "error"
        self._say(f"live session finished: {summary or ('exit code ' + str(code))}", level)
        candidate = self.workspace.newest("transcript")
        if candidate is not None:
            self._say(f"reloading the transcript the session wrote: {candidate.name}")
            self.load_path(candidate.path)
            self._say("press Play to animate the session that was just recorded")

    # --- stream hook from the shell -----------------------------------------
    def handle_stream_event(self, event: Any) -> None:
        """Ignore a broadcast this panel itself emitted."""
        del event

    def stop_worker(self) -> None:
        self.replayer.stop()
        if self.runner.running:
            self.runner.stop()


__all__ = ["OrchestratorPanel", "StreamTable", "transcript_summary_for"]
