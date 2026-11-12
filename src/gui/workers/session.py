"""Stream session workers: paced transcript replay and live orchestrator sessions.

The Lllayer panel needs a stream that behaves like a stream: events arrive one after the
other with the spacing they were recorded with, and the canvas pulses the link each event
travelled over. Two workers provide that, and both are optional: if the source is missing
the panel simply reports it in the console and stays usable.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from PyQt6.QtCore import QObject, QTimer, pyqtSignal

from gui.model.stream import (
    KIND_SYSTEM,
    StreamEvent,
    load_transcript,
    live_command,
    replay,
    transcript_summary,
)

from gui.model.profile import PROFILE_FILE, load_profile

from .runner import CommandWorker, ToolPlan, classify_line, python_path_for, verdict_of_line


class TranscriptReplayer(QObject):
    """Deliver recorded events at the pace their timestamps say, or at a set cadence."""

    event_delivered = pyqtSignal(object)
    progressed = pyqtSignal(int, int)
    state_changed = pyqtSignal(str)

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._events: List[StreamEvent] = []
        self._cursor = 0
        self._speed = 1.0
        self._state = "idle"
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._deliver_next)

    @property
    def state(self) -> str:
        return self._state

    @property
    def events(self) -> List[StreamEvent]:
        """Copy of the events currently held for replay."""
        return list(self._events)

    @property
    def cursor(self) -> int:
        return self._cursor

    @property
    def total(self) -> int:
        return len(self._events)

    def set_speed(self, speed: float) -> None:
        self._speed = max(0.25, float(speed))

    def load(self, path: Path, max_events: int = 400) -> Dict[str, Any]:
        """Read one transcript file and hold its events for replay."""
        self._timer.stop()
        self._events = load_transcript(Path(path), max_events=max_events)
        self._cursor = 0
        self._set_state("loaded")
        return transcript_summary(self._events)

    def load_events(self, events: Iterable[StreamEvent]) -> None:
        self._timer.stop()
        self._events = list(events)
        self._cursor = 0
        self._set_state("loaded")

    def start(self) -> None:
        if not self._events:
            self._set_state("empty")
            return
        if self._state == "finished" or self._cursor >= len(self._events):
            self._cursor = 0
        self._set_state("running")
        self._schedule(0.0)

    def pause(self) -> None:
        if self._state == "running":
            self._timer.stop()
            self._set_state("paused")

    def stop(self) -> None:
        self._timer.stop()
        self._set_state("stopped")

    def step(self, count: int = 1) -> None:
        """Deliver the next events at once, ignoring the recorded spacing."""
        for _ in range(max(1, int(count))):
            if self._cursor >= len(self._events):
                break
            self._emit_cursor()

    def restart(self) -> None:
        self._cursor = 0
        self.start()

    def _schedule(self, delay_seconds: float) -> None:
        delay = max(0.0, float(delay_seconds)) / self._speed
        self._timer.start(int(delay * 1000.0) if delay > 0 else 0)

    def _deliver_next(self) -> None:
        if self._cursor >= len(self._events):
            self._set_state("finished")
            return
        _event, delay = self._pending_step()
        self._emit_cursor()
        if self._cursor >= len(self._events):
            self._set_state("finished")
            return
        self._schedule(delay)

    def _pending_step(self):
        """The ``(event, delay)`` pair the model prescribes for the cursor position."""
        event = self._events[self._cursor]
        for pair in replay([event], speed=self._speed):
            return pair
        return event, 0.0

    def _emit_cursor(self) -> None:
        event = self._events[self._cursor]
        self._cursor += 1
        self.event_delivered.emit(event)
        self.progressed.emit(self._cursor, len(self._events))

    def _set_state(self, state: str) -> None:
        self._state = state
        self.state_changed.emit(state)


class SessionRunner(QObject):
    """Run a real orchestrator session and turn its JSON lines into stream events.

    The command is the one the workspace profile declares, so the workbench is a viewer
    of a genuine session rather than a simulation of one. When the run cannot start, the
    failure reaches the console through the same signal path.
    """

    event_arrived = pyqtSignal(object)
    text_arrived = pyqtSignal(str, str)
    state_changed = pyqtSignal(str)
    run_finished = pyqtSignal(int, str)

    def __init__(self, root: Path, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._root = Path(root)
        self._worker: Optional[CommandWorker] = None
        self._turn = 0

    @property
    def running(self) -> bool:
        return self._worker is not None and self._worker.isRunning()

    def start(self, config: Path, extra_args: Optional[List[str]] = None, json_output: bool = True) -> bool:
        """Start a live session; returns False when the command could not be planned.

        ``json_output`` mirrors the ``--json`` flag of the CLI: with it the run prints the
        machine readable result, without it the human readable verdict line the console
        colouring already knows how to read.
        """
        if self.running:
            return False
        program, args = live_command(self._root, Path(config))
        if not args:
            self.text_arrived.emit(
                f"this workspace profile declares no live session command: set session_module in {PROFILE_FILE}",
                "error",
            )
            return False
        if not json_output:
            args = [item for item in args if item != "--json"]
        if extra_args:
            args = [*args, *extra_args]
        plan = ToolPlan(
            label="orchestrator session",
            program=program,
            args=args,
            cwd=str(self._root),
            env={"PYTHONPATH": python_path_for(self._root)},
            hint=f"python -m {load_profile(self._root).session_module} --config config/topology.json --json",
        )
        self._worker = CommandWorker(plan, self)
        self._worker.line_ready.connect(self._on_line)
        self._worker.ended.connect(self._on_ended)
        self._worker.started.connect(lambda text: self.text_arrived.emit(f"$ {text}", "info"))
        self._worker.finished.connect(self._worker.deleteLater)
        self._worker.start()
        self.state_changed.emit("running")
        return True

    def stop(self) -> None:
        if self._worker is not None:
            self._worker.stop()
            self.state_changed.emit("stopping")

    def _on_line(self, line: str, level: str) -> None:
        self.text_arrived.emit(line, level)
        event = self._to_event(line)
        if event is not None:
            self.event_arrived.emit(event)

    def _on_ended(self, code: int, summary: str) -> None:
        self.state_changed.emit("finished" if code == 0 else "failed")
        self.run_finished.emit(code, summary)

    def _to_event(self, line: str) -> Optional[StreamEvent]:
        stripped = line.strip()
        if not stripped.startswith("{"):
            return None
        try:
            record = json.loads(stripped)
        except ValueError:
            return None
        if not isinstance(record, dict):
            return None
        self._turn += 1
        kind = KIND_SYSTEM
        verdict = verdict_of_line(line)
        if verdict:
            kind = "control"
        return StreamEvent(
            turn=int(record.get("turn", self._turn)),
            sender=str(record.get("sender", record.get("from", "orchestrator"))),
            recipient=str(record.get("recipient", record.get("to", "console"))),
            envelope=record,
            kind=kind,
            offset_seconds=0.0,
            stamp_text=str(record.get("timestamp", f"live turn {self._turn:03d}")),
        )


__all__ = ["SessionRunner", "TranscriptReplayer"]
