"""Stream model for the orchestrator panel: events, pacing, model catalog.

Two feeds share one event shape. A recorded transcript from ``runtime_state`` is
replayed with real timestamps, which is what makes the animated stream and the
canvas pulses reproducible on a machine without any model installed. A live session,
when one is launched from the panel, pushes the very same events as they arrive.

The model catalog is also read from disk rather than hard coded: whatever the
orchestrator topology and the local configuration files declare becomes a choice in
the per node assignment UI, so a topology with a Prover SMT node or a new Llama
checkpoint shows the new entry without a code change.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .profile import load_profile

KIND_MESSAGE = "message"
KIND_CONTROL = "control"
KIND_FAULT = "fault"
KIND_SYSTEM = "system"

_LEVEL_TONE = {"info": "info", "warn": "warn", "error": "fail", "ok": "pass", "smt": "smt"}


@dataclass
class StreamEvent:
    """One envelope in the stream: who speaks, to whom, with what payload."""

    turn: int
    sender: str
    recipient: str
    envelope: Dict[str, Any] = field(default_factory=dict)
    kind: str = KIND_MESSAGE
    offset_seconds: float = 0.0
    stamp_text: str = ""

    @property
    def channel(self) -> str:
        return f"{self.sender}->{self.recipient}"

    def preview(self, width: int = 160) -> str:
        body = self.envelope.get("payload", self.envelope)
        if isinstance(body, dict):
            # A message payload usually carries the sentence the node actually sent, so the
            # stream reads as a conversation instead of as a JSON dump.
            for key in ("naturalLanguage", "message", "text", "statement"):
                value = body.get(key)
                if isinstance(value, str) and value:
                    body = value
                    break
        text = body if isinstance(body, str) else json.dumps(body, ensure_ascii=False, sort_keys=True)
        flat = " ".join(str(text).split())
        if len(flat) <= width:
            return flat
        return flat[: max(0, width - 3)] + "..."

    def verdict(self) -> str:
        for key in ("status", "verdict"):
            value = self.envelope.get(key)
            if isinstance(value, str) and value:
                return value
        solver = self.envelope.get("smt_result")
        if isinstance(solver, str) and solver:
            return solver
        if isinstance(solver, dict):
            for key in ("status", "z3Status", "verdict"):
                value = solver.get(key)
                if isinstance(value, str) and value:
                    return value
        payload = self.envelope.get("payload")
        if isinstance(payload, dict):
            for key in ("z3Status", "status", "verdict"):
                value = payload.get(key)
                if isinstance(value, str) and value:
                    return value
        return ""

    def tone(self) -> str:
        verdict = self.verdict().upper()
        if verdict in ("UNSAT", "PASS", "OK"):
            return "pass"
        if verdict in ("SAT", "FAIL", "INVALID"):
            return "fail"
        if self.kind == KIND_FAULT:
            return "warn"
        if self.kind == KIND_SYSTEM:
            return "info"
        return "info"

    def level(self) -> str:
        tone = self.tone()
        for level, mapped in _LEVEL_TONE.items():
            if mapped == tone:
                return level
        return "info"

    def to_line(self) -> str:
        stamp = self.stamp_text or f"turn {self.turn:03d}"
        verdict = self.verdict()
        tail = f" [{verdict}]" if verdict else ""
        return f"{stamp} {self.channel} {self.preview()}{tail}"


def _parse_iso(text: str) -> float:
    """Seconds since the epoch for an ISO style timestamp, or zero when unparseable."""
    try:
        return datetime.fromisoformat(text.strip().replace("Z", "+00:00")).timestamp()
    except ValueError:
        return 0.0


def _stamp_of(record: Dict[str, Any]) -> Tuple[float, str]:
    for key in ("timestamp", "ts", "at", "time"):
        value = record.get(key)
        if isinstance(value, (int, float)):
            return float(value), time.strftime("%H:%M:%S", time.localtime(float(value)))
        if isinstance(value, str) and value:
            return _parse_iso(value), value
    return 0.0, ""


def load_transcript(path: Path, max_events: int = 400, default_gap: float = 0.6) -> List[StreamEvent]:
    """Read a transcript file and return the ordered stream of events it recorded.

    Recorded transcripts do not always carry timestamps. When none is present the
    stream falls back to ``default_gap`` seconds between events, so the animation
    still has a rhythm instead of dumping every line at once.
    """
    with open(path, "r", encoding="utf-8") as handle:
        document = json.load(handle)
    records: Sequence[Any]
    if isinstance(document, dict):
        records = document.get("records", document.get("events", []))
    elif isinstance(document, list):
        records = document
    else:
        raise ValueError(f"{path} holds no transcript records")

    timed = any(
        isinstance(record, dict) and _stamp_of(record)[0] for record in records
    )
    events: List[StreamEvent] = []
    previous_stamp: Optional[float] = None
    for index, record in enumerate(records):
        if len(events) >= max_events:
            break
        if not isinstance(record, dict):
            events.append(
                StreamEvent(
                    turn=index,
                    sender="source",
                    recipient="console",
                    envelope={"payload": str(record)},
                    kind=KIND_SYSTEM,
                    offset_seconds=default_gap,
                )
            )
            continue
        stamp, stamp_text = _stamp_of(record)
        delta = default_gap
        if timed:
            delta = 0.0
            if previous_stamp is not None and stamp and stamp >= previous_stamp:
                delta = min(2.0, stamp - previous_stamp)
        if stamp:
            previous_stamp = stamp
        envelope = record.get("envelope")
        if not isinstance(envelope, dict):
            envelope = {
                key: value
                for key, value in record.items()
                if key not in ("turn", "sender", "recipient", "timestamp")
            }
        else:
            # Session transcripts keep the solver result and the routing decision beside the
            # envelope; carrying them into the envelope gives the stream its verdict and lets
            # a blocked message show up as a fault instead of as a normal exchange.
            for key in ("smt_result", "routing_authorized", "terminationReason"):
                if key in record and key not in envelope:
                    envelope[key] = record[key]
        kind = str(record.get("kind", KIND_MESSAGE))
        if record.get("routing_authorized") is False:
            kind = KIND_FAULT
        events.append(
            StreamEvent(
                turn=int(record.get("turn", index)),
                sender=str(record.get("sender", record.get("from", "unknown"))),
                recipient=str(record.get("recipient", record.get("to", "unknown"))),
                envelope=envelope,
                kind=kind,
                offset_seconds=delta,
                stamp_text=stamp_text or f"turn {int(record.get('turn', index)):03d}",
            )
        )
    return events


def replay(events: Iterable[StreamEvent], speed: float = 1.0, max_gap: float = 1.2) -> Iterable[Tuple[StreamEvent, float]]:
    """Yield ``(event, delay)`` pairs so a caller can pace the animation faithfully."""
    factor = speed if speed and speed > 0 else 1.0
    for event in events:
        yield event, min(max_gap, event.offset_seconds / factor)


def transcript_summary(events: Sequence[StreamEvent]) -> Dict[str, Any]:
    channels: Dict[str, int] = {}
    for event in events:
        channels[event.channel] = channels.get(event.channel, 0) + 1
    verdicts: Dict[str, int] = {}
    for event in events:
        verdict = event.verdict()
        if verdict:
            verdicts[verdict] = verdicts.get(verdict, 0) + 1
    return {
        "events": len(events),
        "channels": channels,
        "verdicts": verdicts,
        "turns": len({event.turn for event in events}),
    }


# --- model catalog ------------------------------------------------------------
def model_catalog(config_paths: Iterable[Path], extra_dirs: Iterable[Path] = ()) -> List[Dict[str, str]]:
    """Collect assignable models from configuration files, in declaration order.

    Every entry carries its provenance so the UI can say where a model name came
    from, which matters when a topology references a node type such as a Prover SMT
    checker rather than an LLM checkpoint.
    """
    entries: List[Dict[str, str]] = []
    seen: set = set()

    def add(name: str, role: str, origin: str) -> None:
        clean = str(name).strip()
        if not clean or clean in seen:
            return
        seen.add(clean)
        entries.append({"name": clean, "role": role or "llm", "origin": origin})

    for path in config_paths:
        file_path = Path(path)
        if not file_path.is_file():
            continue
        try:
            with open(file_path, "r", encoding="utf-8") as handle:
                document = json.load(handle)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        _harvest(document, str(file_path.name), add)

    for directory in extra_dirs:
        dir_path = Path(directory)
        if not dir_path.is_dir():
            continue
        for candidate in sorted(dir_path.glob("*.json")):
            try:
                with open(candidate, "r", encoding="utf-8") as handle:
                    document = json.load(handle)
            except (OSError, ValueError, json.JSONDecodeError):
                continue
            _harvest(document, str(candidate.name), add)
    return entries


_MODEL_KEYS = ("model", "model_name", "modelName", "checkpoint", "backend", "engine", "llm", "foundation")
_MODELLIST_KEYS = ("models", "candidates", "allowed_models", "model_catalog", "roster")
_ROLE_KEYS = ("role", "kind", "task", "profile", "persona")
_SUFFIX_NOISE = (".json", ".jsonl", ".smt2", ".smt", ".gf", ".txt", ".py", ".v", ".sv", ".yaml", ".yml", ".md", ".csv", ".rpt", ".vcd")


def _looks_like_model_name(value: str) -> bool:
    """Accept only something that can plausibly name a model, not a path or a constant."""
    if not value or len(value) > 96:
        return False
    if any(ch in value for ch in " \t\n(){}[]<>|,;=\"'"):
        return False
    if value.lower().endswith(_SUFFIX_NOISE):
        return False
    if re.fullmatch(r"[A-Z0-9_]+", value):
        return False
    return True


def _harvest(document: Any, origin: str, add) -> None:
    """Pull declared model names out of a configuration document.

    Strict by intent: a name is only taken from a key that means model, or from a list
    that means a set of models. File references and configuration constants stay out,
    so the assignment combo box in the panel never fills with noise.
    """
    if isinstance(document, dict):
        role = None
        for key in _ROLE_KEYS:
            value = document.get(key)
            if isinstance(value, str) and value:
                role = value
                break
        for key in _MODEL_KEYS:
            value = document.get(key)
            if isinstance(value, str) and _looks_like_model_name(value):
                add(value, role or "llm", origin)
        for key in _MODELLIST_KEYS:
            value = document.get(key)
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, str) and _looks_like_model_name(item):
                        add(item, role or "llm", origin)
                    elif isinstance(item, dict):
                        name = next(
                            (item[k] for k in _MODEL_KEYS if isinstance(item.get(k), str)),
                            "",
                        )
                        if _looks_like_model_name(str(name)):
                            add(str(name), str(item.get("role") or role or "llm"), origin)
        for value in document.values():
            if isinstance(value, (dict, list)):
                _harvest(value, origin, add)
    elif isinstance(document, list):
        for item in document:
            _harvest(item, origin, add)


def assignment_path(root: Path) -> Path:
    """Where the GUI keeps its own per node model assignment file."""
    return Path(root) / "config" / "models.json"


def load_assignments(path: Path) -> Dict[str, Any]:
    """Read the assignment file, returning empty sections when it does not exist.

    The file is GUI owned: it stores the model names a person typed for each node, plus
    the profile list they offered to the combo boxes. Nothing in the existing stack reads
    it, so adding it cannot change a run of the trainer or of the orchestrator.
    """
    try:
        with open(path, "r", encoding="utf-8") as handle:
            document = json.load(handle)
    except (OSError, ValueError, json.JSONDecodeError):
        return {"profiles": [], "assignments": {}}
    if not isinstance(document, dict):
        return {"profiles": [], "assignments": {}}
    profiles = document.get("profiles")
    assignments = document.get("assignments")
    return {
        "profiles": [item for item in profiles if isinstance(item, dict)] if isinstance(profiles, list) else [],
        "assignments": {str(k): str(v) for k, v in assignments.items()} if isinstance(assignments, dict) else {},
    }


def save_assignments(path: Path, profiles: Sequence[Dict[str, str]], assignments: Dict[str, str]) -> Path:
    """Write the assignment file with stable key order, so diffs stay readable."""
    body = {
        "profiles": sorted(
            ({"name": str(item.get("name", "")), "role": str(item.get("role", "llm"))} for item in profiles),
            key=lambda item: item["name"],
        ),
        "assignments": {key: assignments[key] for key in sorted(assignments)},
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(body, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return target


def live_command(root: Path, config: Path) -> Tuple[str, List[str]]:
    """Command line of a live orchestrator session, for a process based runner.

    The module to run comes from the workspace profile. A workspace that declares none
    gets an empty argument list, which callers read as "no live session available".
    """
    executable = "python3" if not str(root).lower().startswith("c:") else "python"
    module = load_profile(root).session_module
    if not module:
        return executable, []
    return executable, [
        "-m",
        module,
        "--config",
        str(config),
        "--json",
    ]


__all__ = [
    "KIND_CONTROL",
    "KIND_FAULT",
    "KIND_MESSAGE",
    "KIND_SYSTEM",
    "StreamEvent",
    "assignment_path",
    "live_command",
    "load_assignments",
    "load_transcript",
    "model_catalog",
    "replay",
    "save_assignments",
    "transcript_summary",
]
