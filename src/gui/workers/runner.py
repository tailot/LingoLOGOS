"""Command runner and scanner workers.

Two jobs live here:

* run an external program (a Verilator simulation, a make target) without ever blocking
  the interface, and report each output line to a log console as it arrives;
* rescan the workspace filesystem from a background thread, so opening a panel or
  pressing refresh never stalls on a big export_artifacts tree.

The runner never trusts the tool: a program that is missing, that cannot start, or that
exits with an error lands in the same signal path as a normal run, with a level tag the
console colours differently.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from PyQt6.QtCore import QThread, pyqtSignal

from gui.model.profile import DEFAULT_PROFILE, PROFILE_FILE, load_profile

_VERDICT = re.compile(r"(FINAL VERDICT|Verdict)\s*:\s*(PASS|FAIL)", re.IGNORECASE)
_ERRORISH = re.compile(r"(error|failed|exception|traceback|cannot|no such)", re.IGNORECASE)
_WARNISH = re.compile(r"(warning|warn|deprecated)", re.IGNORECASE)

# Values of the built in workspace profile, kept as module constants for callers that
# want the defaults. Every function below reads the profile of the workspace it is
# given instead, so a ``workbench.json`` at the root changes what these plans run.
MAKE_TARGETS = dict(DEFAULT_PROFILE.make_targets)

# The export is a workspace script, not a make recipe: it is the one production step
# the workbench can run on its own, on any host, because it only moves files that an
# OpenLane run already left on disk.
EXPORT_SCRIPT = DEFAULT_PROFILE.export_script
OPENLANE_RUNS = DEFAULT_PROFILE.openlane_runs


def export_script_path(root: Path) -> Path:
    """Absolute path of the artifact export script of this workspace."""
    return Path(root) / load_profile(root).export_script


def openlane_runs_dir(root: Path) -> Path:
    """Directory an OpenLane run leaves its run folders in."""
    return Path(root) / load_profile(root).openlane_runs


def find_openlane_runs(root: Path) -> List[str]:
    """Run directories present on disk, oldest first, empty when there is none."""
    if not load_profile(root).openlane_runs:
        return []
    directory = openlane_runs_dir(root)
    if not directory.is_dir():
        return []
    runs = [
        path
        for path in directory.iterdir()
        if path.is_dir() and not path.name.startswith((".", "_"))
    ]
    try:
        runs.sort(key=lambda path: path.stat().st_mtime)
    except OSError:
        runs.sort(key=lambda path: str(path))
    return [str(path) for path in runs]


def newest_openlane_run(root: Path) -> str:
    """The most recent OpenLane run of this workspace, or an empty string."""
    runs = find_openlane_runs(root)
    return runs[-1] if runs else ""


@dataclass
class ExportOptions:
    """What the export button will ask the script for, kept apart for the tooltip."""

    dry_run: bool = False
    desktop: bool = False
    run: str = ""
    runs_dir: str = ""

    def as_args(self) -> List[str]:
        args: List[str] = []
        if self.runs_dir:
            args += ["--runs-dir", str(self.runs_dir)]
        if self.run:
            args += ["--run", str(self.run)]
        if self.dry_run:
            args += ["--dry-run", "--allow-missing"]
        if self.desktop:
            args += ["--desktop"]
        return args


def build_export_plan(
    root: Path,
    dry_run: bool = False,
    desktop: bool = False,
    run: str = "",
    runs_dir: str = "",
) -> ToolPlan:
    """Plan the artifact export of section 4: one script, one tree, one manifest.

    The script is host agnostic: it copies and generates files, so the workbench runs
    it with its own interpreter. What changes between hosts is whether a run exists to
    export, so the plan says that out loud instead of pretending the button is magic.
    """
    root = Path(root)
    profile = load_profile(root)
    label = "export plan (dry run)" if dry_run else "export artifacts"
    if not profile.has_export:
        return ToolPlan(
            label=label,
            program=sys.executable,
            cwd=str(root),
            available=False,
            reason="this workspace profile declares no export script",
            hint=f"set export_script in {PROFILE_FILE} at the workspace root",
        )
    script = export_script_path(root)
    if not script.is_file():
        return ToolPlan(
            label=label,
            program=sys.executable,
            cwd=str(root),
            available=False,
            reason=f"the export script is missing: {profile.export_script}",
            hint=f"python3 {profile.export_script} --dry-run",
        )
    runs = find_openlane_runs(root)
    if not run and not runs_dir and not runs:
        note = (
            f"no OpenLane run under {profile.openlane_runs}: the export would report every "
            "copied artifact as MISSING and generate the board views only"
        )
        make_hint = f" for example: make -f {profile.makefile} openlane" if profile.has_makefile else ""
        hint = f"build one from the shell that has OpenLane,{make_hint}" if make_hint else "build one from the shell that has OpenLane"
        if not dry_run:
            return ToolPlan(
                label=label,
                program=sys.executable,
                args=[str(script)],
                cwd=str(root),
                available=False,
                reason=note,
                hint=hint,
            )
    else:
        chosen = run or runs_dir or (runs[-1] if runs else "")
        note = f"exporting {Path(str(chosen)).name}" if chosen else "exporting the newest run"
    options = ExportOptions(dry_run=dry_run, desktop=desktop, run=run, runs_dir=runs_dir)
    return ToolPlan(
        label=label,
        program=sys.executable,
        args=[str(script), *options.as_args()],
        cwd=str(root),
        available=True,
        reason=note,
    )


def export_note(root: Path) -> str:
    """One line about what an export run here would find, for a tooltip or a log."""
    runs = find_openlane_runs(root)
    if not runs:
        return f"no OpenLane run under {load_profile(root).openlane_runs or 'any folder'}"
    return f"{len(runs)} OpenLane run directories, newest {Path(runs[-1]).name}"


@dataclass
class ToolPlan:
    """A command, its working directory, its extra environment, and whether it can run here."""

    label: str
    program: str
    args: List[str] = field(default_factory=list)
    cwd: str = ""
    available: bool = True
    reason: str = ""
    hint: str = ""
    env: Dict[str, str] = field(default_factory=dict)

    @property
    def command_text(self) -> str:
        parts = [self.program, *self.args]
        return " ".join(part for part in parts if part)

    def blocked_because(self) -> str:
        if self.available:
            return ""
        return self.reason or f"{self.label} is not available on this host"


def python_path_entries(root: Path) -> str:
    """The workspace folders the profile puts on PYTHONPATH, without any inherited value."""
    return os.pathsep.join(str(Path(root) / entry) for entry in load_profile(root).session_python_path)


def python_path_for(root: Path) -> str:
    """PYTHONPATH value that makes the session module importable from the workspace.

    The absolute folders of the profile come first and any value the caller already had
    is kept behind them, so an installed deployment is unaffected.
    """
    entry = python_path_entries(root)
    existing = os.environ.get("PYTHONPATH", "")
    if not entry:
        return existing
    return entry if not existing else entry + os.pathsep + existing


def verdict_of_line(line: str) -> Optional[str]:
    """Return PASS or FAIL when a line carries the final verdict of a testbench."""
    match = _VERDICT.search(line)
    if not match:
        return None
    return match.group(2).upper()


def classify_line(line: str) -> str:
    """Map a raw output line to the log level a console should display it with."""
    verdict = verdict_of_line(line)
    if verdict:
        return "ok" if verdict == "PASS" else "error"
    if _ERRORISH.search(line):
        return "error"
    if _WARNISH.search(line):
        return "warn"
    return "info"


def probe_verilator(root: Path) -> ToolPlan:
    """Describe whether Verilator can be run from inside this workbench session.

    On Windows the workspace drives Verilator through WSL or Docker (see its make file),
    so the button is disabled here on purpose and the reason states the command to run
    in that shell.
    """
    root = Path(root)
    profile = load_profile(root)
    found = shutil.which("verilator")
    if sys.platform.startswith("win"):
        hint = f"make -f {profile.makefile} sim-topology" if profile.has_makefile else "run the simulation from that shell"
        return ToolPlan(
            label="Verilator simulation",
            program="verilator",
            cwd=str(root),
            available=False,
            reason=(
                "Verilator is disabled in the interface on Windows: this repository runs "
                "it through WSL or Docker, so the workbench cannot launch it directly here."
            ),
            hint=f"run it from that shell instead, for example: {hint}",
        )
    if not found:
        return ToolPlan(
            label="Verilator simulation",
            program="verilator",
            cwd=str(root),
            available=False,
            reason="the verilator executable was not found on PATH",
            hint="install Verilator or run the make target from a shell that has it",
        )
    return ToolPlan(
        label="Verilator simulation",
        program=found,
        cwd=str(root),
        available=True,
        reason=f"found at {found}",
    )


def build_make_plan(root: Path, target: str) -> ToolPlan:
    """Plan a make invocation for one of the make targets of the workspace."""
    root = Path(root)
    profile = load_profile(root)
    make = shutil.which("make") or ("wsl" if sys.platform.startswith("win") else "make")
    if not profile.has_makefile:
        return ToolPlan(
            label=target,
            program=make,
            args=[],
            cwd=str(root),
            available=False,
            reason="this workspace profile declares no make file",
            hint=f"set makefile in {PROFILE_FILE} at the workspace root",
        )
    makefile = root / profile.makefile
    if sys.platform.startswith("win") and shutil.which("make") is None:
        return ToolPlan(
            label=target,
            program=make,
            args=[],
            cwd=str(root),
            available=False,
            reason="no make executable on this Windows host; the make file expects WSL or Docker",
            hint=f"make -f {profile.makefile} {target}",
        )
    args = ["-f", str(makefile), target]
    return ToolPlan(label=target, program=make, args=args, cwd=str(root), available=makefile.is_file(), reason="" if makefile.is_file() else f"{makefile} is missing")


def build_verilator_plan(root: Path, source: Path, top_module: str = "") -> ToolPlan:
    """Plan a direct Verilator run of one testbench file, mirroring the Makefile flags."""
    root = Path(root)
    probe = probe_verilator(root)
    if not probe.available:
        return probe
    profile = load_profile(root)
    objdir = root / (profile.verilator_objdir or "obj_dir_tb")
    args = ["--binary", "--timing", "-Mdir", str(objdir)]
    if top_module:
        args += ["--top-module", top_module]
    args.append(str(source))
    return ToolPlan(
        label=f"verilator {Path(source).name}",
        program=probe.program,
        args=args,
        cwd=str(root / profile.simulation_cwd) if profile.simulation_cwd else str(root),
        available=True,
        reason=probe.reason,
    )


class CommandWorker(QThread):
    """Run one program, streaming its merged output to the console as it arrives."""

    line_ready = pyqtSignal(str, str)
    ended = pyqtSignal(int, str)
    started = pyqtSignal(str)

    def __init__(self, plan: ToolPlan, parent=None) -> None:
        super().__init__(parent)
        self._plan = plan
        self._process: Optional[subprocess.Popen] = None
        self._verdict: Optional[str] = None
        self._tail: List[str] = []

    @property
    def plan(self) -> ToolPlan:
        return self._plan

    @property
    def verdict(self) -> Optional[str]:
        """The verdict the last run reported, when the tool printed one."""
        return self._verdict

    def run(self) -> None:  # noqa: D401 - QThread entry point
        plan = self._plan
        self.started.emit(plan.command_text)
        if not plan.available:
            self.line_ready.emit(plan.blocked_because(), "error")
            if plan.hint:
                self.line_ready.emit(plan.hint, "warn")
            self.ended.emit(-1, "blocked")
            return
        try:
            environment = None
            if plan.env:
                environment = dict(os.environ)
                environment.update(plan.env)
            self._process = subprocess.Popen(
                [plan.program, *plan.args],
                cwd=plan.cwd or None,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
        except OSError as exc:
            self.line_ready.emit(f"cannot start {plan.program}: {exc}", "error")
            self.ended.emit(-1, "start failed")
            return
        handle = self._process.stdout
        if handle is None:
            self.line_ready.emit("the program produced no output stream", "warn")
            self.ended.emit(-1, "no stream")
            return
        for raw in handle:
            line = raw.rstrip("\n")
            if not line:
                continue
            verdict = verdict_of_line(line)
            if verdict:
                self._verdict = verdict
            self._tail.append(line)
            if len(self._tail) > 400:
                del self._tail[0]
            self.line_ready.emit(line, classify_line(line))
        code = int(self._process.wait())
        self.ended.emit(code, self.summary())

    def summary(self) -> str:
        if self._verdict:
            return f"FINAL VERDICT: {self._verdict}"
        return f"{len(self._tail)} output lines"

    def stop(self) -> None:
        """Ask the process to stop; the wait loop in run reports the exit code."""
        process = self._process
        if process is not None and process.poll() is None:
            try:
                process.terminate()
            except OSError:
                pass

    def tail(self, count: int = 40) -> List[str]:
        return self._tail[-count:]


class ScanWorker(QThread):
    """Rescan the workspace tree in the background and report the role counts."""

    scanned = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, workspace, parent=None) -> None:
        super().__init__(parent)
        self._workspace = workspace

    def run(self) -> None:
        try:
            scan = self._workspace.rescan()
        except OSError as exc:
            self.failed.emit(f"the workspace scan failed: {exc}")
            return
        self.scanned.emit(scan)


class ParseWorker(QThread):
    """Parse one artifact in the background, so a huge source file never blocks a panel."""

    parsed = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, loader, path: Path, parent=None) -> None:
        super().__init__(parent)
        self._loader = loader
        self._path = Path(path)

    def run(self) -> None:
        try:
            result = self._loader(self._path)
        except Exception as exc:  # a parser must never take the workbench down
            self.failed.emit(f"{self._path.name}: {type(exc).__name__}: {exc}")
            return
        self.parsed.emit(result)


def plan_table(plans: Sequence[ToolPlan]) -> Dict[str, str]:
    """Human readable map of label to availability note, for the diagnostics card."""
    return {plan.label: (plan.reason or plan.hint or "not available on this host") if not plan.available else plan.reason for plan in plans}


__all__ = [
    "CommandWorker",
    "EXPORT_SCRIPT",
    "MAKE_TARGETS",
    "OPENLANE_RUNS",
    "ExportOptions",
    "ParseWorker",
    "ScanWorker",
    "ToolPlan",
    "build_export_plan",
    "build_make_plan",
    "build_verilator_plan",
    "classify_line",
    "export_note",
    "export_script_path",
    "find_openlane_runs",
    "newest_openlane_run",
    "openlane_runs_dir",
    "plan_table",
    "probe_verilator",
    "python_path_entries",
    "python_path_for",
    "verdict_of_line",
]
