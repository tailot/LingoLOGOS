"""Background workers: file scanning, external tool runs, paced stream replay.

Every long or blocking action the workbench performs lives here, so a panel never
freezes the interface and never lets a failing tool bring the application down. Workers
talk to panels through Qt signals only, and each one is allowed to fail quietly by
reporting the failure as a log line.
"""

from __future__ import annotations

from .runner import (
    MAKE_TARGETS,
    CommandWorker,
    ExportOptions,
    ParseWorker,
    ScanWorker,
    ToolPlan,
    build_export_plan,
    build_make_plan,
    build_verilator_plan,
    classify_line,
    export_note,
    find_openlane_runs,
    newest_openlane_run,
    plan_table,
    probe_verilator,
    python_path_entries,
    python_path_for,
    verdict_of_line,
)
from .session import SessionRunner, TranscriptReplayer

__all__ = [
    "MAKE_TARGETS",
    "CommandWorker",
    "ExportOptions",
    "ParseWorker",
    "ScanWorker",
    "SessionRunner",
    "ToolPlan",
    "TranscriptReplayer",
    "build_export_plan",
    "build_make_plan",
    "build_verilator_plan",
    "classify_line",
    "export_note",
    "find_openlane_runs",
    "newest_openlane_run",
    "plan_table",
    "probe_verilator",
    "python_path_entries",
    "python_path_for",
    "verdict_of_line",
]
