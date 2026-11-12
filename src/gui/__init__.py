"""LingoLOGOS Workbench: the PyQt6 desktop front end of the neuro-symbolic stack.

The package is a leaf of the dependency graph, and a closed one. Nothing in the host
project imports anything from here, and nothing here imports anything from the host
project, not at module level and not inside a function either. The workbench talks to
the project through two channels only: the files it reads from the workspace, and the
commands it launches as separate processes. Which folders and which commands, is data
declared by the workspace profile (:mod:`gui.model.profile`, optionally overridden by a
``workbench.json`` at the workspace root), so a workbench import can never drag torch or
z3 into a process that does not want them, and a missing command degrades one panel
instead of breaking the application.

Offline rule: the workbench ships no remote asset. No CDN, no web font, no downloaded
icon, no network call. Visual identity comes from the Qt widgets themselves, from
QPainter primitives, from the standard icons of ``QStyle``, and from a local
QStyleSheet built in :mod:`gui.theme`.

Modules of the workbench, one tab each in the shell window:
    1. Topology canvas: generic node-based editor over mixed datasets.
    2. Lllayer panel: multi model orchestration and timestamped live stream.
    3. Board viewer: multi die hardware view derived from the filesystem.
    4. Dataset manager: mixed payload tables in json, jsonl and txt form.
    5. SV analyzer: SystemVerilog testbench inspection and Verilator hand off.
"""

__all__ = ["main", "run"]

__version__ = "0.1.0"


def __getattr__(name: str):
    """Import the Qt heavy surface only when an entry point is actually requested."""
    if name in ("main", "run"):
        from .app import main as _main

        return _main
    raise AttributeError(f"module gui has no attribute {name}")
