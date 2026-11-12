"""Application entry point: build the shell, apply the local theme, run the Qt loop.

The offline rule is enforced, not just documented. The workbench never imports a web
engine, and :func:`assert_offline` fails loudly if one ever appears in the process, so a
regression that would reach for a remote asset shows up at start up instead of silently
degrading on an air gapped host.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from PyQt6.QtCore import QT_VERSION_STR, PYQT_VERSION_STR
from PyQt6.QtWidgets import QApplication

import gui
from gui import theme
from gui.paths import discover_workspace, open_workspace
from gui.window import WorkbenchWindow

WEB_MODULES = ("PyQt6.QtWebEngineWidgets", "PyQt6.QtWebEngineCore", "PyQt6.QtWebKit", "PyQt6.QtWebKitWidgets")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Command line of the workbench: which workspace to open and how loud to be."""
    parser = argparse.ArgumentParser(
        prog="lingologos-workbench",
        description="Desktop workbench over the LingoLOGOS neuro-symbolic workspace.",
    )
    parser.add_argument(
        "--workspace",
        default="",
        help="workspace root to open; by default the repository that holds the current directory",
    )
    parser.add_argument("--verbose", action="store_true", help="print the artifact role inventory at start up")
    parser.add_argument("--version", action="store_true", help="print the workbench and toolkit versions and exit")
    return parser.parse_args(list(argv) if argv is not None else None)


def resolve_root(requested: str) -> Tuple[Optional[Path], str]:
    """Find the workspace to open, and say why it could not be found when that happens."""
    if requested:
        candidate = Path(requested).expanduser()
        if not candidate.is_dir():
            return None, f"the workspace path {candidate} is not a directory"
        return candidate, ""
    try:
        found = discover_workspace(Path.cwd())
    except FileNotFoundError as exc:
        return None, f"{exc}: pass --workspace PATH"
    return found, ""


def assert_offline() -> str:
    """Return the name of a forbidden module already present in this process, or an empty string."""
    loaded = [name for name in WEB_MODULES if name in sys.modules]
    return loaded[0] if loaded else ""


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Start the workbench; returns the Qt loop exit code."""
    args = parse_args(argv)
    if args.version:
        print(f"LingoLOGOS Workbench {gui.__version__} on PyQt {PYQT_VERSION_STR}, Qt {QT_VERSION_STR}")
        return 0
    root, reason = resolve_root(args.workspace)
    if root is None:
        print(f"cannot open a workspace: {reason}", file=sys.stderr)
        return 2

    own_loop = QApplication.instance() is None
    argv_for_qt: List[str] = [sys.argv[0]]
    app = QApplication.instance() or QApplication(argv_for_qt)
    app.setApplicationName("LingoLOGOS Workbench")
    app.setOrganizationName("LingoLOGOS")
    app.setStyleSheet(theme.stylesheet())

    offender = assert_offline()
    if offender:
        print(f"refusing to start: {offender} is loaded, and this workbench is offline by design", file=sys.stderr)
        return 3

    workspace = open_workspace(root)
    window = WorkbenchWindow(workspace)
    window.show()
    print(f"LingoLOGOS Workbench {gui.__version__} open on {root}")
    if args.verbose:
        print("  " + (workspace.profile.describe() if workspace.profile else "no workspace profile"))
        for row in workspace.role_summary():
            print("  " + " | ".join(str(cell) for cell in row))
    print("no web resource is loaded: interface, icons and fonts come from Qt alone")
    if not own_loop:
        return 0
    return int(app.exec())


def run(argv: Optional[Sequence[str]] = None) -> int:
    """Alias of :func:`main` for callers that prefer the verb."""
    return main(argv)


__all__ = ["WEB_MODULES", "assert_offline", "main", "parse_args", "resolve_root", "run"]
