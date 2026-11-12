"""Workspace profile: the single place where the workbench names its host project.

The workbench is a front end that reads files and launches commands. It never imports
the project it sits in, not even lazily. Everything it needs to know about that project
that is not a file format is data, and that data lives here: where the conventional
folders are, which command starts a live session, which script exports the fabrication
artifacts, which make file drives a simulation.

A profile is a plain mapping. The built in defaults describe the layout this repository
ships with, and an optional ``workbench.json`` at the workspace root is merged over them,
so another workspace can point the same interface at its own folders and commands without
a line of the interface changing. A key set to an empty string switches the matching
feature off: the button that would need it stays visible and says why it cannot run.

A profile that cannot be read never stops the workbench. The defaults stay in force and
the reasons are collected in :attr:`Profile.problems`, which the shell prints to its
console.

This module is Qt free, like the rest of the model layer, and it is the only module of the
workbench that is allowed to spell a host project identifier in code. A test enforces it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

PROFILE_FILE = "workbench.json"

# A directory is taken for a workspace root when it holds all of these, or when it holds a
# profile file, which is the explicit way to declare a root.
DEFAULT_ROOT_MARKERS: Tuple[str, ...] = ("pyproject.toml", "src")

# Folder keys of the profile, all relative to the workspace root.
DIRECTORY_KEYS: Tuple[str, ...] = (
    "config",
    "topologies",
    "datasets",
    "artifacts",
    "hardware",
    "rtl",
    "sim",
    "runtime_state",
    "sources",
)

DEFAULT_DIRECTORIES: Dict[str, str] = {
    "config": "config",
    "topologies": "topologies",
    "datasets": "datasets",
    "artifacts": "export_artifacts",
    "hardware": "asic_chip",
    "rtl": "asic_chip/rtl",
    "sim": "asic_chip/sim",
    "runtime_state": "runtime_state",
    "sources": "src",
}

DEFAULT_MAKE_TARGETS: Dict[str, str] = {
    "lint-topology": "Verilator lint of the topology router and its board",
    "sim-topology": "Topology testbench simulation, prints FINAL VERDICT",
    "sim-tb": "Formal core testbench simulation",
    "export": "Regenerate the hardware export artifacts",
}

_TEXT_KEYS: Tuple[str, ...] = (
    "name",
    "session_module",
    "export_script",
    "openlane_runs",
    "makefile",
    "simulation_cwd",
    "verilator_objdir",
)
_LIST_KEYS: Tuple[str, ...] = ("session_python_path",)
_MAP_KEYS: Tuple[str, ...] = ("directories", "make_targets")


@dataclass(frozen=True)
class Profile:
    """What the workbench knows about the project around it, as data.

    Every path is relative to the workspace root and uses forward slashes. An empty
    string for a command or a script means that the workspace does not offer it.
    """

    name: str = "default"
    directories: Dict[str, str] = field(default_factory=lambda: dict(DEFAULT_DIRECTORIES))
    session_module: str = "lingologos_orchestrator.cli"
    session_python_path: Tuple[str, ...] = ("src",)
    export_script: str = "asic_chip/scripts/export_artifacts.py"
    openlane_runs: str = "asic_chip/openlane/runs"
    makefile: str = "asic_chip/Makefile"
    simulation_cwd: str = "asic_chip"
    verilator_objdir: str = "asic_chip/sim/obj_dir_tb"
    make_targets: Dict[str, str] = field(default_factory=lambda: dict(DEFAULT_MAKE_TARGETS))
    problems: Tuple[str, ...] = ()
    source: Optional[Path] = None

    def directory(self, key: str) -> str:
        """Relative folder for a profile key, falling back to the built in default."""
        return self.directories.get(key) or DEFAULT_DIRECTORIES.get(key, "")

    @property
    def has_session(self) -> bool:
        """Whether this workspace declares a command for a live orchestrator session."""
        return bool(self.session_module)

    @property
    def has_export(self) -> bool:
        return bool(self.export_script)

    @property
    def has_makefile(self) -> bool:
        return bool(self.makefile)

    def describe(self) -> str:
        """One line for the console and the verbose start up report."""
        origin = self.source.name if self.source else "built in defaults"
        return f"workspace profile {self.name} ({origin})"


DEFAULT_PROFILE = Profile(name="lingologos")


def _clean_text(key: str, value: Any, problems: List[str]) -> Optional[str]:
    if isinstance(value, str):
        return value.strip()
    problems.append(f"{key} must be a string, the value was ignored")
    return None


def _clean_list(key: str, value: Any, problems: List[str]) -> Optional[Tuple[str, ...]]:
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return tuple(item.strip() for item in value if item.strip())
    problems.append(f"{key} must be a list of strings, the value was ignored")
    return None


def _clean_map(key: str, value: Any, problems: List[str]) -> Optional[Dict[str, str]]:
    if not isinstance(value, dict):
        problems.append(f"{key} must be an object, the value was ignored")
        return None
    cleaned: Dict[str, str] = {}
    for name, item in value.items():
        if not isinstance(item, str):
            problems.append(f"{key}.{name} must be a string, the entry was ignored")
            continue
        cleaned[str(name)] = item.strip()
    return cleaned


def merge_profile(base: Profile, document: Mapping[str, Any], source: Optional[Path] = None) -> Profile:
    """Return ``base`` with the recognised keys of ``document`` laid over it.

    Unknown keys and values of the wrong type are skipped and reported in ``problems``;
    they never raise, because a profile is configuration and a typo in it must not take
    the interface down.
    """
    problems: List[str] = list(base.problems)
    changes: Dict[str, Any] = {}
    for key, value in document.items():
        if key in _TEXT_KEYS:
            text = _clean_text(key, value, problems)
            if text is not None:
                changes[key] = text
        elif key in _LIST_KEYS:
            items = _clean_list(key, value, problems)
            if items is not None:
                changes[key] = items
        elif key in _MAP_KEYS:
            mapping = _clean_map(key, value, problems)
            if mapping is None:
                continue
            if key == "directories":
                unknown = sorted(set(mapping) - set(DIRECTORY_KEYS))
                if unknown:
                    problems.append(f"unknown directory keys ignored: {', '.join(unknown)}")
                merged = dict(base.directories)
                merged.update({name: item for name, item in mapping.items() if name in DIRECTORY_KEYS and item})
                changes[key] = merged
            else:
                changes[key] = mapping
        else:
            problems.append(f"unknown key {key} ignored")
    return replace(base, problems=tuple(problems), source=source, **changes)


def load_profile(root: Optional[Path]) -> Profile:
    """Profile of the workspace at ``root``: the defaults, plus its ``workbench.json``."""
    if root is None:
        return DEFAULT_PROFILE
    path = Path(root) / PROFILE_FILE
    if not path.is_file():
        return DEFAULT_PROFILE
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return replace(DEFAULT_PROFILE, problems=(f"{PROFILE_FILE} could not be read: {exc}",), source=path)
    if not isinstance(document, dict):
        return replace(DEFAULT_PROFILE, problems=(f"{PROFILE_FILE} must hold a JSON object at the top level",), source=path)
    return merge_profile(DEFAULT_PROFILE, document, path)


def is_workspace_root(candidate: Path) -> bool:
    """Whether a directory is a workspace root: a profile file, or the default markers."""
    if (candidate / PROFILE_FILE).is_file():
        return True
    return all((candidate / marker).exists() for marker in DEFAULT_ROOT_MARKERS)


__all__ = [
    "DEFAULT_DIRECTORIES",
    "DEFAULT_MAKE_TARGETS",
    "DEFAULT_PROFILE",
    "DEFAULT_ROOT_MARKERS",
    "DIRECTORY_KEYS",
    "PROFILE_FILE",
    "Profile",
    "is_workspace_root",
    "load_profile",
    "merge_profile",
]
