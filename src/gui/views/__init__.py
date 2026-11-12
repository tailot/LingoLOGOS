"""Module panels of the workbench.

Every panel follows three conventions so the shell can drive them without knowing
what it does: ``attach_console(log)`` hands the panel the shared log console,
``refresh(scan)`` re-reads whatever artifacts the workspace scan found, and a panel
that reacts to the message stream exposes ``handle_stream_event(event)``.

The panel classes are exported lazily: importing ``gui.views`` stays cheap, and a
panel that cannot be loaded for any reason does not take the whole interface down.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any, List

__all__ = ["CanvasPanel", "DatasetPanel", "HardwarePanel", "OrchestratorPanel", "SourcePanel"]

_PANELS = {
    "CanvasPanel": ("canvas", "CanvasPanel"),
    "DatasetPanel": ("dataset", "DatasetPanel"),
    "HardwarePanel": ("hardware", "HardwarePanel"),
    "OrchestratorPanel": ("lllayer", "OrchestratorPanel"),
    "SourcePanel": ("sv", "SourcePanel"),
}


def __getattr__(name: str) -> Any:
    entry = _PANELS.get(name)
    if entry is None:
        raise AttributeError(f"module gui.views has no attribute {name}")
    module = import_module(f".{entry[0]}", __name__)
    return getattr(module, entry[1])


def panel_names() -> List[str]:
    """Names of the panels the shell mounts, in tab order."""
    return list(__all__)
