"""Visual identity of the workbench, built from Qt primitives only.

Everything here is local: the style sheet is a Python string, the icons are the
standard pixmaps Qt ships, the fonts are whatever fixed pitch face the host already
has. No web font, no CDN, no downloaded icon, no .qrc resource file. That keeps the
workbench byte-for-byte reproducible on an offline machine and free of third party
asset licensing questions.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QFont, QFontDatabase, QIcon, QPen
from PyQt6.QtWidgets import QStyle

# --- palette -----------------------------------------------------------------
# Dark engineering console: low glare for long layout sessions, and a distinct hue
# per semantic class so a verdict, a fault and a selection never read the same.
PALETTE: Dict[str, str] = {
    "window": "#0f141b",
    "panel": "#161d27",
    "panel_alt": "#1b2532",
    "frame": "#26313f",
    "grid": "#223042",
    "text": "#d8e2ee",
    "text_dim": "#8b9bad",
    "accent": "#39b8ff",
    "accent_soft": "#1d3a52",
    "pass": "#3ecf8e",
    "fail": "#ff5d6c",
    "warn": "#ffc65c",
    "smt": "#b98bff",
    "gf": "#5cd6c0",
    "data": "#ffab5c",
    "link": "#4f7fa8",
    "link_hot": "#7fd7ff",
}


def color(key: str) -> QColor:
    """Return a palette entry as QColor, so views never hardcode hex values."""
    return QColor(PALETTE[key])


def pen(key: str, width: float = 1.0, style: Qt.PenStyle = Qt.PenStyle.SolidLine) -> QPen:
    """Return a themed pen for QPainter work in the canvas and board views."""
    handle = QPen(color(key))
    handle.setWidthF(width)
    handle.setStyle(style)
    handle.setCapStyle(Qt.PenCapStyle.RoundCap)
    handle.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    return handle


# --- syntax roles ------------------------------------------------------------
# The SystemVerilog analyzer emits style keys and never a colour. This map is the
# single place where a style key becomes a hue, so the analyzer stays free of
# presentation and the workbench stays free of hardcoded hex values.
SYNTAX: Dict[str, str] = {
    "comment": "text_dim",
    "string": "gf",
    "param": "data",
    "constant": "accent",
    "system": "warn",
    "mask": "fail",
    "verdict": "pass",
    "decl": "smt",
    "keyword": "text",
}


def style_color(key: str) -> QColor:
    """Colour for a syntax style key, falling back to plain text."""
    return color(SYNTAX.get(key, "text"))


# --- fonts -------------------------------------------------------------------
# Preference order only. Each name is a hint: the first one the host already has
# wins, and if none is present Qt returns its own fixed pitch default. Nothing is
# fetched, nothing is installed.
_MONO_HINTS: List[str] = [
    "JetBrains Mono",
    "Cascadia Mono",
    "Consolas",
    "DejaVu Sans Mono",
    "Liberation Mono",
    "Menlo",
    "Courier New",
]


def mono_font(point_size: float = 9.5, bold: bool = False) -> QFont:
    """Return a readable fixed pitch font from the families already on this host."""
    available = set(QFontDatabase.families())
    for hint in _MONO_HINTS:
        if hint in available:
            handle = QFont(hint)
            handle.setStyleHint(QFont.StyleHint.Monospace)
            handle.setPointSizeF(point_size)
            handle.setBold(bold)
            return handle
    handle = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)
    handle.setPointSizeF(point_size)
    handle.setBold(bold)
    return handle


def ui_font(point_size: float = 9.5, bold: bool = False) -> QFont:
    """Return the application default face at a workbench size."""
    handle = QFontDatabase.systemFont(QFontDatabase.SystemFont.GeneralFont)
    handle.setPointSizeF(point_size)
    handle.setBold(bold)
    return handle


# --- icons -------------------------------------------------------------------
# The icon set of the workbench is the set Qt distributes. Mapping semantic names
# onto StandardPixmap keys means the toolbar still has glyphs on a headless build
# and no asset file travels with the repository.
_ICON_KEYS: Dict[str, str] = {
    "open": "SP_FileIcon",
    "save": "SP_DialogSaveButton",
    "folder": "SP_DirIcon",
    "refresh": "SP_BrowserReload",
    "run": "SP_MediaPlay",
    "stop": "SP_MediaStop",
    "add": "SP_FileDialogNewFolder",
    "remove": "SP_TrashIcon",
    "check": "SP_DialogApplyButton",
    "cancel": "SP_DialogCancelButton",
    "error": "SP_MessageBoxCritical",
    "warn": "SP_MessageBoxWarning",
    "info": "SP_MessageBoxInformation",
    "chip": "SP_ComputerIcon",
    "network": "SP_DriveNetIcon",
    "up": "SP_ArrowUp",
    "down": "SP_ArrowDown",
    "left": "SP_ArrowLeft",
    "right": "SP_ArrowRight",
}


def standard_icon(widget_style: QStyle, name: str) -> Optional[QIcon]:
    """Return a themed standard icon, or None when the name is unknown to Qt."""
    key = _ICON_KEYS.get(name)
    if key is None:
        return None
    pixmap_key = getattr(QStyle.StandardPixmap, key, None)
    if pixmap_key is None:
        return None
    return widget_style.standardIcon(pixmap_key)


# --- style sheet -------------------------------------------------------------
def stylesheet() -> str:
    """Return the local QStyleSheet for the whole shell."""
    p = PALETTE
    return f"""
QMainWindow#WorkbenchShell {{ background: {p['window']}; }}
QWidget {{ color: {p['text']}; }}
QMainWindow::separator {{ background: {p['frame']}; width: 1px; height: 1px; }}

QMenuBar {{ background: {p['panel']}; color: {p['text']}; }}
QMenuBar::item:selected {{ background: {p['accent_soft']}; }}
QMenu {{ background: {p['panel_alt']}; border: 1px solid {p['frame']}; }}
QMenu::item:selected {{ background: {p['accent_soft']}; }}
QStatusBar {{ background: {p['panel']}; color: {p['text_dim']}; }}
QStatusBar::item {{ border: none; }}

QTabWidget#ModuleTabs::pane {{ border: 1px solid {p['frame']}; background: {p['panel']}; }}
QTabBar#ModuleTabs::tab {{
    background: {p['panel_alt']}; color: {p['text_dim']};
    padding: 7px 14px; border: 1px solid {p['frame']}; border-bottom: none;
}}
QTabBar#ModuleTabs::tab:selected {{ background: {p['window']}; color: {p['accent']}; }}

QGroupBox#Card {{
    border: 1px solid {p['frame']}; border-radius: 6px; margin-top: 10px;
    background: {p['panel']}; padding-top: 6px;
}}
QGroupBox#Card::title {{ subcontrol-origin: margin; left: 8px; color: {p['text_dim']}; }}

QLabel#PanelTitle {{ color: {p['accent']}; font-weight: bold; }}
QLabel#MutedLabel {{ color: {p['text_dim']}; }}
QLabel#Chip {{
    border: 1px solid {p['frame']}; border-radius: 9px; padding: 1px 8px;
    background: {p['panel_alt']}; color: {p['text_dim']};
}}
QLabel#ChipPass {{ border-color: {p['pass']}; color: {p['pass']}; background: {p['panel_alt']}; }}
QLabel#ChipFail {{ border-color: {p['fail']}; color: {p['fail']}; background: {p['panel_alt']}; }}
QLabel#ChipWarn {{ border-color: {p['warn']}; color: {p['warn']}; background: {p['panel_alt']}; }}

QTextEdit#LogConsole, QPlainTextEdit#LogConsole, QPlainTextEdit#SourceView {{
    background: {p['window']}; color: {p['text']};
    border: 1px solid {p['frame']};
}}

QTableView, QTreeView, QListWidget {{
    background: {p['window']}; alternate-background-color: {p['panel']};
    gridline-color: {p['frame']}; border: 1px solid {p['frame']};
    selection-background-color: {p['accent_soft']}; selection-color: {p['text']};
}}
QHeaderView::section {{
    background: {p['panel_alt']}; color: {p['text_dim']};
    border: none; border-right: 1px solid {p['frame']}; padding: 4px;
}}
QTableCornerButton::section {{ background: {p['panel_alt']}; border: none; }}

QToolBar {{ background: {p['panel']}; border: none; spacing: 4px; padding: 4px; }}
QToolButton, QPushButton {{
    background: {p['panel_alt']}; color: {p['text']};
    border: 1px solid {p['frame']}; border-radius: 4px; padding: 5px 10px;
}}
QToolButton:hover, QPushButton:hover {{ border-color: {p['accent']}; }}
QToolButton:pressed, QPushButton:pressed {{ background: {p['accent_soft']}; }}
QToolButton:disabled, QPushButton:disabled {{ color: {p['text_dim']}; background: {p['panel']}; }}
QPushButton#Primary {{ border-color: {p['accent']}; color: {p['accent']}; }}
QPushButton#Danger {{ border-color: {p['fail']}; color: {p['fail']}; }}
QToolButton#CardFold, QToolButton#CardFold:hover, QToolButton#CardFold:pressed {{
    background: transparent; border: none; padding: 0px; color: {p['text_dim']};
}}
QToolButton#CardFold:hover {{ color: {p['accent']}; }}
QGroupBox#Card[folded="true"] {{
    background: {p['panel_alt']}; padding-top: 0px;
}}
QGroupBox#Card[folded="true"]::title {{ color: {p['accent']}; }}

QLineEdit, QPlainTextEdit#Editable, QComboBox, QSpinBox, QDoubleSpinBox {{
    background: {p['window']}; color: {p['text']};
    border: 1px solid {p['frame']}; border-radius: 4px; padding: 4px;
}}
QComboBox:focus, QLineEdit:focus {{ border-color: {p['accent']}; }}
QComboBox QAbstractItemView {{
    background: {p['panel_alt']}; border: 1px solid {p['frame']};
    selection-background-color: {p['accent_soft']};
}}

QSplitter::handle {{ background: {p['frame']}; }}
QScrollBar:vertical {{ background: {p['panel']}; width: 11px; }}
QScrollBar::handle:vertical {{ background: {p['frame']}; border-radius: 5px; min-height: 24px; }}
QScrollBar:horizontal {{ background: {p['panel']}; height: 11px; }}
QScrollBar::handle:horizontal {{ background: {p['frame']}; border-radius: 5px; min-width: 24px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}

QDockWidget {{ color: {p['text_dim']}; titlebar-close-icon: none; titlebar-normal-icon: none; }}
QDockWidget::title {{ background: {p['panel_alt']}; padding: 5px; }}

QProgressBar {{
    border: 1px solid {p['frame']}; border-radius: 4px; background: {p['window']};
    text-align: center; color: {p['text_dim']};
}}
QProgressBar::chunk {{ background: {p['accent_soft']}; }}
"""
