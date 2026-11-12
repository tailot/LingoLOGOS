"""Offline rule of the workbench: no web resource, anywhere in the interface.

The workbench is pure desktop: native PyQt6 widgets, QPainter drawing, Qt shipped standard
icons and local style sheets. This gate reads every source file of the package and refuses
any remote asset, any web toolkit, and any network call, with the single documented
exception of the guard list in ``gui/app.py`` that proves the rule at start up.
"""

import os
import re
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

GUI_ROOT = Path(__file__).resolve().parents[1] / "src" / "gui"

WEB_TOOLKITS = (
    "QtWebEngine",
    "QtWebEngineWidgets",
    "QtWebEngineCore",
    "QtWebKit",
    "QtWebKitWidgets",
    "QWebEngineView",
    "QQuickWebViewer",
)
NETWORK_TOKENS = (
    "urllib.request",
    "urlopen",
    "http.client",
    "socket.",
    "requests.",
    "QNetworkAccessManager",
    "QNetworkRequest",
    "QNetworkReply",
    "QDesktopServices",
)
REMOTE_ASSET_TOKENS = (
    "http://",
    "https://",
    "//cdn.",
    "googleapis",
    "unpkg",
    "jsdelivr",
    "@font-face",
    "<html",
    "<link ",
    "<script",
    "url((",
    "url('",
    'url("',
)


def gui_sources():
    return sorted(GUI_ROOT.rglob("*.py"))


def loaded_modules():
    """``gui.app`` and ``gui.theme`` are the two modules of the interface that load Qt.

    They are asked for inside the tests that need them, so a host where the Qt runtime
    cannot load skips those two and still runs every source scan of this gate.
    """
    try:
        from gui import app, theme
    except ImportError as missing:
        pytest.skip(f"Qt is not loadable here: {missing}")
    return app, theme


def test_the_workbench_package_is_present_and_its_sources_are_read():
    assert GUI_ROOT.is_dir()
    sources = gui_sources()
    assert len(sources) >= 10, "the workbench package should hold its modules"
    assert all(path.name.endswith(".py") for path in sources)


def test_no_python_source_names_a_web_toolkit_or_a_network_call():
    offenders = []
    for path in gui_sources():
        text = path.read_text(encoding="utf-8")
        if path.name == "app.py":
            # The offline guard itself has to name the forbidden modules to detect them.
            marker = text.index("WEB_MODULES")
            end = text.index(")", marker)
            text = text[:marker] + text[end:]
        for token in WEB_TOOLKITS + NETWORK_TOKENS:
            if token in text:
                offenders.append((str(path.relative_to(GUI_ROOT)), token))
    assert offenders == [], f"web or network usage found in the interface: {offenders}"


def test_no_python_source_references_a_remote_asset():
    offenders = []
    for path in gui_sources():
        text = path.read_text(encoding="utf-8")
        for token in REMOTE_ASSET_TOKENS:
            if token in text:
                offenders.append((str(path.relative_to(GUI_ROOT)), token))
    assert offenders == [], f"remote asset reference found in the interface: {offenders}"


def test_the_start_up_guard_list_covers_the_web_toolkits_of_this_qt_version():
    app, theme = loaded_modules()
    assert "PyQt6.QtWebEngineWidgets" in app.WEB_MODULES
    assert "PyQt6.QtWebKitWidgets" in app.WEB_MODULES
    assert theme.stylesheet(), "the style sheet is local text and must not be empty"
    assert app.assert_offline() == "", "a web toolkit must not be importable in this session"


def test_the_style_sheet_is_local_text_only():
    _, theme = loaded_modules()
    sheet = theme.stylesheet()
    for token in REMOTE_ASSET_TOKENS:
        assert token not in sheet
    assert "url(" not in sheet.replace("url((", "url("), "no image or font is fetched from disk or net"
    assert "gridline-color" in sheet


def test_icons_come_from_the_qt_style_and_not_from_files():
    text = "\n".join(path.read_text(encoding="utf-8") for path in gui_sources())
    assert "QIcon.fromTheme" not in text, "a theme icon lookup would need an installed icon set"
    assert "QIcon(" not in text, "icons are standard pixmaps, never loaded images"
    assert ".svg" not in text and ".png" not in text, "no image file is shipped or fetched"


def test_no_resource_folder_is_needed_by_the_interface():
    assert not (GUI_ROOT / "static").exists()
    assert not (GUI_ROOT / "assets").exists()
    assert list(GUI_ROOT.rglob("*.qrc")) == []
    assert list(GUI_ROOT.rglob("*.ttf")) == []
    assert list(GUI_ROOT.rglob("*.woff*")) == []


@pytest.mark.parametrize("name", ["index.html", "main.js", "styles.css"])
def test_the_interface_ships_no_web_document(name):
    assert list(GUI_ROOT.rglob(name)) == []
