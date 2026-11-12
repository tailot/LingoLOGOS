"""Filesystem layer of the workbench: what a workspace is and what it holds.

Covers:
  1. Workspace discovery from a nested path and the derived directory set.
  2. The artifact role inventory a panel asks a role for.
  3. The suffix families that decide which module claims a dropped file.
  4. The start up argument handling of the entry point, including the offline guard.
"""

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

from gui.paths import (
    DATASET_SUFFIXES,
    FORMAL_SUFFIXES,
    GRAMMAR_SUFFIXES,
    HDL_SUFFIXES,
    LAYOUT_SUFFIXES,
    TOPOLOGY_SUFFIXES,
    discover_workspace,
    open_workspace,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
# Roles that only ever come from tracked sources, so they must exist in any checkout.
TRACKED_ROLES = (
    "topology",
    "dataset",
    "corpus",
    "grammar",
    "formal",
    "rtl",
    "testbench",
)
# Roles that come from generated silicon artifacts: checked when the export ran.
GENERATED_ROLES = ("board", "netlist", "def", "gds", "lef", "sdc", "report", "transcript")
ARTIFACTS_DIR = REPO_ROOT / "export_artifacts"
needs_generated_artifacts = pytest.mark.skipif(
    not ARTIFACTS_DIR.is_dir(),
    reason="no export_artifacts directory in this checkout: the silicon roles were never exported",
)


@pytest.fixture(scope="module")
def workspace():
    return open_workspace(REPO_ROOT)


def test_discover_workspace_walks_up_to_the_repository_root():
    assert discover_workspace(Path(__file__).resolve().parent) == REPO_ROOT


def test_discover_workspace_refuses_a_stray_directory(tmp_path):
    with pytest.raises(FileNotFoundError):
        discover_workspace(tmp_path)


def test_open_workspace_derives_the_known_directories(workspace):
    assert workspace.root == REPO_ROOT
    assert workspace.src_dir.name == "src"
    assert workspace.config_dir.name == "config"
    assert workspace.datasets_dir.name == "datasets"
    assert workspace.makefile.name == "Makefile"
    assert workspace.asic_dir.is_dir()
    assert workspace.rtl_dir == workspace.asic_dir / "rtl"
    assert workspace.sim_dir == workspace.asic_dir / "sim"
    assert workspace.reports_dir == workspace.artifacts_dir / "factory" / "reports"


def test_scan_fills_every_role_of_a_tracked_source(workspace):
    scan = workspace.rescan() or workspace.scan
    for role in TRACKED_ROLES:
        assert scan.get(role), f"no {role} artifact found under {REPO_ROOT}"
    assert len(workspace.of_role("topology")) >= 2
    assert sum(scan.values(), [])


@needs_generated_artifacts
def test_generated_silicon_roles_are_readable_when_the_export_ran(workspace):
    scan = workspace.rescan() or workspace.scan
    for role in GENERATED_ROLES:
        assert scan.get(role), f"export_artifacts has no {role} artifact"
    assert workspace.newest("board") is not None
    assert workspace.newest("board").suffix in (".v", ".sv", ".json")


def test_unknown_role_yields_nothing_and_no_key_error(workspace):
    assert workspace.of_role("does_not_exist") == []
    assert workspace.newest("does_not_exist") is None


def test_artifact_row_leads_with_the_role_and_carries_the_name(workspace):
    artifact = workspace.newest("testbench")
    assert artifact is not None
    row = artifact.as_row()
    assert len(row) == 4
    assert row[0] == artifact.role == "testbench"
    assert row[1] == artifact.name
    assert artifact.suffix == ".sv"
    assert artifact.size_bytes > 0
    assert artifact.modified_utc.startswith("20")


def test_relative_paths_stay_inside_the_workspace(workspace):
    artifact = workspace.newest("topology")
    assert artifact is not None
    text = workspace.relative(artifact.path)
    assert text and not Path(text).is_absolute()
    assert (workspace.root / text).is_file()


def test_role_summary_is_one_row_per_non_empty_role(workspace):
    rows = workspace.role_summary()
    assert rows
    assert all(len(row) == 3 for row in rows)
    assert any(row[0] == "topology" and row[1].isdigit() for row in rows)


def test_newest_name_hint_filters_the_candidates(workspace):
    hinted = workspace.newest("topology", name_hint="federated")
    assert hinted is not None and "federated" in hinted.path.name
    assert hinted in workspace.of_role("topology")
    assert workspace.newest("topology", name_hint="no_such_file_here") is None


def test_the_manifest_of_a_tree_is_read_by_the_one_reader_of_the_workbench(workspace):
    from gui.model.exportdoc import load_manifest, manifest_path

    assert manifest_path(workspace.root) == workspace.artifacts_dir / "manifest.json"
    manifest = load_manifest(workspace.root)
    if not (workspace.artifacts_dir / "manifest.json").is_file():
        assert manifest is None
        return
    assert manifest is not None and manifest.is_export
    assert manifest.total == len(manifest.entries)


def test_suffix_families_are_the_routers_of_a_dropped_file():
    assert TOPOLOGY_SUFFIXES == (".json",)
    assert DATASET_SUFFIXES == (".json", ".jsonl", ".txt")
    assert HDL_SUFFIXES == (".sv", ".v")
    assert set(LAYOUT_SUFFIXES) == {".gds", ".lef", ".def"}
    assert set(FORMAL_SUFFIXES) == {".smt2", ".smt", ".sdc"}
    assert GRAMMAR_SUFFIXES == (".gf",)


def entry_point():
    """The start up helpers live in ``gui.app``, which loads Qt.

    Importing it here rather than at the top of the file keeps the filesystem tests above
    runnable on a host where the Qt runtime cannot load; the three tests below skip there.
    """
    try:
        from gui import app
    except ImportError as missing:
        pytest.skip(f"Qt is not loadable here: {missing}")
    return app


def test_resolve_root_accepts_the_repository_and_refuses_a_stray_path():
    app = entry_point()
    root, reason = app.resolve_root(str(REPO_ROOT))
    assert root == REPO_ROOT and reason == ""
    root, reason = app.resolve_root(str(REPO_ROOT / "definitely_not_here"))
    assert root is None and "not a directory" in reason
    root, reason = app.resolve_root("")
    assert root == REPO_ROOT and reason == ""


def test_parse_args_defaults_to_the_discovered_workspace():
    app = entry_point()
    args = app.parse_args([])
    assert args.workspace == "" and args.verbose is False and args.version is False
    args = app.parse_args(["--workspace", str(REPO_ROOT), "--verbose"])
    assert args.workspace == str(REPO_ROOT) and args.verbose is True


def test_offline_guard_reports_no_web_toolkit_in_this_process():
    app = entry_point()
    assert app.assert_offline() == ""
