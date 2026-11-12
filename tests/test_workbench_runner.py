"""Command runners of the workbench: what can run here, and what the UI may offer.

Covers:
  1. The PYTHONPATH a node runner must inject, mirroring the isolation invariant.
  2. Verdict extraction and log level classification of a raw output line.
  3. Verilator availability: disabled in the interface on Windows, with the make line to use.
  4. Make and Verilator command plans, on a simulated Linux host and on this Windows host.
  5. The plan table the diagnostics card renders, and the worker classes the panels drive.
  6. The artifact export plan: the script it runs, the runs it finds, and what blocks it.
"""

import os
import sys
from pathlib import Path

import pytest
from PyQt6.QtCore import QThread

from gui.workers import runner
from gui.workers.runner import (
    EXPORT_SCRIPT,
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
    export_script_path,
    find_openlane_runs,
    newest_openlane_run,
    plan_table,
    probe_verilator,
    python_path_for,
    verdict_of_line,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def linux_host(monkeypatch):
    """Pretend the workbench runs on a host that can launch Verilator directly."""
    monkeypatch.setattr(runner.sys, "platform", "linux")
    monkeypatch.setattr(runner.shutil, "which", lambda name: f"/usr/bin/{name}")


@pytest.fixture
def windows_host(monkeypatch):
    """Pretend the plain Windows case: no make, no Verilator on PATH."""
    monkeypatch.setattr(runner.sys, "platform", "win32")
    monkeypatch.setattr(runner.shutil, "which", lambda name: None)


def test_the_workspace_source_folder_goes_first_on_pythonpath(monkeypatch):
    monkeypatch.delenv("PYTHONPATH", raising=False)
    assert python_path_for(REPO_ROOT) == str(REPO_ROOT / "src")
    monkeypatch.setenv("PYTHONPATH", f"somewhere{os.pathsep}elsewhere")
    value = python_path_for(REPO_ROOT)
    assert value.startswith(str(REPO_ROOT / "src") + os.pathsep)
    assert value.endswith("elsewhere"), "an existing value is kept behind the injected entry"


def test_only_the_final_verdict_line_is_read_as_a_verdict():
    assert verdict_of_line("FINAL VERDICT: PASS") == "PASS"
    assert verdict_of_line("final verdict: fail") == "FAIL"
    assert verdict_of_line("Verdict : PASS") == "PASS"
    assert verdict_of_line("reading the scoreboard") is None
    assert verdict_of_line("") is None


def test_output_lines_are_mapped_to_the_levels_the_console_understands():
    assert classify_line("FINAL VERDICT: PASS") == "ok"
    assert classify_line("FINAL VERDICT: FAIL") == "error"
    assert classify_line("%Warning-WIDTH: nothing matches") == "warn"
    assert classify_line("Traceback (most recent call last):") == "error"
    assert classify_line("%Error: cannot open file") == "error"
    assert classify_line("Info: nothing unusual here") == "info"
    for line in ("FINAL VERDICT: PASS", "%Warning-X", "plain text", "boom failed"):
        assert classify_line(line) in ("ok", "warn", "error", "info")


def test_a_plan_tells_its_command_and_why_it_is_blocked():
    ready = ToolPlan(label="lint", program="make", args=["-f", "Makefile", "lint-topology"])
    assert ready.command_text == "make -f Makefile lint-topology"
    assert ready.available is True and ready.blocked_because() == ""
    blocked = ToolPlan(label="sim", program="verilator", available=False, reason="not installed")
    assert blocked.blocked_because() == "not installed"
    silent = ToolPlan(label="odd", program="x", available=False)
    assert "odd" in silent.blocked_because()


def test_the_make_targets_of_the_repository_are_the_documented_four():
    assert set(MAKE_TARGETS) == {"lint-topology", "sim-topology", "sim-tb", "export"}
    assert all(description.strip() for description in MAKE_TARGETS.values())


def test_verilator_is_disabled_in_the_interface_on_windows():
    plan = probe_verilator(REPO_ROOT)
    if not sys.platform.startswith("win"):
        pytest.skip("this host is not Windows: the interface does not disable Verilator here")
    assert plan.available is False
    assert "Windows" in plan.reason and "WSL" in plan.reason and "Docker" in plan.reason
    assert "asic_chip/Makefile sim-topology" in plan.hint
    assert plan.command_text == "verilator"
    assert plan.blocked_because() == plan.reason


def test_verilator_is_planned_when_a_directly_usable_host_has_it(linux_host):
    plan = probe_verilator(REPO_ROOT)
    assert plan.available is True
    assert plan.program == "/usr/bin/verilator"
    assert plan.cwd == str(REPO_ROOT)
    assert "found at" in plan.reason


def test_verilator_is_reported_missing_when_the_executable_is_absent(linux_host, monkeypatch):
    monkeypatch.setattr(runner.shutil, "which", lambda name: None)
    plan = probe_verilator(REPO_ROOT)
    assert plan.available is False
    assert "PATH" in plan.reason


def test_a_direct_testbench_run_mirrors_the_makefile_flags(linux_host):
    source = REPO_ROOT / "asic_chip" / "sim" / "topology_testbench.sv"
    plan = build_verilator_plan(REPO_ROOT, source, top_module="topology_testbench")
    assert plan.available is True
    assert plan.args[:3] == ["--binary", "--timing", "-Mdir"]
    assert "--top-module" in plan.args and "topology_testbench" in plan.args
    assert plan.args[-1] == str(source)
    assert plan.cwd == str(REPO_ROOT / "asic_chip")
    assert "topology_testbench.sv" in plan.label


def test_a_direct_testbench_run_stays_blocked_on_windows(windows_host):
    source = REPO_ROOT / "asic_chip" / "sim" / "topology_testbench.sv"
    plan = build_verilator_plan(REPO_ROOT, source)
    assert plan.available is False
    assert plan.args == []
    assert plan.hint


def test_make_plans_point_at_the_asic_makefile(linux_host):
    plan = build_make_plan(REPO_ROOT, "sim-topology")
    assert plan.program == "/usr/bin/make"
    assert plan.args == ["-f", str(REPO_ROOT / "asic_chip" / "Makefile"), "sim-topology"]
    assert plan.cwd == str(REPO_ROOT)
    assert plan.available is True and plan.reason == ""


def test_a_make_plan_says_when_the_makefile_is_absent(linux_host):
    plan = build_make_plan(REPO_ROOT / "no_such_workspace_folder", "export")
    assert plan.available is False
    assert "is missing" in plan.reason


def test_windows_without_make_offers_the_make_line_itself(windows_host):
    plan = build_make_plan(REPO_ROOT, "lint-topology")
    assert plan.available is False
    assert "make executable" in plan.reason
    assert plan.hint == "make -f asic_chip/Makefile lint-topology"
    assert plan.args == []


def test_the_diagnostics_table_reads_availability_out_of_the_plans():
    plans = [
        ToolPlan(label="ready one", program="make", reason=""),
        ToolPlan(label="blocked one", program="x", available=False, reason="no tool"),
        ToolPlan(label="hinted one", program="y", available=False, hint="try make"),
        ToolPlan(label="silent one", program="z", available=False),
    ]
    table = plan_table(plans)
    assert table["ready one"] == ""
    assert table["blocked one"] == "no tool"
    assert table["hinted one"] == "try make"
    assert "not available" in table["silent one"]


def test_the_workers_are_threads_so_the_interface_never_freezes():
    for worker in (CommandWorker, ScanWorker, ParseWorker):
        assert issubclass(worker, QThread)
        assert "run" in vars(worker), "the work happens off the user interface thread"
    assert hasattr(CommandWorker, "stop")
    assert hasattr(CommandWorker, "tail")


# --- the artifact export of section 4 ---------------------------------------------
@pytest.fixture
def export_workspace(tmp_path):
    """A workspace holding the export script and one OpenLane run directory."""
    script = tmp_path / "asic_chip" / "scripts" / "export_artifacts.py"
    script.parent.mkdir(parents=True)
    script.write_text("# the repository export script\n", encoding="utf-8")
    runs = tmp_path / "asic_chip" / "openlane" / "runs"
    (runs / "2026.01.04_09-00-00").mkdir(parents=True)
    return tmp_path


def test_the_repository_ships_the_export_script_the_plan_wants_to_run():
    assert EXPORT_SCRIPT == "asic_chip/scripts/export_artifacts.py"
    assert export_script_path(REPO_ROOT) == REPO_ROOT / "asic_chip" / "scripts" / "export_artifacts.py"
    if not (REPO_ROOT / EXPORT_SCRIPT).is_file():
        pytest.skip("this checkout has no export script")
    assert build_export_plan(REPO_ROOT, dry_run=True).available is True


def test_the_export_plan_runs_the_script_with_the_interpreter_of_the_workbench(export_workspace):
    plan = build_export_plan(export_workspace)
    assert plan.available is True
    assert plan.program == sys.executable, "the export moves files, so it runs on any host"
    assert plan.args[0] == str(export_workspace / EXPORT_SCRIPT)
    assert plan.cwd == str(export_workspace)
    assert plan.label == "export artifacts"
    assert "exporting 2026.01.04_09-00-00" in plan.reason
    assert "--dry-run" not in plan.args


def test_the_plan_check_variant_adds_the_flags_that_write_nothing(export_workspace):
    plan = build_export_plan(export_workspace, dry_run=True)
    assert plan.available is True
    assert "dry run" in plan.label
    assert "--dry-run" in plan.args and "--allow-missing" in plan.args


def test_the_openlane_runs_are_found_oldest_first(export_workspace):
    runs = export_workspace / "asic_chip" / "openlane" / "runs"
    older = runs / "2025.12.01_10-00-00"
    older.mkdir()
    os.utime(runs / "2026.01.04_09-00-00", (1_600_000_000, 1_600_000_000))
    os.utime(older, (1_700_000_000, 1_700_000_000))
    (runs / "_scratch").mkdir()
    (runs / ".hidden").mkdir()
    found = find_openlane_runs(export_workspace)
    assert [Path(entry).name for entry in found] == ["2026.01.04_09-00-00", "2025.12.01_10-00-00"]
    assert newest_openlane_run(export_workspace).endswith("2025.12.01_10-00-00")
    assert "2 OpenLane run directories" in export_note(export_workspace)


def test_an_export_with_no_run_to_copy_says_so_before_starting(tmp_path):
    script = tmp_path / "asic_chip" / "scripts" / "export_artifacts.py"
    script.parent.mkdir(parents=True)
    script.write_text("# placeholder\n", encoding="utf-8")
    assert find_openlane_runs(tmp_path) == []
    assert "no OpenLane run" in export_note(tmp_path)
    plan = build_export_plan(tmp_path)
    assert plan.available is False
    assert "no OpenLane run under asic_chip/openlane/runs" in plan.reason
    assert "make -f asic_chip/Makefile openlane" in plan.hint
    check = build_export_plan(tmp_path, dry_run=True)
    assert check.available is True, "the dry run still answers what would be exported"


def test_the_export_plan_is_blocked_when_the_script_is_absent(tmp_path):
    plan = build_export_plan(tmp_path)
    assert plan.available is False
    assert "export script is missing" in plan.reason
    assert plan.hint == "python3 asic_chip/scripts/export_artifacts.py --dry-run"
    assert plan.args == []


def test_an_explicit_run_or_runs_directory_makes_the_plan_runnable(export_workspace):
    plan = build_export_plan(export_workspace, run="2026.01.04_09-00-00")
    assert plan.available is True
    assert "--run" in plan.args and plan.args[plan.args.index("--run") + 1] == "2026.01.04_09-00-00"
    elsewhere = build_export_plan(export_workspace, runs_dir=str(export_workspace / "elsewhere"))
    assert elsewhere.available is True
    assert "--runs-dir" in elsewhere.args
    assert "exporting" in elsewhere.reason


def test_the_export_options_write_the_flags_in_the_order_the_script_expects():
    options = ExportOptions(dry_run=True, desktop=True, run="r1", runs_dir="d1")
    assert options.as_args() == ["--runs-dir", "d1", "--run", "r1", "--dry-run", "--allow-missing", "--desktop"]
    assert ExportOptions().as_args() == []
