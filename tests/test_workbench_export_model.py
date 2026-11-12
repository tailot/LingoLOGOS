"""The export manifest model: reading what the section 4 export really wrote.

Covers:
  1. The manifest contract keys the export script writes and this model reads back.
  2. Entry level views: origin wording, size wording, present flags, gap flags, tones.
  3. Manifest level counts: per status, per group, per family, required gaps.
  4. Reading from disk: the manifest path of a workspace, and absent or broken files.
  5. Tolerance: a half written entry still becomes a row, and never invents an artifact.
"""

import json
from pathlib import Path

import pytest

from gui.model.exportdoc import (
    BOARD_FAMILY,
    FACTORY_FAMILY,
    KNOWN_STATUSES,
    MANIFEST_NAME,
    MANIFEST_SCHEMA,
    STATUS_COPIED,
    STATUS_GENERATED,
    STATUS_MISSING,
    ExportEntry,
    ExportManifest,
    family_of,
    group_of,
    load_manifest,
    manifest_path,
    parse_entry,
    parse_manifest,
    read_manifest,
    tone_of,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPORT_SCRIPT = REPO_ROOT / "asic_chip" / "scripts" / "export_artifacts.py"


def sample_document() -> dict:
    """A manifest shaped exactly like the one export_artifacts.py writes."""
    return {
        "schema": MANIFEST_SCHEMA,
        "generated_utc": "2026-01-04T09:12:33Z",
        "tool": "export_artifacts.py",
        "design": "lingologos_top",
        "run": {
            "directory": "asic_chip/openlane/runs/2026.01.04_09-00-00",
            "name": "2026.01.04_09-00-00",
            "openlane_version": "2.3.11",
            "pdk": "sky130A",
            "clock_period": "11.000",
            "signoff_sdc_file": "factory/sdc/lingologos_top.sdc",
            "final_views_present": True,
            "flow_error_log_bytes": 0,
        },
        "destination": "export_artifacts",
        "destination_host_path": "/home/user/LingoLOGOS/export_artifacts",
        "entries": [
            {
                "artifact": "factory/gds/lingologos_top.gds",
                "status": STATUS_COPIED,
                "purpose": "tapeout layout, the GDSII the streamout step wrote",
                "source": "runs/.../67-final/52-gds/lygds/lingologos_top.gds",
                "size_bytes": 2048576,
                "sha256": "a" * 64,
                "optional": False,
            },
            {
                "artifact": "factory/reports/metrics.json",
                "status": STATUS_GENERATED,
                "purpose": "per corner setup and hold summary taken from the run",
                "generated_from": ["metrics.json"],
                "size_bytes": 512,
                "sha256": "b" * 64,
                "optional": False,
            },
            {
                "artifact": "factory/lvs/lingologos_top.lvs.rpt",
                "status": STATUS_MISSING,
                "purpose": "LVS report of the signoff LVS step",
                "looked_at": ["runs/.../signoff/lvs/report", "runs/.../lvs"],
                "optional": True,
            },
            {
                "artifact": "visualization_board/netlists/lingologos_synth.v",
                "status": STATUS_GENERATED,
                "purpose": "gate level netlist of the design, for the board viewer",
                "generated_from": ["lingologos_top.v"],
                "size_bytes": 4096,
                "sha256": "c" * 64,
                "optional": False,
            },
            {
                "artifact": "visualization_board/system_board/board_complete.v",
                "status": STATUS_GENERATED,
                "purpose": "board level structural wrapper with the dies of topology.json",
                "generated_from": ["topology.json", "lingologos_top.v"],
                "size_bytes": 9000,
                "sha256": "d" * 64,
                "optional": False,
            },
        ],
        "missing": 1,
        "missing_required": 0,
    }


@pytest.fixture
def manifest() -> ExportManifest:
    return parse_manifest(sample_document())


# --- the contract on both sides -------------------------------------------------
def test_the_model_reads_the_same_schema_and_statuses_the_export_writes():
    assert MANIFEST_NAME == "manifest.json"
    assert MANIFEST_SCHEMA == "lingologos-export-v1"
    assert KNOWN_STATUSES == (STATUS_COPIED, STATUS_GENERATED, STATUS_MISSING)
    if not EXPORT_SCRIPT.is_file():
        pytest.skip("this checkout has no export script to compare the contract with")
    source = EXPORT_SCRIPT.read_text(encoding="utf-8")
    assert '"schema": "%s"' % MANIFEST_SCHEMA in source
    for status in KNOWN_STATUSES:
        assert '"%s"' % status in source
    for key in ("artifact", "purpose", "missing_required"):
        assert '"%s"' % key in source
    for key in ("generated_from", "looked_at", "size_bytes", "sha256", "optional"):
        assert key in source


# --- entries --------------------------------------------------------------------
def test_families_and_groups_split_the_two_documented_subtrees():
    assert family_of("factory/gds/x.gds") == FACTORY_FAMILY
    assert family_of("visualization_board/netlists/x.v") == BOARD_FAMILY
    assert family_of("other/thing") == "other"
    assert group_of("factory/reports/metrics.json") == "factory/reports"
    assert group_of("factory") == "factory"
    assert group_of("") == ""


def test_status_tone_and_presence_are_the_same_everywhere():
    assert tone_of(STATUS_COPIED) == "pass"
    assert tone_of(STATUS_GENERATED) == "pass"
    assert tone_of(STATUS_MISSING) == "fail"
    assert tone_of("weird") == "warn"
    copied = ExportEntry("factory/gds/a.gds", STATUS_COPIED)
    gap = ExportEntry("factory/lvs/a.rpt", STATUS_MISSING)
    assert copied.present is True and gap.present is False
    assert gap.is_gap is True and gap.tone == "fail"
    assert ExportEntry("x", STATUS_MISSING, optional=True).is_gap is False


def test_an_entry_says_where_its_bytes_came_from_in_the_export_words():
    copied = ExportEntry("factory/gds/a.gds", STATUS_COPIED, source="runs/x/gds/a.gds")
    assert copied.describe_origin() == "runs/x/gds/a.gds"
    made = ExportEntry(
        "visualization_board/system_board/board_complete.v",
        STATUS_GENERATED,
        generated_from=["topology.json", "lingologos_top.v"],
    )
    assert made.describe_origin() == "generated from topology.json, lingologos_top.v"
    missing = ExportEntry("factory/lvs/a.rpt", STATUS_MISSING, looked_at=["runs/a", "runs/b"])
    assert missing.describe_origin() == "looked for in runs/a, runs/b"
    assert ExportEntry("x", STATUS_MISSING, purpose="nothing found").describe_origin() == "nothing found"


def test_sizes_are_reported_as_the_manifest_records_them():
    assert ExportEntry("x", STATUS_COPIED, size_bytes=None).describe_size() == "-"
    assert ExportEntry("x", STATUS_COPIED, size_bytes=0).describe_size() == "0 B"
    assert ExportEntry("x", STATUS_COPIED, size_bytes=1024).describe_size() == "1.0 KB"
    assert ExportEntry("x", STATUS_COPIED, size_bytes=2048576).describe_size() == "2.0 MB"
    assert ExportEntry("x", STATUS_COPIED, size_bytes=3 * 1024 ** 3).describe_size() == "3.0 GB"


def test_a_row_carries_the_four_columns_the_export_table_shows():
    entry = ExportEntry("factory/gds/a.gds", STATUS_COPIED, source="runs/x/a.gds", size_bytes=10)
    assert entry.as_row() == ["factory/gds/a.gds", STATUS_COPIED, "10 B", "runs/x/a.gds"]
    digest = ExportEntry("factory/gds/a.gds", STATUS_COPIED, sha256="e" * 64, purpose="layout")
    assert digest.as_digest_row() == ["a.gds", STATUS_COPIED, "e" * 16, "layout"]
    assert ExportEntry("factory/gds/a.gds").as_digest_row()[2] == "-"


# --- the whole document ---------------------------------------------------------
def test_the_manifest_counts_artifacts_per_status_and_never_invents_one(manifest):
    assert manifest.is_export is True
    assert manifest.total == 5
    assert manifest.present == 4
    assert manifest.counts() == {
        STATUS_COPIED: 1,
        STATUS_GENERATED: 3,
        STATUS_MISSING: 1,
    }
    assert manifest.gap_entries()[0].artifact == "factory/lvs/lingologos_top.lvs.rpt"
    assert manifest.missing_required_count == 0
    assert manifest.complete is True


def test_run_identity_comes_from_the_run_block_of_the_manifest(manifest):
    assert manifest.run_name() == "2026.01.04_09-00-00"
    assert manifest.run_directory().endswith("2026.01.04_09-00-00")
    assert manifest.run_is_present is True
    detail = manifest.run_detail()
    assert "openlane 2.3.11" in detail and "pdk sky130A" in detail and "clock ns 11.000" in detail
    assert ExportManifest().run_name() == "-"
    assert ExportManifest().run_detail() == "no run detail recorded"


def test_entries_group_by_subtree_and_the_families_summarise_them(manifest):
    groups = manifest.by_group()
    assert list(groups) == [
        "factory/gds",
        "factory/reports",
        "factory/lvs",
        "visualization_board/netlists",
        "visualization_board/system_board",
    ]
    assert [entry.status for entry in groups["factory/gds"]] == [STATUS_COPIED]
    assert manifest.families() == [(FACTORY_FAMILY, 2, 3), (BOARD_FAMILY, 2, 2)]
    assert manifest.summary_rows() == [
        ["foundry and signoff", "2 of 3", FACTORY_FAMILY],
        ["board and simulation", "2 of 2", BOARD_FAMILY],
    ]


def test_the_headline_and_the_text_dump_state_the_facts_of_the_export(manifest):
    headline = manifest.headline()
    assert "design lingologos_top" in headline
    assert "run 2026.01.04_09-00-00" in headline
    assert "copied 1, generated 3, missing 1" in headline
    text = manifest.as_text()
    assert text.startswith("export manifest schema " + MANIFEST_SCHEMA)
    assert "generated 2026-01-04T09:12:33Z by export_artifacts.py" in text
    assert "same tree on the host at /home/user/LingoLOGOS/export_artifacts" in text
    assert "factory/gds/lingologos_top.gds 2.0 MB" in text
    assert "tapeout layout, the GDSII the streamout step wrote" in text
    assert "required artifacts missing: 0" in text


def test_a_run_with_a_missing_required_artifact_is_not_complete():
    document = sample_document()
    document["entries"][0]["status"] = STATUS_MISSING
    document["missing"] = 2
    document["missing_required"] = 1
    broken = parse_manifest(document)
    assert broken.complete is False
    assert broken.missing_required_count == 1


# --- reading the file from disk -------------------------------------------------
def test_manifest_path_finds_the_manifest_of_a_workspace(tmp_path):
    assert manifest_path(tmp_path) == tmp_path / "export_artifacts" / MANIFEST_NAME
    export = tmp_path / "export_artifacts"
    export.mkdir()
    (export / MANIFEST_NAME).write_text("{}", encoding="utf-8")
    assert manifest_path(tmp_path) == export / MANIFEST_NAME
    (tmp_path / MANIFEST_NAME).write_text("{}", encoding="utf-8")
    assert manifest_path(tmp_path) == tmp_path / MANIFEST_NAME


def test_load_manifest_reads_the_export_tree_of_a_workspace(tmp_path):
    export = tmp_path / "export_artifacts"
    export.mkdir()
    (export / MANIFEST_NAME).write_text(json.dumps(sample_document()), encoding="utf-8")
    manifest = load_manifest(tmp_path)
    assert manifest is not None
    assert manifest.total == 5
    assert manifest.design == "lingologos_top"
    assert manifest.path == export / MANIFEST_NAME


def test_an_absent_or_broken_manifest_reads_as_no_manifest_at_all(tmp_path):
    assert load_manifest(tmp_path) is None
    assert read_manifest(tmp_path / MANIFEST_NAME) is None
    broken = tmp_path / MANIFEST_NAME
    broken.write_text("{ not json", encoding="utf-8")
    assert read_manifest(broken) is None
    empty = tmp_path / "export_artifacts"
    empty.mkdir()
    (empty / MANIFEST_NAME).write_text(json.dumps({"schema": MANIFEST_SCHEMA}), encoding="utf-8")
    assert load_manifest(tmp_path) is None


# --- tolerance ------------------------------------------------------------------
def test_a_half_written_entry_becomes_a_row_without_inventing_an_artifact():
    bare = parse_entry("visualization_board/netlists/lingologos_synth.v")
    assert bare.status == STATUS_MISSING and bare.present is False
    assert bare.artifact == "visualization_board/netlists/lingologos_synth.v"
    odd = parse_entry({"artifact": "factory/gds/a.gds", "status": "MAYBE", "size_bytes": "text"})
    assert odd.status == STATUS_MISSING and odd.size_bytes is None
    assert parse_entry(None).artifact == ""
    assert parse_entry(None).status == STATUS_MISSING


def test_a_manifest_document_that_is_not_a_dict_still_reads_as_empty():
    empty = parse_manifest(["nonsense"])
    assert empty.entries == [] and empty.total == 0 and empty.is_export is True
    assert empty.counts() == {STATUS_COPIED: 0, STATUS_GENERATED: 0, STATUS_MISSING: 0}
    assert empty.headline().startswith("design -")
    assert ExportManifest(entries=[]).complete is False
