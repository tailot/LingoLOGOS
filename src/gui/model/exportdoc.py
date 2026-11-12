"""The export manifest: what the fabrication and board export really wrote.

The workbench shows the production tree of section 4 of the interface contract the
same way it shows everything else, by reading it from disk. The export script that
the workspace profile names writes one ``manifest.json`` next to the two subtrees it
produces, and that file is the only honest description of the export: it records,
per artifact, the destination path, whether the file was copied or generated, where
it came from, how big it is, and its digest, and it says how many required artifacts
are missing.

This module turns that document into plain Python objects. It is Qt free like the
rest of :mod:`gui.model`, so the same reading code answers a widget, a pytest and a
command line question. It never repairs a number: an entry the export wrote as
MISSING is reported as MISSING here too.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

MANIFEST_NAME = "manifest.json"
MANIFEST_SCHEMA = "lingologos-export-v1"

STATUS_COPIED = "COPIED"
STATUS_GENERATED = "GENERATED"
STATUS_MISSING = "MISSING"
KNOWN_STATUSES = (STATUS_COPIED, STATUS_GENERATED, STATUS_MISSING)

FACTORY_FAMILY = "factory"
BOARD_FAMILY = "visualization_board"
FAMILY_LABELS = {
    FACTORY_FAMILY: "foundry and signoff",
    BOARD_FAMILY: "board and simulation",
}

STATUS_TONE = {STATUS_COPIED: "pass", STATUS_GENERATED: "pass", STATUS_MISSING: "fail"}


def manifest_path(root: Any) -> Path:
    """The manifest of an exported tree, given the workspace or export root."""
    candidate = Path(str(root))
    direct = candidate / MANIFEST_NAME
    if direct.is_file():
        return direct
    return candidate / "export_artifacts" / MANIFEST_NAME


def family_of(artifact: str) -> str:
    """Export subtree of an artifact path: factory or visualization_board."""
    parts = [part for part in str(artifact).replace("\\", "/").split("/") if part]
    if not parts:
        return ""
    head = parts[0].lower()
    if head == FACTORY_FAMILY or head.startswith(FACTORY_FAMILY):
        return FACTORY_FAMILY
    if head == BOARD_FAMILY or head.startswith("board"):
        return BOARD_FAMILY
    return head


def group_of(artifact: str) -> str:
    """Two level group used by the tree: factory/gds, factory/reports, and so on."""
    parts = [part for part in str(artifact).replace("\\", "/").split("/") if part]
    if len(parts) >= 2:
        return f"{parts[0]}/{parts[1]}"
    return parts[0] if parts else ""


def tone_of(status: str) -> str:
    """The console tone the interface colours an export status with."""
    return STATUS_TONE.get(str(status).upper(), "warn")


# --------------------------------------------------------------------------- entries
@dataclass
class ExportEntry:
    """One row of the manifest: an artifact, its state, and where it came from."""

    artifact: str
    status: str = STATUS_MISSING
    purpose: str = ""
    source: str = ""
    size_bytes: Optional[int] = None
    sha256: str = ""
    optional: bool = False
    generated_from: List[str] = field(default_factory=list)
    looked_at: List[str] = field(default_factory=list)

    @property
    def name(self) -> str:
        return str(self.artifact).replace("\\", "/").rsplit("/", 1)[-1]

    @property
    def family(self) -> str:
        return family_of(self.artifact)

    @property
    def group(self) -> str:
        return group_of(self.artifact)

    @property
    def present(self) -> bool:
        return self.status in (STATUS_COPIED, STATUS_GENERATED)

    @property
    def tone(self) -> str:
        return tone_of(self.status)

    @property
    def is_gap(self) -> bool:
        """A missing artifact that the export contract still requires."""
        return self.status == STATUS_MISSING and not self.optional

    def describe_origin(self) -> str:
        """Where the bytes came from, in the words the export itself used."""
        if self.source:
            return self.source
        if self.generated_from:
            return "generated from " + ", ".join(Path(str(p)).name for p in self.generated_from)
        if self.looked_at:
            return "looked for in " + ", ".join(str(p) for p in self.looked_at)
        return self.purpose

    def describe_size(self) -> str:
        if self.size_bytes is None:
            return "-"
        value = float(self.size_bytes)
        for unit in ("B", "KB", "MB", "GB"):
            if value < 1024.0 or unit == "GB":
                return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
            value /= 1024.0
        return f"{self.size_bytes} B"

    def as_row(self) -> List[str]:
        """The four columns the export table shows."""
        return [self.artifact, self.status, self.describe_size(), self.describe_origin()]

    def as_digest_row(self) -> List[str]:
        return [self.name, self.status, self.sha256[:16] or "-", self.purpose]


@dataclass
class ExportManifest:
    """The whole export document, plus the counts the interface reports as chips."""

    schema: str = MANIFEST_SCHEMA
    generated_utc: str = ""
    tool: str = ""
    design: str = ""
    destination: str = ""
    destination_host_path: str = ""
    run: Dict[str, Any] = field(default_factory=dict)
    entries: List[ExportEntry] = field(default_factory=list)
    missing: int = 0
    missing_required: int = 0
    path: Optional[Path] = None

    # --- what the export says about itself -----------------------------------
    @property
    def is_export(self) -> bool:
        return bool(self.entries) or self.schema == MANIFEST_SCHEMA

    @property
    def total(self) -> int:
        return len(self.entries)

    @property
    def present(self) -> int:
        return sum(1 for entry in self.entries if entry.present)

    @property
    def complete(self) -> bool:
        """Nothing required is missing, and at least one artifact came out."""
        return self.total > 0 and self.missing_required_count == 0

    @property
    def missing_required_count(self) -> int:
        return sum(1 for entry in self.entries if entry.is_gap)

    def counts(self) -> Dict[str, int]:
        """Artifact count per status, always carrying the three documented keys."""
        table = {status: 0 for status in KNOWN_STATUSES}
        for entry in self.entries:
            table[entry.status] = table.get(entry.status, 0) + 1
        return table

    # --- the run the export was taken from ------------------------------------
    def run_name(self) -> str:
        return str(self.run.get("name") or "-")

    def run_directory(self) -> str:
        return str(self.run.get("directory") or "-")

    def run_detail(self) -> str:
        """One line about the OpenLane run: version, PDK and clock, as recorded."""
        bits: List[str] = []
        for key, label in (
            ("openlane_version", "openlane"),
            ("pdk", "pdk"),
            ("clock_period", "clock ns"),
            ("final_views_present", "final views"),
        ):
            value = self.run.get(key)
            if value is None or value == "":
                continue
            bits.append(f"{label} {value}")
        return ", ".join(bits) if bits else "no run detail recorded"

    @property
    def run_is_present(self) -> bool:
        value = self.run.get("final_views_present")
        return bool(value) if isinstance(value, bool) else bool(self.run.get("name"))

    # --- grouped views --------------------------------------------------------
    def by_group(self) -> Dict[str, List[ExportEntry]]:
        """Entries grouped by export subtree, in the order the manifest wrote them."""
        groups: Dict[str, List[ExportEntry]] = {}
        for entry in self.entries:
            groups.setdefault(entry.group, []).append(entry)
        return groups

    def families(self) -> List[Tuple[str, int, int]]:
        """Rows of family, present, total, for the two subtrees and any other one."""
        table: Dict[str, List[int]] = {}
        for entry in self.entries:
            row = table.setdefault(entry.family or "other", [0, 0])
            row[1] += 1
            if entry.present:
                row[0] += 1
        return [(family, row[0], row[1]) for family, row in table.items()]

    def gap_entries(self) -> List[ExportEntry]:
        return [entry for entry in self.entries if entry.status == STATUS_MISSING]

    def summary_rows(self) -> List[List[str]]:
        """Rows for the compact table: family, present of total, label."""
        rows: List[List[str]] = []
        for family, present, total in self.families():
            label = FAMILY_LABELS.get(family, family)
            rows.append([label, f"{present} of {total}", family])
        return rows

    def headline(self) -> str:
        """One line the console and a chip can print verbatim."""
        counts = self.counts()
        bits = [
            f"design {self.design or '-'}",
            f"run {self.run_name()}",
            (
                f"copied {counts[STATUS_COPIED]}, generated {counts[STATUS_GENERATED]}, "
                f"missing {counts[STATUS_MISSING]}"
            ),
        ]
        return " | ".join(bits)

    def as_text(self) -> str:
        """Plain text dump of the manifest, for a log console or a terminal."""
        lines = [
            f"export manifest schema {self.schema}",
            f"generated {self.generated_utc or '-'} by {self.tool or '-'}",
            f"design {self.design or '-'} from run {self.run_name()}",
            self.run_detail(),
            f"destination {self.destination or '-'}",
        ]
        if self.destination_host_path:
            lines.append(f"same tree on the host at {self.destination_host_path}")
        for group, entries in self.by_group().items():
            lines.append(group)
            for entry in entries:
                lines.append(f"  {entry.status:<9} {entry.artifact} {entry.describe_size()}")
                if entry.purpose:
                    lines.append(f"  {'':<9} {entry.purpose}")
        lines.append(self.headline())
        lines.append(
            "required artifacts missing: %d" % self.missing_required_count
        )
        return "\n".join(lines)


# --------------------------------------------------------------------------- reading
def _clean_text(value: Any) -> str:
    return "" if value is None else str(value)


def _clean_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def parse_entry(data: Any) -> ExportEntry:
    """One manifest entry, tolerant of a key the export script did not write."""
    if isinstance(data, str):
        return ExportEntry(artifact=data, status=STATUS_MISSING)
    record = data if isinstance(data, dict) else {}
    raw_generated = record.get("generated_from") or []
    raw_look = record.get("looked_at") or []
    status = _clean_text(record.get("status")).upper() or STATUS_MISSING
    return ExportEntry(
        artifact=_clean_text(record.get("artifact") or record.get("dest")),
        status=status if status in KNOWN_STATUSES else STATUS_MISSING,
        purpose=_clean_text(record.get("purpose")),
        source=_clean_text(record.get("source")),
        size_bytes=_clean_int(record.get("size_bytes")),
        sha256=_clean_text(record.get("sha256")),
        optional=bool(record.get("optional")),
        generated_from=[_clean_text(item) for item in raw_generated] if isinstance(raw_generated, list) else [],
        looked_at=[_clean_text(item) for item in raw_look] if isinstance(raw_look, list) else [],
    )


def parse_manifest(data: Any, path: Optional[Path] = None) -> ExportManifest:
    """Build the manifest object out of the decoded JSON document."""
    record = data if isinstance(data, dict) else {}
    raw_entries = record.get("entries")
    entries = [parse_entry(item) for item in raw_entries] if isinstance(raw_entries, list) else []
    raw_run = record.get("run")
    return ExportManifest(
        schema=_clean_text(record.get("schema")) or MANIFEST_SCHEMA,
        generated_utc=_clean_text(record.get("generated_utc")),
        tool=_clean_text(record.get("tool")),
        design=_clean_text(record.get("design")),
        destination=_clean_text(record.get("destination")),
        destination_host_path=_clean_text(record.get("destination_host_path")),
        run=dict(raw_run) if isinstance(raw_run, dict) else {},
        entries=entries,
        missing=_clean_int(record.get("missing")) or 0,
        missing_required=_clean_int(record.get("missing_required")) or 0,
        path=Path(path) if path is not None else None,
    )


def read_manifest(path: Any) -> Optional[ExportManifest]:
    """Read one manifest file, or return None when it is absent or unreadable."""
    target = Path(str(path))
    if not target.is_file():
        return None
    try:
        with open(target, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    manifest = parse_manifest(data, target)
    return manifest if manifest.is_export else None


def load_manifest(root: Any) -> Optional[ExportManifest]:
    """Read the manifest of a workspace, from the export tree it documents."""
    return read_manifest(manifest_path(root))


__all__ = [
    "BOARD_FAMILY",
    "FACTORY_FAMILY",
    "FAMILY_LABELS",
    "KNOWN_STATUSES",
    "MANIFEST_NAME",
    "MANIFEST_SCHEMA",
    "STATUS_COPIED",
    "STATUS_GENERATED",
    "STATUS_MISSING",
    "ExportEntry",
    "ExportManifest",
    "family_of",
    "group_of",
    "load_manifest",
    "manifest_path",
    "parse_entry",
    "parse_manifest",
    "read_manifest",
    "tone_of",
]
