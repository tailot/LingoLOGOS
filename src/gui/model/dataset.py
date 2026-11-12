"""Dataset and training payload model: mixed format, three editable fields.

A training corpus in this repository arrives as a JSON array of payloads, a JSONL
stream, a GF style text corpus, or a single payload file. All four shapes land in one
document type whose records expose the same three editable fields:

    natural prompt      the surface sentence the model is trained on
    target GF grammar   the formal target the parser must produce
    SMT-LIB constraint  the formal assertion set that must stay UNSAT

The field names shown to the person are English labels, while the keys and values
inside the payload stay byte identical, because they are data and not prose. Aliases
are remembered per record, so saving a file never renames a key the trainer expects.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

FIELD_NATURAL = "natural_prompt"
FIELD_AST = "target_grammar"
FIELD_SMT = "smt_constraint"

FIELD_LABELS: Dict[str, str] = {
    FIELD_NATURAL: "Natural Prompt",
    FIELD_AST: "Target GF Grammar",
    FIELD_SMT: "SMT-LIB Constraint",
}

FIELD_ORDER: Tuple[str, ...] = (FIELD_NATURAL, FIELD_AST, FIELD_SMT)

ALIASES: Dict[str, Tuple[str, ...]] = {
    FIELD_NATURAL: ("naturalLanguage", "natural_language", "prompt", "sentence", "text", "input", "utterance"),
    FIELD_AST: (
        "abstractSyntaxTree",
        "abstract_syntax_tree",
        "targetGrammar",
        "grammar",
        "gf",
        "target",
        "tree",
    ),
    FIELD_SMT: ("smtConstraints", "smt_constraints", "smt2", "smt", "constraint", "assertions"),
}

JSON_SUFFIXES = (".json",)
JSONL_SUFFIXES = (".jsonl", ".ndjson")
TEXT_SUFFIXES = (".txt", ".gf", ".corpus")


@dataclass
class DatasetRecord:
    """One training example: payload mapping plus, for text lines, the raw line.

    ``meta`` and ``container`` remember the wrapper a payload was found in, so a file
    shaped like a seed payload keeps its envelope when the table saves it back.
    """

    index: int
    payload: Dict[str, Any] = field(default_factory=dict)
    raw: str = ""
    keys: Dict[str, str] = field(default_factory=dict)
    format: str = "json"
    meta: Dict[str, Any] = field(default_factory=dict)
    container: str = ""

    def field_key(self, field_name: str) -> Optional[str]:
        return self.keys.get(field_name)

    def get(self, field_name: str, default: str = "") -> str:
        key = self.keys.get(field_name)
        if key is None:
            return default
        value = self.payload.get(key, default)
        return _render(value)

    def set(self, field_name: str, value: str) -> None:
        """Edit a field, creating the key with the canonical alias when it is absent."""
        key = self.keys.get(field_name)
        if key is None:
            key = ALIASES.get(field_name, (field_name,))[0]
            self.keys[field_name] = key
        current = self.payload.get(key)
        if isinstance(current, (dict, list)):
            try:
                self.payload[key] = json.loads(value)
                return
            except json.JSONDecodeError:
                pass
        self.payload[key] = value

    def extra_fields(self) -> List[str]:
        used = set(self.keys.values())
        return [k for k in self.payload if k not in used]

    def search_blob(self) -> str:
        """Text to search through: the live payload, so an edit is found immediately."""
        if self.payload:
            return json.dumps(self.payload, ensure_ascii=False, sort_keys=True)
        return self.raw

    def as_written(self) -> Dict[str, Any]:
        """Rebuild the mapping this record was read from, envelope included."""
        if not self.meta and not self.container:
            return self.payload
        merged = dict(self.meta)
        if self.container:
            merged[self.container] = self.payload
        else:
            merged.update(self.payload)
        return merged


@dataclass
class DatasetDocument:
    """A whole loaded corpus: records, format, field alias map, and provenance."""

    path: Optional[Path] = None
    format: str = "json"
    records: List[DatasetRecord] = field(default_factory=list)
    keys_by_field: Dict[str, List[str]] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)
    truncated_at: Optional[int] = None

    def __len__(self) -> int:
        return len(self.records)

    def window(self, offset: int, size: int) -> List[DatasetRecord]:
        """Paginated slice used by the table view; offsets are clamped, never raising."""
        if offset < 0:
            offset = 0
        if size <= 0:
            return []
        return self.records[offset : offset + size]

    def keys_of(self, field_name: str) -> List[str]:
        return list(self.keys_by_field.get(field_name, []))

    def stats(self) -> Dict[str, Any]:
        present = {name: sum(1 for r in self.records if r.keys.get(name)) for name in FIELD_ORDER}
        statuses: Dict[str, int] = {}
        for record in self.records:
            status = _status_of(record)
            if status:
                statuses[status] = statuses.get(status, 0) + 1
        return {
            "path": str(self.path) if self.path else "<unsaved>",
            "format": self.format,
            "records": len(self.records),
            "fields": present,
            "formal_statuses": statuses,
            "extra_keys": sorted({k for r in self.records for k in r.extra_fields()}),
            "notes": list(self.notes),
        }

    def schema(self) -> Dict[str, Tuple[str, int]]:
        """Key to ``(sample type, occurrences)`` map: feeds the port derivation of a node."""
        found: Dict[str, Tuple[str, int]] = {}
        for record in self.records:
            for key, value in record.payload.items():
                sample = type(value).__name__
                kind, count = found.get(key, (sample, 0))
                found[key] = (sample if kind == sample else "mixed", count + 1)
        return found

    def to_payload(self) -> Any:
        """Serialisable view of the document, in the shape its format expects."""
        if self.format == "text":
            return [record.get(FIELD_NATURAL) for record in self.records]
        if self.format == "payload" and len(self.records) == 1:
            return self.records[0].as_written()
        return [record.as_written() for record in self.records]


def _render(value: Any) -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)


def _status_of(record: DatasetRecord) -> str:
    payload = record.payload
    for key in record.keys.values():
        value = payload.get(key)
        if isinstance(value, dict):
            status = value.get("z3Status") or value.get("status")
            if isinstance(status, str):
                return status
    for candidate in ("z3Status", "status"):
        status = payload.get(candidate)
        if isinstance(status, str):
            return status
    return ""


def _bind_fields(record: DatasetRecord) -> DatasetRecord:
    """Attach the three canonical fields to whatever key names the record really uses."""
    for field_name, aliases in ALIASES.items():
        for alias in aliases:
            if alias in record.payload:
                record.keys[field_name] = alias
                break
    return record


def _coerce_document(document: Any, document_notes: List[str]) -> List[Tuple[Dict[str, Any], Dict[str, Any], str]]:
    """Split a JSON document into ``(payload, envelope, container key)`` triples.

    A payload wrapped in an envelope, as a seed payload is, keeps its envelope in the
    returned metadata so the same file shape comes back out on save.
    """
    if isinstance(document, list):
        return [
            (item if isinstance(item, dict) else {ALIASES[FIELD_NATURAL][0]: item}, {}, "")
            for item in document
        ]
    if isinstance(document, dict):
        for container in ("records", "payloads", "corpus", "examples", "samples", "data"):
            inner = document.get(container)
            if isinstance(inner, list):
                document_notes.append(f"records read from the {container} array of the document")
                envelope = {key: value for key, value in document.items() if key != container}
                return [
                    (item if isinstance(item, dict) else {ALIASES[FIELD_NATURAL][0]: item}, envelope, "")
                    for item in inner
                ]
        inner = document.get("payload")
        if isinstance(inner, dict):
            document_notes.append("single payload read from the payload object of the document")
            envelope = {key: value for key, value in document.items() if key != "payload"}
            return [(inner, envelope, "payload")]
        return [(document, {}, "")]
    document_notes.append("unsupported top level JSON node, no record created")
    return []


def load_dataset(path: Path, limit: Optional[int] = None) -> DatasetDocument:
    """Load a dataset file of any supported format into one document.

    :param limit: optional cap on how many records to read, for huge corpora.
    """
    file_path = Path(path)
    suffix = file_path.suffix.lower()
    notes: List[str] = []
    if suffix in JSON_SUFFIXES:
        try:
            document = _read_json(file_path)
        except json.JSONDecodeError as exc:
            notes.append(f"the file is not valid JSON: {exc}")
            document = None
        except OSError as exc:
            notes.append(f"the file could not be read: {exc}")
            document = None
        payloads = _coerce_document(document, notes)
        format_name = "json"
        if isinstance(document, dict) and "payload" in document:
            format_name = "payload"
        records = [
            _bind_fields(
                DatasetRecord(
                    index=index,
                    payload=payload,
                    meta=envelope,
                    container=container,
                    format=format_name,
                )
            )
            for index, (payload, envelope, container) in enumerate(payloads)
        ]
    elif suffix in JSONL_SUFFIXES:
        records = []
        position = 0
        for line_number, line in enumerate(_lines(file_path)):
            if not line.strip():
                continue
            if limit is not None and position >= limit:
                break
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                notes.append(f"line {line_number + 1} is not valid JSON and was kept as a raw prompt")
                payload = {ALIASES[FIELD_NATURAL][0]: line}
            body = payload if isinstance(payload, dict) else {ALIASES[FIELD_NATURAL][0]: payload}
            records.append(
                _bind_fields(DatasetRecord(index=position, payload=body, raw=line, format="jsonl"))
            )
            position += 1
        format_name = "jsonl"
    elif suffix in TEXT_SUFFIXES or file_path.name.lower().endswith(".corpus"):
        natural_key = ALIASES[FIELD_NATURAL][0]
        records = []
        position = 0
        for line in _lines(file_path):
            if not line.strip():
                continue
            if limit is not None and position >= limit:
                break
            payload = {natural_key: line}
            records.append(_bind_fields(DatasetRecord(index=position, payload=payload, raw=line, format="text")))
            position += 1
        format_name = "text"
    else:
        raise ValueError(f"{file_path} has no supported dataset reader for its suffix")

    keys_by_field: Dict[str, List[str]] = {}
    for record in records:
        for field_name, key in record.keys.items():
            bucket = keys_by_field.setdefault(field_name, [])
            if key not in bucket:
                bucket.append(key)

    if limit is not None and len(records) >= limit:
        notes.append(f"reading stopped after {limit} records")

    return DatasetDocument(
        path=file_path,
        format=format_name,
        records=records,
        keys_by_field=keys_by_field,
        notes=notes,
        truncated_at=limit,
    )


def _lines(file_path: Path) -> Iterable[str]:
    with open(file_path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            yield line.rstrip("\n")


def _read_json(file_path: Path) -> Any:
    with open(file_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def save_dataset(document: DatasetDocument, path: Optional[Path] = None, format: Optional[str] = None) -> Path:
    """Write the document back in its own format, or in the requested one."""
    target = Path(path) if path is not None else document.path
    if target is None:
        raise ValueError("no destination path given and the document was never loaded from a file")
    format_name = format or document.format
    if format_name in ("payload", "json"):
        body: Any
        if format_name == "payload" and len(document.records) == 1:
            body = document.records[0].as_written()
        elif document.format == "payload" and len(document.records) == 1:
            body = document.records[0].as_written()
        else:
            body = [record.as_written() for record in document.records]
        text = json.dumps(body, ensure_ascii=False, indent=2)
    elif format_name == "jsonl":
        text = "\n".join(json.dumps(record.as_written(), ensure_ascii=False) for record in document.records)
    else:
        text = "\n".join(record.get(FIELD_NATURAL) for record in document.records)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text + "\n", encoding="utf-8")
    document.path = target
    document.format = format_name
    return target


def filter_records(document: DatasetDocument, needle: str, field_name: Optional[str] = None) -> List[int]:
    """Return the indices of records matching a substring, optionally in one field."""
    if not needle:
        return list(range(len(document.records)))
    lowered = needle.lower()
    hits: List[int] = []
    for record in document.records:
        haystack = record.get(field_name) if field_name in FIELD_ORDER else record.search_blob()
        if lowered in haystack.lower():
            hits.append(record.index)
    return hits


def dataset_as_text(document: DatasetDocument, limit: int = 40) -> str:
    """Flat listing for the log console and for scripts."""
    lines = [f"{document.stats()['path']} [{document.format}] {len(document.records)} records"]
    for record in document.records[:limit]:
        lines.append(f"  #{record.index}: {record.get(FIELD_NATURAL)}")
    if len(document.records) > limit:
        lines.append(f"  ... {len(document.records) - limit} more")
    return "\n".join(lines)


def graph_payload_fields(document: DatasetDocument) -> Dict[str, Any]:
    """Representative payload mapping of a corpus, for canvas port derivation."""
    merged: Dict[str, Any] = {}
    for record in document.records[:200]:
        for key, value in record.payload.items():
            if key in merged:
                continue
            merged[key] = value
    return merged


__all__ = [
    "ALIASES",
    "FIELD_AST",
    "FIELD_LABELS",
    "FIELD_NATURAL",
    "FIELD_ORDER",
    "FIELD_SMT",
    "DatasetDocument",
    "DatasetRecord",
    "dataset_as_text",
    "filter_records",
    "graph_payload_fields",
    "load_dataset",
    "save_dataset",
]
