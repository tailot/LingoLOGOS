"""Dataset and payload model of the workbench: three editable fields over any corpus.

Covers:
  1. Field binding through the repository aliases, so the English column labels map onto
     the byte identical payload keys of the training engine.
  2. Editing a record without renaming keys and without dropping unknown fields.
  3. The json, jsonl, txt and single payload readers and writers, and their round trips.
  4. Pagination, filtering, statistics and schema inference of a loaded corpus.
  5. The honest verdict for Hugging Face style corpora that carry none of the three fields.
"""

import json
from pathlib import Path

import pytest

from gui.model.dataset import (
    ALIASES,
    FIELD_AST,
    FIELD_LABELS,
    FIELD_NATURAL,
    FIELD_ORDER,
    FIELD_SMT,
    DatasetDocument,
    dataset_as_text,
    filter_records,
    graph_payload_fields,
    load_dataset,
    save_dataset,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASETS_DIR = REPO_ROOT / "datasets"


def write(tmp_path, text, name):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


PAYLOAD = {
    "naturalLanguage": "para la frase di prova",
    "abstractSyntaxTree": "(pred (arg))",
    "smtConstraints": ["(assert true)"],
    "extraFlag": 7,
}


def test_the_three_editable_fields_are_the_ones_the_engine_writes():
    assert FIELD_ORDER == (FIELD_NATURAL, FIELD_AST, FIELD_SMT)
    assert FIELD_LABELS[FIELD_NATURAL] == "Natural Prompt"
    assert "naturalLanguage" in ALIASES[FIELD_NATURAL]
    assert "abstractSyntaxTree" in ALIASES[FIELD_AST]
    assert "smtConstraints" in ALIASES[FIELD_SMT]


def test_a_payload_document_binds_the_three_fields_through_its_aliases(tmp_path):
    document = load_dataset(write(tmp_path, json.dumps([PAYLOAD], ensure_ascii=False), "one.json"))
    assert len(document) == 1
    record = document.records[0]
    assert record.get(FIELD_NATURAL) == PAYLOAD["naturalLanguage"]
    assert record.get(FIELD_AST) == PAYLOAD["abstractSyntaxTree"]
    assert "assert true" in record.get(FIELD_SMT)
    assert document.keys_of(FIELD_NATURAL) == ["naturalLanguage"]
    assert record.field_key(FIELD_SMT) == "smtConstraints"
    assert record.extra_fields() == ["extraFlag"]
    assert "para la frase di prova" in record.search_blob()
    assert record.keys.get(FIELD_NATURAL) == "naturalLanguage"
    assert record.index == 0 and record.format in ("json", "payload")


def test_editing_a_field_that_is_absent_creates_it_under_the_canonical_alias(tmp_path):
    document = load_dataset(write(tmp_path, json.dumps([{"extraFlag": 1}]), "bare.json"))
    record = document.records[0]
    assert record.get(FIELD_NATURAL) == ""
    record.set(FIELD_NATURAL, "written now")
    assert record.get(FIELD_NATURAL) == "written now"
    assert record.as_written()["naturalLanguage"] == "written now"


def test_a_structured_field_takes_json_and_falls_back_to_the_plain_text(tmp_path):
    document = load_dataset(write(tmp_path, json.dumps([PAYLOAD], ensure_ascii=False), "one.json"))
    record = document.records[0]
    record.set(FIELD_SMT, '["(assert false)"]')
    assert record.get(FIELD_SMT) == '["(assert false)"]' or "(assert false)" in record.get(FIELD_SMT)
    assert isinstance(record.payload["smtConstraints"], list)
    record.set(FIELD_SMT, "(assert raw)")
    assert record.payload["smtConstraints"] == "(assert raw)"


def test_an_unreadable_suffix_is_refused_with_an_english_message(tmp_path):
    path = write(tmp_path, "a,b\n1,2\n", "table.csv")
    with pytest.raises(ValueError) as captured:
        load_dataset(path)
    assert "suffix" in str(captured.value)


def test_editing_keeps_the_payload_key_and_the_untouched_fields(tmp_path):
    document = load_dataset(write(tmp_path, json.dumps([PAYLOAD], ensure_ascii=False), "one.json"))
    record = document.records[0]
    record.set(FIELD_NATURAL, "replaced prompt")
    assert record.get(FIELD_NATURAL) == "replaced prompt"
    written = record.as_written()
    assert written["naturalLanguage"] == "replaced prompt"
    assert written["abstractSyntaxTree"] == PAYLOAD["abstractSyntaxTree"]
    assert written["smtConstraints"] == PAYLOAD["smtConstraints"]
    assert written["extraFlag"] == 7
    stats = document.stats()
    assert stats["fields"][FIELD_NATURAL] == 1 and stats["fields"][FIELD_AST] == 1
    assert stats["extra_keys"] == ["extraFlag"]


def test_jsonl_json_and_text_readers_all_produce_records(tmp_path):
    jsonl = write(
        tmp_path,
        "\n".join(json.dumps({"naturalLanguage": f"row {n}"}) for n in range(3)),
        "rows.jsonl",
    )
    document = load_dataset(jsonl)
    assert document.format == "jsonl" and len(document) == 3
    assert [r.get(FIELD_NATURAL) for r in document.records] == ["row 0", "row 1", "row 2"]

    text = write(tmp_path, "first line\nsecond line\n\n", "plain.txt")
    plain = load_dataset(text)
    assert plain.format in ("text", "txt")
    assert [r.get(FIELD_NATURAL) for r in plain.records] == ["first line", "second line"]


def test_a_single_object_payload_is_read_as_one_record(tmp_path):
    document = load_dataset(write(tmp_path, json.dumps(PAYLOAD, ensure_ascii=False), "single.json"))
    assert len(document) == 1
    assert document.records[0].get(FIELD_AST) == PAYLOAD["abstractSyntaxTree"]


def test_pagination_clamps_instead_of_raising(tmp_path):
    path = write(
        tmp_path,
        "\n".join(json.dumps({"naturalLanguage": f"row {n}"}) for n in range(5)),
        "rows.jsonl",
    )
    document = load_dataset(path)
    assert len(document.window(0, 2)) == 2
    assert len(document.window(4, 2)) == 1
    assert document.window(99, 5) == []
    assert len(document.window(-3, 2)) == 2
    assert document.window(0, 0) == []


def test_limit_marks_the_document_as_truncated(tmp_path):
    path = write(
        tmp_path,
        "\n".join(json.dumps({"naturalLanguage": f"row {n}"}) for n in range(6)),
        "rows.jsonl",
    )
    document = load_dataset(path, limit=2)
    assert len(document) == 2
    assert document.truncated_at == 6 or (document.truncated_at or 0) >= 2


def test_filtering_is_case_insensitive_and_can_be_narrowed_to_one_field(tmp_path):
    path = write(
        tmp_path,
        json.dumps(
            [
                {"naturalLanguage": "Alpha prompt", "abstractSyntaxTree": "(a)"},
                {"naturalLanguage": "beta prompt", "abstractSyntaxTree": "ALPHA rule"},
            ]
        ),
        "pairs.json",
    )
    document = load_dataset(path)
    assert filter_records(document, "ALPHA") == [0, 1]
    assert filter_records(document, "alpha", FIELD_NATURAL) == [0]
    assert filter_records(document, "alpha", FIELD_AST) == [1]
    assert filter_records(document, "nothing here") == []


def test_a_broken_file_reports_instead_of_crashing(tmp_path):
    path = write(tmp_path, "{ this is not json", "broken.json")
    document = load_dataset(path)
    assert isinstance(document, DatasetDocument)
    assert document.notes or len(document) == 0


def test_a_broken_line_of_a_jsonl_file_is_kept_as_a_raw_prompt(tmp_path):
    path = write(
        tmp_path,
        '{"naturalLanguage": "good line"}\n{ not json at all\n{"naturalLanguage": "also good"}',
        "rows.jsonl",
    )
    document = load_dataset(path)
    assert len(document) == 3
    assert document.notes, "a malformed line has to be reported"
    assert document.records[1].get(FIELD_NATURAL).startswith("{")


def test_round_trip_through_every_writable_format(tmp_path):
    source = write(tmp_path, json.dumps([PAYLOAD], ensure_ascii=False), "one.json")
    document = load_dataset(source)
    document.records[0].set(FIELD_NATURAL, "round trip prompt")

    for name, fmt in (("again.json", "json"), ("again.jsonl", "jsonl"), ("again.txt", "txt")):
        target = tmp_path / name
        save_dataset(document, target, fmt)
        assert target.is_file()
        back = load_dataset(target)
        assert len(back) == 1
        assert back.records[0].get(FIELD_NATURAL) == "round trip prompt"

    payload_target = tmp_path / "again_payload.json"
    save_dataset(document, payload_target, "payload")
    with open(payload_target, "r", encoding="utf-8") as handle:
        body = json.load(handle)
    assert isinstance(body, dict)
    assert body["naturalLanguage"] == "round trip prompt"
    assert body["extraFlag"] == 7


def test_the_saved_file_keeps_the_engine_keys_byte_identical(tmp_path):
    document = load_dataset(write(tmp_path, json.dumps([PAYLOAD], ensure_ascii=False), "one.json"))
    target = tmp_path / "kept.json"
    save_dataset(document, target, "json")
    with open(target, "r", encoding="utf-8") as handle:
        body = json.load(handle)
    assert list(body[0]) == ["naturalLanguage", "abstractSyntaxTree", "smtConstraints", "extraFlag"]
    assert body[0]["smtConstraints"] == ["(assert true)"]
    assert "natural_prompt" not in json.dumps(body)


def test_text_export_reads_as_a_human_listing(tmp_path):
    document = load_dataset(
        write(tmp_path, json.dumps([PAYLOAD]), "one.json")
    )
    text = dataset_as_text(document)
    assert isinstance(text, str) and text
    assert "para" in text or "Natural" in text


def test_schema_inference_reports_types_and_counts(tmp_path):
    document = load_dataset(
        write(
            tmp_path,
            json.dumps(
                [
                    {"naturalLanguage": "a", "score": 1},
                    {"naturalLanguage": "b", "score": 2.5},
                ]
            ),
            "two.json",
        )
    )
    schema = document.schema()
    assert schema["naturalLanguage"] == ("str", 2)
    assert schema["score"][0] == "mixed" and schema["score"][1] == 2
    fields = graph_payload_fields(document)
    assert isinstance(fields, dict) and fields


def test_a_seed_payload_of_this_repository_binds_all_three_fields():
    candidates = sorted(DATASETS_DIR.rglob("seed_payload.json")) if DATASETS_DIR.is_dir() else []
    if not candidates:
        pytest.skip("no seed_payload.json under datasets in this checkout")
    bound = None
    for path in candidates:
        document = load_dataset(path)
        present = document.stats()["fields"]
        if all(present[name] for name in FIELD_ORDER):
            bound = (path, document)
            break
    assert bound is not None, "no seed payload carries all three fields of the payload schema"
    path, document = bound
    assert str(path).startswith(str(DATASETS_DIR))
    assert document.stats()["records"] >= 1


def test_an_sft_corpus_honestly_reports_that_it_binds_none_of_the_three_fields():
    candidates = sorted((DATASETS_DIR / "multilingual").rglob("*.json")) if (DATASETS_DIR / "multilingual").is_dir() else []
    if not candidates:
        pytest.skip("no multilingual corpus in this checkout")
    document = load_dataset(candidates[0], limit=25)
    present = document.stats()["fields"]
    assert len(document) > 0
    assert all(present[name] == 0 for name in FIELD_ORDER)
    assert document.stats()["extra_keys"], "an SFT corpus still carries its own keys"
