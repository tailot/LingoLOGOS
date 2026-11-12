"""Lllayer stream model of the workbench: timestamps, verdicts and per node models.

Covers:
  1. The stream event itself: channel, preview, verdict detection, tone and console level.
  2. Transcript readers for the shapes the orchestrator writes, with and without timestamps.
  3. The pacing helper that drives the animated stream, including the speed factor.
  4. The model catalog: only plausible model names are offered, each with its provenance.
  5. The GUI owned per node assignment file, which nothing else in the stack reads.
"""

import json
from pathlib import Path

import pytest

from gui.model.stream import (
    KIND_CONTROL,
    KIND_FAULT,
    KIND_MESSAGE,
    KIND_SYSTEM,
    StreamEvent,
    assignment_path,
    live_command,
    load_assignments,
    load_transcript,
    model_catalog,
    replay,
    save_assignments,
    transcript_summary,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_DIR = REPO_ROOT / "runtime_state"
needs_transcripts = pytest.mark.skipif(
    not RUNTIME_DIR.is_dir() or not list(RUNTIME_DIR.glob("*transcript*.json")),
    reason="no orchestrator transcript in this checkout: run a session to produce one",
)


def write(tmp_path, payload, name="transcript.json"):
    path = tmp_path / name
    if isinstance(payload, str):
        path.write_text(payload, encoding="utf-8")
    else:
        path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def timed_records():
    return [
        {
            "turn": 0,
            "sender": "agente_alfa",
            "recipient": "agente_beta",
            "timestamp": "2026-01-01T10:00:00Z",
            "envelope": {"payload": {"naturalLanguage": "primo messaggio"}},
        },
        {
            "turn": 1,
            "sender": "agente_beta",
            "recipient": "arbitro_logico",
            "timestamp": "2026-01-01T10:00:02Z",
            "envelope": {"payload": {"naturalLanguage": "secondo messaggio"}},
            "smt_result": {"status": "UNSAT"},
        },
        {
            "turn": 2,
            "sender": "arbitro_logico",
            "recipient": "agente_alfa",
            "timestamp": "2026-01-01T10:00:09Z",
            "envelope": {"payload": {"naturalLanguage": "blocco"}},
            "routing_authorized": False,
        },
    ]


def test_an_event_names_its_channel_and_reads_as_one_line():
    event = StreamEvent(
        turn=3,
        sender="alfa",
        recipient="beta",
        envelope={"payload": {"naturalLanguage": "una frase di prova"}},
        stamp_text="10:11:12",
    )
    assert event.channel == "alfa->beta"
    assert event.kind == KIND_MESSAGE
    assert event.preview() == "una frase di prova"
    line = event.to_line()
    assert line.startswith("10:11:12 alfa->beta")
    assert "una frase di prova" in line


def test_a_long_preview_is_shortened_instead_of_wrapping():
    event = StreamEvent(turn=0, sender="a", recipient="b", envelope={"payload": "x" * 200})
    short = event.preview(40)
    assert len(short) == 40 and short.endswith("...")
    assert len(event.preview(400)) == 200


def test_the_preview_falls_back_to_the_envelope_when_there_is_no_payload():
    event = StreamEvent(turn=0, sender="a", recipient="b", envelope={"beta": 2, "alpha": 1})
    assert event.preview() == '{"alpha": 1, "beta": 2}'


def test_the_verdict_comes_from_the_envelope_the_solver_or_the_payload():
    direct = StreamEvent(turn=0, sender="a", recipient="b", envelope={"status": "PASS"})
    assert direct.verdict() == "PASS" and direct.tone() == "pass"
    solver = StreamEvent(turn=0, sender="a", recipient="b", envelope={"smt_result": {"z3Status": "UNSAT"}})
    assert solver.verdict() == "UNSAT" and solver.tone() == "pass"
    nested = StreamEvent(turn=0, sender="a", recipient="b", envelope={"payload": {"z3Status": "SAT"}})
    assert nested.verdict() == "SAT" and nested.tone() == "fail"
    quiet = StreamEvent(turn=0, sender="a", recipient="b", envelope={})
    assert quiet.verdict() == "" and quiet.tone() == "info"
    for event in (direct, solver, nested, quiet):
        assert event.level() in ("ok", "error", "warn", "info")
    assert direct.level() != "error"


def test_a_blocked_routing_shows_up_as_a_fault():
    fault = StreamEvent(turn=1, sender="a", recipient="b", kind=KIND_FAULT)
    assert fault.tone() == "warn"
    assert KIND_SYSTEM and KIND_CONTROL


def test_a_transcript_of_records_carries_timestamps_gaps_and_kinds(tmp_path):
    events = load_transcript(write(tmp_path, timed_records()))
    assert len(events) == 3
    assert [event.turn for event in events] == [0, 1, 2]
    assert events[0].channel == "agente_alfa->agente_beta"
    assert events[0].offset_seconds == 0.0
    assert events[1].offset_seconds == pytest.approx(2.0)
    assert events[2].offset_seconds == pytest.approx(2.0), "gaps are capped so the replay stays watchable"
    assert events[0].stamp_text == "2026-01-01T10:00:00Z"
    assert events[1].verdict() == "UNSAT" and events[1].tone() == "pass"
    assert events[2].kind == KIND_FAULT
    summary = transcript_summary(events)
    assert summary["events"] == 3 and summary["turns"] == 3
    assert summary["channels"]["agente_alfa->agente_beta"] == 1
    assert summary["verdicts"] == {"UNSAT": 1}


def test_an_untimed_transcript_gets_a_rhythm_instead_of_zero_gaps(tmp_path):
    records = [
        {"turn": 0, "sender": "a", "recipient": "b", "envelope": {"payload": "one"}},
        {"turn": 1, "sender": "b", "recipient": "c", "envelope": {"payload": "two"}},
    ]
    events = load_transcript(write(tmp_path, records))
    assert [event.offset_seconds for event in events] == [pytest.approx(0.6), pytest.approx(0.6)]
    assert events[0].stamp_text == "turn 000"
    custom = load_transcript(write(tmp_path, records), default_gap=1.5)
    assert custom[1].offset_seconds == pytest.approx(1.5)


def test_a_record_without_a_dict_envelope_becomes_the_envelope(tmp_path):
    path = write(
        tmp_path,
        [{"turn": 4, "from": "alfa", "to": "beta", "naturalLanguage": "frase", "note": "x"}],
    )
    events = load_transcript(path)
    assert events[0].sender == "alfa" and events[0].recipient == "beta"
    assert events[0].envelope["naturalLanguage"] == "frase"
    assert "turn" not in events[0].envelope


def test_both_transcript_shapes_are_read_and_a_stranger_is_refused(tmp_path):
    wrapped = load_transcript(write(tmp_path, {"records": timed_records()}))
    assert len(wrapped) == 3
    alternative = load_transcript(write(tmp_path, {"events": timed_records()}))
    assert len(alternative) == 3
    with pytest.raises(ValueError) as captured:
        load_transcript(write(tmp_path, "5"))
    assert "transcript records" in str(captured.value)


def test_a_non_object_record_is_kept_as_a_system_line(tmp_path):
    events = load_transcript(write(tmp_path, ["plain sentence"]))
    assert len(events) == 1
    assert events[0].kind == KIND_SYSTEM
    assert events[0].envelope == {"payload": "plain sentence"}


def test_max_events_caps_a_huge_transcript(tmp_path):
    path = write(tmp_path, timed_records() * 4)
    assert len(load_transcript(path, max_events=4)) == 4


def test_the_pacing_helper_scales_and_clamps_the_gaps(tmp_path):
    events = load_transcript(write(tmp_path, timed_records()))
    paced = list(replay(events))
    assert [round(delay, 3) for _, delay in paced] == [0.0, 1.2, 1.2]
    uncapped = list(replay(events, max_gap=5.0))
    assert [round(delay, 3) for _, delay in uncapped] == [0.0, 2.0, 2.0]
    fast = list(replay(events, speed=2.0, max_gap=5.0))
    assert round(fast[1][1], 3) == pytest.approx(1.0)
    assert all(delay <= 0.5 for _, delay in list(replay(events, max_gap=0.5)))
    assert round(list(replay(events, speed=0.0, max_gap=5.0))[1][1], 3) == pytest.approx(2.0)


CATALOG_DOCUMENT = {
    "model": "Llama-3-8B-Instruct",
    "models": ["Mistral-7B-Instruct", "Prover-SMT"],
    "engine": {"name": "z3", "role": "prover"},
    "dataset": "datasets/agente_alfa/seed_payload.json",
    "status": "UNSAT",
    "retries": 3,
}


def test_the_catalog_collects_model_names_with_their_provenance(tmp_path):
    path = write(tmp_path, CATALOG_DOCUMENT, "models_config.json")
    catalog = model_catalog([path])
    names = [entry["name"] for entry in catalog]
    assert "Llama-3-8B-Instruct" in names
    assert "Mistral-7B-Instruct" in names
    assert names.count("Mistral-7B-Instruct") == 1, "the catalog never lists the same model twice"
    for entry in catalog:
        assert entry["origin"] == "models_config.json"
        assert entry["role"]
    assert "datasets/agente_alfa/seed_payload.json" not in names
    assert "UNSAT" not in names
    assert "3" not in names


def test_the_catalog_also_scans_a_directory_of_configurations(tmp_path):
    folder = tmp_path / "conf"
    folder.mkdir()
    (folder / "b.json").write_text(json.dumps({"model_name": "Mistral-7B"}), encoding="utf-8")
    (folder / "a.json").write_text(json.dumps({"checkpoint": "Llama-3-8B"}), encoding="utf-8")
    (folder / "broken.json").write_text("{ nope", encoding="utf-8")
    catalog = model_catalog([], [folder])
    assert [entry["name"] for entry in catalog] == ["Llama-3-8B", "Mistral-7B"]


def test_missing_configuration_files_are_simply_skipped(tmp_path):
    assert model_catalog([tmp_path / "nowhere.json"], [tmp_path / "no_dir"]) == []


def test_the_assignment_file_round_trips_and_defaults_to_empty_sections(tmp_path):
    path = assignment_path(tmp_path)
    assert path == tmp_path / "config" / "models.json"
    empty = load_assignments(path)
    assert empty == {"profiles": [], "assignments": {}}
    save_assignments(
        path,
        [{"name": "Mistral-7B", "role": "llm"}, {"name": "Llama-3-8B", "role": "llm"}],
        {"arbitro_logico": "Prover-SMT", "agente_alfa": "Llama-3-8B"},
    )
    back = load_assignments(path)
    assert [profile["name"] for profile in back["profiles"]] == ["Llama-3-8B", "Mistral-7B"]
    assert back["assignments"] == {"agente_alfa": "Llama-3-8B", "arbitro_logico": "Prover-SMT"}
    assert list(back["assignments"]) == ["agente_alfa", "arbitro_logico"], "keys stay sorted for clean diffs"


def test_a_damaged_assignment_file_never_breaks_the_panel(tmp_path):
    broken = write(tmp_path, "{ not json", "models.json")
    assert load_assignments(broken) == {"profiles": [], "assignments": {}}
    wrong = write(tmp_path, [1, 2, 3], "models.json")
    assert load_assignments(wrong) == {"profiles": [], "assignments": {}}
    junk = write(tmp_path, {"profiles": "nope", "assignments": [1]}, "models.json")
    assert load_assignments(junk) == {"profiles": [], "assignments": {}}


def test_the_live_session_command_targets_the_orchestrator_module():
    program, arguments = live_command(REPO_ROOT, REPO_ROOT / "config" / "topology.json")
    assert program in ("python", "python3")
    assert arguments[:2] == ["-m", "lingologos_orchestrator.cli"]
    assert "--config" in arguments and "--json" in arguments
    assert str(REPO_ROOT / "config" / "topology.json") in arguments


@needs_transcripts
def test_a_real_transcript_of_this_repository_streams_without_losing_itself():
    paths = sorted(RUNTIME_DIR.glob("*transcript*.json"))
    assert paths
    for path in paths:
        events = load_transcript(path)
        assert events, f"{path.name} holds no stream records"
        assert all(event.channel for event in events)
        assert all(event.preview(80) for event in events)
        summary = transcript_summary(events)
        assert summary["events"] == len(events)
        paced = list(replay(events))
        assert len(paced) == len(events)
        assert all(delay >= 0.0 for _, delay in paced)
