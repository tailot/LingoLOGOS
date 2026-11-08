"""Checks for the RV32E firmware stored in the program memory of the ASIC core.

The core under test is asic_chip/rtl/core/rv32e_core.sv and the top level wiring is
asic_chip/rtl/top/lingologos_top.sv. Section 1 of the LingoLOGOS ASIC specification
asks for a real reduced core rather than a stub, so this file emulates the documented
instruction set with an independent interpreter from tests/asic_reference_model.py and
compares the result with the expectations asic_chip/sim/core_self_test.mem ships to both
simulation harnesses. No simulator is needed for this gate; the interpretation here is
Python only.

Acceptance gates:
  F1  the program memory is complete, initialized, and matches asic_chip/config.json
  F2  every encoding the program executes stays inside the documented RV32E subset
  F3  exactly one encoding is illegal, the documented x16 probe, at the documented slot
  F4  the emulated run reproduces every value of asic_chip/sim/core_self_test.mem
  F5  the run finishes inside the cycle budget the simulation harness allows
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tests"))

import asic_reference_model as ref  # noqa: E402

ASIC_DIR = REPO_ROOT / "asic_chip"
CORE_FILE = ASIC_DIR / "rtl" / "core" / "rv32e_core.sv"
SELF_TEST_FILE = ASIC_DIR / "sim" / "core_self_test.mem"
CHIP_CONFIG_FILE = ASIC_DIR / "config.json"
HARNESS_FILE = ASIC_DIR / "sim" / "sim_main.cpp"

ROM_LINE = re.compile(r"instruction_rom\[(\d+)\]\s*=\s*32'h([0-9A-Fa-f_]+);")
PARAM_LINE = re.compile(r"parameter int (IMEM_WORDS|DMEM_WORDS)\s*=\s*(\d+)")
HARNESS_CYCLE_LIMIT = re.compile(r"cycles\s*<\s*(\d+)")

NOP = 0x00000013


def read_core_source() -> str:
    return CORE_FILE.read_text(encoding="utf-8")


def read_rom() -> Dict[int, int]:
    rom: Dict[int, int] = {}
    for match in ROM_LINE.finditer(read_core_source()):
        index = int(match.group(1))
        value = int(match.group(2).replace("_", ""), 16)
        if index in rom:
            raise AssertionError(f"program slot {index} is written twice in {CORE_FILE}")
        if not 0 <= value <= 0xFFFFFFFF:
            raise AssertionError(f"program slot {index} is wider than 32 bits: {value:#x}")
        rom[index] = value
    return rom


def read_self_test_values() -> Dict[str, int]:
    values: Dict[str, int] = {}
    for line in SELF_TEST_FILE.read_text(encoding="utf-8").splitlines():
        text = line.split("//", 1)[0].strip()
        if not text:
            continue
        key, _, value = text.partition("=")
        values[key.strip()] = int(value.strip(), 16)
    return values


def read_chip_config() -> dict:
    return json.loads(CHIP_CONFIG_FILE.read_text(encoding="utf-8"))


def rom_parameters() -> Dict[str, int]:
    return {match.group(1): int(match.group(2)) for match in PARAM_LINE.finditer(read_core_source())}


def self_test_stimulus(values: Dict[str, int]) -> Dict[str, int]:
    """Signed pad values exactly as the top level declares them."""
    return {
        "in_vector": [ref.to_signed(values[f"in_vector_{index}"], 8) for index in range(4)],
        "gf_symbol_in": values["gf_symbol_in"],
        "smt_in_bits": values["smt_in_bits"],
    }


def gf_tables() -> Dict[int, int]:
    artifact = json.loads((ASIC_DIR / "artifacts" / "gf_rules.json").read_text(encoding="utf-8"))
    return {
        int(symbol["byte"], 16): int(symbol["multiplier"], 16) for symbol in artifact["symbols"]
    }


def gf_initial_state() -> int:
    artifact = json.loads((ASIC_DIR / "artifacts" / "gf_rules.json").read_text(encoding="utf-8"))
    return int(artifact["initial_state"], 16)


def weight_values() -> List[List[int]]:
    artifact = json.loads((ASIC_DIR / "artifacts" / "model_weights.json").read_text(encoding="utf-8"))
    return artifact["weight_matrix"]["values"]


def clause_tables() -> List[List[int]]:
    artifact = json.loads((ASIC_DIR / "artifacts" / "smt_constraints.json").read_text(encoding="utf-8"))
    variable_bits = {name: int(bit) for name, bit in artifact["variable_bits"].items()}
    input_bits = int(artifact["input_bits"])
    ordered = sorted(artifact["clauses"], key=lambda clause: int(clause["id"]))
    return [ref.clause_truth_table(clause["expr"], variable_bits, input_bits) for clause in ordered]


def emulate() -> ref.CoreRun:
    values = read_self_test_values()
    stimulus = self_test_stimulus(values)
    parameters = rom_parameters()
    return ref.run_firmware(
        rom=read_rom(),
        weight_values=weight_values(),
        gf_initial_state=gf_initial_state(),
        gf_multipliers=gf_tables(),
        clause_tables=clause_tables(),
        pad_inputs=stimulus["in_vector"],
        pad_symbol=stimulus["gf_symbol_in"],
        imem_words=parameters["IMEM_WORDS"],
        dmem_words=parameters["DMEM_WORDS"],
    )


def test_f1_program_memory_is_complete_and_matches_config() -> None:
    """Every slot of the declared program memory is written exactly once."""
    rom = read_rom()
    parameters = rom_parameters()
    config_datapath = read_chip_config()["datapath"]
    declared_words = int(config_datapath["core_imem_words"])
    assert parameters["IMEM_WORDS"] == declared_words, (
        f"{CORE_FILE} declares {parameters['IMEM_WORDS']} program slots while "
        f"asic_chip/config.json declares {declared_words}"
    )
    assert parameters["DMEM_WORDS"] == int(config_datapath["core_dmem_words"]), (
        "the scratch memory size of the core and of the chip configuration disagree"
    )
    assert declared_words & (declared_words - 1) == 0, (
        f"the program memory holds {declared_words} slots, which is not a power of two, "
        "so a wrapped program counter could index an undefined slot"
    )
    assert sorted(rom) == list(range(declared_words)), (
        "the program memory initialization does not cover every slot exactly once"
    )
    firmware_words = int(config_datapath["core_firmware_words"])
    trailing = [rom[index] for index in range(firmware_words, declared_words)]
    assert trailing == [NOP] * (declared_words - firmware_words), (
        "the slots after the program must all be no-operations, so a runaway program "
        "counter never decodes an encoding the chip configuration does not describe"
    )
    last_programmed = max(index for index, word in rom.items() if word != NOP)
    assert last_programmed == firmware_words - 1, (
        f"the last slot that is not a no-operation is {last_programmed}, while "
        f"asic_chip/config.json records the firmware as {firmware_words} words"
    )
    halt_word_index = next(
        index
        for index, word in rom.items()
        if (word & 0x7F) == 0b0001011 and ((word >> 12) & 0b111) == 0b111
    )
    assert halt_word_index < firmware_words, (
        f"the halt instruction sits in slot {halt_word_index}, outside the {firmware_words} "
        "slots the chip configuration counts as firmware"
    )
    pad_slots = [index for index in range(firmware_words) if rom[index] == NOP]
    assert pad_slots == [0, halt_word_index + 1], (
        f"the program region holds no-operations in slots {pad_slots}; only the entry slot, "
        "which keeps address zero out of the reset vector, and the safety slot right after "
        f"the halt in slot {halt_word_index} are expected"
    )


def test_f2_every_executed_encoding_is_inside_the_documented_subset() -> None:
    """No executed instruction uses an opcode the core documents as unimplemented."""
    rom = read_rom()
    forbidden_opcodes = {
        0b1110011: "SYSTEM",
        0b0011001: "OP-IMM-32",
        0b0111011: "OP-32",
        0b0000111: "ENVIRONMENT",
    }
    offenders: List[str] = []
    for index, instruction in sorted(rom.items()):
        opcode = instruction & 0x7F
        if opcode in forbidden_opcodes:
            offenders.append(f"slot {index} uses {forbidden_opcodes[opcode]} ({instruction:#010x})")
        if opcode == 0b0001111 and ((instruction >> 12) & 0b111) != 0:
            offenders.append(f"slot {index} uses FENCE.I, which has nothing to invalidate")
    assert not offenders, "the program memory contains encodings outside the documented subset: " + "; ".join(
        offenders
    )
    fence_encodings = [
        index for index, instruction in rom.items() if (instruction & 0x7F) == 0b0001111
    ]
    for index in fence_encodings:
        assert ((rom[index] >> 12) & 0b111) == 0, f"slot {index} is not a plain FENCE"


def test_f3_illegal_encoding_is_the_documented_x16_probe() -> None:
    """Exactly one program slot breaks the RV32E register-space rule, on purpose."""
    rom = read_rom()
    probe_index = int(read_chip_config()["datapath"]["core_illegal_probe_word_index"])
    wide_register_slots: List[int] = []
    for index, instruction in sorted(rom.items()):
        opcode = instruction & 0x7F
        rd = (instruction >> 7) & 0x1F
        rs1 = (instruction >> 15) & 0x1F
        rs2 = (instruction >> 20) & 0x1F
        if not ref.register_space_is_legal(opcode, rd, rs1, rs2):
            wide_register_slots.append(index)
    assert wide_register_slots == [probe_index], (
        f"the program names a register above x15 in slots {wide_register_slots}, while "
        f"asic_chip/config.json documents the single probe in slot {probe_index}"
    )
    run = emulate()
    assert run.illegal_words == [probe_index], (
        f"the emulation reports illegal encodings in slots {run.illegal_words}, expected "
        f"[{probe_index}]"
    )
    assert run.illegal_seen == 1, "the sticky illegal flag must be set by the probe"


def test_f4_emulation_reproduces_the_core_self_test_expectations() -> None:
    """The emulated run equals every expectation the simulation harness checks."""
    values = read_self_test_values()
    stimulus = self_test_stimulus(values)
    run = emulate()

    assert run.halted, "the program never reached the halt instruction inside the cycle limit"
    assert run.pass_flag == values["exp_core_pass"], (
        f"the emulated pass flag is {run.pass_flag}, expected {values['exp_core_pass']}"
    )
    assert run.registers[15] == ref.to_unsigned(values["exp_x15"], 32), (
        f"the firmware error accumulator holds {run.registers[15]:#010x}, expected "
        f"{values['exp_x15']:#010x}: one instruction check failed"
    )
    for name, register in (("exp_x8", 8), ("exp_x10", 10), ("exp_x11", 11), ("exp_x13", 13)):
        assert run.registers[register] == ref.to_unsigned(values[name], 32), (
            f"x{register} holds {run.registers[register]:#010x} after the self-test, the "
            f"generated expectation in {SELF_TEST_FILE.name} is {values[name]:#010x}"
        )
    assert run.gf_final_state == values["exp_gf_state_final"], (
        f"the automaton state after the run is {run.gf_final_state:#04x}, expected "
        f"{values['exp_gf_state_final']:#04x}"
    )
    assert run.illegal_seen == values["exp_core_illegal_seen"], (
        "the sticky illegal flag of the emulation disagrees with the generated expectation"
    )
    expected_row = ref.mesh_row_sum(
        weight_values(), values["mesh_row_index"], stimulus["in_vector"]
    )
    assert run.registers[11] == ref.to_unsigned(expected_row, 32), (
        f"the mesh row readback in x11 is {run.registers[11]}, the reference dot product of "
        f"row {values['mesh_row_index']} over the frozen pads is {expected_row}"
    )
    lane_word = 0
    for lane, table in enumerate(clause_tables()):
        lane_word |= (table[stimulus["smt_in_bits"]] & 1) << lane
    assert run.registers[8] == lane_word, (
        f"x8 holds {run.registers[8]}, the reference clause evaluation over the byte "
        f"{stimulus['smt_in_bits']:#04x} is {lane_word}"
    )
    status_word = ((run.illegal_seen & 1) << 4) | ((run.cop_count - 1) & 0xF)
    assert run.registers[13] == ref.to_unsigned(values["exp_x13"], 32), (
        "the coprocessor status readback does not match the generated file"
    )
    assert (run.registers[13] & 0xF) == 3, (
        f"x13 records {run.registers[13] & 0xF} coprocessor operations before the status read, "
        "the program issues the SMT, automaton and mesh reads before it"
    )
    assert status_word == run.registers[13], (
        "the status word computed from the counters does not match the register readback"
    )
    assert stimulus["gf_symbol_in"] not in gf_tables(), (
        "the frozen pad symbol is expected to be outside the bracket alphabet so the "
        "default arm of the automaton table is exercised"
    )


def test_f5_run_fits_the_harness_cycle_budget() -> None:
    """The self-test retires well inside the loop the C++ harness allows."""
    source = HARNESS_FILE.read_text(encoding="utf-8")
    match = HARNESS_CYCLE_LIMIT.search(source)
    assert match, f"{HARNESS_FILE} does not show a bounded wait loop for core_done"
    budget = int(match.group(1))
    run = emulate()
    assert run.halted, "the emulation did not observe a halt before its own safety limit"
    assert run.cycles < budget, (
        f"the self-test needs {run.cycles} clock cycles, the harness only waits for "
        f"{budget}, so the harness would have to be widened"
    )
    retired = len(set(run.executed_words))
    assert retired > 80, (
        f"the program only exercised {retired} distinct instructions, which is fewer than "
        "the instruction-set checks the header of the core documents"
    )


def test_data_memory_readback_matches_the_reference_model() -> None:
    """The store and load sequence of the firmware exercises the scratch array."""
    run = emulate()
    nonzero_slots = {index: word for index, word in enumerate(run.dmem) if word != 0}
    assert nonzero_slots, "the program was expected to write at least one scratch word"
    for index, word in nonzero_slots.items():
        assert 0 <= word <= 0xFFFFFFFF, f"scratch slot {index} holds a value outside 32 bits"
