"""Gate tests for the LingoLOGOS ASIC tree under asic_chip/.

What these tests prove and what they do not
-------------------------------------------
The tests in this file check that the artifacts, the generated SystemVerilog, the
committed golden vectors, the simulation harnesses, the pin plan and the four
documents describe the same design. Every expected value is recomputed here by
``tests/asic_reference_model.py``, which is written with deliberately different
algorithms from ``asic_chip/scripts/lingologos_compiler.py``:

* the compiler builds the mesh with a non-adjacent-form shift-add tree, this file
  evaluates a plain dot product;
* the compiler parses the clause language with recursive descent, the reference
  model compiles it with shunting-yard to reverse polish form and evaluates a mask;
* the compiler walks the GF(2^8) transitions with its own polynomial helpers, the
  reference model has its own GF code and its own irreducibility test;
* the firmware expectations come from an independent RV32E interpreter in
  ``tests/test_asic_core_firmware.py``, not from the generator.

Nothing here runs a hardware simulator. A Verilator or Icarus run would compare the
RTL against these expectations; the check that ``tests`` can offer without a toolchain
is that the netlist text, when re-evaluated by a second method, produces the same
numbers the committed vectors claim. ``asic_chip/doc/verification_report.md`` records
which levels have been executed and which have not.

Gates in this file
------------------
1  chip configuration against the RTL port list
2  generated mesh netlist against an independent dot product
3  generated GF table against field arithmetic and the grammar alphabet
4  generated clause netlist against the reference masks, all 256 input bytes
5  golden vector rows recomputed without the generator
6  generator determinism and the committed-file check mode
7  manifest digests against the committed files
8  self-test memory against the reference engines
9  testbench literals against the self-test memory
10 C++ harness against the declared port list
11 pinout agreement across RTL, configuration, datasheet and pin order plan
12 OpenLane setup against the chip configuration
13 datasheet numbers against measured values
14 language and ASCII hygiene for the whole asic_chip tree
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
ASIC_DIR = REPO_ROOT / "asic_chip"
TESTS_DIR = Path(__file__).resolve().parent

for extra in (str(TESTS_DIR), str(ASIC_DIR / "scripts")):
    if extra not in sys.path:
        sys.path.insert(0, extra)

import asic_reference_model as ref  # noqa: E402  the independent model
import lingologos_compiler as compiler  # noqa: E402  only to re-run it deterministically

WEIGHTS_ARTIFACT = ASIC_DIR / "artifacts" / "model_weights.json"
GF_ARTIFACT = ASIC_DIR / "artifacts" / "gf_rules.json"
SMT_ARTIFACT = ASIC_DIR / "artifacts" / "smt_constraints.json"
GRAMMAR_ARTIFACT = REPO_ROOT / "artifacts" / "grammar.json"
CHIP_CONFIG = ASIC_DIR / "config.json"
OPENLANE_CONFIG = ASIC_DIR / "openlane" / "config.json"
OPENLANE_SDC = ASIC_DIR / "openlane" / "lingologos.sdc"
PIN_ORDER = ASIC_DIR / "openlane" / "pin_order.cfg"
TOP_RTL = ASIC_DIR / "rtl" / "top" / "lingologos_top.sv"
CORE_RTL = ASIC_DIR / "rtl" / "core" / "rv32e_core.sv"
MESH_RTL = ASIC_DIR / "rtl" / "generated" / "weight_rom_mesh.sv"
GF_RTL = ASIC_DIR / "rtl" / "generated" / "gf_automaton_1cycle.sv"
SMT_RTL = ASIC_DIR / "rtl" / "generated" / "smt_evaluator.sv"
MANIFEST = ASIC_DIR / "rtl" / "generated" / "MANIFEST.json"
GOLDEN_MEM = ASIC_DIR / "sim" / "golden_vectors.mem"
SELFTEST_MEM = ASIC_DIR / "sim" / "core_self_test.mem"
HARNESS_CPP = ASIC_DIR / "sim" / "sim_main.cpp"
TESTBENCH_SV = ASIC_DIR / "sim" / "testbench.sv"
DATASHEET = ASIC_DIR / "doc" / "datasheet_lingologos.md"
INTEGRATION_GUIDE = ASIC_DIR / "doc" / "integration_guide.md"
VERIFICATION_REPORT = ASIC_DIR / "doc" / "verification_report.md"
BUILD_INSTRUCTIONS = ASIC_DIR / "doc" / "build_instructions.md"


# -----------------------------------------------------------------------------
# small readers
# -----------------------------------------------------------------------------

def read_text(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    assert "\ufeff" not in text, f"{path} starts with a byte order mark"
    return text


def read_json(path: Path) -> Dict[str, object]:
    return json.loads(read_text(path))


def chip_config() -> Dict[str, object]:
    return read_json(CHIP_CONFIG)


def config_paths(config: Dict[str, object], key: str) -> List[str]:
    """A path entry in the chip configuration may be a string or a list of strings."""
    value = config["paths"][key]
    return [value] if isinstance(value, str) else list(value)


def hex_byte(text: str) -> int:
    return int(str(text), 16)


def parse_sv_literal(text: str) -> int:
    """Read a SystemVerilog sized literal such as 32'h0000_0009, 8'sh0A or 4'd3."""
    match = re.fullmatch(r"(\d+)'([sS]?)([hdHbB])([0-9A-Fa-f_]+)", text.strip())
    assert match, f"not a sized literal: {text}"
    digits = match.group(4).replace("_", "")
    base = {"h": 16, "d": 10, "b": 2}[match.group(3).lower()]
    return int(digits, base)


def parse_int_token(raw: str) -> int:
    """Read a parameter value written either as 128 or as 8'h01."""
    raw = raw.strip()
    if "'" in raw:
        _, _, digits = raw.partition("'")
        digits = digits.replace("_", "")
        base = {"h": 16, "d": 10, "b": 2, "o": 8}[digits[0].lower()]
        return int(digits[1:], base)
    return int(raw)


# -----------------------------------------------------------------------------
# SystemVerilog port parsing
# -----------------------------------------------------------------------------

TOP_MODULE = re.compile(
    r"module\s+lingologos_top\s*#\s*\((?P<params>.*?)\)\s*\((?P<ports>.*?)\)\s*;",
    re.S,
)
PORT_DECL = re.compile(
    r"(?P<direction>input|output)\s+wire\s+logic\s+"
    r"(?P<signed>signed\s+)?"
    r"(?:\[(?P<high>\d+):(?P<low>\d+)\]\s*)?"
    r"(?P<name>[A-Za-z_][A-Za-z0-9_]*)"
)
DEBUG_REGION = re.compile(r"`ifdef\s+(?P<macro>\w+)(?P<body>.*?)`endif", re.S)
PARAM_DECL = re.compile(r"parameter\s+[^\n=]*?(?P<name>[A-Z_][A-Z0-9_]*)\s*=\s*(?P<value>[^,\n)]+)")


def split_ports(text: str) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], str]:
    """Return the public port list, the debug port list and the debug macro name."""
    match = TOP_MODULE.search(text)
    assert match, "the top level module header was not found"
    ports_text = match.group("ports")
    debug_region = DEBUG_REGION.search(ports_text)
    macro = ""
    debug_text = ""
    if debug_region:
        macro = debug_region.group("macro")
        debug_text = debug_region.group("body")
        ports_text = ports_text[: debug_region.start()] + ports_text[debug_region.end() :]
    public = [entry for entry in _port_entries(ports_text)]
    debug = [entry for entry in _port_entries(debug_text)]
    return public, debug, macro


def _port_entries(text: str) -> List[Dict[str, object]]:
    entries: List[Dict[str, object]] = []
    for match in PORT_DECL.finditer(text):
        if match.group("high") is not None:
            width = int(match.group("high")) - int(match.group("low")) + 1
        else:
            width = 1
        entries.append(
            {
                "name": match.group("name"),
                "direction": match.group("direction"),
                "width": width,
                "signed": bool(match.group("signed")),
            }
        )
    return entries


def top_parameters(text: str) -> Dict[str, int]:
    header = TOP_MODULE.search(text)
    assert header
    values: Dict[str, int] = {}
    for match in PARAM_DECL.finditer(header.group("params")):
        values[match.group("name")] = parse_int_token(match.group("value"))
    return values


# -----------------------------------------------------------------------------
# a tiny safe evaluator for the generated assign netlists
# -----------------------------------------------------------------------------

# The generated right hand sides may only use identifiers, indexing, shifts, the
# arithmetic plus/minus, the boolean operators and the conditional. An asterisk is not
# in the set, which is what keeps the claim "there is no multiplier in the datapath"
# checkable instead of decorative.
SAFE_EXPRESSION = re.compile(r"[0-9A-Za-z_()\[\]<<>+\-|&~^?:,.\s]*")
TERRIBLE = re.compile(r"__|import|lambda|exec|eval|open|global")


def _rewrite_ternary(expr: str) -> str:
    depth = 0
    question = -1
    colon = -1
    for index, char in enumerate(expr):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        elif depth == 0 and char == "?" and question < 0:
            question = index
        elif depth == 0 and char == ":" and question >= 0 and colon < 0:
            colon = index
    if question < 0:
        return expr
    if colon < 0:
        raise ValueError(f"conditional operator without a colon: {expr}")
    condition = expr[:question].strip()
    when_set = expr[question + 1 : colon].strip()
    when_clear = expr[colon + 1 :].strip()
    return (
        f"({_rewrite_ternary(when_set)} if {_rewrite_ternary(condition)} "
        f"else {_rewrite_ternary(when_clear)})"
    )


def python_expression(text: str) -> str:
    """Translate a generated SystemVerilog right hand side into Python source."""
    expr = text.strip().rstrip(";").strip()
    expr = re.sub(r"\b\d+'b([01])\b", r"\1", expr)
    expr = expr.replace("<<<", "<<").replace(">>>", ">>")
    expr = _rewrite_ternary(expr)
    expr = re.sub(r"\$(\w+)\(", r"_svfn_\1(", expr)
    matched = SAFE_EXPRESSION.fullmatch(expr)
    if matched is None or matched.group(0) != expr:
        raise ValueError(f"expression contains unexpected characters: {text}")
    if TERRIBLE.search(expr):
        raise ValueError(f"expression contains a rejected word: {text}")
    return expr


class _BitBus:
    """Gives the generated ``in_bits[3]`` selects a meaning in Python."""

    def __init__(self, value: int) -> None:
        self.value = value

    def __getitem__(self, index: int) -> int:
        return (self.value >> index) & 1


def _identity(value: int) -> int:
    return value


def evaluate_netlist(
    declarations: Sequence[Tuple[str, str]],
    reads: Sequence[str],
    input_byte: int,
) -> List[int]:
    """Evaluate a list of single assignment wire declarations in file order."""
    env: Dict[str, object] = {"in_bits": _BitBus(input_byte), "_svfn_signed": _identity}
    for name, expression in declarations:
        env[name] = eval(python_expression(expression), {"__builtins__": {}}, env)  # noqa: S307
    return [int(env[name]) for name in reads]


# -----------------------------------------------------------------------------
# generated netlist parsers
# -----------------------------------------------------------------------------

MESH_WIRE = re.compile(r"wire logic signed \[(\d+):(\d+)\] (\w+) = (.*);")
MESH_ASSIGN = re.compile(r"assign (\w+) = (.*);")
MESH_PORT = re.compile(r"(input|output)\s+logic signed \[(\d+):(\d+)\] (\w+)")

GF_ARM = re.compile(r"16'h([0-9A-Fa-f]{4}):\s*next_state = 8'h([0-9A-Fa-f]{2});")
GF_DEFAULT = re.compile(r"default:\s*next_state = 8'h([0-9A-Fa-f]{2});")

SMT_WIRE = re.compile(r"wire logic (\w+) = (.*);")
SMT_ASSIGN = re.compile(r"assign clause_results\[(\d+)\] = (\w+);")

# A procedural block is what turns combinational logic into a latch, so the generated
# nets must not contain one. The pattern is anchored at the start of a code line so
# that a comment saying "no always block" does not look like a violation.
ALWAYS_BLOCK = re.compile(r"^\s*(always_ff|always_comb|always)\b", re.M)


def parse_mesh(text: str) -> Dict[str, object]:
    ports = [
        {
            "name": match.group(4),
            "direction": match.group(1),
            "width": int(match.group(2)) - int(match.group(3)) + 1,
        }
        for match in MESH_PORT.finditer(text)
    ]
    wires = [
        (
            match.group(3),
            int(match.group(1)) - int(match.group(2)) + 1,
            match.group(4),
        )
        for match in MESH_WIRE.finditer(text)
    ]
    assigns = [(match.group(1), match.group(2)) for match in MESH_ASSIGN.finditer(text)]
    return {"ports": ports, "wires": wires, "assigns": assigns}


def mesh_row_values(mesh: Dict[str, object], row: int, inputs: Sequence[int]) -> int:
    """Evaluate one generated output row for the given INT8 lane values."""
    env: Dict[str, object] = {"_svfn_signed": _identity}
    for index, port in enumerate([p for p in mesh["ports"] if p["direction"] == "input"]):
        env[port["name"]] = inputs[index]
    for name, width, expression in mesh["wires"]:
        value = eval(python_expression(expression), {"__builtins__": {}}, env)  # noqa: S307
        env[name] = ref.to_signed(ref.to_unsigned(value, int(width)), int(width))
    target = f"out_vec_{row}"
    for name, expression in mesh["assigns"]:
        if name == target:
            value = eval(python_expression(expression), {"__builtins__": {}}, env)  # noqa: S307
            return ref.to_signed(ref.to_unsigned(value, 20), 20)
    raise AssertionError(f"the generated mesh has no assign for {target}")


def parse_gf_table(text: str) -> Tuple[Dict[Tuple[int, int], int], int]:
    arms: Dict[Tuple[int, int], int] = {}
    for match in GF_ARM.finditer(text):
        pair = int(match.group(1), 16)
        state = (pair >> 8) & 0xFF
        symbol = pair & 0xFF
        assert (state, symbol) not in arms, f"duplicate case arm for state 0x{state:02X} symbol 0x{symbol:02X}"
        arms[(state, symbol)] = int(match.group(2), 16)
    defaults = GF_DEFAULT.findall(text)
    assert len(defaults) == 1, f"the case table must have exactly one default arm, found {len(defaults)}"
    return arms, int(defaults[0], 16)


def parse_smt_netlist(text: str) -> Dict[str, object]:
    wires = [(match.group(1), match.group(2)) for match in SMT_WIRE.finditer(text)]
    assigns = [(int(match.group(1)), match.group(2)) for match in SMT_ASSIGN.finditer(text)]
    return {"wires": wires, "assigns": assigns}


def smt_lane_words(netlist: Dict[str, object]) -> Dict[int, int]:
    """Evaluate the generated clause netlist over all 256 input bytes, once."""
    reads = [name for _, name in netlist["assigns"]]
    lanes = [lane for lane, _ in netlist["assigns"]]
    words: Dict[int, int] = {}
    for byte_value in range(256):
        values = evaluate_netlist(netlist["wires"], reads, byte_value)
        word = 0
        for lane, value in zip(lanes, values):
            word |= (value & 1) << lane
        words[byte_value] = word
    return words


# -----------------------------------------------------------------------------
# vector and self-test memory readers
# -----------------------------------------------------------------------------

VECTOR_COLUMNS = [
    "in0", "in1", "in2", "in3", "gf_sym", "smt_bits",
    "exp_out0", "exp_out1", "exp_out2", "exp_out3", "exp_gf_state", "exp_smt",
]
COLUMN_WIDTHS = [2, 2, 2, 2, 2, 2, 5, 5, 5, 5, 2, 1]


def parse_golden_vectors() -> List[List[str]]:
    rows: List[List[str]] = []
    for line in read_text(GOLDEN_MEM).splitlines():
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        tokens = line.split()
        assert len(tokens) == len(VECTOR_COLUMNS), f"expected 12 columns: {line}"
        for token, width in zip(tokens, COLUMN_WIDTHS):
            assert len(token) == width, f"column width drift in {line}: {token}"
            assert re.fullmatch(r"[0-9A-F]+", token), f"non hex column in {line}: {token}"
        rows.append(tokens)
    assert rows, "the golden vector file has no data rows"
    return rows


def parse_key_values(path: Path) -> Dict[str, int]:
    values: Dict[str, int] = {}
    for line in read_text(path).splitlines():
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = int(value.strip(), 16)
    return values


# -----------------------------------------------------------------------------
# artifact helpers, read through the independent model where possible
# -----------------------------------------------------------------------------

def weight_matrix() -> List[List[int]]:
    artifact = read_json(WEIGHTS_ARTIFACT)
    return [[int(value) for value in row] for row in artifact["weight_matrix"]["values"]]


def gf_tables() -> Tuple[int, int, Dict[int, int]]:
    artifact = read_json(GF_ARTIFACT)
    polynomial = hex_byte(artifact["polynomial"])
    initial = hex_byte(artifact["initial_state"])
    multipliers = {hex_byte(symbol["byte"]): hex_byte(symbol["multiplier"]) for symbol in artifact["symbols"]}
    return polynomial, initial, multipliers


def clause_tables() -> List[List[int]]:
    artifact = read_json(SMT_ARTIFACT)
    variable_bits = {name: int(bit) for name, bit in artifact["variable_bits"].items()}
    return [
        ref.clause_truth_table(clause["expr"], variable_bits, int(artifact["input_bits"]))
        for clause in artifact["clauses"]
    ]


def gf_lane_word(byte_value: int, tables: Sequence[Sequence[int]]) -> int:
    word = 0
    for lane, table in enumerate(tables):
        word |= (table[byte_value] & 1) << lane
    return word


# -----------------------------------------------------------------------------
# gate 1
# -----------------------------------------------------------------------------

def test_chip_config_matches_the_rtl_port_list() -> None:
    """The single source of truth for widths has to agree with the RTL text."""
    config = chip_config()
    public, debug, macro = split_ports(read_text(TOP_RTL))
    assert config["top_module"] == "lingologos_top"
    assert len(public) == 18, "the datasheet pin table lists 18 user ports"
    assert len(debug) == int(config["ports"]["debug_ports"])
    assert macro == config["ports"]["debug_macro"] == "LL_DEBUG_PORTS"

    inputs = sum(port["width"] for port in public if port["direction"] == "input")
    outputs = sum(port["width"] for port in public if port["direction"] == "output")
    assert inputs == int(config["ports"]["input_bits"])
    assert outputs == int(config["ports"]["output_bits"])
    assert inputs + outputs == int(config["ports"]["total_bits"])
    assert inputs + outputs == int(config["package"]["signal_pins"])
    assert int(config["package"]["signal_pins"]) + int(config["package"]["power_pins"]) == 160

    debug_bits = sum(port["width"] for port in debug)
    assert debug_bits == 302

    names = [port["name"] for port in public + debug]
    assert len(set(names)) == len(names), "a port name is declared twice"
    assert all(port["name"].startswith("dbg_") for port in debug)
    assert not any(port["name"].startswith("dbg_") for port in public)

    datapath = config["datapath"]
    matrix = weight_matrix()
    assert int(datapath["mesh_rows"]) == len(matrix)
    assert int(datapath["mesh_columns"]) == len(matrix[0])

    top_params = top_parameters(read_text(TOP_RTL))
    assert int(datapath["core_imem_words"]) == top_params["IMEM_WORDS"]
    assert int(datapath["core_dmem_words"]) == top_params["DMEM_WORDS"]
    assert top_params["GF_INITIAL_STATE"] == gf_tables()[1]

    core_text = read_text(CORE_RTL)
    core_params = dict(
        (match.group(1), int(match.group(2)))
        for match in re.finditer(r"parameter\s+int\s+(IMEM_WORDS|IMEM_BITS|DMEM_WORDS)\s*=\s*(\d+)", core_text)
    )
    assert core_params["IMEM_WORDS"] == top_params["IMEM_WORDS"]
    assert core_params["DMEM_WORDS"] == top_params["DMEM_WORDS"]
    assert "localparam int IMEM_ADDR_BITS = IMEM_BITS + 2;" in core_text
    assert int(datapath["core_pc_bits"]) == core_params["IMEM_BITS"] + 2
    pc_width = next(port["width"] for port in debug if port["name"] == "dbg_core_pc")
    assert pc_width == core_params["IMEM_BITS"] + 2

    _, _, multipliers = gf_tables()
    gf_artifact = read_json(GF_ARTIFACT)
    assert int(datapath["gf_reachable_states"]) == int(gf_artifact["state_budget"])
    assert int(datapath["gf_defined_transitions"]) == int(gf_artifact["state_budget"]) * len(multipliers)
    assert hex_byte(datapath["gf_field_polynomial"]) == ref.DEFAULT_FIELD_POLYNOMIAL
    assert f"0x{ref.DEFAULT_FIELD_POLYNOMIAL:03X}" == datapath["gf_field_polynomial"]

    for group in ("paths",):
        for key, value in config[group].items():
            entries = value if isinstance(value, list) else [value]
            for entry in entries:
                assert (ASIC_DIR / entry).exists(), f"config path {key} points at a missing file: {entry}"


# -----------------------------------------------------------------------------
# gate 2
# -----------------------------------------------------------------------------

def test_generated_mesh_netlist_evaluates_like_a_dot_product() -> None:
    """A shift-add tree built from a different identity has to match the dot product."""
    mesh = parse_mesh(read_text(MESH_RTL))
    matrix = weight_matrix()
    config = chip_config()
    datapath = config["datapath"]

    assert len(mesh["ports"]) == int(datapath["mesh_columns"]) + int(datapath["mesh_rows"])
    assert all(port["width"] == int(datapath["input_bits_per_lane"]) for port in mesh["ports"] if port["direction"] == "input")
    assert all(port["width"] == int(datapath["accumulator_bits"]) for port in mesh["ports"] if port["direction"] == "output")
    assert len(mesh["assigns"]) == int(datapath["mesh_rows"])
    assert not any("*" in expression for _, expression in mesh["assigns"]), "a multiplier appeared in the netlist"
    assert not any("*" in expression for _, _, expression in mesh["wires"])
    assert not ALWAYS_BLOCK.search(read_text(MESH_RTL)), "a procedural block appeared in the mesh netlist"

    rows: List[Sequence[int]] = []
    rows.append([127, 127, 127, 127])
    rows.append([-128, -128, -128, -128])
    rows.append([-128, 127, -128, 127])
    rows.append([1, 1, 1, 1])
    rows.append([0, 0, 0, 0])
    rows.append([10, -5, 2, 100])
    for index in range(24):
        rows.append([((index * 37 + 11 * row) % 255) - 128 for row in range(4)])

    for inputs in rows:
        for row in range(len(matrix)):
            netlist_value = mesh_row_values(mesh, row, inputs)
            reference_value = ref.mesh_row_sum(matrix, row, inputs)
            assert netlist_value == reference_value, (
                f"row {row} of the generated mesh gives {netlist_value} for {inputs}, "
                f"the dot product gives {reference_value}"
            )

    # The accumulator headroom claim in gate H1 of the generator, recomputed here
    # without any of the generator helpers: the largest magnitude a single row can
    # reach is the sum of the absolute weights of that row times the largest input
    # magnitude, and every input magnitude is at most 128.
    limit = (1 << (int(datapath["accumulator_bits"]) - 1)) - 1
    worst_row_bound = max(sum(abs(weight) for weight in row) for row in matrix) * 128
    assert worst_row_bound == 24960, "the datasheet quotes 24960 as the largest exact output"
    assert worst_row_bound <= limit, f"worst case {worst_row_bound} does not fit {limit}"
    largest = 0
    for probe in rows:
        for row in range(len(matrix)):
            largest = max(largest, abs(ref.mesh_row_sum(matrix, row, probe)))
    assert largest <= worst_row_bound


# -----------------------------------------------------------------------------
# gate 3
# -----------------------------------------------------------------------------

def test_generated_gf_table_matches_field_arithmetic_and_the_grammar_alphabet() -> None:
    """Every arm of the case table is re-derived with the reference field code."""
    polynomial, initial, multipliers = gf_tables()
    assert ref.gf2_is_irreducible(polynomial), "the declared field polynomial is reducible"
    arms, default_state = parse_gf_table(read_text(GF_RTL))

    grammar = read_json(GRAMMAR_ARTIFACT)
    terminals = grammar["grammar_metadata"]["terminals"]
    alphabet = {ord(token) for token in terminals}
    assert alphabet == set(multipliers), "the automaton alphabet is not the grammar terminal set"
    assert alphabet == {0x28, 0x29, 0x5B, 0x5D}

    assert default_state == initial, "the default arm must load the initial state"
    assert len(arms) == len(multipliers) * 255, f"expected 1020 arms, found {len(arms)}"

    reachable = {initial}
    frontier = {initial}
    while frontier:
        nxt = set()
        for state in frontier:
            for multiplier in multipliers.values():
                nxt.add(ref.gf_mul(state, multiplier, polynomial))
        frontier = nxt - reachable
        reachable |= frontier
    assert len(reachable) == 255, f"the symbol set reaches {len(reachable)} states, not 255"
    assert 0 not in reachable, "zero must not be reachable from the initial state"

    assert {state for state, _ in arms} == reachable
    for (state, symbol), next_state in arms.items():
        expected = ref.gf_mul(state, multipliers[symbol], polynomial)
        assert next_state == expected, (
            f"state 0x{state:02X} with symbol 0x{symbol:02X} enters 0x{next_state:02X}, "
            f"field arithmetic gives 0x{expected:02X}"
        )
    assert 0 not in set(arms.values())

    for multiplier in sorted(set(multipliers.values())):
        assert ref.gf_order(multiplier, polynomial) == 255
    inverse = ref.gf_inverse(0x02, polynomial)
    assert inverse == 0x8E, "the close bracket multiplier is not the measured inverse of 0x02"

    symbols_of = {}
    for state, symbol in arms:
        symbols_of.setdefault(state, set()).add(symbol)
    for state in reachable:
        assert symbols_of[state] == set(multipliers), (
            f"state 0x{state:02X} is defined for {sorted(symbols_of[state])}, "
            "not for the whole alphabet"
        )
    for state in reachable:
        for symbol in range(256):
            if symbol not in multipliers:
                assert (state, symbol) not in arms, "an unexpected symbol must fall in the default arm"
    for symbol in multipliers:
        assert (0x00, symbol) not in arms, "an unreachable state must not occupy a case arm"


# -----------------------------------------------------------------------------
# gate 4
# -----------------------------------------------------------------------------

def test_generated_smt_netlist_matches_the_reference_masks_on_every_byte() -> None:
    """The clause network is a mask built by a different parser, checked byte by byte."""
    text = read_text(SMT_RTL)
    netlist = parse_smt_netlist(text)
    artifact = read_json(SMT_ARTIFACT)
    variable_bits = {name: int(bit) for name, bit in artifact["variable_bits"].items()}
    lanes = int(artifact["result_lanes"])

    assert not ALWAYS_BLOCK.search(text), "a clause network with an always block would not be latch free"
    assert "<=" not in text, "a non blocking assignment inside the clause network"
    assert len(netlist["assigns"]) == lanes
    assert [lane for lane, _ in netlist["assigns"]] == list(range(lanes))
    built = compiler.build_smt_netlist(artifact)
    clause_wires = [name for name, _ in netlist["wires"] if name.startswith("cl")]
    assert len(clause_wires) == sum(int(clause["nodes"]) for clause in built["clauses"])
    assert len([name for name, _ in netlist["wires"] if name.startswith("v_")]) == len(variable_bits)

    # Three independent derivations of the same lane word: the reference shunting-yard
    # model, the clause parser inside the generator, and the generated netlist text
    # evaluated by the small evaluator in this file.
    reference_tables = [
        ref.clause_truth_table(str(clause["expr"]), variable_bits, int(artifact["input_bits"]))
        for clause in artifact["clauses"]
    ]
    generator_tables = [clause["truth_table"] for clause in built["clauses"]]
    assert generator_tables == reference_tables, (
        "the clause parser in the generator and the reference model disagree"
    )

    netlist_words = smt_lane_words(netlist)
    for byte_value in range(256):
        expected = gf_lane_word(byte_value, reference_tables)
        assert netlist_words[byte_value] == expected, (
            f"input byte 0x{byte_value:02X}: the netlist gives lane word "
            f"0x{netlist_words[byte_value]:X} and the reference model gives 0x{expected:X}"
        )
    assert {word for word in netlist_words.values()} == {
        gf_lane_word(byte_value, reference_tables) for byte_value in range(256)
    }

    assert int(chip_config()["datapath"]["smt_result_lanes"]) == lanes
    assert int(chip_config()["datapath"]["smt_input_bits"]) == int(artifact["input_bits"])

    # The logic depth quoted in the chip configuration is measured here on the
    # dependency graph of the generated wires, which is the depth a synthesis tool
    # would actually see, and not on the tree that the generator parsed.
    wire_names = {name for name, _ in netlist["wires"]}
    wire_expression = {name: expression for name, expression in netlist["wires"]}

    def cone_depth(name: str, visited: List[str]) -> int:
        if name in visited:
            raise AssertionError(f"a clause netlist wire depends on itself: {name}")
        if name not in wire_expression or name.startswith("v_"):
            return 0
        children = [
            token
            for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", wire_expression[name])
            if token in wire_names
        ]
        nested = visited + [name]
        return 1 + max((cone_depth(child, nested) for child in children), default=0)

    lane_depths = [cone_depth(source, []) for _, source in netlist["assigns"]]
    assert max(lane_depths) == int(chip_config()["datapath"]["smt_max_logic_depth"]), (
        f"the generated netlist is {max(lane_depths)} levels deep, the configuration says "
        f"{chip_config()['datapath']['smt_max_logic_depth']}"
    )
    assert all(depth > 0 for depth in lane_depths), "a result lane is tied to a primary input"

    lane_rows = re.findall(r"^\| (\d) \| `([^`]+)` \| (\d+) \| (\d+) \|$", read_text(DATASHEET), re.M)
    assert len(lane_rows) == lanes, "the datasheet clause table does not list every result lane"
    for lane_text, quoted, nodes, depth in lane_rows:
        lane = int(lane_text)
        assert quoted == str(built["clauses"][lane]["expr"]), (
            f"the datasheet quotes a different expression for lane {lane}"
        )
        assert int(nodes) == int(built["clauses"][lane]["nodes"]), (
            f"the datasheet quotes {nodes} nodes for lane {lane}, the parser counted "
            f"{built['clauses'][lane]['nodes']}"
        )
        assert int(depth) == lane_depths[lane], (
            f"the datasheet quotes depth {depth} for lane {lane}, the generated netlist is "
            f"{lane_depths[lane]} levels deep"
        )


# -----------------------------------------------------------------------------
# gate 5
# -----------------------------------------------------------------------------

def test_golden_vectors_are_reproduced_without_the_generator() -> None:
    """Every column of every committed row is recomputed from the artifacts."""
    rows = parse_golden_vectors()
    matrix = weight_matrix()
    polynomial, initial, multipliers = gf_tables()
    tables = clause_tables()

    state = initial
    for index, row in enumerate(rows):
        inputs = [ref.to_signed(int(token, 16), 8) for token in row[0:4]]
        symbol = int(row[4], 16)
        smt_byte = int(row[5], 16)

        for lane in range(4):
            assert inputs[lane] >= 0 - 128 and inputs[lane] <= 127
            expected = ref.to_unsigned(ref.mesh_row_sum(matrix, lane, inputs), 20)
            assert int(row[6 + lane], 16) == expected, (
                f"row {index} mesh lane {lane}: file says {row[6 + lane]}, "
                f"the dot product gives 0x{expected:05X}"
            )

        state = ref.gf_mul(state, multipliers[symbol], polynomial) if symbol in multipliers else initial
        assert int(row[10], 16) == state, f"row {index} automaton state: file {row[10]}, model 0x{state:02X}"

        word = gf_lane_word(smt_byte, tables)
        assert int(row[11], 16) == word, f"row {index} clause lanes: file {row[11]}, model 0x{word:X}"

    assert len(rows) == 304
    assert rows[0][10] == "01", "the first row is the reset row and must show the initial state"
    visited = {row[10] for row in rows}
    assert len(visited) == 255
    assert len({row[5] for row in rows}) == 256, "the exhaustive input sweep must cover all 256 bytes"
    assert rows[1][0:4] == ["0A", "FB", "02", "64"], "the frozen self-test stimulus must be a vector row"


# -----------------------------------------------------------------------------
# gate 6
# -----------------------------------------------------------------------------

def test_generator_is_deterministic_and_check_mode_passes(tmp_path: Path) -> None:
    """A scratch regeneration must match the committed files byte for byte."""
    rtl_dir = tmp_path / "rtl"
    sim_dir = tmp_path / "sim"
    code = compiler.main(
        [
            "--out-dir", str(rtl_dir),
            "--sim-dir", str(sim_dir),
        ]
    )
    assert code == 0, "the generator returned a failing verdict"
    pairs = [
        (MESH_RTL, rtl_dir / "weight_rom_mesh.sv"),
        (GF_RTL, rtl_dir / "gf_automaton_1cycle.sv"),
        (SMT_RTL, rtl_dir / "smt_evaluator.sv"),
        (GOLDEN_MEM, sim_dir / "golden_vectors.mem"),
        (SELFTEST_MEM, sim_dir / "core_self_test.mem"),
    ]
    for committed, scratch in pairs:
        assert scratch.is_file(), f"the generator did not write {scratch.name}"
        left = committed.read_bytes()
        right = scratch.read_bytes()
        assert left == right, f"{committed.relative_to(REPO_ROOT)} is stale with respect to the artifacts"

    check = subprocess.run(
        [sys.executable, str(ASIC_DIR / "scripts" / "lingologos_compiler.py"), "--check"],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        check=False,
    )
    assert check.returncode == 0, check.stdout + check.stderr
    assert "FINAL VERDICT: PASS" in check.stdout
    assert "Stale generated files" not in check.stdout


# -----------------------------------------------------------------------------
# gate 7
# -----------------------------------------------------------------------------

def test_manifest_digests_match_the_committed_files() -> None:
    """The digest in the manifest is the hash of the bytes on disk.

    Digests are computed over bytes, so anyone can reproduce them with sha256sum, and
    a generated file whose line endings depend on the host that wrote it would not
    hash to the same value on Linux and on Windows.
    """
    manifest = read_json(MANIFEST)
    assert manifest["generator"] == "asic_chip/scripts/lingologos_compiler.py"
    assert all(bool(value) for value in manifest["gates"].values()), manifest["gates"]
    for name, digest in manifest["output_digests"].items():
        raw = (ASIC_DIR / "rtl" / "generated" / name).read_bytes()
        assert hashlib.sha256(raw).hexdigest()[:16] == digest, f"{name} changed after the manifest"
    for name, digest in manifest["artifact_digests"].items():
        raw = (ASIC_DIR / "artifacts" / name).read_bytes()
        assert hashlib.sha256(raw).hexdigest()[:16] == digest, f"{name} changed after the manifest"
    golden = GOLDEN_MEM.read_bytes()
    assert hashlib.sha256(golden).hexdigest()[:16] == manifest["golden_vectors_digest"]
    selftest = SELFTEST_MEM.read_bytes()
    assert hashlib.sha256(selftest).hexdigest()[:16] == manifest["self_test_digest"]

    numbers = manifest["measurements"]
    assert numbers["H1_mesh"]["accumulator_limit"] == 524287
    assert numbers["H2_gf"]["defined_transitions"] == 1020
    assert numbers["H4_vectors"]["rows"] == 304
    assert numbers["H4_vectors"]["seed"] == 42


# -----------------------------------------------------------------------------
# gate 8
# -----------------------------------------------------------------------------

def test_self_test_memory_matches_the_reference_engines() -> None:
    values = parse_key_values(SELFTEST_MEM)
    matrix = weight_matrix()
    polynomial, initial, multipliers = gf_tables()
    tables = clause_tables()

    stimulus = [values[f"in_vector_{lane}"] for lane in range(4)]
    signed = [ref.to_signed(value, 8) for value in stimulus]
    assert all(-128 <= value <= 127 for value in signed)

    row_sums = [ref.mesh_row_sum(matrix, lane, signed) for lane in range(4)]
    assert len(set(row_sums)) == 4, "the four rows must differ, otherwise the readback check proves nothing"
    row_index = values["mesh_row_index"]
    assert 0 <= row_index < len(matrix)
    assert ref.to_unsigned(row_sums[row_index], 32) == values["exp_x11"]
    for lane in range(4):
        assert ref.to_unsigned(row_sums[lane], 32) != values["exp_x11"] or lane == row_index

    assert gf_lane_word(values["smt_in_bits"], tables) == values["exp_x8"]

    symbol = values["gf_step_symbol"]
    assert symbol in multipliers
    assert ref.gf_mul(initial, multipliers[symbol], polynomial) == values["exp_x10"]
    assert values["gf_symbol_in"] not in multipliers, "the pad symbol must idle the automaton"
    assert values["exp_gf_state_final"] == initial

    assert values["exp_x13"] & 0xF == 3, "the status counter is read before the read itself is counted"
    assert values["exp_cop_count"] == (values["exp_x13"] & 0xF) + 1, (
        "the counter register also counts the request the status word was read on"
    )
    assert values["exp_x13"] >> 4 == values["exp_core_illegal_seen"]
    assert values["exp_x15"] == 0
    assert values["exp_core_pass"] == 1
    assert len(values) == 17, "the self-test memory changed shape"


# -----------------------------------------------------------------------------
# gate 9
# -----------------------------------------------------------------------------

def test_testbench_literals_match_the_self_test_memory() -> None:
    text = read_text(TESTBENCH_SV)
    values = parse_key_values(SELFTEST_MEM)

    for lane, key in enumerate(("in_vector_0", "in_vector_1", "in_vector_2", "in_vector_3")):
        literals = re.findall(rf"{key}\s*=\s*(\d+'[sS]?[hH][0-9A-Fa-f_]{{2}})", text)
        assert literals, f"the testbench never drives {key}"
        values_seen = [parse_sv_literal(token) for token in literals]
        assert values[key] in values_seen, (
            f"{key} is driven with {values_seen} and never with the self-test value 0x{values[key]:02X}"
        )
    for key in ("gf_symbol_in", "smt_in_bits"):
        literals = re.findall(rf"^\s*{key}\s*=\s*(\d+'[sS]?[hH][0-9A-Fa-f_]{{2}});", text, re.M)
        assert literals, f"the testbench never drives {key}"
        values_seen = [parse_sv_literal(token) for token in literals]
        assert values[key] in values_seen, (
            f"{key} is driven with {values_seen} and never with the self-test value 0x{values[key]:02X}"
        )

    expected = {
        "out_vector_2": ("exp_x11", 20),
        "dbg_reg_x8": ("exp_x8", 32),
        "dbg_reg_x10": ("exp_x10", 32),
        "dbg_reg_x11": ("exp_x11", 32),
        "dbg_reg_x13": ("exp_x13", 32),
        "dbg_reg_x15": ("exp_x15", 32),
        "dbg_cop_count": ("exp_cop_count", 4),
        "gf_next_state": ("exp_gf_state_final", 8),
        "core_pass": ("exp_core_pass", 1),
        "core_illegal_seen": ("exp_core_illegal_seen", 1),
    }
    for signal, (key, width) in expected.items():
        pattern = rf"{signal}\s*!==\s*((?:\d+)'[sS]?[hdHb][0-9A-Fa-f_]+)"
        match = re.search(pattern, text)
        assert match, f"the testbench never checks {signal}"
        literal = parse_sv_literal(match.group(1))
        want = values[key] & ((1 << width) - 1)
        assert literal == want, (
            f"the testbench expects {signal} to be 0x{literal:X}, the self-test memory says 0x{want:X}"
        )

    waits = [int(token) for token in re.findall(r"#(\d+);", text)]
    assert waits, "the testbench has no timed control at all"
    assert max(waits) >= 20000, "the watchdog no longer leaves room for the self-test"

    data_rows = 0
    for line in read_text(GOLDEN_MEM).splitlines():
        if line and not line.startswith("//"):
            data_rows += 1
    stated = re.search(r"GOLDEN_ROWS\s*=\s*(\d+)", text)
    assert stated, "the testbench does not state how many golden rows it loads"
    assert int(stated.group(1)) == data_rows, (
        f"the testbench loads {stated.group(1)} golden rows and the vector file holds {data_rows}"
    )


def test_harness_cycle_budget_covers_the_measured_self_test() -> None:
    text = read_text(HARNESS_CPP)
    match = re.search(r"cycles\s*<\s*(\d+)", text)
    assert match, "the C++ harness has no cycle bound"
    budget = int(match.group(1))

    datapath = chip_config()["datapath"]
    imem_words = int(datapath["core_imem_words"])
    dmem_words = int(datapath["core_dmem_words"])
    rom = {
        int(match.group(1)): parse_int_token("32'h" + match.group(2))
        for match in re.finditer(r"instruction_rom\[(\d+)\]\s*=\s*32'h([0-9A-Fa-f_]+);", read_text(CORE_RTL))
    }
    assert len(rom) == imem_words, "the program memory is not a full image"
    assert set(rom) == set(range(imem_words)), "the program memory image has gaps"

    polynomial, initial, multipliers = gf_tables()
    run = ref.run_firmware(
        rom,
        weight_matrix(),
        initial,
        multipliers,
        clause_tables(),
        (0, 0, 0, 0),
        0x00,
        imem_words,
        dmem_words,
    )
    assert run.halted, "the reference interpreter did not halt the firmware"
    assert run.cycles < budget, f"the self-test needs {run.cycles} cycles and the harness waits for {budget}"


# -----------------------------------------------------------------------------
# gate 10
# -----------------------------------------------------------------------------

def test_harness_touches_only_declared_top_level_ports() -> None:
    public, debug, _ = split_ports(read_text(TOP_RTL))
    declared = {port["name"] for port in public + debug}
    text = read_text(HARNESS_CPP)
    used = {match.group(1) for match in re.finditer(r"chip->([A-Za-z_]\w*)", text)}
    methods = {"eval", "trace", "final", "expected", "width"}
    names = used - methods
    unknown = sorted(names - declared)
    assert not unknown, f"the C++ harness touches signals that are not ports: {unknown}"

    for port in public:
        assert port["name"] in names, f"the harness ignores the pad signal {port['name']}"
    driven = {match.group(1) for match in re.finditer(r"chip->([A-Za-z_]\w*)\s*=", text)}
    for port in public:
        if port["direction"] == "input":
            assert port["name"] in driven, f"the harness never drives {port['name']}"
    debug_used = sorted(name for name in names if name.startswith("dbg_"))
    assert debug_used, "the harness reads no debug port, so the self-test checks less than it claims"
    for name in debug_used:
        assert name in {port["name"] for port in debug}, f"{name} is not a debug port"

    instantiates = {
        match.group(1) for match in re.finditer(r"\.([A-Za-z_]\w*)\s*\(", read_text(TESTBENCH_SV))
    }
    assert declared <= instantiates, "the testbench leaves a top level port unconnected"


# -----------------------------------------------------------------------------
# gate 11
# -----------------------------------------------------------------------------

PIN_ROW = re.compile(
    r"^\|\s*(?P<first>\d+)(?:-(?P<last>\d+))?\s*\|\s*(?P<side>left|top|right|bottom)\s*\|"
    r"\s*(?P<signal>[A-Za-z_][A-Za-z0-9_]*)\s*\|\s*(?P<width>\d+)\s*\|"
    r"\s*(?P<direction>input|output|power)\s*\|",
    re.M,
)
PIN_PLAN_SIDES = {"N": "top", "E": "right", "W": "left", "S": "bottom"}
# The pin plan is read by the pin constraint grammar of OpenLane 2, not by the line
# oriented format of OpenLane 1. That grammar has no comment token, and the only
# characters it will lex inside a pattern, besides letters, digits and underscores, are
# these. A dash inside brackets, which reads as a character range to a human, is a
# token recognition error there.
PIN_PLAN_PATTERN_CHARS = set("\\.*+?()[]|")


def test_pinout_agrees_in_rtl_config_datasheet_and_pin_plan() -> None:
    """One pin table, four consumers: RTL, config, datasheet, OpenLane pin order."""
    config = chip_config()
    public, debug, _ = split_ports(read_text(TOP_RTL))
    rtl_width = {port["name"]: int(port["width"]) for port in public}
    rtl_direction = {port["name"]: str(port["direction"]) for port in public}

    rows = list(PIN_ROW.finditer(read_text(DATASHEET)))
    assert rows, "the datasheet pin table could not be parsed"
    expected_pins = 160
    covered: List[int] = []
    datasheet_side: Dict[str, str] = {}
    signal_rows = 0
    power_rows: Dict[str, int] = {}
    for match in rows:
        first = int(match.group("first"))
        last = int(match.group("last") or match.group("first"))
        signal = match.group("signal")
        width = int(match.group("width"))
        direction = match.group("direction")
        assert last >= first, f"pin range is reversed in the datasheet: {signal}"
        span = last - first + 1
        covered.extend(range(first, last + 1))
        if direction == "power":
            power_rows[signal] = span
            continue
        signal_rows += 1
        assert span == width, f"{signal} spans {span} pins but is declared {width} bits wide"
        assert signal in rtl_width, f"{signal} is in the pin table but not in the RTL port list"
        assert rtl_width[signal] == width, f"{signal} width differs between datasheet and RTL"
        assert rtl_direction[signal] == direction, f"{signal} direction differs between datasheet and RTL"
        assert signal not in datasheet_side, f"{signal} appears twice in the pin table"
        datasheet_side[signal] = match.group("side")

    assert len(set(covered)) == len(covered), "pin ranges overlap in the datasheet"
    assert sorted(covered) == list(range(1, expected_pins + 1)), "pin numbers are not a full, non overlapping 1..160 tiling"
    assert signal_rows == len(rtl_width), "the pin table does not cover every user port exactly once"
    assert set(power_rows) == {"VDD", "VSS"}
    assert sum(power_rows.values()) == int(config["package"]["power_pins"])
    assert all(count == 7 for count in power_rows.values())

    # The tool matches its own bit level pin names against these patterns anchored at
    # both ends, so the gate builds the same names from the RTL and applies the same
    # anchoring. A pattern that matches nothing, or that leaves a pin unplaced, is what
    # stopped the first run at the custom I/O placement step.
    bit_names: Dict[str, str] = {}
    for name, width in rtl_width.items():
        if width == 1:
            bit_names[name] = name
        else:
            for index in range(width):
                bit_names[f"{name}[{index}]"] = name

    plan = []
    side: Optional[str] = None
    for raw in read_text(PIN_ORDER).splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            side = PIN_PLAN_SIDES.get(line[1:].strip())
            assert side, f"the pin plan has a directive the pin grammar does not know: {line}"
            continue
        stray = sorted({c for c in line if not (c.isalnum() or c == "_" or c in PIN_PLAN_PATTERN_CHARS)})
        assert not stray, f"the pin plan pattern {line} uses {stray}, which the pin grammar cannot lex"
        assert side is not None, f"the pin plan pattern {line} appears before any side directive"
        plan.append((side, line))
    assert plan, "the pin order plan has no patterns"

    hits: Dict[str, List[str]] = {name: [] for name in rtl_width}
    bit_side: Dict[str, str] = {}
    for side, pattern in plan:
        matcher = re.compile(pattern)
        found = [bit for bit in bit_names if matcher.fullmatch(bit)]
        assert found, f"the pin order pattern {pattern} matches no pin of the netlist"
        for bit in found:
            assert bit not in bit_side, f"{bit} is matched twice in the pin order plan"
            bit_side[bit] = side
            hits[bit_names[bit]].append(side)
    unplaced = sorted(set(bit_names) - set(bit_side))
    assert not unplaced, f"{len(unplaced)} pins of the netlist are in no pin order pattern, for example {unplaced[:5]}"
    for name, sides in hits.items():
        assert sides, f"{name} is matched by no pin order pattern"
        assert len(set(sides)) == 1, f"{name} is placed on more than one side in the pin order plan"
        assert sides[0] == datasheet_side[name], f"{name} is placed on {sides[0]} in the plan and {datasheet_side[name]} in the datasheet"
    for side in ("left", "top", "right", "bottom"):
        bits = sum(1 for placed_on in bit_side.values() if placed_on == side)
        declared = sum(
            int(match.group("width"))
            for match in rows
            if match.group("direction") != "power" and match.group("side") == side
        )
        assert bits == declared, f"the {side} side carries {bits} bits in the plan and {declared} in the datasheet"

    openlane = read_json(OPENLANE_CONFIG)
    assert openlane["FP_PIN_ORDER_CFG"] == "dir::pin_order.cfg"
    assert (OPENLANE_CONFIG.parent / "pin_order.cfg").is_file()


# -----------------------------------------------------------------------------
# gate 12
# -----------------------------------------------------------------------------

def test_bringup_constraint_set_differs_from_the_spec_only_in_the_period() -> None:
    """The bring-up set exists to reach a routed layout, not to move the goalposts."""
    config = chip_config()
    spec = read_json(OPENLANE_CONFIG)
    bringup_path = OPENLANE_CONFIG.parent / "config.bringup.json"
    assert bringup_path.is_file(), "no bring-up configuration next to config.json"
    bringup = read_json(bringup_path)

    assert float(bringup["CLOCK_PERIOD"]) == float(config["bringup_clock_period_ns"])
    assert float(bringup["CLOCK_PERIOD"]) > float(spec["CLOCK_PERIOD"])
    assert bringup["SDC"] == ["dir::lingologos.bringup.sdc"]
    differing = {
        key
        for key in set(spec) | set(bringup)
        if not key.startswith("NOTE_") and spec.get(key) != bringup.get(key)
    }
    assert differing == {"CLOCK_PERIOD", "SDC", "PNR_SDC_FILE", "SIGNOFF_SDC_FILE"}, (
        "the bring-up set changed more than the period and the three keys that name the "
        f"constraint file holding it: {sorted(differing)}"
    )
    for one in (spec, bringup):
        assert one["PNR_SDC_FILE"] == one["SIGNOFF_SDC_FILE"] == one["SDC"][0], (
            "a set whose place and route and signoff constraint files are unnamed times the "
            "design against the generic file openlane falls back to"
        )
    for one in (spec, bringup):
        assert one["SETUP_VIOLATION_CORNERS"] == ["*"], (
            "a setup violation in a corner outside this list is a warning and nothing more, and "
            "what OpenLane installs for a sky130 PDK by default is the typical corner only"
        )
        assert one["HOLD_VIOLATION_CORNERS"] == ["*"], (
            "the hold violations of the complete run were all in the corners it was measured at"
        )
        for absent in ("QUIT_ON_TIMING_VIOLATIONS", "RUN_CVC", "QUIT_ON_LINTER_ERRORS",
                       "DIODE_INSERTION_STRATEGY"):
            assert absent not in one, (
                f"{absent} is a key this OpenLane does not read, or one it rewrites other keys "
                "from: it stays in a configuration only as a warning in the log and a control "
                "nobody has"
            )
        assert one["RUN_HEURISTIC_DIODE_INSERTION"] is True, (
            "the antenna check of the complete run failed on 54 pins and 46 nets, and the "
            "heuristic diode insertion step is the one the legacy strategy key switched off"
        )
        assert one["DIODE_ON_PORTS"] == "in"
        assert int(one["HEURISTIC_ANTENNA_THRESHOLD"]) < 90
        assert int(one["GRT_ANTENNA_ITERS"]) > 3 and int(one["GRT_ANTENNA_MARGIN"]) > 10
        assert one["RUN_POST_GRT_DESIGN_REPAIR"] is True
        assert one["RUN_POST_GRT_RESIZER_TIMING"] is True
        # The run at the 30.0 ns gate measured the gap these two margins have to cover: the
        # resizer closed hold at +0.100201 ns in nom_tt_025C_1v80 and the same corner read
        # -0.058548 ns at signoff on the extracted parasitics of the routed layout.
        assert float(one["PL_RESIZER_HOLD_SLACK_MARGIN"]) >= 0.30, (
            "the installed default placement resizer margin stops a hair past zero, and the loss "
            "between the last hold repair and signoff measured 0.16 ns on this design"
        )
        assert float(one["GRT_RESIZER_HOLD_SLACK_MARGIN"]) >= 0.30, (
            "the resizer after the global routing is the last one the Classic flow runs, so its "
            "installed default margin of 0.05 ns decides how much hold the routed layout keeps"
        )
        for trades_setup in ("PL_RESIZER_ALLOW_SETUP_VIOS", "GRT_RESIZER_ALLOW_SETUP_VIOS"):
            assert trades_setup not in one, (
                "a resizer told to allow setup violations buys the hold side with the setup side, "
                "and setup is what the signoff of the complete runs failed on"
            )

    sdc = read_text(OPENLANE_CONFIG.parent / "lingologos.bringup.sdc")
    clock = re.search(r"create_clock\s+-name\s+(\w+)\s+-period\s+([\d.]+)", sdc)
    assert clock, "the bring-up constraint file does not declare a clock"
    assert clock.group(1) == config["clock_port"]
    assert float(clock.group(2)) == float(config["bringup_clock_period_ns"])

    script = read_text(ASIC_DIR / "scripts" / "build_gds.sh")
    assert "config.bringup.json" in script, "the flow entry point cannot select the bring-up set"
    assert "LL_OPENLANE_CONFIG" in script, "the choice is not overridable from the environment"
    assert 'FLOW_STATUS=$?' in script, (
        "a flow that ends in deferred signoff errors still wrote a GDSII and the reports that say "
        "what failed, and an entry point that dies at that moment never prints them"
    )
    assert 'exit "$FLOW_STATUS"' in script, "the status of the flow must reach the caller"


def test_setup_installs_openlane_only_on_an_interpreter_with_prebuilt_wheels() -> None:
    """A newer interpreter has no wheel for libparse, and its source build failed here."""
    setup = read_text(ASIC_DIR / "scripts" / "setup_openlane_flow.sh")

    candidates = re.search(r"for candidate in ([^\n]+?)\s+do", setup)
    assert candidates, "the setup script does not name the interpreters it will accept"
    offered = candidates.group(1).split()
    assert offered[0] == '"${LL_PYTHON:-}"', "an explicit interpreter must win over the search"
    assert "python3.12" in offered, "the newest interpreter with a libparse wheel must be offered"
    for too_new_name in ("python3.13", "python3.14"):
        assert too_new_name not in offered, f"{too_new_name} has no prebuilt libparse wheel"
    assert "python3" in offered[-1], "the distribution default stays last"

    assert "(3, 13)" in setup, (
        "the ceiling that rejects an interpreter without wheels has moved"
    )
    assert 'rm -rf "$VENV"' in setup, (
        "a virtual environment left on an interpreter pip cannot install into is reused"
    )
    assert "--no-ppa" in setup and "deadsnakes" in setup, (
        "Ubuntu 26.04 carries no older interpreter, so the route to python3.12 is documented in the script"
    )
    assert "python-build-standalone" in setup and "cpython-3.12." in setup, (
        "the script gives up instead of obtaining an interpreter with wheels itself"
    )
    assert "LL_PYTHON_URL" in setup, "the interpreter the script downloads cannot be replaced"
    assert "python3.12-tk" in setup and "import tkinter" in setup, (
        "openlane imports tkinter before it reads a design, and an interpreter can be the "
        "right version and still not carry that module"
    )

    instructions = read_text(ASIC_DIR / "doc" / "build_instructions.md")
    assert "ppa:deadsnakes/ppa" in instructions
    assert "python-build-standalone" in instructions, (
        "the interpreter the script fetches when apt fails is not in the instructions"
    )
    assert "cp312" in instructions, "the wheel ceiling is not recorded in the instructions"
    report = read_text(ASIC_DIR / "doc" / "verification_report.md")
    assert "No rule to make target" in report, (
        "the source build that failed on this machine is not recorded as a result"
    )
    assert "CPython 3.12.16" in report, (
        "the install that proves the interpreter ceiling was not recorded as executed"
    )


def test_setup_and_flow_drive_the_volare_and_openlane_that_are_actually_installed() -> None:
    """The volare installed here has no --global option, and its openlane is not on PATH."""
    setup = read_text(ASIC_DIR / "scripts" / "setup_openlane_flow.sh")
    entry = read_text(ASIC_DIR / "scripts" / "build_gds.sh")

    assert 'volare" --global' not in setup, (
        "volare 0.20 answers --global with an error of its own, which the discarded stderr "
        "of this script used to turn into a complaint about the network"
    )
    assert '"$VENV/bin/volare" ls-remote --pdk sky130' in setup, (
        "the version listing must be the subcommand of the volare this flow installs"
    )
    assert 'volare" fetch --pdk-root "$PDK_ROOT" --pdk sky130 $PDK_FETCH_ARGS' in setup, (
        "the PDK root and the family are options of the subcommand there, not of the command"
    )
    assert "LL_PDK_LIBRARIES" in setup, (
        "one PDK version with every library measured 16 GiB on this machine against 2.1 GiB for "
        "the set this flow reads, so the large set has to be a choice and never the default"
    )
    assert 'volare" enable --pdk-root' in setup, (
        "enabling is what leaves a sky130A under the PDK root for the flow to find"
    )

    assert "export LL_OPENLANE_BIN" in setup, (
        "the setup installs openlane somewhere PATH does not reach and does not say where"
    )
    assert "find_openlane2" in entry and "LL_OPENLANE_BIN" in entry, (
        "the flow entry point cannot reach a tool installed in a virtual environment"
    )
    assert '"$HOME/openlane-venv/bin/openlane"' in entry, (
        "the environment the one command setup creates is not among the places searched"
    )
    assert '--dockerized "$DESIGN_DIR/$FLOW_CONFIG"' in entry, (
        "a pip install of openlane carries no tools, so the container image has to be asked for"
    )
    assert "${PDK_ROOT:-}" in entry, (
        "PDK_ROOT has a default in the setup and none in the entry point run on its own"
    )

    # The flow has run, so the tree has to talk about the run it has, not only about the
    # one it hopes for. These assertions hold the documentation to what the tools printed.
    run_dir = "asic_chip/openlane/runs/RUN_2026-11-09_16-15-02"
    report = read_text(ASIC_DIR / "doc" / "verification_report.md")
    instructions = read_text(ASIC_DIR / "doc" / "build_instructions.md")
    datasheet = read_text(DATASHEET)
    assert run_dir in entry, (
        "the entry point still looks for a run directory only where the tool has never written one"
    )
    assert run_dir in instructions and "01-verilator-lint" in instructions, (
        "the instructions do not say which steps of the first real run left logs, and where"
    )
    assert "25-odb-customioplacement" in report and "token recognition error at: '-'" in report, (
        "the step that stopped the first run, and the error it printed, are not in the report"
    )
    second_run = "asic_chip/openlane/runs/RUN_2026-11-09_18-43-18"
    assert second_run in report and second_run in instructions and second_run in entry, (
        "the run that went through all 78 steps of the Classic flow is not named where a reader "
        "would look for it"
    )
    third_run = "asic_chip/openlane/runs/RUN_2026-11-09_21-14-43"
    assert third_run in report and third_run in instructions and third_run in entry, (
        "the run whose manufacturability report says Antenna Passed, LVS Passed and DRC Passed is "
        "not named where a reader would look for it"
    )
    assert "Antenna Passed" in report and "Antenna Passed" in instructions, (
        "a reader of these documents still finds only the failed antenna check of the first "
        "complete run and never the one that passed"
    )
    assert "PL_RESIZER_HOLD_SLACK_MARGIN" in entry and "0.40" in entry, (
        "the entry point does not say which hold margin the constraint sets ask the two resizers "
        "for, which is the knob the second complete run measured the need for"
    )
    assert "-0.4906" in datasheet and "32.0 ns" in datasheet, (
        "the slack by which the second layout missed the slow corner, and the period the next run "
        "is held to, are not in the datasheet"
    )
    assert "Final result: Circuits match uniquely." in report, (
        "the line netgen printed when the extracted layout was compared against the synthesized "
        "netlist is not quoted where the outcome of that check is recorded"
    )
    assert "circuits match uniquely" in instructions, (
        "the instructions do not say what the layout versus schematic check concluded"
    )
    for fact in ("Antenna Failed", "SETUP_VIOLATION_CORNERS", "HOLD_VIOLATION_CORNERS"):
        assert fact in report, f"the signoff outcome {fact!r} of that run is missing from the report"
        assert fact in instructions, f"the signoff outcome {fact!r} of that run is missing from the instructions"
    assert "169,306" in datasheet and "17,552" in datasheet, (
        "the area and instance count a real place and route measured for this design are not in "
        "the datasheet, only the pre-layout synthesis figures"
    )
    assert "149097.996800" in report, (
        "the only area figure a real tool produced for this design is not recorded as measured"
    )
    assert "149,098 square micrometers" in datasheet and "11,015" in datasheet, (
        "the pre-layout cell count and cell area of that synthesis are not in the datasheet"
    )
    assert "#W" in instructions and "@min_distance" in instructions, (
        "the grammar the pin plan file has to be written in is documented nowhere a reader looks"
    )
    for name in ("config.json", "config.bringup.json"):
        openlane_config = read_json(ASIC_DIR / "openlane" / name)
        assert (
            openlane_config["PNR_SDC_FILE"]
            == openlane_config["SIGNOFF_SDC_FILE"]
            == openlane_config["SDC"][0]
        ), (
            "openlane warns that the place and route and the signoff constraint files are "
            "undefined and then times the design against a generic file instead of this one"
        )

    instructions_text = instructions
    assert "ls-remote --pdk sky130" in instructions_text, (
        "the volare spelling the script now uses is not in the instructions"
    )
    assert "No such option '--global'" in instructions, (
        "the spelling that fails, and the error it prints, are not recorded where a reader looks"
    )
    assert "No module named 'tkinter'" in instructions, (
        "the interpreter that installs openlane correctly and still cannot start it is not documented"
    )


def test_openlane_setup_agrees_with_the_chip_configuration() -> None:
    config = chip_config()
    openlane = read_json(OPENLANE_CONFIG)
    public, _, _ = split_ports(read_text(TOP_RTL))
    inputs = {port["name"] for port in public if port["direction"] == "input"}

    assert openlane["DESIGN_NAME"] == config["top_module"]
    assert openlane["PDK"] == config["pdk"]
    assert openlane["STD_CELL_LIBRARY"] == config["std_cell_library"]
    assert openlane["CLOCK_PORT"] == config["clock_port"]
    assert openlane["CLOCK_PORT"] in inputs
    assert float(openlane["CLOCK_PERIOD"]) == float(config["clock_period_ns"])
    assert openlane["SETUP_VIOLATION_CORNERS"] == ["*"], (
        "a negative slack fails the flow only in a corner the configuration named, and what "
        "OpenLane installs for a sky130 PDK by default is the typical corner alone"
    )
    assert openlane["HOLD_VIOLATION_CORNERS"] == ["*"]
    assert openlane["ERROR_ON_LINTER_ERRORS"] is True, (
        "the lint error check of the run that reached the end of the flow came back clear, so "
        "there is no measured reason to relax it, and the 14 warnings it reported are gated "
        "apart from this"
    )
    assert openlane["MACROS"] == []
    assert "LL_DEBUG_PORTS" not in json.dumps(openlane["SYNTH_DEFINES"])

    for entry in openlane["VERILOG_FILES"]:
        assert entry.startswith("dir::"), f"unexpected path token in {entry}"
        target = OPENLANE_CONFIG.parent / entry[len("dir::"):]
        assert target.is_file(), f"the OpenLane source list points at a missing file: {entry}"
    listed = {(OPENLANE_CONFIG.parent / entry[len("dir::"):]).resolve() for entry in openlane["VERILOG_FILES"]}
    required = set()
    for key in ("rtl_top", "rtl_core", "rtl_generated"):
        for entry in config_paths(config, key):
            required.add((ASIC_DIR / entry).resolve())
    assert required <= listed, "the OpenLane source list is missing a file the chip config declares"

    sdc = read_text(OPENLANE_SDC)
    clock = re.search(r"create_clock\s+-name\s+(\w+)\s+-period\s+([\d.]+)", sdc)
    assert clock, "the constraint file does not declare a clock"
    assert clock.group(1) == config["clock_port"]
    assert float(clock.group(2)) == float(config["clock_period_ns"])
    assert openlane["SDC"] == ["dir::lingologos.sdc"]
    assert (OPENLANE_CONFIG.parent / "lingologos.sdc").is_file()


# -----------------------------------------------------------------------------
# gate 13
# -----------------------------------------------------------------------------

def _switching_activity() -> Dict[str, float]:
    """Mean toggles per clock transition, with the reset row as the starting state."""
    rows = parse_golden_vectors()
    previous_out = ["00000"] * 4
    previous_gf = "00"
    previous_smt = "0"
    totals = {"out": 0, "gf": 0, "smt": 0}
    for row in rows[1:]:
        totals["out"] += sum(
            bin(int(new, 16) ^ int(old, 16)).count("1")
            for new, old in zip(row[6:10], previous_out)
        )
        totals["gf"] += bin(int(row[10], 16) ^ int(previous_gf, 16)).count("1")
        totals["smt"] += bin(int(row[11], 16) ^ int(previous_smt, 16)).count("1")
        previous_out = row[6:10]
        previous_gf = row[10]
        previous_smt = row[11]
    transitions = len(rows) - 1
    return {
        "out": totals["out"] / transitions,
        "gf": totals["gf"] / transitions,
        "smt": totals["smt"] / transitions,
    }


def test_datasheet_numbers_are_measured_values_not_claims() -> None:
    """Every number the datasheet attributes to a measurement has to be reproducible."""
    datasheet = read_text(DATASHEET)
    config = chip_config()
    public, debug, _ = split_ports(read_text(TOP_RTL))
    rows = parse_golden_vectors()
    values = parse_key_values(SELFTEST_MEM)
    matrix = weight_matrix()
    _, initial, multipliers = gf_tables()

    input_bits = sum(port["width"] for port in public if port["direction"] == "input")
    output_bits = sum(port["width"] for port in public if port["direction"] == "output")
    debug_bits = sum(port["width"] for port in debug)
    worst_row_bound = max(sum(abs(weight) for weight in row) for row in matrix) * 128
    mesh_comments = read_text(MESH_RTL)
    term_bounds: Dict[int, int] = {}
    for match in re.finditer(r"W\[(\d+)\]\[\d+\] = [^\n]*?bound (\d+) \* \|X_\d+\|", mesh_comments):
        row_index = int(match.group(1))
        term_bounds[row_index] = term_bounds.get(row_index, 0) + int(match.group(2))
    tree_bound = 128 * max(term_bounds.values())
    assert len(term_bounds) == len(matrix), "the generated mesh comments do not bound every row"

    required = [
        f"{len(public)} ports, {input_bits} input bits, {output_bits} output bits",
        f"{len(debug)} ports, {debug_bits} bits",
        f"{input_bits + output_bits} bits total",
        "QFN-160",
        "4.0 ns",
        "250 MHz",
        "524287",
        f"{worst_row_bound}",
        f"| {tree_bound} |",
        f"| {524287 / tree_bound:.2f} |",
        "1020",
        "255",
        f"{len(rows)} rows",
        "NOT MEASURED",
    ]
    for token in required:
        assert token in datasheet, f"the datasheet does not state {token}"

    manifest = read_json(MANIFEST)
    for name, digest in manifest["artifact_digests"].items():
        assert f"`{digest}`" in datasheet, f"the datasheet quotes a digest other than {name} {digest}"

    for key in ("exp_x8", "exp_x10", "exp_x11", "exp_x13"):
        assert f"0x{values[key]:08X}" in datasheet, f"the datasheet does not report {key}"

    activity = _switching_activity()
    for name, key, width in (
        ("out_vector_0..3", "out", 80),
        ("gf_next_state", "gf", 8),
        ("smt_results", "smt", 4),
    ):
        assert f"| `{name}` | {width} | {activity[key]:.3f} | {activity[key] / width:.4f} |" in datasheet, (
            f"the activity table in the datasheet disagrees with the vector file for {name}"
        )

    self_test_rows = parse_key_values(SELFTEST_MEM)
    for lane in range(4):
        expected = ref.to_unsigned(ref.mesh_row_sum(matrix, lane, [ref.to_signed(v, 8) for v in
                                                                  (self_test_rows["in_vector_0"], self_test_rows["in_vector_1"],
                                                                   self_test_rows["in_vector_2"], self_test_rows["in_vector_3"])]), 20)
        assert f"`{expected:05X}`" in datasheet, f"the datasheet row sum table is missing lane {lane}"

    assert ref.gf_order(0x02, ref.DEFAULT_FIELD_POLYNOMIAL) == 255
    assert "0x8E" in datasheet, "the datasheet must report the measured inverse of 0x02"

    for path in (DATASHEET, INTEGRATION_GUIDE, VERIFICATION_REPORT, BUILD_INSTRUCTIONS):
        assert path.is_file(), f"section 9 of the specification requires {path.name}"
    report = read_text(VERIFICATION_REPORT)
    for phrase in ("Verilator", "OpenLane", "not run", "NOT MEASURED"):
        assert phrase in report, f"the verification report is silent about {phrase}"
    guide = read_text(INTEGRATION_GUIDE)
    for phrase in ("gf_symbol_in", "LL_DEBUG_PORTS", "0x00", "deviation"):
        assert phrase in guide, f"the integration guide is silent about {phrase}"
    instructions = read_text(BUILD_INSTRUCTIONS)
    for phrase in ("Verilator", "OpenLane", "lingologos_compiler.py", "pytest"):
        assert phrase in instructions, f"the build instructions are silent about {phrase}"


# -----------------------------------------------------------------------------
# gate 14
# -----------------------------------------------------------------------------

TEXT_SUFFIXES = {".py", ".sv", ".v", ".cpp", ".hpp", ".h", ".json", ".md", ".cfg", ".sdc", ".sh", ".mem", ".yml"}
ITALIAN_WORDS = re.compile(
    r"\b(perche|poiche|siccome|quindi|affinche|iniziale|attenzione|valore|valori|"
    r"stato|stati|della|delle|dello|nella|nello|questa|questo|quando|anche|bens)\b",
    re.IGNORECASE,
)
LEGACY_MARKERS = re.compile(r"TODO|FIXME|XXX not implemented")


def test_asic_chip_text_is_ascii_and_english() -> None:
    """The repository language policy applies to the ASIC tree as much as to src."""
    offenders: List[str] = []
    italian: List[str] = []
    stale: List[str] = []
    for path in sorted(ASIC_DIR.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix not in TEXT_SUFFIXES and path.name != "Makefile":
            continue
        raw = path.read_bytes()
        assert b"\x00" not in raw, f"{path} is not a text file"
        try:
            text = raw.decode("ascii")
        except UnicodeDecodeError:
            bad = sorted({byte for byte in raw if byte > 127})
            offenders.append(f"{path.relative_to(REPO_ROOT)}: bytes {bad}")
            continue
        if path.suffix in {".md", ".py", ".sv", ".cpp", ".json", ".cfg", ".sdc", ".sh"} or path.name == "Makefile":
            for match in ITALIAN_WORDS.finditer(text):
                italian.append(f"{path.relative_to(REPO_ROOT)}: {match.group(0)}")
        if path.suffix in {".py", ".sv", ".cpp", ".md"} and LEGACY_MARKERS.search(text):
            stale.append(str(path.relative_to(REPO_ROOT)))
        assert "\r\n" not in raw.decode("latin-1"), f"{path} mixes CRLF line endings"

    assert not offenders, f"non ASCII text in the ASIC tree: {offenders}"
    assert not italian, f"non English prose in the ASIC tree: {italian}"
    assert not stale, f"unfinished markers left in the ASIC tree: {stale}"


def test_no_generated_artifact_is_committed_twice_or_missing() -> None:
    """The generated set is exactly the set the chip configuration declares."""
    config = chip_config()
    declared = {ASIC_DIR / p for p in config_paths(config, "rtl_generated")}
    present = {path for path in (ASIC_DIR / "rtl" / "generated").iterdir() if path.is_file()}
    assert present == declared | {MANIFEST}, f"unexpected or missing files in rtl/generated: {sorted(p.name for p in present)}"
    for name, schema in config["artifact_schemas"].items():
        path = ASIC_DIR / "artifacts" / name
        if not path.is_file():
            path = ASIC_DIR / "rtl" / "generated" / name
        assert path.is_file(), f"the artifact {name} is missing"
        assert read_json(path)["schema"] == schema, f"{name} does not carry the declared schema"


def test_the_reference_model_mux_semantics_are_the_documented_ones() -> None:
    """Pin the argument order of the one asymmetric operator of the clause DSL.

    AND, OR and XOR do not change when two operands are swapped, so a clause model can
    carry a mux bug for a long time without showing it on a symmetric expression. The
    DSL documents the prefix call as MUX(select, when-set, when-clear) and the infix
    form as value MUX select, which is also the order emitted by the generated netlist
    in rtl/generated/smt_evaluator.sv, where the select is the test of the ternary.
    """
    variable_bits = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
    prefix = ref.clause_truth_table("MUX(B, A, C)", variable_bits, 8)
    infix = ref.clause_truth_table("A MUX B", variable_bits, 8)
    clear_is_one = ref.clause_truth_table("MUX(B, A, 1)", variable_bits, 8)
    set_is_one = ref.clause_truth_table("MUX(B, 1, C)", variable_bits, 8)
    for byte_value in range(256):
        bit = {name: (byte_value >> position) & 1 for name, position in variable_bits.items()}
        assert prefix[byte_value] == (bit["A"] if bit["B"] else bit["C"]), (
            f"MUX(B, A, C) is wrong at 0x{byte_value:02X}"
        )
        assert infix[byte_value] == (bit["A"] if bit["B"] else 0), (
            f"A MUX B is wrong at 0x{byte_value:02X}"
        )
        assert clear_is_one[byte_value] == (bit["A"] if bit["B"] else 1), (
            f"MUX(B, A, 1) is wrong at 0x{byte_value:02X}"
        )
        assert set_is_one[byte_value] == (1 if bit["B"] else bit["C"]), (
            f"MUX(B, 1, C) is wrong at 0x{byte_value:02X}"
        )

    netlist = read_text(SMT_RTL)
    ternary = re.findall(r"wire logic (\w+) = in_bits\((\d+)\) \? (\w+) : (\w+);", netlist)
    if not ternary:
        ternary = re.findall(r"wire logic (\w+) = (\w+) \? (\w+) : (\w+);", netlist)
    assert ternary, "the generated clause netlist has no mux to check the order against"
