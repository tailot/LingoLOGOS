"""Filesystem derived hardware model: dies, links, masks, artifacts, fault sites.

The board view never assumes a die count. It reads the board wrapper the export
produced, or it falls back to a topology document, and it takes the number of dies,
their node ids, their routing masks, their channel slots and their link set from what
is actually written in those files. A two die board, a four die board and an N die
board therefore all render from the same code path.

Parsing is lexical rather than a synthesizable Verilog frontend: the workbench needs
the *shape* of the board, which die, which id, which mask, which link, which pin, and
the structural Verilog the export emits is regular enough for anchored patterns over
comment stripped text. Anything that cannot be read is recorded in ``warnings`` and
shown in the view, never silently dropped.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

_WIDTH_LITERAL = re.compile(r"^\s*(?P<width>\d+)?'\s*(?P<base>[bodh])\s*(?P<digits>[0-9a-fA-F_xzXZ?]+)\s*$")
_PLAIN_INT = re.compile(r"^\s*(\d+)\s*$")
_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)
_LINE_COMMENT = re.compile(r"//[^\n]*")


def strip_comments(text: str) -> str:
    """Blank out comments while keeping every line number of the original text."""
    def blank(match: re.Match) -> str:
        return "".join(ch if ch == "\n" else " " for ch in match.group(0))

    return _LINE_COMMENT.sub(blank, _BLOCK_COMMENT.sub(blank, text))


def parse_verilog_literal(text: str) -> Tuple[Optional[int], Optional[int]]:
    """Return ``(width, value)`` for a Verilog literal, or ``(None, None)`` if unreadable."""
    cleaned = (text or "").strip().rstrip(",")
    match = _WIDTH_LITERAL.match(cleaned)
    if match:
        width = int(match.group("width")) if match.group("width") else None
        digits = match.group("digits").replace("_", "")
        base = {"b": 2, "o": 8, "d": 10, "h": 16}[match.group("base")]
        if any(ch in digits.lower() for ch in "xz?"):
            return width, None
        try:
            return width, int(digits, base)
        except ValueError:
            return width, None
    plain = _PLAIN_INT.match(cleaned)
    if plain:
        return None, int(plain.group(1))
    return None, None


def resolve_range(text: Optional[str], parameters: Dict[str, int]) -> Tuple[int, str]:
    """Resolve a declared range such as ``[NODE_ID_WIDTH-1:0]`` into a bit width."""
    if not text:
        return 1, "scalar"
    inner = text.strip().strip("[]").strip()
    if ":" not in inner:
        return 1, "array"
    high, _, low = inner.partition(":")

    def bound(token: str) -> Optional[int]:
        token = token.strip()
        if token.isdigit():
            return int(token)
        if token in parameters:
            return parameters[token]
        if "-" in token:
            left, _, right = token.partition("-")
            left_value, right_value = bound(left), bound(right)
            if left_value is not None and right_value is not None:
                return left_value - right_value
        width, value = parse_verilog_literal(token)
        return value if value is not None else None

    high_value, low_value = bound(high), bound(low)
    if high_value is None or low_value is None:
        return 1, f"unresolved {inner}"
    return abs(high_value - low_value) + 1, f"[{high.strip()}:{low.strip()}]"


# --- structural records ------------------------------------------------------
@dataclass
class PinSpec:
    """One board connector of one die, with the direction of the pin it hangs off."""

    name: str
    direction: str
    width: int
    dtype: str
    die_name: str = ""
    raw_range: str = ""
    slot: Optional[int] = None

    @property
    def label(self) -> str:
        return self.name if self.slot is None else f"{self.name}[{self.slot}]"


@dataclass
class BoardDie:
    """A die on the board: identity from the wrapper plus its routing register."""

    index: int
    name: str
    node_id: int
    total_nodes: int
    routing_mask: int
    mask_width: int
    standalone: bool = False
    pins: List[PinSpec] = field(default_factory=list)
    chip_module: str = ""
    switch_module: str = ""
    chip_instance: str = ""
    switch_instance: str = ""
    parameters: Dict[str, int] = field(default_factory=dict)
    channel_slots: int = 0
    tied_slots: List[int] = field(default_factory=list)
    line: int = 0

    def mask_bits(self) -> List[int]:
        return [(self.routing_mask >> bit) & 1 for bit in range(self.mask_width)]

    def enabled_slots(self) -> List[int]:
        return [bit for bit in range(self.mask_width) if (self.routing_mask >> bit) & 1]

    def is_enabled(self, target_index: int) -> bool:
        return 0 <= target_index < self.mask_width and bool((self.routing_mask >> target_index) & 1)

    def pins_grouped(self) -> Dict[str, List[PinSpec]]:
        """Group the connector stack by function so the schematic can draw port bundles."""
        groups: Dict[str, List[PinSpec]] = {}
        for pin in self.pins:
            key = re.sub(r"_\d+$", "", pin.name)
            key = re.sub(r"\[(\d+)\]$", "", key)
            groups.setdefault(key, []).append(pin)
        return groups


@dataclass
class BoardLink:
    """One full duplex channel between two dies, as the board wrapper wires it."""

    index: int
    source_index: int
    target_index: int
    source_name: str
    target_name: str
    signal_count: int = 0
    slot_source: Optional[int] = None
    slot_target: Optional[int] = None
    line: int = 0

    @property
    def label(self) -> str:
        return f"link {self.index}: {self.source_name} -> {self.target_name}"

    def signals(self) -> List[str]:
        """Names of the three data wires plus one handshake wire of such a channel."""
        return ["valid", "dest", "data"] + (["ready"] if self.signal_count >= 4 else [])


@dataclass
class BoardDescriptor:
    """The whole board as discovered: parameters, dies, links, board level connectors."""

    name: str = "board"
    source: Optional[Path] = None
    parameters: Dict[str, int] = field(default_factory=dict)
    dies: List[BoardDie] = field(default_factory=list)
    links: List[BoardLink] = field(default_factory=list)
    top_ports: List[PinSpec] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    provenance: str = "verilog-board"

    @property
    def die_count(self) -> int:
        return len(self.dies)

    @property
    def max_nodes(self) -> int:
        widths = [die.mask_width for die in self.dies] or [1]
        return int(self.parameters.get("MAX_NODES", max(widths)))

    def die_by_name(self, name: str) -> Optional[BoardDie]:
        for die in self.dies:
            if die.name == name:
                return die
        return None

    def adjacency_matrix(self) -> List[List[int]]:
        """Routing masks of every die as a square matrix, N by N for N dies."""
        size = self.die_count
        matrix = [[0] * size for _ in range(size)]
        for row, die in enumerate(self.dies):
            for column, bit in enumerate(die.mask_bits()[:size]):
                matrix[row][column] = bit
        return matrix

    def channel_pairs(self) -> List[Tuple[BoardLink, BoardLink]]:
        """Group the two directional halves of every physical channel together."""
        forward = {(l.source_index, l.target_index): l for l in self.links}
        pairs: List[Tuple[BoardLink, BoardLink]] = []
        seen: set = set()
        for link in self.links:
            key = tuple(sorted((link.source_index, link.target_index)))
            if key in seen:
                continue
            seen.add(key)
            reverse = forward.get((link.target_index, link.source_index))
            pairs.append((link, reverse) if reverse is not None else (link, link))
        return pairs

    def summary(self) -> Dict[str, Any]:
        return {
            "board": self.name,
            "dies": self.die_count,
            "links": len(self.links),
            "parameters": dict(self.parameters),
            "max_nodes": self.max_nodes,
            "pins": len(self.top_ports),
            "warnings": list(self.warnings),
            "provenance": self.provenance,
        }


# --- instance extraction ------------------------------------------------------
_INSTANCE = re.compile(
    r"(?P<module>[A-Za-z_]\w*)\s+"
    r"(?:#\s*\((?:[^()]|\([^()]*\))*\)\s*)?"
    r"(?P<inst>[A-Za-z_]\w*)\s*"
    r"\((?P<body>(?:[^()]|\([^()]*\))*)\)",
    re.DOTALL,
)
_PORT_CONNECT = re.compile(r"\.(?P<field>\w+)\s*\((?P<expr>[^,()]*|\([^()]*\))\)")
_KEYWORD_STATEMENTS = {
    # Reserved words that can sit where an instance pattern expects a module name, so a
    # task declaration or a typed function header is never read as a hardware instance.
    "module", "endmodule", "function", "endfunction", "task", "endtask", "automatic",
    "begin", "end", "if", "else", "for", "foreach", "forever", "assign", "repeat",
    "always", "always_ff", "always_comb", "always_latch", "initial", "final", "generate",
    "case", "return", "wait", "while", "do", "typedef", "struct", "enum", "union",
    "interface", "class", "covergroup", "cover", "assert", "assume", "property",
    "sequence", "program", "package", "import", "export", "extern", "let", "rand",
    "randc", "const", "static", "local", "protected", "virtual", "this", "new",
    "int", "integer", "logic", "bit", "byte", "reg", "wire", "real", "realtime",
    "string", "chandle", "void", "shortint", "int unsigned", "longint", "time",
    "input", "output", "inout", "localparam", "parameter", "defparam", "posedge",
    "negedge", "edge", "var", "genvar", "event", "type",
}


def find_instances(text: str) -> List[Tuple[str, str, Dict[str, str], int]]:
    """Return ``(module, instance, port connections, line)`` for every instance block.

    Comment stripped text is expected, and the pattern tolerates a parameter header
    spread over several lines, which is how the export writes a switch instance.
    """
    results: List[Tuple[str, str, Dict[str, str], int]] = []
    for match in _INSTANCE.finditer(text):
        module = match.group("module")
        instance = match.group("inst")
        if module.lower() in _KEYWORD_STATEMENTS or instance.lower() in _KEYWORD_STATEMENTS:
            continue
        if module == instance:
            continue
        connections: Dict[str, str] = {}
        for connect in _PORT_CONNECT.finditer(match.group("body")):
            connections[connect.group("field")] = connect.group("expr").strip()
        line = text.count("\n", 0, match.start()) + 1
        results.append((module, instance, connections, line))
    return results


def read_parameters(text: str) -> Dict[str, int]:
    """Collect every ``parameter`` and ``localparam`` with a readable integer value."""
    parameters: Dict[str, int] = {}
    pattern = re.compile(
        r"\b(?:parameter|localparam)\s+"
        r"(?:(?:integer|int|logic|bit|byte|wire|real|reg)\s*)?"
        r"(?:\[[^\]]*\]\s*)?"
        r"(?P<name>[A-Za-z_]\w*)\s*"
        r"(?:\[[^\]]*\]\s*)?"
        r"=\s*(?P<value>[^;,)\n]+)"
    )
    for match in pattern.finditer(text):
        _, value = parse_verilog_literal(match.group("value"))
        if value is not None:
            parameters[match.group("name")] = value
    return parameters


_LINK_COMMENT = re.compile(
    r"//\s*link\s+(?P<index>\d+)\s*:\s*node\s+(?P<src>\d+)\s*\((?P<srcname>[^)]*)\)\s*"
    r"to\s+node\s+(?P<dst>\d+)\s*\((?P<dstname>[^)]*)\)"
)
_PORT = re.compile(
    r"^\s*(?P<direction>input|output|inout)\s+(?P<kind>wire|logic|reg|var|)\s*"
    r"(?P<signed>signed)?\s*(?P<range>(?:\[[^\]]*\]\s*)+)?(?P<name>[A-Za-z_]\w*)"
    r"(?P<slot>(?:\s*\[\s*\d+\s*\])+)?\s*,?\s*$"
)
_ARRAY_PORTS = re.compile(r"^\s*(?P<direction>input|output|inout)\s+.*?\b(?P<name>[A-Za-z_]\w*)\s*\[\s*(?P<slot>\d+)\s*\]")
_TIE_OFF = re.compile(
    r"assign\s+(?P<die>[A-Za-z_]\w*)_link_(?P<which>rx|tx)_(?P<signal>\w+)\s*\[\s*(?P<slot>\d+)\s*\]\s*=\s*(?P<value>1'b0|\d+'d0|\d+'d0);"
)


def _die_name_at(board: BoardDescriptor, index: int) -> str:
    for die in board.dies:
        if die.index == index or die.node_id == index:
            return die.name
    return f"node_{index}"


def parse_board_verilog(path: Path) -> BoardDescriptor:
    """Read the exported board wrapper and return the board it describes.

    Die count, node ids, routing masks, channel slots, link channels and pin stacks
    are all results of the read: the function contains no die count and no agent name.
    """
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    clean = strip_comments(text)
    board = BoardDescriptor(name=Path(path).stem, source=Path(path), provenance="verilog-board")

    header = re.search(r"^\s*module\s+(?P<name>\w+)", clean, re.MULTILINE)
    if header:
        board.name = header.group("name")
    board.parameters = read_parameters(clean)

    instances = find_instances(clean)
    switches = [(m, i, c, ln) for m, i, c, ln in instances if i.startswith("u_switch_")]
    chips = [(m, i, c, ln) for m, i, c, ln in instances if i.startswith("u_chip_")]
    names: List[str] = []
    for _, instance, _, _ in chips or switches:
        name = instance.split("_", 2)[-1]
        if name not in names:
            names.append(name)
    if not names:
        for _, instance, _, _ in switches + chips:
            names.append(instance.split("_", 2)[-1])

    mask_width = int(board.parameters.get("MAX_NODES", 0))
    for position, name in enumerate(names):
        board.dies.append(
            BoardDie(
                index=position,
                name=name,
                node_id=position,
                total_nodes=len(names),
                routing_mask=0,
                mask_width=mask_width or 1,
            )
        )
    by_name = {die.name: die for die in board.dies}

    for module, instance, connections, line in switches:
        die_name = instance[len("u_switch_"):]
        die = by_name.get(die_name)
        if die is None:
            board.warnings.append(f"switch instance {instance} at line {line} has no matching die")
            continue
        die.switch_module = module
        die.switch_instance = instance
        die.line = line
        node_id_value = parse_verilog_literal(connections.get("node_id", ""))[1]
        if node_id_value is not None:
            die.node_id = node_id_value
        total_value = parse_verilog_literal(connections.get("total_nodes", ""))[1]
        if total_value is not None:
            die.total_nodes = total_value
        mask_literal = connections.get("routing_mask", "")
        width, mask_value = parse_verilog_literal(mask_literal)
        if mask_value is None:
            board.warnings.append(f"routing_mask of {die_name} is not a readable literal: {mask_literal}")
        else:
            die.routing_mask = mask_value
            if width:
                die.mask_width = max(die.mask_width, width)
        die.standalone = connections.get("standalone", "").replace(" ", "") in ("1'b1", "1", "true")
        if die.node_id != die.index:
            board.warnings.append(
                f"{die_name}: node_id {die.node_id} differs from position {die.index} in the file"
            )

    for module, instance, connections, line in chips:
        die = by_name.get(instance[len("u_chip_"):])
        if die is not None:
            die.chip_module = module
            die.chip_instance = instance
            if die.line == 0:
                die.line = line

    for line_text in text.splitlines():
        match = _PORT.match(line_text)
        if match:
            direction = match.group("direction")
            ranges = re.findall(r"\[[^\]]*\]", match.group("range") or "")
            raw_range = ranges[0] if ranges else ""
            width, dtype = resolve_range(raw_range, board.parameters)
            signed = (match.group("signed") or "").strip()
            port_name = match.group("name")
            pin = PinSpec(
                name=port_name,
                direction=direction,
                width=width,
                dtype=f"signed {dtype}" if signed else dtype,
                raw_range=raw_range,
            )
            board.top_ports.append(pin)
            owner = next((d for d in board.dies if port_name.startswith(f"{d.name}_")), None)
            if owner is not None:
                pin.die_name = owner.name
                pin.name = port_name[len(owner.name) + 1:]
                owner.pins.append(pin)
            continue
        array_match = _ARRAY_PORTS.match(line_text)
        if array_match and board.dies:
            port_name = array_match.group("name")
            owner = next((d for d in board.dies if port_name.startswith(f"{d.name}_")), None)
            if owner is None:
                continue
            direction = array_match.group("direction")
            slot = int(array_match.group("slot"))
            short = port_name[len(owner.name) + 1:]
            pin = PinSpec(
                name=short,
                direction=direction,
                width=owner_channel_width(short, board.parameters),
                dtype="channel",
                die_name=owner.name,
                slot=slot,
            )
            owner.channel_slots = max(owner.channel_slots, slot + 1)
            if slot < owner.mask_width and not owner.is_enabled(slot):
                owner.tied_slots.append(slot)
            board.top_ports.append(pin)
            owner.pins.append(pin)

    for match in _TIE_OFF.finditer(clean):
        die = by_name.get(match.group("die"))
        if die is not None:
            slot = int(match.group("slot"))
            if slot not in die.tied_slots:
                die.tied_slots.append(slot)

    if not board.dies:
        board.warnings.append("no u_chip_* or u_switch_* instance found in the board wrapper")

    link_matches = list(_LINK_COMMENT.finditer(text))
    for position, match in enumerate(link_matches):
        end = link_matches[position + 1].start() if position + 1 < len(link_matches) else len(text)
        segment = text[match.end() : end]
        signals = len([ln for ln in segment.splitlines() if ln.strip().startswith("assign")])
        src_index = int(match.group("src"))
        dst_index = int(match.group("dst"))
        src_name = (match.group("srcname") or "").strip() or _die_name_at(board, src_index)
        dst_name = (match.group("dstname") or "").strip() or _die_name_at(board, dst_index)
        board.links.append(
            BoardLink(
                index=int(match.group("index")),
                source_index=src_index,
                target_index=dst_index,
                source_name=src_name,
                target_name=dst_name,
                signal_count=signals,
                slot_source=dst_index,
                slot_target=src_index,
                line=text.count("\n", 0, match.start()) + 1,
            )
        )

    if not board.links:
        for die in board.dies:
            for bit in die.enabled_slots():
                if bit == die.node_id:
                    continue
                board.links.append(
                    BoardLink(
                        index=len(board.links),
                        source_index=die.node_id,
                        target_index=bit,
                        source_name=die.name,
                        target_name=_die_name_at(board, bit),
                        signal_count=0,
                        slot_source=bit,
                        slot_target=die.node_id,
                    )
                )
        if board.links:
            board.warnings.append("links derived from routing masks because the wrapper has no link comment")
    return board


def owner_channel_width(signal: str, parameters: Dict[str, int]) -> int:
    """Bit width of one channel wire, from the naming of the channel signal."""
    if "data" in signal:
        return int(parameters.get("DATA_WIDTH", 64))
    if "dest" in signal:
        return int(parameters.get("NODE_ID_WIDTH", 4))
    return 1


def board_from_topology_json(path: Path) -> BoardDescriptor:
    """Synthesize the same board shape straight from a topology document.

    Used when the export has not run yet: die count, ids and masks are still real
    values, they simply come from the topology instead of the board wrapper.
    """
    with open(path, "r", encoding="utf-8") as handle:
        document = json.load(handle)
    board = BoardDescriptor(name=Path(path).stem, source=Path(path), provenance="topology-json")
    names: List[str] = []
    for instance in document.get("instances", []) or []:
        if isinstance(instance, dict) and instance.get("id"):
            names.append(str(instance["id"]))
    for node in document.get("nodes", []) or []:
        if isinstance(node, dict) and node.get("node_id"):
            names.append(str(node["node_id"]))
    mask_width = max(len(names), 1)
    board.parameters = {"MAX_NODES": mask_width, "NODE_ID_WIDTH": max(1, mask_width.bit_length())}
    for position, name in enumerate(names):
        board.dies.append(
            BoardDie(
                index=position,
                name=name,
                node_id=position,
                total_nodes=len(names),
                routing_mask=0,
                mask_width=mask_width,
                channel_slots=mask_width,
            )
        )
    by_name = {die.name: die for die in board.dies}

    channels: List[Tuple[str, str]] = []
    for rule in document.get("routingMatrix", []) or []:
        if not isinstance(rule, dict):
            continue
        source = str(rule.get("from", ""))
        for target in rule.get("to", []) or []:
            channels.append((source, str(target)))
    for rule in document.get("routing_rules", []) or []:
        if isinstance(rule, dict):
            channels.append((str(rule.get("source", "")), str(rule.get("target", ""))))
    for node in document.get("nodes", []) or []:
        if isinstance(node, dict):
            source = str(node.get("node_id", ""))
            for target in node.get("outbound_channels", []) or []:
                channels.append((source, str(target)))

    for source, target in channels:
        src_die, dst_die = by_name.get(source), by_name.get(target)
        if src_die is None or dst_die is None:
            board.warnings.append(f"channel {source} -> {target} has no matching die")
            continue
        src_die.routing_mask |= 1 << dst_die.index
        board.links.append(
            BoardLink(
                index=len(board.links),
                source_index=src_die.index,
                target_index=dst_die.index,
                source_name=src_die.name,
                target_name=dst_die.name,
                signal_count=0,
                slot_source=dst_die.index,
                slot_target=src_die.index,
            )
        )
    return board


def load_board(path: Path) -> BoardDescriptor:
    """Dispatch on suffix: a ``.v`` / ``.sv`` wrapper or a ``.json`` topology document."""
    if Path(path).suffix.lower() == ".json":
        return board_from_topology_json(Path(path))
    return parse_board_verilog(Path(path))


# --- fault and miswiring plan from a testbench --------------------------------
@dataclass
class FaultSite:
    """One place where a testbench drives a fault signal, with its line number."""

    signal: str
    line: int
    value: str
    note: str = ""


@dataclass
class FaultPlan:
    """What a testbench says about faults: miswire sites, masks, verdicts, tasks."""

    source: Optional[Path] = None
    parameters: Dict[str, int] = field(default_factory=dict)
    masks: Dict[str, int] = field(default_factory=dict)
    mask_widths: Dict[str, int] = field(default_factory=dict)
    fault_signals: Dict[str, List[FaultSite]] = field(default_factory=dict)
    verdicts: List[Tuple[int, str]] = field(default_factory=list)
    tasks: List[Tuple[str, int]] = field(default_factory=list)
    die_instances: List[Tuple[str, int]] = field(default_factory=list)

    @property
    def die_count(self) -> int:
        declared = self.parameters.get("DIES") or self.parameters.get("NODES")
        if declared:
            return int(declared)
        return len(self.die_instances) or len(self.masks)

    def summary(self) -> Dict[str, Any]:
        return {
            "parameters": dict(self.parameters),
            "masks": dict(self.masks),
            "fault_sites": {name: len(sites) for name, sites in self.fault_signals.items()},
            "verdicts": len(self.verdicts),
            "tasks": len(self.tasks),
            "dies": self.die_count,
        }


_FAULT_ASSIGN = re.compile(r"^\s*(?P<signal>\w*(?:miswire|fault|inject|error)\w*)\s*=\s*(?P<value>[^;]+);")
_VERDICT_LINE = re.compile(r"(FINAL VERDICT\s*:\s*(?:PASS|FAIL))")
_TASK = re.compile(r"^\s*task\s+(?:automatic\s+)?(?P<name>\w+)", re.MULTILINE)
_MASK_DECL = re.compile(r"(?P<name>\w*(?:MASK|MESH)\w*)\s*=\s*(?P<value>\d+'b[01_xz]+)")
_TB_INSTANCE = re.compile(r"(?P<module>lingologos\w*)\s+(?P<inst>die\d+|u_switch_\w+|u_chip_\w+)\s*\(")


def parse_fault_plan(path: Path) -> FaultPlan:
    """Read fault behaviour out of a SystemVerilog testbench."""
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    plan = FaultPlan(source=Path(path))
    for index, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("//"):
            continue
        mask = _MASK_DECL.search(stripped)
        if mask:
            width, value = parse_verilog_literal(mask.group("value"))
            if value is not None:
                plan.masks[mask.group("name")] = value
                plan.mask_widths[mask.group("name")] = width or max(1, value.bit_length())
        fault = _FAULT_ASSIGN.match(stripped)
        if fault:
            plan.fault_signals.setdefault(fault.group("signal"), []).append(
                FaultSite(signal=fault.group("signal"), line=index, value=fault.group("value").strip())
            )
        verdict = _VERDICT_LINE.search(stripped)
        if verdict:
            plan.verdicts.append((index, verdict.group(1)))
    plan.parameters = read_parameters(strip_comments(text))
    for match in _TASK.finditer(text):
        plan.tasks.append((match.group("name"), text.count("\n", 0, match.start()) + 1))
    for match in _TB_INSTANCE.finditer(strip_comments(text)):
        plan.die_instances.append((match.group("inst"), text.count("\n", 0, match.start()) + 1))
    return plan


# --- artifact index ----------------------------------------------------------
@dataclass
class ArtifactIndex:
    """Groups the workspace scan by role, with the queries the views ask."""

    roles: Dict[str, Sequence[Any]] = field(default_factory=dict)

    @classmethod
    def build(cls, scan: Dict[str, Sequence[Any]]) -> "ArtifactIndex":
        return cls(roles={role: list(items) for role, items in scan.items()})

    def paths_of(self, role: str) -> List[Path]:
        return [item.path for item in self.roles.get(role, [])]

    def first(self, role: str) -> Optional[Path]:
        items = self.roles.get(role, [])
        return items[0].path if items else None

    def newest(self, role: str, hint: str = "") -> Optional[Path]:
        """Return the path of the newest artifact of a role, like ``first`` does."""
        items = [item for item in self.roles.get(role, []) if hint.lower() in item.path.name.lower()]
        if not items:
            return None
        return max(items, key=lambda item: item.path.stat().st_mtime).path

    def counts(self) -> Dict[str, int]:
        return {role: len(items) for role, items in self.roles.items()}

    def missing_roles(self, required: Iterable[str]) -> List[str]:
        return [role for role in required if not self.roles.get(role)]


__all__ = [
    "ArtifactIndex",
    "BoardDescriptor",
    "BoardDie",
    "BoardLink",
    "FaultPlan",
    "FaultSite",
    "PinSpec",
    "board_from_topology_json",
    "find_instances",
    "load_board",
    "owner_channel_width",
    "parse_board_verilog",
    "parse_fault_plan",
    "parse_verilog_literal",
    "read_parameters",
    "resolve_range",
    "strip_comments",
]
