"""Static analysis of a SystemVerilog source: symbols, masks, tasks, verdicts.

The analyzer is lexical and tolerant by design. It is not a synthesizable frontend, it
extracts the handles a testbench author cares about when reading one: ``localparam``
values, mesh masks, module and instance names, task signatures, fault signals, and the
verdict lines a simulation prints. Unreadable constructs are counted, never dropped.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .hardware import find_instances, parse_verilog_literal, read_parameters, strip_comments

_KEYWORDS = (
    "module", "endmodule", "input", "output", "inout", "wire", "logic", "reg", "var",
    "localparam", "parameter", "typedef", "enum", "struct", "packed", "signed",
    "begin", "end", "if", "else", "for", "foreach", "while", "do", "case", "endcase",
    "function", "endfunction", "task", "endtask", "automatic", "return", "posedge",
    "negedge", "always", "always_ff", "always_comb", "initial", "final", "assign",
    "integer", "bit", "byte", "int", "string", "real", "time", "void", "rand",
    "class", "endclass", "interface", "endinterface", "generate", "endgenerate",
    "repeat", "wait", "fork", "join", "assert", "assume", "cover", "constraint",
)

_COMMENT = re.compile(r"/\*.*?\*/|//[^\n]*", re.DOTALL)
_STRING = re.compile(r'"(?:\\.|[^"\\])*"')
_LOCALPARAM = re.compile(
    r"(?P<kind>localparam|parameter)\s+"
    r"(?:(?:int|integer|logic|bit|byte|wire|real|reg)\s*)?"
    r"(?:\[[^\]]*\]\s*)?"
    r"(?P<name>[A-Za-z_]\w*)\s*"
    r"(?:\[[^\]]*\]\s*)?"
    r"=\s*(?P<value>[^;]+);"
)
_MASK_NAME = re.compile(r"\b[A-Za-z_]\w*(?:MASK|MESH)\w*\b")
_TASK = re.compile(r"^\s*task\s+(?P<automatic>automatic\s+)?(?P<name>\w+)\s*(?:\((?P<args>[^)]*)\))?", re.MULTILINE)
_FUNCTION = re.compile(
    r"^\s*function\s+(?P<ret>[\w\s\[\]$]+?)\s+(?P<name>\w+)\s*(?:\((?P<args>[^)]*)\))?", re.MULTILINE
)
_MODULE = re.compile(r"^\s*module\s+(?P<name>\w+)", re.MULTILINE)
_VERDICT = re.compile(r"(FINAL VERDICT|Verdict)\s*:\s*(PASS|FAIL)")
_FAULT = re.compile(r"\b((?:[A-Za-z_]\w*)?(?:miswire|fault|inject|error)[A-Za-z_]\w*)\b")
_DIRECTIVE = re.compile(r"\$\w+|`\w+")


@dataclass
class SvSymbol:
    """One extracted symbol with the line it sits on."""

    kind: str
    name: str
    value: str = ""
    line: int = 0
    width: Optional[int] = None
    note: str = ""

    def as_row(self) -> List[str]:
        rendered = self.value if len(self.value) < 90 else self.value[:89] + "..."
        return [self.kind, self.name, rendered, str(self.line)]


@dataclass
class SvAnalysis:
    """All extraction results of one SystemVerilog document."""

    source: Optional[Path] = None
    text: str = ""
    symbols: List[SvSymbol] = field(default_factory=list)
    modules: List[SvSymbol] = field(default_factory=list)
    instances: List[SvSymbol] = field(default_factory=list)
    tasks: List[SvSymbol] = field(default_factory=list)
    functions: List[SvSymbol] = field(default_factory=list)
    masks: List[SvSymbol] = field(default_factory=list)
    faults: List[SvSymbol] = field(default_factory=list)
    verdicts: List[Tuple[int, str]] = field(default_factory=list)

    @classmethod
    def parse(cls, text: str, source: Optional[Path] = None) -> "SvAnalysis":
        document = cls(source=source, text=text)
        document._scan_symbols()
        document._scan_structure()
        document._scan_verdicts()
        return document

    def _line_of(self, offset: int) -> int:
        return self.text.count("\n", 0, offset) + 1

    # --- scans -----------------------------------------------------------------
    def _scan_symbols(self) -> None:
        for match in _LOCALPARAM.finditer(self.text):
            raw_value = match.group("value").strip()
            width, value = parse_verilog_literal(raw_value)
            name = match.group("name")
            kind = "mask" if ("MASK" in name.upper() or "MESH" in name.upper()) else match.group("kind")
            symbol = SvSymbol(
                kind=kind,
                name=name,
                value=raw_value,
                line=self._line_of(match.start()),
                width=width,
                note="" if value is None else str(value),
            )
            self.symbols.append(symbol)
            if kind == "mask":
                self.masks.append(symbol)
        for match in _MASK_NAME.finditer(strip_comments(self.text)):
            name = match.group(0)
            if any(existing.name == name for existing in self.masks):
                continue
            if re.search(rf"\b{re.escape(name)}\s*=", self.text):
                continue
            self.masks.append(SvSymbol(kind="mask-ref", name=name, line=self._line_of(match.start())))

    def _scan_structure(self) -> None:
        clean = strip_comments(self.text)
        for match in _MODULE.finditer(self.text):
            self.modules.append(SvSymbol("module", match.group("name"), line=self._line_of(match.start())))
        for module, instance, connections, line in find_instances(clean):
            self.instances.append(
                SvSymbol(
                    kind="instance",
                    name=instance,
                    value=module,
                    line=line,
                    note=", ".join(f"{key}={value}" for key, value in sorted(connections.items())[:4]),
                )
            )
        for match in _TASK.finditer(self.text):
            self.tasks.append(
                SvSymbol(
                    kind="task",
                    name=match.group("name"),
                    value=(match.group("args") or "").strip(),
                    line=self._line_of(match.start()),
                    note="automatic" if match.group("automatic") else "",
                )
            )
        for match in _FUNCTION.finditer(self.text):
            self.functions.append(
                SvSymbol(
                    kind="function",
                    name=match.group("name"),
                    value=(match.group("args") or "").strip(),
                    line=self._line_of(match.start()),
                    note=(match.group("ret") or "").strip(),
                )
            )
        for name in sorted(set(_FAULT.findall(clean))):
            first = re.search(rf"\b{re.escape(name)}\b", clean)
            if first is not None:
                self.faults.append(SvSymbol("fault", name, line=self._line_of(first.start())))

    def _scan_verdicts(self) -> None:
        for index, line in enumerate(self.text.splitlines(), start=1):
            match = _VERDICT.search(line)
            if match:
                self.verdicts.append((index, match.group(0)))

    # --- queries ---------------------------------------------------------------
    def symbol_named(self, name: str) -> Optional[SvSymbol]:
        for symbol in self.symbols:
            if symbol.name == name:
                return symbol
        return None

    def mesh_masks(self) -> List[SvSymbol]:
        return [s for s in self.masks if "MESH" in s.name.upper()]

    def parameter_map(self) -> Dict[str, int]:
        return read_parameters(strip_comments(self.text))

    def summary(self) -> Dict[str, Any]:
        return {
            "path": str(self.source) if self.source else "<memory>",
            "lines": self.text.count("\n") + 1,
            "modules": len(self.modules),
            "instances": len(self.instances),
            "tasks": len(self.tasks),
            "functions": len(self.functions),
            "localparams": len(self.symbols),
            "masks": len(self.masks),
            "fault_signals": len(self.faults),
            "verdict_lines": len(self.verdicts),
        }

    def keyword_pattern(self) -> re.Pattern:
        """Regex matching every SystemVerilog keyword, for the highlighter."""
        return re.compile(r"\b(?:" + "|".join(re.escape(k) for k in _KEYWORDS) + r")\b")

    def local_symbol_pattern(self) -> re.Pattern:
        """Regex matching a whole localparam statement, so it can be selected as one unit."""
        return _LOCALPARAM

    def local_range_pattern(self) -> re.Pattern:
        """Anchored pattern used to find a localparam statement on one single line."""
        return re.compile(
            r"^\s*(?:localparam|parameter)\s+[^=]+=\s*[^;]+;?\s*$"
        )

    def rule_table(self) -> List[Tuple[re.Pattern, str]]:
        """Highlighting rules as ``(pattern, style key)`` pairs, in application order.

        Style keys are resolved to colours by the view through the theme palette, so
        this analyzer holds no presentation value at all.
        """
        return [
            (_COMMENT, "comment"),
            (_STRING, "string"),
            (_LOCALPARAM, "param"),
            (re.compile(r"\b[A-Z][A-Z0-9_]{2,}\b"), "constant"),
            (_DIRECTIVE, "system"),
            (_MASK_NAME, "mask"),
            (re.compile(r"\b(?:FINAL VERDICT|Verdict)\b"), "verdict"),
            (re.compile(r"\b(?:module|task|function|endmodule|endtask|endfunction)\b"), "decl"),
            (self.keyword_pattern(), "keyword"),
        ]


def load_sv(path: Path) -> SvAnalysis:
    """Read a file from disk and analyze it."""
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    return SvAnalysis.parse(text, Path(path))


__all__ = ["SvAnalysis", "SvSymbol", "load_sv"]
