#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# Export the fabrication views and the board views of one LingoLOGOS OpenLane run.
#
# The contract this script implements is the one written in
# doc/TECHNICAL_DOCUMENTATION.md under "Fabrication and board artifact export":
# one OpenLane run directory in, one directory tree out, and one manifest that
# records, per artifact, where the file came from, how big it is and what its
# purpose is. Nothing here recomputes a number: a report a tool wrote is copied
# byte for byte, and the two generated files say so in their own headers.
#
# The tree it writes is the one the export specification asks for:
#
#   export_artifacts/
#     factory/gds/                     GDSII of the run, one file per streamout tool
#     factory/lef/                     abstract LEF of the design
#     factory/def/                     DEF of the routed design
#     factory/sdc/                     constraint set the run was signed off against
#     factory/reports/                 DRC, LVS, antenna, signoff and timing summaries
#     visualization_board/netlists/    gate level Verilog and the structured .nl file
#     visualization_board/system_board/board_complete.v
#     manifest.json                    what was exported, from where, with which digest
#
# Two places where the export specification and the tools disagree, and what this
# script does about it, is written next to the entries concerned below: the lef and
# def pair (the traceability table of the specification names one factory/lef_def
# directory while its own tree names two), and the netlist (the specification names
# one gate level file while OpenLane writes the synthesized one and the routed one).
#
# It has run against real runs. What was measured, on this repository, is recorded
# in asic_chip/doc/build_instructions.md and in section 2.5 of
# doc/TECHNICAL_DOCUMENTATION.md.
#
# Run it from anywhere; the paths below are resolved from this file, not from the
# working directory. From inside WSL, which is where the runs of this design are:
#
#   python3 asic_chip/scripts/export_artifacts.py
#   python3 asic_chip/scripts/export_artifacts.py --desktop
#   python3 asic_chip/scripts/export_artifacts.py \
#     --runs-dir /home/tailot/LingoLOGOS/asic_chip/openlane/runs
# -----------------------------------------------------------------------------

import argparse
import datetime
import fnmatch
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from typing import Dict, List, Optional, Sequence, Tuple

SCRIPT_PATH = os.path.realpath(__file__)
ASIC_DIR = os.path.dirname(os.path.dirname(SCRIPT_PATH))
REPO_ROOT = os.path.dirname(ASIC_DIR)

DEFAULT_RUNS_DIR = os.path.join(ASIC_DIR, "openlane", "runs")
DEFAULT_OUT_DIR = os.path.join(REPO_ROOT, "export_artifacts")
DEFAULT_TOPOLOGY = os.path.join(REPO_ROOT, "config", "topology.json")
CHIP_CONFIG = os.path.join(ASIC_DIR, "config.json")

RTL_TOP = os.path.join(ASIC_DIR, "rtl", "top", "lingologos_top.sv")
RTL_ROUTER = os.path.join(ASIC_DIR, "rtl", "core", "lingologos_topology_router.sv")
RTL_CORE = [os.path.join(ASIC_DIR, "rtl", "core", "rv32e_core.sv"),
            os.path.join(ASIC_DIR, "rtl", "core", "rv32e_regfile.sv")]
RTL_GENERATED = [os.path.join(ASIC_DIR, "rtl", "generated", "weight_rom_mesh.sv"),
                 os.path.join(ASIC_DIR, "rtl", "generated", "gf_automaton_1cycle.sv"),
                 os.path.join(ASIC_DIR, "rtl", "generated", "smt_evaluator.sv")]

# Keywords that introduce a statement which is not a module instance. The Verilog
# reader below is a reader of a subset, and this list is how it tells the two apart
# without a grammar: a statement that begins with one of these is skipped.
NON_INSTANCE_KEYWORDS = {
    "assign", "always", "always_ff", "always_comb", "always_latch", "initial",
    "final", "if", "else", "case", "casex", "casez", "for", "while", "repeat",
    "forever", "begin", "end", "wire", "logic", "reg", "bit", "byte", "int",
    "integer", "real", "localparam", "parameter", "typedef", "struct", "union",
    "enum", "packed", "function", "endfunction", "task", "endtask", "generate",
    "endgenerate", "genvar", "input", "output", "inout", "module", "endmodule",
    "posedge", "negedge", "or", "and", "not", "xor", "nand", "nor", "return",
    "void", "string", "modport", "interface", "clocking", "property", "assert",
}

# The export matrix: one entry per file the export writes, in the order it writes
# them. dest is relative to the export root, kind is copy (a file the run wrote),
# bundle (a report composed here from data the run recorded) or gen (a file
# generated here), candidates are patterns inside the run directory written
# component by component, with {design} standing for the design name and ** for
# any depth, and optional marks an artifact a flow variant may legitimately not
# write, which is reported but does not decide the exit status.
ARTIFACT_MATRIX: List[Dict[str, object]] = [
    {
        "dest": "factory/gds/lingologos_top.gds",
        "kind": "copy",
        "purpose": "tapeout layout, the GDSII the streamout step wrote",
        "candidates": [
            "final/gds/{design}.gds",
            "results/signoff/{design}.gds",
            "**/[0-9]*-klayout-streamout/{design}.klayout.gds",
            "**/[0-9]*-magic-streamout/{design}.gds",
        ],
    },
    {
        "dest": "factory/gds/lingologos_top.magic.gds",
        "kind": "copy",
        "optional": True,
        "purpose": "GDSII as written by Magic, the view the Magic DRC run checked",
        "candidates": [
            "final/mag_gds/{design}.magic.gds",
            "**/[0-9]*-magic-streamout/*magic*.gds",
        ],
    },
    {
        "dest": "factory/gds/lingologos_top.klayout.gds",
        "kind": "copy",
        "optional": True,
        "purpose": "GDSII as written by KLayout, the view the XOR check compared",
        "candidates": [
            "final/klayout_gds/{design}.klayout.gds",
            "**/[0-9]*-klayout-streamout/*.gds",
        ],
    },
    {
        "dest": "factory/lef/lingologos_top.lef",
        "kind": "copy",
        "purpose": "abstract LEF of the design, for floorplanning and SoC integration",
        "note": "the sky130 technology LEF and the standard cell LEF are not copied: "
                "they belong to the PDK, not to this design, and the foundry ships them",
        "candidates": [
            "final/lef/{design}.lef",
            "results/signoff/{design}.lef",
            "**/[0-9]*-magic-writelef/{design}.lef",
        ],
    },
    {
        "dest": "factory/def/lingologos_top.def",
        "kind": "copy",
        "purpose": "DEF of the routed design, the most advanced DEF the run wrote",
        "note": "the traceability table of the export specification names one "
                "factory/lef_def directory while its own directory tree names "
                "factory/lef and factory/def; the tree wins, because that is the "
                "layout every other artifact in this export is filed under",
        "candidates": [
            "final/def/{design}.def",
            "results/routing/{design}.def",
            "results/signoff/{design}.def",
            "**/[0-9]*-{design}.def",
        ],
    },
    {
        "dest": "factory/sdc/lingologos_top.sdc",
        "kind": "copy",
        "purpose": "constraint set the run was signed off against",
        "candidates": [
            "final/sdc/{design}.sdc",
            "results/signoff/{design}.sdc",
            "**/[0-9]*-openroad-checksdcfiles/{design}.sdc",
        ],
        "extra_sources": "SIGNOFF_SDC_FILE of resolved.json is tried last of all",
    },
    {
        "dest": "factory/reports/drc_violations.rpt",
        "kind": "copy",
        "purpose": "DRC report of the signoff DRC step",
        "candidates": [
            "**/[0-9]*-magic-drc/reports/drc_violations.magic.rpt",
            "reports/signoff/*drc*.rpt",
        ],
    },
    {
        "dest": "factory/reports/drc_violations.klayout.xml",
        "kind": "copy",
        "optional": True,
        "purpose": "the same check in the other tool, kept in the format it was written in",
        "candidates": [
            "**/[0-9]*-klayout-drc/reports/drc_violations.klayout.xml",
        ],
    },
    {
        "dest": "factory/reports/lvs_report.rpt",
        "kind": "copy",
        "purpose": "LVS report of the signoff LVS step",
        "candidates": [
            "**/[0-9]*-netgen-lvs/reports/lvs.netgen.rpt",
            "reports/signoff/*lvs*.rpt",
        ],
    },
    {
        "dest": "factory/reports/antenna_final.rpt",
        "kind": "copy",
        "optional": True,
        "purpose": "antenna report of the last antenna check of the run",
        "candidates": [
            "**/[0-9]*-openroad-checkantennas*/reports/antenna.rpt",
            "**/[0-9]*-openroad-checkantennas/reports/antenna.rpt",
        ],
    },
    {
        "dest": "factory/reports/signoff_summary.rpt",
        "kind": "copy",
        "purpose": "the pass or fail summary Misc.ReportManufacturability printed",
        "candidates": [
            "**/[0-9]*-misc-reportmanufacturability/manufacturability.rpt",
            "reports/*manufacturability*.rpt",
        ],
    },
    {
        "dest": "factory/reports/run_metrics.json",
        "kind": "copy",
        "optional": True,
        "purpose": "the metrics the run recorded, copied so the reports beside it can be checked",
        "candidates": ["final/metrics.json", "metrics.json"],
    },
    {
        "dest": "factory/reports/sta_signoff.rpt",
        "kind": "bundle",
        "purpose": "per corner setup and hold summary taken from the metrics of the run",
    },
    {
        "dest": "visualization_board/netlists/lingologos_synth.v",
        "kind": "copy",
        "purpose": "gate level Verilog from the synthesis step",
        "candidates": [
            "**/[0-9]*-yosys-synthesis/{design}.nl.v",
            "results/synthesis/{design}.nl.v",
        ],
    },
    {
        "dest": "visualization_board/netlists/lingologos_routed.v",
        "kind": "copy",
        "purpose": "gate level Verilog after fill insertion, the netlist the layout implements",
        "note": "the export specification names one gate level netlist and describes it "
                "as both post synthesis and post routing; the two are different files "
                "with different instance counts, so both are exported and labelled",
        "candidates": [
            "final/nl/{design}.nl.v",
            "results/routing/{design}.nl.v",
            "**/[0-9]*-openroad-fillinsertion/{design}.nl.v",
        ],
    },
    {
        "dest": "visualization_board/netlists/lingologos_powered.v",
        "kind": "copy",
        "optional": True,
        "purpose": "the same netlist with VPWR and VGND attached, the view LVS compared",
        "candidates": [
            "final/pnl/{design}.pnl.v",
            "**/[0-9]*-openroad-fillinsertion/{design}.pnl.v",
        ],
    },
    {
        "dest": "visualization_board/netlists/lingologos_top.nl",
        "kind": "gen",
        "purpose": "structured node and edge netlist generated from the most complete "
                   "Verilog of the design; no upstream step writes a .nl file, so this "
                   "file is generated here and its format is documented in its header",
    },
    {
        "dest": "visualization_board/system_board/board_complete.v",
        "kind": "gen",
        "purpose": "board level structural wrapper: the dies config/topology.json "
                   "declares, their pin groups brought out to board signals, and the "
                   "link wires between them",
    },
]


def sha256_of(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        return handle.read()


def is_wsl() -> bool:
    if os.environ.get("WSL_DISTRO_NAME"):
        return True
    try:
        return "microsoft" in read_text("/proc/version").lower()
    except OSError:
        return False


def host_path_of(path: str) -> Optional[str]:
    """Host visible form of a path, through the wslpath of the distribution."""
    try:
        done = subprocess.run(["wslpath", "-w", path], capture_output=True, text=True)
    except OSError:
        return None
    if done.returncode != 0:
        return None
    return done.stdout.strip()


def desktop_dir() -> Optional[str]:
    """Desktop of the Windows host, as this process can reach it."""
    profile = os.environ.get("USERPROFILE")
    if profile:
        return os.path.join(profile, "Desktop", "LingoLOGOS_Export")
    if is_wsl():
        name = os.environ.get("LOGNAME") or os.environ.get("USER") or ""
        if name:
            return "/mnt/c/Users/%s/Desktop/LingoLOGOS_Export" % name
    return None


def translate_host_path(text: str) -> Optional[str]:
    """WSL form of a Windows path, through wslpath -u."""
    try:
        done = subprocess.run(["wslpath", "-u", text], capture_output=True, text=True)
    except OSError:
        return None
    if done.returncode != 0:
        return None
    return done.stdout.strip()


def find_runs(runs_dir: str) -> List[str]:
    """Run directories, newest first. OpenLane 2 names them RUN_date_time, an
    OpenLane 1 run carries the -tag name given on the command line."""
    try:
        names = os.listdir(runs_dir)
    except OSError:
        return []
    found = [os.path.join(runs_dir, n) for n in names
             if os.path.isdir(os.path.join(runs_dir, n))]
    return sorted(found, key=lambda p: (os.path.getmtime(p), os.path.basename(p)),
                  reverse=True)


def resolved_config(run: str) -> Dict[str, object]:
    path = os.path.join(run, "resolved.json")
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def step_number(path: str, run: str) -> int:
    """Leading step number of an OpenLane 2 step directory, 0 when there is none.
    A nested step directory, 43-openroad-repairantennas/2-openroad-checkantennas,
    takes the number of its parent so the later stage of a repair loop sorts later."""
    try:
        relative = os.path.relpath(path, run)
    except ValueError:
        return 0
    for part in relative.split(os.sep):
        match = re.match(r"^(\d+)-", part)
        if match:
            return int(match.group(1))
    return 0


def children(path: str) -> List[str]:
    """Immediate children of a directory, sorted, empty when it cannot be read."""
    try:
        return sorted(os.listdir(path))
    except OSError:
        return []


def base_subtrees(path: str) -> List[str]:
    """The directory itself and every directory under it, for a ** component."""
    out = [path]
    for root, dirs, _ in os.walk(path):
        for name in dirs:
            out.append(os.path.join(root, name))
    return out


def resolve_candidate(run: str, design: str, pattern: str) -> List[str]:
    """Files of the run matching a pattern written component by component.

    A component matches a child of the match of the component before it, which is
    the rule every shell glob follows, and it is matched case sensitively and one
    name at a time: the leading component of a pattern for a step directory of
    OpenLane 2 is therefore the run itself, and ** means any depth. Walking by
    hand rather than through glob has a second reason, which is that glob on
    Windows folds case, and the step directory names of a run made in Linux must
    not be matched as if they were written in a case insensitive file system. A
    component with no wildcard is looked up by name in the listing of the directory
    in front of it for the same reason: os.path.exists on a hand built path would
    answer yes on Windows to final/Gds while the run of the flow has final/gds, and
    the export would then be correct on one machine and empty on the other.
    """
    run = str(run)
    parts = pattern.replace("{design}", design).split("/")
    current = [run]
    for part in parts:
        if not part:
            continue
        following: List[str] = []
        for base in current:
            if part == "**":
                following.extend(base_subtrees(base))
            elif any(character in part for character in "*?["):
                if not os.path.isdir(base):
                    continue
                following.extend(os.path.join(base, name) for name in children(base)
                                 if fnmatch.fnmatchcase(name, part))
            else:
                following.extend(os.path.join(base, name) for name in children(base)
                                 if name == part)
        current = following
    return [path for path in current if os.path.exists(path)]


def pick_source(run: str, design: str,
                candidates: Sequence[str]) -> Tuple[Optional[str], List[str]]:
    """Best match over all candidates: final views first, then results, then the
    highest numbered step directory. Tried patterns are returned for the report."""
    run = str(run)
    ranked: List[Tuple[Tuple[int, int], str]] = []
    for pattern in candidates:
        for path in resolve_candidate(run, design, pattern):
            tail = path[len(run):].replace(os.sep, "/")
            # The final view directory is what OpenLane 2 leaves as the answer of the
            # flow, results is the per step output, and a step directory of an OpenLane
            # 1 flow or a nested repair step is ranked by its step number so the last
            # stage of a loop wins over the first one.
            rank = 2 if "/final/" in tail else (1 if "/results/" in tail else 0)
            ranked.append(((rank, step_number(path, run)), path))
    if not ranked:
        return None, list(candidates)
    ranked.sort(key=lambda item: (-item[0][0], -item[0][1], item[1]))
    return ranked[0][1], list(candidates)


def metrics_of(run: str) -> Dict[str, float]:
    for path in (os.path.join(run, "final", "metrics.json"),
                 os.path.join(run, "metrics.json")):
        if os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8", errors="replace") as handle:
                    data = json.load(handle)
            except (OSError, ValueError):
                return {}
            if isinstance(data, dict):
                return {k: v for k, v in data.items() if isinstance(v, (int, float))}
    return {}


def build_sta_report(run: str, design: str, metrics: Dict[str, float]) -> str:
    """A signoff timing summary: one line per corner, read from the metrics of this
    run, plus the place where the full per corner reports of OpenSTA live."""
    lines = [
        "LingoLOGOS static timing signoff summary, design %s" % design,
        "run directory: %s" % run,
        "",
        "generated by asic_chip/scripts/export_artifacts.py from the metrics of the run;",
        "the numbers below are the numbers the run recorded, not recomputed here.",
        "",
    ]

    def value(kind: str, corner_local: Optional[str]) -> Optional[float]:
        key = "timing__%s__corner:%s" % (kind, corner_local) if corner_local \
            else "timing__%s" % kind
        found = metrics.get(key)
        return found if isinstance(found, (int, float)) else None

    corner_names = set()
    for key in metrics:
        match = re.match(r"^timing__(?:setup|hold)__[a-z]+__corner:(.+)$", key)
        if match:
            corner_names.add(match.group(1))
    corners = sorted(corner_names)
    if not corners:
        lines.append("The run wrote no per corner timing metrics, so this summary is empty.")
    else:
        header = ("corner".ljust(24) + "setup wns".rjust(11) + "setup tns".rjust(11)
                  + "hold wns".rjust(10) + "hold tns".rjust(10)
                  + "worst setup slack".rjust(19) + "worst hold slack".rjust(18)
                  + "closed".rjust(8))
        lines.append(header)
        lines.append("-" * len(header))
        for corner in corners:
            setup_wns = value("setup__wns", corner)
            setup_tns = value("setup__tns", corner)
            hold_wns = value("hold__wns", corner)
            hold_tns = value("hold__tns", corner)
            setup_ws = value("setup__ws", corner)
            hold_ws = value("hold__ws", corner)

            def fmt(number: Optional[float]) -> str:
                return "-" if number is None else "%.4f" % number

            # Timing is closed for a corner when the worst slack of both checks is
            # non negative. Worst slack is what OpenLane 2 records per corner; the
            # worst negative slack pair is consulted only when no slack is recorded.
            if setup_ws is not None and hold_ws is not None:
                closed = setup_ws >= 0.0 and hold_ws >= 0.0
            else:
                closed = (setup_wns is not None and setup_wns <= 0.0
                          and hold_wns is not None and hold_wns <= 0.0)
            lines.append(corner.ljust(24) + fmt(setup_wns).rjust(11)
                         + fmt(setup_tns).rjust(11) + fmt(hold_wns).rjust(10)
                         + fmt(hold_tns).rjust(10) + fmt(setup_ws).rjust(19)
                         + fmt(hold_ws).rjust(18)
                         + ("yes" if closed else "no").rjust(8))
        pooled = []
        # The unsuffixed timing metrics of a run are already the worst over the
        # corners, so they are reported as written; when a value exists only per
        # corner, the worst of those is the minimum, the most negative number.
        for kind in ("setup__wns", "setup__tns", "setup__ws",
                     "hold__wns", "hold__tns", "hold__ws"):
            gathered = [metrics[k] for k in metrics
                        if k == "timing__%s" % kind or k.startswith("timing__%s__corner:" % kind)]
            if gathered:
                pooled.append((kind, min(gathered)))
        if pooled:
            lines.append("")
            lines.append("worst value over all corners, which is what signoff is judged on:")
            for kind, number in pooled:
                lines.append("  timing__%s = %.4f" % (kind, number))
    lines.append("")
    lines.append("full OpenSTA reports of this run, by corner:")
    sta_dirs = resolve_candidate(run, design, "**/[0-9]*-openroad-stapostpnr")
    if not sta_dirs:
        lines.append("  no openroad-stapostpnr step directory was found in the run")
    for directory in sta_dirs:
        for corner in sorted(children(directory)):
            corner_dir = os.path.join(directory, corner)
            if not os.path.isdir(corner_dir):
                continue
            reports = sorted(name for name in children(corner_dir) if name.endswith(".rpt"))
            lines.append("  %s: %s" % (corner, ", ".join(reports) if reports else "no report files"))
    lines.append("")
    error_log = os.path.join(run, "error.log")
    if os.path.isfile(error_log):
        size = os.path.getsize(error_log)
        lines.append("The flow exit status is not part of a report. The error log of this run is")
        lines.append("error.log of %d bytes, which is %s" % (
            size,
            "empty, so no deferred error from a signoff checker was recorded."
            if size == 0 else
            "not empty, which is where OpenLane 2 records a deferred error from a signoff checker."))
    else:
        lines.append("The flow exit status is not part of a report. error.log is absent: this run")
        lines.append("wrote no error log, which is what OpenLane 1 does.")
    lines.append("")
    return "\n".join(lines)


def strip_comments(text: str) -> str:
    """Remove block comments, line comments and the attribute groups tools such as
    Yosys put in front of a cell. All three are decoration: they carry no
    connectivity, and leaving them in would hide the statement they decorate."""
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
    text = re.sub(r"\(\*.*?\*\)", " ", text, flags=re.S)
    text = re.sub(r"//[^\n]*", "", text)
    return text


def strip_directive_blocks(text: str) -> str:
    """Remove lines that start with a backtick directive, and the lines between an
    ifdef and its endif. The debug port group of the top level is inside such a
    block, and the synthesis flow never defines that macro, so the block is absent
    from the netlist as well and dropping it is the faithful reading."""
    out: List[str] = []
    depth = 0
    for line in text.split("\n"):
        if re.match(r"^`\s*(ifdef|ifndef)\b", line):
            depth += 1
            continue
        if re.match(r"^`\s*(else|elif)\b", line):
            continue
        if re.match(r"^`\s*endif\b", line):
            depth = max(0, depth - 1)
            continue
        if depth:
            continue
        if line.lstrip().startswith("`"):
            continue
        out.append(line)
    return "\n".join(out)


def verilog_source(path: str) -> str:
    """A Verilog file read for structural reading, not for simulation: comments,
    attributes and preprocessor blocks out, one string in."""
    return strip_directive_blocks(strip_comments(read_text(path)))


def unescape(text: str) -> str:
    """Verilog escaped identifiers, \\name , become name, and the space that ends
    the escape goes with it."""
    text = re.sub(r"\\([A-Za-z0-9_$]+)\s", r"\1 ", text)
    return text.strip()


def match_paren(text: str, index: int) -> int:
    """Index of the parenthesis closing the one at index, or -1 when the group is open."""
    depth = 0
    for position in range(index, len(text)):
        if text[position] == "(":
            depth += 1
        elif text[position] == ")":
            depth -= 1
            if depth == 0:
                return position
    return -1


def preceding_character(text: str, index: int) -> str:
    """The last non whitespace character before index, or empty at the start."""
    position = index - 1
    while position >= 0 and text[position].isspace():
        position -= 1
    return text[position] if position >= 0 else ""


def module_region(text: str, module: str) -> Optional[Tuple[str, str]]:
    """Port list text and body text of one module.

    A module may declare parameters before its ports, as the top level of this
    design does, and a parameter list is a parenthesis group too. The two are told
    apart by the character standing in front of the group, which is a hash for a
    parameter list and anything else for a port list: the scan walks past a hashed
    group and stops at the next one, which is the port list.
    """
    match = re.search(r"\bmodule\s+" + re.escape(module) + r"\b", text)
    if not match:
        for candidate in re.finditer(r"\bmodule\s+([A-Za-z_][A-Za-z0-9_$]*)", text):
            if candidate.group(1) == module:
                match = candidate
                break
    if not match:
        return None
    position = match.end()
    port_open = -1
    while position < len(text):
        character = text[position]
        if character == "(":
            if preceding_character(text, position) == "#":
                close = match_paren(text, position)
                if close < 0:
                    return None
                position = close + 1
                continue
            port_open = position
            break
        if character == ";":
            break
        position += 1
    if port_open < 0:
        return None
    port_close = match_paren(text, port_open)
    if port_close < 0:
        return None
    ports = text[port_open + 1:port_close]
    end = re.search(r"\bendmodule\b", text[port_close:])
    if not end:
        return None
    return ports, text[port_close + 1:port_close + end.start()]


class Port(object):
    def __init__(self, name: str, direction: str, range_decl: str, signed: bool = False):
        self.name = name
        self.direction = direction
        self.range_decl = range_decl
        self.signed = signed

    def width(self) -> int:
        match = re.match(r"^\s*\[\s*(\d+)\s*:\s*(\d+)\s*\]$", self.range_decl)
        if not match:
            return 1
        high, low = int(match.group(1)), int(match.group(2))
        return abs(high - low) + 1


def parse_ports(text: str, module: str) -> List[Port]:
    """Ports of a module, read from both the ANSI header and the non ANSI
    declarations a netlist writer puts in the body instead."""
    region = module_region(text, module)
    if region is None:
        return []
    ports, body = region
    found: List[Port] = []
    seen = set()

    def add(direction: str, rest: str) -> None:
        rest = rest.replace("\n", " ").strip()
        range_match = re.search(r"\[[^\]]*\]", rest)
        range_decl = range_match.group(0) if range_match else ""
        signed = bool(re.search(r"\bsigned\b", rest))
        name_match = re.search(r"([A-Za-z_][A-Za-z0-9_$]*)\s*$", rest)
        if not name_match:
            return
        name = name_match.group(1)
        if name in seen:
            return
        seen.add(name)
        found.append(Port(name, direction, range_decl, signed))

    depth = 0
    chunk: List[str] = []
    for character in ports:
        if character in "([{":
            depth += 1
        elif character in ")]}":
            depth -= 1
        if character == "," and depth == 0:
            yield_chunk = "".join(chunk)
            chunk = []
            direction = re.search(r"\b(input|output|inout)\b", yield_chunk)
            if direction:
                add(direction.group(1), yield_chunk[direction.end():])
        else:
            chunk.append(character)
    direction = re.search(r"\b(input|output|inout)\b", "".join(chunk))
    if direction:
        add(direction.group(1), "".join(chunk)[direction.end():])

    for match in re.finditer(
            r"(?m)^[ \t]*(input|output|inout)\b([^;]*?)([A-Za-z_][A-Za-z0-9_$]*)[ \t]*;",
            body):
        add(match.group(1), match.group(2) + " " + match.group(3))
    return found


def module_names(text: str) -> List[str]:
    """Modules of a text already run through verilog_source, in source order."""
    return [m.group(1) for m in re.finditer(r"\bmodule\s+([A-Za-z_][A-Za-z0-9_$]*)", text)]


def split_top_level(text: str) -> List[str]:
    """Split on commas at bracket depth zero."""
    parts: List[str] = []
    depth = 0
    chunk: List[str] = []
    for character in text:
        if character in "([{":
            depth += 1
        elif character in ")]}":
            depth -= 1
        if character == "," and depth == 0:
            parts.append("".join(chunk).strip())
            chunk = []
        else:
            chunk.append(character)
    if "".join(chunk).strip():
        parts.append("".join(chunk).strip())
    return parts


def statements(body: str) -> List[str]:
    """The semicolon terminated statements of a module body, with braces, brackets
    and parentheses respected so a semicolon inside a group does not end one."""
    out: List[str] = []
    depth = 0
    chunk: List[str] = []
    for character in body:
        if character in "([{":
            depth += 1
        elif character in ")]}":
            depth -= 1
        if character == ";" and depth <= 0:
            out.append("".join(chunk))
            chunk = []
        else:
            chunk.append(character)
    return [s.strip() for s in out if s.strip()]


def parse_instances(text: str, module: str) -> List[Tuple[str, str, List[Tuple[str, str]]]]:
    """Module instances of one module body, as (cell type, instance name, connections).
    A connection is (port, net token as written). This is a reader of a subset, not a
    Verilog parser: it recognises TYPE [#(...)] NAME ( ... ); statements and skips the
    statements that begin with a keyword."""
    region = module_region(text, module)
    if region is None:
        return []
    _, body = region
    found: List[Tuple[str, str, List[Tuple[str, str]]]] = []
    for statement in statements(body):
        head = re.match(r"([A-Za-z_][A-Za-z0-9_$]*)[\s#]", statement)
        if not head:
            continue
        cell = head.group(1)
        if cell in NON_INSTANCE_KEYWORDS:
            continue
        open_index = statement.find("(", head.end())
        if open_index < 0:
            continue
        between = statement[head.end():open_index].strip()
        if between.startswith("#"):
            # The first group is the override list of the instance, not the port
            # connections, so the connections live in the group that follows it.
            parameter_close = match_paren(statement, open_index)
            if parameter_close < 0:
                continue
            open_index = statement.find("(", parameter_close + 1)
            if open_index < 0:
                continue
            between = statement[parameter_close + 1:open_index].strip()
        close_index = match_paren(statement, open_index)
        if close_index < 0:
            continue
        name_match = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_$]*)", unescape(between))
        if not name_match:
            continue
        inside = statement[open_index + 1:close_index]
        connections: List[Tuple[str, str]] = []
        for part in split_top_level(inside):
            part = unescape(part)
            named = re.match(r"^\.\s*([A-Za-z_][A-Za-z0-9_$]*)\s*\((.*)\)$", part, re.S)
            if named:
                connections.append((named.group(1), " ".join(named.group(2).split())))
            elif part:
                connections.append(("", " ".join(part.split())))
        found.append((cell, name_match.group(1), connections))
    return found


def build_node_netlist(sources: Sequence[Tuple[str, str]], top: str) -> str:
    """The structured .nl file. One section per module, one node line per port and
    per instance, one edge line per connection, in source order, so two runs of the
    same input produce the same bytes."""
    lines = [
        "# LingoLOGOS structured node and edge netlist, schema lingologos-nl-v1",
        "# generated by asic_chip/scripts/export_artifacts.py, not by a tool of the flow",
        "# top module: %s" % top,
        "#",
        "# node lines name one vertex of the connectivity graph:",
        "#   node port      <module>.<port> <direction> <bits>      a pin of a module",
        "#   node instance  <module>.<inst> cell:<cell type>       one instantiation of a module",
        "# edge lines name one connection of that graph, always instance to net token:",
        "#   edge <module>.<inst> .<port> <net token>              named connection",
        "#   edge <module>.<inst> <position> <net token> positional connection, a netlist written",
        "#                                                                   without port names",
        "# A module section is opened by module and closed by end. A module defined by an",
        "# earlier source is skipped here, so the file names every module once.",
        "#",
    ]
    for path, label in sources:
        try:
            digest = sha256_of(path)
        except OSError:
            digest = "unreadable"
        lines.append("# source %s (%s) sha256=%s" % (path, label, digest))
    lines.append("")
    seen: set = set()
    for path, label in sources:
        try:
            text = verilog_source(path)
        except OSError:
            continue
        for module in module_names(text):
            if module in seen:
                lines.append("# skipped %s, defined by an earlier source" % module)
                continue
            seen.add(module)
            ports = parse_ports(text, module)
            instances = parse_instances(text, module)
            lines.append("module %s" % module)
            lines.append("  origin %s %s" % (path, label))
            lines.append("  nodes %d %d" % (len(ports), len(instances)))
            for port in ports:
                lines.append("  node port %s.%s %s %d"
                             % (module, port.name, port.direction, port.width()))
            for cell, inst, _ in instances:
                lines.append("  node instance %s.%s cell:%s" % (module, inst, cell))
            for cell, inst, connections in instances:
                position = 0
                for port, net in connections:
                    if port:
                        lines.append("  edge %s.%s .%s %s" % (module, inst, port, net))
                    else:
                        lines.append("  edge %s.%s %d %s positional"
                                     % (module, inst, position, net))
                        position += 1
            lines.append("end %s" % module)
            lines.append("")
    return "\n".join(lines)


def router_parameters() -> Dict[str, int]:
    found: Dict[str, int] = {}
    if not os.path.isfile(RTL_ROUTER):
        return found
    for match in re.finditer(r"parameter\s+int\s+([A-Z_0-9]+)\s*=\s*(\d+)",
                             verilog_source(RTL_ROUTER)):
        found[match.group(1)] = int(match.group(2))
    text = verilog_source(RTL_ROUTER)
    # The switch declares the node count port as "input logic [3:0] total_nodes", so the
    # range sits in front of the name. The other order is read as well, because a module
    # may declare the range after the name.
    total = re.search(r"\[\s*(\d+)\s*:\s*0\s*\]\s*(?:\w+\s+)?total_nodes", text)
    if not total:
        total = re.search(r"total_nodes\s*\[\s*(\d+)\s*:\s*0\s*\]", text)
    if total:
        found["TOTAL_NODES_WIDTH"] = int(total.group(1)) + 1
    return found


def board_topology(topology_path: str) -> Tuple[List[str], Dict[str, List[str]]]:
    try:
        with open(topology_path, "r", encoding="utf-8", errors="replace") as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        return [], {}
    instances = data.get("instances") or []
    dies = [str(item.get("id")) for item in instances if isinstance(item, dict) and item.get("id")]
    routing: Dict[str, List[str]] = {}
    for rule in data.get("routingMatrix") or []:
        if not isinstance(rule, dict) or rule.get("from") is None:
            continue
        source = str(rule["from"])
        targets = rule.get("to") or []
        if isinstance(targets, str):
            targets = [targets]
        routing.setdefault(source, [])
        for target in targets:
            if str(target) not in routing[source]:
                routing[source].append(str(target))
    return dies, routing


def safe(name: str) -> str:
    """A Verilog identifier out of a topology id, which may hold other characters."""
    cleaned = re.sub(r"[^A-Za-z0-9_]", "_", name)
    if not re.match(r"^[A-Za-z_]", cleaned):
        cleaned = "n_" + cleaned
    return cleaned


def build_board_wrapper(dies: Sequence[str], routing: Dict[str, List[str]],
                        chip_ports: List[Port], router_params: Dict[str, int],
                        topology_path: str, design: str, port_source: str) -> str:
    """SystemVerilog board wrapper: one die per instance the topology declares, one
    topology switch per die, and the wires of the routing matrix as the only link
    channels that exist. Node identifiers are the position of the instance in
    config/topology.json, which is the order the control layer assigns, and channel
    index k of a switch is the wire towards node k, as the header of
    lingologos_topology_router.sv states, so the receive channels of a die are the
    transmit channels of its neighbours and nothing else.
    """
    node_width = router_params.get("NODE_ID_WIDTH", 4)
    data_width = router_params.get("DATA_WIDTH", 64)
    max_nodes = router_params.get("MAX_NODES", 16)
    total_width = router_params.get("TOTAL_NODES_WIDTH", 4)
    links = [(source, target) for source in dies
             for target in routing.get(source, []) if target in dies]
    power = [port for port in chip_ports if port.name in ("VPWR", "VGND")]
    envelope = [port for port in chip_ports
                if port.name not in ("VPWR", "VGND", "clk", "rst_n")]
    lines = [
        "/*",
        " * LingoLOGOS board level structural wrapper, generated by",
        " * asic_chip/scripts/export_artifacts.py. Do not edit by hand: rerun the",
        " * export and the file moves with the topology it reads.",
        " *",
        " * Source of the shape: %s" % topology_path,
        " *   dies, in node id order: %s" % ", ".join(dies),
    ]
    for die in dies:
        lines.append(" *   links out of %s: %s" % (die, ", ".join(routing.get(die, [])) or "none"))
    lines += [
        " *",
        " * Pin list of a die, read from %s:" % port_source,
        " * that is the envelope asic_chip/doc/datasheet_lingologos.md documents, and",
        " * every pin of it is brought out to a board connector named after the die, so",
        " * three dies on one board are three visible groups of pins, not one merged",
        " * blob whose provenance nobody can recover. A connector keeps the direction",
        " * of the pin it is wired to, because it hangs off that pin directly.",
        " *",
        " * What this file is for: schematic and board level visualization, and a",
        " * description a standard EDA tool can elaborate. It is not part of the chip:",
        " * asic_chip/openlane/config.json does not list it, so no synthesis or place",
        " * and route run sees it, and lingologos_topology_router.sv is not in that",
        " * source list either, which the header of that module states as well.",
        " *",
        " * What is not modelled here, and is not claimed by it: the board clock",
        " * source, the reset network, the power distribution, the package and the",
        " * link medium between dies. Those are board design work, not chip work.",
        " *",
        " * The message side of the switch, msg_valid and friends, is exposed as a",
        " * board connector because the die has no envelope pin for it yet: the",
        " * package of asic_chip/doc/datasheet_lingologos.md has no link pin.",
        " *",
        " * The name ends in .v because the export specification asks for that name.",
        " * The declarations are SystemVerilog, because the link ports of the switch",
        " * are packed two dimensional arrays, which Verilog 2001 has no notation for.",
        " * Verilator reads a .v file as SystemVerilog by default, and this file passes",
        " * verilator --lint-only with the warning flags of asic_chip/Makefile, that is",
        " * -Wall -Wno-UNSIGNED -Wno-UNUSED -Wno-WIDTHTRUNC, on the container image",
        " * verilator/verilator:latest, which is how the lint targets of that Makefile",
        " * are run on this machine.",
        " */",
        "`default_nettype none",
        "/* verilator lint_off DECLFILENAME */",
        "module lingologos_board #(",
        "    parameter int NODE_ID_WIDTH = %d," % node_width,
        "    parameter int DATA_WIDTH    = %d," % data_width,
        "    parameter int MAX_NODES     = %d" % max_nodes,
        ") (",
        "    input  wire board_clk,",
        "    input  wire board_rst_n,",
    ]
    port_lines: List[str] = []
    if power:
        port_lines.append("    // the two supply rails of the board, shared by every die")
        for port in power:
            port_lines.append("    inout  wire %s" % port.name)
        port_lines.append("    // one connector per pin of a die, named after the die")
    for die in dies:
        name = safe(die)
        for port in envelope:
            # The connector of a die carries the direction of the pin it is wired
            # to: what the die drives leaves the board as an output, and what the
            # board drives into the die arrives as an input. The board module
            # connects each connector straight to that pin of the die instance, so
            # the two directions have to agree or the connector would be driven
            # from both ends.
            range_decl = ""
            if port.signed:
                # signed comes before the range in a declaration, not after it.
                range_decl += "signed "
            if port.range_decl:
                range_decl += port.range_decl.replace(" ", "") + " "
            port_lines.append("    %-6s wire %s%s_%s" % (port.direction, range_decl, name, port.name))
        port_lines.append("    output wire %s_routing_error" % name)
        port_lines.append("    input  wire %s_msg_valid" % name)
        port_lines.append("    output wire %s_msg_ready" % name)
        port_lines.append("    input  wire [NODE_ID_WIDTH-1:0] %s_msg_dest" % name)
        port_lines.append("    input  wire [DATA_WIDTH-1:0] %s_msg_data" % name)
        port_lines.append("    output wire %s_rx_valid" % name)
        port_lines.append("    input  wire %s_rx_ready" % name)
        port_lines.append("    output wire [DATA_WIDTH-1:0] %s_rx_data" % name)
    lines.append(",\n".join(port_lines))
    lines.append(");")
    lines.append("")
    lines.append("    // The topology declares %d die(s), so node ids run 0 to %d, and"
                 % (len(dies), len(dies) - 1))
    lines.append("    // the %d bit total_nodes field of every switch is told that number."
                 % total_width)
    lines.append("")
    for die in dies:
        name = safe(die)
        lines.append("    // link channels of die %s, index k is the wire towards node %s" % (die, "k"))
        for signal in ("tx_valid", "tx_ready", "rx_valid", "rx_ready"):
            lines.append("    wire [MAX_NODES-1:0] %s_link_%s;" % (name, signal))
        for signal in ("tx_dest", "rx_dest"):
            lines.append("    wire [MAX_NODES-1:0][NODE_ID_WIDTH-1:0] %s_link_%s;" % (name, signal))
        for signal in ("tx_data", "rx_data"):
            lines.append("    wire [MAX_NODES-1:0][DATA_WIDTH-1:0] %s_link_%s;" % (name, signal))
    lines.append("")
    driven_tx: Dict[str, set] = {safe(d): set() for d in dies}
    driven_rx: Dict[str, set] = {safe(d): set() for d in dies}
    for source, target in links:
        driven_rx[safe(target)].add(dies.index(source))
        driven_tx[safe(source)].add(dies.index(target))
    lines += [
        "    // Every channel of every switch is one bit of a MAX_NODES wide port, and the",
        "    // switch reads all of them, so every one needs exactly one driver. A channel",
        "    // a link on this board drives gets that driver below; every other channel is",
        "    // tied idle, one channel at a time. Tying the whole port idle instead would",
        "    // overlap the channel a link drives, and two strong drivers on one bit of a",
        "    // net resolve to x in a four state simulator, which is the difference between",
        "    // an idle channel and a broken one.",
    ]
    for die in dies:
        name = safe(die)
        for index in range(max_nodes):
            if index not in driven_rx[name]:
                lines.append("    assign %s_link_rx_valid[%d] = 1'b0;" % (name, index))
                lines.append("    assign %s_link_rx_dest[%d]  = %d'd0;" % (name, index, node_width))
                lines.append("    assign %s_link_rx_data[%d]  = %d'd0;" % (name, index, data_width))
            if index not in driven_tx[name]:
                lines.append("    assign %s_link_tx_ready[%d] = 1'b0;" % (name, index))
    lines.append("")
    lines.append("    // the wires of the routing matrix, one full duplex channel per rule")
    for index, (source, target) in enumerate(links):
        src, dst = safe(source), safe(target)
        from_id, to_id = dies.index(source), dies.index(target)
        lines.append("    // link %d: node %d (%s) to node %d (%s)"
                     % (index, from_id, source, to_id, target))
        lines.append("    assign %s_link_rx_valid[%d] = %s_link_tx_valid[%d];"
                     % (dst, from_id, src, to_id))
        lines.append("    assign %s_link_rx_dest[%d]  = %s_link_tx_dest[%d];"
                     % (dst, from_id, src, to_id))
        lines.append("    assign %s_link_rx_data[%d]  = %s_link_tx_data[%d];"
                     % (dst, from_id, src, to_id))
        lines.append("    assign %s_link_tx_ready[%d] = %s_link_rx_ready[%d];"
                     % (src, to_id, dst, from_id))
    lines.append("")
    for index, die in enumerate(dies):
        name = safe(die)
        mask = 0
        for target in routing.get(die, []):
            if target in dies:
                mask |= 1 << dies.index(target)
        mask_text = "".join("1" if mask & (1 << bit) else "0"
                            for bit in range(max_nodes - 1, -1, -1))
        lines.append("    // die %d: %s" % (index, die))
        lines.append("    %s u_chip_%s (" % (design, name))
        lines.append("        .clk     (board_clk),")
        lines.append("        .rst_n   (board_rst_n),")
        for port in power:
            lines.append("        .%s   (%s)," % (port.name, port.name))
        for port in envelope:
            lines.append("        .%s   (%s_%s)," % (port.name, name, port.name))
        lines[-1] = lines[-1].rstrip(",")
        lines.append("    );")
        lines.append("")
        lines.append("    lingologos_topology_router #(")
        lines.append("        .NODE_ID_WIDTH (NODE_ID_WIDTH),")
        lines.append("        .DATA_WIDTH    (DATA_WIDTH),")
        lines.append("        .MAX_NODES     (MAX_NODES)")
        lines.append("    ) u_switch_%s (" % name)
        lines.append("        .clk           (board_clk),")
        lines.append("        .rst_n         (board_rst_n),")
        lines.append("        .standalone    (1'b0),")
        lines.append("        .node_id       (%d'd%d)," % (node_width, index))
        lines.append("        .total_nodes   (%d'd%d)," % (total_width, len(dies)))
        lines.append("        .routing_mask  (%d'b%s)," % (max_nodes, mask_text))
        lines.append("        .tx_valid      (%s_msg_valid)," % name)
        lines.append("        .tx_ready      (%s_msg_ready)," % name)
        lines.append("        .tx_dest       (%s_msg_dest)," % name)
        lines.append("        .tx_data       (%s_msg_data)," % name)
        lines.append("        .rx_valid      (%s_rx_valid)," % name)
        lines.append("        .rx_ready      (%s_rx_ready)," % name)
        lines.append("        .rx_data       (%s_rx_data)," % name)
        lines.append("        .link_tx_valid (%s_link_tx_valid)," % name)
        lines.append("        .link_tx_ready (%s_link_tx_ready)," % name)
        lines.append("        .link_tx_dest  (%s_link_tx_dest)," % name)
        lines.append("        .link_tx_data  (%s_link_tx_data)," % name)
        lines.append("        .link_rx_valid (%s_link_rx_valid)," % name)
        lines.append("        .link_rx_dest  (%s_link_rx_dest)," % name)
        lines.append("        .link_rx_data  (%s_link_rx_data)," % name)
        lines.append("        .link_rx_ready (%s_link_rx_ready)," % name)
        lines.append("        .routing_error (%s_routing_error)" % name)
        lines.append("    );")
        lines.append("")
    lines += [
        "    // The board connector and the pin of the die are two names for one net. A",
        "    // connector the board drives feeds the die, and a connector the die drives",
        "    // feeds the board, which is why the directions of the connector list above",
        "    // are the ones the die has, seen from outside the die. Getting that the other",
        "    // way round drives a port the board declares as an input, which no",
        "    // elaborator accepts, and it is the one mistake a file like this makes most",
        "    // often. The order of the connector list is the order the source of the",
        "    // envelope declares its ports in, which the header above names.",
        "",
        "endmodule",
        "/* verilator lint_on DECLFILENAME */",
        "`default_nettype wire",
        "",
    ]
    return "\n".join(lines)


class Exporter(object):
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.entries: List[Dict[str, object]] = []

    def record(self, dest: str, status: str, purpose: str, **extra: object) -> None:
        entry: Dict[str, object] = {"artifact": dest, "status": status, "purpose": purpose}
        entry.update(extra)
        self.entries.append(entry)

    def write_generated(self, target: str, text: str) -> None:
        """Generated text goes out with LF endings. The gates of this repository read
        ASIC text as LF only, and a tool reading the export should not have to guess
        which line ending the person who ran the export happened to be standing on."""
        if self.args.dry_run:
            return
        parent = os.path.dirname(target)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(target, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)

    def copy(self, entry: Dict[str, object], run: str, design: str, out_root: str) -> None:
        dest = str(entry["dest"])
        purpose = str(entry["purpose"])
        candidates = list(entry.get("candidates") or [])
        target = os.path.join(out_root, dest.replace("/", os.sep))
        source: Optional[str] = None
        extra: List[str] = []
        if entry.get("extra_sources") and run:
            signoff = resolved_config(run).get("SIGNOFF_SDC_FILE")
            # An absolute path outside the run directory: tried last of all, and not
            # run through the pattern matcher, which reads a candidate as a path
            # relative to the run.
            if isinstance(signoff, str) and os.path.isfile(signoff):
                extra.append(signoff)
        if run:
            source, _ = pick_source(run, design, candidates)
        if source is None and extra:
            source = extra[0]
        if source is None:
            self.record(dest, "MISSING", purpose,
                        looked_at=candidates + extra,
                        optional=bool(entry.get("optional")))
            return
        size = os.path.getsize(source)
        if not self.args.dry_run:
            parent = os.path.dirname(target)
            if parent:
                os.makedirs(parent, exist_ok=True)
            shutil.copy2(source, target)
        self.record(dest, "COPIED", purpose,
                    source=source, size_bytes=size, sha256=sha256_of(source))

    def generate(self, dest: str, design: str, run: Optional[str], out_root: str) -> None:
        target = os.path.join(out_root, dest.replace("/", os.sep))
        if dest.endswith(".nl"):
            sources = self.nl_sources(design, run)
            if not sources:
                self.record(dest, "MISSING",
                            "no Verilog of the design in the run and no RTL in the repository")
                return
            text = build_node_netlist(sources, design)
            self.write_generated(target, text)
            self.record(dest, "GENERATED",
                        "structured node and edge netlist built from %s"
                        % ", ".join(os.path.basename(p) for p, _ in sources),
                        generated_from=[p for p, _ in sources],
                        size_bytes=len(text.encode("utf-8")))
            return
        dies, routing = board_topology(self.args.topology)
        if not dies:
            self.record(dest, "MISSING",
                        "the topology file declares no instance: %s" % self.args.topology)
            return
        # The envelope of the die is the RTL top level, not the netlist: the netlist of
        # the routed design carries VPWR and VGND as ports, which are the straps of the
        # layout and not what a board designer wires, and the RTL order is the order the
        # datasheet documents. The netlist serves as the envelope only when no RTL top
        # level is readable, and then the two rails become board connectors.
        chip_ports: List[Port] = []
        port_source = ""
        if os.path.isfile(RTL_TOP):
            chip_ports = parse_ports(verilog_source(RTL_TOP), design)
            port_source = RTL_TOP
        if not chip_ports:
            for path, _ in self.nl_sources(design, run):
                chip_ports = parse_ports(verilog_source(path), design)
                if chip_ports:
                    port_source = path
                    break
        if not chip_ports:
            self.record(dest, "MISSING", "the port list of %s could not be read" % design)
            return
        text = build_board_wrapper(dies, routing, chip_ports, router_parameters(),
                                   self.args.topology, design, port_source)
        self.write_generated(target, text)
        self.record(dest, "GENERATED",
                    "board wrapper for %d die(s) declared by %s" % (len(dies), self.args.topology),
                    generated_from=[self.args.topology, RTL_ROUTER],
                    size_bytes=len(text.encode("utf-8")))

    def nl_sources(self, design: str, run: Optional[str]) -> List[Tuple[str, str]]:
        """Most complete Verilog description available, then the RTL it came from."""
        found: List[Tuple[str, str]] = []
        seen: set = set()

        def add(path: Optional[str], label: str) -> None:
            if path and os.path.isfile(path) and path not in seen:
                seen.add(path)
                found.append((path, label))

        if run:
            for pattern, label in (
                    ("final/pnl/%s.pnl.v" % design, "post layout netlist with the power nets attached"),
                    ("final/nl/%s.nl.v" % design, "post layout gate level netlist"),
                    ("**/[0-9]*-openroad-fillinsertion/%s.nl.v" % design, "post layout gate level netlist"),
                    ("**/[0-9]*-yosys-synthesis/%s.nl.v" % design, "post synthesis gate level netlist"),
            ):
                if "**" in pattern:
                    matches = resolve_candidate(run, design, pattern)
                    matches.sort(key=lambda p: (-step_number(p, run), p))
                    add(matches[0] if matches else None, label)
                else:
                    add(os.path.join(run, pattern.replace("{design}", design)), label)
        for path in [RTL_TOP] + RTL_GENERATED + RTL_CORE:
            add(path, "register transfer level source")
        add(RTL_ROUTER, "topology switch of the board, not part of the chip")
        return found

    def run(self) -> int:
        runs_dir = self.args.runs_dir
        if self.args.run:
            run = self.args.run
            if not os.path.isdir(run):
                print("[EXPORT] the run directory %s does not exist" % run)
                return 1
        else:
            runs = find_runs(runs_dir)
            if not runs:
                print("[EXPORT] no run directory under %s" % runs_dir)
                print("[EXPORT] run the flow first, make -f asic_chip/Makefile openlane,")
                print("[EXPORT] or point --runs-dir at the runs of another checkout")
                return 1
            run = runs[0]
        resolved = resolved_config(run)
        design = self.args.design or str(resolved.get("DESIGN_NAME") or "lingologos_top")
        print("[EXPORT] design %s" % design)
        print("[EXPORT] run %s" % run)
        for key in ("OPENLANE_VERSION", "PDK", "CLOCK_PERIOD", "SIGNOFF_SDC_FILE"):
            if key in resolved:
                print("[EXPORT]   %s = %s" % (key, resolved[key]))
        out_root = self.args.out
        host_form = None
        if is_wsl():
            host_form = host_path_of(out_root)
        print("[EXPORT] destination %s" % out_root)
        if host_form:
            print("[EXPORT] the same directory on the host: %s" % host_form)
        if not self.args.dry_run:
            os.makedirs(out_root, exist_ok=True)
        for entry in ARTIFACT_MATRIX:
            kind = str(entry["kind"])
            dest = str(entry["dest"])
            if kind == "copy":
                self.copy(entry, run, design, out_root)
            elif kind == "bundle":
                text = build_sta_report(run, design, metrics_of(run))
                target = os.path.join(out_root, dest.replace("/", os.sep))
                self.write_generated(target, text)
                self.record(dest, "GENERATED", str(entry["purpose"]),
                            generated_from=["metrics and step directories of %s" % run],
                            size_bytes=len(text.encode("utf-8")))
            else:
                self.generate(dest, design, run, out_root)
        copied = [e for e in self.entries if e["status"] == "COPIED"]
        generated = [e for e in self.entries if e["status"] == "GENERATED"]
        missing = [e for e in self.entries if e["status"] == "MISSING"]
        required_missing = [e for e in missing if not e.get("optional")]
        manifest = {
            "schema": "lingologos-export-v1",
            "generated_utc": datetime.datetime.now(datetime.timezone.utc)
                             .strftime("%Y-%m-%dT%H:%M:%SZ"),
            "tool": "asic_chip/scripts/export_artifacts.py",
            "design": design,
            "run": {
                "directory": run,
                "name": os.path.basename(run),
                "openlane_version": resolved.get("OPENLANE_VERSION"),
                "pdk": resolved.get("PDK"),
                "clock_period": resolved.get("CLOCK_PERIOD"),
                "signoff_sdc_file": resolved.get("SIGNOFF_SDC_FILE"),
                "final_views_present": os.path.isdir(os.path.join(run, "final")),
                "flow_error_log_bytes": (os.path.getsize(os.path.join(run, "error.log"))
                                         if os.path.isfile(os.path.join(run, "error.log")) else None),
            },
            "destination": out_root,
            "destination_host_path": host_form,
            "entries": self.entries,
            "missing": len(missing),
            "missing_required": len(required_missing),
        }
        if not self.args.dry_run:
            with open(os.path.join(out_root, "manifest.json"), "w",
                      encoding="utf-8", newline="\n") as handle:
                json.dump(manifest, handle, indent=2, sort_keys=False)
                handle.write("\n")
        print("[EXPORT] copied %d, generated %d, missing %d"
              % (len(copied), len(generated), len(missing)))
        for entry in self.entries:
            if entry["status"] == "COPIED":
                print("[OK]   %s  (%d bytes)" % (entry["artifact"], entry.get("size_bytes", 0)))
            elif entry["status"] == "GENERATED":
                print("[GEN]  %s  (%d bytes)" % (entry["artifact"], entry.get("size_bytes", 0)))
            else:
                print("[MISS] %s" % entry["artifact"])
                if entry.get("optional"):
                    print("       optional artifact, the flow of this run wrote no such report")
                if entry.get("looked_at"):
                    print("       looked at: %s" % ", ".join(str(p) for p in entry["looked_at"]))
        if required_missing:
            print("[EXPORT] the required artifacts above make the exit status 1; the export")
            print("[EXPORT] writes no placeholder for an artifact it could not read")
            if not self.args.allow_missing:
                return 1
        return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export the fabrication and board artifacts of one OpenLane run")
    parser.add_argument("--runs-dir", default=DEFAULT_RUNS_DIR,
                        help="directory holding the OpenLane run directories")
    parser.add_argument("--run", default=None,
                        help="use this run directory instead of the newest one")
    parser.add_argument("--design", default="",
                        help="design name; read from resolved.json when it is empty")
    parser.add_argument("--out", default=DEFAULT_OUT_DIR,
                        help="directory receiving the exported tree")
    parser.add_argument("--host-out", default=None,
                        help="write to this host directory instead of --out. A Windows path "
                             "is translated with wslpath when this process runs inside WSL")
    parser.add_argument("--desktop", action="store_true",
                        help="write to Desktop/LingoLOGOS_Export of the Windows host")
    parser.add_argument("--topology", default=DEFAULT_TOPOLOGY,
                        help="topology file whose instances and routing matrix shape the board")
    parser.add_argument("--allow-missing", action="store_true",
                        help="exit zero even when an artifact is missing")
    parser.add_argument("--dry-run", action="store_true",
                        help="print what would be written and write nothing")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.desktop:
        target = desktop_dir()
        if not target:
            print("[EXPORT] no Windows desktop is reachable from this process")
            return 1
        args.out = target
    if args.host_out:
        if re.match(r"^[A-Za-z]:[\\/]", args.host_out) and is_wsl():
            translated = translate_host_path(args.host_out)
            if not translated:
                print("[EXPORT] wslpath could not translate %s" % args.host_out)
                return 1
            args.out = translated
        else:
            args.out = args.host_out
    return Exporter(args).run()


if __name__ == "__main__":
    sys.exit(main(None))
