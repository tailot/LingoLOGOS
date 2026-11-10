# LingoLOGOS ASIC build instructions

Everything in `asic_chip/` builds from one place: `asic_chip/scripts/lingologos_compiler.py`.
Edit an artifact, rerun the generator, and the generated RTL, the golden vector table, the
core self test table and the manifest all move together. Hand editing a generated file is
the one workflow this flow is designed to make impossible: `--check` compares bytes and
fails.

All commands are written to run **from the repository root**, because the file lists in
`asic_chip/config.json`, in the Makefile and in the C++ harness are relative to it.

## 1. What you need, per step

| Step | Tool | Version the flow was written against | Needed for |
|---|---|---|---|
| Generate RTL and vectors | CPython | 3.10, 3.11, 3.12 or 3.14, standard library only | every other step |
| Run the gate suite | pytest | 9.1.1 | release checks, CI |
| Lint and simulate | Verilator | 5.006 or later | the C++ harness, the SystemVerilog testbench, latch inference |
| Second simulator opinion | Icarus Verilog | 11 or later, `-g2012` | `sim-iverilog` |
| Logic synthesis checks | Yosys | 0.38 or later | optional, memory inference and cell mapping only |
| Place and route | OpenLane | NOT INSTALLED here; OpenLane 1 runs as `flow.tcl`, OpenLane 2 as `openlane` | GDSII, timing, power, area |
| PDK | SkyWater `sky130A`, `sky130_fd_sc_hd` | OpenLane managed install | the same |

In the checkout where this tree was written CPython 3.14.5, pytest 9.1.1 and a reachable
Docker engine were present, and Verilator 5.054 ran in the image
`verilator/verilator:latest`, as described at the end of section 4. Verilator, Icarus,
Yosys, `make` and OpenLane are not installed on the host itself. The lint and simulation
steps below were therefore run in the container, not on the host; the Icarus path and the
layout step of section 5 were **not run** anywhere.
`asic_chip/doc/verification_report.md` records exactly which claims that limits.

## 2. Step 1, regenerate the datapath

```sh
python asic_chip/scripts/lingologos_compiler.py
```

Expected tail of the output:

```
Final gates: { "H1_weight_mesh_fits": true, ... }
FINAL VERDICT: PASS
```

Files it writes, all committed:

| Output | What it is |
|---|---|
| `asic_chip/rtl/generated/weight_rom_mesh.sv` | the 4 by 4 constant product netlist, shift-add trees only |
| `asic_chip/rtl/generated/gf_automaton_1cycle.sv` | the one cycle GF(2^8) state table, 1020 arms |
| `asic_chip/rtl/generated/smt_evaluator.sv` | the clause network, wires only |
| `asic_chip/rtl/generated/MANIFEST.json` | digests of the artifacts and of every generated file, plus the gate measurements |
| `asic_chip/sim/golden_vectors.mem` | 304 stimulus rows with expected results |
| `asic_chip/sim/core_self_test.mem` | the 17 key and value pairs the core self test checks |

Useful flags:

| Flag | Effect |
|---|---|
| `--check` | regenerate in memory, compare the bytes with what is committed, and additionally check `RTL_matches_artifacts`. Exits non zero on any difference |
| `--out-dir DIR` | write the generated modules somewhere else, for experiment |
| `--sim-dir DIR` | write the two memory tables somewhere else |
| `--lenient` | let a failing gate print its report instead of stopping the run; the final verdict is still FAIL |

A gate that fails means the artifacts describe a datapath that does not fit the chip
contract, for example a weight whose shift-add tree would overflow the 20 bit
accumulator. Fix the artifact, do not relax the gate.

## 3. Step 2, the Python gate suite

```sh
python -m pytest -v tests/
```

This is the check `AGENTS.md` requires before any change is considered done. The ASIC
gates live in `tests/test_asic_compiler.py` and `tests/test_asic_core_firmware.py` and run
on plain Python with no HDL tool, so they also run in CI on Python 3.10, 3.11 and 3.12.
They read the committed files and recompute the expected values from
`tests/asic_reference_model.py`, which does not import the generator.

The generator itself is also reachable through make, if `make` is installed:

```sh
make -f asic_chip/Makefile gen
make -f asic_chip/Makefile check
make -f asic_chip/Makefile pytest
```

## 4. Step 3, simulation with Verilator

Lint only, which is also the latch and width inference check:

```sh
verilator --lint-only -O2 --timescale 1ns/1ps -Wall -Wno-UNSIGNED -Wno-UNUSED -Wno-WIDTHTRUNC \
  -DLL_DEBUG_PORTS --top-module lingologos_top \
  asic_chip/rtl/generated/weight_rom_mesh.sv asic_chip/rtl/generated/gf_automaton_1cycle.sv \
  asic_chip/rtl/generated/smt_evaluator.sv asic_chip/rtl/core/rv32e_regfile.sv \
  asic_chip/rtl/core/rv32e_core.sv asic_chip/rtl/top/lingologos_top.sv
```

The C++ harness of section 8 of the specification, the one that replays
`golden_vectors.mem` and runs the core self test:

```sh
verilator --cc --exe --build -O2 --timescale 1ns/1ps -Wall -Wno-UNSIGNED -Wno-UNUSED \
  -Wno-WIDTHTRUNC -DLL_DEBUG_PORTS --trace --top-module lingologos_top \
  -Mdir asic_chip/sim/obj_dir \
  asic_chip/rtl/generated/weight_rom_mesh.sv asic_chip/rtl/generated/gf_automaton_1cycle.sv \
  asic_chip/rtl/generated/smt_evaluator.sv asic_chip/rtl/core/rv32e_regfile.sv \
  asic_chip/rtl/core/rv32e_core.sv asic_chip/rtl/top/lingologos_top.sv \
  asic_chip/sim/sim_main.cpp
asic_chip/sim/obj_dir/Vlingologos_top asic_chip/sim/golden_vectors.mem asic_chip/sim/core_self_test.mem asic_chip/sim/lingologos.vcd
```

The harness takes three arguments, in that order: golden vector file, self test file, VCD
path. It prints one line per vector row group, the self test register readback, and a
final `PASS` or `FAIL` line, with a non zero exit status on failure. It allows 400 cycles
for the self test; the run reports 91 and the reference model counts 89.

The same checks through the SystemVerilog testbench, when a C++ build is not wanted:

```sh
verilator --binary --timing -O2 --timescale 1ns/1ps -Wall -Wno-UNSIGNED -Wno-UNUSED \
  -Wno-WIDTHTRUNC -DLL_DEBUG_PORTS --top-module lingologos_testbench \
  -Mdir asic_chip/sim/obj_dir_tb \
  asic_chip/rtl/generated/weight_rom_mesh.sv asic_chip/rtl/generated/gf_automaton_1cycle.sv \
  asic_chip/rtl/generated/smt_evaluator.sv asic_chip/rtl/core/rv32e_regfile.sv \
  asic_chip/rtl/core/rv32e_core.sv asic_chip/rtl/top/lingologos_top.sv \
  asic_chip/sim/testbench.sv
asic_chip/sim/obj_dir_tb/Vlingologos_testbench
```

With Icarus instead:

```sh
iverilog -g2012 -D LL_DEBUG_PORTS -s lingologos_testbench \
  -o asic_chip/sim/obj_dir_vvp/lingologos.vvp \
  asic_chip/rtl/generated/weight_rom_mesh.sv asic_chip/rtl/generated/gf_automaton_1cycle.sv \
  asic_chip/rtl/generated/smt_evaluator.sv asic_chip/rtl/core/rv32e_regfile.sv \
  asic_chip/rtl/core/rv32e_core.sv asic_chip/rtl/top/lingologos_top.sv \
  asic_chip/sim/testbench.sv
vvp asic_chip/sim/obj_dir_vvp/lingologos.vvp
```

`LL_DEBUG_PORTS` exists only for simulation. It exposes the `dbg_` group, 12 ports and 302
bits, so the harness can see inside the pipeline. Synthesis and OpenLane never define it,
so those ports never become pads; `asic_chip/openlane/pin_order.cfg` has no row for them.

Build products `asic_chip/sim/obj_dir`, `obj_dir_tb`, `obj_dir_vvp`, `*.vcd` and `*.vvp`
are ignored by `.gitignore` and removed by `make -f asic_chip/Makefile clean`.

### When Verilator is not on the host, use the official image

The three commands above ran in `verilator/verilator:latest`, the image this tree was checked
in: `Verilator 5.054 2026-11-01 rev v5.054`, amd64, 252,551,017 compressed bytes. Two facts
about that image save time. Its working directory is `/work`, so mounting the checkout there
keeps every relative path in the file lists valid. Its entrypoint is
`/usr/local/bin/verilator-wrap.sh`, which already invokes the tool, so the argument list
after the image name starts at the flags: writing the word `verilator` there makes the tool
look for a module named `verilator` and stop with `Cannot find file containing module`.

Lint, from the repository root:

```sh
docker run --rm -v "$PWD:/work" -w /work verilator/verilator:latest \
  --lint-only -O2 --timescale 1ns/1ps -Wall -Wno-UNSIGNED -Wno-UNUSED -Wno-WIDTHTRUNC \
  -DLL_DEBUG_PORTS --top-module lingologos_top \
  asic_chip/rtl/generated/weight_rom_mesh.sv asic_chip/rtl/generated/gf_automaton_1cycle.sv \
  asic_chip/rtl/generated/smt_evaluator.sv asic_chip/rtl/core/rv32e_regfile.sv \
  asic_chip/rtl/core/rv32e_core.sv asic_chip/rtl/top/lingologos_top.sv
```

On Windows PowerShell, pass the quoted Windows path of the checkout instead of `"$PWD"`, and
write the command on one line instead of using a backslash continuation.

The image also carries `make` and a C++ compiler, so the make targets run there unchanged.
That is the shortest route, and it is the one used for the recorded results:

```sh
docker run --rm -v "$PWD:/work" -w /work --entrypoint sh verilator/verilator:latest \
  -c 'make -f asic_chip/Makefile lint && make -f asic_chip/Makefile sim-tb && make -f asic_chip/Makefile sim-verilator && make -f asic_chip/Makefile sim-topology'
```

To start the built model instead of the tool, replace the entrypoint:

```sh
docker run --rm -v "$PWD:/work" -w /work \
  --entrypoint ./asic_chip/sim/obj_dir_tb/Vlingologos_testbench verilator/verilator:latest
```

The build directories and the VCD written from inside the container belong to the invoking
user on Linux; on other platforms `make -f asic_chip/Makefile clean` removes them.

## 5. Step 4, place and route with OpenLane

The declared entry point is:

```sh
sh asic_chip/scripts/build_gds.sh
```

What it does, in order:

1. checks that a configuration file exists, `asic_chip/openlane/config.bringup.json` by
   default and `asic_chip/openlane/config.json` when `LL_OPENLANE_CONFIG` names it;
2. runs `python3 asic_chip/scripts/lingologos_compiler.py --check`, so a stale
   `rtl/generated` directory stops the flow before a layout is attempted;
3. looks for `flow.tcl` on `PATH`, then at `$OPENLANE_ROOT/flow.tcl` for OpenLane 1, then for
   `openlane` in `$LL_OPENLANE_BIN`, in `~/openlane-venv/bin` and on `PATH` for OpenLane 2; if
   none is found it prints that a place and route has already run in this checkout but no
   OpenLane is reachable now, and exits with status 127;
4. runs the flow, keeping the exit status the tool returned instead of letting the script die
   where the flow stopped;
5. prints the newest run directory, the GDSII files below it and the reports worth opening, and
   exits with the status of the flow, so a caller sees the same number the tool returned.

Sourcing `asic_chip/env.sh` sets the variables the flow expects (`OPENLANE_ROOT`,
`OPENLANE_VERSION`, `PDK_ROOT`, `PDK=sky130A`, `STD_CELL_LIBRARY=sky130_fd_sc_hd`) and defines
the `ll-gen` and `ll-check` aliases. It installs nothing, and it is a POSIX shell script,
so on Windows source it inside WSL or a container rather than in PowerShell.

The OpenLane 2 route installs on the host with pip and runs its steps inside a container.
The upstream page for that route, under Windows, asks for WSL 2 plus Docker Desktop with WSL
integration, a quad core CPU at 2.0 GHz or above, 8 GiB of RAM as the minimum and 16 GiB
recommended. In outline, from a shell inside WSL:

```sh
python3 -m pip install openlane volare
python3 -m openlane --dockerized --smoke-test
```

The smoke test pulls the OpenLane 2 container image, and volare then fetches the sky130A
PDK into the PDK root. Both of those downloads have happened on the machine the RTL was
written on, inside Ubuntu 26.04 on WSL 2 with Docker Desktop, and the sizes they left
behind are the sizes to plan for: the image tagged `ghcr.io/efabless/openlane2:2.3.10`,
which is the one openlane 2.3.10 asked for, occupies 7.39 GB of docker disk usage, and one
sky130 version of open_pdks occupies 2.1 GiB with the default library set, the set the
place and route of this design reads. Asking volare for every library with `-l all` takes
the same version to 16 GiB, because it then also carries the magic, netgen, klayout, vs and
xdrc variants of that commit, and nothing in this flow needs them; the setup script takes
the default set and only adds libraries when `LL_PDK_LIBRARIES` names them. The upstream
figure of 700 MiB to 1.5 GiB per version sits inside the default set, not the large one.

`build_gds.sh` hands a configuration file to whichever entry point it finds: the bring-up
set `asic_chip/openlane/config.bringup.json` by default, and `asic_chip/openlane/config.json`
when `LL_OPENLANE_CONFIG=config.json`. Both carry OpenLane 1 variable names, which OpenLane 2
keeps in its Classic flow, and that mapping has now run three times on 2026-11-09 with openlane
2.3.10. The first attempt accepted every key in the bring-up file and completed the Verilator
lint, the Yosys synthesis, the netlist checkers, the floorplan, the tap and cap rows, the power
distribution network, the global placement and the OpenROAD pin placement, 24 steps of the
Classic flow, before the custom I/O placement step, the 25th, stopped on `pin_order.cfg`. The
next section is about the format that file has to be in. After that file was rewritten the same
command ran all 78 steps in 16 min 59 s and left what the first attempt had not: three GDSII
files, a DRC report at zero violations in Magic and in KLayout, an LVS report saying the
circuits match uniquely, an IR drop report, a static timing report per corner, and a signoff
summary whose antenna line said Failed on 54 pins and 46 nets. That run was constrained to
8.0 ns and exited non zero on deferred setup and hold violations. The third run, with the diode
knobs spelled in OpenLane 2 names instead of the OpenLane 1 strategy number that had silently
disabled them, ran all 78 steps in 12 min 12 s at a 30.0 ns gate and printed `Antenna Passed`,
`LVS Passed` and `DRC Passed` in the manufacturability summary; it still exited non zero, on
deferred setup errors in one of the nine corners and hold errors in seven. The end of this
section says what the gate is now, which knobs decide it and what is still open, and section 2.5
of `asic_chip/doc/verification_report.md` holds the numbers of all three runs.

### The pin plan file, and the grammar that reads it

`asic_chip/openlane/pin_order.cfg` is read by the custom I/O placement step, the one that
runs because `FP_PIN_ORDER_CFG` is set, and that step parses it with the pin constraint
grammar of OpenLane 2 rather than the line oriented format of OpenLane 1. Three consequences
of that, each of them seen as a failure in a real run here rather than read out of a manual:

* the grammar has no comment token. A line that starts with `#` is a side directive, and the
  banner the OpenLane 1 file used, `#W` followed by a dash, stops the step with
  `token recognition error at: '-'` and `An exception occurred: Syntax Error at 1:2` in the
  log of that step;
* the directives are `#W`, `#N`, `#E` and `#S`, meaning left, top, right and bottom, and every
  pattern after one of them belongs to that side until the next directive. A directive written
  `#N_R` fills that side in the opposite order, and the annotations `@bit_major`, `@bus_major`
  and `@min_distance=<value>` choose the sorting along the side and the spacing between pins;
* a pattern is a regular expression which the tool anchors at both ends and matches against the
  **bit level** pin names of the netlist. Those names are `in_vector_0[3]`, not `in_vector_0`,
  because a bus port becomes one pin per bit. The characters the grammar will lex inside a
  pattern, besides letters, digits and underscores, are `\ . + * ? ( ) [ ] |`, and a dash is not
  among them: `[0-9]` is a syntax error there and `\d+` is what works. A pattern that matches no
  pin, and a pin that no pattern matches, both fail the same step.

The pinout gate in `tests/test_asic_compiler.py` rebuilds those bit level names from the RTL and
applies the same anchoring and the same character restriction, so a plan that the step would
reject fails the suite without needing a run. The file in this repository places the 146 user
signals as the pin table in `doc/datasheet_lingologos.md` declares them, 40 bits on the left,
40 on the top, 40 on the right and 26 on the bottom. The step that reads it passed in the
complete run, and the parser of that step was also driven by hand inside the container image
against the names of that netlist: 146 bit level pins seen, 146 placed, none unmatched in either
direction, 40 on each of the left, top and right and 26 on the bottom.

### One command that sets the toolchain up as well

`build_gds.sh` only runs the flow: it checks the configuration file, runs the generator in
`--check` mode, and stops with status 127 when neither `flow.tcl` nor `openlane` is on `PATH`.
Installing the toolchain is a separate step, once per machine, and
`asic_chip/scripts/setup_openlane_flow.sh` is that step plus the flow, written for a POSIX
shell inside a Linux distribution:

```sh
sh asic_chip/scripts/setup_openlane_flow.sh
```

It refuses to start when the Docker engine does not answer, installs the host packages only
when one of them is missing, creates `~/openlane-venv` with `openlane` and `volare` inside,
downloads the sky130 version volare lists first, and then runs
`python3 -m openlane --dockerized --smoke-test`. A PDK version is an open_pdks commit and this
file does not hardcode one: it asks volare which versions it publishes and takes the first one
listed. With the volare 0.20.6 this flow installs, asking is the subcommand
`volare ls-remote --pdk sky130`, which prints one 40 character commit per line, and the download
is `volare fetch --pdk-root "$PDK_ROOT" --pdk sky130 <hash>` followed by `volare enable`
with the same two options. A `-l <set>` goes into the fetch only when `LL_PDK_LIBRARIES` names
a library set, and `all` is the 16 GiB one described above, which this flow does not need. Versions live under `$PDK_ROOT/volare/sky130/versions/<hash>`, and
enabling one symlinks each variant of the family, `sky130A` and `sky130B`, directly under
`$PDK_ROOT`, which is the name the flow checks; the enabled version is recorded in
`$PDK_ROOT/volare/sky130/current`. The spelling some older guides show, `volare --global search`
and `volare --global download`, is not a thing in this version: it answers
`Error: No such option '--global'.` and `Error: No such command 'search'.`, which is why the
script leaves the output of that listing visible instead of discarding it. The interpreter for the virtual
environment is picked rather than taken for granted: `python3.12` is preferred, then 3.11, 3.10
and 3.9, and `python3` last. The reason is the two compiled dependencies of `openlane`, whose
published wheel sets were read from the Python package index: `klayout` 0.29.12 ships manylinux
wheels for cp38 through cp313, `libparse` 0.3.1 only through cp312, so 3.12 is the newest
interpreter the flow installs from wheels. On a newer interpreter the script stops with status
127 and prints the commands that fix it, rather than watching pip compile two C extensions: that
was tried on this machine through the default `python3` and both builds failed, as recorded in
`verification_report.md`. Ubuntu 26.04 ships `python3` at 3.14.4 and its own archive carries no
older interpreter, so there `python3.12` and `python3.12-venv` come from `ppa:deadsnakes/ppa`,
which the script adds with `sudo add-apt-repository` unless `--no-ppa` is given, and
`python3.12-tk` comes along with them. That last package is not a luxury: `openlane` 2.3.10
imports `tkinter` while it starts up, in `openlane/common/tcl.py`, and on this machine that import
was seen failing with `ModuleNotFoundError: No module named 'tkinter'` in an environment where
`openlane` itself was installed correctly. Debian and Ubuntu ship the module apart from the
interpreter, so the script tests `import tkinter` on the interpreter it picked, installs the `tk`
package of that interpreter when the test fails, and stops with status 127 if it still fails. When neither
the distribution archive nor that PPA yields 3.12, the script downloads a standalone build of
CPython 3.12 from the `python-build-standalone` project into
`~/.local/share/lingologos/python` and uses that, a route that needs no sudo and no third party
apt entry. `LL_PYTHON` names an interpreter you obtained yourself instead, and `LL_PYTHON_URL`
replaces the pinned archive the script downloads; `PDK_ROOT` says where the PDK goes, `~/pdk`
unless the environment says otherwise, and `LL_PDK_LIBRARIES` names the library set of the PDK
version that volare should fetch, which is volare's own default unless you name one such as
`all`. That last route was run in this distribution
before it was written down: on such a 3.12.16, `pip install openlane volare` installed
`openlane` 2.3.10, `klayout` 0.29.12 and `libparse` 0.3.1 from wheels and imported all three,
while the same install on the default `python3` failed. A virtual environment left behind on an
interpreter that cannot install these packages is replaced, because it holds nothing but them. The last step is `build_gds.sh`
unchanged, so the stale RTL gate stays in the path. The script is idempotent: a rerun with
everything present skips straight to the flow. Its flags are `--flow-only`, `--skip-smoke`,
`--spec` and `--no-ppa`, and the PowerShell twin forwards them as `-FlowOnly`, `-SkipSmoke`,
`-Spec` and `-NoPpa`.

To run the flow again afterwards, from a shell inside the distribution, `sh asic_chip/scripts/build_gds.sh`
from the repository root is enough, and nothing has to be activated first. That entry point looks for
`openlane` in `$LL_OPENLANE_BIN` first, then in `~/openlane-venv/bin`, then on `PATH`, and one found
in the virtual environment is invoked with `--dockerized`: a pip install of `openlane` carries no tools
of its own, and the container image named in its own `__main__`, `ghcr.io/efabless/openlane2` tagged
with the package version, carries them. That mode mounts your home directory and the PDK root at
identical paths, and the working directory too when it is not under your home, so a checkout that sits
inside the distribution is read there without any translation; a checkout under `/mnt/c` would be read,
but every file operation in it is slow, which is the reason `-Copy` puts the tree in the distribution in
the first place. `PDK_ROOT` names the PDK for the container, and the entry point falls back to `~/pdk`
when the environment does not name it and a `sky130A` sits there.

On Windows the same sequence is driven by one PowerShell command, which additionally starts
Docker Desktop when the engine is not answering and installs an Ubuntu distribution when none
exists, two things a shell inside WSL cannot do. It installs the checkout that holds the file it
is started from, so the directory it is started from does not matter:

```powershell
powershell -ExecutionPolicy Bypass -File asic_chip\scripts\setup_openlane_flow.ps1 -Copy
```

Started instead from a shell inside WSL, through `powershell.exe`, the path of the file needs
quoting or forward slashes, since that shell removes backslashes: written the way a Windows shell
writes it, it arrives as one long name that does not exist. The file stops before copying when the
directory it belongs to is not a checkout, which is how a start from a stray copy shows up.
`-Copy` puts the checkout inside the distribution with `robocopy` addressed to
`\\wsl.localhost\<distro>\<home>\LingoLOGOS`, which is the distribution filesystem and not
`/mnt/c`; a run performs a very large number of small file operations and `/mnt/c` makes every
one of them slow. The working tree is copied as it stands, so files that are not committed yet
come along. `-FromCommit <revision>` copies a snapshot through `git archive` and `tar` instead,
and then anything uncommitted is missing, which is why it is not the default. The path inside
the distribution defaults to `~/LingoLOGOS`. In Docker Desktop, Settings, Resources, WSL
Integration must be enabled for that distribution: the script cannot change that setting, and
without it `docker` inside WSL cannot reach the engine. `-FlowOnly` skips the setup and runs
only the flow, `-Spec` runs the flow at the declared 4.0 ns budget instead of the 32.0 ns
bring-up gate it uses by default, `-SkipSmoke` leaves out the upstream smoke test, which is
worth doing on a machine where the container image and the PDK are already known good, and
`-NoPpa` stops the script from adding the interpreter archive; the `--flow-only`,
`--skip-smoke`, `--spec` and `--no-ppa` spellings do the same on the POSIX side.
Both scripts have now run inside the Ubuntu distribution named above, and every step of them
except the smoke test has completed there: the docker check, the choice of interpreter, the
install of `openlane` and `volare`, and the PDK download that created `~/pdk/sky130A` and its
`sky130B` sibling as links. The flow itself also ran, three times, inside the copy of the tree
that lives in that distribution, that is `/home/<user>/LingoLOGOS`. The attempt that stopped at
the custom I/O placement step left 24 steps of logs under
`asic_chip/openlane/runs/RUN_2026-11-09_16-15-02`, which is why the pin plan section above
exists; the run that followed the rewrite of that file left 78 numbered step directories under
`asic_chip/openlane/runs/RUN_2026-11-09_18-43-18`, 1.2 GB of logs, netlists, extracted
parasitics, reports and three GDSII files; the third complete run left the same shape of
directory under `asic_chip/openlane/runs/RUN_2026-11-09_21-14-43`. Those directories are not
deleted by anything here, and the newest one is what `build_gds.sh` reports at the end of a run.
That copy is the one the flow runs from, and the robocopy route of the PowerShell script adds
files to it rather than replacing it, so a run directory stays where the tool wrote it and a
configuration edited on Windows arrives with the next `-Copy`. The download cost of a first run
measured here is 7.39 GB of docker disk for the container image and 2.1 GiB for one PDK version
with the libraries this flow reads, against the 700 MiB to 1.5 GiB upstream states for a
version. What those runs support, and what they do not, is stated in section 2.5 of
`asic_chip/doc/verification_report.md`; this document does not claim a run that closed signoff,
because no run here has.

Outputs land in `asic_chip/openlane/runs/production_build/` when the OpenLane 1 entry point
ran, because `build_gds.sh` passes `-tag production_build`: the final GDSII, the timing
report at `reports/report.csv`, and the logs. OpenLane 2 is invoked without a tag and names its
own run directory: the runs created `asic_chip/openlane/runs/RUN_2026-11-09_16-15-02`,
`asic_chip/openlane/runs/RUN_2026-11-09_18-43-18` and
`asic_chip/openlane/runs/RUN_2026-11-09_21-14-43`, under the `runs/` directory next to the
configuration file it was handed. `build_gds.sh` prints the newest run directory under that one,
or under a `runs/` in the working directory if that is where the tool put it, together with the
`.gds` files below it, rather than naming a path it has not seen. Inside a run directory, the
steps are numbered in the order they ran, `01-verilator-lint` through `78-misc-gds-summary` in a
complete run, and each one holds its own log; the `resolved.json` at the root is the
configuration the tool settled on, with every default filled in, which is where an ignored or
renamed key becomes visible. The numbering of the later steps moves between runs of the same
flow, because the number of enabled steps depends on the knobs: the run constrained to 8.0 ns
wrote its layout in `56-magic-streamout` and `57-klayout-streamout`, and the run constrained to
30.0 ns wrote it in `59-magic-streamout` and `60-klayout-streamout`. OpenLane 2 writes no
`results/` and no `reports/` tree at the top of that directory, so the files worth opening are
inside the step directories, and the names below are those of the run constrained to 30.0 ns:
`59-magic-streamout/lingologos_top.magic.gds`, `59-magic-streamout/lingologos_top.gds` and
`60-klayout-streamout/lingologos_top.klayout.gds` for the layout, `65-magic-drc` and
`66-klayout-drc` for the two rule checks, `63-klayout-xor/xor.xml` for the cross check between
the two streamouts, `71-netgen-lvs/reports/lvs.netgen.rpt` for connectivity and latch-up,
`57-openroad-stapostpnr/<corner>/` for the nine timing corners, each with `max.rpt`, `min.rpt`,
`summary.rpt`, `clock.rpt`, `skew.max.rpt`, `power.rpt` and the `ws.max.rpt`, `wns.max.rpt`,
`tns.max.rpt`, `ws.min.rpt`, `wns.min.rpt` and `violator_list.rpt` the timing checker reads,
`58-openroad-irdropreport` for the static IR drop,
`48-openroad-checkantennas-1/reports/antenna_summary.rpt` for the antenna ratios before and after
the repair, and at the root of the run the `flow.log`, `warning.log` and `error.log` of the flow
itself plus `77-misc-reportmanufacturability/manufacturability.rpt`, the summary line that names
each signoff check Passed or Failed. `build_gds.sh` prints the paths of the last three of those
after a run, because they are where a failed gate says what it failed on.

The constraint set is selected by `build_gds.sh`, and the default is not the specification.
`config.json` carries the declared 250 MHz budget at 4.0 ns, and no place and route has ever
closed that period on this RTL. Two runs have bounded what the RTL costs: routed against an
8.0 ns gate it missed setup by 4.4409 ns in the worst typical corner and by 15.7205 ns in the
worst slow corner, which put that layout about 12.44 ns from closing in the typical corner and
about 23.72 ns in the slow one, and routed against a 30.0 ns gate the same RTL missed setup in
the slow corner only, by 0.4906 ns, while the typical corners came out with 14.4 to 15.5 ns to
spare. 4.0 ns is therefore far outside what has been routed, and `LL_OPENLANE_CONFIG=config.json`
or `-Spec` is the way to route at it, now with the measured gap in front of whoever does it.
`config.bringup.json` and `lingologos.bringup.sdc` are the same configuration at the period the
flow is held to, and that period is 32.0 ns: it was 8.0 ns for the first two runs of 2026-11-09
and 30.0 ns for the third, whose slow corner needed 30.49 ns, so 32.0 ns leaves a little under
1.5 ns against the rerouting a different period causes. That figure is a choice about what the
flow is held to, not a rating of the part. OpenLane 1 reads only the `config.json` of the design
directory, so there the period is applied as `-override_configs "CLOCK_PERIOD=32.0
SDC=dir::lingologos.bringup.sdc"`, with the figure taken from `LL_BRINGUP_PERIOD`; OpenLane 2 is
handed the bring-up file directly.

A timing violation becomes an error through a corner list, not through a key with the word `QUIT`
in it. `SETUP_VIOLATION_CORNERS` and `HOLD_VIOLATION_CORNERS` are both `["*"]` in the two
configuration files here, so a violation in any of the nine corners of the post route timing step
fails the run. The default OpenLane installs for a sky130 or gf180mcu PDK is
`TIMING_VIOLATION_CORNERS = ["*tt*"]`, set in `openlane/config/pdk_compat.py` line 231, and that is
why the complete run turned its typical corner violations into errors and its far worse slow corner
violations into warnings. Max slew and max cap violations are reported numbers and never gate a run
in OpenLane 2.3.10, whatever the corner. `QUIT_ON_TIMING_VIOLATIONS`, `RUN_CVC` and
`QUIT_ON_LINTER_ERRORS` are out of both configuration files: that version of OpenLane has no first
key at all, removed the step the second one names, and renamed the third to
`ERROR_ON_LINTER_ERRORS`, spelled with that name in both files and left at its default true: the
run that reached the end of the flow printed `Check for Lint errors clear.` and `14 Lint warnings
found.`, and warnings are a separate check, `Checker.LintWarnings`, which reported them without
failing the run.

A hold violation in this flow is repaired before detailed routing and never after it, which is the
shape the third run reported. The last hold repair lives in `openlane/scripts/openroad/rsz_timing_postgrt.tcl`:
that script re-runs the global router, estimates parasitics from that routing, and then calls
`repair_timing -hold -hold_margin $::env(GRT_RESIZER_HOLD_SLACK_MARGIN)`. The margin defaults to
0.05 ns, and the same kind of margin for the earlier resizer steps, `PL_RESIZER_HOLD_SLACK_MARGIN`,
defaults to 0.10 ns, so the repair stops as soon as the slack is barely positive. The third run
showed what that costs: the metrics of the timing step immediately after that resizer read
`timing__hold__ws` of +0.100201 ns with zero violations in `nom_tt_025C_1v80` on global-route
parasitics, while the same corner at signoff, on the SPEF extracted from the final routing, read
-0.058548 ns. Detailed routing and extraction therefore take about 0.16 ns from those paths and
nothing downstream puts it back. Both margins are 0.40 ns in both configuration files now, so the
repair overfixes instead of stopping at zero, and the `Checker.HoldViolations` block of `error.log`
in the next run is where the answer is: it prints `Hold violations found in the following corners:`
and names them, and that block, not `OpenROAD.STAPostPNR`, is what makes a run exit non zero.

The two configuration files also name the antenna repair knobs directly, which is what the failed
antenna check of the complete run was about. `RUN_HEURISTIC_DIODE_INSERTION` is on with
`HEURISTIC_ANTENNA_THRESHOLD` 60, `DIODE_ON_PORTS` is `in`, `GRT_ANTENNA_ITERS` is 5 and
`GRT_ANTENNA_MARGIN` is 25, `RUN_POST_GRT_DESIGN_REPAIR` and `RUN_POST_GRT_RESIZER_TIMING` are on,
and the floorplan leaves room to insert all of that at `FP_CORE_UTIL` 40 and `PL_TARGET_DENSITY`
0.45. Neither file carries `DIODE_INSERTION_STRATEGY` any more. OpenLane 2 accepts that OpenLane 1
key, deletes it and derives the native knobs from it, in `openlane/config/config.py` lines 918 to
943, and for value 3 the derivation turns off the heuristic diode insertion and the diodes on ports
that the same file asks for a few lines away. The complete run had exactly that setting: two diode
steps were skipped, `flow.log` says `Gating variable ... set to 'False'` twice, the repair that did
run used its default 3 iterations and 10 percent margin, and the signoff summary reported 54 antenna
pin violations on 46 nets. The run that followed the removal of that key reported `Antenna Passed`,
with an empty violation table in `48-openroad-checkantennas-1/reports/antenna_summary.rpt` and zero
design rule errors left by the detailed router. The cost of that fix is measured rather than
estimated: 16898 instances of class `antenna_cell`, a die area of 393012 square micrometres against
the 350447 of the previous run, standard cell area 204806 against 169306, and 660126 micrometres of
wire over 134302 vias in the final route, which closed in eight detailed routing iterations from an
initial 10890 errors. `HEURISTIC_ANTENNA_THRESHOLD` 60 is the aggressive end of that trade and 90 is
the looser end; the report of the run decides, not this file.

`asic_chip/openlane/lingologos.sdc` declares
the 4.0 ns budget, the 0.15 ns setup and 0.05 ns hold uncertainty, an 0.80 ns input and
0.30 ns output delay, a `sky130_fd_sc_hd__buf_4` driver and 0.030 pF and 0.025 pF loads;
`lingologos.bringup.sdc` repeats that file with the period at 32.0 ns and carries comments saying
that its input and output delays are absolute values which do not scale with the period, and that
its input delay minimum is smaller than the clock insertion delay the flow measured, which is where
its hold violations came from. Both files name their constraint file three times, as `SDC`,
`PNR_SDC_FILE` and `SIGNOFF_SDC_FILE`, because OpenLane 2 otherwise times the placement and routing
steps against generic constraints and says so in the log. The slack OpenSTA reported for the routed
netlist has replaced the predicted figures in `doc/datasheet_lingologos.md`; the cells that stay NOT
MEASURED there are power as a rating and everything that would have to be measured at 4.0 ns.

## 6. Make targets

```sh
make -f asic_chip/Makefile help
```

| Target | Needs | Does |
|---|---|---|
| `help` | nothing | prints the target list |
| `gen` | Python | runs `asic_chip/scripts/lingologos_compiler.py` |
| `check` | Python | runs the same with `--check` |
| `sim-verilator` | Verilator, a C++ compiler | builds and runs the C++ harness, dumps `asic_chip/sim/lingologos.vcd` |
| `sim-tb` | Verilator | builds and runs `sim/testbench.sv` |
| `sim-topology` | Verilator | builds and runs `sim/topology_testbench.sv`: four dies of the topology switch, wired as `config/topology.json` declares |
| `sim-iverilog` | Icarus Verilog | compiles and runs the same testbench with `vvp` |
| `lint` | Verilator | lint only, no simulation. It runs `lint-topology` as well, so the topology switch is covered by this target |
| `lint-topology` | Verilator | lints the topology switch alone, and then with its bench on top |
| `pytest` | Python, pytest | `python -m pytest -v tests/` |
| `openlane` | Python, OpenLane, PDK | `sh asic_chip/scripts/build_gds.sh`, at the 32.0 ns bring-up gate |
| `openlane-spec` | Python, OpenLane, PDK | the same script with `LL_OPENLANE_CONFIG=config.json`, at the declared 4.0 ns budget |
| `setup-openlane` | Docker engine, network, sudo | `sh asic_chip/scripts/setup_openlane_flow.sh`, then the same flow |
| `clean` | nothing | removes `obj_dir` variants, VCD, VVP and OpenLane run directories |

A target whose tool is missing prints a one line message naming the tool, points at this
file, and exits with status 127. That is deliberate: a missing tool must be a visible
failure, never a silent skip. In the checkout this tree was written in, the Verilator targets
ran inside the container image described at the end of section 4, `lint-topology` and
`sim-topology` among them on 2026-11-10, with the output of those two recorded in section 2.4.1
of `asic_chip/doc/verification_report.md`; `sim-iverilog` was never run
at all, and nothing in that paragraph ran on the host: the `openlane` and `setup-openlane` recipes call the two scripts
of section 5, and those scripts have both run, three times for the flow, which is where the
layout numbers of the datasheet come from. `asic_chip/doc/verification_report.md` states which
claim each of those facts supports.

## 7. Platform notes

- PowerShell has no `make` here. Run the generator and pytest directly, or use WSL, Git
  Bash or a container for the make targets, which are POSIX shell recipes. The Verilator
  image carries its own `make` and C++ compiler, so `lint`, `sim-tb`, `sim-verilator`,
  `lint-topology` and `sim-topology` run
  there unchanged, as shown at the end of section 4.
- The generator writes LF line endings explicitly and UTF-8 without a byte order mark, on
  every platform, so the digests in `MANIFEST.json` match `sha256sum` output on Windows
  and Linux alike. If a checkout is configured to convert line endings on checkout, set it
  to keep LF for `asic_chip/rtl/generated`, `asic_chip/sim` and the SDC, or the digest
  gate will fail with an apparent content difference.
- The Python flow needs no third party package beyond pytest: the generator uses only the
  standard library, so `requirements.txt` does not grow for the ASIC tree.
- CI runs `pytest -v tests/` on Python 3.10, 3.11 and 3.12, which includes the ASIC gates
  because they need no HDL tool. Verilator and OpenLane steps are not in CI.

## 8. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `--check` exits non zero listing a generated file | the artifact changed and the RTL is stale, or a generated file was hand edited. Run the generator without `--check`, inspect the diff, then rerun `--check` |
| `MANIFEST.json` digest does not match the file | line endings were converted by a checkout setting, see the platform note above |
| generator prints a failing gate such as `H1_weight_mesh_fits: false` | an artifact value exceeds the datapath contract, for example a weight whose shift-add tree would overflow the 20 bit accumulator. Fix the artifact |
| `verilator not found` and exit 127 | Verilator is not on the host. Use the container route at the end of section 4; a simulation claim in the documentation only applies where the tool actually ran |
| `Cannot find file containing module: 'verilator'` from a container run | the image entrypoint already invokes the tool, so the command line repeated the word `verilator`. Start the argument list at the flags |
| the container writes build products the host cannot delete | they belong to the user the container ran as. `make -f asic_chip/Makefile clean` from the same route removes them |
| Verilator reports `WIDTHTRUNC` on the mesh wires | expected and disabled by `-Wno-WIDTHTRUNC`: the trees are sized to the 20 bit accumulator and the truncation is the declared wrap |
| Verilator infers a latch in `rtl/generated` | impossible by construction, those modules contain only `wire` declarations and continuous assignments; if it happens, the generator changed, and the gate suite will have failed first |
| `iverilog` reports `sorry: SystemVerilog ... not supported` | use a Verilator run instead, Icarus supports only part of SystemVerilog 2012 |
| the custom I/O placement step of OpenLane 2 exits 65 with `token recognition error at: '-'` and `An exception occurred: Syntax Error at 1:2` | `openlane/pin_order.cfg` was written the way OpenLane 1 reads it, and OpenLane 2 parses that file with a grammar that has no comment token. `#W`, `#N`, `#E`, `#S` directives and anchored patterns over bit level pin names are what it wants, and section 5 explains it; the first run here failed exactly this way and the file in this repository is the fix |
| the same step exits 65 with a message naming a pin | a pin of the netlist is in no pattern, or a pattern matches no pin. The step says which, and the pinout gate in `tests/test_asic_compiler.py` makes the same match against the RTL ports without a run |
| the flow warns `'PNR_SDC_FILE' is not defined. Using generic fallback SDC`, and the same for `SIGNOFF_SDC_FILE` | OpenLane 2 wants its own names for the place and route and signoff constraint files, and a configuration written with the OpenLane 1 `SDC` key leaves both undefined. Both name the same file as `SDC` in both configurations here; without them the placement and routing steps time the design against generic constraints instead of this one |
| the log carries `'KEY' has been removed: ...`, `Key 'KEY' provided is unused by the current flow.` or `An unknown key 'KEY' was provided.` | the configuration names something this OpenLane does not read, and the flow carries on without it. `openlane/config/removals.py` lists the removed keys with the reason upstream gave; `openlane/config/pdk_compat.py` is where the PDK compatibility layer sets keys such as `TIMING_VIOLATION_CORNERS`. The `resolved.json` of a run shows the value the tool settled on for every key, which is where a name that does nothing stops being a guess |
| an OpenLane 1 knob in a configuration has the opposite effect of what the file says | some legacy keys are not renamed but recompiled: `DIODE_INSERTION_STRATEGY` is deleted and the native antenna knobs are derived from it in `openlane/config/config.py` lines 918 to 943, and value 3 sets `RUN_HEURISTIC_DIODE_INSERTION` false and `DIODE_ON_PORTS` to `none`. `flow.log` prints `Gating variable ... set to 'False'` for every step that was skipped for that reason. Write the native knobs and leave the legacy one out, which is what both configuration files here do |
| the flow finishes and the summary says `Antenna Failed`, with a count of pin violations on nets | nothing crashed: a manufacturability report named `NN-misc-reportmanufacturability/manufacturability.rpt` is a signoff table and one of its lines is a verdict, and `NN-openroad-checkantennas-1/reports/antenna_summary.rpt` names the nets and their ratios before and after the repair. The step numbers differ between runs, because they count how many steps ran: the run constrained to 8.0 ns had them at 74 and 45, the run constrained to 30.0 ns at 77 and 48. The knobs that change the outcome are `RUN_HEURISTIC_DIODE_INSERTION`, `HEURISTIC_ANTENNA_THRESHOLD`, `DIODE_ON_PORTS`, `GRT_ANTENNA_ITERS` and `GRT_ANTENNA_MARGIN`, with the floorplan density knobs deciding whether there is room to insert the diodes. The run of 2026-11-09 constrained to 8.0 ns failed on 54 pins and 46 nets with the legacy strategy key in the file; the run that followed the removal of that key printed `Antenna Passed` with an empty table, and the section above says what replaced the key and what the diodes cost in area |
| the flow runs all its steps, prints `ERROR ... One or more deferred errors were encountered` and `OpenLane will now quit.`, and exits non zero with the reports already on disk | that is the shape of a signoff failure in OpenLane 2, not of a crash: `Checker.SetupViolations`, `Checker.HoldViolations` and the antenna checker raise a deferred error at the end so the run still writes its GDSII, its DRC and its LVS reports. Which timing corners count as an error is the corner list, `SETUP_VIOLATION_CORNERS` and `HOLD_VIOLATION_CORNERS`, and the list OpenLane sets for a sky130 PDK is `["*tt*"]` unless the configuration says otherwise; with the list at `["*"]` the third run reported `Setup violations found in the following corners:` naming one corner and `Hold violations found in the following corners:` naming seven, and the same corners named in `warning.log` as warnings are the corners the list excludes. Read `error.log` and `warning.log` at the root of the run directory, and inside `NN-openroad-stapostpnr/<corner>/` read `violator_list.rpt` for the offending paths with their slack, `ws.max.rpt`, `wns.max.rpt` and `tns.max.rpt` for the setup numbers and `ws.min.rpt`, `wns.min.rpt` and `tns.min.rpt` for the hold ones; `summary.rpt` in that directory is the per corner table. `build_gds.sh` prints those paths and returns the same exit status |
| a complete run reports hold violations and the setup violations are gone | that is what the third run did, and it is the input delay of the constraint file against the clock insertion delay of the tree, not a broken register. In `clock.rpt` of a corner that run measured 2.1845 ns of insertion delay at a register in `max_ss_100C_1v60` while the constraint file declares a minimum external input delay of 0.30 ns, so a path from a port can arrive before the capturing edge. Every violator in that run was spelled `[hold in-reg]` in `violator_list.rpt`, an input port to a register, 10 of them at worst slack -0.1540 ns in the slow corner and one at -0.0024 ns in the fast corner; the run constrained to 8.0 ns had 41 of the same shape. Two things answer it, and both are in the configuration now: `RUN_POST_GRT_RESIZER_TIMING` is on, and `GRT_RESIZER_HOLD_SLACK_MARGIN` and `PL_RESIZER_HOLD_SLACK_MARGIN` are 0.40 ns instead of the installed 0.05 and 0.10 ns, because the resizer only runs before detailed routing and its margin decides how much of the routing degradation it leaves behind. If the overfix does not close them, widening that input delay minimum is the honest change, and it changes an assumption about the board that the datasheet has to record |
| the log fills with warnings that look alarming and the reports are fine | four of them are known and benign on this design: `[STA-0441] set_input_delay relative to a clock defined on the same port/pin not allowed`, once per constrained port in every step that re-reads the constraints, 43 of them in the run constrained to 8.0 ns and 56 in the one constrained to 30.0 ns, because the constraint file puts an input delay on the clock port itself and OpenLane base constraint files do the same; `[GRT-0097] No global routing found for nets`, which the detailed router resolves; `[DRT-0349] LEF58_ENCLOSURE ... Skipping for layer mcon`, a rule the writer declines for one contact layer; and `[DRT-0120] Large net net301 has 162 pins`, the clock net, which in the third run named a net of 112 pins and said `and 25 similar warnings`. `Checker.WireLength skipped: no threshold configured` and `'VSRC_LOC_FILES' was not given a value` say a check has no threshold or no source location, not that it failed, and so do the `No max slew violations found` and `No max cap violations found` lines of the signoff checker: the third run exceeded the slew limit in five corners and the capacitance limit in three, reported them as warnings, and gates a run on those only if `MAX_SLEW_VIOLATION_CORNERS` or `MAX_CAP_VIOLATION_CORNERS` names a corner, which neither configuration file does |
| the `resolved.json` of a run names a PDK path under `~/.volare` while the setup installed `~/pdk/sky130A` | the container was handed no `PDK_ROOT`, so openlane used the volare default root. Both roots on this machine hold a sky130 version, at different open_pdks commits, and all three runs of 2026-11-09 resolved under `~/.volare`, at the sky130 version `0fe599b2afb6708d281543108caf8310912f54af`. Export `PDK_ROOT` when the installed one must be the one used, and read the `PDK_ROOT` and `PDK` values of `resolved.json` to see which one a run actually used |
| `flow.tcl` found but the PDK is missing | run the PDK install step of your OpenLane version, or point `PDK_ROOT` at an existing sky130A install |
| Docker client present, engine unreachable | start the daemon; nothing in this flow runs a container until it answers. `powershell -ExecutionPolicy Bypass -File asic_chip\scripts\setup_openlane_flow.ps1` starts it and reports what is missing |
| `docker` inside WSL reports `could not be found in this WSL 2 distro` | Docker Desktop is running but its WSL integration is off for that distribution. Settings, Resources, WSL Integration, enable it; no script can change that setting for you |
| `pip install openlane volare` ends in `Failed to build klayout libparse` and `error: failed-wheel-build-for-install` | the virtual environment sits on an interpreter newer than the published wheels of those two packages reach: `klayout` 0.29.12 stops at cp313, `libparse` 0.3.1 at cp312, and the source distributions of both failed to compile here, `klayout` on missing `zlib.h`, `curl/curl.h`, `png.h` and `expat.h`, `libparse` on `make: *** No rule to make target 'patch'`. Get `python3.12`, point `LL_PYTHON` at it, and rerun |
| `apt-get install -y python3.12` reports `Unable to locate package python3.12` | the distribution ships only a newer `python3` and its own archive has no older interpreter, which is the state of Ubuntu 26.04. `setup_openlane_flow.sh` then adds `ppa:deadsnakes/ppa` and retries; `--no-ppa` skips that repository and expects `LL_PYTHON` |
| the copy into the distribution looks like a failure | robocopy statuses 1 and 2 mean files were copied and extra files exist, both normal. The script stops only at a status of 8 or more |
| `openlane --help`, the smoke test or the flow stops with `ModuleNotFoundError: No module named 'tkinter'` | `openlane` 2.3.10 imports `tkinter` at startup, in `openlane/common/tcl.py`, and Debian and Ubuntu ship that module apart from the interpreter. Install the `tk` package of the interpreter the virtual environment sits on, `python3.12-tk` here; `setup_openlane_flow.sh` checks the import and installs it, then stops with status 127 if it is still absent |
| `[FAIL] volare listed no sky130 version` with `Error: No such option '--global'.` or `Error: No such command 'search'.` printed above it | that is the OpenLane 1 era spelling of the version listing, which volare 0.20 does not have. The subcommands are `ls-remote`, `fetch` and `enable`, each taking `--pdk-root` and `--pdk` itself, and `setup_openlane_flow.sh` now uses those. When the listing really is empty and volare says nothing else, the network or a GitHub request limit is the cause |
| `build_gds.sh` ends with `no flow.tcl, and no openlane in $LL_OPENLANE_BIN, in $HOME/openlane-venv/bin or on PATH` and exit 127 | the tool is installed somewhere the entry point does not look. It tries `$LL_OPENLANE_BIN`, then `~/openlane-venv/bin/openlane`, then `PATH`, so an `openlane` living in another virtual environment is reached with `LL_OPENLANE_BIN=/path/to/venv/bin/openlane sh asic_chip/scripts/build_gds.sh` |
