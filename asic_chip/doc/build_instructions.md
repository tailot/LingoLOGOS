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
  -c 'make -f asic_chip/Makefile lint && make -f asic_chip/Makefile sim-tb && make -f asic_chip/Makefile sim-verilator'
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

1. checks that `asic_chip/openlane/config.json` exists;
2. runs `python3 asic_chip/scripts/lingologos_compiler.py --check`, so a stale
   `rtl/generated` directory stops the flow before a layout is attempted;
3. looks for `flow.tcl` on `PATH`, then `$OPENLANE_ROOT/flow.tcl`, then `openlane` for
   OpenLane 2; if none is found it prints that no place and route has been run and exits
   with status 127.

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
PDK into the PDK root. The same documentation puts a PDK build at 700 MiB to 1.5 GiB per
version. Tags published for the image at the time of writing are
`ghcr.io/efabless/openlane2:2.0.4-amd64` and `ghcr.io/efabless/openlane2:2.0.3`; the
archived OpenLane 1 image `efabless/openlane:superstable` is 629,969,982 bytes compressed on
the amd64 channel, according to the registry. None of these downloads has been performed in
this checkout, so nothing here asserts that the flow completes.

`build_gds.sh` hands `asic_chip/openlane/config.json` to whichever entry point it finds.
That file carries OpenLane 1 variable names, which OpenLane 2 keeps in its Classic flow; the
mapping to the native OpenLane 2 spellings is documented upstream but has never been
exercised on this design, and a native OpenLane 2 flow would want its own configuration
file. Read `config.json` as written for the OpenLane 1 naming.

Outputs land in `asic_chip/openlane/runs/production_build/`: the final GDSII, the timing
report at `reports/report.csv`, and the logs. `asic_chip/openlane/lingologos.sdc` declares
the 4.0 ns budget, the 0.15 ns setup and 0.05 ns hold uncertainty, an 0.80 ns input and
0.30 ns output delay, a `sky130_fd_sc_hd__buf_4` driver and 0.030 pF and 0.025 pF loads.
Those numbers are a budget until OpenSTA reports a slack for the routed netlist, at which
point the datasheet cells marked NOT MEASURED get replaced by the report values.

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
| `sim-iverilog` | Icarus Verilog | compiles and runs the same testbench with `vvp` |
| `lint` | Verilator | lint only, no simulation |
| `pytest` | Python, pytest | `python -m pytest -v tests/` |
| `openlane` | Python, OpenLane, PDK | `sh asic_chip/scripts/build_gds.sh` |
| `clean` | nothing | removes `obj_dir` variants, VCD, VVP and OpenLane run directories |

A target whose tool is missing prints a one line message naming the tool, points at this
file, and exits with status 127. That is deliberate: a missing tool must be a visible
failure, never a silent skip. In the checkout this tree was written in, the Verilator targets
ran inside the container image described at the end of section 4; `sim-iverilog` and
`openlane` were never run at all, and `asic_chip/doc/verification_report.md` states which
claim each of those facts supports.

## 7. Platform notes

- PowerShell has no `make` here. Run the generator and pytest directly, or use WSL, Git
  Bash or a container for the make targets, which are POSIX shell recipes. The Verilator
  image carries its own `make` and C++ compiler, so `lint`, `sim-tb` and `sim-verilator` run
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
| OpenLane reports `PIN_NOT_FOUND` or an unpinned port | `openlane/pin_order.cfg` and the top level port list disagree; the pinout gate in `tests/test_asic_compiler.py` fails too, fix the plan |
| `flow.tcl` found but the PDK is missing | run the PDK install step of your OpenLane version, or point `PDK_ROOT` at an existing sky130A install |
| Docker client present, engine unreachable | start the daemon; nothing in this flow runs a container until it answers |
