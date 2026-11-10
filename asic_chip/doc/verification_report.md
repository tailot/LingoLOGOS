# LingoLOGOS ASIC verification report

Design under verification: `lingologos_top` (`asic_chip/rtl/top/lingologos_top.sv`).
Report date: 2026-11-08, revised 2026-11-09 to record the Verilator runs of section 2.4.
Written in the checkout where the RTL was authored.

## 0. How to read this report

Every number in this report carries one of three labels, and the labels are not
interchangeable.

| Label | Meaning |
|---|---|
| EXECUTED | A program ran in this checkout and produced the quoted output. The command is given. |
| CROSS-CHECKED | Two implementations written independently agree on the value: the Python generator and the Python reference model, and since section 2.4 also the Verilator run of the same RTL |
| NOT RUN / NOT MEASURED | The tool that would produce this number is not installed here, or the run that would produce it has not run in this checkout. Where a run did happen and did not close, the row says which part ran and what it reported. |

This report used to say that nothing in it claims a post-layout, timing or power number. Since the
second and third runs of 2026-11-09 that sentence is no longer true, and section 2.5 is where the
numbers that replaced it come from: area, utilization, wire length, DRC, LVS, static IR drop and static
timing across nine corners, each quoted with the period and the corner it was measured at. None of it is
a characterization. No run has closed signoff, no frequency is claimed for the part, and the power
figure is an estimate under the activity model of a tool. The antenna check failed on the second run and
passed on the third, whose own manufacturability report reads `Antenna Passed`, `LVS Passed` and
`DRC Passed`; what neither of those two runs closed is timing.

## 1. Tool inventory of the machine that produced this checkout

| Tool | Status in this checkout | How it was determined |
|---|---|---|
| CPython | 3.14.5, EXECUTED | `python --version` |
| pytest | 9.1.1, EXECUTED | `python -m pytest --version` |
| git | 2.53.0.windows.1, EXECUTED | `git --version` |
| Docker Engine | EXECUTED, daemon reachable | `docker images` lists the pulled image, `docker run` starts containers |
| Verilator | EXECUTED inside a container, `verilator/verilator:latest`, `Verilator 5.054 2026-11-01 rev v5.054`, amd64 image of 252,551,017 compressed bytes | lint, both simulation paths and the build of both C++ models were run in that image, section 2.4 |
| Verilator on the host PATH | NOT INSTALLED | `Get-Command verilator` finds nothing, so the Verilator targets of the Makefile cannot run as host commands on this machine |
| Icarus Verilog (`iverilog`, `vvp`) | NOT INSTALLED | `Get-Command iverilog` finds nothing, and no Icarus run was attempted here |
| Yosys | NOT INSTALLED | `Get-Command yosys` finds nothing |
| `make` | NOT INSTALLED on the host; present inside the Verilator image | `Get-Command make` finds nothing on the host, so no make target ran here as a host command. The image has `make`, and `lint`, `sim-tb` and `sim-verilator` ran through it there with exit 0, section 2.4 |
| WSL 2 distributions | EXECUTED, `docker-desktop` and `Ubuntu` | `wsl -l -q`. Ubuntu is `Ubuntu 26.04.1 LTS` with `Python 3.14.4` as `python3`, created by the Ubuntu provisioning step of `asic_chip/scripts/setup_openlane_flow.ps1`. Docker Desktop WSL integration is now enabled for it, and inside the distribution `docker info` answers with engine `29.5.3`, which is the failure this tree used to record as blocking. That same PowerShell command copied the working tree into `/home/<user>/LingoLOGOS` and ran `asic_chip/scripts/setup_openlane_flow.sh` inside the distribution, which reached the apt step, installed `build-essential`, `make`, `git`, `python3-pip` and `python3.14-venv`, created `/home/<user>/openlane-venv` on `python3.14.4`, and stopped in `pip install openlane volare`. Rerun once the interpreter ceiling recorded below was in place, the same command printed `[setup] interpreter for the virtual environment: Python 3.12.16 (python3.12)`, rebuilt that environment on it, and got past `pip install openlane volare`; it then stopped at the PDK listing recorded in the volare row further down. The copy is the branch that uses `robocopy` addressed to `\\wsl.localhost\Ubuntu\home\<user>\...`; the `git archive` branch through `wsl -- tar -xf -` fed by `cmd` was exercised separately and returned 0. At that point no PDK, no `/home/<user>/pdk`, no `~/.volare` and no OpenLane run existed there. Later runs of the same command installed `python3.12-tk`, fetched a sky130 version into `/home/<user>/pdk`, pulled the container image and ran the flow of the last step, and section 2.5 records what came of it |
| `openlane`, `klayout` and `libparse` package metadata | EXECUTED over HTTPS against the Python package index | `openlane` 2.3.10 declares `requires-python` `>=3.8,<4` and depends on `klayout >=0.29.0,<0.30.0` and on `libparse >=0.3.1,<1`. The files published for the releases those pins resolve to: `klayout` 0.29.12 ships manylinux and musllinux wheels for cp38 through cp313, none for cp314; `libparse` 0.3.1 ships wheels for cp36 through cp312 and none for cp313 or cp314. The narrower set decides, so 3.12 is the newest interpreter that installs both from wheels. `antlr4-python3-runtime` 4.10, pulled in by `ioplace-parser`, is published as a source distribution only, but it is pure Python. That is why `setup_openlane_flow.sh` prefers `python3.12`, 3.11, 3.10 and 3.9 over `python3`, stops with status 127 on 3.13 or newer, and accepts `LL_PYTHON` as an override |
| Install of `openlane` on CPython 3.12.16 | EXECUTED, PASSED | Same distribution, in `/tmp`, on a standalone CPython 3.12.16 fetched from the `python-build-standalone` release `20261026` rather than on an interpreter from apt: `pip install openlane volare` exited 0 and installed `openlane-2.3.10`, `klayout-0.29.12`, `libparse-0.3.1`, `volare-0.20.6` and 27 further packages, with both compiled pins taken from wheels and only the pure Python `antlr4-python3-runtime-4.9.3` built locally. `/tmp/llv312/bin/python3 -c "import klayout, libparse, openlane; print(openlane.__version__)"` printed `2.3.10`, and the environment put `openlane`, `openlane.steps`, `openlane.env_info`, `openlane.state`, `openlane.config` and `volare` on its bin path. This is what `setup_openlane_flow.sh` now arranges for, and it is still not an OpenLane run: no PDK, no container image and no run directory were involved |
| Which checkout `asic_chip/scripts/setup_openlane_flow.ps1` installs | EXECUTED | Started with `-File` from a directory that is neither the checkout nor related to it, the file printed `[setup] checkout to install: C:\Users\tailot\Desktop\progetti\LingoLOGOS`, which is the directory holding the file and not the current directory, and a copy of itself placed two levels under a temporary directory printed that temporary directory instead. It resolves that root from its own path and stops before touching anything when the root is not a checkout: the copy of itself under a temporary directory printed `[FAIL] '...\ll_probe\stray' is not the checkout this file belongs to, no asic_chip\config.json` and `[FAIL] run the setup_openlane_flow.ps1 of the checkout you want installed`, exit 1. Both probes were run with `-RepoRoot` pointed at a path that does not exist inside the distribution, so they ended at `[FAIL] /nonexistent_ll_probe/asic_chip/scripts/setup_openlane_flow.sh is not present inside 'Ubuntu'`, exit 1, before any package, download or run |
| sky130A PPA package index for this release | EXECUTED over HTTPS | `https://ppa.launchpadcontent.net/deadsnakes/ppa/ubuntu/dists/resolute/main/binary-amd64/Packages.gz`, 200 answered, and its index carries `Package: python3.12` and `Package: python3.12-venv` for `resolute`, source `python3.12`. The plain Ubuntu archive for the same release offers neither, which the failed `apt-get install -y python3.13` and `python3.12` lines of the run above confirm from the other side. With that repository configured, `apt-cache policy python3.12-tk` inside the distribution answers `Candidate: 3.12.16-1+resolute1` from `ppa.launchpadcontent.net/deadsnakes/ppa`, while `apt-cache policy python3-tk` offers only the `3.14.3-0ubuntu2` of the release itself, so the tk module has to come from the same PPA as the interpreter |
| Install of `openlane` on `python3.14.4` | EXECUTED, FAILED, and it is the reason the interpreter is selected | In the Ubuntu distribution, in a virtual environment on `python3.14.4` with pip `26.3`: `pip install openlane volare` resolved, fell back to the source distributions of the two compiled pins, and ended in `Failed to build klayout libparse` with `error: failed-wheel-build-for-install`. `klayout` failed on `fatal error: zlib.h: No such file or directory` and the same for `curl/curl.h`, `png.h` and `expat.h`, then on `cannot find -lcurl`, `-lexpat` and `-lpng` at link time. `libparse` failed before compiling anything, from `make: *** No rule to make target 'patch'`, raised as `subprocess.CalledProcessError: Command '['make', 'patch']' returned non-zero exit status 2`. No published wheel would have avoided either failure for cp314, and none exists for `libparse` at cp313 either, so the script now refuses such an interpreter instead of trying |
| Install of `openlane` in `/home/<user>/openlane-venv` | EXECUTED, PASSED | On the route `setup_openlane_flow.sh` itself takes, with `python3.12` `3.12.16-1+resolute1` from the PPA as the interpreter, `pip install openlane volare` exited 0 in that environment and put `openlane` 2.3.10, `klayout` 0.29.12, `libparse` 0.3.1 and `volare` 0.20.6 into it. `~/openlane-venv/bin/volare --version` prints `Volare v0.20.6`, and `~/openlane-venv/bin/python3 -m openlane.env_info` prints distro `ubuntu 26.04`, `python 3.12.16`, container engine `docker 29.5.3`, rootless `False`, and reports `tkinter` as absent, which is the next row |
| `tkinter` for the `openlane` entry point | EXECUTED, FAILED on the packaged interpreter, and the script installs it now | `~/openlane-venv/bin/openlane --help` ends in `ModuleNotFoundError: No module named 'tkinter'` raised from `openlane/common/tcl.py`, and `/usr/bin/python3.12 -c "import tkinter"` fails the same way in that distribution, where `openlane.env_info` had already listed `tkinter` as not found. `apt-cache policy python3.12-tk` shows the remedy one apt call away. `setup_openlane_flow.sh` installs `python3.12-tk` next to the interpreter and tests the import on the interpreter it picked, stopping with status 127 when the test still fails, so an environment that cannot start `openlane` is no longer discovered only after the PDK download. Whether a `python-build-standalone` interpreter carries `tkinter` was not checked |
| Command line of the installed `volare` | EXECUTED, and it is not the spelling the script used until now | `volare --version` prints `Volare v0.20.6` and `volare --help` lists `build, enable, fetch, ls, ls-remote, output, path, prune, rm`: there is no `search` command and no `--global` option, and asking for either prints `Error: No such option '--global'.` or `Error: No such command 'search'.`, which the discarded stderr of the script had been turning into a complaint about the network. The options belong to the subcommands: `--pdk-root` defaults to `$PDK_ROOT` or `~/.volare`, `--pdk` defaults to `sky130`, `-l/--include-libraries` accepts `all`, and `fetch` and `enable` take the version as an optional argument. `volare ls-remote --pdk sky130` printed 40 character commits one per line, first `c6d73a35f524070e85faff4a6a9eef49553ebc2b`. The script reads that list and calls `fetch --pdk-root "$PDK_ROOT" --pdk sky130` with that version, adding `-l` only when `LL_PDK_LIBRARIES` names a library set, then `enable` with the same two options. Both calls have run here: that listing, a `fetch` of the newest version with `-l all`, which put 16 GiB on disk under `/home/<user>/pdk`, and the `enable` that created `/home/<user>/pdk/sky130A` and `/home/<user>/pdk/sky130B` as links into the version directory. The same version without `-l all` measured 2.1 GiB on this machine, which is inside the 700 MiB to 1.5 GiB upstream states for a version, so the large set is no longer the default |
| What volare 0.20 and openlane 2.3.10 do, as read from the installed packages | READ, not executed | `volare/common.py` puts versions under `$PDK_ROOT/volare/sky130/versions/<hash>` and falls back to `~/.volare` when `PDK_ROOT` is not in the environment. `enable()` in `volare/manage.py` symlinks each variant of the family from the version directory directly into `$PDK_ROOT` and records the version in `$PDK_ROOT/volare/sky130/current`, so an enabled `sky130` yields the `$PDK_ROOT/sky130A` this flow checks. `run_in_container` in `openlane/container.py` mounts `$HOME`, then the PDK root, then the working directory when that is outside `$HOME`, all at identical paths, sets `PDK_ROOT` inside the container, pulls the image `openlane/__main__.py` names, `ghcr.io/efabless/openlane2` tagged with the package version, and passes `--user <uid>:<gid>` when the engine is docker and not rootless, which `openlane.env_info` reports as the state here |
| OpenLane, sky130A PDK | `openlane` INSTALLED in `~/openlane-venv`, PDK INSTALLED in `~/pdk` and in `~/.volare`, container image PULLED | When neither the tool nor the PDK was installed yet, `asic_chip/scripts/build_gds.sh` reported `[FAIL] neither flow.tcl nor openlane was found on PATH` and exited 127, on the host and again inside the Ubuntu distribution, where it also printed the bring-up banner naming `config.bringup.json` at 8.0 ns and the generator `--check` gate ended in `FINAL VERDICT: PASS`. That message is now worded as the three places the entry point looks, and `~/openlane-venv` is one of them and does hold `openlane` 2.3.10. The flow has since run, and section 2.5 records how far it got and what it left behind |

Consequence, stated plainly: **the RTL has now been linted and simulated cycle by cycle by
a real simulator, Verilator 5.054, inside a container image, and synthesized into
`sky130_fd_sc_hd` by the Yosys step of a real OpenLane 2 run.** What has not been run here is a
place and route that finished, static timing on a routed netlist, power and area extraction after
layout, and a second simulator. Everything marked EXECUTED below was executed either by Python on
the host, by the commands of section 2.4 in the Verilator container, or by section 2.5 in the
OpenLane container.

## 2. What was executed

### 2.1 The generator and its own gates

| Command | Result |
|---|---|
| `python asic_chip/scripts/lingologos_compiler.py` | regenerated the three generated modules, `sim/golden_vectors.mem`, `sim/core_self_test.mem` and `rtl/generated/MANIFEST.json`; `FINAL VERDICT: PASS` |
| `python asic_chip/scripts/lingologos_compiler.py --check` | same gates plus `RTL_matches_artifacts: true`; `FINAL VERDICT: PASS` |
| gate 6 of the Python suite (`test_generator_is_deterministic_and_check_mode_passes`) | regenerates into a temporary directory and proves the committed bytes are reproducible and that no committed output is stale |

The five internal gates of the generator, with the values it printed:

| Gate | What it proves | Measured result |
|---|---|---|
| H1_weight_mesh_fits | every product is a shift-add tree that fits the accumulator | 4 by 4 mesh, 20 bit accumulator, limit 524287, largest shift 7, worst partial sum bound 35072, headroom ratio 14.95 |
| H2_gf_table_closed | the automaton is a closed table over a field | polynomial `0x11D` is irreducible, 255 reachable states, state budget 255, 1020 defined arms, symbol orders [255] |
| H3_smt_netlist_latch_free | the clause network is combinational | 4 clauses, 4 result lanes, no unused lane, deepest cone 3 levels, nodes per clause 3, 2, 4, 3 |
| H4_golden_vectors_in_range | the vector table is legal stimulus | 304 rows, seed 42, all 256 clause input bytes present, final GF state `0x8E`, three bracket words recorded |
| H5_core_self_test_consistent | the core self test agrees with the engines | input `[10, -5, 2, 100]`, row sums `[9245, -1606, 4170, -925]`, SMT readback `0x09`, GF state `0x02` after the step and `0x01` after idle, status word `0x13`. The two counter views, the 3 inside that word and the 4 in the register once the status request itself has been counted, are the `exp_x13` and `exp_cop_count` pairs of `sim/core_self_test.mem` |
| RTL_matches_artifacts (`--check` only) | the committed RTL still says what the artifacts say | true |

### 2.2 The Python gate suite

`python -m pytest -v tests/` was run after the last change. The full suite reports 205
tests, all PASSED, zero failures, zero errors; the last run printed `205 passed in 27.06s`,
of which about 16 seconds belong to the synthesis run that `tests/test_grammar_artifacts.py`
performs once per session.
Of those, 26 tests
target `asic_chip/`: 20 in `tests/test_asic_compiler.py` and 6 in
`tests/test_asic_core_firmware.py`. They are
not unit tests of the generator: with one exception they never import it, they read the
committed files and recompute the answers from a separate reference model
(`tests/asic_reference_model.py`), which implements the dot product, the GF(2^8) field,
the clause masks and the RV32E instruction semantics by itself.

| Gate | Test | What it checks | Status |
|---|---|---|---|
| 1 | `test_chip_config_matches_the_rtl_port_list` | `config.json` ports, widths and file lists agree with the port declarations in the RTL, and every path in `paths` exists | EXECUTED, PASSED |
| 2 | `test_generated_mesh_netlist_evaluates_like_a_dot_product` | the generated `wire` netlist is evaluated by a small interpreter in the test and compared with a plain dot product over the artifact weights | EXECUTED, PASSED |
| 3 | `test_generated_gf_table_matches_field_arithmetic_and_the_grammar_alphabet` | all 1020 arms of the table are recomputed with `gf_mul` in the model, the alphabet matches `artifacts/grammar.json`, and there is exactly one default arm | EXECUTED, PASSED |
| 4 | `test_generated_smt_netlist_matches_the_reference_masks_on_every_byte` | three derivations of each lane word agree for all 256 input bytes: the shunting-yard mask model, the generator parser, and the netlist text evaluated by the test interpreter; latch freedom checked by the absence of any `always` block and of any non blocking assignment; lane depth measured on the netlist dependency graph and compared with `config.json` and with the datasheet table | EXECUTED, PASSED |
| 5 | `test_golden_vectors_are_reproduced_without_the_generator` | every column of all 304 rows recomputed from the artifacts, including the GF chain and the SMT lane words | EXECUTED, PASSED |
| 6 | `test_generator_is_deterministic_and_check_mode_passes` | regeneration into a scratch directory is byte identical, `--check` passes, and no committed file is stale | EXECUTED, PASSED |
| 7 | `test_manifest_digests_match_the_committed_files` | the digests in `MANIFEST.json` are recomputed over the raw bytes on disk, so `sha256sum` on the checkout reproduces them | EXECUTED, PASSED |
| 8 | `test_self_test_memory_matches_the_reference_engines` | `core_self_test.mem` agrees with the model engines and with the RTL firmware constants | EXECUTED, PASSED |
| 9 | `test_testbench_literals_match_the_self_test_memory`, `test_harness_cycle_budget_covers_the_measured_self_test` | the literals baked into `testbench.sv` and `sim_main.cpp` equal the 17 key and value pairs of the memory file, the row count the testbench loads equals the number of data lines in the vector file, and the 400 cycle budget of the C++ harness covers the run: 89 cycles counted by the emulator, 91 by the harness itself | EXECUTED, PASSED |
| 10 | `test_harness_touches_only_declared_top_level_ports` | `sim_main.cpp` and `testbench.sv` drive and sample only ports that exist at the top level, and the `dbg_` group is inside the `LL_DEBUG_PORTS` region | EXECUTED, PASSED |
| 11 | `test_pinout_agrees_in_rtl_config_datasheet_and_pin_plan` | the 18 public ports, 51 input bits, 95 output bits, 146 total bits, the datasheet pin table and the 37 lines of `openlane/pin_order.cfg` all agree; every non clock input pin in the plan matches exactly one port glob | EXECUTED, PASSED |
| 12 | `test_openlane_setup_agrees_with_the_chip_configuration` | the SDC period, uncertainty, load and driving cell, and the OpenLane config design name and file list agree with `config.json` | EXECUTED, PASSED |
| 13 | `test_datasheet_numbers_are_measured_values_not_claims` | every quantitative claim in the datasheet is recomputed: mesh example words, GF table figures, SMT node and depth table, vector row count, switching activity over the 303 transitions, accumulator bound and headroom, and the three artifact digests quoted at the end of the document | EXECUTED, PASSED |
| 14 | `test_asic_chip_text_is_ascii_and_english` | every text file under `asic_chip/` is ASCII, contains no Italian words and no leftover task or defect marker, and uses LF line endings | EXECUTED, PASSED |
| extra | `test_no_generated_artifact_is_committed_twice_or_missing` | the generated set matches `config.json` `paths.rtl_generated` exactly, with no duplicate or stray file | EXECUTED, PASSED |
| extra | `test_the_reference_model_mux_semantics_are_the_documented_ones` | the `select ? set : clear` semantics of `MUX` are checked byte by byte in the model and the generated RTL uses the same conditional form, so the mask identity bug listed in section 4 cannot come back | EXECUTED, PASSED |
| F1 to F5 plus data readback | `tests/test_asic_core_firmware.py` | the firmware ROM is complete and inside the documented subset, the illegal encoding is the documented `x16` probe, the emulated run reproduces `core_self_test.mem`, the run fits the harness budget, and data memory readback matches the model | EXECUTED, PASSED |

### 2.3 The core, emulated in Python and confirmed in simulation

`tests/asic_reference_model.py` contains a small RV32E emulator (`run_firmware`) that
executes the 97 word firmware of `rv32e_core.sv` word by word, with its own decode, its
own illegal instruction rule, its own alignment rule and its own model of the
coprocessor window. This is what produced the core self test expectations, and it is what
gate F4 re-checks on every run.

| Item | Value | Label |
|---|---|---|
| Firmware size | 97 words in a 128 word ROM, 9 bit program counter | EXECUTED |
| Cycles to completion | 89 | CROSS-CHECKED, emulator |
| Distinct instructions executed | 89 | CROSS-CHECKED, emulator |
| Coprocessor accesses | 4 | CROSS-CHECKED, emulator; EXECUTED, the counter reads 4 at the debug port in both simulator paths |
| Illegal encodings | 1, at ROM slot 79: `32'h00080093`, `ADDI x1, x16, 0` | CROSS-CHECKED, emulator |
| `x8`, `x10`, `x11`, `x13`, `x15` at the end | `0x00000009`, `0x00000002`, `0x0000104A`, `0x00000013`, `0x00000000` | CROSS-CHECKED, emulator |
| GF state after the step and after idle | `0x02`, then `0x01` | CROSS-CHECKED, emulator |
| C++ harness cycle bound | 400 cycles, 4.4 times the 91 cycles the harness itself measured | EXECUTED, source level check plus the run of section 2.4 |

### 2.4 The RTL under Verilator 5.054, in a container

The three runs below are the expansion of the `lint`, `sim-verilator` and `sim-tb` targets
of `asic_chip/Makefile`: the same `VOPT` flags, the same `VDEF` define, the same file list,
the same binary names and arguments. They ran in the image
`verilator/verilator:latest` with the checkout mounted at `/work`, so the relative paths in
the vector tables resolve exactly as they do on the host. The image entrypoint already
starts the tool, so the argument list in the table starts at the flags.

| Step | What ran | Result, quoted from the tool |
|---|---|---|
| lint | `--lint-only -O2 --timescale 1ns/1ps -Wall -Wno-UNSIGNED -Wno-UNUSED -Wno-WIDTHTRUNC -DLL_DEBUG_PORTS --top-module lingologos_top` over the six RTL files | exit 0, no warning left. `Built from 0.343 MB sources in 8 modules, into 0.152 MB in 3 C++ files`, `Walltime 0.107 s`, `allocated 27.086 MB` |
| C++ harness | `--cc --exe --build` with the same flags plus `--trace --top-module lingologos_top -Mdir asic_chip/sim/obj_dir`, then `./asic_chip/sim/obj_dir/Vlingologos_top asic_chip/sim/golden_vectors.mem asic_chip/sim/core_self_test.mem asic_chip/sim/lingologos.vcd` | build exit 0, `Built from 0.336 MB sources in 8 modules, into 0.379 MB in 9 C++ files`, build part of the walltime `7.275 s`. Run exit 0: `[INFO] phase A, 304 golden rows`, `[INFO] phase B, RV32E instruction-set self-test`, `[INFO] self-test took 91 clock cycles`, `checked 304 vector rows plus the core self-test, 0 mismatches`, `FINAL VERDICT: PASS` |
| SystemVerilog testbench | `--binary --timing` with the same flags, `--top-module lingologos_testbench -Mdir asic_chip/sim/obj_dir_tb`, then `./asic_chip/sim/obj_dir_tb/Vlingologos_testbench` | build exit 0, `Built from 0.435 MB sources in 9 modules, into 0.241 MB in 7 C++ files`, `Walltime 6.400 s`. Run exit 0: `phase A, 304 golden rows`, `self-test finished at pc 174, coprocessor counter 4`, `checked 304 vector rows plus the core self-test, 0 mismatches`, `FINAL VERDICT: PASS`, `$finish at 4us; walltime 0.013 s; speed 319.818 us/s` |

What that buys:

- The 304 golden rows now agree with the RTL in two independent simulators of the same
  design (the C++ harness and the SystemVerilog testbench), which is what L4 asks for.
- The core self test passes against the RTL: `x8`, `x10`, `x11`, `x13`, `x15`, the sticky
  illegal instruction flag and the final automaton state all match
  `sim/core_self_test.mem`, so the illegal encoding path of firmware word 79 is now proven
  by a simulator and not only by the Python emulator.
- The halted program counter reads byte address `0x174`, which is word 93 of the 97 word
  ROM, the halt loop.
- The C++ harness writes `asic_chip/sim/lingologos.vcd`: 248,815 bytes, 17,359 lines, the
  waveform a power tool would later be fed. It is a build product and `.gitignore` keeps it
  out of the commit.
- The cycle figure of the self test is 91 as counted by the harness loop and 89 as counted
  by the emulator; the two counters use different start and stop conditions. The harness
  bound is 400 cycles, which covers both.
- The make targets were run in the same image too, which carries its own `make` and C++
  compiler: `make -f asic_chip/Makefile lint`, then `sim-tb`, then `sim-verilator`, exit 0
  for all three, each ending in `FINAL VERDICT: PASS`. That run reused the two build
  directories the commands above had already produced, so it re checked and re ran them
  instead of recompiling, and the log says `make[1]: Nothing to be done for 'default.'` for
  both. The lint line of the table is the fresh one: `Walltime 0.098 s`, `allocated 27.152
  MB`.

### 2.5 The OpenLane 2 flow, run three times on 2026-11-09

Three runs, in the copy of this tree that the PowerShell setup command puts in
`/home/<user>/LingoLOGOS`, inside Ubuntu 26.04.1 on WSL 2, with Docker engine 29.5.3 answering
inside the distribution. The entry point was `sh asic_chip/scripts/build_gds.sh` with
`LL_OPENLANE_BIN` naming `~/openlane-venv/bin/openlane`, so `openlane` 2.3.10 ran the Classic flow,
78 steps, in the container image `ghcr.io/efabless/openlane2:2.3.10`, which occupies 7.39 GB of
docker disk usage. The configuration handed to all three runs was
`asic_chip/openlane/config.bringup.json`, the bring-up set, at an 8.0 ns gate for the first two runs
and a 30.0 ns gate for the third. The first run, `asic_chip/openlane/runs/RUN_2026-11-09_16-15-02`,
166 MB, stopped at step 25. The second, `asic_chip/openlane/runs/RUN_2026-11-09_18-43-18`, 1.2 GB,
went through all 78 steps; its own `resolved.json` confirms `CLOCK_PERIOD` 8, and its figures are in
the subsection headed Run 2 below. The third, `asic_chip/openlane/runs/RUN_2026-11-09_21-14-43`,
also ran all 78 steps, with `resolved.json` confirming `CLOCK_PERIOD` 30.0; the subsection headed
Run 3 holds its figures, and every post-layout figure elsewhere in this repository that is marked
with a 30.0 ns period comes from it.

#### Run 1, stopped at the custom I/O placement step

| Steps of the flow | Result | What they left |
|---|---|---|
| `01-verilator-lint` | completed | the Verilator lint of the six RTL files inside the OpenLane image, with the flow configured not to stop on linter errors |
| `02` to `04`, the timing construct, error and warning checkers | completed | `warning.log` of the run, three lines, and `error.log`, seven |
| `05-yosys-jsonheader`, `06-yosys-synthesis` | completed | the mapped netlist `lingologos_top.nl.v`, 11,015 cells on 10,938 nets, and in the log `Chip area for module '\lingologos_top': 149097.996800` square micrometers against `sky130_fd_sc_hd__tt_025C_1v80` |
| `07` to `09`, the unmapped cell, synthesis check and assign statement checkers | completed | nothing to stop on, no unmapped cell and no inferred latch |
| `10-openroad-checksdcfiles` | completed | the 8.0 ns constraint file read by OpenROAD. `resolved.json` shows `PNR_SDC_FILE` as null, which is the source of the warning `'PNR_SDC_FILE' is not defined. Using generic fallback SDC`: the steps after it timed the design against a generic constraint file rather than this one, and both configuration files now name the same file under that key |
| `11` to `22` | completed | the floorplan, the tap and endcap rows, the power distribution network and the routing obstructions |
| `23-openroad-globalplacementskipio`, `24-openroad-ioplacement` | completed | a global placement and the OpenROAD pin placement |
| `25-odb-customioplacement` | FAILED, exit 65 | `line 1:2 token recognition error at: '-'` and `An exception occurred: Syntax Error at 1:2: token recognition error at: '-'` |

Nothing after step 25 ran in that directory: no GDSII, no `reports/report.csv`, no post-layout
timing report, no DRC, no LVS and no circuit level verification. Two facts about it are worth a line
of their own, because both changed what the tree does, and they come before the second run because
the first of them is what the second run fixed.

- **Why step 25 failed.** That step parses `asic_chip/openlane/pin_order.cfg` with the pin
  constraint grammar of OpenLane 2, and the file was written in the line oriented format of
  OpenLane 1, a comment banner followed by `left <glob>` rows. The grammar has no comment token, so
  the dash in the banner is a syntax error at the second character of the first line. Section 5 of
  `build_instructions.md` now describes the grammar as the installed tool accepts it. The file was
  rewritten in it, and the rewrite was checked inside that same image, started with
  `--entrypoint python3`, against the port names of the netlist this run produced: 146 ports in the
  netlist and 146 pins matched by the patterns, each one exactly once, 40 north, 40 east, 40 west
  and 26 south, nothing unmatched in either direction. That check does not run on the host: the
  `antlr4-python3-runtime` in `~/openlane-venv` is 4.9.3, while the parser `ioplace-parser` ships
  was generated for ANTLR 4.10, and handing it the file ends in `TypeError: ord() expected string
  of length 1, but int found`. The pinout gate of the Python suite reproduces the matching, the
  anchoring and the character restriction without needing a run.
- **Which PDK the run used.** `resolved.json` records `PDK_ROOT` as
  `/home/<user>/.volare/volare/sky130/versions/0fe599b2afb6708d281543108caf8310912f54af`, not
  `/home/<user>/pdk`: that launch did not have `PDK_ROOT` in its environment, and volare falls back
  to `~/.volare`. Both roots hold a version now, `~/pdk` the newest one `ls-remote` lists,
  `c6d73a35f524070e85faff4a6a9eef49553ebc2b`, enabled with its `sky130A` and `sky130B` links, and
  `~/.volare` the version an earlier download had put there. The second run of the same day resolved the PDK
  under `~/.volare` too, at `0fe599b2afb6708d281543108caf8310912f54af`, so neither run used what
  `volare enable` had installed under `~/pdk`. `build_gds.sh` defaults `PDK_ROOT` to `~/pdk` when
  the environment names nothing, and the next launch will record in its own `resolved.json` which
  root it actually used.

#### Run 2, the same command after the pin file was rewritten

All 78 steps ran in 16 min 59 s, and the run wrote what the first one had not. OpenLane 2 writes
no `results/` or `reports/` tree at the top of a run directory: each step leaves its output in its
own numbered directory, and the files worth opening are these.

| What | Where in `RUN_2026-11-09_18-43-18` | What it says |
|---|---|---|
| GDSII | `56-magic-streamout/lingologos_top.magic.gds`, `56-magic-streamout/lingologos_top.gds`, `57-klayout-streamout/lingologos_top.klayout.gds` | three files, from the two streamout steps |
| DRC, magic | `62-magic-drc/reports/drc_violations.magic.rpt` | `[INFO] COUNT: 0` |
| DRC, klayout | `63-klayout-drc/reports/drc_violations.klayout.json` | every rule at zero, `"total": 0` |
| Streamout cross check | `60-klayout-xor/xor.xml` | the XOR of the magic and klayout streamouts ran and reported no error |
| LVS, netgen | `68-netgen-lvs/reports/lvs.netgen.rpt` | `Final result: Circuits match uniquely.`, 37,818 device instances on 12,448 nets, 3 disconnected pins, series and parallel devices merged to 12,573 devices |
| Antenna | `74-misc-reportmanufacturability/manufacturability.rpt`, `45-openroad-checkantennas-1/reports/antenna_summary.rpt` | FAILED: 54 pin violations on 46 nets after the repair. Worst ratio 9.44 before it, 5.20 after, 468 antenna cells in the final netlist |
| Disconnected pins | `47-odb-reportdisconnectedpins` | 3, none critical |
| Area and instances | `51-openroad-fillinsertion/or_metrics_out.json` | die 350,447, core 330,268, standard-cell area 169,306 square micrometers, core utilization 0.512633, 17,552 instances, 148 I/O pins |
| Wire length | `49-odb-reportwirelength/or_metrics_out.json`, `wire_lengths.csv` | longest net 1,529.65 micrometers, 12,180 rows of per net length |
| IR drop | `55-openroad-irdropreport` | worst drop 0.0135214 V on VPWR, worst voltage 1.78648 V, worst 0.00820593 V on VGND |
| Power estimate | `54-openroad-stapostpnr/nom_tt_025C_1v80/power.rpt` | total 1.507948e-01 W at the 8.0 ns period, 78.5 percent of it combinational, under the default activity model of the tool and not from the stimulus vectors of section 2.3 |
| Signoff summary | `74-misc-reportmanufacturability/manufacturability.rpt` | `Antenna Failed`, `LVS Passed`, `DRC Passed` |
| Exit | the terminal of the launch | non zero, `ERROR ... One or more deferred errors were encountered` then `OpenLane will now quit.` |

The static timing step wrote a directory per corner, nine corners, each with its own reports. The
figures below are the worst slack and the count of violating paths of `54-openroad-stapostpnr`,
taken from `summary.rpt` and from `max.rpt` and `min.rpt` inside each corner directory, at the 8.0
ns period that run was constrained to.

| Corner | Setup worst slack | Setup total negative slack | Setup paths | Hold worst slack | Hold paths | Max cap | Max slew |
|---|---|---|---|---|---|---|---|
| overall | -15.7205 ns | about -23,845 ns | 15,089 | -0.1700 ns | 41 | 56 | 1,774 |
| `nom_tt_025C_1v80` | -4.1255 ns | -2,061.6 ns | 2,478 | -0.0431 ns | 3 | 0 | 0 |
| `min_tt_025C_1v80` | -3.8033 ns | -1,420.7 ns | 2,410 | -0.0155 ns | 2 | 0 | 0 |
| `max_tt_025C_1v80` | -4.4409 ns | -2,716.1 ns | 2,485 | -0.0532 ns | 8 | 0 | 0 |
| `nom_ss_100C_1v60` | -15.0799 ns | about -22,617 ns | 2,572 | -0.1359 ns | 10 | 41 | 1,223 |
| `min_ss_100C_1v60` | -14.5466 ns | about -21,396 ns | not counted here | -0.0828 ns | 7 | 24 | 781 |
| `max_ss_100C_1v60` | -15.7205 ns | about -23,845 ns | not counted here | -0.1700 ns | 11 | 56 | 1,774 |
| `ff_n40C_1v95`, all three | positive, +0.0880 ns at worst | none | 0 | positive, +0.0144 ns at worst | 0 | 0 | 0 |

The worst setup path of `max_tt_025C_1v80` runs from `_17335_` to `_16903_` on the net `core_pc[3]`,
through `rebuffer51`, `xor2_4`, `a21oi_1`, `a211oi_2`, `o311ai_4`, `clone600`, `and4b_4`, `a22o_1`,
`a2111o_1` and a `dlymetal6s2s_1` delay cell, with the clock tree delivering that register through
`clkbuf_0_clk` and `clkbuf_leaf_157_clk` at about 1.23 ns of insertion delay. The worst hold path of
that corner starts at the input port `in_vector_2[6]` and ends at the register `_16656_`, with a data
arrival of 1.211 ns against a clock arriving at about 1.23 ns plus the hold time and 0.05 ns of
uncertainty. Every one of the 41 violating hold paths is of that input to register shape; none is
register to register.

- **What made the run exit non zero, and what did not.** The exit is not a crash. It is
  `Checker.SetupViolations` and `Checker.HoldViolations` in `openlane/steps/checker.py`, lines 568
  and 575, raising `DeferredStepError` at line 585 once the flow has finished everything it can
  still do. What decides error or warning is a corner list, and the list OpenLane installs for a
  sky130 or gf180mcu PDK is `TIMING_VIOLATION_CORNERS = ["*tt*"]`, set in
  `openlane/config/pdk_compat.py` at line 231 and present in the `resolved.json` of this run at line
  1316. That is why the three tt corners produced the errors while the ss corners, whose slack was
  four times worse, produced warnings: `SETUP_VIOLATION_CORNERS` and `HOLD_VIOLATION_CORNERS` were
  null in that run and inherited the list. Both are set to `["*"]` in both configuration files now,
  so from here every corner that fails is an error. Max slew and max cap were never going to be
  errors at all, whatever the corner: their corner list is the empty string override of `checker.py`
  lines 616 and 627, so the 1,774 slew and 56 cap violations of the slow corner are reported numbers
  and not gates, and this tree reports them as that.
- **The names in the older configuration that do nothing.** `QUIT_ON_TIMING_VIOLATIONS` does not
  exist in OpenLane 2.3.10; `RUN_CVC` is listed in `openlane/config/removals.py` at line 41 with the
  note that upstream no longer supports CVC inside OpenLane; `QUIT_ON_LINTER_ERRORS` is a deprecated
  name of `ERROR_ON_LINTER_ERRORS`, `checker.py` lines 346 to 353. A configuration carrying them
  only produces the warnings `has been removed` and `provided is unused by the current flow`, which
  is what the log of both runs shows. All three are gone from `config.json` and
  `config.bringup.json`.
- **Why the antenna check failed, and what the tree changed for it.** The two configuration files
  carried `DIODE_INSERTION_STRATEGY: 3`, an OpenLane 1 knob. `openlane/config/config.py` lines 918 to
  943 deletes that key and rewrites the native knobs from it, and for value 3 the result was
  `RUN_HEURISTIC_DIODE_INSERTION: false`, `DIODE_ON_PORTS: "none"`, with only the global router
  repair left on. That is exactly what `resolved.json` of this run records at lines 152, 355 and 356,
  and what `flow.log` says twice with `Gating variable ... set to 'False'`: the heuristic diode
  insertion step and the diodes on ports step were skipped, and the repair that did run used its
  default 3 iterations and 10 percent margin. 46 nets stayed over the ratio. The legacy key is out of
  both configurations, and the native knobs are now what they were meant to be: heuristic insertion
  on with `HEURISTIC_ANTENNA_THRESHOLD` 60, `DIODE_ON_PORTS` on the inputs, `GRT_ANTENNA_ITERS` 5,
  `GRT_ANTENNA_MARGIN` 25, `RUN_POST_GRT_DESIGN_REPAIR` and `RUN_POST_GRT_RESIZER_TIMING` on, and a
  floorplan with room to repair, `FP_CORE_UTIL` 40 and `PL_TARGET_DENSITY` 0.45.
- **Why the bring-up gate moved from 8.0 ns to 30.0 ns.** At 8.0 ns the routed design missed setup
  by 4.4409 ns in the worst tt corner and by 15.7205 ns in the worst slow corner, which puts the
  first order limit of this layout at about 12.44 ns in tt and about 23.72 ns in ss, roughly 80 and
  42 MHz. 30.0 ns is the gate chosen so that a rerun has margin against the worst corner rather than
  against the typical one; it is a choice about what the flow is held to, not a rating of the part.
  The 4.0 ns declared budget stays in `config.json` and `lingologos.sdc`, and has never been taken
  through place and route. The hold side of the same run is a constraint problem rather than a speed
  problem: the clock tree produced 1.2 to 1.35 ns of insertion delay, and the bring-up constraint
  file asks for an input delay minimum of 0.30 ns, which is smaller than the insertion delay, so an
  input launched at the same edge can arrive before the capturing clock edge. The post route resizer
  is now enabled to fix those paths, and the constraint file says in its own comments that its input
  and output delays are absolute values that do not scale with the period.
- **Warnings that are not problems.** 43 lines of `[STA-0441] set_input_delay relative to a clock
  defined on the same port/pin not allowed` come from the constraint file applying an input delay to
  the clock port itself; OpenLane reads its own base constraint file the same way, and the warning
  does not change any measurement reported above. The other recurring warnings are `[GRT-0097] No
  global routing found for nets`, `[DRT-0349] LEF58_ENCLOSURE ... Skipping for layer mcon`,
  `[DRT-0120] Large net net301 has 162 pins`, `Checker.WireLength skipped: no threshold configured`,
  `'VSRC_LOC_FILES' was not given a value` and `Cell 'lingologos_top' has (3) input pin(s) without
  antenna gate information`.

#### Run 3, the same command with the diode knobs native and the gate at 30.0 ns

`asic_chip/openlane/runs/RUN_2026-11-09_21-14-43`. Nothing about the launch changed: the same script,
the same image, the same PDK under `~/.volare` at `0fe599b2afb6708d281543108caf8310912f54af`. What
changed is the configuration: `DIODE_INSERTION_STRATEGY` gone and the diode knobs named directly, the
corner lists at `["*"]`, and `CLOCK_PERIOD` at 30.0 instead of 8.0. All 78 steps ran in 12 min 12 s,
and `resolved.json` records `RUN_HEURISTIC_DIODE_INSERTION` true, `DIODE_ON_PORTS` `in`,
`GRT_ANTENNA_ITERS` 5 and `GRT_ANTENNA_MARGIN` 25, which is what the file asks for rather than what a
legacy key derived from it. The step numbering of this directory is three higher in the back half of
the flow than the previous run, because three more steps were enabled. The exit status was 2.

| What | Where in `RUN_2026-11-09_21-14-43` | What it says |
|---|---|---|
| GDSII | `59-magic-streamout/lingologos_top.magic.gds`, `59-magic-streamout/lingologos_top.gds`, `60-klayout-streamout/lingologos_top.klayout.gds` | three files, from the two streamout steps |
| DRC, magic | `65-magic-drc/reports/drc_violations.magic.rpt` | `[INFO] COUNT: 0` |
| DRC, klayout | `66-klayout-drc/reports/drc_violations.klayout.json` | every rule at zero, `"total": 0` |
| Router rule count | `46-openroad-detailedrouting/or_metrics_out.json` | `route__drc_errors` 0 at the eighth iteration, from 10,890 at the first; 660,126 micrometres of wire over 134,302 vias on 11,718 nets |
| Streamout cross check | `63-klayout-xor/xor.xml` | the XOR of the two streamouts ran and reported no error |
| LVS, netgen | `71-netgen-lvs/reports/lvs.netgen.rpt` | `Final result: Circuits match uniquely.`, 13,563 devices on 11,990 nets, 3 disconnected pins. The device count is not the same measure as the 37,818 of the previous run, which also quoted 12,573 devices after collapsing series and parallel ones; the verdict is what is comparable between the two |
| Antenna | `77-misc-reportmanufacturability/manufacturability.rpt`, `48-openroad-checkantennas-1/reports/antenna_summary.rpt` | `Antenna Passed`, and the violation table of the check that runs after the detailed router is empty |
| Area and instances | `54-openroad-fillinsertion/or_metrics_out.json` | die 393,012, core 371,917, standard-cell area 204,806 square micrometres, core utilization 0.550678, 34,149 instances, 148 I/O pins |
| What the antenna repair cost | same file, `design__instance__count__class:antenna_cell` | 16,898 instances of class `antenna_cell`, against the 468 diodes the previous run inserted |
| IR drop | `58-openroad-irdropreport/or_metrics_out.json` | `nom_tt_025C_1v80`: worst VPWR voltage 1.79663 V, worst drop 0.00336923 V, worst VGND 0.00378131 V |
| Power estimate | `57-openroad-stapostpnr/nom_tt_025C_1v80/power.rpt` | total 3.926652e-02 W at the 30.0 ns period, 43.0 percent internal and 57.0 percent switching, leakage 9.765576e-08 W; combinational 3.031464e-02 W, sequential 5.650650e-03 W, clock 3.301227e-03 W. Default activity model, not the stimulus vectors of section 2.3 |
| Signoff summary | `77-misc-reportmanufacturability/manufacturability.rpt` | `Antenna Passed`, `LVS Passed`, `DRC Passed` |
| Exit | the terminal of the launch | 2, after `ERROR ... One or more deferred errors were encountered` and `OpenLane will now quit.` |

The static timing step wrote a directory per corner, nine corners. The figures below are from
`57-openroad-stapostpnr/<corner>/`, from `ws.max.rpt`, `wns.max.rpt`, `tns.max.rpt`, `ws.min.rpt`,
`tns.min.rpt` and `violator_list.rpt`, at the 30.0 ns period this run was constrained to.

| Corner | Setup worst slack | Setup total negative slack | Setup paths | Hold worst slack | Hold total negative slack | Hold paths |
|---|---|---|---|---|---|---|
| `max_ss_100C_1v60` | -0.4906 ns | -13.5139 ns | 35 | -0.1540 ns | -0.9162 ns | 10 |
| `nom_ss_100C_1v60` | +0.3841 ns | none | 0 | -0.1380 ns | -0.6480 ns | 9 |
| `min_ss_100C_1v60` | +1.1885 ns | none | 0 | -0.1067 ns | -0.3071 ns | 7 |
| `max_tt_025C_1v80` | +14.4003 ns | none | 0 | -0.0989 ns | -0.3077 ns | 8 |
| `nom_tt_025C_1v80` | +14.9556 ns | none | 0 | -0.0585 ns | -0.2102 ns | 5 |
| `min_tt_025C_1v80` | +15.4798 ns | none | 0 | -0.0283 ns | -0.0917 ns | 5 |
| `max_ff_n40C_1v95` | +20.1103 ns | none | 0 | -0.0024 ns | -0.0024 ns | 1 |
| `nom_ff_n40C_1v95` | +20.5284 ns | none | 0 | +0.0254 ns | none | 0 |
| `min_ff_n40C_1v95` | +20.8555 ns | none | 0 | +0.0536 ns | none | 0 |

The setup side is now one corner only. The worst path of `max_ss_100C_1v60`, read from the
`[setup reg-reg]` entries of `violator_list.rpt` and the path group of `max.rpt`, runs from
`_17334_/Q` through the net `core_pc[2]` into `_08377_`, an `or3b_2`, then `_08403_` an `a21oi_2` and
`_08404_` an `a211o_1`, and ends at `_17287_/D`; the clock reaches that register through
`clkbuf_0_clk`, a `clkbuf_16`, then `clkbuf_2_3_0_clk`, `clkbuf_5_26__f_clk` and
`clkbuf_leaf_164_clk`, with 2.1845 ns of insertion delay. Constrained to 30.0 ns, that corner needs
30.4906 ns, and it is the only one of the nine that does not have slack to spare at that period.

The hold side is a different problem, and its shape did not change between the two complete runs: in
this run, as in the previous one, every violating path is an input port to a register, spelled
`[hold in-reg]` in `violator_list.rpt`, and none is register to register. The worst is
`in_vector_1[4]` to `_16662_/D` at -0.154013 ns in `max_ss_100C_1v60`, with a data arrival of
1.972405 ns: the port goes through `input22` and `buf_6`, then `_09823_` an `a21o_1`, `_09828_` a
`nand2_1`, `_09841_` an `and3_1` and `_09842_` a `nor2_2`, onto the net `mesh_raw_1[12]`, with the
declared 0.300000 ns of minimum external input delay and the propagated clock tree of that corner.
Behind it, `in_vector_2[6]` to `_16656_/D` at -0.129536 ns, `in_vector_0[0]` to `_16670_/D` at
-0.127176 ns and `in_vector_2[1]` to `_16651_/D` at -0.126910 ns.

- **What made this run exit non zero.** `error.log` names two checks. The setup checker printed
  `Setup violations found in the following corners:` with one entry, `max_ss_100C_1v60`, and the hold
  checker printed `Hold violations found in the following corners:` with seven, the three ss corners,
  the three tt corners and `max_ff_n40C_1v95`; the two clean ff corners are named in `warning.log`
  instead, which is what the corner lists at `["*"]` produce. Max slew exceeded its limit in five
  corners and max capacitance in three, and both stayed warnings because
  `MAX_SLEW_VIOLATION_CORNERS` and `MAX_CAP_VIOLATION_CORNERS` are empty lists in the installed
  defaults and neither configuration file overrides them. `Antenna Passed`, `LVS Passed` and
  `DRC Passed` mean the physical gates are closed; the exit is timing, and its value is 2.
- **The measured gap between the hold repair and the hold signoff.** The resizer had closed hold at
  the point where it runs: the metrics of `45-openroad-stamidpnr-3`, the timing step right after
  `OpenROAD.ResizerTimingPostGRT`, report `timing__hold__ws` of +0.100201 ns,
  `timing__hold_r2r__ws` of +0.402233 ns and zero hold violations in `nom_tt_025C_1v80`, on parasitics
  estimated from the global route. The same corner at signoff, on the SPEF that `OpenROAD.RCX`
  extracted from the final routing, reads -0.058548 ns. Detailed routing and extraction therefore take
  about 0.16 ns from those input paths, and the Classic flow puts nothing after
  `OpenROAD.DetailedRouting` to put it back: the last hold repair lives in
  `openlane/scripts/openroad/rsz_timing_postgrt.tcl`, which re-runs the global router, estimates
  parasitics from that routing and calls `repair_timing -hold -hold_margin
  $::env(GRT_RESIZER_HOLD_SLACK_MARGIN)`, and that margin was the installed default of 0.05 ns. Both
  hold margins, `GRT_RESIZER_HOLD_SLACK_MARGIN` and `PL_RESIZER_HOLD_SLACK_MARGIN`, are now 0.40 ns in
  `config.json` and in `config.bringup.json`, so the repair aims at 0.40 ns of slack instead of
  stopping a few hundredths above zero. That is a response to a measured mechanism and not a verified
  fix: the next run decides, and `Checker.HoldViolations` in its `error.log` is where the answer is.
- **Why the gate moved from 30.0 ns to 32.0 ns.** At 30.0 ns the slow corner was 0.4906 ns short, so
  that layout needed 30.4906 ns, and 32.0 ns leaves a little under 1.5 ns against the movement a rerun
  produces. That movement is real and not small: the first-order requirement of the typical corner
  measured 12.44 ns after the run at the 8.0 ns gate and 15.60 ns after this one, and the slow corner
  23.72 ns and then 30.49 ns. It is not only a reroute. The layout of this run has 16,898 instances of
  class `antenna_cell` where the previous run had 468 diodes, a die 12 percent larger at 393,012 square
  micrometres and standard-cell area 21 percent larger at 204,806, so the antenna repair the previous
  run failed is part of what this design now pays for its own speed. `CLOCK_PERIOD` 4.0 ns in
  `config.json` stays unrouted, and nothing here states a maximum frequency.
- **What is still owed.** A run that exits 0 has not happened at any period. The physical gates closed
  here and closed in the run before it as well; what has never closed is timing at all nine corners,
  first at 8.0 ns in three of them and now at 30.0 ns in one for setup and seven for hold. The next
  launch, at 32.0 ns with the 0.40 ns hold margins, is the attempt, and its own numbers replace the
  ones above. Until then the honest summary of this design is that it routes, it is manufacturable,
  and it has not been shown to meet a clock.


## 3. What was not run

This is the part that a reader must not infer past.

| Validation level | Intended tool | Status | Why |
|---|---|---|---|
| L4 lint and latch inference, on the host, through `make` | `verilator --lint-only` via `make -f asic_chip/Makefile lint` | NOT RUN on this host | neither Verilator nor `make` is on the host PATH; the same command line ran in a container, section 2.4 |
| L4 cycle accurate run of the 304 golden rows against Verilog | Verilator with `sim/sim_main.cpp` | EXECUTED in the container, section 2.4 | 304 rows and the self test, 0 mismatches |
| L4 the same run through a SystemVerilog testbench | `verilator --binary --timing` | EXECUTED in the container, section 2.4 | same checks, same verdict |
| L4 a second simulator | `iverilog -g2012` plus `vvp` | NOT RUN | Icarus is not installed here |
| L5 synthesis, latches, inferred memories | Yosys or OpenLane synthesis | EXECUTED inside the OpenLane container, section 2.5 | 11,015 cells on 10,938 nets in `sky130_fd_sc_hd`, 149,098 square micrometers of cell area, and the steps that look for unmapped cells, latches and suspicious assigns found nothing to stop on. Pre-layout only |
| L5 place and route, GDSII | OpenLane 2, `sh asic_chip/scripts/build_gds.sh` | EXECUTED end to end on the second and third runs of 2026-11-09, section 2.5 | floorplan, tap and endcap insertion, the power distribution network, placement, clock tree synthesis, routing, extraction, both streamouts: three GDSII files in each of the two complete runs, the largest step directory tree 1.2 GB. The first run of the same day stopped at step 25 on the pin order file |
| Signoff DRC and LVS | Magic, KLayout and netgen, through the OpenLane 2 flow | EXECUTED, section 2.5 | DRC zero violations in both tools and LVS `Circuits match uniquely` in both complete runs. The commercial Calibre flow named in the specification was never attempted. The antenna check FAILED on the run at the 8.0 ns gate, 54 pins over 46 nets, and the manufacturability report of the run at the 30.0 ns gate reads `Antenna Passed` |
| Static timing at the declared 250 MHz budget, `T_clk` 4.0 ns | OpenLane and OpenSTA on the routed netlist with the SDF of the run | NOT RUN, and now known not to be met | the layout that exists misses setup by 4.4409 ns at 8.0 ns in the worst typical corner and by 15.7205 ns in the worst slow corner, so a 4.0 ns period is about 8 ns away from what this layout does. `LL_OPENLANE_CONFIG=config.json` would select that constraint set, and nothing has ever been routed against it |
| Static timing at the bring-up gate, 8.0 ns for two runs and 30.0 ns for the third | OpenLane and OpenSTA, nine corners | EXECUTED at 8.0 ns and at 30.0 ns, NOT MEASURED at 32.0 ns | section 2.5 holds both corner by corner tables. At the 8.0 ns gate: worst setup slack -15.7205 ns, worst hold slack -0.1700 ns on 41 input to register paths. At the 30.0 ns gate: worst setup slack -0.4906 ns in one corner, worst hold slack -0.1540 ns on 45 violations spread over seven corners, and `Antenna Passed`, `LVS Passed`, `DRC Passed`. The gate is 32.0 ns now, with 0.40 ns of hold margin asked of the resizers, and that rerun has not happened |
| Power, dynamic and static | OpenLane plus an activity annotated power run | NOT MEASURED as a rating | the flow wrote a per corner `report_power` estimate, 1.507948e-01 W in `nom_tt_025C_1v80` at the 8.0 ns gate and 3.926652e-02 W at the 30.0 ns gate, both under the default activity model of the tool with no SDF back-annotation and no value change dump behind it. The Verilator run of section 2.4 produced `asic_chip/sim/lingologos.vcd` (248,815 bytes, 17,359 lines) that such a run could read, and the figures in datasheet section 5.2 are stimulus statistics of the vector file, not currents |
| Area, wire length, utilization | OpenLane reports | MEASURED on both layouts that exist, section 2.5 | at the 30.0 ns gate: die 393,012, core 371,917, standard-cell area 204,806 square micrometers, utilization 0.550678, 34,149 instances, 148 I/O, 660,126 micrometers of routed wire over 134,302 vias. At the 8.0 ns gate: die 350,447, core 330,268, standard-cell area 169,306, utilization 0.512633, 17,552 instances, longest net 1,529.65 micrometers |
| Static IR drop | OpenROAD, the `openroad-irdropreport` step of each run | EXECUTED twice, section 2.5 | at the 8.0 ns gate, worst drop 0.0135214 V on VPWR with 1.78648 V left and 0.00820593 V on VGND; at the 30.0 ns gate, worst drop 0.00336923 V on VPWR with 1.79663 V left and 0.00378131 V on VGND |
| ERC with `cvc` | `cvc` | NOT RUN, and not runnable through this flow | OpenLane 2.3.10 has removed the step, `openlane/config/removals.py` line 41, and the `RUN_CVC` key that asked for it is out of both configuration files |
| `datasheet_lingologos.pdf` | a PDF toolchain | NOT GENERATED | no PDF toolchain in this checkout; `datasheet_lingologos.md` is the source of record |

The clock figure that appears across the ASIC tree is a **declared budget**, not a measured one:
`openlane/lingologos.sdc` sets a 4.0 ns period and the same value is written in `config.json` and in
the datasheet. What a run has now measured is that this layout does not meet it. The pair
`config.bringup.json` and `lingologos.bringup.sdc`, which `build_gds.sh` uses by default, exists so
that the flow is held to a period it can actually close and reports the slack it really has; that
gate was 8.0 ns for the first two runs of 2026-11-09, 30.0 ns for the third and is 32.0 ns now,
because the slack those runs reported says the slow corner needed about 23.72 ns in the second and
about 30.49 ns in the third. `LL_OPENLANE_CONFIG=config.json`, or the
`--spec` flag of the setup scripts, selects the declared 4.0 ns budget, and nothing has ever been
routed against it. Neither number is a rating of the part: what is measured is a worst slack at a
stated period and corner, section 2.5 is that table, and the highest frequency this tree claims for
lingologos_top is none.

One sentence of this report used to say that a run at the declared period would stop before writing
a GDSII "because `QUIT_ON_TIMING_VIOLATIONS` is true". That key does not exist in OpenLane 2.3.10 and
never gated anything, and the claim was wrong in its direction as well: the run of 2026-11-09 wrote
its GDSII files first and then reported the violations, and what decided error or warning was the
corner list `TIMING_VIOLATION_CORNERS = ["*tt*"]` that OpenLane installs for a sky130 PDK. Both
configuration files now set `SETUP_VIOLATION_CORNERS` and `HOLD_VIOLATION_CORNERS` to `["*"]`, so a
violation in any corner fails the run, and both files have been stripped of the keys that do nothing
in this version.

## 4. Defects and document errors the verification layer actually caught

Items 1 to 6 were caught by the Python gates, items 7 to 12 by the lint of section 2.4 and
items 13 to 16 by the two simulation paths of section 2.4. Items 17 to 20 were caught by the
first real place and route attempt of section 2.5, and item 17 in particular is a mistake no
gate of this tree could see, because it is about what the tool expects rather than about what
the design computes. All of them are fixed in this checkout. Items 1 to 6 and 14 to 16 have a
Python test that fails if they come back; items 7 to 13 stop the lint or the build of the same
flow, so they cannot come back unnoticed either.

1. **Wrong boolean identity in the reference model, found by cross-checking.** The model
   built the clause mask for `MUX` as `((set ^ clear) & select) | clear`. The correct
   identity is `((set ^ clear) & select) ^ clear`. With `OR` the model was wrong for some
   inputs, and gate 5 reported the disagreement as `row 2 clause lanes: file 5, model
   0xD`. The generator and the RTL were right; the reference model was the bug. It stayed
   hidden because the self test byte `0x5A` happens to be invariant under the wrong
   identity and because the AND, OR and XOR tables are symmetric. The dedicated
   regression test is `test_the_reference_model_mux_semantics_are_the_documented_ones`.
2. **Datasheet quoted the wrong combinational depth for one clause.** Lane 1,
   `XOR(A, C) MUX B`, was listed as depth 1. The generated netlist is 2 levels deep for
   that lane. Gate 4 now measures depth on the wire dependency graph of the generated
   module, which is what a synthesis tool would see, and compares it with the datasheet
   table and with `config.json`.
3. **Datasheet quoted a stale artifact digest.** The digest printed for
   `artifacts/gf_rules.json` was an older value. Gate 13 now recomputes and compares all
   three artifact digests quoted in the datasheet.
4. **Line ending drift broke digest reproducibility.** The generated files were written
   with platform line endings, so the digests in `MANIFEST.json`, computed over the text
   the generator held in memory, did not match the bytes on disk. The generator now writes
   LF explicitly, and gate 7 digests the raw file bytes, so `sha256sum asic_chip/rtl/generated/*.sv`
   reproduces the manifest.
5. **Switching activity figures did not match the committed vectors.** The datasheet said
   3.488 toggles per transition on `gf_next_state` and 1.350 on `smt_results`, from an
   ad hoc script. Recomputing over the 303 transitions of `sim/golden_vectors.mem` gives
   3.492 and 1.353, and the derived fractions are 0.4827, 0.4365 and 0.3383. The datasheet
   now carries the recomputed numbers, and gate 13 compares the whole table.
6. **Scratch files left in the flow directory.** Two temporary files used while assembling
   the firmware had been written next to the generator and were removed; gate 15 asserts
   that the generated file set matches the declared set exactly.
7. **A part select of a parenthesized expression, which is not legal.** The branch and jump
   cone of `rtl/core/rv32e_core.sv` built the next program counter as
   `(pc_ext + 32'd4)[IMEM_ADDR_BITS-1:0]`. The elaboration stopped with `syntax error,
   unexpected '['` at three lines of that file. The sum now goes into a named 32 bit signal
   and the slice is taken of the signal.
8. **Two comments were read as directives.** Two comments explained how the lint tool reacts
   to comments, and began with the very word the tool looks for, so the tool parsed them as
   pragma text and failed the build with `BADVLTPRAGMA`. Both are now written as labelled
   usage lines that do not start with that word.
9. **Silent widening in the generated mesh, reported under `-Wall`.** Four signature
   extension wires widen `$signed(in_vec_c)` from 8 to 20 bits. The widening is the intent,
   but a flow that runs with `-Wall` has to say so on purpose, so the generator now wraps
   exactly those four declarations in a documented width lint pair.
10. **A combinational loop through the register file.** The register file offered a write
    through read path, which closed a loop `reg_wdata` to `rdata_b` to `operand_b` back to
    `reg_wdata`, and the lint reported the loop twice as `UNOPTFLAT`. The write through is
    not needed by this core, which completes the write on the clock edge before the next
    instruction reads, so it was removed. x0 still reads zero because the write is guarded
    by `waddr != 5'd0`, and the read side is now a plain indexed read.
11. **Blocking writes inside a sequential block.** The three store paths of the core, byte,
    half word and word, wrote `data_ram` with blocking assignments inside `always_ff`,
    reported as `BLKSEQ`. They are non blocking now.
12. **A dangling port at the top level.** The core instance left its instruction observation
    output unconnected, `PINCONNECTEMPTY`. The wire is named and connected, so the port can
    be probed and is not silently absent.
13. **A testbench that would not elaborate.** The sign extension helper derived its
    replication count and its bit range from a run time parameter, which the language
    requires to be an elaboration constant; the same file compared 20 bit pads against 32 bit
    results five times, and its clock had both a declaration level initializer and a
    procedural writer, which the lint reports as `PROCASSINIT`. All three are fixed.
14. **The testbench checked 1024 rows instead of 304.** It counted the rows of the vector
    file by scanning for an unused sentinel word, but the memory loader fills the tail of a
    partially covered array, so the scan never stopped. Rows 304 to 1023 were compared
    against zeros and passed vacuously, which is exactly the kind of quiet coverage loss a
    golden vector run is supposed to avoid. The row count is now a named parameter and gate
    9 fails when that parameter and the number of data lines in the file disagree.
15. **The checker sampled the registered outputs one delta too early.** The vector file says
    the mesh columns and the automaton state are sampled one clock edge after the row is
    driven, but a process that wakes up on that edge reads the value the register held
    before the edge. Every row of phase A reported the previous row's values. The checker
    now settles one time unit after the edge, which is the documented sampling point.
16. **The coprocessor counter expectation was off by one, in both verification paths.** The
    status word the firmware reads into x13 carries the counter as it stood when that
    request was decoded, so its low nibble is 3, while the counter register has also counted
    the status request itself and reads 4 afterwards. `sim/sim_main.cpp` and
    `sim/testbench.sv` both compared the live counter with 3, and the run reported
    `got 0x4 want 0x3`. The RTL is self consistent; the checkers were wrong. The self test
    memory now carries its own key `exp_cop_count`, gate 8 asserts it equals the counter
    field of x13 plus one, and gate 9 compares it with the literal each checker uses.
17. **The pin order file was written for a different tool.** The tree shipped
    `openlane/pin_order.cfg` the way OpenLane 1 reads it, a comment banner followed by
    `left <glob>` rows. OpenLane 2 parses that file with a pin constraint grammar that has no
    comment token at all, matches anchored regular expressions against the **bit level** pin names
    of the netlist, and cannot lex a dash inside a pattern. The first run of the flow reached the
    step that reads it and failed there with `line 1:2 token recognition error at: '-'`, exit 65,
    after 24 steps of work and a PDK download. The Python gate that checks the pinout could not see
    this: it parsed the file with its own parser and compared the result with the RTL port names,
    not with the names the tool produces from them. The file is now written in the grammar of the
    installed tool, and that gate expands the RTL buses into bit level names, anchors every pattern
    the way the step does, rejects characters the grammar cannot lex and fails on a pin left
    unplaced or a pattern that matches nothing, so this class of mistake is caught before a run.
18. **The setup script called a volare command line that does not exist.** `volare --global search
    sky130` is the spelling of an older guide; volare 0.20.6 has no `search` command and no
    `--global` option, and the script sent its stderr to `/dev/null`, so what reached the user was
    a complaint about the network. The calls are `ls-remote`, `fetch` and `enable` now, each taking
    `--pdk-root` and `--pdk`, and the tool output is left visible.
19. **The setup script assumed `openlane` could start.** `openlane` 2.3.10 imports `tkinter` during
    its startup, in `openlane/common/tcl.py`, and Debian and Ubuntu ship that module apart from the
    interpreter. On this distribution the import failed in an environment where `openlane` itself
    was installed correctly, and the script found that out only after the PDK download. It installs
    the `tk` package of the interpreter it selects and tests the import first now.
20. **The PDK fetch asked for fourteen times the libraries the flow reads.** The fetch passed `-l
    all`, which measured 16 GiB on disk against 2.1 GiB for the set the place and route of this
    design uses. The default is volare's own set now, and `LL_PDK_LIBRARIES` names a larger one on
    purpose.

21. **An OpenLane 1 knob silently disabled the antenna repair this design needs.** Both
    configuration files carried `DIODE_INSERTION_STRATEGY: 3`, taken from the OpenLane 1 guides.
    `openlane/config/config.py` lines 918 to 943 accepts that key, deletes it, and derives the native
    knobs from it; for value 3 the derivation sets `RUN_HEURISTIC_DIODE_INSERTION` false and
    `DIODE_ON_PORTS` to `none`, which is the opposite of what the same files were asking for two
    lines away. The flow log of the complete run says twice `Gating variable ... set to 'False'`, two
    diode steps were skipped, and the signoff summary reported 54 antenna pin violations on 46 nets
    afterwards. A key that looks like configuration and behaves like a reset of other configuration
    is the defect; the legacy key is gone and the native knobs are written directly.
22. **Three configuration keys were load bearing in no version at all.** `QUIT_ON_TIMING_VIOLATIONS`
    does not exist in OpenLane 2.3.10, `RUN_CVC` names a step upstream removed, and
    `QUIT_ON_LINTER_ERRORS` is a deprecated spelling of `ERROR_ON_LINTER_ERRORS`. OpenLane warns
    about each and continues, so the tree carried for months a control it believed it had: the
    sentence in this report that a run at the declared period would stop before writing a GDSII
    "because `QUIT_ON_TIMING_VIOLATIONS` is true" described a key that was never read. The real gate
    is a corner list, `TIMING_VIOLATION_CORNERS`, and OpenLane sets it to `["*tt*"]` for a sky130
    PDK in `openlane/config/pdk_compat.py` line 231, which is why the complete run failed on the
    typical corners and treated the slow corners, four times worse, as warnings.
23. **The bring-up gate was a guess, and the runs measured it.** 8.0 ns was chosen before any
    place and route existed, as a period a first run might close. The layout of the second run needed
    about 12.44 ns in the typical corner and about 23.72 ns in the slow one; the layout of the third,
    which paid for the antenna repair the second run failed, needed about 15.60 ns and about
    30.49 ns. The gate is 32.0 ns now, and it is still a choice about what the flow is held to rather
    than a figure of the part, which is how the datasheet labels it.
24. **Hold can be repaired and then lost again, because nothing repairs it after the last routing.**
    In both complete runs the resizer closed hold where it runs, and signoff reported violations on
    the same paths: at the 30.0 ns gate the metrics of the timing step that follows
    `OpenROAD.ResizerTimingPostGRT` read +0.100201 ns of hold slack with zero violations in
    `nom_tt_025C_1v80`, while that corner reads -0.058548 ns at signoff on the SPEF extracted from the
    final routing. The last hold repair lives in `openlane/scripts/openroad/rsz_timing_postgrt.tcl`,
    which re-runs the global router and estimates parasitics from that routing rather than from the
    real one, and it repairs only up to `GRT_RESIZER_HOLD_SLACK_MARGIN`, an installed default of
    0.05 ns. A design repaired to a few hundredths of a nanosecond does not survive extraction. The
    tree answers with a margin, `PL_RESIZER_HOLD_SLACK_MARGIN` and `GRT_RESIZER_HOLD_SLACK_MARGIN` at
    0.40 ns in both configuration files, and with a constraint note: the violations are all
    `[hold in-reg]`, input ports against a clock tree that inserted 2.1845 ns, against an SDC that
    promises 0.30 ns of minimum input delay.

Items 7, 8, 11 and 12 changed no behaviour of the design. Item 9 states an intended widening
that was previously silent, item 10 removed a read path the core never needed, and item 13
to 16 changed the verification side, where 14, 15 and 16 would each have quietly weakened
level L4 had they stayed. Of 17 to 20, item 17 is the one that stopped a run, 18 and 19 stopped
a setup, and 20 cost only disk. Of 21 to 24, 21 and 22 are the ones that made a completed run
report failures the configuration believed it had prevented, 23 replaced an unmeasured gate with a
measured one, and 24 names the mechanism that left both complete runs failing hold.

Two further notes that are not defects but are easy to overread:

- The illegal instruction path is proven by the emulator and now by simulation as well.
  Firmware word 79 encodes `ADDI x1, x16, 0`; `register_space_ok` in `rtl/core/rv32e_core.sv`
  rejects any 5 bit register field of 16 or above when that field is used; the sticky
  `core_illegal_seen` flag reaches bit 4 of the coprocessor status word, which the firmware
  reads into `x13`. Both simulation paths of section 2.4 assert the flag and the resulting
  `x13`, and gate F3 of the Python suite checks that the probe slot is the only illegal
  encoding in the ROM.
- The GF tracker in `artifacts/gf_rules.json` cannot separate round from square brackets:
  both use the same two multipliers, so the machine recognizes the commutative image of
  Dyck-2 and accepts `(]`. This is a consequence of the one byte state budget, and it is
  stated in both the datasheet and the artifact.

## 5. Traceability against the specification validation levels

| Level in the specification | What it asks | Status here |
|---|---|---|
| L0 | artifact and schema validity | EXECUTED, gates 1, 3, 13, 15 |
| L1 | generator self gates H1 to H5 | EXECUTED, section 2.1 |
| L2 | reference model agreement on every row and byte | EXECUTED, gates 2, 4, 5, 8 |
| L3 | reproducibility and digests | EXECUTED, gates 6, 7 |
| L4 | HDL simulation against the golden vectors | EXECUTED in the container of section 2.4, Verilator 5.054: 304 rows plus the core self test, 0 mismatches, in two independent paths. NOT RUN on the host and NOT RUN with a second simulator |
| L5 | synthesis, timing, power, area, GDSII | Synthesis, floorplan, placement, clock tree synthesis, routing, extraction, streamout, DRC, LVS and static timing EXECUTED on the two complete runs of section 2.5: three GDSII files each, DRC zero violations in both tools, LVS `Circuits match uniquely`, area and utilization measured, static timing reported over nine corners at 8.0 ns and at 30.0 ns. Both runs exited non zero on setup and hold violations; the antenna check FAILED on the first of them and passed on the second. Power as a rating, and timing at the declared 4.0 ns budget, NOT RUN and NOT MEASURED |

## 6. What a reviewer must still do before trusting the silicon claims

1. Section 2.4 ran the lint and both simulation paths in the container image
   `verilator/verilator:latest`. Repeat them on a host install of Verilator 5.006 or later
   with `make -f asic_chip/Makefile lint`, `sim-verilator` and `sim-tb`, so the result does
   not depend on a container. Any failure there invalidates section 4.3 of the datasheet.
2. Install Icarus Verilog and run `sim-iverilog` to get a second simulator opinion.
3. Rerun the place and route that already ran end to end, twice. From the copy inside the distribution,
   `sh asic_chip/scripts/build_gds.sh`; from Windows, the PowerShell command
   `asic_chip\scripts\setup_openlane_flow.ps1 -Copy`, which installs what is missing and then
   runs the flow in one command. Those runs did leave a GDSII, DRC reports, an LVS report and a corner
   by corner timing table, and the datasheet has taken those numbers with the period and corner they
   were measured at. What is still owed is a run that closes: the 32.0 ns gate, the 0.40 ns hold
   margins of section 2.5, exit status zero and nothing in the setup and hold checkers of `error.log`.
   The antenna check was clean on the third run. After that, and not before, someone should take the
   declared 4.0 ns budget through place and route knowing that the layout that exists today is about
   30.5 ns away from it in the slow corner.
4. Re-run `python asic_chip/scripts/lingologos_compiler.py --check` after any artifact
   edit; a stale `rtl/generated` directory stops the OpenLane script by design.
5. Record the run identifiers, tool versions and the exact VCD used for power in this
   report, in the same wording, so a later reader can tell executed from declared.
