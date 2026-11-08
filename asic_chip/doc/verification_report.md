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
| NOT RUN / NOT MEASURED | The tool that would produce this number is not installed here, and the run has not run in this checkout. |

Nothing in this report claims a post-layout number, a timing number, a power number or
a gate count. Those need OpenLane and a signoff tool, and neither was run.

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
| OpenLane, sky130A PDK | NOT INSTALLED | `asic_chip/scripts/build_gds.sh` reports `[FAIL] neither flow.tcl nor openlane was found on PATH` and exits 127 |

Consequence, stated plainly: **the RTL has now been linted and simulated cycle by cycle by
a real simulator, Verilator 5.054, inside a container image.** What has not been run here
is synthesis, place and route, static timing, power and area extraction, and a second
simulator. Everything marked EXECUTED below was executed either by Python on the host or by
the commands of section 2.4 in the container.

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

`python -m pytest -v tests/` was run after the last change. The full suite reports 181
tests, all PASSED, zero failures, zero errors; the last run printed `181 passed in 7.36s`.
Of those, 23 tests
target `asic_chip/`: 17 in `tests/test_asic_compiler.py` and 6 in
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


## 3. What was not run

This is the part that a reader must not infer past.

| Validation level | Intended tool | Status | Why |
|---|---|---|---|
| L4 lint and latch inference, on the host, through `make` | `verilator --lint-only` via `make -f asic_chip/Makefile lint` | NOT RUN on this host | neither Verilator nor `make` is on the host PATH; the same command line ran in a container, section 2.4 |
| L4 cycle accurate run of the 304 golden rows against Verilog | Verilator with `sim/sim_main.cpp` | EXECUTED in the container, section 2.4 | 304 rows and the self test, 0 mismatches |
| L4 the same run through a SystemVerilog testbench | `verilator --binary --timing` | EXECUTED in the container, section 2.4 | same checks, same verdict |
| L4 a second simulator | `iverilog -g2012` plus `vvp` | NOT RUN | Icarus is not installed here |
| L5 synthesis, latches, inferred memories | Yosys or OpenLane synthesis | NOT RUN | not installed |
| L5 place and route, GDSII | OpenLane 1 or 2, `sh asic_chip/scripts/build_gds.sh` | NOT RUN | neither `flow.tcl` nor `openlane` is on `PATH`; the script exits 127 and says so |
| Static timing at 250 MHz, `T_clk` 4.0 ns | OpenLane and OpenSTA on the routed netlist with `back-annotated SDF` | NOT RUN, NOT MEASURED | no netlist exists |
| Power, dynamic and static | OpenLane plus `cvc` and a VCD at the declared operating point | NOT MEASURED | no layout and no extraction was run; the Verilator run of section 2.4 produced `asic_chip/sim/lingologos.vcd` (248,815 bytes, 17,359 lines) that such a tool could read, and the figures in datasheet section 5.2 are stimulus statistics of the vector file, not currents |
| Area, wire length, utilization | OpenLane reports | NOT MEASURED | no layout |
| ERC and IR drop | `cvc` | NOT RUN | not installed |
| Signoff DRC and LVS | Calibre or equivalent | NOT RUN | commercial tool, never attempted |
| `datasheet_lingologos.pdf` | a PDF toolchain | NOT GENERATED | no PDF toolchain in this checkout; `datasheet_lingologos.md` is the source of record |

The clock figure that appears across the ASIC tree is a **declared budget**, not a
measured one: `openlane/lingologos.sdc` sets a 4.0 ns period and the same value is written
in `config.json` and in the datasheet. Whether the design meets it is unknown until a
synthesis and timing run exists.

## 4. Defects and document errors the verification layer actually caught

Items 1 to 6 were caught by the Python gates, items 7 to 12 by the lint of section 2.4 and
items 13 to 16 by the two simulation paths of section 2.4. All of them are fixed in this
checkout. Items 1 to 6 and 14 to 16 have a Python test that fails if they come back; items 7
to 13 stop the lint or the build of the same flow, so they cannot come back unnoticed either.

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

Items 7, 8, 11 and 12 changed no behaviour of the design. Item 9 states an intended widening
that was previously silent, item 10 removed a read path the core never needed, and item 13
to 16 changed the verification side, where 14, 15 and 16 would each have quietly weakened
level L4 had they stayed.

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
| L5 | synthesis, timing, power, area, GDSII | NOT RUN, no OpenLane |

## 6. What a reviewer must still do before trusting the silicon claims

1. Section 2.4 ran the lint and both simulation paths in the container image
   `verilator/verilator:latest`. Repeat them on a host install of Verilator 5.006 or later
   with `make -f asic_chip/Makefile lint`, `sim-verilator` and `sim-tb`, so the result does
   not depend on a container. Any failure there invalidates section 4.3 of the datasheet.
2. Install Icarus Verilog and run `sim-iverilog` to get a second simulator opinion.
3. Install OpenLane with the sky130A PDK, run `asic_chip/scripts/build_gds.sh`, and only
   then fill in area, timing and power in the datasheet, replacing every `NOT MEASURED`
   cell with the report value and its run directory.
4. Re-run `python asic_chip/scripts/lingologos_compiler.py --check` after any artifact
   edit; a stale `rtl/generated` directory stops the OpenLane script by design.
5. Record the run identifiers, tool versions and the exact VCD used for power in this
   report, in the same wording, so a later reader can tell executed from declared.
