# LingoLOGOS ASIC datasheet

Design name: `lingologos_top`.
Target library: SkyWater sky130A, standard cell library `sky130_fd_sc_hd`.
Declared core frequency: 250 MHz (T_clk = 4.0 ns).
Status of this document: **design description, not a characterization report.** Every
number in a table labelled measured was produced by the generator
`asic_chip/scripts/lingologos_compiler.py` or by the reference interpreter
`tests/asic_reference_model.py` in this checkout. Every entry in a section headed
"NOT MEASURED" is a declared design target or a tool estimate whose basis is named where it
appears, and no silicon backs it. One place and route run with its extraction, signoff checks and
static timing report exists, from 2026-11-09, and this document quotes it only with the period and
corner it was measured at. See `verification_report.md` for the exact list of what has been run and
what has not.

---

## 1. General description

LingoLOGOS is a mixed function core that pairs a reduced RISC-V processor with three
fixed-function hardware engines:

| Engine | Role | Clock cost |
|---|---|---|
| Weight ROM mesh | 4 by 4 signed INT8 matrix multiply, weights are constants in the netlist | combinational cone, registered output, 1 cycle latency |
| GF(2^8) automaton | bracket structure tracker over the field defined by 0x11D | one state update per clock |
| SMT evaluator | four Boolean clauses lifted from the constraint language | combinational, 0 cycles, valid while the input is held |
| RV32E core | brings up and self-tests the three engines through a custom instruction port | one instruction per clock |

The three engines are always live. The core does not gate them; it reads them back
through a coprocessor port. The design is intended for a sensor-class part that
classifies bracket-structured event streams without software.

### 1.1 Top level identifiers

| Property | Value | Source |
|---|---|---|
| Top module | `lingologos_top` | `asic_chip/rtl/top/lingologos_top.sv` |
| User signal count | 18 ports, 51 input bits, 95 output bits, 146 bits total | parsed from the RTL, checked by `tests/test_asic_compiler.py` |
| Debug ports | 12 ports, 302 bits, present only with `LL_DEBUG_PORTS` defined | same |
| Package declared | QFN-160, 146 signal bits plus 14 power pads | `asic_chip/config.json` |
| Clock period declared | 4.0 ns | `asic_chip/openlane/lingologos.sdc` |
| Reset | `rst_n`, active low, asynchronous assert, released before `core_start` | top level |

### 1.2 Package note, measured against the specification

Section 7 of the specification offers a QFN-48 or QFN-64 package. The interface the same
specification requires does not fit either package: the top level needs 146 user signal
bits, and a QFN-64 offers at most 64 pads. This design therefore declares QFN-160 and
states the discrepancy in `asic_chip/config.json` instead of shrinking the interface. The
pad assignment in section 3 follows the placement intent in
`asic_chip/openlane/pin_order.cfg`; it is a project artifact, not a taped-out pad list.

---

## 2. Timing

### 2.1 Declared clock target, and what one routed layout measured against it

| Item | Declared value | Basis |
|---|---|---|
| `clk` frequency | 250 MHz | specification section 5 |
| T_clk | 4.0 ns | `create_clock -name clk -period 4.0` |
| Setup uncertainty | 0.15 ns propagated, 0.05 ns drive | constraint file |
| Clock transition | 0.25 ns | constraint file |
| Input delay budget | 0.80 ns | constraint file |
| Output delay budget | 0.30 ns | constraint file |
| Driving cell | `sky130_fd_sc_hd__buf_4` on pin X | constraint file |
| Output load | 0.030 pF, 0.025 pF on the four input lanes | constraint file |
| `rst_n` | false path to the clock domain | constraint file |
| Bring-up gate, the period the flow is held to | 32.0 ns, one period, not a product figure | `asic_chip/openlane/config.bringup.json` and `lingologos.bringup.sdc` |

**The 250 MHz target is declared, and it is now known not to be met by the layouts that exist.**
Synthesis ran on 2026-11-09, as step 06 of the OpenLane 2 flow recorded in
`asic_chip/doc/verification_report.md`: the netlist mapped into `sky130_fd_sc_hd` holds 11,015
cells on 10,938 nets, and its standard-cell area is 149,098 square micrometers, 0.149 square
millimeters. That is a pre-layout figure. The same machine then ran the flow twice more on
2026-11-09, both times through all 78 steps of the Classic flow, and both runs produced a clock tree,
a route, a parasitic extraction and a static timing report: one at the 8.0 ns gate in force at the time
and one at the 30.0 ns gate that replaced it. The numbers those two runs reported are the only timing
numbers this datasheet can stand behind:

| Corner | Worst setup slack at 8.0 ns | Worst hold slack at 8.0 ns | Worst setup slack at 30.0 ns | Worst hold slack at 30.0 ns |
|---|---|---|---|---|
| `nom_tt_025C_1v80` | -4.1255 ns | -0.0431 ns | +14.9556 ns | -0.0585 ns |
| `max_tt_025C_1v80` | -4.4409 ns, 2,485 violating paths | -0.0532 ns | +14.4003 ns | -0.0989 ns |
| `min_ss_100C_1v60` | -14.5466 ns | -0.0828 ns | +1.1885 ns | -0.1067 ns |
| `max_ss_100C_1v60` | -15.7205 ns, total negative slack about -23,845 ns | -0.1700 ns, 41 violating paths | -0.4906 ns, total negative slack -13.5139 ns, 35 violating paths | -0.1540 ns, 10 violating paths |
| `ff_n40C_1v95`, all three | positive, +0.0880 ns at best | positive | positive, +20.1103 ns at worst | +0.0536 ns at best, -0.0024 ns in `max_ff_n40C_1v95` |

The layout of the second run closes at roughly 12.4 ns in the typical corner and roughly 23.7 ns in the
slow corner, about 80 MHz and 42 MHz. The layout of the third closes at roughly 15.6 ns and roughly
30.5 ns, about 64 MHz and 33 MHz. Neither closes at 4.0 ns: that period is about 8.4 ns away from what
the first layout reached in the typical corner and about 26.5 ns away from what the second one reached in
the slow corner, the tightest pair in the table. The hold failures were 41 paths in the second run and
45 violations spread over seven corners in the third, every one of them from an input port to a register
against the 1.2 to 1.35 ns of clock insertion delay the first clock tree produced and the 2.1845 ns of
the second, and none register to register. No maximum operating frequency of the part is claimed here:
what is measured is a worst slack at a stated period and corner, and the 4.0 ns set was never taken
through place and route. 250 MHz on sky130 for a design whose longest combinational cone runs through a
4 by 4 multiply-accumulate and a 1020 entry case table was an optimistic target, and on these layouts it
is a target that was tried and missed. The bring-up gate is now 32.0 ns, above the 30.49 ns the slow
corner of the third run showed it needed, and both resizers are now asked for 0.40 ns of hold slack
instead of the 0.05 ns OpenLane defaults to: that run measured hold closed at +0.100201 ns where the
resizer runs and -0.058548 ns at signoff in the same corner, so detailed routing and extraction took
about 0.16 ns out of those paths and nothing repairs hold after them.

### 2.2 Latency of each output, as implemented

| Output | Cycles after the input is driven | Nature |
|---|---|---|
| `out_vector_0..3` | 1 | mesh cone is combinational, the top level registers the result |
| `gf_next_state` | 1 | state register, updated every clock |
| `smt_results` | 0 | combinational, valid while `smt_in_bits` is held |
| `core_done`, `core_pass` | self-test dependent | 89 cycles counted by the reference interpreter, 91 counted by the harness itself when the run was executed |
| `core_illegal_seen` | sticky | set by the first illegal encoding, never cleared |

The self-test cycle count is 89 as counted by the reference interpreter and 91 as counted by
the C++ harness when it ran, section 2.4 of `verification_report.md`; the two counters start
and stop on different events. Inside the harness the wait happens against a 400 cycle bound,
which covers both.

### 2.3 Cycle by cycle contract the harnesses rely on

1. Inputs change while `clk` is low.
2. `smt_results` is correct in that same half cycle, combinational.
3. On the following rising edge, `out_vector_*` take the values computed from the inputs
   of that cycle, and `gf_next_state` takes the state produced by the symbol of that
   cycle.
4. `gf_next_state` therefore reports the state of the register, not the next state the
   automaton cone is currently computing.

Both `asic_chip/sim/sim_main.cpp` and `asic_chip/sim/testbench.sv` implement exactly this
contract, and `tests/test_asic_compiler.py` checks that both agree with
`asic_chip/sim/golden_vectors.mem`.

---

## 3. Pinout, QFN-160

Numbering runs counter-clockwise from pin 1 on the left side, which is the order the
OpenLane 1 pin placer walks the die. Bits are enumerated from the least significant bit
upward, which is the order the netlist enumerates a bus.

| Pins | Side | Signal | Width | Direction | Notes |
|---|---|---|---|---|---|
| 1-8 | left | in_vector_0 | 8 | input | INT8, two's complement |
| 9-16 | left | in_vector_1 | 8 | input | INT8, two's complement |
| 17-24 | left | in_vector_2 | 8 | input | INT8, two's complement |
| 25-32 | left | in_vector_3 | 8 | input | INT8, two's complement |
| 33-40 | left | gf_symbol_in | 8 | input | raw symbol byte fed to the automaton every cycle |
| 41-60 | top | out_vector_0 | 20 | output | signed accumulator, row 0 |
| 61-80 | top | out_vector_1 | 20 | output | signed accumulator, row 1 |
| 81-100 | right | out_vector_2 | 20 | output | signed accumulator, row 2 |
| 101-120 | right | out_vector_3 | 20 | output | signed accumulator, row 3 |
| 121-128 | bottom | smt_in_bits | 8 | input | predicate variables A to H, bit 0 is A |
| 129-136 | bottom | gf_next_state | 8 | output | automaton state register |
| 137-140 | bottom | smt_results | 4 | output | clause lane 0 in bit 0 |
| 141 | bottom | clk | 1 | input | 250 MHz declared |
| 142 | bottom | rst_n | 1 | input | active low |
| 143 | bottom | core_start | 1 | input | pulse to start the built-in self-test |
| 144 | bottom | core_done | 1 | output | high after the halt instruction |
| 145 | bottom | core_pass | 1 | output | self-test verdict, high on pass |
| 146 | bottom | core_illegal_seen | 1 | output | sticky illegal encoding flag |
| 147-153 | bottom | VDD | 7 | power | core supply pads |
| 154-160 | bottom | VSS | 7 | power | core ground pads |

Signal bits accounted for: 40 on the left, 40 on the top, 40 on the right, 26 on the
bottom, 146 total, matching the port list of the RTL. Power pads fill the remaining 14
slots of the 160 pin frame. Debug ports are not bonded: they exist only in the simulation
build.

### 3.1 Debug ports, simulation only

`dbg_core_running` (1), `dbg_core_pc` (9), `dbg_cop_count` (4), `dbg_cop_last_rdata` (32),
`dbg_reg_x7`, `dbg_reg_x8`, `dbg_reg_x10`, `dbg_reg_x11`, `dbg_reg_x12`, `dbg_reg_x13`,
`dbg_reg_x14`, `dbg_reg_x15` (32 each). 12 ports, 302 bits. They are wrapped in
``` `ifdef LL_DEBUG_PORTS ``` and the synthesis flow passes an empty define list, so a
synthesized netlist never contains them.

---

## 4. Signal description and coprocessor contract

### 4.1 Weight mesh

Weights are the fixed reference matrix of `asic_chip/artifacts/model_weights.json`,
emitted as constants in the netlist through non-adjacent form shift-add terms. The mesh
is not a trainable array and it does not hold the quantized weights of the trained model;
see section 8.

| Property | Measured value |
|---|---|
| Dimensions | 4 by 4 |
| Input precision | INT8, range -128 to 127 |
| Accumulator width | 20 bits, signed range -524288 to 524287 |
| Largest shift used | 2^7 |
| Shift-add terms in the netlist | 25 across the four rows |
| Largest exact output magnitude over all input combinations | 24960 |
| Conservative bound used by gate H1 | 35072 |
| Headroom ratio used by gate H1 | 14.95 |

Mesh example, the frozen self-test stimulus `in_vector = (10, -5, 2, 100)`:

| Row | Measured value | Hex as it appears on the pads |
|---|---|---|
| 0 | 9245 | `0241D` |
| 1 | -1606 | `FF9BA` |
| 2 | 4170 | `0104A` |
| 3 | -925 | `FFC63` |

Saturation corners from the vector file: all four inputs at +127 gives
`01BC8 FDEC3 FEFA1 02EA2`, all four at -128 gives `FE400 02180 01080 FD100`. Both stay
far inside the 20 bit range.

### 4.2 GF(2^8) automaton

| Property | Measured value |
|---|---|
| Field polynomial | 0x11D, irreducible, proved by the gcd criterion in the reference model |
| Initial state | 0x01 |
| Reachable states with the four bracket symbols | 255, the whole nonzero field |
| Transitions written into the case table | 1020, which is 255 states times 4 symbols |
| Order of the multiplier 0x02 | 255 |
| Measured inverse of 0x02 | 0x8E |
| Default arm of the table | initial state, so any symbol outside the alphabet resets the tracker |

Symbol map: `(` = 0x28 multiplies by 0x02, `)` = 0x29 multiplies by 0x8E,
`[` = 0x5B multiplies by 0x02, `]` = 0x5D multiplies by 0x8E. The four byte values are the
terminals of `artifacts/grammar.json`.

Across the 304 rows of `golden_vectors.mem` the chain visits all 255 states, and 36
distinct symbol bytes appear, which exercises the default arm as well as the alphabet.

**Known limitation, stated in the artifact.** Because round and square brackets share the
same two multipliers, the automaton recognizes the commutative image of the Dyck-2
language: it accepts `(]` and `][`. The invariant it maintains is a field product, so it
cannot distinguish nesting order. This follows from the one-byte state budget declared in
the specification, not from a bug in the table.

### 4.3 SMT evaluator

| Property | Measured value |
|---|---|
| Input bits | 8, variables A to H mapped to bits 0 to 7 |
| Result lanes | 4, all four used, no lane tied to zero |
| Clauses | 4, from `asic_chip/artifacts/smt_constraints.json` |
| Deepest combinational cone | 3 levels |
| Node counts per clause | 3, 2, 4, 3 |
| Truth table entries per clause | 256, one per input byte |

| Lane | Expression | Nodes | Depth |
|---|---|---|---|
| 0 | `(A AND B) OR (NOT C)` | 3 | 2 |
| 1 | `XOR(A, C) MUX B` | 2 | 2 |
| 2 | `(A OR B) AND NOT (C XOR D)` | 4 | 3 |
| 3 | `MUX(E, XOR(A, B), AND(C, D))` | 3 | 2 |

Over the 304 vector rows all 256 possible input bytes appear, so each lane is exercised
over its whole domain. The observed lane word values are 0, 1, 2, 3, 5, 7, 8, 9, 10, 12,
13 and 14; four of the sixteen lane combinations are unreachable for this clause set,
which follows from the clauses and not from the netlist.

**Scope note.** These clauses are a hardware test network written in the DSL documented in
`asic_chip/artifacts/smt_constraints.json`. They are not the compiled form of the Z3
assertions that `datasets/multilingual/tools/logic_map.py` produces.

### 4.4 Coprocessor instructions

Four custom instructions plus a halt, `opcode = 0001011`, distinguished by `funct3`:

| `funct3` | Mnemonic | Value written to `rd` |
|---|---|---|
| 001 | `LLSMT rd, rs1` | the four clause lanes evaluated on the low byte of `rs1` |
| 010 | `LLGFS rd, rs1` | the state the automaton would enter on the low byte of `rs1` |
| 011 | `LLRDM rd, rs1` | the selected mesh accumulator row, row index from bits 1 to 0 |
| 100 | `LLCTL rd, rs1` | bit 4 illegal flag, bits 3 to 0 coprocessor operation count |
| 111 | `LLHALT rs1` | stops the core, `core_pass` takes bit 0 of `rs1` |

The counter read by `LLCTL` counts the custom operations retired **before** the status
read, so the self-test expects 3, not 4. The status word the self-test expects is `0x13`:
illegal flag set in bit 4, counter 3 in the low nibble.

### 4.5 Core

| Property | Value | Source |
|---|---|---|
| Instruction set | RV32E subset, 16 registers | `asic_chip/rtl/core/rv32e_core.sv` |
| Program memory | 128 words of 32 bits, initialized in the netlist | RTL `initial` block |
| Program counter | 9 bits, byte addresses | RTL and `config.json` |
| Scratch memory | 64 words of 32 bits | RTL |
| Firmware size | 97 words, slots 97 to 127 are no-operations | measured from the RTL |
| Illegal encoding probe | program slot 79, `ADDI x1, x16, 0` | `config.json` |
| Self-test length | 89 clock cycles, 89 instructions retired | reference interpreter |
| Self-test verdict | pass flag 1, error accumulator x15 = 0 | reference interpreter |
| Register readbacks | x8 = 0x00000009, x10 = 0x00000002, x11 = 0x0000104A, x13 = 0x00000013 | reference interpreter and `core_self_test.mem` |

Unimplemented and flagged illegal by the decoder: SYSTEM instructions, the 64-bit
`OP-IMM-32` and `OP-32` opcodes, `FENCE.I`, misaligned loads and stores, and any encoding
that names a register above x15 in a field the opcode actually reads. `FENCE` is accepted
and does nothing.

---

## 5. Power

**NOT MEASURED** as a characterized power figure. Synthesis, placement, routing, parasitic
extraction and a static IR drop analysis ran twice on 2026-11-09, and section 2.5 of
`asic_chip/doc/verification_report.md` records what they left, but the power numbers they produced are
estimates under the switching activity model of the tool rather than measured currents. The static
timing step of each run wrote a power report per corner. At the 8.0 ns gate the `nom_tt_025C_1v80` one
totalled 1.507948e-01 W, split 7.252268e-02 W internal, 7.827200e-02 W switching and 1.031524e-07 W
leakage, with 78.5 percent of it in combinational logic and 7.6 percent in the clock. At the 30.0 ns
gate the same corner totals 3.926652e-02 W, split 1.688897e-02 W internal, 2.237745e-02 W switching and
9.765576e-08 W leakage, with 77.2 percent combinational, 14.4 percent sequential and 8.4 percent clock.
The ratio between the two totals, 0.26, is close to the 8 over 30 = 0.27 that the longer period
predicts for a switching dominated estimate, but the two runs routed different layouts, so the pair is
not a clean measurement of frequency scaling either. Neither report had an SDF back-annotation or a
value change dump behind it: the activity in both is the default model of the tool and not the vectors
of section 5.2, and the figures are quoted as what a tool printed at a stated period, not as a figure
the part consumes. The two runs also reported a static IR drop: 0.0135214 V worst on VPWR, leaving
1.78648 V, and 0.00820593 V worst on VGND at the 8.0 ns gate; 0.00336923 V worst on VPWR, leaving
1.79663 V, and 0.00378131 V worst on VGND at the 30.0 ns gate. The design was simulated, and that run
wrote a waveform a power tool could read, `asic_chip/sim/lingologos.vcd` with 248,815 bytes, but nothing
was extracted from it. There is no milliwatt figure this datasheet stands behind as a rating, and
publishing one would be a fabrication.

### 5.1 Method that will be used once the flow is complete

The report will come from an OpenLane run that closes signoff. `asic_chip/openlane/config.json`
used to declare a `cvc` ERC and IR drop check under `RUN_CVC`; OpenLane 2.3.10 has removed that
step, the key is reported as an unknown key and the configuration no longer carries it. What the
flow does give today is the static analysis of the `openroad-irdropreport` step of a run, the
per-corner `power.rpt` of the static timing step, and the area and wire length reports of the placement
and routing steps. Power at a stated activity will be taken from an SDF annotated timing back-annotation
plus a value change dump activity file at the declared operating point, which has not been done.

### 5.2 Measured stimulus statistics available today

These are properties of the verification vectors, not of any transistor. They are recorded
here because they are the input to the future estimate, and because they are the only
activity numbers that exist in this checkout. The chain starts at the reset row of the
file and every count is taken over the 303 transitions between the 304 rows, so the unit
is one toggle on one signal bit per clock of driven stimulus.

| Signal group | Width | Mean bit toggles per transition | Fraction of bits toggling |
|---|---|---|---|
| `out_vector_0..3` | 80 | 38.617 | 0.4827 |
| `gf_next_state` | 8 | 3.492 | 0.4365 |
| `smt_results` | 4 | 1.353 | 0.3383 |

The rows of `golden_vectors.mem` are dominated by pseudorandom input, so these figures
describe a near worst-case toggle environment rather than an application profile.

### 5.3 Declared operating point, unproven

| Item | Declared | Status |
|---|---|---|
| Core supply | 1.8 V, the nominal `sky130_fd_sc_hd` rail | NOT MEASURED |
| Junction temperature range | not characterized | NOT MEASURED |
| Absolute maximum ratings | not characterized | NOT MEASURED |
| Standby current | not characterized | NOT MEASURED |

Any absolute maximum rating for a sky130 part must come from the PDK device tables and
from a completed run. The PDK is installed on the machine that runs the flow, sky130A
version `0fe599b2afb6708d281543108caf8310912f54af` in the volare tree and `c6d73a35f524070e85faff4a6a9eef49553ebc2b`
under `~/pdk`, and no characterization of the part against those tables has been done.

---

## 6. Area

**Measured for the two layouts that exist, neither of which closed signoff.** `asic_chip/openlane/config.json`
declares `FP_CORE_UTIL 40` and `PL_TARGET_DENSITY 0.45`, and the die frame implied by a QFN-160
package is a packaging choice, not a layout result. Both runs of 2026-11-09 that went through all 78
steps of the OpenLane 2 Classic flow left these figures, from the fill insertion report recorded in
`asic_chip/doc/verification_report.md`. The run at the 8.0 ns gate: a die area of 350,447 square
micrometers, a core area of 330,268 square micrometers, 169,306 square micrometers of standard-cell area
inside it, a core utilization of 0.512633, 17,552 instances and 148 I/O pins. The run at the 30.0 ns
gate: die 393,012, core 371,917, standard-cell area 204,806, core utilization 0.550678, 34,149
instances, 148 I/O pins, and 660,126 micrometers of routed wire over 134,302 vias on 11,718 nets, which
the first run never reported as a total. Before placement the mapped netlist of step 06 held 11,015
cells on 10,938 nets over 149,098 square micrometers of cell area, which counts the constant logic
synthesis made out of the program memory. The difference between 11,015 cells and the 17,552 instances
of the first run is what that flow added: 20,266 fill cells, 4,686 tap cells, 468 diodes inserted for
antenna repair, 726 timing repair buffers, 361 clock buffers and 219 clock inverters. The same classes
in the second layout, read from the metrics of its fill insertion step, are 21,734 fill, 5,311 tap,
16,898 instances of class `antenna_cell`, 325 timing repair buffers, 373 clock buffers and 227 clock
inverters. That antenna figure is the price of the check the first run failed, and it is the main reason
the die grew 12 percent between the two. The longest routed net of the first run measured 1,529.65
micrometers. None of this is a finished part: the first run reported antenna violations and the timing
figures of section 2.1 and exited non zero on them, and the second passed the antenna check, with
`Antenna Passed`, `LVS Passed` and `DRC Passed` in its own manufacturability report, and still exited
non zero on setup in one corner and hold in seven.

A structural note that is not a measurement either: the program memory is emitted as a
128 by 32 bit read only array and synthesis turns it into constant logic. A production
part would replace it with a compiled ROM macro, which `asic_chip/openlane/config.json`
records under `NOTE_ROM`.

---

## 7. Validation levels

| Level | What it proves | Run in this checkout |
|---|---|---|
| L0 | artifacts and chip configuration agree with each other | yes, `tests/test_asic_compiler.py` |
| L1 | generator gates H1 to H5 pass | yes, `--check` mode |
| L2 | an independent reference model reproduces every generated expectation | yes, 304 vector rows and the 17 key and value pairs of the core self test recomputed |
| L3 | RTL semantics reproduce the reference model | yes, by the Python interpreter and by the Verilator run of section 2.4 of `verification_report.md`, which reproduces every column of every row |
| L4 | Verilator runs the C++ harness and the SystemVerilog testbench | yes, in the container image `verilator/verilator:latest`, Verilator 5.054: lint clean under `-Wall`, 304 rows plus the core self test with 0 mismatches in both paths. NOT RUN on the host, NOT RUN with Icarus |
| L5 | synthesis, floorplan, placement, route, DRC, LVS, STA | partly, on 2026-11-09: two OpenLane 2 runs went through all 78 steps of the Classic flow, each produced a GDSII, and both passed DRC with zero violations in Magic and KLayout and LVS with a unique circuit match. The second also reported `Antenna Passed`, the check the first one failed with 54 pins over 46 nets. Both reported setup and hold violations, the first at the 8.0 ns gate and the second at the 30.0 ns gate, which made the flow exit non zero both times. STA at the declared 4.0 ns has never been run. Details in section 2.5 of `verification_report.md` |

Full details in `verification_report.md`.

---

## 8. Corrections to the specification, measured rather than assumed

1. Section 4 of the specification quotes the inverse of 0x02 in GF(2^8) as 0x8C. The
   measured inverse modulo 0x11D is **0x8E**, and 0x02 has multiplicative order **255**, so
   the closed state space is the 255 nonzero field elements, not 256.
2. Section 7 offers QFN-48 or QFN-64. The interface needs 146 signal bits, so this design
   declares **QFN-160** and records the conflict.
3. Section 3 describes the mesh weights as the trained network weights. The netlist carries
   the fixed reference matrix from the specification text; the trained tensor in
   `artifacts/grammar_weights.pt` is not the source of the constants in the RTL.
4. Section 5 asks for a one cycle GF automaton. That is what is built, but with a one-byte
   state the reachable set is the multiplicative group, so the automaton recognizes the
   Parikh image of the bracket language rather than the language itself.
5. Section 6 describes a combinational mesh. This design registers the mesh output at the
   top level, one cycle of latency, which keeps the multiply-accumulate cone off the
   pad-to-pad path. The deviation is documented in `integration_guide.md`.

---

## 9. Ordering, configuration and files

| Item | Value |
|---|---|
| Chip configuration | `asic_chip/config.json`, schema `lingologos-chip-config-1` |
| Weight artifact | `asic_chip/artifacts/model_weights.json`, schema `lingologos-weight-mesh-1` |
| Automaton artifact | `asic_chip/artifacts/gf_rules.json`, schema `lingologos-gf-automaton-1` |
| Clause artifact | `asic_chip/artifacts/smt_constraints.json`, schema `lingologos-smt-clauses-1` |
| Generated RTL | `asic_chip/rtl/generated/`, digests in `MANIFEST.json` |
| Golden vectors | `asic_chip/sim/golden_vectors.mem`, schema `lingologos-golden-1`, 304 rows |
| Self-test vectors | `asic_chip/sim/core_self_test.mem`, schema `lingologos-selftest-1`, 22 keys |
| Build guide | `asic_chip/doc/build_instructions.md` |
| Integration guide | `asic_chip/doc/integration_guide.md` |
| Verification report | `asic_chip/doc/verification_report.md` |

Generator digests of the three artifacts, recorded in
`asic_chip/rtl/generated/MANIFEST.json`: `model_weights.json` `f7ff9dc8c2229eb3`,
`gf_rules.json` `85c612cdcf543dd5`, `smt_constraints.json` `7d7d5a25c4a2915c`.
