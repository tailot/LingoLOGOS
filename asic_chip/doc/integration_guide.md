# LingoLOGOS integration guide

Audience: the engineer who has to put `lingologos_top` on a board, drive it from an FPGA
or a microcontroller, or re-generate its RTL with different constants.

Every claim below describes what the RTL in this checkout actually does. Where the
behavior deviates from the specification, section 8 states the deviation and why. Where a
property has not been characterized, the guide says so instead of inventing a number.

---

## 1. Files an integrator touches

| File | Role |
|---|---|
| `asic_chip/config.json` | single source of truth for widths, latencies, package, paths |
| `asic_chip/artifacts/model_weights.json` | the 4 by 4 weight matrix |
| `asic_chip/artifacts/gf_rules.json` | field polynomial, initial state, symbol multipliers |
| `asic_chip/artifacts/smt_constraints.json` | the four Boolean clauses and the variable map |
| `asic_chip/scripts/lingologos_compiler.py` | turns the three artifacts into RTL plus vectors |
| `asic_chip/rtl/generated/` | generated SystemVerilog, never edited by hand |
| `asic_chip/rtl/core/rv32e_core.sv` | core, decoder, program ROM with the self-test |
| `asic_chip/rtl/top/lingologos_top.sv` | pad facing interface, output registers, coprocessor mux |
| `asic_chip/sim/golden_vectors.mem` | 304 expected input and output rows |
| `asic_chip/sim/core_self_test.mem` | 22 expected self-test values |
| `asic_chip/openlane/` | synthesis, constraint and pin order configuration |

Rule: the three files in `rtl/generated/` are outputs. If you edit them by hand, the
digest check in `asic_chip/rtl/generated/MANIFEST.json` fails and
`tests/test_asic_compiler.py` reports it.

---

## 2. Board connection

### 2.1 Supplies and package

The declared package is QFN-160 with 7 VDD and 7 VSS pads on the bottom side, in the pin
ranges 147 to 153 and 154 to 160. The declared core rail is the nominal
`sky130_fd_sc_hd` supply, 1.8 V. **No absolute maximum rating, no ESD class, no IOH, VOL
or input leakage figure exists in this checkout**: the PDK is not installed and no
characterization has been run. A board designed around this part today must treat the
supply decoupling and the pad driver strength as provisional.

### 2.2 Clock

`clk` is a single-ended input on pin 141. The declared period is 4.0 ns, and the input
delay budget in `asic_chip/openlane/lingologos.sdc` is 0.80 ns, which implies a source
that launches within 20 percent of the period. The SDC names `sky130_fd_sc_hd__buf_4` as
the driving cell: a weaker driver than that on a long trace invalidates the budget.

The design has one clock domain and no clock gating. There is no PLL, no clock divider and
no source synchronous strobe. If you drive the part slower than 250 MHz nothing breaks,
because the datapath has no timing-critical path that depends on the rate; only the
throughput changes.

### 2.3 Reset

`rst_n` (pin 142) is active low, asynchronously asserted and must be released
synchronously: hold it low for at least two clock periods, then release it with a rising
edge of `clk` at least one period before `core_start` is pulsed. `rst_n` is declared a
false path against the clock in the constraint file, which means the constraint set does
not check the recovery time of the reset tree. Until static timing has actually been run,
treat the two period figure as good practice rather than a checked limit.

### 2.4 Start

`core_start` (pin 143) is a single cycle pulse. It loads the program counter with 4 and
sets the core running. It is ignored while the core is already running. A second pulse
after `core_done` restarts the self-test from the same frozen inputs; the state of the
GF automaton is whatever the cycles before the pulse left behind, so pulse
`gf_symbol_in` with 0x00 for one cycle if a deterministic starting state is required.

---

## 3. Driving the datapath

### 3.1 Input lanes

`in_vector_0..3` are 8 bit two's complement. Range -128 to 127. The mesh has no saturation
logic and no overflow flag: the 20 bit accumulator cannot wrap for any input combination
in range (worst magnitude 24960 against a limit of 524287, gate H1), so any wrap you ever
see means an input driver is putting garbage on the lane.

### 3.2 When to sample the outputs

| Output | Valid |
|---|---|
| `smt_results` | combinational, valid as soon as `smt_in_bits` settles |
| `out_vector_0..3` | the row computed from the inputs of the previous cycle, so sample after the rising edge |
| `gf_next_state` | the state register, updated on every rising edge |

### 3.3 The automaton input hazard, read this once

`gf_symbol_in` is sampled **every clock cycle**, not on a strobe. The automaton advances
with whatever byte is on that bus, and any byte outside the four symbol values drives the
`default` arm of the case table, which loads the initial state 0x01.

Consequences for an integrator:

1. If your source does not have a symbol this cycle, drive 0x00 on `gf_symbol_in`. 0x00 is
   not in the alphabet, so the state is held at the initial state, which is the documented
   idle behavior and matches the frozen self-test stimulus.
2. Do not leave `gf_symbol_in` floating or tri-sted in simulation; it will resolve to X and
   the state register becomes X.
3. If you need a symbol strobe, you must build a small external state machine that holds the
   symbol byte for exactly one cycle. There is no enable pin on the pad side.
4. The core can also advance the automaton, through the `LLGFS` instruction, using the byte
   in `rs1` instead of the pad. See section 6.3.

### 3.4 SMT input

`smt_in_bits` maps variables A to H onto bits 0 to 7. Bit 0 is A. The four output lanes
are `smt_results[0]` for clause 0 and so on. There is no clock, no acknowledge and no
pipeline: the lanes are a function of the input byte. If you need them registered,
register them outside the chip.

---

## 4. Reading the self-test

`core_done` goes high when the halt instruction retires, measured 89 cycles after
`core_start` for the shipped firmware. `core_pass` is bit 0 of the register passed to the
halt, and `core_illegal_seen` is the sticky flag set by the deliberate illegal encoding in
program slot 79. The expected result of the shipped firmware is:

| Signal or register | Expected |
|---|---|
| `core_pass` | 1 |
| `core_illegal_seen` | 1 |
| `dbg_reg_x8` (SMT readback for byte 0x5A) | 0x00000009 |
| `dbg_reg_x10` (GF state after one step on 0x28) | 0x00000002 |
| `dbg_reg_x11` (mesh row 2 readback) | 0x0000104A, which is 4170 |
| `dbg_reg_x13` (coprocessor status) | 0x00000013, counter field 3 with the illegal flag set; the counter register itself reads 4, section 6.2 |
| `dbg_reg_x15` (error accumulator) | 0x00000000 |
| `gf_next_state` at the end of the self-test | 0x01 |

The debug registers only exist with `LL_DEBUG_PORTS` defined, so a board can only observe
`core_done`, `core_pass` and `core_illegal_seen`. A production self-test therefore reads
the verdict through those three pins, and the register level detail stays in simulation.

---

## 5. Throughput and cadence

One vector per clock is the nominal rate. The mesh cone and the automaton are independent,
and the SMT cone has no state, so the three engines do not interact: an input lane change
has no effect on `gf_next_state` and vice versa.

| Quantity | Value | Basis |
|---|---|---|
| Vectors per second at the declared 250 MHz | 250 million | declared, not measured in silicon |
| Cycles per self-test | 89 | measured by the reference interpreter |
| Cycles per core instruction | 1 | single cycle core, no pipeline, no stall |
| Cycles per coprocessor read | 1 | measured, see the self-test trace |

The program ROM holds 128 instruction slots, and 97 are used. A longer firmware requires
enlarging `IMEM_WORDS` and `IMEM_BITS` in `rv32e_core.sv`, and `asic_chip/config.json`
plus the tests read those parameters, so change all three together.

---

## 6. Programming the core

### 6.1 What the decoder accepts

RV32E subset, 16 registers. `LUI`, `AUIPC`, `JAL`, `JALR`, the six branches `BEQ`, `BNE`,
`BLT`, `BGE`, `BLTU`, `BGEU`, the five loads `LB`, `LH`, `LW`, `LBU`, `LHU`, the three
stores `SB`, `SH`, `SW`, `ADDI`, `SLTI`, `SLTIU`, `XORI`, `ORI`, `ANDI`, `SLLI`, `SRLI`,
`SRAI`, `ADD`, `SUB`, `XOR`, `OR`, `AND`, `SLL`, `SRL`, `SLT`, `SLTU`, `FENCE` accepted as
a no-op, plus the five custom instructions. The register file has 16 entries, so an
encoding that names x16 or higher in a field the opcode actually reads is illegal, and the
illegal flag is raised instead of the register being written. The field matters, not the
bit pattern: the immediate of `ADDI x4, x0, 80`, which is `0x05000213` in program slot 21,
puts a 1 in bit 24 of the instruction, and that is legal, because `ADDI` reads no `rs2`
field and the check only applies to fields the opcode decodes.

The decoder rejects, and sets the sticky flag on: `funct3` values 010 and 011 in the branch
group, 011, 110 and 111 in the load group, 101 and above in the store group, a shift form
whose `funct7` field is neither 0x00 nor 0x20, `FENCE.I`, the custom encodings with
`funct3` 000, 101 and 110, SYSTEM, `OP-IMM-32`, `OP-32`, every unmapped opcode, misaligned
loads and stores, and the wide register references described above.

### 6.2 Custom instruction encoding

`opcode = 0b0001011` (custom-0), `rd` and `rs1` as usual, `funct3` selects the operation.
No other bit of the encoding is inspected, so bits 31:20 are free for assembler use.

| Bits | Field |
|---|---|
| 6:0 | `0001011` |
| 11:7 | `rd`, written with the readback |
| 14:12 | `funct3`: 001 SMT, 010 GF step, 011 mesh row, 100 status, 111 halt |
| 19:15 | `rs1`, the operand byte or the row index |
| 31:20 | ignored by the decoder |

`LLHALT` writes nothing, clears `running`, sets `done` and takes `core_pass` from bit 0 of
`rs1`. The status read of `LLCTL` is defined as the counter value **before** the current
read is counted, which is why the shipped self-test reads 3 after four coprocessor
operations. The counter register has counted that status read as well, so the debug port
`dbg_cop_count` reads 4 once the run is over: the two views differ by exactly the request
that produced the status word. Both figures are in `sim/core_self_test.mem`, as `exp_x13`
and `exp_cop_count`.

### 6.3 Coprocessor readback banks

| `funct3` | Result written to `rd` |
|---|---|
| 001 | `smt_evaluator` run on the low byte of `rs1`, not the byte on the pad |
| 010 | the state the automaton enters on the low byte of `rs1`; this also advances the live automaton with that byte instead of the pad byte |
| 011 | the selected row of the mesh, computed from the live pad inputs, sign extended to 32 bits, row index from bits 1:0 of `rs1` |
| 100 | bit 4 illegal flag, bits 3:0 operation counter |
| 111 | nothing, halts |

### 6.4 Firmware map of the shipped self-test

Slots 97 to 127 hold no-operations and exist so the ROM is fully initialized. Slot 0 is a
no-operation on purpose: the program counter starts at 4, so address 0 stays a benign
landing pad for a runaway jump. The table is a direct render of the `initial` block in
`rv32e_core.sv`; the comments are the assembler annotations and a few pseudo-mnemonics.

| Slot | Byte address | Word | Note |
|---|---|---|---|
| 0 | 0 | `0x00000013` | NOP, reset slot: keeps the trap slot out of address 0 |
| 1 | 4 | `0x00000793` | ADDI x15, x0, 0   error accumulator, must be zero at the end |
| 2 | 8 | `0x00500093` | ADDI x1, x0, 5 |
| 3 | 12 | `0x00300113` | ADDI x2, x0, 3 |
| 4 | 16 | `0x002081B3` | ADD  x3, x1, x2, expected 8 |
| 5 | 20 | `0x00800213` | ADDI x4, x0, 8 |
| 6 | 24 | `0x0041C733` | XOR  x14, x3, x4 |
| 7 | 28 | `0x00E7E7B3` | OR   x15, x15, x14 |
| 8 | 32 | `0x402081B3` | SUB  x3, x1, x2, expected 2 |
| 9 | 36 | `0x00200213` | ADDI x4, x0, 2 |
| 10 | 40 | `0x0041C733` | XOR  x14, x3, x4 |
| 11 | 44 | `0x00E7E7B3` | OR   x15, x15, x14 |
| 12 | 48 | `0x001121B3` | SLT  x3, x2, x1, expected 1 |
| 13 | 52 | `0x00100213` | ADDI x4, x0, 1 |
| 14 | 56 | `0x0041C733` | XOR  x14, x3, x4 |
| 15 | 60 | `0x00E7E7B3` | OR   x15, x15, x14 |
| 16 | 64 | `0x0030B193` | SLTIU x3, x1, 3, expected 0 |
| 17 | 68 | `0x00000213` | ADDI x4, x0, 0 |
| 18 | 72 | `0x0041C733` | XOR  x14, x3, x4 |
| 19 | 76 | `0x00E7E7B3` | OR   x15, x15, x14 |
| 20 | 80 | `0x00409193` | SLLI x3, x1, 4, expected 80 |
| 21 | 84 | `0x05000213` | ADDI x4, x0, 80 |
| 22 | 88 | `0x0041C733` | XOR  x14, x3, x4 |
| 23 | 92 | `0x00E7E7B3` | OR   x15, x15, x14 |
| 24 | 96 | `0xFF900193` | ADDI x3, x0, -7 |
| 25 | 100 | `0x4011D193` | SRAI x3, x3, 1, expected -4, arithmetic shift keeps the sign |
| 26 | 104 | `0xFFC00213` | ADDI x4, x0, -4 |
| 27 | 108 | `0x0041C733` | XOR  x14, x3, x4 |
| 28 | 112 | `0x00E7E7B3` | OR   x15, x15, x14 |
| 29 | 116 | `0x0020C1B3` | XOR  x3, x1, x2, expected 6 |
| 30 | 120 | `0x0011C193` | XORI x3, x3, 1, expected 7 |
| 31 | 124 | `0x00700213` | ADDI x4, x0, 7 |
| 32 | 128 | `0x0041C733` | XOR  x14, x3, x4 |
| 33 | 132 | `0x00E7E7B3` | OR   x15, x15, x14 |
| 34 | 136 | `0x001231B7` | LUI  x3, 0x00123, expected 0x00123000 |
| 35 | 140 | `0x00400293` | ADDI x5, x0, 4, byte address 4 of the scratch RAM |
| 36 | 144 | `0x0032A023` | SW   x3, 0(x5) |
| 37 | 148 | `0x0002A303` | LW   x6, 0(x5) |
| 38 | 152 | `0x0061C733` | XOR  x14, x3, x6 |
| 39 | 156 | `0x00E7E7B3` | OR   x15, x15, x14 |
| 40 | 160 | `0xED400193` | ADDI x3, x0, -300 |
| 41 | 164 | `0x0032A423` | SW   x3, 8(x5), store word at byte address 12 |
| 42 | 168 | `0x00829303` | LH   x6, 8(x5), low half, sign extended back to -300 |
| 43 | 172 | `0x00334733` | XOR  x14, x6, x3 |
| 44 | 176 | `0x00E7E7B3` | OR   x15, x15, x14 |
| 45 | 180 | `0x0082C303` | LBU  x6, 8(x5), low byte of -300 is 0xD4 |
| 46 | 184 | `0x0D400213` | ADDI x4, x0, 212, 0xD4 is the expected zero extended byte |
| 47 | 188 | `0x00434733` | XOR  x14, x6, x4 |
| 48 | 192 | `0x00E7E7B3` | OR   x15, x15, x14 |
| 49 | 196 | `0x00000463` | BEQ  x0, x0, SKIP_POISON_A, taken branch must skip the poison |
| 50 | 200 | `0x00178793` | ADDI x15, x15, 1, poison, must never execute |
| 51 | 204 | `0x00000613` | ADDI x12, x0, 0 |
| 52 | 208 | `0x00209463` | BNE  x1, x2, SKIP_POISON_B, taken branch must skip the poison |
| 53 | 212 | `0x00278793` | ADDI x15, x15, 2, poison, must never execute |
| 54 | 216 | `0x00114463` | BLT  x2, x1, SKIP_POISON_C, 3 is less than 5 |
| 55 | 220 | `0x00478793` | ADDI x15, x15, 4, poison, must never execute |
| 56 | 224 | `0x0020F463` | BGEU x1, x2, SKIP_POISON_D, 5 is above or equal to 3 unsigned |
| 57 | 228 | `0x00878793` | ADDI x15, x15, 8, poison, must never execute |
| 58 | 232 | `0x00115663` | BGE  x2, x1, POISON_E, 3 is not above or equal to 5: must not be taken |
| 59 | 236 | `0x00100613` | ADDI x12, x0, 1, the fall-through path, harmless |
| 60 | 240 | `0x00000463` | BEQ  x0, x0, AFTER_E, jumps over the poison sitting on the branch target |
| 61 | 244 | `0x01078793` | ADDI x15, x15, 16, poison, reached only if a false branch was taken |
| 62 | 248 | `0x0080066F` | JAL  x12, SKIP_POISON_F, unconditional jump with the link in x12 |
| 63 | 252 | `0x02078793` | ADDI x15, x15, 32, poison, must never execute |
| 64 | 256 | `0x0FC00693` | ADDI x13, x0, 252, the expected link of the jump at slot 62 |
| 65 | 260 | `0x40D60733` | SUB  x14, x12, x13 |
| 66 | 264 | `0x00E7E7B3` | OR   x15, x15, x14 |
| 67 | 268 | `0x17C00693` | ADDI x13, x0, 380, absolute byte address of the subroutine entry |
| 68 | 272 | `0x00068767` | JALR x14, x13, 0, call the subroutine, link in x14 |
| 69 | 276 | `0x00000613` | ADDI x12, x0, 0, the return must land on this line |
| 70 | 280 | `0x00000463` | BEQ  x0, x0, AFTER_JALR, jumps over the poison below |
| 71 | 284 | `0x04078793` | ADDI x15, x15, 64, poison, reached only if the return missed the caller |
| 72 | 288 | `0x00700713` | ADDI x14, x0, 7, the subroutine body must have added 7 to x11 |
| 73 | 292 | `0x00E5C733` | XOR  x14, x11, x14 |
| 74 | 296 | `0x00E7E7B3` | OR   x15, x15, x14 |
| 75 | 300 | `0x00000617` | AUIPC x12, 0, x12 becomes the byte address of this very instruction |
| 76 | 304 | `0x12C00693` | ADDI x13, x0, 300, the expected program counter mark |
| 77 | 308 | `0x00D64733` | XOR  x14, x12, x13 |
| 78 | 312 | `0x00E7E7B3` | OR   x15, x15, x14 |
| 79 | 316 | `0x00080093` | ADDI x1, x16, 0, x16 does not exist in RV32E, must be flagged illegal |
| 80 | 320 | `0x05A00393` | ADDI x7, x0, 90, 0x5A is an SMT input byte |
| 81 | 324 | `0x0003940B` | LLSMT x8, x7, four clause lanes for that byte |
| 82 | 328 | `0x02800393` | ADDI x7, x0, 40, 0x28 is the round open bracket |
| 83 | 332 | `0x0003A50B` | LLGFS x10, x7, one GF(2^8) step, state in x10 |
| 84 | 336 | `0x00200393` | ADDI x7, x0, 2, mesh row index |
| 85 | 340 | `0x0003B58B` | LLRDM x11, x7, read back row 2 of the mesh |
| 86 | 344 | `0x0000468B` | LLCTL x13, x0, coprocessor counter, must read back three |
| 87 | 348 | `0x01043713` | SLTIU x14, x8, 16, the SMT readback must fit the four result lanes |
| 88 | 352 | `0x00174713` | XORI x14, x14, 1 |
| 89 | 356 | `0x00E7E7B3` | OR   x15, x15, x14 |
| 90 | 360 | `0x00153713` | SLTIU x14, x10, 1, every reachable GF state is nonzero |
| 91 | 364 | `0x00E7E7B3` | OR   x15, x15, x14 |
| 92 | 368 | `0x0017B713` | SLTIU x14, x15, 1, x14 becomes 1 exactly when no check failed |
| 93 | 372 | `0x0007700B` | LLHALT x14, stop and report pass as bit 0 of x14 |
| 94 | 376 | `0x00000013` | NOP, safety slot, the core is already stopped |
| 95 | 380 | `0x00758593` | ADDI x11, x11, 7, body of the subroutine the jump and link test calls |
| 96 | 384 | `0x00070067` | JALR x0, x14, 0, return to the caller through the link held in x14 |

`tests/test_asic_core_firmware.py` emulates this table with an independent RV32E
interpreter and compares the resulting register values with
`asic_chip/sim/core_self_test.mem`. The test also pins down invariants the listing has to
keep: slot 0 and the slot right after the halt are no-operations, the halt is the last
programmed slot, the only wide register reference is slot 79, and no SYSTEM or 64-bit
opcode appears anywhere.

### 6.5 Changing the firmware

The self-test expectations are not derived from the RTL by a tool; they are written by the
generator into `core_self_test.mem` from the stimulus in `SELF_TEST_STIMULUS`, and
`asic_chip/sim/testbench.sv` repeats the important subset as literals. If you rewrite the
firmware, all four places must move together:

1. the `initial` block in `rv32e_core.sv`,
2. `SELF_TEST_STIMULUS` and the expected values in
   `asic_chip/scripts/lingologos_compiler.py`, then re-run the generator,
3. the literals in `asic_chip/sim/testbench.sv` and, if the readback changes,
   `asic_chip/sim/sim_main.cpp`,
4. the expectations in `tests/test_asic_core_firmware.py`.

The test suite fails loudly if only some of them move, which is the point of keeping the
comparison in a separate interpreter instead of in the generator that produced the RTL.

---

## 7. Regenerating RTL and vectors

```
make -f asic_chip/Makefile gen      # rewrite rtl/generated and sim
make -f asic_chip/Makefile check    # regenerate and compare digests against MANIFEST
make -f asic_chip/Makefile pytest   # run the Python gates
```

or directly:

```
python asic_chip/scripts/lingologos_compiler.py
python asic_chip/scripts/lingologos_compiler.py --check
python asic_chip/scripts/lingologos_compiler.py --out-dir build/asic --vectors 128 --seed 7
```

The generator is deterministic: the same artifacts, the same `--vectors` count and the
same `--seed` reproduce byte-identical files. `tests/test_asic_compiler.py` regenerates
into a temporary directory and compares bytes against the committed files, so any drift
between artifacts and committed RTL shows up as a test failure rather than as a surprise
in synthesis.

### 7.1 Changing the weights

Edit `weight_matrix.values` in `asic_chip/artifacts/model_weights.json`. If the dimensions
change, update `dimensions`, and update `mesh_rows`, `mesh_cols`, `mesh_input_bits` in
`asic_chip/config.json` and `TOP_MESH_ROWS`, `TOP_MESH_COLS`, `TOP_IN_BITS` in the
generator. Then check gate H1: `accumulator_bits` must stay wide enough for the new
worst case, which H1 computes as `sum of abs(weight) times 128` plus the non-adjacent form
partial term. A 5 by 5 matrix with 8 bit inputs still fits comfortably in 20 bits; going
to 16 bit weights or inputs does not, and H1 fails with the numbers printed rather than
producing an RTL that wraps.

### 7.2 Changing the automaton

Edit the `symbols` list in `asic_chip/artifacts/gf_rules.json`. Two invariants are enforced
by gate H2 and must not be broken: the polynomial must be irreducible, and the reachable
state count must stay within `state_budget`. Adding a symbol whose multiplier is 0x00 makes
the transition table collapse onto a single state and the order computation reports it;
using a multiplier of 0x01 makes the symbol invisible to the tracker, which the reachable
state count exposes.

### 7.3 Changing the clauses

Edit the `clauses` list in `asic_chip/artifacts/smt_constraints.json`. The DSL grammar is
documented in the artifact itself and in the `ClauseParser` docstring: precedence from
loosest to tightest is OR, XOR, AND, NOT, `AND`, `OR`, `XOR` and `MUX` may be a prefix call
with parentheses or an infix operator, `MUX` in infix form means `select ? value : 0`, and
variables are the letters `A` to `H` mapped by `variable_bits`. The parser rejects a
construct that would need a latch: there is no feedback, no register and no storage element
expressible in the DSL. Gate H3 then counts the levels of the resulting network and refuses
a network whose cone feeds itself.

---

## 8. Deviations from the specification, and why

| Specification | This implementation | Why |
|---|---|---|
| Combinational mesh from pad to pad | mesh cone combinational, result registered in the top level, one cycle latency | keeps a multiply-accumulate cone out of any pad-to-pad path, and the 4.0 ns budget is already aggressive for sky130 |
| QFN-48 or QFN-64 | QFN-160 | 146 user signal bits do not fit in 64 pads |
| Inverse of 0x02 is 0x8C | 0x8E, measured | 0x8C is wrong modulo 0x11D, and using it would make the close bracket a no-op |
| 256 automaton states | 255 reachable, 1020 transitions written | 0x00 is unreachable from 0x01 with nonzero multipliers, and the default arm of the table sends any unexpected symbol to the initial state |
| Mesh weights are the trained weights | fixed reference matrix from the specification text | `artifacts/grammar_weights.pt` is a trained torch tensor, not a quantized INT8 netlist constant; wiring it in is a separate, undone step |
| The SMT evaluator evaluates the Z3 constraints | four hand written clauses in a documented DSL | `dataset/tools/logic_map.py` emits Boolean maps for a different interface; compiling those into this netlist is future work |
| One cycle GF automaton | built as specified, but with a one byte state | a one byte state cannot count nesting, so the recognized language is the commutative image of Dyck-2, not Dyck-2 |
| Program memory | 128 by 32 bit array initialized in the netlist | synthesis turns it into constant logic; production needs a compiled ROM macro, recorded under `NOTE_ROM` in `asic_chip/openlane/config.json` |
| The core is a debug aid that reads the engines | the core also drives the engines: `LLGFS` advances the live automaton with the byte in `rs1`, and the mesh readback uses the live pad inputs | avoids a second state copy that could disagree with the pad-facing state; documented in section 6.3 |
| One SMT evaluator | two instances of the same generated module, one fed by `smt_in_bits`, one fed by the core operand | keeps the pad path free of a multiplexer on the input, at the cost of duplicate logic |

---

## 9. Bringing the design up in simulation

| Flow | Command | Notes |
|---|---|---|
| Verilator, C++ harness | `make -f asic_chip/Makefile sim-verilator` | builds `sim/sim_main.cpp` against the top level with `LL_DEBUG_PORTS` defined |
| Verilator, SystemVerilog testbench | `make -f asic_chip/Makefile sim-tb` | uses `$readmemh` on `asic_chip/sim/golden_vectors.mem`, so run it from the repository root |
| Icarus Verilog | `make -f asic_chip/Makefile sim-iverilog` | same vectors, `vvp` output |
| Lint | `make -f asic_chip/Makefile lint` | Verilator lint only, no simulation |

Both harnesses read the vector files by path relative to the repository root, which is why
the Makefile targets exist; running them by hand from another directory makes the harness
report a missing file rather than a wrong value. The three Verilator rows ran in the
container image `verilator/verilator:latest`, Verilator 5.054, with the checkout mounted at
`/work`; section 2.4 of `verification_report.md` quotes what they printed, and the container
route is written out at the end of section 4 of `build_instructions.md`. The Icarus row has
never run: no Icarus binary is installed here.

The C++ harness waits up to 400 cycles for `core_done`; the self-test needs 89 cycles by the
reference interpreter and 91 as the harness counted them. If
a firmware change pushes the count past 400, raise the bound in `sim_main.cpp` and the
watchdog in `testbench.sv` together, and note that `tests/test_asic_core_firmware.py` reads
the bound out of `sim_main.cpp` and asserts that the measured cycle count stays under it.

---

## 10. Physical implementation notes

`asic_chip/openlane/config.json` is written for OpenLane 1 with the sky130A PDK: design
name `lingologos_top`, the generated modules plus the core and the top as the source list,
`CLOCK_PORT clk`, `CLOCK_PERIOD 4.0`, `FP_PIN_ORDER_CFG` pointing at `pin_order.cfg`, and the
antenna and timing gate knobs written under the names OpenLane 2 reads. Beside it sit
`config.bringup.json` and `lingologos.bringup.sdc`, the same configuration at a 32.0 ns period and with
both resizer hold slack margins, `PL_RESIZER_HOLD_SLACK_MARGIN` and `GRT_RESIZER_HOLD_SLACK_MARGIN`, at
0.40 ns instead of the installed 0.10 and 0.05. The entry point runs the bring-up pair by default so
that a route produces a layout and a measured slack rather than stopping at the declared budget: two end
to end runs measured the gap. At the 8.0 ns gate the first missed setup by 4.4409 ns in the worst
typical corner and by 15.7205 ns in the worst slow one; at the 30.0 ns gate the second closed every
corner except 0.4906 ns of setup in the slow corner, and lost hold on input paths by up to 0.1540 ns.
The gate is now above both gaps, and those margins are what the second run measured routing and
extraction to be worth. `LL_OPENLANE_CONFIG=config.json` routes
at the declared 250 MHz budget instead, and nothing has ever been routed against it. An integrator
reading a report has to check which of the two configurations produced it, and which corner: what
turns a slack into a failure is the corner list `SETUP_VIOLATION_CORNERS` and
`HOLD_VIOLATION_CORNERS`, both set to every corner in both files here, while OpenLane's own default
for a sky130 PDK is the typical corner only.

`asic_chip/scripts/build_gds.sh` runs the generator with `--check` before touching the layout flow,
and exits 127 with an explicit message when neither OpenLane 1 nor OpenLane 2 is reachable. When a
flow did run, the script prints the newest run directory, the GDSII files below it and the reports
that name each signoff check, and returns the exit status of the flow rather than the status of its
own last command. Two place and routes have been run for this design, on 2026-11-09, and both
returned non zero: deferred setup and hold violations are reported after the GDSII, the DRC reports and
the LVS report have been written, and in the first of the two the antenna check failed as well. Section
2.5 of `asic_chip/doc/verification_report.md` lists what those runs measured and what they leave open.

For a production hardening pass, the items that need attention before any timing or power
claim can be written into the datasheet: a route that closes, meaning an antenna check that
passes, which the run at the 8.0 ns gate failed on 54 pins and 46 nets and the run at the 30.0 ns
gate passed, and a signoff with zero setup and hold violations at the period that is meant to be
quoted; the ROM macro
replacement, an assessment of the 1020 arm case table (it synthesizes into a PLA-like
structure and is the most likely critical path, and the worst setup path of both routed
layouts does run through the `core_pc` cone), the reset false path that needs to become a real
checked path, and the absence of an I/O characterization.

---

## 11. What an integrator must not assume today

1. No maximum frequency claim. Static timing has been run twice, on routed layouts and over nine
   corners. At the 8.0 ns bring-up period of the first run it reported 15,089 setup violating paths and
   41 hold violating paths; at the 30.0 ns gate of the second, 35 setup violating paths in one corner and
   45 hold violations spread over seven corners. The declared 250 MHz budget has never been routed at
   all, and a slack at a stated period is not a rating.
2. No power, ESD, latch-up or pad drive claim. Area, utilization, wire length, DRC, LVS, IR drop and
   a per corner power estimate exist from two runs, each labeled with the period and corner it
   came from and with the activity model of the tool behind the power figure; nothing has been
   characterized, and there are no pads to have a drive or latch-up figure for.
3. No radiation, temperature or voltage corner behavior.
4. The pin table in the datasheet is generated from the RTL port list and the placement
   intent file; it is not a package ballout from a foundry.
5. The self-test verdict covers the decoder, the memory path, the coprocessor port, the
   automaton cone and the mesh readback. It does not cover pad behavior, since there are no
   pads yet.
