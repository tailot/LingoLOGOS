// -----------------------------------------------------------------------------
// SystemVerilog testbench for the LingoLOGOS ASIC top level.
//
// This is the second of the two validation paths. The primary one is the C++ harness
// in asic_chip/sim/sim_main.cpp, which section 8 of the specification asks for; this
// file exists so the same checks can run inside a Verilator --binary build or in
// Icarus Verilog without any C++, and so the vectors can be loaded with $readmemh
// directly from SystemVerilog.
//
// Run from the repository root:
//   build: verilator --binary --timing -DLL_DEBUG_PORTS --top-module lingologos_testbench \
//       -Iasic_chip/rtl/generated asic_chip/sim/testbench.sv \
//       asic_chip/rtl/generated/*.sv asic_chip/rtl/core/*.sv asic_chip/rtl/top/*.sv
//   run:   obj_dir/Vlingologos_testbench
// The generated RTL contains no timing constructs, so --timing is only needed by the
// testbench itself and a plain cycle based testbench works without it too. The command
// line above is introduced by a label on purpose: a source comment whose first word is
// the name of the tool is read as a lint pragma by that tool, so the label keeps this
// one a comment.
// -----------------------------------------------------------------------------

`timescale 1ns / 1ps

// The module name and the file name differ on purpose: testbench.sv is the name the
// flow documentation and the build files use, while the module carries the testbench
// in its name. The pragma pair records the mismatch instead of leaving it as a lint
// report surprise.
/* verilator lint_off DECLFILENAME */
module lingologos_testbench;

    localparam int IN_BITS   = 8;
    localparam int ACC_BITS  = 20;
    localparam int SMT_BITS  = 8;
    localparam int SMT_LANES = 4;

    // GOLDEN_ROWS is the number of data rows the vector file currently holds. The
    // build gate in tests/test_asic_compiler.py compares this parameter with the
    // number of data lines in the file, so the two cannot drift apart. MAX_ROWS is
    // only the capacity of the token array: a shorter file leaves the remaining slots
    // at the value the loader writes into them, so the row count is stated here rather
    // than derived from a scan for an unused sentinel word.
    localparam int GOLDEN_ROWS = 304;
    localparam int MAX_ROWS  = 1024;
    localparam int MAX_TOKENS = MAX_ROWS * 12;

    logic clk;
    logic rst_n = 1'b0;
    logic core_start = 1'b0;

    logic signed [IN_BITS-1:0]  in_vector_0;
    logic signed [IN_BITS-1:0]  in_vector_1;
    logic signed [IN_BITS-1:0]  in_vector_2;
    logic signed [IN_BITS-1:0]  in_vector_3;
    logic signed [ACC_BITS-1:0] out_vector_0;
    logic signed [ACC_BITS-1:0] out_vector_1;
    logic signed [ACC_BITS-1:0] out_vector_2;
    logic signed [ACC_BITS-1:0] out_vector_3;
    logic [7:0] gf_symbol_in;
    logic [7:0] gf_next_state;
    logic [SMT_BITS-1:0] smt_in_bits;
    logic [SMT_LANES-1:0] smt_results;
    logic core_done;
    logic core_pass;
    logic core_illegal_seen;
    logic dbg_core_running;
    logic [8:0] dbg_core_pc;
    logic [3:0] dbg_cop_count;
    logic [31:0] dbg_cop_last_rdata;
    logic [31:0] dbg_reg_x7;
    logic [31:0] dbg_reg_x8;
    logic [31:0] dbg_reg_x10;
    logic [31:0] dbg_reg_x11;
    logic [31:0] dbg_reg_x12;
    logic [31:0] dbg_reg_x13;
    logic [31:0] dbg_reg_x14;
    logic [31:0] dbg_reg_x15;

    lingologos_top dut (
        .clk               (clk),
        .rst_n             (rst_n),
        .core_start        (core_start),
        .in_vector_0       (in_vector_0),
        .in_vector_1       (in_vector_1),
        .in_vector_2       (in_vector_2),
        .in_vector_3       (in_vector_3),
        .out_vector_0      (out_vector_0),
        .out_vector_1      (out_vector_1),
        .out_vector_2      (out_vector_2),
        .out_vector_3      (out_vector_3),
        .gf_symbol_in      (gf_symbol_in),
        .gf_next_state     (gf_next_state),
        .smt_in_bits       (smt_in_bits),
        .smt_results       (smt_results),
        .core_done         (core_done),
        .core_pass         (core_pass),
        .core_illegal_seen (core_illegal_seen),
        .dbg_core_running  (dbg_core_running),
        .dbg_core_pc       (dbg_core_pc),
        .dbg_cop_count     (dbg_cop_count),
        .dbg_cop_last_rdata (dbg_cop_last_rdata),
        .dbg_reg_x7        (dbg_reg_x7),
        .dbg_reg_x8        (dbg_reg_x8),
        .dbg_reg_x10       (dbg_reg_x10),
        .dbg_reg_x11       (dbg_reg_x11),
        .dbg_reg_x12       (dbg_reg_x12),
        .dbg_reg_x13       (dbg_reg_x13),
        .dbg_reg_x14       (dbg_reg_x14),
        .dbg_reg_x15       (dbg_reg_x15)
    );

    // The columns of golden_vectors.mem are loaded as a flat stream of hex words,
    // twelve per row, in the order the generator documents in the file header.
    logic [31:0] tokens [0:MAX_TOKENS-1];
    int row_count;

    int failures = 0;

    // Expected mesh columns, held at the width of the pads so that a comparison of the
    // checker never compares two different widths.
    logic signed [ACC_BITS-1:0] want_vector [0:3];

    // extend keeps the low bits of a hex column and copies the top kept bit upward,
    // which is the sign extension the generator documented for the vector file. The
    // replication count and the range of a part select have to be elaboration time
    // constants, so the mask is built with a shift and a single bit select instead of
    // with a replicated range.
    function automatic logic signed [31:0] extend(input logic [31:0] value, input int bits);
        logic [31:0] mask;
        mask = (32'sd1 << bits) - 32'sd1;
        extend = (value & mask) | (value[bits-1] ? ~mask : 32'sd0);
    endfunction

    initial begin
        for (int index = 0; index < MAX_TOKENS; index = index + 1) begin
            tokens[index] = 32'hxxxx_xxxx;
        end
        $readmemh("asic_chip/sim/golden_vectors.mem", tokens);
        if (tokens[0] === 32'hxxxx_xxxx || tokens[11] === 32'hxxxx_xxxx) begin
            $display("[FAIL] asic_chip/sim/golden_vectors.mem did not load a first row");
            $fatal(1, "FINAL VERDICT: FAIL");
        end
        row_count = GOLDEN_ROWS;
        $display("[INFO] phase A, %0d golden rows from asic_chip/sim/golden_vectors.mem", row_count);

        rst_n = 1'b0;
        in_vector_0 = '0;
        in_vector_1 = '0;
        in_vector_2 = '0;
        in_vector_3 = '0;
        gf_symbol_in = 8'h00;
        smt_in_bits = 8'h00;
        repeat (4) @(posedge clk);
        rst_n = 1'b1;
        @(posedge clk);

        for (int row = 0; row < row_count; row = row + 1) begin
            @(negedge clk);
            in_vector_0   = tokens[row*12 + 0][IN_BITS-1:0];
            in_vector_1   = tokens[row*12 + 1][IN_BITS-1:0];
            in_vector_2   = tokens[row*12 + 2][IN_BITS-1:0];
            in_vector_3   = tokens[row*12 + 3][IN_BITS-1:0];
            gf_symbol_in  = tokens[row*12 + 4][7:0];
            smt_in_bits   = tokens[row*12 + 5][7:0];
            #1;
            if (smt_results !== tokens[row*12 + 11][SMT_LANES-1:0]) begin
                failures = failures + 1;
                $display("[FAIL] row %0d smt_results (combinational): got %b want %b",
                         row, smt_results, tokens[row*12 + 11][SMT_LANES-1:0]);
            end
            want_vector[0] = extend(tokens[row*12 + 6], ACC_BITS);
            want_vector[1] = extend(tokens[row*12 + 7], ACC_BITS);
            want_vector[2] = extend(tokens[row*12 + 8], ACC_BITS);
            want_vector[3] = extend(tokens[row*12 + 9], ACC_BITS);
            @(posedge clk);
            // The output register stage and the automaton state register take their new
            // value on this edge. A process that wakes up on the edge reads them before
            // the update has landed, so the checker settles one time unit later, which
            // is where the file header says these columns are sampled.
            #1;
            if (out_vector_0 !== want_vector[0]) begin
                failures = failures + 1;
                $display("[FAIL] row %0d out_vector_0: got %0d want %0d", row, out_vector_0,
                         want_vector[0]);
            end
            if (out_vector_1 !== want_vector[1]) begin
                failures = failures + 1;
                $display("[FAIL] row %0d out_vector_1: got %0d want %0d", row, out_vector_1,
                         want_vector[1]);
            end
            if (out_vector_2 !== want_vector[2]) begin
                failures = failures + 1;
                $display("[FAIL] row %0d out_vector_2: got %0d want %0d", row, out_vector_2,
                         want_vector[2]);
            end
            if (out_vector_3 !== want_vector[3]) begin
                failures = failures + 1;
                $display("[FAIL] row %0d out_vector_3: got %0d want %0d", row, out_vector_3,
                         want_vector[3]);
            end
            if (gf_next_state !== tokens[row*12 + 10][7:0]) begin
                failures = failures + 1;
                $display("[FAIL] row %0d gf_next_state: got %02h want %02h", row, gf_next_state,
                         tokens[row*12 + 10][7:0]);
            end
        end

        // Phase B. The values below mirror asic_chip/sim/core_self_test.mem, which the
        // generator writes; tests/test_asic_compiler.py compares this block against
        // that file, so the numbers cannot drift apart.
        in_vector_0   = 8'sh0A;
        in_vector_1   = 8'shFB;
        in_vector_2   = 8'sh02;
        in_vector_3   = 8'sh64;
        gf_symbol_in  = 8'h00;
        smt_in_bits   = 8'h5A;
        repeat (4) @(posedge clk);

        if (out_vector_2 !== 20'sh104A) begin
            failures = failures + 1;
            $display("[FAIL] frozen pads out_vector_2 before the core starts: got %0d want 4170",
                     out_vector_2);
        end

        core_start = 1'b1;
        wait (core_done == 1'b1);
        @(posedge clk);
        #1;
        $display("[INFO] self-test finished at pc %03h, coprocessor counter %0d",
                 dbg_core_pc, dbg_cop_count);

        if (core_pass !== 1'b1) begin
            failures = failures + 1;
            $display("[FAIL] core_pass is not set");
        end
        if (core_illegal_seen !== 1'b1) begin
            failures = failures + 1;
            $display("[FAIL] core_illegal_seen is not set, the firmware executes a x16 encoding on purpose");
        end
        if (dbg_reg_x8 !== 32'h0000_0009) begin
            failures = failures + 1;
            $display("[FAIL] x8 after the SMT readback: got %08h want 00000009", dbg_reg_x8);
        end
        if (dbg_reg_x10 !== 32'h0000_0002) begin
            failures = failures + 1;
            $display("[FAIL] x10 after the GF step: got %08h want 00000002", dbg_reg_x10);
        end
        if (dbg_reg_x11 !== 32'h0000_104A) begin
            failures = failures + 1;
            $display("[FAIL] x11 after the mesh row readback: got %08h want 0000104A", dbg_reg_x11);
        end
        if (dbg_reg_x13 !== 32'h0000_0013) begin
            failures = failures + 1;
            $display("[FAIL] x13 after the status readback: got %08h want 00000013", dbg_reg_x13);
        end
        if (dbg_reg_x15 !== 32'h0000_0000) begin
            failures = failures + 1;
            $display("[FAIL] x15 error accumulator is not zero: got %08h", dbg_reg_x15);
        end
        // The counter register counts the status request itself as well: the status
        // word that the firmware reads in x13 carries the counter of the three earlier
        // requests, because the readback is combinational and the increment lands on
        // the clock edge of the request it belongs to.
        if (dbg_cop_count !== 4'd4) begin
            failures = failures + 1;
            $display("[FAIL] coprocessor counter: got %0d want 4", dbg_cop_count);
        end
        if (gf_next_state !== 8'h01) begin
            failures = failures + 1;
            $display("[FAIL] gf_next_state after the self-test: got %02h want 01", gf_next_state);
        end

        core_start = 1'b0;
        if (failures > 0) begin
            $display("checked %0d vector rows, %0d mismatches", row_count, failures);
            $fatal(1, "FINAL VERDICT: FAIL");
        end
        $display("checked %0d vector rows plus the core self-test, 0 mismatches", row_count);
        $display("FINAL VERDICT: PASS");
        $finish;
    end

    // One process owns the clock: it gives the signal its first value and then toggles
    // it, so the same signal is never written by a declaration initializer and by a
    // procedural block at the same time.
    initial begin
        clk = 1'b0;
        forever begin
            #5 clk = ~clk;
        end
    end

    // Watchdog: a stuck core must fail the run instead of hanging the simulator.
    initial begin
        #20000;
        $display("[FAIL] watchdog expired with core_running=%b pc=%03h", dbg_core_running, dbg_core_pc);
        $fatal(1, "FINAL VERDICT: FAIL");
    end

endmodule
/* verilator lint_on DECLFILENAME */
