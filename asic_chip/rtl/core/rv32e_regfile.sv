// -----------------------------------------------------------------------------
// RV32E register file: sixteen 32-bit registers, x0 hardwired to zero.
//
// RV32E is the embedded variant of RV32I: the register set is x0..x15, so every
// 5-bit register field with a value of 16 or above is an illegal encoding for this
// core. The read ports therefore guard the upper half of the 5-bit space instead of
// indexing past the end of the array, and the core qualifies writes with the
// legality check.
//
// The read ports are pure flop reads. There is deliberately no write through path:
// in a single cycle core every instruction commits its register write at the clock
// edge at the end of its own cycle, so the instruction that follows already sees the
// new value from the array. Feeding wdata into the read multiplexers instead would
// create a combinational loop, wdata from the arithmetic unit back into the operand
// that produces it, and a loop is not synthesizable hardware. Verilator reports such
// a path as UNOPTFLAT.
// -----------------------------------------------------------------------------

module rv32e_regfile #(
    parameter int REGS = 16
) (
    input  wire logic          clk,
    input  wire logic          rst_n,
    input  wire logic          we,
    input  wire logic [4:0]    waddr,
    input  wire logic [31:0]   wdata,
    input  wire logic [4:0]    raddr_a,
    input  wire logic [4:0]    raddr_b,
    output      logic [31:0]   rdata_a,
    output      logic [31:0]   rdata_b,
    output      logic [REGS-1:0][31:0] reg_dump
);

    logic [31:0] registers [REGS];

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int index = 0; index < REGS; index = index + 1) begin
                registers[index] <= 32'h0000_0000;
            end
        end else if (we && (waddr != 5'd0) && (waddr[4] == 1'b0)) begin
            registers[waddr[3:0]] <= wdata;
        end
    end

    always_comb begin
        if (raddr_a[4]) begin
            rdata_a = 32'h0000_0000;
        end else begin
            rdata_a = registers[raddr_a[3:0]];
        end
    end

    always_comb begin
        if (raddr_b[4]) begin
            rdata_b = 32'h0000_0000;
        end else begin
            rdata_b = registers[raddr_b[3:0]];
        end
    end

    for (genvar index = 0; index < REGS; index = index + 1) begin : g_dump
        assign reg_dump[index] = registers[index];
    end

endmodule
