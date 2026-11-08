// -----------------------------------------------------------------------------
// LingoLOGOS ASIC top level.
//
// Four blocks, all on-chip, exactly as section 1 of the LingoLOGOS ASIC
// specification asks for:
//   * weight_rom_mesh      combinatorial INT8 mesh of fixed weights, no multiplier
//   * gf_automaton_1cycle  combinatorial GF(2^8) transition table, one cycle per symbol
//   * smt_evaluator        combinational clause network, zero cycle latency
//   * rv32e_core           RV32E control processor with a coprocessor port
// The three generated blocks come from rtl/generated, which the generator rebuilds
// from the artifacts; the core is hand written in rtl/core.
//
// Latency, as the datasheet timing table states:
//   out_vector_*     one clock after in_vector_*      (registered output stage)
//   gf_next_state    one clock after gf_symbol_in     (state register)
//   smt_results      same cycle as smt_in_bits        (pure combinational)
// The mesh outputs go through a register stage so the pad ring sees flops instead of
// a long combinational cone; the specification draws the mesh as combinational and
// this stage is the one deviation, recorded in asic_chip/doc/integration_guide.md.
//
// Coprocessor mux, selected by rv32e_core.cop_sel:
//   0  SMT evaluator result for the byte in the core register
//   1  GF automaton next state for the symbol in the core register
//   2  signed readback of the selected mesh output row
//   3  coprocessor counter in bits 3:0 and the sticky illegal flag in bit 4
//
// Two shares are worth stating plainly. The GF automaton is shared: a step
// instruction from the core advances the same state register the pads see, for one
// cycle overriding the pad symbol. The clause network is not shared: it is
// instantiated twice, one copy driven by the pads and one by the core, because a
// single copy would let a core read disturb the pad output for a cycle. Copy two
// costs 8 input bits of gates, which is why the split is affordable.
//
// The dbg_ ports exist only in a simulation build compiled with LL_DEBUG_PORTS
// defined. Synthesis and the OpenLane flow see the pad list and nothing else.
// -----------------------------------------------------------------------------

module lingologos_top #(
    parameter logic [7:0] GF_INITIAL_STATE = 8'h01,
    parameter int IMEM_WORDS = 128,
    parameter int DMEM_WORDS = 64
) (
    input  wire logic               clk,
    input  wire logic               rst_n,
    input  wire logic               core_start,

    input  wire logic signed [7:0]  in_vector_0,
    input  wire logic signed [7:0]  in_vector_1,
    input  wire logic signed [7:0]  in_vector_2,
    input  wire logic signed [7:0]  in_vector_3,
    output wire logic signed [19:0] out_vector_0,
    output wire logic signed [19:0] out_vector_1,
    output wire logic signed [19:0] out_vector_2,
    output wire logic signed [19:0] out_vector_3,

    input  wire logic [7:0]         gf_symbol_in,
    output wire logic [7:0]         gf_next_state,
    input  wire logic [7:0]         smt_in_bits,
    output wire logic [3:0]         smt_results,

    output wire logic               core_done,
    output wire logic               core_pass,
    output wire logic               core_illegal_seen
`ifdef LL_DEBUG_PORTS
    , output wire logic              dbg_core_running
    , output wire logic [8:0]        dbg_core_pc
    , output wire logic [3:0]        dbg_cop_count
    , output wire logic [31:0]       dbg_cop_last_rdata
    , output wire logic [31:0]       dbg_reg_x7
    , output wire logic [31:0]       dbg_reg_x8
    , output wire logic [31:0]       dbg_reg_x10
    , output wire logic [31:0]       dbg_reg_x11
    , output wire logic [31:0]       dbg_reg_x12
    , output wire logic [31:0]       dbg_reg_x13
    , output wire logic [31:0]       dbg_reg_x14
    , output wire logic [31:0]       dbg_reg_x15
`endif
);

    // -----------------------------------------------------------------------
    // Weight mesh: combinational cone, registered on the way out.
    // -----------------------------------------------------------------------
    wire logic signed [19:0] mesh_raw_0;
    wire logic signed [19:0] mesh_raw_1;
    wire logic signed [19:0] mesh_raw_2;
    wire logic signed [19:0] mesh_raw_3;

    weight_rom_mesh u_weight_rom_mesh (
        .in_vec_0   (in_vector_0),
        .in_vec_1   (in_vector_1),
        .in_vec_2   (in_vector_2),
        .in_vec_3   (in_vector_3),
        .out_vec_0  (mesh_raw_0),
        .out_vec_1  (mesh_raw_1),
        .out_vec_2  (mesh_raw_2),
        .out_vec_3  (mesh_raw_3)
    );

    logic signed [19:0] mesh_reg_0;
    logic signed [19:0] mesh_reg_1;
    logic signed [19:0] mesh_reg_2;
    logic signed [19:0] mesh_reg_3;

    assign out_vector_0 = mesh_reg_0;
    assign out_vector_1 = mesh_reg_1;
    assign out_vector_2 = mesh_reg_2;
    assign out_vector_3 = mesh_reg_3;

    // -----------------------------------------------------------------------
    // Control core and coprocessor wiring
    // -----------------------------------------------------------------------
    wire              cop_req;
    wire        [1:0] cop_sel;
    wire        [31:0] cop_wdata;
    logic       [31:0] cop_rdata;
    wire        [3:0] cop_count;
    wire        [31:0] cop_last_rdata;
    wire              core_running;
    wire        [8:0] core_pc;
    wire [31:0]       core_instruction;
    wire [15:0][31:0] core_reg_dump;

    // core_instruction is the decoded word inside the core. The pin plan of the chip
    // has no instruction bus, so this wire is connected to the instance only so that a
    // waveform viewer or a hierarchical reference in a testbench has a name to point
    // at; no logic at this level reads it, which is why the lint report calls it
    // unused and the build flags accept that warning.

    rv32e_core #(
        .IMEM_WORDS (IMEM_WORDS),
        .DMEM_WORDS (DMEM_WORDS)
    ) u_core (
        .clk               (clk),
        .rst_n             (rst_n),
        .core_start        (core_start),
        .cop_req           (cop_req),
        .cop_sel           (cop_sel),
        .cop_wdata         (cop_wdata),
        .cop_rdata         (cop_rdata),
        .cop_count         (cop_count),
        .cop_last_rdata    (cop_last_rdata),
        .core_running      (core_running),
        .core_done         (core_done),
        .core_pass         (core_pass),
        .core_illegal_seen (core_illegal_seen),
        .core_pc           (core_pc),
        .core_instruction  (core_instruction),
        .core_reg_dump     (core_reg_dump)
    );

    // -----------------------------------------------------------------------
    // GF(2^8) automaton: one table lookup per clock, shared with the core.
    // -----------------------------------------------------------------------
    logic [7:0] gf_state;
    wire logic [7:0] gf_next_comb;

    wire core_gf_step = cop_req && (cop_sel == 2'd1);
    wire [7:0] gf_symbol_applied = core_gf_step ? cop_wdata[7:0] : gf_symbol_in;

    gf_automaton_1cycle u_gf_automaton (
        .current_state (gf_state),
        .symbol_in     (gf_symbol_applied),
        .next_state    (gf_next_comb)
    );

    assign gf_next_state = gf_state;

    // -----------------------------------------------------------------------
    // SMT clause evaluator: combinational, no clock, no latch. Two copies, one
    // for the pads and one for the core request, so neither can disturb the other.
    // -----------------------------------------------------------------------
    wire logic [3:0] smt_pad_results;
    wire logic [3:0] smt_core_results;

    smt_evaluator u_smt_evaluator (
        .in_bits        (smt_in_bits),
        .clause_results (smt_pad_results)
    );

    smt_evaluator u_smt_evaluator_core (
        .in_bits        (cop_wdata[7:0]),
        .clause_results (smt_core_results)
    );

    assign smt_results = smt_pad_results;

    // -----------------------------------------------------------------------
    // Coprocessor readback mux
    // -----------------------------------------------------------------------
    logic signed [19:0] mesh_row_selected;
    always_comb begin
        unique case (cop_wdata[1:0])
            2'd0: mesh_row_selected = mesh_reg_0;
            2'd1: mesh_row_selected = mesh_reg_1;
            2'd2: mesh_row_selected = mesh_reg_2;
            2'd3: mesh_row_selected = mesh_reg_3;
        endcase
    end

    always_comb begin
        case (cop_sel)
            2'd0:    cop_rdata = {28'h0000_000, smt_core_results};
            2'd1:    cop_rdata = {24'h0000_0000, gf_next_comb};
            2'd2:    cop_rdata = {{12{mesh_row_selected[19]}}, mesh_row_selected};
            default: cop_rdata = {24'h0000_0000, 3'b000, core_illegal_seen, cop_count};
        endcase
    end

    // -----------------------------------------------------------------------
    // Sequential stage: the only storage in the datapath.
    // -----------------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            mesh_reg_0 <= 20'sh00000;
            mesh_reg_1 <= 20'sh00000;
            mesh_reg_2 <= 20'sh00000;
            mesh_reg_3 <= 20'sh00000;
            gf_state   <= GF_INITIAL_STATE;
        end else begin
            mesh_reg_0 <= mesh_raw_0;
            mesh_reg_1 <= mesh_raw_1;
            mesh_reg_2 <= mesh_raw_2;
            mesh_reg_3 <= mesh_raw_3;
            gf_state   <= gf_next_comb;
        end
    end

`ifdef LL_DEBUG_PORTS
    // The register window exposes only the registers the instruction-set self-test
    // deposits its results in. It exists so a C++ harness can read a plain 32-bit
    // port instead of reaching into the register file hierarchy, and it disappears
    // entirely from a synthesis build.
    assign dbg_core_running   = core_running;
    assign dbg_core_pc        = core_pc;
    assign dbg_cop_count      = cop_count;
    assign dbg_cop_last_rdata = cop_last_rdata;
    assign dbg_reg_x7         = core_reg_dump[7];
    assign dbg_reg_x8         = core_reg_dump[8];
    assign dbg_reg_x10        = core_reg_dump[10];
    assign dbg_reg_x11        = core_reg_dump[11];
    assign dbg_reg_x12        = core_reg_dump[12];
    assign dbg_reg_x13        = core_reg_dump[13];
    assign dbg_reg_x14        = core_reg_dump[14];
    assign dbg_reg_x15        = core_reg_dump[15];
`endif

endmodule
