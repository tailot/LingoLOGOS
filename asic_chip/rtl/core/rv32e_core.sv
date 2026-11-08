// -----------------------------------------------------------------------------
// RV32E single-cycle core with the LingoLOGOS coprocessor port.
//
// Section 1 of the LingoLOGOS ASIC specification calls for a reduced RISC-V core
// with no external memory interface. This core follows that request literally: the
// program lives in an on-chip read-only array and the data memory is an on-chip
// scratch array, so there is no bus, no DRAM controller and no ROM macro to
// integrate. Both arrays are small on purpose; the firmware is an instruction-set
// self-test, not an application.
//
// Supported instruction set, the RV32E base subset:
//   LUI, AUIPC, JAL, JALR, BEQ/BNE/BLT/BGE/BLTU/BGEU,
//   LB/LH/LW/LBU/LHU, SB/SH/SW,
//   ADDI/SLTI/SLTIU/XORI/ORI/ANDI, SLLI/SRLI/SRAI,
//   ADD/SUB/AND/OR/XOR/SLL/SLT/SLTU/SRL/SRA,
//   FENCE (treated as a no-op, there is no external memory to order against).
// Not implemented, and reported as an illegal instruction: every 64-bit operation
// (OP-IMM-32, OP-32), all system and privileged instructions, FENCE.I, any register
// field above x15, which is the RV32E register-space rule, and load or store
// accesses that are not naturally aligned. The scratch array is a bank of full-width
// words read through a fixed part-select, so an unaligned access would need a barrel
// shifter this design does not have; it is reported instead of performed.
//
// Coprocessor port, opcode 0b0001011 (custom-0), by funct3:
//   3'b001  LLSMT  rd, rs1   four SMT clause results for the byte in rs1
//   3'b010  LLGFS  rd, rs1   one GF(2^8) automaton step with the byte in rs1
//   3'b011  LLRDM  rd, rs1   signed readback of mesh row rs1[1:0]
//   3'b100  LLCTL  rd        coprocessor counter plus the sticky illegal flag
//   3'b111  LLHALT rs1       stop the core, pass flag = rs1[0]
// The first four drive cop_req for one cycle and take their result from cop_rdata in
// the same cycle, which is what "single-cycle operation, zero wait states" means here.
// -----------------------------------------------------------------------------

module rv32e_core #(
    // IMEM_WORDS is the number of 32 bit instruction slots and is a power of two, so
    // a program counter that runs off the end of the program wraps onto the NOP slots
    // instead of decoding an out of range address.
    parameter int IMEM_WORDS = 128,
    parameter int IMEM_BITS  = 7,
    parameter int DMEM_WORDS = 64
) (
    input  wire logic        clk,
    input  wire logic        rst_n,
    input  wire logic        core_start,

    // Coprocessor request port, one cycle request and one cycle result.
    output      logic        cop_req,
    output      logic [1:0]  cop_sel,
    output      logic [31:0] cop_wdata,
    input  wire logic [31:0] cop_rdata,
    output      logic [3:0]  cop_count,
    output      logic [31:0] cop_last_rdata,

    // Status of the self-test run.
    output      logic        core_running,
    output      logic        core_done,
    output      logic        core_pass,
    output      logic        core_illegal_seen,
    // The program counter is a byte address, so it needs two bits more than the word
    // index of the program memory.
    output      logic [IMEM_BITS+1:0] core_pc,
    output      logic [31:0] core_instruction,
    output      logic [15:0][31:0] core_reg_dump
);

    localparam int IMEM_ADDR_BITS = IMEM_BITS + 2;

    localparam logic [6:0] OPC_LUI       = 7'b0110111;
    localparam logic [6:0] OPC_AUIPC     = 7'b0010111;
    localparam logic [6:0] OPC_JAL       = 7'b1101111;
    localparam logic [6:0] OPC_JALR      = 7'b1100111;
    localparam logic [6:0] OPC_BRANCH    = 7'b1100011;
    localparam logic [6:0] OPC_LOAD      = 7'b0000011;
    localparam logic [6:0] OPC_STORE     = 7'b0100011;
    localparam logic [6:0] OPC_MISC_MEM  = 7'b0001111;
    localparam logic [6:0] OPC_OP_IMM    = 7'b0010011;
    localparam logic [6:0] OPC_OP        = 7'b0110011;
    localparam logic [6:0] OPC_CUSTOM    = 7'b0001011;
    localparam logic [6:0] OPC_SYSTEM    = 7'b1110011;
    localparam logic [6:0] OPC_OP_IMM_32 = 7'b0011001;
    localparam logic [6:0] OPC_OP_32     = 7'b0111011;

    localparam logic [1:0] COP_SMT   = 2'd0;
    localparam logic [1:0] COP_GF    = 2'd1;
    localparam logic [1:0] COP_MESH  = 2'd2;
    localparam logic [1:0] COP_STAT  = 2'd3;

    // ---------------------------------------------------------------------------
    // Program memory. The listing below is the hand assembled RV32E self-test; the
    // disassembly is in the comment column and in asic_chip/doc/integration_guide.md.
    // Slots after the program are NOPs so a runaway program counter never decodes a
    // spurious illegal instruction.
    // ---------------------------------------------------------------------------
    logic [31:0] instruction_rom [IMEM_WORDS];

    initial begin
    instruction_rom[0] = 32'h00000013;  // NOP   0 x0 reset slot: keeps the trap slot out of address 0
    instruction_rom[1] = 32'h00000793;  // ADDI x15, x0, 0   error accumulator, must be zero at the end
    instruction_rom[2] = 32'h00500093;  // ADDI x1, x0, 5
    instruction_rom[3] = 32'h00300113;  // ADDI x2, x0, 3
    instruction_rom[4] = 32'h002081B3;  // ADD  x3, x1, x2   8
    instruction_rom[5] = 32'h00800213;  // ADDI x4, x0, 8
    instruction_rom[6] = 32'h0041C733;  // XOR  x14, x3, x4
    instruction_rom[7] = 32'h00E7E7B3;  // OR   x15, x15, x14
    instruction_rom[8] = 32'h402081B3;  // SUB  x3, x1, x2   2
    instruction_rom[9] = 32'h00200213;  // ADDI x4, x0, 2
    instruction_rom[10] = 32'h0041C733;  // XOR  x14, x3, x4
    instruction_rom[11] = 32'h00E7E7B3;  // OR   x15, x15, x14
    instruction_rom[12] = 32'h001121B3;  // SLT  x3, x2, x1   1
    instruction_rom[13] = 32'h00100213;  // ADDI x4, x0, 1
    instruction_rom[14] = 32'h0041C733;  // XOR  x14, x3, x4
    instruction_rom[15] = 32'h00E7E7B3;  // OR   x15, x15, x14
    instruction_rom[16] = 32'h0030B193;  // SLTIU x3, x1, 3   0, 5 is not below 3
    instruction_rom[17] = 32'h00000213;  // ADDI x4, x0, 0
    instruction_rom[18] = 32'h0041C733;  // XOR  x14, x3, x4
    instruction_rom[19] = 32'h00E7E7B3;  // OR   x15, x15, x14
    instruction_rom[20] = 32'h00409193;  // SLLI x3, x1, 4   5 shifted left by 4 is 80
    instruction_rom[21] = 32'h05000213;  // ADDI x4, x0, 80
    instruction_rom[22] = 32'h0041C733;  // XOR  x14, x3, x4
    instruction_rom[23] = 32'h00E7E7B3;  // OR   x15, x15, x14
    instruction_rom[24] = 32'hFF900193;  // ADDI x3, x0, -7
    instruction_rom[25] = 32'h4011D193;  // SRAI x3, x3, 1   -4, arithmetic shift keeps the sign
    instruction_rom[26] = 32'hFFC00213;  // ADDI x4, x0, -4
    instruction_rom[27] = 32'h0041C733;  // XOR  x14, x3, x4
    instruction_rom[28] = 32'h00E7E7B3;  // OR   x15, x15, x14
    instruction_rom[29] = 32'h0020C1B3;  // XOR  x3, x1, x2   5 xor 3 = 6
    instruction_rom[30] = 32'h0011C193;  // XORI x3, x3, 1   6 xor 1 = 7
    instruction_rom[31] = 32'h00700213;  // ADDI x4, x0, 7
    instruction_rom[32] = 32'h0041C733;  // XOR  x14, x3, x4
    instruction_rom[33] = 32'h00E7E7B3;  // OR   x15, x15, x14
    instruction_rom[34] = 32'h001231B7;  // LUI  x3, 0x00123   0x00123000
    instruction_rom[35] = 32'h00400293;  // ADDI x5, x0, 4   byte address 4 of the scratch RAM
    instruction_rom[36] = 32'h0032A023;  // SW   x3, 0(x5)   store word at byte address 4
    instruction_rom[37] = 32'h0002A303;  // LW   x6, 0(x5)
    instruction_rom[38] = 32'h0061C733;  // XOR  x14, x3, x6
    instruction_rom[39] = 32'h00E7E7B3;  // OR   x15, x15, x14
    instruction_rom[40] = 32'hED400193;  // ADDI x3, x0, -300
    instruction_rom[41] = 32'h0032A423;  // SW   x3, 8(x5)   store word at byte address 12
    instruction_rom[42] = 32'h00829303;  // LH   x6, 8(x5)   low half, sign extended back to -300
    instruction_rom[43] = 32'h00334733;  // XOR  x14, x6, x3
    instruction_rom[44] = 32'h00E7E7B3;  // OR   x15, x15, x14
    instruction_rom[45] = 32'h0082C303;  // LBU  x6, 8(x5)   low byte of -300 is 0xD4
    instruction_rom[46] = 32'h0D400213;  // ADDI x4, x0, 212   0xD4, the expected zero extended byte
    instruction_rom[47] = 32'h00434733;  // XOR  x14, x6, x4
    instruction_rom[48] = 32'h00E7E7B3;  // OR   x15, x15, x14
    instruction_rom[49] = 32'h00000463;  // BEQ  x0, x0, SKIP_POISON_A   taken branch must skip the poison
    instruction_rom[50] = 32'h00178793;  // ADDI x15, x15, 1   poison, must never execute
    instruction_rom[51] = 32'h00000613;  // ADDI x12, x0, 0
    instruction_rom[52] = 32'h00209463;  // BNE  x1, x2, SKIP_POISON_B   taken branch must skip the poison
    instruction_rom[53] = 32'h00278793;  // ADDI x15, x15, 2   poison, must never execute
    instruction_rom[54] = 32'h00114463;  // BLT  x2, x1, SKIP_POISON_C   3 < 5 is true
    instruction_rom[55] = 32'h00478793;  // ADDI x15, x15, 4   poison, must never execute
    instruction_rom[56] = 32'h0020F463;  // BGEU x1, x2, SKIP_POISON_D   5 >= 3 unsigned is true
    instruction_rom[57] = 32'h00878793;  // ADDI x15, x15, 8   poison, must never execute
    instruction_rom[58] = 32'h00115663;  // BGE  x2, x1, POISON_E   3 >= 5 is false: the branch must not be taken
    instruction_rom[59] = 32'h00100613;  // ADDI x12, x0, 1   the fall-through path, harmless
    instruction_rom[60] = 32'h00000463;  // BEQ  x0, x0, AFTER_E   jump over the poison sitting on the branch target
    instruction_rom[61] = 32'h01078793;  // ADDI x15, x15, 16   poison, reached only if a false branch was taken
    instruction_rom[62] = 32'h0080066F;  // JAL  x12, SKIP_POISON_F   unconditional jump, the link address goes to x12
    instruction_rom[63] = 32'h02078793;  // ADDI x15, x15, 32   poison, must never execute
    instruction_rom[64] = 32'h0FC00693;  // LINKADDI x13, SKIP_POISON_F   the link of a jump is its own address plus four
    instruction_rom[65] = 32'h40D60733;  // SUB  x14, x12, x13
    instruction_rom[66] = 32'h00E7E7B3;  // OR   x15, x15, x14
    instruction_rom[67] = 32'h17C00693;  // LIW  x13, SUB_ENTRY   absolute byte address of the subroutine entry
    instruction_rom[68] = 32'h00068767;  // JALR x14, x13, 0   call the subroutine, link in x14
    instruction_rom[69] = 32'h00000613;  // ADDI x12, x0, 0   the return must land on this line
    instruction_rom[70] = 32'h00000463;  // BEQ  x0, x0, AFTER_JALR   jump over the poison below
    instruction_rom[71] = 32'h04078793;  // ADDI x15, x15, 64   poison, reached only if the return missed the caller
    instruction_rom[72] = 32'h00700713;  // ADDI x14, x0, 7   the subroutine body must have added 7 to x11
    instruction_rom[73] = 32'h00E5C733;  // XOR  x14, x11, x14
    instruction_rom[74] = 32'h00E7E7B3;  // OR   x15, x15, x14
    instruction_rom[75] = 32'h00000617;  // AUIPC x12, 0   x12 = byte address of this very instruction
    instruction_rom[76] = 32'h12C00693;  // LIW  x13, PC_MARK
    instruction_rom[77] = 32'h00D64733;  // XOR  x14, x12, x13
    instruction_rom[78] = 32'h00E7E7B3;  // OR   x15, x15, x14
    instruction_rom[79] = 32'h00080093;  // ADDI x1, x16, 0: x16 does not exist in RV32E, must trap
    instruction_rom[80] = 32'h05A00393;  // ADDI x7, x0, 90   0x5A, an SMT input byte
    instruction_rom[81] = 32'h0003940B;  // LLSMT x8, x7   four clause lanes for that byte
    instruction_rom[82] = 32'h02800393;  // ADDI x7, x0, 40   0x28, the round open bracket
    instruction_rom[83] = 32'h0003A50B;  // LLGFS x10, x7   one GF(2^8) step, state in x10
    instruction_rom[84] = 32'h00200393;  // ADDI x7, x0, 2   mesh row index
    instruction_rom[85] = 32'h0003B58B;  // LLRDM x11, x7   read back row 2 of the mesh
    instruction_rom[86] = 32'h0000468B;  // LLCTL x13, x0   coprocessor counter, must read back three
    instruction_rom[87] = 32'h01043713;  // SLTIU x14, x8, 16   the SMT readback must fit the 4 result lanes
    instruction_rom[88] = 32'h00174713;  // XORI x14, x14, 1
    instruction_rom[89] = 32'h00E7E7B3;  // OR   x15, x15, x14
    instruction_rom[90] = 32'h00153713;  // SLTIU x14, x10, 1   every reachable GF state is nonzero
    instruction_rom[91] = 32'h00E7E7B3;  // OR   x15, x15, x14
    instruction_rom[92] = 32'h0017B713;  // SLTIU x14, x15, 1   x14 = 1 exactly when no check failed
    instruction_rom[93] = 32'h0007700B;  // LLHALT x14   stop and report pass = x14[0]
    instruction_rom[94] = 32'h00000013;  // NOP   safety slot, the core is already stopped
    instruction_rom[95] = 32'h00758593;  // ADDI x11, x11, 7   body of the subroutine the jump-and-link test calls
    instruction_rom[96] = 32'h00070067;  // JALR x0, x14, 0   return to the caller through the link held in x14
    instruction_rom[97] = 32'h00000013;  // NOP   unused program slot
    instruction_rom[98] = 32'h00000013;  // NOP   unused program slot
    instruction_rom[99] = 32'h00000013;  // NOP   unused program slot
    instruction_rom[100] = 32'h00000013;  // NOP   unused program slot
    instruction_rom[101] = 32'h00000013;  // NOP   unused program slot
    instruction_rom[102] = 32'h00000013;  // NOP   unused program slot
    instruction_rom[103] = 32'h00000013;  // NOP   unused program slot
    instruction_rom[104] = 32'h00000013;  // NOP   unused program slot
    instruction_rom[105] = 32'h00000013;  // NOP   unused program slot
    instruction_rom[106] = 32'h00000013;  // NOP   unused program slot
    instruction_rom[107] = 32'h00000013;  // NOP   unused program slot
    instruction_rom[108] = 32'h00000013;  // NOP   unused program slot
    instruction_rom[109] = 32'h00000013;  // NOP   unused program slot
    instruction_rom[110] = 32'h00000013;  // NOP   unused program slot
    instruction_rom[111] = 32'h00000013;  // NOP   unused program slot
    instruction_rom[112] = 32'h00000013;  // NOP   unused program slot
    instruction_rom[113] = 32'h00000013;  // NOP   unused program slot
    instruction_rom[114] = 32'h00000013;  // NOP   unused program slot
    instruction_rom[115] = 32'h00000013;  // NOP   unused program slot
    instruction_rom[116] = 32'h00000013;  // NOP   unused program slot
    instruction_rom[117] = 32'h00000013;  // NOP   unused program slot
    instruction_rom[118] = 32'h00000013;  // NOP   unused program slot
    instruction_rom[119] = 32'h00000013;  // NOP   unused program slot
    instruction_rom[120] = 32'h00000013;  // NOP   unused program slot
    instruction_rom[121] = 32'h00000013;  // NOP   unused program slot
    instruction_rom[122] = 32'h00000013;  // NOP   unused program slot
    instruction_rom[123] = 32'h00000013;  // NOP   unused program slot
    instruction_rom[124] = 32'h00000013;  // NOP   unused program slot
    instruction_rom[125] = 32'h00000013;  // NOP   unused program slot
    instruction_rom[126] = 32'h00000013;  // NOP   unused program slot
    instruction_rom[127] = 32'h00000013;  // NOP   unused program slot
    end

    // ---------------------------------------------------------------------------
    // Fetch
    // ---------------------------------------------------------------------------
    logic [IMEM_ADDR_BITS-1:0] pc;
    logic [IMEM_ADDR_BITS-1:0] next_pc;
    logic                 running;
    logic                 done;
    logic                 pass_flag;
    logic                 illegal_seen;
    logic [3:0]           cop_counter;
    logic [31:0]          cop_last;

    // Byte address to word index: the two low bits of a byte address are always zero
    // because every instruction is four bytes wide.
    wire [IMEM_BITS-1:0] rom_index = pc[IMEM_ADDR_BITS-1:2];
    wire [31:0] instruction = instruction_rom[rom_index];
    wire [31:0] pc_ext      = {{(32-IMEM_ADDR_BITS){1'b0}}, pc};

    assign core_pc          = pc;
    assign core_instruction = instruction;
    assign core_running     = running;
    assign core_done        = done;
    assign core_pass        = pass_flag;
    assign core_illegal_seen = illegal_seen;
    assign cop_count        = cop_counter;
    assign cop_last_rdata   = cop_last;

    // ---------------------------------------------------------------------------
    // Decode
    // ---------------------------------------------------------------------------
    wire [6:0] opcode = instruction[6:0];
    wire [2:0] funct3 = instruction[14:12];
    wire [6:0] funct7 = instruction[31:25];
    wire [4:0] rd_idx  = instruction[11:7];
    wire [4:0] rs1_idx = instruction[19:15];
    wire [4:0] rs2_idx = instruction[24:20];
    wire [4:0] shamt   = instruction[24:20];

    wire [31:0] imm_i = {{20{instruction[31]}}, instruction[31:20]};
    wire [31:0] imm_s = {{21{instruction[31]}}, instruction[30:25], instruction[11:7]};
    wire [31:0] imm_b = {{20{instruction[31]}}, instruction[7], instruction[30:25],
                         instruction[11:8], 1'b0};
    wire [31:0] imm_u = {instruction[31:12], 12'b0};
    wire [31:0] imm_j = {{11{instruction[31]}}, instruction[31], instruction[19:12],
                         instruction[20], instruction[30:21], 1'b0};

    // The RV32E rule: no register field may name a register above x15. Only the
    // fields the opcode actually reads count as register fields; on OP-IMM the bits
    // 24:20 belong to the immediate, and on STORE and BRANCH the bits 11:7 do, so a
    // legal immediate must not be mistaken for an out-of-range register name.
    wire uses_rd  = (opcode == OPC_LUI)   || (opcode == OPC_AUIPC) || (opcode == OPC_JAL)
                 || (opcode == OPC_JALR)  || (opcode == OPC_LOAD)  || (opcode == OPC_OP_IMM)
                 || (opcode == OPC_OP)    || (opcode == OPC_CUSTOM);
    wire uses_rs1 = (opcode == OPC_BRANCH) || (opcode == OPC_LOAD) || (opcode == OPC_STORE)
                 || (opcode == OPC_OP_IMM) || (opcode == OPC_OP)   || (opcode == OPC_JALR)
                 || (opcode == OPC_CUSTOM);
    wire uses_rs2 = (opcode == OPC_BRANCH) || (opcode == OPC_STORE) || (opcode == OPC_OP);
    wire register_space_ok = !((uses_rd && rd_idx[4] != 1'b0)
                            || (uses_rs1 && rs1_idx[4] != 1'b0)
                            || (uses_rs2 && rs2_idx[4] != 1'b0));

    logic encoding_ok;
    always_comb begin
        encoding_ok = 1'b0;
        unique case (opcode)
            OPC_LUI, OPC_AUIPC, OPC_JAL: begin
                encoding_ok = 1'b1;
            end
            OPC_JALR: begin
                encoding_ok = 1'b1;
            end
            OPC_BRANCH: begin
                encoding_ok = (funct3 != 3'b010) && (funct3 != 3'b011);
            end
            OPC_LOAD: begin
                encoding_ok = (funct3 != 3'b011) && (funct3 != 3'b110) && (funct3 != 3'b111);
            end
            OPC_STORE: begin
                encoding_ok = (funct3 == 3'b000) || (funct3 == 3'b001) || (funct3 == 3'b010);
            end
            OPC_OP_IMM: begin
                // ADDI, SLTI, SLTIU, XORI, ORI and ANDI take any immediate. The three
                // shift forms share this opcode and need a clean upper immediate, the
                // way the RV32E shift encodings are defined.
                if (funct3 == 3'b001) begin
                    encoding_ok = (funct7 == 7'b0000000);                       // SLLI
                end else if (funct3 == 3'b101) begin
                    encoding_ok = (funct7 == 7'b0000000) || (funct7 == 7'b0100000);  // SRLI, SRAI
                end else begin
                    encoding_ok = 1'b1;
                end
            end
            OPC_OP: begin
                if ((funct3 == 3'b000) || (funct3 == 3'b101)) begin
                    encoding_ok = (funct7 == 7'b0000000) || (funct7 == 7'b0100000);
                end else if ((funct3 == 3'b001) || (funct3 == 3'b010) || (funct3 == 3'b011)
                          || (funct3 == 3'b100) || (funct3 == 3'b110) || (funct3 == 3'b111)) begin
                    encoding_ok = (funct7 == 7'b0000000);
                end else begin
                    encoding_ok = 1'b0;
                end
            end
            OPC_MISC_MEM: begin
                // FENCE is a no-op here, FENCE.I has nothing to invalidate.
                encoding_ok = (funct3 == 3'b000);
            end
            OPC_CUSTOM: begin
                encoding_ok = (funct3 == 3'b001) || (funct3 == 3'b010) || (funct3 == 3'b011)
                           || (funct3 == 3'b100) || (funct3 == 3'b111);
            end
            default: begin
                // OPC_SYSTEM, OPC_OP_IMM_32, OPC_OP_32 and every unmapped opcode.
                encoding_ok = 1'b0;
            end
        endcase
    end

    wire instruction_ok = encoding_ok && register_space_ok;
    // The executing signal combines the encoding rules with the alignment rules of
    // the data memory and is declared with them, below.

    // ---------------------------------------------------------------------------
    // Register file
    // ---------------------------------------------------------------------------
    wire [31:0] rs1_value;
    wire [31:0] rs2_value;
    logic       reg_we;
    logic [31:0] reg_wdata;

    rv32e_regfile #(.REGS(16)) u_regfile (
        .clk      (clk),
        .rst_n    (rst_n),
        .we       (reg_we),
        .waddr    (rd_idx),
        .wdata    (reg_wdata),
        .raddr_a  (rs1_idx),
        .raddr_b  (rs2_idx),
        .rdata_a  (rs1_value),
        .rdata_b  (rs2_value),
        .reg_dump (core_reg_dump)
    );

    // For the immediate forms the second ALU operand is the immediate, which keeps a
    // single adder in the critical path of both the arithmetic and the address path.
    wire [31:0] operand_b = (opcode == OPC_OP) ? rs2_value : imm_i;

    // ---------------------------------------------------------------------------
    // ALU
    // ---------------------------------------------------------------------------
    // Only the register-register form can subtract: in the immediate form bit 30 of
    // the instruction is part of the immediate, so a negative ADDI would otherwise be
    // decoded as a subtract.
    wire subtract_request = (opcode == OPC_OP) && instruction[30];

    logic [31:0] alu_result;
    always_comb begin
        unique case (funct3)
            3'b000: alu_result = subtract_request ? (rs1_value - operand_b)
                                                  : (rs1_value + operand_b);
            3'b001: alu_result = rs1_value << shamt;
            3'b010: alu_result = 32'($signed(rs1_value) < $signed(operand_b));
            3'b011: alu_result = 32'(rs1_value < operand_b);
            3'b100: alu_result = rs1_value ^ operand_b;
            3'b101: alu_result = instruction[30] ? 32'($signed(rs1_value) >>> shamt)
                                                 : (rs1_value >> shamt);
            3'b110: alu_result = rs1_value | operand_b;
            3'b111: alu_result = rs1_value & operand_b;
            default: alu_result = 32'h0000_0000;
        endcase
    end

    wire branch_taken =
          ((funct3 == 3'b000) && (rs1_value == rs2_value))
       || ((funct3 == 3'b001) && (rs1_value != rs2_value))
       || ((funct3 == 3'b100) && ($signed(rs1_value) <  $signed(rs2_value)))
       || ((funct3 == 3'b101) && ($signed(rs1_value) >= $signed(rs2_value)))
       || ((funct3 == 3'b110) && (rs1_value <  rs2_value))
       || ((funct3 == 3'b111) && (rs1_value >= rs2_value));

    // ---------------------------------------------------------------------------
    // Data memory: on-chip scratch array, byte-addressed, access wraps at the end.
    // ---------------------------------------------------------------------------
    localparam int DMEM_INDEX_BITS = $clog2(DMEM_WORDS);
    logic [31:0] data_ram [DMEM_WORDS];

    wire [31:0] load_address  = rs1_value + imm_i;
    wire [31:0] store_address = rs1_value + imm_s;
    wire [DMEM_INDEX_BITS-1:0] load_index  = load_address[DMEM_INDEX_BITS+1:2];
    wire [DMEM_INDEX_BITS-1:0] store_index = store_address[DMEM_INDEX_BITS+1:2];
    wire [31:0] read_word = data_ram[load_index];

    // Natural alignment: halfword accesses need a two byte alignment, word accesses
    // a four byte alignment, byte accesses are always aligned. An access that is not
    // aligned is reported as an illegal instruction and is not performed.
    wire [31:0] access_address = (opcode == OPC_STORE) ? store_address : load_address;
    logic aligned_ok;
    always_comb begin
        unique case (funct3)
            3'b001, 3'b101: aligned_ok = (access_address[0] == 1'b0);
            3'b010:         aligned_ok = (access_address[1:0] == 2'b00);
            default:        aligned_ok = 1'b1;
        endcase
    end
    wire memory_access = (opcode == OPC_LOAD) || (opcode == OPC_STORE);
    wire legal_now     = instruction_ok && (memory_access ? aligned_ok : 1'b1);
    wire executing     = running && legal_now;

    // Byte lane selection. The halfword lane drops the lowest address bit so the
    // sixteen bit part-select always stays inside the word; an access that would
    // need the dropped bit is already reported as illegal.
    wire [1:0] load_byte_lane  = load_address[1:0];
    wire [1:0] load_half_lane  = {load_address[1], 1'b0};
    wire [1:0] store_byte_lane = store_address[1:0];
    wire [1:0] store_half_lane = {store_address[1], 1'b0};

    logic [31:0] load_result;
    always_comb begin
        unique case (funct3)
            3'b000: load_result = 32'($signed({{24{read_word[(load_byte_lane*8)+7]}},
                                               read_word[(load_byte_lane*8) +: 8]}));
            3'b001: load_result = 32'($signed({{16{read_word[(load_half_lane*8)+15]}},
                                               read_word[(load_half_lane*8) +: 16]}));
            3'b010: load_result = read_word;
            3'b100: load_result = 32'({{24{1'b0}}, read_word[(load_byte_lane*8) +: 8]});
            3'b101: load_result = 32'({{16{1'b0}}, read_word[(load_half_lane*8) +: 16]});
            default: load_result = 32'h0000_0000;
        endcase
    end

    wire do_store = executing && (opcode == OPC_STORE);
    always_ff @(posedge clk) begin
        if (do_store) begin
            // A store writes the low bits of rs2 into the selected lane of the word,
            // which is the RISC-V rule for aligned accesses. The assignments are the
            // delayed form, so the write commits at the clock edge: the load path of
            // the next cycle sees it, and the array never has a combinational read of
            // a value being written in the same expression.
            unique case (funct3)
                3'b000: data_ram[store_index][(store_byte_lane*8) +: 8]  <= rs2_value[7:0];
                3'b001: data_ram[store_index][(store_half_lane*8) +: 16] <= rs2_value[15:0];
                3'b010: data_ram[store_index] <= rs2_value;
                default: ;
            endcase
        end
    end

    // ---------------------------------------------------------------------------
    // Coprocessor request
    // ---------------------------------------------------------------------------
    logic cop_fire;
    always_comb begin
        cop_fire = 1'b0;
        cop_sel  = COP_SMT;
        unique case (funct3)
            3'b001: begin cop_fire = 1'b1; cop_sel = COP_SMT;  end
            3'b010: begin cop_fire = 1'b1; cop_sel = COP_GF;   end
            3'b011: begin cop_fire = 1'b1; cop_sel = COP_MESH; end
            3'b100: begin cop_fire = 1'b1; cop_sel = COP_STAT; end
            default: ;
        endcase
    end
    assign cop_req   = executing && (opcode == OPC_CUSTOM) && cop_fire;
    assign cop_wdata = rs1_value;

    wire halt_request = executing && (opcode == OPC_CUSTOM) && (funct3 == 3'b111);

    // ---------------------------------------------------------------------------
    // Write-back and next program counter
    // ---------------------------------------------------------------------------
    always_comb begin
        reg_we    = 1'b0;
        reg_wdata = 32'h0000_0000;
        if (executing) begin
            unique case (opcode)
                OPC_LUI:      begin reg_we = 1'b1; reg_wdata = imm_u; end
                OPC_AUIPC:    begin reg_we = 1'b1; reg_wdata = pc_ext + imm_u; end
                OPC_JAL:      begin reg_we = 1'b1; reg_wdata = pc_ext + 32'd4; end
                OPC_JALR:     begin reg_we = 1'b1; reg_wdata = pc_ext + 32'd4; end
                OPC_LOAD:     begin reg_we = 1'b1; reg_wdata = load_result; end
                OPC_OP_IMM, OPC_OP: begin
                    reg_we = 1'b1; reg_wdata = alu_result;
                end
                OPC_CUSTOM: begin
                    if (cop_fire) begin
                        reg_we = 1'b1; reg_wdata = cop_rdata;
                    end
                end
                default: begin
                    // OPC_BRANCH, OPC_STORE and OPC_MISC_MEM write no register.
                    reg_we = 1'b0;
                end
            endcase
        end
    end

    // jump_target is the full 32 bit address a taken branch or jump would land on,
    // and next_pc is its low IMEM_ADDR_BITS bits. The truncation is taken from a
    // named signal rather than from a parenthesized expression because SystemVerilog
    // allows a part select only on a name or a member access, so strict parsers
    // reject the form that puts the brackets straight after a closing parenthesis.
    // The wrap onto the trailing NOP slots of the program is unchanged.
    logic [31:0] jump_target;

    always_comb begin
        jump_target = pc_ext + 32'd4;
        if ((opcode == OPC_BRANCH) && branch_taken) begin
            jump_target = pc_ext + imm_b;
        end else if (executing && (opcode == OPC_JAL)) begin
            jump_target = pc_ext + imm_j;
        end else if (executing && (opcode == OPC_JALR)) begin
            jump_target = (rs1_value + imm_i) & ~32'b1;
        end
        next_pc = jump_target[IMEM_ADDR_BITS-1:0];
    end

    // ---------------------------------------------------------------------------
    // Sequential state
    // ---------------------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            pc           <= '0;
            running      <= 1'b0;
            done         <= 1'b0;
            pass_flag    <= 1'b0;
            illegal_seen <= 1'b0;
            cop_counter  <= 4'd0;
            cop_last     <= 32'h0000_0000;
        end else begin
            if (!running && !done && core_start) begin
                running <= 1'b1;
            end
            if (running) begin
                if (!legal_now) begin
                    illegal_seen <= 1'b1;
                end
                if (cop_req) begin
                    cop_counter <= cop_counter + 4'd1;
                    cop_last    <= cop_rdata;
                end
                if (halt_request) begin
                    running   <= 1'b0;
                    done      <= 1'b1;
                    pass_flag <= rs1_value[0];
                end else begin
                    pc <= next_pc;
                end
            end
        end
    end

endmodule
