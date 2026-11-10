// -----------------------------------------------------------------------------
// SystemVerilog testbench for the topology switch of the Edge-NSAC board.
//
// This bench exercises lingologos_topology_router.sv on its own, without the chip,
// because the module is not instantiated in lingologos_top.sv yet: it is lint clean
// and simulated here, and it is not in the source list of the chip configuration, so
// no synthesis run sees it. That distinction is recorded in the module header and in
// asic_chip/doc/verification_report.md.
//
// Four dies are wired together, three of them declared as neighbours of each other
// exactly as config/topology.json does, the fourth one present only to be the far end
// of a wire that the sender describes differently. All driving of the device under
// test happens on the falling edge of the clock, which is the middle of the interval
// that the rising edge samples, and every sample of a signal that the device drives
// happens a few nanoseconds after that falling edge, so no value is ever read in the
// same instant in which it is written.
//
// Run from the repository root through the make target sim-topology, whose command
// line is recorded in asic_chip/doc/build_instructions.md and in the report.
// -----------------------------------------------------------------------------

`timescale 1ns / 1ps

// The module name and the file name differ on purpose, as in the testbench of the top
// level: the file name is the one the flow documentation uses. The pragma pair records
// the mismatch instead of leaving it as a lint report surprise.
/* verilator lint_off DECLFILENAME */
module lingologos_topology_testbench;

    localparam int IDW   = 4;
    localparam int DW    = 64;
    localparam int NODES = 16;
    localparam int DIES  = 4;

    // The three working dies name each other as directly wired neighbours, which is what
    // the routing matrix of the declared topology produces for a complete graph. Node 3
    // has no declared neighbour, so a frame that reaches it is a board fault.
    localparam logic [NODES-1:0] M0_MESH = 16'b0000_0000_0000_0110;
    localparam logic [NODES-1:0] M1_MESH = 16'b0000_0000_0000_0101;
    localparam logic [NODES-1:0] M2_MESH = 16'b0000_0000_0000_0011;
    localparam logic [NODES-1:0] M3_NONE = 16'b0000_0000_0000_0000;

    // Frame payloads, distinct so that a misrouted word cannot pass as a correct one.
    localparam logic [DW-1:0] LOOP_WORD  = 64'hdead_beef_0000_0001;
    localparam logic [DW-1:0] W01        = 64'h0000_0000_0000_0010;
    localparam logic [DW-1:0] W02        = 64'h0000_0000_0000_0011;
    localparam logic [DW-1:0] W10        = 64'h0000_0000_0000_0020;
    localparam logic [DW-1:0] W12        = 64'h0000_0000_0000_0021;
    localparam logic [DW-1:0] W20        = 64'h0000_0000_0000_0030;
    localparam logic [DW-1:0] W21        = 64'h0000_0000_0000_0031;
    localparam logic [DW-1:0] W_REFUSE   = 64'h0000_0000_0000_0040;
    localparam logic [DW-1:0] W_BACK_A   = 64'h0000_0000_0000_0050;
    localparam logic [DW-1:0] W_BACK_B   = 64'h0000_0000_0000_0051;
    localparam logic [DW-1:0] W_BACK_C   = 64'h0000_0000_0000_0052;
    localparam logic [DW-1:0] W_BACK_D   = 64'h0000_0000_0000_0053;
    localparam logic [DW-1:0] W_BACK_E   = 64'h0000_0000_0000_0054;
    localparam logic [DW-1:0] W_FAR      = 64'h0000_0000_0000_0060;

    logic clk;
    logic rst_n;

    // Strap and register fields, one row per die.
    logic [DIES-1:0]              cfg_standalone;
    logic [DIES-1:0][3:0]         cfg_total_nodes;
    logic [DIES-1:0][NODES-1:0]   cfg_mask;
    logic [DIES-1:0][IDW-1:0]     cfg_node_id;

    // Ports of the local core of each die.
    logic [DIES-1:0]              inj_valid;
    logic [DIES-1:0]              inj_ready;
    logic [DIES-1:0][IDW-1:0]     inj_dest;
    logic [DIES-1:0][DW-1:0]      inj_data;
    logic [DIES-1:0]              out_valid;
    logic [DIES-1:0]              out_ready;
    logic [DIES-1:0][DW-1:0]      out_data;
    logic [DIES-1:0]              err_pulse;

    // Link channels of each die, indexed by the node id the channel is strapped to. They
    // are unpacked arrays because that is the shape the module declares for them.
    logic [NODES-1:0]          t0v, t0r, r0v, r0r;
    logic [NODES-1:0][IDW-1:0] t0d, r0d;
    logic [NODES-1:0][DW-1:0]  t0w, r0w;
    logic [NODES-1:0]          t1v, t1r, r1v, r1r;
    logic [NODES-1:0][IDW-1:0] t1d, r1d;
    logic [NODES-1:0][DW-1:0]  t1w, r1w;
    logic [NODES-1:0]          t2v, t2r, r2v, r2r;
    logic [NODES-1:0][IDW-1:0] t2d, r2d;
    logic [NODES-1:0][DW-1:0]  t2w, r2w;
    logic [NODES-1:0]          t3v, t3r, r3v, r3r;
    logic [NODES-1:0][IDW-1:0] t3d, r3d;
    logic [NODES-1:0][DW-1:0]  t3w, r3w;

    // The fault injection flag lands the wire of channel 2 of die 0 on die 3 instead of
    // on die 2, which is what a board wired differently from the mask looks like.
    logic miswire;

    lingologos_topology_router #(.NODE_ID_WIDTH(IDW), .DATA_WIDTH(DW), .MAX_NODES(NODES)) die0 (
        .clk(clk), .rst_n(rst_n),
        .standalone(cfg_standalone[0]), .node_id(cfg_node_id[0]),
        .total_nodes(cfg_total_nodes[0]), .routing_mask(cfg_mask[0]),
        .tx_valid(inj_valid[0]), .tx_ready(inj_ready[0]),
        .tx_dest(inj_dest[0]), .tx_data(inj_data[0]),
        .rx_valid(out_valid[0]), .rx_ready(out_ready[0]), .rx_data(out_data[0]),
        .link_tx_valid(t0v), .link_tx_ready(t0r), .link_tx_dest(t0d), .link_tx_data(t0w),
        .link_rx_valid(r0v), .link_rx_dest(r0d), .link_rx_data(r0w), .link_rx_ready(r0r),
        .routing_error(err_pulse[0])
    );

    lingologos_topology_router #(.NODE_ID_WIDTH(IDW), .DATA_WIDTH(DW), .MAX_NODES(NODES)) die1 (
        .clk(clk), .rst_n(rst_n),
        .standalone(cfg_standalone[1]), .node_id(cfg_node_id[1]),
        .total_nodes(cfg_total_nodes[1]), .routing_mask(cfg_mask[1]),
        .tx_valid(inj_valid[1]), .tx_ready(inj_ready[1]),
        .tx_dest(inj_dest[1]), .tx_data(inj_data[1]),
        .rx_valid(out_valid[1]), .rx_ready(out_ready[1]), .rx_data(out_data[1]),
        .link_tx_valid(t1v), .link_tx_ready(t1r), .link_tx_dest(t1d), .link_tx_data(t1w),
        .link_rx_valid(r1v), .link_rx_dest(r1d), .link_rx_data(r1w), .link_rx_ready(r1r),
        .routing_error(err_pulse[1])
    );

    lingologos_topology_router #(.NODE_ID_WIDTH(IDW), .DATA_WIDTH(DW), .MAX_NODES(NODES)) die2 (
        .clk(clk), .rst_n(rst_n),
        .standalone(cfg_standalone[2]), .node_id(cfg_node_id[2]),
        .total_nodes(cfg_total_nodes[2]), .routing_mask(cfg_mask[2]),
        .tx_valid(inj_valid[2]), .tx_ready(inj_ready[2]),
        .tx_dest(inj_dest[2]), .tx_data(inj_data[2]),
        .rx_valid(out_valid[2]), .rx_ready(out_ready[2]), .rx_data(out_data[2]),
        .link_tx_valid(t2v), .link_tx_ready(t2r), .link_tx_dest(t2d), .link_tx_data(t2w),
        .link_rx_valid(r2v), .link_rx_dest(r2d), .link_rx_data(r2w), .link_rx_ready(r2r),
        .routing_error(err_pulse[2])
    );

    lingologos_topology_router #(.NODE_ID_WIDTH(IDW), .DATA_WIDTH(DW), .MAX_NODES(NODES)) die3 (
        .clk(clk), .rst_n(rst_n),
        .standalone(cfg_standalone[3]), .node_id(cfg_node_id[3]),
        .total_nodes(cfg_total_nodes[3]), .routing_mask(cfg_mask[3]),
        .tx_valid(inj_valid[3]), .tx_ready(inj_ready[3]),
        .tx_dest(inj_dest[3]), .tx_data(inj_data[3]),
        .rx_valid(out_valid[3]), .rx_ready(out_ready[3]), .rx_data(out_data[3]),
        .link_tx_valid(t3v), .link_tx_ready(t3r), .link_tx_dest(t3d), .link_tx_data(t3w),
        .link_rx_valid(r3v), .link_rx_dest(r3d), .link_rx_data(r3w), .link_rx_ready(r3r),
        .routing_error(err_pulse[3])
    );

    // The wiring of the board. A channel of a die lands on the channel that names the
    // sending die at the receiving end, so a link is only a wire and no die has to know
    // the identity of the peer beyond the strap of that peer. Channels that are not
    // wired stay idle, which is the honest model of a board where two dies are not
    // neighbours of each other.
    always_comb begin
        r0v = '0; r0d = '0; r0w = '0;
        r1v = '0; r1d = '0; r1w = '0;
        r2v = '0; r2d = '0; r2w = '0;
        r3v = '0; r3d = '0; r3w = '0;
        t0r = '0; t1r = '0; t2r = '0; t3r = '0;

        // Die 0 to die 1.
        r1v[0] = t0v[1];
        r1d[0] = t0d[1];
        r1w[0] = t0w[1];
        t0r[1] = r1r[0];

        // Die 1 to die 0.
        r0v[1] = t1v[0];
        r0d[1] = t1d[0];
        r0w[1] = t1w[0];
        t1r[0] = r0r[1];

        // Die 1 to die 2 and back.
        r2v[1] = t1v[2];
        r2d[1] = t1d[2];
        r2w[1] = t1w[2];
        t1r[2] = r2r[1];
        r1v[2] = t2v[1];
        r1d[2] = t2d[1];
        r1w[2] = t2w[1];
        t2r[1] = r1r[2];

        // Die 2 to die 0.
        r0v[2] = t2v[0];
        r0d[2] = t2d[0];
        r0w[2] = t2w[0];
        t2r[0] = r0r[2];

        // Die 0 to die 2, or to die 3 when the fault injection flag is set.
        if (!miswire) begin
            r2v[0] = t0v[2];
            r2d[0] = t0d[2];
            r2w[0] = t0w[2];
            t0r[2] = r2r[0];
        end else begin
            r3v[0] = t0v[2];
            r3d[0] = t0d[2];
            r3w[0] = t0w[2];
            t0r[2] = r3r[0];
        end
    end

    // Watchers on the rising edge, which is the edge where a handshake completes: a
    // delivery is counted when valid and ready are both high as the clock rises, which is
    // also the moment the frame leaves the register. Watching the falling edge instead
    // would miss a frame whose ready signal was raised between a falling edge and the
    // rising edge that carries it away, and a scoreboard that misses a frame cannot tell
    // a lost frame from a frame that is still in flight.
    //
    // The counters below are a scoreboard, not a register file: a scoreboard has to be
    // visible to the stimulus processes in the same time step, which is what blocking
    // assignment gives and a non blocking one would hide. The pragma pair records that
    // choice instead of leaving it as a lint report surprise, the same way the module
    // name pragma at the top of this file records the file name mismatch. This block is
    // never synthesized: a testbench is not part of any source list of the flow.
    /* verilator lint_off BLKSEQ */
    localparam int RX_SLOTS = 10;

    int             rx_count [DIES];
    int             err_count [DIES];
    logic [DW-1:0]  rx_word [DIES][0:RX_SLOTS-1];
    logic [NODES-1:0] driven_channels [DIES];
    logic           link_activity [DIES];

    always @(posedge clk) begin
        for (int d = 0; d < DIES; d = d + 1) begin
            if (out_valid[d] && out_ready[d]) begin
                if (rx_count[d] < RX_SLOTS) begin
                    rx_word[d][rx_count[d]] = out_data[d];
                end
                rx_count[d] = rx_count[d] + 1;
            end
            if (err_pulse[d]) begin
                err_count[d] = err_count[d] + 1;
            end
        end
        driven_channels[0] = driven_channels[0] | t0v;
        driven_channels[1] = driven_channels[1] | t1v;
        driven_channels[2] = driven_channels[2] | t2v;
        driven_channels[3] = driven_channels[3] | t3v;
        if (|t0v) link_activity[0] = 1'b1;
        if (|t1v) link_activity[1] = 1'b1;
        if (|t2v) link_activity[2] = 1'b1;
        if (|t3v) link_activity[3] = 1'b1;
    end
    /* verilator lint_on BLKSEQ */

    int failures = 0;
    int checks = 0;

    task automatic expect_true(input bit condition, input string label);
        checks = checks + 1;
        if (!condition) begin
            failures = failures + 1;
            $display("[FAIL] %s", label);
        end
    endtask

    // Land in the middle of the interval that the next rising edge samples, so that a
    // signal read here is exactly the value the rising edge is going to see.
    task automatic align;
        @(negedge clk);
        #3;
    endtask

    // Hand one frame to the injection port of a die and report whether the port refused
    // it on the first attempt, which is what back pressure looks like to the core.
    task automatic inject_frame(input int die, input logic [IDW-1:0] dest,
                                input logic [DW-1:0] word, output bit accepted,
                                output bit refused_once);
        inj_valid[die] = 1'b1;
        inj_dest[die]  = dest;
        inj_data[die]  = word;
        accepted       = 1'b0;
        refused_once   = 1'b0;
        #3;
        forever begin
            if (inj_ready[die]) begin
                accepted = 1'b1;
                break;
            end
            refused_once = 1'b1;
            @(negedge clk);
            #3;
        end
        // The frame is taken by the stage of the injection port at the rising edge that
        // follows this sample point. The port has to be released after that edge and
        // before the next one, which is the falling edge that comes in between.
        @(negedge clk);
        inj_valid[die] = 1'b0;
    endtask

    // Wait until a die has taken delivery of the given number of frames, bounded so that
    // a broken fabric fails the verdict instead of hanging the simulator.
    task automatic wait_for_rx(input int die, input int want);
        int guard;
        guard = 0;
        align;
        while (rx_count[die] < want && guard < 300) begin
            @(negedge clk);
            #3;
            guard = guard + 1;
        end
    endtask

    initial begin
        clk = 1'b0;
        forever begin
            #5 clk = ~clk;
        end
    end

    initial begin
        #20000;
        $display("[FAIL] watchdog: the topology bench did not finish, deliveries are %0d %0d %0d %0d",
                 rx_count[0], rx_count[1], rx_count[2], rx_count[3]);
        $fatal(1, "FINAL VERDICT: FAIL");
    end

    initial begin
        rst_n = 1'b0;
        miswire = 1'b0;
        cfg_standalone  = 4'b1111;
        cfg_total_nodes = {4'd1, 4'd1, 4'd1, 4'd1};
        cfg_node_id     = {4'd3, 4'd2, 4'd1, 4'd0};
        cfg_mask        = {M3_NONE, M3_NONE, M3_NONE, M3_NONE};
        inj_valid = '0;
        inj_dest  = '0;
        inj_data  = '0;
        out_ready = 4'b1111;
        for (int d = 0; d < DIES; d = d + 1) begin
            rx_count[d] = 0;
            err_count[d] = 0;
            driven_channels[d] = '0;
            link_activity[d] = 1'b0;
            for (int w = 0; w < RX_SLOTS; w = w + 1) begin
                rx_word[d][w] = '0;
            end
        end

        align;
        align;
        rst_n = 1'b1;
        align;

        // Step A: a single die board loops back. TOTAL_NODES is one here, so the module
        // has to loop back even though no mask bit is set and the destination names a
        // node that does not exist on this board.
        begin
            bit accepted;
            bit refused_once;
            inject_frame(0, 4'd9, LOOP_WORD, accepted, refused_once);
            wait_for_rx(0, 1);
            expect_true(accepted, "step A: the injection port of a standalone die refused a frame");
            expect_true(!refused_once, "step A: a one die board back pressed its own loopback");
            expect_true(rx_count[0] == 1, "step A: the loopback did not deliver exactly once");
            expect_true(rx_word[0][0] === LOOP_WORD, "step A: the looped word changed on the way");
            expect_true(!link_activity[0], "step A: a standalone die drove a link channel");
            expect_true(err_count[0] == 0, "step A: a standalone die reported a routing error");
            $display("[INFO] step A, loopback on a one die board: deliveries %0d, errors %0d",
                     rx_count[0], err_count[0]);
        end

        // Step B: the declared board. Every working die names the other two as wired
        // neighbours, so each of the six directed pairs of the declared topology is a
        // single hop and no die has to relay anything.
        begin
            bit accepted;
            bit refused_once;
            int base0;
            int base1;
            int base2;
            base0 = rx_count[0];
            base1 = rx_count[1];
            base2 = rx_count[2];
            cfg_standalone  = 4'b0000;
            cfg_total_nodes = {4'd4, 4'd3, 4'd3, 4'd3};
            cfg_mask        = {M3_NONE, M2_MESH, M1_MESH, M0_MESH};
            align;
            inject_frame(0, 4'd1, W01, accepted, refused_once);
            inject_frame(0, 4'd2, W02, accepted, refused_once);
            inject_frame(1, 4'd0, W10, accepted, refused_once);
            inject_frame(1, 4'd2, W12, accepted, refused_once);
            inject_frame(2, 4'd0, W20, accepted, refused_once);
            inject_frame(2, 4'd1, W21, accepted, refused_once);
            wait_for_rx(0, base0 + 2);
            wait_for_rx(1, base1 + 2);
            wait_for_rx(2, base2 + 2);
            expect_true(rx_count[0] == base0 + 2, "step B: die 0 did not receive its two frames");
            expect_true(rx_count[1] == base1 + 2, "step B: die 1 did not receive its two frames");
            expect_true(rx_count[2] == base2 + 2, "step B: die 2 did not receive its two frames");
            expect_true(rx_count[3] == 0, "step B: the undeclared die received a frame");
            expect_true(((rx_word[0][base0] === W10) && (rx_word[0][base0 + 1] === W20))
                        || ((rx_word[0][base0] === W20) && (rx_word[0][base0 + 1] === W10)),
                        "step B: die 0 received the wrong pair of words");
            expect_true(((rx_word[1][base1] === W01) && (rx_word[1][base1 + 1] === W21))
                        || ((rx_word[1][base1] === W21) && (rx_word[1][base1 + 1] === W01)),
                        "step B: die 1 received the wrong pair of words");
            expect_true(((rx_word[2][base2] === W02) && (rx_word[2][base2 + 1] === W12))
                        || ((rx_word[2][base2] === W12) && (rx_word[2][base2 + 1] === W02)),
                        "step B: die 2 received the wrong pair of words");
            expect_true(err_count[0] == 0 && err_count[1] == 0 && err_count[2] == 0,
                        "step B: a declared edge was refused");
            expect_true(driven_channels[0] == M0_MESH,
                        "step B: die 0 drove a channel its mask does not name");
            $display("[INFO] step B words at die 0: %h %h, deliveries %0d %0d %0d",
                     rx_word[0][base0], rx_word[0][base0 + 1],
                     rx_count[0], rx_count[1], rx_count[2]);
        end

        // Step C: a mask that does not name the destination. The frame is consumed by the
        // die that injected it and a single pulse comes out, so the port of the core is
        // not left hanging by a frame that cannot go anywhere.
        begin
            bit accepted;
            bit refused_once;
            int errors_before;
            int delivered_before;
            errors_before = err_count[0];
            delivered_before = rx_count[2];
            cfg_mask[0] = 16'b0000_0000_0000_0010;
            align;
            inject_frame(0, 4'd2, W_REFUSE, accepted, refused_once);
            align;
            align;
            expect_true(err_count[0] == errors_before + 1, "step C: the refusal pulse did not arrive");
            expect_true(rx_count[2] == delivered_before, "step C: a refused frame was delivered anyway");
            expect_true(!refused_once, "step C: the injection port stayed busy after an unroutable frame");
            expect_true(err_count[2] == 0, "step C: die 2 reported an error it should not");
            $display("[INFO] step C, one frame with no neighbour for its destination: errors at die 0 = %0d",
                     err_count[0]);
            cfg_mask[0] = M0_MESH;
            align;
        end

        // Step D: back pressure. The delivery port of die 1 is held down, and the fabric
        // has to soak up as many frames as it can buffer before the injection port of
        // die 0 starts refusing them. One frame reaches the delivery register of die 1,
        // the next one sits in the capture register of that channel, the third one waits
        // on the wire in the send register of die 0, the fourth one waits in the
        // injection stage of die 0, and only the fifth one is refused by the port.
        begin
            bit accepted;
            bit refused_once;
            int delivered_before;
            delivered_before = rx_count[1];
            out_ready[1] = 1'b0;
            inject_frame(0, 4'd1, W_BACK_A, accepted, refused_once);
            expect_true(accepted && !refused_once, "step D: the first frame was refused too early");
            inject_frame(0, 4'd1, W_BACK_B, accepted, refused_once);
            expect_true(accepted && !refused_once, "step D: the second frame was refused too early");
            inject_frame(0, 4'd1, W_BACK_C, accepted, refused_once);
            expect_true(accepted && !refused_once, "step D: the third frame was refused too early");
            inject_frame(0, 4'd1, W_BACK_D, accepted, refused_once);
            expect_true(accepted && !refused_once, "step D: the fourth frame was refused too early");
            align;
            expect_true(rx_count[1] == delivered_before,
                        "step D: a frame was handed to the core of die 1 while its port was blocked");
            expect_true(out_valid[1] && !out_ready[1],
                        "step D: die 1 did not hold a frame waiting for the core of that die");
            // The fabric is full, so the fifth frame has to be refused by the port.
            inj_valid[0] = 1'b1;
            inj_dest[0]  = 4'd1;
            inj_data[0]  = W_BACK_E;
            #3;
            expect_true(!inj_ready[0], "step D: the fifth frame was accepted although the fabric was full");
            repeat (3) begin
                @(negedge clk);
                #3;
                expect_true(!inj_ready[0], "step D: the port freed while the peer stayed blocked");
            end
            expect_true(rx_count[1] == delivered_before,
                        "step D: a frame arrived while the delivery port was still blocked");
            $display("[INFO] step D, blocked port of die 1: four frames absorbed, the fifth one refused");

            // Release the blocked die: the port frees, the fifth frame is handed over, and
            // all five words have to arrive in the order they were injected.
            out_ready[1] = 1'b1;
            while (!inj_ready[0]) begin
                @(negedge clk);
                #3;
            end
            @(negedge clk);
            inj_valid[0] = 1'b0;
            wait_for_rx(1, delivered_before + 5);
            $display("[INFO] step D words at die 1: %h %h %h %h %h",
                     rx_word[1][delivered_before],     rx_word[1][delivered_before + 1],
                     rx_word[1][delivered_before + 2], rx_word[1][delivered_before + 3],
                     rx_word[1][delivered_before + 4]);
            expect_true(rx_count[1] == delivered_before + 5,
                        "step D: not all five buffered frames arrived at die 1");
            expect_true(rx_word[1][delivered_before]     === W_BACK_A,
                        "step D: the first buffered frame is not the first delivered");
            expect_true(rx_word[1][delivered_before + 1] === W_BACK_B,
                        "step D: the second buffered frame arrived out of order");
            expect_true(rx_word[1][delivered_before + 2] === W_BACK_C,
                        "step D: the third buffered frame arrived out of order");
            expect_true(rx_word[1][delivered_before + 3] === W_BACK_D,
                        "step D: the fourth buffered frame arrived out of order");
            expect_true(rx_word[1][delivered_before + 4] === W_BACK_E,
                        "step D: the frame that was refused by the port is not the last one");
            expect_true(err_count[0] == 1, "step D: die 0 reported more errors than the declared refusal");
            $display("[INFO] step D, deliveries at die 1: %0d frames, order preserved, no loss",
                     rx_count[1]);
        end

        // Step E: the board and the mask disagree. The wire of channel 2 of die 0 is
        // landed on die 3, so die 0 forwards a frame for node 2 onto a wire that does not
        // lead to node 2 and die 3 refuses a frame that is not addressed to it.
        begin
            bit accepted;
            bit refused_once;
            int guard;
            miswire = 1'b1;
            align;
            inject_frame(0, 4'd2, W_FAR, accepted, refused_once);
            // The refusal of a frame that is not addressed to a die comes later than the
            // refusal of a frame that never left the die which injected it: this one has
            // to cross the wire and be read by the capture register of the far die first.
            guard = 0;
            align;
            while (err_count[3] == 0 && guard < 20) begin
                @(negedge clk);
                #3;
                guard = guard + 1;
            end
            expect_true(err_count[3] == 1,
                        "step E: the die at the far end of the wrong wire stayed silent");
            expect_true(err_count[0] == 1, "step E: die 0 reported an error it should not");
            expect_true(rx_count[2] == 2, "step E: the intended destination received the frame");
            expect_true(rx_count[3] == 0, "step E: the wrong wire delivered a frame");
            expect_true(rx_word[3][0] === '0, "step E: the refusing die latched a frame anyway");
            $display("[INFO] step E, a wire that does not match the mask: errors at die 3 = %0d",
                     err_count[3]);
            miswire = 1'b0;
            align;
        end

        align;
        if (failures > 0) begin
            $display("checked %0d topology assertions, %0d mismatches", checks, failures);
            $fatal(1, "FINAL VERDICT: FAIL");
        end
        $display("checked %0d topology assertions on four dies, 0 mismatches", checks);
        $display("deliveries per die: %0d %0d %0d %0d, refusal pulses: %0d %0d %0d %0d",
                 rx_count[0], rx_count[1], rx_count[2], rx_count[3],
                 err_count[0], err_count[1], err_count[2], err_count[3]);
        $display("FINAL VERDICT: PASS");
        $finish;
    end

endmodule
/* verilator lint_on DECLFILENAME */
