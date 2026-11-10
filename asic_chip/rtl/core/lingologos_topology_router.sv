// lingologos_topology_router.sv
// -----------------------------------------------------------------------------
// Adjacency masked topology switch for the LingoLOGOS Edge-NSAC board.
//
// This module owns the fabric half of TOPOLOGY_CTRL. STANDALONE, NODE_ID,
// TOTAL_NODES and ROUTING_MASK are decoded here. DATASET_ID is deliberately absent:
// it selects the corpus shard the firmware loads, it never reaches a datapath, and
// giving it a wire here would suggest the fabric can act on a shard number.
//
// A die exposes one injection port, one receive port, and one full duplex link
// channel per node id, where channel index k is the wire towards node k.
// ROUTING_MASK bit k names the node ids this die is wired to, so delivery is single
// hop: a frame leaves on the channel whose index is the destination, and a die that
// is not adjacent to the destination refuses the frame and pulses routing_error
// rather than relaying it. That is a decision, not an omission. A 16 bit mask cannot
// carry a path, and a frame relayed across a board wired differently from the mask is
// far harder to diagnose than one that stops at the die which cannot carry it. The
// topology the orchestrator negotiates from config/topology.json sets one mask bit per
// declared routing rule, so every declared edge is a direct link and single hop
// delivery covers the declared plan. The gate test named
// test_every_declared_edge_is_a_direct_link in tests/test_topology_router.py is what
// keeps that claim checked instead of merely asserted.
//
// Every output is a register or a function of local registers only, so chaining two
// dies never creates a combinational path between them. A frame appears on its output
// channel two cycles after it was accepted: one cycle to capture, one to route. Back
// pressure is real. A blocked receive port holds the frame on the wire that carried
// it, and a busy link channel holds the frame on the injection port.
//
// Status in this repository: lint clean and simulated under the flags of the Makefile,
// not synthesized. The chip configuration of asic_chip/config.json does not list this
// file under rtl_core, which is what keeps it out of the OpenLane source list until the
// package has pins for the link channels and for the straps of NODE_ID.
// -----------------------------------------------------------------------------

module lingologos_topology_router #(
    parameter int NODE_ID_WIDTH = 4,
    parameter int DATA_WIDTH    = 64,
    parameter int MAX_NODES     = 16
)(
    input  logic                                    clk,
    input  logic                                    rst_n,

    // Fields of TOPOLOGY_CTRL as decoded by the control and status layer.
    input  logic                                    standalone,
    input  logic [NODE_ID_WIDTH-1:0]                node_id,
    input  logic [3:0]                              total_nodes,
    input  logic [MAX_NODES-1:0]                    routing_mask,

    // Injection port of the local core.
    input  logic                                    tx_valid,
    output logic                                    tx_ready,
    input  logic [NODE_ID_WIDTH-1:0]                tx_dest,
    input  logic [DATA_WIDTH-1:0]                   tx_data,

    // Receive port of the local core.
    output logic                                    rx_valid,
    input  logic                                    rx_ready,
    output logic [DATA_WIDTH-1:0]                   rx_data,

    // One full duplex link channel per node id: index k is the wire to node k. The
    // destination travels with the frame, so a link carries no protocol state and a
    // board wrapper is only a wire between two identically configured dies.
    output logic [MAX_NODES-1:0]                    link_tx_valid,
    input  logic [MAX_NODES-1:0]                    link_tx_ready,
    output logic [MAX_NODES-1:0][NODE_ID_WIDTH-1:0] link_tx_dest,
    output logic [MAX_NODES-1:0][DATA_WIDTH-1:0]    link_tx_data,
    input  logic [MAX_NODES-1:0]                    link_rx_valid,
    input  logic [MAX_NODES-1:0][NODE_ID_WIDTH-1:0] link_rx_dest,
    input  logic [MAX_NODES-1:0][DATA_WIDTH-1:0]    link_rx_data,
    output logic [MAX_NODES-1:0]                    link_rx_ready,

    // One cycle pulse, set when this die refused a frame. A refusal happens when the
    // mask of this die does not name the destination of an injected frame as a direct
    // neighbour, or when a frame arrives on a link for a destination that is not this
    // die. Both cases mean the board and the mask disagree.
    output logic                                    routing_error
);

    // Stage registers of the injection port.
    logic                        inj_valid_q;
    logic [NODE_ID_WIDTH-1:0]    inj_dest_q;
    logic [DATA_WIDTH-1:0]       inj_data_q;

    // Capture registers, one per incoming link channel.
    logic                        cap_valid_q [MAX_NODES];
    logic [NODE_ID_WIDTH-1:0]    cap_dest_q  [MAX_NODES];
    logic [DATA_WIDTH-1:0]       cap_data_q  [MAX_NODES];

    // Send registers, one per outgoing link channel.
    logic                        send_valid_q [MAX_NODES];
    logic [NODE_ID_WIDTH-1:0]    send_dest_q  [MAX_NODES];
    logic [DATA_WIDTH-1:0]       send_data_q  [MAX_NODES];

    // Receive register of the local core, and the one cycle error pulse.
    logic                        rx_valid_q;
    logic [DATA_WIDTH-1:0]       rx_data_q;
    logic                        error_q;

    // Control wires, all of them functions of the stage registers alone.
    logic                        rx_free_now;
    logic                        rx_taken;
    logic                        rx_load;
    logic [DATA_WIDTH-1:0]       rx_data_mux;
    logic [DATA_WIDTH-1:0]       cap_data_mux;
    logic                        inj_to_rx;
    logic                        inj_to_link;
    logic                        inj_refuse;
    logic                        error_next;
    logic [MAX_NODES-1:0]        cap_to_rx;
    logic [MAX_NODES-1:0]        cap_refuse;
    logic [MAX_NODES-1:0]        cap_ready;
    logic [MAX_NODES-1:0]        send_free;
    logic [MAX_NODES-1:0]        send_load;

    // A board of one die is standalone whatever the strapping says, so TOTAL_NODES is
    // not a decorative field: it forces the loopback path on a single chip board.
    wire loopback_mode = standalone || (total_nodes <= 4'd1);

    // Two combinational blocks, not one. The block below drives the capture side of the
    // incoming links and reads stage registers and the receive port only, so the ready
    // signal a die presents to the board never depends on the ready signal it receives
    // from the board. That separation is what keeps a chain of dies acyclic, which the
    // header of this file claims and which Verilator reports as UNOPTFLAT when it is
    // violated. The second block drives the send side, where depending on link_tx_ready
    // is exactly the point of back pressure.
    always_comb begin
        rx_free_now  = 1'b0;
        rx_taken     = 1'b0;
        cap_data_mux = '0;
        cap_to_rx    = '0;
        cap_refuse   = '0;
        cap_ready    = '0;

        rx_free_now = ~rx_valid_q || (rx_valid_q && rx_ready);

        // A frame that arrived from the board gets the receive register before a frame
        // the local core injects into itself. The local core can wait on its injection
        // port, while an arriving frame cannot be unreceived.
        for (int i = 0; i < MAX_NODES; i = i + 1) begin
            if (cap_valid_q[i] && (cap_dest_q[i] == node_id) && rx_free_now && !rx_taken) begin
                cap_to_rx[i] = 1'b1;
                rx_taken     = 1'b1;
                cap_data_mux = cap_data_q[i];
            end
        end
        for (int i = 0; i < MAX_NODES; i = i + 1) begin
            if (cap_valid_q[i] && (cap_dest_q[i] != node_id)) begin
                cap_refuse[i] = 1'b1;
            end
            // A capture register frees in the same cycle its frame is taken away or
            // dropped, so the wire that carried it can present ready again at once.
            cap_ready[i] = ~cap_valid_q[i] || cap_to_rx[i] || cap_refuse[i];
            link_rx_ready[i] = cap_ready[i];
        end
    end

    always_comb begin
        inj_to_rx   = 1'b0;
        inj_to_link = 1'b0;
        inj_refuse  = 1'b0;
        tx_ready    = 1'b0;
        rx_load     = 1'b0;
        rx_data_mux = '0;
        error_next  = 1'b0;
        send_free   = '0;
        send_load   = '0;

        for (int i = 0; i < MAX_NODES; i = i + 1) begin
            // A send channel frees in the same cycle its frame is taken by the peer, so
            // a new frame can be written into it without inserting a bubble. The link
            // outputs are driven here, one channel at a time, because an unpacked array
            // of registers is not assignment compatible with a packed output vector.
            send_free[i] = ~send_valid_q[i] || (send_valid_q[i] && link_tx_ready[i]);
            link_tx_valid[i] = send_valid_q[i];
            link_tx_dest[i]  = send_dest_q[i];
            link_tx_data[i]  = send_data_q[i];
        end

        if (inj_valid_q) begin
            if (loopback_mode || (inj_dest_q == node_id)) begin
                if (rx_free_now && !rx_taken) begin
                    inj_to_rx = 1'b1;
                end
            end else if (routing_mask[inj_dest_q]) begin
                if (send_free[inj_dest_q]) begin
                    inj_to_link           = 1'b1;
                    send_load[inj_dest_q] = 1'b1;
                end
            end else begin
                inj_refuse = 1'b1;
                error_next = 1'b1;
            end
        end

        // A frame of an arriving channel claims the receive register, and an injected
        // frame takes it only when no arriving frame is already claiming it.
        if (inj_to_rx) begin
            rx_load     = 1'b1;
            rx_data_mux = inj_data_q;
        end
        if (rx_taken) begin
            rx_load     = 1'b1;
            rx_data_mux = cap_data_mux;
        end

        // The injection port accepts when its stage is empty, or when the frame it holds
        // leaves during this cycle, in which case the new frame replaces it.
        tx_ready   = ~inj_valid_q || inj_to_rx || inj_to_link || inj_refuse;
        error_next = error_next || (|cap_refuse);
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            inj_valid_q <= 1'b0;
            inj_dest_q  <= '0;
            inj_data_q  <= '0;
            rx_valid_q  <= 1'b0;
            rx_data_q   <= '0;
            error_q     <= 1'b0;
            for (int i = 0; i < MAX_NODES; i = i + 1) begin
                cap_valid_q[i]  <= 1'b0;
                cap_dest_q[i]   <= '0;
                cap_data_q[i]   <= '0;
                send_valid_q[i] <= 1'b0;
                send_dest_q[i]  <= '0;
                send_data_q[i]  <= '0;
            end
        end else begin
            error_q <= error_next;

            if (tx_valid && tx_ready) begin
                inj_valid_q <= 1'b1;
                inj_dest_q  <= tx_dest;
                inj_data_q  <= tx_data;
            end else if (inj_to_rx || inj_to_link || inj_refuse) begin
                inj_valid_q <= 1'b0;
            end

            for (int i = 0; i < MAX_NODES; i = i + 1) begin
                if (link_rx_valid[i] && cap_ready[i]) begin
                    cap_valid_q[i] <= 1'b1;
                    cap_dest_q[i]  <= link_rx_dest[i];
                    cap_data_q[i]  <= link_rx_data[i];
                end else if (cap_to_rx[i] || cap_refuse[i]) begin
                    cap_valid_q[i] <= 1'b0;
                end
            end

            for (int i = 0; i < MAX_NODES; i = i + 1) begin
                if (send_load[i]) begin
                    send_valid_q[i] <= 1'b1;
                    send_dest_q[i]  <= inj_dest_q;
                    send_data_q[i]  <= inj_data_q;
                end else if (send_valid_q[i] && link_tx_ready[i]) begin
                    send_valid_q[i] <= 1'b0;
                end
            end

            if (rx_load) begin
                rx_valid_q <= 1'b1;
                rx_data_q  <= rx_data_mux;
            end else if (rx_valid_q && rx_ready) begin
                rx_valid_q <= 1'b0;
            end
        end
    end

    assign rx_valid      = rx_valid_q;
    assign rx_data       = rx_data_q;
    assign routing_error = error_q;

endmodule
