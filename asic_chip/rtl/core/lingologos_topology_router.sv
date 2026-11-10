// lingologos_topology_router.sv
// Dynamic topology router and loopback module for LingoLOGOS Edge-NSAC board.
// Supports standalone single-chip loopback mode and multi-chip mesh forwarding.

module lingologos_topology_router #(
    parameter int NODE_ID_WIDTH = 4,
    parameter int DATA_WIDTH    = 64
)(
    input  logic                     clk,
    input  logic                     rst_n,
    input  logic                     standalone_mode,
    input  logic [NODE_ID_WIDTH-1:0] current_node_id,
    input  logic                     tx_valid_local,
    input  logic [NODE_ID_WIDTH-1:0] tx_dest_node,
    input  logic [DATA_WIDTH-1:0]    tx_data_local,
    output logic                     tx_ready_local,
    output logic                     rx_valid_local,
    output logic [DATA_WIDTH-1:0]    rx_data_local,
    input  logic                     rx_ready_local,
    input  logic                     rx_valid_ext,
    input  logic [DATA_WIDTH-1:0]    rx_data_ext,
    output logic                     tx_valid_ext,
    output logic [DATA_WIDTH-1:0]    tx_data_ext
);

    always_comb begin
        if (standalone_mode) begin
            rx_valid_local = tx_valid_local;
            rx_data_local  = tx_data_local;
            tx_ready_local = rx_ready_local;
            tx_valid_ext   = 1'b0;
            tx_data_ext    = '0;
        end else begin
            if (tx_dest_node == current_node_id) begin
                rx_valid_local = tx_valid_local;
                rx_data_local  = tx_data_local;
                tx_ready_local = rx_ready_local;
                tx_valid_ext   = 1'b0;
                tx_data_ext    = '0;
            end else begin
                tx_valid_ext   = tx_valid_local;
                tx_data_ext    = tx_data_local;
                tx_ready_local = 1'b1;
                rx_valid_local = rx_valid_ext;
                rx_data_local  = rx_data_ext;
            end
        end
    end

endmodule
