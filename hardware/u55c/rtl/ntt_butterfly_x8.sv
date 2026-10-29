`timescale 1ns/1ps
module ntt_butterfly_x8 #(
    parameter int MUL_LATENCY = 4,
    parameter int LANES = 8
) (
    input  logic clk,
    input  logic rst_n,
    input  logic in_valid,
    input  logic [LANES-1:0][63:0] u,
    input  logic [LANES-1:0][63:0] v,
    input  logic [LANES-1:0][63:0] w_mont,
    input  logic [63:0] q,
    input  logic [63:0] q_inv,
    output logic out_valid,
    output logic [LANES-1:0][63:0] sum,
    output logic [LANES-1:0][63:0] diff
);
    localparam int MRED_LATENCY = 3 * MUL_LATENCY + 1;
    logic [63:0] u_delay [0:LANES-1][0:MRED_LATENCY-1];
    logic [63:0] q_delay [0:MRED_LATENCY-1];
    logic [63:0] product [0:LANES-1];
    logic lane_valid [0:LANES-1];
    integer i, j;

    generate
        genvar g;
        for (g = 0; g < LANES; g = g + 1) begin : multipliers
            mred64_pipeline #(.MUL_LATENCY(MUL_LATENCY)) mul (
                .clk, .rst_n, .in_valid, .x(v[g]), .y(w_mont[g]),
                .q, .q_inv, .out_valid(lane_valid[g]), .r(product[g])
            );
        end
    endgenerate

    always @(posedge clk) begin
        if (!rst_n) begin
            for (i = 0; i < LANES; i = i + 1)
                for (j = 0; j < MRED_LATENCY; j = j + 1)
                    u_delay[i][j] <= '0;
            for (j = 0; j < MRED_LATENCY; j = j + 1)
                q_delay[j] <= '0;
        end else begin
            q_delay[0] <= q;
            for (i = 0; i < LANES; i = i + 1)
                u_delay[i][0] <= u[i];
            for (j = 1; j < MRED_LATENCY; j = j + 1) begin
                q_delay[j] <= q_delay[j-1];
                for (i = 0; i < LANES; i = i + 1)
                    u_delay[i][j] <= u_delay[i][j-1];
            end
        end
    end

    assign out_valid = lane_valid[0];
    generate
        for (g = 0; g < LANES; g = g + 1) begin : modular_add_sub
            wire [64:0] add_wide = {1'b0, u_delay[g][MRED_LATENCY-1]} +
                                   {1'b0, product[g]};
            wire [64:0] sub_wide = {1'b0, u_delay[g][MRED_LATENCY-1]} +
                                   {1'b0, q_delay[MRED_LATENCY-1]} -
                                   {1'b0, product[g]};
            assign sum[g] = add_wide >= {1'b0, q_delay[MRED_LATENCY-1]} ?
                            add_wide - {1'b0, q_delay[MRED_LATENCY-1]} : add_wide;
            assign diff[g] = u_delay[g][MRED_LATENCY-1] >= product[g] ?
                             u_delay[g][MRED_LATENCY-1] - product[g] : sub_wide;
        end
    endgenerate
endmodule
