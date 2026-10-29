`timescale 1ns/1ps
module mred64_pipeline #(
    parameter int MUL_LATENCY = 4
) (
    input  logic        clk,
    input  logic        rst_n,
    input  logic        in_valid,
    input  logic [63:0] x,
    input  logic [63:0] y,
    input  logic [63:0] q,
    input  logic [63:0] q_inv,
    output logic        out_valid,
    output logic [63:0] r
);
    localparam int LATENCY = 3 * MUL_LATENCY + 1;
    logic [LATENCY-1:0] valid_pipe;

    (* keep = "true", shreg_extract = "no" *) logic [127:0] xy_pipe [0:MUL_LATENCY-1];
    (* keep = "true", shreg_extract = "no" *) logic [63:0] q_s1 [0:MUL_LATENCY-1];
    (* keep = "true", shreg_extract = "no" *) logic [63:0] qi_s1[0:MUL_LATENCY-1];

    (* keep = "true", shreg_extract = "no" *) logic [127:0] mi_pipe [0:MUL_LATENCY-1];
    (* keep = "true", shreg_extract = "no" *) logic [63:0] xyhi_s2[0:MUL_LATENCY-1];
    (* keep = "true", shreg_extract = "no" *) logic [63:0] q_s2[0:MUL_LATENCY-1];

    (* keep = "true", shreg_extract = "no" *) logic [127:0] mq_pipe [0:MUL_LATENCY-1];
    (* keep = "true", shreg_extract = "no" *) logic [63:0] xyhi_s3[0:MUL_LATENCY-1];
    (* keep = "true", shreg_extract = "no" *) logic [63:0] q_s3[0:MUL_LATENCY-1];

    integer i;
    logic [64:0] reduced;

    always @(posedge clk) begin
        if (!rst_n) begin
            valid_pipe <= '0;
            r <= '0;
        end else begin
            valid_pipe <= {valid_pipe[LATENCY-2:0], in_valid};

            xy_pipe[0] <= x * y;
            q_s1[0] <= q;
            qi_s1[0] <= q_inv;
            for (i = 1; i < MUL_LATENCY; i = i + 1) begin
                xy_pipe[i] <= xy_pipe[i-1];
                q_s1[i] <= q_s1[i-1];
                qi_s1[i] <= qi_s1[i-1];
            end

            mi_pipe[0] <= xy_pipe[MUL_LATENCY-1][63:0] * qi_s1[MUL_LATENCY-1];
            xyhi_s2[0] <= xy_pipe[MUL_LATENCY-1][127:64];
            q_s2[0] <= q_s1[MUL_LATENCY-1];
            for (i = 1; i < MUL_LATENCY; i = i + 1) begin
                mi_pipe[i] <= mi_pipe[i-1];
                xyhi_s2[i] <= xyhi_s2[i-1];
                q_s2[i] <= q_s2[i-1];
            end

            mq_pipe[0] <= mi_pipe[MUL_LATENCY-1][63:0] * q_s2[MUL_LATENCY-1];
            xyhi_s3[0] <= xyhi_s2[MUL_LATENCY-1];
            q_s3[0] <= q_s2[MUL_LATENCY-1];
            for (i = 1; i < MUL_LATENCY; i = i + 1) begin
                mq_pipe[i] <= mq_pipe[i-1];
                xyhi_s3[i] <= xyhi_s3[i-1];
                q_s3[i] <= q_s3[i-1];
            end

            reduced = {1'b0, xyhi_s3[MUL_LATENCY-1]} +
                      {1'b0, q_s3[MUL_LATENCY-1]} -
                      {1'b0, mq_pipe[MUL_LATENCY-1][127:64]};
            if (reduced >= {1'b0, q_s3[MUL_LATENCY-1]})
                reduced = reduced - {1'b0, q_s3[MUL_LATENCY-1]};
            r <= reduced[63:0];
        end
    end

    assign out_valid = valid_pipe[LATENCY-1];
endmodule
