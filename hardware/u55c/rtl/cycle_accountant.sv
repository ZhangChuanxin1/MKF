`timescale 1ns/1ps
module cycle_accountant (
    input logic clk, input logic rst_n, input logic clear,
    input logic operation_active,
    input logic [2:0] requested_class,
    input logic hbm_request, input logic hbm_ready,
    output logic [63:0] decomposition, external_product, ntt_intt,
    output logic [63:0] modup_down, hbm_stall, other, total,
    output logic consistent
);
    localparam [2:0] DECOMP=0, EXT=1, NTT=2, MOD=3, HBM=4, OTHER=5;
    logic [2:0] charged_class;
    always @* begin
        if (hbm_request && !hbm_ready) charged_class = HBM;
        else charged_class = requested_class;
        consistent = total == decomposition + external_product + ntt_intt +
                              modup_down + hbm_stall + other;
    end
    always @(posedge clk) begin
        if (!rst_n || clear) begin
            decomposition<=0; external_product<=0; ntt_intt<=0;
            modup_down<=0; hbm_stall<=0; other<=0; total<=0;
        end else if (operation_active) begin
            total <= total + 1;
            case (charged_class)
              DECOMP: decomposition <= decomposition + 1;
              EXT: external_product <= external_product + 1;
              NTT: ntt_intt <= ntt_intt + 1;
              MOD: modup_down <= modup_down + 1;
              HBM: hbm_stall <= hbm_stall + 1;
              default: other <= other + 1;
            endcase
        end
    end
endmodule
