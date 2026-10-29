`timescale 1ns/1ps

module tifs_stage_counter_axis(
    (* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME s_axi, PROTOCOL AXI4LITE, DATA_WIDTH 32, ADDR_WIDTH 8" *)
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi AWADDR" *) input  wire [7:0] s_axi_awaddr,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi AWVALID" *) input wire s_axi_awvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi AWREADY" *) output reg s_axi_awready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi WDATA" *) input wire [31:0] s_axi_wdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi WSTRB" *) input wire [3:0] s_axi_wstrb,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi WVALID" *) input wire s_axi_wvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi WREADY" *) output reg s_axi_wready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi BRESP" *) output wire [1:0] s_axi_bresp,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi BVALID" *) output reg s_axi_bvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi BREADY" *) input wire s_axi_bready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi ARADDR" *) input wire [7:0] s_axi_araddr,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi ARVALID" *) input wire s_axi_arvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi ARREADY" *) output reg s_axi_arready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi RDATA" *) output reg [31:0] s_axi_rdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi RRESP" *) output wire [1:0] s_axi_rresp,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi RVALID" *) output reg s_axi_rvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi RREADY" *) input wire s_axi_rready,
    (* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 ap_clk CLK" *)
    (* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME ap_clk, ASSOCIATED_BUSIF s_axi:phase0:phase1:phase2:phase3, ASSOCIATED_RESET ap_rst_n" *) input wire ap_clk,
    (* X_INTERFACE_INFO = "xilinx.com:signal:reset:1.0 ap_rst_n RST" *)
    (* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME ap_rst_n, POLARITY ACTIVE_LOW" *) input wire ap_rst_n,

    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 phase0 TDATA" *) input wire [7:0] phase0_tdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 phase0 TVALID" *) input wire phase0_tvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 phase0 TREADY" *) output wire phase0_tready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 phase1 TDATA" *) input wire [7:0] phase1_tdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 phase1 TVALID" *) input wire phase1_tvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 phase1 TREADY" *) output wire phase1_tready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 phase2 TDATA" *) input wire [7:0] phase2_tdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 phase2 TVALID" *) input wire phase2_tvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 phase2 TREADY" *) output wire phase2_tready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 phase3 TDATA" *) input wire [7:0] phase3_tdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 phase3 TVALID" *) input wire phase3_tvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 phase3 TREADY" *) output wire phase3_tready,
    input wire k0_arvalid,input wire k0_arready,input wire k0_rvalid,input wire k0_rready,input wire k0_rlast,
    input wire k0_awvalid,input wire k0_awready,input wire k0_wvalid,input wire k0_wready,input wire k0_wlast,input wire k0_bvalid,input wire k0_bready,
    input wire k1_arvalid,input wire k1_arready,input wire k1_rvalid,input wire k1_rready,input wire k1_rlast,
    input wire k1_awvalid,input wire k1_awready,input wire k1_wvalid,input wire k1_wready,input wire k1_wlast,input wire k1_bvalid,input wire k1_bready,
    input wire k2_arvalid,input wire k2_arready,input wire k2_rvalid,input wire k2_rready,input wire k2_rlast,
    input wire k2_awvalid,input wire k2_awready,input wire k2_wvalid,input wire k2_wready,input wire k2_wlast,input wire k2_bvalid,input wire k2_bready,
    input wire k3_arvalid,input wire k3_arready,input wire k3_rvalid,input wire k3_rready,input wire k3_rlast,
    input wire k3_awvalid,input wire k3_awready,input wire k3_wvalid,input wire k3_wready,input wire k3_wlast,input wire k3_bvalid,input wire k3_bready
);
assign s_axi_bresp=2'b00; assign s_axi_rresp=2'b00;
reg armed,done,seen_active; reg [1:0] selected;
reg [2:0] current_stage,current_timing;
reg [15:0] read_outstanding,write_response_pending;
reg [63:0] decomp_cycles,external_cycles,ntt_cycles,mod_cycles,stall_cycles,other_cycles,total_cycles;
reg [63:0] timing_tensor,timing_decomp,timing_external,timing_mod,timing_other;
wire [7:0] phase_data = selected==0?phase0_tdata:selected==1?phase1_tdata:selected==2?phase2_tdata:phase3_tdata;
wire phase_valid = selected==0?phase0_tvalid:selected==1?phase1_tvalid:selected==2?phase2_tvalid:phase3_tvalid;
wire phase_ready = armed;
wire phase_fire = phase_valid && phase_ready;
wire [2:0] sampled_stage = phase_fire ? phase_data[5:3] : current_stage;
wire [2:0] sampled_timing = phase_fire ? phase_data[2:0] : current_timing;
assign phase0_tready = armed && selected==0;
assign phase1_tready = armed && selected==1;
assign phase2_tready = armed && selected==2;
assign phase3_tready = armed && selected==3;
wire arvalid=selected==0?k0_arvalid:selected==1?k1_arvalid:selected==2?k2_arvalid:k3_arvalid;
wire arready=selected==0?k0_arready:selected==1?k1_arready:selected==2?k2_arready:k3_arready;
wire rvalid=selected==0?k0_rvalid:selected==1?k1_rvalid:selected==2?k2_rvalid:k3_rvalid;
wire rready=selected==0?k0_rready:selected==1?k1_rready:selected==2?k2_rready:k3_rready;
wire rlast=selected==0?k0_rlast:selected==1?k1_rlast:selected==2?k2_rlast:k3_rlast;
wire awvalid=selected==0?k0_awvalid:selected==1?k1_awvalid:selected==2?k2_awvalid:k3_awvalid;
wire awready=selected==0?k0_awready:selected==1?k1_awready:selected==2?k2_awready:k3_awready;
wire wvalid=selected==0?k0_wvalid:selected==1?k1_wvalid:selected==2?k2_wvalid:k3_wvalid;
wire wready=selected==0?k0_wready:selected==1?k1_wready:selected==2?k2_wready:k3_wready;
wire wlast=selected==0?k0_wlast:selected==1?k1_wlast:selected==2?k2_wlast:k3_wlast;
wire bvalid=selected==0?k0_bvalid:selected==1?k1_bvalid:selected==2?k2_bvalid:k3_bvalid;
wire bready=selected==0?k0_bready:selected==1?k1_bready:selected==2?k2_bready:k3_bready;
wire memory_stall=(arvalid&&!arready)||(awvalid&&!awready)||(wvalid&&!wready)||(rvalid&&!rready)||(bvalid&&!bready)||
                  ((read_outstanding!=0)&&!rvalid)||((write_response_pending!=0)&&!bvalid);

task clear_counts; begin decomp_cycles<=0;external_cycles<=0;ntt_cycles<=0;mod_cycles<=0;stall_cycles<=0;other_cycles<=0;total_cycles<=0;timing_tensor<=0;timing_decomp<=0;timing_external<=0;timing_mod<=0;timing_other<=0;read_outstanding<=0;write_response_pending<=0;current_stage<=0;current_timing<=0;seen_active<=0;done<=0; end endtask
always @(posedge ap_clk) begin
 if(!ap_rst_n) begin s_axi_awready<=0;s_axi_wready<=0;s_axi_bvalid<=0;s_axi_arready<=0;s_axi_rvalid<=0;s_axi_rdata<=0;armed<=0;selected<=0;clear_counts;end
 else begin
  s_axi_awready<=0;s_axi_wready<=0;s_axi_arready<=0;
  if(s_axi_awvalid&&s_axi_wvalid&&!s_axi_bvalid) begin s_axi_awready<=1;s_axi_wready<=1;s_axi_bvalid<=1;
   if(s_axi_awaddr[7:0]==8'h00&&s_axi_wstrb[0]) begin
    if(s_axi_wdata[0]) begin armed<=0;clear_counts;end
    if(s_axi_wdata[1]) begin selected<=s_axi_wdata[9:8];armed<=1;clear_counts;end
   end
  end
  if(s_axi_bvalid&&s_axi_bready)s_axi_bvalid<=0;
  if(s_axi_arvalid&&!s_axi_rvalid) begin s_axi_arready<=1;s_axi_rvalid<=1;case(s_axi_araddr[7:0])
   8'h00:s_axi_rdata<={21'd0,selected,7'd0,done,armed}; 8'h04:s_axi_rdata<={29'd0,current_stage};
   8'h10:s_axi_rdata<=decomp_cycles[31:0];8'h14:s_axi_rdata<=decomp_cycles[63:32];
   8'h18:s_axi_rdata<=external_cycles[31:0];8'h1c:s_axi_rdata<=external_cycles[63:32];
   8'h20:s_axi_rdata<=ntt_cycles[31:0];8'h24:s_axi_rdata<=ntt_cycles[63:32];
   8'h28:s_axi_rdata<=mod_cycles[31:0];8'h2c:s_axi_rdata<=mod_cycles[63:32];
   8'h30:s_axi_rdata<=stall_cycles[31:0];8'h34:s_axi_rdata<=stall_cycles[63:32];
   8'h38:s_axi_rdata<=other_cycles[31:0];8'h3c:s_axi_rdata<=other_cycles[63:32];
   8'h40:s_axi_rdata<=total_cycles[31:0];8'h44:s_axi_rdata<=total_cycles[63:32];
   8'h48:s_axi_rdata<=timing_tensor[31:0];8'h4c:s_axi_rdata<=timing_tensor[63:32];
   8'h50:s_axi_rdata<=timing_decomp[31:0];8'h54:s_axi_rdata<=timing_decomp[63:32];
   8'h58:s_axi_rdata<=timing_external[31:0];8'h5c:s_axi_rdata<=timing_external[63:32];
   8'h60:s_axi_rdata<=timing_mod[31:0];8'h64:s_axi_rdata<=timing_mod[63:32];
   8'h68:s_axi_rdata<=timing_other[31:0];8'h6c:s_axi_rdata<=timing_other[63:32];default:s_axi_rdata<=32'h0;endcase
  end
  if(s_axi_rvalid&&s_axi_rready)s_axi_rvalid<=0;
  if(armed) begin
   case({arvalid&&arready,rvalid&&rready&&rlast}) 2'b10:read_outstanding<=read_outstanding+1;2'b01:read_outstanding<=read_outstanding-1;default:;endcase
   case({wvalid&&wready&&wlast,bvalid&&bready}) 2'b10:write_response_pending<=write_response_pending+1;2'b01:write_response_pending<=write_response_pending-1;default:;endcase
   if(phase_fire) begin
    current_stage<=phase_data[5:3]; current_timing<=phase_data[2:0];
    if(phase_data[5:3]!=3'd0) seen_active<=1;
    else if(seen_active) begin armed<=0;done<=1;end
   end
   if((seen_active || (phase_fire && phase_data[5:3]!=3'd0)) && sampled_stage!=3'd0) begin total_cycles<=total_cycles+1;
    if(memory_stall||sampled_stage==3'd5)stall_cycles<=stall_cycles+1;
    else begin
     case(sampled_stage)3'd1:decomp_cycles<=decomp_cycles+1;3'd2:external_cycles<=external_cycles+1;3'd3:ntt_cycles<=ntt_cycles+1;3'd4:mod_cycles<=mod_cycles+1;default:other_cycles<=other_cycles+1;endcase
     case(sampled_timing)3'd1:timing_tensor<=timing_tensor+1;3'd2:timing_decomp<=timing_decomp+1;3'd3:timing_external<=timing_external+1;3'd4:timing_mod<=timing_mod+1;default:timing_other<=timing_other+1;endcase
    end
   end
  end
 end
end
endmodule
