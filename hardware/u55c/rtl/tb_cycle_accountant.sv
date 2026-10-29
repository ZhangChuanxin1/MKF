`timescale 1ns/1ps
module tb_cycle_accountant;
logic clk=0,rst_n=0,clear=0,active=0,hbm_request=0,hbm_ready=1;
logic[2:0] cls=5; logic[63:0]d,e,n,m,h,o,t;logic ok;
cycle_accountant dut(.clk,.rst_n,.clear,.operation_active(active),
 .requested_class(cls),.hbm_request,.hbm_ready,.decomposition(d),
 .external_product(e),.ntt_intt(n),.modup_down(m),.hbm_stall(h),
 .other(o),.total(t),.consistent(ok));
always #1 clk=~clk;
task run(input[2:0] c,input integer count);integer i;begin
 @(negedge clk);cls=c;active=1;for(i=0;i<count;i=i+1)@(posedge clk);
 @(negedge clk);active=0;end endtask
initial begin repeat(3)@(posedge clk);rst_n=1;@(posedge clk);
 run(0,3);run(1,5);run(2,7);run(3,11);run(5,13);
 @(negedge clk);cls=1;active=1;hbm_request=1;hbm_ready=0;repeat(17)@(posedge clk);
 @(negedge clk);active=0;hbm_request=0;hbm_ready=1;@(posedge clk);
 if(d!=3||e!=5||n!=7||m!=11||h!=17||o!=13||t!=56||!ok)
  $fatal(1,"COUNTER FAIL d=%0d e=%0d n=%0d m=%0d h=%0d o=%0d t=%0d ok=%0d",d,e,n,m,h,o,t,ok);
 $display("COUNTER PASS d=%0d e=%0d n=%0d m=%0d h=%0d o=%0d total=%0d",d,e,n,m,h,o,t);$finish;
end
endmodule
