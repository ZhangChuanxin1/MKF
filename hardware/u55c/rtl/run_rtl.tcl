# U55C out-of-context arithmetic resource/timing flow, not a board build.
# vivado -mode batch -source run_rtl.tcl -tclargs TOP MODE CLOCK_NS
# TOP: mred64_pipeline | ntt_butterfly_x8 | cycle_accountant | tifs_stage_counter
# MODE: synth | route. MKF_RTL_BUILD_DIR must name a fresh output directory.
if {$argc != 3} { error "usage: TOP synth|route CLOCK_NS" }
set top [lindex $argv 0]
set mode [lindex $argv 1]
set period [lindex $argv 2]
if {$top ni {mred64_pipeline ntt_butterfly_x8 cycle_accountant tifs_stage_counter}} { error "unsupported top" }
if {$mode ni {synth route}} { error "unsupported mode" }
if {![string is double -strict $period] || $period <= 0} { error "positive clock period required" }
if {![info exists ::env(MKF_RTL_BUILD_DIR)]} { error "MKF_RTL_BUILD_DIR required" }
set here [file normalize [file dirname [info script]]]
set destination [file normalize $::env(MKF_RTL_BUILD_DIR)]
if {[file exists $destination]} { error "output directory already exists; choose a fresh path" }
file mkdir $destination
cd $destination
switch -- $top {
    ntt_butterfly_x8 { read_verilog -sv [list [file join $here mred64_pipeline.sv] [file join $here ntt_butterfly_x8.sv]] }
    tifs_stage_counter { read_verilog [file join $here tifs_stage_counter.v] }
    default { read_verilog -sv [file join $here ${top}.sv] }
}
set module_top $top
set clock_port clk
if {$top eq "tifs_stage_counter"} { set module_top tifs_stage_counter_axis; set clock_port ap_clk }
synth_design -top $module_top -part xcu55c-fsvh2892-2L-e -mode out_of_context
create_clock -name clk -period $period [get_ports $clock_port]
report_utilization -file synthesis_utilization.rpt
if {$mode eq "route"} {
    opt_design
    place_design
    route_design
    report_timing_summary -delay_type min_max -file routed_timing.rpt
    report_utilization -file routed_utilization.rpt
    write_checkpoint routed.dcp
}
exit
