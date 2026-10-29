# Source-only U55C arithmetic HLS wrapper; no board operation.
# vitis_hls -f run_micro.tcl -tclargs TARGET MODE CLOCK_NS
# TARGET: mred | ntt | ntt_serial; MODE: csynth | csim
# Set MKF_MICRO_BUILD_DIR to a fresh local output directory.
if {$argc != 3} { error "usage: TARGET MODE CLOCK_NS" }
set target [lindex $argv 0]
set mode [lindex $argv 1]
set period [lindex $argv 2]
if {$mode ni {csim csynth}} { error "mode must be csim or csynth" }
if {![string is double -strict $period] || $period <= 0} { error "positive clock period required" }
if {![info exists ::env(MKF_MICRO_BUILD_DIR)]} { error "MKF_MICRO_BUILD_DIR required" }
set here [file normalize [file dirname [info script]]]
set destination [file normalize $::env(MKF_MICRO_BUILD_DIR)]
if {[file exists $destination]} { error "output directory already exists; choose a fresh path" }
set includes [file join $here .. hls include]
set kernels [file join $here .. hls kernels]
switch -- $target {
    mred { set source mred_synth.cpp; set top mred_synth }
    ntt { set source par1_ntt_kernel.cpp; set top par1_ntt_kernel }
    ntt_serial { set source par1_ntt_serial_kernel.cpp; set top par1_ntt_kernel }
    default { error "unsupported arithmetic target" }
}
if {$mode eq "csim" && $target ne "ntt"} {
    error "only ntt has a bundled HLS testbench; do not claim untested csim"
}
if {$mode eq "csim"} {
    if {![info exists ::env(MKF_NTT_FIXTURE_DIR)]} { error "MKF_NTT_FIXTURE_DIR required" }
    foreach name {ntt_input.bin ntt_expected.bin intt_expected.bin ntt_twiddle.bin intt_twiddle.bin ntt_constants.bin} {
        if {![file isfile [file join $::env(MKF_NTT_FIXTURE_DIR) $name]]} { error "missing exact NTT fixture: $name" }
    }
}
set cflags [format {-std=c++14 -I"%s" -I"%s"} $includes $kernels]
open_project $destination
set_top $top
add_files [file join $here $source] -cflags $cflags
if {$mode eq "csim"} { add_files -tb [file join $here tb_par1_ntt.cpp] -cflags $cflags }
open_solution solution1 -flow_target vivado
set_part {xcu55c-fsvh2892-2L-e}
create_clock -period $period -name default
if {$mode eq "csim"} {
    csim_design -argv [format {"%s"} $::env(MKF_NTT_FIXTURE_DIR)]
} else { csynth_design }
exit
