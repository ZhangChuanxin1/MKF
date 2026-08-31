# U55C-only HLS entry point.
# Usage: vitis_hls -f run_hls.tcl -tclargs TARGET MODE CLOCK_NS
# TARGET: kklss_ckks | mkf_ckks | kklss_bfv | mkf_bfv
# MODE:   csim | csynth | export

if {$argc != 3} {
    error "usage: run_hls.tcl TARGET MODE CLOCK_NS"
}
if {![info exists ::env(MKF_HLS_BUILD_DIR)]} {
    error "MKF_HLS_BUILD_DIR is required"
}

set target [lindex $argv 0]
set mode [lindex $argv 1]
set clock_ns [lindex $argv 2]
if {$mode ni {csim csynth export}} {
    error "unsupported mode: $mode"
}
if {![string is double -strict $clock_ns] || $clock_ns <= 0.0} {
    error "CLOCK_NS must be a positive number"
}

set script_dir [file normalize [file dirname [info script]]]
set repo_root [file normalize [file join $script_dir .. .. ..]]
set build_root [file normalize $::env(MKF_HLS_BUILD_DIR)]
set kernel_dir [file join $script_dir kernels]
set include_dir [file join $script_dir include]
set tb_dir [file join $script_dir tb]
set fixture_dir [file join $repo_root fixtures generated]

set source ""
set top ""
set testbench ""
set csim_argv ""
set required_fixtures {}

switch -- $target {
    kklss_ckks {
        set source [file join $kernel_dir par1_kklss_ckks_kernel.cpp]
        set top par1_kklss_ckks_kernel
        set testbench [file join $tb_dir tb_par1_kklss_ckks.cpp]
        set f0 [file join $fixture_dir kklss_ckks_p1_n16]
        set f1 [file join $fixture_dir mkf_ckks_p1_n16]
        set csim_argv [format {"%s" "%s"} $f0 $f1]
        set required_fixtures [list $f0 $f1]
    }
    mkf_ckks {
        set source [file join $kernel_dir par1_mkf_ckks_kernel.cpp]
        set top par1_mkf_ckks_kernel
        set testbench [file join $tb_dir tb_par1_mkf_ckks.cpp]
        set f0 [file join $fixture_dir mkf_ckks_p1_n16]
        set csim_argv [format {"%s"} $f0]
        set required_fixtures [list $f0]
    }
    kklss_bfv {
        set source [file join $kernel_dir par1_kklss_bfv_kernel.cpp]
        set top par1_kklss_bfv_kernel
        set testbench [file join $tb_dir tb_par1_kklss_bfv.cpp]
        set f0 [file join $fixture_dir kklss_bfv_p1_n16]
        set f1 [file join $fixture_dir mkf_bfv_p1_n16]
        set csim_argv [format {"%s" "%s"} $f0 $f1]
        set required_fixtures [list $f0 $f1]
    }
    mkf_bfv {
        set source [file join $kernel_dir par1_mkf_bfv_kernel.cpp]
        set top par1_mkf_bfv_kernel
        set testbench [file join $tb_dir tb_par1_mkf_bfv.cpp]
        set f0 [file join $fixture_dir mkf_bfv_p1_n16]
        set csim_argv [format {"%s"} $f0]
        set required_fixtures [list $f0]
    }
    default {
        error "unsupported target: $target"
    }
}

foreach required [list $source $include_dir] {
    if {![file exists $required]} {
        error "missing required path: $required"
    }
}
if {$mode eq "csim"} {
    if {![file exists $testbench]} {
        error "missing testbench: $testbench"
    }
    foreach fixture $required_fixtures {
        if {![file isdirectory $fixture]} {
            error "missing generated fixture directory: $fixture"
        }
    }
}

file mkdir $build_root
set project [file join $build_root ${target}_${mode}_${clock_ns}ns]
set cflags [format {-std=c++14 -I"%s" -I"%s"} $include_dir $kernel_dir]

open_project -reset $project
set_top $top
add_files $source -cflags $cflags
if {$mode eq "csim"} {
    add_files -tb $testbench -cflags $cflags
}
open_solution -reset solution1 -flow_target vivado
set_part {xcu55c-fsvh2892-2L-e}
create_clock -period $clock_ns -name default
config_interface -m_axi_addr64

if {$mode eq "csim"} {
    csim_design -argv $csim_argv
} else {
    csynth_design
    if {$mode eq "export"} {
        config_export -format ip_catalog -rtl verilog -vendor mkf.artifact -library hls -version 1.0
        export_design
    }
}
exit
