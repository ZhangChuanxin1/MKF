# Arithmetic microkernels and RTL

These are device-neutral arithmetic sources with newly isolated U55C-only
wrappers. They were not synthesized/simulated by the packaging task. Only
include paths were adjusted for the new source layout. A successful tool
launch alone is not a correctness/performance PASS.

## HLS estimates

Activate Vitis HLS 2022.2 and explicitly choose a fresh `.local/` output:

```powershell
$env:MKF_MICRO_BUILD_DIR = [IO.Path]::GetFullPath('.local/mred-u55c-first')
vitis_hls -f hardware/u55c/micro/run_micro.tcl -tclargs mred csynth 10.0
```

Other targets: `ntt`, `ntt_serial`. C synthesis is resource-heavy: run one
target at a time. The wrapper fixes `xcu55c-fsvh2892-2L-e`.

Only `ntt` has a bundled HLS C-sim bench. It requires six exact fixture files:
`ntt_input.bin`, `ntt_expected.bin`, `intt_expected.bin`, `ntt_twiddle.bin`,
`intt_twiddle.bin`, `ntt_constants.bin`. These are not interchangeable with
the four multiplication-fixture directories and no generator for these six
files is supplied here. C-sim is blocked until the exact fixture contract is
provided; set `MKF_NTT_FIXTURE_DIR` explicitly. The other two HLS targets are
source/csynth-only, not claimed bit-exact validated blocks.

## RTL out-of-context synthesis/route

```powershell
$env:MKF_RTL_BUILD_DIR = [IO.Path]::GetFullPath('.local/mred-rtl-first')
vivado -mode batch -source hardware/u55c/rtl/run_rtl.tcl `
  -tclargs mred64_pipeline synth 10.0
```

Use `route` instead of `synth` only when ready for the heavier OOC flow.
Other tops: `ntt_butterfly_x8`, `cycle_accountant`, `tifs_stage_counter`.
RTL testbench sources are included for arithmetic/counter inspection; the
wrapper does not automatically run a simulator. A routed OOC report is not
full-device timing closure, an XRT design or a board measurement. Never scale
these estimates to fill missing U55C latency/throughput/resource table cells.
