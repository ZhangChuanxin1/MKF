# U55C HLS sources

This directory fixes the hardware target to AMD Alveo U55C, device part
`xcu55c-fsvh2892-2L-e`, with platform
`xilinx_u55c_gen3x16_xdma_3_202210_1` and Vitis/Vivado 2022.2.

The four HLS tops implement P1, ring degree 16384 and 16-party primitive
multiplication with relinearization for KKLSS/MKF and MK-CKKS/MK-BFV.

Run them through the repository wrapper:

```powershell
pwsh -NoProfile -File scripts/run_u55c_hls.ps1 `
  -Target mkf_ckks -Mode csim -ClockProfile smoke
```

`run_hls.tcl` resolves kernels, headers, testbenches and generated fixtures
only from this repository. The U55C device part is fixed in the Tcl script and
cannot be overridden from the command line.

This is not a board-complete design. The current top-level interfaces still
need U55C XRT integration, connectivity/HBM mapping, a `phase_events`
consumer, link and route, timing closure and board correctness validation.

Separate arithmetic microkernel and RTL sources are indexed in
[micro/README.md](micro/README.md). They are source/OOC research paths, not
additional completed U55C measurements.
