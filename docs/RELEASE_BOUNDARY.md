# Release and claim boundary

This artifact deliberately separates source availability from measured
hardware evidence.

| Claim | Metric | Baseline | Controlled variables | Measurement boundary | Current status |
|---|---|---|---|---|---|
| MKF software correctness | Decrypt-and-compare result | Scheme reference | P1, ring degree, parties, inputs | Go unit test | Runnable |
| KKLSS software correctness | Decrypt-and-compare result | Scheme reference | P1, ring degree, parties, inputs | Go unit test | Runnable |
| Same-CPU algorithmic comparison | Wall-clock latency | KKLSS CPU-ST | Parameters, host, one logical CPU, trial schedule | Timed Go operation | Script source included; no result released here |
| Same-U55C primitive comparison | Kernel latency | KKLSS on the same U55C design | Part, toolchain, clock, interface, fixtures | Board kernel interval | Not measured in this release |
| HLS feasibility | HLS latency/resource estimate | Target-specific HLS solution | Exact part and requested clock | C synthesis report | User-run; not board evidence |
| End-to-end ResNet-20 acceleration | End-to-end latency and correctness | Matched CPU/network baseline | Full operand/runtime/HBM contract | Complete network invocation | Not implemented for U55C here |

## Included hardware boundary

The included hardware is limited to four P1 primitive HLS tops:

- `par1_kklss_ckks_kernel`
- `par1_mkf_ckks_kernel`
- `par1_kklss_bfv_kernel`
- `par1_mkf_bfv_kernel`

Each current top exposes one `gmem0` AXI master and a `phase_events` AXI
stream. A board design still needs a physical event consumer, XRT host,
connectivity and HBM bank mapping, co-simulation/emulation, link and route,
routed timing closure, and board-level correctness checks.

## Prohibited interpretation

- C simulation does not establish FPGA performance.
- C synthesis estimates do not establish routed frequency or board latency.
- A requested 3.333 ns clock does not establish an achieved 300 MHz clock.
- CPU stage allocation or projected values must not be relabeled as FPGA
  measurements.
- The primitive HLS tops do not establish a complete ResNet-20 accelerator.
