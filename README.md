# MKFHE FPGA: U55C Reproducibility Artifact

This repository contains software experiments and U55C-targeted HLS sources
for KKLSS and MKF multiplication with relinearization in MK-CKKS and MK-BFV.

## Fixed scope

- FPGA card: AMD Alveo U55C
- Device part: `xcu55c-fsvh2892-2L-e`
- Platform: `xilinx_u55c_gen3x16_xdma_3_202210_1`
- Toolchain target: Vitis/Vivado 2022.2
- Primitive parameter profile: P1, ring degree 16384, 16 parties
- HLS targets: `kklss_ckks`, `mkf_ckks`, `kklss_bfv`, and `mkf_bfv`

The hardware release is **HLS-only**. It supports source validation, C
simulation, C synthesis, and optional HLS IP export. It does not include an
XRT host, HBM connectivity plan, linked accelerator image, routed timing
result, or board-measured latency. See [docs/RELEASE_BOUNDARY.md](docs/RELEASE_BOUNDARY.md).

## Quick start

Requirements for the local software checks:

- PowerShell 7
- Go 1.23 or a compatible Go release
- Python 3.10 or newer

Run the publication and software smoke checks:

```powershell
pwsh -NoProfile -File scripts/preflight.ps1
pwsh -NoProfile -File scripts/run_suite.ps1 -Profile smoke
```

Run the four software correctness tests:

```powershell
pwsh -NoProfile -File scripts/run_suite.ps1 -Profile software
```

The flooding replay additionally requires NumPy:

```powershell
py -3 -m pip install -r requirements-software.txt
```

## U55C HLS

Activate Vitis/Vivado 2022.2 and make the exact U55C platform visible to
`platforminfo`. Generate the bit-exact fixtures locally; generated payloads
are intentionally ignored by Git because they are large.

```powershell
pwsh -NoProfile -File scripts/generate_fixtures.ps1 -AllowLargeOutput
pwsh -NoProfile -File scripts/preflight.ps1 -RequireU55C
pwsh -NoProfile -File scripts/run_suite.ps1 -Profile u55c-csim
```

HLS synthesis is explicit because it is resource intensive:

```powershell
pwsh -NoProfile -File scripts/run_suite.ps1 `
  -Profile u55c-csynth `
  -ClockProfile smoke `
  -AllowHeavySynthesis
```

The `release300` profile requests a 3.333 ns HLS constraint. It is not an
achieved 300 MHz result. An achieved clock requires implementation and
non-negative routed WNS/WHS on the exact U55C design.

## Repository layout

```text
config/                 fixed U55C identity and release status
docs/                   experiment map and evidence boundaries
experiments/            CPU profiling, flooding, ResNet-20 and security tools
fixtures/               documentation; generated fixtures are untracked
hardware/u55c/hls/      four U55C-targeted HLS kernels and testbenches
scripts/                preflight, fixture generation and run orchestration
software/kklss/         KKLSS reference implementation and its license
software/mkf/           MKF implementation and tests
```

## Evidence status

| Layer | Status in this repository | Meaning |
|---|---|---|
| Publication scan | Runnable locally | Rejects personal paths, unsupported device identities, secrets and generated artifacts |
| Software compile/smoke | Runnable locally | Compilation and lightweight contracts only |
| CPU correctness | Runnable locally | Same-parameter cryptographic correctness tests |
| U55C HLS C simulation | User-run | Requires locally generated bit-exact fixtures and Vitis HLS |
| U55C HLS synthesis | User-run | Produces HLS estimates, not board measurements |
| U55C link and route | Not included | No achieved-frequency or routed-resource claim |
| U55C board measurement | Not included | No latency, throughput, power or HBM claim |

No U55C performance table or board-measured value is embedded in this
repository. The CPU profiling patch kit contains a clearly labeled local
sample precheck that is not a paper measurement. A result is reportable only
when its source revision, configuration, fixture hashes, toolchain,
measurement boundary, and raw output are all retained together.

## Licensing

The bundled KKLSS implementation retains its upstream CC BY-NC 2.0 license at
`software/kklss/LICENSE`; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
The repository owner must select and add a license for the original MKF and
hardware sources before public redistribution. No license is implied merely
by publishing the files.

## Uploading

Follow [UPLOAD_GUIDE_zh.md](UPLOAD_GUIDE_zh.md). Use a fresh, neutral repository
for double-blind review; a clean branch does not erase sensitive objects from
older Git history.
