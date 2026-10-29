# MKF

Research code for *MKF: A Software-Hardware Co-design for Efficient Multi-Key
Fully Homomorphic Encryption*.

## Overview

This repository contains research code associated with the MKF manuscript.
The experimental scope covers multiplication with relinearization in
multi-key CKKS (MK-CKKS) and multi-key BFV (MK-BFV), with KKLSS as the
comparison implementation. The repository also includes noise and depth
experiments, stage-profiling tools, uniform-flooding simulations,
security-estimation scripts, encrypted-inference research code, and figure
reproduction scripts.

The hardware sources target the AMD Alveo U55C. The current release provides
primitive HLS kernels, arithmetic microkernels, and RTL components; it does
not provide a complete board-level accelerator or independently reproduce
every performance result discussed in the manuscript. The scope of each
experiment is documented in the [experiment catalog](docs/EXPERIMENT_CATALOG.md).

## Repository organization

This first source version separates CPU, FPGA, MNIST and ResNet-20 experiment
entries while retaining existing implementations in their original module
locations. See the [project structure](docs/PROJECT_STRUCTURE.md) and the
machine-readable `config/experiment_layout.json` index. Reserved components
are explicitly labeled `PENDING_IMPLEMENTATION`; directory presence does not
imply that a complete accelerator or CNN implementation is provided.

| Experiment group | Entry |
|---|---|
| CPU correctness, latency, stages and noise/depth | [experiments/cpu](experiments/cpu/README.md) |
| U55C primitives, resources and rotation | [experiments/fpga](experiments/fpga/README.md) |
| MNIST comparison | [experiments/cnn/mnist](experiments/cnn/mnist/README.md) |
| ResNet-20/CIFAR-10 | [experiments/cnn/resnet20](experiments/cnn/resnet20/README.md) |

```text
software/
  kklss/               KKLSS baseline and profiling instrumentation
  mkf/                 MKF implementations, tests, and research backends
hardware/u55c/
  hls/                 Multiplication/relinearization kernels and testbenches
  micro/               Modular-reduction and NTT HLS sources
  rtl/                 Arithmetic and stage-counter RTL
  host/                Reserved XRT host implementation
  connectivity/        Reserved validated HBM/stream mapping
  integration/         Reserved complete accelerator build
  validation/          Reserved board-level validation
  cnn/                 Reserved MNIST and ResNet-20 hardware integration
experiments/           CPU profiling, noise/depth, flooding, and security tools
figures/               Plotting scripts and classified figure inputs
datasets/              Instructions for preparing external input data
config/                U55C target configuration
scripts/               Validation and experiment entry points
tests/                 Lightweight artifact tests
docs/                  Reproduction procedures and evidence boundaries
```

## Requirements

The local compile and contract checks were validated with Go 1.23.4 and
Python 3.13.0 on Windows/amd64. The orchestration scripts require PowerShell
7. Python utilities require Python 3.10 or later. Go module versions and
checksums are retained in each module's `go.mod` and `go.sum`; the
cryptographic library dependency is Lattigo v2.3.0.

Optional Python dependencies are separated by workload:

| File | Workload |
|---|---|
| `requirements-software.txt` | Uniform-flooding replay |
| `requirements-figures.txt` | Figure reproduction with NumPy and Matplotlib |
| `requirements-resnet.txt` | Plaintext ResNet-20 reference with PyTorch |

U55C HLS experiments additionally require Vitis/Vivado 2022.2. Concrete
security estimation requires the separate SageMath and estimator environment
specified in the [runbook](docs/RUNBOOK.md).

## Software validation

Run the following commands from this directory in PowerShell. The `py -3`
syntax uses the Windows Python launcher.

```powershell
py -3 scripts/publication_scan.py --verify-manifest
py -3 scripts/verify_project_layout.py
py -3 -m unittest discover -s tests -v
pwsh -NoProfile -File scripts/run_suite.ps1 -Profile smoke
```

The smoke profile compiles both Go modules and executes lightweight level
and benchmark-contract checks. It does not run a complete encrypted network,
hardware synthesis, or a performance evaluation. Selected software
correctness tests can be invoked separately:

```powershell
pwsh -NoProfile -File scripts/run_suite.ps1 -Profile software
```

For an isolated Python environment:

```powershell
py -3 -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements-software.txt
pwsh -NoProfile -File scripts/run_suite.ps1 -Profile software -PythonExe .venv/Scripts/python.exe
```

Install the figure or ResNet dependency file only when needed. On systems
without the Windows launcher, invoke Python utilities with `python3` and
pass the corresponding executable through `-PythonExe`.

## Experimental procedures

The [runbook](docs/RUNBOOK.md) gives commands, input requirements, and
measurement boundaries. The following table summarizes the available paths.

| Experiment | Implementation or entry point | Reproduction scope |
|---|---|---|
| Software correctness | MKF and KKLSS Go tests | Selected cryptographic correctness tests |
| CPU multiplication/relinearization | MKF profiler; KKLSS benchmark and stage tests | Explicitly selected profiling configurations |
| Noise and multiplicative depth | Scheme-specific Go noise tests | Observed precision, noise budget, and correctness |
| Uniform flooding | `experiments/flooding` | Conditional arithmetic-feasibility replay |
| Concrete security estimation | `experiments/security` | Candidate-specific estimator inputs and scripts |
| MNIST encrypted CNN | `software/kklss/cnn` | Upstream KKLSS path; local dataset required |
| ResNet-20/CIFAR-10 | Plaintext reference tools and MKF Go backends | Reference, contract, and opt-in functional research paths |
| U55C primitives | `scripts/run_u55c_hls.ps1` | HLS C simulation and synthesis entry points |
| Arithmetic microbenchmarks | `hardware/u55c/micro` and `rtl` | Source-level and out-of-context implementation flows |
| Manuscript figures | `figures/mkf` and `figures/kklss` | Reproduction of the supplied figure inputs |

Crypto profiling, noise experiments, and CNN execution require explicit
`-AllowHeavy` opt-in. HLS synthesis requires `-AllowHeavySynthesis`. Execute
resource-intensive experiments serially. New outputs are placed in ignored
`.local/` directories rather than overwriting retained evidence.

### Implementation identities

The historical `cmd/mkfbench` profiler uses an aggregate-output path. The
paper-facing `true_mkf_*` CKKS code retains the vector-secret ciphertext
representation and repeated-Gamma materialization. Their outputs and timing
boundaries are not interchangeable. The
[construction-identity document](docs/CONSTRUCTION_IDENTITIES.md) identifies
these paths and the production functionality excluded from this release.

The included inference code does not establish a complete matched MKF/KKLSS
MNIST benchmark or a qualified end-to-end ResNet-20 CPU/U55C comparison.
Those results require their own construction, asset, correctness, and
measurement records.

## U55C hardware configuration

| Item | Configuration |
|---|---|
| Accelerator card | AMD Alveo U55C |
| FPGA part | `xcu55c-fsvh2892-2L-e` |
| Platform identity | `xilinx_u55c_gen3x16_xdma_3_202210_1` |
| Toolchain target | Vitis/Vivado 2022.2 |
| Primitive parameter scope | P1, polynomial degree 16384, 16 parties |
| HLS targets | KKLSS/MKF for MK-CKKS and MK-BFV |
| Requested clock periods | `smoke`: 10 ns; `release300`: 3.333 ns |

Generate the primitive fixtures and run the HLS checks only in a correctly
configured toolchain environment:

```powershell
pwsh -NoProfile -File scripts/generate_fixtures.ps1 -AllowLargeOutput
pwsh -NoProfile -File scripts/run_suite.ps1 -Profile u55c-csim
pwsh -NoProfile -File scripts/run_suite.ps1 -Profile u55c-csynth -ClockProfile smoke -AllowHeavySynthesis
```

A requested clock constraint is not an achieved clock frequency. HLS
estimates and out-of-context RTL reports are not board measurements. The
current sources still require an XRT host, validated connectivity and HBM
mapping, a physical consumer for `phase_events`, full-design link and route,
timing closure, and board-level correctness validation. No script
automatically programs, resets, or reboots the card. See the
[hardware documentation](hardware/u55c/README.md) and
[release boundary](docs/RELEASE_BOUNDARY.md).

## Data provenance and result interpretation

External datasets, checkpoints, folded weights, and generated key/ciphertext
payloads are prepared separately according to the
[dataset instructions](datasets/README.md). Third-party license and
attribution requirements remain applicable.

Historical noise CSVs are retained as existing observations, not as a newly
validated, uniform experiment matrix. The plotting inputs are also separate
from measurement evidence: the MKF CPU arrays are derived, the MKF FPGA
arrays are author-supplied projections, and the KKLSS inputs include
author-revised figure values. These files support visual reproduction,
not independent verification of hardware performance. Their classifications
are recorded in the [figure provenance](figures/README.md).

Same-CPU algorithmic improvement, same-U55C co-design improvement, and
cross-platform system acceleration require distinct comparisons. Reportable
results must bind the implementation revision, parameters, inputs,
correctness checks, platform, trial policy, and timing boundary to the raw
observations. Proportional stage allocations and projected values cannot
replace stage-counter or board measurements.

The [local validation record](docs/VALIDATION_20261028.md) lists the checks
completed for this source release and the experiments that were not run.

## License and publication

The bundled KKLSS implementation retains its upstream CC BY-NC 2.0 license.
The included plaintext ResNet model retains its upstream notices. See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and the accompanying license
files. A license for the original MKF and hardware sources has not yet been
specified; publication alone does not grant such a license.

Instructions for publishing a reviewed copy are provided in the
[GitHub upload guide](docs/UPLOAD.md). Sensitive data, generated payloads,
tool outputs, and local execution logs should not be committed.
