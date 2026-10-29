# Initial project structure

Project name: `MKF`. The repository/project root is the `MKF` source tree;
implementation identifiers and Go module names remain unchanged.

This initial source version creates experiment and implementation locations
for CPU, U55C and CNN work. Existing source files remain in their canonical
locations: Go modules, include paths and fixture contracts are unchanged.

```text
software/
  kklss/cnn/                      Existing MNIST baseline and weights
  mkf/
    mkf_cpu/                     Existing primitive/network research code
    cnn/
      mnist/                     Future MKF CNN implementation
      resnet20/                  Index to existing mkf_cpu research paths
hardware/u55c/
  hls/                           Existing P1 multiplication/relinearization
  micro/                         Existing arithmetic microkernels
  rtl/                           Existing RTL components
  host/                          Future XRT host
  connectivity/                  Future validated stream/HBM mapping
  integration/                   Future complete accelerator build
  validation/                    Future board correctness/measurement harness
  cnn/
    mnist/                       Future full CNN kernel/integration
    resnet20/                    Future full network kernel/integration
experiments/
  cpu/
    correctness/                 Correctness index
    latency/                     Historical profiling/comparison index
    stage_breakdown/             Instrumentation and stage-test index
    noise_depth/                 Noise/depth harness index
    kklss_exact_tests/            Existing profiling kit
  fpga/
    primitives/                  HLS primitive experiment index
    resources/                   HLS/OOC resource-flow index
    rotation/                    Software/fixture index; hardware pending
  cnn/
    mnist/
      kklss/                     Baseline experiment entry
      mkf/                       Future matched MKF experiment
    resnet20/
      plaintext/                 Reference and calibration entry
      cpu/                       Partial encrypted-network research entry
      fpga/                      Future production U55C experiment
  flooding/                      Existing conditional-feasibility replay
  security/                      Existing security-estimator inputs
  noise/                         Historical observations/source index
  resnet20/                      Existing plaintext tools, not moved
datasets/
  mnist/                         MNIST requirements
  cifar10/                       CIFAR requirements
  checkpoints/                   Checkpoint/folded-weight requirements
  local/                         User-created local inputs, ignored
figures/                         Existing plot sources/classified inputs
evidence/                        Provenance policy, not generated results
config/experiment_layout.json    Directory/source/status index
```

Each reserved directory has an English README so it survives a Git commit.
No dummy kernel, fabricated input, checkpoint, benchmark CSV or success
receipt is introduced to make a missing component appear complete.

## Status semantics

| Status | Meaning |
|---|---|
| `SOURCE_AVAILABLE` | Source/harness exists; successful execution or publication eligibility is not implied |
| `PARTIAL` | Research source exists, but implementation or evidence gates remain incomplete |
| `REQUIRES_EXTERNAL_ASSETS` | Matching external inputs must be supplied and validated |
| `PENDING_IMPLEMENTATION` | Documentation-only location; no executable implementation is provided |

Canonical source references in the JSON index are relative to the artifact
root. They are not duplicate implementations. Review construction and
validation scope before changing an implementation status.

## Lightweight check

From the artifact root, without Go, FPGA tools or PowerShell 7:

```powershell
py -3 scripts/verify_project_layout.py
py -3 scripts/publication_scan.py --verify-manifest
py -3 -m unittest discover -s tests -v
```

These checks verify the scaffold, not the experiments' cryptographic or
hardware correctness.

For this initial layout revision, all 23 declared components, the publication
scan, source manifest, and 11 packaging/layout unit tests passed. Existing
cryptographic implementation files and figure data were not modified. No
complete encrypted-network or hardware experiment was launched to create
this source layout.
