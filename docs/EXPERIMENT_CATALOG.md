# Complete experiment-category index

All 15 categories identified in the local suite are indexed here. Included
source does not imply matching or completed measurements.

| ID | Experiment | Code / entry | Boundary |
|---|---|---|---|
| 00 | Parameters/environment | `config/u55c.json`, module `params.go`, preflight | Construction-specific parameter/security review still required |
| 01 | Correctness/noise/depth | Go noise tests; `run_experiment.ps1 -Experiment noise` | Actual opt-in tests; historical CSVs not freshly revalidated |
| 02 | CPU multiplication + relinearization scaling | MKF `cmd/mkfbench`; KKLSS scheme benchmark tests | Historical paths available; no newly matched full matrix |
| 03 | FPGA scaling | `figures/mkf`, `figures/kklss`; primitive HLS | Plot inputs are not new board results |
| 04 | Stage attribution/cycles | Both `stageprof` trees; stage tests; `mkf_cycle_counters.hpp`; RTL counters | CPU timers, semantic counters and physical cycles are different metrics |
| 05 | Gadget decomposition | CPU stage tests; HLS decomposition code | No independently qualified U55C microbenchmark result |
| 06 | Resource/HBM/frequency | HLS configuration; U55C RTL OOC wrapper | Full XRT/HBM/route/board evidence absent |
| 07 | Rotation | `rotation_fixture_test.go`; `true_mkf_ckks_test.go` | Source/exporter included, production U55C latency absent |
| 08 | Flooding | `experiments/flooding` | Conditional arithmetic replay, not a proof-grade bound |
| 09 | Concrete lattice security | `experiments/security` | Exact Sage/estimator required; no RLWE/KDM/protocol proof implied |
| 10 | MNIST CNN | `software/kklss/cnn`; IDX converter | KKLSS source/weights included, dataset local; no complete matched MKF CNN |
| 11 | ResNet-20 functional reference | Plaintext tools; MKF `resnet20_*`, `true_mkf_*` | Data/checkpoint/folded weights separate; full production exports excluded |
| 12 | ResNet-20 matched CPU/FPGA | `p2gn16_matched_cpu_benchmark_test.go`, module's benchmark launcher | BLOCKED pending production exports, assets and independent seals; do not time a fallback |
| 13 | Analytical complexity | Paper-facing MKF and KKLSS source audit | Operation counts are not latency |
| 14 | Modular reduction/NTT | `hardware/u55c/micro`, `rtl` | Arithmetic sources and U55C wrappers included; not synthesized during packaging |

Evidence labels: `source`, `local-check`, `historical-data`,
`author-figure-data`, `derived`, and `blocked` are distinct.
Retain `claim -> metric -> baseline -> controls -> boundary -> raw result`.
Same-CPU algorithmic, same-U55C co-design and cross-platform combined gains
are separate comparisons; never substitute one for another.
