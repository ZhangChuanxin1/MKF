# Construction identities

| Path | Role | Do not infer |
|---|---|---|
| `software/kklss/{mkckks,mkbfv,mkrlwe}` | KKLSS baseline | Every historical number came from this exact revision |
| MKF `ckks.go`, `bfv.go`, `common.go`, `cmd/mkfbench` | Historical aggregate-domain profiler | Compact aggregate output proves/times the full vector-output paper construction |
| MKF `true_mkf_ckks*.go` | Paper-facing vector-secret/repeated-Gamma CKKS path | A complete matched BFV/CPU/FPGA matrix exists |
| `guarded_compact_*`, `p2gn16_*`, `resnet20_*` | Guarded-network research/contracts | Functional, security, key-export and observer gates pass by default |
| `p2gn16_production_export_disabled.go`, boundary-export stubs | Deliberately unavailable production exports | The original full-functional deployment is reproduced by this reduced source package |
| `hardware/u55c/hls` | P1 primitive kernels/fixture path | A complete production network or matched paper construction was measured |

Before a comparison bind both source hashes, parameter primes, input/output
key semantics, correctness, online/offline boundaries, threading and trial
policy. A compile or a selected correctness test does not close that chain.
Packaging does not select or promote a construction.
