# CPU latency experiments

Status: `PARTIAL`.

The historical MKF aggregate-output profiler and the KKLSS benchmarks are available. Use `scripts/run_experiment.ps1 -Experiment cpu-mkf` only with explicit heavy-job opt-in. These paths are not a completed, construction-matched performance matrix; retain the implementation and measurement boundaries.

## Canonical source references

Paths below are relative to the artifact root; existing source files are not moved or copied:

- `software/mkf/cmd/mkfbench/main.go`
- `software/kklss/mkckks/mkckks_benchmark_test.go`
- `software/kklss/mkbfv/mkbfv_bench_test.go`

The machine-readable index is `config/experiment_layout.json`. A directory or source reference does not establish completed measurement evidence.
