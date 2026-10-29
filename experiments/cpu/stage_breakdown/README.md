# CPU stage attribution

Status: `SOURCE_AVAILABLE`.

Stage instrumentation and the KKLSS P1/n=16 stage tests are available. The wrapper `scripts/run_experiment.ps1 -Experiment cpu-kklss-stages` uses the fixed harness policy. Instrumented stage timers must not be replaced with proportional allocations from total latency.

## Canonical source references

Paths below are relative to the artifact root; existing source files are not moved or copied:

- `software/mkf/stageprof/stageprof.go`
- `software/kklss/stageprof/stageprof.go`
- `experiments/cpu/kklss_exact_tests`

The machine-readable index is `config/experiment_layout.json`. A directory or source reference does not establish completed measurement evidence.
