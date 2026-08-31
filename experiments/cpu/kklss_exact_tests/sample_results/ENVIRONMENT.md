# Sample-result environment

These files are a local functional precheck, not the final paper measurement.

- OS: macOS (arm64)
- Go: 1.22.12
- Scheduler: `GOMAXPROCS=1`
- Parameter set: KKLSS `PN14QP439` (`P1`)
- Parties: 16
- Warm-up runs: 5
- Measured runs: 10
- Garbage collection: explicit `runtime.GC()` before each measured interval

The manuscript's final KKLSS and MKF CPU-ST values must be collected on the
same declared server and software environment.
