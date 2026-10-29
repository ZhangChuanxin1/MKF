# Runbook

Run PowerShell commands from the repository root. Heavy commands require
`-AllowHeavy`. Run FHE/HLS/Vivado jobs serially. `GOMAXPROCS=1` limits Go
execution but does not pin a CPU or guarantee zero page faults.

## Build/correctness

```powershell
pwsh -NoProfile -File scripts/run_suite.ps1 -Profile smoke
pwsh -NoProfile -File scripts/run_suite.ps1 -Profile software
```

## Historical MKF CPU profiler

```powershell
pwsh -NoProfile -File scripts/run_experiment.ps1 -Experiment cpu-mkf `
  -Scheme ckks -Parameter P1 -Parties 2 -Trials 10 -Warmup 5 -AllowHeavy
```

Choose BFV/P2/larger parties explicitly. Key generation/encryption are outside
the timed operation. This is the historical aggregate-output path, not the
completed paper-construction matched benchmark. Raw CSV/log/environment and
manifest identity are retained.

## KKLSS profiling

```powershell
pwsh -NoProfile -File scripts/run_experiment.ps1 -Experiment cpu-kklss-stages `
  -Scheme ckks -Parameter P1 -Parties 16 -Trials 10 -Warmup 5 -AllowHeavy
```

Repeat with BFV. These tests fix P1/n=16, five warm-ups, ten observations;
the wrapper rejects other settings. They do not independently decrypt every
timed result and are not a publication-qualified comparison by themselves.

Original scaling benchmarks are also present. They run both parameter sets
and powers of two up to `-n`; Go calibration is not a zero-warmup policy:

```powershell
$env:GOMAXPROCS='1'
Push-Location software/kklss
go test ./mkckks -run '^$' -bench '^BenchmarkMKCKKS$' -benchtime=10x -count=1 -timeout=24h -args -n=16
go test ./mkbfv -run '^$' -bench '^BenchmarkMKBFV$' -benchtime=10x -count=1 -timeout=24h -args -n=16
Pop-Location
```

Retain source hashes, CPU/affinity, memory/page faults, trial statistics,
validation and boundaries before claiming a same-CPU speedup.

## Noise/depth

```powershell
pwsh -NoProfile -File scripts/run_experiment.ps1 -Experiment noise `
  -Implementation mkf -Scheme ckks -Parameter P1 -Parties 2 `
  -Trials 3 -MaxDepth 1 -AllowHeavy
```

Repeat for KKLSS, BFV, P2 and other party counts. Increasing MaxDepth requests
an actual experiment, not guaranteed correctness. Historical CKKS chain mode
is selected; guarded/compact modes are separate source paths, not silently
substituted by this wrapper.

## Flooding

```powershell
py -3 -m pip install -r requirements-software.txt
py -3 experiments/flooding/simulate_uniform_flooding.py `
  --input experiments/flooding/preflood_calibration_canonical_p1p2_raw.csv `
  --out-dir .local/flooding-review --mc-trials 4
```

Use a fresh output directory. This is conditional model replay, not new
calibration or proof of a statistical-security bound.

## MNIST

Prepare the [dataset](../datasets/README.md), then:

```powershell
pwsh -NoProfile -File scripts/run_experiment.ps1 -Experiment mnist -AllowHeavy
```

Upstream `TestCNN` uses image index 2. It is not 10,000-image accuracy or a
complete matched MKF CNN benchmark.

## ResNet-20

```powershell
py -3 experiments/resnet20/plain_reference_trace.py --help
py -3 experiments/resnet20/calibrate_activation_scales.py --help
```

Install requirements-resnet.txt; supply checkpoint, CIFAR batch and folded
weights. Use explicit `--source-file`, `--checkpoint`, `--folded-weight-root`,
`--data-root`, `--test-batch`, `--output` as needed; defaults preserve the
historical asset layout. Frozen model source is included under third_party.
Plaintext time is not FHE time. Go functional tests remain opt-in; production
exports are excluded, so no full-network timing shortcut is advertised.

## Security (Linux/Sage)

The P2GN16 script requires SageMath **10.5** and estimator revision
`d8c00b483f70b6298690386b1a3278b5167a2e42`. From that separately provisioned
estimator checkout (for the revision gate and imports):

```bash
sage -python /absolute/path/to/artifact/experiments/security/p2g_n16_qp_security_estimate.sage.py
```

Retain versions, command and raw output. P3 scripts are candidate inputs, not
passing profiles. Ordinary-LWE estimates do not establish RLWE/KDM/protocol security.

## U55C

```powershell
pwsh -NoProfile -File scripts/generate_fixtures.ps1 -AllowLargeOutput
pwsh -NoProfile -File scripts/run_suite.ps1 -Profile u55c-csim
pwsh -NoProfile -File scripts/run_suite.ps1 -Profile u55c-csynth `
  -ClockProfile smoke -AllowHeavySynthesis
```

Activate Vitis/Vivado 2022.2 and the configured platform. No board programming
occurs. Separate [microkernel/RTL wrappers](../hardware/u55c/micro/README.md)
are source-only until actually run and validated.

## Figures

```powershell
py -3 -m pip install -r requirements-figures.txt
pwsh -NoProfile -File scripts/run_experiment.ps1 -Experiment figures
```

The wrapper copies sources to fresh `.local/` directories before plotting;
tracked files remain unchanged. Read [provenance](../figures/README.md).
