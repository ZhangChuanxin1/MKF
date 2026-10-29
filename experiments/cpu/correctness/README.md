# CPU correctness

Status: `SOURCE_AVAILABLE`.

Selected MKF and KKLSS scheme correctness tests are retained in the canonical Go modules. Use the software profile in `scripts/run_suite.ps1` from the artifact root. A selected test PASS is not a matched performance result.

## Canonical source references

Paths below are relative to the artifact root; existing source files are not moved or copied:

- `software/mkf/mkf_cpu/ckks_test.go`
- `software/mkf/mkf_cpu/bfv_test.go`
- `software/kklss/mkckks/mkckks_test.go`
- `software/kklss/mkbfv/mkbfv_test.go`

The machine-readable index is `config/experiment_layout.json`. A directory or source reference does not establish completed measurement evidence.
