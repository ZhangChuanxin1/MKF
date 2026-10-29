# Noise/depth source and historical CSVs

Harnesses: MKF `mkf_cpu/noise_experiment_test.go` and KKLSS
`{mkckks,mkbfv}/noise_experiment_test.go`. Use the opt-in noise command in the
runbook for fresh observations.

`historical/` contains byte-identical pre-existing CSV copies: distinct chain
variants, partial/depth-one checks and split runs. Full original manifests,
construction matching and exclusions were not re-established by packaging.
These files are not one newly validated matrix. Do not merge partial/rejected
rows to fill missing cells. Precision, noise budget, correctness and depth
are distinct; adding a prime requires fresh parameter/security assessment.
