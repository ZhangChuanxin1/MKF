# KKLSS MNIST CNN

Status: `REQUIRES_EXTERNAL_ASSETS`.

The implementation, tests and included weight/bias files remain in `software/kklss/cnn`. Prepare the MNIST test CSV according to `datasets/README.md` before invoking the opt-in MNIST entry. The included correctness test uses one selected image; it is not a complete accuracy evaluation.

## Canonical source references

Paths below are relative to the artifact root; existing source files are not moved or copied:

- `software/kklss/cnn/cnn.go`
- `software/kklss/cnn/cnn_test.go`
- `software/kklss/cnn/cnn_bench_test.go`
- `scripts/convert_mnist_idx.py`

The machine-readable index is `config/experiment_layout.json`. A directory or source reference does not establish completed measurement evidence.
