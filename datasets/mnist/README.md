# MNIST assets

Status: `REQUIRES_EXTERNAL_ASSETS`.

Provide the original MNIST test IDX files locally and use the converter documented in `datasets/README.md`. Generated CSVs are ignored. The included baseline weight/bias text files remain at their canonical implementation paths; they are not copied into this directory.

## Canonical source references

Paths below are relative to the artifact root; existing source files are not moved or copied:

- `scripts/convert_mnist_idx.py`
- `software/kklss/cnn/data/k1.txt`
- `software/kklss/cnn/data/FC1.txt`
- `software/kklss/cnn/data/FC2.txt`
- `software/kklss/cnn/data/B1.txt`
- `software/kklss/cnn/data/B2.txt`

The machine-readable index is `config/experiment_layout.json`. A directory or source reference does not establish completed measurement evidence.
