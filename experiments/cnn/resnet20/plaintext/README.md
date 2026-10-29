# ResNet-20 plaintext reference

Status: `REQUIRES_EXTERNAL_ASSETS`.

The plaintext reference and activation-calibration tools remain in `experiments/resnet20`. Supply the matching checkpoint, CIFAR batches, folded weights and calibrated scales through explicit asset paths. Plaintext execution time is not encrypted-inference latency.

## Canonical source references

Paths below are relative to the artifact root; existing source files are not moved or copied:

- `experiments/resnet20/plain_reference_trace.py`
- `experiments/resnet20/calibrate_activation_scales.py`

The machine-readable index is `config/experiment_layout.json`. A directory or source reference does not establish completed measurement evidence.
