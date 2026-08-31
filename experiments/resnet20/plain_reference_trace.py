#!/usr/bin/env python3
"""Offline, reproducible ResNet-20/CIFAR-10 plaintext accuracy/range trace.

This is a numerical/reference gate for the MKF experiment.  It is not an FHE
implementation and its elapsed time must never be reported as FHE performance.
The model definition, checkpoint, and CIFAR Python batch are all loaded from
local files; this script never calls torch.hub or downloads anything.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import pickle
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch


SCRIPT_PATH = Path(__file__).resolve()
WORKSPACE_ROOT = SCRIPT_PATH.parents[2]
REFERENCE_ROOT = WORKSPACE_ROOT / "third_party" / "resnet20_reference"
SOURCE_COMMIT = "786c16252c0fc58ee9adac063f8337cc4a7a497a"
LOW_MEMORY_COMMIT = "c818bd3519794191dfcf78c252bc209ee30e3e4a"
SOURCE_ROOT = REFERENCE_ROOT / f"pytorch-cifar-models-{SOURCE_COMMIT}"
SOURCE_FILE = SOURCE_ROOT / "pytorch_cifar_models" / "resnet.py"
SOURCE_ARCHIVE = REFERENCE_ROOT / f"pytorch-cifar-models-{SOURCE_COMMIT}.zip"
CHECKPOINT = REFERENCE_ROOT / "checkpoint" / "cifar10_resnet20-4118986f.pt"
LOW_MEMORY_ROOT = REFERENCE_ROOT / f"LowMemoryFHEResNet20-{LOW_MEMORY_COMMIT}"
FOLDED_WEIGHT_ROOT = LOW_MEMORY_ROOT / "weights"
EXPECTED_SOURCE_FILE_SHA256 = "885b8fd2ccfb8ee27c02d3a382e2951622fdcbc2af2bbca375aaf71d40f5f246"
EXPECTED_CHECKPOINT_SHA256 = "4118986f0df73003d572b0e397f0ac7b3f60af1f31aff3d2da164536e36f6ec8"
EXPECTED_CIFAR_ARCHIVE_BYTES = 170_498_071
EXPECTED_CIFAR_ARCHIVE_MD5 = "c58f30108f718f92721af3b95e74349a"
CIFAR10_CHECKPOINT_MEAN: Tuple[float, float, float] = (0.4914, 0.4822, 0.4465)
CIFAR10_CHECKPOINT_STD: Tuple[float, float, float] = (0.2023, 0.1994, 0.2010)
DEFAULT_DATA_ROOT = (
    WORKSPACE_ROOT
    / "experiments"
    / "resnet20_p3"
    / "data"
    / "cifar10_official_20260825"
)

CHEBYSHEV_DEGREE = 59
SITE_SCALES: Tuple[float, ...] = (
    0.90,
    1.00,
    0.52,
    0.55,
    0.36,
    0.63,
    0.42,
    0.57,
    0.40,
    0.76,
    0.37,
    0.63,
    0.25,
    0.63,
    0.40,
    0.57,
    0.33,
    0.69,
    0.10,
)
SITE_NAMES: Tuple[str, ...] = (
    "stem.conv1_bn1",
    "layer1.block1.conv1_bn1",
    "layer1.block1.residual_sum",
    "layer1.block2.conv1_bn1",
    "layer1.block2.residual_sum",
    "layer1.block3.conv1_bn1",
    "layer1.block3.residual_sum",
    "layer2.block1.conv1_bn1",
    "layer2.block1.residual_sum",
    "layer2.block2.conv1_bn1",
    "layer2.block2.residual_sum",
    "layer2.block3.conv1_bn1",
    "layer2.block3.residual_sum",
    "layer3.block1.conv1_bn1",
    "layer3.block1.residual_sum",
    "layer3.block2.conv1_bn1",
    "layer3.block2.residual_sum",
    "layer3.block3.conv1_bn1",
    "layer3.block3.residual_sum",
)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def md5_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_arrays(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def artifact(path: Path) -> Dict[str, Any]:
    result: Dict[str, Any] = {"path": str(path)}
    if path.is_file():
        result.update({"exists": True, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    else:
        result["exists"] = False
    return result


def require_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"missing {description}: {path}")


def load_model(source_file: Path, checkpoint: Path, dtype: torch.dtype) -> torch.nn.Module:
    """Construct pretrained=False from the frozen source and load locally."""
    require_file(source_file, "frozen model source")
    require_file(checkpoint, "frozen checkpoint")
    spec = importlib.util.spec_from_file_location("frozen_cifar_resnet", source_file)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import frozen model source: {source_file}")
    module = importlib.util.module_from_spec(spec)
    # The frozen module populates dynamic constructors through
    # sys.modules[__name__] at import time.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    model = module.cifar10_resnet20(pretrained=False)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    incompat = model.load_state_dict(state, strict=True)
    if incompat.missing_keys or incompat.unexpected_keys:
        raise RuntimeError(
            f"strict checkpoint load failed: missing={incompat.missing_keys}, "
            f"unexpected={incompat.unexpected_keys}"
        )
    model.eval()
    model.to(device="cpu", dtype=dtype)
    return model


def build_mask(starting_padding: int, ending_padding: int, window_length: int, max_length: int) -> np.ndarray:
    """Literal translation of the frozen export notebook's build_mask."""
    mask: List[int] = [0] * starting_padding
    while len(mask) < max_length - ending_padding:
        mask.extend([1] * window_length)
        mask.append(0)
    mask = mask[:max_length]
    if len(mask) < max_length:
        mask.extend([0] * (max_length - len(mask)))
    for offset in range(ending_padding):
        mask[max_length - offset - 1] = 0
    return np.asarray(mask, dtype=np.float64)


def check_frozen_stem(model: torch.nn.Module, weight_root: Path, tolerance: float) -> Dict[str, Any]:
    """Gate checkpoint conv1/BN against all 144 masks and 16 bias blocks."""
    require_file(weight_root / "conv1bn1-bias.bin", "frozen folded stem bias")
    conv = model.conv1.weight.detach().to(dtype=torch.float32, device="cpu").numpy()
    bn = model.bn1
    gamma = bn.weight.detach().to(dtype=torch.float32, device="cpu")
    variance = bn.running_var.detach().to(dtype=torch.float32, device="cpu")
    mean = bn.running_mean.detach().to(dtype=torch.float32, device="cpu")
    beta = bn.bias.detach().to(dtype=torch.float32, device="cpu")
    alpha = (gamma / torch.sqrt(variance + float(bn.eps))).numpy()
    folded_bias = (beta - gamma * mean / torch.sqrt(variance + float(bn.eps))).numpy()

    width = 32
    one_plane = width * width
    masks = (
        build_mask(width + 1, 0, width - 1, one_plane),
        build_mask(width, 0, one_plane, one_plane),
        build_mask(width, 0, width - 1, one_plane),
        build_mask(1, 0, width - 1, one_plane),
        build_mask(0, 0, one_plane, one_plane),
        build_mask(0, 1, width - 1, one_plane),
        build_mask(1, width - 1, width - 1, one_plane),
        build_mask(0, width, one_plane, one_plane),
        build_mask(0, width + 1, width - 1, one_plane),
    )
    masks = tuple(np.tile(mask, 16) for mask in masks)

    maximum = 0.0
    worst_file: Optional[str] = None
    checked_values = 0
    file_count = 0
    for output_channel in range(16):
        for kernel_index in range(9):
            path = weight_root / f"conv1bn1-ch{output_channel}-k{kernel_index + 1}.bin"
            require_file(path, "frozen folded stem kernel")
            observed = np.loadtxt(path, dtype=np.float64)
            expected_blocks = [
                np.full(one_plane, float(alpha[output_channel] * conv[output_channel, input_channel].reshape(-1)[kernel_index]), dtype=np.float64)
                for input_channel in range(3)
            ]
            expected_blocks.extend(np.zeros(one_plane, dtype=np.float64) for _ in range(13))
            expected = np.concatenate(expected_blocks) * masks[kernel_index]
            if observed.shape != expected.shape:
                raise RuntimeError(f"stem file shape mismatch for {path}: {observed.shape} != {expected.shape}")
            difference = float(np.max(np.abs(observed - expected)))
            if difference > maximum:
                maximum = difference
                worst_file = path.name
            checked_values += observed.size
            file_count += 1

    bias_path = weight_root / "conv1bn1-bias.bin"
    observed_bias = np.loadtxt(bias_path, dtype=np.float64)
    expected_bias = np.repeat(folded_bias.astype(np.float64), one_plane)
    if observed_bias.shape != expected_bias.shape:
        raise RuntimeError(
            f"stem bias shape mismatch for {bias_path}: {observed_bias.shape} != {expected_bias.shape}"
        )
    block_differences = [
        float(
            np.max(
                np.abs(
                    observed_bias[channel * one_plane : (channel + 1) * one_plane]
                    - float(folded_bias[channel])
                )
            )
        )
        for channel in range(16)
    ]
    bias_maximum = max(block_differences)
    passed = maximum <= tolerance and bias_maximum <= tolerance and file_count == 144
    return {
        "pass": passed,
        "tolerance": tolerance,
        "kernel_files_checked": file_count,
        "kernel_values_checked": checked_values,
        "kernel_max_abs_diff": maximum,
        "kernel_worst_file": worst_file,
        "bias_blocks_checked": 16,
        "bias_values_checked": int(observed_bias.size),
        "bias_block_max_abs_diffs": block_differences,
        "bias_max_abs_diff": bias_maximum,
        "fold_formula": "alpha=gamma/sqrt(running_var+eps); W_fold=alpha*W; b_fold=beta-alpha*running_mean",
    }


def chebyshev_coefficients(degree: int) -> np.ndarray:
    """Match Lattigo v2.3 ckks.Approximate on [-1,1].

    The interpolation uses degree+1 first-kind roots and stores the series as
    c[0] + sum_{j=1}^degree c[j] T_j(x).
    """
    count = degree + 1
    indices = np.arange(count, dtype=np.float64)
    theta = (indices + 0.5) * np.pi / float(count)
    nodes = np.cos(theta)
    values = np.maximum(nodes, 0.0)
    coefficients = np.empty(count, dtype=np.float64)
    coefficients[0] = np.sum(values) / float(count)
    for degree_index in range(1, count):
        coefficients[degree_index] = (
            2.0
            * np.sum(values * np.cos(float(degree_index) * theta))
            / float(count)
        )
    return coefficients


def chebyshev_eval_tensor(x: torch.Tensor, coefficients: torch.Tensor) -> torch.Tensor:
    """Clenshaw evaluation of the frozen ReLU interpolant.

    ReLU(x)=(x+|x|)/2 makes every odd coefficient above c1 vanish in
    exact arithmetic. The root-interpolation construction leaves only
    float64 roundoff (below 7e-16) in those positions. Evaluating the even
    series at y=2*x^2-1 is algebraically identical because
    T_{2j}(x)=T_j(y), and halves plaintext calibration/evaluation time. The
    published coefficient vector and its hash remain unchanged.
    """
    odd_tail = coefficients[3::2]
    if coefficients.numel() >= 4 and (
        odd_tail.numel() == 0 or float(torch.max(torch.abs(odd_tail)).item()) <= 1e-12
    ):
        even_coefficients = coefficients[::2]
        y = 2.0 * x * x - 1.0
        return coefficients[1] * x + _chebyshev_clenshaw(y, even_coefficients)
    return _chebyshev_clenshaw(x, coefficients)


def _chebyshev_clenshaw(x: torch.Tensor, coefficients: torch.Tensor) -> torch.Tensor:
    """Evaluate c0 + sum(cj*Tj(x)) for an arbitrary coefficient vector."""
    b_k_plus_1 = torch.zeros_like(x)
    b_k_plus_2 = torch.zeros_like(x)
    for coefficient_index in range(coefficients.numel() - 1, 0, -1):
        coefficient = coefficients[coefficient_index]
        b_k = 2.0 * x * b_k_plus_1 - b_k_plus_2 + coefficient
        b_k_plus_2 = b_k_plus_1
        b_k_plus_1 = b_k
    return x * b_k_plus_1 - b_k_plus_2 + coefficients[0]


class RangeTracker:
    def __init__(self, mode: str, site_scales: Sequence[float] = SITE_SCALES) -> None:
        self.mode = mode
        self.site_scales = tuple(float(value) for value in site_scales)
        if len(self.site_scales) != len(SITE_NAMES):
            raise ValueError(
                f"expected {len(SITE_NAMES)} site scales, received {len(self.site_scales)}"
            )
        if any(not math.isfinite(value) or value <= 0.0 for value in self.site_scales):
            raise ValueError("all activation site scales must be finite and positive")
        self.next_site = 0
        self.records: List[Dict[str, Any]] = [
            {
                "id": f"A{index + 1:02d}",
                "name": SITE_NAMES[index],
                "scale": self.site_scales[index],
                "calls": 0,
                "value_count": 0,
                "finite_count": 0,
                "nonfinite_count": 0,
                "out_of_minus1_plus1_count": 0,
                "scaled_min": None,
                "scaled_max": None,
            }
            for index in range(len(self.site_scales))
        ]

    def begin_forward(self) -> None:
        if self.next_site != 0:
            raise RuntimeError(f"tracker {self.mode} began with dangling site index {self.next_site}")

    def update(self, scaled: torch.Tensor) -> int:
        if self.next_site >= len(self.records):
            raise RuntimeError(f"more than {len(self.records)} activation calls in one forward")
        site_index = self.next_site
        self.next_site += 1
        record = self.records[site_index]
        detached = scaled.detach()
        finite = torch.isfinite(detached)
        finite_count = int(finite.sum().item())
        total = detached.numel()
        nonfinite = total - finite_count
        outside = int(((detached < -1.0) | (detached > 1.0)).sum().item())
        if finite_count:
            finite_values = detached[finite]
            batch_min = float(finite_values.min().item())
            batch_max = float(finite_values.max().item())
            record["scaled_min"] = batch_min if record["scaled_min"] is None else min(record["scaled_min"], batch_min)
            record["scaled_max"] = batch_max if record["scaled_max"] is None else max(record["scaled_max"], batch_max)
        record["calls"] += 1
        record["value_count"] += total
        record["finite_count"] += finite_count
        record["nonfinite_count"] += nonfinite
        record["out_of_minus1_plus1_count"] += outside
        return site_index

    def end_forward(self) -> None:
        if self.next_site != len(self.records):
            raise RuntimeError(
                f"tracker {self.mode} observed {self.next_site} activation calls; expected {len(self.records)}"
            )
        self.next_site = 0

    def finalize(self) -> List[Dict[str, Any]]:
        finalized: List[Dict[str, Any]] = []
        for record in self.records:
            item = dict(record)
            denominator = item["value_count"]
            bad = item["out_of_minus1_plus1_count"] + item["nonfinite_count"]
            item["out_of_interval_or_nonfinite_count"] = bad
            item["out_of_interval_or_nonfinite_fraction"] = bad / denominator if denominator else None
            finalized.append(item)
        return finalized


def traced_activation(
    z: torch.Tensor,
    tracker: RangeTracker,
    mode: str,
    coefficients: torch.Tensor,
) -> torch.Tensor:
    site_index = tracker.next_site
    scale = tracker.site_scales[site_index]
    scaled = z * scale
    tracker.update(scaled)
    if mode == "exact":
        return torch.relu(z)
    if mode == "chebyshev":
        return chebyshev_eval_tensor(scaled, coefficients) / scale
    raise ValueError(f"unsupported activation mode: {mode}")


def forward_traced(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    mode: str,
    tracker: RangeTracker,
    coefficients: torch.Tensor,
) -> torch.Tensor:
    tracker.begin_forward()
    out = model.bn1(model.conv1(inputs))
    out = traced_activation(out, tracker, mode, coefficients)
    for layer in (model.layer1, model.layer2, model.layer3):
        for block in layer:
            identity = out
            branch = block.bn1(block.conv1(out))
            branch = traced_activation(branch, tracker, mode, coefficients)
            branch = block.bn2(block.conv2(branch))
            if block.downsample is not None:
                identity = block.downsample(out)
            out = traced_activation(branch + identity, tracker, mode, coefficients)
    out = model.avgpool(out)
    out = torch.flatten(out, 1)
    logits = model.fc(out)
    tracker.end_forward()
    return logits


def load_cifar_test_batch(path: Path) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    require_file(path, "CIFAR-10 Python test_batch")
    with path.open("rb") as handle:
        payload = pickle.load(handle, encoding="bytes")
    data = payload.get(b"data", payload.get("data"))
    labels = payload.get(b"labels", payload.get("labels"))
    if data is None or labels is None:
        raise RuntimeError(f"unrecognized CIFAR-10 pickle keys in {path}")
    raw = np.asarray(data, dtype=np.uint8)
    targets = np.asarray(labels, dtype=np.int64)
    if raw.shape != (10000, 3072) or targets.shape != (10000,):
        raise RuntimeError(f"unexpected CIFAR-10 test shapes: data={raw.shape}, labels={targets.shape}")
    images = raw.reshape(10000, 3, 32, 32)
    metadata = {
        "kind": "official_cifar10_python_test_batch",
        "storage_layout": "uint8 planar RGB reshaped to 3x32x32; numerical preprocessing is recorded separately",
        "sample_count": 10000,
        "test_batch": artifact(path),
        "data_and_label_sha256": sha256_arrays(raw, targets),
    }
    return images, targets, metadata


def make_synthetic(samples: int, seed: int) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    generator = np.random.default_rng(seed)
    images = generator.integers(0, 256, size=(samples, 3, 32, 32), dtype=np.uint8)
    labels = np.arange(samples, dtype=np.int64) % 10
    return images, labels, {
        "kind": "deterministic_synthetic_smoke_only",
        "storage_layout": "deterministic uint8 planar RGB; numerical preprocessing is recorded separately; labels=index mod 10",
        "sample_count": samples,
        "seed": seed,
        "data_and_label_sha256": sha256_arrays(images, labels),
        "accuracy_is_publication_evidence": False,
    }


def find_test_batch(root: Path) -> Optional[Path]:
    candidates = sorted(root.rglob("test_batch")) if root.is_dir() else []
    return candidates[0] if candidates else None


def iter_batches(images: np.ndarray, labels: np.ndarray, batch_size: int) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
    for start in range(0, len(images), batch_size):
        end = min(start + batch_size, len(images))
        yield images[start:end], labels[start:end]


def preprocess_input(raw_batch: np.ndarray, dtype: torch.dtype, mode: str) -> torch.Tensor:
    """Apply the explicitly selected data-owner preprocessing before encryption."""
    input_batch = torch.from_numpy(raw_batch.copy()).to(dtype=dtype).div_(255.0)
    if mode == "raw_rgb01":
        return input_batch
    if mode == "checkpoint_normalized":
        mean = torch.tensor(CIFAR10_CHECKPOINT_MEAN, dtype=dtype).view(1, 3, 1, 1)
        std = torch.tensor(CIFAR10_CHECKPOINT_STD, dtype=dtype).view(1, 3, 1, 1)
        return input_batch.sub_(mean).div_(std)
    raise ValueError(f"unsupported preprocessing mode: {mode}")


def load_site_scales(path_text: Optional[str]) -> Tuple[Tuple[float, ...], Dict[str, Any]]:
    """Load a frozen training-only calibration artifact, or retain the legacy tuple."""
    if path_text is None:
        return SITE_SCALES, {
            "kind": "legacy_embedded_scales",
            "publication_status": "rejected_for_checkpoint_normalized_inference",
            "reason": "not calibrated for the checkpoint-normalized CIFAR-10 input semantics",
        }

    path = Path(path_text).resolve()
    require_file(path, "frozen activation-scale calibration artifact")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "mkf-resnet20-degree59-training-calibration-v1":
        raise RuntimeError(f"unsupported activation-scale artifact schema in {path}")
    if payload.get("status") != "PASS":
        raise RuntimeError(f"activation-scale artifact is not PASS: {payload.get('status')}")
    if payload.get("activation", {}).get("degree") != CHEBYSHEV_DEGREE:
        raise RuntimeError("activation-scale artifact degree does not match this evaluator")
    if tuple(payload.get("activation", {}).get("site_names", ())) != SITE_NAMES:
        raise RuntimeError("activation-scale artifact site names/order do not match this evaluator")
    values = tuple(float(value) for value in payload.get("activation", {}).get("site_scales", ()))
    if len(values) != len(SITE_NAMES):
        raise RuntimeError(f"activation-scale artifact contains {len(values)} scales; expected {len(SITE_NAMES)}")
    if any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise RuntimeError("activation-scale artifact contains a non-finite or non-positive scale")
    if payload.get("dataset", {}).get("partition") != "official CIFAR-10 training batches 1-5":
        raise RuntimeError("activation scales were not calibrated exclusively on the declared training partition")
    return values, {
        "kind": "frozen_training_only_calibration",
        "artifact": artifact(path),
        "calibration_dataset_partition": payload["dataset"]["partition"],
        "calibration_data_and_label_sha256": payload["dataset"]["data_and_label_sha256"],
        "safety_utilization": payload["calibration_algorithm"]["safety_utilization"],
    }


def run(args: argparse.Namespace) -> Dict[str, Any]:
    if args.dtype != "float32":
        raise ValueError("this frozen trace currently accepts only --dtype float32")
    dtype = torch.float32
    torch.set_num_threads(args.threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.use_deterministic_algorithms(True)

    source_file = Path(args.source_file).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    folded_root = Path(args.folded_weight_root).resolve()
    require_file(source_file, "frozen model source")
    require_file(checkpoint, "frozen checkpoint")
    source_digest = sha256_file(source_file)
    checkpoint_digest = sha256_file(checkpoint)
    if source_digest != EXPECTED_SOURCE_FILE_SHA256:
        raise RuntimeError(
            f"frozen model source SHA-256 mismatch: {source_digest} != {EXPECTED_SOURCE_FILE_SHA256}"
        )
    if checkpoint_digest != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError(
            f"frozen checkpoint SHA-256 mismatch: {checkpoint_digest} != {EXPECTED_CHECKPOINT_SHA256}"
        )
    model = load_model(source_file, checkpoint, dtype)
    stem_gate = check_frozen_stem(model, folded_root, args.stem_tolerance)
    if not stem_gate["pass"]:
        raise RuntimeError(f"checkpoint-to-frozen-stem gate failed: {stem_gate}")

    if args.synthetic_smoke:
        images, labels, dataset = make_synthetic(args.synthetic_samples, args.synthetic_seed)
    else:
        archive = Path(args.cifar_archive).resolve()
        require_file(archive, "official CIFAR-10 Python archive")
        if archive.stat().st_size != EXPECTED_CIFAR_ARCHIVE_BYTES:
            raise RuntimeError(
                f"official CIFAR-10 archive size mismatch: {archive.stat().st_size} != {EXPECTED_CIFAR_ARCHIVE_BYTES}"
            )
        archive_md5 = md5_file(archive)
        if archive_md5 != EXPECTED_CIFAR_ARCHIVE_MD5:
            raise RuntimeError(
                f"official CIFAR-10 archive MD5 mismatch: {archive_md5} != {EXPECTED_CIFAR_ARCHIVE_MD5}"
            )
        test_batch = Path(args.test_batch).resolve() if args.test_batch else find_test_batch(Path(args.data_root).resolve())
        if test_batch is None:
            raise FileNotFoundError(
                "official CIFAR-10 test_batch not found; pass --test-batch PATH, "
                "or use --synthetic-smoke only for a non-accuracy smoke gate"
            )
        images, labels, dataset = load_cifar_test_batch(test_batch)

    site_scales, scale_source = load_site_scales(args.site_scales_file)
    coefficient_array = chebyshev_coefficients(CHEBYSHEV_DEGREE)
    coefficient_tensor = torch.tensor(coefficient_array, dtype=dtype, device="cpu")
    dense_grid = np.linspace(-1.0, 1.0, 200001, dtype=np.float64)
    approximation = np.polynomial.chebyshev.chebval(dense_grid, coefficient_array)
    max_grid_error = float(np.max(np.abs(approximation - np.maximum(dense_grid, 0.0))))

    modes = ("exact", "chebyshev") if args.mode == "both" else (args.mode,)
    trackers = {mode: RangeTracker(mode, site_scales) for mode in modes}
    predictions: Dict[str, List[np.ndarray]] = {mode: [] for mode in modes}
    correct = {mode: 0 for mode in modes}
    manual_module_max_diff = 0.0
    batch_count = 0
    start_time = time.perf_counter()
    with torch.inference_mode():
        for raw_batch, label_batch in iter_batches(images, labels, args.batch_size):
            input_batch = preprocess_input(raw_batch, dtype, args.preprocessing)
            label_tensor = torch.from_numpy(label_batch)
            for mode in modes:
                logits = forward_traced(model, input_batch, mode, trackers[mode], coefficient_tensor)
                predicted = logits.argmax(dim=1).to(dtype=torch.int64).cpu()
                predictions[mode].append(predicted.numpy())
                correct[mode] += int((predicted == label_tensor).sum().item())
                if mode == "exact" and batch_count == 0:
                    module_logits = model(input_batch)
                    manual_module_max_diff = max(
                        manual_module_max_diff,
                        float(torch.max(torch.abs(logits - module_logits)).item()),
                    )
            batch_count += 1
            if args.progress_every and batch_count % args.progress_every == 0:
                print(
                    f"plaintext trace progress: batches={batch_count}, samples={min(batch_count * args.batch_size, len(images))}/{len(images)}",
                    file=sys.stderr,
                    flush=True,
                )
    elapsed = time.perf_counter() - start_time

    merged_predictions = {
        mode: np.concatenate(chunks).astype(np.int64, copy=False)
        for mode, chunks in predictions.items()
    }
    results: Dict[str, Any] = {}
    for mode in modes:
        mode_predictions = merged_predictions[mode]
        results[mode] = {
            "top1_correct": correct[mode],
            "sample_count": int(len(labels)),
            "top1_accuracy_fraction": correct[mode] / float(len(labels)),
            "prediction_sha256": sha256_arrays(mode_predictions),
            "activation_sites": trackers[mode].finalize(),
        }
    agreement: Optional[Dict[str, Any]] = None
    if set(modes) == {"exact", "chebyshev"}:
        same = int(np.sum(merged_predictions["exact"] == merged_predictions["chebyshev"]))
        agreement = {
            "same_prediction_count": same,
            "sample_count": int(len(labels)),
            "fraction": same / float(len(labels)),
        }

    range_gate_modes: Dict[str, Any] = {}
    for mode in modes:
        bad = sum(
            site["out_of_interval_or_nonfinite_count"]
            for site in results[mode]["activation_sites"]
        )
        range_gate_modes[mode] = {
            "pass": bad == 0,
            "out_of_interval_or_nonfinite_count_across_sites": bad,
        }
    range_gate_pass = all(value["pass"] for value in range_gate_modes.values())

    archive_path = Path(args.cifar_archive).resolve()
    report: Dict[str, Any] = {
        "schema": "mkf-resnet20-plaintext-reference-trace-v3",
        "status": "SMOKE_PASS" if args.synthetic_smoke else ("PASS" if range_gate_pass else "COMPLETED_WITH_RANGE_VIOLATIONS"),
        "evidence_scope": "plaintext numerical/accuracy trace; not FHE performance",
        "runtime": {
            "python": platform.python_version(),
            "python_executable": sys.executable,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "dtype": args.dtype,
            "device": "cpu",
            "torch_threads": args.threads,
            "batch_size": args.batch_size,
            "batch_count": batch_count,
            "deterministic_algorithms": True,
            "plaintext_reference_wall_clock_seconds_not_fhe_performance": elapsed,
        },
        "provenance": {
            "script": artifact(SCRIPT_PATH),
            "model_source_commit": SOURCE_COMMIT,
            "model_source_file": artifact(source_file),
            "model_source_archive": artifact(SOURCE_ARCHIVE),
            "checkpoint": artifact(checkpoint),
            "low_memory_reference_commit": LOW_MEMORY_COMMIT,
            "folded_weight_root": str(folded_root),
            "cifar_archive": artifact(archive_path),
        },
        "checkpoint_to_frozen_stem_gate": stem_gate,
        "dataset": dataset,
        "input_preprocessing": {
            "mode": args.preprocessing,
            "executed_by": "data owner before CKKS encoding/encryption",
            "base_conversion": "uint8 RGB / 255",
            "mean": list(CIFAR10_CHECKPOINT_MEAN) if args.preprocessing == "checkpoint_normalized" else None,
            "std": list(CIFAR10_CHECKPOINT_STD) if args.preprocessing == "checkpoint_normalized" else None,
            "folded_into_stem_weights": False,
        },
        "activation": {
            "degree": CHEBYSHEV_DEGREE,
            "interval": [-1.0, 1.0],
            "site_count": len(site_scales),
            "site_scales": list(site_scales),
            "site_scale_source": scale_source,
            "coefficient_convention": "Lattigo-v2.3-compatible first-kind-root interpolation; c0+sum(cj*Tj(x))",
            "coefficient_float64_sha256": sha256_arrays(coefficient_array),
            "plaintext_evaluation": "parity-reduced Clenshaw using T_2j(x)=T_j(2*x^2-1); coefficient vector unchanged",
            "max_abs_analytically_zero_odd_coefficient_above_c1": float(np.max(np.abs(coefficient_array[3::2]))),
            "dense_200001_point_max_abs_error_vs_relu": max_grid_error,
            "per_site_max_abs_output_error_bound_on_interval": [max_grid_error / scale for scale in site_scales],
            "site_operation": "Chebyshev_s(z)=P59(s*z)/s; exact mode uses ReLU(z)",
        },
        "scaled_preactivation_range_gate": {
            "pass": range_gate_pass,
            "enforced_for_publication": not args.synthetic_smoke,
            "modes": range_gate_modes,
        },
        "manual_exact_forward_vs_frozen_module_first_batch_max_abs_diff": manual_module_max_diff if "exact" in modes else None,
        "results": results,
        "prediction_agreement_exact_vs_chebyshev": agreement,
        "publication_boundaries": {
            "elapsed_is_fhe_performance": False,
            "synthetic_accuracy_is_publication_evidence": not args.synthetic_smoke,
            "encrypted_accuracy_measured": False,
            "fpga_latency_measured": False,
        },
    }
    if args.synthetic_smoke:
        expected_calls = 1 if args.batch_size >= args.synthetic_samples else math.ceil(args.synthetic_samples / args.batch_size)
        for mode in modes:
            observed_calls = [record["calls"] for record in results[mode]["activation_sites"]]
            if len(observed_calls) != 19 or any(count != expected_calls for count in observed_calls):
                raise RuntimeError(
                    f"synthetic smoke activation count failed for {mode}: {observed_calls}, expected {expected_calls}"
                )
        report["synthetic_smoke_gate"] = {
            "pass": True,
            "activation_sites_per_forward": 19,
            "expected_calls_per_site": expected_calls,
        }
    return report


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("exact", "chebyshev", "both"), default="both")
    parser.add_argument("--source-file", default=str(SOURCE_FILE))
    parser.add_argument("--checkpoint", default=str(CHECKPOINT))
    parser.add_argument("--folded-weight-root", default=str(FOLDED_WEIGHT_ROOT))
    parser.add_argument("--stem-tolerance", type=float, default=1e-6)
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--test-batch", default=None)
    parser.add_argument("--cifar-archive", default=str(DEFAULT_DATA_ROOT / "cifar-10-python.tar.gz"))
    parser.add_argument(
        "--preprocessing",
        choices=("checkpoint_normalized", "raw_rgb01"),
        default="checkpoint_normalized",
        help="data-owner preprocessing applied before CKKS encoding; raw_rgb01 is retained only for LowMemory C++ compatibility audits",
    )
    parser.add_argument(
        "--site-scales-file",
        default=None,
        help="PASS training-only calibration JSON; omitted means the rejected legacy embedded tuple",
    )
    parser.add_argument("--synthetic-smoke", action="store_true")
    parser.add_argument("--synthetic-samples", type=int, default=4)
    parser.add_argument("--synthetic-seed", type=int, default=20260825)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--progress-every", type=int, default=10, help="print plaintext progress every N batches; 0 disables")
    parser.add_argument("--dtype", choices=("float32",), default="float32")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.batch_size <= 0 or args.threads <= 0 or args.synthetic_samples <= 0 or args.progress_every < 0:
        parser.error("--batch-size, --threads, and --synthetic-samples must be positive")
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    report = run(args)
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    summary = {
        "status": report["status"],
        "dataset_kind": report["dataset"]["kind"],
        "samples": report["dataset"]["sample_count"],
        "results": {
            mode: value["top1_accuracy_fraction"] for mode, value in report["results"].items()
        },
        "prediction_agreement": report["prediction_agreement_exact_vs_chebyshev"],
        "stem_gate": report["checkpoint_to_frozen_stem_gate"]["pass"],
        "output": str(output),
    }
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
