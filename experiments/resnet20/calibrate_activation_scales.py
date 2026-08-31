#!/usr/bin/env python3
"""Freeze degree-59 activation scales using only CIFAR-10 training data.

This is a deterministic plaintext calibration/range gate. It never reads the
CIFAR-10 test batch, never measures FHE performance, and never optimizes test
accuracy. The output is consumed read-only by ``plain_reference_trace.py``.
"""

from __future__ import annotations

import argparse
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

import plain_reference_trace as reference


SCRIPT_PATH = Path(__file__).resolve()


def load_training_batches(root: Path) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    images: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    batch_artifacts: List[Dict[str, Any]] = []
    for index in range(1, 6):
        path = root / f"data_batch_{index}"
        reference.require_file(path, f"official CIFAR-10 data_batch_{index}")
        with path.open("rb") as handle:
            payload = pickle.load(handle, encoding="bytes")
        raw = np.asarray(payload.get(b"data", payload.get("data")), dtype=np.uint8)
        targets = np.asarray(payload.get(b"labels", payload.get("labels")), dtype=np.int64)
        if raw.shape != (10_000, 3072) or targets.shape != (10_000,):
            raise RuntimeError(
                f"unexpected training shape in {path}: data={raw.shape}, labels={targets.shape}"
            )
        images.append(raw.reshape(10_000, 3, 32, 32))
        labels.append(targets)
        batch_artifacts.append(reference.artifact(path))
    merged_images = np.concatenate(images, axis=0)
    merged_labels = np.concatenate(labels, axis=0)
    return merged_images, merged_labels, {
        "kind": "official_cifar10_python_training_batches",
        "partition": "official CIFAR-10 training batches 1-5",
        "sample_count": int(merged_labels.size),
        "batch_artifacts": batch_artifacts,
        "data_and_label_sha256": reference.sha256_arrays(merged_images, merged_labels),
        "test_batch_read_by_calibrator": False,
    }


def iter_batches(images: np.ndarray, batch_size: int) -> Iterable[np.ndarray]:
    for start in range(0, len(images), batch_size):
        yield images[start : min(start + batch_size, len(images))]


def trace_ranges(
    model: torch.nn.Module,
    images: np.ndarray,
    mode: str,
    scales: Sequence[float],
    coefficients: torch.Tensor,
    preprocessing: str,
    batch_size: int,
    progress_every: int,
    pass_name: str,
) -> Tuple[List[Dict[str, Any]], float]:
    tracker = reference.RangeTracker(mode, scales)
    start = time.perf_counter()
    with torch.inference_mode():
        for batch_index, raw_batch in enumerate(iter_batches(images, batch_size), start=1):
            inputs = reference.preprocess_input(raw_batch, torch.float32, preprocessing)
            reference.forward_traced(model, inputs, mode, tracker, coefficients)
            if progress_every and batch_index % progress_every == 0:
                print(
                    f"calibration {pass_name}: batches={batch_index}, "
                    f"samples={min(batch_index * batch_size, len(images))}/{len(images)}",
                    file=sys.stderr,
                    flush=True,
                )
    return tracker.finalize(), time.perf_counter() - start


def max_abs(record: Dict[str, Any]) -> float:
    if record["nonfinite_count"]:
        return math.inf
    return max(abs(float(record["scaled_min"])), abs(float(record["scaled_max"])))


def run(args: argparse.Namespace) -> Dict[str, Any]:
    if not 0.0 < args.safety_utilization < 1.0:
        raise ValueError("--safety-utilization must lie strictly between zero and one")
    if args.max_iterations <= 0 or args.batch_size <= 0 or args.threads <= 0:
        raise ValueError("iteration, batch, and thread counts must be positive")

    torch.set_num_threads(args.threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.use_deterministic_algorithms(True)

    source_file = Path(args.source_file).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    folded_root = Path(args.folded_weight_root).resolve()
    training_root = Path(args.training_root).resolve()
    archive = Path(args.cifar_archive).resolve()
    reference.require_file(archive, "official CIFAR-10 Python archive")
    if archive.stat().st_size != reference.EXPECTED_CIFAR_ARCHIVE_BYTES:
        raise RuntimeError("official CIFAR-10 archive size mismatch")
    if reference.md5_file(archive) != reference.EXPECTED_CIFAR_ARCHIVE_MD5:
        raise RuntimeError("official CIFAR-10 archive MD5 mismatch")
    if reference.sha256_file(source_file) != reference.EXPECTED_SOURCE_FILE_SHA256:
        raise RuntimeError("frozen model source SHA-256 mismatch")
    if reference.sha256_file(checkpoint) != reference.EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("frozen checkpoint SHA-256 mismatch")

    model = reference.load_model(source_file, checkpoint, torch.float32)
    stem_gate = reference.check_frozen_stem(model, folded_root, args.stem_tolerance)
    if not stem_gate["pass"]:
        raise RuntimeError(f"checkpoint-to-frozen-stem gate failed: {stem_gate}")
    images, _labels, dataset = load_training_batches(training_root)

    coefficient_array = reference.chebyshev_coefficients(reference.CHEBYSHEV_DEGREE)
    coefficients = torch.tensor(coefficient_array, dtype=torch.float32)
    all_one_scales = (1.0,) * len(reference.SITE_NAMES)
    exact_records, exact_seconds = trace_ranges(
        model,
        images,
        "exact",
        all_one_scales,
        coefficients,
        args.preprocessing,
        args.batch_size,
        args.progress_every,
        "exact-initialization",
    )
    exact_raw_maxima = [max_abs(record) for record in exact_records]
    if any(not math.isfinite(value) or value <= 0.0 for value in exact_raw_maxima):
        raise RuntimeError(f"invalid exact-path calibration maxima: {exact_raw_maxima}")
    scales = [args.safety_utilization / value for value in exact_raw_maxima]

    iterations: List[Dict[str, Any]] = []
    calibration_pass = False
    for iteration in range(1, args.max_iterations + 1):
        records, elapsed = trace_ranges(
            model,
            images,
            "chebyshev",
            scales,
            coefficients,
            args.preprocessing,
            args.batch_size,
            args.progress_every,
            f"chebyshev-fixed-point-{iteration}",
        )
        observed = [max_abs(record) for record in records]
        finite = all(math.isfinite(value) for value in observed)
        within_target = finite and all(
            value <= args.safety_utilization + args.range_tolerance for value in observed
        )
        iterations.append(
            {
                "iteration": iteration,
                "input_site_scales": list(scales),
                "scaled_preactivation_max_abs": observed,
                "records": records,
                "all_finite": finite,
                "all_within_safety_utilization": within_target,
                "plaintext_range_trace_wall_clock_seconds_not_fhe_performance": elapsed,
            }
        )
        if within_target:
            calibration_pass = True
            break
        if not finite:
            break
        scales = [
            scale * min(1.0, args.safety_utilization / max(value, np.finfo(np.float64).tiny))
            for scale, value in zip(scales, observed)
        ]

    final_records = iterations[-1]["records"] if iterations else []
    final_maxima = iterations[-1]["scaled_preactivation_max_abs"] if iterations else []
    report: Dict[str, Any] = {
        "schema": "mkf-resnet20-degree59-training-calibration-v1",
        "status": "PASS" if calibration_pass else "BLOCKED",
        "evidence_scope": "plaintext training-only activation-range calibration; not FHE performance",
        "runtime": {
            "python": platform.python_version(),
            "python_executable": sys.executable,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "device": "cpu",
            "dtype": "float32",
            "torch_threads": args.threads,
            "batch_size": args.batch_size,
            "deterministic_algorithms": True,
            "exact_initialization_wall_clock_seconds_not_fhe_performance": exact_seconds,
        },
        "provenance": {
            "script": reference.artifact(SCRIPT_PATH),
            "reference_script": reference.artifact(reference.SCRIPT_PATH),
            "model_source_commit": reference.SOURCE_COMMIT,
            "model_source_file": reference.artifact(source_file),
            "checkpoint": reference.artifact(checkpoint),
            "cifar_archive": reference.artifact(archive),
            "folded_weight_root": str(folded_root),
        },
        "checkpoint_to_frozen_stem_gate": stem_gate,
        "dataset": dataset,
        "input_preprocessing": {
            "mode": args.preprocessing,
            "mean": list(reference.CIFAR10_CHECKPOINT_MEAN),
            "std": list(reference.CIFAR10_CHECKPOINT_STD),
            "executed_by": "data owner before CKKS encoding/encryption",
        },
        "calibration_algorithm": {
            "name": "training-max fixed-point range calibration",
            "test_labels_or_test_images_used": False,
            "initialization": "s_i = safety_utilization / max_abs(z_i) on exact-ReLU full training trace",
            "iteration": "run degree-59 polynomial path on full training set; shrink only violating s_i by safety_utilization / observed_max_abs(s_i*z_i)",
            "scale_increases_after_initialization": False,
            "safety_utilization": args.safety_utilization,
            "reserved_interval_headroom_fraction": 1.0 - args.safety_utilization,
            "range_tolerance": args.range_tolerance,
            "maximum_iterations": args.max_iterations,
            "completed_iterations": len(iterations),
        },
        "activation": {
            "degree": reference.CHEBYSHEV_DEGREE,
            "interval": [-1.0, 1.0],
            "site_names": list(reference.SITE_NAMES),
            "site_scales": list(scales),
            "exact_path_raw_preactivation_max_abs": exact_raw_maxima,
            "final_polynomial_path_scaled_preactivation_max_abs": final_maxima,
            "final_records": final_records,
            "coefficient_float64_sha256": reference.sha256_arrays(coefficient_array),
            "coefficient_convention": "Lattigo-v2.3-compatible first-kind-root interpolation; c0+sum(cj*Tj(x))",
            "plaintext_evaluation": "parity-reduced Clenshaw using T_2j(x)=T_j(2*x^2-1); coefficient vector unchanged",
            "max_abs_analytically_zero_odd_coefficient_above_c1": float(np.max(np.abs(coefficient_array[3::2]))),
        },
        "iterations": iterations,
        "publication_boundaries": {
            "calibration_used_test_partition": False,
            "accuracy_measured_in_calibration": False,
            "elapsed_is_fhe_performance": False,
            "encrypted_accuracy_measured": False,
            "fpga_latency_measured": False,
        },
    }
    return report


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-file", default=str(reference.SOURCE_FILE))
    parser.add_argument("--checkpoint", default=str(reference.CHECKPOINT))
    parser.add_argument("--folded-weight-root", default=str(reference.FOLDED_WEIGHT_ROOT))
    parser.add_argument("--stem-tolerance", type=float, default=1e-6)
    parser.add_argument(
        "--training-root",
        default=str(reference.DEFAULT_DATA_ROOT / "extracted" / "cifar-10-batches-py"),
    )
    parser.add_argument(
        "--cifar-archive",
        default=str(reference.DEFAULT_DATA_ROOT / "cifar-10-python.tar.gz"),
    )
    parser.add_argument("--preprocessing", choices=("checkpoint_normalized",), default="checkpoint_normalized")
    parser.add_argument("--safety-utilization", type=float, default=0.90)
    parser.add_argument("--range-tolerance", type=float, default=1e-6)
    parser.add_argument("--max-iterations", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--progress-every", type=int, default=20)
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    report = run(args)
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": report["status"],
                "samples": report["dataset"]["sample_count"],
                "iterations": report["calibration_algorithm"]["completed_iterations"],
                "test_partition_used": report["calibration_algorithm"]["test_labels_or_test_images_used"],
                "output": str(output),
            },
            sort_keys=True,
        )
    )
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
