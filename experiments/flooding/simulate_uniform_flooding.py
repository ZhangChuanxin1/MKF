#!/usr/bin/env python3
"""Conditional fixed-offset flooding feasibility simulation.

The statistical-distance calculation is a rigorous upper bound conditional
on an admissible coefficient-offset bound D_delta.  The experiment deliberately sets
D_delta = 2 * B_pre_obs, where B_pre_obs is the largest coefficient error
seen in three cryptographic calibration trials.  Consequently, the output is
an arithmetic feasibility study, not a proof that the observed maximum is a
2^-128 bound.

Monte Carlo uses a normalized continuous-uniform approximation to the
discrete uniform sampler.  The deterministic n*W correctness bound and the
closed-form statistical-distance bound are authoritative; Monte Carlo is
reported only as a typical-case arithmetic check.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
from collections import defaultdict
from pathlib import Path

import numpy as np


TARGET_CKKS_PRECISION = 20.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def log2_int(value: int) -> float:
    if value <= 0:
        return float("-inf")
    bits = value.bit_length()
    shift = max(bits - 53, 0)
    top = value >> shift
    return math.log2(top) + shift


def logadd2(a: float, b: float) -> float:
    hi = max(a, b)
    lo = min(a, b)
    if hi == float("-inf"):
        return hi
    return hi + math.log2(1.0 + 2.0 ** (lo - hi))


def quantile(values: list[float], q: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=np.float64), q))


def read_calibration(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("empty calibration CSV")
    return rows


def select_worst(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[(row["scheme"], row["parameter"], int(row["parties"]))].append(row)

    selected: list[dict[str, object]] = []
    for (scheme, parameter, parties), group in sorted(groups.items()):
        if parameter not in {"P1", "P2"}:
            raise ValueError(
                f"unsupported profile {parameter!r}: this script requires explicit "
                "unguarded P1/P2 inputs and never strips profile suffixes"
            )
        b_pre = max(int(row["pre_max_coeff"]) for row in group)
        base = {
            "scheme": scheme,
            "parameter": parameter,
            "set": parameter,
            "parties": parties,
            "trials": len(group),
            "ring_degree": int(group[0]["ring_degree"]),
            "b_pre_obs": b_pre,
            "log2_b_pre_obs": log2_int(b_pre),
            "log2_scale_or_delta": float(group[0]["log2_scale_or_delta"]),
            "log2_active_q": float(group[0]["log2_active_q"]),
        }
        if scheme == "CKKS":
            base["pre_margin_bits"] = min(
                float(row["decoded_precision_bits"]) for row in group
            )
        else:
            base["pre_margin_bits"] = (
                float(group[0]["log2_scale_or_delta"]) - 1.0 - log2_int(b_pre)
            )
        selected.append(base)
    return selected


def derive_profile(row: dict[str, object], security_lambda: int) -> dict[str, object]:
    n = int(row["parties"])
    degree = int(row["ring_degree"])
    b_pre = int(row["b_pre_obs"])
    ceil_log_n = (n - 1).bit_length()
    d_delta = 2 * b_pre
    exponent = security_lambda + ceil_log_n

    # W = ceil((N D_delta 2^exponent - 1)/2).
    numerator = degree * d_delta * (1 << exponent) - 1
    width = (numerator + 1) // 2
    aggregate_cap = n * width

    log2_per_share_sd = (
        math.log2(degree)
        + log2_int(d_delta)
        - log2_int(2 * width + 1)
    )
    log2_joint_sd = math.log2(n) + log2_per_share_sd
    log2_total_coeff = logadd2(log2_int(b_pre), log2_int(aggregate_cap))

    if row["scheme"] == "CKKS":
        log2_mask_slot_bound = (
            math.log2(degree)
            + log2_int(aggregate_cap)
            - float(row["log2_scale_or_delta"])
        )
        log2_pre_slot_error = -float(row["pre_margin_bits"])
        deterministic_post = -logadd2(log2_pre_slot_error, log2_mask_slot_bound)
    else:
        deterministic_post = (
            float(row["log2_scale_or_delta"]) - 1.0 - log2_total_coeff
        )

    no_wrap_headroom = (
        float(row["log2_active_q"]) - 1.0 - log2_total_coeff
    )
    return {
        **row,
        "lambda": security_lambda,
        "d_delta_obs": d_delta,
        "log2_d_delta_obs": log2_int(d_delta),
        "width_w": width,
        "log2_width_w": log2_int(width),
        "aggregate_cap_nw": aggregate_cap,
        "log2_aggregate_cap_nw": log2_int(aggregate_cap),
        "log2_per_share_sd_bound": log2_per_share_sd,
        "log2_joint_sd_bound": log2_joint_sd,
        "deterministic_post_margin_bits": deterministic_post,
        "deterministic_target_pass": (
            deterministic_post >= TARGET_CKKS_PRECISION
            if row["scheme"] == "CKKS"
            else deterministic_post > 0.0
        ),
        "mask_only_no_wrap_headroom_bits": no_wrap_headroom,
    }


def simulate_n64(
    profile: dict[str, object],
    rng: np.random.Generator,
    mc_trials: int,
) -> dict[str, object]:
    n = int(profile["parties"])
    degree = int(profile["ring_degree"])
    width = int(profile["width_w"])
    log2_width = float(profile["log2_width_w"])
    b_pre = int(profile["b_pre_obs"])
    margins: list[float] = []
    aggregate_max_logs: list[float] = []

    if n != 64:
        raise ValueError("Monte Carlo is intentionally restricted to n=64")

    twist = None
    if profile["scheme"] == "CKKS":
        indices = np.arange(degree, dtype=np.float64)
        twist = np.exp(-1j * np.pi * indices / degree)

    for _ in range(mc_trials):
        normalized_sum = np.zeros(degree, dtype=np.float64)
        for _party in range(n):
            normalized_sum += rng.uniform(-1.0, 1.0, size=degree)

        coeff_max = float(np.max(np.abs(normalized_sum)))
        aggregate_max_logs.append(log2_width + math.log2(coeff_max))

        if profile["scheme"] == "CKKS":
            evaluations = np.fft.fft(normalized_sum * twist)[: degree // 2]
            normalized_slot_max = float(np.max(np.abs(evaluations)))
            log2_mask_slot = (
                log2_width
                + math.log2(normalized_slot_max)
                - float(profile["log2_scale_or_delta"])
            )
            log2_pre_slot = -float(profile["pre_margin_bits"])
            margins.append(-logadd2(log2_pre_slot, log2_mask_slot))
        else:
            log2_mask_coeff = log2_width + math.log2(coeff_max)
            log2_total = logadd2(log2_int(b_pre), log2_mask_coeff)
            margins.append(
                float(profile["log2_scale_or_delta"]) - 1.0 - log2_total
            )

    if profile["scheme"] == "CKKS":
        failures = sum(value < TARGET_CKKS_PRECISION for value in margins)
    else:
        failures = sum(value <= 0.0 for value in margins)

    return {
        "scheme": profile["scheme"],
        "set": profile["set"],
        "parties": n,
        "ring_degree": degree,
        "mc_trials": mc_trials,
        "mc_failures": failures,
        "mc_min_margin_bits": min(margins),
        "mc_median_margin_bits": quantile(margins, 0.5),
        "mc_q01_margin_bits": quantile(margins, 0.01),
        "mc_max_aggregate_coeff_log2": max(aggregate_max_logs),
        "mc_q99_aggregate_coeff_log2": quantile(aggregate_max_logs, 0.99),
        "sampler_note": (
            "continuous-normalized approximation to discrete uniform; "
            "deterministic cap is exact and SD is a rigorous conditional upper bound"
        ),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("no rows to write")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def render_summary(
    profiles: list[dict[str, object]],
    mc_rows: list[dict[str, object]],
    security_lambda: int,
    seed: int,
) -> str:
    worst = {
        (str(row["scheme"]), str(row["set"])): row
        for row in profiles
        if int(row["parties"]) == 64
    }
    mc_map = {(str(row["scheme"]), str(row["set"])): row for row in mc_rows}
    lines = [
        "# Conditional uniform-flooding feasibility simulation",
        "",
        f"- Security target: lambda={security_lambda}.",
        "- Calibration: three fresh cryptographic trials per profile and party count.",
        "- Admissible-offset proxy: D_delta=2*B_pre_obs.",
        "- Sampler: independent coefficient-wise discrete uniform [-W,W].",
        "- Monte Carlo: normalized continuous approximation, n=64 only.",
        f"- RNG: NumPy PCG64DXSM, seed=0x{seed:X}.",
        "",
        "| Scheme | Set | Pre margin (bit) | log2 W | log2(nW) | Deterministic post (bit) | MC min (bit) | Failures |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key in [("CKKS", "P1"), ("CKKS", "P2"), ("BFV", "P1"), ("BFV", "P2")]:
        profile = worst[key]
        mc = mc_map[key]
        lines.append(
            "| {scheme} | {set_name} | {pre:.2f} | {w:.2f} | {cap:.2f} | "
            "{post:.2f} | {mc_min:.2f} | {fail}/{trials} |".format(
                scheme=key[0],
                set_name=key[1],
                pre=float(profile["pre_margin_bits"]),
                w=float(profile["log2_width_w"]),
                cap=float(profile["log2_aggregate_cap_nw"]),
                post=float(profile["deterministic_post_margin_bits"]),
                mc_min=float(mc["mc_min_margin_bits"]),
                fail=int(mc["mc_failures"]),
                trials=int(mc["mc_trials"]),
            )
        )
    lines.extend(
        [
            "",
            "The whole-polynomial per-share SD bound includes the ring-degree factor N;",
            "the n-share hybrid bound is at most 2^-lambda conditional on D_delta.",
            "The deterministic aggregate mask cap is nW and has zero tail failure.",
            "",
            "This study does not certify B_pre_obs as an admissible 2^-lambda bound.",
            "It evaluates arithmetic feasibility only. CKKS failure is caused by loss",
            "of precision, not by mask-only modular wraparound.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--lambda", dest="security_lambda", type=int, default=128)
    parser.add_argument("--mc-trials", type=int, default=128)
    parser.add_argument("--seed", type=lambda value: int(value, 0), default=0xF100D128)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    calibration = read_calibration(args.input)
    selected = select_worst(calibration)
    profiles = [derive_profile(row, args.security_lambda) for row in selected]

    rng = np.random.Generator(np.random.PCG64DXSM(args.seed))
    mc_rows = [
        simulate_n64(row, rng, args.mc_trials)
        for row in profiles
        if int(row["parties"]) == 64
    ]

    profile_path = args.out_dir / "uniform_flooding_profiles.csv"
    mc_path = args.out_dir / "uniform_flooding_mc_n64.csv"
    summary_path = args.out_dir / "uniform_flooding_summary.md"
    provenance_path = args.out_dir / "provenance.json"
    write_csv(profile_path, profiles)
    write_csv(mc_path, mc_rows)
    summary_path.write_text(
        render_summary(profiles, mc_rows, args.security_lambda, args.seed),
        encoding="utf-8",
    )

    provenance = {
        "schema": "mkf-uniform-flooding-feasibility-v1",
        "status": "CONDITIONAL_ARITHMETIC_FEASIBILITY_ONLY",
        "security_target_lambda": args.security_lambda,
        "mc_trials": args.mc_trials,
        "rng": "NumPy PCG64DXSM",
        "seed_hex": hex(args.seed),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "input": str(args.input.resolve()),
        "input_sha256": sha256(args.input),
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256(Path(__file__)),
        "outputs": {
            profile_path.name: sha256(profile_path),
            mc_path.name: sha256(mc_path),
            summary_path.name: sha256(summary_path),
        },
        "limitations": [
            "B_pre_obs is a three-trial observed maximum, not a proof-grade offset bound",
            "Monte Carlo does not test 2^-128 statistical distance",
            "continuous normalized uniform approximates the discrete sampler",
            "deterministic nW and the algebraic SD upper bound are the authoritative calculations",
        ],
    }
    provenance_path.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
