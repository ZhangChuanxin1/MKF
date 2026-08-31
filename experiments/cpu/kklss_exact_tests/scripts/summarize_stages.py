#!/usr/bin/env python3
"""Validate and summarize KKLSS mutually exclusive stage profiles."""

import argparse
import csv
import statistics
from pathlib import Path


STAGES = (
    ("Tensor-product generation", "tensor_ns"),
    ("Gadget decomposition", "decomposition_ns"),
    ("External-product accumulation", "external_product_ns"),
    ("modulus", "modulus_management_ns"),
    ("Memory and data movement", "memory_movement_ns"),
    ("Other operations", "other_ns"),
    ("Total latency", "total_ns"),
)
COMPONENT_COLUMNS = tuple(column for _, column in STAGES[:-1])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckks", required=True, type=Path)
    parser.add_argument("--bfv", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    return parser.parse_args()


def read_and_validate(path, expected_scheme):
    with path.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    if len(rows) != 10:
        raise ValueError(f"{path}: expected 10 trials, found {len(rows)}")
    trials = [int(row["trial"]) for row in rows]
    if sorted(trials) != list(range(10)):
        raise ValueError(f"{path}: trials must be exactly 0 through 9")
    for row in rows:
        if row["scheme"] != expected_scheme:
            raise ValueError(f"{path}: unexpected scheme {row['scheme']}")
        if row["param"] != "P1" or row["n"] != "16":
            raise ValueError(f"{path}: expected P1 and n=16")
        values = [int(row[column]) for column in COMPONENT_COLUMNS]
        total = int(row["total_ns"])
        if any(value < 0 for value in values) or total < 0:
            raise ValueError(f"{path}: negative timing in trial {row['trial']}")
        if sum(values) != total:
            raise ValueError(
                f"{path}: stages do not sum to total in trial {row['trial']}"
            )
    return rows


def stage_label(scheme, generic_label):
    if generic_label != "modulus":
        return generic_label
    if scheme == "KKLSS-CKKS":
        return "ModUp/ModDown and rescaling"
    return "ModUp/ModDown and scale-and-round"


def summarize(rows):
    scheme = rows[0]["scheme"]
    output = []
    for generic_label, column in STAGES:
        values = [int(row[column]) / 1_000_000 for row in rows]
        output.append(
            {
                "scheme": scheme,
                "parameter": "P1",
                "parties": "16",
                "samples": "10",
                "warmup": "5",
                "stage": stage_label(scheme, generic_label),
                "mean_ms": f"{statistics.fmean(values):.6f}",
                "median_ms": f"{statistics.median(values):.6f}",
                "stddev_ms": f"{statistics.stdev(values):.6f}",
                "min_ms": f"{min(values):.6f}",
                "max_ms": f"{max(values):.6f}",
            }
        )
    return output


def main():
    args = parse_args()
    rows = []
    rows.extend(summarize(read_and_validate(args.ckks, "KKLSS-CKKS")))
    rows.extend(summarize(read_and_validate(args.bfv, "KKLSS-BFV")))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        "scheme", "parameter", "parties", "samples", "warmup", "stage",
        "mean_ms", "median_ms", "stddev_ms", "min_ms", "max_ms",
    )
    with args.out.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"validated 20 trials; wrote {args.out}")


if __name__ == "__main__":
    main()
