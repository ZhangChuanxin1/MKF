#!/usr/bin/env python3
"""Validate and summarize uninstrumented KKLSS end-to-end timings."""

import argparse
import csv
import statistics
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckks", required=True, type=Path)
    parser.add_argument("--bfv", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    return parser.parse_args()


def summarize(path, expected_scheme):
    with path.open(newline="", encoding="utf-8") as source:
        rows = [
            row for row in csv.DictReader(source)
            if row["scheme"] == expected_scheme
        ]
    if len(rows) != 10:
        raise ValueError(f"{path}: expected 10 trials, found {len(rows)}")
    if sorted(int(row["trial"]) for row in rows) != list(range(10)):
        raise ValueError(f"{path}: trials must be exactly 0 through 9")
    for row in rows:
        if row["scheme"] != expected_scheme:
            raise ValueError(f"{path}: unexpected scheme {row['scheme']}")
        if row["parameter"] != "P1" or row["parties"] != "16":
            raise ValueError(f"{path}: expected P1 and n=16")
    values = [int(row["elapsed_ns"]) / 1_000_000 for row in rows]
    if any(value < 0 for value in values):
        raise ValueError(f"{path}: negative timing")
    return {
        "scheme": expected_scheme,
        "parameter": "P1",
        "parties": "16",
        "samples": "10",
        "warmup": "5",
        "mean_ms": f"{statistics.fmean(values):.6f}",
        "median_ms": f"{statistics.median(values):.6f}",
        "stddev_ms": f"{statistics.stdev(values):.6f}",
        "min_ms": f"{min(values):.6f}",
        "max_ms": f"{max(values):.6f}",
    }


def main():
    args = parse_args()
    rows = [
        summarize(args.ckks, "KKLSS-CKKS"),
        summarize(args.bfv, "KKLSS-BFV"),
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        "scheme", "parameter", "parties", "samples", "warmup",
        "mean_ms", "median_ms", "stddev_ms", "min_ms", "max_ms",
    )
    with args.out.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"validated 20 trials; wrote {args.out}")


if __name__ == "__main__":
    main()
