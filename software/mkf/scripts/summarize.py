#!/usr/bin/env python3
"""Aggregate raw CPU benchmark samples into median/mean/stddev/p95 CSV."""

import argparse
import csv
import math
import statistics
from collections import defaultdict


def percentile(values, q):
    ordered = sorted(values)
    if not ordered:
        return math.nan
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="raw CSV from rnsbench or stageprof")
    parser.add_argument("output", help="summary CSV")
    args = parser.parse_args()

    groups = defaultdict(list)
    with open(args.input, newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        if "ns_per_op" in reader.fieldnames:
            value_column = "ns_per_op"
            key_columns = [
                "scheme", "parameter", "logN", "logQP", "q_count",
                "p_count", "operation",
            ]
        elif "elapsed_ns" in reader.fieldnames and "stage" in reader.fieldnames:
            value_column = "elapsed_ns"
            key_columns = ["scheme", "parameter", "parties", "stage"]
        else:
            raise SystemExit("unsupported CSV schema")

        for row in reader:
            groups[tuple(row[column] for column in key_columns)].append(
                float(row[value_column])
            )

    with open(args.output, "w", newline="", encoding="utf-8") as target:
        fieldnames = key_columns + [
            "samples", "median_ns", "mean_ns", "stddev_ns", "p95_ns"
        ]
        writer = csv.DictWriter(target, fieldnames=fieldnames)
        writer.writeheader()
        for key in sorted(groups):
            values = groups[key]
            row = dict(zip(key_columns, key))
            row.update(
                samples=len(values),
                median_ns=f"{statistics.median(values):.3f}",
                mean_ns=f"{statistics.fmean(values):.3f}",
                stddev_ns=f"{statistics.stdev(values) if len(values) > 1 else 0.0:.3f}",
                p95_ns=f"{percentile(values, 0.95):.3f}",
            )
            writer.writerow(row)


if __name__ == "__main__":
    main()
