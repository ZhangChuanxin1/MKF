#!/usr/bin/env python3
"""Summarize end-to-end MKF CPU samples and mutually exclusive stages."""

import argparse
import csv
import statistics


STAGES = {
    "Tensor-product generation": "tensor_ns",
    "Gadget decomposition": "decomposition_ns",
    "External-product accumulation": "external_product_ns",
    "ModUp/ModDown and rescaling": "modup_moddown_rescale_ns",
    "Memory and data movement": "memory_copy_ns",
    "Other operations": "other_ns",
    "Total latency": "total_ns",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")
    args = parser.parse_args()

    with open(args.input, newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))

    groups = {}
    for row in rows:
        groups.setdefault(
            (row["scheme"], row["parameter"], row["parties"]), []
        ).append(row)

    fields = [
        "scheme", "parameter", "parties", "samples", "stage",
        "mean_ms", "median_ms", "stddev_ms", "min_ms", "max_ms",
        "all_valid", "decomposition_count", "external_product_poly_count",
    ]
    with open(args.output, "w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        for key in sorted(groups):
            samples = groups[key]
            for stage, column in STAGES.items():
                values = [float(row[column]) / 1e6 for row in samples]
                writer.writerow({
                    "scheme": key[0],
                    "parameter": key[1],
                    "parties": key[2],
                    "samples": len(values),
                    "stage": stage,
                    "mean_ms": f"{statistics.fmean(values):.6f}",
                    "median_ms": f"{statistics.median(values):.6f}",
                    "stddev_ms": f"{statistics.stdev(values) if len(values) > 1 else 0:.6f}",
                    "min_ms": f"{min(values):.6f}",
                    "max_ms": f"{max(values):.6f}",
                    "all_valid": all(row["validation_passed"] == "true" for row in samples),
                    "decomposition_count": samples[0]["decomposition_count"],
                    "external_product_poly_count": samples[0]["external_product_poly_count"],
                })


if __name__ == "__main__":
    main()
