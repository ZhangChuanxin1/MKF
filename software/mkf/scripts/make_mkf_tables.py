#!/usr/bin/env python3
"""Create compact full-matrix and P1/n=16 paper-table CSVs."""

import argparse
import csv


def load(path):
    with open(path, newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def totals(rows):
    return {
        (row["scheme"], row["parameter"], int(row["parties"])): row
        for row in rows
        if row["stage"] == "Total latency"
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("p1_summary")
    parser.add_argument("p2_summary")
    parser.add_argument("matrix_output")
    parser.add_argument("breakdown_output")
    args = parser.parse_args()

    rows = load(args.p1_summary) + load(args.p2_summary)
    total = totals(rows)
    with open(args.matrix_output, "w", newline="", encoding="utf-8") as target:
        fields = [
            "parameter", "parties", "mkf_ckks_mean_ms", "mkf_bfv_mean_ms",
            "mkf_ckks_median_ms", "mkf_bfv_median_ms",
            "ckks_max_error", "bfv_exact", "decomposition_count",
            "external_product_poly_count",
        ]
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        for parameter in ("P1", "P2"):
            for parties in (2, 4, 8, 16, 32, 64):
                ckks = total[("MKF-CKKS", parameter, parties)]
                bfv = total[("MKF-BFV", parameter, parties)]
                raw_name = (
                    args.p1_summary.replace("_summary.csv", "_raw.csv")
                    if parameter == "P1"
                    else args.p2_summary.replace("_summary.csv", "_raw.csv")
                )
                raw = [r for r in load(raw_name) if r["scheme"] == "MKF-CKKS" and int(r["parties"]) == parties]
                writer.writerow({
                    "parameter": parameter,
                    "parties": parties,
                    "mkf_ckks_mean_ms": ckks["mean_ms"],
                    "mkf_bfv_mean_ms": bfv["mean_ms"],
                    "mkf_ckks_median_ms": ckks["median_ms"],
                    "mkf_bfv_median_ms": bfv["median_ms"],
                    "ckks_max_error": max(float(r["max_error"]) for r in raw),
                    "bfv_exact": bfv["all_valid"],
                    "decomposition_count": ckks["decomposition_count"],
                    "external_product_poly_count": ckks["external_product_poly_count"],
                })

    names = [
        "Tensor-product generation", "Gadget decomposition",
        "External-product accumulation", "ModUp/ModDown and rescaling",
        "Memory and data movement", "Other operations", "Total latency",
    ]
    selected = {
        (r["scheme"], r["stage"]): r
        for r in rows
        if r["parameter"] == "P1" and r["parties"] == "16"
    }
    with open(args.breakdown_output, "w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=["stage", "mkf_ckks_cpu_st_mean_ms", "mkf_bfv_cpu_st_mean_ms"])
        writer.writeheader()
        for stage in names:
            writer.writerow({
                "stage": stage,
                "mkf_ckks_cpu_st_mean_ms": selected[("MKF-CKKS", stage)]["mean_ms"],
                "mkf_bfv_cpu_st_mean_ms": selected[("MKF-BFV", stage)]["mean_ms"],
            })


if __name__ == "__main__":
    main()
