#!/usr/bin/env python3
"""Merge raw benchmark CSV files with identical headers."""

import argparse
import csv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output")
    parser.add_argument("inputs", nargs="+")
    args = parser.parse_args()

    fieldnames = None
    rows = []
    for path in args.inputs:
        with open(path, newline="", encoding="utf-8") as source:
            reader = csv.DictReader(source)
            if fieldnames is None:
                fieldnames = reader.fieldnames
            elif reader.fieldnames != fieldnames:
                raise SystemExit(f"header mismatch: {path}")
            rows.extend(reader)

    with open(args.output, "w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
