# Local packaging validation — 2026-10-28

This records checks actually performed during assembly, not new manuscript
performance measurements. Local outputs live in ignored `.local/` folders.

## Completed

- Publication content/path scan: PASS; no forbidden platform identity,
  personal path, matched token/key pattern or oversized tracked file found.
  This is a bounded scanner, not proof of anonymity or absence of every secret.
- Source manifest generation and verification: PASS; every published source
  is hashed. Unlisted source files and changed source bytes fail verification.
- Packaging/IDX converter unit tests: PASS. Tests cover IDX validation,
  CSV format, overwrite refusal, manifest edits/unlisted files, local-data
  exclusions, source layout, U55C configuration and arithmetic wrapper identity.
- PowerShell parser: PASS for all packaged `.ps1` files.
- Go 1.23.4, Windows/amd64: compile-only tests for both complete modules PASS.
- Lightweight `TestP2GN16LattigoLevelSemanticsNoCrypto` and
  `TestP2GN16MatchedCPUContractNoFullRun`: PASS. These are not full-network runs.
- Both plotting scripts executed in separate ignored output folders; each
  produced PDF/PNG/SVG and a CSV byte-identical to its packaged input CSV.
  Generated PNGs were visually inspected; original plot style was preserved.

Plot replay used the local isolated plotting environment. Optional requirement
files specify compatibility ranges, not a claim of bit-identical rendering on
all Python/Matplotlib/font installations. Go dependencies retain their exact
module versions/checksums.

## Not performed / not established

- Full correctness/benchmark/noise matrices, MNIST inference, full encrypted
  ResNet-20, security-estimator execution and new accuracy measurements.
- Vitis/Vivado HLS C-sim, synthesis, RTL simulation/route, XRT integration or
  board measurements; Vitis HLS/Vivado were not available on this host.
- Selection of an original-source license, legal redistribution review of
  external datasets/models, or end-to-end double-blind anonymity certification.
- No Git commit, remote push, history rewrite or board operation was performed.

HLS/RTL wrappers are U55C-configured sources, not successful hardware results.
The local checks do not close the construction/evidence gaps in the catalog.
