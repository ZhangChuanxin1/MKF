#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]] || [[ $# -gt 2 ]]; then
  echo "Usage: $0 /absolute/path/to/MKHE-KKLSS [/absolute/path/to/output]" >&2
  exit 2
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
package_dir="$(cd "$script_dir/.." && pwd)"
source_dir="$1"
go_bin="${GO_BIN:-go}"
if ! command -v "$go_bin" >/dev/null 2>&1; then
  echo "Go not found. Put go in PATH or set GO_BIN=/absolute/path/to/go." >&2
  exit 1
fi
if [[ $# -eq 2 ]]; then
  output_dir="$2"
else
  output_dir="$package_dir/runs/$(date +%Y%m%d_%H%M%S)"
fi

if [[ ! -f "$source_dir/stageprof/stageprof.go" ]]; then
  echo "Instrumentation is not installed. Run install_into_kklss.sh first." >&2
  exit 1
fi

mkdir -p "$output_dir"
output_dir="$(cd "$output_dir" && pwd)"

(
  cd "$source_dir"

  echo "[0/5] Resolving upstream Go dependencies"
  "$go_bin" mod download

  echo "[1/5] Running upstream correctness tests"
  GOMAXPROCS=1 "$go_bin" test ./mkckks ./mkbfv -run '^(TestCKKS|TestMKBFV)$' -count=1

  echo "[2/5] Running CKKS end-to-end cross-check"
  GOMAXPROCS=1 \
  KKLSS_TOTAL_OUT="$output_dir/kklss_ckks_p1_n16_total_raw.csv" \
    "$go_bin" test ./mkckks -run '^TestProfileExactP1N16$' -count=1 -v

  echo "[3/5] Running BFV end-to-end cross-check"
  GOMAXPROCS=1 \
  KKLSS_TOTAL_OUT="$output_dir/kklss_bfv_p1_n16_total_raw.csv" \
    "$go_bin" test ./mkbfv -run '^TestProfileExactP1N16$' -count=1 -v

  echo "[4/5] Running CKKS stage profile"
  GOMAXPROCS=1 \
  KKLSS_STAGE_OUT="$output_dir/kklss_ckks_p1_n16_stage_raw.csv" \
    "$go_bin" test ./mkckks -run '^TestProfileStagesExactP1N16$' -count=1 -v

  echo "[5/5] Running BFV stage profile"
  GOMAXPROCS=1 \
  KKLSS_STAGE_OUT="$output_dir/kklss_bfv_p1_n16_stage_raw.csv" \
    "$go_bin" test ./mkbfv -run '^TestProfileStagesExactP1N16$' -count=1 -v

  {
    date -u '+utc_time=%Y-%m-%dT%H:%M:%SZ'
    uname -a
    "$go_bin" version
    "$go_bin" env GOOS GOARCH
    printf 'GOMAXPROCS=1\n'
    printf 'source_dir=%s\n' "$source_dir"
  } > "$output_dir/environment.txt"
)

python3 "$script_dir/summarize_totals.py" \
  --ckks "$output_dir/kklss_ckks_p1_n16_total_raw.csv" \
  --bfv "$output_dir/kklss_bfv_p1_n16_total_raw.csv" \
  --out "$output_dir/kklss_p1_n16_total_summary.csv"

python3 "$script_dir/summarize_stages.py" \
  --ckks "$output_dir/kklss_ckks_p1_n16_stage_raw.csv" \
  --bfv "$output_dir/kklss_bfv_p1_n16_stage_raw.csv" \
  --out "$output_dir/kklss_p1_n16_stage_summary.csv"

echo "Results written to: $output_dir"
