#!/usr/bin/env bash
set -euo pipefail

# Paper-facing ResNet-20 CPU measurement launcher. This script deliberately
# fails in the Go harness until both audited KKLSS-CPU and MKF-CPU backends are
# registered. It never falls back to the functional executor or inferred data.

if [[ $# -ne 2 ]]; then
  echo "usage: $0 LOGICAL_CPU OUTPUT_DIRECTORY" >&2
  exit 2
fi

core="$1"
output_directory="$2"
case "$core" in
  ''|*[!0-9]*) echo "LOGICAL_CPU must be one non-negative integer" >&2; exit 2 ;;
esac

script_directory="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
module_root="$(cd -- "$script_directory/.." && pwd)"
kklss_root="$(cd -- "$module_root/../kklss" && pwd)"

mkdir -p "$output_directory"
output_directory="$(cd -- "$output_directory" && pwd)"
cd -- "$module_root"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
raw="$output_directory/resnet20_cpu_raw_${timestamp}.csv"
summary="$output_directory/resnet20_cpu_summary_${timestamp}.json"
log="$output_directory/resnet20_cpu_${timestamp}.log"
environment="$output_directory/resnet20_cpu_environment_${timestamp}.txt"
source_manifest="$output_directory/resnet20_cpu_source_manifest_${timestamp}.sha256"
asset_manifest="$output_directory/resnet20_cpu_asset_manifest_${timestamp}.sha256"
checksums="$output_directory/resnet20_cpu_result_checksums_${timestamp}.sha256"

: "${MKF_RESNET20_ACTIVATION_SCALES_JSON:?must name the frozen activation-scale JSON}"
: "${MKF_RESNET20_CIFAR_FIXTURE_DIR:?must name the frozen CIFAR fixture directory}"
: "${MKF_RESNET20_WEIGHTS_DIR:?must name the frozen checkpoint-derived weight directory}"
: "${MKF_RESNET20_FC_BIAS_PATH:?must name the frozen checkpoint FC-bias file}"

{
  sha256sum go.mod go.sum "scripts/$(basename -- "${BASH_SOURCE[0]}")"
  find mkf_cpu -maxdepth 1 -type f -name '*.go' -print0 | LC_ALL=C sort -z | xargs -0 sha256sum
  find "$kklss_root" -type f \( -name '*.go' -o -name 'go.mod' -o -name 'go.sum' \) -print0 | LC_ALL=C sort -z | xargs -0 sha256sum
} >"$source_manifest"

hash_labeled_file() {
  local path="$1"
  local label="$2"
  printf '%s  %s\n' "$(sha256sum "$path" | awk '{print $1}')" "$label"
}

{
  hash_labeled_file "$MKF_RESNET20_ACTIVATION_SCALES_JSON" "activation_scales/$(basename -- "$MKF_RESNET20_ACTIVATION_SCALES_JSON")"
  hash_labeled_file "$MKF_RESNET20_FC_BIAS_PATH" "checkpoint_bias/$(basename -- "$MKF_RESNET20_FC_BIAS_PATH")"
  while IFS= read -r -d '' relative; do
    hash_labeled_file "$MKF_RESNET20_CIFAR_FIXTURE_DIR/$relative" "cifar_fixture/$relative"
  done < <(cd -- "$MKF_RESNET20_CIFAR_FIXTURE_DIR" && find . -type f -printf '%P\0' | LC_ALL=C sort -z)
  while IFS= read -r -d '' relative; do
    hash_labeled_file "$MKF_RESNET20_WEIGHTS_DIR/$relative" "weights/$relative"
  done < <(cd -- "$MKF_RESNET20_WEIGHTS_DIR" && find . -type f -name '*.bin' -printf '%P\0' | LC_ALL=C sort -z)
} >"$asset_manifest"

source_manifest_sha256="$(sha256sum "$source_manifest" | awk '{print $1}')"
asset_manifest_sha256="$(sha256sum "$asset_manifest" | awk '{print $1}')"
parameter_source_sha256="$(sha256sum mkf_cpu/params.go | awk '{print $1}')"
cpu_model="$(grep -m1 '^model name' /proc/cpuinfo | cut -d: -f2- | sed 's/^[[:space:]]*//')"
lattigo_module="$(go list -m -f '{{.Path}} {{.Version}}' github.com/ldsec/lattigo/v2)"

test -n "$cpu_model"
test -n "$lattigo_module"

{
  uname -a
  echo "cpu_model=$cpu_model"
  go version
  echo "lattigo_module=$lattigo_module"
  echo "logical_cpu=$core"
  echo "gomaxprocs=1"
  echo "warmups=0"
  echo "measured_trials=10"
  echo "raw_csv=$raw"
  echo "summary_json=$summary"
  echo "source_manifest=$source_manifest"
  echo "source_manifest_sha256=$source_manifest_sha256"
  echo "asset_manifest=$asset_manifest"
  echo "asset_manifest_sha256=$asset_manifest_sha256"
  echo "parameter_source_sha256=$parameter_source_sha256"
} >"$environment"

taskset -c "$core" env \
  GOMAXPROCS=1 \
  MKF_RESNET20_CPU_MODEL="$cpu_model" \
  MKF_RESNET20_LATTIGO_MODULE="$lattigo_module" \
  MKF_RESNET20_SOURCE_MANIFEST_SHA256="$source_manifest_sha256" \
  MKF_RESNET20_ASSET_MANIFEST_SHA256="$asset_manifest_sha256" \
  MKF_RESNET20_PARAMETER_SOURCE_SHA256="$parameter_source_sha256" \
  MKF_RUN_FULL_RESNET20_CPU_BENCHMARK=1 \
  MKF_RESNET20_CPU_RAW_CSV="$raw" \
  MKF_RESNET20_CPU_SUMMARY_JSON="$summary" \
  go test ./mkf_cpu \
    -run '^TestFullResNet20CPUBenchmark$' \
    -count=1 -timeout=0 -v 2>&1 | tee "$log"

test -s "$raw"
test -s "$summary"
sha256sum "$raw" "$summary" "$log" "$environment" "$source_manifest" "$asset_manifest" >"$checksums"
echo "RESNET20_CPU_BENCHMARK_PASS raw=$raw summary=$summary log=$log environment=$environment source_manifest=$source_manifest asset_manifest=$asset_manifest checksums=$checksums"
