#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 /absolute/path/to/MKHE-KKLSS" >&2
  exit 2
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
package_dir="$(cd "$script_dir/.." && pwd)"
source_dir="$1"
go_bin="${GO_BIN:-go}"

if ! command -v patch >/dev/null 2>&1; then
  echo "Required command not found: patch" >&2
  exit 1
fi
if ! command -v "$go_bin" >/dev/null 2>&1; then
  echo "Go not found. Put go in PATH or set GO_BIN=/absolute/path/to/go." >&2
  exit 1
fi
go_path="$(command -v "$go_bin")"
gofmt_bin="${GOFMT_BIN:-$(dirname "$go_path")/gofmt}"
if [[ ! -x "$gofmt_bin" ]]; then
  echo "gofmt not found. Set GOFMT_BIN=/absolute/path/to/gofmt." >&2
  exit 1
fi

if [[ ! -f "$source_dir/go.mod" ]] || ! grep -q '^module mk-lattigo$' "$source_dir/go.mod"; then
  echo "Not a compatible MKHE-KKLSS root: $source_dir" >&2
  exit 1
fi

for required in \
  mkckks/evaluator.go \
  mkbfv/evaluator.go \
  mkrlwe/keyswitch.go; do
  if [[ ! -f "$source_dir/$required" ]]; then
    echo "Missing upstream file: $source_dir/$required" >&2
    exit 1
  fi
done

for target in \
  stageprof/stageprof.go \
  mkckks/profile_exact_test.go \
  mkckks/profile_stages_exact_test.go \
  mkbfv/profile_exact_test.go \
  mkbfv/profile_stages_exact_test.go; do
  if [[ -e "$source_dir/$target" ]]; then
    echo "Refusing to overwrite existing file: $source_dir/$target" >&2
    exit 1
  fi
done

patch_file="$package_dir/patches/kklss_stage_instrumentation.patch"
patch --dry-run -p1 -d "$source_dir" < "$patch_file"
patch -p1 -d "$source_dir" < "$patch_file"

mkdir -p "$source_dir/stageprof"
cp "$package_dir/stageprof/stageprof.go" "$source_dir/stageprof/stageprof.go"
cp "$package_dir/mkckks/profile_exact_test.go" "$source_dir/mkckks/profile_exact_test.go"
cp "$package_dir/mkckks/profile_stages_exact_test.go" "$source_dir/mkckks/profile_stages_exact_test.go"
cp "$package_dir/mkbfv/profile_exact_test.go" "$source_dir/mkbfv/profile_exact_test.go"
cp "$package_dir/mkbfv/profile_stages_exact_test.go" "$source_dir/mkbfv/profile_stages_exact_test.go"

"$gofmt_bin" -w \
  "$source_dir/stageprof/stageprof.go" \
  "$source_dir/mkckks/evaluator.go" \
  "$source_dir/mkckks/profile_exact_test.go" \
  "$source_dir/mkckks/profile_stages_exact_test.go" \
  "$source_dir/mkrlwe/keyswitch.go" \
  "$source_dir/mkrlwe/keyswitch_hoisted.go" \
  "$source_dir/mkbfv/basis_extension.go" \
  "$source_dir/mkbfv/evaluator.go" \
  "$source_dir/mkbfv/keyswitch.go" \
  "$source_dir/mkbfv/keyswitch_hoisted.go" \
  "$source_dir/mkbfv/profile_exact_test.go" \
  "$source_dir/mkbfv/profile_stages_exact_test.go"

echo "KKLSS profiling files installed in: $source_dir"
