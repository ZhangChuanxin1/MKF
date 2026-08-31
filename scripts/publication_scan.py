#!/usr/bin/env python3
"""Fail-closed publication scan for the U55C artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path


EXPECTED_PART = "xcu55c-fsvh2892-2L-e"
EXPECTED_PLATFORM = "xilinx_u55c_gen3x16_xdma_3_202210_1"
EXPECTED_TARGETS = ["kklss_ckks", "mkf_ckks", "kklss_bfv", "mkf_bfv"]
MAX_TRACKABLE_BYTES = 90 * 1024 * 1024


def joined(*pieces: str) -> str:
    return "".join(pieces)


FORBIDDEN_BYTES = {
    "unsupported-device-short": joined("v", "80").encode(),
    "unsupported-device-part": joined("xc", "v", "80").encode(),
    "unsupported-device-family": joined("vc", "1902").encode(),
    "unsupported-device-full": joined("xc", "vc", "1902").encode(),
    "private-user-short": joined("z", "cx").encode(),
    "private-user-name": joined("chuan", "xin").encode(),
    "private-user-surname": joined("zh", "ang").encode(),
    "private-home-path": joined("/", "home", "/").encode(),
    "windows-user-path": joined(":\\", "users", "\\").encode(),
    "privileged-board-tool": joined("hot", "boot").encode(),
    "direct-register-token": joined("bar", "2").encode(),
}

BLOCKED_SUFFIXES = {
    ".bit", ".dcp", ".jou", ".log", ".pdi", ".rpt", ".str", ".vcd",
    ".wdb", ".xclbin", ".xo", ".xsa",
}
BLOCKED_DIRECTORY_NAMES = {".xil", "_x", "build", "results", "runs"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        parts = {part.lower() for part in relative.parts}
        if parts.intersection({".git", ".local", ".pytest_cache", ".venv", "__pycache__"}):
            continue
        if path.is_file():
            files.append(path)
    return sorted(files, key=lambda item: item.as_posix())


def scan(root: Path) -> tuple[list[dict[str, object]], list[dict[str, str]], list[Path]]:
    checks: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    files = source_files(root)

    def check(name: str, ok: bool, detail: str) -> None:
        checks.append({"name": name, "ok": bool(ok), "detail": detail})
        if not ok:
            failures.append({"name": name, "detail": detail})

    required = [
        root / "README.md",
        root / "config" / "u55c.json",
        root / "hardware" / "u55c" / "hls" / "run_hls.tcl",
        root / "scripts" / "run_suite.ps1",
        root / "software" / "mkf" / "go.mod",
        root / "software" / "kklss" / "go.mod",
        root / "software" / "kklss" / "LICENSE",
    ]
    for path in required:
        check(f"required:{path.relative_to(root).as_posix()}", path.is_file(), "present" if path.is_file() else "missing")

    nested_git = [path for path in root.rglob(".git") if path != root / ".git"]
    check("no_nested_git", not nested_git, ", ".join(str(path.relative_to(root)) for path in nested_git) or "none")

    for path in files:
        relative = path.relative_to(root)
        rel_lower = relative.as_posix().lower()
        for label, token in FORBIDDEN_BYTES.items():
            check(f"path:{label}:{relative.as_posix()}", token.decode() not in rel_lower, rel_lower)
        blocked_parts = sorted(BLOCKED_DIRECTORY_NAMES.intersection(part.lower() for part in relative.parts[:-1]))
        check(f"path:blocked-directory:{relative.as_posix()}", not blocked_parts, ",".join(blocked_parts) or "none")
        check(f"path:blocked-suffix:{relative.as_posix()}", path.suffix.lower() not in BLOCKED_SUFFIXES, path.suffix.lower() or "none")
        check(f"path:size:{relative.as_posix()}", path.stat().st_size <= MAX_TRACKABLE_BYTES, str(path.stat().st_size))

        raw = path.read_bytes()
        lowered = raw.lower()
        for label, token in FORBIDDEN_BYTES.items():
            index = lowered.find(token)
            check(f"content:{label}:{relative.as_posix()}", index < 0, "none" if index < 0 else f"byte_offset={index}")
        secret_patterns = {
            "github-token": re.compile(rb"gh[pousr]_[A-Za-z0-9_]{20,}"),
            "private-key": re.compile(joined("BEGIN ", "PRIVATE ", "KEY").encode()),
            "credential-url": re.compile(rb"https?://[^/\s:@]+:[^/\s@]+@"),
        }
        for label, pattern in secret_patterns.items():
            found = pattern.search(raw)
            check(f"secret:{label}:{relative.as_posix()}", found is None, "none" if found is None else f"byte_offset={found.start()}")

    try:
        config = json.loads((root / "config" / "u55c.json").read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - fail closed with exact parse error
        config = {}
        check("config_parse", False, str(exc))
    else:
        check("config_parse", True, "valid JSON")
    check("config_exact_part", config.get("device_part") == EXPECTED_PART, str(config.get("device_part")))
    check("config_exact_platform", config.get("platform_name") == EXPECTED_PLATFORM, str(config.get("platform_name")))
    check("config_exact_targets", config.get("hls_targets") == EXPECTED_TARGETS, repr(config.get("hls_targets")))

    tcl_path = root / "hardware" / "u55c" / "hls" / "run_hls.tcl"
    tcl = tcl_path.read_text(encoding="utf-8") if tcl_path.is_file() else ""
    set_parts = re.findall(r"set_part\s+\{([^}]+)\}", tcl)
    check("hls_one_fixed_part", set_parts == [EXPECTED_PART], repr(set_parts))
    check("hls_relative_source_root", "[file dirname [info script]]" in tcl, "self-contained path resolution")

    platform_pattern = re.compile(r"xilinx_[A-Za-z0-9_]+")
    observed_platforms: set[str] = set()
    for path in files:
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        observed_platforms.update(platform_pattern.findall(text))
    check("only_expected_platform", observed_platforms <= {EXPECTED_PLATFORM}, repr(sorted(observed_platforms)))

    return checks, failures, files


def write_manifest(root: Path, files: list[Path]) -> None:
    manifest = root / "PUBLICATION_MANIFEST.sha256"
    records = []
    for path in files:
        if path == manifest:
            continue
        records.append(f"{sha256(path)}  {path.relative_to(root).as_posix()}")
    manifest.write_text("\n".join(records) + "\n", encoding="utf-8", newline="\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--write-manifest", action="store_true")
    args = parser.parse_args()

    root = (args.root or Path(__file__).resolve().parent.parent).resolve()
    checks, failures, files = scan(root)
    report = {
        "schema": "mkf-u55c-publication-scan-v1",
        "status": "PASS" if not failures else "FAIL",
        "file_count": len(files),
        "checks": checks,
        "failures": failures,
    }
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered + "\n", encoding="utf-8", newline="\n")
    if args.write_manifest and not failures:
        write_manifest(root, files)
    print(f"publication scan: {report['status']} files={len(files)} failures={len(failures)}")
    if failures:
        for failure in failures[:50]:
            print(f"FAIL {failure['name']}: {failure['detail']}", file=sys.stderr)
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
