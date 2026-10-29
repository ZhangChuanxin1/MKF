#!/usr/bin/env python3
"""Check the declared source scaffold without executing an experiment."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATUSES = {'SOURCE_AVAILABLE', 'PARTIAL', 'PENDING_IMPLEMENTATION', 'REQUIRES_EXTERNAL_ASSETS'}
REQUIRED = {
    'experiments/cpu/correctness', 'experiments/cpu/latency',
    'experiments/cpu/stage_breakdown', 'experiments/cpu/noise_depth',
    'experiments/fpga/primitives', 'experiments/fpga/resources', 'experiments/fpga/rotation',
    'experiments/cnn/mnist/kklss', 'experiments/cnn/mnist/mkf',
    'experiments/cnn/resnet20/plaintext', 'experiments/cnn/resnet20/cpu',
    'experiments/cnn/resnet20/fpga', 'hardware/u55c/host',
    'hardware/u55c/connectivity', 'hardware/u55c/integration', 'hardware/u55c/validation',
    'hardware/u55c/cnn/mnist', 'hardware/u55c/cnn/resnet20',
}


def audit(root: Path = ROOT, config: dict | None = None) -> list[str]:
    root = root.resolve()
    config = config if config is not None else json.loads(
        (root/'config/experiment_layout.json').read_text(encoding='utf-8'))
    problems = []
    entries = config.get('entries', [])
    ids, paths = set(), set()

    def resolve(relative: str) -> Path | None:
        path = (root/relative).resolve()
        if path == root or root not in path.parents:
            problems.append(f'path outside artifact: {relative}')
            return None
        return path

    for entry in entries:
        name, relative = entry['id'], entry['path']
        if name in ids or relative in paths:
            problems.append(f'duplicate component: {name}')
        ids.add(name)
        paths.add(relative)
        state = entry['status']
        if state not in STATUSES:
            problems.append(f'unknown status: {name}')
        path = resolve(relative)
        if path is None:
            continue
        readme = path/'README.md'
        if not readme.is_file():
            problems.append(f'missing component README: {relative}')
        elif f'Status: `{state}`.' not in readme.read_text(encoding='utf-8'):
            problems.append(f'README/status mismatch: {relative}')
        refs = entry.get('source_refs', [])
        for ref in refs:
            source = resolve(ref)
            if source is not None and not source.exists():
                problems.append(f'missing source reference: {ref}')
        if state == 'SOURCE_AVAILABLE' and not refs:
            problems.append(f'source status without references: {relative}')
        if state == 'PENDING_IMPLEMENTATION':
            if refs:
                problems.append(f'pending component has implementation references: {relative}')
            if path.is_dir():
                non_docs = [p for p in path.rglob('*') if p.is_file() and p.suffix.lower() not in {'.md', '.json'}]
                if non_docs:
                    problems.append(f'pending component contains non-documentation files: {relative}')
    for relative in sorted(REQUIRED-paths):
        problems.append(f'required component absent: {relative}')
    return problems


def main() -> int:
    config = json.loads((ROOT/'config/experiment_layout.json').read_text(encoding='utf-8'))
    problems = audit(ROOT, config)
    counts = {state: sum(e['status'] == state for e in config['entries']) for state in sorted(STATUSES)}
    print(json.dumps({'status': 'FAIL' if problems else 'PASS', 'components': len(config['entries']),
                      'component_status_counts': counts, 'failures': problems}, indent=2))
    return int(bool(problems))


if __name__ == '__main__':
    raise SystemExit(main())
