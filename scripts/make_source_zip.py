#!/usr/bin/env python3
"""Create a new reviewed-source ZIP outside the artifact, without local outputs."""
import argparse
from pathlib import Path
import zipfile

import publication_scan
import verify_project_layout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    destination = args.output.resolve()
    if destination == root or root in destination.parents:
        raise SystemExit('ZIP must be outside the artifact directory')
    checks, failures, files = publication_scan.scan(root)
    failures += publication_scan.verify_manifest(root, files)
    failures += [{'name': 'project-layout', 'detail': detail}
                 for detail in verify_project_layout.audit(root)]
    if failures:
        raise SystemExit(f'refusing ZIP: {len(failures)} publication/manifest failures')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, mode='x', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, path.relative_to(root).as_posix())
    print(f'Created {destination.name}: {len(files)} reviewed files, local outputs excluded')


if __name__ == '__main__':
    main()
