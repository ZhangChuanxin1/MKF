#!/usr/bin/env python3
"""Convert local MNIST test IDX files to the exact legacy CNN CSV format."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import struct
from pathlib import Path


def read_payload(path: Path) -> bytes:
    raw = path.read_bytes()
    return gzip.decompress(raw) if raw[:2] == b'\x1f\x8b' else raw


def decode(images: bytes, labels: bytes, expected_count: int = 10000) -> tuple[bytes, bytes]:
    if len(images) < 16 or len(labels) < 8:
        raise ValueError('truncated IDX header')
    magic, count, rows, columns = struct.unpack('>IIII', images[:16])
    label_magic, label_count = struct.unpack('>II', labels[:8])
    if (magic, label_magic) != (2051, 2049):
        raise ValueError('wrong IDX magic numbers')
    if (count, label_count, rows, columns) != (expected_count, expected_count, 28, 28):
        raise ValueError('expected matching test counts and 28x28 images')
    if len(images) != 16 + count * 784 or len(labels) != 8 + count:
        raise ValueError('IDX payload size mismatch')
    if any(label > 9 for label in labels[8:]):
        raise ValueError('MNIST labels must be in [0,9]')
    return images[16:], labels[8:]


def render_csv(pixels: bytes, labels: bytes) -> bytes:
    if len(pixels) != len(labels) * 784:
        raise ValueError('pixel/label count mismatch')
    # Legacy Go reader treats a trailing newline as an extra empty image.
    return '\n'.join(','.join(map(str, [label, *pixels[i*784:(i+1)*784]]))
                     for i, label in enumerate(labels)).encode('ascii')


def convert(images: Path, labels: Path, output: Path, expected_count: int = 10000) -> dict:
    metadata = output.with_name(output.name + '.metadata.json')
    if output.exists() or metadata.exists():
        raise FileExistsError('refusing to overwrite CSV or metadata')
    pixels, classes = decode(read_payload(images), read_payload(labels), expected_count)
    content = render_csv(pixels, classes)
    record = {
        'schema': 'mkf-mnist-idx-conversion-v1', 'count': len(classes),
        'rows': 28, 'columns': 28, 'normalization': 'raw uint8; Go reader divides by 255',
        'images_sha256': hashlib.sha256(images.read_bytes()).hexdigest(),
        'labels_sha256': hashlib.sha256(labels.read_bytes()).hexdigest(),
        'csv_sha256': hashlib.sha256(content).hexdigest(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('xb') as handle:
        handle.write(content)
    with metadata.open('x', encoding='utf-8', newline='\n') as handle:
        json.dump(record, handle, indent=2)
        handle.write('\n')
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images', required=True, type=Path)
    parser.add_argument('--labels', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(convert(args.images, args.labels, args.output), indent=2))


if __name__ == '__main__':
    main()
