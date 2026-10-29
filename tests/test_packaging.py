import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


idx = load('mnist_converter', ROOT / 'scripts/convert_mnist_idx.py')
scan = load('publication_scan', ROOT / 'scripts/publication_scan.py')


class PackagingTests(unittest.TestCase):
    def test_idx_format(self):
        images = struct.pack('>IIII', 2051, 2, 28, 28) + bytes([255]) * 1568
        labels = struct.pack('>II', 2049, 2) + bytes([3, 8])
        pixels, classes = idx.decode(images, labels, expected_count=2)
        csv = idx.render_csv(pixels, classes)
        self.assertEqual(len(csv.splitlines()), 2)
        self.assertEqual(len(csv.splitlines()[0].split(b',')), 785)
        self.assertTrue(csv.startswith(b'3,255,'))
        self.assertFalse(csv.endswith(b'\n'))

    def test_bad_idx_rejected(self):
        with self.assertRaises(ValueError):
            idx.decode(b'broken', b'broken')
        images = struct.pack('>IIII', 2051, 1, 28, 28) + bytes(784)
        with self.assertRaises(ValueError):
            idx.decode(images, struct.pack('>II', 2049, 1) + bytes([10]), 1)

    def test_converter_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            images, labels, output = base/'images', base/'labels', base/'mnist.csv'
            images.write_bytes(struct.pack('>IIII',2051,1,28,28)+bytes(784))
            labels.write_bytes(struct.pack('>II',2049,1)+bytes([1]))
            record = idx.convert(images, labels, output, expected_count=1)
            self.assertEqual(record['count'], 1)
            with self.assertRaises(FileExistsError):
                idx.convert(images, labels, output, expected_count=1)

    def test_manifest_detects_edit_and_extra_file(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = base/'source.txt'
            source.write_text('original', encoding='utf-8')
            scan.write_manifest(base, [source])
            self.assertFalse(scan.verify_manifest(base, scan.source_files(base)))
            source.write_text('changed', encoding='utf-8')
            self.assertTrue(scan.verify_manifest(base, scan.source_files(base)))
            (base/'extra.txt').write_text('extra', encoding='utf-8')
            self.assertTrue(any(item['name']=='manifest-unlisted' for item in scan.verify_manifest(base, scan.source_files(base))))

    def test_config_and_docs(self):
        config = json.loads((ROOT/'config/u55c.json').read_text(encoding='utf-8'))
        self.assertEqual(config['device_part'], 'xcu55c-fsvh2892-2L-e')
        for relative in ['README.md','README_zh.md','docs/RUNBOOK.md','docs/EXPERIMENT_CATALOG.md','datasets/README.md','figures/README.md']:
            self.assertTrue((ROOT/relative).is_file(), relative)

    def test_microkernel_tops_and_counter_clock(self):
        hardware = ROOT/'hardware/u55c'
        self.assertIn('void par1_ntt_kernel(', (hardware/'micro/par1_ntt_serial_kernel.cpp').read_text())
        tcl = (hardware/'micro/run_micro.tcl').read_text()
        self.assertIn('ntt_serial { set source par1_ntt_serial_kernel.cpp; set top par1_ntt_kernel }', tcl)
        counter = (hardware/'rtl/tifs_stage_counter.v').read_text()
        wrapper = (hardware/'rtl/run_rtl.tcl').read_text()
        self.assertIn('module tifs_stage_counter_axis', counter)
        self.assertIn('set module_top tifs_stage_counter_axis; set clock_port ap_clk', wrapper)

    def test_local_data_is_excluded_but_sources_are_not(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            excluded = ['fixtures/generated/payload.bin', 'datasets/local/data.bin',
                        'software/kklss/cnn/data/mnist_test.csv', '.local/private.log',
                        'figures/mkf/preview.png']
            included = ['software/kklss/cnn/data/k1.txt', 'figures/mkf/source.py']
            for name in excluded + included:
                path = base/name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('local test', encoding='utf-8')
            observed = {path.relative_to(base).as_posix() for path in scan.source_files(base)}
            self.assertEqual(observed, set(included))


if __name__ == '__main__':
    unittest.main()
