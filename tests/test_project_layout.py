import copy
import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('project_layout', ROOT/'scripts/verify_project_layout.py')
layout = importlib.util.module_from_spec(spec)
spec.loader.exec_module(layout)


class LayoutTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((ROOT/'config/experiment_layout.json').read_text(encoding='utf-8'))

    def test_declared_layout(self):
        self.assertEqual(layout.audit(ROOT, self.config), [])

    def test_project_identity_and_byte_policy(self):
        project = json.loads((ROOT/'config/project.json').read_text(encoding='utf-8'))
        self.assertEqual(project['name'], 'MKF')
        self.assertTrue((ROOT/'README.md').read_text(encoding='utf-8').startswith('# MKF\n'))
        self.assertIn('* -text', (ROOT/'.gitattributes').read_text(encoding='utf-8'))

    def test_missing_reference_rejected(self):
        config = copy.deepcopy(self.config)
        config['entries'][0]['source_refs'].append('software/does_not_exist.go')
        self.assertTrue(any('missing source reference' in e for e in layout.audit(ROOT, config)))

    def test_escape_rejected(self):
        config = copy.deepcopy(self.config)
        config['entries'][0]['source_refs'].append('../outside_source.go')
        self.assertTrue(any('path outside artifact' in e for e in layout.audit(ROOT, config)))

    def test_unknown_status_rejected(self):
        config = copy.deepcopy(self.config)
        config['entries'][0]['status'] = 'UNSUPPORTED_SUCCESS_CLAIM'
        self.assertTrue(any('unknown status' in e for e in layout.audit(ROOT, config)))


if __name__ == '__main__':
    unittest.main()
