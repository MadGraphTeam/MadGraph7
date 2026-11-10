"""Pure synthetic receipt checks and mutations of retained fixed-QP text."""
import copy
import importlib.util
import json
from pathlib import Path
import unittest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

assessment = load('assessment_under_test', HERE / 'assessment.py')
parser = load('frozen_journal_for_test', ROOT / 'docs/benchmarkfolder/experiments/adaptive-qp-checked-trap-proposal-1/parser.py')

class ReceiptTests(unittest.TestCase):
    def setUp(self):
        # These hashes are visibly synthetic test values, never DSO bindings.
        self.arguments = {'source_index_path': '/synthetic/index.json',
                          'source_index_sha256': 'b' * 64, 'build_command': ['synthetic-never-executed']}
        self.record = {'complete': True, 'loader_or_native_test_executed': False,
            'root_admission_received': True, 'cpu_exclusive': False, 'cpu': 11,
            'jobs': 1, 'automatic_retries': 0, 'overall_seconds': 1800,
            'address_space_limit_bytes_per_build_process': 8589934592,
            'address_space_limit_bytes_per_cache_copy_process': 4294967296,
            'commands': [{'label': label, 'command': ['synthetic-never-executed'],
                          'progress': {'process_reaped': True, 'returncode': 0}}
                         for label in ('copy-cache', 'release-build')],
            'protected_inputs_before': {'/synthetic/index.json': 'b' * 64},
            'protected_inputs_after': {'/synthetic/index.json': 'b' * 64},
            'elapsed_seconds': 1.0,
            'private_library_not_loaded': {'path': '/synthetic/not-real.so', 'sha256': 'a' * 64, 'bytes': 1}}

    def test_valid_synthetic_receipt_shape(self):
        self.assertEqual(assessment.accept_build_receipt(self.record, **self.arguments)['bytes'], 1)

    def test_invalid_receipts_rejected(self):
        for key, value in [('complete', False), ('loader_or_native_test_executed', True),
                           ('address_space_limit_bytes_per_build_process', 4294967296),
                           ('root_admission_received', False), ('cpu_exclusive', True),
                           ('cpu', 12), ('jobs', 2), ('automatic_retries', 1), ('overall_seconds', 900),
                           ('protected_inputs_after', {}), ('elapsed_seconds', 1800),
                           ('elapsed_seconds', float('nan')), ('elapsed_seconds', True),
                           ('private_library_not_loaded', {})]:
            with self.subTest(key=key, value=value):
                record = copy.deepcopy(self.record)
                record[key] = value
                with self.assertRaises(ValueError):
                    assessment.accept_build_receipt(record, **self.arguments)

    def test_child_or_command_change_rejected(self):
        for kind in ('unreaped', 'failed', 'argv'):
            with self.subTest(kind=kind):
                record = copy.deepcopy(self.record)
                if kind == 'argv':
                    record['commands'][1]['command'] = ['wrong']
                else:
                    record['commands'][1]['progress']['process_reaped' if kind == 'unreaped' else 'returncode'] = False if kind == 'unreaped' else 101
                with self.assertRaises(ValueError):
                    assessment.accept_build_receipt(record, **self.arguments)

class FixedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        prepared = json.loads((HERE / 'prepared.json').read_text())
        source = next(r['journal'] for r in prepared['comparators'] if r['mode'] == 'forced-qp')
        cls.reference = parser.parse(ROOT / source)

    def setUp(self):
        self.capture = copy.deepcopy(self.reference)
        self.controls = {i: {'H': 1, 'T': 2, 'U': 0, 'accuracy': [-1.0, -1.0]} for i in (1, 2)}

    def qualify(self, omissions=0):
        return assessment.qualify_fixed(self.capture, self.controls, parser, omissions=omissions)

    def test_closed_fixed_capture_is_unchecked_success(self):
        result = self.qualify()
        self.assertTrue(result['fixed_qp_outputs_qualified'])
        self.assertFalse(result['stability_checked'])
        self.assertFalse(result['per_call_working_precision_qualified'])

    def test_w03_rejects_incomplete_sum(self):
        self.assertFalse(self.qualify(1)['fixed_qp_outputs_qualified'])

    def test_group_failures_rejected(self):
        for label, wrong in [('GROUP_STABLE', 'F'), ('ACTUAL_BACKEND', '8'),
                             ('CHECK_PASS', 'T'), ('CACHE_ENABLED', 'F')]:
            with self.subTest(label=label):
                self.capture = copy.deepcopy(self.reference)
                self.capture['points'][1]['groups'][0]['data'][label][0][1] = wrong
                self.assertFalse(self.qualify()['fixed_qp_outputs_qualified'])

    def test_raw_nan_rejected(self):
        self.capture['points'][1]['groups'][0]['data']['RAW_FINITE'][0][3] = '7FFF000000000000'
        self.assertFalse(self.qualify()['fixed_qp_outputs_qualified'])

    def test_missing_native_and_wrong_mode_rejected(self):
        for change in ('native', 'mode'):
            with self.subTest(change=change):
                self.capture = copy.deepcopy(self.reference)
                group = self.capture['points'][1]['groups'][0]
                if change == 'native':
                    group['native'] = []
                else:
                    group['key'][3] = 5
                self.assertFalse(self.qualify()['fixed_qp_outputs_qualified'])

    def test_accuracy_sentinel_and_driver_status_rejected(self):
        self.controls[1]['accuracy'] = [0.0, 0.0]
        self.assertFalse(self.qualify()['fixed_qp_outputs_qualified'])
        self.controls[1]['accuracy'] = [-1.0, -1.0]
        self.controls[1]['H'] = 3
        self.assertFalse(self.qualify()['fixed_qp_outputs_qualified'])

    def test_partial_capture_rejected(self):
        self.capture['complete'] = False
        self.assertFalse(self.qualify()['fixed_qp_outputs_qualified'])

    def test_inherited_failure_latch_is_not_physical_input(self):
        self.capture['points'][1]['data']['FPE_QP_BEGIN'][0][1] = 'T'
        self.assertTrue(self.qualify()['fixed_qp_outputs_qualified'])

if __name__ == '__main__':
    unittest.main()
