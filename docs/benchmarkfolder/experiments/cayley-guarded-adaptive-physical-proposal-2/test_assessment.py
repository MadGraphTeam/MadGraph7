"""Pure synthetic receipt checks and mutations of retained fixed-QP text."""
import copy
import importlib.util
import json
import math
from pathlib import Path
import struct
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
benchmark = load('frozen_regression_comparison_for_test', ROOT / 'tools/tirust_benchmark.py')

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

    def test_synthetic_QP_to_DP_change_is_numeric_agreement_not_QP_coverage(self):
        # Synthetic path change only. No new host/native result is fabricated.
        before = copy.deepcopy(self.reference)
        for point in before['points'].values():
            point['data']['PUBLIC_STATUS'][0][1] = '323'
        after = copy.deepcopy(before)
        for point in after['points'].values():
            point['data']['PUBLIC_STATUS'][0][1] = '219'
            point['data']['ACCURACY'] = [['R8', i, '0000000000000000'] for i in (1, 2)]
            point['groups'] = []
            point['model_snapshots'] = []
        numeric = assessment.compare_amplitudes(before, after, parser, benchmark)
        self.assertTrue(numeric['passed'])
        self.assertTrue(numeric['ordinary_target_passed'])
        inputs = parser.compare_inputs_and_born(before, after)
        self.assertTrue(inputs['inputs_and_born_exact'])
        self.assertFalse(inputs['all_input_records_comparable'])
        changes = assessment.classify_paths(before, after, parser)
        self.assertTrue(all(p['accepted_precision_changed'] and p['observed_call_count_changed'] for p in changes['points']))
        raw = assessment.previously_served_preserved(before, after, parser)
        self.assertEqual(len(raw['missing_groups_are_coverage_changes']), 400)
        self.assertFalse(raw['preserved_exact'])
        self.assertTrue(raw['matched_groups_bit_exact'])

    def test_numerical_tolerance_allows_ulp_change_but_rejects_large_change(self):
        altered = copy.deepcopy(self.reference)
        row = altered['points'][1]['data']['AMPLITUDE'][1]
        value = float(parser.finite_value(row[2:]))
        encode = lambda v: f'{struct.unpack("<Q", struct.pack("<d", v))[0]:016X}'
        row[2] = encode(math.nextafter(value, math.inf))
        self.assertTrue(assessment.compare_amplitudes(self.reference, altered, parser, benchmark)['passed'])
        row[2] = encode(value * 1.01 if value else 1.0)
        self.assertFalse(assessment.compare_amplitudes(self.reference, altered, parser, benchmark)['passed'])

class GateSeparationTests(unittest.TestCase):
    def setUp(self):
        self.observations = {mode: {
            'numerical_to_closed_adaptive': {'masked': True, 'trapped': True},
            'host_stability_acceptance': {'masked': True, 'trapped': True},
            'observed_QP_path_coverage': {'masked': False, 'trapped': False},
            'IEEE_policy_worker_completion': {'masked': True, 'trapped': True},
            'policy_numerical_agreement': True} for mode in ('checked', 'forced-qp')}

    def test_all_DP_host_acceptance_does_not_claim_QP_coverage(self):
        gates = assessment.separate_gates(self.observations, True, True)['checked']
        self.assertTrue(gates['numerical_agreement'])
        self.assertTrue(gates['host_stability_acceptance'])
        self.assertTrue(gates['trap_completion'])
        self.assertFalse(gates['QP_path_coverage'])

    def test_fault_control_failure_does_not_relabel_numeric_or_host_acceptance(self):
        gates = assessment.separate_gates(self.observations, True, False)['checked']
        self.assertTrue(gates['numerical_agreement'])
        self.assertTrue(gates['host_stability_acceptance'])
        self.assertFalse(gates['trap_completion'])

    def test_clean_trap_completion_does_not_infer_host_stability(self):
        self.observations['checked']['host_stability_acceptance']['masked'] = False
        gates = assessment.separate_gates(self.observations, True, True)['checked']
        self.assertTrue(gates['trap_completion'])
        self.assertFalse(gates['host_stability_acceptance'])

    def test_transport_failure_rescinds_every_gate(self):
        for row in assessment.separate_gates(self.observations, False, True).values():
            self.assertFalse(any(row.values()))

if __name__ == '__main__':
    unittest.main()
