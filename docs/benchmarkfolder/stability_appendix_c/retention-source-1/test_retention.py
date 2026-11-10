"""Synthetic checks of the fixed-population report contract and local paths."""
from pathlib import Path
import tempfile
import types
import unittest
import copy

HERE = Path(__file__).resolve().parent


def load(path, name):
    module = types.ModuleType(name)
    module.__file__ = str(path)
    exec(compile(path.read_bytes(), str(path), 'exec'), module.__dict__)
    return module


V = load(HERE / 'verify.py', 'retention_under_test')
M = load(HERE.parent / 'source5/plot_metrics.py', 'frozen_metrics_under_test')
C = load(HERE / 'capture.py', 'capture_under_test')


def population():
    cases = []; records = []
    for case in V.CASE_IDS:
        lanes = []
        for backend in V.BACKENDS:
            lanes.append({'backend': backend, 'values': [None] * 100})
            records.append({'request': {'id': case + '-backend' + str(backend)},
                'state': 'unexecuted', 'native_started': False,
                'primary_call_outcomes': [{'state': 'unexecuted'} for _ in range(300)]})
        cases.append({'id': case, 'lanes': lanes})
    return {'cases': cases}, records


class SummaryTests(unittest.TestCase):
    def test_all_missing_retains_twenty_rows_and_unknown_unit_interval(self):
        data, records = population()
        result = V.summarize_records(data, records, M)
        self.assertEqual(len(result['rows']), 20)
        for row in result['rows']:
            self.assertEqual((row['intended_points'], row['available_points'], row['unavailable_points']), (100, 0, 100))
            self.assertEqual(row['primary_outcome_counts'], {'unexecuted': 300})
            self.assertEqual((row['tail_at_1e3']['lower'], row['tail_at_1e3']['upper'], row['tail_at_1e3']['fraction']), (0, 1, None))

    def test_points_calls_split_orders_and_group_failures_keep_distinct_denominators(self):
        data, records = population()
        data['cases'][0]['lanes'][0]['values'] = [0, 'inf', 1e-3] + [None] * 97
        record = records[0]
        record['diagnostics'] = {'primary_call_outcomes': [{'state': 'rejected'} for _ in range(300)],
            'rows': [{'phase': phase, 'order': order, 'status': status}
                     for phase, order, status in ((0, 0, 130), (0, 0, 110), (1, 0, 110), (1, 1, 110), (1, 0, 120))],
            'loop_group_calls': [{'phase': 1, 'coverage': True, 'failed_groups': 2,
                'nonfinite_coefficient_groups': 0, 'nonfinite_result_groups': 1,
                'fpe_dp': False, 'fpe_qp': False, 'fpe_dp_before': True, 'fpe_qp_before': False}]}
        row = V.summarize_records(data, records, M)['rows'][0]
        self.assertEqual((row['available_points'], row['unavailable_points'], row['zero_points'], row['infinite_points']), (3, 97, 1, 1))
        self.assertEqual(row['return_code_counts'], {'110': 1, '120': 1})
        self.assertEqual(row['bootstrap_return_code_counts'], {'110': 1, '130': 1})
        self.assertEqual(row['tail_at_1e3']['known_exceedances'], 1)
        self.assertEqual(row['group_failure_call_counts']['active_group_failure'], 1)
        self.assertEqual(row['group_failure_call_counts']['floating_point_failure_latch'], 1)

    def test_missing_call_slots_cannot_disappear(self):
        data, records = population(); records[0]['primary_call_outcomes'].pop()
        with self.assertRaisesRegex(ValueError, '300 intended'):
            V.summarize_records(data, records, M)


class LocalPathTests(unittest.TestCase):
    def test_failed_guard_card_endpoints_are_observed_without_following(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertEqual(C.card_observation(root / 'missing'), {'status': 'missing'})
            (root / 'file').write_text('card')
            self.assertEqual(C.card_observation(root / 'file'), {'status': 'regular'})
            (root / 'alias').symlink_to('/a/nonexistent/external/card')
            self.assertEqual(C.card_observation(root / 'alias'),
                {'status': 'symlink', 'readlink': '/a/nonexistent/external/card'})
            self.assertEqual(C.card_observation(root), {'status': 'nonregular'})

    def test_only_regular_capsule_local_files_are_admitted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); (root / 'file').write_text('x')
            self.assertEqual(V.local_path(root, 'file'), root / 'file')
            for name in ('../file', '/file', './file', 'a//file'):
                with self.assertRaises(ValueError): V.local_path(root, name)
            (root / 'alias').symlink_to('file')
            with self.assertRaises(ValueError): V.local_path(root, 'alias')


class AdmissionTests(unittest.TestCase):
    def bindings(self):
        index = {'root_admitted_run_terminal_sha256': 'a' * 64,
            'source5_index_sha256': V.SOURCE5_SHA,
            'payloads': {'receipts/run-terminal.json': {'sha256': 'a' * 64, 'bytes': 8},
                         'receipts/prepare-terminal.json': {'sha256': V.PREPARATION_SHA, 'bytes': 9}}}
        retained = {'run_terminal_sha256': 'a' * 64, 'source5_index_sha256': V.SOURCE5_SHA,
            'original_hashes_rechecked_after_capture': True,
            'original_allowlist': {name: {'logical_payload': name, **item} for name, item in index['payloads'].items()}}
        return index, retained

    def test_exact_admission_and_original_payload_joins(self):
        V.check_admission(*self.bindings())

    def test_inconsistent_root_pin_or_original_allowlist_cannot_pass_replay(self):
        index, retained = self.bindings()
        for mutate in (lambda x, r: x.update(root_admitted_run_terminal_sha256='b' * 64),
                       lambda x, r: r.update(source5_index_sha256='b' * 64),
                       lambda x, r: r['original_allowlist'].pop('receipts/run-terminal.json'),
                       lambda x, r: r['original_allowlist']['receipts/run-terminal.json'].update(bytes=7)):
            changed, historical = copy.deepcopy(index), copy.deepcopy(retained)
            mutate(changed, historical)
            with self.assertRaises(ValueError): V.check_admission(changed, historical)


if __name__ == '__main__':
    unittest.main()
