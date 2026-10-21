"""New release collector completion/failure guards; no numerical subprocesses."""
from contextlib import ExitStack, nullcontext
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / ('docs/benchmarkfolder/policy-v3/final-pins-20261021-v1/'
                 'core-correctness/core_collect.py')
SPEC = importlib.util.spec_from_file_location('ninjars_final_core_collect_test', SOURCE)
core = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(core)
LAUNCH_SPEC = importlib.util.spec_from_file_location('ninjars_final_core_launch_test', SOURCE.with_name('launch.py'))
launch = importlib.util.module_from_spec(LAUNCH_SPEC)
LAUNCH_SPEC.loader.exec_module(launch)


class FinalCoreCollectionTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT / '.benchmark-work',
                                               prefix='final-core-collection-test-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.report, self.work = self.root / 'report', self.root / 'work'
        self.config = {'cases': [{'id': cid} for cid in core.ORDER]}
        self.specification = self.root / 'specification.json'
        self.specification.write_text('{}\n')
        points = self.root / 'points.dat'; points.write_text('synthetic fixture\n')
        self.spec = {'configuration': self.config, 'sources': {}, 'backends': [8],
                     'scheduled_sources': ['candidate'],
                     'report': str(self.report), 'work': str(self.work),
                     'fresh_parity_scope': 'synthetic fixture only',
                     'cohort': {'cases': {cid: {'points': {'path': str(points)}} for cid in core.ORDER},
                                'candidate_cohort': {'generation_revision': '1' * 40,
                                                     'numerical_revision': '2' * 40,
                                                     'runtime_identity_sha256': '3' * 64}}}
        self.handoff = self.root / 'handoff.json'
        self.handoff.write_text(json.dumps({'root_authorized': True,
            'phase': 'final-pins-core-correctness', 'cpu': core.CPU,
            'specification_sha256': core.supplement.digest(self.specification)}))
        self.args = SimpleNamespace(specification=self.specification, handoff=self.handoff,
                                    execute_numerical=True)

    def run_collector(self, fail_case=None, runtime=None):
        checked_status = []
        self.owner_labels = []
        def owned(report, label):
            self.owner_labels.append(label)
            return nullcontext()
        def run_case(case, *args):
            if case['id'] == fail_case:
                raise RuntimeError('numerical worker failed')
        def checked_series(report, minimum):
            status = core.read(report / 'run-status.json')
            self.assertEqual(status['status'], 'complete')
            self.assertEqual(status['requested_cases'], list(core.ORDER))
            self.assertEqual(status['completed_cases'], list(core.ORDER))
            checked_status.append(status)
        targets = {'fixture': {'reducer_runtime': {
            'identity_sha256': runtime or self.spec['cohort']['candidate_cohort']['runtime_identity_sha256']}}}
        with ExitStack() as stack:
            patches = [(core.policy, 'verified', lambda value, schema: self.spec),
                       (core, 'configuration', lambda: self.config),
                       (core, 'check_sources', lambda spec: {'numerical_revision': '2' * 40}),
                       (core, 'strict_records', lambda report, cid: None),
                       (core.os, 'sched_getaffinity', lambda pid: {core.CPU}),
                       (core.supplement, 'checked', lambda item: Path(item['path'])),
                       (core.supplement, 'bound', lambda path: {'path': str(path)}),
                       (core.supplement, 'harness', lambda *args: nullcontext()),
                       (core.supplement, 'input_roots', lambda *args: nullcontext()),
                       (core.pair, 'owned_preparation', owned),
                       (core.workers, 'run_case', run_case),
                       (core.workers, 'checked_series', checked_series),
                       (core.supplement, 'bind_collected', lambda *args: {'targets': targets})]
            for target, name, replacement in patches:
                stack.enter_context(mock.patch.object(target, name, replacement))
            stack.enter_context(mock.patch.dict(os.environ, core.THREAD_ENV))
            result = core.collect(self.args)
        return result, checked_status

    def test_complete_metadata_precedes_unchanged_series_checker(self):
        result, checked = self.run_collector()
        self.assertEqual(len(checked), 1)
        self.assertEqual(result['status'], 'complete')
        self.assertFalse(result['performance_acceptance_gate'])
        self.assertFalse(result['final_dataset_claim'])
        self.assertEqual(self.owner_labels, list(core.ORDER))

    def test_interrupted_numerical_case_retains_exact_incomplete_coverage(self):
        with self.assertRaisesRegex(RuntimeError, 'numerical worker failed'):
            self.run_collector(fail_case=core.ORDER[2])
        status = core.read(self.report / 'run-status.json')
        self.assertEqual(status['status'], 'failed')
        self.assertEqual(status['completed_cases'], list(core.ORDER[:2]))
        self.assertEqual(status['requested_cases'], list(core.ORDER))
        self.assertFalse((self.report / 'raw-result.json').exists())

    def test_wrong_collected_runtime_cannot_publish_success(self):
        with self.assertRaisesRegex(ValueError, 'another full runtime'):
            self.run_collector(runtime='4' * 64)
        self.assertEqual(core.read(self.report / 'run-status.json')['status'], 'failed')
        self.assertFalse((self.report / 'raw-result.json').exists())

    def test_new_release_cannot_reuse_old_phase_handoff(self):
        value = core.read(self.handoff); value['phase'] = 'nrs-first-core-correctness'
        self.handoff.write_text(json.dumps(value))
        with mock.patch.object(core.policy, 'verified', return_value=self.spec), \
                mock.patch.object(core, 'check_sources') as sources:
            with self.assertRaisesRegex(ValueError, 'root numerical handoff'):
                core.collect(self.args)
        sources.assert_not_called()
        self.assertFalse(self.report.exists())

    def test_launcher_reaps_exact_flat_spawn_identity(self):
        self.report.mkdir(); attempt = self.root / 'attempt'; attempt.mkdir()
        recorded = {'event': 'spawn', 'pid': 12345, 'pgid': 12345, 'start_ticks': 999}
        (self.report / 'process-ledger.jsonl').write_text(json.dumps(recorded) + '\n')
        actual = {**recorded, 'state': 'S'}
        with mock.patch.object(launch, 'identity', side_effect=[actual, None, None]), \
                mock.patch.object(launch.os, 'killpg') as signal_group:
            actions = launch.cleanup_verified_workers(self.report, attempt)
        signal_group.assert_called_once_with(recorded['pgid'], launch.signal.SIGTERM)
        self.assertEqual(actions, [{'action': 'TERM', 'identity': actual}])

    def test_launcher_refuses_reused_or_unregistered_flat_identity(self):
        self.report.mkdir(); recorded = {'event': 'spawn', 'pid': 12345, 'pgid': 12345, 'start_ticks': 999}
        for old_ticks in (999, None):
            attempt = self.root / ('attempt-' + str(old_ticks)); attempt.mkdir()
            (self.report / 'process-ledger.jsonl').write_text(json.dumps({**recorded, 'start_ticks': old_ticks}) + '\n')
            actual = {**recorded, 'start_ticks': 1000, 'state': 'S'}
            with self.subTest(old_ticks=old_ticks), mock.patch.object(launch, 'identity', return_value=actual), \
                    mock.patch.object(launch.os, 'killpg') as signal_group:
                actions = launch.cleanup_verified_workers(self.report, attempt)
            signal_group.assert_not_called()
            self.assertEqual(actions[0]['action'], 'refused-stale-identity')


if __name__ == '__main__':
    unittest.main()
