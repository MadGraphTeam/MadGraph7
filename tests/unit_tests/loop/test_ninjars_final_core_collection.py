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
                       (core.operations, 'owned_processes', lambda *args: nullcontext()),
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


if __name__ == '__main__':
    unittest.main()
