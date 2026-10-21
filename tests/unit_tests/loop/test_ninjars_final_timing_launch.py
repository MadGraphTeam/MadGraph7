"""Actual Python-child ownership tests; no generated executable or reducer runs."""
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / 'docs/benchmarkfolder/policy-v3/final-pins-20261021-v1/timing-launch/launch.py'
SPEC = importlib.util.spec_from_file_location('final_timing_launch_test', SOURCE)
launch = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launch)


class TimingLaunchTest(unittest.TestCase):
    def setUp(self):
        (ROOT / '.benchmark-work').mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=ROOT / '.benchmark-work', prefix='timing-owner-test-')
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        self.report, self.attempt = self.path / 'report', self.path / 'attempt'
        self.report.mkdir()

    def read(self, name):
        return json.loads((self.attempt / name).read_text())

    def test_real_child_session_environment_and_terminal_record(self):
        command = [sys.executable, '-c', 'import os,json; print(json.dumps({k:os.environ[k] for k in '
                   "('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','RAYON_NUM_THREADS','MKL_NUM_THREADS')}))"]
        with mock.patch.object(launch, 'terminal_ledger', return_value={'synthetic_python_fixture': True}):
            code = launch.run_owned(command, self.report, self.attempt, {'fixture': 'Python only'})
        self.assertEqual(code, 0)
        initial = self.read('launch.json'); terminal = self.read('result.json')
        self.assertEqual(initial['controller_identity']['pid'], initial['controller_identity']['pgid'])
        self.assertGreater(initial['controller_identity']['start_ticks'], 0)
        self.assertEqual(json.loads((self.attempt / 'stdout').read_text()), {k: '1' for k in launch.pair.THREADS})
        self.assertEqual(terminal['status'], 'complete')
        self.assertTrue(terminal['controller_absent'] and terminal['controller_group_absent'])
        self.assertFalse(terminal['automatic_retry'] or terminal['in_place_resume'])

    def test_failed_child_is_preserved_without_retry_or_success(self):
        code = launch.run_owned([sys.executable, '-c', 'raise SystemExit(7)'],
                                self.report, self.attempt, {'fixture': 'Python only'})
        self.assertEqual(code, 7)
        terminal = self.read('result.json')
        self.assertEqual(terminal['status'], 'execution-not-complete')
        self.assertEqual(terminal['returncode'], 7)
        self.assertTrue(terminal['controller_absent'] and terminal['controller_group_absent'])
        self.assertFalse(terminal['automatic_retry'])

    def test_interrupt_between_popen_return_and_assignment_is_owned_and_reaped(self):
        real_popen = subprocess.Popen

        def interrupting_popen(*args, **kwargs):
            child = real_popen(*args, **kwargs)
            os.kill(os.getpid(), signal.SIGTERM)
            return child

        with mock.patch.object(launch.subprocess, 'Popen', side_effect=interrupting_popen):
            with self.assertRaisesRegex(KeyboardInterrupt, 'Deferred interruption'):
                launch.run_owned([sys.executable, '-c', 'import time; time.sleep(30)'],
                                 self.report, self.attempt, {'fixture': 'Python only'})
        initial, terminal = self.read('launch.json'), self.read('result.json')
        self.assertGreater(initial['controller_identity']['start_ticks'], 0)
        self.assertEqual(terminal['interruptions'][0]['signal'], signal.SIGTERM)
        self.assertEqual(terminal['status'], 'execution-not-complete')
        self.assertTrue(terminal['controller_absent'] and terminal['controller_group_absent'])
        with self.assertRaises(ProcessLookupError):
            os.killpg(initial['controller_identity']['pgid'], 0)

    def test_outer_success_requires_terminal_custody(self):
        with mock.patch.object(launch, 'terminal_ledger', side_effect=ValueError('missing reap')):
            with self.assertRaisesRegex(ValueError, 'missing reap'):
                launch.run_owned([sys.executable, '-c', 'pass'], self.report, self.attempt, {})
        self.assertEqual(self.read('result.json')['status'], 'execution-not-complete')

    def ledger(self, events):
        (self.report / 'process-ledger.jsonl').write_text(''.join(json.dumps(e) + '\n' for e in events))
        (self.report / 'run-status.json').write_text(json.dumps(
            {'status': 'complete', 'requested_cases': ['fixture'], 'completed_cases': ['fixture']}))

    def test_terminal_custody_rejects_missing_mismatched_and_failed_reaps(self):
        spawn = {'event': 'spawn', 'pid': 99991, 'pgid': 99991, 'start_ticks': 111}
        reap = {**spawn, 'event': 'reaped', 'reaped': True, 'returncode': 0}
        for events in ([spawn], [spawn, {**reap, 'start_ticks': 222}],
                       [spawn, {**reap, 'returncode': 1}], [spawn, {**reap, 'returncode': False}]):
            with self.subTest(events=events):
                self.ledger(events)
                with self.assertRaises(ValueError):
                    launch.terminal_ledger(self.report)

    def test_terminal_custody_refuses_a_surviving_owned_group(self):
        spawn = {'event': 'spawn', 'pid': 99991, 'pgid': 99991, 'start_ticks': 111}
        self.ledger([spawn, {**spawn, 'event': 'reaped', 'reaped': True, 'returncode': 0}])
        with mock.patch.object(launch.owner, 'identity', return_value=None), \
                mock.patch.object(launch.os, 'killpg', return_value=None):
            with self.assertRaisesRegex(RuntimeError, 'process group remains'):
                launch.terminal_ledger(self.report)

    def test_no_quiet_handoff_starts_no_controller(self):
        specification = self.path / 'spec.json'; specification.write_text('{}')
        handoff = self.path / 'handoff.json'; handoff.write_text('{}')
        quiet = self.path / 'quiet.json'; quiet.write_text(json.dumps({'root_authorized': True}))
        with mock.patch.dict(launch.SPECS, {'core': (specification, launch.supplement.digest(specification))}), \
                mock.patch.object(launch.subprocess, 'Popen') as popen:
            with self.assertRaisesRegex(ValueError, 'quiet handoff'):
                launch.prepare('core', handoff, quiet)
        popen.assert_not_called()

    def test_exact_only_ninja_nrs_specs_are_bound(self):
        self.assertEqual(set(launch.SPECS), {'core', 'loop'})
        for filename, digest in launch.SPECS.values():
            self.assertEqual(launch.supplement.digest(filename), digest)
            spec = launch.read(filename)
            self.assertEqual(spec['pair_targets'], [['baseline', 6], ['candidate', 8]])
            self.assertEqual(spec['repetitions'], 7)

    def prepare_fixture(self, stage, targets=None, release=True):
        physical = self.path / 'physical.json'; physical.write_text('{}')
        specification = self.path / 'spec.json'
        value = {'configuration': launch.supplement.bound(physical),
                 'correction': launch.supplement.bound(physical),
                 'pair_targets': targets or [['baseline', 6], ['candidate', 8]],
                 'repetitions': 7, 'report': str(self.path / 'new-report'),
                 'work': str(self.path / 'new-work'), 'candidate_cohort': {'fixture': True}}
        specification.write_text(json.dumps(value))
        digest = launch.supplement.digest(specification)
        handoff = self.path / 'handoff.json'
        required = {'root_authorized': True, 'phase': launch.pair.PAIR_SCHEMA if stage == 'core' else 'fixture-loop-schema',
                    'cpu': 207, 'specification_sha256': digest}
        if stage == 'loop':
            required['quiet_cpu207'] = True
        handoff.write_text(json.dumps(required))
        quiet = self.path / 'quiet.json'
        quiet.write_text(json.dumps({'schema': 'final-timing-quiet-handoff-v1',
            'root_authorized': True, 'stage': stage, 'cpu': 207, 'quiet_cpu207': True,
            'all_other_numerical_and_compiler_work_released': True,
            'specification_sha256': digest, 'collector_handoff_sha256': launch.supplement.digest(handoff),
            'release_evidence': [launch.supplement.bound(physical)] if release else []}))
        dummy_loop = SimpleNamespace(validate_spec=lambda data: (data, {}), SCHEMA='fixture-loop-schema')
        with mock.patch.dict(launch.SPECS, {stage: (specification, digest)}), \
                mock.patch.object(launch.pair, 'validate_spec', side_effect=lambda data: data), \
                mock.patch.object(launch.pair, 'preflight'), \
                mock.patch.object(launch.policy, 'validate_correction'), \
                mock.patch.object(launch, 'load', return_value=dummy_loop), \
                mock.patch.object(launch.os, 'sched_getaffinity', return_value={6, 7, 207}), \
                mock.patch.object(launch.subprocess, 'Popen') as popen:
            result = launch.prepare(stage, handoff, quiet)
        popen.assert_not_called()
        return result

    def test_both_commands_use_the_real_collectors_spec_option(self):
        for stage in ('core', 'loop'):
            with self.subTest(stage=stage):
                result = self.prepare_fixture(stage)
                command = result['command']
                self.assertIn('--spec', command)
                self.assertIn('--execute-numerical', command)
                self.assertEqual(result['stage'], stage)
                self.assertNotIn('--backends', command)

    def test_backend_injection_is_rejected_before_spawn(self):
        for stage in ('core', 'loop'):
            with self.subTest(stage=stage), self.assertRaisesRegex(ValueError, 'Only the reviewed'):
                self.prepare_fixture(stage, targets=[['baseline', 6], ['candidate', 8], ['candidate', 7]])

    def test_quiet_boolean_without_release_evidence_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'release evidence'):
            self.prepare_fixture('core', release=False)


if __name__ == '__main__':
    unittest.main()
