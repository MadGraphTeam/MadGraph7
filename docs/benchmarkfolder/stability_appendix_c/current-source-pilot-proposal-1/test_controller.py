"""Pure/mock admission tests; no compiler, loader or amplitude invocation."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import types
import tempfile
import unittest
from unittest.mock import Mock, patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('current_raw_proposal', HERE / 'controller.py')
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)


def checkpoint():
    lanes = [{'source': source, 'backend': backend, 'status': 'complete', 'error': None,
              'matched_saved_points': 100, 'numerical_agreement_passed': True,
              'ordinary_target_passed': backend == 1}
             for source, backend in [('baseline', 1)] + [('candidate', b) for b in c.BACKENDS]]
    return {'schema': 'madloop-tirust-one-case-closure-v1', 'campaign': 'entry-latch-cache-guard-1',
            'manifest_sha256': c.MANIFEST_SHA, 'case': {'id': c.CASE}, 'file_count': 0, 'files': {},
            'full_matrix_complete': False, 'numerical_result': {'case': c.CASE, 'status': 'passed',
                'returncode': 0, 'raw_replay_passed': True, 'lanes': lanes}}


class Planning(unittest.TestCase):
    def test_default_and_nonexecuting_stage_plans_never_admit_or_run(self):
        for argv in ([], ['plan'], ['prepare'], ['run'], ['plan', '--execute']):
            with (self.subTest(argv=argv), patch.object(c, 'components', side_effect=AssertionError('Not inert')),
                    patch.object(c, 'readonly', side_effect=AssertionError('Not inert')),
                    patch.object(c, 'prepare', side_effect=AssertionError('Not inert')),
                    patch.object(c, 'workers', side_effect=AssertionError('Not inert'))):
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    self.assertEqual(c.main(argv), 0)
                value = json.loads(output.getvalue())
                self.assertFalse(value['execute'])
                self.assertEqual(value['commands_executed'], 0)

    def test_fixed_five_worker_coverage_and_nonlocking_first_order(self):
        rows = c.requests()
        self.assertEqual([row['backend'] for row in rows], [1, 6, 7, 8, 9])
        self.assertEqual([row['lock_required'] for row in rows], [False, False, False, True, True])
        self.assertEqual(len({row['id'] for row in rows}), 5)
        self.assertEqual(sum(row['primary_calls_expected'] for row in rows), 1500)
        self.assertEqual(sum(row['bootstrap_calls_maximum'] for row in rows), 80)

    def test_protocol_has_same_saved_stream_for_every_backend(self):
        value = c.protocol()
        self.assertEqual(value['points']['sha256'], 'b46ce200bb330826d2a4cff69f61f828d3105c0566444fa548b1de75622ee8dd')
        self.assertEqual(value['points']['metadata_sha256'], 'e4a262966bc0de149e3b16a56f85058c56ba068ead2b73f3586101ff38c320e4')
        self.assertEqual(value['limits']['request_wall_seconds'], 180)
        self.assertEqual(value['limits']['prepare_wall_seconds'], 120)
        self.assertEqual(value['limits']['loader_wall_seconds'], 30)

    def test_canonical_private_build_hash_survives_json_key_order(self):
        first = {'z': {'b': 2, 'a': 1}, 'a': 3}
        second = {'a': 3, 'z': {'a': 1, 'b': 2}}
        self.assertEqual(c.canonical(first), c.canonical(second))
        self.assertEqual(c.canonical(first), c.canonical(json.loads(c.canonical(first))))


class ClosedAdmission(unittest.TestCase):
    def setUp(self):
        self.binding = {'case': c.CASE, 'state': 'closed'}

    def test_one_case_closure_does_not_claim_entire_matrix_complete(self):
        value = checkpoint()
        c.verify_case_checkpoint(value, self.binding)
        self.assertFalse(value['full_matrix_complete'])
        self.assertFalse(value['numerical_result']['lanes'][2]['ordinary_target_passed'])

    def test_open_donor_rejected_before_export_or_checksum_reads(self):
        with (patch.object(c, 'read', return_value={'case': c.CASE, 'state': 'active'}),
                patch.object(c, 'digest', side_effect=AssertionError('Opened beyond gate'))):
            with self.assertRaisesRegex(ValueError, 'Closed owner checkpoint'):
                c.closed_donor({}, None, None, None)

    def test_incomplete_failed_or_wrong_case_checkpoints_rejected(self):
        mutations = (
            lambda x: x['case'].update(id='other'),
            lambda x: x['numerical_result'].update(status='running'),
            lambda x: x['numerical_result'].update(returncode=1),
            lambda x: x['numerical_result'].update(raw_replay_passed=False),
            lambda x: x['numerical_result']['lanes'][3].update(status='failed'),
            lambda x: x['numerical_result']['lanes'][3].update(matched_saved_points=99),
            lambda x: x['numerical_result']['lanes'][3].update(numerical_agreement_passed=False),
            lambda x: x['numerical_result']['lanes'].pop(),
        )
        for mutate in mutations:
            value = checkpoint(); mutate(value)
            with self.subTest(value=str(value)[:80]), self.assertRaises(ValueError):
                c.verify_case_checkpoint(value, self.binding)

    def test_private_modules_bind_environment_to_fresh_work(self):
        p, helpers, classifier = c.components()
        self.assertEqual(p.WORK, c.WORK)
        self.assertEqual(p.REPORT, c.REPORT)
        self.assertEqual(set(helpers), {'links', 'cleanup', 'points', 'plot_metrics'})
        self.assertEqual(classifier.TEMPLATE_SHA, '3223f38d2e61243f2b2849d08982d17eadf2a93ae1dc2edaccaa44515467494b')
        with patch.dict(p.os.environ, {'LD_LIBRARY_PATH': '/forbidden', 'LD_PRELOAD': '/forbidden'}):
            environment = p.environment(c.WORK / 'scratch')
        self.assertNotIn('LD_LIBRARY_PATH', environment)
        self.assertNotIn('LD_PRELOAD', environment)
        self.assertTrue(all(str(c.WORK) in environment[name] for name in
            ('XDG_CONFIG_HOME', 'XDG_CACHE_HOME', 'XDG_DATA_HOME', 'XDG_STATE_HOME', 'MPLCONFIGDIR')))

    def test_no_old_high_level_campaign_admission_or_generation_calls(self):
        import ast
        tree = ast.parse((HERE / 'controller.py').read_bytes())
        forbidden = {'protocol', 'donor', 'verify_generation_identity', 'prepare', 'inherit',
                     'inherited_preparation', 'run', 'freeze_cut_points', 'generate_cut_points'}
        invoked = {node.func.attr for node in ast.walk(tree) if isinstance(node, ast.Call)
                   and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name)
                   and node.func.value.id == 'p'}
        self.assertFalse(invoked & forbidden)

    def test_metric_adapter_uses_current_provenance_and_all_100_unknown_slots(self):
        p, helpers, unused = c.components()
        value = c.protocol()
        dataset = c.point_metrics(p, value, [], helpers['plot_metrics'].relative_spread)
        self.assertEqual(dataset['provenance']['manifest_sha256'], c.MANIFEST_SHA)
        self.assertEqual(dataset['provenance']['protocol_sha256'], c.digest(HERE / 'protocol.json'))
        self.assertEqual(dataset['cases'][0]['point_ids'], list(range(1, 101)))
        self.assertEqual([lane['backend'] for lane in dataset['cases'][0]['lanes']], [1, 6, 7, 8, 9])
        self.assertTrue(all(lane['values'] == [None] * 100 for lane in dataset['cases'][0]['lanes']))


class DiagnosticFailures(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.directory = self.root / 'SubProcesses/P0_fixture'
        self.directory.mkdir(parents=True)
        self.links = types.SimpleNamespace(assert_private_path=lambda output, path: path)

    def test_zero_exit_missing_dimensions_is_failed_with_fixed_unavailability(self):
        record = {'state': 'complete'}
        c.collect_diagnostics(record, lambda *args: self.fail('Missing dimensions reached parser'),
                              self.root, self.directory, self.root / 'evidence', {}, self.links)
        self.assertEqual(record['state'], 'failed')
        self.assertIn('all 100', record['diagnostic_unavailable'])

    def test_incomplete_zero_exit_retains_parsed_data_then_rejects(self):
        (self.directory / 'stability-dimensions.dat').write_text('1 100 1 1 1\n')
        for group_complete in (True, False):
            record = {'state': 'complete'}
            parsed = {'structurally_complete': False, 'loop_groups_structurally_complete': group_complete,
                      'rows': [{'status': 110, 'raw_DP_diagnostic_eligible': False}]}
            with self.assertRaisesRegex(ValueError, 'incomplete or structurally invalid'):
                c.collect_diagnostics(record, lambda *args: parsed, self.root, self.directory,
                                      self.root / 'evidence', {}, self.links)
            self.assertEqual(record['diagnostics'], parsed)
            self.assertIn('stability-dimensions.dat', record['raw_diagnostics'])

    def test_native_failure_is_retained_without_becoming_protocol_failure(self):
        (self.directory / 'stability-dimensions.dat').write_text('1 100 1 1 1\n')
        record = {'state': 'complete'}
        parsed = {'structurally_complete': True, 'loop_groups_structurally_complete': True,
                  'rows': [{'status': 110, 'raw_DP_diagnostic_eligible': False,
                            'reason': 'Active loop group reported numerical failure'}]}
        c.collect_diagnostics(record, lambda *args: parsed, self.root, self.directory,
                              self.root / 'evidence', {}, self.links)
        self.assertEqual(record['state'], 'complete')
        self.assertFalse(record['diagnostics']['rows'][0]['raw_DP_diagnostic_eligible'])

    def test_cpu13_adapter_and_clean_outer_environment_on_launch_failure(self):
        p, helpers, unused = c.components()
        signals = types.SimpleNamespace(check=lambda: None)
        with (patch.object(p.subprocess, 'Popen', side_effect=OSError('mock launch failure')) as popen,
              patch.dict(p.os.environ, {'LD_LIBRARY_PATH': '/outer-forbidden', 'LD_PRELOAD': '/outer-forbidden'})):
            record = p.command(['/not-run/stability-check'], self.root, self.root / 'command',
                               30, helpers['cleanup'], signals,
                               target_environment={'LD_LIBRARY_PATH': '/target-only'})
        self.assertEqual(popen.call_count, 1)
        self.assertEqual(record['argv'][:3], ['taskset', '-c', '13'])
        self.assertEqual(record['cpu'], 13)
        self.assertFalse(record['cpu_exclusive'])
        self.assertFalse(record['process_started'])
        self.assertEqual(record['target_environment']['LD_LIBRARY_PATH'], '/target-only')
        self.assertFalse(any(key.startswith('LD_') for key in record['setup_environment']))
        self.assertIn('mock launch failure', record['error']['message'])


class MockLifecycle(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.saved = {}
        self.links = types.SimpleNamespace(
            assert_private_path=lambda output, path: path,
            verify_tree=Mock(), verify_external_graph=Mock(),
            _resolve=lambda path, missing: (path, 'missing', []))
        def copy(master, output, snapshot, audit):
            (output / 'SubProcesses/P0_fixture').mkdir(parents=True)
            return {'master': str(master), 'expected_master': snapshot, 'audit': audit}
        self.links.prepare_worker = Mock(side_effect=copy)
        self.p = types.SimpleNamespace(
            save=lambda path, value: self.saved.update({Path(path).name: value}),
            signal_requests=lambda cleanup: contextlib.nullcontext(types.SimpleNamespace()),
            worker_postguard=Mock(return_value={'passed': True}),
            numerical_lock=Mock(side_effect=AssertionError('Dryrun acquired a lock')),
            command=Mock(side_effect=AssertionError('Dryrun started a child')))
        def prelaunch(value, request, case, output, helpers, record):
            record['card_write_intent'] = ['Cards/MadLoopParams.dat']
            return output / case['donor']['subprocess']
        self.p.prelaunch_worker = Mock(side_effect=prelaunch)
        self.helpers = {'links': self.links, 'cleanup': object()}
        self.case = {'master_snapshot': {'root': str(self.root / 'master')},
            'donor': {'subprocess': 'SubProcesses/P0_fixture'},
            'runtime_closure': {'files': {}},
            'donor_context': {'master': str(self.root / 'closed-export'), 'expected_master': {}, 'audit': {}}}
        self.classifier = types.SimpleNamespace(parsers=lambda *args, **kwargs: {'parse_diagnostics': object()})

    def test_dryrun_copies_all_five_workers_with_zero_child_or_lock_calls(self):
        with (patch.object(c, 'prepared', return_value=('source', {}, self.p, self.helpers,
                self.classifier, self.case, object())), patch.object(c, 'WORK', self.root / 'work'),
              patch.object(c, 'REPORT', self.root / 'work/evidence'), patch.object(c, 'production'),
              patch.object(c, 'seal', return_value='source'), patch.object(c, 'digest', return_value='digest')):
            c.workers('dryrun')
        terminal = self.saved['dryrun-terminal.json']
        self.assertEqual(terminal['status'], 'passed')
        self.assertEqual([row['request']['backend'] for row in terminal['results']], [1, 6, 7, 8, 9])
        self.assertEqual(self.links.prepare_worker.call_count, 5)
        self.assertEqual(self.p.prelaunch_worker.call_count, 5)
        self.assertEqual(self.p.worker_postguard.call_count, 5)
        self.assertTrue(all(row['protected_guards_passed'] and not row['native_started'] for row in terminal['results']))
        self.p.command.assert_not_called(); self.p.numerical_lock.assert_not_called()

    def test_protected_input_failure_stops_later_workers_and_invalidates_metrics(self):
        self.links.verify_tree.side_effect = ValueError('mock changed source')
        with (patch.object(c, 'prepared', return_value=('source', {}, self.p, self.helpers,
                self.classifier, self.case, object())), patch.object(c, 'WORK', self.root / 'work'),
              patch.object(c, 'REPORT', self.root / 'work/evidence'), patch.object(c, 'production'),
              patch.object(c, 'seal', return_value='source'), patch.object(c, 'digest', return_value='digest')):
            with self.assertRaisesRegex(ValueError, 'retains failures'):
                c.workers('dryrun')
        terminal = self.saved['dryrun-terminal.json']
        self.assertEqual(terminal['status'], 'failed')
        self.assertFalse(terminal['results'][0]['readonly_guard_passed'])
        self.assertIn('source_admission_error', terminal['results'][0])
        self.assertEqual([row['state'] for row in terminal['results'][1:]], ['unexecuted'] * 4)
        self.links.prepare_worker.assert_not_called(); self.p.command.assert_not_called()

    def test_compiler_and_loader_failures_still_run_protected_finally_guards(self):
        for failed_stage in ('compiler', 'loader'):
            with self.subTest(stage=failed_stage):
                root = self.root / failed_stage
                data = {'export': str(root / 'export'), 'subprocess': 'SubProcesses/P0_fixture',
                        'compiled': {'worker_render_parameters': {}}}
                admitted = {'source_index_sha256': 'source', 'donor': data, 'donor_snapshot': {},
                            'audit': {}, 'group_layout': {'include': '      INTEGER I\n'},
                            'compiler': {'path': '/mock/compiler'}}
                p = types.SimpleNamespace(**self.p.__dict__)
                p.render = lambda *args: '      PROGRAM MOCK\n      END\n'
                p.MAKEFILE = 'private mock makefile\n'
                p.environment = lambda cwd: {'PATH': '/mock'}
                p.native_environment = lambda manifest: {}
                p.private_changes = lambda *args: {}
                p.elf_interpreter = lambda executable: Path('/mock/loader')
                calls = []
                def command(argv, cwd, evidence, *args, **kwargs):
                    calls.append(str(argv[0]))
                    if len(calls) == 1 and failed_stage == 'loader':
                        return {'error': None}
                    raise RuntimeError('mock ' + failed_stage + ' failure')
                p.command = command
                old_tree = self.links.verify_tree.call_count
                old_external = self.links.verify_external_graph.call_count
                with (patch.object(c, 'readonly', return_value=admitted), patch.object(c, 'protocol',
                        return_value={'limits': {'prepare_wall_seconds': 120, 'loader_wall_seconds': 30}}),
                      patch.object(c, 'components', return_value=(p, self.helpers, object())),
                      patch.object(c, 'WORK', root / 'work'), patch.object(c, 'REPORT', root / 'work/evidence'),
                      patch.object(c, 'production') as production, patch.object(c, 'seal', return_value='source'),
                      patch.object(c.shutil, 'which', return_value='/mock/bash')):
                    with self.assertRaisesRegex(ValueError, 'Preparation failed'):
                        c.prepare()
                terminal = self.saved['prepare-terminal.json']
                self.assertEqual(terminal['status'], 'failed')
                self.assertIn('mock ' + failed_stage + ' failure', terminal['error']['message'])
                self.assertIsNone(terminal['protected_guard_error'])
                self.assertGreater(self.links.verify_tree.call_count, old_tree)
                self.assertGreater(self.links.verify_external_graph.call_count, old_external)
                production.assert_called()


if __name__ == '__main__':
    unittest.main()
