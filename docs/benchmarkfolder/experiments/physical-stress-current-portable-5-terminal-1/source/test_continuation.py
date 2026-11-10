"""Pure policy and mocked/temp-filesystem controller tests; no native runs."""
import contextlib
import copy
import importlib.util
import io
import json
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


p = load('continuation_policy_tested', HERE / 'policy.py')
r = load('continuation_controller_tested', HERE / 'run.py')


def fixture():
    plan = copy.deepcopy(r.PLAN); keys = [p.key(request) for request in plan['requests']]
    rows = []
    for i, request in enumerate(plan['requests']):
        rows.append({**{name: request[name] for name in ('case', 'label', 'source', 'backend')},
            'request_id': keys[i], 'state': 'host_checked_complete' if i < 4 else ('deferred_before_launch' if i == 4 else 'not_started'),
            'host_checked_complete_stream': i < 4, 'qualified_final_points': [1, 2] if i < 4 else [],
            'raw_final_points_observed': [1, 2] if i < 4 else [], 'warmup_calls_observed': 2 if i < 4 else 0})
    summary = {'request_rows': rows, 'stage_status': 'failed', 'completed_orchestration': 4, 'host_checked_complete_streams': 4}
    stage = {'status': 'failed', 'exception': "TimeoutError('Lock wait expired before execution')"}
    accounting = {'requested': 24, 'completed': 4, 'not_completed': copy.deepcopy(plan['requests'][4:]), 'native_stage_failure': stage['exception']}
    abort = {'request': copy.deepcopy(plan['requests'][4]), 'state': 'deferred-before-launch', 'exception': stage['exception'], 'numerically_qualified': False}
    return [plan, summary, stage, accounting, abort, keys[:4]]


class Policy(unittest.TestCase):
    def rejected(self, mutate):
        args = fixture(); mutate(args)
        with self.assertRaises((ValueError, KeyError)): p.select(*args)

    def test_exact_twenty_suffix_not_first_four(self):
        args = fixture(); selected = p.select(*args)
        self.assertEqual(selected, args[0]['requests'][4:])
        self.assertEqual(len(selected), 20)
        selected[0]['backend'] = -1
        self.assertEqual(args[0]['requests'][4]['backend'], 8)

    def test_reject_passed_or_active_ancestor(self):
        for state in ('passed', 'running'):
            with self.subTest(state=state): self.rejected(lambda a: a[2].update(status=state))

    def test_reject_other_timeout(self):
        self.rejected(lambda a: a[2].update(exception="TimeoutError('native request expired')"))

    def test_reject_during_request_abort(self):
        self.rejected(lambda a: a[4].update(state='aborted-during-request'))

    def test_reject_different_deferred_request(self):
        self.rejected(lambda a: a[4].update(request=a[0]['requests'][5]))

    def test_reject_partial_or_rejected_first_four(self):
        self.rejected(lambda a: a[1]['request_rows'][2].update(state='physical_rejected', host_checked_complete_stream=False))

    def test_reject_missing_first_four_point(self):
        self.rejected(lambda a: a[1]['request_rows'][3].update(qualified_final_points=[1]))

    def test_reject_hidden_later_launch(self):
        self.rejected(lambda a: a[5].append(p.key(a[0]['requests'][4])))

    def test_reject_partial_later_observation(self):
        self.rejected(lambda a: a[1]['request_rows'][5].update(warmup_calls_observed=1))
        self.rejected(lambda a: a[1]['request_rows'][4].update(raw_final_points_observed=[1]))

    def test_reject_reordered_requests(self):
        self.rejected(lambda a: a[1]['request_rows'].reverse())

    def test_reject_forged_completion_counts(self):
        self.rejected(lambda a: a[1].update(host_checked_complete_streams=24))
        self.rejected(lambda a: a[3].update(completed=5))

    def test_reject_skipped_not_completed_id(self):
        self.rejected(lambda a: a[3]['not_completed'].pop())

    def test_reject_identity_changed(self):
        self.rejected(lambda a: a[1]['request_rows'][4].update(backend=9))

    def test_combination_preserves_failure_and_no_truth_claim(self):
        args = fixture(); rows = args[1]['request_rows']; rows[7]['state'] = 'physical_rejected'
        combined = p.combine(args[0], rows[:4], rows[4:])
        self.assertEqual(combined['coverage_count'], 24)
        self.assertEqual(combined['request_rows'][7]['state'], 'physical_rejected')
        self.assertIsNone(combined['coefficient_agreement'])
        self.assertFalse(combined['independent_physical_accuracy_qualified'])
        self.assertFalse(combined['performance_eligible'])

    def test_combination_rejects_overlap_or_missing_or_reorder(self):
        args = fixture(); rows = args[1]['request_rows']
        for right in (rows[3:], rows[5:], list(reversed(rows[4:]))):
            with self.subTest(size=len(right)):
                with self.assertRaises(ValueError): p.combine(args[0], rows[:4], right)


class Controller(unittest.TestCase):
    def test_default_prints_only_and_does_not_seal_or_execute(self):
        with patch.object(r.sys, 'argv', ['run.py']), patch.object(r, 'seal', side_effect=AssertionError('seal called')), \
             patch.object(r, 'inherit', side_effect=AssertionError('inherit called')), patch.object(r, 'run', side_effect=AssertionError('run called')), \
             contextlib.redirect_stdout(io.StringIO()) as output:
            r.main()
        value = json.loads(output.getvalue())
        self.assertEqual(value['commands_executed'], 0); self.assertEqual(value['conditional_requests'], 20)
        self.assertTrue(value['sealed_for_execution'])

    def test_source_only_execute_rejected_before_namespaces(self):
        with patch.object(r, 'read', return_value={'sealed_for_execution': False}):
            with self.assertRaisesRegex(RuntimeError, 'Source-only proposal'): r.seal()

    def test_environment_rebases_writable_paths(self):
        donor = str(r.OLD_WORK)
        with patch.object(r.base, 'environment', return_value={'MPLCONFIGDIR': donor + '/matplotlib', 'XDG_STATE_HOME': donor + '/xdg/state', 'PATH': '/bin'}):
            actual = r.environment(r.WORK)
        self.assertEqual(actual['MPLCONFIGDIR'], str(r.WORK / 'matplotlib'))
        self.assertEqual(actual['XDG_STATE_HOME'], str(r.WORK / 'xdg/state'))
        self.assertEqual(actual['PATH'], '/bin')

    def test_caps_are_original_caps(self):
        self.assertEqual(r.CAPS, {'inherit': 600, 'run': 20 * 380})
        for request in r.PLAN['requests']:
            self.assertEqual(request['inclusive_worker_seconds'], 380)
            self.assertEqual(request['startup_seconds'], 180); self.assertEqual(request['traversal_seconds'], 180)

    def test_readonly_inherit_calls_guard_twice_no_probe(self):
        with tempfile.TemporaryDirectory() as temporary:
            report, work = Path(temporary) / 'report', Path(temporary) / 'work'
            with patch.object(r, 'REPORT', report), patch.object(r, 'WORK', work), \
                 patch.object(r, 'verify_inherited_inputs', return_value={'closed': 'mocked'}) as guard, \
                 patch.object(r.base, 'cmd', side_effect=AssertionError('subprocess called')), \
                 patch.object(r.base, 'runtime', side_effect=AssertionError('loader called')):
                r.inherit()
            self.assertEqual(guard.call_count, 2)
            state = json.loads((report / 'inherited-checkpoint.json').read_bytes())
            self.assertEqual(state['new_setup_launches'], 0)
            self.assertEqual(state['new_mass_probe_launches'], 0)
            self.assertEqual(state['inherited_completed_requests'], 4)

    def test_readonly_inherit_rejects_changed_guard(self):
        with tempfile.TemporaryDirectory() as temporary:
            with patch.object(r, 'REPORT', Path(temporary) / 'report'), patch.object(r, 'WORK', Path(temporary) / 'work'), \
                 patch.object(r, 'verify_inherited_inputs', side_effect=[{'closed': 1}, {'closed': 2}]):
                with self.assertRaisesRegex(RuntimeError, 'checkpoint changed'): r.inherit()

    def test_original_closed_report_must_match_capsule(self):
        retained = {'original_closed_root': str(r.OLD_REPORT), 'closed_files': {'a': {'sha256': 'abc', 'size': 3}}, 'closed_directories': ['.', 'd']}
        snapshot = {'entries': {'a': {'kind': 'regular_file', 'sha256': 'abc', 'size': 3}, 'd': {'kind': 'directory'}}}
        r.verify_closed_report(retained, snapshot)
        snapshot['entries']['a']['sha256'] = 'changed'
        with self.assertRaisesRegex(RuntimeError, 'byte binding'): r.verify_closed_report(retained, snapshot)

    def test_original_closed_report_rejects_unbound_extra_or_link(self):
        retained = {'original_closed_root': str(r.OLD_REPORT), 'closed_files': {}, 'closed_directories': ['.']}
        for item in ({'kind': 'symlink'}, {'kind': 'regular_file', 'sha256': 'x', 'size': 1}):
            with self.subTest(item=item):
                with self.assertRaisesRegex(RuntimeError, 'inventory differs'): r.verify_closed_report(retained, {'entries': {'x': item}})

    def test_inherited_card_alias_rejected_before_write(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); output = root / 'worker'; output.mkdir(); donor = root / 'inherited-card'; donor.write_text('unchanged')
            (output / 'Cards').mkdir(); (output / 'Cards/param_card.dat').symlink_to(donor)
            directory = output / 'SubProcesses/P0_test'; directory.mkdir(parents=True)
            with self.assertRaises(r.links.LinkClosureError): r.private_destinations(output, directory)
            self.assertEqual(donor.read_text(), 'unchanged')

    def test_unrecognized_private_output_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary); directory = output / 'SubProcesses/P0_test'; directory.mkdir(parents=True)
            context = {'expected_worker': {'entries': {'SubProcesses': {}, 'SubProcesses/P0_test': {}}}}
            (directory / 'unexpected.txt').write_text('x')
            with self.assertRaisesRegex(RuntimeError, 'Unexpected worker output'): r.allowed_worker_changes(output, directory, context)

    @contextlib.contextmanager
    def mocked_request(self, *, guard_side_effect=None, prepare_error=None, native_error=None, post_error=None):
        with tempfile.TemporaryDirectory() as temporary, contextlib.ExitStack() as stack:
            root = Path(temporary); report = root / 'report'; report.mkdir(); work = root / 'work'; work.mkdir()
            (report / 'inherited-checkpoint.json').write_text('{}')
            points = root / 'points.dat'; points.write_text('mock points')
            checksum = lambda data: hashlib.sha256(data).hexdigest()
            entry = {'case': {'id': 'mock'}, 'process_subdirectory': 'SubProcesses/P0_mock',
                     'physical_card_sha256': checksum(b'card'), 'point_sha256': checksum(b'mock points')}
            request = {'case': 'mock', 'source': 'candidate', 'label': 'candidate-9-auto', 'backend': 9, 'points': str(points)}
            evidence = report / 'runs/mock/candidate-9-auto'
            stack.enter_context(patch.object(r, 'PLAN', {'cases': [entry]}))
            stack.enter_context(patch.object(r, 'REPORT', report)); stack.enter_context(patch.object(r, 'WORK', work))
            original_read = r.read
            def read(path):
                if Path(path).name == 'closed-master-audit.json': return {'masters': {'candidate/mock': {}}, 'external_graph': {}}
                if Path(path).name == 'preflight.json': return {'runtime': {'files': {}}}
                return original_read(path)
            stack.enter_context(patch.object(r, 'read', side_effect=read))
            guard = stack.enter_context(patch.object(r, 'verify_inherited_inputs', side_effect=guard_side_effect,
                 return_value={'ancestor_index_sha256': 'mocked', 'closed': True}))
            stack.enter_context(patch.object(r.base, 'build_record', return_value={'executable_sha256': checksum(b'executable')}))
            def prepare(master, output, expected, audit):
                if prepare_error: raise prepare_error
                (output / 'Cards').mkdir(parents=True); (output / 'Cards/param_card.dat').write_bytes(b'card')
                (output / 'Cards/MadLoopParams.dat').write_text('numerical')
                directory = output / entry['process_subdirectory']; directory.mkdir(parents=True)
                (directory / 'check').write_bytes(b'executable')
                return {'expected_worker': {'entries': {}}}
            copied = stack.enter_context(patch.object(r.links, 'prepare_worker', side_effect=prepare))
            stack.enter_context(patch.object(r, 'private_destinations'))
            stack.enter_context(patch.object(r.base, 'configure_worker', return_value={}))
            stack.enter_context(patch.object(r.base, 'scientific_sources', return_value={'f': 'sha'}))
            stack.enter_context(patch.object(r.base.bench, 'numerical_card_inputs', return_value={'card': 'sha'}))
            stack.enter_context(patch.object(r.base.bench, 'resolved_parameter_card', side_effect=lambda directory: directory.parents[1] / 'Cards/param_card.dat'))
            stack.enter_context(patch.object(r.base, 'runtime', return_value={'files': {}}))
            def native(*args):
                r.save(evidence / 'result.json', {'numerically_qualified': True, 'runtime_identity_failure': False})
                r.save(report / 'execution/request-mock-candidate-9-auto.json', {'cleanup': {'survivors': []}, 'cleanup_exception': None})
                if native_error: raise native_error
            launched = stack.enter_context(patch.object(r.base, 'cmd', side_effect=native))
            stack.enter_context(patch.object(r, 'allowed_worker_changes', return_value=(set(), {})))
            checked = stack.enter_context(patch.object(r.links, 'assert_worker_inputs', side_effect=post_error))
            yield request, evidence, guard, copied, launched, checked

    def test_request_preguard_prevents_copy_and_native(self):
        with self.mocked_request(guard_side_effect=RuntimeError('guard failed')) as (request, evidence, guard, copied, launched, checked):
            with self.assertRaisesRegex(RuntimeError, 'guard failed'): r.run_request(request)
            copied.assert_not_called(); launched.assert_not_called()

    def test_request_copy_failure_still_guards_originals(self):
        with self.mocked_request(prepare_error=RuntimeError('copy refused')) as (request, evidence, guard, copied, launched, checked):
            with self.assertRaisesRegex(RuntimeError, 'copy refused'): r.run_request(request)
            self.assertEqual(guard.call_count, 2); launched.assert_not_called(); checked.assert_not_called()
            self.assertTrue((evidence / 'inherited-after.json').is_file())

    def test_request_clean_completion_checks_private_and_original_inputs(self):
        with self.mocked_request() as (request, evidence, guard, copied, launched, checked):
            result = r.run_request(request)
            self.assertTrue(result['numerically_qualified']); self.assertEqual(guard.call_count, 2)
            launched.assert_called_once(); checked.assert_called_once()
            self.assertTrue(json.loads((evidence / 'completion.json').read_bytes())['inputs_unchanged'])

    def test_request_native_failure_still_checks_private_and_original_inputs(self):
        with self.mocked_request(native_error=RuntimeError('controller failed')) as (request, evidence, guard, copied, launched, checked):
            with self.assertRaisesRegex(RuntimeError, 'Fatal cleanup'): r.run_request(request)
            self.assertEqual(guard.call_count, 2); checked.assert_called_once()
            self.assertFalse((evidence / 'completion.json').exists())

    def test_request_private_postguard_revokes_passing_result_and_guards_originals(self):
        with self.mocked_request(post_error=RuntimeError('private changed')) as (request, evidence, guard, copied, launched, checked):
            with self.assertRaisesRegex(RuntimeError, 'private changed'): r.run_request(request)
            self.assertEqual(guard.call_count, 2)
            completion = json.loads((evidence / 'completion.json').read_bytes())
            self.assertFalse(completion['result']['numerically_qualified']); self.assertFalse(completion['inputs_unchanged'])

    def test_request_original_postguard_revokes_passing_result(self):
        with self.mocked_request(guard_side_effect=[{'ancestor_index_sha256': 'mocked', 'closed': True}, {'closed': 'changed'}]) as (request, evidence, guard, copied, launched, checked):
            with self.assertRaisesRegex(RuntimeError, 'Inherited input changed'): r.run_request(request)
            self.assertFalse(json.loads((evidence / 'completion.json').read_bytes())['result']['numerically_qualified'])

    def test_run_executes_exact_twenty_suffix_with_original_deadlines(self):
        args = fixture(); selected = p.select(*args)
        with tempfile.TemporaryDirectory() as temporary:
            report = Path(temporary)
            with patch.object(r, 'REPORT', report), patch.object(r, 'frozen', return_value={'checkpoint': {'requests': selected}}), \
                 patch.object(r.base, 'acquire', return_value=None) as acquire, patch.object(r, 'run_request', return_value={'numerically_qualified': True}) as request, \
                 patch.object(r.L, 'Deadline', side_effect=lambda seconds: contextlib.nullcontext()) as deadline:
                r.run()
            self.assertEqual([call.args[0] for call in request.call_args_list], selected)
            self.assertEqual(acquire.call_count, 20); self.assertEqual(deadline.call_count, 20)
            self.assertTrue(all(call.args[0] == 380 for call in deadline.call_args_list))
            summary = json.loads((report / 'summary.json').read_bytes())
            self.assertEqual(summary['request_count'], 20); self.assertIsNone(summary['full24_coefficient_agreement'])

    def test_run_lock_timeout_does_not_start_request_or_deadline(self):
        selected = p.select(*fixture())
        with tempfile.TemporaryDirectory() as temporary:
            report = Path(temporary)
            with patch.object(r, 'REPORT', report), patch.object(r, 'frozen', return_value={'checkpoint': {'requests': selected}}), \
                 patch.object(r.base, 'acquire', side_effect=TimeoutError('Lock wait expired before execution')), \
                 patch.object(r, 'run_request') as request, patch.object(r.L, 'Deadline') as deadline:
                with self.assertRaises(TimeoutError): r.run()
            request.assert_not_called(); deadline.assert_not_called()
            abort = json.loads((report / 'aborted-request.json').read_bytes())
            self.assertEqual(abort['request'], selected[0]); self.assertEqual(abort['state'], 'deferred-before-launch')


class ActualClosedBinding(unittest.TestCase):
    """Actual immutable input reads, plus isolated mock mutations; no stages."""

    def test_closed_terminal_capsule_and_all_inherited_inputs_no_launch(self):
        with patch.object(r.base, 'cmd', side_effect=AssertionError('child command forbidden')), \
             patch.object(r.base, 'runtime', side_effect=AssertionError('loader forbidden')), \
             patch.object(r.L, 'capture', side_effect=AssertionError('query child forbidden')):
            state = r.verify_inherited_inputs()
        self.assertEqual(len(state['builds']), 8)
        self.assertEqual(state['requests'], r.PLAN['requests'][4:])
        self.assertEqual(state['ancestor_index_sha256'], '6fc2035649ba0387f0d4457784ae883592f13bb7e3b6f9c33b99e038b81f7146')
        self.assertFalse(r.WORK.exists()); self.assertFalse(r.REPORT.exists())

    def test_closed_retention_size_and_root_marker_match_actual_observation(self):
        capsule = HERE.with_name('physical-stress-current-portable-4-terminal-2')
        retained = r.read(capsule / 'retention.json'); audit = r.read(HERE / 'closed-master-audit.json')
        r.verify_closed_report(retained, audit['closed_report'])
        self.assertEqual(len(retained['closed_files']), 1208)
        self.assertEqual(len(retained['closed_directories']), 26)

    def test_actual_terminal_preserves_threshold_disagreement(self):
        summary = r.read(HERE.with_name('physical-stress-current-portable-4-terminal-2') / 'qualification-summary.json')
        self.assertEqual(summary['host_checked_complete_streams'], 4)
        self.assertEqual(len(summary['comparisons']), 6)
        self.assertEqual(sum(not comparison['passed'] for comparison in summary['comparisons']), 5)
        self.assertFalse(summary['six_lane_all_cases_agree'])

    def test_source_inventory_rejects_changed_binding_bytes(self):
        original = r.sha
        def changed(path):
            return '0' * 64 if Path(path) == HERE / 'binding.json' else original(path)
        with patch.object(r, 'sha', side_effect=changed):
            with self.assertRaisesRegex(RuntimeError, 'Source changed: binding.json'): r.seal()

    def test_actual_ancestor_audit_confirmation_and_summary_pin_mutations_rejected(self):
        original_read = r.read
        original_binding = original_read(HERE / 'binding.json')
        capsule = HERE.with_name('physical-stress-current-portable-4-terminal-2')
        authenticated_shape = {'verified': True, 'index_sha256': original_binding['ancestor_index_sha256'],
                              'preparation_index_sha256': original_binding['preparation_index_sha256'],
                              'summary': original_read(capsule / 'qualification-summary.json')}
        for pin, message in (('ancestor_index_sha256', 'Ancestor capsule differs'),
                             ('ancestor_verifier_sha256', 'Ancestor capsule differs'),
                             ('closed_master_audit_sha256', 'Closed audit/confirmation differs'),
                             ('root_terminal_confirmation_sha256', 'Closed audit/confirmation differs'),
                             ('ancestor_summary_sha256', 'Ancestor summary binding differs')):
            with self.subTest(pin=pin):
                mutated = {**original_binding, pin: '0' * 64}
                with patch.object(r, 'read', side_effect=lambda path: mutated if Path(path) == HERE / 'binding.json' else original_read(path)), \
                     patch.object(r, 'load', return_value=SimpleNamespace(verify=lambda root: authenticated_shape)):
                    with self.assertRaisesRegex(RuntimeError, message): r.verify_inherited_inputs()

    def test_root_confirmation_failure_rejected_even_with_matched_mock_hash(self):
        original_read = r.read; original_sha = r.sha
        path = HERE / 'root-terminal-confirmation.json'; binding = original_read(HERE / 'binding.json')
        confirmation = {**original_read(path), 'owned_cleanup_complete': False}
        capsule = HERE.with_name('physical-stress-current-portable-4-terminal-2')
        shape = {'verified': True, 'index_sha256': binding['ancestor_index_sha256'], 'preparation_index_sha256': binding['preparation_index_sha256'],
                 'summary': original_read(capsule / 'qualification-summary.json')}
        with patch.object(r, 'read', side_effect=lambda value: confirmation if Path(value) == path else original_read(value)), \
             patch.object(r, 'load', return_value=SimpleNamespace(verify=lambda root: shape)):
            with self.assertRaisesRegex(RuntimeError, 'Missing genuine terminal4 confirmation'): r.verify_inherited_inputs()


if __name__ == '__main__': unittest.main()
