"""Source-only successor tests: temporary inputs, mocked ownership, no children."""
import ast
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import types
import unittest
from unittest.mock import Mock, patch


HERE = Path(__file__).resolve().parent
OLD = HERE.with_name('current-cms-replay-supervisor-1')
PROPOSAL = HERE.with_name('current-cms-replay-proposal-1')
PINNED_SOURCE1_INDEX = 'edf6d8cacfe1ad72cdb94584cb6d9a49e022e976c52d0668d670b702c28c84d6'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


with patch.object(subprocess, 'Popen', side_effect=AssertionError('Source import must not launch children')):
    R = load('cms_successor2_adapter_tests', HERE / 'run.py')
    N = load('cms_successor2_unchanged_native_tests', OLD / 'native_job.py')


class NoChildTests(unittest.TestCase):
    def setUp(self):
        self.no_child = patch.object(
            subprocess, 'Popen', side_effect=AssertionError('No real child admitted by source tests'))
        self.no_child.start()
        self.addCleanup(self.no_child.stop)


class FrozenContractTests(NoChildTests):
    def test_source1_index_and_all_26_payloads_remain_exact(self):
        self.assertEqual(digest(OLD / 'source-index.json'), PINNED_SOURCE1_INDEX)
        files = json.loads((OLD / 'source-index.json').read_bytes())['files']
        self.assertEqual(len(files), 26)
        for name, item in files.items():
            with self.subTest(path=name):
                self.assertEqual(digest(OLD / name), item['sha256'])
                self.assertEqual((OLD / name).stat().st_size, item['size'])

    def test_default_dry_run_has_zero_commands_and_creates_no_namespace(self):
        with patch.object(R.sys, 'argv', ['run.py']), patch.object(R, 'seal') as seal, \
                patch.object(Path, 'mkdir', side_effect=AssertionError('Dry-run attempted a filesystem write')), \
                contextlib.redirect_stdout(io.StringIO()) as output:
            R.main()
        plan = json.loads(output.getvalue())
        self.assertEqual(plan['commands_executed'], 0)
        self.assertEqual(plan['requests'], 6)
        self.assertTrue(plan['source_only'])
        seal.assert_not_called()

    def test_only_inherit_and_run_are_public_stages(self):
        self.assertEqual(tuple(R.STAGES), ('inherit', 'run'))
        self.assertEqual(R.CAPS, {'inherit': 600, 'run': 3720})
        for forbidden in ('prepare', 'preflight'):
            with self.subTest(stage=forbidden), \
                    patch.object(R.sys, 'argv', ['run.py', forbidden]), \
                    contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                R.main()

    def test_execute_requires_explicit_stage(self):
        with patch.object(R.sys, 'argv', ['run.py', '--execute']), \
                self.assertRaisesRegex(RuntimeError, 'Explicit stage'):
            R.main()

    def test_frozen_six_requests_keep_order_caps_points_cards_and_policy(self):
        original = json.loads((PROPOSAL / 'proposal.json').read_bytes())
        expected_order = [('baseline', 1), ('candidate', 1), ('candidate', 6),
                          ('candidate', 7), ('candidate', 8), ('candidate', 9)]
        self.assertEqual([(x['source'], x['backend']) for x in R.PLAN['requests']], expected_order)
        for inherited, expected in zip(R.PLAN['requests'], original['requests']):
            for key in ('id', 'backend', 'inclusive_seconds', 'lock_required',
                        'numerical_card', 'point_indices', 'precision_request',
                        'protocol_phase_timeout_seconds', 'source', 'stdin'):
                self.assertEqual(inherited[key], expected[key], key)
            self.assertEqual(inherited['inclusive_seconds'], 620)
            self.assertEqual(inherited['protocol_phase_timeout_seconds'], 300)
            self.assertEqual(inherited['precision_request'], 'auto')
            self.assertEqual(inherited['point_indices'], list(range(1, 22)))
            self.assertEqual(digest(inherited['points']), original['points_sha256'])
            self.assertEqual(digest(inherited['numerical_card']), digest(expected['numerical_card']))
        for key in ('points_sha256', 'physical_card_sha256', 'strict_host_accuracy_limit',
                    'ordinary_target', 'regression_tolerance', 'expected_CMS_model',
                    'expected_observation', 'worker_policy'):
            self.assertEqual(R.PLAN[key], original[key], key)
        self.assertEqual(R.PLAN['strict_host_accuracy_limit'], 1e-3)
        self.assertEqual(N.POINT_COUNT, 21)
        self.assertEqual(N.ACCURACY_LIMIT, 1e-3)
        self.assertEqual(N.PHASE_TIMEOUT_SECONDS, 300)

    def test_master_reader_roots_remain_old_and_owned_outputs_are_new(self):
        self.assertEqual(R.base.WORK.name, 'current-cms-replay-1')
        self.assertEqual(R.base.REPORT.name, 'current-cms-replay-qualification-1')
        self.assertEqual(R.WORK.name, 'current-cms-replay-2')
        self.assertEqual(R.REPORT.name, 'current-cms-replay-qualification-2')
        self.assertEqual(R.L.WORK, R.WORK)
        self.assertEqual(R.L.REPORT, R.REPORT)
        self.assertEqual(R.q.WORK, R.WORK)
        self.assertEqual(R.q.REPORT, R.REPORT)

    def test_effective_request_and_writable_environment_paths_use_successor(self):
        for request, original in zip(R.PLAN['requests'], R.base.PLAN['requests']):
            for name in ('cwd', 'fifo_argument_appended_by_worker'):
                self.assertTrue(request[name].startswith(str(R.WORK) + '/'))
                self.assertTrue(original[name].startswith(str(R.OLD_WORK) + '/'))
        sample = {'MPLCONFIGDIR': str(R.OLD_WORK / 'matplotlib'),
                  'XDG_STATE_HOME': str(R.OLD_WORK / 'xdg/state'),
                  'XDG_CACHE_HOME': str(R.OLD_WORK / 'xdg/cache'),
                  'OMP_NUM_THREADS': '1', 'PWD': str(R.WORK / 'worker')}
        with patch.object(R.base, 'environment', return_value=sample):
            actual = R.environment(R.WORK / 'worker')
        for name in ('MPLCONFIGDIR', 'XDG_STATE_HOME', 'XDG_CACHE_HOME'):
            self.assertTrue(actual[name].startswith(str(R.WORK) + '/'))
        self.assertEqual(actual['OMP_NUM_THREADS'], '1')
        self.assertEqual(actual['PWD'], str(R.WORK / 'worker'))

    def test_no_generation_build_or_mass_commands_are_exposed(self):
        source = (HERE / 'run.py').read_text()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                self.assertNotIn(node.func.attr, ('prepare', 'prepare_process', 'materialize',
                                                 'build_probe', 'preflight'))
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                self.assertNotEqual(node.value, '--masses')


class InheritanceTests(NoChildTests):
    def test_inherit_is_read_only_setup_with_exact_checkpoint_and_failed_pin(self):
        checkpoint = {'ancestor_index_sha256': R.BINDING['ancestor_index_sha256'],
                      'audit_sha256': R.BINDING['audit_sha256'],
                      'builds': {'baseline': {'build_sha256': 'b' * 64},
                                 'candidate': {'preflight_sha256': 'c' * 64}}}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.object(R, 'REPORT', root / 'report'), patch.object(R, 'WORK', root / 'work'), \
                    patch.object(R, 'verify_inherited_inputs', side_effect=[checkpoint, copy.deepcopy(checkpoint)]) as guard, \
                    patch.object(R.base, 'cmd', side_effect=AssertionError('Inheritance invoked a command')):
                R.inherit()
                receipt = json.loads((R.REPORT / 'inherited-checkpoint.json').read_bytes())
            self.assertEqual(guard.call_count, 2)
            self.assertEqual(receipt['checkpoint'], checkpoint)
            self.assertEqual(receipt['ancestor_failed_native_attempt'], R.BINDING['ancestor_index_sha256'])
            self.assertEqual(receipt['new_ambient_directory_snapshot'], R.BINDING['audit_sha256'])
            self.assertFalse(receipt['source2_executed_setup'])
            self.assertEqual(receipt['subprocess_launch_count'], 0)
            self.assertEqual(receipt['mass_probe_launch_count'], 0)
            self.assertEqual(receipt['CMS_model_probe_launch_count'], 0)
            self.assertEqual(set(receipt['inherited_stages']), {'prepare', 'preflight'})
            self.assertEqual(list((root / 'work').iterdir()), [])

    def test_changed_checkpoint_or_reused_namespace_denies_inheritance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.object(R, 'REPORT', root / 'report'), patch.object(R, 'WORK', root / 'work'), \
                    patch.object(R, 'verify_inherited_inputs', side_effect=[{'pin': 1}, {'pin': 2}]):
                with self.assertRaisesRegex(RuntimeError, 'checkpoint changed'):
                    R.inherit()
                self.assertFalse((R.REPORT / 'inherited-checkpoint.json').exists())
                with patch.object(R, 'verify_inherited_inputs') as guard, \
                        self.assertRaisesRegex(RuntimeError, 'Fresh successor'):
                    R.inherit()
                guard.assert_not_called()

    def test_frozen_rechecks_the_recorded_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = Path(tmp)
            (report / 'inherited-checkpoint.json').write_text(json.dumps({'checkpoint': {'pin': 1}}))
            with patch.object(R, 'REPORT', report), \
                    patch.object(R, 'verify_inherited_inputs', return_value={'pin': 1}):
                self.assertEqual(R.frozen(), {'checkpoint': {'pin': 1}})
            with patch.object(R, 'REPORT', report), \
                    patch.object(R, 'verify_inherited_inputs', return_value={'pin': 2}), \
                    self.assertRaisesRegex(RuntimeError, 'Inherited checkpoint changed'):
                R.frozen()

    def test_closed_report_inventory_and_bytes_are_exact(self):
        for change in ('content', 'addition', 'deletion', 'empty-directory', 'symlink'):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                (root / 'builds').mkdir()
                item = root / 'builds/build.json'
                item.write_bytes(b'closed build receipt')
                retention = {'original_closed_root': str(root), 'closed_directories': ['builds'],
                             'closed_files': {'builds/build.json': {'bytes': item.stat().st_size,
                                                                 'sha256': digest(item)}}}
                with patch.object(R, 'OLD_REPORT', root):
                    R.verify_original_evidence(retention)
                    if change == 'content': item.write_bytes(b'changed build receipt')
                    elif change == 'addition': (root / 'new.json').write_text('{}')
                    elif change == 'deletion': item.unlink()
                    elif change == 'empty-directory': (root / 'extra').mkdir()
                    else: (root / 'alias').symlink_to(item)
                    with self.assertRaisesRegex(RuntimeError, 'donor evidence'):
                        R.verify_original_evidence(retention)

    @contextlib.contextmanager
    def verification_fixture(self, ancestor=None):
        ancestor = ancestor or {'index_sha256': R.BINDING['ancestor_index_sha256'],
                                'summary': {'overall_status': 'failed', 'amplitude_launch_count': 0}}
        verifier = types.SimpleNamespace(verify=Mock(return_value=ancestor))
        audit = {'schema': 'cms-replay-link-closure-audit-v1',
                 'source1_master_guard_inventories': {
                     name: {'root': '/synthetic/' + name, 'entries': {}}
                     for name in ('baseline_export', 'candidate_export', 'sealed_supervisor1', 'sealed_proposal1')}}
        pres = {s: {'runtime': {'files': {'/synthetic/' + s + '.so': s + '-runtime'}},
                    'probe_runtime': {'files': {'/synthetic/' + s + '-probe.so': s + '-probe-runtime'}},
                    'complex_model': {'CMSParam': 1.0, 'source': s}}
                for s in ('baseline', 'candidate')}
        def fixture_read(path):
            if Path(path) == R.AUDIT_PATH: return audit
            if Path(path) == R.CAPSULE / 'retention.json': return {'synthetic_retention': True}
            raise AssertionError('Unmocked read from inherited evidence: ' + str(path))
        def fixture_sha(path): return hashlib.sha256(str(path).encode()).hexdigest()
        with contextlib.ExitStack() as stack:
            mocks = {}
            for obj, name, options in (
                    (R, 'seal', {}), (R, 'load', {'return_value': verifier}),
                    (R, 'read', {'side_effect': fixture_read}), (R, 'sha', {'side_effect': fixture_sha}),
                    (R, 'verify_original_evidence', {}), (R.links, 'verify_tree', {}),
                    (R.links, 'verify_external_graph', {}), (R.base, 'frozen', {}),
                    (R.base, 'check_hashes', {}),
                    (R.base, 'build_record', {'side_effect': lambda s, e: {
                        'executable_sha256': s + '-executable', 'probe': {'sha256': s + '-probe'}}}),
                    (R.base, 'preflight_record', {'side_effect': lambda s, c: (pres[s], s + '-preflight')})):
                mocks[name] = stack.enter_context(patch.object(obj, name, **options))
            yield types.SimpleNamespace(mocks=mocks, preflights=pres, sha=fixture_sha)

    def test_inherited_build_and_preflight_receipts_are_preserved_without_children(self):
        with self.verification_fixture() as fixture:
            result = R.verify_inherited_inputs()
        self.assertEqual(result['ancestor_index_sha256'], R.BINDING['ancestor_index_sha256'])
        self.assertEqual(result['donor_source_index_sha256'], PINNED_SOURCE1_INDEX)
        self.assertEqual(result['source_freeze_sha256'], fixture.sha(R.OLD_REPORT / 'source-freeze.json'))
        self.assertEqual(fixture.mocks['verify_tree'].call_count, 4)
        fixture.mocks['verify_external_graph'].assert_called_once()
        for source in ('baseline', 'candidate'):
            record = result['builds'][source]
            expected_build = R.OLD_REPORT / 'builds' / (source + '-' + R.PLAN['case']['id']) / 'build.json'
            self.assertEqual(record['build_sha256'], fixture.sha(expected_build))
            self.assertEqual(record['preflight_sha256'], source + '-preflight')
            self.assertEqual(record['runtime'], fixture.preflights[source]['runtime'])
            self.assertEqual(record['probe_runtime'], fixture.preflights[source]['probe_runtime'])
            self.assertEqual(record['complex_model'], fixture.preflights[source]['complex_model'])
        self.assertEqual(fixture.mocks['check_hashes'].call_count, 2)

    def test_failed_ancestor_pin_status_and_zero_amplitudes_are_required(self):
        base = {'index_sha256': R.BINDING['ancestor_index_sha256'],
                'summary': {'overall_status': 'failed', 'amplitude_launch_count': 0}}
        for mutation in ('index', 'status', 'amplitude'):
            ancestor = copy.deepcopy(base)
            if mutation == 'index': ancestor['index_sha256'] = '0' * 64
            elif mutation == 'status': ancestor['summary']['overall_status'] = 'passed'
            else: ancestor['summary']['amplitude_launch_count'] = 1
            with self.subTest(mutation=mutation), self.verification_fixture(ancestor) as fixture, \
                    self.assertRaisesRegex(RuntimeError, 'Failed ancestor'):
                R.verify_inherited_inputs()
            fixture.mocks['build_record'].assert_not_called()

    def test_main_inherit_receipt_has_600_second_bound_and_no_native_execution(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'source'
            source.mkdir()
            (source / 'source-index.json').write_text('{}')
            report = root / 'report'
            def synthetic_inherit():
                report.mkdir()
                (report / 'inherited-checkpoint.json').write_text('{}')
            with patch.object(R, 'HERE', source), patch.object(R, 'REPORT', report), \
                    patch.object(R, 'WORK', root / 'work'), patch.object(R, 'seal'), \
                    patch.object(R, 'inherit', side_effect=synthetic_inherit), patch.object(R, 'frozen'), \
                    patch.object(R.L, 'Deadline', side_effect=lambda seconds: contextlib.nullcontext()) as deadline, \
                    patch.object(R.L.cleanup, 'SignalRequests', return_value=contextlib.nullcontext(Mock())), \
                    patch.object(R.sys, 'argv', ['run.py', 'inherit', '--execute']):
                R.main()
            deadline.assert_called_once_with(600)
            receipt = json.loads((report / 'stage-inherit.json').read_bytes())
            self.assertEqual(receipt['status'], 'passed')
            self.assertEqual(receipt['cap_seconds_excluding_lock_wait'], 600)
            self.assertFalse(receipt['native_execution'])
            self.assertEqual(receipt['ancestor_failed_native_attempt'], R.BINDING['ancestor_index_sha256'])


class RequestTests(NoChildTests):
    @contextlib.contextmanager
    def request_fixture(self, *, runtime=None, guards=None, worker_error=None, cleanup_survivors=None):
        with tempfile.TemporaryDirectory() as tmp, contextlib.ExitStack() as stack:
            root = Path(tmp)
            old = root / 'closed-work1'
            work = root / 'fresh-work2'
            report = root / 'fresh-report2'
            report.mkdir()
            (report / 'inherited-checkpoint.json').write_text('{}')
            request = copy.deepcopy(R.PLAN['requests'][0])
            plan = copy.deepcopy(R.PLAN)
            points = root / 'points.dat'
            points.write_bytes(b'synthetic frozen full21 points; never evaluated\n')
            request['points'] = str(points)
            plan['points_sha256'] = digest(points)
            case = request['case']
            master = old / 'exports' / request['source'] / case
            process = master / 'SubProcesses/P0_epem_mupmum'
            process.mkdir(parents=True)
            (master / 'Cards').mkdir()
            (master / 'Cards/param_card.dat').write_bytes(b'physical card\n')
            (master / 'Cards/MadLoopParams.dat').write_bytes(b'numerical card\n')
            (master / 'Cards/ident_card.dat').write_bytes(b'ident card\n')
            (process / 'check').write_bytes(b'inert worker executable bytes; never launched\n')
            resources = master / 'SubProcesses/MadLoop5_resources'
            resources.mkdir()
            (master / 'SubProcesses/MadLoopParams.dat').symlink_to('../Cards/MadLoopParams.dat')
            (resources / 'MadLoopParams.dat').symlink_to('../MadLoopParams.dat')
            (resources / 'param_card.dat').symlink_to('../../Cards/param_card.dat')
            (process / 'MadLoop5_resources').symlink_to('../MadLoop5_resources')
            plan['cases'][0]['physical_card_sha256'] = digest(master / 'Cards/param_card.dat')
            output = work / 'workers' / request['label'] / case
            directory = output / 'SubProcesses/P0_epem_mupmum'
            evidence = report / 'runs' / case / request['label']
            events = []
            build = {'executable_sha256': digest(process / 'check')}
            expected_runtime = {'/synthetic/runtime.so': 'f' * 64}
            pre = {'runtime': {'files': expected_runtime}, 'complex_model': {'CMSParam': 1.0}}
            audit_path = root / 'audit.json'
            audit_path.write_text(json.dumps({'source1_master_guard_inventories': {'baseline_export': {'entries': {}}}}))
            def prepare(donor, private, expected, audit):
                events.append('copy-and-rebase')
                self.assertEqual(Path(donor), master)
                self.assertEqual(Path(private), output)
                self.assertFalse(private.exists())
                private.parent.mkdir(parents=True)
                shutil.copytree(donor, private, symlinks=True)
                return {'expected_worker': {'entries': {p.relative_to(private).as_posix(): {} for p in private.rglob('*')}}}
            def configure(private, req):
                events.append('configure')
                self.assertTrue((private / 'SubProcesses/P0_epem_mupmum/check').is_file())
                self.assertEqual((directory / 'MadLoop5_resources').resolve(), output / 'SubProcesses/MadLoop5_resources')
                (private / 'SubProcesses/MadLoopParams.dat').write_bytes(b'configured only private card\n')
                return {'requested': {'mlstabthres': 1e-3}}
            def command(argv, label, cwd, seconds):
                events.append('native-controller-mock')
                self.assertEqual(Path(argv[2]), OLD / 'native_job.py')
                self.assertEqual(Path(cwd), directory)
                self.assertEqual(seconds, 610)
                (evidence / 'result.json').write_text(json.dumps({'numerically_qualified': True, 'orchestration_complete': True}))
                execution = report / 'execution' / (label + '.json')
                execution.parent.mkdir(parents=True, exist_ok=True)
                execution.write_text(json.dumps({'cleanup': {'survivors': cleanup_survivors or []}, 'cleanup_exception': None}))
                if worker_error is not None: raise worker_error
            mocks = {}
            settings = [(R, 'PLAN', {'new': plan}), (R, 'OLD_WORK', {'new': old}),
                        (R, 'WORK', {'new': work}), (R, 'REPORT', {'new': report}),
                        (R, 'AUDIT_PATH', {'new': audit_path}),
                        (R, 'verify_inherited_inputs', {'side_effect': guards or [{'pin': 1}, {'pin': 1}]}),
                        (R.base, 'build_record', {'return_value': build}),
                        (R.base, 'preflight_record', {'return_value': (pre, 'preflight-receipt')}),
                        (R.links, 'prepare_worker', {'side_effect': prepare}),
                        (R.base, 'configure_worker', {'side_effect': configure}),
                        (R.base, 'scientific_sources', {'return_value': {'worker.f': 'science-sha'}}),
                        (R.base.bench, 'numerical_card_inputs', {'return_value': {'numerical': 'sealed-card-sha'}}),
                        (R.base.bench, 'resolved_parameter_card', {'return_value': output / 'Cards/param_card.dat'}),
                        (R.base, 'runtime', {'return_value': {'files': expected_runtime if runtime is None else runtime}}),
                        (R.base, 'cmd', {'side_effect': command}),
                        (R, 'allowed_worker_changes', {'return_value': ({'Cards/MadLoopParams.dat'}, set())}),
                        (R.links, 'assert_worker_inputs', {})]
            for obj, name, options in settings:
                value = stack.enter_context(patch.object(obj, name, **options))
                mocks[name] = value
            yield types.SimpleNamespace(root=root, request=request, output=output, directory=directory,
                                        evidence=evidence, master=master, events=events, mocks=mocks)

    def test_copy_and_private_destinations_precede_configuration_and_native_mock(self):
        with self.request_fixture() as fixture:
            before = {p.relative_to(fixture.master).as_posix(): digest(p)
                      for p in fixture.master.rglob('*') if p.is_file() and not p.is_symlink()}
            result = R.run_request(fixture.request)
            self.assertTrue(result['numerically_qualified'])
            self.assertEqual(fixture.events, ['copy-and-rebase', 'configure', 'native-controller-mock'])
            self.assertEqual(fixture.mocks['verify_inherited_inputs'].call_count, 2)
            self.assertEqual(before, {p.relative_to(fixture.master).as_posix(): digest(p)
                                      for p in fixture.master.rglob('*') if p.is_file() and not p.is_symlink()})
            specification = json.loads((fixture.evidence / 'specification.json').read_bytes())
            self.assertEqual(specification['cwd'], str(fixture.directory))
            self.assertIn(str(fixture.directory / 'check'), specification['expected_runtime'])
            self.assertEqual(specification['phase_timeout_seconds'], 300)
            self.assertEqual(specification['host_accuracy_limit'], 1e-3)
            self.assertEqual(specification['points_count'], 21)
            completion = json.loads((fixture.evidence / 'completion.json').read_bytes())
            self.assertTrue(completion['inputs_unchanged'])

    def test_pre_request_guard_fails_before_copy_configuration_or_native(self):
        with self.request_fixture(guards=[RuntimeError('inherited pre-guard failed')]) as fixture:
            with self.assertRaisesRegex(RuntimeError, 'pre-guard'):
                R.run_request(fixture.request)
            fixture.mocks['prepare_worker'].assert_not_called()
            fixture.mocks['configure_worker'].assert_not_called()
            fixture.mocks['cmd'].assert_not_called()

    def test_runtime_equality_rejects_added_missing_and_changed_runtime_before_native(self):
        for runtime in ({}, {'/synthetic/runtime.so': '0' * 64},
                        {'/synthetic/runtime.so': 'f' * 64, '/synthetic/extra.so': 'e' * 64}):
            with self.subTest(runtime=runtime), self.request_fixture(runtime=runtime) as fixture:
                with self.assertRaisesRegex(RuntimeError, 'Relocated runtime differs'):
                    R.run_request(fixture.request)
                fixture.mocks['cmd'].assert_not_called()
                self.assertEqual(fixture.mocks['verify_inherited_inputs'].call_count, 2)
                self.assertFalse((fixture.evidence / 'completion.json').exists())

    def test_post_request_inherited_guard_revokes_qualified_native_result(self):
        with self.request_fixture(guards=[{'pin': 1}, {'pin': 2}]) as fixture:
            with self.assertRaisesRegex(RuntimeError, 'Inherited inputs changed'):
                R.run_request(fixture.request)
            result = json.loads((fixture.evidence / 'completion.json').read_bytes())
            self.assertFalse(result['result']['numerically_qualified'])
            self.assertFalse(result['inputs_unchanged'])
            self.assertEqual(result['fatal_infrastructure_failure'], 'post-request input guard')

    def test_worker_post_guard_failure_still_checks_inherited_masters_and_revokes_result(self):
        with self.request_fixture() as fixture:
            fixture.mocks['assert_worker_inputs'].side_effect = ValueError('private worker changed')
            with self.assertRaisesRegex(ValueError, 'private worker changed'):
                R.run_request(fixture.request)
            self.assertEqual(fixture.mocks['verify_inherited_inputs'].call_count, 2)
            result = json.loads((fixture.evidence / 'completion.json').read_bytes())
            self.assertFalse(result['result']['numerically_qualified'])
            self.assertFalse(result['inputs_unchanged'])

    def test_copy_failure_still_checks_inherited_masters_and_never_launches(self):
        with self.request_fixture() as fixture:
            fixture.mocks['prepare_worker'].side_effect = ValueError('copy admission failed')
            with self.assertRaisesRegex(ValueError, 'copy admission failed'):
                R.run_request(fixture.request)
            self.assertEqual(fixture.mocks['verify_inherited_inputs'].call_count, 2)
            fixture.mocks['configure_worker'].assert_not_called()
            fixture.mocks['cmd'].assert_not_called()

    def test_owned_timeout_cannot_preserve_a_qualified_mock_result(self):
        with self.request_fixture(worker_error=TimeoutError('inclusive cap')) as fixture:
            self.assertFalse(R.run_request(fixture.request)['numerically_qualified'])
            completion = json.loads((fixture.evidence / 'completion.json').read_bytes())
            self.assertFalse(completion['result']['numerically_qualified'])

    def test_cleanup_survivor_is_fatal_and_cannot_write_qualified_completion(self):
        with self.request_fixture(cleanup_survivors=[42]) as fixture:
            with self.assertRaisesRegex(RuntimeError, 'Fatal cleanup'):
                R.run_request(fixture.request)
            self.assertEqual(fixture.mocks['verify_inherited_inputs'].call_count, 2)
            self.assertFalse((fixture.evidence / 'completion.json').exists())


class RequestOrderingTests(NoChildTests):
    def test_run_keeps_six_request_order_and_acquires_lock_before_inclusive_cap(self):
        events = []
        deadlines = []
        locks = []
        def acquire(request):
            events.append(('acquire', request['id']))
            if request['lock_required']:
                lock = Mock()
                locks.append(lock)
                return lock
        @contextlib.contextmanager
        def deadline(seconds):
            deadlines.append(seconds)
            events.append(('deadline', seconds))
            yield
        def request(req):
            events.append(('request', req['id']))
            return {'numerically_qualified': False}
        with tempfile.TemporaryDirectory() as tmp:
            report = Path(tmp)
            (report / 'inherited-checkpoint.json').write_text('{}')
            with patch.object(R, 'REPORT', report), patch.object(R, 'frozen'), \
                    patch.object(R.base, 'acquire', side_effect=acquire), \
                    patch.object(R, 'run_request', side_effect=request), \
                    patch.object(R.L, 'Deadline', side_effect=deadline), \
                    patch.object(R.time, 'monotonic', return_value=0.):
                R.run()
            ids = [r['id'] for r in R.PLAN['requests']]
            self.assertEqual(events, [event for identity in ids for event in
                                      (('acquire', identity), ('deadline', 620), ('request', identity))])
            self.assertEqual(deadlines, [620] * 6)
            self.assertEqual(len(locks), 2)
            for lock in locks: lock.close.assert_called_once()
            summary = json.loads((report / 'summary.json').read_bytes())
            self.assertEqual([row['request']['id'] for row in summary['requests']], ids)
            self.assertEqual(summary['ancestor_failed_native_attempt'], R.BINDING['ancestor_index_sha256'])
            self.assertFalse(summary['performance_eligible'])

    def test_fatal_request_stops_before_any_later_request(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = Path(tmp)
            with patch.object(R, 'REPORT', report), patch.object(R, 'frozen'), \
                    patch.object(R.base, 'acquire', return_value=None), \
                    patch.object(R, 'run_request', side_effect=RuntimeError('fatal master guard')) as request, \
                    patch.object(R.L, 'Deadline', return_value=contextlib.nullcontext()), \
                    self.assertRaisesRegex(RuntimeError, 'fatal master guard'):
                R.run()
            request.assert_called_once_with(R.PLAN['requests'][0])
            aborted = json.loads((report / 'aborted-request.json').read_bytes())
            self.assertEqual(aborted['request']['id'], R.PLAN['requests'][0]['id'])
            self.assertFalse(aborted['numerically_qualified'])
            self.assertFalse((report / 'summary.json').exists())


class WorkerArchiveTests(NoChildTests):
    @contextlib.contextmanager
    def fixture(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'worker'
            directory = output / 'SubProcesses/P0_epem_mupmum'
            directory.mkdir(parents=True)
            (output / 'Cards').mkdir()
            (output / 'Cards/MadLoopParams.dat').write_text('frozen numerical card\n')
            (output / 'SubProcesses/MadLoop5_resources').mkdir()
            context = {'expected_worker': R.links.tree_snapshot(output)}
            yield output, directory, context

    def put(self, path, content='archived evidence\n'):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)

    def test_success_and_warmup_refusal_archives_are_typed_and_private(self):
        for phase in ('success', 'warmup-refusal'):
            with self.subTest(phase=phase), self.fixture() as (output, directory, context):
                session = directory / 'worker-raw/session-abcd123_'
                self.put(session / 'startup/startup.json', '{}\n')
                self.put(session / 'startup/warmup.dat')
                if phase == 'success':
                    self.put(session / 'previous-outputs/warmup.dat')
                    self.put(session / '000000/command.json', '{}\n')
                    for name in ('dimensions', 'warmup', 'timing', 'values', 'timed_status'):
                        self.put(session / ('000000/' + name + '.dat'))
                        self.put(directory / (name + '.dat'))
                    self.put(output / 'SubProcesses/MadLoop5_resources/ML5_0_HelFilter.dat')
                changed, added = R.allowed_worker_changes(output, context)
                relative = (session / 'startup/startup.json').relative_to(output).as_posix()
                self.assertEqual(added[relative], 'regular_file')
                self.assertEqual(added[session.relative_to(output).as_posix()], 'directory')
                self.assertIn('Cards/MadLoopParams.dat', changed)
                if phase == 'success':
                    self.assertEqual(added[(session / '000000/command.json').relative_to(output).as_posix()],
                                     'regular_file')
                self.assertTrue(all(kind in ('regular_file', 'directory') for kind in added.values()))

    def test_unexpected_archive_json_nesting_and_symlinks_are_rejected(self):
        for name in ('worker-raw/session-abcd123_/startup/unknown.json',
                     'worker-raw/startup.json',
                     'worker-raw/session-abcd123_/previous-outputs/command.json',
                     'worker-raw/session-abcd123_/000000/nested/values.dat',
                     'worker-raw/session-abcd123_/startup/warmup.dat'):
            with self.subTest(path=name), self.fixture() as (output, directory, context):
                path = directory / name
                if name.endswith('startup/warmup.dat'):
                    target = output.parent / 'inherited-raw'
                    target.write_text('must not read or write through this alias\n')
                    path.parent.mkdir(parents=True)
                    path.symlink_to(target)
                else:
                    self.put(path)
                with self.assertRaisesRegex(RuntimeError, 'Unexpected worker output'):
                    R.allowed_worker_changes(output, context)

    def test_archive_file_and_container_types_come_from_the_fixed_grammar(self):
        cases = [('values.dat', 'directory'),
                 ('worker-raw/session-abcd1234/startup/startup.json', 'directory'),
                 ('worker-raw', 'regular_file'),
                 ('worker-raw/session-abcd1234', 'regular_file')]
        for name, kind in cases:
            with self.subTest(path=name, kind=kind), self.fixture() as (output, directory, context):
                path = directory / name
                if kind == 'directory':
                    path.mkdir(parents=True)
                else:
                    self.put(path)
                with self.assertRaisesRegex(RuntimeError, 'Unexpected worker output'):
                    R.allowed_worker_changes(output, context)


class NativeMapIdentityTests(NoChildTests):
    def mapped_fixture(self, path, permissions='r-xp'):
        info = path.stat()
        device = f'{os.major(info.st_dev):02x}:{os.minor(info.st_dev):02x}'
        return f'1000-2000 {permissions} 00000000 {device} {info.st_ino} {path}\n'

    def test_bound_elf_passes_but_unknown_or_changed_elf_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'synthetic-elf'
            path.write_bytes(b'\x7fELF' + bytes(60))
            raw = self.mapped_fixture(path)
            self.assertTrue(N.classify_maps(raw, {str(path.resolve()): digest(path)})['exact_expected'])
            for expected in ({}, {str(path.resolve()): '0' * 64}):
                with self.subTest(expected=expected), \
                        self.assertRaisesRegex(ValueError, 'Unexpected ELF bytes/path'):
                    N.classify_maps(raw, expected)

    def test_unknown_non_elf_mapping_keeps_the_unchanged_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'unknown-data'
            path.write_bytes(b'unclassified immutable data')
            with self.assertRaisesRegex(ValueError, 'Unclassified non-ELF mapping'):
                N.classify_maps(self.mapped_fixture(path, 'r--p'), {})


if __name__ == '__main__':
    unittest.main(verbosity=2)
