"""Software controls for the bounded six-case timing authority; no native calls."""
from contextlib import contextmanager, ExitStack
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import signal
import tempfile
import tarfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / 'docs/benchmarkfolder/policy-v3/final-pins-20261021-v1/light-timing/light_pair.py'
module_spec = importlib.util.spec_from_file_location('tested_light_pair', SOURCE)
light = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(light)


class LightTiming(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=ROOT, prefix='.test-light-pair-')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def write(self, name, data):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data.encode() if isinstance(data, str) else data)
        return path

    def stream(self, name='parent.dat', nfinal=2, count=100):
        return self.write(name, str(count) + '\n' + ''.join(
            '%d %d 0 0\n' % (point, momentum)
            for point in range(count) for momentum in range(nfinal + 2)))

    def driver_fixture(self, nborn, nfinal=2):
        prefix = 'ML5_TEST_'
        def render(name):
            return (ROOT / 'tools' / name).read_text().replace('@PREFIX@', prefix).replace(
                '@NEXTERNAL@', str(nfinal + 2)).replace('@NBORN@', str(nborn)).encode()
        fresh, worker = render('ninjars_benchmark_driver.f'), render('ninjars_benchmark_worker.f')
        field = lambda raw: {'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}
        driver = 'SubProcesses/P0_test/check_sa.f'
        files = {driver: field(fresh), 'SubProcesses/P0_test/loop_num.f': field(b'original numerator')}
        anchor = {'compiled_manifest': {'files': files}, 'compilation': {'driver_sha256': field(fresh)['sha256']}}
        changed = copy.deepcopy(files); changed[driver] = field(worker)
        manifest = {'files': changed, 'files_sha256': light.bench.content_digest(changed)}
        return {'nfinal': nfinal}, anchor, {'driver_sha256': field(worker)['sha256']}, manifest, self.write('worker.f', worker)

    def raw_fixture(self, points=10, loops=3):
        directory = self.root / 'raw'; directory.mkdir()
        self.write('raw/timing.dat', '.01 .6\n')
        self.write('raw/dimensions.dat', '1 1 1\n')
        self.write('raw/values.dat', ''.join('%d %d 228 1 2 3 4 1e-12\n' % (p, o)
            for p in range(1, points + 1) for o in range(2)))
        self.write('raw/timed_status.dat', '228 %d\n' % (points * loops))
        parsed = light.workers.parse_outputs(directory, 8, points, loops)
        return {**parsed, 'backend': 8, 'loops': loops, 'raw_capture': {
            name: light.supplement.bound(directory / name)
            for name in (*light.workers.OUTPUTS, 'dimensions.dat')}}

    def test_fixed_prefix_preserves_event_bytes_and_declares_actual_ten(self):
        parent = self.stream(nfinal=1)
        prefix = light.prefix_bytes(parent, 1, 10)
        self.assertEqual(prefix, b'10\n' + b''.join(parent.read_bytes().splitlines(keepends=True)[1:31]))
        self.assertEqual(len(prefix.splitlines()), 31)

    def test_prefix_rejects_bad_count_or_malformed_parent_dimensions(self):
        parent = self.stream()
        for count in (0, 101, True, 10.):
            with self.subTest(count=count), self.assertRaises(ValueError): light.prefix_bytes(parent, 2, count)
        for data in (b'100\n0 0 0 0\n', parent.read_bytes() + b'1 2 3 4\n'):
            parent.write_bytes(data)
            with self.assertRaisesRegex(ValueError, 'dimensions'): light.prefix_bytes(parent, 2, 10)

    def test_worker_delta_accepts_born_and_loop_induced_actual_dimensions(self):
        for nborn in (0, 1, 3):
            with self.subTest(nborn=nborn): light.check_worker_delta(*self.driver_fixture(nborn))

    def test_worker_cannot_rebind_born_dimension_even_with_refreshed_manifest(self):
        case, anchor, build, manifest, worker = self.driver_fixture(1)
        data = worker.read_bytes().replace(b'NSQSO_BORN=1', b'NSQSO_BORN=0')
        self.assertNotEqual(data, worker.read_bytes()); worker.write_bytes(data)
        build['driver_sha256'] = hashlib.sha256(data).hexdigest()
        key = 'SubProcesses/P0_test/check_sa.f'
        manifest['files'][key] = {'sha256': build['driver_sha256'], 'bytes': len(data)}
        manifest['files_sha256'] = light.bench.content_digest(manifest['files'])
        with self.assertRaisesRegex(ValueError, 'original driver'):
            light.check_worker_delta(case, anchor, build, manifest, worker)

    def test_numerator_or_fresh_driver_changes_cannot_hide_in_new_hashes(self):
        case, anchor, build, manifest, worker = self.driver_fixture(0)
        manifest['files']['SubProcesses/P0_test/loop_num.f']['sha256'] = 'a' * 64
        manifest['files_sha256'] = light.bench.content_digest(manifest['files'])
        with self.assertRaisesRegex(ValueError, 'Numerical source'):
            light.check_worker_delta(case, anchor, build, manifest, worker)
        case, anchor, build, manifest, worker = self.driver_fixture(1)
        anchor['compilation']['driver_sha256'] = 'b' * 64
        with self.assertRaisesRegex(ValueError, 'original driver'):
            light.check_worker_delta(case, anchor, build, manifest, worker)

    def test_actual_ten_raw_rows_and_all_timed_evaluations_are_parsed(self):
        record = self.raw_fixture(); inputs = {}
        light.check_raw(record, 10, inputs)
        self.assertEqual(len(record['rows']), 20)
        self.assertEqual(record['timed_status_counts'], {'228': 30})
        self.assertEqual(len(inputs), 4)
        with self.assertRaisesRegex(RuntimeError, 'rows'): light.check_raw(record, 100, {})

    def test_refreshed_raw_custody_cannot_hide_missing_row_or_timed_count(self):
        record = self.raw_fixture()
        raw = self.root / 'raw/values.dat'; raw.write_text(''.join(raw.read_text().splitlines(True)[:-1]))
        record['raw_capture']['values.dat'] = light.supplement.bound(raw)
        with self.assertRaisesRegex(RuntimeError, 'rows'): light.check_raw(record, 10, {})
        record = self.raw_fixture_after_removing_old()
        raw = self.root / 'raw/timed_status.dat'; raw.write_text('228 100\n')
        record['raw_capture']['timed_status.dat'] = light.supplement.bound(raw)
        with self.assertRaisesRegex(RuntimeError, 'count'): light.check_raw(record, 10, {})

    def raw_fixture_after_removing_old(self):
        import shutil
        shutil.rmtree(self.root / 'raw')
        return self.raw_fixture()

    def test_published_raw_summary_and_capture_directory_are_bound(self):
        record = self.raw_fixture(); changed = copy.deepcopy(record)
        changed['seconds_per_point'] *= 2
        with self.assertRaisesRegex(ValueError, 'Published record'): light.check_raw(changed, 10, {})
        other = self.write('other/dimensions.dat', '1 1 1\n')
        record['raw_capture']['dimensions.dat'] = light.supplement.bound(other)
        with self.assertRaisesRegex(ValueError, 'Mixed raw'): light.check_raw(record, 10, {})

    def valid_spec_fixture(self):
        cases = [{'id': cid, 'nfinal': 1 if cid in ('qcd-1', 'ew-1', 'supplement-gg-h') else 2} for cid in light.CASE_IDS]
        config = {'cases': cases, 'points_per_case': 10, 'repetitions': 7, 'seed': 410083}
        archive = 'b' * 64
        runtime = light.policy.seal({'schema': 'reducer-runtime-signature-v2', 'libraries': {'libninjars.a': archive}})
        cohort = {'numerical_revision': 'revision', 'generation_revision': 'revision', 'generator_identity_sha256': 'generator',
            'runtime_identity_sha256': runtime['identity_sha256'], 'archive_sha256': archive}
        parents, prefixes, roots, origins = {}, {}, {}, {}
        card = light.supplement.bound(self.write('physical.dat', 'same physical card'))
        for case in cases:
            cid = case['id']; parent = self.stream(cid + '/parent.dat', case['nfinal'])
            parents[cid] = light.supplement.bound(parent)
            prefixes[cid] = light.supplement.bound(self.write(cid + '/ten.dat', light.prefix_bytes(parent, case['nfinal'], 10)))
            roots[cid] = {source: light.policy.relative(self.root / source) for source, _ in light.TARGETS}
            origins[cid] = {source: {'physical_card': card, 'reducer_runtime': runtime,
                'generation': {'source_identity': {'revision': 'revision', 'tree_state_sha256': 'generator'}},
                'compilation': {}, 'compiled_manifest': {}} for source, _ in light.TARGETS}
        production = {'tracked-source.f': 'a' * 64}
        parent_specs = {name: (self.write(name + '.json', json.dumps({'production_sources': production})), 'unused')
            for name in ('core', 'loop')}
        source = {'collector': light.supplement.bound(SOURCE)}
        proof = light.policy.seal({'schema': 'representative-policy-validation-v2',
            'reducer_runtimes': {'candidate_corrected/8': {runtime['identity_sha256']: {}}}})
        descriptor = light.supplement.bound(self.write('cohort.json', '{}'))
        qualification = light.supplement.bound(self.write('proof.json', json.dumps(proof)))
        seeds = {cid: 930083 if cid.startswith('supplement-') else 410083 for cid in light.CASE_IDS}
        spec = light.policy.seal({'schema': light.SCHEMA, 'configuration': config, 'origins': origins,
            'output_roots': roots, 'parent_points': parents, 'points': prefixes, 'sources': source,
            'pair_targets': [list(t) for t in light.TARGETS], 'role_map': light.ROLES,
            'repetitions': 7, 'minimum_seconds': .5, 'cpu': 207, 'wall_budget_seconds': 600., 'timeout': 120.,
            'sampling_label': light.SAMPLING_LABEL, 'parent_sampling_seeds': seeds,
            'old_full8_runtime_gate_required_for_completion': False, 'new_cuttools_or_collier_evaluations': 0,
            'historical_parent_specs_source_only': {k: light.supplement.bound(p) for k, (p, _) in parent_specs.items()},
            'cohort_descriptor': descriptor, 'runtime_qualification': qualification,
            'production_sources': production, 'candidate_cohort': cohort})
        stack = ExitStack(); self.addCleanup(stack.close)
        for name, value in (('parent_inputs', (config, origins, roots, parents)), ('source_files', source), ('parent_seeds', seeds)):
            stack.enter_context(mock.patch.object(light, name, return_value=value))
        stack.enter_context(mock.patch.object(light, 'PARENTS', parent_specs))
        stack.enter_context(mock.patch.object(light.pair, 'check_production'))
        stack.enter_context(mock.patch.object(light, 'committed_production', return_value=production))
        stack.enter_context(mock.patch.object(light.pair, 'candidate_cohort', return_value=cohort))
        stack.enter_context(mock.patch.object(light.release, 'validate_cohort', return_value=cohort))
        stack.enter_context(mock.patch.object(light.policy, 'original_identity'))
        return spec

    def test_exact_light_scope_and_distinct_parent_seeds_are_accepted(self):
        spec = self.valid_spec_fixture()
        self.assertEqual(light.validate_spec(spec, live=False)[0], spec)
        self.assertNotEqual(spec['parent_sampling_seeds']['qcd-1'], spec['parent_sampling_seeds']['supplement-gg-h'])

    def test_swapped_or_deceptively_rehashed_prefix_is_rejected(self):
        spec = self.valid_spec_fixture(); cid = 'qcd-2'
        path = light.policy.path(spec['points'][cid]['path'])
        lines = path.read_bytes().splitlines(keepends=True)
        lines[1:5], lines[5:9] = lines[5:9], lines[1:5]
        path.write_bytes(b''.join(lines)); spec['points'][cid] = light.supplement.bound(path)
        spec = light.policy.seal(spec)
        with self.assertRaisesRegex(ValueError, 'authorized parent prefix'): light.validate_spec(spec, live=False)

    def test_other_target_sampling_timer_or_source_plan_cannot_be_resealed(self):
        spec = self.valid_spec_fixture()
        for key, value in (('pair_targets', [['baseline', 1], ['candidate', 8]]), ('repetitions', 3),
                ('minimum_seconds', .05), ('wall_budget_seconds', 601.), ('cpu', 190), ('timeout', 121.),
                ('sampling_label', '100 point timing'), ('new_cuttools_or_collier_evaluations', 1),
                ('old_full8_runtime_gate_required_for_completion', True), ('sources', {})):
            changed = light.policy.seal({**spec, key: value})
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'plan changed'):
                light.validate_spec(changed, live=False)

    def test_matching_handoff_needs_quiet_release_evidence_and_literal_booleans(self):
        spec = self.write('spec.json', '{}'); evidence = light.supplement.bound(self.write('released.json', '{}'))
        value = {'root_authorized': True, 'phase': light.SCHEMA, 'cpu': 207,
            'specification_sha256': light.workers.digest(spec), 'quiet_cpu207': True,
            'all_other_numerical_and_compiler_work_released': True, 'release_evidence': [evidence]}
        handoff = self.write('handoff.json', json.dumps(value))
        self.assertEqual(light.validate_handoff(spec, handoff), value)
        for key, wrong in (('root_authorized', 1), ('quiet_cpu207', False), ('release_evidence', []),
                ('specification_sha256', 'f' * 64)):
            handoff.write_text(json.dumps({**value, key: wrong}))
            with self.subTest(key=key), self.assertRaises(ValueError): light.validate_handoff(spec, handoff)
        handoff.write_text(json.dumps(value)); light.policy.path(evidence['path']).write_text('changed')
        with self.assertRaises(ValueError): light.validate_handoff(spec, handoff)

    def test_snapshot_cannot_publish_partial_or_relabel_ten_as_hundred(self):
        report = self.root / 'report'; report.mkdir()
        config = {'points_per_case': 10}; spec = {'candidate_cohort': {}, 'sources': {}}
        spec_path = self.write('spec.json', '{}')
        descriptor = light.workers.protocol_descriptor()
        protocol = {'input_spec': light.supplement.bound(spec_path), 'configuration': config,
            'pair_targets': [list(t) for t in light.TARGETS], 'cpu': 207, 'repetitions': 7,
            'backends': [6, 8], 'sources': 'both', 'role_map': light.ROLES, 'fresh_parity': False,
            'candidate_cohort': {}, 'protocol_descriptor': descriptor,
            'protocol_identity_sha256': light.bench.content_digest(descriptor)}
        self.write('report/protocol.json', json.dumps(protocol))
        complete = {'status': 'complete', 'requested_cases': list(light.CASE_IDS),
            'completed_cases': list(light.CASE_IDS), 'actual_points': 10}
        with mock.patch.object(light, 'validate_spec', return_value=(spec, config)):
            for status in ({**complete, 'status': 'failed'}, {**complete, 'completed_cases': list(light.CASE_IDS[:-1])},
                    {**complete, 'actual_points': 100}):
                self.write('report/run-status.json', json.dumps(status))
                with self.subTest(status=status), self.assertRaisesRegex(ValueError, 'incomplete'):
                    light.snapshot(report)
            self.write('report/run-status.json', json.dumps(complete))
            self.write('report/protocol.json', json.dumps({**protocol, 'repetitions': 3}))
            with self.assertRaisesRegex(ValueError, 'protocol'): light.snapshot(report)

    def test_snapshot_requires_actual_full_dispatch_and_owned_terminal_receipts(self):
        report = self.root / 'report'; report.mkdir(); spec_path = self.write('spec.json', '{}')
        config = {'points_per_case': 10}; spec = {'candidate_cohort': {}, 'sources': {}}
        descriptor = light.workers.protocol_descriptor()
        self.write('report/protocol.json', json.dumps({'input_spec': light.supplement.bound(spec_path),
            'configuration': config, 'pair_targets': [list(t) for t in light.TARGETS], 'cpu': 207,
            'repetitions': 7, 'backends': [6, 8], 'sources': 'both', 'role_map': light.ROLES,
            'fresh_parity': False, 'candidate_cohort': {}, 'protocol_descriptor': descriptor,
            'protocol_identity_sha256': light.bench.content_digest(descriptor)}))
        self.write('report/run-status.json', json.dumps({'status': 'complete', 'requested_cases': list(light.CASE_IDS),
            'completed_cases': list(light.CASE_IDS), 'actual_points': 10}))
        self.write('report/dispatch.jsonl', '')
        with mock.patch.object(light, 'validate_spec', return_value=(spec, config)):
            with self.assertRaisesRegex(ValueError, 'complete adjacent alternating'): light.snapshot(report)
            self.write('report/process-ledger.jsonl', json.dumps({'event': 'spawn', 'pid': 12345,
                'pgid': 12345, 'start_ticks': 555}) + '\n')
            with mock.patch.object(light.observed, 'check_dispatch'), mock.patch.object(light, 'check_source_checkpoints'):
                with self.assertRaisesRegex(ValueError, 'terminal owned'): light.snapshot(report)

    def test_deadline_interrupt_unwinds_owned_context_and_retains_failed_status(self):
        work, report = self.root / 'work', self.root / 'report'
        spec = {'report': light.policy.relative(report), 'work': light.policy.relative(work), 'sources': {},
            'points': {}, 'candidate_cohort': {}, 'sampling_label': light.SAMPLING_LABEL,
            'output_roots': {'qcd-1': {}}, 'run_id': 'synthetic'}
        spec_path = self.write('spec.json', json.dumps(spec)); handoff = self.write('handoff.json', '{}')
        events, handlers, timers = [], {}, []
        @contextmanager
        def context(*args, **kwargs):
            events.append('enter')
            try: yield
            finally: events.append('exit')
        class Timer:
            def __init__(self, seconds, function): self.function = function; self.seconds = seconds; timers.append(self)
            def start(self): events.append('timer-start')
            def cancel(self): events.append('timer-cancel')
            def join(self, timeout): events.append('timer-join')
        def kill(pid, signum): handlers[signum](signum, None)
        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(light, 'validate_spec', return_value=(spec, {'cases': [{'id': 'qcd-1'}]})))
            stack.enter_context(mock.patch.object(light, 'validate_handoff'))
            stack.enter_context(mock.patch.object(light.os, 'sched_getaffinity', return_value={207}))
            stack.enter_context(mock.patch.dict(light.os.environ, {key: '1' for key in light.pair.THREADS}))
            stack.enter_context(mock.patch.object(light.pair, 'check_production'))
            stack.enter_context(mock.patch.object(light.subprocess, 'check_output', return_value='orchestration\n'))
            for owner, name in ((light.supplement, 'harness'), (light.supplement, 'input_roots'), (light.pair, 'owned_preparation')):
                stack.enter_context(mock.patch.object(owner, name, context))
            stack.enter_context(mock.patch.object(light.signal, 'getsignal', return_value='prior'))
            stack.enter_context(mock.patch.object(light.signal, 'signal', side_effect=lambda s, h: handlers.update({s: h})))
            stack.enter_context(mock.patch.object(light.os, 'kill', side_effect=kill))
            stack.enter_context(mock.patch.object(light.threading, 'Timer', Timer))
            stack.enter_context(mock.patch.object(light.pair, 'run_pair_case', side_effect=lambda *a: timers[0].function()))
            with self.assertRaisesRegex(TimeoutError, 'wall budget'): light.collect(spec_path, handoff)
        status = json.loads((report / 'run-status.json').read_text())
        self.assertEqual(status['status'], 'failed'); self.assertEqual(status['completed_cases'], [])
        self.assertEqual(events.count('enter'), 3); self.assertEqual(events.count('exit'), 3)
        self.assertIn('timer-cancel', events); self.assertIn('timer-join', events)
        self.assertTrue(all(value == 'prior' for value in handlers.values()))
        self.assertFalse((report / 'result.json').exists())

    def test_wrong_thread_limit_blocks_collection_before_report_or_worker_start(self):
        spec_path = self.write('spec.json', '{}'); handoff = self.write('handoff.json', '{}')
        report = self.root / 'report'
        spec = {'report': light.policy.relative(report)}
        environment = {key: '1' for key in light.pair.THREADS}
        environment['OMP_NUM_THREADS'] = '2'
        with mock.patch.object(light, 'validate_spec', return_value=(spec, {})), \
                mock.patch.object(light, 'validate_handoff'), \
                mock.patch.object(light.os, 'sched_getaffinity', return_value={207}), \
                mock.patch.dict(light.os.environ, environment), \
                mock.patch.object(light.pair, 'run_pair_case') as run:
            with self.assertRaisesRegex(ValueError, 'one-thread'): light.collect(spec_path, handoff)
        run.assert_not_called(); self.assertFalse(report.exists())

    def checkpoint_fixture(self):
        source = self.write('numerical.f', 'exact production bytes')
        production = {str(source.relative_to(ROOT)): light.workers.digest(source)}
        spec = {'production_sources': production, 'candidate_cohort': {
            'numerical_revision': 'numerical-commit', 'generator_identity_sha256': 'generator'}}
        event = {'index': 0, 'case': 'qcd-1', 'phase': 'measurement', 'event': 'start',
            'source': 'candidate', 'backend': 8, 'repetition': 0}
        with mock.patch.object(light.pair, 'check_production'), \
                mock.patch.object(light.subprocess, 'check_output', return_value='old-orchestration\n'):
            record = light.record_source_checkpoint(spec, self.root / 'report', event)
        return spec, {**event, 'source_checkpoint': record}

    def test_live_production_rejection_precedes_dispatch_and_checkpoint_publication(self):
        report = self.root / 'report'; report.mkdir()
        dispatch = light.GuardedDispatch(report, {})
        with mock.patch.object(light.pair, 'check_production', side_effect=ValueError('live production delta')):
            with self.assertRaisesRegex(ValueError, 'live production delta'):
                dispatch.event({'id': 'qcd-1'}, 'measurement', 'start', ('candidate', 8), 0)
        self.assertFalse((report / 'source-checkpoints').exists())
        self.assertEqual((report / 'dispatch.jsonl').read_text(), '')

    def test_observed_inventory_rechecks_actual_bytes_after_live_guard(self):
        source = self.write('numerical.f', 'actual changed bytes')
        spec = {'production_sources': {str(source.relative_to(ROOT)): 'a' * 64}}
        with mock.patch.object(light.pair, 'check_production'):
            with self.assertRaisesRegex(ValueError, 'after guard'):
                light.record_source_checkpoint(spec, self.root / 'report', {'index': 0})
        self.assertFalse((self.root / 'report').exists())

    def test_recorded_source_context_rejects_missing_or_rebound_before_after_fields(self):
        spec, event = self.checkpoint_fixture()
        with mock.patch.object(light, 'committed_production', return_value=spec['production_sources']):
            light.check_source_checkpoints([event], spec)
            missing = dict(event); del missing['source_checkpoint']
            with self.assertRaises((KeyError, ValueError)): light.check_source_checkpoints([missing], spec)
            path = light.policy.path(event['source_checkpoint']['path']); original = json.loads(path.read_text())
            for field, value in (('event', 'complete'), ('index', 1), ('case', 'ew-1'), ('repetition', 1),
                    ('backend', 6), ('numerical_revision', 'another-commit')):
                path.write_text(json.dumps({**original, field: value}))
                rebound = {**event, 'source_checkpoint': light.supplement.bound(path)}
                with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'context'):
                    light.check_source_checkpoints([rebound], spec)

    def test_historical_context_uses_recorded_commits_despite_current_head_change(self):
        spec, event = self.checkpoint_fixture()
        expected = spec['production_sources']; revisions = []
        def committed(revision):
            revisions.append(revision)
            self.assertIn(revision, ('numerical-commit', 'old-orchestration'))
            return expected
        with mock.patch.object(light, 'committed_production', side_effect=committed), \
                mock.patch.object(light.subprocess, 'check_output', side_effect=AssertionError('current HEAD must not be read')):
            light.check_source_checkpoints([event], spec)
        self.assertEqual(revisions, ['numerical-commit', 'old-orchestration'])
        with mock.patch.object(light, 'committed_production', side_effect=lambda rev:
                expected if rev == 'numerical-commit' else {'different.f': 'a' * 64}):
            with self.assertRaisesRegex(ValueError, 'historical production'): light.check_source_checkpoints([event], spec)

    def test_committed_inventory_resolves_historical_symlinks_without_live_files(self):
        archive = io.BytesIO(); data = b'historical committed numerical bytes'
        with tarfile.open(fileobj=archive, mode='w') as stream:
            member = tarfile.TarInfo('madgraph/original.py'); member.size = len(data)
            stream.addfile(member, io.BytesIO(data))
            link = tarfile.TarInfo('madgraph/alias.py'); link.type = tarfile.SYMTYPE; link.linkname = 'original.py'
            stream.addfile(link)
        commands = []
        def git(command, **kwargs):
            commands.append(command)
            self.assertIn('historical', command); self.assertNotIn('HEAD', command)
            if command[1] == 'ls-tree': return b'madgraph/original.py\0madgraph/alias.py\0'
            if command[1] == 'archive': return archive.getvalue()
            self.fail('Unexpected source command')
        with mock.patch.object(light.subprocess, 'check_output', side_effect=git):
            inventory = light.committed_production.__wrapped__('historical')
        digest = hashlib.sha256(data).hexdigest()
        self.assertEqual(inventory, {'madgraph/original.py': digest, 'madgraph/alias.py': digest})
        self.assertEqual(len(commands), 2)


if __name__ == '__main__': unittest.main()
