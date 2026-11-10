"""Pure fault controls and a fake runner: no compiler, loader or native calls."""
from contextlib import contextmanager
from pathlib import Path
from itertools import combinations
from unittest import mock
import copy
import json
import tempfile
import unittest
import prefix_reader as reader
import run as control


class MemoryCapsule:
    def __init__(self):
        self.files = {}
        names = [reader.worker_name(spec) for spec in reader.PREFIX_SPECS]
        self.inventory = {'actual_attempt_directories': names + ['closed-runtime'],
                          'original_files': {}, 'actual_attempt_symlinks': {}}
        self.runtime = {'/original/lib%d.so' % i: str(i) * 64 for i in range(15)}
        self.binding = {'runtime': self.runtime, 'target_interpreter': '/original/lib0.so'}
        manifest = {'entries': [{'resolved_path': path, 'source_path': path, 'sha256': sha,
                                'soname': Path(path).name} for path, sha in self.runtime.items()]}
        old_binding = {**self.binding, 'native_library_path': '/original/run4/closed-runtime'}
        self.inventory['actual_attempt_symlinks'] = {'closed-runtime/' + e['soname']: e['resolved_path']
                                                    for e in manifest['entries']}
        self.put('controller/source-index.json', {})
        self.put('controller/bindings.json', old_binding)
        self.put('controller/donor-closure.json', manifest)
        self.put('controller/inherited-build2.json', {**self.binding, 'executable': '/original/build2/diagnostic'})
        self.put('actual-run4/result.json', {'passed': False, 'workers_completed': 9,
            'failure': reader.ORIGINAL_FAILURE, 'elapsed_seconds': float(reader.OLD_ELAPSED),
            'recompiled': False, 'original_donor_runtime_preserved': True,
            'source_index_sha256': reader.digest(self.raw('controller/source-index.json')),
            'build_sha256': reader.digest(self.raw('controller/inherited-build2.json'))})
        self.fixtures = {'fixtures/selected-inputs.json': reader.digest(b'fixed-fixture')}
        self.put('controller/fixtures/selected-inputs.json', b'fixed-fixture')
        alias_proof = {'passed': True, 'library_count': 15, 'alias_directory': old_binding['native_library_path'],
                       'libraries': [{**e, 'alias_target': e['resolved_path']} for e in manifest['entries']]}
        self.put('actual-run4/closed-runtime-proof.json', alias_proof)
        loader = {'native': False, 'returncode': 0, 'process_reaped': True, 'seconds_cap': 30,
                  'target_loader_environment_at_execve': {'LD_LIBRARY_PATH': old_binding['native_library_path'],
                                                          'LD_TRACE_LOADED_OBJECTS': '1'}}
        for prefix in ['actual-run4'] + ['actual-run4/' + name for name in names]:
            self.put(prefix + '/loader.json', loader)
            self.put(prefix + '/loader.log', '\n'.join(self.runtime).encode())
        workers = []
        for spec in reader.PREFIX_SPECS:
            name = reader.worker_name(spec)
            prefix = 'actual-run4/' + name
            for leaf in ('closed-runtime-pretrace.json', 'closed-runtime-preworker.json'):
                self.put(prefix + '/' + leaf, alias_proof)
            maps = b'original runtime map'
            execution = {'native': True, 'returncode': 0, 'process_spawned': True, 'process_reaped': True,
                         'seconds_cap': 180, 'cwd': '/original/run4/' + name,
                         'target_loader_environment_at_execve': {'LD_LIBRARY_PATH': old_binding['native_library_path']},
                         'mapping_capture': {'complete_snapshot_sha256': reader.digest(maps)}}
            self.put(prefix + '/worker.json', execution)
            self.put(prefix + '/maps.txt', maps)
            self.put(prefix + '/mapped-runtime.json', {'expected_runtime_sha256': self.runtime,
                'mapped_files_sha256_after_exit': self.runtime,
                'mapped_elf_paths': list(self.runtime) + ['/original/build2/diagnostic'],
                'executable': '/original/build2/diagnostic', 'cpu_allowed_list': '12',
                'maps_text': maps.decode(), 'maps_sha256': reader.digest(maps)})
            self.put(prefix + '/inputs.json', {})
            self.put(prefix + '/outputs.json', {})
            trace, stdout = ('trace-' + name).encode(), ('stdout-' + name).encode()
            self.put(prefix + '/trace.txt', trace)
            self.put(prefix + '/worker.log', stdout)
            self.put(prefix + '/warnings.json', [])
            parsed = {'backend': spec[0], 'mode': spec[1], 'observed': spec[2], 'host_checked': False,
                      'public_calls': 5, 'diagnostic_calls': 3, 'warnings': [], 'public_raw': ['same-pair'],
                      'trace_sha256': reader.digest(trace), 'stdout_sha256': reader.digest(stdout)}
            parsed['points'] = [{'point': p, 'status': 120, 'accuracy': [-1., -1.], 'host_checked': False,
                                'inputs': {'shared_input_sha256': reader.digest(str(p).encode())} if spec[2] else None}
                               for p in (1, 93, 39)]
            workers.append({'worker': name, 'trace_sha256': reader.digest(trace),
                            'stdout_sha256': reader.digest(stdout), 'parsed': parsed})
        self.preflight = {'passed': True, 'parser_sha256': 'parser-pin', 'fixture_sha256': self.fixtures,
            'closed_failed_run4_retention_index_sha256': reader.PREFIX_SHA,
            'candidate_parser_full_workers_passed': 10, 'matrix_workers_required': 16,
            'original_source4_attempt_remains_failed': True, 'original_source4_parser_failure': reader.ORIGINAL_FAILURE,
            'adjudication_executed': False, 'scientific_acceptance': False,
            'public_plain_observed_bytes_equal': True, 'all_partial_common_inputs_equal': True, 'workers': workers}
        checks=[]
        for left,right in combinations([w['parsed'] for w in workers if w['parsed']['observed']],2):
            if left['backend']!=right['backend'] and left['mode']!=right['mode']:continue
            checks.extend({'left':[left['backend'],left['mode']],'right':[right['backend'],right['mode']],
                           'point':p,'shared_input_equal':True} for p in (1,93,39))
        self.preflight['partial_common_input_checks']=checks

    def put(self, name, value):
        self.files[name] = value if isinstance(value, bytes) else json.dumps(value).encode()
        self.inventory['original_files'][name] = {}

    def raw(self, name):
        return self.files[name]

    def json(self, name):
        return reader.loads(self.raw(name))

    def validate(self):
        return reader.validate(self, self.preflight, 'parser-pin', self.fixtures, self.binding)


class PrefixTests(unittest.TestCase):
    def setUp(self):
        self.c = MemoryCapsule()

    def test_complete_ten_prefix_has_exact_unexpanded_budget(self):
        value = self.c.validate()
        self.assertEqual(len(value['workers']), 10)
        self.assertEqual(value['new_phase_budget_seconds_decimal'], '2847.43727759690955')
        self.assertEqual(reader.NEW_BUDGET + reader.OLD_ELAPSED, reader.TOTAL_BUDGET)

    def reject(self, mutate):
        mutate(self.c)
        with self.assertRaises((ValueError, KeyError)):
            self.c.validate()

    def test_missing_prefix_worker(self):
        self.reject(lambda c: c.preflight['workers'].pop())

    def test_extra_prefix_worker(self):
        self.reject(lambda c: c.preflight['workers'].append(copy.deepcopy(c.preflight['workers'][0])))

    def test_duplicate_prefix_worker(self):
        self.reject(lambda c: c.preflight['workers'].__setitem__(1, copy.deepcopy(c.preflight['workers'][0])))

    def test_directory_extra_or_missing(self):
        for mutate in (lambda c: c.inventory['actual_attempt_directories'].append('b9-m4-o0'),
                       lambda c: c.inventory['actual_attempt_directories'].remove('b8-m4-o1')):
            self.setUp(); self.reject(mutate)

    def test_started_worker_inventory_extra(self):
        self.reject(lambda c: c.inventory['original_files'].__setitem__('actual-run4/b9-m4-o0/worker.json', {}))

    def test_original_result_status_or_budget_cannot_change(self):
        for key, value in [('passed', True), ('workers_completed', 10), ('elapsed_seconds', 1),
                           ('recompiled', True), ('failure', None)]:
            self.setUp()
            self.reject(lambda c, k=key, v=value: c.put('actual-run4/result.json', {**c.json('actual-run4/result.json'), k: v}))

    def test_prefix_seal_parser_fixture_or_scope_cannot_change(self):
        for key, value in [('closed_failed_run4_retention_index_sha256', 'bad'), ('parser_sha256', 'bad'),
                           ('fixture_sha256', {}), ('adjudication_executed', True), ('scientific_acceptance', True),
                           ('candidate_parser_full_workers_passed', 9), ('public_plain_observed_bytes_equal', False),
                           ('all_partial_common_inputs_equal',False),('partial_common_input_checks',[])]:
            self.setUp(); self.reject(lambda c, k=key, v=value: c.preflight.__setitem__(k, v))

    def test_raw_trace_stdout_parsed_identity_parity_tamper(self):
        mutations = [lambda c: c.put('actual-run4/b1-m4-o0/trace.txt', b'tampered'),
                     lambda c: c.put('actual-run4/b1-m4-o0/worker.log', b'tampered'),
                     lambda c: c.preflight['workers'][0]['parsed'].__setitem__('backend', 9),
                     lambda c: c.preflight['workers'][0]['parsed'].__setitem__('host_checked', True),
                     lambda c: c.preflight['workers'][0]['parsed'].__setitem__('public_raw', ['different'])]
        for mutate in mutations:
            self.setUp(); self.reject(mutate)

    def test_reaped_returncode_runtime_map_and_alias_tamper(self):
        for leaf, key, value in [('worker.json', 'process_reaped', False), ('worker.json', 'returncode', 1),
                                 ('mapped-runtime.json', 'cpu_allowed_list', '13'),
                                 ('mapped-runtime.json', 'mapped_files_sha256_after_exit', {})]:
            self.setUp()
            name = 'actual-run4/b1-m4-o0/' + leaf
            self.reject(lambda c, n=name, k=key, v=value: c.put(n, {**c.json(n), k: v}))
        self.setUp(); self.reject(lambda c: c.inventory['actual_attempt_symlinks'].clear())

    def test_inherited_point_ids_status_accuracy_hostflag_common_input_tamper(self):
        for key,value in [('point',2),('status',110),('accuracy',[0.,0.]),('host_checked',True)]:
            self.setUp();self.reject(lambda c,k=key,v=value:c.preflight['workers'][0]['parsed']['points'][0].__setitem__(k,v))
        self.setUp()
        self.reject(lambda c:c.preflight['workers'][1]['parsed']['points'][0]['inputs'].__setitem__('shared_input_sha256','f'*64))

    def test_six_suffix_and_exact_final_matrix(self):
        self.assertEqual(reader.SUFFIX_SPECS, ((8,5,0),(8,5,1),(9,4,0),(9,4,1),(9,5,0),(9,5,1)))
        workers = [{'backend': b, 'mode': m, 'observed': o} for b, m, o in reader.MATRIX_SPECS]
        reader.require_complete_matrix(workers)
        for changed in (workers[:-1], workers + [workers[0]], workers[:-1] + [workers[0]], workers[::-1]):
            with self.assertRaises(ValueError): reader.require_complete_matrix(changed)

    def test_outer_capsule_seal_and_member_tamper(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = b'{"original_files":{}}'
            (root/'retention-inventory.json').write_bytes(raw)
            index = {'schema': 'qcd-n3-genuine-qp-failed-run4-retention-v1', 'files': {
                'retention-inventory.json': {'bytes': len(raw), 'sha256': reader.digest(raw)}}}
            encoded = json.dumps(index).encode(); (root/'evidence-index.json').write_bytes(encoded)
            with self.assertRaisesRegex(ValueError, 'index changed'): reader.Capsule(root)
            with mock.patch.object(reader, 'PREFIX_SHA', reader.digest(encoded)):
                reader.Capsule(root)
                (root/'unlisted.txt').write_bytes(b'extra')
                with self.assertRaisesRegex(ValueError,'outer membership'):reader.Capsule(root)
                (root/'unlisted.txt').unlink()
                (root/'directory-alias').symlink_to(root,target_is_directory=True)
                with self.assertRaisesRegex(ValueError,'symlinked'):reader.Capsule(root)
                (root/'directory-alias').unlink()
                (root/'retention-inventory.json').write_bytes(raw+b' ')
                with self.assertRaisesRegex(ValueError, 'member changed'): reader.Capsule(root)

    def test_capsule_symlink_ancestor_rejected_before_payload_read(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'real').mkdir();(root/'alias').symlink_to(root/'real',target_is_directory=True)
            with self.assertRaisesRegex(ValueError,'ancestor is symlinked'):reader.Capsule(root/'alias'/'missing-capsule')

    def test_duplicate_json_keys_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Duplicate JSON'): reader.loads('{"worker":1,"worker":2}')

    def test_load_rejects_preflight_parser_or_fixture_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            here=Path(directory);(here/'fixtures').mkdir()
            (here/'fixtures/selected-inputs.json').write_bytes(b'fixed-fixture')
            (here/'analyze.py').write_bytes(b'fixed-parser')
            preflight=copy.deepcopy(self.c.preflight)
            preflight['parser_sha256']=reader.digest(b'fixed-parser')
            encoded=json.dumps(preflight).encode()
            (here/'closed-run4-preflight.json').write_bytes(encoded)
            (here/'source-index.json').write_text(json.dumps({'files':{'closed-run4-preflight.json':{
                'bytes':len(encoded),'sha256':reader.digest(encoded)}}}))
            with mock.patch.object(reader,'Capsule',return_value=self.c):
                self.assertEqual(len(reader.load(here,self.c.binding)['workers']),10)
                (here/'closed-run4-preflight.json').write_bytes(encoded+b' ')
                with self.assertRaisesRegex(ValueError,'preflight bytes changed'):reader.load(here,self.c.binding)
                (here/'closed-run4-preflight.json').write_bytes(encoded)
                (here/'analyze.py').write_bytes(b'changed-parser')
                with self.assertRaisesRegex(ValueError,'parser or fixture'):reader.load(here,self.c.binding)
                (here/'analyze.py').write_bytes(b'fixed-parser')
                (here/'fixtures/selected-inputs.json').write_bytes(b'changed-fixture')
                with self.assertRaisesRegex(ValueError,'fixture'):reader.load(here,self.c.binding)


class ControllerTests(unittest.TestCase):
    def fake_run(self, directory, *, fail_at=None, clock=lambda: 0):
        root=Path(directory); work=root/'work'; work.mkdir()
        (root/'donor-closure.json').write_text('{}')
        inherited=MemoryCapsule().validate()
        proof={'executable_sha256':'x','source_index_sha256':'source2'}
        calls=[]
        @contextmanager
        def unlocked(): yield
        def command(argv, folder, label, **kwargs):
            (folder/(label+'.log')).write_text('fake loader')
        def worker(spec, attempt, **kwargs):
            calls.append(spec)
            if spec==fail_at: raise ValueError('fake worker failure')
            return {'backend':spec[0], 'mode':spec[1], 'observed':spec[2]}
        patches=[mock.patch.object(control,'HERE',root), mock.patch.object(control,'WORK',work),
            mock.patch.object(control,'numerical_lock',unlocked), mock.patch.object(control,'source_check',return_value={'runtime':{}}),
            mock.patch.object(control,'inherited_build',return_value=(root/'build.json',proof,root/'exe')),
            mock.patch.object(control,'sha',return_value='fakehash'), mock.patch.object(control.prefix_reader,'load',return_value=inherited),
            mock.patch.object(control.donor_closure,'create',return_value={'passed':True}),
            mock.patch.object(control,'target_command',return_value=['fake-only']), mock.patch.object(control,'run_command',side_effect=command),
            mock.patch.object(control,'validate_loader'),mock.patch.object(control,'run_worker',side_effect=worker),
            mock.patch.object(control.analyze,'adjudicate',return_value={'all_shared_input_gates_pass':True}),
            mock.patch.object(control.time,'monotonic',side_effect=clock)]
        from contextlib import ExitStack
        with ExitStack() as stack:
            for patch in patches: stack.enter_context(patch)
            if fail_at is not None:
                with self.assertRaisesRegex(ValueError,'fake worker failure'):control.run()
            else:control.run()
        return calls,json.loads((work/'run5/result.json').read_text())

    def test_fake_runner_launches_only_literal_six_then_adjudicates(self):
        with tempfile.TemporaryDirectory() as directory:
            calls,result=self.fake_run(directory)
        self.assertEqual(calls,list(reader.SUFFIX_SPECS))
        self.assertEqual((result['inherited_workers'],result['new_workers_parsed'],result['workers_completed']),(10,6,16))
        self.assertTrue(result['passed']);self.assertTrue(result['original_source4_attempt_remains_failed'])
        self.assertEqual(result['combined_elapsed_seconds'],float(reader.OLD_ELAPSED))

    def test_fake_runner_stops_first_failure_without_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            calls,result=self.fake_run(directory,fail_at=reader.SUFFIX_SPECS[1])
        self.assertEqual(calls,list(reader.SUFFIX_SPECS[:2]))
        self.assertEqual((result['inherited_workers'],result['new_workers_parsed']),(10,1))
        self.assertFalse(result['passed'])

    def test_busy75_does_not_create_run5(self):
        @contextmanager
        def busy(): raise SystemExit(75);yield
        with tempfile.TemporaryDirectory() as directory,mock.patch.object(control,'WORK',Path(directory)),\
             mock.patch.object(control,'numerical_lock',busy),mock.patch.object(control,'source_check') as check:
            with self.assertRaises(SystemExit) as stopped:control.run()
            self.assertEqual(stopped.exception.code,75);check.assert_not_called()
            self.assertFalse((Path(directory)/'run5').exists())

    def test_prefix_validation_counts_against_phase_budget(self):
        calls=iter([0,float(reader.NEW_BUDGET)+1,float(reader.NEW_BUDGET)+1])
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(TimeoutError,'prefix verification'):
                self.fake_run(directory,clock=lambda:next(calls))
            result=json.loads((Path(directory)/'work/run5/result.json').read_text())
            self.assertFalse(result['passed']);self.assertEqual(result['new_workers_parsed'],0)

    def test_combined_budget_guard_also_rejects_accounting_drift(self):
        # Inject drift into the retained-time accounting only. The phase stays
        # below its cap; the independent combined-budget guard must still fail.
        from decimal import Decimal
        ticks=iter([0]*9+[2847])
        with tempfile.TemporaryDirectory() as directory,mock.patch.object(reader,'OLD_ELAPSED',Decimal('200')):
            _,result=self.fake_run(directory,clock=lambda:next(ticks))
        self.assertLess(result['elapsed_seconds'],result['new_phase_budget_seconds'])
        self.assertGreater(result['combined_elapsed_seconds'],3000)
        self.assertFalse(result['passed'])


if __name__=='__main__': unittest.main(verbosity=2)
