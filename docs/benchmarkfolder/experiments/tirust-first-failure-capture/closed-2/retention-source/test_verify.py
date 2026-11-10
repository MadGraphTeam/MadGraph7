"""Pure synthetic retention parsing; no native process, source admission or I/O."""
import importlib.util
import json
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('capture_retention_verify', Path(__file__).with_name('verify.py'))
V = importlib.util.module_from_spec(spec)
spec.loader.exec_module(V)


def input_blob(case_id=1, method=1, group=None):
    case = V.SPECS[case_id - 1]
    cid, _, point, groups, n, rank = case
    h = [cid, 1, point, method, group or groups[0], n, rank,
         4 * n, n * n, n, 4 * V.math.comb(rank + 4, 4), 0, 1, int(method == 2), 0]
    return ('MLTCAP_INPUT_V1\n' + ' '.join(map(str, h)) + '\n' +
            '8000000000000000\n' * (1 + 4 * n + n * n + 2 * n)).encode('ascii')


def event(kind, **updates):
    row = dict(record_type=kind, case_id=1, phase=1, call=77, point=25,
               method=1, mode=1, group=59, sqso=1, n=5, rank=4)
    row.update(updates)
    return row


def event_blob(*rows):
    init = dict(record_type='init', case_id=1, schema=1, input_limit_per_case=6,
                tensor_output_read_policy='status0_only')
    return ('\n'.join(json.dumps(row) for row in (init,) + rows) + '\n').encode()


def frozen_controller():
    # Frozen source only. Neither candidate points nor native work are opened.
    path = Path(__file__).resolve().parent.parent / 'source-proposal-1/controller.py'
    return path.read_bytes()


def replay_row(stage, repetition, status=0, count=0, **updates):
    row = dict(record_type=stage,
               input_key=dict(case_id=1, phase=1, point=25, method=1, group=59),
               input_path='/original-path-is-an-identifier-only/input.dat',
               repetition=repetition, status=status, evaluation_count=count,
               context_present=True, error_required=1, error_truncated=False,
               error_encoding='latin1-bytes', error='')
    if stage == 'destroy': row.update(context_present=False, destroy_succeeded=status == 0)
    row.update(updates)
    return row


def replay_manifest():
    return [dict(case_id=1, phase=1, point=25, method=1, group=59, rank=4,
                 state='captured', path='/original-path-is-an-identifier-only/input.dat')]


def replay_blob(rows):
    return ('\n'.join(json.dumps(row) for row in rows) + '\n').encode()


class InputTests(unittest.TestCase):
    def test_all_twelve_slots_and_raw_negative_zero(self):
        self.assertEqual(len(V.expected_slots()), 12)
        for slot in V.expected_slots():
            parsed = V.parse_input(input_blob(slot['case_id'], slot['method'], slot['group']))
            self.assertEqual(V.key(parsed['input_key']), V.key(slot))
            self.assertEqual(parsed['mu2_bits'], '8000000000000000')
            self.assertEqual(len(parsed['mass_bits']), 2 * slot['nprop'])

    def test_no_missing_extra_nonascii_or_cr_records(self):
        good = input_blob()
        for bad in (good[:-1], good + b'\n', good + b'junk\n', good.replace(b'\n', b'\r\n'),
                    good.replace(b'8000000000000000', b'\xff000000000000000', 1), good[:-18]):
            with self.subTest(bad=bad[-20:]), self.assertRaises(ValueError): V.parse_input(bad)

    def test_unregistered_keys_shapes_and_controls_rejected(self):
        good = input_blob(); lines = good.decode().splitlines()
        for column, value in ((0, 0), (1, 0), (2, 24), (3, 4), (4, 60), (5, 6),
                              (6, 5), (7, 21), (8, 24), (9, 4), (10, 281), (11, 1),
                              (12, 0), (13, 1), (14, 1)):
            header = lines[1].split(); header[column] = str(value)
            bad = '\n'.join([lines[0], ' '.join(header)] + lines[2:]) + '\n'
            with self.subTest(column=column), self.assertRaises(ValueError): V.parse_input(bad.encode())

    def test_duplicate_json_keys_and_nonstandard_constants_rejected(self):
        for bad in (b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":Infinity}'):
            with self.assertRaises(ValueError): V.json_value(bad)


class CaptureTests(unittest.TestCase):
    def test_fortran_guard_without_c_input_stays_missing(self):
        raw = event_blob(event('group_enter', entry_buffers_valid=True,
                               pl_bits=['0' * 16] * 20, mass_bits=['0' * 16] * 10),
                         event('fortran_stage', stage=1, detail=0),
                         event('group_end', stable=False))
        result = V.parse_capture_events(raw, 1)
        self.assertEqual(len(result['primary_slots']), 6)
        self.assertFalse(any(slot['c_boundaries'] for slot in result['primary_slots']))

    def test_failed_c_boundary_has_error_but_no_output_read(self):
        row = event('c_boundary', status=1, forwarded_ordinal=88, input_shape_valid=True,
                    full_request=True, error_required=7, error_truncated=False,
                    error_encoding='latin1-bytes', error='failed')
        raw = event_blob(event('group_enter', entry_buffers_valid=False), row,
                         event('group_end', stable=False))
        result = V.parse_capture_events(raw, 1)
        self.assertEqual(len(result['primary_slots'][0]['c_boundaries']), 1)
        row['output_bits'] = ['0' * 16] * 560
        with self.assertRaises(ValueError):
            V.parse_capture_events(event_blob(event('group_enter', entry_buffers_valid=False), row), 1)

    def test_success_requires_complete_words_and_partial_lifecycle_is_retained(self):
        row = event('c_boundary', status=0, forwarded_ordinal=2, input_shape_valid=True,
                    full_request=True, error_required=1, error_truncated=False,
                    error_encoding='latin1-bytes', error='', output_bits=['0' * 16] * 560)
        parsed = V.parse_capture_events(event_blob(event('group_enter', entry_buffers_valid=False), row), 1)
        self.assertIsNotNone(parsed['open_group'])
        row['output_bits'].pop()
        with self.assertRaises(ValueError):
            V.parse_capture_events(event_blob(event('group_enter', entry_buffers_valid=False), row), 1)

    def test_error_capacity_truncation_and_binary_bytes(self):
        row = dict(error_required=5000, error_truncated=True, error_encoding='latin1-bytes',
                   error='\x00\xff' + 'a' * 4093)
        V.error_copy(row, exact=True)
        row['error_truncated'] = False
        with self.assertRaises(ValueError): V.error_copy(row, exact=True)

    def test_lifecycle_and_key_mismatch_rejected(self):
        for raw in (event_blob(event('group_end', stable=True)),
                    event_blob(event('group_enter', entry_buffers_valid=False),
                               event('group_enter', entry_buffers_valid=False)),
                    event_blob(event('group_enter', entry_buffers_valid=False, method=2, mode=1))):
            with self.assertRaises(ValueError): V.parse_capture_events(raw, 1)


class FileTests(unittest.TestCase):
    def test_file_hashes_sizes_and_complete_inventory(self):
        blobs = {'index.json': b'index', 'retained.txt': b'abc'}
        def read(name): return blobs[name]
        read.paths = set(blobs)
        index = dict(schema='madloop-tirust-first-failure-capture-closed-v1',
                     files={'retained.txt': dict(sha256=V.sha(b'abc'), bytes=3)})
        self.assertEqual(V.verify_files(index, read), {'retained.txt': b'abc'})
        read.paths.add('unexpected')
        with self.assertRaises(ValueError): V.verify_files(index, read)
        read.paths.remove('unexpected'); blobs['retained.txt'] = b'abd'
        with self.assertRaises(ValueError): V.verify_files(index, read)

    def test_no_original_absolute_or_parent_traversal_open(self):
        for name in ('/original/point.dat', '../point.dat', 'a/../point.dat', 'a//b', 'a\\b'):
            with self.assertRaises(ValueError): V.logical_path(name)

    def test_external_terminal_observation_binds_exact_bytes(self):
        proof = dict(observed_terminal=True, session_id=62191, exit_code=1, sha256=V.sha(b'failed'))
        V.terminal_confirmation(proof, b'failed')
        for changed in (dict(proof, observed_terminal=False), dict(proof, exit_code=True),
                        dict(proof, sha256=V.sha(b'changed'))):
            with self.assertRaises(ValueError): V.terminal_confirmation(changed, b'failed')


class ClosedStageTests(unittest.TestCase):
    def empty_capture(self):
        prepare = json.dumps(dict(source_index_sha256=V.SOURCE_INDEX_SHA256,
                                  status='passed', amplitude_evaluations=0)).encode()
        results = []
        for cid, case, *_ in V.SPECS:
            inputs = [{k: v for k, v in slot.items() if k != 'phase'} | dict(state='missing', replayed=False)
                      for slot in V.expected_slots() if slot['case_id'] == cid]
            results.append(dict(case=case, state='unexecuted', native_started=False, inputs=inputs))
        capture = json.dumps(dict(source_index_sha256=V.SOURCE_INDEX_SHA256,
                                  preparation_sha256=V.sha(prepare), status='failed',
                                  maximum_unique_inputs=12, maximum_standalone_evaluations=24,
                                  missing_inputs_are_not_replaced=True, automatic_retry=False,
                                  results=results)).encode()
        blobs = {'receipts/prepare-terminal.json': prepare, 'receipts/capture-terminal.json': capture}
        registry = dict(originals={}, replay_status='not-executed',
                        capture_terminal_confirmation=dict(observed_terminal=True, session_id=62191,
                                                           exit_code=1, sha256=V.sha(capture)))
        return blobs, registry

    def test_unexecuted_capture_and_replay_keep_all_twelve_missing(self):
        blobs, registry = self.empty_capture()
        result = V.derive_closed(blobs, registry)
        self.assertEqual(result['expected_input_slots'], 12)
        self.assertEqual(result['missing_or_unavailable_input_slots'], 12)
        self.assertEqual(result['replay']['evaluate_count'], 0)
        self.assertFalse(result['numerical_acceptance_claimed'])

    def test_pre_admission_failure_is_not_a_native_tensor_result(self):
        blobs, registry = self.empty_capture()
        name = 'receipts/replay-admission-failure-1.txt'; blobs[name] = b'export inventory differs\n'
        registry.update(replay_status='pre-admission-failed', replay_admission_confirmation=dict(
            observed_terminal=True, session_id=1059, exit_code=1, logical=name,
            sha256=V.sha(blobs[name]), native_started=False))
        result = V.derive_closed(blobs, registry)
        self.assertEqual(result['replay']['status'], 'pre-admission-failed')
        self.assertEqual(result['replay']['evaluate_count'], 0)
        registry['replay_admission_confirmation']['native_started'] = True
        with self.assertRaises(ValueError): V.derive_closed(blobs, registry)

    def test_absent_replay_cannot_hide_started_or_native_files(self):
        blobs, registry = self.empty_capture()
        blobs['receipts/replay-started.json'] = b'{}'
        with self.assertRaises(ValueError): V.derive_closed(blobs, registry)
        blobs.pop('receipts/replay-started.json'); blobs['replay/some/output.jsonl'] = b''
        with self.assertRaises(ValueError): V.derive_closed(blobs, registry)

    def test_complete_portable_callback_never_opens_original_paths(self):
        blobs, registry = self.empty_capture()
        source = Path(__file__).resolve().parent.parent / 'source-proposal-1'
        seal = V.json_value((source / 'source-index.json').read_bytes())
        blobs['source/source-index.json'] = (source / 'source-index.json').read_bytes()
        for name in seal['files']:
            blobs['source/' + name] = (source / name).read_bytes()
        blobs['retention.json'] = json.dumps(registry).encode()
        index = dict(schema='madloop-tirust-first-failure-capture-closed-v1',
                     files={name: dict(sha256=V.sha(blob), bytes=len(blob)) for name, blob in blobs.items()})
        blobs['index.json'] = json.dumps(index).encode()
        opened = []
        def read(name):
            self.assertFalse(name.startswith('/'))
            self.assertNotIn('..', name.split('/'))
            opened.append(name)
            return blobs[name]
        read.paths = set(blobs)
        result = V.verify(read_bytes=read)
        self.assertTrue(result['verified'])
        self.assertEqual(result['summary']['missing_or_unavailable_input_slots'], 12)
        self.assertEqual(set(opened), set(blobs))

    def test_original_registry_resolution_is_a_lookup_not_an_open(self):
        original = '/this/path/must/not/be/opened.dat'
        blob = input_blob()
        blobs = {'capture/input.dat': blob}
        registry = {original: dict(logical='capture/input.dat', sha256=V.sha(blob), bytes=len(blob))}
        self.assertEqual(V.original_payload(dict(path=original, sha256=V.sha(blob), size=len(blob)),
                                            registry, blobs), blob)
        with self.assertRaises(ValueError):
            V.original_payload(dict(path=original + '.absent', sha256=V.sha(blob), size=len(blob)), registry, blobs)


class ReplayTests(unittest.TestCase):
    def test_exact_frozen_controller_summary_and_failure_output_policy(self):
        rows = [replay_row('create', 1), replay_row('register', 1),
                replay_row('evaluate', 1, status=1, count=1),
                replay_row('evaluate', 2, count=2, output_bits=['0' * 16] * 560),
                replay_row('destroy', 2, count=2)]
        result = V.verify_replay(replay_blob(rows), replay_manifest(), frozen_controller())
        self.assertEqual(result['evaluate_count'], 2)
        self.assertEqual(result['failed_stages'], 1)
        self.assertTrue(result['protocol_complete'])
        self.assertFalse(result['numerical_acceptance_claimed'])
        rows[2]['output_bits'] = ['0' * 16] * 560
        with self.assertRaises(ValueError):
            V.verify_replay(replay_blob(rows), replay_manifest(), frozen_controller())

    def test_setup_failure_has_no_invented_evaluations(self):
        rows = [replay_row('create', 1, status=2, context_present=False)]
        result = V.verify_replay(replay_blob(rows), replay_manifest(), frozen_controller())
        self.assertEqual(result['evaluate_count'], 0)
        self.assertEqual(result['missing_evaluations'], 2)
        self.assertEqual(result['inputs'][0]['branch'], 'context_setup_failed')

    def test_partial_prefix_and_unknown_status_stay_explicit(self):
        rows = [replay_row('create', 1), replay_row('register', 1),
                replay_row('evaluate', 1, status=77, count=1)]
        result = V.verify_replay(replay_blob(rows), replay_manifest(), frozen_controller())
        self.assertFalse(result['protocol_complete'])
        self.assertFalse(result['qualification_passed'])
        self.assertEqual(result['unknown_statuses'], [dict(row=3, status=77)])

    def test_destroy_failure_context_lifetime_is_unknown(self):
        rows = [replay_row('create', 1), replay_row('register', 1, status=2),
                replay_row('destroy', 2, status=2, context_present=None)]
        result = V.verify_replay(replay_blob(rows), replay_manifest(), frozen_controller())
        self.assertEqual(result['evaluate_count'], 0)
        rows[-1]['context_present'] = False
        with self.assertRaises(ValueError):
            V.verify_replay(replay_blob(rows), replay_manifest(), frozen_controller())

    def test_controller_source_pin_is_required(self):
        with self.assertRaises(ValueError): V.load_replay_summary(frozen_controller() + b'\n')


class AdditiveAdmissionTests(unittest.TestCase):
    def test_exact_two_template_hash_join_is_required(self):
        names = ('madgraph/iolibs/template_files/loop_optimized/TIRUST_interface.inc',
                 'madgraph/iolibs/template_files/loop_optimized/loop_matrix_standalone.inc')
        changes = {name: dict(old='a' * 64, new='b' * 64) for name in names}
        source = dict(inputs={name: 'b' * 64 for name in names})
        self.assertEqual(V.template_changes(dict(approved_template_changes=changes), source, changes), changes)
        for bad in ({names[0]: changes[names[0]]}, changes | {'unreviewed': dict(old='a' * 64, new='b' * 64)},
                    changes | {names[0]: dict(old='a' * 64, new='c' * 64)}):
            with self.assertRaises(ValueError):
                V.template_changes(dict(approved_template_changes=bad), source, changes)
        source['inputs'][names[0]] = 'c' * 64
        with self.assertRaises(ValueError):
            V.template_changes(dict(approved_template_changes=changes), source, changes)

    def test_additive_source_cannot_be_selected_by_an_unknown_seal(self):
        with self.assertRaises(ValueError):
            V.verify_source2({'source2/source-index.json': b'{}'}, {}, b'capture', b'prepare', [])


if __name__ == '__main__': unittest.main()
