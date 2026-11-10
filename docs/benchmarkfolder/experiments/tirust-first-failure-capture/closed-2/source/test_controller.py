"""Pure/mock/temporary-filesystem tests. No compiler, loader or native calls."""
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import struct
import tempfile
import types
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('capture_controller_tests', HERE / 'controller.py')
c = importlib.util.module_from_spec(spec); spec.loader.exec_module(c)


class Planning(unittest.TestCase):
    def test_all_dry_stages_are_inert(self):
        for argv in ([], ['plan', '--execute'], ['prepare'], ['capture'], ['replay']):
            with (self.subTest(argv=argv), mock.patch.object(c, 'admit', side_effect=AssertionError('admission')),
                  mock.patch.object(c, 'read', side_effect=AssertionError('read')),
                  contextlib.redirect_stdout(io.StringIO()) as output):
                self.assertEqual(c.main(argv), 0)
            value = json.loads(output.getvalue())
            self.assertEqual(value['commands_executed'], 0)
            self.assertEqual(value['maximum_unique_inputs'], 12)
            self.assertEqual(value['maximum_standalone_evaluations'], 24)
            self.assertEqual(value['native_caps_seconds'], {'capture_per_case': 180, 'replay_total': 60})
            self.assertFalse(value['automatic_retry'])

    def test_execute_requires_inventory_before_helpers_or_native(self):
        with mock.patch.object(c, 'seal', side_effect=ValueError('unsealed')):
            for stage in ('prepare', 'capture', 'replay'):
                with self.subTest(stage=stage), self.assertRaisesRegex(ValueError, 'unsealed'):
                    c.main([stage, '--execute'])

    def test_exact_target_registry(self):
        rows = [row for spec in c.SPECS for row in c.expected_inputs(spec)]
        self.assertEqual(len(rows), 12)
        self.assertEqual(len({(r['case'], r['point'], r['method'], r['group']) for r in rows}), 12)
        self.assertEqual({r['point'] for r in rows if r['case_id'] == 1}, {25})
        self.assertEqual({r['group'] for r in rows if r['case_id'] == 2}, {408, 422})
        self.assertEqual({(r['nprop'], r['rank']) for r in rows}, {(5, 4), (6, 5)})

    def test_makefile_has_no_numerical_rebuild_dependencies(self):
        self.assertIn('capture-build:\n', c.MAKEFILE)
        self.assertIn('$(filter-out TIRUST_interface.o TIR_interface.o,$(PROCESS))', c.MAKEFILE)
        self.assertIn('-Wl,--wrap=ml_tirust_evaluate_dp', c.MAKEFILE)
        self.assertNotIn('$(MAKE)', c.MAKEFILE)
        recipes = [line for line in c.MAKEFILE.splitlines() if line.startswith('\t')]
        self.assertEqual(len(recipes), 8)
        self.assertEqual(sum('-c ' in line for line in recipes), 6)
        self.assertTrue(all('-c ' in line for line in recipes if '$(CC)' in line))
        self.assertTrue(all(line.startswith('\t$(FC)') for line in recipes if ' -o stability-check ' in line
                            or ' -o capture-replay/stability-check ' in line))


class Inputs(unittest.TestCase):
    def test_expected_missing_requests_are_never_dropped(self):
        closed = {'results': [{'case': spec['case'], 'inputs': [dict(row, state='missing')
                                  for row in c.expected_inputs(spec)]} for spec in c.SPECS]}
        self.assertEqual(len(c.capture_rows(closed)), 12)
        self.assertEqual(c.request_manifest(c.capture_rows(closed)), '')
        closed['results'][0]['inputs'].pop()
        with self.assertRaisesRegex(ValueError, 'dropped an expected input'): c.capture_rows(closed)

    def test_closed_capture_rejects_changed_keys_or_case_order(self):
        closed = {'results': [{'case': spec['case'], 'inputs': [dict(row, state='missing')
                                  for row in c.expected_inputs(spec)]} for spec in c.SPECS]}
        closed['results'][0]['inputs'][0]['group'] = 999
        with self.assertRaisesRegex(ValueError, 'key differs'): c.capture_rows(closed)
        closed['results'].reverse()
        with self.assertRaisesRegex(ValueError, 'both registered cases'): c.capture_rows(closed)

    def test_manifest_binds_only_present_closed_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = Path(tmp) / 'request.dat'; first.write_text('MLTCAP_INPUT_V1\nopaque\n')
            rows = [{'state': 'captured', 'path': str(first), 'sha256': c.digest(first)}, {'state': 'missing'}]
            self.assertEqual(c.request_manifest(rows), str(first) + '\n')
            first.write_text('changed\n')
            with self.assertRaisesRegex(ValueError, 'Closed capture input changed'): c.request_manifest(rows)

    def test_manifest_rejects_duplicate_paths_oversized_or_nonabsolute(self):
        row = {'state': 'captured', 'path': '/tmp/input.dat', 'sha256': 'unused'}
        for rows in ([row, row], [dict(row, path='/tmp/a%d' % i) for i in range(13)],
                     [dict(row, path='relative')], [dict(row, path='/tmp/a\n/tmp/b')]):
            with self.subTest(rows=rows), self.assertRaises(ValueError): c.request_manifest(rows)

    def test_capture_collects_all_six_slots_without_filling_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp); spec = c.SPECS[0]
            first = output / c.expected_inputs(spec)[0]['name']; first.write_text('MLTCAP_INPUT_V1\nraw\n')
            links = types.SimpleNamespace(assert_private_path=lambda root, path: path)
            rows = c.collect_inputs(spec, output, links, output)
            self.assertEqual(len(rows), 6)
            self.assertEqual([row['state'] for row in rows], ['captured'] + ['missing'] * 5)
            self.assertEqual(rows[0]['sha256'], c.digest(first))

    def test_capture_rejects_symlink_or_wrong_header(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); output = root / 'worker'; output.mkdir(); spec = c.SPECS[0]
            first = output / c.expected_inputs(spec)[0]['name']; first.write_text('wrong header\n')
            links = types.SimpleNamespace(assert_private_path=lambda root, path: path)
            with self.assertRaisesRegex(ValueError, 'Unknown capture input header'):
                c.collect_inputs(spec, output, links, output)
            first.unlink(); external = root / 'external'; external.write_text('MLTCAP_INPUT_V1\n')
            first.symlink_to(external)
            with self.assertRaisesRegex(ValueError, 'new regular file'):
                c.collect_inputs(spec, output, links, output)


class Replay(unittest.TestCase):
    @staticmethod
    def fixture(spec=None):
        request = dict(c.expected_inputs(spec or c.SPECS[0])[0], state='captured', path='/closed/request.dat')
        key = {name: request[name] for name in ('case_id', 'point', 'method', 'group')}; key['phase'] = 1
        rows = []
        for stage, repetition, count in (('create', 1, 0), ('register', 1, 0), ('evaluate', 1, 1),
                                          ('evaluate', 2, 2), ('destroy', 2, 2)):
            row = {'record_type': stage, 'repetition': repetition, 'status': 0, 'input_key': dict(key),
                   'input_path': request['path'], 'evaluation_count': count, 'context_present': stage != 'destroy'}
            if stage == 'evaluate': row['output_bits'] = ['0000000000000000'] * (8 * c.math.comb(request['rank'] + 4, 4))
            rows.append(row)
        return request, rows

    @staticmethod
    def summarize(request, records):
        return c.replay_summary('\n'.join(map(json.dumps, records)), [request])

    def test_failed_stages_retained_without_failed_output(self):
        request, records = self.fixture(); records[2].update(status=1, error='Unstable'); del records[2]['output_bits']
        result = self.summarize(request, records)
        self.assertEqual(result['evaluate_count'], 2)
        self.assertEqual(result['failed_stages'], 1)
        self.assertEqual(result['missing_evaluations'], 0)
        self.assertFalse(result['numerical_acceptance_claimed'])
        self.assertTrue(result['qualification_passed'])

    def test_missing_evaluations_stay_explicit(self):
        request, records = self.fixture(); result = self.summarize(request, records[:1])
        self.assertEqual(result['missing_evaluations'], 2)
        self.assertFalse(result['protocol_complete'])
        self.assertEqual(len(result['inputs'][0]['missing_required_stages']), 4)
        empty = self.summarize(request, [])
        self.assertEqual(len(empty['inputs'][0]['missing_stages']), 5)

    def test_failed_or_non_evaluation_outputs_rejected(self):
        for mutation in ('failed-output', 'non-eval-output', 'malformed-word', 'wrong-count', 'unknown-stage', 'bad-repetition'):
            request, records = self.fixture()
            if mutation == 'failed-output': records[2]['status'] = 1
            elif mutation == 'non-eval-output': records[0]['output_bits'] = []
            elif mutation == 'malformed-word': records[2]['output_bits'][0] = 'NaN'
            elif mutation == 'wrong-count': records[2]['output_bits'].pop()
            elif mutation == 'unknown-stage': records[0]['record_type'] = 'unknown'
            else: records[2]['repetition'] = 3
            with self.subTest(mutation=mutation), self.assertRaises(ValueError): self.summarize(request, records)

    def test_evaluation_budget_enforced(self):
        request, records = self.fixture()
        with self.assertRaisesRegex(ValueError, 'Duplicate replay manifest key'):
            c.replay_summary('', [request] * 12)
        with self.assertRaisesRegex(ValueError, 'manifest budget exceeded'):
            c.replay_summary('', [request] * 13)

    def test_valid_context_and_registration_failure_branches_keep_missing_work(self):
        for branch in ('create-failure', 'null-success-context', 'registration-failure'):
            request, rows = self.fixture()
            if branch == 'create-failure': rows[0]['status'] = 2; rows = rows[:1]
            elif branch == 'null-success-context': rows[0]['context_present'] = False; rows = rows[:1]
            else:
                rows[1]['status'] = 3; rows[-1]['evaluation_count'] = 0; rows = [rows[0], rows[1], rows[-1]]
            with self.subTest(branch=branch):
                result = self.summarize(request, rows)
                self.assertTrue(result['protocol_complete'])
                self.assertEqual(result['evaluate_count'], 0)
                self.assertEqual(result['missing_evaluations'], 2)
                self.assertEqual(len(result['inputs'][0]['missing_stages']), 4 if len(rows) == 1 else 2)

    def test_duplicate_wrong_key_path_order_status_and_counter_rejected(self):
        for mutation in ('duplicate', 'wrong-key', 'wrong-path', 'reordered', 'boolean-status', 'large-status', 'count-jump', 'eval-before-register'):
            request, rows = self.fixture()
            if mutation == 'duplicate': rows.insert(1, dict(rows[0]))
            elif mutation == 'wrong-key': rows[2]['input_key']['group'] = 999
            elif mutation == 'wrong-path': rows[2]['input_path'] = '/other/request.dat'
            elif mutation == 'reordered': rows[2], rows[3] = rows[3], rows[2]
            elif mutation == 'boolean-status': rows[2]['status'] = True
            elif mutation == 'large-status': rows[2]['status'] = 2 ** 31
            elif mutation == 'count-jump': rows[2]['evaluation_count'] = 2
            else: rows.pop(1)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError): self.summarize(request, rows)

    def test_unknown_status_retained_as_unqualified(self):
        request, rows = self.fixture(); rows[2]['status'] = 99; del rows[2]['output_bits']
        result = self.summarize(request, rows)
        self.assertEqual(result['unknown_statuses'], [{'row': 3, 'status': 99}])
        self.assertFalse(result['qualification_passed'])

    def test_both_exact_tensor_output_sizes_are_required(self):
        for spec, count in zip(c.SPECS, (560, 1008)):
            request, rows = self.fixture(spec)
            self.assertEqual(len(rows[2]['output_bits']), count)
            self.assertTrue(self.summarize(request, rows)['qualification_passed'])


class SourceAndRuntime(unittest.TestCase):
    def test_unqualified_runtime_fails_without_overwriting_raw_native_status(self):
        for observation in ({'verified': False}, {}, {'verified': 1}):
            with self.subTest(observation=observation):
                record = {'state': 'complete', 'command': {'returncode': 0, 'error': None}, 'inputs': ['retained']}
                pilot = types.SimpleNamespace(runtime_binding=lambda *args: observation)
                with self.assertRaisesRegex(ValueError, 'not qualified'):
                    c.observe_runtime(pilot, Path('/evidence'), Path('/executable'), {}, record)
                self.assertEqual(record['state'], 'failed')
                self.assertIn('runtime_qualification_error', record)
                self.assertEqual(record['command'], {'returncode': 0, 'error': None})
                self.assertEqual(record['inputs'], ['retained'])

    def test_runtime_observation_exception_retained_and_qualified_observation_accepted(self):
        record = {'state': 'complete'}
        pilot = types.SimpleNamespace(runtime_binding=mock.Mock(side_effect=ValueError('no sampled maps')))
        with self.assertRaisesRegex(ValueError, 'no sampled maps'):
            c.observe_runtime(pilot, Path('/evidence'), Path('/executable'), {}, record)
        self.assertEqual(record['runtime_qualification_error']['message'], 'no sampled maps')
        record = {'state': 'complete'}
        pilot.runtime_binding = lambda *args: {'verified': True}
        c.observe_runtime(pilot, Path('/evidence'), Path('/executable'), {}, record)
        self.assertEqual(record['state'], 'complete')
        self.assertNotIn('runtime_qualification_error', record)

    @staticmethod
    def elf_fixture():
        strings = b'\0ml_tirust_evaluate_dp\0__wrap_ml_tirust_evaluate_dp\0'
        symbols = struct.pack('<IBBHQQ', 0, 0, 0, 0, 0, 0)
        symbols += struct.pack('<IBBHQQ', 1, 0, 0, 0, 0, 0)
        symbols += struct.pack('<IBBHQQ', strings.index(b'__wrap'), 0, 0, 1, 0, 0)
        reloc = struct.pack('<QQq', 8075, (1 << 32) | 4, -4)
        first = 64 + 4 * 64; string_start = first + len(symbols); relocation_start = string_start + len(strings)
        header = bytearray(64); header[:6] = b'\x7fELF\x02\x01'
        struct.pack_into('<Q', header, 40, 64); struct.pack_into('<HH', header, 58, 64, 4)
        table = struct.pack('<IIQQQQIIQQ', *([0] * 10))
        table += struct.pack('<IIQQQQIIQQ', 0, 2, 0, 0, first, len(symbols), 2, 0, 8, 24)
        table += struct.pack('<IIQQQQIIQQ', 0, 3, 0, 0, string_start, len(strings), 0, 0, 1, 0)
        table += struct.pack('<IIQQQQIIQQ', 0, 4, 0, 0, relocation_start, len(reloc), 1, 0, 8, 24)
        return bytes(header) + table + symbols + strings + reloc

    def test_static_archive_import_and_call_relocation(self):
        member = self.elf_fixture()
        header = ('%-16s%-12s%-6s%-6s%-8s%-10s`\n' % ('mtirust.o/', '0', '0', '0', '644', len(member))).encode()
        archive = b'!<arch>\n' + header + member + (b'\n' if len(member) % 2 else b'')
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'libtirust.a'; path.write_bytes(archive)
            result = c.static_abi(path)
            self.assertEqual(result['dp_relocations'], [{'symbol': 'ml_tirust_evaluate_dp', 'offset': 8075, 'type': 4}])
            self.assertEqual(result['member_sha256'], hashlib.sha256(member).hexdigest())
        with self.assertRaisesRegex(ValueError, 'exactly one'): c.archive_member(archive, 'absent.o')
        with self.assertRaisesRegex(ValueError, 'Expected ELF64'): c.elf_symbols(b'not ELF')

    def test_final_wrapper_map_must_reference_actual_mtirust_member(self):
        with tempfile.TemporaryDirectory() as tmp:
            executable = Path(tmp) / 'stability-check'; executable.write_bytes(self.elf_fixture())
            linkmap = Path(tmp) / 'capture-link.map'
            linkmap.write_text('__wrap_ml_tirust_evaluate_dp capture-observer-c.o\n  /private/libtirust.a(mtirust.o)\nnext_symbol other.o\n')
            self.assertTrue(c.linked_abi(executable, linkmap)['verified'])
            linkmap.write_text('__wrap_ml_tirust_evaluate_dp capture-observer-c.o\nnext_symbol other.o\n')
            with self.assertRaisesRegex(ValueError, 'does not bind'): c.linked_abi(executable, linkmap)

    def test_helper_executes_only_once_read_hashed_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'helper.py'; blob = b'VALUE = 42\n'; path.write_bytes(blob)
            module = c.module_from_bytes(path, hashlib.sha256(blob).hexdigest())
            self.assertEqual(module.VALUE, 42)
            path.write_text('raise AssertionError("unbound execution")\n')
            with self.assertRaisesRegex(ValueError, 'Pinned Python source differs'):
                c.module_from_bytes(path, hashlib.sha256(blob).hexdigest())

    def test_loader_closure_allows_only_bound_target_subset(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); loader = root / 'ld.so'; dso = root / 'libmadloop_tirust.so'; other = root / 'unused.so'
            for path in (loader, dso, other): path.write_text(path.name)
            executable = root / 'stability-check'; executable.write_text('opaque executable')
            old = {'loader': str(loader), 'files': {str(path): c.digest(path) for path in (loader, dso, other)}}
            pilot = types.SimpleNamespace(elf_interpreter=lambda path: loader)
            result = c.closure(pilot, executable, old, str(loader) + '\n' + str(dso))
            self.assertEqual(set(result), {str(loader), str(dso)})
            unbound = root / 'unbound.so'; unbound.write_text('unbound')
            with self.assertRaisesRegex(ValueError, 'outside the immutable'):
                c.closure(pilot, executable, old, str(dso) + '\n' + str(unbound))
            with self.assertRaisesRegex(ValueError, 'Unresolved'):
                c.closure(pilot, executable, old, 'not found')

    def test_wrong_target_loader_rejected(self):
        pilot = types.SimpleNamespace(elf_interpreter=lambda path: Path('/wrong/loader'))
        with self.assertRaisesRegex(ValueError, 'changed interpreter'):
            c.closure(pilot, Path('/unused/executable'), {'loader': '/correct/loader'}, '')


if __name__ == '__main__': unittest.main()
