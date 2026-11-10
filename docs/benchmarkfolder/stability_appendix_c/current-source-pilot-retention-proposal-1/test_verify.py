"""Pure synthetic retention checks; no actual run/capsule or native operation."""
import ctypes
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import types
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent


def load(path, name):
    module = types.ModuleType(name); module.__file__ = str(path)
    exec(compile(path.read_bytes(), str(path), 'exec'), module.__dict__)
    return module


v = load(HERE / 'verify.py', 'proposed_retention_verifier')
codec_blob = (HERE.parent / 'retention-source-1/journal_codec.py').read_bytes()
assert v.sha(codec_blob) == v.CODEC_SHA
codec = v.checked_module(codec_blob, 'test_pinned_codec')
source5 = (HERE.parent / 'source5/pilot.py').read_bytes()
controller = (HERE.parent / 'current-source-pilot-proposal-1/controller.py').read_bytes()
metrics = load(HERE.parent / 'source5/plot_metrics.py', 'test_frozen_metrics')
# This exact checked fixture module supplies mocked receipt/source bytes only.
fixture_path = HERE.parent / 'current-source-classifier-proposal-1/test_classifier.py'
fixture = load(fixture_path, 'checked_classifier_fixtures')


def encoded(blob):
    zipped = gzip.compress(blob, mtime=0)
    item = {'path': 'chunks/chunk-000000.gz', 'raw_bytes': len(blob),
            'raw_sha256': v.sha(blob), 'gzip_bytes': len(zipped), 'gzip_sha256': v.sha(zipped)}
    return {'schema': codec.SCHEMA, 'raw_bytes': len(blob), 'raw_sha256': v.sha(blob), 'chunks': [item]}, zipped


class PureRetention(unittest.TestCase):
    def setUp(self):
        self.popen = patch.object(subprocess, 'Popen', side_effect=AssertionError('No child allowed')).start()
        self.cdll = patch.object(ctypes, 'CDLL', side_effect=AssertionError('No loader allowed')).start()
        self.addCleanup(patch.stopall)

    def test_duplicate_json_and_nonstandard_constants_rejected(self):
        for text in ('{"a":1,"a":2}', '{"a":NaN}', '{"a":Infinity}'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                v.json_value(text)

    def test_paths_reject_escape_and_alias_spellings(self):
        for name in ('/absolute', '../x', 'a/../b', 'a//b', 'a/./b', 'a\\b', 'a:b', 'a\0b'):
            with self.subTest(name=name), self.assertRaises(ValueError): v.safe_path(name)

    def test_index_source_and_terminal_bindings_precede_payload_reads(self):
        for field in ('source_index_sha256', 'root_admitted_run_terminal_sha256'):
            index = {'schema': 'madloop-current-source-raw-stability-capsule-v1',
                     'source_index_sha256': v.CURRENT_SHA, 'root_admitted_run_terminal_sha256': v.RUN_SHA}
            index[field] = '0' * 64
            def reader(name):
                self.assertEqual(name, 'index.json')
                return json.dumps(index).encode()
            reader.paths = {'index.json'}
            with self.subTest(field=field), self.assertRaises(ValueError): v.verify(read_bytes=reader)

    def test_physical_sizes_types_and_total_budget_precede_payload_reads(self):
        for descriptors in ({'one': {'bytes': True, 'sha256': 'a' * 64}},
                            {'one': {'bytes': v.MAX_RAW, 'sha256': 'a' * 64},
                             'two': {'bytes': 1, 'sha256': 'b' * 64}}):
            index = {'schema': 'madloop-current-source-raw-stability-capsule-v1',
                     'source_index_sha256': v.CURRENT_SHA, 'root_admitted_run_terminal_sha256': v.RUN_SHA,
                     'files': descriptors}
            def reader(name):
                self.assertEqual(name, 'index.json')
                return json.dumps(index).encode()
            reader.paths = set(descriptors) | {'index.json'}
            with self.subTest(descriptors=descriptors), self.assertRaises(ValueError): v.verify(read_bytes=reader)

    def test_callback_codec_roundtrip_and_empty_journal(self):
        for blob in (b'', b'B 1\nG hex\nC 1\n' * 100):
            descriptor, zipped = encoded(blob)
            self.assertEqual(v.decode_journal(descriptor, lambda name: zipped, codec), blob)

    def test_codec_rejects_truncation_trailing_multimember_and_overexpansion(self):
        descriptor, zipped = encoded(b'closed journal\n')
        for bad in (zipped[:-1], zipped + b'x', zipped + gzip.compress(b'x', mtime=0), gzip.compress(b'closed journal\nx', mtime=0)):
            mutated = json.loads(json.dumps(descriptor))
            mutated['chunks'][0].update(gzip_bytes=len(bad), gzip_sha256=v.sha(bad))
            with self.subTest(size=len(bad)), self.assertRaises(ValueError):
                v.decode_journal(mutated, lambda name: bad, codec)

    def test_codec_rejects_duplicate_paths_and_false_sizes_before_read(self):
        descriptor, unused = encoded(b'one')
        descriptor['chunks'] *= 2
        with self.assertRaisesRegex(ValueError, 'Duplicate chunk'):
            v.decode_journal(descriptor, lambda name: self.fail('Must not read'), codec)
        descriptor, unused = encoded(b'one'); descriptor['raw_bytes'] = True
        with self.assertRaises(ValueError): v.decode_journal(descriptor, lambda name: self.fail('Must not read'), codec)

    def test_missing_dimensions_does_not_fabricate_diagnostics(self):
        record = {'native_started': True, 'state': 'failed', 'diagnostic_unavailable':
                  'Dimensions not produced; all 100 intended points remain unavailable'}
        texts = {name: '' for name in v.RAW_NAMES}; texts['stdout.txt'] = 'warning retained\n'
        self.assertIsNone(v.replay_diagnostics(record, lambda *args: self.fail('No dimensions'), texts, {}))
        record['diagnostics'] = {}
        with self.assertRaises(ValueError): v.replay_diagnostics(record, lambda *args: None, texts, {})

    def test_parser_exception_and_partial_assignment_are_distinct(self):
        texts = {name: 'fixture' for name in v.RAW_NAMES}; texts['stdout.txt'] = ''
        def bad(*args): raise ValueError('fixture parse')
        record = {'state': 'failed', 'diagnostic_error': {'class': 'ValueError', 'message': 'fixture parse'}}
        self.assertIsNone(v.replay_diagnostics(record, bad, texts, {}))
        partial = {'structurally_complete': False, 'loop_groups_structurally_complete': True, 'rows': []}
        record = {'state': 'failed', 'diagnostics': partial, 'diagnostic_error':
                  {'class': 'ValueError', 'message': 'Raw diagnostic journals are incomplete or structurally invalid'}}
        self.assertEqual(v.replay_diagnostics(record, lambda *args: partial, texts, {}), partial)
        record.pop('diagnostics')
        with self.assertRaises(ValueError): v.replay_diagnostics(record, lambda *args: partial, texts, {})

    def test_qualified_parser_preserves_legacy_and_failure_annotations(self):
        qualification = fixture.c.qualify_sources(**fixture.binding_inputs())
        parser = fixture.c.parsers(source5, qualification=qualification)['parse_diagnostics']
        legacy = fixture.c.parsers(source5)['parse_diagnostics']
        raw = fixture.fixture_namespace['diagnostic_text']()
        groups = fixture.journal(before=(1, 0))
        parsed = parser('1 100 1 1 1', raw, '', groups, fixture.LAYOUT)
        old = legacy('1 100 1 1 1', raw, '', groups, fixture.LAYOUT)
        self.assertEqual(len(parsed['primary_call_outcomes']), 300)
        self.assertFalse(old['rows'][2]['raw_DP_diagnostic_eligible'])
        self.assertTrue(parsed['rows'][2]['raw_DP_diagnostic_eligible'])
        call = parsed['loop_group_calls'][1]
        self.assertFalse(call['legacy_source5_eligible'])
        self.assertTrue(call['incoming_software_latch_cleared'])
        self.assertFalse(call['hardware_ieee_state_observed'])
        failed = parser('1 100 1 1 1', raw, '', fixture.journal(before=(1, 0), failed=True), fixture.LAYOUT)
        self.assertFalse(failed['rows'][2]['raw_DP_diagnostic_eligible'])

    def test_combined_guard_and_all_fixed_unknown_slots(self):
        manifest = {'candidate': {'exporter_identity': {'revision': 'synthetic'}}}
        value = {'cases': [{'id': v.CASE, 'process': 'synthetic source-only'}]}
        results = []
        for request in v.requests():
            rows = [{'phase': 1, 'point': 1, 'method': method, 'order': 0, 'status': 110,
                     'raw_DP_diagnostic_eligible': True, 'reason': None, 'coefficients': [1., 2., 3., 4.]}
                    for method in (1, 2, 3)]
            results.append({'request': request, 'native_started': False, 'state': 'unexecuted',
                'runtime_binding': {'verified': True}, 'readonly_guard_passed': True,
                'protected_guards_passed': request['backend'] != 9, 'diagnostics': {'rows': rows}})
        dataset = v.metric_replay(controller, source5, manifest, value, results, metrics.relative_spread, 'a' * 64)
        lanes = dataset['cases'][0]['lanes']
        self.assertEqual([len(lane['values']) for lane in lanes], [100] * 5)
        self.assertEqual([sum(x is not None for x in lane['values']) for lane in lanes], [1, 1, 1, 1, 0])
        self.assertIn('Runtime or read-only', dataset['provenance']['unavailable_reasons'][v.CASE + '-backend9'][0]['reason'])
        # Failed/unexecuted terminal state is preserved, not used as an extra
        # rejection gate absent from the frozen numerical aggregation.
        summary = v.summarize(dataset, results, metrics)
        self.assertEqual(len(summary['rows']), 5)
        self.assertEqual(summary['rows'][-1]['tail_at_1e3']['upper'], 1.)
        self.assertFalse(summary['numerical_acceptance_claimed'])

    def test_runtime_map_completeness_and_locale_permissions(self):
        exe, lib = '/private/stability-check', '/nix/store/runtime/libmadloop_tirust.so'
        expected = {lib: 'b' * 64}; case = {'runtime_closure': {'files': expected}, 'executable_sha256': 'a' * 64}
        files = {exe: 'a' * 64, **expected}; name = v.MAP_NAMES[0]
        sample = {'path': name, 'elf_files': files, 'read_only_locale': []}
        record = {'command': {'target_argv': [exe]}, 'runtime_binding': {'verified': True, 'backend_observed': False, 'samples': [sample]}}
        raw = ''.join('100-200 r-xp 0000 00:00 1 ' + path + '\n' for path in files)
        v.runtime_binding(record, {name: raw}, case)
        with self.assertRaises(ValueError): v.runtime_binding(record, {name: raw + '200-300 r--p 0000 00:00 1 /unexpected\n'}, case)
        locale = '/nix/store/locale/lib/locale/locale-archive'; sample['read_only_locale'] = [locale]
        v.runtime_binding(record, {name: raw + '200-300 r--p 0000 00:00 1 ' + locale + '\n'}, case)
        with self.assertRaises(ValueError): v.runtime_binding(record, {name: raw + '200-300 rw-p 0000 00:00 1 ' + locale + '\n'}, case)


if __name__ == '__main__':
    unittest.main()
