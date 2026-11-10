"""Seven pure/mock tests; no native execution or active campaign reads."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location('stress_console_diagnostics', HERE / 'verify_evidence.py')
d = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(d)


class DiagnosticsTests(unittest.TestCase):
    def test_runtime_flags_are_not_signal_or_register_observations(self):
        raw = b'ordinary\nNote: The following floating-point exceptions are signalling: IEEE_INVALID_FLAG IEEE_DIVIDE_BY_ZERO\n'
        result = d.diagnostics(raw)
        self.assertEqual(result['runtime_reported_flags'], ['IEEE_DIVIDE_BY_ZERO', 'IEEE_INVALID_FLAG'])
        self.assertEqual(result['lines'][0]['line'], 2)
        self.assertNotIn('signal_diagnostic', result['lines'][0]['kinds'])

    def test_exact_line_numbered_warning_reduction_and_signal_evidence(self):
        raw = (b'\n ##W03 WARNING unstable point\n'
               b' ## The last QP reduction was deemed unstable because a floating point exception was encountered.\n'
               b' ERROR in OneLOop c0: all input equal zero, returning 0\n'
               b'Program received signal SIGFPE: Floating-point exception.\n')
        result = d.diagnostics(raw)
        self.assertEqual([r['line'] for r in result['lines']], [2, 3, 4, 5])
        self.assertEqual(result['lines'][0]['text'], ' ##W03 WARNING unstable point')
        self.assertIn('W03', result['lines'][0]['kinds'])
        self.assertIn('reduction_exception_or_library_diagnostic', result['lines'][1]['kinds'])
        self.assertIn('reduction_exception_or_library_diagnostic', result['lines'][2]['kinds'])
        self.assertIn('signal_diagnostic', result['lines'][3]['kinds'])

    def test_missing_console_is_distinct_from_present_empty_console(self):
        absent, empty = d.diagnostics(None), d.diagnostics(b'')
        self.assertFalse(absent['stdout_present'])
        self.assertIsNone(absent['stdout_sha256'])
        self.assertTrue(empty['stdout_present'])
        self.assertEqual(empty['stdout_sha256'], d.sha(b''))
        self.assertEqual(empty['lines'], [])

    def test_source_quote_never_fabricates_line_171_or_deadline(self):
        needle = 'raise ProtocolError(f"Worker protocol timeout; see {self.log}")'
        raw = ('\n' * 170 + '    ' + needle + '\n').encode()
        self.assertEqual(d.source_line(raw, 'worker.py', needle)['line'], 171)
        self.assertIsNone(d.source_line(b'different\n', 'worker.py', needle))
        with self.assertRaises(ValueError): d.source_line(raw + raw, 'worker.py', needle)

    def test_timeout_coverage_and_mapless_native_warmup_remain_distinct(self):
        ids = [f'original/{i}' for i in range(4)] + [f'continuation/{i}' for i in range(18)] + [
            'qcd_n4_uux_zzgg/candidate-9-auto', 'ew_n3_epem_ttxa/candidate-7-auto']
        rows = []
        for rid in ids:
            rows.append({'request_id': rid, 'case': rid.split('/')[0], 'backend': 1,
                         'state': 'host_checked_complete', 'complete_runtime_map': True,
                         'warmup_calls_observed': 3, 'warmup_trace': {'rows': []},
                         'final_trace': {'rows': []}})
        timeout = rows[-2]
        timeout.update(backend=9, state='physical_rejected', warmup_calls_observed=0,
                       native_failure={'class': 'ProtocolError', 'message': 'Worker protocol timeout; see logfile'},
                       native_entry_observations={'READY_verified_by_frozen_worker': False, 'RUN_command_journal_count': 0})
        collier = rows[-1]
        collier.update(backend=7, state='controller_failed', complete_runtime_map=False,
                       warmup_trace={'rows': [{'order': order, 'status': status}
                           for status in (217, 217, 410) for order in (0, 1)]})
        base = {'request_rows': rows[4:], 'state_counts': {'host_checked_complete': 8, 'physical_rejected': 11, 'controller_failed': 1},
                'aggregate': {'request_rows': rows, 'host_checked_complete_streams': 12,
                              'comparisons': [{'passed': n >= 12} for n in range(22)]}}
        result = d.derive(base, lambda _: b'', b'no quoted worker source\n', b'no quoted native source\n')
        self.assertEqual(result['continuation_native_process_activity_observed'], 20)
        self.assertEqual(result['continuation_complete_runtime_maps'], 19)
        target = result['qcd_n4_tirust_timeout']
        self.assertEqual(target['completed_warmup_rows'], 0)
        self.assertEqual(target['completed_final_rows'], 0)
        self.assertEqual(target['RUN_command_journal_count'], 0)
        self.assertFalse(target['READY_observed'])
        self.assertFalse(target['internal_in_progress_amplitude_calls_instrumented'])
        self.assertTrue(target['no_completed_rows_does_not_imply_no_in_progress_amplitude_calls'])
        self.assertIsNone(target['worker_timeout_source'])
        self.assertFalse(result['hardware_register_capture'])
        self.assertEqual(result['aggregate_failing_comparisons'], 12)
        self.assertEqual(result['request_rows'][-1]['warmup_statuses_by_call'], [217, 217, 410])
        self.assertEqual(result['request_rows'][-1]['state'], 'controller_failed')

    def test_changed_base_pin_and_uninventoried_reads_are_rejected(self):
        def changed(name): return b'changed immutable evidence'
        changed.paths = {'base/evidence-index.json', 'base/verify_evidence.py'}
        with self.assertRaisesRegex(ValueError, 'Closed base identity differs'): d.verify(read_bytes=changed)
        with tempfile.TemporaryDirectory(prefix='stress-console-reader-') as td:
            root = Path(td) / 'diagnostics'; root.mkdir()
            for name in (d.BASE, d.ANCESTOR, d.PREPARATION): (root.with_name(name)).mkdir()
            (root.with_name(d.BASE) / 'known.txt').write_text('explicit inventory')
            read = d.reader(root)
            self.assertEqual(read('base/known.txt'), b'explicit inventory')
            with self.assertRaises(ValueError): read('base/../unknown')
            with self.assertRaises(ValueError): read('base/not-in-inventory')

    def test_absent_ancestor_map_field_is_unknown_without_changing_current_count(self):
        ids = [f'original/{n}' for n in range(4)] + [f'current/{n}' for n in range(18)] + [
            'qcd_n4_uux_zzgg/candidate-9-auto', 'ew_n3_epem_ttxa/candidate-7-auto']
        rows = [{'request_id': rid, 'case': rid.split('/')[0], 'backend': 1,
                 'state': 'host_checked_complete', 'warmup_calls_observed': 1,
                 'warmup_trace': {'rows': [{'order': 0, 'status': 217}]}} for rid in ids]
        for row in rows[4:]: row['complete_runtime_map'] = True
        rows[-1]['complete_runtime_map'] = False
        base = {'request_rows': rows[4:], 'state_counts': {},
                'aggregate': {'request_rows': rows, 'host_checked_complete_streams': 12, 'comparisons': []}}
        summary = d.derive(base, lambda _: b'', b'', b'')
        self.assertEqual([r['complete_runtime_map'] for r in summary['request_rows'][:4]], [None] * 4)
        self.assertTrue(all('omits this field' in r['runtime_map_field_scope'] for r in summary['request_rows'][:4]))
        self.assertTrue(all('Explicit admitted' in r['runtime_map_field_scope'] for r in summary['request_rows'][4:]))
        self.assertEqual(summary['continuation_complete_runtime_maps'], 19)
        self.assertEqual(summary['continuation_native_process_activity_observed'], 20)


if __name__ == '__main__': unittest.main()
