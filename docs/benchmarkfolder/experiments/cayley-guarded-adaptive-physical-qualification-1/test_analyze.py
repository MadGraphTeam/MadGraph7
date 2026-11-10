"""Offline analyzer tests; use closed ancestor text and synthetic failures only."""
import importlib.util
import json
from pathlib import Path
import signal
import tempfile
import unittest
from unittest.mock import Mock


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


analyze = load('guarded_offline_analyzer_under_test', HERE / 'analyze.py')
OLD_SOURCE = ROOT / analyze.OLD_SOURCE
parser = load('closed_journal_for_offline_tests', OLD_SOURCE / 'parser.py')
controls = load('closed_controls_for_offline_tests', OLD_SOURCE / 'control_log.py')
assessment = load('closed_assessment_for_offline_tests', ROOT / analyze.SOURCE / 'assessment.py')
CLOSED_RUN = ROOT / '.tirust-build/adaptive-qp-checked-trap-proposal-1/run1'


class PayloadTests(unittest.TestCase):
    def test_relative_payload_names(self):
        self.assertEqual(analyze.relative_name('.tirust-build/x/log.txt'), '.tirust-build/x/log.txt')
        for name in ('/tmp/lib.so', '../escape', 'x/../escape', 'x//y', './x', '.', ''):
            with self.subTest(name=name), self.assertRaises(ValueError):
                analyze.relative_name(name)

    def test_nonutf8_is_rejected_without_loading_sources(self):
        with self.assertRaises(UnicodeDecodeError):
            analyze.reanalyze({'text': b'\xff'})

    def test_missing_sources_is_a_capture_error(self):
        with self.assertRaisesRegex(ValueError, 'Missing frozen source'):
            analyze.reanalyze({'text': b'plain text'})

    def test_nonterminal_checkpoint_is_not_an_execution_receipt(self):
        blobs = {name: b'pass\n' for name in (
            analyze.SOURCE + '/prepared.json', analyze.SOURCE + '/controller.py',
            analyze.SOURCE + '/assessment.py', analyze.OLD_SOURCE + '/controller.py',
            analyze.OLD_SOURCE + '/parser.py', analyze.OLD_SOURCE + '/control_log.py', analyze.BENCHMARK)}
        blobs[analyze.RUN + '/execution.json'] = json.dumps({'execution_complete': False}).encode()
        with self.assertRaisesRegex(ValueError, 'terminal publication'):
            analyze.reanalyze(blobs)

    def test_json_key_normalization_is_explicit(self):
        self.assertEqual(analyze.canonical({1: {'observed': {2: 'x'}}}),
                         {'1': {'observed': {'2': 'x'}}})


class WorkerFailureTests(unittest.TestCase):
    def inspect(self, *, code=-signal.SIGFPE, journal=None, control='adaptive-trapped',
                kind='physical', retained_error=None, stdout_present=True):
        with tempfile.TemporaryDirectory(prefix='guarded-analyzer-test-') as temporary:
            base = Path(temporary)
            directory = base / analyze.RUN / 'synthetic-worker'
            directory.mkdir(parents=True)
            (directory / 'checked-control.log').write_bytes((CLOSED_RUN / control / 'checked-control.log').read_bytes())
            if journal is not None:
                (directory / 'mass-observer.hex').write_text(journal)
            if stdout_present:
                (directory / 'stdout.log').write_text('')
            row = {'kind': kind, 'mode': 'checked' if kind == 'physical' else None,
                   'policy': 'trapped' if kind == 'physical' else 'invalid', 'exit_code': code}
            if retained_error is not None:
                row['offline_assessment_failure'] = retained_error
            return analyze.inspect_worker(base, 'synthetic-worker', row, parser, controls, assessment)

    def test_missing_trapped_journal_is_retained_failure(self):
        result, capture = self.inspect()
        self.assertIsNone(capture)
        self.assertIn('absent after interrupted worker', result['journal_disposition'])
        self.assertNotIn('portable_parse_failure', result)
        self.assertFalse(result['physical_outputs_qualified'])
        self.assertFalse(result['control_assessment']['physical_worker_completed_with_requested_IEEE_policy'])

    def test_missing_journal_after_zero_exit_is_required_output_failure(self):
        result, capture = self.inspect(code=0, control='adaptive-masked')
        self.assertIsNone(capture)
        self.assertEqual(result['journal_disposition'], 'missing required journal after zero exit')
        self.assertFalse(result['physical_outputs_qualified'])

    def test_retained_missing_journal_error_keeps_original_path(self):
        expected = analyze.RUN + '/synthetic-worker/mass-observer.hex'
        failure = {'type': 'FileNotFoundError', 'message': "Missing '/historical/root/" + expected + "'"}
        result, _ = self.inspect(retained_error=failure)
        self.assertEqual(result['missing_journal_path'], expected)
        self.assertTrue(result['retained_missing_journal_failure_matches'])
        self.assertEqual(result['retained_offline_assessment_failure'], failure)
        wrong = {'type': 'FileNotFoundError', 'message': "Missing '/different/path'"}
        result, _ = self.inspect(retained_error=wrong)
        self.assertFalse(result['retained_missing_journal_failure_matches'])

    def test_interrupted_journal_prefix_is_preserved_and_unqualified(self):
        result, capture = self.inspect(journal=parser.FORMAT + '\nBEGIN 1 FFFFF\n')
        self.assertIsNotNone(capture)
        self.assertFalse(capture['complete'])
        self.assertEqual(result['journal_disposition'], 'interrupted prefix; unqualified')
        self.assertFalse(result['numerical_assessment']['checked_outputs_qualified'])

    def test_malformed_prefix_retains_parse_failure(self):
        result, capture = self.inspect(journal=parser.FORMAT + '\nUNKNOWN\n')
        self.assertIsNone(capture)
        self.assertEqual(result['portable_parse_failure']['type'], 'ValueError')
        self.assertFalse(result['physical_outputs_qualified'])

    def test_missing_stdout_is_visible_not_imputed(self):
        result, _ = self.inspect(stdout_present=False)
        self.assertFalse(result['stdout_present'])
        self.assertEqual(result['W03'], [])

    def test_missing_stdout_cannot_establish_omission_free_acceptance(self):
        with tempfile.TemporaryDirectory(prefix='guarded-analyzer-test-') as temporary:
            base = Path(temporary)
            directory = base / analyze.RUN / 'synthetic-worker'
            directory.mkdir(parents=True)
            (directory / 'checked-control.log').write_bytes(
                (CLOSED_RUN / 'adaptive-masked/checked-control.log').read_bytes())
            (directory / 'mass-observer.hex').write_text('synthetic parser input\n')
            journal = Mock()
            journal.parse.return_value = {'complete': True}
            journal.summarize.return_value = {'complete': True}
            journal.qualify_checked.side_effect = AssertionError('Missing stdout must not be qualified')
            row = {'kind': 'physical', 'mode': 'checked', 'policy': 'masked', 'exit_code': 0}
            result, capture = analyze.inspect_worker(
                base, 'synthetic-worker', row, journal, controls, assessment)
            self.assertIsNotNone(capture)
            self.assertFalse(result['physical_outputs_qualified'])
            self.assertNotIn('numerical_assessment', result)
            self.assertIn('omission scan unavailable', result['numerical_assessment_disposition'])
            journal.qualify_checked.assert_not_called()

    def test_closed_fault_control_needs_no_journal(self):
        result, capture = self.inspect(control='fault-invalid', kind='fault')
        self.assertIsNone(capture)
        self.assertTrue(result['control_assessment']['fault_control_demonstrated'])
        self.assertIn('not applicable', result['journal_disposition'])


class PublicationAndGateTests(unittest.TestCase):
    def test_failed_publication_cannot_restore_healthy_transport(self):
        record = {'execution_complete': False, 'closure_failure': {'type': 'TimeoutError'}}
        self.assertFalse(analyze.published_transport(record, True))

    def test_claimed_success_requires_final_closure(self):
        with self.assertRaises(ValueError):
            analyze.published_transport({'execution_complete': True}, True)
        self.assertTrue(analyze.published_transport(
            {'execution_complete': True, 'protected_inputs_after': 'unchanged'}, True))

    def test_automatic_DP_acceptance_does_not_require_QP_coverage(self):
        gates = {'checked': {'numerical_agreement': True, 'host_stability_acceptance': True,
                            'trap_completion': True, 'masked_trapped_numerical_agreement': True,
                            'QP_path_coverage': False},
                 'forced-qp': {'numerical_agreement': True, 'QP_path_coverage': False,
                               'trap_completion': True, 'masked_trapped_numerical_agreement': True}}
        result = analyze.composite_gates(gates)
        self.assertTrue(result['checked_trap_gate_qualified'])
        self.assertFalse(result['forced_qp_trap_gate_qualified'])

    def test_no_captures_does_not_invent_comparisons(self):
        comparisons, pending = analyze.compare_captures(
            {'comparators': []}, {}, {}, {}, parser, None, assessment, None)
        self.assertEqual(comparisons, {})
        self.assertFalse(pending['checked']['policy_numerical_agreement'])
        self.assertFalse(pending['forced-qp']['numerical_to_closed_adaptive']['trapped'])


if __name__ == '__main__':
    unittest.main()
