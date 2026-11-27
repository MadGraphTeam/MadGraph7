"""Explicit TIRust selection and rejection of failed or stale amplitude results."""

from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

from madgraph import MadGraph5Error
from madgraph.core import base_objects
from madgraph.various import process_checks


class TestTIRustProcessChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.process = self.root / 'SubProcesses/P0_fixture'
        self.process.mkdir(parents=True)
        self.result = self.process / 'result.dat'
        self.result.write_text('stale result')

    def test_failed_check_never_reads_previous_result(self):
        child = SimpleNamespace(returncode=1, communicate=lambda: (b'invalid partial request', None))
        with mock.patch.object(process_checks.subprocess, 'Popen', return_value=child), \
                mock.patch.object(process_checks.LoopMatrixElementEvaluator, 'parse_check_output') as parse:
            with self.assertRaisesRegex(MadGraph5Error, './check failed'):
                process_checks.LoopMatrixElementEvaluator.get_me_value(
                    'fixture', 0, str(self.root), skip_compilation=True, verbose=False)
        parse.assert_not_called()
        self.assertFalse(self.result.exists())

    def test_successful_exit_without_fresh_output_is_not_a_zero_amplitude(self):
        child = SimpleNamespace(returncode=0, communicate=lambda: (b'STOP', None))
        with mock.patch.object(process_checks.subprocess, 'Popen', return_value=child):
            with self.assertRaisesRegex(MadGraph5Error, 'fresh result.dat'):
                process_checks.LoopMatrixElementEvaluator.get_me_value(
                    'fixture', 0, str(self.root), skip_compilation=True, verbose=False)

    def test_fresh_success_is_parsed_after_execution(self):
        def run():
            self.assertFalse(self.result.exists())
            self.result.write_text('fresh result')
            return b'completed', None
        child = SimpleNamespace(returncode=0, communicate=run)
        with mock.patch.object(process_checks.subprocess, 'Popen', return_value=child), \
                mock.patch.object(process_checks.LoopMatrixElementEvaluator, 'parse_check_output',
                                  side_effect=lambda output, format: output.read()):
            result = process_checks.LoopMatrixElementEvaluator.get_me_value(
                'fixture', 0, str(self.root), skip_compilation=True, verbose=False)
        self.assertEqual(result, 'fresh result')

    def test_stability_requires_explicit_tirust_selection(self):
        process = base_objects.Process({'has_born': True})
        checker = SimpleNamespace(loop_optimized_output=True,
            check_matrix_element_stability=mock.Mock(return_value={}))
        for options, expected in (({}, [6]), ({'MLReductionLib':[9]}, [9])):
            with self.subTest(options=options), \
                    mock.patch.object(process_checks, 'generate_loop_matrix_element', return_value=({}, process)), \
                    mock.patch.object(process_checks, 'LoopMatrixElementTimer', return_value=checker):
                process_checks.check_stability({'model':None}, options={'reuse':True},
                    tir={'ninja_dir':'installed', 'tirust_dir':'installed'}, MLOptions=options)
            passed = checker.check_matrix_element_stability.call_args.kwargs['MLOptions']
            self.assertEqual(passed['MLReductionLib'], expected)
            self.assertEqual(passed['TIRustGlobalCache'], 0)
