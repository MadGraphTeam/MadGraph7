"""Default backend selection and ownership of caller-supplied check options."""

import copy
from types import SimpleNamespace
import unittest
from unittest import mock

from madgraph.core import base_objects
from madgraph.various import ninjars_install, process_checks


class TestReductionDefaults(unittest.TestCase):
    def test_default_selection_returns_independent_lists(self):
        with mock.patch.object(ninjars_install, 'is_available', return_value=True):
            first = ninjars_install.default_reduction_library(True, 'installed')
            first.append(1)
            self.assertEqual(
                ninjars_install.default_reduction_library(True, 'installed'), [8])
        with mock.patch.object(ninjars_install, 'is_available', return_value=False):
            self.assertEqual(
                ninjars_install.default_reduction_library(True, 'missing'), [6])
        with mock.patch.object(ninjars_install, 'is_available') as available:
            self.assertEqual(
                ninjars_install.default_reduction_library(False, 'installed'), [6])
            available.assert_not_called()

    def run_check(self, name, mloptions, has_born=True, reuse=False,
                  optimized=True, timing_result=None):
        process = base_objects.Process({'has_born': has_born})
        matrix = process if reuse else {'processes': [process]}
        timer = SimpleNamespace(loop_optimized_output=optimized,
            time_matrix_element=mock.Mock(return_value=timing_result),
            check_matrix_element_stability=mock.Mock(return_value={}))
        with mock.patch.object(process_checks, 'generate_loop_matrix_element',
                               return_value=({}, matrix)), \
                mock.patch.object(process_checks, 'LoopMatrixElementTimer',
                                  return_value=timer), \
                mock.patch.object(ninjars_install, 'is_available', return_value=True):
            result = getattr(process_checks, name)(
                {'model': None, 'has_born': not has_born},
                options={'reuse': reuse}, tir={'ninjars_dir': 'installed'},
                MLOptions=mloptions)
        return result, timer

    def test_timing_and_profile_defaults_use_generated_born_status(self):
        for name in ('check_timing', 'check_profile'):
            for has_born in (True, False):
                for reuse in (True, False):
                    for options in ({}, {'MLReductionLib': []}):
                        with self.subTest(name=name, born=has_born,
                                          reuse=reuse, options=options):
                            original = copy.deepcopy(options)
                            result, timer = self.run_check(name, options,
                                has_born=has_born, reuse=reuse, timing_result={})
                            expected = [8] if has_born else [6]
                            selected = timer.time_matrix_element.call_args.kwargs['MLOptions']
                            self.assertEqual(selected['MLReductionLib'], expected)
                            self.assertEqual(options, original)
                            if name == 'check_profile':
                                self.assertEqual(timer.check_matrix_element_stability
                                    .call_args.kwargs['MLOptions']['MLReductionLib'], expected)

    def test_explicit_selection_and_options_survive_each_check(self):
        for name in ('check_timing', 'check_profile', 'check_stability'):
            for has_born in (True, False):
                options = {'MLReductionLib': [7, 6, 1], 'COLLIERGlobalCache': 9}
                original = copy.deepcopy(options)
                with self.subTest(name=name, born=has_born):
                    _, timer = self.run_check(name, options,
                        has_born=has_born, timing_result={})
                    invocation = (timer.check_matrix_element_stability if name ==
                                  'check_stability' else timer.time_matrix_element)
                    passed = invocation.call_args.kwargs['MLOptions']
                    self.assertEqual(passed['MLReductionLib'], [7, 6, 1])
                    self.assertEqual(options, original)
                    # Downstream per-library setup also owns its backend list.
                    passed['MLReductionLib'].append(8)
                    self.assertEqual(options, original)

    def test_stability_comparison_defaults_do_not_leak_to_next_call(self):
        options = {}
        _, timer = self.run_check('check_stability', options)
        passed = timer.check_matrix_element_stability.call_args.kwargs['MLOptions']
        self.assertEqual(passed['MLReductionLib'], [8])
        self.assertEqual(passed['COLLIERGlobalCache'], 0)
        self.assertEqual(options, {})
        _, timer = self.run_check('check_timing', options, has_born=False,
                                 timing_result={})
        self.assertEqual(timer.time_matrix_element.call_args.kwargs
                         ['MLOptions']['MLReductionLib'], [6])
        self.assertEqual(options, {})

    def test_stability_backend_setup_does_not_replace_callers_list(self):
        options = {'MLReductionLib': [1, 6]}
        process = base_objects.Process()
        timer = SimpleNamespace(tir_dir={}, output_path='/unused',
                                setup_process=mock.Mock(return_value=None))
        with mock.patch.object(base_objects.Process, 'shell_string', return_value='0_test'):
            result = process_checks.LoopMatrixElementTimer.check_matrix_element_stability(
                timer, process, options={'reuse': True, 'npoints': 0, 'split_orders': -1},
                MLOptions=options)
        self.assertIsNone(result)
        self.assertEqual(timer.setup_process.call_args.args[4]['MLReductionLib'], 1)
        self.assertEqual(options, {'MLReductionLib': [1, 6]})

    def test_nonoptimized_checks_report_cuttools_without_changing_options(self):
        for name in ('check_timing', 'check_profile'):
            options = {'MLReductionLib': [8]}
            with self.subTest(name=name):
                result, timer = self.run_check(name, options, optimized=False,
                                              timing_result={})
                timing = result[0] if name == 'check_profile' else result
                self.assertEqual(timing['reduction_tool'], 1)
                self.assertEqual(timer.time_matrix_element.call_args.kwargs['MLOptions'], {})
                self.assertEqual(options, {'MLReductionLib': [8]})

    def test_profile_returns_failure_without_mutating_options(self):
        options = {}
        result, timer = self.run_check('check_profile', options, timing_result=None)
        self.assertEqual(result, (None, None))
        timer.check_matrix_element_stability.assert_not_called()
        self.assertEqual(options, {})
