"""Pure admission checks; native lifecycle tests are a separately admitted stage."""
import importlib.util
import json
import os
from pathlib import Path
import re
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('lazy_clear_run', HERE / 'run.py')
run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run)
target_spec = importlib.util.spec_from_file_location('lazy_clear_exec_target', HERE / 'exec_target.py')
exec_target = importlib.util.module_from_spec(target_spec)
target_spec.loader.exec_module(exec_target)


class SourceChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.binding = json.loads((HERE / 'bindings.json').read_text())
        cls.original = (Path(cls.binding['process']) / 'TIRUST_interface.f').read_text()

    def test_only_one_generated_procedure_changes(self):
        candidate = run.replacement(self.original)
        delta = json.loads((HERE / 'replacement.json').read_text())
        self.assertEqual(candidate.replace(delta['new'], delta['old']), self.original)
        self.assertEqual(candidate.count(delta['new']), 1)

    def test_partial_or_ambiguous_input_fails(self):
        for value in ('', self.original + self.original, run.replacement(self.original)):
            with self.assertRaises(ValueError):
                run.replacement(value)

    def test_existing_public_api_and_shared_context_are_used(self):
        body = run.wrapper(self.original, True)
        self.assertEqual(body.count('TIRUST_CONTEXT_GENERATION().GE.0'), 2)
        self.assertNotIn('LAST_GENERATION', body)
        self.assertNotIn('MLReductionLib', body)
        self.assertNotIn('RETURN', body)
        self.assertEqual(body.count('CALL TIRUST_CLEAR_CACHE()'), 2)

    def test_control_fragment_is_unchanged(self):
        body = run.wrapper(self.original, False)
        self.assertNotIn('TIRUST_CONTEXT_GENERATION', body)
        self.assertEqual(body.count('CALL TIRUST_CLEAR_CACHE()'), 2)

    def test_cards_only_change_the_backend(self):
        original = (HERE / 'fixtures/MadLoopParams.dat').read_text()
        for backend in (1, 9):
            candidate = run.card(backend)
            self.assertEqual(re.sub(r'(?m)(^#MLReductionLib\s*\n)[^\n]+', r'\g<1>9', candidate), original)
            self.assertIn('\n#CTModeRun\n-1\n', candidate)
            self.assertIn('\n#LoopPoleMode\n0\n', candidate)

    def test_numerical_parity_is_strict(self):
        run.values_equal(b'1 2 3\n', b'1 2 3\n')
        for other in (b'1 2 4\n', b'1 2 3', b'1 2 3 \n'):
            with self.assertRaises(ValueError):
                run.values_equal(b'1 2 3\n', other)

    def test_runtime_parser_keeps_direct_interpreter(self):
        actual = run.runtime_paths('linux-vdso.so.1 (0x1)\nlibx.so => /tmp/x.so (0x2)\n/tmp/ld.so (0x3)\n')
        self.assertEqual(actual, {'/tmp/x.so', '/tmp/ld.so'})

    def test_c_driver_has_no_numerical_interposition(self):
        text = (HERE / 'observer.c').read_text()
        self.assertNotIn('evaluate_dp', text)
        self.assertNotIn('evaluate_mp', text)
        self.assertNotIn('fenv', text)
        self.assertNotIn('float', text)
        self.assertNotIn('double', text)

    def test_bounded_scope(self):
        b = self.binding
        self.assertEqual(b['main_backend_ids'], [1, 9])
        self.assertEqual(b['pairs'], 7)
        self.assertFalse(b['cpu_exclusive'])
        self.assertFalse(b['production_change_authorized'])
        self.assertFalse(b['performance_acceptance_eligible'])
        self.assertEqual(len(b['objects']), 18)
        self.assertEqual(len(b['runtime']), 15)
        self.assertEqual(b['global_wall_seconds'], 900 - b['prior_elapsed_seconds'])

    def test_seed_and_epoch_lifecycle_cases_are_present(self):
        text = (HERE / 'lifecycle.f90').read_text()
        for value in ('call a_clear()', 'call b_clear()', 'call tirust_clear_cache(status)',
                      'call tirust_tensor_evaluate(', 'call tirust_error_message(',
                      'call ieee_get_rounding_mode(', 'call ieee_get_flag(',
                      'tirust_context_generation()<=second', 'counts(7)/=0'):
            self.assertIn(value, text)

    def test_clean_outer_and_target_only_loader_environment(self):
        with patch.dict(os.environ, {'LD_LIBRARY_PATH': '/wrong', 'LD_PRELOAD': '/wrong.so',
                'LD_TRACE_LOADED_OBJECTS': '1', 'LD_AUDIT': '/wrong-audit.so', 'LD_DEBUG': 'libs'}):
            self.assertFalse(any(k.startswith('LD_') for k in run.outer_environment()))
            normal = exec_target.target_environment('/pinned-providers', False)
            trace = exec_target.target_environment('/pinned-providers', True)
        self.assertEqual({k: v for k, v in normal.items() if k.startswith('LD_')},
            {'LD_LIBRARY_PATH': '/pinned-providers'})
        self.assertEqual({k: v for k, v in trace.items() if k.startswith('LD_')},
            {'LD_LIBRARY_PATH': '/pinned-providers', 'LD_TRACE_LOADED_OBJECTS': '1'})

    def test_final_exec_receives_target_only_environment(self):
        with patch.object(exec_target.sys, 'argv', ['helper', '/providers', '0', '/program', 'arg']), \
             patch.object(exec_target.os, 'execve') as actual:
            exec_target.main()
        path, argv, env = actual.call_args.args
        self.assertEqual(path, '/program')
        self.assertEqual(argv, ['/program', 'arg'])
        self.assertEqual(env['LD_LIBRARY_PATH'], '/providers')
        self.assertNotIn('LD_TRACE_LOADED_OBJECTS', env)

    def test_lock_precedes_new_namespace(self):
        source = (HERE / 'run.py').read_text()
        self.assertLess(source.index('fcntl.flock(lock,'), source.index('WORK.mkdir('))

    def test_observed_plain_parity_has_both_arm_joins(self):
        source = (HERE / 'run.py').read_text()
        self.assertIn("observed_values[backend, 'original']", source)
        self.assertIn("observed_values[backend, 'candidate']", source)
        self.assertIn('values_equal(observed_values[backend, arm], raw)', source)

    def test_export_cards_layout_and_ident_identity(self):
        source = (HERE / 'run.py').read_text()
        self.assertIn("cards = directory / 'Cards'", source)
        self.assertIn("cards / 'param_card.dat'", source)
        self.assertIn("cards / 'ident_card.dat'", source)
        self.assertNotIn("directory / 'param_card.dat'", source)
        self.assertEqual(run.sha(HERE / 'fixtures/ident_card.dat'), self.binding['fixture_ident_card_sha256'])
        self.assertIn("CALL SETPARA('param_card.dat')", (HERE / 'worker.f').read_text())

    def test_reused_artifacts_are_bound_and_prior_failure_is_closed(self):
        b = self.binding
        for item in [*b['reuse_artifacts'].values(), b['reuse_loader_trace']]:
            self.assertEqual(b['dependencies'][item['path']], item['sha256'])
            self.assertEqual(run.sha(Path(item['path'])), item['sha256'])
        self.assertEqual(run.sha(HERE / 'prior-failed-run-index.json'), b['prior_failed_run_index_sha256'])


if __name__ == '__main__':
    unittest.main()
