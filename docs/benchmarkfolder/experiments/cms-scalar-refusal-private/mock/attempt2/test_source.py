#!/usr/bin/env python3
"""Offline source/recipe tests only; native process creation is forbidden."""
import ast
import contextlib
import copy
import importlib.util
import io
import json
import os
from pathlib import Path
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


P = load('attempt2_source_preparation', HERE/'prepare.py')
C = load('attempt2_source_control', HERE/'control.py')


class LauncherCorrection(unittest.TestCase):
    def setUp(self):
        self.old = json.loads((P.PRIOR/'prepared.json').read_text())
        self.new = json.loads((HERE/'prepared.json').read_text())

    def test_prior_evidence_exact(self):
        for name, digest in P.PRIOR_HASHES.items():
            self.assertEqual(P.sha(P.PRIOR/name), digest)

    def test_exact_recipe_and_unchanged_runtime(self):
        expected = P.proposal(self.old)
        for key in expected:
            if key != 'inputs':
                self.assertEqual(self.new[key], expected[key], key)
        for name, digest in self.old['inputs'].items():
            self.assertEqual(self.new['inputs'][name], digest)
        library_path = ':'.join(self.old['runtime_directories'])
        for old, new in zip(self.old['loader_checks'], self.new['loader_checks'], strict=True):
            self.assertEqual(old['expected'], new['expected'])
            self.assertEqual(new['job']['argv'], [old['loader'], '--library-path', library_path, '--list', str(P.ROOT/old['executable'])])
        for job, check in zip(self.new['mock_jobs'], self.new['loader_checks'], strict=True):
            self.assertEqual(job['argv'], [check['loader'], '--library-path', library_path, str(P.ROOT/check['executable'])])

    def test_guards_reject_recipe_and_environment_drift(self):
        for category in ('loader', 'mock', 'environment'):
            changed = copy.deepcopy(self.old)
            if category == 'loader': changed['loader_checks'][0]['job']['argv'].append('--bad')
            if category == 'mock': changed['mock_jobs'][0]['argv'].append('unexpected')
            if category == 'environment': changed['compiler_environment']['LD_LIBRARY_PATH'] = 'bad'
            with self.subTest(category=category), self.assertRaises(ValueError):
                P.proposal(changed)

    def test_execution_helper_unchanged(self):
        old = ast.parse((P.PRIOR/'control.py').read_text())
        new = ast.parse((HERE/'control.py').read_text())
        for name in ('load', 'save', 'inventory', 'execute'):
            before = next(x for x in old.body if isinstance(x, ast.FunctionDef) and x.name == name)
            after = next(x for x in new.body if isinstance(x, ast.FunctionDef) and x.name == name)
            self.assertEqual(ast.dump(before), ast.dump(after))
        text = (HERE/'control.py').read_text()
        self.assertNotIn("env['LD_LIBRARY_PATH'] =", text)
        self.assertIn("env = base.environment(spec)", text)
        self.assertEqual(C.CMS, P.CMS)
        self.assertEqual(C.ROOT, P.ROOT)
        self.assertEqual(C.HERE, HERE)

    def test_default_and_explicit_verify_are_read_only(self):
        before = {str(p.relative_to(HERE)):P.sha(p) for p in HERE.rglob('*') if p.is_file()}
        for argv in (['control.py'], ['control.py', 'verify']):
            stream = io.StringIO()
            with mock.patch.dict(os.environ, self.old['compiler_environment'], clear=True), \
                 mock.patch('sys.argv', argv), \
                 mock.patch('subprocess.Popen', side_effect=AssertionError('Native launch forbidden')), \
                 contextlib.redirect_stdout(stream):
                C.main()
            result = json.loads(stream.getvalue())
            self.assertFalse(result['process_launched'])
            self.assertEqual(result['loader_checks'], 2)
            self.assertEqual(result['mock_cases'], 2)
        after = {str(p.relative_to(HERE)):P.sha(p) for p in HERE.rglob('*') if p.is_file()}
        self.assertEqual(before, after)
        self.assertFalse((HERE/'run.json').exists())


if __name__ == '__main__':
    unittest.main(verbosity=2)
