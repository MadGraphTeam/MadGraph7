#!/usr/bin/env python3
"""Synthetic offline validator checks; no subprocesses or native execution."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location('classifier_mock_validator', HERE / 'validate.py')
V = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(V)
GOOD = ' CLASSIFIER_MOCK_COMPLETE           8           8\n CLASSIFIER_MOCK_COMPLETE          16           8\n COMPLETE_CLASSIFIER_MOCK_ONLY\n'


class ValidatorTests(unittest.TestCase):
    def test_frozen_sources(self):
        V.verify_sources()

    def test_complete(self):
        result = V.validate_text(GOOD)
        self.assertEqual(result['completed_overloads_total'], 16)
        self.assertFalse(result['actual_analytic_execution_qualified'])

    def test_missing_duplicate_reordered(self):
        lines = GOOD.splitlines(keepends=True)
        for malformed in (''.join(lines[:-1]), ''.join(lines[1:]), GOOD + lines[-1],
                          ''.join((lines[1], lines[0], lines[2]))):
            with self.subTest(text=malformed), self.assertRaises(ValueError):
                V.validate_text(malformed)

    def test_bad_counts_or_precision(self):
        for old, new in (('16', '8'), ('8           8', '8           7')):
            with self.assertRaises(ValueError):
                V.validate_text(GOOD.replace(old, new, 1))

    def test_errors_and_skips(self):
        for prefix in ('MOCK_FAILURE 8 101\n', 'ERROR STOP 1\n', 'SUBNORMAL_FIXTURE_UNSUPPORTED 8\n', '\n'):
            with self.subTest(prefix=prefix), self.assertRaises(ValueError):
                V.validate_text(prefix + GOOD)
        with self.assertRaises(ValueError):
            V.validate_text(GOOD, 'unexpected runtime warning\n')

    def test_truncated_and_control(self):
        for malformed in (GOOD[:-1], GOOD + '\x00', GOOD + '\x1b[0m', GOOD + 'x\n'):
            with self.subTest(text=malformed), self.assertRaises(ValueError):
                V.validate_text(malformed)

    def test_strict_event_and_closure(self):
        with tempfile.TemporaryDirectory(dir=HERE) as directory:
            path = Path(directory);out = path / 'stdout.log';err = path / 'stderr.log'
            out.write_text(GOOD);err.write_text('')
            event = {'returncode': 0, 'timed_out': False,
                     'output_files': {p.name: V.sha(p) for p in (out, err)}}
            self.assertTrue(V.validate_files(out, err, event, path)['passed'])
            for field, value in (('returncode', 1), ('returncode', False), ('timed_out', True),
                                 ('timed_out', None), ('error', 'failure'), ('interrupted', True),
                                 ('output_files', {})):
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    V.validate_files(out, err, dict(event, **{field: value}), path)
            (path / 'fort.99').write_text('unexpected')
            with self.assertRaises(ValueError):
                V.validate_files(out, err, event, path)


if __name__ == '__main__':
    unittest.main(verbosity=2)
