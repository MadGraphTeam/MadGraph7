#!/usr/bin/env python3
"""Reject lost numerical/IEEE/trap evidence using retained tiny-fixture logs."""
import gzip
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
module_spec = importlib.util.spec_from_file_location('fixture_analysis', HERE / 'analyze.py')
analysis = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(analysis)

class AnalysisTests(unittest.TestCase):
    def setUp(self):
        work = ROOT / '.tirust-build/cayley-zero-denominator-guard-proposal-1/parser-tests'
        work.mkdir(exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=work)
        self.raw = Path(self.tmp.name) / 'raw'
        shutil.copytree(HERE / 'evidence/tiny-1/raw', self.raw)

    def tearDown(self):
        self.tmp.cleanup()

    def mutate_masked(self, predicate, transform):
        path = self.raw / 'opt3-masked.txt.gz'
        with gzip.open(path, 'rt') as f:
            lines = f.read().splitlines()
        for i, line in enumerate(lines):
            v = line.split()
            if predicate(v):
                transform(v)
                lines[i] = ' '.join(v)
                break
        else:
            self.fail('mutation target absent')
        with gzip.open(path, 'wt') as f:
            f.write('\n'.join(lines) + '\n')

    def test_retained_passes(self):
        self.assertTrue(analysis.analyze(self.raw)['passed'])

    def test_nonzero_value_difference_rejected(self):
        self.mutate_masked(lambda v: v[2] == 'candidate' and v[14] == 'some' and v[4] == '3ff0000000000000',
                           lambda v: v.__setitem__(15, '3ff0000000000001'))
        with self.assertRaises(AssertionError):
            analysis.analyze(self.raw)

    def test_existing_flags_cleared_rejected(self):
        self.mutate_masked(lambda v: v[2] == 'candidate' and v[5] == '48', lambda v: v.__setitem__(9, '0'))
        with self.assertRaises(AssertionError):
            analysis.analyze(self.raw)

    def test_blanket_nonfinite_guard_rejected(self):
        self.mutate_masked(lambda v: v[2] == 'candidate' and v[4] == '7ff0000000000000' and v[14] == 'some',
                           lambda v: v.__setitem__(14, 'none'))
        with self.assertRaises(AssertionError):
            analysis.analyze(self.raw)

    def test_negativezero_division_not_skipped_rejected(self):
        self.mutate_masked(lambda v: v[2] == 'candidate' and v[3] == '3ff0000000000000' and v[4] == '8000000000000000' and v[5] == '0',
                           lambda v: v.__setitem__(9, '4'))
        with self.assertRaises(AssertionError):
            analysis.analyze(self.raw)

    def test_missing_row_rejected(self):
        path = self.raw / 'opt3-masked.txt.gz'
        with gzip.open(path, 'rt') as f:
            lines = f.readlines()
        with gzip.open(path, 'wt') as f:
            f.writelines(lines[:-1])
        with self.assertRaises(AssertionError):
            analysis.analyze(self.raw)

    def test_nonzero_overflow_trap_suppressed_rejected(self):
        path = self.raw / 'opt3-traps.jsonl'
        records = [json.loads(x) for x in path.read_text().splitlines()]
        for record in records:
            if record['id'] == 'overflow_control' and record['mode'] == 'candidate':
                record['returncode'] = 0
                break
        path.write_text(''.join(json.dumps(r) + '\n' for r in records))
        with self.assertRaises(AssertionError):
            analysis.analyze(self.raw)

if __name__ == '__main__':
    unittest.main()
