"""Metadata grammar controls; no MP arithmetic or timing execution."""
from pathlib import Path
import runpy
import unittest

check = runpy.run_path(str(Path(__file__).with_name('run.py')))['check_float_metadata']


class TraceFloatMetadata(unittest.TestCase):
    def test_rug_zero_spellings_and_finite_values(self):
        for text, negative in [('0', False), ('-0', True), ('0.0e0', False), ('-0.0e0', True),
                               ('1.234567890123456789e-420', False), ('-2e+1300', True)]:
            with self.subTest(text=text):
                check({'precision': 256, 'negative': negative, 'decimal': text})

    def test_wrong_zero_sign_nonfinite_or_invalid_metadata_rejected(self):
        bad = [dict(precision=256, negative=False, decimal='-0'),
               dict(precision=256, negative=True, decimal='0'),
               dict(precision=256, negative=False, decimal='NaN'),
               dict(precision=256, negative=False, decimal='inf'),
               dict(precision=256, negative=False, decimal='1e2 trailing'),
               dict(precision=0, negative=False, decimal='0'),
               dict(precision=True, negative=False, decimal='0'),
               dict(precision=256, negative=0, decimal='0')]
        for value in bad:
            with self.subTest(value=value), self.assertRaises(AssertionError):
                check(value)


if __name__ == '__main__':
    unittest.main()
