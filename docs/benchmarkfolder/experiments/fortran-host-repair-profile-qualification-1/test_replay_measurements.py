"""Small synthetic checks for the file-free measurement replay component."""
import hashlib
import unittest
import replay_measurements as replay


class MeasurementReplayChecks(unittest.TestCase):
    def test_plain_bytes_mapping_and_source_seal_required(self):
        for bad in ([], {'source/source-index.json': bytearray(b'bad')},
                    {'source/source-index.json': b'{}'}):
            with self.assertRaises(ValueError):replay.replay(bad)

    def test_changed_parser_rejected_before_execution(self):
        data = {'source/analyze.py': b"raise RuntimeError('must not execute')\n"}
        with self.assertRaisesRegex(ValueError, 'Changed approved pure parser'):
            replay._module(data, 'source/analyze.py', replay.ANALYZE_SHA256)

    def test_parser_imports_use_only_approved_preloaded_modules(self):
        allowed = b'import math\nx=math.sqrt(4)\n'
        result = replay._module({'p': allowed}, 'p', hashlib.sha256(allowed).hexdigest())
        self.assertEqual(result['x'], 2)
        forbidden = b'import os\n'
        with self.assertRaisesRegex(ValueError, 'Unapproved pure-parser import'):
            replay._module({'p': forbidden}, 'p', hashlib.sha256(forbidden).hexdigest())

    def test_duplicate_json_and_nonfinite_constants_rejected(self):
        for value in (b'{"a":1,"a":2}', b'{"a":NaN}', b'{"a":Infinity}'):
            with self.assertRaises(ValueError):replay._json({'x': value}, 'x')

    def test_json_comparison_preserves_boolean_numeric_distinctions(self):
        replay._same({1: [0.0]}, {'1': [0.0]}, 'same')
        for left, right in ((True, 1), (False, 0), (1, 1.0)):
            with self.assertRaises(ValueError):replay._same(left, right, 'typed')

    def test_literal_provenance_join_is_not_a_filesystem_read(self):
        payload = b'file-free\n'
        row = {'sha256': hashlib.sha256(payload).hexdigest(), 'bytes': len(payload)}
        pin = {'files': {'historical/input.dat': row}}
        replay._bound_input({'x': payload}, pin, 'x', '/unavailable/root/historical/input.dat')
        with self.assertRaises(ValueError):
            replay._bound_input({'x': payload+b'changed'}, pin, 'x', '/unavailable/root/historical/input.dat')
        pin['files']['input.dat'] = row
        with self.assertRaisesRegex(ValueError, 'Ambiguous'):
            replay._bound_input({'x': payload}, pin, 'x', '/unavailable/root/historical/input.dat')

    def test_timing_and_decimal_words(self):
        self.assertEqual(replay._timing('0.01 1.2\n'), [0.01, 1.2])
        for text in ('0 1', '-1 1', 'nan 1', '1 inf', '1 2 3'):
            with self.assertRaises(ValueError):replay._timing(text)
        row = replay._decimal_rows('1 0 219 -0.0 1.0 2.0 3.0 0.0\n')[0]
        self.assertEqual(row['words'][0], '8000000000000000')
        for text in ('1 0 219 0 0\n', '1 0 219 nan 0 0 0 0\n'):
            with self.assertRaises(ValueError):replay._decimal_rows(text)


if __name__ == '__main__':unittest.main()
