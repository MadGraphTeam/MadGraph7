"""Synthetic exact-word tests only; never read a native trace or load a library."""
from fractions import Fraction as F
import importlib.util
import json
from pathlib import Path
import struct
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('kinematic_decoder', HERE / 'decode_trace.py')
D = importlib.util.module_from_spec(spec)
spec.loader.exec_module(D)


def dp(value):
    return [struct.pack('>d', value).hex().upper()]


def qp_integer(value):
    if not value:
        return ['0000000000000000', '0000000000000000']
    sign = int(value < 0)
    number = abs(value)
    exponent = number.bit_length() - 1
    fraction = (number << (112 - exponent)) - (1 << 112)
    bits = (sign << 127) | ((exponent + 16383) << 112) | fraction
    return [f'{bits & ((1 << 64)-1):016X}', f'{bits >> 64:016X}']


def fixture(case, mode, stop=None, native_status=0):
    derivation, route, _ = D.source_binding(case)
    target = derivation['target_identities'][0]
    n, rank, gid = 4, target['rank'], target['reduction_identifier']
    if case == 'cms':
        p = [[10, 0, 0, 10], [10, 0, 0, -10], [8, 1, 0, 6], [11, -1, 0, -6]]
        corners = [p[0], p[1], [-v for v in p[3]], [-v for v in p[2]]]
        pair = (2, 4)
    else:
        p = [[10, 0, 0, 10], [10, 0, 0, -10], [7, 1, 0, 3], [6, -1, 0, -3], [6, 0, 0, 0]]
        corners = [[-v for v in p[k]] for k in (2, 3, 4)] + [[p[0][k] + p[1][k] for k in range(4)]]
        pair = (1, 2)
    r = [sum(v[k] for v in corners) for k in range(4)]
    shifts = [r] + [[sum(corners[j][k] for j in range(end)) for k in range(4)] for end in (1, 2, 3)]
    masses = [25, 0] * n
    lines = ['KT_SCHEMA_V2 7 64 128 64 32']
    suffix, encode = ('DP', dp) if mode == 1 else ('QP', qp_integer)

    def event(name, values, i=0, j=0, branch=0, precision=suffix, encoder=encode):
        h = [7, 1, gid, mode, rank, n, i, j, branch, 1]
        words = [w for v in values for w in encoder(v)]
        lines.append('KT_' + name + '_' + precision + ' ' + ' '.join(map(str, h)) + ' ' + ' '.join(words))

    event('LOOP', [v for x in corners for v in x] + masses, precision='DP', encoder=dp)
    if mode == 4:
        event('LOOP', [v for x in corners for v in x])
    for k, vector in enumerate(p, 1):
        sign = route['external_provenance'][k-1]['stored_helas_sign']
        event('EXTERNAL', [v for x in vector for v in (sign*x, 0)], i=k)
    if stop == 'loop':
        return ('\n'.join(lines) + '\n').encode(), derivation, route
    event('RESIDUAL', r + [v for x in corners for v in x])
    if stop == 'residual':
        return ('\n'.join(lines) + '\n').encode(), derivation, route
    event('MATRIX', [shifts[j][k] for k in range(4) for j in range(n)] + masses)
    i, j = pair
    diff = D.squared(D.subtract(shifts[i-1], shifts[j-1]))
    event('RAW', [diff, 0, diff-50, 0], i, j)
    h = [7, 1, gid, mode, rank, n, i, j, 0, 1]
    lines.append('KT_SCALE_' + suffix + ' ' + ' '.join(map(str, h)) + ' ' + ' '.join(encode(100)+dp(1e-8)))
    event('FINAL', [diff-50, 0], i, j)
    inv = [D.squared(D.subtract(shifts[i], shifts[j])) for j in range(n) for i in range(n)]
    event('TIRUST', [v for x in shifts for v in x] + masses + inv + [100])
    if stop != 'native_inputs':
        lines.append('KT_STATUS_' + suffix + ' ' + ' '.join(map(str, [7, 1, gid, mode, rank, n, 0, 0, 0, 1, native_status])))
    return ('\n'.join(lines) + '\n').encode(), derivation, route


class DecoderTests(unittest.TestCase):
    def test_dp_and_qp_one_and_quad_low_bit(self):
        self.assertEqual(D.ieee(['3FF0000000000000'], 64, 'low-first')['value'], F(1))
        self.assertEqual(D.ieee(['0000000000000001', '3FFF000000000000'], 128, 'low-first')['value'], 1 + F(1, 1 << 112))

    def test_subnormals_and_signed_zero_preserved(self):
        self.assertEqual(D.ieee(['0000000000000001'], 64, 'low-first')['value'], F(1, 1 << 1074))
        tiny = D.ieee(['0000000000000001', '0000000000000000'], 128, 'low-first')
        self.assertEqual(tiny['value'], F(1, 1 << 16494))
        self.assertEqual(D.ieee(['8000000000000000'], 64, 'low-first')['sign'], 1)
        json.dumps(D.wire(tiny), allow_nan=False)

    def test_nonfinite_is_retained_without_arithmetic(self):
        nan = D.ieee(['0000000000000001', '7FFF000000000000'], 128, 'low-first')
        self.assertEqual(nan['kind'], 'nan')
        self.assertEqual(nan['payload_hex'], '0x1')
        self.assertIsNone(nan['value'])
        self.assertEqual(D.ieee(['FFF0000000000000'], 64, 'low-first')['kind'], 'infinity')

    def test_explicit_abi_and_bad_word_refused(self):
        for order, words in [(None, ['3FF0000000000000']), ('high-first', ['3FF0000000000000']), ('low-first', ['0'])]:
            with self.subTest(order=order, words=words), self.assertRaises(ValueError):
                D.ieee(words, 64, order)

    def test_mixed_precision_scale_record_and_count(self):
        words = qp_integer(100) + dp(1e-8)
        decoded = D.payload('KT_SCALE_QP', words, 4, 'low-first')
        self.assertEqual(decoded['reference_normalization'][0]['value'], 100)
        self.assertEqual(decoded['osthres_dp'][0]['bits'], 64)
        with self.assertRaises(ValueError):
            D.payload('KT_SCALE_QP', words + ['0000000000000000'], 4, 'low-first')

    def test_both_routed_pairs_dp_and_qp_preserve_nonzero_residual(self):
        for case in ('cms', 'ew'):
            for mode in (1, 4):
                with self.subTest(case=case, mode=mode):
                    raw, derivation, route = fixture(case, mode)
                    result = D.analyze(raw, case, 'low-first', derivation, route)
                    call = result['calls'][0]
                    self.assertEqual(call['matrix_shift_pair']['difference_from_symbolic_routed_vector'], [0]*4)
                    self.assertEqual(call['conservation']['captured_zero_seeded_residual'], [1, 0, 0, 0])
                    self.assertNotEqual(call['exact_routing_on_captured_external_words']['square_shift_due_to_r'], 0)
                    self.assertTrue(call['matrix_masses_equal_dp_squared_inputs_exactly'])
                    self.assertEqual(call['critical_native_invariant'], call['matrix_shift_pair']['exact_square'])
                    self.assertFalse(call['numerically_qualified'])
                    json.dumps(D.wire(result), allow_nan=False)

    def test_early_returns_and_refusals_not_promoted(self):
        for stop in ('loop', 'residual', 'native_inputs'):
            raw, derivation, route = fixture('ew', 4, stop=stop)
            call = D.analyze(raw, 'ew', 'low-first', derivation, route)['calls'][0]
            self.assertIsNone(call['native_status'])
            self.assertFalse(call['numerically_qualified'])
            self.assertIn('unresolved', call['missing_boundary_scope'])
        raw, derivation, route = fixture('ew', 4, native_status=17)
        call = D.analyze(raw, 'ew', 'low-first', derivation, route)['calls'][0]
        self.assertEqual(call['native_status'], 17)
        self.assertFalse(call['numerically_qualified'])

    def test_captured_branch_stays_authoritative_when_ideal_predicate_differs(self):
        raw, derivation, route = fixture('cms', 1)
        final = next(line for line in raw.decode().splitlines() if line.startswith('KT_FINAL_DP'))
        words = final.split()
        words[9] = '3'
        before = ' '.join(['KT_ZERO_BEFORE_DP'] + words[1:])
        after = ' '.join(['KT_ZERO_AFTER_DP'] + words[1:])
        raw = raw.replace(final.encode(), (before + '\n' + after + '\n' + final).encode())
        result = D.analyze(raw, 'cms', 'low-first', derivation, route)
        decision = result['calls'][0]['threshold_decisions'][0]
        self.assertEqual(decision['original_branches']['ZERO'], 'taken')
        self.assertFalse(decision['exact_rational_predicates_not_fortran_replay']['ZERO'])
        self.assertFalse(result['accuracy_qualified'])
        interrupted = raw.replace((after + '\n').encode(), b'')
        call = D.analyze(interrupted, 'cms', 'low-first', derivation, route)['calls'][0]
        self.assertEqual(call['threshold_decisions'][0]['original_branches']['ZERO'], 'incomplete')

    def test_schema_without_calls_is_explicitly_unqualified(self):
        derivation, route, _ = D.source_binding('cms')
        result = D.analyze(b'KT_SCHEMA_V2 7 64 128 64 32\n', 'cms', 'low-first', derivation, route)
        self.assertTrue(result['no_selected_calls_observed'])
        self.assertFalse(result['accuracy_qualified'])

    def test_modes_two_and_five_refused(self):
        raw, _, _ = fixture('cms', 1)
        raw = raw.replace(b'KT_LOOP_DP 7 1 3 1 ', b'KT_LOOP_DP 7 1 3 2 ')
        with self.assertRaisesRegex(ValueError, 'mode1/4'):
            D.parse(raw, 'low-first')


if __name__ == '__main__':
    unittest.main()
