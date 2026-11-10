"""Source-derived branch-state regressions; no numerical executable is loaded."""
from fractions import Fraction
import copy
import unittest
import analyze
import test_protocol as fixture


def mutate_momentum(rows, kind, part, data):
    row = fixture.find(rows, kind, part)
    row['data'] = data + row['data'][len(data):]


class TransitionTests(unittest.TestCase):
    def test_born_must_use_original_public_point(self):
        rows = fixture.sample()
        mutate_momentum(rows, 10, 0, fixture.D(1))
        with self.assertRaisesRegex(ValueError, 'Born is not using original public'):
            fixture.parse(rows)

    def test_dp_repair_must_start_from_original_public_point(self):
        rows = fixture.sample()
        mutate_momentum(rows, 20, 0, fixture.D(1))
        with self.assertRaisesRegex(ValueError, 'DP repair does not start from original public'):
            fixture.parse(rows)

    def test_qp_repair_must_losslessly_widen_original_point(self):
        rows = fixture.sample()
        mutate_momentum(rows, 21, 0, fixture.q(1))
        with self.assertRaisesRegex(ValueError, 'QP repair does not start from losslessly widened'):
            fixture.parse(rows)

    def test_qp_low_bits_are_not_narrowed_to_dp(self):
        rows = fixture.sample()
        # Keep all three required original DP states at1, then perturb only
        # QP-before below DP resolution. Exact rational comparison must reject.
        mutate_momentum(rows, 1, 0, fixture.D(1))
        mutate_momentum(rows, 10, 0, fixture.D(1))
        mutate_momentum(rows, 20, 0, fixture.D(1))
        mutate_momentum(rows, 21, 0, fixture.q(Fraction(1) + Fraction(1, 2**100)))
        with self.assertRaisesRegex(ValueError, 'QP repair does not start from losslessly widened'):
            analyze.observed_inputs([r for r in rows if r['call']==101], 9, 4)

    def test_qp_before_signed_zero_mismatch_is_rejected(self):
        rows = fixture.sample()
        mutate_momentum(rows, 21, 0, (1<<127).to_bytes(16,'little'))
        with self.assertRaisesRegex(ValueError, 'QP repair does not start from losslessly widened'):
            fixture.parse(rows)

    def test_exact_widening_normal_subnormal_and_signed_zero(self):
        import struct
        values=[0.0,-0.0,1.0,-1.0,1.5,2.**-1022,float.fromhex('0x0.0000000000001p-1022'),
                float.fromhex('0x1.fffffffffffffp1023')]
        raw=struct.pack('<'+'d'*len(values),*values)
        widened=analyze.widen_binary64(raw)
        self.assertEqual(analyze.binary(raw,64),analyze.binary(widened,128))
        self.assertEqual(widened[:16],bytes(16))
        self.assertEqual(widened[16:32],(1<<127).to_bytes(16,'little'))
        self.assertEqual(widened[32:48],(16383<<112).to_bytes(16,'little'))

    def test_nonfinite_or_misaligned_widening_is_rejected(self):
        for raw in (b'partial',fixture.D(float('nan')),fixture.D(float('inf'))):
            with self.subTest(raw=raw),self.assertRaises(ValueError):
                analyze.widen_binary64(raw)

    def test_legitimate_repair_difference_is_retained_separately(self):
        rows = fixture.sample()
        mutate_momentum(rows, 20, 1, fixture.D(1))
        parsed = fixture.parse(rows)
        transition = parsed['points'][0]['inputs']['input_transitions']
        self.assertFalse(transition['born_and_dp_repaired_inputs_identical'])
        self.assertEqual(transition['dp_repair_changed_components'], [0])
        self.assertNotEqual(transition['born_entry_dp_hex'], transition['dp_repair_after_hex'])
        self.assertEqual(transition['born_entry_dp_hex'], transition['public_original_dp_hex'])

    def test_changed_dp_after_remains_in_cross_worker_common_input_digest(self):
        before = fixture.parse(fixture.sample())
        rows = fixture.sample(); mutate_momentum(rows, 20, 1, fixture.D(1))
        after = fixture.parse(rows)
        self.assertNotEqual(before['points'][0]['inputs']['shared_input_sha256'],
                            after['points'][0]['inputs']['shared_input_sha256'])
        workers = [fixture.parse(fixture.sample(b, m, o), b, m, o)
                   for b in analyze.BACKENDS for m in (4, 5) for o in (0, 1)]
        index = next(i for i, w in enumerate(workers) if (w['backend'], w['mode'], w['observed']) == (9, 4, 1))
        workers[index] = after
        result = analyze.adjudicate(workers)
        failed = [r for r in result['comparisons'] if not r['shared_physical_born_repair_numerator_equal']]
        self.assertTrue(failed)
        self.assertTrue(all(row['coefficients'] is None for row in failed))

    def test_all_original_and_repaired_transition_bytes_are_retained(self):
        parsed = fixture.parse(fixture.sample())
        transition = parsed['points'][0]['inputs']['input_transitions']
        dp = ('public_original_dp_hex', 'born_entry_dp_hex', 'born_return_dp_hex',
              'dp_repair_before_hex', 'dp_repair_after_hex')
        qp = ('qp_repair_before_hex', 'qp_repair_after_hex', 'mp_construction_before_hex', 'mp_construction_after_hex')
        self.assertTrue(all(len(bytes.fromhex(transition[name])) == 160 for name in dp))
        self.assertTrue(all(len(bytes.fromhex(transition[name])) == 320 for name in qp))


if __name__ == '__main__':
    unittest.main(verbosity=2)
