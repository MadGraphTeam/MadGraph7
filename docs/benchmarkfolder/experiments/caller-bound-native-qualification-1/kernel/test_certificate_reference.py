import unittest
from fractions import Fraction as F
from certificate_reference import (ZERO, pow2, certificate, exact_error_bound,
                                   source_chord_bound_exp, difference_bound_exp)
from derive_masks import pair_class


class CertificateTests(unittest.TestCase):
    def test_shell_complement_is_protected(self):
        self.assertEqual(pair_class([1,1,0,-1],[1,1,-1,-1],[True]*4),1)
        self.assertEqual(pair_class([1,1,-1,-1],[1,1,-1,-1],[True]*4),1)

    def test_offshell_singleton_unknown_composite_not_automatic_acceptance(self):
        self.assertEqual(pair_class([0,0,1,0],[1,1,-1,-1],[True,True,False,True]),0)
        self.assertEqual(pair_class([1,0,-1,0],[1,1,-1,-1],[True]*4),2)

    def test_resolved_transverse_dp_qp(self):
        q = [F(0), pow2(-18), F(0), F(0)]
        raw = -pow2(-36)
        for p, emin in [(53, -1022), (113, -16382)]:
            self.assertTrue(certificate(q, raw, [-36]*4, p, emin))
            self.assertLess(exact_error_bound(q, [-36]*4, p, emin), abs(raw))

    def test_all_protected_unknown_and_missing_contracts_fall_back(self):
        q, r = [F(0), F(1), F(0), F(0)], F(-1)
        for cls in [0, 1, 3, 99]:
            self.assertFalse(certificate(q, r, [ZERO]*4, pair_class=cls))
        for kw in [dict(mass_snap_seen=True), dict(have_bounds=False),
                   dict(environment_valid=False)]:
            self.assertFalse(certificate(q, r, [ZERO]*4, **kw))

    def test_lightlike_and_cancellation_neighborhood(self):
        q = [F(1), F(1), pow2(-40), F(0)]
        self.assertFalse(certificate(q, -pow2(-80), [ZERO]*4))
        self.assertFalse(certificate(q, F(0), [ZERO]*4))

    def test_boost_changes_actual_source_bound(self):
        q, r = [F(0), pow2(-18), F(0), F(0)], -pow2(-36)
        small = source_chord_bound_exp([10]*4, [ZERO]*4, [1]*4, 32)
        huge = source_chord_bound_exp([1010]*4, [ZERO]*4, [1]*4, 32)
        self.assertTrue(certificate(q, r, [small]*4))
        self.assertFalse(certificate(q, r, [huge]*4))

    def test_source_cancellation_does_not_cancel_multiplicity(self):
        b = source_chord_bound_exp([10], [ZERO], [2], 2)
        self.assertNotEqual(b, ZERO)
        self.assertGreaterEqual(pow2(b), 4*pow2(-53)*2*pow2(10))

    def test_zero_addition_count_still_carries_external_error(self):
        b = source_chord_bound_exp([10], [-43], [1], 0)
        self.assertGreaterEqual(pow2(b), pow2(-43))

    def test_subtraction_bound_includes_both_sources(self):
        b = difference_bound_exp(10, 10, -40, -41)
        self.assertGreaterEqual(pow2(b), pow2(-40)+pow2(-41)+pow2(-53)*2*pow2(10))

    def test_unsupported_error_and_operation_bounds(self):
        self.assertFalse(certificate([0,1,0,0], -1, [100001]*4))
        with self.assertRaises(ValueError):
            source_chord_bound_exp([1], [ZERO], [-1], 1)
        with self.assertRaises(ValueError):
            source_chord_bound_exp([1], [ZERO], [1], 2**52)

    def test_certificate_is_sufficient_against_exact_bound(self):
        # Fixed synthetic dyadic grid; no measured point inputs or reducer calls.
        for p, emin in [(53,-1022),(113,-16382)]:
            for qexp in [-400,-40,-18,0,40,400]:
                q = [F(0),pow2(qexp),pow2(qexp-3),F(0)]
                raw = -sum(x*x for x in q)
                for err in [qexp-60,qexp-30,qexp-3,qexp+3]:
                    if certificate(q,raw,[err]*4,p,emin):
                        self.assertLess(exact_error_bound(q,[err]*4,p,emin),abs(raw))

    def test_direct_invariant_avoids_mass_roundtrip_loss(self):
        # Exact binary64 toy: the tiny term vanishes on subtraction from mass.
        t, m = -2.0**-45, 2.0**20
        legacy = ((t-m)-m)+m+m
        self.assertEqual(legacy, 0.0)
        self.assertNotEqual(t, legacy)


if __name__ == '__main__':
    unittest.main()
