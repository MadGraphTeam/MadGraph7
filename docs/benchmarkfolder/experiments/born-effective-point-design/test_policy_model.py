"""Test proposal lifecycle invariants only; no physical or Fortran execution."""
import unittest
from policy_model import RescueCall


class Lifecycle(unittest.TestCase):
    def prepared(self):
        call = RescueCall()
        call.original_born('B(P)', 'REF(P)')
        call.prepare('C128', 'round64(C128)', repair_ok=True)
        return call

    def test_ordinary_success_does_no_rescue_work(self):
        call = RescueCall()
        call.original_born('original_bits', 'original_ref')
        self.assertEqual((call.published_born, call.reference), ('original_bits', 'original_ref'))
        self.assertEqual((call.prepare_count, call.born_count, call.cache_invalidations), (0, 0, 0))
        self.assertFalse(call.numerator_called)

    def test_finite_born_is_replaced_at_qp_transition(self):
        call = self.prepared()
        call.stage_policy(True, 'B_MP(C)', 'REF_MP(C)')
        self.assertEqual((call.published_born, call.interference_view), ('B_MP(C)', 'C128'))
        self.assertFalse(call.numerator_done)
        self.assertFalse(call.numerator_called)

    def test_nonfinite_trigger_does_not_certify_a_loop(self):
        call = self.prepared()
        call.nonfinite_born_trigger = True
        call.stage_policy(True, 'B_MP(C)', 'REF_MP(C)')
        with self.assertRaises(ValueError):
            call.accept(independently_checked=True)
        call.finish_numerator()
        self.assertFalse(call.accept(independently_checked=False))

    def test_false_option_keeps_actual_dp_view(self):
        call = self.prepared()
        call.stage_policy(False, 'B_DP(round(C))', 'REF_DP(round(C))')
        self.assertEqual(call.interference_view, 'round64(C128)')
        self.assertEqual(call.policy, 'promoted_dp')
        self.assertNotEqual(call.interference_view, call.canonical)

    def test_policy_switch_invalidates_born_coefficients_and_cache(self):
        call = self.prepared()
        call.stage_policy(False, 'DPborn', 'DPref')
        call.finish_numerator()
        call.stage_policy(True, 'MPborn', 'MPref')
        self.assertFalse(call.numerator_done)
        self.assertEqual((call.born_count, call.cache_invalidations), (2, 2))
        self.assertEqual(call.prepare_count, 1)
        self.assertEqual(call.reference, 'MPref')

    def test_same_policy_reuses_only_born_and_coordinates(self):
        call = self.prepared()
        call.stage_policy(True, 'born', 'ref')
        self.assertEqual(call.stage_policy(True, 'ignored', 'ignored'), 'reuse')
        self.assertEqual(call.born_count, 1)
        self.assertFalse(call.numerator_done)

    def test_rotation_uses_canonical_point(self):
        call = self.prepared()
        call.stage_policy(True, 'born', 'ref')
        call.finish_numerator()
        call.rotate(1)
        call.rotate(2)
        self.assertEqual(call.events[-2:], [('rotate_from_canonical', 'C128', 1), ('rotate_from_canonical', 'C128', 2)])
        self.assertFalse(call.numerator_done)
        self.assertEqual(call.prepare_count, 1)

    def test_finite_failed_repair_is_rejected(self):
        call = RescueCall()
        self.assertEqual(call.prepare('finite_input_retained', 'finite_dp', repair_ok=False), 'failed')
        self.assertFalse(call.ready)
        self.assertIsNone(call.published_born)

    def test_conversion_and_born_failure_never_publish_old_born(self):
        a = RescueCall()
        a.original_born('old', 'oldref')
        a.prepare('finite_C', 'overflow', repair_ok=True, conversion_ok=False)
        self.assertTrue(a.failed)
        self.assertIsNone(a.reference)
        b = self.prepared()
        b.stage_policy(True, 'invalid', 'invalid', born_ok=False)
        self.assertFalse(b.ready)
        self.assertIsNone(b.published_born)

    def test_loop_induced_and_unsupported_keep_legacy_path(self):
        for call in (RescueCall(loop_induced=True), RescueCall(supported=False)):
            self.assertEqual(call.prepare('C', 'C8', repair_ok=True), 'legacy')
            self.assertEqual(call.stage_policy(True, 'born', 'ref'), 'legacy')
            self.assertEqual(call.prepare_count, 0)

    def test_new_call_never_inherits_certificate(self):
        old = self.prepared()
        old.stage_policy(True, 'born', 'ref')
        old.finish_numerator()
        old.accept(independently_checked=True)
        new = RescueCall()
        self.assertFalse(new.ready or new.numerator_done or new.checked)


if __name__ == '__main__':
    unittest.main()
