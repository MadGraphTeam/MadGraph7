"""Pure parser/source regressions. No compiler, reducer or native child."""
import copy
import json
from pathlib import Path
import unittest

import analyze
import prepare

HERE = Path(__file__).resolve().parent


class Protocol(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = json.loads((HERE / 'cases.json').read_bytes())
        cls.expected = json.loads((HERE / 'expected.json').read_bytes())
        cls.golden = analyze.golden(cls.cases, cls.expected)

    def changed(self, before, after):
        self.assertIn(before, self.golden, 'negative mutation does not exist')
        return self.golden.replace(before, after, 1)

    def rejects(self, text):
        with self.assertRaises(ValueError):
            analyze.parse(text, self.cases, self.expected)

    def test_frozen_240_case_corpus(self):
        self.assertEqual(prepare.corpus(), self.cases)
        self.assertEqual(prepare.expected(self.cases), self.expected)
        self.assertEqual(prepare.render(self.cases), (HERE / 'inputs.txt').read_text())
        self.assertEqual(len(self.cases), 240)
        self.assertEqual(sum(all(r['have']) for r in self.expected), 180)
        for mode in (1, 2, 4, 5):
            masks = {c['sticky_mask'] for c in self.cases if c['mode'] == mode and
                     c['label'].startswith(('sticky-mask-', 'positive-slot-'))}
            self.assertEqual(masks, set(range(32)))

    def test_pure_golden_pass_is_not_native_evidence(self):
        result = analyze.parse(self.golden, self.cases, self.expected)
        self.assertEqual(result['cases'], 240)
        self.assertFalse(result['full_sloop_lifetime_coverage'])
        self.assertFalse(result['physical_amplitude_qualified'])
        self.assertEqual(result['tensor_or_scientific_evaluations'], 0)

    def test_missing_extra_duplicate_rows_rejected(self):
        self.rejects(self.golden.replace('ENDCASE 1\n', '', 1))
        self.rejects(self.golden + 'DONE 240\n')
        self.rejects(self.golden.replace('CASE 1 1 1 1\n', 'CASE 1 1 1 1\nCASE 1 1 1 1\n', 1))

    def test_lost_or_changed_case_rejected(self):
        self.rejects(self.changed('CASE 1 1 1 1', 'CASE 2 1 1 1'))
        self.rejects(self.changed('READY 240', 'READY 239'))
        self.rejects(self.changed('DONE 240', 'DONE 239'))

    def test_flags_halting_rounding_mutation_rejected(self):
        self.rejects(self.changed('POST 0 0 0 0 0 0 0 0 0 0 0',
                                  'POST 1 0 0 0 0 0 0 0 0 0 0'))
        self.rejects(self.changed('POST 0 0 0 0 0 0 0 0 0 0 0',
                                  'POST 0 0 0 0 0 1 0 0 0 0 0'))
        self.rejects(self.changed('POST 0 0 0 0 0 0 0 0 0 0 0',
                                  'POST 0 0 0 0 0 0 0 0 0 0 1'))

    def test_token_and_environment_transport_rejected(self):
        token = next(line for line in self.golden.splitlines() if line.startswith('TOKEN '))
        words = token.split(); words[2] = f'{int(words[2],16)^1:016X}'
        self.rejects(self.changed(token, ' '.join(words)))
        self.rejects(self.changed('RECORDED_ENV 4B4500001F80033F', 'RECORDED_ENV 0000000000000000'))

    def test_slots_bounds_class_rejected(self):
        self.rejects(self.changed('SLOTS 1 1 2 3', 'SLOTS 1 3 2 1'))
        bound = next(line for line in self.golden.splitlines() if line.startswith('BOUNDS '))
        words = bound.split(); words[-1] = str(int(words[-1]) + 1)
        self.rejects(self.changed(bound, ' '.join(words)))
        self.rejects(self.changed('CLASS 1 1 1', 'CLASS 1 1 0'))

    def test_legacy_forward_and_mass_transport_rejected(self):
        self.rejects(self.changed('FORWARD 1 1 1 2 3 2 0 1', 'FORWARD 1 0 1 2 3 2 0 1'))
        self.rejects(self.changed('MASS 1 3FF0000000000000 8000000000000000',
                                  'MASS 1 3FF0000000000000 0000000000000000'))

    def test_dp_qp_memory_word_mutations_rejected(self):
        for tag in ('DPMOM ', 'QPMOM '):
            old = next(line for line in self.golden.splitlines() if line.startswith(tag))
            words = old.split(); words[-1] = f'{int(words[-1],16)^1:0{len(words[-1])}X}'
            self.rejects(self.changed(old, ' '.join(words)))

    def test_missing_precision_rejections_are_retained(self):
        for mode in (1, 2, 4, 5):
            subset = [e for e in self.expected if e['mode'] == mode]
            self.assertTrue(any(e['label'] == 'direct-fresh-false' and not any(e['have']) for e in subset))
            self.assertTrue(any(e['label'] == 'external-snan-bits' and not any(e['have']) for e in subset))
            self.assertTrue(any(e['label'] == 'last-significand-bit' and all(e['have']) for e in subset))


class Source(unittest.TestCase):
    def test_exact_generated_wrapper_and_aloha_spans(self):
        full = (HERE / 'full-CT_interface.f').read_text()
        got, _ = prepare.span(full, r'^      SUBROUTINE ML5_0_LOOP_3_PROVENANCE\(', r'^      END\s*$')
        self.assertEqual(got, (HERE / 'wrapper.f').read_text())
        module, _ = prepare.span((HERE / 'full-aloha_functions.f').read_text(),
                                  r'^      module ALOHA_OBJECT\s*$', r'^      end module ALOHA_OBJECT\s*$')
        self.assertEqual(module, (HERE / 'aloha_object.f').read_text())
        self.assertIn('PARAMETER (NSQUAREDSO=0)', got)

    def test_fixture_scope_and_capture_order(self):
        driver = (HERE / 'driver.f90').read_text()
        recorder = (HERE / 'recording.f90').read_text().lower()
        self.assertLess(driver.index('call capture(post_flags'), driver.index('! Word observations'))
        self.assertLess(driver.index('token_after=read_kinematic_environment_token()'), driver.index('! Word observations'))
        self.assertLess(driver.index("'READY',ncases"), driver.index('open(newunit=fifo_unit'))
        self.assertIn('call restore_environment()', driver)
        self.assertNotIn('write(', recorder)
        self.assertNotIn('use mtirust', recorder)
        self.assertNotIn('call ml_tirust', recorder)
        self.assertIn('call ml5_0_loop_3(w1,w2,w3,m1,m2,m3,rank,squaredsoindex,loopnum)', recorder)


if __name__ == '__main__':
    unittest.main(verbosity=2)
