"""Physical and numerical conditioning of the independent CMS fixtures."""
import decimal
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

TOOLS = Path(__file__).resolve().parents[3] / 'tools'
sys.path.insert(0, str(TOOLS))
from tirust_complex_stress import two_body, gram_diagnostic, diagnostic_rows, parse_forced_qp
import tirust_complex_stress as stress
from tirust_benchmark_points import validate_points


class ComplexStressFixtures(unittest.TestCase):
    def test_fixed_qp_cannot_start_incompatible_persistent_worker(self):
        with self.assertRaisesRegex(ValueError, 'forced stability'):
            stress.observe('baseline', 8, 'qp', {}, 0)

    def test_massive_and_massless_two_body_conservation(self):
        for masses in ([0., 0., 0., 0.], [0.01, 0.02, 1., 2.]):
            for energy in (1000., 91., 160., sum(masses[2:])+1e-6):
                for angle in (1., 1e-3, 1e-6, 1e-8):
                    with self.subTest(masses=masses, energy=energy, angle=angle):
                        point = two_body('p', 'fixture', masses, energy, angle)
                        self.assertTrue(validate_points([point], masses[2:], incoming_masses=masses[:2]))

    def test_gram_ladder_is_nonzero_and_resolves_dp_cancellation(self):
        previous = decimal.Decimal(1)
        for angle in (1e-3, 1e-6, 1e-8):
            point = two_body('p', 'small-gram', [0.]*4, 1000., angle)
            normalized = decimal.Decimal(gram_diagnostic(point)['normalized_determinant'])
            expected = decimal.Decimal(str(angle))**2/2
            self.assertGreater(normalized, 0)
            self.assertLess(normalized, previous)
            self.assertLess(abs(normalized/expected-1), decimal.Decimal('1e-6'))
            previous = normalized

    def test_rejects_below_threshold_inputs(self):
        for masses in ([0., 0., 2., 3.], [2., 3., 0., 0.]):
            with self.assertRaisesRegex(ValueError, 'threshold'):
                two_body('p', 'threshold', masses, 5., 1.)

    def test_failed_nonfinite_stress_rows_remain_unqualified(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory/'raw').mkdir()
            (directory/'raw/values.dat').write_text(
                '1 0 216 1.0 0.2 -0.1 -0.2 1e-12\n'
                '2 0 420 1.0 NaN NaN Infinity 1.0\n')
            result = diagnostic_rows(directory, 6, [12,21])
            self.assertEqual([r['point'] for r in result['accepted_rows']], [12])
            self.assertEqual([r['point'] for r in result['rejected_rows']], [21])
            self.assertEqual(result['rejected_rows'][0]['raw_values'][1], 'NaN')

    def test_forced_qp_is_explicitly_unchecked(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory/'raw').mkdir()
            (directory/'raw/values.dat').write_text(
                '1 0 120 1.0 0.2 -0.1 -0.2 -1\n'
                '1 1 120 1.0 0.2 -0.1 -0.2 -1\n')
            (directory/'raw/dimensions.dat').write_text('1 1 1\n')
            (directory/'raw/timed_status.dat').write_text('120 1\n')
            result = parse_forced_qp(directory, 8, 1)
            self.assertIs(result['stability_checked'], False)
            self.assertEqual(result['backend_precision_counts'], {'8:QP_unchecked':1})
            ordinary = diagnostic_rows(directory, 8)
            self.assertFalse(ordinary['accepted_rows'])
            self.assertEqual(len(ordinary['rejected_rows']), 2)

    def test_inherited_alternate_card_is_not_labeled_primary_policy(self):
        with tempfile.TemporaryDirectory() as temporary:
            report = Path(temporary)
            point = report / 'points' / (stress.CASE['id'] + '.dat')
            point.parent.mkdir()
            point.write_text('1\n500 0 0 500\n')
            checksum = stress.campaign.digest(point)
            point.with_suffix('.json').write_text(json.dumps(
                {'sha256': checksum, 'points': [{'id':'p'}]}))
            card = report / 'inputs/baseline' / stress.CASE['id'] / 'param_card.dat'
            card.parent.mkdir(parents=True)
            card.write_text('physical card fixture\n')
            card_hash = stress.campaign.digest(card)
            run = report / 'run'
            run.mkdir()
            inputs = {'points_sha256':checksum, 'param_card_sha256':card_hash,
                      'generation':{'case':stress.CASE, 'parameter_card_sha256':card_hash},
                      'settings':{'osthres':1e-24, 'mlstabthres':1e-3}}
            (run / 'baseline-inputs.json').write_text(json.dumps(inputs))
            record = {'source':'baseline', 'case':stress.CASE,
                      'points_sha256':checksum, 'precision_request':'qp',
                      'alternate_onshell_threshold':None}
            with mock.patch.object(stress, 'REPORT', report):
                with self.assertRaisesRegex(ValueError, 'numerical policy'):
                    stress.diagnostic_binding(run, record)
                record['alternate_onshell_threshold'] = 1e-24
                binding = stress.diagnostic_binding(run, record)
                self.assertEqual(binding['resolved_onshell_threshold'], 1e-24)
