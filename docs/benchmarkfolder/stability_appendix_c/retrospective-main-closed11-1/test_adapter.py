import importlib.util
import math
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('closed_main_adapter', Path(__file__).with_name('adapter.py'))
A = importlib.util.module_from_spec(spec)
spec.loader.exec_module(A)


def lane(backend=1, status=211, precision='DP'):
    rows = []
    for point in range(1, 101):
        row = {'point': point, 'order': 0, 'actual_backend': backend, 'precision': precision,
               'status': status, 'coefficients': [1., 2., 3., 4.], 'accuracy': 1e-8}
        rows.extend([row, row | {'order': 1}])
    return {'backend': backend, 'status': 'complete', 'performance_eligible': False,
            'observation': {'rows': rows}}


class SyntheticTests(unittest.TestCase):
    def test_duplicates_collapse_to_one_hundred_physical_points(self):
        self.assertEqual([r['point'] for r in A.collapse(lane())], list(range(1, 101)))

    def test_aliases_are_precision_dependent(self):
        for status, (backend, precision) in A.STATUS.items():
            self.assertEqual(len(A.collapse(lane(backend, status, precision))), 100)
        self.assertEqual(A.STATUS[219], (9, 'DP'))
        self.assertEqual(A.STATUS[329], (1, 'QP'))
        self.assertEqual(A.STATUS[322], (8, 'QP'))
        self.assertEqual(A.STATUS[323], (9, 'QP'))

    def test_wrong_alias_rejected(self):
        for item in (lane(9,329,'QP'), lane(8,323,'QP'), lane(7,327,'QP')):
            with self.assertRaises(ValueError):
                A.collapse(item)

    def test_unchecked_and_initialization_statuses_rejected(self):
        for status in (110,120,239,349):
            with self.assertRaises(ValueError):
                A.collapse(lane(9,status,'DP'))

    def test_duplicate_value_difference_rejected(self):
        item = lane()
        item['observation']['rows'][1]['coefficients'] = [1.,2.1,3.,4.]
        with self.assertRaises(ValueError):
            A.collapse(item)

    def test_duplicate_signed_zero_not_silently_collapsed(self):
        self.assertFalse(A.same_value(0., -0.))

    def test_missing_order_cannot_reduce_denominator(self):
        item = lane()
        item['observation']['rows'].pop()
        with self.assertRaises(ValueError):
            A.collapse(item)

    def test_incomplete_lane_rejected(self):
        item = lane()
        item['status'] = 'failed'
        with self.assertRaises(ValueError):
            A.collapse(item)

    def test_nonfinite_coefficient_or_estimate_rejected(self):
        for kind in ('coefficient','estimate'):
            item = lane()
            for row in item['observation']['rows'][:2]:
                if kind == 'coefficient':
                    row['coefficients'] = [1.,math.nan,3.,4.]
                else:
                    row['accuracy'] = math.inf
            with self.assertRaises(ValueError):
                A.collapse(item)

    def test_strict_json_rejects_duplicate_nonfinite(self):
        for raw in ('{"a":1,"a":2}', '{"a":NaN}', '{"a":Infinity}'):
            with self.assertRaises(ValueError):
                A.strict_json(raw)


class ActualClosedInputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.store, cls.verification, cls.plotting = A.verified_inputs()
        cls.datasets, cls.outcomes = A.normalized(cls.store, cls.plotting)

    def test_all_four_common_one_hundred_point_cases(self):
        self.assertTrue(self.verification['full_matrix_regression_passed'])
        self.assertEqual(len(self.datasets), 8)
        for data in self.datasets.values():
            case = data['cases'][0]
            self.assertEqual(case['point_ids'], list(range(1,101)))
            self.assertEqual([lane['backend'] for lane in case['lanes']], list(A.BACKENDS))
            self.assertTrue(all(len(lane['values']) == 100 for lane in case['lanes']))

    def test_tirust_precision_counts(self):
        expected = {'qcd_n2_gg_ttx': {'DP':100}, 'qcd_n3_uux_zgg': {'DP':87,'QP':13},
                    'ew_n2_epem_mupmum': {'DP':100}, 'qcd_n4_uux_zzgg': {'DP':14,'QP':86}}
        for case, counts in expected.items():
            self.assertEqual(self.outcomes[case]['accepted_outcomes']['9']['accepted_precision_counts'], counts)

    def test_current_reference_self_comparison_is_zero(self):
        for key, data in self.datasets.items():
            if key.endswith('finite-disagreement-current-cuttools'):
                self.assertEqual(data['cases'][0]['lanes'][0]['values'], [0.]*100)

    def test_all_current_finite_comparisons_pass_regression(self):
        for key, data in self.datasets.items():
            if key.endswith('finite-disagreement-current-cuttools'):
                for lane in data['cases'][0]['lanes']:
                    self.assertEqual(self.plotting.tail_bounds(lane['values'],3e-6)['fraction'],0.)

    def test_target_misses_are_preserved(self):
        zgg = self.datasets['qcd_n3_uux_zgg-finite-disagreement-current-cuttools']['cases'][0]
        for lane in zgg['lanes'][1:]:
            self.assertEqual(self.plotting.tail_bounds(lane['values'],1e-8)['known_exceedances'],28)
        for case in ('qcd_n3_uux_zgg','qcd_n4_uux_zzgg'):
            self.assertTrue(any(not outcome['comparison_to_starting_baseline']['ordinary_1e8_target_passed']
                                for outcome in self.outcomes[case]['accepted_outcomes'].values()))

    def test_policy_record_is_not_presented_as_card_bytes(self):
        for data in self.datasets.values():
            self.assertFalse(data['provenance']['actual_policy_binding_available'])
            self.assertNotIn('policy_binding',data['provenance'])
            for record in data['provenance']['recorded_policy_and_native_identity_by_backend'].values():
                self.assertFalse(record['effective_numerical_card_bytes_retained'])
                self.assertEqual(record['recorded_settings']['mlstabthres'],1e-6)
                self.assertEqual(record['W03_count_case_insensitive'],0)

    def test_exact_empirical_steps_and_zero_ninja_ratio_mask(self):
        for data in self.datasets.values():
            grid = self.plotting.threshold_grid(data)
            for lane in data['cases'][0]['lanes']:
                for value in lane['values']:
                    if isinstance(value,(float,int)) and 1e-16 <= value <= 1.:
                        self.assertIn(value,grid)
            self.assertIsNone(self.plotting.tail_ratio([0.]*100,[0.]*100,1e-8))


if __name__ == '__main__':
    unittest.main()
