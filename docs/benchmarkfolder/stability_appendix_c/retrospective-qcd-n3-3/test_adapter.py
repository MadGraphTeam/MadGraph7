import copy
import importlib.util
import math
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('retrospective_adapter', Path(__file__).with_name('adapter.py'))
A = importlib.util.module_from_spec(spec); spec.loader.exec_module(A)


def lane():
    rows = []
    for point in range(1, 101):
        row = {'point': point, 'order': 0, 'actual_backend': 1, 'precision': 'DP',
               'status': 211, 'coefficients': [1., 2., 3., 4.], 'accuracy': 1e-8}
        rows.extend([row, row | {'order': 1}])
    return {'backend': 1, 'observation': {'rows': rows}}


class AdapterTests(unittest.TestCase):
    def test_duplicates_collapse_to_physical_points(self):
        rows = A.collapse(lane())
        self.assertEqual(len(rows), 100)
        self.assertEqual([r['point'] for r in rows], list(range(1, 101)))

    def test_duplicate_value_difference_rejected(self):
        sample = lane(); sample['observation']['rows'][1]['coefficients'] = [1., 2.1, 3., 4.]
        with self.assertRaises(ValueError): A.collapse(sample)

    def test_signed_zero_difference_not_silently_collapsed(self):
        self.assertFalse(A.same_value(0., -0.))

    def test_missing_order_cannot_reduce_denominator(self):
        sample = lane(); sample['observation']['rows'].pop()
        with self.assertRaises(ValueError): A.collapse(sample)

    def test_status_precision_alias_mismatch_rejected(self):
        sample = lane(); sample['observation']['rows'][0]['precision'] = 'QP'
        sample['observation']['rows'][1]['precision'] = 'QP'
        with self.assertRaises(ValueError): A.collapse(sample)

    def test_nonfinite_coefficients_rejected(self):
        sample = lane()
        for row in sample['observation']['rows'][:2]: row['coefficients'] = [1., math.nan, 3., 4.]
        with self.assertRaises(ValueError): A.collapse(sample)

    def test_strict_json_rejects_duplicate_and_nonfinite_values(self):
        for source in ('{"a":1,"a":2}', '{"a":NaN}'):
            with self.assertRaises(ValueError): A.strict_json(source)


if __name__ == '__main__': unittest.main()
