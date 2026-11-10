"""Pure statistical contracts. No native process or plot artifact is created."""
import importlib.util
import math
from pathlib import Path
import sys
import types
import unittest
from unittest import mock

spec = importlib.util.spec_from_file_location('stability_plot_metrics', Path(__file__).with_name('plot_metrics.py'))
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)


def dataset():
    return {'schema': 'madloop-stability-point-metrics-v1',
            'metric': 'raw-dp-finite-consistency', 'provenance': {'closed_index_sha256': 'a' * 64},
            'cases': [{'id': 'example', 'point_ids': [1, 2, 3],
                       'lanes': [{'backend': b, 'values': [0, 1e-5, None]} for b in M.BACKENDS]}]}


class StatisticsTests(unittest.TestCase):
    def test_three_probe_definition_is_not_regression_metric(self):
        self.assertAlmostEqual(M.relative_spread([1, 2, 3]), 1)
        self.assertAlmostEqual(M.symmetric_disagreement(1, 3), .5)

    def test_ordinary_arithmetic_matches_legacy_bitwise(self):
        for values in ([1., 1. + 2.**-50, 1. - 2.**-49],
                       [-2.3, -2.1, -2.2], [1., -.25, .5], [-1e16, 1., 1e16]):
            spread, mean = max(values) - min(values), abs(sum(values) / len(values))
            expected = spread / mean if mean else (0. if spread == 0 else math.inf)
            self.assertEqual(M.relative_spread(values).hex(), expected.hex())

    def test_zero_and_zero_mean_are_distinct(self):
        self.assertEqual(M.relative_spread([0, 0, 0]), 0)
        self.assertEqual(M.relative_spread([1, 1, 1]), 0)
        self.assertTrue(math.isinf(M.relative_spread([-1, 0, 1])))
        self.assertTrue(math.isinf(M.symmetric_disagreement(-1, 1)))

    def test_overflow_normalization_preserves_finite_ratio(self):
        self.assertAlmostEqual(M.relative_spread([1e308, -1e308, 1e308]), 6)
        self.assertAlmostEqual(M.symmetric_disagreement(1e308, 1e308), 0)

    def test_subnormal_mean_is_not_erased(self):
        self.assertEqual(M.relative_spread([5e-324, 0., 0.]), 3.)
        self.assertEqual(M.relative_spread([5e-324, 5e-324, 1e-323]), .75)

    def test_nonfinite_probe_values_rejected(self):
        for values in ([0, 1], [0, 1, math.nan], [0, 1, math.inf]):
            with self.assertRaises(ValueError): M.relative_spread(values)

    def test_threshold_equality_is_not_exceedance(self):
        row = M.tail_bounds([0, 1e-6, 1e-5], 1e-6)
        self.assertEqual(row['known_exceedances'], 1)
        self.assertEqual(row['fraction'], 1 / 3)

    def test_unavailable_values_stay_in_denominator(self):
        row = M.tail_bounds([0, 1e-5, None, None], 1e-6)
        self.assertEqual((row['lower'], row['upper'], row['fraction']), (.25, .75, None))

    def test_all_missing_population_has_full_uncertainty(self):
        row = M.tail_bounds([None] * 100, 1e-6)
        self.assertEqual((row['lower'], row['upper']), (0, 1))

    def test_zero_is_reported_without_floor(self):
        row = M.tail_bounds([0] * 100, 1e-16)
        self.assertEqual(row['exact_zero_metrics'], 100)
        self.assertEqual((row['lower'], row['upper']), (0, 0))

    def test_json_infinity_is_explicit_and_exceeds_every_finite_target(self):
        self.assertEqual(M.tail_bounds(['inf', 0], 1e5)['fraction'], .5)
        for value in (math.inf, math.nan, -1, True):
            with self.assertRaises(ValueError): M.metric_value(value)

    def test_ratio_masks_zero_ninja_and_incomplete_populations(self):
        self.assertIsNone(M.tail_ratio([1, 1], [0, 0], .1))
        self.assertIsNone(M.tail_ratio([1, None], [1, 0], .1))
        self.assertEqual(M.tail_ratio([1, 1], [1, 0], .1), 2)
        self.assertEqual(M.tail_ratio([0, 0], [1, 0], .1), 0)

    def test_duplicate_split_order_rows_are_rejected(self):
        data = dataset(); data['cases'][0]['point_ids'] = [1, 1, 2]
        with self.assertRaises(ValueError): M.validate_dataset(data)

    def test_all_five_intended_lanes_required(self):
        data = dataset(); data['cases'][0]['lanes'].pop()
        with self.assertRaises(ValueError): M.validate_dataset(data)

    def test_backend_id_must_be_an_integer_not_float_or_bool(self):
        for value in (True, 6.0):
            data = dataset(); data['cases'][0]['lanes'][1]['backend'] = value
            with self.assertRaises(ValueError): M.validate_dataset(data)

    def test_vertical_markers_depend_on_metric_and_bound_policy(self):
        data = dataset()
        self.assertEqual([x[0] for x in M.target_markers(data)], [1e-3])
        data['metric'] = 'accepted-reducer-disagreement'
        self.assertEqual([x[0] for x in M.target_markers(data)], [3e-6, 1e-8])
        data['metric'] = 'accepted-composite-internal-estimate'
        self.assertEqual([x[0] for x in M.target_markers(data)], [1e-3])
        data['provenance']['policy_binding'] = {'MLStabThres': 1e-6, 'sha256': 'a'*64}
        self.assertEqual([x[0] for x in M.target_markers(data)], [1e-3, 1e-6])

    def test_exact_empirical_jumps_and_regression_target_are_in_grid(self):
        data = dataset(); data['metric'] = 'accepted-reducer-disagreement'
        values = [0.] * 98 + [2e-6, 2.8064206471279104e-6]
        data['cases'][0]['point_ids'] = list(range(100))
        for lane in data['cases'][0]['lanes']: lane['values'] = values
        grid = M.threshold_grid(data)
        critical = max(values)
        self.assertIn(critical, grid)
        self.assertIn(3e-6, grid)
        self.assertEqual(M.tail_bounds(values, 3e-6)['fraction'], 0.)
        self.assertEqual(M.tail_bounds(values, critical)['fraction'], 0.)
        self.assertEqual(M.tail_bounds(values, math.nextafter(critical, 0.))['fraction'], .01)

    def test_tied_metrics_jump_together_and_zero_or_infinite_values_add_no_breakpoint(self):
        data = dataset()
        for lane in data['cases'][0]['lanes']: lane['values'] = [1e-7, 1e-7, 'inf']
        grid = M.threshold_grid(data)
        self.assertEqual(grid.count(1e-7), 1)
        self.assertNotIn(0., grid)
        self.assertTrue(all(math.isfinite(x) for x in grid))
        self.assertEqual(M.tail_bounds([1e-7, 1e-7, 'inf'], 1e-7)['fraction'], 1 / 3)

    def test_metric_semantics_and_provenance_are_required(self):
        for key, value in (('metric', 'true-error'), ('provenance', {})):
            data = dataset(); data[key] = value
            with self.assertRaises(ValueError): M.validate_dataset(data)

    def test_missing_rows_cannot_shrink_population(self):
        data = dataset(); data['cases'][0]['lanes'][0]['values'].pop()
        with self.assertRaises(ValueError): M.validate_dataset(data)

    def test_summary_preserves_n_resolution_and_missingness(self):
        result = M.summarize(dataset(), [1e-6])
        case = result['cases'][0]
        self.assertEqual((case['n'], case['resolution']), (3, 1 / 3))
        self.assertIsNone(case['lanes'][0]['ratio_over_Ninja'][0])
        self.assertEqual(case['lanes'][0]['tail'][0]['unavailable'], 1)


class PresentationTests(unittest.TestCase):
    """Inspect emitted plot labels without matplotlib or filesystem writes."""

    def render_labels(self, data):
        matplotlib = types.ModuleType('matplotlib')
        matplotlib.use = mock.Mock()
        pyplot = types.ModuleType('matplotlib.pyplot')
        figure = mock.MagicMock()
        top, ratio, note = mock.MagicMock(), mock.MagicMock(), mock.MagicMock()
        figure.add_subplot.side_effect = (top, ratio, note)
        pyplot.figure = mock.Mock(return_value=figure)
        pyplot.close = mock.Mock()
        matplotlib.pyplot = pyplot
        prefix = mock.Mock()
        prefix.with_suffix.return_value.exists.return_value = False
        with mock.patch.dict(sys.modules, {'matplotlib': matplotlib,
                                          'matplotlib.pyplot': pyplot}), \
                mock.patch.object(M, 'Path', return_value=prefix):
            summary = M.render(data, 'unused', input_sha256='b' * 64)
        labels = [call.kwargs['label'] for call in top.step.call_args_list
                  if 'label' in call.kwargs]
        title = figure.suptitle.call_args.args[0].replace('\n', ' ')
        self.assertIn('Host forced-DP/card-selected backend', title)
        return labels, note.text.call_args.args[2], summary

    def test_each_backend_label_reports_zero_and_unavailable_counts(self):
        data = dataset()
        data['cases'][0]['lanes'][1]['values'] = [0, 0, 1e-5]
        labels, footer, summary = self.render_labels(data)
        self.assertEqual(labels, ['CutTools (zero:1; unavailable:1/3)',
                                  'Ninja (zero:2; unavailable:0/3)',
                                  'COLLIER (zero:1; unavailable:1/3)',
                                  'NinjaRS (zero:1; unavailable:1/3)',
                                  'TIRust (zero:1; unavailable:1/3)'])
        self.assertIn('Zero and unavailable counts in the legend.', footer)
        self.assertIn('upper bounds from unavailable outcomes, not measured failures', footer)
        self.assertEqual(summary, dict(M.summarize(data, M.threshold_grid(data)),
                                       input_sha256='b' * 64))

    def test_wholly_unavailable_backend_is_explicitly_labelled(self):
        data = dataset()
        data['cases'][0]['point_ids'] = list(range(100))
        for lane in data['cases'][0]['lanes']:
            lane['values'] = [None] * 100 if lane['backend'] == 9 else [0] * 100
        labels, _, summary = self.render_labels(data)
        self.assertIn('TIRust (zero:0; unavailable:100/100)', labels)
        self.assertIn('CutTools (zero:100; unavailable:0/100)', labels)
        row = summary['cases'][0]['lanes'][-1]['tail'][0]
        self.assertEqual((row['lower'], row['upper'], row['fraction']), (0, 1, None))
        for lane in summary['cases'][0]['lanes']:
            self.assertTrue(all(value is None for value in lane['ratio_over_Ninja']))


if __name__ == '__main__': unittest.main()
