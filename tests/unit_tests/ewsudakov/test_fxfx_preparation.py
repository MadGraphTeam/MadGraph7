"""Reuse event preparation while preserving xi-dependent recoil and decay inputs."""

import math
from types import SimpleNamespace
import unittest
from unittest import mock

from madgraph.interface.reweight_interface import ReweightInterface
from madgraph.various import fxfx_ewsudakov as fx, lhe_parser
from tests.unit_tests.ewsudakov import test_sudakov_factor_limits as limits


@unittest.skipIf(limits.np is None, 'NumPy is required for FxFx event fixtures')
class TestFxFxPreparation(unittest.TestCase):
    xis = [1., 1.5, 3.]

    def setUp(self):
        fx.fks_reset_accounting()
        self.addCleanup(fx.fks_reset_accounting)
        event_id = mock.patch.object(fx, 'CURRENT_EVENT_ID', 0)
        event_id.start()
        self.addCleanup(event_id.stop)
        # The ISR pair invariant is 8000 GeV^2: xi=1 keeps the jet,
        # while xi>=1.5 merges it. The longitudinal boost exercises CM prep.
        self.event = limits.ww_event([
            limits.np.array([50., 20., math.sqrt(336.), 42.])], beta_z=.27)
        self.event.wgt = -2.
        self.event.parse_reweight()
        self.event.reweight_data = {'1001': -2., '1101': 3.}
        self.event.reweight_order = ['1001', '1101']

    def driver(self):
        driver = limits.FactorDriver()
        driver.inc_sudakov = True
        driver._sudakov_ickkw = 3
        driver.sudakov_xi_scan = list(self.xis)
        driver._configure_sudakov_weight_ids(self.event.parse_reweight(), self.xis)
        driver.density_prod_id_to_path = driver.density_decay_id_to_path = {'process': True}
        driver.density_prod_f2pylib = driver.density_decay_f2pylib = {'module': True}
        return driver

    def module(self):
        catalogue = {}
        driver = self.driver()
        for xi in self.xis:
            driver.mW2_cluster_scale = xi
            clustered, _ = driver._cluster_fxfx_event(self.event)
            _, order = clustered.get_tag_and_order()
            tag = (tuple(order[0]), tuple(sorted(order[1])))
            catalogue[tag] = tag

        def scalar(tag, p, coupling):
            correction = len(p) * .01
            return limits.np.array([1., correction, correction * 3., correction * 2.,
                                    correction * 2., correction * 3.])

        def density(tag, p, coupling, indices, dims):
            dimension = math.prod(dims)
            return {'total_dim': dimension, 'nres': len(dims),
                    'density_delta': limits.np.full(dimension, len(p) * .01),
                    'density_delta_var': limits.np.full(dimension, len(p) * .005),
                    'born_diag': limits.np.ones(dimension)}

        return SimpleNamespace(original_pdg_list_dict=catalogue,
                               ewsudakov=mock.Mock(side_effect=scalar),
                               density_sudakov=mock.Mock(side_effect=density))

    @staticmethod
    def matrix_inputs(record):
        def matrix(event, pdgs, dims):
            momenta = limits.np.array([[p.E, p.px, p.py, p.pz] for p in event])
            record.append((list(pdgs), momenta))
            return limits.np.eye(math.prod(dims))
        return matrix

    def independent_scan(self, driver, module):
        result = {}
        for k, xi in enumerate(self.xis):
            driver.mW2_cluster_scale = xi
            driver._current_xi_idx = k
            result.update(driver._compute_density_ewsudakov_reweight(self.event, module))
        return result

    def check_scan(self):
        source = str(self.event)
        module = self.module()
        expected_inputs = []
        fx.fks_reset_accounting()
        fx.CURRENT_EVENT_ID = 1
        with mock.patch.object(fx.DensityHelper, 'compute_density_matrix',
                               side_effect=self.matrix_inputs(expected_inputs)):
            expected = self.independent_scan(self.driver(), module)
        expected_methods = dict(fx.FKS_METHOD_COUNTS)
        expected_outcomes = dict(fx.FKS_QUALITY_COUNTS)
        fx.fks_reset_accounting()
        driver = self.driver()
        actual_inputs = []
        module.ewsudakov.reset_mock()
        module.density_sudakov.reset_mock()
        with mock.patch.object(driver, '_apply_forced_clustering',
                               wraps=driver._apply_forced_clustering) as forced, \
                mock.patch.object(fx, 'FXFX_CLUSTER_STEP_RE', wraps=fx.FXFX_CLUSTER_STEP_RE) as pattern, \
                mock.patch.object(fx.DensityHelper, 'compute_density_matrix',
                                  side_effect=self.matrix_inputs(actual_inputs)), \
                mock.patch.object(lhe_parser.Event, '__str__',
                                  side_effect=AssertionError('Clustering must not serialize events')):
            actual = ReweightInterface.calculate_weight(driver, self.event, module)
        self.assertEqual(actual, expected)
        self.assertEqual(fx.FKS_METHOD_COUNTS, expected_methods)
        self.assertEqual(fx.FKS_QUALITY_COUNTS, expected_outcomes)
        self.assertEqual(forced.call_count, 1)
        self.assertEqual(pattern.findall.call_count, 1 if self.event.tag else 0)
        self.assertEqual(module.ewsudakov.call_count, len(self.xis))
        self.assertEqual(module.density_sudakov.call_count, len(self.xis))
        self.assertEqual(len(actual_inputs), len(expected_inputs))
        for (actual_pdgs, actual_p), (expected_pdgs, expected_p) in zip(actual_inputs, expected_inputs):
            self.assertEqual(actual_pdgs, expected_pdgs)
            limits.np.testing.assert_array_equal(actual_p, expected_p)
        self.assertIsNone(driver._fxfx_event_cache)
        self.assertEqual(str(self.event), source)
        return driver, module

    def test_cached_scan_matches_independent_weights_momenta_and_accounting(self):
        driver, module = self.check_scan()
        self.assertEqual([len(call.args[1]) for call in module.density_sudakov.call_args_list], [5, 4, 4])
        self.assertEqual(len(driver.fxfx_clustered_final_states), 2)

    def test_no_fxfx_history_keeps_each_xi_copy_independent(self):
        self.event.tag = ''
        self.check_scan()

    def test_scalar_fallback_shares_forced_preparation_and_history(self):
        driver = self.driver()
        module = self.module()
        with mock.patch.object(fx.ResonanceIdentifier, 'find_resonances', return_value=[]), \
                mock.patch.object(driver, '_apply_forced_clustering',
                                  wraps=driver._apply_forced_clustering) as forced, \
                mock.patch.object(fx, 'FXFX_CLUSTER_STEP_RE', wraps=fx.FXFX_CLUSTER_STEP_RE) as pattern:
            result = ReweightInterface.calculate_weight(driver, self.event, module)
        self.assertEqual(forced.call_count, 1)
        self.assertEqual(pattern.findall.call_count, 1)
        self.assertEqual(module.ewsudakov.call_count, len(self.xis))
        module.density_sudakov.assert_not_called()
        for k, nlegs in enumerate((5, 4, 4)):
            for source, weight in self.event.parse_reweight().items():
                for v, factor in enumerate((1. + nlegs * .03, 1. + nlegs * .02, 1. + nlegs * .01)):
                    self.assertAlmostEqual(result[driver._sudakov_weight_map[k, v, source]], weight * factor)

    def test_a_later_scan_reprepares_even_the_same_event_object(self):
        driver = self.driver()
        module = self.module()
        with mock.patch.object(fx.DensityHelper, 'compute_density_matrix',
                               side_effect=self.matrix_inputs([])), \
                mock.patch.object(driver, '_apply_forced_clustering',
                                  wraps=driver._apply_forced_clustering) as forced:
            first = ReweightInterface.calculate_weight(driver, self.event, module)
            self.event.tag = ''
            second = ReweightInterface.calculate_weight(driver, self.event, module)
        self.assertEqual(forced.call_count, 2)
        self.assertNotEqual(first, second)
        self.assertIsNone(driver._fxfx_event_cache)

    def test_failure_clears_preparation_before_retry(self):
        driver = self.driver()
        module = self.module()
        with mock.patch.object(driver, '_apply_forced_clustering',
                               wraps=driver._apply_forced_clustering) as forced:
            with mock.patch.object(driver, '_prepare_fxfx_sudakov_inputs',
                                   side_effect=RuntimeError('invalid canonical order')):
                with self.assertRaisesRegex(RuntimeError, 'invalid canonical order'):
                    ReweightInterface.calculate_weight(driver, self.event, module)
            self.assertIsNone(driver._fxfx_event_cache)
            with mock.patch.object(fx.DensityHelper, 'compute_density_matrix',
                                   side_effect=self.matrix_inputs([])):
                ReweightInterface.calculate_weight(driver, self.event, module)
        self.assertEqual(forced.call_count, 2)
        self.assertIsNone(driver._fxfx_event_cache)

    def test_preparation_is_only_used_for_its_own_event(self):
        driver = self.driver()
        other = limits.ww_event([], beta_z=-.3)
        expected_other = driver._cluster_fxfx_event(other, return_groups=True)
        with driver._fxfx_event_preparation(self.event):
            original = driver._cluster_fxfx_event(self.event, return_groups=True)
            actual_other = driver._cluster_fxfx_event(other, return_groups=True)
            repeated = driver._cluster_fxfx_event(self.event, return_groups=True)
        for actual, expected in ((actual_other, expected_other), (repeated, original)):
            self.assertEqual(actual[1:], expected[1:])
            limits.np.testing.assert_array_equal(
                [[p.E, p.px, p.py, p.pz] for p in actual[0]],
                [[p.E, p.px, p.py, p.pz] for p in expected[0]])


if __name__ == '__main__':
    unittest.main()
