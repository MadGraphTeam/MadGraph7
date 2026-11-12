"""Both Sudakov modes use the same CM frame and bijective particle ordering."""

import itertools
import math
from types import SimpleNamespace
import unittest
from unittest import mock

from madgraph.interface.reweight_interface import ReweightInterface
from madgraph.various import fxfx_ewsudakov as fx
from tests.unit_tests.ewsudakov import test_sudakov_factor_limits as limits
from tests.unit_tests.ewsudakov.test_fxfx_decay_transport import boost, make_event


@unittest.skipIf(limits.np is None, 'NumPy is required for Sudakov momentum checks')
class TestSudakovPreparation(unittest.TestCase):
    def setUp(self):
        fx.fks_reset_accounting()
        self.addCleanup(fx.fks_reset_accounting)

    def event(self, pdgs, transform=False):
        np = limits.np
        if len(pdgs) == 3:
            final = [[200., 200., 0., 0.],
                     [200., -100., math.sqrt(30000.), 0.],
                     [200., -100., -math.sqrt(30000.), 0.]]
        else:
            final = [[150., 150., 0., 0.], [150., -150., 0., 0.],
                     [150., 0., 150., 0.], [150., 0., -150., 0.]]
        cm = np.array([[300., 0., 0., 300.], [300., 0., 0., -300.]] + final)
        momenta = cm.copy()
        if transform:
            rotation = np.array([[1., 0., 0., 0.], [0., .8, 0., .6],
                                 [0., 0., 1., 0.], [0., -.6, 0., .8]])
            lab = boost([-.12, .07, -.3]) @ rotation
            momenta = (lab @ cm.T).T
        event = make_event([(pid, status, p) for pid, status, p in zip(
            [2, -2] + list(pdgs), [-1, -1] + [1] * len(pdgs), momenta)])
        event.tag = '<clustering type="born"></clustering>'
        event.wgt = -2.
        event.parse_reweight()
        event.reweight_data = {'1001': -2., '1101': 3.}
        event.reweight_order = ['1001', '1101']
        return event, cm

    def evaluate(self, event, canon, mode):
        driver = limits.FactorDriver()
        driver.banner = SimpleNamespace(get_detail=lambda *args:
            'u u~ > ' + ' '.join('a' for p in event if p.status == 1))
        names = driver.SUDAKOV_LO_VARIANT_NAMES if mode == 0 else driver.SUDAKOV_VARIANT_NAMES
        driver._configure_sudakov_weight_ids(event.parse_reweight(), [1.5], names)
        tag = ((2, -2), tuple(sorted(p.pid for p in event if p.status == 1)))
        module = SimpleNamespace(original_pdg_list_dict={tag: ((2, -2), canon)},
                                 ewsudakov=mock.Mock(return_value=[10., 1., 2., 3., 4., 5.]))
        if mode == 0:
            result = ReweightInterface._compute_ewsudakov_reweight(driver, event, module)
        else:
            result = driver._compute_ewsudakov_fxfx_reweight(event, module)
        factors = [1.2, 1.3, 1.1, 1.4, 1.5] if mode == 0 else [1.2, 1.3, 1.1]
        limits.TestSudakovFactorLimits.assert_factors(self, driver, result, factors, event)
        module.ewsudakov.assert_called_once()
        self.assertAlmostEqual(module.ewsudakov.call_args.args[2], math.sqrt(4 * math.pi * event.aqcd))
        return module.ewsudakov.call_args.args[1]

    def check_permutations(self, pdgs, transform=False):
        event, cm = self.event(pdgs, transform)
        original = str(event)
        for canon in sorted(set(itertools.permutations(pdgs))):
            mode_inputs = []
            for mode in (0, 3):
                with self.subTest(canon=canon, mode=mode, transform=transform):
                    inputs = self.evaluate(event, list(canon), mode)
                    mode_inputs.append(inputs)
                    limits.np.testing.assert_allclose(inputs[:2], cm[:2], atol=1e-10, rtol=1e-12)
                    # Every particle of each identity must appear exactly once;
                    # identical particles may be interchanged, never duplicated.
                    for pid in set(pdgs):
                        actual = sorted(tuple(p) for p, label in zip(inputs[2:], canon) if label == pid)
                        expected = sorted(tuple(p) for p, label in zip(cm[2:], pdgs) if label == pid)
                        limits.np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-12)
            limits.np.testing.assert_array_equal(mode_inputs[0], mode_inputs[1])
        self.assertEqual(str(event), original)

    def test_three_distinct_particles_including_non_self_inverse_permutations(self):
        self.check_permutations([11, -11, 22])

    def test_separated_identical_particles_are_neither_lost_nor_duplicated(self):
        self.check_permutations([11, 22, -11, 22])

    def test_boost_and_rotation_agree_between_modes(self):
        self.check_permutations([11, -11, 22], transform=True)

    def test_shared_helper_records_the_frame_operations_for_decays(self):
        event, cm = self.event([11, -11, 22], transform=True)
        module = SimpleNamespace(original_pdg_list_dict={
            ((2, -2), (-11, 11, 22)): ((2, -2), [-11, 22, 11])})
        prep = limits.FactorDriver()._prepare_sudakov_inputs(event, module, event.copy())
        self.assertEqual([kind for kind, momentum in prep['frame_transform']],
                         ['boost_to_cm', 'rotate_to_z'])
        for original, expected in zip(event, cm):
            actual = fx._apply_fxfx_frame_transform(fx.FourMomentum(original), prep['frame_transform'])
            limits.np.testing.assert_allclose([actual.E, actual.px, actual.py, actual.pz],
                                             expected, atol=1e-10, rtol=1e-12)
        self.assertEqual(prep['perm'], [2, 0, 1])
        self.assertEqual(prep['pdg_order'], [2, -2, -11, 22, 11])

    def test_incompatible_canonical_particle_counts_raise_before_the_kernel(self):
        event, _ = self.event([11, -11, 22])
        for canon in ([-11, -11, 22], [11, -11], [11, -11, 22, 22]):
            for mode in (0, 3):
                with self.subTest(canon=canon, mode=mode):
                    with self.assertRaisesRegex(RuntimeError, 'expected .*got'):
                        self.evaluate(event, canon, mode)

    def test_shared_preparation_preserves_the_mode_specific_small_invariant_cut(self):
        np = limits.np
        pz = 299.9
        px = math.sqrt(300.**2 - pz**2)
        event = make_event([(2, -1, np.array([300., 0., 0., 300.])),
                            (-2, -1, np.array([300., 0., 0., -300.])),
                            (11, 1, np.array([300., px, 0., pz])),
                            (-11, 1, np.array([300., -px, 0., -pz]))])
        event.tag = '<clustering type="born"></clustering>'
        event.wgt = 2.
        event.parse_reweight()
        event.reweight_data, event.reweight_order = {'1001': 2.}, ['1001']
        self.evaluate(event, [11, -11], 0)
        driver = limits.FactorDriver()
        module = SimpleNamespace(ewsudakov=mock.Mock(), original_pdg_list_dict={
            ((2, -2), (-11, 11)): ((2, -2), [11, -11])})
        result = driver._compute_ewsudakov_fxfx_reweight(event, module)
        module.ewsudakov.assert_not_called()
        limits.TestSudakovFactorLimits.assert_factors(self, driver, result, [1., 1., 1.], event)

    def test_h_event_inputs_use_the_actual_size_after_isr_clustering(self):
        np = limits.np
        event = make_event([
            (21, -1, np.array([300., 0., 0., 300.])),
            (21, -1, np.array([300., 0., 0., -300.])),
            (21, 1, np.array([295., math.sqrt(87000.), 0., -5.])),
            (21, 1, np.array([295., -math.sqrt(87000.), 0., -5.])),
            (21, 1, np.array([10., 0., 0., 10.])),
        ])
        event.wgt = 2.
        event.parse_reweight()
        event.reweight_data, event.reweight_order = {'1001': 2.}, ['1001']
        original = str(event)
        for merge in (False, True):
            with self.subTest(merge=merge):
                driver = limits.FactorDriver()
                driver.banner = SimpleNamespace(get_detail=lambda *args: 'g g > g g')
                driver._configure_sudakov_weight_ids(event.parse_reweight(), [1.5],
                                                      driver.SUDAKOV_LO_VARIANT_NAMES)
                final = (21,) * (2 if merge else 3)
                module = SimpleNamespace(original_pdg_list_dict={((21, 21), final): ((21, 21), final)},
                                         ewsudakov=mock.Mock(return_value=[10., 1., 2., 3., 4., 5.]))
                mothers = [{'number': 1, 'id': 21}] if merge else []
                with mock.patch('madgraph.interface.reweight_interface.fks_common.combine_ij',
                                return_value=mothers):
                    result = ReweightInterface._compute_ewsudakov_reweight(driver, event, module)
                limits.TestSudakovFactorLimits.assert_factors(
                    self, driver, result, [1.2, 1.3, 1.1, 1.4, 1.5], event)
                module.ewsudakov.assert_called_once()
                momenta = module.ewsudakov.call_args.args[1]
                self.assertEqual(momenta.shape, (len(final) + 2, 4))
                np.testing.assert_allclose(momenta[:2].sum(axis=0), momenta[2:].sum(axis=0), atol=1e-10)
                if merge:
                    energy = math.sqrt(87000.)
                    np.testing.assert_allclose(momenta, [[energy, 0., 0., energy],
                        [energy, 0., 0., -energy], [energy, energy, 0., 0.],
                        [energy, -energy, 0., 0.]], atol=1e-10)
        self.assertEqual(str(event), original)


if __name__ == '__main__':
    unittest.main()
