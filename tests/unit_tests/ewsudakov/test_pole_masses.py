"""Run-local pole masses must reach the kinematic helpers without rereading cards."""

import math
import unittest
from unittest import mock

from madgraph.various import banner, fxfx_ewsudakov as fx
from tests.unit_tests.ewsudakov.test_fks_mappings import _mk_event


DEFAULTS = {24: 80.385, 23: 91.1876, 6: 173.3, 25: 125.}


class TestPoleMasses(unittest.TestCase):
    def setUp(self):
        state = mock.patch.multiple(fx, MW_POLE=DEFAULTS[24], MZ_POLE=DEFAULTS[23],
                                    MT_POLE=DEFAULTS[6], MH_POLE=DEFAULTS[25],
                                    _FKS_COUNT_ACTIVE=False)
        state.start()
        self.addCleanup(state.stop)

    def make_banner(self, masses):
        value = banner.Banner()
        value['slha'] = 'BLOCK MASS\n' + ''.join(' %d %.12g\n' % item for item in masses.items())
        value.get_detail = mock.Mock(wraps=value.get_detail)
        return value

    def interface(self, masses):
        driver = fx.FxFxEWSudakovMixin()
        driver.banner = self.make_banner(masses)
        return driver

    def assert_masses(self, expected):
        actual = {24: fx.MW_POLE, 23: fx.MZ_POLE, 6: fx.MT_POLE, 25: fx.MH_POLE}
        self.assertEqual(actual, expected)
        self.assertEqual({pdg: fx._get_mother_mass(pdg) for pdg in expected}, expected)

    def test_interfaces_restore_their_masses_without_rereading_cards(self):
        first_masses = {24: 80., 23: 90., 6: 170., 25: 120.}
        second_masses = {24: 110., 23: 120., 6: 180., 25: 130.}
        first, second = self.interface(first_masses), self.interface(second_masses)
        for driver, masses in ((first, first_masses), (second, second_masses)):
            driver._init_pole_masses_from_banner()
            self.assert_masses(masses)
            self.assertEqual(driver.banner.get_detail.call_count, 4)
            driver.banner.get_detail.side_effect = AssertionError('Card must not be reread')
        for _ in range(5):
            for driver, masses in ((first, first_masses), (second, second_masses), (first, first_masses)):
                driver._init_pole_masses_from_banner()
                self.assert_masses(masses)
                self.assertEqual(driver.banner.get_detail.call_count, 4)

    def test_reimporting_on_same_interface_reads_new_banner_once(self):
        first_masses = dict(DEFAULTS)
        second_masses = {24: 85., 23: 95., 6: 175., 25: 135.}
        driver = self.interface(first_masses)
        driver._init_pole_masses_from_banner()
        original_banner = driver.banner
        driver.banner = self.make_banner(second_masses)
        for _ in range(5):
            driver._init_pole_masses_from_banner()
            self.assert_masses(second_masses)
        self.assertEqual(original_banner.get_detail.call_count, 4)
        self.assertEqual(driver.banner.get_detail.call_count, 4)

    def test_missing_entries_use_defaults_instead_of_previous_run(self):
        previous = self.interface({24: 110., 23: 120., 6: 180., 25: 130.})
        previous._init_pole_masses_from_banner()
        for entries in ({24: 82.}, {}):
            with self.subTest(entries=entries):
                current = self.interface(entries)
                current._init_pole_masses_from_banner()
                self.assert_masses({**DEFAULTS, **entries})
                current._init_pole_masses_from_banner()
                self.assertEqual(current.banner.get_detail.call_count, 4)

    def test_missing_banner_does_not_prevent_later_initialization(self):
        previous = self.interface({24: 110., 23: 120., 6: 180., 25: 130.})
        previous._init_pole_masses_from_banner()
        driver = fx.FxFxEWSudakovMixin()
        driver._init_pole_masses_from_banner()
        self.assert_masses(DEFAULTS)
        driver.banner = self.make_banner({24: 82.})
        for _ in range(5):
            driver._init_pole_masses_from_banner()
            self.assert_masses({**DEFAULTS, 24: 82.})
        self.assertEqual(driver.banner.get_detail.call_count, 4)

    def test_active_masses_reach_invariant_cut_and_on_shell_projection(self):
        first = self.interface({24: 80.})
        second = self.interface({24: 110.})
        # All massless pair invariants are at least 10000 GeV^2: above
        # 80^2 but below 110^2. The same event changes cut with the run.
        transverse = math.sqrt(100.**2 - 50.**2)
        momenta = [[100., 0., 0., 100.], [100., 0., 0., -100.],
                   [100., transverse, 0., 50.], [100., -transverse, 0., -50.]]
        for driver, mass, small in ((first, 80., False), (second, 110., True), (first, 80., False)):
            with self.subTest(mass=mass):
                driver._init_pole_masses_from_banner()
                self.assertEqual(driver._has_small_invariants(momenta, [-1, -1, 1, 1]), small)
                event = _mk_event([
                    (21, -1, 250., 0., 0., 250.), (21, -1, 250., 0., 0., -250.),
                    (-13, 1, 150., 150., 0., 0.), (14, 1, 250., -250., 0., 0.),
                    (21, 1, 100., 100., 0., 0.),
                ])
                method = fx.fxfx_merge_particles_kinematics(event, 2, 3, [{'number': 3, 'id': 24}])
                self.assertIn(method, fx.FKS_SUCCESS_METHODS)
                self.assertEqual(event[2].mass, mass)
                momentum = fx.FourMomentum(event[2])
                invariant_mass = math.sqrt(momentum.E**2 - momentum.px**2
                                           - momentum.py**2 - momentum.pz**2)
                self.assertAlmostEqual(invariant_mass, mass, places=7)
                event.check_kinematics_only()
        self.assertEqual(first.banner.get_detail.call_count, 4)
        self.assertEqual(second.banner.get_detail.call_count, 4)


if __name__ == '__main__':
    unittest.main()
