################################################################################
#
# Copyright (c) 2009 The MadGraph5_aMC@NLO Development team and Contributors
#
# This file is a part of the MadGraph5_aMC@NLO project, an application which
# automatically generates Feynman diagrams and matrix elements for arbitrary
# high-energy processes in the Standard Model and beyond.
#
# It is subject to the MadGraph5_aMC@NLO license which should accompany this
# distribution.
#
# For more information, visit madgraph.phys.ucl.ac.be and amcatnlo.web.cern.ch
#
################################################################################
"""Massive incoming particles in mg7, as madevent has them (genps.f).

The phase-space mappings of madspace build the incoming momenta massless. The
Integrand puts massive ones on their mass shell, with the same sum, and the
flux 1/(2 sqrt(lambda(s, m1^2, m2^2))) instead of 1/(2 s) (the
incoming_on_shell kernel); for monochromatic beams the run's s and beam
rapidity are those of the massive beam momenta (MadgraphProcess._massive_beams).
Without this, w+ w- > t t~ with 2 x 500 GeV W beams was 10% low.
"""
import math
import os
import sys
import unittest

_file_path = os.path.dirname(os.path.realpath(__file__))
_root = os.path.abspath(os.path.join(_file_path, '..', '..', '..'))
sys.path.insert(0, _root)

try:
    import numpy as np
except ImportError:
    np = None

from tests.unit_tests.various.test_eva_mg7 import _madspace


@unittest.skipIf(np is None, 'numpy required')
class TestIncomingOnShell(unittest.TestCase):

    def on_shell(self, p, masses):
        ms = _madspace()
        if ms is None:
            self.skipTest('madspace build required')
        vec = ms.batch_four_vec_array(p.shape[1])
        fb = ms.FunctionBuilder(
            ms.NamedTypes([('p', vec)]),
            ms.NamedTypes([('p_out', vec), ('flux_ratio', ms.batch_float)]))
        out = fb.incoming_on_shell(fb.input(0), ms.Value(list(masses)))
        fb.output(0, out[0])
        fb.output(1, out[1])
        runtime = ms.FunctionRuntime(fb.function(), ms.default_context())
        return [np.from_dlpack(t) for t in runtime.call([p])]

    def test_on_shell_with_the_same_sum(self):
        rng = np.random.default_rng(7)
        n = 1000
        masses = (80.419, 91.188)
        # massless incoming momenta along +-z (some mirrored: the first one
        # moving to -z), of a pair above threshold, and two outgoing momenta
        e1 = 10 ** rng.uniform(1., 3.5, n)
        e2 = 10 ** rng.uniform(1., 3.5, n)
        mirror = rng.uniform(size=n) < 0.3
        p = np.zeros((n, 4, 4))
        p[:, 0, 0], p[:, 0, 3] = e1, np.where(mirror, -e1, e1)
        p[:, 1, 0], p[:, 1, 3] = e2, np.where(mirror, e2, -e2)
        p[:, 2] = [500., 30., -40., 100.]
        p[:, 3] = [400., -30., 40., -50.]
        p_out, flux = self.on_shell(p, masses)
        p_in = p_out[:, :2]
        s = (p[:, 0, 0] + p[:, 1, 0]) ** 2 - (p[:, 0, 3] + p[:, 1, 3]) ** 2
        above = s > sum(masses) ** 2
        self.assertTrue(above.any() and (~above).any())
        a = above
        # on their mass shell, along the beam axis, in the same directions in
        # the rest frame of the pair (in the lab a slow massive one can move
        # backwards, as in madevent, which builds them in that frame too)
        e_tot = p[:, 0, 0] + p[:, 1, 0]
        z_tot = p[:, 0, 3] + p[:, 1, 3]
        rest_z = lambda q: (e_tot * q[:, 3] - z_tot * q[:, 0])
        for leg, mass in enumerate(masses):
            m2 = p_in[a, leg, 0] ** 2 - p_in[a, leg, 3] ** 2
            np.testing.assert_allclose(m2, mass ** 2, rtol=1e-8)
            self.assertTrue(np.all(p_in[a, leg, 1:3] == 0.))
            self.assertTrue(np.all(np.sign(rest_z(p_in[:, leg])[a])
                                   == np.sign(rest_z(p[:, leg])[a])))
        # the same total momentum, and the outgoing momenta untouched
        np.testing.assert_allclose(p_in[a].sum(axis=1), p[a, :2].sum(axis=1),
                                   rtol=1e-12, atol=1e-9)
        np.testing.assert_array_equal(p_out[:, 2:], p[:, 2:])
        # the flux ratio s / sqrt(lambda)
        m1, m2 = masses
        lam = (s - (m1 + m2) ** 2) * (s - (m1 - m2) ** 2)
        np.testing.assert_allclose(flux[a], s[a] / np.sqrt(lam[a]), rtol=1e-12)
        # below threshold: no weight, momenta as they were
        self.assertTrue(np.all(flux[~a] == 0.))
        np.testing.assert_array_equal(p_out[~a], p[~a])

    def test_monochromatic_beams_come_out_as_madevent_has_them(self):
        """With the run's s and rapidity from _massive_beams, the mapping's
        massless pair goes on shell into exactly the beam momenta."""
        from madgraph.mg7.madgraph_process import MadgraphProcess
        mw = 80.419
        process = self.process([{'incoming': [24, -24], 'outgoing': [6, -6]}],
                               lambda pid: mw)
        process._massive_beams()
        p1, p2 = math.sqrt(400. ** 2 - mw ** 2), math.sqrt(600. ** 2 - mw ** 2)
        self.assertAlmostEqual(process.e_cm ** 2,
                               2 * mw ** 2 + 2 * (400. * 600. + p1 * p2), delta=1e-6)
        # the massless incoming pair of the mapping: CM at sqrt(s), boosted
        half, y = process.e_cm / 2, process.beam_rapidity
        p = np.zeros((1, 3, 4))
        for leg, sign in ((0, 1.), (1, -1.)):
            p[0, leg] = [half * math.cosh(y) + sign * half * math.sinh(y), 0., 0.,
                         sign * half * math.cosh(y) + half * math.sinh(y)]
        p[0, 2] = p[0, 0] + p[0, 1]
        p_out, flux = self.on_shell(p, (mw, mw))
        np.testing.assert_allclose(p_out[0, 0], [400., 0., 0., p1], rtol=1e-12)
        np.testing.assert_allclose(p_out[0, 1], [600., 0., 0., -p2], rtol=1e-12)
        # massless beams keep s = 4 E1 E2
        process.get_mass = lambda pid: 0.
        process.e_cm = 1.
        process._massive_beams()
        self.assertEqual(process.e_cm, 1.)

    @staticmethod
    def process(subprocess_data, get_mass, energies=(400., 600.)):
        """A MadgraphProcess with only what _massive_beams reads, as init_beam
        leaves it for monochromatic beams."""
        from madgraph.mg7.madgraph_process import MadgraphProcess
        process = object.__new__(MadgraphProcess)
        process.subprocess_data = subprocess_data
        process.get_mass = get_mass
        process.beam_energies = energies
        process.e_cm = 2. * math.sqrt(energies[0] * energies[1])
        process.beam_rapidity = 0.5 * math.log(energies[0] / energies[1])
        process.massive_beams = True
        return process

    def test_each_subprocess_has_the_s_of_its_incoming_masses(self):
        """Different incoming masses in different subprocesses (madevent: one
        stot per P directory): each subprocess gets the s and rapidity of its
        own massive beams, the process keeps 4 E1 E2."""
        mmu = 0.10566
        masses = {13: mmu, -13: mmu, 22: 0., 24: 80.419, -24: 80.419}
        process = self.process(
            [{'incoming': [-13, 13], 'outgoing': [6, -6]},
             {'incoming': [22, 22], 'outgoing': [6, -6]},
             {'incoming': [24, -24], 'outgoing': [6, -6]}],
            lambda pid: masses[pid])
        e_cm, y = process.e_cm, process.beam_rapidity
        process._massive_beams()
        self.assertEqual((process.e_cm, process.beam_rapidity), (e_cm, y))
        for m in (0., mmu, 80.419):
            s, rapidity = process.beam_kinematics([m, m])
            p1 = math.sqrt(400. ** 2 - m ** 2)
            p2 = math.sqrt(600. ** 2 - m ** 2)
            self.assertAlmostEqual(s ** 2, 2 * m ** 2 + 2 * (400. * 600. + p1 * p2),
                                   delta=1e-6)
            self.assertAlmostEqual(rapidity, 0.5 * math.log((1000. + p1 - p2)
                                                            / (1000. - p1 + p2)))
        self.assertEqual(process.beam_kinematics([0., 0.]), (e_cm, y))
        # a subprocess whose masses are above the beam energies is refused
        masses[24] = masses[-24] = 500.
        with self.assertRaises(ValueError):
            process._massive_beams()
        # other beams: the process' s for every subprocess
        process.massive_beams = False
        self.assertEqual(process.beam_kinematics([500., 500.]), (e_cm, y))


if __name__ == '__main__':
    unittest.main()
