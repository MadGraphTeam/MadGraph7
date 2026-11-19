################################################################################
#
# Copyright (c) 2026 The MadGraph7 Development team and Contributors
#
# This file is a part of the MadGraph7 project, an application which
# automatically generates Feynman diagrams and matrix elements for arbitrary
# high-energy processes in the Standard Model and beyond.
#
# It is subject to the MadGraph7 license which should accompany this
# distribution.
#
################################################################################
"""The GPD library of GPD processes (Template/LO/Source/GPD/gpd_models.f),
compiled with a small driver: the analytic models (toy, toy2, toy3), their
support and symmetries, continuity at x = +-xi, the t models (exponential
and Regge), the grid model (bilinear interpolation of a user table) and the
settings read from gpd_card.dat. The double-distribution model needs the
PDFs of a madevent output; it is checked in the benchmark report (rows
B1-B6)."""

from __future__ import absolute_import

import math
import os
import shutil
import subprocess
import sys
import tempfile

root_path = os.path.split(os.path.dirname(os.path.realpath(__file__)))[0]
sys.path.append(os.path.join(root_path, os.path.pardir, os.path.pardir))

import unittest as std_unittest

import tests.unit_tests as unittest

import madgraph

GPD_SRC = os.path.join(madgraph.MG5DIR, 'Template', 'LO', 'Source', 'GPD')

DRIVER = """      PROGRAM GPDDRV
C     reads 'ipart x xi t mu' lines, writes H E Ht Et
      IMPLICIT NONE
      INTEGER IPART
      DOUBLE PRECISION X, XI, T, MU, G(4)
 10   READ(*,*,END=20) IPART, X, XI, T, MU
      CALL GPD_EVAL(IPART, X, XI, T, MU, G)
      WRITE(*,'(A,1P,4E25.16)') 'G ', G
      GOTO 10
 20   END
"""


@std_unittest.skipIf(shutil.which('gfortran') is None, 'gfortran is needed')
class GPDLibraryTest(unittest.TestCase):

    tmp = None

    @classmethod
    def setUpClass(cls):
        # three levels down: no ../../Cards or ../../../Cards of a real output
        cls.tmp = tempfile.mkdtemp()
        cls.rundir = os.path.join(cls.tmp, 'a', 'b', 'run')
        os.makedirs(cls.rundir)
        with open(os.path.join(cls.rundir, 'drv.f'), 'w') as f:
            f.write(DRIVER)
        cmd = ['gfortran', '-O1', '-ffixed-line-length-132', '-I' + GPD_SRC,
               os.path.join(GPD_SRC, 'gpd_models.f'), os.path.join(GPD_SRC, 'gpd_dd_dummy.f'),
               'drv.f', '-o', 'drv']
        subprocess.check_call(cmd, cwd=cls.rundir)

    @classmethod
    def tearDownClass(cls):
        if cls.tmp:
            shutil.rmtree(cls.tmp)

    def evaluate(self, card, points):
        with open(os.path.join(self.rundir, 'gpd_card.dat'), 'w') as f:
            for name, value in card.items():
                f.write("  %s = %s  ! test\n" % (value, name))
        inp = "".join("%d %r %r %r %r\n" % p for p in points)
        out = subprocess.run([os.path.join(self.rundir, 'drv')], input=inp, capture_output=True,
                             text=True, cwd=self.rundir, check=True).stdout
        vals = [[float(v) for v in l.split()[1:]] for l in out.splitlines() if l.startswith('G ')]
        self.assertEqual(len(vals), len(points))
        return vals, out

    def close(self, a, b, tol=1e-14):
        self.assertTrue(abs(a - b) <= tol * max(1.0, abs(b)), "%r != %r" % (a, b))

    def test_toy3_closed_form(self):
        xs = [-0.95, -0.6, -0.2, 0.0, 0.1, 0.45, 0.8]
        vals, out = self.evaluate({'gpd_model': 'toy3'}, [(21, x, 0.2, 0.0, 3.0) for x in xs])
        self.assertIn('GPD: model 4', out)
        for x, g in zip(xs, vals):
            s = (1 - x * x) ** 2
            for got, ref in zip(g, [s * (1 + 0.5 * x * x), 0.3 * s, x * s, 0.2 * x * s]):
                self.close(got, ref)

    def test_support_and_symmetry(self):
        """Zero for |x| >= 1; gluon H, E even and Ht, Et odd in x."""
        for model in ('toy', 'toy3'):
            pts = [(21, x, 0.3, 0.0, 3.0) for x in (0.7, -0.7, 0.15, -0.15, 1.0, -1.0, 1.2)]
            vals, out = self.evaluate({'gpd_model': model}, pts)
            for k, sign in ((0, 1), (1, 1), (2, -1), (3, -1)):
                self.close(vals[0][k], sign * vals[1][k])
                self.close(vals[2][k], sign * vals[3][k])
            for g in vals[4:]:
                self.assertEqual(g, [0.0, 0.0, 0.0, 0.0])

    def test_toy_continuous_with_a_kink_at_xi(self):
        """The toy model is continuous at x = +-xi, with a kink there (as
        double distributions have)."""
        xi, h = 0.25, 1e-7
        for ipart in (21, 2):
            pts = [(ipart, s * xi + d, xi, 0.0, 3.0) for s in (1, -1) for d in (-2 * h, -h, 0.0, h, 2 * h)]
            vals, out = self.evaluate({'gpd_model': 'toy'}, pts)
            for block in (vals[:5], vals[5:]):
                h1 = [g[0] for g in block]
                self.assertTrue(abs(h1[1] - h1[3]) < 1e-5 * abs(h1[2]))
                left = (h1[1] - h1[0]) / h
                right = (h1[4] - h1[3]) / h
                self.assertGreater(abs(left - right), 1e-3)

    def test_t_models(self):
        x, xi, t = 0.12, 0.05, -0.4
        card = {'gpd_model': 'toy2', 'gpd_tmodel': 1, 'gpd_tslope_g': 2.25, 'gpd_tslope_q': 1.5}
        vals, out = self.evaluate(card, [(21, x, xi, t, 3.0), (2, x, xi, t, 3.0)])
        self.close(vals[0][0], (1 - x * x) * math.exp(2.25 * t), 1e-13)
        self.close(vals[1][0], (1 - x * x) * math.exp(1.5 * t), 1e-13)
        card.update({'gpd_tmodel': 2, 'gpd_alphap_g': 0.115})
        vals, out = self.evaluate(card, [(21, x, xi, t, 3.0), (21, 0.01, xi, t, 3.0)])
        self.close(vals[0][0], (1 - x * x) * math.exp(2.25 * t) * x ** (-0.115 * t), 1e-13)
        # |beta| -> max(|x|, xi) for the non-dd models
        self.close(vals[1][0], (1 - 1e-4) * math.exp(2.25 * t) * xi ** (-0.115 * t), 1e-13)

    def test_grid_model(self):
        """A table of a function bilinear in (x, xi) is reproduced exactly
        inside the grid; zero outside its x range; one block per parton."""
        xg = [-0.9, -0.3, -0.05, 0.05, 0.3, 0.9]
        xig = [0.01, 0.1, 0.4]
        f = lambda x, xi: 0.7 + 0.3 * x - 1.1 * xi + 0.5 * x * xi
        path = os.path.join(self.rundir, 'grid.dat')
        with open(path, 'w') as out:
            out.write("# test grid\n%d %d\n" % (len(xg), len(xig)))
            out.write(" ".join(repr(x) for x in xg) + "\n")
            out.write(" ".join(repr(x) for x in xig) + "\n")
            for pdg, fname, scale in ((21, 'H', 1.0), (21, 'Ht', -2.0), (2, 'E', 3.0)):
                out.write("# block\n%d %s\n" % (pdg, fname))
                out.write(" ".join(repr(scale * f(x, xi)) for xi in xig for x in xg) + "\n")
        pts = [(21, 0.3, 0.1, 0.0, 3.0), (21, 0.17, 0.27, 0.0, 3.0), (2, -0.5, 0.03, 0.0, 3.0),
               (21, 0.95, 0.1, 0.0, 3.0), (1, 0.17, 0.27, 0.0, 3.0)]
        vals, out = self.evaluate({'gpd_model': 'grid', 'gpd_grid': "'grid.dat'"}, pts)
        self.close(vals[0][0], f(0.3, 0.1), 1e-14)
        self.close(vals[1][0], f(0.17, 0.27), 1e-14)
        self.close(vals[1][2], -2.0 * f(0.17, 0.27), 1e-14)
        self.assertEqual(vals[1][1], 0.0)
        self.close(vals[2][1], 3.0 * f(-0.5, 0.03), 1e-14)
        self.assertEqual(vals[3], [0.0, 0.0, 0.0, 0.0])
        self.assertEqual(vals[4], [0.0, 0.0, 0.0, 0.0])

    def test_card_settings(self):
        """gpd_xtol below 1e-10 is raised (the x quadrature cannot do
        better); unknown entries are ignored."""
        vals, out = self.evaluate({'gpd_model': 'toy2', 'gpd_xtol': 1e-12, 'gpd_gap': 'none',
                                   'gpd_flux': 'chff'}, [(21, 0.3, 0.1, 0.0, 3.0)])
        self.assertIn('raised to 1e-10', out)
        self.close(vals[0][0], 0.91)


if __name__ == '__main__':
    unittest.main()
