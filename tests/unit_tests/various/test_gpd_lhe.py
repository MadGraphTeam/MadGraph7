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
"""The event record of GPD processes (madgraph/various/gpd_lhe.py): the
elastic protons added after the unweighting, with exact 2 -> 2 kinematics
gamma p -> X p' at the W, M_X and t of the event; and the run card of the
lpp = 5 proton (madgraph/various/banner.py)."""

from __future__ import absolute_import

import gzip
import math
import os
import random
import shutil
import sys
import tempfile

root_path = os.path.split(os.path.dirname(os.path.realpath(__file__)))[0]
sys.path.append(os.path.join(root_path, os.path.pardir, os.path.pardir))

import tests.unit_tests as unittest

import madgraph.various.banner as banner
import madgraph.various.gpd_lhe as gpd_lhe

MP = gpd_lhe.MPROTON


def dot(a, b):
    return a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3]


def minus(a, b):
    return [x - y for x, y in zip(a, b)]


def plus(a, b):
    return [x + y for x, y in zip(a, b)]


class GPDRecoilTest(unittest.TestCase):
    """gpd_lhe.recoil: exact gamma(q) + p -> X + p'."""

    def close(self, a, b, tol, msg=None):
        self.assertTrue(abs(a - b) <= tol, "%r != %r within %g %s" % (a, b, tol, msg or ''))

    def kinematics(self, eg=200.0, eb=6800.0, mx=3.0969, xi=None):
        q = [eg, 0.0, 0.0, eg]
        pz = math.sqrt(eb * eb - MP * MP)
        pin = [eb, 0.0, 0.0, -pz]
        W2 = dot(plus(q, pin), plus(q, pin))
        kx = [0.0, 0.0, 0.0, 0.0]
        # any X of mass mx inside the gamma p system
        tot = plus(q, pin)
        kx = [tot[0] * 0.999, 0.0, 0.0, 0.0]
        kx[3] = math.copysign(math.sqrt(max(kx[0] ** 2 - mx * mx, 0.0)), tot[3])
        return q, pin, kx, W2

    def test_conservation_and_mass_shells(self):
        q, pin, kx, W2 = self.kinematics()
        for t in (-0.05, -0.4, -1.5):
            pts = []
            for phi in (0.3, 2.0, 5.5):
                kx_new, pout, tused = gpd_lhe.recoil(q, pin, kx, t, phi)
                self.close(tused, t, 1e-9)
                self.close(dot(minus(pin, pout), minus(pin, pout)), t, 1e-6)
                self.close(dot(pout, pout), MP * MP, 1e-6)
                self.close(math.sqrt(dot(kx_new, kx_new)), math.sqrt(dot(kx, kx)), 1e-6)
                for i in range(4):
                    self.close(q[i] + pin[i], kx_new[i] + pout[i], 1e-7)
                pts.append(math.atan2(pout[2], pout[1]))
            # the azimuth follows phi (up to the orientation of the beam axis)
            d1 = (pts[1] - pts[0]) % (2 * math.pi)
            d2 = (pts[2] - pts[1]) % (2 * math.pi)
            self.assertTrue(abs(d1 - 1.7) < 1e-9 or abs(d1 - (2 * math.pi - 1.7)) < 1e-9, d1)
            self.assertTrue(abs(d2 - 3.5) < 1e-9 or abs(d2 - (2 * math.pi - 3.5)) < 1e-9, d2)

    def test_t_moved_into_the_physical_range(self):
        """|t| below |t_min| (the massless t_min of the matrix element) is
        moved to the physical t_min."""
        # near the threshold (W = 5.3 GeV), where t_min is well conditioned
        q, pin, kx, W2 = self.kinematics(eg=1e-3)
        kx_new, pout, tused = gpd_lhe.recoil(q, pin, kx, -1e-12, 0.0)
        # t_min of gamma p -> X p at W (forward p'), independently
        W = math.sqrt(W2)
        MX2 = dot(kx, kx)
        lam = lambda a, b, c: a * a + b * b + c * c - 2 * (a * b + a * c + b * c)
        ein, pin_cm = (W2 + MP * MP) / (2 * W), math.sqrt(lam(W2, 0.0, MP * MP)) / (2 * W)
        eout, pout_cm = (W2 + MP * MP - MX2) / (2 * W), math.sqrt(lam(W2, MX2, MP * MP)) / (2 * W)
        tmin = 2 * MP * MP - 2 * (ein * eout - pin_cm * pout_cm)
        self.assertLess(tused, -1e-12)
        self.assertLess(tmin, -0.05)
        # the boost of the 6.8 TeV proton to the gamma p frame (gamma ~ 1300)
        # costs ~1e-9 GeV on its momentum
        self.close(tused, tmin, 1e-7 * abs(tmin))


EVENT = """ 4      1 +7.5000000e+02 9.11880000e+01 7.54677100e-03 1.18000000e-01
       22 -1    0    0    0    0 +0.0000000000e+00 +0.0000000000e+00 +2.0000000000e+02 2.0000000000e+02 0.0000000000e+00 0.0000e+00 9.0000e+00
       23 -1    0    0    0    0 +0.0000000000e+00 +0.0000000000e+00 %+.10e %.10e 0.0000000000e+00 0.0000e+00 9.0000e+00
      -13  1    1    2    0    0 +1.0000000000e+00 +0.5000000000e+00 %+.10e %.10e 1.0566000000e-01 0.0000e+00 9.0000e+00
       13  1    1    2    0    0 -1.0000000000e+00 -0.5000000000e+00 %+.10e %.10e 1.0566000000e-01 0.0000e+00 9.0000e+00
#gpd   -2.5000000000000E-01  -3.0000000000000E-01
"""


def make_event(eb=6800.0, eg=200.0, epair=0.0123):
    """gamma(+z) + pair(-z) -> mu+ mu-, back to back in the transverse plane."""
    ppair = epair
    pz_tot = eg - ppair
    e_tot = eg + epair
    m = 0.10566
    # mu+ and mu- with opposite transverse momenta, sharing pz and E
    pt2 = 1.0 + 0.25
    em = e_tot / 2
    pzm = pz_tot / 2
    # adjust: the muon energy fixed by its momentum (mass shell)
    e1 = math.sqrt(pt2 + pzm ** 2 + m * m)
    # rescale the pair energy so that energy is conserved exactly
    epair = 2 * e1 - eg
    pz_tot = eg - epair
    pzm = pz_tot / 2
    e1 = math.sqrt(pt2 + pzm ** 2 + m * m)
    epair = 2 * e1 - eg
    return (EVENT % (-epair, epair, pzm, e1, pzm, e1)).splitlines(True)


class GPDEventRecordTest(unittest.TestCase):

    def check_event(self, lines, eb):
        head = lines[0].split()
        nup = int(head[0])
        parts = [l.split() for l in lines[1:1 + nup]]
        mom = lambda w: [float(w[9]), float(w[6]), float(w[7]), float(w[8])]
        ins = [mom(w) for w in parts if int(w[1]) == -1]
        outs = [mom(w) for w in parts if int(w[1]) == 1]
        for k in range(4):
            self.assertTrue(abs(sum(p[k] for p in ins) - sum(p[k] for p in outs)) < 1e-9 * eb)
        return parts

    def test_rewrite_event(self):
        eb = (6800.0, 6800.0)
        lines = make_event()
        new, tused, moved = gpd_lhe.rewrite_event(lines, eb, random.Random(3))
        self.assertFalse(moved)
        # the pair moves along -z: beam 2, t = t2
        self.assertTrue(abs(tused + 0.3) < 1e-9, tused)
        parts = self.check_event(new, eb[0])
        pdgs = [(int(w[0]), int(w[1])) for w in parts]
        self.assertEqual(pdgs, [(22, -1), (2212, -1), (-13, 1), (13, 1), (2212, 1)])
        self.assertEqual(int(new[0].split()[0]), 5)
        for w in parts:
            if int(w[0]) == 2212:
                self.assertTrue(abs(float(w[10]) - MP) < 1e-9)
        info = [l for l in new if l.startswith('#gpd')]
        self.assertEqual(len(info), 1)
        self.assertIn('beam= 2', info[0])
        # an event without the pair (or without its #gpd line) is unchanged
        plain = [l for l in lines if not l.startswith('#gpd')]
        self.assertEqual(gpd_lhe.rewrite_event(plain, eb, random.Random(3))[0], plain)

    def test_add_protons(self):
        tmp = tempfile.mkdtemp()
        try:
            path = os.path.join(tmp, 'events.lhe.gz')
            with gzip.open(path, 'wt') as f:
                f.write("<LesHouchesEvents version=\"3.0\">\n<init>\n</init>\n")
                for k in range(3):
                    f.write("<event>\n")
                    f.writelines(make_event(eg=100.0 + 50 * k))
                    f.write("</event>\n")
                f.write("</LesHouchesEvents>\n")
            nev, nmoved = gpd_lhe.add_protons(path, (6800.0, 6800.0), seed=1)
            self.assertEqual((nev, nmoved), (3, 0))
            text = gzip.open(path, 'rt').read()
            self.assertEqual(text.count('<event>'), 3)
            states = [tuple(l.split()[:2]) for l in text.splitlines() if len(l.split()) == 13]
            self.assertEqual(states.count(('2212', '-1')), 3)
            self.assertEqual(states.count(('2212', '1')), 3)
            self.assertEqual(states.count(('23', '-1')), 0)
        finally:
            shutil.rmtree(tmp)

    def test_rewrite_event_ions(self):
        """Nuclei (heavy-ion GPD processes): coherent, the ion recoils as a
        whole; incoherent, a nucleon of momentum p_A/A recoils and the
        remnant (A - 1) is written; momentum conserved exactly."""
        eb = (522080.0, 522080.0)
        pb = dict(pdg=gpd_lhe.ion_pdg(82, 126), mass=193.729, A=208, Z=82)
        self.assertEqual(pb['pdg'], 1000822080)
        lines = make_event(eg=50.0, epair=3e-3)
        new, tused, moved = gpd_lhe.rewrite_event(lines, eb, random.Random(5), beams=[pb, pb])
        parts = self.check_event(new, eb[0])
        self.assertEqual([(int(w[0]), int(w[1])) for w in parts],
                         [(22, -1), (1000822080, -1), (-13, 1), (13, 1), (1000822080, 1)])
        for w in parts[-1:]:
            self.close(float(w[10]), 193.729, 1e-9)
        info = [l for l in new if l.startswith('#gpd')][0]
        self.assertIn('A= 208 nucl= coherent', info)
        self.assertTrue(abs(tused + 0.3) < 1e-6, tused)
        new, tused, moved = gpd_lhe.rewrite_event(lines, eb, random.Random(5), beams=[pb, pb],
                                                  incoherent=True)
        parts = self.check_event(new, eb[0])
        pdgs = [(int(w[0]), int(w[1])) for w in parts]
        self.assertEqual(pdgs[:4], [(22, -1), (1000822080, -1), (-13, 1), (13, 1)])
        self.assertIn(pdgs[4], [(2212, 1), (2112, 1)])
        self.assertEqual(pdgs[5], (1000812070 if pdgs[4][0] == 2212 else 1000822070, 1))
        self.assertEqual(int(new[0].split()[0]), 6)
        self.assertIn('nucl= incoherent', [l for l in new if l.startswith('#gpd')][0])
        # the remnant carries (A - 1)/A of the ion
        self.close(float(parts[5][9]), eb[1] * 207 / 208, 1e-9)

    def close(self, a, b, tol):
        self.assertTrue(abs(a - b) <= tol * max(1.0, abs(b)), "%r != %r" % (a, b))


class GPDRunCardTest(unittest.TestCase):

    def test_lpp5(self):
        card = banner.RunCardLO()
        card.set('lpp1', 5)
        card.set('lpp2', 5)
        self.assertEqual(card['lpp1'], 5)
        self.assertEqual(card['lpp2'], 5)
        self.assertEqual(card.get_idbmup(5), 2212)
        self.assertEqual(card.get_idbmup(5, beam=2), 2212)

    def test_lpp5_ions(self):
        """Heavy ions on GPD beams (lpp = 5): allowed, with the ion code."""
        card = banner.RunCardLO()
        card.set('lpp1', 5)
        card.set('lpp2', 5)
        card.set('nb_proton2', 82)
        card.set('nb_neutron2', 126)
        card.set('mass_ion2', 193.729)
        card.set('ebeam2', 522080.0)
        card.check_validity()
        self.assertEqual(card.get_idbmup(5, beam=2), 1000822080)
        self.assertEqual(card.get_idbmup(5, beam=1), 2212)

    def test_lpp5_no_systematics(self):
        """GPD beams switch off the systematics reweighting (it assumes collinear
        PDFs and stopped with an exception after the events of a GPD run), also
        with an electron on beam 1; other beams keep it."""
        for lpp1, lpp2 in ((5, 5), (0, 5)):
            card = banner.RunCardLO()
            card.set('lpp1', lpp1)
            card.set('lpp2', lpp2)
            card.set('use_syst', True)
            card.check_validity()
            self.assertFalse(card['use_syst'])
        card = banner.RunCardLO()
        card.set('lpp1', 1)
        card.set('lpp2', 1)
        card.set('use_syst', True)
        card.check_validity()
        self.assertTrue(card['use_syst'])


if __name__ == '__main__':
    unittest.main()
