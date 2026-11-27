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
"""Acceptance tests of exclusive photoproduction with GPDs (the 'gpd' label):
standalone matrix elements, the madevent cross section and event record, and
the refusals. The references are those of the benchmark report of the
gpd_proc branch (docs/benchmark.pdf, rows L2, M1, M5, S1-S4, D7)."""
from __future__ import division
from __future__ import absolute_import

import cmath
import glob
import gzip
import logging
import math
import os
import shutil
import subprocess
import tempfile
import unittest

import madgraph.interface.master_interface as MGCmd
import madgraph.interface.madevent_interface as MECmd
import madgraph.various.banner as banner
from madgraph import MG5DIR, InvalidCmd, MadGraph5Error

pjoin = os.path.join


def jpsi_closed_form(xi, mc=1.55, aewm1=132.507, a_s=0.118, ldme=1.16):
    """avg |M|^2 of gamma p -> J/psi p at t = 0 with H_g = 1 - x^2 (toy2):
    (32 pi^2/3) alpha alpha_s^2 e_c^2 |R(0)|^2 (1 - xi^2) |calH|^2 / m^3,
    calH = -2 + (1-xi^2)/xi [ln((1-xi)/(1+xi)) - i pi], |R(0)|^2 = 4 pi <O>/18."""
    alpha = 1 / aewm1
    r2 = 4 * math.pi * ldme / 18
    h = -2 + (1 - xi * xi) / xi * complex(math.log((1 - xi) / (1 + xi)), -math.pi)
    return 32 * math.pi ** 2 / 3 * alpha * a_s ** 2 * (2 / 3) ** 2 * r2 * (1 - xi * xi) * abs(h) ** 2 / mc ** 3


def jpsi_point(xi, mc=1.55, pz=50.0, M=None):
    """gamma along -z, the pair 2 xi P along +z (P = (pz,0,0,pz)), J/psi of
    mass M (default 2 mc)."""
    M = M or 2 * mc
    qP = M * M / (4 * xi)
    eq = qP / (2 * pz)
    q = [eq, 0.0, 0.0, -eq]
    pair = [2 * xi * pz, 0.0, 0.0, 2 * xi * pz]
    return [q, pair, [a + b for a, b in zip(q, pair)]]


def set_gpd_card(path, settings):
    lines = open(path).read().splitlines()
    for k, v in settings.items():
        for i, l in enumerate(lines):
            w = l.split('!')[0].split('=')
            if len(w) == 2 and w[1].strip() == k:
                lines[i] = "  %-8s = %s" % (v, k)
    open(path, 'w').write("\n".join(lines) + "\n")


class TestCmdGPD(unittest.TestCase):
    """GPD processes in standalone and madevent outputs."""

    def setUp(self):
        self.path = tempfile.mkdtemp(prefix='acc_test_gpd')
        self.out_dir = pjoin(self.path, 'GPDPROC')

    def tearDown(self):
        shutil.rmtree(self.path)

    def new_cmd(self, model='sm_onia-c_mass'):
        cmd = MGCmd.MasterCmd()
        cmd.no_notification()
        cmd.exec_cmd('set automatic_html_opening False', precmd=True, errorhandling=False)
        cmd.exec_cmd('import model %s' % model)
        return cmd

    def test_gpd_standalone(self):
        """gamma p -> J/psi p (gluon GPD, toy2): the standalone |M|^2 against
        the closed form of appendix B at three xi (1e-8), with the J/psi at
        2 m_c and at its PDG mass (the onium reshuffling scales the hard
        momenta at fixed xi, which leaves the closed form unchanged)."""
        cmd = self.new_cmd()
        cmd.exec_cmd('generate a gpd_g > Jpsi(1|3S11)')
        cmd.exec_cmd('output standalone_fortran %s -f' % self.out_dir)
        set_gpd_card(pjoin(self.out_dir, 'Cards', 'gpd_card.dat'), {'gpd_model': 'toy2'})
        pdir = glob.glob(pjoin(self.out_dir, 'SubProcesses', 'P[0-9]*_*'))[0]
        xis = (0.3, 0.05, 0.002, 0.3, 0.05)
        masses = (3.1, 3.1, 3.1, 3.0969, 3.0969)
        with open(pjoin(pdir, 'gpd_points.dat'), 'w') as f:
            for xi, M in zip(xis, masses):
                f.write("%.17e %.17e %.17e\n" % (xi, 0.0, 3.1))
                for v in jpsi_point(xi, M=M):
                    f.write(" ".join("%.17e" % c for c in v) + "\n")
        subprocess.check_call(['make', 'check_gpd'], cwd=pdir, stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL)
        out = subprocess.check_output(['./check_gpd'], cwd=pdir).decode()
        me2 = [float(l.split()[1]) for l in out.splitlines() if l.startswith('ME2')]
        self.assertEqual(len(me2), len(xis))
        for xi, val in zip(xis, me2):
            ref = jpsi_closed_form(xi)
            self.assertLess(abs(val - ref) / ref, 1e-8, "xi=%g: %s against %s" % (xi, val, ref))

    def test_gpd_coherent_standalone(self):
        """'a gpd > Jpsi(1|3S11) / z w+ w-' in standalone: the coherent
        directory (P1_agpd_...) sums the CFFs of the four channels per
        helicity; its |M|^2 equals the square, with the proton weights, of
        the CFFs of the channel directories summed by hand (1e-12), and it
        differs from the sum of the channel |M|^2 (the interference)."""
        cmd = self.new_cmd()
        cmd.exec_cmd('generate a gpd > Jpsi(1|3S11) / z w+ w-')
        cmd.exec_cmd('output standalone_fortran %s -f' % self.out_dir)
        sub = pjoin(self.out_dir, 'SubProcesses')
        coh = [d for d in os.listdir(sub) if d.startswith('P') and '_agpd_' in d]
        chans = sorted(d for d in os.listdir(sub) if d.startswith('P') and '_agpd' in d and d not in coh)
        self.assertEqual(len(coh), 1)
        self.assertEqual(len(chans), 4)
        xi, t = 0.3, -0.2
        res = {}
        for d in coh + chans:
            pdir = pjoin(sub, d)
            with open(pjoin(pdir, 'gpd_points.dat'), 'w') as f:
                f.write("%.17e %.17e %.17e\n" % (xi, t, 2.0))
                for v in jpsi_point(xi):
                    f.write(" ".join("%.17e" % c for c in v) + "\n")
            subprocess.check_call(['make', 'check_gpd'], cwd=pdir, stdout=subprocess.DEVNULL,
                                  stderr=subprocess.DEVNULL)
            out = subprocess.check_output(['./check_gpd'], cwd=pdir).decode()
            me2 = [float(l.split()[1]) for l in out.splitlines() if l.startswith('ME2')][0]
            cff = []
            for l in out.splitlines():
                if l.startswith('CFF'):
                    v = [float(a) for a in l.split()[2:]]
                    cff.append([complex(v[2 * g], v[2 * g + 1]) for g in range(4)])
            res[d] = (me2, cff)
        mp = 0.93827208816
        tm = t / (4 * mp * mp)
        total = 0.0
        for k in range(len(res[coh[0]][1])):
            h, e, ht, et = (sum(res[d][1][k][g] for d in chans) for g in range(4))
            total += 2 * ((1 - xi * xi) * (abs(h) ** 2 + abs(ht) ** 2) - (xi * xi + tm) * abs(e) ** 2
                          - xi * xi * tm * abs(et) ** 2
                          - 2 * xi * xi * ((h * e.conjugate()).real + (ht * et.conjugate()).real))
        total /= 4.0
        self.assertLess(abs(res[coh[0]][0] / total - 1), 1e-12)
        incoherent = sum(res[d][0] for d in chans)
        self.assertGreater(abs(res[coh[0]][0] / incoherent - 1), 1e-4)

    def test_gpd_madevent(self):
        """pp -> p J/psi p at 13.6 TeV (toy2, exp(2.25 t) per amplitude, both
        photon emitters, gap survival of 2207.03012 eq. (10) by default):
        madevent against the gamma-UPC EDFF flux with the survival convolved
        with the closed form (751.35 pb, benchmark row M8; 805.43 pb without
        the survival, row M1), and the event record gamma p -> J/psi p with
        exact momentum conservation."""
        cmd = self.new_cmd()
        cmd.exec_cmd('generate a gpd_g > Jpsi(1|3S11)')
        cmd.exec_cmd('output madevent %s -f' % self.out_dir)
        card = pjoin(self.out_dir, 'Cards', 'run_card.dat')
        rc = banner.RunCardLO(card)
        for k, v in dict(lpp1=5, lpp2=5, ebeam1=6800, ebeam2=6800, pdlabel1='nn23lo1', pdlabel2='nn23lo1',
                         fixed_ren_scale=True, fixed_fac_scale=True, scale=91.188, dsqrt_q2fact1=91.188,
                         dsqrt_q2fact2=91.188, nevents=2000, use_syst=False, nhel=0).items():
            rc.set(k, v, user=True)
        rc.write(card)
        # the per-point quadrature (the madevent default is mc), whose errors
        # are those of the phase-space integration alone
        set_gpd_card(pjoin(self.out_dir, 'Cards', 'gpd_card.dat'),
                     {'gpd_model': 'toy2', 'gpd_tmodel': '1', 'gpd_tslope_g': '2.25',
                      'gpd_x_integration': 'quadrature'})
        me = MECmd.MadEventCmdShell(me_dir=self.out_dir)
        me.no_notification()
        me.exec_cmd('set automatic_html_opening False')
        me.exec_cmd('generate_events -f')
        val = me.results.current['cross']
        err = me.results.current['error']
        target = 751.35
        self.assertLess(abs(val - target) / err, 4., "%s +- %s against %s pb" % (val, err, target))
        lhe = glob.glob(pjoin(self.out_dir, 'Events', '*', 'unweighted_events.lhe.gz'))[0]
        nev = 0
        with gzip.open(lhe, 'rt') as f:
            ev = None
            for line in f:
                if line.startswith('<event'):
                    ev = []
                elif line.startswith('</event'):
                    n = int(ev[0].split()[0])
                    parts = [l.split() for l in ev[1:1 + n]]
                    self.assertEqual(sorted(int(w[0]) for w in parts), [22, 443, 2212, 2212])
                    bal = [0.0] * 4
                    for w in parts:
                        sgn = -1 if int(w[1]) == -1 else 1
                        for k, c in enumerate((9, 6, 7, 8)):
                            bal[k] += sgn * float(w[c])
                    self.assertLess(max(abs(b) for b in bal), 1e-5)
                    nev += 1
                    ev = None
                elif ev is not None:
                    ev.append(line)
        self.assertEqual(nev, 2000)

    def test_gpd_refusals(self):
        """GPDs on both beams and NLO are refused at generation; the options of
        a 'gpd' process reach every channel; a P-wave colour-singlet onium with
        a photon (end-point divergence) is refused at run time."""
        cmd = self.new_cmd()
        self.assertRaises((InvalidCmd, MadGraph5Error), cmd.exec_cmd, 'generate gpd gpd > Jpsi(1|3S11)',
                          errorhandling=False)
        self.assertRaises((InvalidCmd, MadGraph5Error), cmd.exec_cmd, 'generate a gpd > Jpsi(1|3S11) [QCD]',
                          errorhandling=False)
        cmd.exec_cmd('generate a gpd > Jpsi(1|3S11) / z w+ w- @3')
        amps = cmd._curr_amps
        self.assertEqual(len(amps), 4)
        for amp in amps:
            proc = amp.get('process')
            self.assertEqual(proc.get('id'), 3)
            self.assertEqual(set(abs(i) for i in proc.get('forbidden_particles')), set([23, 24]))
        # run time: chi_c0 gamma, double poles at x = -+xi
        cmd.exec_cmd('generate a gpd_g > chic0(1|3P01) a')
        cmd.exec_cmd('output standalone_fortran %s -f' % self.out_dir)
        set_gpd_card(pjoin(self.out_dir, 'Cards', 'gpd_card.dat'), {'gpd_model': 'toy3'})
        onia = pjoin(self.out_dir, 'Cards', 'onia_card.dat')
        lines = open(onia).read().splitlines()
        for i, l in enumerate(lines):
            w = l.split()
            if len(w) > 2 and w[0] == '10441' and 'mass for' in l:
                lines[i] = "  10441  3.1  # mass for chic0(1|3P01)"
        open(onia, 'w').write("\n".join(lines) + "\n")
        pdir = glob.glob(pjoin(self.out_dir, 'SubProcesses', 'P[0-9]*_*'))[0]
        rs = math.sqrt(30.0)
        eg = (30.0 - 3.1 ** 2) / (2 * rs)
        th = 1.0
        pg = [eg, eg * math.sin(th), 0.0, eg * math.cos(th)]
        q, pair = [rs / 2, 0, 0, -rs / 2], [rs / 2, 0, 0, rs / 2]
        pc = [q[k] + pair[k] - pg[k] for k in range(4)]
        with open(pjoin(pdir, 'gpd_points.dat'), 'w') as f:
            f.write("0.2 -0.3 2.0\n")
            for v in (q, pair, pc, pg):
                f.write(" ".join("%.17e" % c for c in v) + "\n")
        subprocess.check_call(['make', 'check_gpd'], cwd=pdir, stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL)
        r = subprocess.run(['./check_gpd'], cwd=pdir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn('double pole', r.stdout.decode())
