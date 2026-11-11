from __future__ import division
from __future__ import absolute_import
import subprocess
import unittest
import os
import re
import shutil
import sys
import logging
import time
import tempfile
import math
import glob
import gzip

logger = logging.getLogger('test_cmd')

import tests.unit_tests.iolibs.test_file_writers as test_file_writers

import madgraph.interface.master_interface as MGCmd
import madgraph.interface.madevent_interface as MECmd
import madgraph.interface.launch_ext_program as launch_ext
import madgraph.iolibs.files as files

import madgraph.various.misc as misc
import madgraph.various.lhe_parser as lhe_parser
import madgraph.various.banner as banner_mod
import madgraph.various.lhe_parser as lhe_parser
import madgraph.various.banner as banner

_file_path = os.path.split(os.path.dirname(os.path.realpath(__file__)))[0]
_pickle_path =os.path.join(_file_path, 'input_files')

from madgraph import MG4DIR, MG5DIR, MadGraph5Error, InvalidCmd

pjoin = os.path.join

#===============================================================================
# TestCmd
#===============================================================================
class TestMadSpin(unittest.TestCase):
    """test that we can launch everything from a single file"""

    def setUp(self):
        
        self.debuging = unittest.debug
        if self.debuging:
            self.path = pjoin(MG5DIR, 'MS_TEST')
            if os.path.exists(self.path):
                shutil.rmtree(self.path)
            os.mkdir(self.path) 
        else:
            self.path = tempfile.mkdtemp(prefix='ms_test_mg5')
        self.run_dir = pjoin(self.path, 'MGPROC') 
        
    
    def tearDown(self):

        if not self.debuging:
            shutil.rmtree(self.path)
        self.assertFalse(self.debuging)

    def test_madspin_onium_decay(self):
        """Spin-correlated decay of a bound state: q q~ > gamma* > J/psi
        (colour singlet) along the beam is transverse, so its MadSpin decay
        into mu+ mu- goes as 1 + cos^2(theta) in the J/psi rest frame (theta
        from the beam), and as 1 with spinmode none. The onia card travels
        in the banner (onium_mass 3.1 = 2 m_c), the total width is the DECAY
        block of the J/psi in the param_card, and the branching ratio is the
        LO partial width over it, or a BR line of that block."""

        if not misc.default_f2py():
            raise unittest.SkipTest('no f2py')

        def madgraph(name, lines):
            # through bin/madgraph, as a user: the proc card of the events then
            # has the generate line MadSpin rebuilds the production from
            script = pjoin(self.path, name)
            with open(script, 'w') as fsock:
                fsock.write('set automatic_html_opening False\n' + '\n'.join(lines) + '\n')
            subprocess.call([pjoin(MG5DIR, 'bin', 'madgraph'), script], cwd=self.path,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                            stdin=subprocess.DEVNULL)

        madgraph('output.mg', ['import model sm_onia-c_mass',
                               'generate p p > Jpsi(1|3S11) QCD=0 / z h',
                               'output madevent %s -f -nojpeg' % self.run_dir])
        card = pjoin(self.run_dir, 'Cards', 'onia_card.dat')
        text = re.sub(r'(?m)^(\s*443\s+)[0-9.Ee+-]+(\s+# mass for)', r'\g<1>3.1\2', open(card).read())
        open(card, 'w').write(text)
        with open(pjoin(self.run_dir, 'Cards', 'param_card.dat'), 'a') as fsock:
            fsock.write('\nDECAY  443  9.26e-05\n')
        madgraph('launch.mg', ['launch %s' % self.run_dir, 'set nevents 400',
                               'set pdlabel nn23lo1', 'set iseed 31'])
        events = pjoin(self.run_dir, 'Events', 'run_01', 'unweighted_events.lhe.gz')
        production = lhe_parser.EventFile(events).cross
        lhe = pjoin(self.path, 'jpsi.lhe.gz')
        shutil.copy(events, lhe)
        self.assertIn('<MGOniaCard>', gzip.open(lhe, 'rt').read())

        def decay(spinmode, tag):
            script = pjoin(self.path, 'ms_%s' % tag)
            with open(script, 'w') as fsock:
                fsock.write('import jpsi.lhe.gz\nset spinmode %s\nset seed 5\n'
                            'decay Jpsi > mu+ mu-\nlaunch\n' % spinmode)
            for d in glob.glob(pjoin(self.path, 'madspin_*')) + glob.glob(pjoin(self.path, 'decay_*')):
                shutil.rmtree(d)
            subprocess.call([pjoin(MG5DIR, 'MadSpin', 'madspin'), script], cwd=self.path,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                            stdin=subprocess.DEVNULL)
            out = pjoin(self.path, 'jpsi_decayed.lhe.gz')
            self.assertTrue(os.path.exists(out), spinmode)
            events = lhe_parser.EventFile(out)
            cross = events.cross
            cos = []
            for event in events:
                psi = [p for p in event if p.pdg == 443]
                mu = [p for p in event if p.pdg == 13 and p.status == 1]
                self.assertEqual(len(psi), 1)
                self.assertEqual(psi[0].status, 2)
                self.assertEqual(len(mu), 1)
                rest = lhe_parser.FourMomentum(mu[0]).boost_to_restframe(
                    lhe_parser.FourMomentum(psi[0]))
                cos.append(rest.pz / math.sqrt(rest.px**2 + rest.py**2 + rest.pz**2))
            os.rename(out, pjoin(self.path, 'jpsi_decayed_%s.lhe.gz' % tag))
            # 1 + a cos^2: <cos^2> = (1/3 + a/5) / (1 + a/3)
            m2 = sum(c * c for c in cos) / len(cos)
            m4 = sum(c ** 4 for c in cos) / len(cos)
            err_m2 = math.sqrt((m4 - m2 * m2) / len(cos))
            return cross, m2, err_m2, len(cos)

        cross, m2, err, n = decay('madspin', 'madspin')
        self.assertEqual(n, 400)
        self.assertLess(abs(m2 - (1. / 3 + 1. / 5) / (1 + 1. / 3)), 4 * err, (m2, err))
        # BR = Gamma_LO(J/psi > mu+ mu-) / Gamma = 4 e_c^2 alpha^2 |R(0)|^2 / M^2 / Gamma
        alpha, M = 1 / 132.507, 3.1
        r2 = 1.16 * 4 * math.pi / 18
        br = 4 * (2. / 3) ** 2 * alpha ** 2 * r2 / M ** 2 / 9.26e-05
        self.assertLess(abs(cross / production / br - 1), 1e-3, (cross, production, br))

        cross, m2, err, n = decay('none', 'none')
        self.assertLess(abs(m2 - 1. / 3), 4 * err, (m2, err))

    def test_madspin_onium_octet_decay(self):
        """Spin-correlated decay of a hadron produced as colour octets, by
        heavy-quark spin symmetry: g g > Jpsi(1|3PJ8) (J = 0, 1, 2 with the
        HQSS LDMEs) gives the J/psi the spin of the (L_z, S_z) density traced
        over L_z, from the uncoupled 3P[8] matrix element. At LO the 3P2[8]
        has J_z = +-2 (S_z = +-1) and the 3P0[8] gives S_z = 0 with 1/3 of its
        rate; sigma_2 / sigma_0 = 4/3, so rho_00 = 1/7 and the mu+ mu- go as
        1 + cos^2(theta)/2. The octets become J/psi (status 2, pdg 443) in
        the decayed events, with the branching ratio of the J/psi."""

        if not misc.default_f2py():
            raise unittest.SkipTest('no f2py')

        def madgraph(name, lines):
            script = pjoin(self.path, name)
            with open(script, 'w') as fsock:
                fsock.write('set automatic_html_opening False\n' + '\n'.join(lines) + '\n')
            subprocess.call([pjoin(MG5DIR, 'bin', 'madgraph'), script], cwd=self.path,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                            stdin=subprocess.DEVNULL)

        madgraph('output.mg', ['import model sm_onia-c_mass',
                               'generate g g > Jpsi(1|3PJ8)',
                               'output madevent %s -f -nojpeg' % self.run_dir])
        with open(pjoin(self.run_dir, 'Cards', 'param_card.dat'), 'a') as fsock:
            fsock.write('\nDECAY  443  9.26e-05\n')
        madgraph('launch.mg', ['launch %s' % self.run_dir, 'set nevents 3000',
                               'set pdlabel nn23lo1', 'set iseed 21'])
        events = pjoin(self.run_dir, 'Events', 'run_01', 'unweighted_events.lhe.gz')
        production = lhe_parser.EventFile(events).cross
        shutil.copy(events, pjoin(self.path, 'octet.lhe.gz'))
        script = pjoin(self.path, 'ms_octet')
        with open(script, 'w') as fsock:
            fsock.write('import octet.lhe.gz\nset spinmode madspin\nset seed 5\n'
                        'decay Jpsi > mu+ mu-\nlaunch\n')
        subprocess.call([pjoin(MG5DIR, 'MadSpin', 'madspin'), script], cwd=self.path,
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                        stdin=subprocess.DEVNULL)
        out = pjoin(self.path, 'octet_decayed.lhe.gz')
        self.assertTrue(os.path.exists(out))
        self.assertTrue(os.path.isdir(pjoin(self.path, 'madspin_me_unc')))
        events = lhe_parser.EventFile(out)
        cross = events.cross
        cos = []
        for event in events:
            self.assertFalse([p for p in event if abs(p.pdg) in (9943003, 9944003, 9945003)])
            psi = [p for p in event if p.pdg == 443]
            mu = [p for p in event if abs(p.pdg) == 13 and p.status == 1]
            self.assertEqual(len(psi), 1)
            self.assertEqual(psi[0].status, 2)
            self.assertEqual(len(mu), 2)
            pair = lhe_parser.FourMomentum(mu[0]) + lhe_parser.FourMomentum(mu[1])
            self.assertLess(abs(pair.mass - psi[0].mass), 1e-4)
            muon = [p for p in mu if p.pdg == 13][0]
            rest = lhe_parser.FourMomentum(muon).boost_to_restframe(
                lhe_parser.FourMomentum(psi[0]))
            cos.append(rest.pz / math.sqrt(rest.px**2 + rest.py**2 + rest.pz**2))
        self.assertEqual(len(cos), 3000)
        # 1 + a cos^2 from <cos^2> = (1/3 + a/5) / (1 + a/3)
        m2 = sum(c * c for c in cos) / len(cos)
        m4 = sum(c ** 4 for c in cos) / len(cos)
        err_m2 = math.sqrt((m4 - m2 * m2) / len(cos))
        a = (m2 - 1. / 3) / (1. / 5 - m2 / 3)
        err_a = err_m2 * (1. / 5 - 1. / 9) / (1. / 5 - m2 / 3) ** 2
        self.assertLess(abs(a - 0.5), 3 * err_a, (a, err_a))
        # BR of the J/psi: its LO partial width at its own onium_mass (3.1)
        alpha, M = 1 / 132.507, 3.1
        r2 = 1.16 * 4 * math.pi / 18
        br = 4 * (2. / 3) ** 2 * alpha ** 2 * r2 / M ** 2 / 9.26e-05
        self.assertLess(abs(cross / production / br - 1), 1e-3, (cross, production, br))

    def test_madspin_onium_radiative(self):
        """Radiative transitions of bound states (MadSpin/onia_radiative.py):
        q q~ > gamma* > psi(2S), transverse along the beam, decayed through
        psi(2S) > chi_cJ gamma (E1), chi_cJ > J/psi gamma (E1), J/psi > mu+
        mu-, for J = 1 and 2 (two decay lines with the same final state). The
        cross section is the production one times the summed products of the
        BR lines, every bound state is on its mass shell (those of the model
        for the daughters), the chain is in the event record, and the first
        photon goes as 1 + alpha_J cos^2(theta) in the psi(2S) rest frame,
        alpha_1 = -1/3, alpha_2 = 1/13. spinmode none is refused."""

        if not misc.default_f2py():
            raise unittest.SkipTest('no f2py')

        def madgraph(name, lines):
            script = pjoin(self.path, name)
            with open(script, 'w') as fsock:
                fsock.write('set automatic_html_opening False\n' + '\n'.join(lines) + '\n')
            subprocess.call([pjoin(MG5DIR, 'bin', 'madgraph'), script], cwd=self.path,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                            stdin=subprocess.DEVNULL)

        madgraph('output.mg', ['import model sm_onia-c_mass',
                               'generate p p > psi(2|3S11) QCD=0 / z h',
                               'output madevent %s -f -nojpeg' % self.run_dir])
        with open(pjoin(self.run_dir, 'Cards', 'param_card.dat'), 'a') as fsock:
            fsock.write('\nDECAY 100443 2.94e-04\n   9.75e-02  2  20443  22\n   9.36e-02  2  445  22\n'
                        'DECAY  20443 8.40e-04\n   3.43e-01  2  443  22\n'
                        'DECAY    445 1.94e-03\n   1.90e-01  2  443  22\n'
                        'DECAY    443 9.26e-05\n   5.961e-02  2  13  -13\n')
        madgraph('launch.mg', ['launch %s' % self.run_dir, 'set nevents 2000',
                               'set pdlabel nn23lo1', 'set iseed 51'])
        events = pjoin(self.run_dir, 'Events', 'run_01', 'unweighted_events.lhe.gz')
        production = lhe_parser.EventFile(events).cross
        # MadSpin unpacks the file it imports: keep the events aside and put the input back
        # before each run
        text = gzip.open(events, 'rt').read()

        def run(spinmode, events_text=text):
            for name in ('psi2s.lhe.gz', 'psi2s.lhe'):
                if os.path.exists(pjoin(self.path, name)):
                    os.remove(pjoin(self.path, name))
            with gzip.open(pjoin(self.path, 'psi2s.lhe.gz'), 'wt') as fsock:
                fsock.write(events_text)
            script = pjoin(self.path, 'ms_rad')
            with open(script, 'w') as fsock:
                fsock.write('import psi2s.lhe.gz\nset spinmode %s\nset seed 13\n'
                            'decay psi(2S) > chic1 a, chic1 > Jpsi a, Jpsi > mu+ mu-\n'
                            'decay psi(2S) > chic2 a, chic2 > Jpsi a, Jpsi > mu+ mu-\n'
                            'launch\n' % spinmode)
            for d in glob.glob(pjoin(self.path, 'madspin_*')) + glob.glob(pjoin(self.path, 'decay_*')):
                shutil.rmtree(d)
            subprocess.call([pjoin(MG5DIR, 'MadSpin', 'madspin'), script], cwd=self.path,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                            stdin=subprocess.DEVNULL)
            return pjoin(self.path, 'psi2s_decayed.lhe.gz')

        out = run('none')
        self.assertFalse(os.path.exists(out))

        # the same events without the DECAY blocks of the bound states: the widths and branching
        # fractions are the PDG 2026 values (madgraph/models/onia_decays.py)
        import madgraph.models.onia_decays as onia_decays
        defaults = onia_decays.PDG_2026
        stripped = re.sub(r'(?ims)^\s*DECAY\s+(100443|20443|445|443)\s.*?(?=^\s*(?:DECAY|BLOCK)\s|</slha>)',
                          '', text)
        slha = stripped.split('</slha>')[0].split('<slha>')[-1]
        self.assertIsNone(re.search(r'(?im)^\s*DECAY\s+(100443|20443|445|443)\s', slha))
        self.assertIsNotNone(re.search(r'(?im)^\s*DECAY\s+23\s', slha))
        out = run('madspin', stripped)
        self.assertTrue(os.path.exists(out))
        br_model = defaults[443][1][(-13, 13)] * (
            defaults[100443][1][(22, 20443)] * defaults[20443][1][(22, 443)]
            + defaults[100443][1][(22, 445)] * defaults[445][1][(22, 443)])
        self.assertLess(abs(lhe_parser.EventFile(out).cross / production / br_model - 1), 1e-6)
        os.rename(out, pjoin(self.path, 'psi2s_decayed_model.lhe.gz'))

        out = run('madspin')
        self.assertTrue(os.path.exists(out))
        events = lhe_parser.EventFile(out)
        br = 5.961e-02 * (9.75e-02 * 0.343 + 9.36e-02 * 0.19)
        self.assertLess(abs(events.cross / production / br - 1), 1e-6)
        masses = {100443: 3.69, 20443: 3.51, 445: 3.56, 443: 3.1}
        cos = {20443: [], 445: []}
        for event in events:
            states = dict((p.pdg, p) for p in event if p.pdg in masses)
            chi = [pdg for pdg in (20443, 445) if pdg in states]
            self.assertEqual(len(chi), 1)
            chi = chi[0]
            self.assertEqual(sorted(states), sorted([100443, chi, 443]))
            for pdg, p in states.items():
                self.assertEqual(p.status, 2)
                self.assertLess(abs(lhe_parser.FourMomentum(p).mass - masses[pdg]), 1e-3 * (1 + p.E))
            photons = [p for p in event if p.pdg == 22]
            self.assertEqual(len(photons), 2)
            first = [g for g in photons if g.mother1 is states[100443]]
            self.assertEqual(len(first), 1)
            self.assertEqual(len([g for g in photons if g.mother1 is states[chi]]), 1)
            self.assertEqual(len([p for p in event if abs(p.pdg) == 13 and p.mother1 is states[443]]), 2)
            rest = lhe_parser.FourMomentum(first[0]).boost_to_restframe(
                lhe_parser.FourMomentum(states[100443]))
            cos[chi].append(rest.pz / math.sqrt(rest.px ** 2 + rest.py ** 2 + rest.pz ** 2))
        self.assertEqual(len(cos[20443]) + len(cos[445]), 2000)
        # the two lines in the ratio of their BR products (binomial, 4 sigma)
        f = 9.75e-02 * 0.343 / (9.75e-02 * 0.343 + 9.36e-02 * 0.19)
        self.assertLess(abs(len(cos[20443]) / 2000. - f), 4 * math.sqrt(f * (1 - f) / 2000))
        # 1 + a cos^2: <cos^2> = (1/3 + a/5) / (1 + a/3)
        for pdg, a in ((20443, -1. / 3), (445, 1. / 13)):
            c = cos[pdg]
            m2 = sum(x * x for x in c) / len(c)
            m4 = sum(x ** 4 for x in c) / len(c)
            err = math.sqrt((m4 - m2 * m2) / len(c))
            self.assertLess(abs(m2 - (1. / 3 + a / 5) / (1 + a / 3)), 4 * err, (pdg, m2, err))

    def test_madspin_onium_decay_2to2(self):
        """MadSpin after a 2 -> 2 onium production, g g > J/psi g with a soft
        jet: every event is decayed (the momentum reshuffling of a strongly
        boosted event works at the precision of the event record), and the
        cross section is the production one times the BR line of the DECAY
        block, with no Breit-Wigner truncation (the onium decays on shell)."""

        if not misc.default_f2py():
            raise unittest.SkipTest('no f2py')

        def madgraph(name, lines):
            script = pjoin(self.path, name)
            with open(script, 'w') as fsock:
                fsock.write('set automatic_html_opening False\n' + '\n'.join(lines) + '\n')
            subprocess.call([pjoin(MG5DIR, 'bin', 'madgraph'), script], cwd=self.path,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                            stdin=subprocess.DEVNULL)

        madgraph('output.mg', ['import model sm_onia-c_mass',
                               'generate g g > Jpsi(1|3S11) g',
                               'output madevent %s -f -nojpeg' % self.run_dir])
        card = pjoin(self.run_dir, 'Cards', 'onia_card.dat')
        text = re.sub(r'(?m)^(\s*443\s+)[0-9.Ee+-]+(\s+# mass for)', r'\g<1>3.1\2', open(card).read())
        open(card, 'w').write(text)
        with open(pjoin(self.run_dir, 'Cards', 'param_card.dat'), 'a') as fsock:
            fsock.write('\nDECAY  443  9.26e-05\n   5.961e-02  2  13  -13\n')
        madgraph('launch.mg', ['launch %s' % self.run_dir, 'set nevents 2000', 'set ptj 5',
                               'set pdlabel nn23lo1', 'set iseed 41'])
        events = pjoin(self.run_dir, 'Events', 'run_01', 'unweighted_events.lhe.gz')
        production = lhe_parser.EventFile(events).cross
        shutil.copy(events, pjoin(self.path, 'psig.lhe.gz'))
        script = pjoin(self.path, 'ms_psig')
        with open(script, 'w') as fsock:
            fsock.write('import psig.lhe.gz\nset spinmode madspin\nset seed 11\n'
                        'decay Jpsi > mu+ mu-\nlaunch\n')
        subprocess.call([pjoin(MG5DIR, 'MadSpin', 'madspin'), script], cwd=self.path,
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                        stdin=subprocess.DEVNULL)
        out = pjoin(self.path, 'psig_decayed.lhe.gz')
        self.assertTrue(os.path.exists(out))
        events = lhe_parser.EventFile(out)
        self.assertLess(abs(events.cross / production / 5.961e-02 - 1), 1e-6)
        n = 0
        for event in events:
            n += 1
            psi = [p for p in event if p.pdg == 443]
            self.assertEqual(len(psi), 1)
            self.assertEqual(psi[0].status, 2)
            self.assertEqual(len([p for p in event if abs(p.pdg) == 13]), 2)
        self.assertEqual(n, 2000)

        # with onia_breit_wigner the J/psi mass follows its Breit-Wigner within BW_cut, and the
        # cross section keeps the share of the Breit-Wigner inside the window
        os.rename(out, pjoin(self.path, 'psig_decayed_onshell.lhe.gz'))
        for d in glob.glob(pjoin(self.path, 'madspin_*')) + glob.glob(pjoin(self.path, 'decay_*')):
            shutil.rmtree(d)
        with open(script, 'w') as fsock:
            fsock.write('import psig.lhe.gz\nset spinmode madspin\nset seed 11\n'
                        'set onia_breit_wigner True\nset BW_cut 15\n'
                        'decay Jpsi > mu+ mu-\nlaunch\n')
        subprocess.call([pjoin(MG5DIR, 'MadSpin', 'madspin'), script], cwd=self.path,
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                        stdin=subprocess.DEVNULL)
        self.assertTrue(os.path.exists(out))
        import MadSpin.decay as madspin_decay
        keep = madspin_decay.bw_retained_fraction(3.1, 9.26e-05, 15)
        events = lhe_parser.EventFile(out)
        self.assertLess(abs(events.cross / production / 5.961e-02 / keep - 1), 1e-6)
        offsets = []
        for event in events:
            psi = [p for p in event if p.pdg == 443][0]
            mu = [p for p in event if abs(p.pdg) == 13]
            pair = lhe_parser.FourMomentum(mu[0]) + lhe_parser.FourMomentum(mu[1])
            if psi.E < 50:      # away from the rounding of very boosted J/psi
                offsets.append(pair.mass - 3.1)
        self.assertTrue(offsets)
        self.assertLess(max(abs(o) for o in offsets), 15 * 9.26e-05 * 1.001)
        # half of a Breit-Wigner lies within one half-width of the pole
        inside = sum(1 for o in offsets if abs(o) < 9.26e-05 / 2) / len(offsets)
        self.assertLess(abs(inside - 0.5 / keep), 0.08)

    def _get_single_decayed_lhe_path(self):
        decayed_events = glob.glob(pjoin(self.run_dir, 'Events', '*decayed*', '*.lhe.gz'))
        self.assertTrue(decayed_events)
        return decayed_events[0]


    def test_hepmc_decay(self):
        """ """
        
        cwd = os.getcwd()
        
        files.cp(pjoin(MG5DIR, 'tests', 'input_files', 'test.hepmc.gz'), self.path)


        fsock = open(pjoin(self.path, 'test_hepmc'),'w')
        text = """
        set spinmode none
        set cross_section {0:1.0}
        set new_wgt BR
        set input_format hepmc
        import ./test.hepmc.gz
        import model %s/tests/input_files/DM_pion %s/tests/input_files/DM_pion/param_pion.dat
        decay k0 > xr xr a
        launch
        """ % (MG5DIR, MG5DIR)
        
        fsock.write(text)
        fsock.close()

        import subprocess
        if logging.getLogger('madgraph').level <= 20:
            stdout=None
            stderr=None
        else:
            devnull =open(os.devnull,'w')
            stdout=devnull
            stderr=devnull

        subprocess.call([pjoin(MG5DIR, 'MadSpin', 'madspin'),
                         pjoin(self.path, 'test_hepmc')],
                        cwd=pjoin(self.path),
                        stdout=stdout,stderr=stderr)
        self.assertTrue(os.path.exists(pjoin(self.path, 'test_decayed.lhe.gz')))
        lhe = lhe_parser.EventFile(pjoin(self.path, 'test_decayed.lhe.gz'))
        self.assertEqual(10, len(lhe))
        
        nb_dec = 0
        nb_photon = 0
        for event in lhe:
            self.assertEqual(event.nexternal, len(event))
            for particle in event:
                if particle.pdg == 130:
                    self.assertEqual(particle.status,2)
                    nb_dec +=1
                if particle.pdg ==22:
                    nb_photon += 1
                    
        self.assertEqual(nb_dec, 116)
        self.assertEqual(nb_photon, 116)

    def test_lhe_none_decay(self):
        """ """
        
        cwd = os.getcwd()
        
        files.cp(pjoin(MG5DIR, 'tests', 'input_files', 'test_spinmode_none.lhe.gz'), self.path)


        fsock = open(pjoin(self.path, 'test_hepmc'),'w')
        text = """
        set spinmode none
        import ./test_spinmode_none.lhe.gz
        decay z > mu+ mu-
        launch
        """
        
        fsock.write(text)
        fsock.close()

        import subprocess
        if logging.getLogger('madgraph').level <= 20:
            stdout=None
            stderr=None
        else:
            devnull =open(os.devnull,'w')
            stdout=devnull
            stderr=devnull

        subprocess.call([pjoin(MG5DIR, 'MadSpin', 'madspin'),
                         pjoin(self.path, 'test_hepmc')],
                        cwd=pjoin(self.path),
                        stdout=stdout,stderr=stderr)

        self.assertTrue(os.path.exists(pjoin(self.path, 'test_spinmode_none_decayed.lhe.gz')))
        # the input is decompressed to be read: it has to be compressed back
        self.assertTrue(os.path.exists(pjoin(self.path, 'test_spinmode_none.lhe.gz')))
        self.assertFalse(os.path.exists(pjoin(self.path, 'test_spinmode_none.lhe')))
        lhe = lhe_parser.EventFile(pjoin(self.path, 'test_spinmode_none_decayed.lhe.gz'))
        self.assertEqual(100, len(lhe))
        
        nb_dec = 0
        nb_muon = 0
        for event in lhe:
            muon_in = 0
            self.assertEqual(event.nexternal, len(event))
            for particle in event:
                if particle.pdg == 23:
                    self.assertEqual(particle.status,2)
                    nb_dec += 1
                if particle.pdg == 13:
                    nb_muon += 1
                    muon_in +=1
            self.assertEqual(muon_in, 1)
        self.assertEqual(nb_dec, 189)
        self.assertEqual(nb_muon, 100)

    def test_madspin_spin_only(self):
        """ """
        
        cwd = os.getcwd()
        
        files.cp(pjoin(MG5DIR, 'tests', 'input_files', 'test_spinmode_none.lhe.gz'), self.path)


        fsock = open(pjoin(self.path, 'test_hepmc'),'w')
        text = """
        import ./test_spinmode_none.lhe.gz
        set onlyhelicity True
        launch
        """
        
        fsock.write(text)
        fsock.close()

        import subprocess
        if logging.getLogger('madgraph').level <= 20:
            stdout=None
            stderr=None
        else:
            devnull =open(os.devnull,'w')
            stdout=devnull
            stderr=devnull

        subprocess.call([pjoin(MG5DIR, 'MadSpin', 'madspin'),
                         pjoin(self.path, 'test_hepmc')],
                        cwd=pjoin(self.path),
                        stdout=stdout,stderr=stderr)

        self.assertTrue(os.path.exists(pjoin(self.path, 'test_spinmode_none_decayed.lhe.gz')))
        lhe = lhe_parser.EventFile(pjoin(self.path, 'test_spinmode_none_decayed.lhe.gz'))
        self.assertEqual(100, len(lhe))
        
        nb_dec = 0
        nb_notdec = 0 
        nb_muon = 0
        pol = {0:0, -1:0,1:0, 9:9}
        for event in lhe:
            muon_in = 0
            self.assertEqual(event.nexternal, len(event))
            for particle in event:
                if particle.status == 2:
                    self.assertEqual(particle.helicity, 9)
                    nb_dec +=1
                    continue
                if particle.pdg == 23:
                    if particle.status == 1:
                        nb_notdec += 1    
                    else: 
                        nb_dec += 1
                if particle.pdg == 13:
                    nb_muon += 1
                    muon_in +=1
                self.assertIn(int(particle.helicity), pol)
                pol[int(particle.helicity)] +=1
            self.assertEqual(muon_in, 0)
        
        self.assertEqual(nb_notdec, 100)
        self.assertEqual(nb_dec, 89)
        self.assertEqual(nb_muon, 0)
        import math
        self.assertLess(abs(pol[1]-pol[-1]), 2 * math.sqrt(pol[1]))
        self.assertLess(pol[0], pol[-1])

    def test_madspin_mixed_flavor_decay_log_summary(self):
        """Check MadSpin log summary for mixed lepton/quark decay definition."""

        cmd_path = pjoin(self.path, 'test_madspin_mixed_flavor.cmd')
        log_path = pjoin(self.path, 'test_madspin_mixed_flavor.log')
        command = """import model sm
set automatic_html_opening False --no_save
set notification_center False --no_save
define l+ = e+ mu+ u d~
define l- = e- mu- u~ d
generate u u~ > z g
output madevent %(path)s
launch
madspin=ON
shower=OFF
analysis=OFF
set nevents 10000
set iseed 1
add madspin_card --before_line="launch" set offshell_min_recoil 0.5
decay w+ > j j
decay w- > j j
decay z > l+ l-
""" % {'path': self.run_dir}
        with open(cmd_path, 'w') as fsock:
            fsock.write(command)

        with open(log_path, 'w') as log_file:
            return_code = subprocess.call(
                [sys.executable, pjoin(_file_path, os.path.pardir, 'bin', 'madgraph'), cmd_path],
                cwd=pjoin(_file_path, os.path.pardir),
                stdout=log_file, stderr=subprocess.STDOUT)
        self.assertEqual(return_code, 0)

        with open(log_path) as log_file:
            log = log_file.read()
            misc.sprint(log)

        # MadSpin's default spinmode is "madspin" (offshell), which routes
        # through the density code path. The density path emits a single
        # summary line of the form:
        #   CRITICAL: MadSpin unweight efficiency: 0.1000
        #            (10000 written / 99958 trials, 10.00 trials/event)
        # rather than the legacy multi-line onshell summary
        # ("Total number of events written", "Average number of trial
        # points per production event", "Branching ratio to allowed
        # decays", ...).
        #
        # Parse the metrics that are still meaningful here individually and
        # leniently, so the test does not break on minor wording/spacing
        # changes of that line:
        #   - the "unweight efficiency" marker must be present (MadSpin ran),
        #   - the number of written events (deterministic: == nevents),
        #   - the trials/event ratio (sampling dependent -> sanity range only).
        self.assertIsNotNone(
            re.search(r'MadSpin\s+unweight\s+efficiency', log),
            msg='MadSpin density-mode summary line not found in log')

        written = re.search(r'([0-9]+)\s+written\b', log)
        self.assertIsNotNone(written,
            msg='MadSpin written-event count not found in log')
        n_written = int(written.group(1))
        self.assertEqual(n_written, 10000,
            msg='Expected 10000 written events, got %d' % n_written)

        # trials/event is sampling dependent: 9.6-12.6 over eight production
        # samples of this card (spinmode madspin; PA takes ~2.8, the legacy
        # onshell path ~4.98). It used to depend strongly on the production
        # sample -- with madevent decays iseed 1-4 gave 4.4-15 (34 on CI), and
        # 4.6-152 before the refine normalisation changed: the offshell
        # max-weight scan reused one draw's production density for all the
        # draws of an event, and the weight itself was unbounded -- a Z drawn
        # up to sqrt(shat) pushes the gluon soft, w ~ 1/chi -- unless
        # offshell_min_recoil cuts it, which this card opts in to (the option
        # is off by default: it biases the off-shell tail). Only sanity-check it is present and in a
        # physically reasonable range rather than pinning the exact value: the
        # bound only catches a broken unweighting.
        trials = re.search(r'([0-9]+(?:\.[0-9]+)?)\s*trials\s*/\s*event', log)
        if trials is not None:
            trials_per_event = float(trials.group(1))
            self.assertGreater(trials_per_event, 1.0)
            self.assertLess(trials_per_event, 500.0)

        # The legacy onshell-mode summary lines (Branching ratio to allowed
        # decays / Number of events with weights larger than max_weight /
        # Number of subprocesses / Number of failures when restoring the
        # Monte Carlo masses) are not emitted by the density code path,
        # so those assertions are intentionally not ported here.

        with gzip.open(self._get_single_decayed_lhe_path(), 'rt') as lhe_file:
            banner_lines = []
            for line in lhe_file:
                if '<event' in line:
                    break
                banner_lines.append(line)
        banner_text = ''.join(banner_lines)
        self.assertNotRegex(banner_text, r'(?mi)^\s*81\s+[0-9eE.+-]+\s+# added\s*$')
        self.assertNotRegex(banner_text, r'(?mi)^\s*82\s+[0-9eE.+-]+\s+# added\s*$')
        self.assertNotRegex(banner_text, r'(?mi)^\s*83\s+[0-9eE.+-]+\s+# added\s*$')
        self.assertNotRegex(banner_text, r'(?mi)^\s*decay\s+81\s+[0-9eE.+-]+\s+# added\s*$')
        self.assertNotRegex(banner_text, r'(?mi)^\s*decay\s+82\s+[0-9eE.+-]+\s+# added\s*$')
        self.assertNotRegex(banner_text, r'(?mi)^\s*decay\s+83\s+[0-9eE.+-]+\s+# added\s*$')

    def test_madspin_mixed_flavor_decay_log_summary_mg7(self):
        """Same check as test_madspin_mixed_flavor_decay_log_summary but with the
        default 'mg7' (madspace) exporter instead of Fortran madevent: drive the
        whole chain through the madevent-style launch command interface
        (output mg7 -> launch -> madspin=ON + decay lines) and verify MadSpin's
        density path runs on the mg7 events and emits its
        'MadSpin unweight efficiency: ...' summary line.

        NB: 'set nevents 500' relies on the mg7 run_card editor accepting the
        legacy madevent name (mapped to generation.events); without it the run
        would generate the full default 100k events and hit the timeout below.
        """
        cmd_path = pjoin(self.path, 'test_madspin_mixed_flavor_mg7.cmd')
        log_path = pjoin(self.path, 'test_madspin_mixed_flavor_mg7.log')
        command = """import model sm
set automatic_html_opening False --no_save
set notification_center False --no_save
define l+ = e+ mu+ u d~
define l- = e- mu- u~ d
generate u u~ > z g
output mg7 %(path)s
launch
madspin=ON
shower=OFF
analysis=OFF
set nevents 500
set iseed 1
decay w+ > j j
decay w- > j j
decay z > l+ l-
""" % {'path': self.run_dir}
        with open(cmd_path, 'w') as fsock:
            fsock.write(command)

        # Bounded and with stdin closed so the run never blocks on a prompt.
        # With 'set nevents 500' honoured the whole run takes 30-110 s on a CI
        # runner, which varies a lot from one run to the next; a timeout of
        # 600 s still catches the regression it is there for (nevents no longer
        # applied -> the full 100k events get decayed) without failing on a
        # slow runner, as 240 s did.
        timeout = 600
        def log_tail(why):
            # the mg7 run log lives in a tmp dir CI does not upload: show its
            # tail in the CI output, so that the reason is visible there
            with open(log_path) as log_file:
                lines = log_file.read().splitlines()
            print('\n===== test_madspin_mixed_flavor_decay_log_summary_mg7: '
                  'madgraph log tail (%s) =====\n%s' % (why, '\n'.join(lines[-150:])),
                  file=sys.stderr, flush=True)

        with open(log_path, 'w') as log_file:
            try:
                return_code = subprocess.call(
                    [sys.executable, pjoin(_file_path, os.path.pardir, 'bin', 'madgraph'), cmd_path],
                    cwd=pjoin(_file_path, os.path.pardir),
                    stdin=subprocess.DEVNULL,
                    stdout=log_file, stderr=subprocess.STDOUT, timeout=timeout)
            except subprocess.TimeoutExpired:
                log_file.flush()
                log_tail('timed out after %d s' % timeout)
                self.fail('mg7 + MadSpin run timed out (is set nevents applied?)')
        with open(log_path) as log_file:
            log = log_file.read()
        if return_code != 0 or not re.search(r'MadSpin\s+unweight\s+efficiency', log):
            log_tail('rc=%s' % return_code)
        self.assertEqual(return_code, 0)

        self.assertIsNotNone(
            re.search(r'MadSpin\s+unweight\s+efficiency', log),
            msg='mg7 + MadSpin: density-mode summary line not found in log')
        # store_result used to save the madevent results database to
        # HTML/results.pkl, which the mg7 output does not have
        self.assertEqual(
            [line for line in log.splitlines() if 'I/O error' in line], [])
        self.assertFalse(os.path.exists(pjoin(self.run_dir, 'crossx.html')))
        # the madmatrix exporter used to print its keyword arguments
        self.assertNotIn('kwargs[', log)

    def test_madspin_wplus_all_all_flavor_balance(self):
        """`w+ > all all` should populate e/mu decay modes with similar rates."""

        cmd_path = pjoin(self.path, 'test_madspin_wplus_all_all.cmd')
        log_path = pjoin(self.path, 'test_madspin_wplus_all_all.log')
        command = """import model sm
set automatic_html_opening False --no_save
set notification_center False --no_save
generate p p > w+ g
output madevent %(path)s
launch
madspin=ON
shower=OFF
analysis=OFF
set nevents 2000
set iseed 1
decay w+ > all all
""" % {'path': self.run_dir}
        with open(cmd_path, 'w') as fsock:
            fsock.write(command)

        with open(log_path, 'w') as log_file:
            return_code = subprocess.call(
                [sys.executable, pjoin(_file_path, os.path.pardir, 'bin', 'madgraph'), cmd_path],
                cwd=pjoin(_file_path, os.path.pardir),
                stdout=log_file, stderr=subprocess.STDOUT)
        self.assertEqual(return_code, 0)

        counts={}
        for event in lhe_parser.EventFile(self._get_single_decayed_lhe_path()):
            for particle in event:
                if particle.status == 1:
                    if particle.pdg in counts:
                        counts[particle.pdg] += 1
                    else:
                        counts[particle.pdg] = 1

        misc.sprint(counts)
        self.assertNotIn(81, counts)
        self.assertNotIn(82, counts)
        self.assertNotIn(83, counts)
        self.assertNotIn(-81, counts)
        self.assertNotIn(-82, counts)
        self.assertNotIn(-83, counts)    
        self.assertGreater(counts[-11], 0)
        self.assertEqual(counts[-11],counts[12])
        self.assertGreater(counts[-13], 0)
        self.assertEqual(counts[-13],counts[14])
        self.assertGreater(counts[-15], 0) 
        self.assertEqual(counts[-15],counts[16])
        self.assertGreater(counts[4], 0)
        self.assertEqual(counts[4], counts[-3])
        self.assertGreater(counts[2], 0)
        self.assertEqual(counts[2], counts[-1])


        lepton_total = counts[-11] + counts[-13]
        self.assertLess(abs(counts[-11] - counts[-13]), 4* math.sqrt(counts[-11]),
            msg='Expected electron/muon counts to be comparable, got %s' % counts)
        self.assertLess(abs(counts[2] - counts[4]), 4* math.sqrt(counts[4]),
            msg='Expected electron/muon counts to be comparable, got %s' % counts)
                
    def test_madspin_wplus_all_all_flavor_balance_2to1(self):
        """`w+ > all all` should populate e/mu decay modes with similar rates."""

        cmd_path = pjoin(self.path, 'test_madspin_wplus_all_all.cmd')
        log_path = pjoin(self.path, 'test_madspin_wplus_all_all.log')
        command = """import model sm
set automatic_html_opening False --no_save
set notification_center False --no_save
generate p p > w+
output madevent %(path)s
launch
madspin=ON
shower=OFF
analysis=OFF
set nevents 2000
set iseed 1
decay w+ > all all
""" % {'path': self.run_dir}
        with open(cmd_path, 'w') as fsock:
            fsock.write(command)

        with open(log_path, 'w') as log_file:
            return_code = subprocess.call(
                [sys.executable, pjoin(_file_path, os.path.pardir, 'bin', 'madgraph'), cmd_path],
                cwd=pjoin(_file_path, os.path.pardir),
                stdout=log_file, stderr=subprocess.STDOUT)
        self.assertEqual(return_code, 0)

        counts={}
        for event in lhe_parser.EventFile(self._get_single_decayed_lhe_path()):
            for particle in event:
                if particle.status == 1:
                    if particle.pdg in counts:
                        counts[particle.pdg] += 1
                    else:
                        counts[particle.pdg] = 1

        misc.sprint(counts)
        self.assertNotIn(81, counts)
        self.assertNotIn(82, counts)
        self.assertNotIn(83, counts)
        self.assertNotIn(-81, counts)
        self.assertNotIn(-82, counts)
        self.assertNotIn(-83, counts)    
        self.assertGreater(counts[-11], 0)
        self.assertEqual(counts[-11],counts[12])
        self.assertGreater(counts[-13], 0)
        self.assertEqual(counts[-13],counts[14])
        self.assertGreater(counts[-15], 0) 
        self.assertEqual(counts[-15],counts[16])
        self.assertGreater(counts[4], 0)
        self.assertEqual(counts[4], counts[-3])
        self.assertGreater(counts[2], 0)
        self.assertEqual(counts[2], counts[-1])


        lepton_total = counts[-11] + counts[-13]
        self.assertLess(abs(counts[-11] - counts[-13]), 4* math.sqrt(counts[-11]),
            msg='Expected electron/muon counts to be comparable, got %s' % counts)
        self.assertLess(abs(counts[2] - counts[4]), 4* math.sqrt(counts[4]),
            msg='Expected electron/muon counts to be comparable, got %s' % counts)
               

    def _run_one_mode(self, mode, particle_to_decay, name_input_file, name_scipt_file):
        cwd = os.getcwd()
        index = name_input_file.find(".lhe")
        name_file_decayed = name_input_file[:index] + "_decayed" + name_input_file[index:]
        path_input_file = pjoin(MG5DIR, 'tests', 'input_files', 'madspin', name_input_file)

        files.cp(path_input_file, self.path)

        fsock = open(pjoin(self.path, name_scipt_file),'w')
        if 'loop_induced' in name_input_file:
            text = f"""
            import {pjoin(self.path, name_input_file)}
            set spinmode {mode}
            decay w+ > all all
            decay w- > all all
            launch
            """
        else:
            text = f"""
            import {pjoin(self.path, name_input_file)}
            set spinmode {mode}
            decay t > w+ b, w+ > all all
            decay t~ > w- b~, w- > all all
            launch
            """
        fsock.write(text)
        fsock.close()

        import subprocess
        if logging.getLogger('madgraph').level <= 20:
            stdout=None
            stderr=None
        else:
            devnull =open(os.devnull,'w')
            stdout=devnull
            stderr=devnull

        returncode = subprocess.call([pjoin(MG5DIR, 'MadSpin', 'madspin'),
                        pjoin(self.path, name_scipt_file)],
                        cwd=pjoin(self.path),
                        stdout=stdout,stderr=stderr)

        # check the exit status first: a crashed MadSpin can still leave partial
        # output behind, and then the assertions below report a confusing
        # symptom instead of the actual failure.
        self.assertEqual(returncode, 0,
                         'MadSpin exited with %s for %s' % (returncode, name_scipt_file))
        self.assertTrue(os.path.exists(pjoin(self.path, name_file_decayed)),
                        'no decayed file produced for %s' % name_scipt_file)
        
        
        lhe = lhe_parser.EventFile(pjoin(self.path, name_file_decayed))
        if mode in ['full', 'madspin'] and 'loop_induced' in name_input_file:
            self.assertEqual(5, len(lhe))
        else:
            self.assertEqual(100, len(lhe))
        
        # we cannot check if the momenta are identical by fixing the seed (it could change whenever we change the code somewhere)
        # we just check that there 8 particles in each event and that the w+- have a status of 2.
        for event in lhe:
            self.assertEqual(event.nexternal, len(event))
            for particle in event:
                if particle.pid in particle_to_decay:
                    self.assertEqual(particle.status, 2,
                                     'pdg %s not marked as decayed for %s'
                                     % (particle.pid, name_scipt_file))

    def test_madspin_loop_induced(self):
        """ Tests that the different modes of MadSpin work for loop-induced processes.
            It checks that there is no crash and that the decayed particles have a status of 2.
        """ 

        self._run_one_mode("PA", [24, -24], 'test_madspin_loop_induced_PA.lhe.gz', 'test_loop_induced_PA')
        self._run_one_mode("full", [24, -24], 'test_madspin_loop_induced_full.lhe.gz', 'test_loop_induced_full')
        self._run_one_mode("onshell", [24, -24], 'test_madspin_loop_induced_onshell.lhe.gz', 'test_loop_induced_onshell')
        self._run_one_mode("madspin", [24, -24], 'test_madspin_loop_induced_madspin.lhe.gz', 'test_loop_induced_madspin')

    def test_madspin_tree_level(self):
        """ Tests that the different modes of MadSpin work for tree-level processes.
            It checks that there is no crash and that the decayed particles have a status of 2.
        """ 

        self._run_one_mode("PA", [6, -6], 'test_madspin_tree_level.lhe.gz', 'test_tree_level_PA')
        self._run_one_mode("full", [6, -6], 'test_madspin_tree_level.lhe.gz', 'test_tree_level_full')
        self._run_one_mode("onshell", [6, -6], 'test_madspin_tree_level.lhe.gz', 'test_tree_level_onshell')
        self._run_one_mode("madspin", [6, -6], 'test_madspin_tree_level.lhe.gz', 'test_tree_level_madspin')
        self._run_one_mode("none", [6, -6], 'test_madspin_tree_level.lhe.gz', 'test_tree_level_none')
        self._run_one_mode("madspin_v1", [6, -6], 'test_madspin_tree_level.lhe.gz', 'test_tree_level_madspin_v1')
        self._run_one_mode("onshell_v1", [6, -6], 'test_madspin_tree_level.lhe.gz', 'test_tree_onshell_v1')

    @staticmethod
    def _decayed_sample(path):
        """The <init> block (sigma*BR) and the event blocks of an LHE file:
        everything that has to come out the same, without the banner."""
        with gzip.open(path, 'rt') as fsock:
            text = fsock.read()
        init = re.search(r'<init>.*?</init>', text, re.S)
        return (init.group(0) if init else None,
                re.findall(r'<event>.*?</event>', text, re.S))

    def test_madspin_mg7_decays_reproducible_with_seed(self):
        """The same card, seed and input give the same decayed sample with the
        default decay_generator mg7.

        Its decay pools used to run with mg7's run.seed = -1 -- a new random
        seed on every run -- so two runs of this card differed from the first
        event on (sigma*BR and the unweighting efficiency too), in every spin
        mode, while decay_generator madevent was reproducible. Both paths are
        covered: the density modes (one forked generation per decaying
        particle) and spinmode none (serial generation).
        """
        import MadSpin.interface_madspin as interface_madspin
        if not interface_madspin.MadSpinInterface._mg7_available():
            self.skipTest('madspace is not built: MadSpin would generate the '
                          'decays with madevent, which is not what is tested')

        for spinmode in ('madspin', 'none'):
            samples = []
            for tag in ('a', 'b'):
                run = pjoin(self.path, '%s_%s' % (spinmode, tag))
                os.mkdir(run)
                files.cp(pjoin(MG5DIR, 'tests', 'input_files', 'ttbar.lhe.gz'),
                         pjoin(run, 'events.lhe.gz'))
                card = pjoin(run, 'madspin_card.dat')
                with open(card, 'w') as fsock:
                    fsock.write('import events.lhe.gz\n'
                                'set spinmode %s\n'
                                'set decay_generator mg7\n'
                                'set seed 77\n'
                                'decay t > w+ b, w+ > e+ ve\n'
                                'decay t~ > w- b~, w- > mu- vm~\n'
                                'launch\n' % spinmode)
                log_path = pjoin(run, 'madspin.log')
                with open(log_path, 'w') as log:
                    returncode = subprocess.call(
                        [sys.executable, pjoin(MG5DIR, 'MadSpin', 'madspin'),
                         card],
                        cwd=run, stdin=subprocess.DEVNULL,
                        stdout=log, stderr=subprocess.STDOUT, timeout=900)
                decayed = pjoin(run, 'events_decayed.lhe.gz')
                if returncode or not os.path.exists(decayed):
                    with open(log_path) as log:
                        print('\n===== %s %s: MadSpin log tail (rc=%s) =====\n%s'
                              % (spinmode, tag, returncode,
                                 '\n'.join(log.read().splitlines()[-100:])),
                              file=sys.stderr, flush=True)
                self.assertEqual(returncode, 0)
                self.assertTrue(os.path.exists(decayed))

                # the mg7 generator really made the pools, each with a fixed
                # seed of its own
                cards = glob.glob(pjoin(run, 'decay_*', 'Cards',
                                        'run_card.toml'))
                self.assertEqual(len(cards), 2, 'no mg7 decay pool in %s' % run)
                seeds = [banner.RunCardMG7(path)['run']['seed']
                         for path in cards]
                self.assertTrue(all(seed > 0 for seed in seeds), seeds)
                self.assertEqual(len(set(seeds)), 2, seeds)

                samples.append(self._decayed_sample(decayed))

            init_a, events_a = samples[0]
            init_b, events_b = samples[1]
            self.assertEqual(len(events_a), 25)
            self.assertEqual(init_a, init_b,
                             'spinmode %s: sigma*BR differs between the two '
                             'runs' % spinmode)
            for i, (event_a, event_b) in enumerate(zip(events_a, events_b)):
                self.assertEqual(event_a, event_b,
                                 'spinmode %s: event %s differs between the '
                                 'two runs' % (spinmode, i))
            self.assertEqual(len(events_a), len(events_b))
