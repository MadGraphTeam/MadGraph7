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
"""FKS generation for processes with bound states (madgraph/fks/fks_onia.py,
arXiv:2402.19221): category checks, constituent pairs, soft partners and the
translation to physical numbering."""

import os
import unittest

import madgraph.fks.fks_onia as fks_onia
import madgraph.interface.master_interface as MGCmd
import madgraph.various.misc as misc

pjoin = os.path.join


def _sm_onia_available():
    import madgraph
    return os.path.isdir(pjoin(madgraph.MG5DIR, 'models', 'sm_onia'))


@unittest.skipUnless(_sm_onia_available(), 'model sm_onia not installed')
class TestFKSOniaGeneration(unittest.TestCase):
    """Generation of NLO processes with bound states (no output)."""

    @classmethod
    def setUpClass(cls):
        cls.cmd = MGCmd.MasterCmd()
        cls.cmd.no_notification()
        cls.cmd.exec_cmd('set automatic_html_opening False --no_save')
        cls.cmd.exec_cmd('import model sm_onia-c_mass')

    def generate(self, line):
        self.cmd.exec_cmd('generate %s' % line)
        return self.cmd._fks_multi_proc['born_processes']

    def assertRefused(self, line, text):
        with self.assertRaises(Exception) as ctx:
            self.cmd.exec_cmd('generate %s' % line)
        self.assertIn(text, str(ctx.exception))

    def test_refusals(self):
        """Outside the category of arXiv:2402.19221: clear messages."""
        self.assertRefused('g g > chic0(1|3P01) chic2(1|3P21) [QCD]',
                           'more than one P-wave quarkonium')
        self.assertRefused('g g > Jpsi(1|3S18) g [QED]', 'QCD corrections only')

    def test_massless_constituents_refused(self):
        cmd = MGCmd.MasterCmd()
        cmd.no_notification()
        cmd.exec_cmd('set automatic_html_opening False --no_save')
        cmd.exec_cmd('import model sm_onia')            # massless charm
        with self.assertRaises(Exception) as ctx:
            cmd.exec_cmd('generate g g > chic0(1|3P01) g [real=QCD]')
        self.assertIn('massive constituents', str(ctx.exception))

    def check_reals(self, born):
        """Constituents adjacent, never FKS partons, and only the onium (its
        first constituent) as a soft partner."""
        model = born.born_amp['process']['model']
        for real in born.real_amps:
            legs = real.process['legs']
            pairs = fks_onia.onium_pairs(legs)          # raises if not adjacent
            constituents = set(n + 1 for pair in pairs for n in pair)
            seconds = set(b + 1 for _, b in pairs)
            for info in real.fks_infos:
                self.assertNotIn(info['i'], constituents)
                self.assertNotIn(info['j'], constituents)
            for i, js in real.fks_j_from_i.items():
                self.assertFalse(seconds & set(js))
                self.assertFalse(i in constituents and js)
                for a, b in pairs:
                    first = legs[a]
                    if a + 1 in js:
                        self.assertTrue(fks_onia.onium_is_soft_partner(first, model))

    def test_octet_s_wave_soft_partner(self):
        born, = self.generate('g g > Jpsi(1|3S18) g [real=QCD]')
        self.check_reals(born)
        gluon = [r for r in born.real_amps
                 if [l['id'] for l in r.process['legs']] == [21, 21, 4, -4, 21, 21]][0]
        # the octet onium (first constituent, leg 3) is a soft partner of both gluons
        self.assertIn(3, gluon.fks_j_from_i[5])
        self.assertIn(3, gluon.fks_j_from_i[6])
        self.assertNotIn(4, gluon.fks_j_from_i[5])

    def test_two_onia_and_massive_partners(self):
        born, = self.generate('g g > Jpsi(1|3S11) Jpsi(1|3S18) [real=QCD]')
        self.check_reals(born)
        real = [r for r in born.real_amps if r.process['legs'][-1]['id'] == 21][0]
        # the singlet S wave (3, 4) is not a partner, the octet (5, 6) is
        self.assertEqual(sorted(real.fks_j_from_i[7]), [1, 2, 5])
        born, = self.generate('b g > chic0(1|3P01) b [real=QCD]')
        self.check_reals(born)
        real = [r for r in born.real_amps if [l['id'] for l in r.process['legs']]
                == [5, 21, 5, 4, -4, 21]][0]
        self.assertEqual(sorted(real.fks_j_from_i[6]), [1, 2, 3, 4])
        self.assertEqual(sorted((i['i'], i['j']) for i in real.fks_infos),
                         [(6, 1), (6, 2), (6, 3)])

    def test_zero_born_p_wave_kept(self):
        """q q~ > chi_cJ vanishes at LO, but its real emission is soft singular
        (P[1] -> 3S1[8]) and is kept with the onium as a soft partner."""
        born, = self.generate('u u~ > chic0(1|3P01) [real=QCD]')
        self.check_reals(born)
        real = [r for r in born.real_amps if r.process['legs'][-1]['id'] == 21][0]
        self.assertEqual(sorted(real.fks_j_from_i[5]), [1, 2, 3])

    def test_leptonium_no_partner(self):
        born, = self.generate('u u~ > tata(1|3S1) g [real=QCD]')
        self.check_reals(born)
        for real in born.real_amps:
            for js in real.fks_j_from_i.values():
                self.assertNotIn(3, js)

    def test_fock_label_keeps_nlo(self):
        """Every Fock state of a label is generated at NLO, not only the last."""
        borns = self.generate('g g > Jpsi [QCD]')
        names = set(l['onium']['name'] for b in borns for l in b.born_amp['process']['legs']
                    if l['onium'])
        self.assertGreater(len(names), 1)
        for b in borns:
            self.assertTrue(b.born_amp['process']['perturbation_couplings'])

    def test_physical_translation(self):
        born, = self.generate('g g > Jpsi(1|3S18) g [real=QCD]')
        real = [r for r in born.real_amps
                if [l['id'] for l in r.process['legs']] == [21, 21, 4, -4, 21, 21]][0]
        model = real.process['model']
        infos, jfromi, colors, tags, charges = fks_onia.to_physical_fks(
            real.process['legs'], model, real.fks_infos, real.fks_j_from_i,
            real.colors, real.particle_tags, real.charges)
        self.assertEqual(colors, [8, 8, 8, 8, 8])
        self.assertEqual(len(tags), 5)
        self.assertEqual(sorted((i['i'], i['j']) for i in infos), [(5, 1), (5, 2), (5, 4)])
        self.assertEqual(sorted(jfromi[5]), [1, 2, 3, 4])
        self.assertEqual(fks_onia.physical_pdgs(real.process['legs']), [21, 21, 9940003, 21, 21])


class TestFKSOniaHelpers(unittest.TestCase):
    """The pure helpers of fks_onia."""

    def legs(self, spec):
        import madgraph.core.base_objects as MG
        out = MG.LegList()
        for n, (pid, onium) in enumerate(spec, 1):
            out.append(MG.Leg({'id': pid, 'number': n, 'state': n > 2, 'onium': onium}))
        return out

    def test_pairs_and_numbers(self):
        base = {'name': 'x', 'N': 1, 'S': 1, 'L': 0, 'J': 1, 'charge': 0., 'mass': 3.1, 'ldme': 1.}
        o1 = dict(base, index=0, id=443, C=1)
        o2 = dict(base, index=1, id=9940003, C=8)
        legs = self.legs([(21, {}), (21, {}), (4, o1), (-4, o1), (4, o2), (-4, o2), (21, {})])
        self.assertEqual(fks_onia.onium_pairs(legs), [(2, 3), (4, 5)])
        self.assertEqual([fks_onia.physical_number(legs, n) for n in (1, 2, 3, 5, 7)],
                         [1, 2, 3, 4, 5])
        self.assertRaises(fks_onia.OniaNLOError, fks_onia.physical_number, legs, 4)
        self.assertEqual(fks_onia.physical_pdgs(legs), [21, 21, 443, 9940003, 21])
        bad = self.legs([(21, {}), (21, {}), (4, o1), (21, {}), (-4, o1)])
        self.assertRaises(fks_onia.OniaNLOError, fks_onia.onium_pairs, bad)

    def test_aux_states(self):
        self.assertEqual(fks_onia.aux_states({'L': 1, 'C': 1}), [(8, '18')])
        self.assertEqual(fks_onia.aux_states({'L': 1, 'C': 8}), [(1, '81'), (8, '88')])
        self.assertEqual(fks_onia.aux_states({'L': 0, 'C': 8}), [])

    def test_perturbation_tag(self):
        self.assertEqual(fks_onia.fock_state_perturbation_tag('all', ['QCD']), ' [QCD]')
        self.assertEqual(fks_onia.fock_state_perturbation_tag('real', ['QCD']), ' [real=QCD]')
        self.assertEqual(fks_onia.fock_state_perturbation_tag('tree', []), '')
