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

    def test_builtin_virtuals(self):
        """[QCD] with bound states: which 2 -> 1 Borns get a built-in virtual
        (BinothLHA_onia.f, Petrelli et al.), which vanish, and which need the
        user hook (fks_onia.builtin_virtual)."""
        model = self.cmd._curr_model
        cases = {'g g > etac(1|1S01) [QCD]': (1, fks_onia.PETRELLI_GG[((0, 0, 0), 1)]),
                 'g g > chic2(1|3P21) [QCD]': (1, fks_onia.PETRELLI_GG[((1, 1, 2), 1)]),
                 'g g > Jpsi(1|3P08) [QCD]': (2, fks_onia.PETRELLI_GG[((1, 1, 0), 8)]),
                 'u u~ > Jpsi(1|3S18) [QCD]': (3, fks_onia.PETRELLI_QQ_3S18),
                 'u u~ > tata(1|3S1) [QCD]': (4, 0.),
                 'u u~ > chic0(1|3P01) [QCD]': (5, 0.),
                 'g g > hc(1|1P11) [QCD]': (5, 0.),
                 'g g > etac(1|1S01) g [QCD]': (0, 0.)}
        for line, (case, bconst) in cases.items():
            borns = self.generate(line)
            for born in borns:
                self.assertTrue(getattr(born, 'onia_virtual', False), line)
                info = fks_onia.builtin_virtual(born.born_amp['process']['legs'], model)
                self.assertEqual(info['case'], case, line)
                self.assertAlmostEqual(info['bconst'], bconst, places=12)
        # eq. (125) of hep-ph/9707223 for 1S0[1], C_F (-5 + pi^2/4) + C_A (1 + 5 pi^2/12)
        self.assertAlmostEqual(fks_onia.PETRELLI_GG[((0, 0, 0), 1)], 11.96020697, places=7)
        # [real=QCD] keeps the user hook
        for born in self.generate('g g > etac(1|1S01) [real=QCD]'):
            self.assertFalse(getattr(born, 'onia_virtual', False))

    def test_missing_singular_borns(self):
        """A real whose soft or collinear region has no generated Born is
        reported (MadFKS would not subtract it): with gluons only in the
        initial state, d g > chic0 d g needs d g > chic0 d (soft gluon) and
        d d~ > chic0 g (final d || initial g). With every massless flavour
        in p and j, and for a coupling of another order (e+ e- > a in a QCD
        calculation), nothing is reported."""
        self.generate('g g > chic0(1|3P01) g [real=QCD]')
        found = self.cmd._fks_multi_proc.warn_missing_singular_borns()
        self.assertIn(('d g > chic0(1|3P01) d g', 'd g > chic0(1|3P01) d'), found)
        self.assertIn(('d g > chic0(1|3P01) d g', 'd d~ > chic0(1|3P01) g'), found)
        for real, born in found:
            self.assertNotEqual(born.split('>')[0].split(), ['g', 'g'])
        self.generate('u u~ > z g [real=QCD]')
        self.assertIn(('u g > z u g', 'u g > z u'),
                      self.cmd._fks_multi_proc.warn_missing_singular_borns())
        self.generate('u u~ > e+ e- [real=QCD]')
        self.assertEqual(self.cmd._fks_multi_proc.warn_missing_singular_borns(), [])
        self.cmd.exec_cmd('define p = g u d s u~ d~ s~')
        self.cmd.exec_cmd('define j = g u d s u~ d~ s~')
        self.generate('p p > chic0(1|3P01) j [real=QCD]')
        self.assertEqual(self.cmd._fks_multi_proc.warn_missing_singular_borns(), [])

    def test_c_parity_zeros(self):
        """Diagrams whose onium projection vanishes by C parity or colour are
        left out of the JAMPs (HelasMatrixElement.get_onia_c_forbidden_
        amplitudes): all of q g > J/psi[1] q (two gluons make a C-even
        singlet), none of q q~ > 3S1[8], the two-gluon and s-channel diagrams
        of g g > J/psi[1] g but not its six three-gluon orderings, the own-line
        diagrams of g g > J/psi J/psi but not the crossed ones, the photon but
        not the Z diagrams of u u~ > (tau tau)[1S0] g; for an auxiliary S wave
        (states=) the rule of that state."""
        import madgraph.core.helas_objects as helas_objects
        cases = [('u g > Jpsi(1|3S11) u', 5, 5), ('u u~ > hc(1|1P11) g', 5, 5),
                 ('g g > hc(1|1P11)', 3, 3), ('u u~ > Jpsi(1|3S18)', 1, 0),
                 ('u u~ > etac(1|1S08)', 1, 1), ('g g > Jpsi(1|3S11) g', 18, 12),
                 ('g g > Jpsi(1|3S11) Jpsi(1|3S11)', 76, 38),
                 ('u u~ > tata(1|1S0) g', 4, 2), ('u u~ > tata(1|3S1)', 2, 0)]
        for process, namp, ndrop in cases:
            self.cmd.exec_cmd('generate %s' % process)
            me = helas_objects.HelasMatrixElement(self.cmd._curr_amps[0], gen_color=True)
            self.assertEqual(sum(len(d['amplitudes']) for d in me['diagrams']), namp, process)
            self.assertEqual(len(me.get_onia_c_forbidden_amplitudes()), ndrop, process)
        # u u~ > 1P1[1] g vanishes; as a 1S0[8] state (the auxiliary S wave of
        # its soft current) its single-gluon diagrams survive
        self.cmd.exec_cmd('generate u u~ > hc(1|1P11) g')
        me = helas_objects.HelasMatrixElement(self.cmd._curr_amps[0], gen_color=True)
        self.assertEqual(len(me.get_onia_c_forbidden_amplitudes()), 5)
        aux = me.get_onia_c_forbidden_amplitudes({0: {'L': 0, 'S': 0, 'C': 8}})
        self.assertLess(len(aux), 5)

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

    def test_colour_transitions_conserve_colour(self):
        """sum_K <P| T_K Q_eff |S'> = 0 over the coloured Born legs, the
        onium included (exact rationals), for P[1] -> S[8], P[8] -> S[1],
        S[8]; the onium's own charge gives T.D = 0."""
        for line, labels in [('u u~ > chic0(1|3P01) g', ['18']),
                             ('g g > chic0(1|3P01) g', ['18']),
                             ('g g > Jpsi(1|3P08) g', ['81', '88']),
                             ('u u~ > Jpsi(1|3P08) g', ['81', '88'])]:
            born, = self.generate(line + ' [real=QCD]')
            amp = born.born_amp
            tm = fks_onia.transition_matrices(amp, amp['process']['model'])
            self.assertEqual([a['label'] for a in tm['aux']], labels)
            for aux in tm['aux']:
                kinds = set(kind for (_, kind) in aux['matrices'])
                self.assertEqual(kinds, {'Delta'} if aux['label'] != '88' else {'D', 'T'})
                for kind in kinds:
                    total = {}
                    for (k, kd), mat in aux['matrices'].items():
                        if kd != kind:
                            continue
                        for key, (re, im) in mat.items():
                            t = total.get(key, (0, 0))
                            total[key] = (t[0] + re, t[1] + im)
                    self.assertTrue(any(aux['matrices'][(k, kd)] for (k, kd) in aux['matrices']
                                        if kd == kind), line)
                    self.assertFalse([v for v in total.values() if v != (0, 0)], (line, kind))
                if aux['label'] == '88':
                    self.assertFalse(aux['matrices'][(tm['onium'], 'D')])
                    self.assertTrue(aux['matrices'][(tm['onium'], 'T')])

    def test_colour_forbidden_born(self):
        """q q~ > chi_c0[1] has an empty projected colour basis: zero colour
        flows (not a colourless process) and zero colour links, while g g >
        h_c (zero by Landau-Yang) keeps its colour basis."""
        import madgraph.core.helas_objects as helas_objects
        born, = self.generate('u u~ > chic0(1|3P01) [real=QCD]')
        me = helas_objects.HelasMatrixElement(born.born_amp)
        self.assertEqual(len(me['color_basis']), 0)
        self.assertTrue(me.is_onia_colour_forbidden())
        self.assertEqual(me.get_color_amplitudes(), [[]])

        class _Born(object):
            def get(self, key):
                return {'base_amplitude': born.born_amp, 'color_basis': me['color_basis']}[key]
        self.assertEqual(fks_onia.born_color_links(_Born()),
                         [{'link': [1, 2], 'matrix': {}, 'ncolor': 1}])
        born, = self.generate('g g > hc(1|1P11) [real=QCD]')
        me = helas_objects.HelasMatrixElement(born.born_amp)
        self.assertEqual(len(me['color_basis']), 1)
        self.assertFalse(me.is_onia_colour_forbidden())


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

    def test_nonet_identities(self):
        """The effective generators of the P-wave soft current against
        explicit SU(3) matrices (arXiv:2402.19221, sect. 3.1): the Casimirs
        C_eff written in born_onia.f, Q_eff(81).Q_eff(88) = 0,
        T.Q_eff(81) = 0 and T.Q_eff(88) = (1/r - r) C_A/2."""
        try:
            import numpy as np
        except ImportError:
            raise unittest.SkipTest('numpy needed')
        lam = np.zeros((8, 3, 3), complex)
        lam[0][0, 1] = lam[0][1, 0] = 1
        lam[1][0, 1], lam[1][1, 0] = -1j, 1j
        lam[2][0, 0], lam[2][1, 1] = 1, -1
        lam[3][0, 2] = lam[3][2, 0] = 1
        lam[4][0, 2], lam[4][2, 0] = -1j, 1j
        lam[5][1, 2] = lam[5][2, 1] = 1
        lam[6][1, 2], lam[6][2, 1] = -1j, 1j
        lam[7] = np.diag([1, 1, -2]) / np.sqrt(3)
        t = lam / 2
        f = np.zeros((8, 8, 8))
        d = np.zeros((8, 8, 8))
        for a in range(8):
            for b in range(8):
                for c in range(8):
                    comm = t[a] @ t[b] - t[b] @ t[a]
                    anti = t[a] @ t[b] + t[b] @ t[a]
                    f[a, b, c] = (-2j * np.trace(comm @ t[c])).real
                    d[a, b, c] = (2 * np.trace(anti @ t[c])).real
        nc, cf, ca = 3., 4. / 3., 3.
        for r in (1., 1.55 / 4.7):
            m = 2 + r + 1 / r
            q18, q81, q88, tt = (np.zeros((8, 9, 9), complex) for _ in range(4))
            for a in range(8):
                q18[a, 0, 1 + a] = m / np.sqrt(2 * nc)
                q81[a, 1 + a, 0] = m / np.sqrt(2 * nc)
                q88[a, 1:, 1:] = m * d[a] / 2 + (1 / r - r) / 2 * (-1j * f[a])
                tt[a, 1:, 1:] = -1j * f[a]
            def dot(x, y):
                return sum(x[a].conj().T @ y[a] for a in range(8))
            ceff18 = m ** 2 / (2 * nc)
            ceff81 = m ** 2 * cf
            ceff88 = m * (-2 / nc + (nc ** 2 - 2) / (2 * nc) * (r + 1 / r))
            octet = np.diag([0.] + [1.] * 8)
            singlet = np.diag([1.] + [0.] * 8)
            self.assertTrue(np.allclose(dot(q18, q18), ceff18 * octet))
            self.assertTrue(np.allclose(dot(q81, q81), ceff81 * singlet))
            self.assertTrue(np.allclose(dot(q88, q88), ceff88 * octet))
            self.assertTrue(np.allclose(dot(q81, q88), 0))
            self.assertTrue(np.allclose(sum(tt[a] @ q81[a] for a in range(8)), 0))
            self.assertTrue(np.allclose(sum(tt[a] @ q88[a] for a in range(8)),
                                        (1 / r - r) / 2 * ca * octet))
