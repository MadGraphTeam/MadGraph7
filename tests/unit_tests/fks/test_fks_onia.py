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

import math
import os
import re
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
        self.assertRefused('g g > Jpsi(1|3S18) g [QED]', 'QCD corrections only')

    def test_two_pwaves_onia_as_sisters(self):
        """Several P-wave quarkonia are supported (JHEP_2Pwave). When the
        onia are the only coloured particles (e+ e- > two P waves), the
        gluon of the real is emitted by the onia, which are its FKS sisters
        (soft only): one sector per onium, j = its first constituent."""
        born, = self.generate('e+ e- > Jpsi(1|3P08) chic2(1|3P21) / z h QED=2 QCD=2 [real=QCD]')
        self.assertEqual(len(born.real_amps), 1)
        real = born.real_amps[0]
        self.assertEqual([l['id'] for l in real.process['legs']], [-11, 11, 4, -4, 4, -4, 21])
        self.assertEqual(sorted((i['i'], i['j']) for i in real.fks_infos), [(7, 3), (7, 5)])
        self.assertEqual(sorted(real.fks_j_from_i[7]), [3, 5])
        infos, j_from_i, _, _, _ = fks_onia.to_physical_fks(
            real.process['legs'], real.process['model'], real.fks_infos, real.fks_j_from_i,
            real.colors, real.particle_tags, real.charges)
        self.assertEqual(sorted((i['i'], i['j']) for i in infos), [(5, 3), (5, 4)])
        self.assertEqual(j_from_i[5], [3, 4])
        # next to elementary coloured partons too, each P wave is the sister
        # of the gluon it emits (its soft cone is mapped about it)
        born, = self.generate('u u~ > chic0(1|3P01) chic2(1|3P21) QED=0 [real=QCD]')
        self.check_reals(born)
        real = [r for r in born.real_amps if r.process['legs'][-1]['id'] == 21][0]
        self.assertEqual([l['id'] for l in real.process['legs']], [2, -2, 4, -4, 4, -4, 21])
        self.assertEqual(sorted((i['i'], i['j']) for i in real.fks_infos), [(7, 1), (7, 2), (7, 3), (7, 5)])

    def test_dipole_matrices(self):
        """The dipole-dipole colour matrices of two P-wave octets (all
        kinds [81], [88] on both sides): exchanging the two onia conjugates
        and transposes the cross matrices and leaves the parallel ones
        unchanged; HKW's 3P[8] + 3P[1] has the [81][18] cross term only (the
        [88] daughter with the singlet spectator is colour forbidden), with
        the daughters 3S1[1] + 3P[1] and 3P[8] + 3S1[8] of JHEP_2Pwave, and no
        parallel term (colour-forbidden Born)."""
        for line in ('e+ e- > Jpsi(1|3P28) psi(2|3P08) / z h QED=2 QCD=2',
                     'e+ e- > Jpsi(1|3P08) chic2(1|3P21) / z h QED=2 QCD=2'):
            born, = self.generate(line + ' [real=QCD]')
            amp = born.born_amp
            model = amp['process']['model']
            p1, p2 = fks_onia.pwave_quarkonia(amp['process']['legs'], model)
            tm1 = fks_onia.transition_matrices(amp, model, pleg=p1)
            tm2 = fks_onia.transition_matrices(amp, model, pleg=p2)
            dm = fks_onia.dipole_matrices(amp, model, p1, p2, tm1, tm2)
            rev = fks_onia.dipole_matrices(amp, model, p2, p1, tm2, tm1)
            self.assertTrue(dm['cross'])
            for (a1, a2, k1, k2), mat in dm['cross'].items():
                other = rev['cross'][(a2, a1, k2, k1)]
                self.assertEqual(sorted(other), sorted((b, g) for (g, b) in mat))
                for (g, b), (re, im) in mat.items():
                    self.assertEqual(other[(b, g)], (re, -im))
            self.assertEqual(set(dm['parallel']),
                             set((a1, a2, k1, k2) for (a2, a1, k2, k1) in rev['parallel']))
            for (a1, a2, k1, k2), mat in dm['parallel'].items():
                self.assertEqual(rev['parallel'][(a2, a1, k2, k1)], mat)
            if 'chic2' in line:
                labels = [(tm1['aux'][a1]['label'], tm2['aux'][a2]['label'])
                          for (a1, a2, _, _) in dm['cross']]
                self.assertEqual(sorted(set(labels)), [('81', '18')])
                self.assertFalse(dm['parallel'])
            else:
                self.assertTrue(dm['parallel'])

    def test_identical_onia_partition(self):
        """Two identical P-wave onia as the only coloured particles: one
        sector survives (the other is equivalent), and both onia stay in the
        partition of fks_Sij, so that the kept sector times its orbit
        factor FKS_FAC_J_D = 2 counts the real once; with elementary
        coloured partons the onia own no sector and are left out."""
        born, = self.generate('e+ e- > Jpsi(1|3P08) Jpsi(1|3P08) / z h QED=2 QCD=2 [real=QCD]')
        real, = born.real_amps
        infos, _, _, _, _ = fks_onia.to_physical_fks(
            real.process['legs'], real.process['model'], real.fks_infos, real.fks_j_from_i,
            real.colors, real.particle_tags, real.charges)
        self.assertEqual(len(infos), 1)
        is_onium = [False, False, True, True, False]
        self.assertEqual(fks_onia.partition_nosector(is_onium, infos), [False] * 5)
        self.assertEqual(fks_onia.partition_nosector(is_onium, [{'j': 1}, {'j': 2}]),
                         is_onium)

    def test_partition_decided_per_real(self):
        """Whether the onia of a real are in the partition of fks_Sij is
        decided over all Borns of the real: u g > Jpsi u g owns the sector
        (gluon, onium) under u g > Jpsi u, and its sector (quark, quark) under
        g g > Jpsi g must count the onium in its S function too, or the S
        functions of the real do not sum to one (it shifted sigma of
        p p > Jpsi[3S1(8)] j by ~5%). A real without a final-state gluon
        (u u~ > Jpsi u u~) owns none, under every Born."""
        self.cmd.exec_cmd('define pp = g u u~')
        borns = self.generate('pp pp > Jpsi(1|3S18) pp [real=QCD]')
        copies = {}
        for born in borns:
            for real in born.real_amps:
                key = fks_onia.real_partition_key(real.process['legs'])
                copies.setdefault(key, []).append(real)
        for key, reals in copies.items():
            owns = any(fks_onia.onium_owns_sector(r.process['legs'], r.fks_infos) for r in reals)
            self.assertEqual([r.onia_partition for r in reals], [owns] * len(reals), key)
        gg, = [b for b in borns if [l['id'] for l in b.born_amp['process']['legs']] == [21, 21, 4, -4, 21]]
        ug = [r for r in gg.real_amps if fks_onia.real_partition_key(r.process['legs'])[0] == (2, 21)
              and r.process['legs'][-1]['id'] == 21]
        self.assertTrue(ug)
        for real in ug:
            legs = real.process['legs']
            self.assertFalse(fks_onia.onium_owns_sector(legs, real.fks_infos))
            self.assertTrue(real.onia_partition)
            infos = fks_onia.to_physical_fks(legs, real.process['model'], real.fks_infos, real.fks_j_from_i,
                                             real.colors, real.particle_tags, real.charges)[0]
            row = [bool(fks_onia.onium_of(l)) for l in fks_onia.physical_legs(legs)]
            self.assertEqual(fks_onia.partition_nosector(row, infos), row)   # the old, per-Born rule
            self.assertEqual(fks_onia.partition_nosector(row, infos, real.onia_partition), [False] * len(row))
        quarks = [r for key, reals in copies.items() if key[0] == (2, -2) and 21 not in key[1] for r in reals]
        self.assertTrue(quarks)
        self.assertFalse(any(r.onia_partition for r in quarks))

    def test_leg_identity_keeps_fock_state(self):
        """Constituents of different onia are not interchangeable FKS legs;
        those of identical onia are (the position 'index' is ignored)."""
        import madgraph.fks.fks_common as fks_common
        born, = self.generate('e+ e- > Jpsi(1|3P08) chic2(1|3P21) / z h QED=2 QCD=2 [real=QCD]')
        legs = born.real_amps[0].process['legs']
        ident = fks_common.external_leg_identity
        self.assertNotEqual(ident(legs[2]), ident(legs[4]))
        born, = self.generate('e+ e- > Jpsi(1|3P08) Jpsi(1|3P08) / z h QED=2 QCD=2 [real=QCD]')
        legs = born.real_amps[0].process['legs']
        self.assertEqual(ident(legs[2]), ident(legs[4]))

    def test_onia_data_by_slot(self):
        """onia.inc numbers the onia by their 'index' (ONIA, PAIRS), which
        the FKS sort need not keep in the order of the legs: B_c- comes
        before B_c+ here. The quantum numbers of every onium must sit in its
        own slot (slot 1 = B_c+, the P wave)."""
        import madgraph.core.helas_objects as helas_objects
        import madgraph.iolibs.export_v4 as export_v4
        born, = self.generate('e+ e- > Bc+(1|1P18) Bc-(1|1S08) / z h QED=2 QCD=2 [real=QCD]')
        amp = born.born_amp
        self.assertEqual([l['onium']['index'] for l in amp['process']['legs'] if l['onium']],
                         [1, 1, 0, 0])
        me = helas_objects.HelasMatrixElement(amp, gen_color=False)

        class Lines(list):
            def writelines(self, lines):
                self.extend(lines.split('\n') if isinstance(lines, str) else lines)
        lines = Lines()
        export_v4.ProcessExporterFortran.write_onia_file(None, lines, me)
        data = dict((m.group(1), m.group(2)) for m in
                    (re.match(r'DATA \((\w+)\(i\),i=1,\d+\)/([^/]*)/', l) for l in lines) if m)
        self.assertEqual(data['ONIA'], '4,3')
        self.assertEqual(data['PAIRS'], '0,0,2,-2,1,-1')
        self.assertEqual(data['L_ONIA'], '1,0')
        self.assertEqual(data['J_ONIA'], '1,0')
        self.assertEqual(data['C_ONIA'], '8,8')

    def test_massless_constituents_refused(self):
        cmd = MGCmd.MasterCmd()
        cmd.no_notification()
        cmd.exec_cmd('set automatic_html_opening False --no_save')
        cmd.exec_cmd('import model sm_onia')            # massless charm
        with self.assertRaises(Exception) as ctx:
            cmd.exec_cmd('generate g g > chic0(1|3P01) g [real=QCD]')
        self.assertIn('massive constituents', str(ctx.exception))

    def check_reals(self, born):
        """Constituents adjacent, never FKS parton i, j only as the first
        constituent of a soft-partner onium (its own soft-only sector), and
        only the onium (its first constituent) as a soft partner."""
        model = born.born_amp['process']['model']
        for real in born.real_amps:
            legs = real.process['legs']
            pairs = fks_onia.onium_pairs(legs)          # raises if not adjacent
            constituents = set(n + 1 for pair in pairs for n in pair)
            seconds = set(b + 1 for _, b in pairs)
            for info in real.fks_infos:
                self.assertNotIn(info['i'], constituents)
                self.assertNotIn(info['j'], seconds)
                if info['j'] in constituents:
                    self.assertTrue(fks_onia.onium_is_soft_partner(legs[info['j'] - 1], model))
                    self.assertEqual(legs[info['i'] - 1]['id'], 21)
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
        # the P wave (first constituent 4) owns a sector next to the b quarks
        self.assertEqual(sorted((i['i'], i['j']) for i in real.fks_infos),
                         [(6, 1), (6, 2), (6, 3), (6, 4)])

    def test_zero_born_p_wave_kept(self):
        """q q~ > chi_cJ vanishes at LO, but its real emission is soft singular
        (P[1] -> 3S1[8]) and is kept with the onium as a soft partner."""
        born, = self.generate('u u~ > chic0(1|3P01) [real=QCD]')
        self.check_reals(born)
        real = [r for r in born.real_amps if r.process['legs'][-1]['id'] == 21][0]
        self.assertEqual(sorted(real.fks_j_from_i[5]), [1, 2, 3])

    def test_virtual_source(self):
        """[QCD] with bound states: the tree model keeps the built-in or user
        virtual (onia_virtual); a loop model gives MadLoop's for S waves
        (virt_process with the constituent legs and their 'onium'), and the
        built-in or user one for P waves (fks_onia.madloop_virtual)."""
        born = self.generate('g g > etac(1|1S01) [QCD]')[0]
        self.assertTrue(born.onia_virtual)
        self.assertIsNone(born.virt_process)
        cmd = MGCmd.MasterCmd()
        cmd.no_notification()
        cmd.exec_cmd('set automatic_html_opening False --no_save')
        cmd.exec_cmd('import model loop_sm_onia-c_mass')
        for line, madloop in [('g g > etac(1|1S01) [QCD]', True),
                              ('u u~ > tata(1|3S1) [QCD]', True),
                              ('g g > chic2(1|3P21) [QCD]', False)]:
            cmd.exec_cmd('generate %s' % line)
            born = cmd._fks_multi_proc['born_processes'][0]
            self.assertEqual(not getattr(born, 'onia_virtual', False), madloop, line)
            if madloop:
                legs = born.virt_process.get('legs')
                self.assertEqual([bool(l.get('onium')) for l in legs],
                                 [bool(l.get('onium')) for l in born.born_amp['process']['legs']])
                self.assertTrue(born.virt_amp.get('loop_diagrams'), line)
            else:
                self.assertIsNone(born.virt_amp)
        ok, why = fks_onia.madloop_virtual(self.cmd._fks_multi_proc['born_processes'][0]
                                           .born_amp['process'])
        self.assertFalse(ok)
        self.assertIn('no QCD loops', why)

    def test_vanishing_born(self):
        """Borns that vanish identically by colour or C parity, or as a
        vanishing 2 -> 1 Born (Landau-Yang: g g > 3S1[8]; colour: q q~ >
        chi_c0[1]) (fks_onia.born_vanishes): their virtual is zero, and the FKS
        output does not use MadLoop for them (export_fks.generate_directories_fks)."""
        import madgraph.fks.fks_helas_objects as fks_helas
        self.cmd.exec_cmd('define p = g u d s u~ d~ s~')
        self.cmd.exec_cmd('define j = g u d s u~ d~ s~')
        for line, nonzero in [('p p > Jpsi(1|3S11) j [real=QCD]', ['gg']),
                              ('p p > etac(1|1S01) [real=QCD]', ['gg']),
                              ('p p > Jpsi(1|3S18) [real=QCD]', ['uux', 'uxu']),
                              ('p p > chic0(1|3P01) [real=QCD]', ['gg'])]:
            self.cmd.exec_cmd('generate %s' % line)
            mp = fks_helas.FKSHelasMultiProcess(self.cmd._fks_multi_proc)
            for me in mp['matrix_elements']:
                name = me.born_me.get('processes')[0].shell_string()
                self.assertEqual(fks_onia.born_vanishes(me.born_me),
                                 not any(name.startswith('0_%s_' % n) for n in nonzero), name)
        self.cmd.exec_cmd('define p = g u c d s u~ c~ d~ s~')
        self.cmd.exec_cmd('define j = p')

    def test_builtin_virtuals(self):
        """[QCD] with bound states: which 2 -> 1 Borns get a built-in virtual
        (BinothLHA_onia.f: Petrelli et al., and g g -> 1P1[8] recomputed),
        which vanish, and which need the user hook (fks_onia.builtin_virtual)."""
        model = self.cmd._curr_model
        cases = {'g g > etac(1|1S01) [QCD]': (1, fks_onia.PETRELLI_GG[((0, 0, 0), 1)]),
                 'g g > chic2(1|3P21) [QCD]': (1, fks_onia.PETRELLI_GG[((1, 1, 2), 1)]),
                 'g g > Jpsi(1|3P08) [QCD]': (2, fks_onia.RECOMPUTED_GG[((1, 1, 0), 8)]),
                 'g g > Jpsi(1|3P28) [QCD]': (2, fks_onia.RECOMPUTED_GG[((1, 1, 2), 8)]),
                 'g g > Jpsi(1|1S08) [QCD]': (2, fks_onia.PETRELLI_GG[((0, 0, 0), 8)]),
                 'u u~ > Jpsi(1|3S18) [QCD]': (3, fks_onia.PETRELLI_QQ_3S18),
                 'u u~ > tata(1|3S1) [QCD]': (4, 0.),
                 'u u~ > chic0(1|3P01) [QCD]': (5, 0.),
                 'g g > hc(1|1P11) [QCD]': (5, 0.),
                 'g g > etac(1|1P18) [QCD]': (6, fks_onia.ONIA_GG_1P18),
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
        # g g -> 3P0[8], 3P2[8], recomputed (not Petrelli's eqs. 130-131):
        # (87 pi^2 - 80)/72 and -31/12 + 5 pi^2/4 + 4 log 2 for N_c = 3
        self.assertAlmostEqual(fks_onia.RECOMPUTED_GG[((1, 1, 0), 8)], (87 * math.pi ** 2 - 80) / 72, places=12)
        self.assertAlmostEqual(fks_onia.RECOMPUTED_GG[((1, 1, 2), 8)],
                               -31. / 12 + 5 * math.pi ** 2 / 4 + 4 * math.log(2), places=12)
        self.assertFalse(set(fks_onia.RECOMPUTED_GG) & set(fks_onia.PETRELLI_GG))
        # g g -> 1P1[8], recomputed: (7 pi^2 - 22 - 14 log 2)/6 for N_c = 3
        self.assertAlmostEqual(fks_onia.ONIA_GG_1P18, 6.2305283799644, places=12)
        # [real=QCD] keeps the user hook
        for born in self.generate('g g > etac(1|1S01) [real=QCD]'):
            self.assertFalse(getattr(born, 'onia_virtual', False))

    def test_vanishing_2to1(self):
        """2 -> 1 Borns and auxiliary S-wave Borns that vanish by symmetry
        (Landau-Yang, C and P parity, helicity, colour) are flagged, so that
        their generated kernels return zero instead of rounding noise."""
        model = self.cmd._curr_model
        cases = {'g g > Jpsi(1|3S18) [QCD]': True, 'g g > chic1(1|3P11) [QCD]': True,
                 'g g > hc(1|1P11) [QCD]': True, 'g g > Jpsi(1|3P18) [QCD]': True,
                 'u u~ > Jpsi(1|3P08) [QCD]': True, 'u u~ > etac(1|1P18) [QCD]': True,
                 'u u~ > etac(1|1S08) [QCD]': True,
                 'g g > etac(1|1S01) [QCD]': False, 'g g > chic0(1|3P01) [QCD]': False,
                 'g g > etac(1|1P18) [QCD]': False, 'u u~ > Jpsi(1|3S18) [QCD]': False,
                 'u u~ > tata(1|3S1) [QCD]': False, 'g g > etac(1|1S01) g [real=QCD]': False}
        for line, zero in cases.items():
            for born in self.generate(line):
                legs = born.born_amp['process']['legs']
                self.assertEqual(fks_onia.vanishing_2to1(legs, model), zero, line)
        # auxiliary S-wave Borns: g g -> 3S1[8] vanishes, g g -> 1S0[8] does not
        for born in self.generate('g g > chic0(1|3P01) [QCD]'):
            legs = born.born_amp['process']['legs']
            self.assertTrue(fks_onia.vanishing_2to1(legs, model, dict(L=0, S=1, J=1, C=8)))
        for born in self.generate('g g > hc(1|1P11) [QCD]'):
            legs = born.born_amp['process']['legs']
            self.assertFalse(fks_onia.vanishing_2to1(legs, model, dict(L=0, S=0, J=0, C=8)))

    def test_flush_routines(self):
        """Born-type onium kernels test each helicity amplitude against
        rounding noise (export_fks.get_onia_flush): the routine <k>M2 forms
        the colour-summed square from the colour flows <k>JF of the kernel's
        colour basis and its colour matrix, and the kernel sets amplitudes
        below ONIA_FLUSH_REL of their bound to zero; real kernels have no
        such test."""
        import madgraph.fks.fks_helas_objects as fks_helas
        import madgraph.iolibs.export_fks as export_fks
        exporter = export_fks.ProcessExporterFortranFKS()
        for line, ncolor in (('g g > Jpsi(1|3S18) [real=QCD]', 2),
                             ('u u~ > Jpsi(1|3S18) [real=QCD]', 1)):
            self.generate(line)
            multi = fks_helas.FKSHelasMultiProcess(self.cmd._fks_multi_proc)
            for me in multi.get('matrix_elements'):
                born_me = me.born_me
                decl, test, init, acc, routines = exporter.get_onia_flush(
                    born_me, 'KN', born_me['color_basis'], None)
                self.assertIn('COMMON/KNRL/ONIA_RLAST', decl)
                self.assertIn('CALL KNM2(AMP_OUT,ONIA_ABSAMP,ONIA_M2,ONIA_SCALE)', test)
                self.assertIn('AMP_OUT(:) = (0D0,0D0)', test)
                self.assertIn('ONIA_ABSAMP(:) = 0D0', init)
                self.assertIn('ABS(AMP(:))*ABS(PROJ)', acc)
                self.assertIn('SUBROUTINE KNM2(AMP,ABSAMP,M2,SCALE)', routines)
                self.assertIn('SUBROUTINE KNJF(AMP,JAMP)', routines)
                self.assertIn('NCOLOR=%d)' % ncolor, routines, line)
                self.assertIn('CF(1,1) = ', routines)
        self.assertLess(exporter.ONIA_FLUSH_REL, 1e-16)

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
        # the octet onium (physical leg 3) owns a soft-only sector
        self.assertEqual(sorted((i['i'], i['j']) for i in infos), [(5, 1), (5, 2), (5, 3), (5, 4)])
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
