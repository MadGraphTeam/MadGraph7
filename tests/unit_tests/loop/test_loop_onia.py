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
"""Unit tests of the one-loop amplitudes of processes with bound states
(quarkonia) in MadLoop: the loop_sm_onia model and its generator."""

import ast
import filecmp
import os
import shutil
import sys
import tempfile

root_path = os.path.split(os.path.dirname(os.path.realpath(__file__)))[0]
mg_root = os.path.join(root_path, os.path.pardir, os.path.pardir)
sys.path.append(mg_root)
sys.path.append(os.path.join(mg_root, 'tools'))

import tests.unit_tests as unittest

import make_onia_model
import madgraph.models.import_boundstates as import_boundstates

_model_dir = os.path.join(os.path.abspath(mg_root), 'models')


class TestLoopSmOniaModel(unittest.TestCase):
    """models/loop_sm_onia is models/loop_sm plus the Fock states of sm_onia,
    as written by tools/make_onia_model.py."""

    loop_sm = os.path.join(_model_dir, 'loop_sm')
    loop_sm_onia = os.path.join(_model_dir, 'loop_sm_onia')

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix='onia_model_')

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def tracked_files(self, path):
        return sorted(f for f in os.listdir(path)
                      if not f.endswith(('.pkl', '.log', '.pyc')) and f != '__pycache__')

    def test_generator_reproduces_the_shipped_model(self):
        """Regenerating loop_sm_onia from loop_sm gives the shipped files."""
        out = os.path.join(self.tmpdir, 'loop_sm_onia')
        kept, dropped = make_onia_model.make_onia_model(
            self.loop_sm, out, os.path.join(self.loop_sm_onia, 'boundstates.py'))
        self.assertEqual((kept, dropped), (192, 0))
        self.assertEqual(self.tracked_files(out), self.tracked_files(self.loop_sm_onia))
        for name in self.tracked_files(out):
            new = open(os.path.join(out, name)).read().splitlines()
            old = open(os.path.join(self.loop_sm_onia, name)).read().splitlines()
            if name == 'boundstates.py':
                # the two provenance lines name the source of the Fock states
                new, old = new[2:], old[2:]
            self.assertEqual(new, old, name)

    def test_only_the_onia_files_differ_from_loop_sm(self):
        """Particles, couplings and counterterms are those of loop_sm."""
        common = [f for f in self.tracked_files(self.loop_sm)]
        differing = [f for f in common if not filecmp.cmp(
            os.path.join(self.loop_sm, f), os.path.join(self.loop_sm_onia, f), shallow=False)]
        self.assertEqual(sorted(differing), ['__init__.py', 'object_library.py'])
        self.assertEqual(sorted(set(self.tracked_files(self.loop_sm_onia)) - set(common)),
                         ['boundstates.py'])
        init = open(os.path.join(self.loop_sm_onia, '__init__.py')).read()
        self.assertIn('dual_mass_scheme = True', init)
        self.assertIn('arXiv:2510.26773', init)

    def test_constituents_are_particles_of_the_model(self):
        """Every constituent is a particle of loop_sm, the leptonia renamed by
        PDG code (loop_sm calls the muon and tau m- and tt-)."""
        names = set(make_onia_model.particle_pdgs(os.path.join(self.loop_sm, 'particles.py')))
        header, blocks = make_onia_model.boundstate_blocks(
            os.path.join(self.loop_sm_onia, 'boundstates.py'))
        self.assertEqual(len(blocks), 192)
        constituents = set(p for c, _, _ in blocks for p in c)
        self.assertTrue(constituents <= names, constituents - names)
        self.assertIn('m-', constituents)
        self.assertIn('tt+', constituents)

    def test_boundstates_load(self):
        """The Fock states load from the UFO directory, with their quantum
        numbers."""
        states = import_boundstates.get_boundstates_ufo(self.loop_sm_onia)
        self.assertEqual(len(states), 192)
        etac = [s for s in states if s.name == 'etac(1|1S01)'][0]
        self.assertEqual((etac.pdg_code, etac.particles, etac.spin, etac.orbital, etac.J,
                          etac.color), (441, ['c', 'c~'], 1, 0, 0, 1))
        psi8 = [s for s in states if s.name == 'Jpsi(1|3S18)'][0]
        self.assertEqual((psi8.spin, psi8.orbital, psi8.J, psi8.color), (3, 0, 1, 8))


if __name__ == '__main__':
    unittest.main()


import fractions

import madgraph
import madgraph.interface.master_interface as MGCmd
import madgraph.loop.loop_helas_objects as loop_helas_objects
import madgraph.loop.loop_onia as loop_onia
import madgraph.core.color_amp as color_amp


class TestLoopOniaGeneration(unittest.TestCase):
    """The generation of one-loop processes with bound states: what the
    validator refuses, and the data of the projection (loop_onia.py)."""

    @classmethod
    def setUpClass(cls):
        cls.cmd = MGCmd.MasterCmd()
        cls.cmd.no_notification()
        cls.cmd.exec_cmd('import model loop_sm_onia-c_mass', printcmd=False)

    def generate(self, process):
        self.cmd.exec_cmd('generate %s' % process, printcmd=False)
        return self.cmd._curr_amps

    def matrix_element(self, process):
        amps = self.generate(process)
        mes = loop_helas_objects.LoopHelasProcess(amps, optimized_output=True)
        return mes.get_matrix_elements()[0]

    def test_refused_processes(self):
        """MadLoop: only [virt=QCD] with S-wave bound states and at most two
        P-wave bound states is accepted. [QCD] and [real=QCD] go to the FKS
        subtraction (fks_onia.check_nlo_onia), which accepts them."""
        for process in ['g g > etac(1|1S01) [real=QCD]', 'g g > etac(1|1S01) [QCD]']:
            self.cmd.exec_cmd('generate %s' % process, printcmd=False, errorhandling=False)
            self.assertTrue(self.cmd._fks_multi_proc['born_processes'])
        refused = ['g g > etac(1|1S01) [sqrvirt=QCD]',
                   'g g > etac(1|1S01) [virt=QED]',
                   'e+ e- > chic0(1|3P01) chic2(1|3P21) hc(1|1P11) [virt=QCD]']
        for process in refused:
            self.assertRaises(madgraph.InvalidCmd, lambda p=process: self.cmd.exec_cmd(
                              'generate %s' % p, printcmd=False, errorhandling=False))

    def test_decays_refused(self):
        """The one-loop virtual of a decaying bound state: alone in the
        initial state, not decaying into a P-wave state if it is one
        itself."""
        refused = ['chic0(1|3P01) > hc(1|1P11) a [virt=QCD]',
                   'etac(1|1S01) g > g g [virt=QCD]',
                   'Jpsi(1|3S18) g > u u~ [virt=QCD]']
        for process in refused:
            self.assertRaises(madgraph.InvalidCmd, lambda p=process: self.cmd.exec_cmd(
                              'generate %s' % p, printcmd=False, errorhandling=False))

    def test_decay_projection_data(self):
        """etac > a a and chic1 > u u~ g: the decaying bound state is one
        incoming particle (its two constituent legs are MadLoop's incoming
        legs), averaged over its 2J+1 states, and its octet index in the
        colour strings is that of its antiquark (the leg of colour 3 once
        crossed)."""
        me = self.matrix_element('etac(1|1S01) > a a [virt=QCD]')
        self.assertEqual(me.get_nexternal_ninitial(), (4, 2))
        self.assertEqual(me.get_physical_nexternal_ninitial(), (3, 1))
        self.assertEqual(me.get_denominator_factor(), 2)
        self.assertEqual(me.get_hel_avg_factor(), 1)
        info = loop_onia.onia_loop_info(me)
        self.assertEqual((info['ninitial'], info['ninitial_phys'], info['ninitial_onia']),
                         (2, 1, 1))
        self.assertEqual(info['mapping'], [1, 1, 2, 3])
        onium = info['onia'][0]
        self.assertTrue(onium['initial'])
        self.assertEqual((onium['q1'], onium['q2']), (1, 2))
        self.assertEqual(loop_onia.octet_leg(onium), 2)
        self.assertEqual(info['phys_helc'], [(0, -1, -1), (0, -1, 1), (0, 1, -1), (0, 1, 1)])
        self.assertTrue(loop_onia.has_initial_onia(me))
        me = self.matrix_element('chic1(1|3P11) > u u~ g [virt=QCD]')
        self.assertEqual(me.get_physical_nexternal_ninitial(), (4, 1))
        self.assertEqual(me.get_hel_avg_factor(), 3)
        self.assertEqual(me.get_denominator_factor(), 3)
        info = loop_onia.onia_loop_info(me)
        self.assertEqual(info['ipwaves'], [1])
        # the soft current of the P -> S pole: the 3S1[8] state, with the
        # colour charges of the final partons
        soft = loop_onia.pwave_soft_current(me, info, self.exporter().get_amp_to_jamp_map)
        self.assertEqual([(s['colour'], s['label']) for s in soft], [(8, '18')])
        self.assertEqual(sorted(set(k for k, kind in soft[0]['links'])), [2, 3, 4])
        # a colour octet: averaged over its 2J+1 states only (its decay LDME
        # carries the colour, as at tree level), its charge on the octet
        # index of the colour strings, its antiquark
        me = self.matrix_element('Jpsi(1|3S18) > u u~ [virt=QCD]')
        self.assertEqual(me.get_denominator_factor(), 3)
        self.assertEqual(me.get_hel_avg_factor(), 3)
        info = loop_onia.onia_loop_info(me)
        self.assertEqual(loop_onia.physical_charges(me, info),
                         {1: (2, 8), 2: (3, 3), 3: (4, -3)})
        # a bound state produced is not affected
        me = self.matrix_element('g g > etac(1|1S01) [virt=QCD]')
        self.assertFalse(loop_onia.has_initial_onia(me))
        self.assertEqual(loop_onia.octet_leg(loop_onia.onia_loop_info(me)['onia'][0]), 3)

    def test_massless_constituents_are_refused(self):
        cmd = MGCmd.MasterCmd()
        cmd.no_notification()
        cmd.exec_cmd('import model loop_sm_onia', printcmd=False)
        self.assertRaises(madgraph.InvalidCmd, lambda: cmd.exec_cmd(
                          'generate g g > etac(1|1S01) [virt=QCD]', printcmd=False,
                          errorhandling=False))

    def test_projection_data(self):
        """g g > etac: the constituent legs, helicities and normalisation."""
        me = self.matrix_element('g g > etac(1|1S01) [virt=QCD]')
        self.assertEqual(me.get_nexternal_ninitial(), (4, 2))
        self.assertEqual(me.get_physical_nexternal_ninitial(), (3, 2))
        self.assertEqual(me.get_helicity_combinations(), 16)
        self.assertEqual(me.get_helicity_combinations(onia_physical=True), 4)
        info = loop_onia.onia_loop_info(me)
        self.assertEqual(info['mapping'], [1, 2, 3, 3])
        self.assertEqual(info['nexternal_phys'], 3)
        self.assertEqual(len(info['onia']), 1)
        onium = info['onia'][0]
        self.assertEqual((onium['q1'], onium['q2'], onium['S'], onium['J'], onium['C']),
                         (3, 4, 0, 0, 1))
        self.assertEqual(info['jz'], [(0,)])
        self.assertEqual(info['nothers'], 4)
        self.assertEqual(info['norm'], fractions.Fraction(1, 3))
        # the configurations sharing the gluon helicities share their index
        for h, o in zip(info['helc'], info['others']):
            same = [o2 for h2, o2 in zip(info['helc'], info['others']) if h2[:2] == h[:2]]
            self.assertEqual(set(same), set([o]))

    def test_physical_helicity_configurations(self):
        """SLOOPMATRIXHEL numbers the physical configurations as MadGraph's
        helicity matrix of the physical process (the tree-level output and
        the FKS Born): a bound state with its J_z = -J..J, the last leg
        fastest. Each has the signature of its other legs."""
        me = self.matrix_element('g g > Jpsi(1|3S11) g [virt=QCD]')
        info = loop_onia.onia_loop_info(me)
        self.assertEqual(info['phys_helc'],
                         [tuple(h) for h in me.get_helicity_matrix(onia_physical=True)])
        self.assertEqual(len(info['phys_helc']), 24)
        self.assertEqual(info['phys_helc'][:3], [(-1, -1, -1, -1), (-1, -1, -1, 1),
                                                 (-1, -1, 0, -1)])
        self.assertEqual(info['phys_leg'], [1, 2, -1, 5])
        for c, o in zip(info['phys_helc'], info['phys_others']):
            for h, o2 in zip(info['helc'], info['others']):
                if o2 == o:
                    self.assertEqual((h[0], h[1], h[4]), (c[0], c[1], c[3]))
        # two bound states
        me = self.matrix_element('e+ e- > Jpsi(1|3S11) etac(1|1S01) / z QED=2 [virt=QCD]')
        info = loop_onia.onia_loop_info(me)
        self.assertEqual(info['phys_helc'],
                         [tuple(h) for h in me.get_helicity_matrix(onia_physical=True)])
        self.assertEqual(len(info['phys_helc']), 12)
        self.assertEqual(info['phys_leg'], [1, 2, -1, -2])

    def test_pwave_projection_data(self):
        """g g > chic2: one P-wave bound state, its five J_z."""
        me = self.matrix_element('g g > chic2(1|3P21) [virt=QCD]')
        self.assertTrue(loop_onia.has_pwave(me))
        info = loop_onia.onia_loop_info(me)
        self.assertEqual(info['ipwave'], 1)
        onium = info['onia'][0]
        self.assertEqual((onium['L'], onium['S'], onium['J'], onium['C']), (1, 1, 2, 1))
        self.assertEqual(info['jz'], [(-2,), (-1,), (0,), (1,), (2,)])
        self.assertFalse(loop_onia.has_pwave(
            self.matrix_element('g g > etac(1|1S01) [virt=QCD]')))

    def test_two_pwave_projection_data(self):
        """e+ e- > chic0 hc: two P-wave bound states, in the order of their dual
        directions."""
        me = self.matrix_element('e+ e- > chic0(1|3P01) hc(1|1P11) [virt=QCD]')
        self.assertEqual(loop_onia.npwave(me), 2)
        info = loop_onia.onia_loop_info(me)
        self.assertEqual((info['npwave'], info['ipwaves'], info['ipwave']), (2, [1, 2], 1))
        self.assertEqual([(o['L'], o['S'], o['J']) for o in info['onia']],
                         [(1, 1, 0), (1, 0, 1)])
        self.assertEqual(info['jz'], [(0, -1), (0, 0), (0, 1)])
        self.assertEqual(loop_onia.npwave(self.matrix_element('g g > chic2(1|3P21) [virt=QCD]')), 1)
        # the rank the P-wave states add, for LOOPMAXCOEFS and the polynomial
        # routines alike (R+2 here: tripled lines and two doubled lines)
        self.assertEqual(loop_onia.loop_rank_increase(me), 2)
        self.assertEqual(loop_onia.loop_rank_increase(
            self.matrix_element('g g > chic2(1|3P21) [virt=QCD]')), 1)
        self.assertEqual(loop_onia.loop_rank_increase(
            self.matrix_element('g g > etac(1|1S01) [virt=QCD]')), 0)

    @staticmethod
    def exporter():
        import madgraph.loop.loop_exporters as loop_exporters
        return loop_exporters.LoopProcessOptimizedExporterFortranSA.__new__(
            loop_exporters.LoopProcessOptimizedExporterFortranSA)

    def born_map(self, me):
        return self.exporter().get_amp_to_jamp_map(me.get_born_color_amplitudes(),
                                                   me.get_number_of_born_amplitudes())

    def soft_current(self, process):
        me = self.matrix_element(process)
        info = loop_onia.onia_loop_info(me)
        return me, loop_onia.pwave_soft_current(me, info, self.exporter().get_amp_to_jamp_map)

    def assert_colour_conservation(self, me, aux):
        """sum_k T_k = -T_P on the transition matrices between Born
        amplitudes: zero for a P-wave singlet, and for an octet but for the T
        part of Q_eff(88) (f_abm d_amc = 0, T^a_{b'a} = 0)."""
        zero = (fractions.Fraction(0), fractions.Fraction(0))
        nborn = me.get_number_of_born_amplitudes()
        ampb = self.born_map(me)
        for a in aux:
            for kind in set(k for _, k in a['links']):
                if a['label'] == '88' and kind == 2:
                    continue
                mats = [loop_onia._to_amplitude_space(x, ampb, a['amp_s'], nborn)
                        for (leg, k), x in a['links'].items() if k == kind]
                self.assertTrue(any(c != zero for x in mats for row in x for c in row))
                for i in range(nborn):
                    for j in range(nborn):
                        self.assertEqual(sum(x[i][j][0] for x in mats), 0)
                        self.assertEqual(sum(x[i][j][1] for x in mats), 0)

    def test_pwave_soft_current_colour(self):
        """The S-wave states and colour of the soft current of a P-wave
        bound state (P -> S pole), and colour conservation of the
        transition matrices."""
        me, aux = self.soft_current('g g > chic0(1|3P01) [virt=QCD]')
        self.assertEqual([(a['label'], a['cnorm']) for a in aux], [('18', 6)])
        self.assertEqual(sorted(aux[0]['links']), [(1, 1), (2, 1)])
        self.assert_colour_conservation(me, aux)
        me, aux = self.soft_current('g g > etac(1|1P18) [virt=QCD]')
        self.assertEqual([(a['label'], a['cnorm']) for a in aux],
                         [('81', fractions.Fraction(1, 6)), ('88', 1)])
        self.assertEqual(sorted(aux[1]['links']), [(1, 1), (1, 2), (2, 1), (2, 2)])
        self.assert_colour_conservation(me, aux)
        # the colour charges of the light quarks: initial quark -t^aT,
        # initial antiquark t^a
        me, aux = self.soft_current('u u~ > chic0(1|3P01) g [virt=QCD]')
        self.assertEqual(sorted(aux[0]['links']), [(1, 1), (2, 1), (4, 1)])
        self.assert_colour_conservation(me, aux)
        # a colour-forbidden S-wave state is left out: e+ e- > 3S1[1] g
        me, aux = self.soft_current('e+ e- > Jpsi(1|3P08) g [virt=QCD]')
        self.assertEqual([a['label'] for a in aux], ['88'])

    def test_two_pwave_soft_colour(self):
        """Two P-wave bound states: the soft current of each state (the other
        one keeps its colour) and the dipole-dipole colour data (term 3 of
        the poles). e+ e- > chic0 hc: the P -> S transitions are colour
        forbidden (8 x 1 has no singlet), the S'S' = 3S1[8] 1S0[8] exchange
        is allowed with the Delta x Delta structure."""
        import madgraph.loop.loop_exporters as loop_exporters
        exporter = loop_exporters.LoopProcessOptimizedExporterFortranSA.__new__(
            loop_exporters.LoopProcessOptimizedExporterFortranSA)
        me = self.matrix_element('e+ e- > chic0(1|3P01) hc(1|1P11) [virt=QCD]')
        info = loop_onia.onia_loop_info(me)
        nborn = me.get_number_of_born_amplitudes()
        for ipw in info['ipwaves']:
            self.assertEqual(loop_onia.pwave_soft_current(
                me, info, exporter.get_amp_to_jamp_map, ipw), [])
        dd = loop_onia.pwave_dipole_dipole(me, info, exporter.get_amp_to_jamp_map)
        self.assertEqual([(d['label1'], d['label2'], d['kind1'], d['kind2'], d['cnorm1'],
                           d['cnorm2']) for d in dd], [('18', '18', 1, 1, 6, 6)])
        x = loop_onia._to_amplitude_space(dd[0]['x'], self.born_map(me), dd[0]['amp_s'], nborn)
        self.assertTrue(any(c != (0, 0) for row in x for c in row))
        # real (no f) and Hermitian-free: X[a][b] = sum conj(c_a) c'_b
        self.assertTrue(all(c[1] == 0 for row in x for c in row))
        self.assertEqual(len(x), nborn)

    def test_flow_colour_matrices(self):
        """The colour data in the Born colour-flow basis gives the colour
        matrices between Born amplitudes: the Born colour matrix, and the
        colour links of the projected Born basis (T_j.T_k inserted after the
        projection) those of the diagrams (inserted before, as in MadFKS),
        for a light quark, gluons, an octet and a singlet state, massive
        quarks and unequal masses."""
        for process in ['u u~ > chic0(1|3P01) g [virt=QCD]',
                        'g g > etac(1|1P18) [virt=QCD]',
                        'g g > Jpsi(1|3S18) g [virt=QCD]',
                        'e+ e- > Bc+(1|1S01) b c~ / z t t~ QED=2 [virt=QCD]']:
            me = self.matrix_element(process)
            info = loop_onia.onia_loop_info(me)
            nborn = me.get_number_of_born_amplitudes()
            ampb = self.born_map(me)
            cm = color_amp.ColorMatrix(me.get('born_color_basis'))
            denom = cm.get_line_denominators()
            num = [cm.get_line_numerators(i, d) for i, d in enumerate(denom)]
            self.assertEqual(loop_onia._to_amplitude_space(
                loop_onia.flow_colour_matrix(cm), ampb, ampb, nborn),
                loop_onia.born_born_color_factors(ampb, num, denom), process)
            reference = loop_onia.physical_color_links(info, loop_onia.color_linked_born_factors(
                me, self.exporter().get_amp_to_jamp_map))
            flows = loop_onia.flow_color_links(me, info)
            self.assertEqual(sorted(flows), sorted(reference), process)
            zero = (fractions.Fraction(0), fractions.Fraction(0))
            nonzero = 0
            for key, cl in reference.items():
                self.assertEqual(loop_onia._to_amplitude_space(flows[key], ampb, ampb, nborn),
                                 cl, (process, key))
                nonzero += any(c != zero for row in cl for c in row)
            self.assertTrue(nonzero, process)

    def test_triplet_polarisations(self):
        """A spin triplet has three J_z, a colour octet the norm 2."""
        me = self.matrix_element('u u~ > Jpsi(1|3S18) [virt=QCD]')
        info = loop_onia.onia_loop_info(me)
        self.assertEqual(info['jz'], [(-1,), (0,), (1,)])
        self.assertEqual(info['norm'], fractions.Fraction(2))

    def test_projected_colour_factors(self):
        """With the colour-singlet projector the s-channel gluon Born diagram
        of g g > etac(1|1S01) has no colour factor, and the loop colour basis
        is projected too."""
        me = self.matrix_element('g g > etac(1|1S01) [virt=QCD]')
        born = me.get_born_diagrams()
        self.assertEqual(len(born), 3)
        import madgraph.loop.loop_exporters as loop_exporters
        exporter = loop_exporters.LoopProcessOptimizedExporterFortranSA.__new__(
            loop_exporters.LoopProcessOptimizedExporterFortranSA)
        nborn = me.get_number_of_born_amplitudes()
        ampb = exporter.get_amp_to_jamp_map(me.get_born_color_amplitudes(), nborn)
        cm = color_amp.ColorMatrix(me.get('born_color_basis'))
        denom = cm.get_line_denominators()
        num = [cm.get_line_numerators(i, d) for i, d in enumerate(denom)]
        cfb = loop_onia.born_born_color_factors(ampb, num, denom)
        zero = (fractions.Fraction(0), fractions.Fraction(0))
        # the s-channel diagram (the three-gluon vertex) is the one without
        # colour factor; the two others have the same colour factor
        s_channel = [i for i, row in enumerate(cfb) if all(c == zero for c in row)]
        self.assertEqual(len(s_channel), 1)
        others = [i for i in range(nborn) if i not in s_channel]
        self.assertEqual(len(set(cfb[i][j] for i in others for j in others)), 1)
        # every colour structure of the loop basis carries the singlet projector
        for col_str in me.get('loop_color_basis').keys():
            self.assertTrue(col_str is not None)


class TestPwaveCoefficientCalls(unittest.TestCase):
    """The double-precision coefficient construction of a P-wave output:
    the dual loop vertices update the value and the derivative of the loop
    wavefunctions, the dual objects carry the names of the PW_DUAL module."""

    def test_calls(self):
        calls = ['# Coefficient construction for loop diagram with ID 4',
                 'CALL VVV1L2P0_1(PL(0,0),W(5),GC_4,ZERO,ZERO,PL(0,1),COEFS)\n'
                 'CALL ML5_0_UPDATE_WL_0_1(WL(1,0,1,0),4,COEFS,4,4,WL(1,0,1,1))',
                 'CALL ML5_0_CREATE_LOOP_COEFS(WL(1,0,1,2),2,4,1,2,1,92,H)']
        out = loop_onia.pwave_dual_names(loop_onia.pwave_coef_construction_calls(calls))
        self.assertEqual(out[0], calls[0])
        self.assertEqual(out[1].split('\n'), [
            'CALL VVV1L2P0_1(PWPL(0,0),PWW(5),GC_4,ZERO,ZERO,PWPL(0,1),COEFS)',
            'CALL ML5_0_PW_SPLIT_COEFS(COEFS,COEFS0,COEFS1)',
            'CALL ML5_0_UPDATE_WL_0_1(WL(1,0,1,0),4,COEFS0,4,4,WL(1,0,1,1))',
            'CALL ML5_0_UPDATE_WL_0_1(DWL(1,0,1,0),4,COEFS0,4,4,DWL(1,0,1,1))',
            'CALL ML5_0_UPDATE_WL_0_1(WL(1,0,1,0),4,COEFS1,4,4,TWL)',
            'CALL ML5_0_PW_ADD_WL(TWL,1,4,4,DWL(1,0,1,1))'])
        self.assertEqual(out[2],
            'CALL ML5_0_PW_CREATE_LOOP_COEFS(WL(1,0,1,2),DWL(1,0,1,2),2,4,1,2,1,92,H)')

    def test_two_pwave_calls(self):
        """Two P-wave states: the Leibniz rule over the four components (WL,
        DWL, DWL2, D2WL), nine updates per vertex."""
        calls = ['CALL ML5_0_UPDATE_WL_0_1(WL(1,0,1,0),4,COEFS,4,4,WL(1,0,1,1))',
                 'CALL ML5_0_CREATE_LOOP_COEFS(WL(1,0,1,2),2,4,1,2,1,92,H)']
        out = loop_onia.pwave_coef_construction_calls(calls, npwave=2)
        u = 'CALL ML5_0_UPDATE_WL_0_1(%s(1,0,1,0),4,%s,4,4,%s)'
        add = 'CALL ML5_0_PW_ADD_WL(TWL,1,4,4,%s(1,0,1,1))'
        self.assertEqual(out[0].split('\n'), [
            'CALL ML5_0_PW_SPLIT_COEFS2(COEFS,COEFS0,COEFS1,COEFS2,COEFS3)',
            u % ('D2WL', 'COEFS0', 'D2WL(1,0,1,1)'),
            u % ('DWL', 'COEFS2', 'TWL'), add % 'D2WL',
            u % ('DWL2', 'COEFS1', 'TWL'), add % 'D2WL',
            u % ('WL', 'COEFS3', 'TWL'), add % 'D2WL',
            u % ('DWL', 'COEFS0', 'DWL(1,0,1,1)'),
            u % ('WL', 'COEFS1', 'TWL'), add % 'DWL',
            u % ('DWL2', 'COEFS0', 'DWL2(1,0,1,1)'),
            u % ('WL', 'COEFS2', 'TWL'), add % 'DWL2',
            u % ('WL', 'COEFS0', 'WL(1,0,1,1)')])
        self.assertEqual(out[1], 'CALL ML5_0_PW_CREATE_LOOP_COEFS2(WL(1,0,1,2),DWL(1,0,1,2),'
                                 'DWL2(1,0,1,2),D2WL(1,0,1,2),2,4,1,2,1,92,H)')

    def test_mp_calls(self):
        """The quadruple-precision calls: MP_ updaters, MPPW names."""
        calls = ['CALL MP_VVV1L2P0_1(PL(0,0),W(5),GC_4,ZERO,ZERO,PL(0,1),COEFS)\n'
                 'CALL MP_ML5_0_UPDATE_WL_0_1(WL(1,0,1,0),4,COEFS,4,4,WL(1,0,1,1))',
                 'CALL MP_ML5_0_CREATE_LOOP_COEFS(WL(1,0,1,2),2,4,1,2,1,92,H)',
                 'CALL MP_FFV1_0(W(4),W(3),W(5),GC_5,AMP(1))']
        out = loop_onia.pwave_dual_names(loop_onia.pwave_coef_construction_calls(calls),
                                         mp=True)
        self.assertEqual(out[0].split('\n'), [
            'CALL MP_VVV1L2P0_1(MPPWPL(0,0),MPPWW(5),GC_4,ZERO,ZERO,MPPWPL(0,1),COEFS)',
            'CALL MP_ML5_0_PW_SPLIT_COEFS(COEFS,COEFS0,COEFS1)',
            'CALL MP_ML5_0_UPDATE_WL_0_1(WL(1,0,1,0),4,COEFS0,4,4,WL(1,0,1,1))',
            'CALL MP_ML5_0_UPDATE_WL_0_1(DWL(1,0,1,0),4,COEFS0,4,4,DWL(1,0,1,1))',
            'CALL MP_ML5_0_UPDATE_WL_0_1(WL(1,0,1,0),4,COEFS1,4,4,TWL)',
            'CALL MP_ML5_0_PW_ADD_WL(TWL,1,4,4,DWL(1,0,1,1))'])
        self.assertEqual(out[1],
            'CALL MP_ML5_0_PW_CREATE_LOOP_COEFS(WL(1,0,1,2),DWL(1,0,1,2),2,4,1,2,1,92,H)')
        self.assertEqual(out[2], 'CALL MP_FFV1_0(MPPWW(4),MPPWW(3),MPPWW(5),GC_5,MPPWAMP(1))')

    def test_two_pwave_mp_calls(self):
        """Two P-wave states in quadruple precision: the MP_ routines of the
        four-component expansion (onia_loop.f)."""
        calls = ['CALL MP_ML5_0_UPDATE_WL_0_1(WL(1,0,1,0),4,COEFS,4,4,WL(1,0,1,1))',
                 'CALL MP_ML5_0_CREATE_LOOP_COEFS(WL(1,0,1,2),2,4,1,2,1,92,H)']
        out = loop_onia.pwave_dual_names(
            loop_onia.pwave_coef_construction_calls(calls, npwave=2), mp=True)
        lines = out[0].split('\n')
        self.assertEqual(len(lines), 15)
        self.assertEqual(lines[0], 'CALL MP_ML5_0_PW_SPLIT_COEFS2(COEFS,COEFS0,COEFS1,COEFS2,COEFS3)')
        self.assertEqual(lines[1],
                         'CALL MP_ML5_0_UPDATE_WL_0_1(D2WL(1,0,1,0),4,COEFS0,4,4,D2WL(1,0,1,1))')
        self.assertEqual(lines[3], 'CALL MP_ML5_0_PW_ADD_WL(TWL,1,4,4,D2WL(1,0,1,1))')
        self.assertEqual(lines[-1],
                         'CALL MP_ML5_0_UPDATE_WL_0_1(WL(1,0,1,0),4,COEFS0,4,4,WL(1,0,1,1))')
        self.assertEqual(out[1], 'CALL MP_ML5_0_PW_CREATE_LOOP_COEFS2(WL(1,0,1,2),DWL(1,0,1,2),'
                                 'DWL2(1,0,1,2),D2WL(1,0,1,2),2,4,1,2,1,92,H)')

    def test_mp_dual_modules(self):
        """The quadruple-precision dual numbers: one direction for one P-wave
        state (mp_dual_variables.f, also the source of the EP ones), both
        directions and the mixed derivative for two (mp_dual2_variables.f),
        with the same module and public names."""
        import madgraph.iolibs.export_v4 as export_v4
        template = os.path.join(mg_root, 'aloha', 'template_files')
        texts = {}
        for name in ('mp_dual_variables.f', 'mp_dual2_variables.f'):
            with open(os.path.join(template, name)) as stream:
                texts[name] = stream.read()
        one, two = texts['mp_dual_variables.f'], texts['mp_dual2_variables.f']
        self.assertIn('complex(kind=qp), dimension(0:1) :: comp', one)
        self.assertIn('complex(kind=qp), dimension(0:3) :: comp', two)
        public = lambda text: [l.strip() for l in text.splitlines()
                               if l.strip().startswith('public ::')]
        self.assertEqual(public(one), public(two))
        for text in (one, two):
            self.assertTrue(text.startswith('      module mp_dual_variables\n'))
        for npwave, name in ((1, 'mp_dual_variables.f'), (2, 'mp_dual2_variables.f')):
            tmp = tempfile.mkdtemp()
            try:
                export_v4.ProcessExporterFortran.write_loop_dual_aloha_functions(tmp, npwave)
                with open(os.path.join(tmp, 'mp_dual_variables.f')) as stream:
                    self.assertEqual(stream.read(), texts[name])
                with open(os.path.join(tmp, 'dual_opts.inc')) as stream:
                    self.assertIn('PARAMETER (NPWAVE=%d)' % npwave, stream.read())
            finally:
                shutil.rmtree(tmp)

    def test_unknown_updater_is_refused(self):
        self.assertRaises(loop_onia.OniaLoopError, loop_onia.pwave_coef_construction_calls,
                          ['CALL ML5_0_DP_TERMINAL_WL_1_1(WL(1,0,1,1),4,COEFS,4,4,'
                           'WL(1,0,1,2),WL_ENTRY_INEXACT)'])

    def test_dual_names(self):
        self.assertEqual(loop_onia.pwave_dual_names(
            ['CALL FFV1_0(W(4),W(3),W(5),GC_5,AMP(1))',
             'AMPL(2,86)=AMPL(2,86)*(6.0D0*UVWFCT_G_0_1EPS)',
             'CALL ML5_0_UPDATE_WL_1_1(WL(1,0,1,2),4,COEFS0,4,4,DWL(1,0,1,3))']),
            ['CALL FFV1_0(PWW(4),PWW(3),PWW(5),GC_5,PWAMP(1))',
             'PWAMPL(2,86)=PWAMPL(2,86)*(6.0D0*UVWFCT_G_0_1EPS)',
             'CALL ML5_0_UPDATE_WL_1_1(WL(1,0,1,2),4,COEFS0,4,4,DWL(1,0,1,3))'])


class TestBornBornColourFactors(unittest.TestCase):
    """born_born_color_factors on a toy colour decomposition."""

    def test_toy(self):
        F = fractions.Fraction
        # two colour flows, matrix [[2, 1/2], [1/2, 2]] (numerators over
        # denominators per line), amplitude 0 in flow 0 with 1, amplitude 1
        # in flow 0 with i and in flow 1 with -1
        num = [[4, 1], [1, 4]]
        den = [2, 2]
        ampb = [[(0, (F(1), False))], [(0, (F(1), True)), (1, (F(-1), False))]]
        cfb = loop_onia.born_born_color_factors(ampb, num, den)
        self.assertEqual(cfb[0][0], (F(2), F(0)))
        # conj(1) * (2 * i + 1/2 * (-1))
        self.assertEqual(cfb[0][1], (F(-1, 2), F(2)))
        self.assertEqual(cfb[1][0], (F(-1, 2), F(-2)))
        # |i|^2 * 2 + 1 * 2 + conj(i)*(-1)*1/2 + (-1)*i*1/2 = 4
        self.assertEqual(cfb[1][1], (F(4), F(0)))


class TestLeptoniumPwave(unittest.TestCase):
    """A P-wave leptonium in a QCD virtual: its constituents have no colour,
    so it has no soft current and no colour links."""

    def test_no_soft_current(self):
        import madgraph.loop.loop_exporters as loop_exporters
        cmd = MGCmd.MasterCmd()
        cmd.no_notification()
        cmd.exec_cmd('import model loop_sm_onia-lepton_masses', printcmd=False)
        cmd.exec_cmd('generate u u~ > Ps(2|3P1) [virt=QCD]', printcmd=False)
        me = loop_helas_objects.LoopHelasProcess(
            cmd._curr_amps, optimized_output=True).get_matrix_elements()[0]
        info = loop_onia.onia_loop_info(me)
        exporter = loop_exporters.LoopProcessOptimizedExporterFortranSA.__new__(
            loop_exporters.LoopProcessOptimizedExporterFortranSA)
        self.assertEqual((info['ipwave'], info['onia'][0]['L']), (1, 1))
        self.assertEqual(loop_onia.pwave_soft_current(me, info, exporter.get_amp_to_jamp_map), [])
        # the only coloured legs are the two quarks, which annihilate
        links = loop_onia.flow_color_links(me, info)
        self.assertEqual(sorted(links), [(1, 2), (2, 1)])
        self.assertTrue(links[(1, 2)])


class TestFortranData(unittest.TestCase):
    """fortran_data: one DATA statement for a small array, one per column for
    a large one."""

    def test_small_array(self):
        self.assertEqual(loop_onia.fortran_data('A', ['1', '2', '3', '4'], (2, 2)),
                         ['DATA A/1,2,3,4/'])

    def test_large_array(self):
        limit = loop_onia.DATA_COLUMN_LIMIT
        try:
            loop_onia.DATA_COLUMN_LIMIT = 4
            values = ['%d' % v for v in range(1, 13)]
            self.assertEqual(loop_onia.fortran_data('A', iter(values), (2, 3, 2)),
                             ['DATA A(1:2,1,1)/1,2/', 'DATA A(1:2,2,1)/3,4/',
                              'DATA A(1:2,3,1)/5,6/', 'DATA A(1:2,1,2)/7,8/',
                              'DATA A(1:2,2,2)/9,10/', 'DATA A(1:2,3,2)/11,12/'])
            self.assertRaises(loop_onia.OniaLoopError, loop_onia.fortran_data,
                              'A', values[:-1], (2, 3, 2))
            self.assertRaises(loop_onia.OniaLoopError, loop_onia.fortran_data,
                              'A', values + ['13'], (2, 3, 2))
        finally:
            loop_onia.DATA_COLUMN_LIMIT = limit


import madgraph.loop.loop_exporters as loop_exporters
import madgraph.various.banner as banner_mod


class TestOniaMadLoopDefaults(unittest.TestCase):
    """MadLoopParams.dat defaults of outputs with bound states: TIRust alone
    (the other libraries are not reliable at threshold) and no helicity
    filter, unless the user set them."""

    class Exporter(object):
        def __init__(self, available):
            self.tir_available_dict = available

    def card(self):
        return banner_mod.MadLoopParam(os.path.join(
            mg_root, 'Template', 'loop_material', 'StandAlone', 'Cards',
            'MadLoopParams.dat'))

    def apply(self, available, card):
        exporter = self.Exporter(available)
        cls = loop_exporters.LoopProcessExporterFortranSA
        exporter.set_onia_reduction_library = \
            cls.set_onia_reduction_library.__get__(exporter)
        return cls.set_onia_madloop_defaults(exporter, card)

    def test_tirust_alone(self):
        card = self.card()
        self.apply({'tirust': True, 'iregi': True, 'collier': True}, card)
        self.assertEqual(str(card['MLReductionLib']), '9')
        self.assertEqual(card['HelicityFilterLevel'], 0)

    def test_without_tirust(self):
        card = self.card()
        before = card['MLReductionLib']
        self.apply({'tirust': False, 'iregi': True}, card)
        self.assertEqual(card['MLReductionLib'], before)
        self.assertEqual(card['HelicityFilterLevel'], 0)

    def test_user_choice_is_kept(self):
        card = self.card()
        card.set('MLReductionLib', '7', user=True)
        card.set('HelicityFilterLevel', 2, user=True)
        self.apply({'tirust': True}, card)
        self.assertEqual(str(card['MLReductionLib']), '7')
        self.assertEqual(card['HelicityFilterLevel'], 2)


    def test_extended_precision_default(self):
        # off by default (#PWaveExtendedPrecision, onia_ep.py)
        self.assertIs(self.card()['PWaveExtendedPrecision'], False)


import madgraph.loop.onia_ep as onia_ep


class TestOniaExtendedPrecision(unittest.TestCase):
    """The extended-precision (EP) code of a P-wave output, derived from its
    quadruple-precision (MP) code (madgraph/loop/onia_ep.py)."""

    MIXED = '\n'.join([
        '      SUBROUTINE ML5_0_ONIA_PREPARE(P)',
        '      REAL*8 P(0:3,5)',
        '      END',
        'C     The MP version',
        '      SUBROUTINE ML5_0_MP_ONIA_PREPARE(P)',
        '      USE ML5_0_PW_DUAL, ONLY: MPPWAMP, ML5_0_PW_DUAL_INIT',
        '      IMPLICIT NONE',
        '      REAL*16 P(0:3,5), X',
        '      COMPLEX*32 C',
        "      INCLUDE 'mp_coupl.inc'",
        '      COMMON/ML5_0_MP_PW_EPS/C',
        '      X=1.0E0_16+REAL(C,KIND=16)+1.0E',
        '     $ -8_16',
        '      CALL MP_UPDATE_AS_PARAM()',
        '      CALL ML5_0_MP_ONIA_CG(1,X)',
        '      END',
        '      REAL*16 FUNCTION ML5_0_MP_MDOT(P1,P2)',
        '      REAL*16 P1(0:3),P2(0:3)',
        '      ML5_0_MP_MDOT=P1(0)*P2(0)',
        '      CONTAINS',
        '      ELEMENTAL LOGICAL FUNCTION FINITE(X)',
        '      REAL*16, INTENT(IN) :: X',
        '      FINITE=X.EQ.X',
        '      END FUNCTION',
        '      END',
        '      SUBROUTINE ML5_0_MP_TIRUST_LOOP(N)',
        '      END',
        '      SUBROUTINE MP_ML5_0_UPDATE_WL_0_1(A)',
        '      COMPLEX*32 A',
        '      A=(0Q0,1Q0)',
        '      END'])

    def test_renaming(self):
        text = onia_ep.to_ep('      CALL ML5_0_MP_ONIA_CG(1,MPPWW,mp_dual,SET_MP_PS)\n'
                             '      X=1.0E0_16+REAL(Y,KIND=16)*2Q0\n'
                             '      REAL*16 MP__ZERO\n      COMPLEX*32 COUP\n'
                             '      Z=qcmplx(1)+qreal(2)\n'
                             '      CALL MP_UPDATE_AS_PARAM()\n', 'ML5_0_')
        self.assertEqual(text,
                         '      CALL ML5_0_EP_ONIA_CG(1,EPPWW,ep_dual,SET_EP_PS)\n'
                         '      X=1.0E0_EPK+REAL(Y,KIND=EPK)*2E0_EPK\n'
                         '      REAL(EPK) EP__ZERO\n      COMPLEX(EPK) COUP\n'
                         '      Z=ecmplx(1)+ereal(2)\n'
                         '      CALL ML5_0_EP_UPDATE_AS_PARAM()\n')
        # names that only contain MP, and DP names, are kept
        same = '      CALL ML5_0_COMPUTE_ACCURACY(COMP_X,DP_AMP,ML5_0_MPLOOPNUM)\n'
        self.assertEqual(onia_ep.to_ep(same), same)

    def test_mp_units(self):
        # the MP routines with an EP version: not the DP ones, nor TIRust's
        # (the template has its own EP routines); internal procedures stay
        # with their host
        names = [name for name, _ in onia_ep.mp_units(self.MIXED)]
        self.assertEqual(names, ['ML5_0_MP_ONIA_PREPARE', 'ML5_0_MP_MDOT',
                                 'MP_ML5_0_UPDATE_WL_0_1'])
        units = dict(onia_ep.mp_units(self.MIXED))
        self.assertTrue(units['ML5_0_MP_ONIA_PREPARE'].startswith('C     The MP version\n'))
        self.assertTrue(units['ML5_0_MP_MDOT'].endswith('      END FUNCTION\n      END'))

    def test_ep_source(self):
        text = '\n'.join(unit for _, unit in onia_ep.mp_units(self.MIXED))
        ep = onia_ep.ep_source(text, 'test', 'ML5_0_')
        lines = ep.split('\n')
        # the kind module in each top-level unit, not in internal procedures
        self.assertEqual(lines.count('      USE EP_PRECISION'), 3)
        self.assertEqual(lines[lines.index('      SUBROUTINE ML5_0_EP_ONIA_PREPARE(P)') + 1],
                         '      USE EP_PRECISION')
        self.assertEqual(lines[lines.index('      REAL(EPK) FUNCTION ML5_0_EP_MDOT(P1,P2)') + 1],
                         '      USE EP_PRECISION')
        self.assertIn('      USE ML5_0_PW_DUAL, ONLY: EPPWAMP, ML5_0_PW_DUAL_INIT', lines)
        self.assertIn("      INCLUDE 'ep_coupl.inc'", lines)
        self.assertIn('      COMMON/ML5_0_EP_PW_EPS/C', lines)
        self.assertIn('     $ -8_EPK', lines)
        self.assertIn('      A=(0E0_EPK,1E0_EPK)', lines)
        self.assertIn('      CALL ML5_0_EP_UPDATE_AS_PARAM()', lines)

    def test_leftover_is_refused(self):
        # a construct the renaming misses: an integer literal of kind 16
        self.assertRaises(onia_ep.OniaEPError, onia_ep.ep_source,
                          '      SUBROUTINE MP_X()\n      I=1_16\n      END\n', 'test')
        # comments and strings are not checked
        onia_ep.check_ep("C     REAL*16\n      WRITE(*,*) 'MP_X'\n", 'test')

    def test_aloha_object(self):
        import madgraph.iolibs.export_v4 as export_v4
        text = onia_ep.ep_aloha_object(export_v4.ProcessExporterFortran.LOOP_DUAL_ALOHA_OBJECT)
        self.assertIn('      use mp_dual_variables\n      use ep_dual_variables\n', text)
        for name in ('EP_ALOHA', 'EP_ALOHA2D'):
            self.assertIn('TYPE %s\n' % name, text)
            self.assertIn('END TYPE %s\n' % name, text)
        self.assertEqual(text.count('type(EP_Dual)'), 4)
        self.assertTrue(text.rstrip().endswith('end module ALOHA_OBJECT'))

    def test_dual_variables(self):
        path = os.path.join(mg_root, 'aloha', 'template_files', 'mp_dual_variables.f')
        with open(path) as stream:
            text = onia_ep.ep_dual_variables(stream.read())
        self.assertTrue(text.startswith(onia_ep.KIND_MODULE))
        self.assertIn('      module ep_dual_variables\n      use ep_precision, only: epk\n', text)
        self.assertIn('      integer, parameter :: qp = epk\n', text)
        self.assertIn('public :: size, ecmplx, ereal', text)
        self.assertNotIn('mp_', text.lower().replace('mp_dual_variables', ''))

    def test_aloha_functions(self):
        # the EP external wavefunctions, from the MP ones of the template
        path = os.path.join(mg_root, 'aloha', 'template_files', 'aloha_functions_mp_dual.f')
        with open(path) as stream:
            ep = onia_ep.ep_source(stream.read(), 'aloha_functions_mp_dual.f')
        for name in ('ixxxxx', 'oxxxxx', 'vxxxxx'):
            self.assertIn('      subroutine ep_%s(' % name, ep)

    def test_tirust_without_extended_inputs(self):
        self.assertFalse(onia_ep.tirust_supports_extended_inputs(''))
        tmp = tempfile.mkdtemp()
        try:
            self.assertFalse(onia_ep.tirust_supports_extended_inputs(tmp))
        finally:
            shutil.rmtree(tmp)

    def test_subprocesses_makefile(self):
        path = os.path.join(mg_root, 'Template', 'loop_material', 'StandAlone',
                            'SubProcesses', 'makefile')
        tmp = tempfile.mkdtemp()
        try:
            target = os.path.join(tmp, 'makefile')
            shutil.copy(path, target)
            onia_ep.patch_subprocesses_makefile(target)
            with open(target) as stream:
                text = stream.read()
            self.assertIn('$(wildcard ep_*.f)', text)
            self.assertIn('$(wildcard $(LOOP_PREFIX)*/ep_*.f)', text)
            self.assertIn('ep_polynomial.o : polynomial.o\n', text)
            # once only
            onia_ep.patch_subprocesses_makefile(target)
            with open(target) as stream:
                self.assertEqual(stream.read(), text)
        finally:
            shutil.rmtree(tmp)
