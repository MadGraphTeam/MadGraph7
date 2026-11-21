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
        """Only [virt=QCD] with S-wave bound states and at most one P-wave
        bound state is accepted."""
        refused = ['g g > etac(1|1S01) [QCD]',
                   'g g > etac(1|1S01) [real=QCD]',
                   'g g > etac(1|1S01) [sqrvirt=QCD]',
                   'g g > etac(1|1S01) [virt=QED]',
                   'g g > chic0(1|3P01) chic2(1|3P21) [virt=QCD]']
        for process in refused:
            self.assertRaises(madgraph.InvalidCmd, lambda p=process: self.cmd.exec_cmd(
                              'generate %s' % p, printcmd=False, errorhandling=False))

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
