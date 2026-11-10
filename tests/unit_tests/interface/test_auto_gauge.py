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
# For more information, visit madgraph.phys.ucl.ac.be and amcatnlo.web.cern.ch
#
################################################################################
"""The automatic choice of the FD gauge for a purely electroweak process
(MadGraphCmd.apply_automatic_gauge and fd_gauge_criterion)."""

from __future__ import absolute_import
import glob
import os
import shutil
import tempfile
import unittest

import aloha
import madgraph
import madgraph.interface.master_interface as master_interface
import madgraph.interface.madgraph_interface as mg_interface
import madgraph.models.import_ufo as import_ufo

pjoin = os.path.join
MG5DIR = madgraph.MG5DIR


class TestAutomaticGauge(unittest.TestCase):
    """Which processes get the FD gauge, how the user overrides it, and what
    the switch leaves behind (history, amplitudes)."""

    def setUp(self):
        self.cmd = master_interface.MasterCmd()
        self.cmd.no_notification()
        self.do('import model sm')

    def tearDown(self):
        # the gauge is a module-level state of ALOHA: leave the default
        self.cmd.exec_cmd('set gauge unitary', precmd=True, postcmd=True)
        aloha.unitary_gauge = True

    def do(self, line):
        """run a line as the user types it (history and post_set included)"""
        self.cmd.exec_cmd(line, precmd=True, postcmd=True)

    def assertGauge(self, gauge, automatic=None):
        self.assertEqual(self.cmd.options['gauge'], gauge)
        self.assertEqual(self.cmd._curr_model._curr_gauge, gauge)
        self.assertEqual(aloha.unitary_gauge, {'unitary': True, 'FD': 3}[gauge])
        for amp in mg_interface.iter_tree_amplitudes(self.cmd._curr_amps):
            self.assertIs(amp.get('process').get('model'), self.cmd._curr_model)
        if automatic is not None:
            self.assertEqual(self.cmd._auto_gauge_active, automatic)

    def gauge_lines(self):
        """the history lines about the gauge, with the process lines"""
        return [line for line in self.cmd.history
                if line.startswith(('set gauge', '# FD gauge', '# --gauge', 'generate',
                                    'add process'))]

    # ------------------------------------------------------------------
    # the criterion
    # ------------------------------------------------------------------
    def test_criterion(self):
        """fd_gauge_criterion on the diagrams of SM processes: a massive
        vector propagator with an end that is not a conserved current"""

        self.do('set auto_gauge False')   # the amplitudes as generated
        expected = [
            ('p p > w+ w- j j QCD=0', True),      # VBS: VVV and HVV ends
            ('p p > w+ w-', True),                # s-channel Z/a: WWV end
            ('p p > h j j QCD=0', True),          # VBF: HVV end
            ('e+ e- > z h', True),                # ZZH end
            ('g b > t e- ve~', True),             # W between t-b and leptons
            ('p p > t t~, (t > b w+, w+ > l+ vl)', True),  # decayed W: t-b end
            ('p p > w+ w-, w+ > e+ ve', True),    # decay chain of the above
            ('e+ e- > ta+ ta-', True),            # massive tau current
            ('e+ e- > mu+ mu-', False),           # Z between massless currents
            ('u d~ > e+ ve g', False),            # quark line + gluon: conserved
            ('p p > z z', False),                 # external Z only
            ('p p > w+ j', False),                # external W only
            ('p p > t t~', False),                # no massive vector
            ('a a > mu+ mu-', False),             # no massive vector
            ]
        for line, result in expected:
            self.do('generate %s' % line)
            self.assertEqual(
                mg_interface.fd_gauge_criterion(self.cmd._curr_amps), result, line)
            self.assertGauge('unitary', automatic=False)

    def test_criterion_loop_amplitude(self):
        """a loop amplitude never qualifies"""

        self.do('import model loop_sm')
        self.do('generate u u~ > w+ w- [virt=QCD]')
        self.assertFalse(mg_interface.fd_gauge_criterion(self.cmd._curr_amps))
        self.assertEqual(self.cmd.options['gauge'], 'unitary')

    # ------------------------------------------------------------------
    # the switch
    # ------------------------------------------------------------------
    def test_switch_to_fd_and_back(self):
        """a purely electroweak process is generated in the FD gauge, a QCD
        one in the unitary gauge, and the history (the proc_card) says so"""

        self.do('generate p p > w+ w-')
        self.assertGauge('FD', automatic=True)
        self.assertEqual(len(self.cmd._curr_amps), 1)
        # the FD model: goldstones merged into the vectors, none left
        self.assertFalse([p for p in self.cmd._curr_model.get('particles')
                          if p.get('pdg_code') in [250, 251]])
        self.assertEqual(self.gauge_lines(),
                         [mg_interface.MadGraphCmd.AUTO_GAUGE_COMMENT,
                          'set gauge FD', 'generate p p > w+ w-'])
        # the first process keeps its number
        self.assertEqual(self.cmd._curr_amps[0].get('process').get('id'), 1)

        self.do('generate p p > t t~')
        self.assertGauge('unitary', automatic=False)
        self.assertEqual(len(self.cmd._curr_amps), 2)
        self.assertEqual(self.gauge_lines(), ['generate p p > t t~'])

    def lines_gauges(self):
        """(process id, gauge of the process, gauge of its model)"""
        return [(amp.get('process').get('id'), amp.get('process').get_gauge(),
                 amp.get('process').get('model')._curr_gauge)
                for amp in self.cmd._curr_amps]

    def test_add_process_mixes_gauges(self):
        """a process that does not need the FD gauge added to one that does:
        the session goes back to the unitary gauge, the first line is
        generated with --gauge=FD, and the proc_card says so"""

        self.do('generate p p > w+ w-')
        self.assertGauge('FD', automatic=True)
        self.do('add process p p > t t~')
        self.assertEqual(self.cmd.options['gauge'], 'unitary')
        self.assertEqual(self.cmd._auto_fd_lines, set([0]))
        self.assertTrue(self.cmd.automatic_fd_gauge_in_use())
        self.assertEqual(self.lines_gauges(), [(1, 'FD', 'FD'), (2, '', 'unitary'),
                                               (2, '', 'unitary')])
        self.assertEqual(self.gauge_lines(),
                         [mg_interface.MadGraphCmd.AUTO_LINE_COMMENT,
                          'generate p p > w+ w- --gauge=FD',
                          'add process p p > t t~'])
        # a polarised one stays in the unitary gauge, the rest is kept
        self.do('add process p p > w+{0} w-')
        self.assertEqual(self.cmd._auto_fd_lines, set([0]))
        self.assertEqual(self.lines_gauges()[-1], (3, '', 'unitary'))

    def test_add_process_needing_fd(self):
        """a process that needs the FD gauge added to one that does not: only
        that line is generated (again) in the FD gauge"""

        self.do('generate p p > t t~')
        self.assertGauge('unitary', automatic=False)
        first = list(self.cmd._curr_amps)
        self.do('add process p p > w+ w-')
        self.assertEqual(self.cmd.options['gauge'], 'unitary')
        self.assertEqual(self.cmd._auto_fd_lines, set([1]))
        self.assertEqual(self.lines_gauges(), [(1, '', 'unitary'), (1, '', 'unitary'),
                                               (2, 'FD', 'FD')])
        # the first line is not generated again
        self.assertIs(self.cmd._curr_amps[0], first[0])
        self.assertIn('add process p p > w+ w- --gauge=FD', self.cmd.history)

    def test_single_gauge_output_fallback(self):
        """an output writing every matrix element into one library gets one
        gauge: FD when every process can be, unitary otherwise"""

        self.do('generate p p > t t~')
        self.do('add process p p > w+ w-')
        self.cmd._export_format = 'pythia8'
        self.cmd._export_dir = None
        self.cmd.single_gauge_for_output([])
        self.assertGauge('FD', automatic=True)
        self.assertEqual(self.cmd._auto_fd_lines, set())
        self.assertEqual(len(self.cmd._curr_amps), 3)

        self.do('generate p p > w+{0} w-')
        self.do('add process p p > w+ w-')
        self.assertEqual(self.cmd._auto_fd_lines, set([1]))
        self.cmd._export_format = 'pythia8'
        self.cmd.single_gauge_for_output([])
        self.assertGauge('unitary', automatic=False)
        self.assertEqual(self.cmd._auto_fd_lines, set())

    def test_decay_chain(self):
        """a decay chain is generated again as a decay chain"""

        self.do('generate p p > w+ w-, w+ > e+ ve, w- > e- ve~')
        self.assertGauge('FD', automatic=True)
        self.assertEqual(len(self.cmd._curr_amps), 1)
        self.assertEqual(len(self.cmd._curr_amps[0].get('decay_chains')), 2)

    def test_no_massive_vector(self):
        """nothing to gain for a process without massive vector boson"""

        self.do('generate a a > mu+ mu-')
        self.assertGauge('unitary', automatic=False)

    def test_polarised_legs_keep_unitary(self):
        """the split of a cross section into polarisations is gauge
        dependent, defined in the unitary gauge"""

        self.do('generate p p > w+{0} w-{T}')
        self.assertGauge('unitary', automatic=False)
        self.do('generate p p > w+ w-, w+ > e+{L} ve')
        self.assertGauge('unitary', automatic=False)

    def test_propagator_polarization(self):
        """{G} & co are pieces of the unitary-gauge propagator"""

        self.do('generate p p > w+{G} w-, w+ > e+ ve')
        self.assertGauge('unitary', automatic=False)

    # ------------------------------------------------------------------
    # the user's word
    # ------------------------------------------------------------------
    def test_multiparticles_survive_the_switch(self):
        """the 'define' lines of the user, the 'z|a' ones included, survive
        the reload of the model behind the automatic choice"""

        self.do('define V z|a')
        self.do('define VQ = z u')
        self.do('define p = p b b~')
        self.do('generate e+ e- > V > w+ w-')
        self.assertGauge('FD', True)
        self.assertEqual(self.cmd._multiparticles['v'], [[23], [22]])
        self.do('generate p p > t t~')
        self.assertGauge('unitary', False)
        self.assertEqual(self.cmd._multiparticles['v'], [[23], [22]])
        self.assertEqual(sorted(self.cmd._multiparticles['vq']), [2, 23])
        self.assertIn(5, self.cmd._multiparticles['p'])
        self.do('define V2 = w+ V')
        self.assertEqual(self.cmd._multiparticles['v2'], [[24, 23], [24, 22]])

    def test_spin_two_keeps_unitary_gauge(self):
        """a spin-2 particle in the diagrams: the FD gauge is not validated
        for it (p p > w+ y in loop_smgrav is wrong by orders of magnitude)"""

        self.do('import model %s' % pjoin(MG5DIR, 'tests', 'input_files',
                                            'loop_smgrav'))
        self.do('generate u d~ > w+ y')
        self.assertGauge('unitary', False)
        self.assertFalse(mg_interface.fd_gauge_criterion(self.cmd._curr_amps))

    def test_auto_gauge_not_in_the_proc_card(self):
        """the default auto_gauge True is not dumped into the history (MadSpin
        and the reweighting replay the 'set' lines of the proc_card with
        their own auto_gauge False); a choice of the user is"""

        self.assertFalse([l for l in self.cmd.history
                          if l.startswith('set auto_gauge')])
        self.do('set auto_gauge False')
        self.assertIn('set auto_gauge False', list(self.cmd.history))

    def test_madspin_skips_the_gauge_lines(self):
        """MadSpin runs the 'set' lines of the proc_card but the gauge ones"""

        import MadSpin.interface_madspin as interface_madspin
        is_gauge = interface_madspin.set_line_is_gauge
        self.assertTrue(is_gauge('set gauge FD'))
        self.assertTrue(is_gauge('set auto_gauge True'))
        self.assertFalse(is_gauge('set complex_mass_scheme True'))
        self.assertFalse(is_gauge('set gauge_fake 1'))

    def test_amplicol_process_survives_the_revert(self):
        """an [amplicol] process added after an automatic FD choice: the
        session goes back to the unitary gauge, the first line keeps the FD
        gauge, and the [amplicol] process is kept"""

        self.do('generate p p > w+ w-')
        self.assertGauge('FD', True)
        self.do('add process g g > g g [amplicol]')
        self.assertEqual(self.cmd.options['gauge'], 'unitary')
        self.assertEqual(self.cmd._auto_fd_lines, set([0]))
        self.assertEqual(len(self.cmd._curr_amps), 1)
        self.assertEqual(self.cmd._curr_amps[0].get('process').get_gauge(), 'FD')
        self.assertEqual(len(self.cmd._curr_amplicol_defs), 1)

    def test_explicit_gauge_wins(self):
        """'set gauge unitary' keeps the unitary gauge, 'set gauge FD' the FD
        one whatever the process, 'set gauge default' hands the choice back"""

        self.do('set gauge unitary')
        self.assertTrue(self.cmd._explicit_gauge)
        self.do('generate p p > w+ w-')
        self.assertGauge('unitary', automatic=False)
        self.assertEqual(self.gauge_lines(),
                         ['set gauge unitary', 'generate p p > w+ w-'])

        self.do('set gauge FD')
        self.do('generate p p > t t~')
        self.assertGauge('FD', automatic=False)

        self.do('set gauge default')
        self.assertFalse(self.cmd._explicit_gauge)
        self.do('generate p p > w+ w-')
        self.assertGauge('FD', automatic=True)
        self.do('generate p p > t t~')
        self.assertGauge('unitary', automatic=False)

    def test_auto_gauge_option(self):
        """'set auto_gauge False' turns the choice off, and drops an FD gauge
        chosen automatically on the spot"""

        self.do('set auto_gauge False')
        self.assertIs(self.cmd.options['auto_gauge'], False)
        self.do('generate p p > w+ w-')
        self.assertGauge('unitary', automatic=False)
        self.assertEqual(self.gauge_lines(), ['generate p p > w+ w-'])

        self.do('set auto_gauge True')
        self.do('generate p p > w+ w-')
        self.assertGauge('FD', automatic=True)
        self.do('set auto_gauge False')
        self.assertEqual(self.cmd.options['gauge'], 'unitary')
        self.assertFalse(self.cmd._auto_gauge_active)
        self.assertEqual(aloha.unitary_gauge, True)
        # the processes generated again in the unitary gauge
        self.assertGauge('unitary', automatic=False)
        self.assertEqual(len(self.cmd._curr_amps), 1)
        self.assertNotIn('set gauge FD', self.cmd.history)

    def test_model_without_feynman_gauge(self):
        """a model without goldstone boson (gauge = [0]) has no FD gauge"""

        self.do('import model %s' % pjoin(MG5DIR, 'tests', 'input_files',
                                            'fourfermion_UFO'))
        self.assertEqual(self.cmd._curr_model.get('gauge'), [0])
        self.do('generate fa fa~ > fb fb~')
        # V between massive fermions: the criterion holds, the model does not
        self.assertTrue(mg_interface.fd_gauge_criterion(self.cmd._curr_amps))
        self.assertGauge('unitary', automatic=False)

    def test_unverified_model_is_given_up(self):
        """an FD model whose goldstone content import_ufo could not check
        goes back to the unitary gauge, once for all for that model"""

        self.cmd.fd_gauge_supported_by_model = lambda model: False
        self.do('generate p p > w+ w-')
        self.assertGauge('unitary', automatic=False)
        self.assertEqual(len(self.cmd._curr_amps), 1)
        self.assertIn('sm', self.cmd._fd_unverified_models)
        self.assertNotIn('set gauge FD', self.cmd.history)
        # not tried again: no reload of the model this time
        model = self.cmd._curr_model
        self.do('generate p p > w+ w-')
        self.assertIs(self.cmd._curr_model, model)
        self.assertGauge('unitary', automatic=False)

    def test_fd_gauge_supported_by_model(self):
        """the tag import_ufo leaves on a model it could not check"""

        model = self.cmd._curr_model
        self.assertTrue(self.cmd.fd_gauge_supported_by_model(model))
        model.get('limitations').append(import_ufo.FD_GAUGE_UNVERIFIED)
        try:
            self.assertFalse(self.cmd.fd_gauge_supported_by_model(model))
        finally:
            model.get('limitations').remove(import_ufo.FD_GAUGE_UNVERIFIED)

    def test_nlo_process_keeps_gauge(self):
        """an NLO (FKS) generation never goes through the automatic choice"""

        self.do('import model loop_sm')
        self.do('generate u u~ > w+ w- [QCD]')
        self.assertEqual(self.cmd.options['gauge'], 'unitary')
        self.assertEqual(aloha.unitary_gauge, True)
        self.assertFalse(self.cmd._auto_gauge_active)

    def test_loop_process_after_automatic_fd(self):
        """a loop process (only supported in the unitary/Feynman gauge) after
        a tree-level one generated in the automatic FD gauge: the FD gauge
        steps aside instead of refusing the process"""

        self.do('import model loop_sm')
        self.do('generate u u~ > w+ w-')
        self.assertGauge('FD', automatic=True)
        self.do('generate u u~ > w+ w- [virt=QCD]')
        self.assertEqual(self.cmd.options['gauge'], 'unitary')
        self.assertEqual(aloha.unitary_gauge, True)
        self.assertFalse(self.cmd._auto_gauge_active)
        self.assertEqual(len(self.cmd._curr_amps), 1)
        self.assertNotIn('set gauge FD', self.cmd.history)

    def test_import_model_without_feynman_gauge_after_automatic_fd(self):
        """importing a model without goldstone boson while in the automatic FD
        gauge: the model is loaded in the unitary gauge"""

        self.do('generate p p > w+ w-')
        self.assertGauge('FD', automatic=True)
        self.do('import model %s' % pjoin(MG5DIR, 'tests', 'input_files',
                                            'fourfermion_UFO'))
        self.assertIsNotNone(self.cmd._curr_model)
        self.assertEqual(self.cmd._curr_model.get('gauge'), [0])
        self.assertEqual(self.cmd.options['gauge'], 'unitary')
        self.assertEqual(aloha.unitary_gauge, True)
        self.do('generate fa fa~ > fb fb~')
        self.assertGauge('unitary', automatic=False)


class TestAutomaticGaugeOutput(unittest.TestCase):
    """What an output records about an FD gauge chosen automatically."""

    def setUp(self):
        self.cmd = master_interface.MasterCmd()
        self.cmd.no_notification()
        self.path = tempfile.mkdtemp(prefix='auto_gauge')
        self.do('import model sm')

    def tearDown(self):
        self.cmd.exec_cmd('set gauge unitary', precmd=True, postcmd=True)
        aloha.unitary_gauge = True
        shutil.rmtree(self.path, ignore_errors=True)

    def do(self, line):
        self.cmd.exec_cmd(line, precmd=True, postcmd=True)

    def test_madevent_output_records_fd_gauge(self):
        """proc_characteristics, proc_card and the citations of the output"""

        out = pjoin(self.path, 'WPWM')
        self.do('generate p p > w+ w-')
        self.do('output madevent %s -f' % out)
        with open(pjoin(out, 'SubProcesses', 'proc_characteristics')) as fsock:
            self.assertIn('gauge = FD', fsock.read())
        with open(pjoin(out, 'Cards', 'proc_card_mg5.dat')) as fsock:
            lines = [l.strip() for l in fsock if l.strip() and
                     not l.startswith('#*')]
        self.assertIn('set gauge FD', lines)
        self.assertLess(lines.index('set gauge FD'),
                        lines.index('generate p p > w+ w-'))
        self.assertNotIn('set gauge unitary', lines)
        with open(pjoin(out, 'citations.log')) as fsock:
            self.assertIn('Hagiwara:2024xdh', fsock.read())

    def test_postponed_output_fixes_the_gauge(self):
        """an 'output --postpone_model' waits for its model: the processes
        written in it fix the gauge of the session until it is written"""

        # 'output --postpone_model' is not usable with the flavour
        # grouping (finalize cannot merge its FLV_Coupling objects)
        self.do('set apply_flavor_grouping False')
        self.do('import model sm')
        out = pjoin(self.path, 'MIX')
        self.do('generate p p > t t~')
        self.do('output standalone %s --postpone_model' % out)
        self.assertTrue(self.cmd.postponed_export_pending())
        # a purely electroweak process keeps the unitary gauge of that output
        self.do('generate p p > w+ w-')
        self.assertEqual(self.cmd.options['gauge'], 'unitary')
        self.assertFalse(self.cmd._auto_gauge_active)
        self.do('output standalone %s -f' % out)
        self.assertFalse(self.cmd.postponed_export_pending())
        self.assertEqual(self.cmd.options['gauge'], 'unitary')
        # the model is written: the choice is free again
        self.do('generate p p > w+ w-')
        self.assertEqual(self.cmd.options['gauge'], 'FD')
        self.assertTrue(self.cmd._auto_gauge_active)

    def test_postponed_output_keeps_automatic_fd_gauge(self):
        """the FD gauge chosen for the first output is kept for a QCD
        process added to it, and stays recorded in the proc_card"""

        # 'output --postpone_model' is not usable with the flavour
        # grouping (finalize cannot merge its FLV_Coupling objects)
        self.do('set apply_flavor_grouping False')
        self.do('import model sm')
        out = pjoin(self.path, 'MIX')
        self.do('generate p p > w+ w-')
        self.assertEqual(self.cmd.options['gauge'], 'FD')
        self.do('output standalone %s --postpone_model' % out)
        self.do('generate p p > t t~')
        self.assertEqual(self.cmd.options['gauge'], 'FD')
        self.assertTrue(self.cmd._auto_gauge_active)
        self.do('output standalone %s -f' % out)
        self.assertFalse(self.cmd.postponed_export_pending())
        # the history is what the proc_card of the output is written from
        lines = [l.strip() for l in self.cmd.history]
        self.assertIn('set gauge FD', lines)
        self.assertLess(lines.index('set gauge FD'),
                        lines.index('generate p p > t t~'))

    def test_loop_process_after_postponed_fd_output(self):
        """a loop process cannot follow an FD-gauge output waiting for its
        model: refused, as is 'set auto_gauge False'; an explicit 'set gauge'
        drops that output"""

        # 'output --postpone_model' is not usable with the flavour
        # grouping (finalize cannot merge its FLV_Coupling objects)
        self.do('set apply_flavor_grouping False')
        self.do('import model sm')
        out = pjoin(self.path, 'MIX')
        self.do('generate p p > w+ w-')
        self.do('output standalone %s --postpone_model' % out)
        self.assertRaises(madgraph.InvalidCmd, self.do,
                          'generate u u~ > u u~ [virt=QCD]')
        self.assertEqual(self.cmd.options['gauge'], 'FD')
        self.assertTrue(self.cmd.postponed_export_pending())
        self.assertRaises(madgraph.InvalidCmd, self.do, 'set auto_gauge False')
        self.assertTrue(self.cmd.options['auto_gauge'])
        self.do('set gauge unitary')
        self.assertFalse(self.cmd.postponed_export_pending())
        self.assertEqual(self.cmd.options['gauge'], 'unitary')

    def test_axial_gauge_output_reverts(self):
        """--axial_gauge=True has no room in the FD gauge: an automatic FD
        gauge steps aside, the processes are generated again"""

        out = pjoin(self.path, 'WPWM')
        self.do('generate p p > w+ w-')
        self.assertEqual(self.cmd.options['gauge'], 'FD')
        self.do('output standalone_fortran %s --axial_gauge=True -f' % out)
        self.assertEqual(self.cmd.options['gauge'], 'unitary')
        self.assertEqual(aloha.unitary_gauge, True)
        self.assertFalse(self.cmd._auto_gauge_active)
        self.assertEqual(len(self.cmd._curr_amps), 1)
        with open(pjoin(out, 'Cards', 'proc_card_mg5.dat')) as fsock:
            self.assertNotIn('set gauge FD', fsock.read())
        with open(pjoin(out, 'citations.log')) as fsock:
            self.assertNotIn('Hagiwara:2024xdh', fsock.read())


class TestProcessGauge(unittest.TestCase):
    """'generate|add process ... --gauge=X': one gauge per process line."""

    def setUp(self):
        self.cmd = master_interface.MasterCmd()
        self.cmd.no_notification()
        self.do('set auto_gauge False')
        self.do('import model sm')

    def tearDown(self):
        self.cmd.exec_cmd('set gauge unitary', precmd=True, postcmd=True)
        aloha.unitary_gauge = True

    def do(self, line):
        self.cmd.exec_cmd(line, precmd=True, postcmd=True)

    def gauges(self):
        return [(amp.get('process').get_gauge('unitary'),
                 amp.get('process').get('model')._curr_gauge)
                for amp in mg_interface.iter_tree_amplitudes(self.cmd._curr_amps)]

    def test_two_gauges_in_one_session(self):
        """each line generated with the model of its gauge, the session and
        aloha staying in the unitary gauge"""

        self.do('generate p p > w+ w- --gauge=FD')
        self.do('add process p p > t t~')
        self.do('add process p p > z z, z > e+ e- --gauge=FD')
        self.assertEqual(self.gauges(), [('FD', 'FD'), ('unitary', 'unitary'),
                                         ('unitary', 'unitary'), ('FD', 'FD'),
                                         ('FD', 'FD')])
        self.assertEqual(self.cmd.options['gauge'], 'unitary')
        self.assertEqual(aloha.unitary_gauge, True)
        self.assertEqual(self.cmd.gauges_in_use(), set(['unitary', 'FD']))
        self.assertIn('generate p p > w+ w- --gauge=FD', self.cmd.history)
        self.assertEqual(self.cmd._generate_info, 'p p > w+ w-')

    def test_twin_model_reused(self):
        """the FD model is imported once per session model"""

        self.do('generate p p > w+ w- --gauge=FD')
        twin = self.cmd._curr_amps[0].get('process').get('model')
        self.do('generate p p > z z --gauge=FD')
        self.assertIs(self.cmd._curr_amps[0].get('process').get('model'), twin)
        self.do('import model sm')
        self.do('generate p p > z z --gauge=FD')
        self.assertIsNot(self.cmd._curr_amps[0].get('process').get('model'), twin)

    def test_same_gauge_option_is_plain(self):
        """--gauge= of the session gauge changes nothing"""

        self.do('generate p p > w+ w- --gauge=unitary')
        self.assertEqual(self.gauges(), [('unitary', 'unitary')])
        self.assertIs(self.cmd._curr_amps[0].get('process').get('model'),
                      self.cmd._curr_model)

    def test_refusals(self):
        """loop processes, spin 2, unknown gauges, a model without FD gauge"""

        self.assertRaises(madgraph.InvalidCmd, self.do, 'generate p p > w+ w- --gauge=axial')
        self.assertRaises(madgraph.InvalidCmd, self.do,
                          'generate u u~ > w+ w- [virt=QCD] --gauge=FD')
        self.do('import model %s' % pjoin(MG5DIR, 'tests', 'input_files',
                                            'fourfermion_UFO'))
        self.assertRaises(madgraph.InvalidCmd, self.do,
                          'generate fa fa~ > fb fb~ --gauge=FD')
        self.do('import model %s' % pjoin(MG5DIR, 'tests', 'input_files',
                                            'loop_smgrav'))
        self.assertRaises(madgraph.InvalidCmd, self.do, 'generate u d~ > w+ y --gauge=FD')
        self.assertFalse(self.cmd._curr_amps)

    def test_decay_chain_gauge(self):
        """the decays of an FD line are FD too"""

        self.do('generate p p > w+ w-, w+ > e+ ve --gauge=FD')
        self.assertEqual(set(self.gauges()), set([('FD', 'FD')]))


class TestProcessGaugeOutput(unittest.TestCase):
    """What an output of processes in two gauges looks like."""

    def setUp(self):
        self.cmd = master_interface.MasterCmd()
        self.cmd.no_notification()
        self.path = tempfile.mkdtemp(prefix='process_gauge')
        self.do('set auto_gauge False')
        self.do('import model sm')
        self.do('generate p p > w+ w- --gauge=FD')
        self.do('add process p p > t t~')

    def tearDown(self):
        self.cmd.exec_cmd('set gauge unitary', precmd=True, postcmd=True)
        aloha.unitary_gauge = True
        shutil.rmtree(self.path, ignore_errors=True)

    def do(self, line):
        self.cmd.exec_cmd(line, precmd=True, postcmd=True)

    def test_madevent_output(self):
        """the FD P directory links Source/DHELAS_FD, the others DHELAS"""

        out = pjoin(self.path, 'MIX')
        self.do('output madevent %s -f' % out)
        self.assertEqual(aloha.unitary_gauge, True)
        helas_fd = pjoin(out, 'Source', 'DHELAS_FD')
        self.assertTrue(os.path.exists(pjoin(helas_fd, 'aloha_functions.f')))
        with open(pjoin(helas_fd, 'aloha_functions.f')) as fsock:
            self.assertIn('DEFINE_GAUGE_DIR', fsock.read().upper())
        with open(pjoin(out, 'Source', 'DHELAS', 'aloha_functions.f')) as fsock:
            self.assertNotIn('DEFINE_GAUGE_DIR', fsock.read().upper())
        with open(pjoin(helas_fd, 'makefile')) as fsock:
            self.assertIn('libdhelas_fd', fsock.read())
        with open(pjoin(out, 'Source', 'makefile')) as fsock:
            self.assertIn('libdhelas_fd', fsock.read())
        pdirs = [os.path.basename(p) for p in
                 glob.glob(pjoin(out, 'SubProcesses', 'P*'))]
        for pdir in pdirs:
            makefile = pjoin(out, 'SubProcesses', pdir, 'makefile')
            with open(makefile) as fsock:
                self.assertEqual('dhelas_fd' in fsock.read(), 'wpwm' in pdir, pdir)
        with open(pjoin(out, 'SubProcesses', 'proc_characteristics')) as fsock:
            text = fsock.read()
        self.assertIn('gauge = unitary', text)
        self.assertIn('=FD', text)
        with open(pjoin(out, 'citations.log')) as fsock:
            self.assertIn('Hagiwara:2024xdh', fsock.read())
        with open(pjoin(out, 'Cards', 'proc_card_mg5.dat')) as fsock:
            self.assertIn('generate p p > w+ w- --gauge=FD', fsock.read())

    def test_unsupported_output_refused(self):
        """a format writing every matrix element into one library"""

        self.assertRaises(madgraph.InvalidCmd, self.do,
                          'output pythia8 %s -f' % pjoin(self.path, 'PY8'))


class TestProcCardOneGauge(unittest.TestCase):
    """The proc_card of events read back by MadSpin and the reweighting."""

    def test_one_gauge_lines(self):
        import madgraph.various.banner as banner_mod
        lines = ['import model sm',
                 '# --gauge=FD chosen automatically for the processes that need it',
                 'generate p p > w+ w- --gauge=FD',
                 'add process p p > t t~',
                 'output madevent PROC']
        self.assertEqual(banner_mod.ProcCard.one_gauge_lines(lines),
                         ['import model sm', lines[1], 'set gauge FD',
                          'generate p p > w+ w-', 'add process p p > t t~',
                          'output madevent PROC'])
        plain = ['import model sm', 'generate p p > t t~', 'output madevent PROC']
        self.assertEqual(banner_mod.ProcCard.one_gauge_lines(plain), plain)
        banner = banner_mod.Banner()
        banner['mg5proccard'] = '\n'.join(lines)
        card = banner.charge_card('proc_card')
        self.assertIn('set gauge FD', list(card))
        self.assertEqual(card.info['generate'], 'p p > w+ w-')
