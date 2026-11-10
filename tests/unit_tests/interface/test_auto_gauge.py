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
                if line.startswith(('set gauge', '# FD gauge', 'generate',
                                    'add process'))]

    # ------------------------------------------------------------------
    # the criterion
    # ------------------------------------------------------------------
    def test_criterion(self):
        """fd_gauge_criterion on the diagrams of a few SM processes"""

        self.do('set auto_gauge False')   # the amplitudes as generated
        # (electroweak_only, massive_vector); None: not evaluated past the
        # first QCD vertex
        expected = [
            ('p p > w+ w- j j QCD=0', (True, True)),      # VBS
            ('p p > w+ w-', (True, True)),                # no QCD vertex at LO
            ('p p > h j j', (True, True)),                # VBF/VH, no QCD at LO in the sm
            ('e+ e- > mu+ mu-', (True, True)),            # Z exchange
            ('p p > w+ w-, w+ > e+ ve', (True, True)),    # decay chain
            ('a a > mu+ mu-', (True, False)),             # no massive vector
            ('p p > t t~', (False, None)),                # QCD
            ('p p > w+ j', (False, None)),                # one QCD vertex
            ]
        for line, (electroweak_only, massive_vector) in expected:
            self.do('generate %s' % line)
            result = mg_interface.fd_gauge_criterion(self.cmd._curr_amps)
            self.assertEqual(result[0], electroweak_only, line)
            if massive_vector is not None:
                self.assertEqual(result[1], massive_vector, line)
            self.assertGauge('unitary', automatic=False)

    def test_criterion_loop_amplitude(self):
        """a loop amplitude never qualifies"""

        self.do('import model loop_sm')
        self.do('generate u u~ > w+ w- [virt=QCD]')
        self.assertEqual(mg_interface.fd_gauge_criterion(self.cmd._curr_amps),
                         (False, False))
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

    def test_add_process_reverts(self):
        """adding a QCD process to an electroweak one brings the whole output
        back to the unitary gauge, every process generated again"""

        self.do('generate p p > w+ w-')
        self.assertGauge('FD', automatic=True)
        self.do('add process p p > t t~')
        self.assertGauge('unitary', automatic=False)
        self.assertEqual(len(self.cmd._curr_amps), 3)
        self.assertEqual([amp.get('process').get('id') for amp in self.cmd._curr_amps],
                         [1, 2, 2])
        self.assertEqual(self.gauge_lines(),
                         ['generate p p > w+ w-', 'add process p p > t t~'])
        # and an electroweak one added to a QCD one changes nothing
        self.do('add process p p > z z')
        self.assertGauge('unitary', automatic=False)
        self.assertEqual(len(self.cmd._curr_amps), 4)

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
        self.do('generate e+ e- > V > e+ e-')
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
        self.assertFalse(mg_interface.fd_gauge_criterion(self.cmd._curr_amps)[0])

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
        # the amplitudes are gone with the model, as after 'set gauge'
        self.assertEqual(len(self.cmd._curr_amps), 0)
        self.assertNotIn('set gauge FD', self.cmd.history)

    def test_model_without_feynman_gauge(self):
        """a model without goldstone boson (gauge = [0]) has no FD gauge"""

        self.do('import model %s' % pjoin(MG5DIR, 'tests', 'input_files',
                                            'fourfermion_UFO'))
        self.assertEqual(self.cmd._curr_model.get('gauge'), [0])
        self.do('generate fa fa~ > fb fb~')
        # V exchange, no QCD: the criterion holds, the model does not
        self.assertEqual(mg_interface.fd_gauge_criterion(self.cmd._curr_amps),
                         (True, True))
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
        self.do('generate p p > z z')
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
