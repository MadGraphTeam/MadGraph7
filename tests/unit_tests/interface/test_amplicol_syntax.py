################################################################################
#
# Copyright (c) 2026 The MadGraph5_aMC@NLO Development team and Contributors
#
# This file is a part of the MadGraph5_aMC@NLO project, an application which
# automatically generates Feynman diagrams and matrix elements for arbitrary
# high-energy processes in the Standard Model and beyond.
#
# It is subject to the MadGraph5_aMC@NLO license which should accompany this
# distribution.
#
# For more information, visit madgraph.phys.ucl.ac.be and amcatnlo.web.cern.ch
#
################################################################################
"""Syntax of the processes computed by AmpliCol

    generate p p > j j j [amplicol]

The process is kept as a definition for the mg7 output, which generates its
matrix elements with pyamplicol; no MG diagrams are generated for it.
"""

import os
import sys

root_path = os.path.split(os.path.dirname(os.path.realpath(__file__)))[0]
sys.path.append(os.path.join(root_path, os.path.pardir, os.path.pardir))

import tests.unit_tests as unittest

import madgraph.interface.madgraph_interface as mg_interface
import madgraph.interface.master_interface as MGCmd

from madgraph import InvalidCmd

split = mg_interface.split_backend_line


class BackendLineSplitTest(unittest.TestCase):
    """split_backend_line, without any model."""

    def test_split(self):
        self.assertEqual(split('p p > j j j [amplicol]'), ('p p > j j j', 'amplicol'))
        self.assertEqual(split('p p > j j j [ AmpliCol ] @2'),
                         ('p p > j j j @2', 'amplicol'))
        self.assertEqual(split('p p > j j j'), ('p p > j j j', ''))
        self.assertEqual(split('p p > t t~ [QCD]'), ('p p > t t~ [QCD]', ''))

    def test_other_brackets_are_refused(self):
        self.assertRaises(InvalidCmd, split, 'p p > j j [amplicol] [QCD]')

    def test_process_type(self):
        process_type = MGCmd.MasterCmd.extract_process_type
        self.assertEqual(process_type('p p > j j j [amplicol]'), ('tree', None, []))
        self.assertEqual(process_type('p p > t t~ [QCD]'), ('NLO', 'all', ['QCD']))


class AmplicolProcessTest(unittest.TestCase):
    """generate/add process with [amplicol], with the sm model."""

    def setUp(self):
        self.cmd = MGCmd.MasterCmd()
        self.cmd.exec_cmd('import model sm', printcmd=False, precmd=True)

    def test_no_diagrams_are_generated(self):
        self.cmd.exec_cmd('generate g g > g g g [amplicol]', printcmd=False,
                          precmd=True)
        self.cmd.exec_cmd('add process p p > e+ e-', printcmd=False, precmd=True)
        self.assertEqual(len(self.cmd._curr_amplicol_defs), 1)
        self.assertEqual(
            self.cmd._curr_amplicol_defs[0].nice_string(prefix=False).split('@')[0].strip(),
            'g g > g g g')
        # the madmatrix process keeps its own process number
        self.assertTrue(self.cmd._curr_amps)
        self.assertEqual(
            {amp.get('process').get('id') for amp in self.cmd._curr_amps}, {2})
        self.cmd.exec_cmd('generate p p > e+ e-', printcmd=False, precmd=True)
        self.assertEqual(len(self.cmd._curr_amplicol_defs), 0)

    def test_unsupported_options_are_refused(self):
        for line in ['p p > j j QED=0 [amplicol]',
                     'p p > z > e+ e- [amplicol]',
                     'p p > j j / b [amplicol]',
                     'z > e+ e- [amplicol]']:
            self.assertRaises(InvalidCmd, self.cmd.do_generate, line)


class AmplicolExportTest(unittest.TestCase):
    """The process definitions as pyamplicol sees them, without pyamplicol."""

    def setUp(self):
        self.cmd = MGCmd.MasterCmd()
        self.cmd.exec_cmd('import model sm', printcmd=False, precmd=True)
        self.cmd.exec_cmd('define q = u d', printcmd=False, precmd=True)
        self.cmd.exec_cmd('define qx = u~ d~', printcmd=False, precmd=True)

    def export(self, line):
        import madgraph.iolibs.export_amplicol as export_amplicol
        self.cmd.exec_cmd('generate %s [amplicol]' % line, printcmd=False,
                          precmd=True)
        procdef = self.cmd._curr_amplicol_defs[0]
        model = self.cmd._curr_model
        return (export_amplicol.process_name(procdef, model,
                                             self.cmd._multiparticles),
                export_amplicol.process_expression(procdef, model),
                export_amplicol.representative_processes([procdef], model))

    def test_flavour_groups_are_expanded(self):
        """MG replaces the light quarks by a flavour group, which pyamplicol
        does not know: the original flavours are passed on."""
        name, expression, processes = self.export('u u~ > e+ e- g')
        self.assertEqual(name, 'uux_epemg')
        self.assertEqual(expression, ('u u~ > e+ e- g', {}))
        self.assertEqual(len(processes), 1)
        name, expression, processes = self.export('q qx > e+ e- g')
        self.assertEqual(name, 'qqx_epemg')
        self.assertEqual(expression, ('mp0 mp1 > e+ e- g',
                                      {'mp0': ['u', 'd'], 'mp1': ['u~', 'd~']}))
        self.assertEqual(len(processes), 4)

    def test_multiparticles(self):
        name, (expression, multiparticles), processes = self.export('p p > j j')
        self.assertEqual(name, 'pp_jj')
        self.assertEqual(expression, 'mp0 mp0 > mp0 mp0')
        self.assertEqual(sorted(multiparticles['mp0']),
                         sorted(['g', 'u', 'd', 's', 'c', 'u~', 'd~', 's~', 'c~']))
        self.assertEqual(len(processes), 81)
