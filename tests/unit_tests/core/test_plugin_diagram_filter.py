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
################################################################################
"""Tests of the diagram filters provided by plugins: the plugin declares
    new_diagram_filter = remove_diag
and it is selected with --diagram_filter=PLUGINNAME (tree and MadLoop
generation) or --tree_diagram_filter / --loop_diagram_filter=PLUGINNAME (NLO
FKS generation)."""

from __future__ import absolute_import

import os
import shutil
import sys
import tempfile

root_path = os.path.split(os.path.dirname(os.path.realpath(__file__)))[0]
sys.path.append(os.path.join(root_path, os.path.pardir, os.path.pardir))

import tests.unit_tests as unittest

import madgraph.interface.master_interface as MGCmd
import madgraph.various.misc as misc
from madgraph import InvalidCmd, MadGraph5Error

HEADER = """
minimal_mg5amcnlo_version = (2, 0, 0)
maximal_mg5amcnlo_version = (99, 0, 0)
latest_validated_version = (99, 0, 0)
"""

KEEP_ALL = "def remove_diag(diag, model=None):\n    return False\n"
REMOVE_ALL = "def remove_diag(diag, model=None):\n    return True\n"


class PluginDiagramFilterTest(unittest.TestCase):
    """A temporary plugin directory is put in sys.path; each test writes the
    plugins it needs with add_plugin."""

    counter = 0

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        PluginDiagramFilterTest.counter += 1
        # a different package name each time, python caches the imports
        self.dirname = 'TESTPLUGINDIR%d' % PluginDiagramFilterTest.counter
        self.plugdir = os.path.join(self.tmp, self.dirname)
        os.mkdir(self.plugdir)
        open(os.path.join(self.plugdir, '__init__.py'), 'w').close()
        sys.path.insert(0, self.tmp)
        self.modules = []

    def tearDown(self):
        sys.path.remove(self.tmp)
        for name in list(sys.modules):
            if name.startswith(self.dirname):
                del sys.modules[name]
        shutil.rmtree(self.tmp)

    def add_plugin(self, name, filter_code=None, init_text=None):
        """plugin 'name' whose new_diagram_filter is remove_diag of
        filter_code (or init_text, the full __init__.py, if given)"""
        path = os.path.join(self.plugdir, name)
        os.mkdir(path)
        if init_text is None:
            with open(os.path.join(path, 'filt.py'), 'w') as fsock:
                fsock.write(filter_code)
            init_text = 'from .filt import remove_diag\n' \
                        'new_diagram_filter = remove_diag\n'
        with open(os.path.join(path, '__init__.py'), 'w') as fsock:
            fsock.write(HEADER + init_text)
        # the version check of the plugin is not what is tested here
        misc.plugin_support['%s.%s' % (self.dirname, name)] = True

    def interface(self, model):
        interface = MGCmd.MasterCmd()
        interface.plugin_path = [self.plugdir]
        interface.exec_cmd('import model %s' % model, printcmd=False,
                           precmd=True)
        return interface

    def generate(self, line, model='sm'):
        interface = self.interface(model)
        interface.exec_cmd('generate %s' % line, printcmd=False, precmd=True,
                           errorhandling=False)
        return interface

    def test_plugin_diagram_filter_lookup(self):
        """the filter of the plugin of that name is returned, the names of the
        plugins providing one are listed"""
        self.add_plugin('keepall', KEEP_ALL)
        self.add_plugin('nofilter', init_text='x = 1\n')
        fct = misc.plugin_diagram_filter([self.plugdir], 'keepall')
        self.assertTrue(callable(fct))
        self.assertFalse(fct(None))
        self.assertEqual(misc.plugin_diagram_filter([self.plugdir], None),
                         ['keepall'])

    def test_plugin_diagram_filter_errors(self):
        """unknown plugin, plugin without filter and non-callable filter
        raise"""
        self.add_plugin('nofilter', init_text='x = 1\n')
        self.add_plugin('notfunction', init_text='new_diagram_filter = "x"\n')
        for name in ['unknown', 'nofilter', 'notfunction']:
            self.assertRaises(MadGraph5Error, misc.plugin_diagram_filter,
                              [self.plugdir], name)

    def test_tree_level(self):
        """--diagram_filter=PLUGINNAME removes the diagrams selected by the
        plugin function (here all or none)"""
        self.add_plugin('keepall', KEEP_ALL)
        self.add_plugin('removeall', REMOVE_ALL)
        interface = self.generate('u u~ > d d~ --diagram_filter=keepall')
        self.assertTrue(len(interface._curr_amps[0].get("diagrams")) > 0)
        interface = self.generate('u u~ > d d~')
        self.assertTrue(len(interface._curr_amps[0].get('diagrams')) > 0)
        self.assertRaises(Exception, self.generate,
                          'u u~ > d d~ --diagram_filter=removeall')

    def test_tree_level_invalid_name(self):
        self.add_plugin('nofilter', init_text='x = 1\n')
        for name in ['unknown', 'nofilter']:
            self.assertRaises(InvalidCmd, self.generate,
                              'u u~ > d d~ --diagram_filter=%s' % name)

    def test_tree_level_filter_receives_diagrams(self):
        """the function is called with the diagrams and the model"""
        calls = []
        def record(diag, model=None):
            calls.append((diag, model))
            return False
        import madgraph.core.diagram_generation as diagram_generation
        import madgraph.core.base_objects as base_objects
        amp = diagram_generation.Amplitude()
        amp.set('process', base_objects.Process())
        diags = base_objects.DiagramList([base_objects.Diagram(),
                                          base_objects.Diagram()])
        res = amp.apply_user_filter(diags, record)
        self.assertEqual(len(calls), 2)
        self.assertEqual(len(res), 2)
        self.assertRaises(MadGraph5Error, amp.apply_user_filter, diags,
                          'not a function')

    def test_loop_level(self):
        """--diagram_filter=PLUGINNAME is applied to the loop diagrams of a
        MadLoop process"""
        self.add_plugin('keepall', KEEP_ALL)
        self.add_plugin('removeall', REMOVE_ALL)
        interface = self.generate('g g > h [virt=QCD] '
                                  '--diagram_filter=keepall', 'loop_sm')
        nloops = len(interface._curr_amps[0].get('loop_diagrams'))
        self.assertTrue(nloops > 0)
        # no loop diagram left: no amplitude
        self.assertRaises(Exception, self.generate, 'g g > h [virt=QCD] '
                                  '--diagram_filter=removeall', 'loop_sm')

    def test_loop_induced(self):
        """... and of a loop-induced process"""
        self.add_plugin('removeall', REMOVE_ALL)
        self.assertRaises(Exception, self.generate,
                          'g g > h [noborn=QCD] --diagram_filter=removeall',
                          'loop_sm')

    def test_fks_tree_and_loop_filters(self):
        """--tree_diagram_filter acts on the Born (and real) diagrams and
        --loop_diagram_filter on the loop diagrams only; --diagram_filter is
        refused"""
        self.add_plugin('keepall', KEEP_ALL)
        self.add_plugin('removeall', REMOVE_ALL)
        proc = 'u u~ > d d~ [QCD] %s'

        interface = self.generate(proc % ('--tree_diagram_filter=keepall '
                                          '--loop_diagram_filter=keepall'),
                                  'loop_sm')
        fksmulti = interface._fks_multi_proc
        nvirt = sum(len(amp.get('loop_diagrams'))
                    for amp in fksmulti.get_virt_amplitudes())
        nborn = sum(len(amp.get('diagrams'))
                    for amp in fksmulti.get_born_amplitudes())
        nreal = sum(len(amp.get('diagrams'))
                    for amp in fksmulti.get_real_amplitudes())
        self.assertTrue(nvirt > 0 and nborn > 0 and nreal > 0)

        # no loop diagram left, the tree-level ones are untouched
        interface = self.generate(proc % '--loop_diagram_filter=removeall',
                                  'loop_sm')
        fksmulti = interface._fks_multi_proc
        self.assertEqual(sum(len(amp.get('loop_diagrams'))
                         for amp in fksmulti.get_virt_amplitudes()), 0)
        self.assertEqual(sum(len(amp.get('diagrams'))
                         for amp in fksmulti.get_born_amplitudes()), nborn)
        self.assertEqual(sum(len(amp.get('diagrams'))
                         for amp in fksmulti.get_real_amplitudes()), nreal)

        # no Born diagram left
        self.assertRaises(Exception, self.generate,
                          proc % '--tree_diagram_filter=removeall', 'loop_sm')

        # the loop filter does not touch the loop diagrams of a tree-level
        # filter alone
        interface = self.generate(proc % '--tree_diagram_filter=keepall',
                                  'loop_sm')
        fksmulti = interface._fks_multi_proc
        self.assertEqual(sum(len(amp.get('loop_diagrams'))
                         for amp in fksmulti.get_virt_amplitudes()), nvirt)

        self.assertRaises(InvalidCmd, self.generate,
                          proc % '--diagram_filter', 'loop_sm')
