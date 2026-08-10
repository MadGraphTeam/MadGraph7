################################################################################
#
# Copyright (c) 2009 The MadGraph5_aMC@NLO Development team and Contributors
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

"""Which output formats a loop-induced ([noborn=...]) process may be sent to.

A loop-induced process does not reach the exporters the way a [virt=...] one
does.  master_interface borrows the MadLoop interface only long enough to
validate the model, then switches *back* to 'MadGraph' and calls
create_loop_induced -- so the process is exported by the ordinary tree-level
output machinery (madgraph_interface.do_output, then ExportV4Factory with
output_type='default', or ExportCPPFactory) even though the amplitude is a
LoopAmplitude and the matrix element a LoopHelasMatrixElement.

Only the 'madevent' formats have a loop-induced exporter to route to.  Every
other format used to hand the loop matrix element to a tree-level exporter and
die deep inside it -- 'output standalone' with an IndexError in write_check_sa,
'output matrix' with "wavefunction_rank has not been computed", 'output mg7'
with a KeyError on the first loop leg, which the mg7 exporter's edge-name map
does not contain.  They now refuse the process up front and point at
[sqrvirt=...], which stays in the MadLoop interface and yields the very same
matrix element as a standalone MadLoop output.

The last two tests are the important ones to keep green: the refusal keys off
the amplitude being a LoopAmplitude, and [virt=]/[sqrvirt=] amplitudes are
LoopAmplitudes too -- they are spared only because they never reach these two
factories.  If that ever stops being true, the escape hatch the error message
recommends would be refused along with everything else.
"""

from __future__ import absolute_import

import os
import shutil
import sys
import tempfile

root_path = os.path.split(os.path.dirname(os.path.realpath(__file__)))[0]
sys.path.append(os.path.join(root_path, os.path.pardir, os.path.pardir))

import tests.unit_tests as unittest

import madgraph.interface.master_interface as MGCmd
import madgraph.iolibs.export_v4 as export_v4
import madgraph.loop.loop_diagram_generation as loop_diagram_generation

from madgraph import InvalidCmd

pjoin = os.path.join

# The cheapest loop-induced process there is: no tree-level diagram exists for
# g g > h in loop_sm, so [noborn=QCD] gives the quark-loop amplitude alone.
LOOP_INDUCED_PROCESS = 'g g > h [noborn=QCD]'
# The same matrix element, generated the way the error message recommends.
SQRVIRT_PROCESS = 'g g > h [sqrvirt=QCD]'
# An ordinary virtual correction, for good measure.
VIRTUAL_PROCESS = 'u u~ > d d~ [virt=QCD]'


#===============================================================================
# TestLoopInducedOutput
#===============================================================================
class TestLoopInducedOutput(unittest.TestCase):
    """The output formats a loop-induced process may and may not be sent to."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix='loop_induced_output')

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    #===========================================================================
    # helpers
    #===========================================================================
    def get_interface(self, process=LOOP_INDUCED_PROCESS):
        """A MasterCmd with 'process' generated, ready to be output."""

        interface = MGCmd.MasterCmd()
        interface.no_notification()
        interface.exec_cmd('import model loop_sm', printcmd=False, precmd=True)
        interface.exec_cmd('generate %s' % process, printcmd=False, precmd=True)
        return interface

    def assert_output_refused(self, format):
        """'output <format>' must raise InvalidCmd, and say what to do instead.

        Anything that is not an InvalidCmd -- an IndexError, a KeyError, a
        MadGraph5Error from inside a tree-level exporter -- means the loop
        matrix element reached an exporter that cannot write it.
        """

        interface = self.get_interface()
        out_dir = pjoin(self.tmpdir, format.replace(' ', '_'))
        try:
            interface.exec_cmd('output %s %s -f' % (format, out_dir),
                               printcmd=False, precmd=True)
        except InvalidCmd as error:
            self.assertTrue('sqrvirt' in str(error),
                            "'output %s' refused %s without pointing at "
                            "[sqrvirt=]: %s" % (format, LOOP_INDUCED_PROCESS,
                                                error))
            return
        except Exception as error:
            raise AssertionError(
                "'output %s' crashed on %s with %s: %s"
                % (format, LOOP_INDUCED_PROCESS, type(error).__name__, error))
        raise AssertionError(
            "'output %s' silently accepted %s; no exporter for that format can "
            "write a LoopHelasMatrixElement" % (format, LOOP_INDUCED_PROCESS))

    def assert_output_succeeds(self, process, format='standalone'):
        """'output <format>' must go through, and write a MadLoop directory."""

        interface = self.get_interface(process)
        out_dir = pjoin(self.tmpdir, 'ok')
        interface.exec_cmd('output %s %s -f' % (format, out_dir),
                           printcmd=False, precmd=True)
        # A MadLoop output, not a tree-level one: this file only exists when a
        # loop exporter ran.
        self.assertTrue(
            os.path.exists(pjoin(out_dir, 'Cards', 'MadLoopParams.dat')),
            '%s did not produce a MadLoop output' % process)

    #===========================================================================
    # loop-induced processes are refused by the formats that cannot serve them
    #===========================================================================
    def test_standalone_factory_refuses_loop_induced(self):
        """ExportV4Factory must not hand a loop-induced process to the
        tree-level standalone exporter."""

        interface = self.get_interface()
        self.assertTrue(isinstance(interface._curr_amps[0],
                                   loop_diagram_generation.LoopAmplitude),
                        "%s did not produce a LoopAmplitude" % LOOP_INDUCED_PROCESS)

        interface._export_format = 'standalone'
        interface._export_dir = pjoin(self.tmpdir, 'factory')
        try:
            exporter = export_v4.ExportV4Factory(interface, False,
                                                 group_subprocesses=False,
                                                 cmd_options={})
        except InvalidCmd:
            return
        raise AssertionError(
            "ExportV4Factory returned %s for a loop-induced process; a "
            "tree-level exporter cannot write a LoopHelasMatrixElement"
            % type(exporter).__name__)

    def test_output_standalone_refuses_loop_induced(self):
        """'output standalone' used to die with an IndexError in write_check_sa."""

        self.assert_output_refused('standalone')

    def test_output_matrix_refuses_loop_induced(self):
        """'output matrix' shares the standalone branch of the factory; it used
        to die with "wavefunction_rank has not been computed"."""

        self.assert_output_refused('matrix')

    def test_output_mg7_refuses_loop_induced(self):
        """'output mg7' -- the default format -- used to die with a KeyError on
        the first loop leg.  mg7/madmatrix has no MadLoop backend at all."""

        self.assert_output_refused('mg7')

    def test_refusal_leaves_an_existing_directory_alone(self):
        """The refusal comes from the exporter factory, which runs before
        copy_template; an already existing output directory must survive it."""

        out_dir = pjoin(self.tmpdir, 'existing')
        os.mkdir(out_dir)
        sentinel = pjoin(out_dir, 'sentinel.txt')
        open(sentinel, 'w').write('do not delete me\n')

        interface = self.get_interface()
        self.assertRaises(InvalidCmd, interface.exec_cmd,
                          'output standalone %s -f' % out_dir)
        self.assertTrue(os.path.exists(sentinel),
                        'the refused output wiped the existing directory')

    #===========================================================================
    # ... but the routes that do work must keep working
    #===========================================================================
    def test_output_madevent_still_accepts_loop_induced(self):
        """madevent is the format loop-induced processes are *for*.

        It has the LoopInducedExporterMEGroup / ...MENoGroup exporters, so it
        must go straight through the refusal above.
        """

        interface = self.get_interface()
        out_dir = pjoin(self.tmpdir, 'me')
        interface.exec_cmd('output madevent %s -f' % out_dir,
                           printcmd=False, precmd=True)
        self.assertTrue(
            os.path.isdir(pjoin(out_dir, 'SubProcesses', 'MadLoop5_resources')),
            'madevent did not produce a loop-induced output')

    def test_sqrvirt_standalone_output_still_works(self):
        """The escape hatch the error message recommends must actually work.

        [sqrvirt=] gives a LoopAmplitude just like [noborn=] does; it is spared
        only because master_interface keeps it in the MadLoop interface, which
        passes output_type='madloop' and never reaches the refusing branch.
        """

        self.assert_output_succeeds(SQRVIRT_PROCESS)

    def test_virt_standalone_output_still_works(self):
        """Same for an ordinary [virt=] process."""

        self.assert_output_succeeds(VIRTUAL_PROCESS)
