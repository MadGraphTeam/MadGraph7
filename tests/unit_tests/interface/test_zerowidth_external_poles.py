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
"""The warning of zerowidth_external about propagators that can go on shell.

zerowidth_external drops the width of every internal propagator of a particle
that is also external. That is right for the s/u-channel top of t a > t a,
whose pole is only at the edge of phase space, and wrong for the final-state
W- > d u~ of u d~ > w+ w+ d u~ QCD=0, whose pole lies inside phase space: the
cross section is then infinite unless a cut (mmjj) removes the pole region.
The output must warn in the second case and stay quiet in the first.
"""

from __future__ import absolute_import

import tempfile
import unittest
from unittest import mock

from madgraph.interface import master_interface
import madgraph.interface.madgraph_interface as madgraph_interface


class TestZerowidthExternalPoleWarning(unittest.TestCase):

    def output_warnings(self, process, zerowidth_external=True):
        """The zerowidth_external warnings printed by `output standalone`."""
        with tempfile.TemporaryDirectory() as directory:
            command = master_interface.MasterCmd()
            lines = ['set automatic_html_opening False --no_save',
                     'set zerowidth_external %s' % zerowidth_external,
                     'import model sm', 'generate %s' % process,
                     'output standalone %s/out' % directory]
            with mock.patch.object(madgraph_interface.logger, 'warning') \
                    as warning:
                try:
                    for line in lines:
                        command.exec_cmd(line, errorhandling=False,
                                         printcmd=False, precmd=True)
                finally:
                    command.exec_cmd('set zerowidth_external True --no_save',
                                     errorhandling=False, printcmd=False,
                                     precmd=True)
            return [str(call[0][0]) for call in warning.call_args_list
                    if 'zerowidth_external:' in str(call[0][0])]

    def test_final_state_w_pole(self):
        """u d~ > w+ w+ d u~: m(d u~) reaches the W mass."""
        warnings = self.output_warnings('u d~ > w+ w+ d u~ QCD=0')
        self.assertEqual(len(warnings), 1)
        self.assertIn('reaches the w+ mass 80.4 GeV', warnings[0])

    def test_pole_of_a_crossing(self):
        """p p > w+ w+ j j: the generic _quark _quark process has only
        t-channel W's; the pole is in its crossing q q~ > w+ w+ q q~."""
        warnings = self.output_warnings('p p > w+ w+ j j QCD=0')
        self.assertEqual(len(warnings), 1)
        self.assertIn('reaches the w+ mass', warnings[0])

    def test_no_warning_without_reachable_pole(self):
        """t a > t a (pole at the edge of phase space), u u > w+ w+ d d
        (t-channel W's only), g g > t t~ g (top + gluon at threshold)."""
        for process in ['t a > t a', 'u u > w+ w+ d d QCD=0',
                        'g g > t t~ g']:
            self.assertEqual(self.output_warnings(process), [], process)

    def test_no_warning_when_widths_kept(self):
        """With zerowidth_external False no width is dropped."""
        self.assertEqual(self.output_warnings('u d~ > w+ w+ d u~ QCD=0',
                                              zerowidth_external=False), [])


if __name__ == '__main__':
    unittest.main()
