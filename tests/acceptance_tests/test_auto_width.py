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
"""compute_widths with madspace and with madevent.

Both integrate the N-body decays of the same amplitudes, so the cards they write
have to agree: every channel within the integration error, and the totals
within it too. The processes are the SM ones with a real W or Z in the final
state (h > W d u~, Z > W e nu, ...), and the particles go in one call, which is
what makes the output of the second particle start from the matrix elements of
the first one if nothing resets them.
"""

from __future__ import absolute_import

import os
import re
import shutil
import tempfile
import unittest

import madgraph.interface.master_interface as mgcmd
from madgraph.iolibs.template_files.mg7 import bootstrap as mg7_bootstrap

pjoin = os.path.join


def read_decays(path):
    """{pid: {'total': GeV, 'channels': {sorted products: GeV}}} of a param_card"""
    decays = {}
    current = None
    with open(path) as stream:
        for line in stream:
            match = re.match(r'DECAY\s+(\d+)\s+([-\d.eE+]+)', line)
            if match:
                current = int(match.group(1))
                decays[current] = {'total': float(match.group(2)), 'channels': {}}
                continue
            match = re.match(r'\s*[-\d.eE+]+\s+\d+\s+((?:-?\d+\s+)+)#\s*([-\d.eE+]+)', line)
            if match and current is not None:
                products = tuple(sorted(int(x) for x in match.group(1).split()))
                decays[current]['channels'][products] = float(match.group(2))
    return decays


@unittest.skipUnless(mg7_bootstrap.madspace_is_installed(),
                     'madspace is not installed')
class TestAutoWidthEngines(unittest.TestCase):
    """The cards of --engine=madspace and --engine=madevent agree."""

    PARTICLES = {6: 't', 23: 'z', 24: 'w+', 25: 'h'}

    @classmethod
    def setUpClass(cls):
        cls.path = tempfile.mkdtemp(prefix='acc_test_auto_width')
        cmd = mgcmd.MasterCmd()
        cmd.no_notification()
        cmd.exec_cmd('set automatic_html_opening False --no_save')
        cmd.exec_cmd('import model sm')
        cls.cards = {}
        for engine in ('madspace', 'madevent'):
            card = pjoin(cls.path, 'param_card_%s.dat' % engine)
            cmd.exec_cmd('compute_widths t w+ z h --body_decay=3 --engine=%s '
                         '--output=%s' % (engine, card))
            cls.cards[engine] = read_decays(card)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.path, ignore_errors=True)

    def test_every_particle_has_a_decay_table_with_both(self):
        for engine, decays in self.cards.items():
            for pid in self.PARTICLES:
                self.assertIn(pid, decays, '%s: no decays for %s' % (engine, pid))
                self.assertTrue(decays[pid]['channels'])

    def test_total_widths(self):
        for pid, name in self.PARTICLES.items():
            madspace = self.cards['madspace'][pid]['total']
            madevent = self.cards['madevent'][pid]['total']
            self.assertAlmostEqual(madspace / madevent, 1, delta=0.01,
                                   msg='total width of %s: madspace %g, madevent %g'
                                   % (name, madspace, madevent))

    def test_same_channels_in_both(self):
        for pid, name in self.PARTICLES.items():
            self.assertEqual(sorted(self.cards['madspace'][pid]['channels']),
                             sorted(self.cards['madevent'][pid]['channels']),
                             'channels of %s' % name)

    def test_channels(self):
        """Channel by channel, to 5%: madevent aims at 1% for each of them (and
        the smallest channels of the Z are only a few 1e-8 GeV)."""
        for pid, name in self.PARTICLES.items():
            madspace = self.cards['madspace'][pid]['channels']
            madevent = self.cards['madevent'][pid]['channels']
            for products, expected in madevent.items():
                self.assertAlmostEqual(
                    madspace[products] / expected, 1, delta=0.05,
                    msg='%s > %s: madspace %g, madevent %g'
                    % (name, products, madspace[products], expected))

    def test_the_decays_of_one_particle_are_not_those_of_another(self):
        """The three-body decays of the Z are 1e-7 GeV: they are small because
        the Z is barely above the W threshold. The widths of the Higgs, with
        the same final states, are a thousand times larger."""
        for engine, decays in self.cards.items():
            for products, width in decays[23]['channels'].items():
                if len(products) == 3:
                    self.assertLess(width, 1e-6, '%s: Z > %s' % (engine, products))
