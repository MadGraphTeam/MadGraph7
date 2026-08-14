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
"""Unit tests for the reweight interface helpers."""
from __future__ import absolute_import

import unittest

import madgraph.interface.reweight_interface as rwgt_interface


class FakeEvent(object):
    """Only what _pdg_for_me_call touches: the concrete per-leg PDGs."""

    def __init__(self, pdgs):
        self.pdgs = list(pdgs)

    def get_pdg(self, momenta):
        return list(self.pdgs)


class FakeModel(dict):
    """Stand-in for a model carrying a merged_particles map."""

    def __init__(self, merged):
        dict.__init__(self)
        self['merged_particles'] = merged


class TestPdgForMeCall(unittest.TestCase):
    """The merged-particle labels carried by the generated process must be
    converted back to the concrete PDGs of the event before they are handed to
    the fortran, for BOTH signs of the merged code.

    merged_particles is keyed by the positive code only, so the membership test
    has to be done on abs(). A subprocess whose grouped legs are all
    anti-particles -- g q~ > w+ q~, get_pdg_order [21,-81,24,-81] -- used to
    keep its -81 labels, which the fortran flavor mapping resolves to "no
    flavour": SMATRIXHEL returned an exact 0 (raising "Invalid matrix element")
    and GET_DENSITY returned an all-zero density matrix.
    """

    # {merged code: members}, as apply_flavor_grouping builds it: POSITIVE keys
    MERGED = {81: [1, 2, 3, 4], 82: [11, 13]}

    def setUp(self):
        self.obj = rwgt_interface.ReweightInterface.__new__(
            rwgt_interface.ReweightInterface)
        self.obj.merged_particles = None
        # __del__ calls do_quit; keep it a no-op on this bare instance
        self.obj.exitted = True
        self.model = FakeModel(self.MERGED)

    def call(self, orig_order, event_pdgs, model=None):
        event = FakeEvent(event_pdgs)
        model = self.model if model is None else model
        return self.obj._pdg_for_me_call(event, orig_order, None, model)

    def test_negative_merged_labels_are_resolved(self):
        """g q~ > w+ q~ -- the regression: every grouped leg is an antiparticle
        so both merged codes are NEGATIVE."""
        # process legs [21,-81,24,-81], event is g d~ > w+ u~
        out = self.call(((21, -81), (24, -81)), [21, -1, 24, -2])
        self.assertEqual(out, [21, -1, 24, -2])

    def test_positive_merged_labels_are_resolved(self):
        """g q > w+ q -- positive merged codes, the case that always worked."""
        out = self.call(((21, 81), (24, 81)), [21, 2, 24, 1])
        self.assertEqual(out, [21, 2, 24, 1])

    def test_mixed_sign_merged_labels_are_resolved(self):
        """q q~ > w+ g -- one leg of each sign."""
        out = self.call(((81, -81), (24, 21)), [2, -1, 24, 21])
        self.assertEqual(out, [2, -1, 24, 21])

    def test_negative_lepton_merged_label_is_resolved(self):
        """The same for the charged-lepton group (82), to pin that the fix is
        not specific to the jet code."""
        out = self.call(((-82, 82), (24, -24)), [-11, 13, 24, -24])
        self.assertEqual(out, [-11, 13, 24, -24])

    def test_no_merged_leg_keeps_the_process_order(self):
        """A process with no grouped leg must keep orig_order untouched, so the
        legs stay in the order the matrix element expects."""
        out = self.call(((21, 21), (24, -24)), [21, 21, -24, 24])
        self.assertEqual(out, [21, 21, 24, -24])

    def test_no_flavor_grouping_keeps_the_process_order(self):
        """Without flavor grouping merged_particles is empty and the event PDGs
        must not be substituted (the pre-flavor-grouping behavior)."""
        out = self.call(((21, -1), (24, -2)), [21, -1, 24, -2],
                        model=FakeModel({}))
        self.assertEqual(out, [21, -1, 24, -2])

    def test_falls_back_to_self_merged_particles(self):
        """When there is no model to consult (the load_from_pickle path) the
        map saved on the instance is used -- with the same sign handling."""
        self.obj.merged_particles = self.MERGED
        out = self.call(((21, -81), (24, -81)), [21, -1, 24, -2], model=None)
        self.assertEqual(out, [21, -1, 24, -2])


if __name__ == '__main__':
    unittest.main()
