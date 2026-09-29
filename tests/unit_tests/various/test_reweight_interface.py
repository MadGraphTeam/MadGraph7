################################################################################
#
# Copyright (c) 2009 The MadGraph7 Development team and Contributors
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
"""Unit tests for the reweight interface helpers."""
from __future__ import absolute_import

import itertools
import os
import shutil
import tempfile
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


class FakeMG5Cmd(object):
    """Records the commands the reweight hands to its MG5 interface."""

    def __init__(self):
        self.commands = []

    def exec_cmd(self, line, *args, **opts):
        self.commands.append(line)


class TestReweightGenerationFoldsCrossings(unittest.TestCase):
    """The reweight generates its own matrix elements from the proc card lines,
    and reads a crossed subprocess folded onto its base (build_cross_resolve).
    Crossing is OFF by default for a generation and a proc card line carries no
    --use_crossing unless the user gave one, so the reweight has to ask for the
    folding itself: without it every crossed subprocess got an rw_me matrix
    element -- a generation and a compilation -- of its own."""

    def setUp(self):
        self.obj = rwgt_interface.ReweightInterface.__new__(
            rwgt_interface.ReweightInterface)
        # __del__ calls do_quit; keep it a no-op on this bare instance
        self.obj.exitted = True
        self.obj.keep_ordering = False
        self.obj.use_eventid = False
        self.obj.flag_density_matrix = False
        self.obj.inc_sudakov = False
        self.obj.nb_rw = 0
        self.obj.path2prefix = {}
        self.obj.mg5cmd = FakeMG5Cmd()
        self.path = tempfile.mkdtemp(prefix='rwgt_crossing')

    def tearDown(self):
        shutil.rmtree(self.path)

    def generate(self, processes):
        """The process definitions of the generate command the reweight
        issues for `processes` (proc card lines)."""
        data = {'path': self.path, 'paths': ['rw_me', 'rw_mevirt'],
                'processes': list(processes)}
        self.obj.create_standalone_tree_directory(data)
        line = [c for c in self.obj.mg5cmd.commands
                if c.startswith('generate')]
        self.assertEqual(len(line), 1)
        line = line[0][len('generate'):]
        return [p.replace('add process', '', 1).strip()
                for p in line.split(';') if p.strip()]

    def test_tree_definitions_fold_their_crossings(self):
        """the regression: every tree definition asks for the folding"""
        self.assertEqual(self.generate(['p p > w+ j', 'p p > w+ j j @2']),
                         ['p p > w+ j --use_crossing=True',
                          'p p > w+ j j @2 --use_crossing=True'])

    def test_keep_ordering_unfolds(self):
        """keep_ordering needs one directory per crossed subprocess, and the
        False comes last so that it also overrides a True from the card."""
        self.obj.keep_ordering = True
        self.assertEqual(self.generate(['p p > w+ j --use_crossing=True']),
             ['p p > w+ j --use_crossing=True --use_crossing=False'])

    def test_density_mode_unfolds(self):
        self.obj.flag_density_matrix = True
        self.assertEqual(self.generate(['p p > w+ j']),
                         ['p p > w+ j --use_crossing=False'])

    def test_use_eventid_unfolds(self):
        """the folded entry point takes no process id"""
        self.obj.use_eventid = True
        self.assertEqual(self.generate(['p p > w+ j']),
                         ['p p > w+ j --use_crossing=False'])

    def test_explicit_choice_is_replayed(self):
        """an explicit --use_crossing on a proc card line is the user's choice,
        sticky for the whole definition: nothing is added to any line."""
        for flag in ['--use_crossing=False', '--use_crossing=True',
                     '--use_crossing']:
            self.obj.mg5cmd = FakeMG5Cmd()
            self.assertEqual(
                self.generate(['p p > w+ j', 'p p > w+ j j %s' % flag]),
                ['p p > w+ j', 'p p > w+ j j %s' % flag])

    def test_perturbative_definition_is_left_alone(self):
        """the LO lines derived from a [...] definition carry no flag, and would
        inherit a True from an earlier tree line of the same definition."""
        self.assertEqual(self.obj.tree_crossing_flag(
            ['p p > w+ j', 'p p > w+ j [QCD]']), '')
        self.obj.inc_sudakov = True
        self.assertEqual(self.obj.tree_crossing_flag(['p p > t t~']), '')


class FakeCrossingModule(object):
    """The two per-matrix-element f2py entry points build_cross_resolve walks,
    for the base g u > e+ e- u (prefix m0_, one flavor) and its crossing 30,
    which swaps slots 1 and 5: u~ u > e+ e- g."""

    def py_m0_get_flavor_layout(self):
        return (1, 5, 36)        # NFLAV, NEXTERNAL, NCROSS

    def py_m0_get_pdg_for_flavor(self, flav_idx):
        return {1: [21, 2, -11, 11, 2],
                31: [-2, 2, -11, 11, 21]}.get(flav_idx, [0] * 5)


class TestFoldedCrossingHelicity(unittest.TestCase):
    """A folded crossed event is evaluated with USERHEL = a row of the BASE
    helicity table, and the generated SMATRIX applies the crossing as tau
    (APPLY_CROSSING_TABLE): the momenta move into the base slots, crossed leg
    perm[k] landing in base slot k, but the NHEL slots stay where they are. The
    row an event needs has therefore entry k = the event's helicity of crossed
    leg perm[k]. The base dictionary read positionally in the crossed leg order
    (right while the table was permuted, sigma) picked another configuration:
    for p p > e+ e- j, with q q~ > e+ e- g folded onto g q > e+ e- q, the third
    event of a madevent sample came out exactly 0 ("Invalid matrix element")."""

    def setUp(self):
        self.obj = rwgt_interface.ReweightInterface.__new__(
            rwgt_interface.ReweightInterface)
        # __del__ calls do_quit; keep it a no-op on this bare instance
        self.obj.exitted = True
        self.obj.is_decay = False
        self.obj.keep_ordering = False
        self.path = tempfile.mkdtemp(prefix='rwgt_crossing')
        with open(os.path.join(self.path, 'crossed_flavors.dat'), 'w') as f:
            f.write('M0_ 1 30\n')

    def tearDown(self):
        shutil.rmtree(self.path)

    def test_crossed_helicity_row_follows_the_crossing(self):
        base = dict((row, i + 1) for i, row in
                    enumerate(itertools.product([-1, 1], repeat=5)))
        cross_data = {}
        self.obj.build_cross_resolve(
            FakeCrossingModule(), ['m0_'], [[21, 2, -11, 11, 2]],
            {'m0_': base}, self.path, False, cross_data, None)

        # the crossed subprocess is reached, in its own leg order
        tag = ((-2, 2), (-11, 11, 21))
        self.assertEqual(list(cross_data), [tag])
        [(order, pdir, hel, procindex, flav_idx, phys)] = cross_data[tag]
        self.assertEqual(order, ([-2, 2], [-11, 11, 21]))
        self.assertEqual((procindex, flav_idx), (1, 31))

        # u~(h1) u(h2) e+(h3) e-(h4) g(h5): the gluon sits in base slot 1 and
        # the u~ in base slot 5, so the row is (h5, h2, h3, h4, h1)
        self.assertEqual(len(hel), 32)
        for h in itertools.product([-1, 1], repeat=5):
            self.assertEqual(hel[h], base[(h[4], h[1], h[2], h[3], h[0])])
        # the event of the report: the old keying gave the row whose leg 1
        # and 5 helicities are swapped, which vanishes for this process
        self.assertNotEqual(hel[(1, -1, 1, -1, -1)], base[(1, -1, 1, -1, -1)])


if __name__ == '__main__':
    unittest.main()
