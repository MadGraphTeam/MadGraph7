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
"""Unit tests of madgraph/iolibs/crossing_table.py (no model import: a stub
model gives the charge conjugates)."""

from __future__ import absolute_import

import unittest

import madgraph.iolibs.crossing_table as crossing_table


class _Particle(object):
    def __init__(self, anti):
        self.anti = anti

    def get_anti_pdg_code(self):
        return self.anti


class _Model(dict):
    """What the table builders read off a model: the charge conjugate of a
    PDG (self-conjugate gluon) and no merged particles."""

    def __init__(self):
        super(_Model, self).__init__()
        particles = {21: _Particle(21)}
        for pdg in (1, 2, 3, 4):
            particles[pdg] = _Particle(-pdg)
            particles[-pdg] = _Particle(pdg)
        self['particle_dict'] = particles
        self['merged_particles'] = {}


UUX_GG = (2, -2, 21, 21)


class TestBuildTableTargets(unittest.TestCase):
    """build_table serves a record's TARGET rows each in its own slot order."""

    def setUp(self):
        self.model = _Model()

    def build(self, records, base_labels=UUX_GG, entries=None):
        entries = entries or [(0, base_labels)]
        return crossing_table.build_table(4, 2, base_labels, entries, records,
                                          self.model)

    def test_target_rows_are_served_as_they_come(self):
        """u g > u g and u g > g u are the same physical process, but a
        consumer passing the momenta in the second order needs a row that
        feeds its slots so: both orders get their own row, each crossing the
        base row onto exactly the target."""
        anti = crossing_table.make_anti(self.model)
        targets = [(2, 21, 2, 21), (2, 21, 21, 2)]
        table = self.build([(None, (2, 21, 2, 21), None, targets)])
        record = table.records[0]
        self.assertTrue(record.complete())
        self.assertEqual([a.pdgs for a in record.assignments], targets)
        for a in record.assignments:
            self.assertEqual(table[a.K].crossed(UUX_GG, anti), a.pdgs)
        self.assertNotEqual(record.assignments[0].K, record.assignments[1].K)

    def test_without_targets_one_row_per_physical_process(self):
        """The historical build: one assignment per physical process (initial
        legs in order, final ones as a set), whatever its slot order."""
        table = self.build([(None, (2, 21, 2, 21), None)])
        record = table.records[0]
        self.assertTrue(record.complete())
        self.assertEqual(len(record.assignments), 1)

    def test_rows_already_in_the_table_are_reused(self):
        """A target the table already has a row for takes that row."""
        targets = [(2, 21, 2, 21)]
        table = self.build([(None, (2, 21, 2, 21), None, targets),
                            (None, (2, 21, 2, 21), None, targets)])
        self.assertEqual(table.records[0].assignments[0].K,
                         table.records[1].assignments[0].K)
        self.assertEqual(len(table), 2)

    def test_unserved_target_makes_the_record_incomplete(self):
        """A target no permutation reaches from a base row is listed, and the
        record is no longer complete: its consumer must not fold it."""
        targets = [(2, 21, 2, 21), (1, 21, 1, 21)]
        table = self.build([(None, (2, 21, 2, 21), None, targets)])
        record = table.records[0]
        self.assertEqual(record.unserved, [(1, 21, 1, 21)])
        self.assertEqual([a.pdgs for a in record.assignments],
                         [(2, 21, 2, 21)])
        self.assertFalse(record.complete())
        self.assertFalse(table.complete())

    def test_empty_targets_serve_nothing(self):
        """An empty target list (a beam-swap partner folded into another
        record's mirror) gets no row at all."""
        table = self.build([(None, (21, 2, 2, 21), None, [])])
        self.assertEqual(table.records[0].assignments, [])
        self.assertEqual(len(table), 1)


if __name__ == '__main__':
    unittest.main()
