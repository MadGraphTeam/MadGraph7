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
"""The planar diagrams the MLM clustering of [amplicol] events follows
(madgraph.mg7.amplicol_clustering), from the vertices of the sm model."""

import json
import unittest

import madgraph.iolibs.export_amplicol as export_amplicol
import madgraph.mg7.amplicol_clustering as clustering
import madgraph.models.import_ufo as import_ufo


def gluon_trace(n):
    """LHE colour tags of n gluons in one trace, in leg order."""
    flow = [[501 + i, 501 + (i - 1) % n] for i in range(n)]
    # LHE tags of incoming legs are those of the outgoing antiparticle swapped
    return [[a, c] if i < 2 else [c, a] for i, (c, a) in enumerate(flow)]


def as_sets(diagrams):
    return {frozenset(diagram) for diagram in diagrams}


class TestPlanarDiagrams(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        model = import_ufo.import_model('sm')
        # through json, as the launcher reads it
        cls.data = json.loads(json.dumps(export_amplicol.clustering_model(model)))
        cls.table = clustering.VertexTable(cls.data)

    def diagrams(self, pdgs, flow, **orders):
        return clustering.planar_diagrams(self.table, pdgs, 2, flow, orders)

    def test_the_vertices_have_no_flavour_grouping(self):
        pdgs = {pdg for vertex, _ in self.data['vertices'] for pdg in vertex}
        self.assertFalse(pdgs & {81, 82, 83, -81, -82, -83})
        # an outgoing u~ and d merge into an outgoing W-
        self.assertEqual(self.table.merges[(-2, 1)], [(-24, {'QED': 1})])
        self.assertEqual(self.table.anti(2), -2)
        self.assertEqual(self.table.color(-2), -3)

    def test_gluons_in_one_trace_have_the_planar_diagrams(self):
        """Catalan(n - 2) cubic planar trees, and (2n - 5)!! without a flow."""
        catalan = {4: 2, 5: 5, 6: 14, 7: 42, 8: 132}
        all_trees = {4: 3, 5: 15, 6: 105}
        for n, count in catalan.items():
            self.assertEqual(len(self.diagrams([21] * n, gluon_trace(n))), count)
        for n, count in all_trees.items():
            self.assertEqual(len(self.diagrams([21] * n, None)), count)

    def test_a_quark_line_keeps_its_gluons_in_order(self):
        """u u~ > g g in the order (u, g2, g3, u~): the s-channel gluon and the
        t-channel quark next to the first gluon, not the crossed one."""
        flow = [[503, 0], [0, 501], [502, 501], [503, 502]]
        self.assertEqual(as_sets(self.diagrams([2, -2, 21, 21], flow)),
                         {frozenset({(12, 21)}), frozenset({(6, 2)})})
        self.assertEqual(len(self.diagrams([2, -2, 21, 21], None)), 3)

    def test_colour_singlets_attach_anywhere_on_the_quark_line(self):
        """u d~ > e+ ve g: the W off the quark line on either side of the
        gluon, within the QED order of the process."""
        flow = [[501, 0], [0, 502], [0, 0], [0, 0], [501, 502]]
        pdgs = [2, -1, -11, 12, 21]
        self.assertEqual(
            as_sets(self.diagrams(pdgs, flow, QED=2)),
            {frozenset({(12, 24), (18, 1)}), frozenset({(12, 24), (14, 2)})})
        self.assertEqual(self.diagrams(pdgs, flow, QED=1), [])

    def test_the_u1_part_of_a_gluon_joins_quark_lines(self):
        """u d > u d through a gluon, in the flow exchanging the colours and in
        the one keeping them on each line (the U(1) part)."""
        exchanged = [[501, 0], [502, 0], [502, 0], [501, 0]]
        kept = [[501, 0], [502, 0], [501, 0], [502, 0]]
        for flow in (exchanged, kept):
            self.assertEqual(as_sets(self.diagrams([2, 1, 2, 1], flow, QED=0)),
                             {frozenset({(10, 21)})})
        # with electroweak vertices, the W exchanges the colours like a gluon,
        # the photon and the Z keep them
        self.assertEqual(as_sets(self.diagrams([2, 1, 2, 1], exchanged)),
                         {frozenset({(10, 21)}), frozenset({(6, 24)})})
        self.assertEqual(
            as_sets(self.diagrams([2, 1, 2, 1], kept)),
            {frozenset({(10, pdg)}) for pdg in (21, 22, 23)})

    def test_the_u1_part_of_a_gluon_has_no_gluon_vertex(self):
        """g g > u u~ with the quarks in a colour singlet has no diagram: the
        U(1) gluon does not couple to gluons."""
        flow = [[501, 502], [502, 501], [503, 0], [0, 503]]
        self.assertEqual(self.diagrams([21, 21, 2, -2], flow), [])

    def test_t_channel_lines_are_given_from_both_beams(self):
        anti = self.table.anti
        # legs 1 and 2 away from leg 0: a t-channel line
        self.assertEqual(clustering.madspace_lines(((6, 2),), 4, anti),
                         [(6, 2), (9, -2)])
        # legs 2 and 3: a final-state line
        self.assertEqual(clustering.madspace_lines(((12, 21),), 4, anti),
                         [(12, 21)])


if __name__ == '__main__':
    unittest.main()
