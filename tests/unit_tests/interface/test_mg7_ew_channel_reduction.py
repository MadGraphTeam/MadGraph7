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
"""The electroweak channel reduction of a madmatrix subprocess: the channels it
keeps, the merging of channels into a representative and the resonance keys
the MadNIS channel dropping protects, on hand-built channel metadata."""

import types
import unittest

from madgraph.mg7 import bootstrap as mg7_bootstrap

MASSES = {22: 0., 23: 91.188, 24: 80.4}
WIDTHS = {22: 0., 23: 2.44, 24: 2.05}


def channel(propagators, vertices, on_shell, diagrams):
    return {
        "propagators": propagators,
        "vertices": vertices,
        "on_shell_propagators": on_shell,
        "dollar_propagators": [],
        "diagrams": [
            {
                "diagram": diagram,
                "permutation": list(range(n_particles)),
                "active_flavors": flavors,
                "active_colors": [0],
                "propagator_pdgs": propagators,
            }
            for diagram, flavors, n_particles in diagrams
        ],
    }


def t_channel_meta():
    """u e- > u e- through a t-channel photon (channel 0) or Z (channel 1)."""
    vertices = [["i0", "o0", "p0"], ["i1", "o1", "p0"]]
    return {
        "incoming": [2, 11],
        "outgoing": [2, 11],
        "diagram_count": 2,
        "channels": [
            channel([22], vertices, [], [(0, [0, 1], 4)]),
            channel([23], vertices, [], [(1, [0], 4)]),
        ],
    }


def s_channel_meta(gamma_flavors):
    """u u~ > e+ e- a, the photon from the u line and the lepton pair from an
    s-channel Z (channel 0, flavor 0) or photon (one channel per entry of
    gamma_flavors)."""
    vertices = [["i0", "o2", "p0"], ["i1", "p0", "p1"], ["o0", "o1", "p1"]]
    return {
        "incoming": [2, -2],
        "outgoing": [-11, 11, 22],
        "diagram_count": 1 + len(gamma_flavors),
        "channels": [channel([2, 23], vertices, [1], [(0, [0], 5)])] + [
            channel([2, 22], vertices, [], [(i + 1, flavors, 5)])
            for i, flavors in enumerate(gamma_flavors)
        ],
    }


def cascade_meta():
    """u u~ > e+ e- e+ through W* > e+ Z*, Z* > e- e+. W and Z cannot both be on
    shell (m_Z > m_W), so the channel has a subchannel for each."""
    vertices = [["i0", "i1", "p0"], ["o0", "p0", "p1"], ["o1", "o2", "p1"]]
    return {
        "incoming": [2, -2],
        "outgoing": [-11, 11, -11],
        "diagram_count": 1,
        "channels": [channel([24, 23], vertices, [0, 1], [(0, [0], 5)])],
    }


@unittest.skipUnless(mg7_bootstrap.madspace_is_installed(),
                     'madspace is not installed')
class EWChannelReductionTest(unittest.TestCase):

    def subprocess(self, meta, mode):
        import madgraph.various.banner as banner_mod
        from madgraph.mg7 import subprocess_madmatrix as smm
        run_card = banner_mod.RunCardMG7()
        run_card["phasespace"]["ew_channel_reduction_mode"] = mode
        process = types.SimpleNamespace(
            run_card=run_card,
            get_mass=lambda pid: MASSES.get(abs(pid), 0.),
            get_width=lambda pid: WIDTHS.get(abs(pid), 0.),
        )
        subproc = object.__new__(smm.MadgraphSubprocess)
        subproc.meta = meta
        subproc.process = process
        subproc.unmerged_meta = None
        subproc.subproc_id = 0
        return subproc, smm.build_multi_channel_data(meta, process)

    def test_t_channel_bosons_are_merged(self):
        for mode in ("full", "partial", "t_channel"):
            subproc, mcdata = self.subprocess(t_channel_meta(), mode)
            reduced = subproc.reduce_ew_channels(mcdata)
            # the photon channel has more flavors and the lighter t-channel
            self.assertEqual(reduced.diagram_indices, [[0]], mode)
            # the Z diagram adds to the photon channel's weight
            self.assertEqual(reduced.amp2_remaps, [[0, 0]], mode)

    def test_s_channel_coverage(self):
        # full: the gamma* channel adds nothing to the Z channel and is dropped
        expected = {"full": [[0]], "partial": [[0], [1]],
                    "t_channel": [[0], [1]]}
        for mode, diagrams in expected.items():
            subproc, mcdata = self.subprocess(s_channel_meta([[0]]), mode)
            reduced = subproc.reduce_ew_channels(mcdata)
            self.assertEqual(reduced.diagram_indices, diagrams, mode)
        subproc, mcdata = self.subprocess(s_channel_meta([[0]]), "full")
        self.assertEqual(
            subproc.reduce_ew_channels(mcdata).amp2_remaps, [[0, -1]]
        )

    def test_s_channel_flavor_coverage(self):
        # the gamma* group is kept for flavor 1, with the channel that has it
        # as representative and the other one merged into it
        subproc, mcdata = self.subprocess(s_channel_meta([[0], [0, 1]]), "full")
        reduced = subproc.reduce_ew_channels(mcdata)
        self.assertEqual(reduced.diagram_indices, [[0], [2]])
        self.assertEqual(reduced.amp2_remaps, [[0, 1, 1]])

    def test_resonances_per_subchannel(self):
        from madgraph.mg7 import subprocess_madmatrix as smm
        _, mcdata = self.subprocess(cascade_meta(), "full")
        topos = mcdata.topologies[0]
        self.assertEqual(len(topos), 2)
        per_topo = [
            {pdg for pdg, _ in smm.topology_structure(topo).resonances}
            for topo in topos
        ]
        self.assertEqual(sorted(map(sorted, per_topo)), [[23], [24]])
        keys = smm.resonance_keys(topos, mcdata.permutations[0],
                                  mcdata.active_flavors[0])
        self.assertEqual({pdg for _, pdg, _ in keys}, {23, 24})
        for topo, pdgs in zip(topos, per_topo):
            keys = smm.resonance_keys([topo], mcdata.permutations[0],
                                      mcdata.active_flavors[0])
            self.assertEqual({pdg for _, pdg, _ in keys}, pdgs)


if __name__ == "__main__":
    unittest.main()
