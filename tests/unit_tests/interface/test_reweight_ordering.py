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
# For more information, visit madgraph.phys.ucl.ac.be and amcatnlo.web.cern.ch
#
################################################################################
"""The reweighting's choice of legs must not depend on the LHE's line order.

Two places picked "leg n" by walking the event's own lines while every other
quantity they were combined with -- the momenta handed to the matrix element,
pdg, POS of GET_DENSITY -- is laid out in the matrix element's leg order:

* ReweightInterface.method_boost_event's ``frame_id`` branch (a non-default
  ``me_frame``). It never ran: it died on an undefined ``FourMomenta`` and on
  ``str.reverse``. Fixed by name only, it would have boosted to the rest frame
  of whatever particle the LHE wrote at that place.
* DensityInterface: its boost counted every LHE line, status-2 ones included,
  against matrix-element leg numbers, and ranking by an observable paired
  ``pdg[i]`` (a leg) with ``event[i]`` (a line) -- so the boost frame, the
  helicity axis and the POS given to GET_DENSITY could all belong to another
  particle.

The tests feed the same physical event written in several line orders and
require identical answers.

The original matrix element must also be put in the frame the events were
generated in, which was read from the wrong place: ``frame_id`` is only computed
when the include files are written, so a run_card read back from an LHE banner
always said 6 (the partonic c.m.), and an mg7 banner has none, while there
``me_frame = []`` means no boost at all. And the frame has to be reached the way
the generation reached it, through the partonic c.m.: a single boost straight
from the lab leaves a Wigner rotation behind.
"""

from __future__ import absolute_import
import math
import unittest

import madgraph.various.banner as banner_mod
import madgraph.various.lhe_parser as lhe_parser
import madgraph.interface.reweight_interface as reweight_interface


def _event(final, resonance=False):
    """g g > final, each final-state particle with its own momentum (fixed by
    its identity, not its place), so that permuting ``final`` permutes lines
    of one and the same physical event."""
    momenta = {('t', 0): (40., 10., 30.), ('t', 1): (-25., 60., -80.),
               ('tx', 0): (-35., -50., 20.), ('tx', 1): (20., -20., 130.)}
    lines = [' %d 1 1.0 100.0 0.0078 0.118' % (2 + len(final) + resonance),
             ' 21 -1 0 0 501 502 0.0 0.0 600.0 600.0 0.0 0. 9',
             ' 21 -1 0 0 503 501 0.0 0.0 -600.0 600.0 0.0 0. 9']
    if resonance:
        lines.append(' 25 2 1 2 0 0 0.0 0.0 0.0 700.0 125.0 0. 9')
    for name in final:
        pid = 6 if name[0] == 't' else -6
        px, py, pz = momenta[name]
        e = math.sqrt(173. ** 2 + px ** 2 + py ** 2 + pz ** 2)
        lines.append(' %d 1 1 2 0 0 %r %r %r %r 173.0 0. 9'
                     % (pid, px, py, pz, e))
    return lhe_parser.Event('<event>\n' + '\n'.join(lines) + '\n</event>')


ORDER = [(21, 21), (6, -6, 6, -6)]
# the four tops, identified by name; the event writes them in these orders
LAYOUTS = [[('t', 0), ('tx', 0), ('t', 1), ('tx', 1)],   # madevent
           [('t', 0), ('t', 1), ('tx', 0), ('tx', 1)],   # aMC@NLO
           [('tx', 0), ('t', 0), ('tx', 1), ('t', 1)]]


def _rest(momenta, legs):
    """the summed three-momentum of ``legs`` (0-based)"""
    return [sum(momenta[leg][k] for leg in legs) for k in (1, 2, 3)]


def _zboost(momenta, rapidity):
    """``momenta`` boosted along z by ``rapidity``: the same event seen from a
    lab frame in which the partonic system moves along the beam"""
    ch, sh = math.cosh(rapidity), math.sinh(rapidity)
    return [(e * ch + pz * sh, px, py, pz * ch + e * sh)
            for e, px, py, pz in momenta]


def _mg7_card(me_frame):
    card = banner_mod.RunCardMG7()
    card['run']['me_frame'] = me_frame
    return card


class _BaseStub(object):
    class _Banner(object):
        def __init__(self, run_card):
            self.run_card = run_card

    def __init__(self, run_card):
        """``run_card`` a RunCardMG7, or a plain dict standing for a legacy
        (LO/NLO) one; a list is taken as the legacy me_frame"""
        if isinstance(run_card, list):
            run_card = {'me_frame': run_card}
        self.banner = self._Banner(run_card)
        self.keep_ordering = False
        self.boost_event = False

    method_boost_event = reweight_interface.ReweightInterface.method_boost_event
    get_me_frame = reweight_interface.ReweightInterface.get_me_frame
    boost_momenta_to_rest_frame = staticmethod(
        reweight_interface.ReweightInterface.boost_momenta_to_rest_frame)
    boost_momenta_to_me_frame = staticmethod(
        reweight_interface.ReweightInterface.boost_momenta_to_me_frame)


class TestFrameIdBoost(unittest.TestCase):
    """me_frame in the reweighting: entry n is leg n of the matrix element,
    counted from 1."""

    def _boosted(self, run_card, layout, resonance=False, rapidity=0.):
        event = _event(layout, resonance)
        all_p = [_zboost(event.get_momenta(ORDER), rapidity)]
        return _BaseStub(run_card).method_boost_event(event, all_p, ORDER, 0)

    def test_the_chosen_legs_end_up_at_rest(self):
        """me_frame = [3, 4]: the first top and first anti-top of the matrix
        element, whichever LHE line they sit on."""
        for layout in LAYOUTS:
            out = self._boosted([3, 4], layout)
            self.assertEqual(len(out), 1)
            for comp in _rest(out[0], [2, 3]):
                self.assertAlmostEqual(comp, 0., places=8)

    def test_the_line_order_does_not_matter(self):
        ref = self._boosted([3, 4], LAYOUTS[0])
        for layout in LAYOUTS[1:]:
            for resonance in (False, True):
                out = self._boosted([3, 4], layout, resonance)
                for a, b in zip(out[0], ref[0]):
                    for x, y in zip(a, b):
                        self.assertAlmostEqual(x, y, places=8)

    def test_a_single_leg_is_exactly_at_rest(self):
        """HELAS takes the frame's z axis only for a momentum exactly at rest;
        the boost arithmetic alone leaves ~1e-14."""
        out = self._boosted([5], LAYOUTS[1], rapidity=0.8)
        self.assertEqual(out[0][4][1:], (0., 0., 0.))
        self.assertAlmostEqual(out[0][4][0], 173., places=8)

    def test_frame_id_is_not_read(self):
        """frame_id is only computed when the include files are written: a
        run_card read back from an LHE banner keeps its default 6 whatever
        me_frame says. me_frame is what decides."""
        out = self._boosted({'me_frame': [5], 'frame_id': 6}, LAYOUTS[0])
        self.assertEqual(out[0][4][1:], (0., 0., 0.))

    def test_the_frame_is_reached_through_the_partonic_cm(self):
        """The same event seen from any longitudinally boosted lab frame must
        end up with the same momenta: the generation went lab -> partonic
        c.m. -> me_frame, and a single boost from the lab would leave a Wigner
        rotation that turns the spin axis of the leg held at rest."""
        ref = self._boosted([5], LAYOUTS[0])
        for rapidity in (-1.1, 0.6):
            out = self._boosted([5], LAYOUTS[0], rapidity=rapidity)
            for a, b in zip(out[0], ref[0]):
                for x, y in zip(a, b):
                    self.assertAlmostEqual(x, y, places=7)
        # and the test is sensitive to it: the single boost straight from the
        # lab does put leg 5 at rest, but rotates everything else
        event = _event(LAYOUTS[0])
        lab = _zboost(event.get_momenta(ORDER), 0.6)
        direct = reweight_interface.ReweightInterface.boost_momenta_to_rest_frame(
            lab, lhe_parser.FourMomentum(lab[4]), 4)
        self.assertGreater(max(abs(x - y) for a, b in zip(direct, ref[0])
                               for x, y in zip(a, b)), 1.)

    def test_the_partonic_cm_is_left_to_the_zboost(self):
        """[1, 2] -- frame_id 6, which madevent and aMC@NLO skip outright,
        decays included -- and [] (no leg selected, so no boost of momenta
        already in the partonic c.m.) both mean the partonic c.m. at LO/NLO,
        and so do legs the matrix element does not have."""
        for me_frame in ([1, 2], [2, 1], [], [7]):
            stub = _BaseStub(me_frame)
            self.assertIsNone(stub.get_me_frame(2, 6), me_frame)
            self.assertIsNone(stub.get_me_frame(1, 6), me_frame)
        self.assertIsNone(_BaseStub({}).get_me_frame(2, 6))

    def test_mg7_default_applies_no_boost(self):
        """mg7's [] means the matrix element saw the lab-frame momenta it
        wrote out, not madevent's partonic c.m."""
        event = _event(LAYOUTS[0])
        lab = _zboost(event.get_momenta(ORDER), 0.6)
        all_p = [list(lab)]
        out = _BaseStub(_mg7_card([])).method_boost_event(event, all_p,
                                                          ORDER, 0)
        self.assertIs(out, all_p)
        self.assertEqual(out[0], lab)

    def test_mg7_frame(self):
        stub = _BaseStub(_mg7_card([5]))
        self.assertEqual(stub.get_me_frame(2, 6), [5])
        out = self._boosted(_mg7_card([5]), LAYOUTS[0], rapidity=0.6)
        self.assertEqual(out[0][4][1:], (0., 0., 0.))
        # unlike at LO, [1, 2] of a decay is a real frame in mg7 -- the
        # decaying particle and its first product -- while [1] is the
        # decaying particle's rest frame, i.e. no frame to reach
        self.assertEqual(_BaseStub(_mg7_card([1, 2])).get_me_frame(1, 4),
                         [1, 2])
        self.assertIsNone(_BaseStub(_mg7_card([1])).get_me_frame(1, 4))
        self.assertIsNone(_BaseStub(_mg7_card([1, 2])).get_me_frame(2, 6))

    def test_the_new_hypothesis_is_not_affected(self):
        """Only the original matrix element (hypp_id 0) is put in the
        generation frame."""
        event = _event(LAYOUTS[0])
        all_p = [event.get_momenta(ORDER)]
        stub = _BaseStub([5])
        out = stub.method_boost_event(event, all_p, ORDER, 1)
        self.assertNotEqual(out[0][4][1:], (0., 0., 0.))


class _DensityStub(object):
    """the DensityInterface pieces that choose, rank and boost"""

    def __init__(self, momenta_boost):
        self.momenta_boost = momenta_boost
        self.keep_ordering = False

    method_boost_event = reweight_interface.DensityInterface.method_boost_event
    chose_particle_user_input = \
        reweight_interface.DensityInterface.chose_particle_user_input
    find_position_particles_default_order = \
        reweight_interface.DensityInterface.find_position_particles_default_order
    find_position_particles_new_order = \
        reweight_interface.DensityInterface.find_position_particles_new_order
    find_position_particles_with_observable = \
        reweight_interface.DensityInterface.find_position_particles_with_observable
    boost_momenta_to_rest_frame = staticmethod(
        reweight_interface.ReweightInterface.boost_momenta_to_rest_frame)


class TestDensityInterfaceLegs(unittest.TestCase):

    PROPERTIES = [p for p in dir(lhe_parser.FourMomentum)
                  if isinstance(getattr(lhe_parser.FourMomentum, p), property)]
    PDG = list(ORDER[0]) + list(ORDER[1])

    def _choose(self, layout, user_input, resonance=False, fortran=False):
        event = _event(layout, resonance)
        lab_p = event.get_momenta(ORDER)
        stub = _DensityStub(user_input)
        return stub.chose_particle_user_input(lab_p, self.PDG, self.PROPERTIES,
                                              ORDER, user_input, 'test',
                                              fortran), lab_p, event

    def test_ranked_choice_is_the_same_particle_in_every_layout(self):
        """'the hardest top' names one particle, however the LHE lists them."""
        chosen = set()
        for layout in LAYOUTS:
            for resonance in (False, True):
                positions, lab_p, _ = self._choose(layout, [[6], 'pt', []],
                                                   resonance)
                self.assertEqual(len(positions), 1)
                self.assertEqual(self.PDG[positions[0]], 6)
                pts = [math.hypot(p[1], p[2]) for p, pid in
                       zip(lab_p, self.PDG) if pid == 6]
                chosen_p = lab_p[positions[0]]
                self.assertEqual(math.hypot(chosen_p[1], chosen_p[2]), max(pts))
                chosen.add(tuple(round(x, 9) for x in chosen_p))
        self.assertEqual(len(chosen), 1)

    def test_default_choice_is_a_leg_of_the_right_flavour(self):
        for layout in LAYOUTS:
            positions, _, _ = self._choose(layout, [[6, -6], '', []])
            self.assertEqual([self.PDG[p] for p in positions], [6, -6])

    def test_boost_puts_the_chosen_legs_at_rest(self):
        """the boost is built from the chosen legs of all_p, in the matrix
        element's order, not from LHE lines -- a status-2 line included"""
        for layout in LAYOUTS:
            for resonance in (False, True):
                user_input = [[6, -6], '', []]
                positions, lab_p, event = self._choose(layout, user_input,
                                                       resonance)
                out = _DensityStub(user_input).method_boost_event(
                                    event, [list(lab_p)], ORDER, 0, positions)
                for comp in _rest(out[0], positions):
                    self.assertAlmostEqual(comp, 0., places=8)

    def test_boost_keeps_the_momenta_it_was_given(self):
        """all_p carries the Monte-Carlo-mass projection; boosting the event
        and reading the momenta back from it, as this did, threw it away."""
        user_input = [[6], '', []]
        positions, lab_p, event = self._choose(LAYOUTS[1], user_input)
        shifted = [tuple(x * 1.01 for x in p) for p in lab_p]
        out = _DensityStub(user_input).method_boost_event(
                                event, [shifted], ORDER, 0, positions)
        # the chosen leg of the *given* momenta is at rest, which it would not
        # be if the event's own momenta had been read back
        self.assertEqual(len(out), 1)
        for comp in _rest(out[0], positions):
            self.assertAlmostEqual(comp, 0., places=8)


if __name__ == '__main__':
    unittest.main()
