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
"""Beam polarisation (polbeam1/polbeam2 of the run_card) in the reweighting.

The standalone matrix elements apply /to_beampol/ to their incoming legs
1 and 2; the reweighting has to fill it with the polarisation madevent used,
in madevent's internal convention, and swap it when the incoming legs of the
library are the beams in the other order.
"""

from __future__ import absolute_import
import unittest

import madgraph
import madgraph.various.banner as banner_mod
import madgraph.various.lhe_parser as lhe_parser
import madgraph.interface.reweight_interface as reweight_interface

RI = reweight_interface.ReweightInterface


class _Model(object):
    SPIN = {11: 2, 13: 2, 22: 3, 23: 3}

    def get_particle(self, pid):
        return {'spin': self.SPIN[abs(pid)]}


class _Module(object):
    def __init__(self):
        self.calls = []

    def py_set_beampol(self, pol1, pol2):
        self.calls.append((pol1, pol2))


class _Stub(object):
    InvalidCmd = madgraph.InvalidCmd

    def __init__(self, run_card, processes=(((11, -11), (13, -13)),), is_decay=False):
        self.run_card = run_card
        self.is_decay = is_decay
        self.model = _Model()
        self.id_to_path = dict(((tuple(sorted(o[0])), o[1]), (o, '', {}))
                               for o in processes)
        self._spin_cache = {}
        self._beampol_set = {}
        self.beampol = None
        self.beampol_legs = None

    get_beampol = RI.get_beampol
    events_beampol = RI.events_beampol
    _is_fermion = RI._is_fermion
    set_event_beampol = RI.set_event_beampol


def _card(pol1, pol2, **opts):
    card = banner_mod.RunCardLO()
    card['lpp1'] = card['lpp2'] = 0
    card['polbeam1'] = pol1
    card['polbeam2'] = pol2
    for key, value in opts.items():
        card[key] = value
    return card


def _event():
    """e- (beam 1, +z) e+ (beam 2, -z) > mu+ mu-"""
    lines = [' 4 1 1.0 100.0 0.0078 0.118',
             ' 11 -1 0 0 0 0 0.0 0.0 250.0 250.0 0.0 0. 9',
             ' -11 -1 0 0 0 0 0.0 0.0 -250.0 250.0 0.0 0. 9',
             ' -13 1 1 2 0 0 30.0 40.0 150.0 250.0 0.0 0. 9',
             ' 13 1 1 2 0 0 -30.0 -40.0 -150.0 250.0 0.0 0. 9']
    return lhe_parser.Event('<event>\n' + '\n'.join(lines) + '\n</event>')


class TestGetBeampol(unittest.TestCase):

    def test_madevent_convention(self):
        """setrun.f: pol = sign(1 + |polbeam|/100, polbeam)"""
        self.assertEqual(_Stub(_card(-80, 30)).get_beampol(), (-1.8, 1.3))
        self.assertEqual(_Stub(_card(0, 100)).get_beampol(), (1., 2.))

    def test_unpolarised_is_none(self):
        self.assertIsNone(_Stub(_card(0, 0)).get_beampol())

    def test_no_polarisation_without_lo_run_card(self):
        self.assertIsNone(_Stub(banner_mod.RunCardNLO()).get_beampol())
        self.assertIsNone(_Stub(None).get_beampol())
        self.assertIsNone(_Stub(_card(-80, 30), is_decay=True).get_beampol())

    def test_leg_with_a_boson_parton_is_unpolarised(self):
        """export_v4.beam_polarization: madevent drops the polarisation of an
        incoming leg as soon as one process has a non spin-1/2 particle on it"""
        stub = _Stub(_card(-80, 30), processes=[((11, -11), (13, -13)),
                                                ((11, 22), (11, 22))])
        self.assertEqual(stub.get_beampol(), (-1.8, 1.3))
        self.assertEqual(stub.beampol_legs, [True, False])
        stub = _Stub(_card(-80, 30), processes=[((22, 22), (13, -13))])
        self.assertIsNone(stub.get_beampol())

    def test_job_without_model_uses_the_pickled_spins(self):
        """a --multicore=wait job is set up from rwgt.pkl and has no model: the
        spins come from the pickle (flavour-merged codes included), and an
        unknown one is an error rather than an unpolarised leg"""
        stub = _Stub(_card(-80, 30), processes=[((82, -82), (13, -13))])
        stub.model = None
        stub._pickled_spins = {82: 2, -82: 2}
        self.assertEqual(stub.get_beampol(), (-1.8, 1.3))
        self.assertEqual(stub.beampol_legs, [True, True])
        stub = _Stub(_card(-80, 30), processes=[((82, -82), (13, -13))])
        stub.model = None
        self.assertRaises(madgraph.InvalidCmd, stub.get_beampol)

    def test_eva_beam_is_left_to_the_pdf(self):
        self.assertEqual(_Stub(_card(-80, 30, pdlabel2='eva')).get_beampol(), (-1.8, 1.))
        self.assertIsNone(_Stub(_card(-80, 30, pdlabel='eva')).get_beampol())


class TestSetEventBeampol(unittest.TestCase):

    def setUp(self):
        self.stub = _Stub(_card(-80, 30))
        self.stub.beampol = self.stub.get_beampol()
        self.event = _event()
        self.module = _Module()

    def _set(self, order, **opts):
        p = self.event.get_momenta(order)
        self.stub.set_event_beampol(self.module, self.event, p,
                                    list(order[0]) + list(order[1]), 2, **opts)

    def test_beams_in_the_library_order(self):
        self._set([(11, -11), (-13, 13)])
        self.assertEqual(self.module.calls, [(-1.8, 1.3)])

    def test_swapped_library_swaps_the_polarisation(self):
        """`e+ e- >` library for an event whose beam 1 is the e-"""
        self._set([(-11, 11), (-13, 13)])
        self.assertEqual(self.module.calls, [(1.3, -1.8)])

    def test_set_only_on_change(self):
        for _ in range(3):
            self._set([(11, -11), (-13, 13)])
        self._set([(-11, 11), (-13, 13)])
        self.assertEqual(self.module.calls, [(-1.8, 1.3), (1.3, -1.8)])

    def test_mirror_leg_takes_the_other_beam(self):
        """e- e- collider, `e- a > e- a` and its mirror: leg 1 (the e-) is
        polarised, with the value of the beam it comes from, and leg 2 (the
        photon) is not -- as the JB of madevent's matrix.f"""
        stub = _Stub(_card(-80, 30), processes=[((11, 22), (11, 22))])
        stub.beampol = stub.get_beampol()
        lines = [' 4 1 1.0 100.0 0.0078 0.118',
                 ' 22 -1 0 0 0 0 0.0 0.0 50.0 50.0 0.0 0. 9',
                 ' 11 -1 0 0 0 0 0.0 0.0 -250.0 250.0 0.0 0. 9',
                 ' 11 1 1 2 0 0 30.0 40.0 -150.0 160.0 0.0 0. 9',
                 ' 22 1 1 2 0 0 -30.0 -40.0 -50.0 140.0 0.0 0. 9']
        event = lhe_parser.Event('<event>\n' + '\n'.join(lines) + '\n</event>')
        order = [(11, 22), (11, 22)]
        stub.set_event_beampol(self.module, event, event.get_momenta(order),
                               [11, 22, 11, 22], 2)
        self.assertEqual(self.module.calls, [(1.3, 1.)])

    def test_unpolarised_does_nothing(self):
        self.stub.beampol = None
        self._set([(11, -11), (-13, 13)])
        self.assertEqual(self.module.calls, [])

    def test_crossed_library_is_refused(self):
        """an incoming leg of the library that is a (flipped) final-state
        particle of the event is no beam"""
        p = self.event.get_momenta([(11, -11), (-13, 13)])
        crossed = [p[0], tuple(-x for x in p[3]), (p[1][0] * -1,) + p[1][1:], p[2]]
        self.assertRaises(madgraph.InvalidCmd, self.stub.set_event_beampol,
                          self.module, self.event, crossed, [11, 13, 11, -13], 2)

    def test_folded_crossing_is_refused(self):
        """a folded crossing (SMATRIXHEL_IDX at an extended flavor index) with
        a polarised beam would hit the Fortran STOP of the crossing-aware
        SMATRIX and end the session: refused in Python instead"""
        self.assertRaises(madgraph.InvalidCmd, self._set, [(11, -11), (-13, 13)],
                          crossed=True)
        self.assertEqual(self.module.calls, [])

    def test_folded_crossing_with_unpolarised_legs_runs(self):
        """only a polarised value reaches the STOP: a crossed event whose
        incoming legs carry no polarisation (beam 1 is polarised, but its leg
        is a photon) still runs"""
        stub = _Stub(_card(30, 0), processes=[((22, 11), (22, 11))])
        stub.beampol = stub.get_beampol()
        self.assertEqual(stub.beampol, (1.3, 1.))
        self.assertEqual(stub.beampol_legs, [False, True])
        lines = [' 4 1 1.0 100.0 0.0078 0.118',
                 ' 22 -1 0 0 0 0 0.0 0.0 250.0 250.0 0.0 0. 9',
                 ' 11 -1 0 0 0 0 0.0 0.0 -50.0 50.0 0.0 0. 9',
                 ' 22 1 1 2 0 0 30.0 40.0 150.0 160.0 0.0 0. 9',
                 ' 11 1 1 2 0 0 -30.0 -40.0 50.0 140.0 0.0 0. 9']
        event = lhe_parser.Event('<event>\n' + '\n'.join(lines) + '\n</event>')
        order = [(22, 11), (22, 11)]
        stub.set_event_beampol(self.module, event, event.get_momenta(order),
                               [22, 11, 22, 11], 2, crossed=True)
        self.assertEqual(self.module.calls, [(1., 1.)])

    def test_loop_library_is_refused(self):
        self.assertRaises(madgraph.InvalidCmd, self._set, [(11, -11), (-13, 13)],
                          virtual=True)

    def test_library_without_the_setter_is_refused(self):
        self.module = object()
        self.assertRaises(madgraph.InvalidCmd, self._set, [(11, -11), (-13, 13)])


if __name__ == '__main__':
    unittest.main()
