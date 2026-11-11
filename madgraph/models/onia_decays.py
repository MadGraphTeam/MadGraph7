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
"""Total widths and branching fractions of the onia (PDG 2026).

The total width (GeV; None where it is not measured) and the branching
fractions {(pdg codes of the decay products, sorted): fraction} of the
colour-singlet onia, from the 2026 Review of Particle Physics (F. Takahashi et
al., Int. J. Mod. Phys. A 41, 2630011 (2026), the Listings: OUR FIT / AVERAGE /
EVALUATION): the measured radiative transitions between onia and the lepton
pairs of the vector states. Keyed by the PDG codes of the bound states.

MadSpin uses them for radiative E1/M1 transitions (e.g. "decay chic1 > Jpsi a,
Jpsi > mu+ mu-") and the decays that end them, wherever the param_card of the
events has no DECAY width or BR line of its own. A model may ship its own
values, all_boundstate_decays in its boundstates.py (the same format), which
take precedence over these; a DECAY block or a BR line of the param_card
always wins.
"""
from __future__ import absolute_import

PDG_2026 = {
    # charmonium
    443:    (9.26e-05, {(-11, 11): 5.971e-02, (-13, 13): 5.961e-02,            # J/psi(1S)
                        (22, 441): 1.82e-02}),
    100443: (2.93e-04, {(-11, 11): 7.95e-03, (-13, 13): 8.0e-03,              # psi(2S)
                        (-15, 15): 3.1e-03,
                        (22, 10441): 1.006e-01, (22, 20443): 9.85e-02, (22, 445): 9.63e-02,
                        (22, 441): 3.6e-03, (22, 100441): 5.4e-04}),
    441:    (3.00e-02, {}),                                                    # eta_c(1S)
    100441: (1.16e-02, {}),                                                    # eta_c(2S)
    10443:  (7.8e-04,  {(22, 441): 6.4e-01}),                                  # h_c(1P)
    10441:  (1.240e-02, {(22, 443): 1.35e-02}),                                # chi_c0(1P)
    20443:  (8.2e-04,  {(22, 443): 3.39e-01}),                                 # chi_c1(1P)
    445:    (1.94e-03, {(22, 443): 1.90e-01}),                                 # chi_c2(1P)
    # bottomonium
    553:    (5.402e-05, {(-11, 11): 2.39e-02, (-13, 13): 2.48e-02,            # Upsilon(1S)
                         (-15, 15): 2.60e-02}),
    100553: (3.198e-05, {(-11, 11): 1.91e-02, (-13, 13): 1.93e-02,            # Upsilon(2S)
                         (-15, 15): 2.00e-02,
                         (22, 10551): 3.8e-02, (22, 20553): 6.9e-02, (22, 555): 7.15e-02,
                         (22, 551): 5.5e-04}),
    200553: (2.032e-05, {(-11, 11): 2.18e-02, (-13, 13): 2.18e-02,            # Upsilon(3S)
                         (-15, 15): 2.29e-02,
                         (22, 110551): 5.9e-02, (22, 120553): 1.26e-01, (22, 100555): 1.31e-01,
                         (22, 10551): 2.7e-03, (22, 20553): 9e-04, (22, 555): 1.00e-02,
                         (22, 551): 5.1e-04}),
    551:    (1.0e-02,  {}),                                                    # eta_b(1S)
    10551:  (None, {(22, 553): 1.94e-02}),                                     # chi_b0(1P)
    20553:  (None, {(22, 553): 3.52e-01}),                                     # chi_b1(1P)
    555:    (None, {(22, 553): 1.80e-01}),                                     # chi_b2(1P)
    110551: (None, {(22, 100553): 1.38e-02, (22, 553): 3.8e-03}),              # chi_b0(2P)
    120553: (None, {(22, 100553): 1.81e-01, (22, 553): 9.9e-02}),              # chi_b1(2P)
    100555: (None, {(22, 100553): 8.9e-02, (22, 553): 6.6e-02}),               # chi_b2(2P)
    10553:  (None, {(22, 551): 5.2e-01}),                                      # h_b(1P)
    110553: (None, {(22, 551): 2.2e-01, (22, 100551): 4.8e-01}),               # h_b(2P)
}

def boundstate_decays(model=None):
    """The widths and branching fractions MadSpin uses for the bound states of
    a model: PDG_2026, with the model's own all_boundstate_decays (if its
    boundstates.py has one) in place of the entries it gives."""
    out = dict(PDG_2026)
    if model is not None:
        try:
            from madgraph.models.import_boundstates import get_boundstate_decays_ufo
            out.update(get_boundstate_decays_ufo(model))
        except Exception:
            pass
    return out
