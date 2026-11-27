"""A propagator inside the decay of an on-shell propagator stays within its window.

MG7 restricts an on-shell propagator (a decay chain, t > b e+ ve) to bw_cutoff
widths around its mass, the e_min and e_max of the propagator. The decays below
it are sampled first, the W of t > b W+ before the top. The W must then leave
room for the top in its window, m_W <= e_max(t) - m_b. Bounded only by the
collision energy, the W could exceed it, and the top was sampled in an empty
range: negative Jacobians, negative energies and NaN matrix elements, 0.35% of
the points of p p > t t~, (t > b e+ ve), (t~ > b~ e- ve~).
"""

import math

import phasespace_helpers
import pytest
import torch
from phasespace_helpers import invariant_mass

import madspace as ms

CM_ENERGY = 13000.0
M_T, W_T = 173.0, 1.4915
M_W, W_W = 80.419, 2.0476
M_B = 4.7
BW_CUTOFF = 15.0
WINDOW = (M_T - BW_CUTOFF * W_T, M_T + BW_CUTOFF * W_T)
BATCH_SIZE = 400_000
SEED = 20261115

# u u~ > g > t t~ with t > b W+, W+ > e+ ve, outgoing b e+ ve t~
ONE_DECAY = (
    [M_B, 0.0, 0.0, M_T],
    [(0.0, 0.0, 21), (M_T, W_T, 6), (M_W, W_W, 24)],
    [["o1", "o2", "p2"], ["o0", "p2", "p1"], ["p1", "o3", "p0"], ["p0", "i0", "i1"]],
    [(0, 1, 2)],
)
# and t~ > b~ W-, W- > e- ve~ too, outgoing b e+ ve b~ e- ve~
TWO_DECAYS = (
    [M_B, 0.0, 0.0, M_B, 0.0, 0.0],
    [
        (0.0, 0.0, 21),
        (M_T, W_T, 6),
        (M_W, W_W, 24),
        (M_T, W_T, -6),
        (M_W, W_W, -24),
    ],
    [
        ["o1", "o2", "p2"],
        ["o0", "p2", "p1"],
        ["o4", "o5", "p4"],
        ["o3", "p4", "p3"],
        ["p1", "p3", "p0"],
        ["p0", "i0", "i1"],
    ],
    [(0, 1, 2), (3, 4, 5)],
)
CHANNELS = [ONE_DECAY, TWO_DECAYS]
IDS = ["one decay", "two decays"]


def mapping(channel, window=True):
    """The channel's mapping, with the tops in their window as MG7 builds an
    on-shell propagator, or without a window."""
    outgoing_masses, propagators, vertices, _ = channel
    return ms.PhaseSpaceMapping(
        ms.Topology(
            ms.Diagram(
                [0.0, 0.0],
                outgoing_masses,
                [
                    ms.Propagator(
                        mass=mass,
                        width=width,
                        integration_order=0,
                        e_min=WINDOW[0] if window and abs(pdg_id) == 6 else 0.0,
                        e_max=WINDOW[1] if window and abs(pdg_id) == 6 else 0.0,
                        pdg_id=pdg_id,
                    )
                    for mass, width, pdg_id in propagators
                ],
                vertices,
            )
        ),
        CM_ENERGY,
    )


def top_masses(channel, p):
    """The invariant masses of the decayed tops, shifted by 2 to the momenta of
    the outgoing particles."""
    return [invariant_mass(p, *(2 + i for i in top)) for top in channel[3]]


@pytest.mark.parametrize("channel", CHANNELS, ids=IDS)
def test_points_are_physical(channel):
    """Every point has a positive weight, positive energies and its tops in
    the window, and each W leaves room for its b quark."""
    p, det = phasespace_helpers.sample(mapping(channel), SEED, BATCH_SIZE)
    assert torch.isfinite(det).all() and torch.isfinite(p).all()
    assert (det > 0.0).all()
    assert (p[:, :, 0] >= 0.0).all()
    for top, m_t in zip(channel[3], top_masses(channel, p)):
        assert (m_t >= WINDOW[0] * (1 - 1e-9)).all()
        assert (m_t <= WINDOW[1] * (1 + 1e-9)).all()
        m_w = invariant_mass(p, 2 + top[1], 2 + top[2])
        assert (m_w + M_B <= m_t * (1 + 1e-9)).all()


@pytest.mark.parametrize("channel", CHANNELS, ids=IDS)
def test_volume_matches_without_window(channel):
    """The volume with the tops in their window is that of the mapping without
    a window restricted to the window afterwards: the bound on the W removes
    only points the window excludes."""
    p, det = phasespace_helpers.sample(mapping(channel), SEED + 1, BATCH_SIZE)
    weights = phasespace_helpers.finite_weight(p, det)
    p0, det0 = phasespace_helpers.sample(
        mapping(channel, window=False), SEED + 2, BATCH_SIZE
    )
    inside = torch.ones_like(det0, dtype=torch.bool)
    for m_t in top_masses(channel, p0):
        inside &= (m_t >= WINDOW[0]) & (m_t <= WINDOW[1])
    weights0 = torch.where(inside, phasespace_helpers.finite_weight(p0, det0), 0.0)
    error = math.hypot(
        weights.std(correction=0).item(), weights0.std(correction=0).item()
    ) / math.sqrt(BATCH_SIZE)
    assert abs(weights.mean().item() - weights0.mean().item()) < 4.0 * error


@pytest.mark.parametrize("channel", CHANNELS, ids=IDS)
def test_inverse_round_trip(channel):
    """The inverse map bounds the W the same way and gives back the random
    numbers and the inverse weight."""
    m = mapping(channel)
    r = phasespace_helpers.rand(
        (20_000, m.random_dim()), generator=phasespace_helpers.generator(SEED + 3)
    )
    p, x1, x2, det = m.map_forward([r])
    r_inv, det_inv = m.map_inverse([p, x1, x2])
    assert torch.max(torch.abs(r_inv - r)).item() < 1e-6
    assert torch.max(torch.abs(det * det_inv - 1.0)).item() < 1e-6
