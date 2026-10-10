"""Pairwise cuts handed from PhaseSpaceMapping to ColorOrderedMapping.

In color_ordered mode PhaseSpaceMapping reorders the pairwise invariant-mass
and delta-R cut matrices of Cuts (indexed by outgoing position) into the order
of the t-channel children and passes them to ColorOrderedMapping, which turns
them into floors on the invariants it samples. A stray extra increment in the
loop building that reordering once skipped every odd child, so any pair cut
involving the second, fourth, ... child silently never reached the mapping.

The process is e+ e- > d u s c: leptonic, so every random number goes straight
to the t-channel mapping and PhaseSpaceMapping returns exactly the momenta of a
ColorOrderedMapping fed the same inputs, and every outgoing parton has its own
flavour, so a cut can name one specific pair.
"""

import math

import numpy as np
import pytest

import madspace as ms

O = ms.Observable
PSM = ms.PhaseSpaceMapping

CM_ENERGY = 1000.0
PIDS = [11, -11, 1, 2, 3, 4]
N_OUT = len(PIDS) - 2
MASSES = [0.0] * len(PIDS)
PAIR_MASS_CUT = 400.0
PT_CUT = 100.0
DELTA_R_CUT = 2.0
SEED = 20260918

# PhaseSpaceMapping adds a few graph operations of its own, which typically
# moves the momenta at the 1e-7 GeV level; a missing floor moves them by up to
# O(100 GeV).
ATOL = 1e-5

SINGLE_CHAIN = [0, 2, 3, 4, 5, 1]  # one chain d u s c
SPLIT_2_2 = [0, 2, 3, 1, 4, 5]  # {d u} | {s c}: central 2->2
SPLIT_1_3 = [0, 2, 1, 3, 4, 5]  # {d} | {u s c}: DoubleT + 2->3 peel


def pair_mass(a, b):
    """Cut on the invariant mass of outgoing children a and b."""
    selection = [[PIDS[a + 2]], [PIDS[b + 2]]]
    return ms.CutItem(O(PIDS, O.obs_pair_mass, selection), min=PAIR_MASS_CUT)


def delta_r(a, b):
    selection = [[PIDS[a + 2]], [PIDS[b + 2]]]
    return ms.CutItem(O(PIDS, O.obs_delta_r, selection), min=DELTA_R_CUT)


def pt_all():
    return ms.CutItem(O(PIDS, O.obs_pt, [O.jet_pids]), min=PT_CUT)


def matrix(pairs, value):
    m = [[0.0] * N_OUT for _ in range(N_OUT)]
    for a, b in pairs:
        m[a][b] = m[b][a] = value
    return m


def random_inputs(random_dim, discrete_dim, n, seed):
    rng = np.random.default_rng(seed)
    r = rng.random((n, random_dim))
    d = rng.integers(0, 2, size=(n, discrete_dim)).astype(np.int32)
    return r, d


def psm_momenta(order, cut_items, r, d):
    mapping = PSM(
        MASSES,
        CM_ENERGY,
        leptonic=True,
        mode=PSM.color_ordered,
        cuts=ms.Cuts(cut_items),
        color_order=order,
    )
    assert mapping.random_dim() == r.shape[1]
    assert mapping.discrete_dim() == d.shape[1]
    inputs = [r, d] if d.shape[1] else [r]
    return np.asarray(mapping.map_forward(inputs)[0])


def co_momenta(order, pt_min, m_inv_min, dr_min, r, d):
    mapping = ms.ColorOrderedMapping(order, 0.8, 0.8, pt_min, m_inv_min, dr_min)
    n = r.shape[0]
    inputs = [r[:, i].copy() for i in range(r.shape[1])]
    inputs += [d[:, j].copy() for j in range(d.shape[1])]
    conditions = [np.full(n, CM_ENERGY)] + [np.zeros(n) for _ in range(N_OUT)]
    out = mapping.map_forward(inputs, conditions)
    return np.stack([np.asarray(p) for p in out[:-1]], axis=1)


def same_points(p, q):
    """Per point: both NaN, or both finite and equal within ATOL.

    The bound narrows ranges, so some points land where nothing is left to
    sample and come out as NaN (weight zero); those count as the same point."""
    nan_p = ~np.all(np.isfinite(p), axis=(1, 2))
    nan_q = ~np.all(np.isfinite(q), axis=(1, 2))
    close = np.all(np.abs(np.nan_to_num(p - q)) <= ATOL, axis=(1, 2))
    return (nan_p & nan_q) | (~nan_p & ~nan_q & close)


# Each case is a pair cut involving an odd child in a topology where the bound
# does shape the sampling: a set's composite mass, a 2->3 peel, or the rest
# system of a chain.
# (color order, cut items, pt_min, m_inv_min, dr_min, id)
CASES = [
    (SPLIT_2_2, [pair_mass(0, 1)], [0.0] * N_OUT,
     matrix([(0, 1)], PAIR_MASS_CUT), matrix([], 0.0), "2+2 m(d,u) set mass"),
    (SPLIT_2_2, [pair_mass(2, 3)], [0.0] * N_OUT,
     matrix([(2, 3)], PAIR_MASS_CUT), matrix([], 0.0), "2+2 m(s,c) set mass"),
    (SINGLE_CHAIN, [pair_mass(1, 2)], [0.0] * N_OUT,
     matrix([(1, 2)], PAIR_MASS_CUT), matrix([], 0.0), "chain m(u,s) peel"),
    (SINGLE_CHAIN, [pair_mass(1, 3)], [0.0] * N_OUT,
     matrix([(1, 3)], PAIR_MASS_CUT), matrix([], 0.0), "chain m(u,c) rest"),
    (SPLIT_1_3, [pt_all(), delta_r(1, 2)], [PT_CUT] * N_OUT,
     matrix([], 0.0), matrix([(1, 2)], DELTA_R_CUT), "1+3 dR(u,s) peel"),
]  # fmt: skip


@pytest.mark.parametrize(
    "order, cut_items, pt_min, m_inv_min, dr_min",
    [c[:5] for c in CASES],
    ids=[c[5] for c in CASES],
)
def test_pair_cut_reaches_color_ordered_mapping(
    order, cut_items, pt_min, m_inv_min, dr_min
):
    """PhaseSpaceMapping must hand ColorOrderedMapping the full pair-cut
    matrices: its momenta equal those of a ColorOrderedMapping built by hand
    with the cut, and differ from one built without it."""
    probe = ms.ColorOrderedMapping(order)
    r, d = random_inputs(probe.random_dim(), probe.discrete_dim(), 2000, SEED)

    p_psm = psm_momenta(order, cut_items, r, d)
    p_with = co_momenta(order, pt_min, m_inv_min, dr_min, r, d)
    no_pair = matrix([], 0.0)
    p_without = co_momenta(order, pt_min, no_pair, no_pair, r, d)

    # Near degenerate configurations the extra operations in
    # PhaseSpaceMapping can be amplified past ATOL, so allow a few points.
    assert np.mean(same_points(p_psm, p_with)) > 0.99

    # Without the comparison below the test would pass for a cut that does
    # not affect this topology at all, and prove nothing. A floor only moves
    # the points where it lies above the kinematic one: all of them for the
    # pair masses here, about half for the delta-R cut.
    assert np.mean(same_points(p_with, p_without)) < 0.9


def sample(order, cuts, n, seed):
    mapping = PSM(
        MASSES,
        CM_ENERGY,
        leptonic=True,
        mode=PSM.color_ordered,
        cuts=cuts,
        color_order=order,
    )
    r, d = random_inputs(mapping.random_dim(), mapping.discrete_dim(), n, seed)
    inputs = [r, d] if mapping.discrete_dim() else [r]
    p_ext, _x1, _x2, det = mapping.map_forward(inputs)
    det = np.nan_to_num(np.asarray(det)) * 2.0 ** mapping.discrete_dim()
    return np.asarray(p_ext), det


def invariant_mass(p, a, b):
    total = p[:, a + 2] + p[:, b + 2]
    m2 = total[:, 0] ** 2 - np.sum(total[:, 1:] ** 2, axis=1)
    return np.sqrt(np.maximum(m2, 0.0))


# The central 2->2 split, whose set masses are the only place these cuts
# enter. The single chain is left out on purpose: its weights have a heavy
# tail with or without cuts, which would make a volume comparison flaky.
@pytest.mark.parametrize("pair", [(0, 1), (2, 3)], ids=["m(d,u)", "m(s,c)"])
def test_pair_cut_on_odd_child_raises_efficiency_not_volume(pair):
    """Handing the cut to the mapping must raise the fraction of points that
    pass it, and must leave the volume of the cut region unchanged."""
    n = 200_000
    a, b = pair

    p_free, det_free = sample(SPLIT_2_2, None, n, SEED)
    external = np.where(invariant_mass(p_free, a, b) >= PAIR_MASS_CUT, det_free, 0.0)

    p_cut, piped = sample(SPLIT_2_2, ms.Cuts([pair_mass(a, b)]), n, SEED + 1)
    assert np.all(invariant_mass(p_cut, a, b)[piped != 0.0] >= PAIR_MASS_CUT)
    assert np.all(piped >= 0.0)

    # with the bound dropped, both fractions are the same (0.84 / 0.17)
    eff_external = np.mean(external > 0.0)
    eff_piped = np.mean(piped > 0.0)
    assert eff_piped > eff_external + 0.03

    error = math.sqrt((external.var() + piped.var()) / n)
    assert error > 0.0
    assert abs(piped.mean() - external.mean()) < 5.0 * error
