"""Angular and rapidity ranges implied by pt and eta cuts.

The partonic centre-of-mass frame is the lab up to a boost along the beam with
rapidity Y = log(x1 / x2) / 2. Two consequences of the cuts follow, and
PhaseSpaceMapping samples only the region they leave:

  * every final-state particle has |y| <= |eta|, and the rapidity of a sum of
    momenta is a weighted mean of theirs, so when every particle has an eta
    cut, |Y| <= the largest of them;
  * in a two-body decay of the partonic system (the root of an s-channel
    topology) both products have pt = p sin(theta), and a product's lab
    rapidity is Y + atanh(beta cos(theta)), so its pt and eta cuts bound
    cos(theta) from both sides. A composite product has no pt cut of its own,
    but its rapidity is bounded like Y when every particle in it has an eta
    cut.

Also checked: sqrt(s_hat) >= sum_i m_T,i minimised over transverse momenta
that add up to zero, which is more than the plain sum when one pt cut
dominates.

Pinned down here: no generated point leaves the windows, the windows are
reached, the integral over the cut region does not change, the mapping still
inverts, and nothing is restricted when the cuts do not imply it.
"""

import math

import numpy as np
import pytest
from pytest import approx

import madspace as ms

O = ms.Observable

CM_ENERGY = 13000.0
BATCH_SIZE = 100_000
SEED = 20261006
M_Z = 91.188

PT_LEPTON = 10.0
ETA_LEPTON = 2.5
PT_JET = 20.0
ETA_JET = 2.5


def s_channel_two_body(masses=(0.0, 0.0)):
    return ms.Topology(
        ms.Diagram(
            [0.0, 0.0],
            list(masses),
            [ms.Propagator(0.0, 0.0)],
            [["i0", "i1", "p0"], ["p0", "o0", "o1"]],
        )
    )


def s_channel_four_body():
    """u u~ > a* > (a* > e+ e-) (g* > g g): the root decays into two
    composites."""
    return ms.Topology(
        ms.Diagram(
            [0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [ms.Propagator(0.0, 0.0), ms.Propagator(0.0, 0.0), ms.Propagator(0.0, 0.0)],
            [
                ["i0", "i1", "p0"],
                ["p0", "p1", "p2"],
                ["p1", "o0", "o1"],
                ["p2", "o2", "o3"],
            ],
        )
    )


DY_PIDS = [2, -2, -11, 11]
ZG_PIDS = [2, -2, 23, 21]
WG_PIDS = [2, -1, -11, 12, 21]
EEGG_PIDS = [2, -2, -11, 11, 21, 21]


def lepton_cuts(pids, pt=PT_LEPTON, eta=ETA_LEPTON):
    items = [ms.CutItem(O(pids, O.obs_pt, [O.lepton_pids]), min=pt)]
    if eta is not None:
        items.append(ms.CutItem(O(pids, O.obs_eta_abs, [O.lepton_pids]), max=eta))
    return items


def jet_cuts(pids, pt=PT_JET, eta=ETA_JET):
    items = [ms.CutItem(O(pids, O.obs_pt, [O.jet_pids]), min=pt)]
    if eta is not None:
        items.append(ms.CutItem(O(pids, O.obs_eta_abs, [O.jet_pids]), max=eta))
    return items


def sample(mapping, seed=SEED, n=BATCH_SIZE):
    rng = np.random.default_rng(seed)
    r = rng.random((n, mapping.random_dim()))
    p_ext, x1, x2, det = mapping.map_forward([r])
    p_ext, x1, x2, det = (np.asarray(a) for a in (p_ext, x1, x2, det))
    ok = np.isfinite(det) & np.all(np.isfinite(p_ext), axis=(1, 2))
    return r, p_ext, x1, x2, np.where(ok, det, 0.0)


def passes(cuts, p_ext):
    return np.asarray(cuts(p_ext)).reshape(-1) > 0.5


def pt(p):
    return np.hypot(p[..., 1], p[..., 2])


def rapidity(p):
    return 0.5 * np.log((p[..., 0] + p[..., 3]) / (p[..., 0] - p[..., 3]))


def eta(p):
    p_mag = np.sqrt(np.sum(p[..., 1:] ** 2, axis=-1))
    return 0.5 * np.log((p_mag + p[..., 3]) / (p_mag - p[..., 3]))


def beam_rapidity(x1, x2):
    return 0.5 * np.log(x1 / x2)


def sqrt_s_hat(x1, x2):
    return CM_ENERGY * np.sqrt(x1 * x2)


def weighted_cut_integral(mapping, cuts, seed, n=400_000):
    """det / s_hat over the cut region: a propagator-shaped test integrand,
    so the low-mass region carries weight."""
    _, p_ext, x1, x2, det = sample(mapping, seed=seed, n=n)
    weight = np.where(passes(cuts, p_ext), det / (CM_ENERGY**2 * x1 * x2), 0.0)
    return weight.mean(), weight.std() / math.sqrt(n)


def assert_same_integral(topology, cuts, n=400_000):
    free = ms.PhaseSpaceMapping(topology, CM_ENERGY)
    piped = ms.PhaseSpaceMapping(topology, CM_ENERGY, cuts=cuts)
    mean_free, err_free = weighted_cut_integral(free, cuts, SEED, n)
    mean_piped, err_piped = weighted_cut_integral(piped, cuts, SEED + 1, n)
    error = math.hypot(err_free, err_piped)
    assert 0.0 < error < 0.02 * mean_free
    assert abs(mean_free - mean_piped) < 5.0 * error
    return mean_free, err_free, mean_piped, err_piped


# --------------------------------------------------------------------------
# Drell-Yan: the two windows are the whole cut
# --------------------------------------------------------------------------


def dy_cuts():
    return ms.Cuts(lepton_cuts(DY_PIDS))


def test_dy_every_generated_point_passes():
    """For u u~ > e+ e- the pt and eta cuts are exactly the cos(theta) and Y
    windows, so with the cuts handed to the mapping every point it generates
    with a weight passes them, while without them most fail."""
    c = dy_cuts()
    _, p_free, _, _, det_free = sample(ms.PhaseSpaceMapping(s_channel_two_body(), CM_ENERGY))
    assert np.mean(passes(c, p_free)[det_free > 0]) < 0.9
    _, p_ext, _, _, det = sample(
        ms.PhaseSpaceMapping(s_channel_two_body(), CM_ENERGY, cuts=c)
    )
    assert np.mean(det > 0) > 0.999
    assert np.mean(passes(c, p_ext)[det > 0]) > 0.999


def test_dy_windows_are_reached():
    """The bounds sampled are the cuts themselves, not something stricter."""
    # every sampled point, not only those with a weight: the weight already
    # contains the cut mask
    _, p_ext, x1, x2, _ = sample(
        ms.PhaseSpaceMapping(s_channel_two_body(), CM_ENERGY, cuts=dy_cuts())
    )
    leptons = p_ext[:, 2:4]
    y = beam_rapidity(x1, x2)
    assert np.abs(y).max() <= ETA_LEPTON + 1e-9
    assert np.abs(y).max() > 0.99 * ETA_LEPTON
    assert np.abs(eta(leptons)).max() <= ETA_LEPTON + 1e-6
    assert np.abs(eta(leptons)).max() > 0.999 * ETA_LEPTON
    assert pt(leptons).min() >= PT_LEPTON * (1 - 1e-9)
    assert pt(leptons).min() < 1.01 * PT_LEPTON


def test_dy_cut_integral_unchanged():
    assert_same_integral(s_channel_two_body(), dy_cuts())


def test_dy_round_trip():
    mapping = ms.PhaseSpaceMapping(s_channel_two_body(), CM_ENERGY, cuts=dy_cuts())
    r, p_ext, x1, x2, det = sample(mapping, n=10_000)
    keep = det > 0
    r_back, det_back = mapping.map_inverse([p_ext, x1, x2], [])
    r_back, det_back = np.asarray(r_back), np.asarray(det_back)
    # the cut mask is part of det but not of det_back
    assert r_back[keep] == approx(r[keep], abs=1e-7)
    assert (det * det_back)[keep] == approx(1.0, rel=1e-7)


# --------------------------------------------------------------------------
# a massive product with no cut: only the decay window applies
# --------------------------------------------------------------------------


@pytest.mark.parametrize("pt_jet", [PT_JET, 200.0])
def test_massive_partner_without_cuts(pt_jet):
    """u u~ > Z g: the Z has no eta cut, so Y is unbounded, but the gluon's pt
    and eta cuts (with the Z's beta != 1 on the other side) still fix the
    cos(theta) range. Nearly every sampled point passes; the few that do not
    are those where |Y| is so large that no angle is left (weight zero)."""
    c = ms.Cuts(jet_cuts(ZG_PIDS, pt=pt_jet))
    topology = s_channel_two_body((M_Z, 0.0))
    _, p_free, _, _, _ = sample(ms.PhaseSpaceMapping(topology, CM_ENERGY))
    assert np.mean(passes(c, p_free)) < 0.9
    _, p_ext, x1, x2, det = sample(ms.PhaseSpaceMapping(topology, CM_ENERGY, cuts=c))
    assert np.mean(passes(c, p_ext)) > 0.99
    assert np.mean(passes(c, p_ext)[det > 0]) > 0.999
    # Y is not restricted
    assert np.abs(beam_rapidity(x1, x2)[det > 0]).max() > ETA_JET + 0.5
    assert_same_integral(topology, c)


def test_balanced_transverse_mass_floor():
    """With pt_g >= 200 GeV the Z must carry 200 GeV of pt too, so
    sqrt(s_hat) >= sqrt(m_Z^2 + 200^2) + 200, above the m_Z + 200 the plain
    sum of transverse masses gives."""
    pt_jet = 200.0
    c = ms.Cuts(jet_cuts(ZG_PIDS, pt=pt_jet))
    topology = s_channel_two_body((M_Z, 0.0))
    _, _, x1, x2, _ = sample(ms.PhaseSpaceMapping(topology, CM_ENERGY, cuts=c))
    floor = math.hypot(M_Z, pt_jet) + pt_jet
    assert sqrt_s_hat(x1, x2).min() >= floor * (1 - 1e-9)
    assert sqrt_s_hat(x1, x2).min() < 1.01 * floor


def test_balanced_floor_with_massless_partners():
    """u d~ > e+ ve g with pt_g >= 200 and only the lepton cut otherwise: the
    neutrino (massless, no cut) and the lepton must make up the gluon's pt,
    at full cost, so sqrt(s_hat) >= 2 * 200 + 0."""
    topology = ms.Topology(
        ms.Diagram(
            [0.0, 0.0],
            [0.0, 0.0, 0.0],
            [ms.Propagator(0.0, 0.0), ms.Propagator(0.0, 0.0)],
            [["i0", "i1", "p0"], ["p0", "p1", "o2"], ["p1", "o0", "o1"]],
        )
    )
    c = ms.Cuts(lepton_cuts(WG_PIDS) + jet_cuts(WG_PIDS, pt=200.0, eta=None))
    _, _, x1, x2, _ = sample(ms.PhaseSpaceMapping(topology, CM_ENERGY, cuts=c))
    assert sqrt_s_hat(x1, x2).min() >= 400.0 * (1 - 1e-9)
    assert sqrt_s_hat(x1, x2).min() < 1.01 * 400.0


# --------------------------------------------------------------------------
# composite products: rapidity bounds without pt bounds
# --------------------------------------------------------------------------


def eegg_cuts(eta_jet=ETA_JET):
    return ms.Cuts(lepton_cuts(EEGG_PIDS) + jet_cuts(EEGG_PIDS, eta=eta_jet))


def test_composite_rapidity_windows():
    """The root of u u~ > (e+ e-) (g g) decays into two composites. With every
    particle eta-cut, |Y| and the rapidities of both composites stay within
    the largest eta cut, and the cut integral does not move."""
    c = eegg_cuts()
    topology = s_channel_four_body()
    _, p_ext, x1, x2, det = sample(ms.PhaseSpaceMapping(topology, CM_ENERGY, cuts=c))
    # every sampled point: the weight already contains the cut mask
    keep = np.all(np.isfinite(p_ext), axis=(1, 2))
    eta_max = max(ETA_LEPTON, ETA_JET)
    assert np.abs(beam_rapidity(x1, x2)[keep]).max() <= eta_max + 1e-9
    for pair in ((2, 3), (4, 5)):
        system = p_ext[keep, pair[0]] + p_ext[keep, pair[1]]
        assert np.abs(rapidity(system)).max() <= eta_max + 1e-6
    assert_same_integral(topology, c)


def test_composite_round_trip():
    mapping = ms.PhaseSpaceMapping(s_channel_four_body(), CM_ENERGY, cuts=eegg_cuts())
    r, p_ext, x1, x2, det = sample(mapping, n=10_000)
    keep = det > 0
    r_back, det_back = mapping.map_inverse([p_ext, x1, x2], [])
    r_back, det_back = np.asarray(r_back), np.asarray(det_back)
    # a handful of points lose ~1e-5 in the inner decays with or without the
    # windows (momentum reconstruction), hence quantiles
    r_diff = np.abs(r_back[keep] - r[keep])
    det_diff = np.abs((det * det_back)[keep] - 1.0)
    assert np.quantile(r_diff, 0.999) < 1e-7 and r_diff.max() < 1e-4
    assert np.quantile(det_diff, 0.99) < 1e-7 and np.quantile(det_diff, 0.999) < 1e-5


# --------------------------------------------------------------------------
# where nothing follows
# --------------------------------------------------------------------------


def test_no_rapidity_window_with_an_uncut_particle():
    """Without an eta cut on the gluons, neither Y nor the gluon system's
    rapidity is bounded, and both go beyond the lepton cut."""
    c = ms.Cuts(lepton_cuts(EEGG_PIDS) + jet_cuts(EEGG_PIDS, eta=None))
    _, p_ext, x1, x2, det = sample(
        ms.PhaseSpaceMapping(s_channel_four_body(), CM_ENERGY, cuts=c)
    )
    keep = det > 0
    assert np.abs(beam_rapidity(x1, x2)[keep]).max() > ETA_LEPTON + 1.0
    gluons = p_ext[keep, 4] + p_ext[keep, 5]
    assert np.abs(rapidity(gluons)).max() > ETA_LEPTON + 1.0


def test_no_cuts_no_change():
    """Without cuts the mapping is the one it always was."""
    topology = s_channel_two_body()
    a = sample(ms.PhaseSpaceMapping(topology, CM_ENERGY))
    b = sample(ms.PhaseSpaceMapping(topology, CM_ENERGY, cuts=ms.Cuts(4)))
    for u, v in zip(a, b):
        assert np.array_equal(u, v)


def test_two_body_decay_window_alone():
    """TwoBodyDecay(com=True, pt_min=...) on its own: cos(theta) stays in
    |cos| <= sqrt(1 - (pt_min / p)^2), the weight is the kept fraction, and it
    inverts."""
    pt_min = 30.0
    decay = ms.TwoBodyDecay(True, pt_min=pt_min)
    n = 10_000
    rng = np.random.default_rng(SEED)
    r_phi, r_cos = rng.random(n), rng.random(n)
    m0 = np.full(n, 100.0)
    zeros = np.zeros(n)
    ones = np.ones(n)
    p1, p2, det = decay.map_forward([r_phi, r_cos, m0, zeros, zeros], [ones, ones])
    p1, det = np.asarray(p1), np.asarray(det)
    plain = ms.TwoBodyDecay(True)
    _, _, det_plain = plain.map_forward([r_phi, r_cos, m0, zeros, zeros], [])
    kept = math.sqrt(1 - (pt_min / 50.0) ** 2)
    assert det == approx(np.asarray(det_plain) * kept, rel=1e-12)
    assert pt(p1).min() >= pt_min * (1 - 1e-9)
    *inputs, det_back = decay.map_inverse([p1, p2], [ones, ones])
    assert np.asarray(inputs[1]) == approx(r_cos, abs=1e-8)
    assert det * np.asarray(det_back) == approx(1.0, rel=1e-8)


# --------------------------------------------------------------------------
# coverage: no event passing the cuts lies outside the windows
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "topology,cuts",
    [
        (s_channel_two_body(), dy_cuts()),
        (s_channel_two_body((M_Z, 0.0)), ms.Cuts(jet_cuts(ZG_PIDS, pt=200.0))),
        (s_channel_four_body(), eegg_cuts()),
    ],
    ids=["dy", "zg", "eegg"],
)
def test_every_passing_event_is_inside_the_windows(topology, cuts):
    """Exactness, pointwise: events generated without the windows and passing
    the cuts all map back, through the windowed mapping's inverse, to random
    numbers inside the unit cube. With the round trip above this means the
    windowed mapping covers the whole cut region with the right density."""
    free = ms.PhaseSpaceMapping(topology, CM_ENERGY)
    windowed = ms.PhaseSpaceMapping(topology, CM_ENERGY, cuts=cuts)
    _, p_ext, x1, x2, det = sample(free, n=400_000)
    keep = passes(cuts, p_ext) & (det > 0)
    assert keep.sum() > 1000
    r_back, _ = windowed.map_inverse([p_ext[keep], x1[keep], x2[keep]], [])
    r_back = np.asarray(r_back)
    assert np.all(np.isfinite(r_back))
    assert r_back.min() > -1e-6 and r_back.max() < 1 + 1e-6

