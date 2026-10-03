"""A pair mass cut bounds every propagator whose decay contains the pair.

test_pair_mass_cut.py covers a propagator that decays into exactly the cut
pair. This file covers the general case: a propagator whose decay holds the
pair together with other particles. For future-pointing momenta the invariant
mass of a sum is at least the sum of the invariant masses, so such a node obeys

    m(node) >= m_cut + sum of the masses of its other final-state particles,

and handing that floor to the sampler is the difference between generating the
region the cut allows and generating almost nothing of it.

The channels come from p p > w+ w+ j j QCD=0 (subprocess u d~ > w+ w+ u d~ in
the output's own particle order: W+, W+, jet, jet), with a 200 GeV cut on the
jet pair:

  * W_CHANNEL: two t-channel quark lines meet in an s-channel W- that decays to
    the jet pair, so the cut is exactly a floor on the W- (100% efficiency).
  * Z_CHANNEL: a Z decays to a jet and an off-shell quark that decays to the
    other jet and a W+. The jet pair is two of the Z's three leaves. Before the
    floor reached such nodes only 0.04% of this channel passed the cut and the
    madspace survey gave up on it ("not enough points passing cuts").

The cut-region volume, integrated with each channel, must match the flat
RAMBO mapping that knows nothing about the cut: the floor follows from the
cut and must not cut anything of its own.

The last section pins down which cuts Cuts reports as floors at all.
"""

import math

import numpy as np
import pytest

import madspace as ms

O = ms.Observable
PSM = ms.PhaseSpaceMapping

CM_ENERGY = 13000.0
M_W, W_W = 80.419, 2.0476
M_Z, W_Z = 91.188, 2.441
BATCH_SIZE = 200_000
# the volume comparisons need the precision; 2M points still take under a second
VOLUME_BATCH_SIZE = 2_000_000
SEED = 20261003

PIDS = [2, -1, 24, 24, 2, -1]
JET_PIDS = [2, -1]
JETS = (4, 5)
W_PLUS = (2, 3)
OUTGOING_MASSES = [M_W, M_W, 0.0, 0.0]
PAIR_MASS_CUT = 200.0

W_CHANNEL = (
    [(0.0, 0.0, -1), (0.0, 0.0, 1), (M_W, W_W, -24)],
    [["i0", "o0", "p0"], ["i1", "o1", "p1"], ["o3", "o2", "p2"], ["p0", "p1", "p2"]],
)
Z_CHANNEL = (
    [(0.0, 0.0, -1), (0.0, 0.0, -1), (M_Z, W_Z, 23)],
    [["o3", "o1", "p0"], ["i0", "o0", "p1"], ["p1", "i1", "p2"], ["p0", "o2", "p2"]],
)


def topology(channel):
    propagators, vertices = channel
    return ms.Topology(
        ms.Diagram(
            [0.0, 0.0],
            OUTGOING_MASSES,
            [
                ms.Propagator(
                    mass=mass, width=width, integration_order=0, pdg_id=pdg_id
                )
                for mass, width, pdg_id in propagators
            ],
            vertices,
        )
    )


def pair_mass_cuts(minimum=PAIR_MASS_CUT):
    return ms.Cuts([ms.CutItem(O(PIDS, O.obs_pair_mass, [JET_PIDS]), min=minimum)])


def sample(mapping, seed, n=BATCH_SIZE):
    rng = np.random.default_rng(seed)
    p_ext, _x1, _x2, det = mapping.map_forward([rng.random((n, mapping.random_dim()))])
    p_ext, det = np.asarray(p_ext), np.asarray(det)
    finite = np.isfinite(det) & np.all(np.isfinite(p_ext), axis=(1, 2))
    return p_ext, np.where(finite, det, 0.0)


def invariant_mass(p, *indices):
    total = sum(p[:, i, :] for i in indices)
    m2 = total[:, 0] ** 2 - np.sum(total[:, 1:] ** 2, axis=1)
    return np.sqrt(np.maximum(m2, 0.0))


def cut_volume(p, det):
    """Mean and standard error of the cut-region volume from one batch."""
    weights = np.where(invariant_mass(p, *JETS) >= PAIR_MASS_CUT, det, 0.0)
    return weights.mean(), weights.std() / math.sqrt(len(weights))


@pytest.fixture(scope="module")
def rambo_volume():
    """The cut-region volume from a mapping with no propagator structure."""
    flat = PSM([0.0, 0.0] + OUTGOING_MASSES, CM_ENERGY, mode=PSM.rambo)
    return cut_volume(*sample(flat, SEED, n=VOLUME_BATCH_SIZE))


# --------------------------------------------------------------------------
# the propagator that decays into exactly the pair
# --------------------------------------------------------------------------


def test_w_channel_samples_only_the_allowed_region():
    p, det = sample(PSM(topology(W_CHANNEL), CM_ENERGY, cuts=pair_mass_cuts()), SEED)
    mjj = invariant_mass(p, *JETS)[det > 0]
    assert mjj.min() >= PAIR_MASS_CUT * (1 - 1e-9)


def test_w_channel_volume_matches_rambo(rambo_volume):
    mapping = PSM(topology(W_CHANNEL), CM_ENERGY, cuts=pair_mass_cuts())
    mean, error = cut_volume(*sample(mapping, SEED + 1, n=VOLUME_BATCH_SIZE))
    ref, ref_error = rambo_volume
    assert abs(mean - ref) < 5.0 * math.hypot(error, ref_error)


# --------------------------------------------------------------------------
# a propagator whose decay holds the pair and more
# --------------------------------------------------------------------------


def test_z_channel_propagator_gets_the_floor():
    """The Z's leaves are the jet pair and a W+, so m(Z) >= cut + m_W."""
    p, det = sample(PSM(topology(Z_CHANNEL), CM_ENERGY, cuts=pair_mass_cuts()), SEED)
    m_z = invariant_mass(p, JETS[0], JETS[1], W_PLUS[1])[det > 0]
    assert m_z.min() >= (PAIR_MASS_CUT + M_W) * (1 - 1e-9)


def test_z_channel_efficiency():
    """The jet pair is not a propagator of its own here, so the floor on the Z
    cannot make every point pass, but it has to lift the channel out of the
    0.04% it had when only exact-pair propagators were bounded."""
    p_free, _ = sample(PSM(topology(Z_CHANNEL), CM_ENERGY), SEED)
    p_cut, det = sample(PSM(topology(Z_CHANNEL), CM_ENERGY, cuts=pair_mass_cuts()), SEED)
    free = np.mean(invariant_mass(p_free, *JETS) >= PAIR_MASS_CUT)
    bounded = np.mean(invariant_mass(p_cut, *JETS)[det > 0] >= PAIR_MASS_CUT)
    assert free < 0.01
    assert bounded > 0.2


def test_z_channel_volume_matches_rambo(rambo_volume):
    mapping = PSM(topology(Z_CHANNEL), CM_ENERGY, cuts=pair_mass_cuts())
    mean, error = cut_volume(*sample(mapping, SEED + 2, n=VOLUME_BATCH_SIZE))
    ref, ref_error = rambo_volume
    assert abs(mean - ref) < 5.0 * math.hypot(error, ref_error)


@pytest.mark.parametrize("mode", [PSM.propagator, PSM.rambo, PSM.chili])
def test_floor_reaches_every_t_channel_mode(mode):
    mapping = PSM(
        topology(Z_CHANNEL), CM_ENERGY, cuts=pair_mass_cuts(), t_channel_mode=mode
    )
    p, det = sample(mapping, SEED, n=20_000)
    m_z = invariant_mass(p, JETS[0], JETS[1], W_PLUS[1])[det > 0]
    assert len(m_z) > 0
    assert m_z.min() >= (PAIR_MASS_CUT + M_W) * (1 - 1e-9)


# --------------------------------------------------------------------------
# which cuts count as floors
# --------------------------------------------------------------------------

JETS4 = [21, 21, 1, 2, 3, 4]
JET_GROUP = [1, 2, 3, 4]


def test_two_group_summed_mass_is_a_pair_floor():
    """"jet-jet-sum-mass" is the same cut as "jet-pair_mass"."""
    summed = O(PIDS, O.obs_mass, [JET_PIDS, JET_PIDS], sum_momenta=True)
    pairwise = O(PIDS, O.obs_pair_mass, [JET_PIDS])
    floors = [
        ms.Cuts([ms.CutItem(obs, min=PAIR_MASS_CUT)]).m_inv_min()
        for obs in (summed, pairwise)
    ]
    assert floors[0] == floors[1]
    assert floors[0][2][3] == PAIR_MASS_CUT


def test_ordered_selection_is_not_a_floor():
    """"jet_1-pt" bounds the leading jet, whichever particle that is, so it
    bounds no particle in particular and must stay a filter."""
    leading_pt = O(
        JETS4, O.obs_pt, [JET_GROUP], order_observable=O.obs_pt, order_indices=[1]
    )
    leading_pair = O(
        JETS4,
        O.obs_pair_mass,
        [JET_GROUP, JET_GROUP],
        order_observable=O.obs_pt,
        order_indices=[1, 2],
    )
    cuts = ms.Cuts(
        [ms.CutItem(leading_pt, min=300.0), ms.CutItem(leading_pair, min=500.0)]
    )
    assert cuts.pt_min() == [0.0] * 4
    assert np.all(np.asarray(cuts.m_inv_min()) == 0.0)


@pytest.mark.parametrize("mode", [ms.Cuts.CutMode.all, ms.Cuts.CutMode.any])
def test_any_mode_over_several_objects_is_not_a_floor(mode):
    """With mode "any" one passing jet (pair) is enough, so no individual jet
    (pair) is bounded."""
    cuts = ms.Cuts([
        ms.CutItem(O(JETS4, O.obs_pt, [JET_GROUP]), min=50.0, mode=mode),
        ms.CutItem(O(JETS4, O.obs_pair_mass, [JET_GROUP]), min=500.0, mode=mode),
    ])
    bounded = mode == ms.Cuts.CutMode.all
    assert cuts.pt_min() == [50.0 if bounded else 0.0] * 4
    m_inv = np.asarray(cuts.m_inv_min())
    off_diagonal = m_inv[~np.eye(4, dtype=bool)]
    assert np.all(off_diagonal == (500.0 if bounded else 0.0))


def test_any_mode_over_one_object_is_a_floor():
    cuts = ms.Cuts([
        ms.CutItem(O(PIDS, O.obs_pair_mass, [JET_PIDS]), min=200.0, mode=ms.Cuts.CutMode.any)
    ])
    assert cuts.m_inv_min()[2][3] == 200.0
