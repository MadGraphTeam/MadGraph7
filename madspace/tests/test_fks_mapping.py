"""FKSMapping: a Born PhaseSpaceMapping followed by the FKS emission, in the lab
frame.

The emission itself is tested in test_fks_radiation.py, against genps_fks.f
and to the conditioning of its variables. These tests cover the composition:
the boosts between the lab frame and the centre-of-mass frame of the Born
(also with a beam rapidity), the incoming momenta of the real emission, the
round trip through both mappings, and that every FKS configuration covers the
phase space of the real process once, momentum fractions included.
"""

import math

import numpy as np
import pytest
import torch
from madspace_testing import generator

import madspace as ms

SQRT_S = 1000.0


def to_torch(x):
    return torch.as_tensor(np.asarray(x), dtype=torch.float64).to(
        torch.get_default_device()
    )


def to_numpy(x):
    return torch.as_tensor(x).cpu().numpy()


def born_mapping(masses, leptonic, beam_rapidity=0.0):
    """RAMBO mapping of a 2 -> n Born with massless incoming partons."""
    return ms.PhaseSpaceMapping(
        [0.0, 0.0] + list(masses),
        SQRT_S,
        leptonic=leptonic,
        mode=ms.PhaseSpaceMapping.TChannelMode.rambo,
        beam_rapidity=beam_rapidity,
    )


def forward(mapping, r):
    """Momenta, momentum fractions (broadcast to the batch, they are constants
    without PDFs) and weight."""
    momenta, x1, x2, det = mapping.map_forward([to_torch(r)])
    batch = len(r)
    x1 = np.broadcast_to(to_numpy(x1), (batch,)).copy()
    x2 = np.broadcast_to(to_numpy(x2), (batch,)).copy()
    return to_numpy(momenta), x1, x2, to_numpy(det)


def inverse(mapping, momenta, x1, x2):
    r, det = mapping.map_inverse([to_torch(momenta), to_torch(x1), to_torch(x2)])
    return to_numpy(r), to_numpy(det)


def random_numbers(mapping, batch, seed, low=0.0, high=1.0):
    r = torch.rand((batch, mapping.random_dim()), generator=generator(seed))
    return low + (high - low) * r.numpy().astype(np.float64)


# (final-state masses of the Born, leptonic, beam rapidity, i_fks, j_fks)
LAYOUTS = [
    ([0.0, 0.0], True, 0.0, 4, 2),
    ([173.0, 173.0], True, 0.0, 4, 3),
    ([173.0, 173.0], True, 0.2, 2, 4),
    ([0.0, 0.0, 0.0], False, 0.0, 5, 3),
    ([173.0, 173.0], False, -0.4, 4, 2),
    ([0.0, 0.0], False, 0.0, 4, 0),
    ([0.0, 0.0, 0.0], False, 0.3, 2, 1),
    ([173.0, 173.0], False, 0.0, 4, 1),
]
IDS = [
    "ee-final",
    "ee-massive",
    "ee-massive-boosted",
    "pp-final",
    "pp-massive-boosted",
    "pp-initial",
    "pp-initial-boosted",
    "pp-ttx-initial",
]


def real_masses(masses, i_fks):
    return np.insert(np.array([0.0, 0.0] + list(masses)), i_fks, 0.0)


@pytest.mark.parametrize("masses,leptonic,beam_rapidity,i_fks,j_fks", LAYOUTS, ids=IDS)
def test_round_trip(masses, leptonic, beam_rapidity, i_fks, j_fks):
    batch = 2000
    born = born_mapping(masses, leptonic, beam_rapidity)
    mapping = ms.FKSMapping(born, i_fks, j_fks)
    assert mapping.random_dim() == born.random_dim() + 3
    assert mapping.particle_count() == len(masses) + 3
    assert mapping.radiation().emitter_mass() == (
        real_masses(masses, i_fks)[j_fks] if j_fks >= 2 else 0.0
    )
    # away from the edges of the unit cube, where the conditioning of the
    # variables (see test_fks_radiation.py) takes over
    r = random_numbers(mapping, batch, seed=1 + i_fks + j_fks, low=0.02, high=0.98)
    momenta, x1, x2, det = forward(mapping, r)
    assert momenta.shape == (batch, len(masses) + 3, 4)
    assert (det > 0).all()

    # the incoming momenta are the beams times the momentum fractions
    if leptonic:
        np.testing.assert_array_equal(x1, 1.0)
        np.testing.assert_array_equal(x2, 1.0)
    assert ((x1 <= 1) & (x2 <= 1)).all()
    e1 = 0.5 * SQRT_S * math.exp(beam_rapidity) * x1
    e2 = 0.5 * SQRT_S * math.exp(-beam_rapidity) * x2
    zero = np.zeros(batch)
    np.testing.assert_allclose(
        momenta[:, 0] / SQRT_S, np.stack([e1, zero, zero, e1], 1) / SQRT_S, atol=1e-13
    )
    np.testing.assert_allclose(
        momenta[:, 1] / SQRT_S, np.stack([e2, zero, zero, -e2], 1) / SQRT_S, atol=1e-13
    )

    # momentum conservation and mass shells in the lab frame, to rounding for
    # any rapidity of the Born
    scale = momenta[..., 0].max(1)
    violation = np.abs(momenta[:, 2:].sum(1) - momenta[:, :2].sum(1)).max(1)
    assert (violation <= 1e-13 * scale).all()
    m2 = momenta[..., 0] ** 2 - (momenta[..., 1:] ** 2).sum(-1)
    np.testing.assert_allclose(
        (m2 - real_masses(masses, i_fks) ** 2) / scale[:, None] ** 2, 0.0, atol=1e-12
    )

    # The random numbers of the Born are resolved to the conditioning of the Born
    # mapping, which for the massless channels here, with momentum fractions
    # down to 1e-7, is 1e-10 (as its own round trip). The emission is resolved
    # as in test_fks_radiation.py.
    r_inv, det_inv = inverse(mapping, momenta, x1, x2)
    np.testing.assert_allclose(r_inv[:, :-3], r[:, :-3], rtol=0, atol=1e-8)
    np.testing.assert_allclose(r_inv[:, -3:], r[:, -3:], rtol=0, atol=1e-11)
    np.testing.assert_allclose(det * det_inv, 1.0, rtol=1e-8)


@pytest.mark.parametrize("masses,leptonic,beam_rapidity,i_fks,j_fks", LAYOUTS, ids=IDS)
def test_frame_of_the_born(masses, leptonic, beam_rapidity, i_fks, j_fks):
    """Without the emission (xi -> 0) the real emission is the Born of the Born
    mapping with a soft i, in the lab frame."""
    batch = 500
    born = born_mapping(masses, leptonic, beam_rapidity)
    mapping = ms.FKSMapping(born, i_fks, j_fks)
    r = random_numbers(mapping, batch, seed=2)
    r[:, -3] = 0.0
    momenta, x1, x2, _ = forward(mapping, r)
    born_momenta, x1_born, x2_born, _ = forward(born, r[:, :-3])
    # xi = stiny xi_max, so i and the recoil differ from the Born at O(1e-6)
    np.testing.assert_allclose(momenta[:, i_fks, 0] / SQRT_S, 0.0, atol=1e-5)
    np.testing.assert_allclose(
        np.delete(momenta, i_fks, axis=1) / SQRT_S, born_momenta / SQRT_S, atol=1e-5
    )
    np.testing.assert_allclose(x1, x1_born, rtol=1e-5)
    np.testing.assert_allclose(x2, x2_born, rtol=1e-5)


@pytest.mark.parametrize("index", [0, 1])
def test_born_channel_and_permutation(index):
    """A Born mapping that follows a diagram (s-channel Z, t-channel
    propagators) with two permutations: the condition passes through."""
    topology = ms.Topology(
        ms.Diagram(
            [0.0, 0.0],
            [173.0, 173.0],
            [ms.Propagator(91.0, 2.5)],
            [["i0", "i1", "p0"], ["p0", "o0", "o1"]],
        )
    )
    born = ms.PhaseSpaceMapping(
        topology, SQRT_S, permutations=[[0, 1, 2, 3], [0, 1, 3, 2]]
    )
    mapping = ms.FKSMapping(born, 4, 3)
    batch = 2000
    r = random_numbers(mapping, batch, seed=50, low=0.02, high=0.98)
    perm = torch.full((batch,), index, dtype=torch.int32).to(torch.get_default_device())
    momenta, x1, x2, det = mapping.map_forward([to_torch(r)], [perm])
    r_inv, det_inv = mapping.map_inverse([momenta, x1, x2], [perm])
    momenta, det = to_numpy(momenta), to_numpy(det)
    assert (det > 0).all()
    violation = np.abs(momenta[:, 2:].sum(1) - momenta[:, :2].sum(1)).max(1)
    assert (violation <= 1e-13 * momenta[..., 0].max(1)).all()
    np.testing.assert_allclose(to_numpy(r_inv), r, rtol=0, atol=1e-8)
    np.testing.assert_allclose(det * to_numpy(det_inv), 1.0, rtol=1e-8)


def assert_same_mean(weight, weight_ref):
    mean, err = weight.mean(), weight.std() / math.sqrt(len(weight))
    mean_ref, err_ref = weight_ref.mean(), weight_ref.std() / math.sqrt(len(weight_ref))
    assert abs(mean - mean_ref) < 5 * math.hypot(err, err_ref)


# (leptonic, j_fks): every emitter of t t~ g, incoming ones only with PDFs
VOLUME_CASES = [(True, 2), (True, 3), (False, 0), (False, 1), (False, 2), (False, 3)]


@pytest.mark.parametrize(
    "leptonic,j_fks",
    VOLUME_CASES,
    ids=[f"{'ee' if lep else 'pp'}-j{j}" for lep, j in VOLUME_CASES],
)
def test_volume(leptonic, j_fks):
    """Each FKS configuration of t t~ g covers its phase space once, as a
    PhaseSpaceMapping of the real process does, momentum fractions included."""
    batch = 400000
    born = born_mapping([173.0, 173.0], leptonic)
    mapping = ms.FKSMapping(born, 4, j_fks)
    *_, det = forward(mapping, random_numbers(mapping, batch, seed=10 + j_fks))
    real = born_mapping([173.0, 173.0, 0.0], leptonic)
    *_, det_real = forward(real, random_numbers(real, batch, seed=20 + j_fks))
    assert_same_mean(det, det_real)


@pytest.mark.parametrize("j_fks", [0, 1])
def test_volume_bounded(j_fks):
    """With a lower bound on the partonic energy of the real emission, an
    incoming emitter covers the part of the phase space above it."""
    batch = 400000
    s_hat_min = 600.0**2
    born = born_mapping([173.0, 173.0], False)
    mapping = ms.FKSMapping(born, 4, j_fks, s_hat_min=s_hat_min)
    _, x1, x2, det = forward(mapping, random_numbers(mapping, batch, seed=30 + j_fks))
    assert (x1 * x2 * SQRT_S**2 >= s_hat_min * (1 - 1e-12)).all()
    real = born_mapping([173.0, 173.0, 0.0], False)
    _, x1_real, x2_real, det_real = forward(
        real, random_numbers(real, batch, seed=40 + j_fks)
    )
    assert_same_mean(det, det_real * (x1_real * x2_real * SQRT_S**2 >= s_hat_min))


def test_invalid_born_mappings():
    with pytest.raises(ValueError):
        ms.FKSMapping(
            ms.PhaseSpaceMapping(
                [0.0, 0.0, 0.0, 0.0],
                SQRT_S,
                mode=ms.PhaseSpaceMapping.TChannelMode.rambo,
                mirror_beams=True,
            ),
            4,
            0,
        )
    cuts = ms.Cuts(
        [
            ms.CutItem(
                observable=ms.Observable([1, -1, 11, -11], "eta", [[11, -11]]), min=0.0
            )
        ]
    )
    with pytest.raises(ValueError):
        ms.FKSMapping(
            ms.PhaseSpaceMapping(
                [0.0, 0.0, 0.0, 0.0],
                SQRT_S,
                mode=ms.PhaseSpaceMapping.TChannelMode.rambo,
                cuts=cuts,
            ),
            4,
            2,
        )
    # an incoming emitter needs momentum fractions
    with pytest.raises(ValueError):
        ms.FKSMapping(born_mapping([0.0, 0.0], True), 4, 0)
