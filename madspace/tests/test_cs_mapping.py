"""CSMapping: a Born PhaseSpaceMapping followed by the emission of a
Catani-Seymour dipole, in the lab frame.

The real emission has to project back onto its Born with the momentum maps of
Catani and Seymour (hep-ph/9605323) and of Catani, Dittmaier, Seymour and
Trocsanyi (hep-ph/0201036), written out here with numpy, and with the decaying
particle of a decay as the spectator (arXiv:1507.04676), the final-final map
with the other final-state particles as the spectator. The tests cover the
five families of dipoles with massless and massive emitters and spectators:
the round trip through the mapping and its inverse, the incoming momenta of
the real emission, its soft limit, the mirror of the beams, that every
dipole covers the phase space of the real process once, momentum fractions
included, and the cutoff on the two variables of the dipole.
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


def born_mapping(masses, leptonic, beam_rapidity=0.0, mirror_beams=False):
    """RAMBO mapping of a 2 -> n Born with massless incoming partons."""
    return ms.PhaseSpaceMapping(
        [0.0, 0.0] + list(masses),
        SQRT_S,
        leptonic=leptonic,
        mode=ms.PhaseSpaceMapping.TChannelMode.rambo,
        beam_rapidity=beam_rapidity,
        mirror_beams=mirror_beams,
    )


def forward(mapping, r, conditions=()):
    """Momenta, momentum fractions (broadcast to the batch) and weight."""
    momenta, x1, x2, det = mapping.map_forward([to_torch(r)], list(conditions))
    batch = len(r)
    x1 = np.broadcast_to(to_numpy(x1), (batch,)).copy()
    x2 = np.broadcast_to(to_numpy(x2), (batch,)).copy()
    return to_numpy(momenta), x1, x2, to_numpy(det)


def inverse(mapping, momenta, x1, x2, conditions=()):
    r, det = mapping.map_inverse(
        [to_torch(momenta), to_torch(x1), to_torch(x2)], list(conditions)
    )
    return to_numpy(r), to_numpy(det)


def random_numbers(mapping, batch, seed, low=0.0, high=1.0):
    # on the CPU, the device of the generator, also when a GPU run makes the
    # GPU the default device
    r = torch.rand(
        (batch, mapping.random_dim()), generator=generator(seed), device="cpu"
    )
    return low + (high - low) * r.numpy().astype(np.float64)


def real_masses(masses, emitted):
    return np.insert(np.array([0.0, 0.0] + list(masses)), emitted, 0.0)


def dot(a, b):
    return a[..., 0] * b[..., 0] - (a[..., 1:] * b[..., 1:]).sum(-1)


def kallen(a, b, c):
    return a * a + b * b + c * c - 2 * (a * b + a * c + b * c)


def lorentz(k, big_k, big_kt):
    """Lambda(K, K~) k of CS eq. 5.143."""
    total = big_k + big_kt
    return (
        k
        - (2 * dot(k, total) / dot(total, total))[..., None] * total
        + (2 * dot(k, big_k) / dot(big_k, big_k))[..., None] * big_kt
    )


def lc_dot(p, m_p, q):
    """p.q for a massless q, |q| (E_p - n.p) with n the direction of q and
    E_p - n.p = (m_p^2 + |p x n|^2) / (E_p + n.p) from the mass shell of p,
    which keeps the digits of p.q for p close to the direction of q."""
    n = q[..., 1:] / np.linalg.norm(q[..., 1:], axis=-1)[..., None]
    pn = (p[..., 1:] * n).sum(-1)
    cross = np.cross(p[..., 1:], n)
    ahead = pn > 0
    stable = (m_p**2 + (cross**2).sum(-1)) / np.where(ahead, p[..., 0] + pn, 1.0)
    return np.linalg.norm(q[..., 1:], axis=-1) * np.where(ahead, stable, p[..., 0] - pn)


def project(momenta, x1, x2, masses, i, j, k):
    """The Born of the dipole V_ij,k of a real emission, with its momentum
    fractions, from the maps of CS and CDST. The products with the massless
    i and incoming partons are taken from the mass shells (lc_dot)."""
    p = momenta
    born = np.delete(p, i, axis=1)
    x = [x1.copy(), x2.copy()]
    bj, bk = j - (j > i), k - (k > i)
    if j >= 2 and k >= 2:
        # CDST eq. 5.8, m_ij = m_j
        q = p[:, i] + p[:, j] + p[:, k]
        q2 = dot(q, q)
        m_j, m_k = masses[j], masses[k]
        s_ij = m_j**2 + 2 * lc_dot(p[:, j], m_j, p[:, i])
        scale = np.sqrt(kallen(q2, m_j**2, m_k**2) / kallen(q2, s_ij, m_k**2))
        p_k = (
            scale[:, None] * (p[:, k] - (dot(q, p[:, k]) / q2)[:, None] * q)
            + ((q2 + m_k**2 - m_j**2) / (2 * q2))[:, None] * q
        )
        born[:, bk] = p_k
        born[:, bj] = q - p_k
    elif (j < 2) != (k < 2):
        # CS eqs. 5.42 and 5.62, CDST eqs. 5.42 and 5.77
        a, f = (j, k) if j < 2 else (k, j)
        bf = f - (f > i)
        m_f = masses[f]
        pi_a = lc_dot(p[:, i], 0.0, p[:, a])
        pf_a = lc_dot(p[:, f], m_f, p[:, a])
        x_cs = 1 - lc_dot(p[:, f], m_f, p[:, i]) / (pi_a + pf_a)
        born[:, a] = x_cs[:, None] * p[:, a]
        born[:, bf] = p[:, i] + p[:, f] - (1 - x_cs)[:, None] * p[:, a]
        x[a] = x[a] * x_cs
    else:
        # CS eqs. 5.137-5.143
        a, b = j, k
        pi_ab = lc_dot(p[:, i], 0.0, p[:, a]) + lc_dot(p[:, i], 0.0, p[:, b])
        x_cs = 1 - pi_ab / lc_dot(p[:, a], 0.0, p[:, b])
        born[:, a] = x_cs[:, None] * p[:, a]
        big_k = p[:, a] + p[:, b] - p[:, i]
        big_kt = born[:, a] + p[:, b]
        for n in range(2, born.shape[1]):
            born[:, n] = lorentz(born[:, n], big_k, big_kt)
        x[a] = x[a] * x_cs
    return born, x[0], x[1]


# (final-state masses of the Born, leptonic, beam rapidity, emitted, emitter,
# spectator)
LAYOUTS = [
    ([0.0, 0.0], True, 0.0, 4, 2, 3),
    ([173.0, 173.0], True, 0.0, 4, 2, 3),
    ([173.0, 0.0], True, 0.2, 4, 3, 2),
    ([0.0, 0.0, 0.0], False, 0.3, 3, 4, 2),
    ([173.0, 173.0], False, -0.4, 4, 2, 0),
    ([0.0, 0.0], False, 0.0, 4, 3, 1),
    ([173.0, 173.0], False, 0.0, 4, 0, 2),
    ([0.0, 0.0], False, 0.3, 4, 1, 3),
    ([173.0, 173.0], False, 0.0, 4, 0, 1),
    ([0.0, 0.0, 0.0], False, -0.3, 3, 1, 0),
]
IDS = [
    "ee-ff",
    "ee-ff-massive",
    "ee-ff-massive-spectator-boosted",
    "pp-ff-boosted",
    "pp-fi-massive-boosted",
    "pp-fi",
    "pp-if-massive",
    "pp-if-boosted",
    "pp-ii-ttx",
    "pp-ii-boosted",
]
FAMILIES = {
    (False, False): "final_final",
    (False, True): "final_initial",
    (True, False): "initial_final",
    (True, True): "initial_initial",
}


@pytest.mark.parametrize(
    "masses,leptonic,beam_rapidity,emitted,emitter,spectator", LAYOUTS, ids=IDS
)
def test_round_trip(masses, leptonic, beam_rapidity, emitted, emitter, spectator):
    batch = 2000
    born = born_mapping(masses, leptonic, beam_rapidity)
    mapping = ms.CSMapping(born, emitted, emitter, spectator)
    family = FAMILIES[emitter < 2, spectator < 2]
    assert mapping.family() == ms.CSMapping.Family(family)
    assert mapping.random_dim() == born.random_dim() + 3
    assert mapping.particle_count() == len(masses) + 3
    r = random_numbers(mapping, batch, seed=1 + emitted + emitter, low=0.02, high=0.98)
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

    # momentum conservation and mass shells in the lab frame
    scale = momenta[..., 0].max(1)
    violation = np.abs(momenta[:, 2:].sum(1) - momenta[:, :2].sum(1)).max(1)
    assert (violation <= 1e-13 * scale).all()
    m2 = momenta[..., 0] ** 2 - (momenta[..., 1:] ** 2).sum(-1)
    np.testing.assert_allclose(
        (m2 - real_masses(masses, emitted) ** 2) / scale[:, None] ** 2, 0.0, atol=1e-12
    )

    # The Born of the dipole is the Born of the Born mapping. Both go into the
    # center-of-mass frame of the real emission first, where the products of
    # the projection do not lose the digits of a large rapidity (the Borns of
    # RAMBO here reach |y| ~ 7, sqrt(s_hat) ~ 1 GeV with lab energies of 500).
    # A final-final dipole of two Born partons close in angle loses digits as
    # s_hat / Q^2, 1e-13 at sqrt(Q^2) ~ sqrt(s_hat) / 100 here.
    born_momenta, x1_born, x2_born, _ = forward(born, r[:, :-3])
    rapidity = beam_rapidity + 0.5 * np.log(x1 / x2)
    real_cm = boost_z(momenta, real_masses(masses, emitted), -rapidity[:, None])
    projected, x1_proj, x2_proj = project(
        real_cm, x1, x2, real_masses(masses, emitted), emitted, emitter, spectator
    )
    born_cm = boost_z(born_momenta, [0.0, 0.0] + list(masses), -rapidity[:, None])
    sqrt_s_hat = SQRT_S * np.sqrt(x1 * x2)[:, None, None]
    np.testing.assert_allclose(
        projected / sqrt_s_hat, born_cm / sqrt_s_hat, rtol=0, atol=1e-12
    )
    np.testing.assert_allclose(x1_proj, x1_born, rtol=1e-13)
    np.testing.assert_allclose(x2_proj, x2_born, rtol=1e-13)

    # The random numbers of the Born are resolved to the conditioning of the
    # Born mapping (6e-11 for the massless 2 -> 2 Borns, see
    # test_fks_mapping.py), those of the emission to 1e-13 (also for partons
    # close to the beam, which the initial-final maps write in components
    # along it).
    r_inv, det_inv = inverse(mapping, momenta, x1, x2)
    np.testing.assert_allclose(r_inv[:, :-3], r[:, :-3], rtol=0, atol=1e-9)
    np.testing.assert_allclose(r_inv[:, -3:], r[:, -3:], rtol=0, atol=1e-12)
    np.testing.assert_allclose(det * det_inv, 1.0, rtol=1e-9)


@pytest.mark.parametrize(
    "masses,leptonic,beam_rapidity,emitted,emitter,spectator", LAYOUTS, ids=IDS
)
def test_soft_limit(masses, leptonic, beam_rapidity, emitted, emitter, spectator):
    """At r = 0 for both variables i is soft and the real emission is the Born
    of the Born mapping, in the lab frame."""
    batch = 500
    born = born_mapping(masses, leptonic, beam_rapidity)
    mapping = ms.CSMapping(born, emitted, emitter, spectator)
    r = random_numbers(mapping, batch, seed=2)
    r[:, -3:-1] = 0.0
    momenta, x1, x2, det = forward(mapping, r)
    born_momenta, x1_born, x2_born, _ = forward(born, r[:, :-3])
    np.testing.assert_allclose(momenta[:, emitted] / SQRT_S, 0.0, atol=1e-14)
    np.testing.assert_allclose(
        np.delete(momenta, emitted, axis=1) / SQRT_S,
        born_momenta / SQRT_S,
        rtol=0,
        atol=1e-13,
    )
    np.testing.assert_allclose(x1, x1_born, rtol=1e-14)
    np.testing.assert_allclose(x2, x2_born, rtol=1e-14)
    np.testing.assert_array_equal(det, 0.0)


def dipole_variables(momenta, masses, i, j, k):
    """The two variables of the dipole V_ij,k at real emissions, y and z_i
    (final-final), 1 - x and z (one incoming), v and 1 - x - v
    (initial-initial), from the products of the momenta."""
    p = momenta
    if j >= 2 and k >= 2:
        pipj = lc_dot(p[:, j], masses[j], p[:, i])
        pipk = lc_dot(p[:, k], masses[k], p[:, i])
        pjpk = dot(p[:, j], p[:, k])
        return pipj / (pipj + pipk + pjpk), pipk / (pipk + pjpk)
    if (j < 2) != (k < 2):
        a, f = (j, k) if j < 2 else (k, j)
        pipa = lc_dot(p[:, i], 0.0, p[:, a])
        pfpa = lc_dot(p[:, f], masses[f], p[:, a])
        pipf = lc_dot(p[:, f], masses[f], p[:, i])
        return pipf / (pipa + pfpa), pipa / (pipa + pfpa)
    pipa = lc_dot(p[:, i], 0.0, p[:, j])
    pipb = lc_dot(p[:, i], 0.0, p[:, k])
    papb = lc_dot(p[:, j], 0.0, p[:, k])
    return pipa / papb, pipb / papb


CUTOFF = 0.02


@pytest.mark.parametrize(
    "masses,leptonic,beam_rapidity,emitted,emitter,spectator", LAYOUTS, ids=IDS
)
def test_cutoff(masses, leptonic, beam_rapidity, emitted, emitter, spectator):
    """With a cutoff both variables of the dipole stay above it, also at
    random numbers of zero, the map and its inverse agree, the inverse gives
    zero below the cutoff, and the dipole covers the phase space of the real
    process where both variables are above it, the region of the mapping
    without a cutoff there."""
    batch = 200000
    seed = 70 + emitted + 2 * emitter + 3 * spectator
    born = born_mapping(masses, leptonic, beam_rapidity)
    mapping = ms.CSMapping(born, emitted, emitter, spectator, cutoff=CUTOFF)
    plain = ms.CSMapping(born, emitted, emitter, spectator)
    assert mapping.cutoff() == CUTOFF and plain.cutoff() == 0.0
    m = real_masses(masses, emitted)
    # the floors and the round trip at the 2000 points of test_round_trip,
    # the random numbers of the emission to 1e-10 (1e-12 there) since
    # var - cutoff loses the digits of cutoff / (var - cutoff)
    zeros = 200
    r = random_numbers(mapping, 2000, seed=seed, low=0.02, high=0.98)
    r[:zeros, -3:-1] = 0.0
    momenta, x1, x2, det = forward(mapping, r)
    first, second = dipole_variables(momenta, m, emitted, emitter, spectator)
    np.testing.assert_array_equal(det[:zeros], 0.0)
    assert (np.minimum(first, second) >= CUTOFF * (1 - 1e-9)).all()
    # the round trip as in test_round_trip, away from r = 0 and from the
    # empty ranges of a Born fraction above 1 - cutoff (det = 0)
    trip = det > 0
    trip[:zeros] = False
    assert trip[zeros:].mean() > 0.9
    r_inv, det_inv = inverse(mapping, momenta[trip], x1[trip], x2[trip])
    np.testing.assert_allclose(r_inv[:, -3:], r[trip, -3:], rtol=0, atol=1e-10)
    np.testing.assert_allclose(det[trip] * det_inv, 1.0, rtol=1e-9)

    # the phase space of the mapping without the cutoff, at random numbers of
    # the emission over the full range
    r = random_numbers(mapping, batch, seed=seed + 1)
    *_, det = forward(mapping, r)
    r_plain = random_numbers(plain, batch, seed=seed + 2)
    momenta, x1, x2, det_plain = forward(plain, r_plain)
    first, second = dipole_variables(momenta, m, emitted, emitter, spectator)
    below = np.minimum(first, second) < CUTOFF
    assert 0 < below.mean() < 0.9
    _, det_inv = inverse(mapping, momenta[below], x1[below], x2[below])
    np.testing.assert_array_equal(det_inv, 0.0)
    assert_same_mean(det, np.where(below, 0.0, det_plain))


def assert_same_mean(weight, weight_ref):
    mean, err = weight.mean(), weight.std() / math.sqrt(len(weight))
    mean_ref, err_ref = weight_ref.mean(), weight_ref.std() / math.sqrt(len(weight_ref))
    assert abs(mean - mean_ref) < 5 * math.hypot(err, err_ref)


# (final-state masses of the Born, leptonic, emitted, emitter, spectator)
VOLUME_CASES = [
    ([0.0, 0.0], True, 4, 2, 3),
    ([173.0, 173.0], True, 4, 2, 3),
    ([173.0, 0.0], True, 4, 3, 2),
    ([173.0, 173.0], False, 4, 2, 0),
    ([173.0, 173.0], False, 4, 0, 2),
    ([173.0, 173.0], False, 4, 0, 1),
    ([0.0, 0.0], False, 4, 3, 1),
    ([0.0, 0.0], False, 4, 1, 3),
    ([0.0, 0.0, 0.0], False, 3, 1, 0),
]
VOLUME_IDS = [
    "ee-jjg-ff",
    "ee-ttg-ff",
    "ee-tjg-ff",
    "pp-ttg-fi",
    "pp-ttg-if",
    "pp-ttg-ii",
    "pp-jjg-fi",
    "pp-jjg-if",
    "pp-jjjg-ii",
]


@pytest.mark.parametrize(
    "masses,leptonic,emitted,emitter,spectator", VOLUME_CASES, ids=VOLUME_IDS
)
def test_volume(masses, leptonic, emitted, emitter, spectator):
    """Each dipole covers the phase space of the real process once, as a
    PhaseSpaceMapping of the real process does, momentum fractions
    included."""
    batch = 400000
    seed = 10 + emitted + 2 * emitter + 3 * spectator
    mapping = ms.CSMapping(born_mapping(masses, leptonic), emitted, emitter, spectator)
    *_, det = forward(mapping, random_numbers(mapping, batch, seed=seed))
    real_final = list(real_masses(masses, emitted)[2:])
    real = born_mapping(real_final, leptonic)
    *_, det_real = forward(real, random_numbers(real, batch, seed=seed + 1))
    assert_same_mean(det, det_real)


def boost_z(p, masses, y):
    """Boost along z by the rapidity y, in light-cone components with the
    smaller one from the mass shell."""
    q = p.copy()
    trans2 = np.asarray(masses) ** 2 + p[..., 1] ** 2 + p[..., 2] ** 2
    ahead = p[..., 3] >= 0
    big = np.where(ahead, p[..., 0] + p[..., 3], p[..., 0] - p[..., 3])
    small = trans2 / np.maximum(big, 1e-300)
    plus = np.where(ahead, big, small) * np.exp(y)
    minus = np.where(ahead, small, big) * np.exp(-y)
    q[..., 0] = 0.5 * (plus + minus)
    q[..., 3] = 0.5 * (plus - minus)
    return q


def mirror(p):
    """The rotation by pi about the x axis of mirror_momenta."""
    q = p.copy()
    q[..., 2:] = -q[..., 2:]
    return q


MIRROR_LAYOUTS = [layout for layout in LAYOUTS if not layout[1]]
MIRROR_IDS = [name for name, layout in zip(IDS, LAYOUTS) if not layout[1]]


@pytest.mark.parametrize(
    "masses,leptonic,beam_rapidity,emitted,emitter,spectator",
    MIRROR_LAYOUTS,
    ids=MIRROR_IDS,
)
def test_mirrored_beams(masses, leptonic, beam_rapidity, emitted, emitter, spectator):
    """With mirror_beams, index 0 is the mapping without, and index 1 its
    rotation by pi about the x axis in the frame of the beams, with the same
    momentum fractions and weight: the dipoles commute with the mirror."""
    batch = 2000
    plain = ms.CSMapping(
        born_mapping(masses, leptonic, beam_rapidity), emitted, emitter, spectator
    )
    mirrored = ms.CSMapping(
        born_mapping(masses, leptonic, beam_rapidity, mirror_beams=True),
        emitted,
        emitter,
        spectator,
    )
    r = random_numbers(plain, batch, seed=60 + emitted + emitter, low=0.02, high=0.98)
    momenta, x1, x2, det = forward(plain, r)
    scale = momenta[..., 0].max(1)[:, None, None]
    masses_real = real_masses(masses, emitted)

    def condition(index):
        if np.ndim(index) == 0:
            index = np.full(batch, index)
        return torch.as_tensor(index, dtype=torch.int32).to(torch.get_default_device())

    p0, x1_0, x2_0, det0 = forward(mirrored, r, [condition(0)])
    np.testing.assert_allclose(p0 / scale, momenta / scale, rtol=0, atol=1e-13)
    np.testing.assert_allclose(det0, det, rtol=1e-12)

    p1, x1_1, x2_1, det1 = forward(mirrored, r, [condition(1)])
    np.testing.assert_allclose(x1_1, x1, rtol=1e-13)
    np.testing.assert_allclose(x2_1, x2, rtol=1e-13)
    np.testing.assert_allclose(det1, det, rtol=1e-12)
    rotated = boost_z(
        mirror(boost_z(momenta, masses_real, -beam_rapidity)),
        masses_real,
        beam_rapidity,
    )
    np.testing.assert_allclose(p1 / scale, rotated / scale, rtol=0, atol=1e-12)

    # the round trip with either index per event
    index = condition((np.random.default_rng(61).random(batch) < 0.5).astype(np.int32))
    p_mix, x1_mix, x2_mix, det_mix = forward(mirrored, r, [index])
    r_inv, det_inv = inverse(mirrored, p_mix, x1_mix, x2_mix, [index])
    np.testing.assert_allclose(r_inv[:, :-3], r[:, :-3], rtol=0, atol=1e-9)
    np.testing.assert_allclose(r_inv[:, -3:], r[:, -3:], rtol=0, atol=1e-12)
    np.testing.assert_allclose(det_mix * det_inv, 1.0, rtol=1e-9)


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
    mapping = ms.CSMapping(born, 4, 3, 0)
    batch = 2000
    r = random_numbers(mapping, batch, seed=50, low=0.02, high=0.98)
    perm = torch.full((batch,), index, dtype=torch.int32).to(torch.get_default_device())
    momenta, x1, x2, det = forward(mapping, r, [perm])
    r_inv, det_inv = inverse(mapping, momenta, x1, x2, [perm])
    assert (det > 0).all()
    violation = np.abs(momenta[:, 2:].sum(1) - momenta[:, :2].sum(1)).max(1)
    assert (violation <= 1e-13 * momenta[..., 0].max(1)).all()
    np.testing.assert_allclose(r_inv, r, rtol=0, atol=1e-8)
    np.testing.assert_allclose(det * det_inv, 1.0, rtol=1e-8)


TOP, BOTTOM, W_MASS = 173.0, 4.7, 80.419


def decay_mapping(masses):
    """The mapping of the decay of masses[0] at rest into the others, through
    a W propagator into the last two for three of them."""
    final = list(masses[1:])
    if len(final) == 2:
        propagators, vertices = [], [["i0", "o0", "o1"]]
    else:
        propagators = [ms.Propagator(mass=W_MASS, width=2.0)]
        vertices = [["p0", "o1", "o2"], ["i0", "o0", "p0"]]
    diagram = ms.Diagram([masses[0]], final, propagators, vertices)
    return ms.PhaseSpaceMapping(ms.Topology(diagram), masses[0], leptonic=True)


def decay_variables(momenta, masses, i, j):
    """y and z_i of the dipole of the emitter j with the decaying particle as
    the spectator, with the recoil system K = p_a - p_i - p_j."""
    p = momenta
    big_k = p[:, 0] - p[:, i] - p[:, j]
    pipj = lc_dot(p[:, j], masses[j], p[:, i])
    pipk = dot(p[:, i], big_k)
    pjpk = dot(p[:, j], big_k)
    return pipj / (pipj + pipk + pjpk), pipk / (pipk + pjpk)


def project_decay(momenta, masses, i, j):
    """The Born of the dipole of the emitter j with the decaying particle as
    the spectator: CDST eq. 5.8 with the recoil system K of the other
    final-state particles as the spectator, each of them Lorentz transformed
    with Lambda(K, K~) (K~ itself if alone)."""
    p = momenta
    others = [n for n in range(1, p.shape[1]) if n not in (i, j)]
    big_k = p[:, others].sum(1)
    q = p[:, 0]
    q2, m_k2, m_j = dot(q, q), dot(big_k, big_k), masses[j]
    s_ij = m_j**2 + 2 * lc_dot(p[:, j], m_j, p[:, i])
    scale = np.sqrt(kallen(q2, m_j**2, m_k2) / kallen(q2, s_ij, m_k2))
    big_kt = (
        scale[:, None] * (big_k - (dot(q, big_k) / q2)[:, None] * q)
        + ((q2 + m_k2 - m_j**2) / (2 * q2))[:, None] * q
    )
    born = np.delete(p, i, axis=1)
    born[:, j - (j > i)] = q - big_kt
    for n in others:
        born[:, n - (n > i)] = (
            big_kt if len(others) == 1 else lorentz(p[:, n], big_k, big_kt)
        )
    return born


# (masses of the decay, emitted, emitter): t > b w+ with a massive and a
# massless b, with i or the recoil system before the emitter, and t > b e+ ve
# with two particles in the recoil system
DECAYS = [
    ([TOP, BOTTOM, W_MASS], 3, 1),
    ([TOP, 0.0, W_MASS], 3, 1),
    ([TOP, W_MASS, BOTTOM], 3, 2),
    ([TOP, BOTTOM, W_MASS], 1, 2),
    ([TOP, BOTTOM, 0.0, 0.0], 4, 1),
]
DECAY_IDS = ["t-bw", "t-bw-massless", "t-wb", "t-gbw", "t-benu"]


@pytest.mark.parametrize("masses,emitted,emitter", DECAYS, ids=DECAY_IDS)
def test_decay_round_trip(masses, emitted, emitter):
    """The emission off a final-state emitter with the decaying particle as
    the spectator: the decaying particle at rest, momentum conservation, mass
    shells, the projection onto the Born of the Born mapping, and the round
    trip."""
    batch = 2000
    born = decay_mapping(masses)
    mapping = ms.CSMapping(born, emitted, emitter, 0)
    assert mapping.family() == ms.CSMapping.Family.final_decay
    m = np.insert(np.array(masses), emitted, 0.0)
    r = random_numbers(mapping, batch, seed=80 + emitted + emitter, low=0.02, high=0.98)
    momenta, x1, x2, det = forward(mapping, r)
    assert momenta.shape == (batch, len(masses) + 1, 4)
    assert (det > 0).all()
    np.testing.assert_array_equal(x1, 1.0)
    rest = np.tile([1.0, 0.0, 0.0, 0.0], (batch, 1))
    np.testing.assert_allclose(momenta[:, 0] / TOP, rest, atol=1e-15)
    violation = np.abs(momenta[:, 1:].sum(1) - momenta[:, 0]).max(1)
    assert (violation <= 1e-13 * TOP).all()
    m2 = momenta[..., 0] ** 2 - (momenta[..., 1:] ** 2).sum(-1)
    np.testing.assert_allclose((m2 - m**2) / TOP**2, 0.0, atol=1e-12)
    # the projection of numpy, with the recoil system from the sum of its
    # momenta, to 1e-11 of the mass of the top
    born_momenta, *_ = forward(born, r[:, :-3])
    np.testing.assert_allclose(
        project_decay(momenta, m, emitted, emitter) / TOP,
        born_momenta / TOP,
        rtol=0,
        atol=1e-11,
    )
    # the azimuth of i about the recoil system loses digits where it is close
    # to the x axis, from which its frame takes its first axis (5e-10 for one
    # of the events of t > w+ b, its W 3 mrad from the axis)
    r_inv, det_inv = inverse(mapping, momenta, x1, x2)
    np.testing.assert_allclose(r_inv[:, :-3], r[:, :-3], rtol=0, atol=1e-9)
    np.testing.assert_allclose(r_inv[:, -3:-1], r[:, -3:-1], rtol=0, atol=1e-11)
    np.testing.assert_allclose(r_inv[:, -1], r[:, -1], rtol=0, atol=1e-9)
    np.testing.assert_allclose(det * det_inv, 1.0, rtol=1e-9)


@pytest.mark.parametrize("masses,emitted,emitter", DECAYS, ids=DECAY_IDS)
def test_decay_soft_limit(masses, emitted, emitter):
    """At r = 0 for both variables i is soft and the decay is its Born."""
    born = decay_mapping(masses)
    mapping = ms.CSMapping(born, emitted, emitter, 0)
    r = random_numbers(mapping, 500, seed=90)
    r[:, -3:-1] = 0.0
    momenta, *_, det = forward(mapping, r)
    born_momenta, *_ = forward(born, r[:, :-3])
    np.testing.assert_allclose(momenta[:, emitted] / TOP, 0.0, atol=1e-14)
    np.testing.assert_allclose(
        np.delete(momenta, emitted, axis=1) / TOP, born_momenta / TOP, atol=1e-13
    )
    np.testing.assert_array_equal(det, 0.0)


@pytest.mark.parametrize("masses,emitted,emitter", DECAYS, ids=DECAY_IDS)
def test_decay_cutoff(masses, emitted, emitter):
    """The cutoff on y and z_i of a decaying spectator, as in test_cutoff."""
    batch = 200000
    seed = 100 + emitted + emitter
    born = decay_mapping(masses)
    mapping = ms.CSMapping(born, emitted, emitter, 0, cutoff=CUTOFF)
    plain = ms.CSMapping(born, emitted, emitter, 0)
    m = np.insert(np.array(masses), emitted, 0.0)
    r = random_numbers(mapping, 2000, seed=seed, low=0.02, high=0.98)
    r[:200, -3:-1] = 0.0
    momenta, x1, x2, det = forward(mapping, r)
    y, z = decay_variables(momenta, m, emitted, emitter)
    np.testing.assert_array_equal(det[:200], 0.0)
    assert (np.minimum(y, z) >= CUTOFF * (1 - 1e-9)).all()
    trip = det > 0
    trip[:200] = False
    r_inv, det_inv = inverse(mapping, momenta[trip], x1[trip], x2[trip])
    np.testing.assert_allclose(r_inv[:, -3:], r[trip, -3:], rtol=0, atol=1e-10)
    np.testing.assert_allclose(det[trip] * det_inv, 1.0, rtol=1e-9)
    *_, det = forward(mapping, random_numbers(mapping, batch, seed=seed + 1))
    momenta, x1, x2, det_plain = forward(
        plain, random_numbers(plain, batch, seed=seed + 2)
    )
    y, z = decay_variables(momenta, m, emitted, emitter)
    below = np.minimum(y, z) < CUTOFF
    assert 0 < below.mean() < 0.9
    _, det_inv = inverse(mapping, momenta[below], x1[below], x2[below])
    np.testing.assert_array_equal(det_inv, 0.0)
    assert_same_mean(det, np.where(below, 0.0, det_plain))


def two_body(s, m1, m2):
    """The two-body phase space at s, sqrt(lambda) / (8 pi s)."""
    return np.sqrt(np.maximum(kallen(s, m1**2, m2**2), 0.0)) / (8 * math.pi * s)


@pytest.mark.parametrize("bottom", [BOTTOM, 0.0])
def test_decay_volume(bottom):
    """The dipole covers the phase space of t > b w+ g once: int dr det is
    int ds / (2 pi) Phi_2(m_t^2; sqrt(s), m_W) Phi_2(s; m_b, 0) over the mass
    s of b g, as the Born mapping covers Phi_2(m_t^2; m_b, m_W)."""
    batch = 400000
    born = decay_mapping([TOP, bottom, W_MASS])
    mapping = ms.CSMapping(born, 3, 1, 0)
    *_, det = forward(mapping, random_numbers(mapping, batch, seed=110))
    *_, det_born = forward(born, random_numbers(born, 10, seed=111))
    np.testing.assert_allclose(det_born, two_body(TOP**2, bottom, W_MASS), rtol=1e-12)
    x, w = np.polynomial.legendre.leggauss(200)
    low, high = bottom**2, (TOP - W_MASS) ** 2
    # s = low + (high - low) t^2 smooths the square root at the upper end
    t = (x + 1) / 2
    s = high - (high - low) * t * t
    jac = (high - low) * 2 * t * w / 2
    volume = np.sum(
        jac * two_body(TOP**2, np.sqrt(s), W_MASS) * two_body(s, bottom, 0.0)
    ) / (2 * math.pi)
    mean, err = det.mean(), det.std() / math.sqrt(batch)
    assert abs(mean - volume) < 5 * err
    assert err < 2e-3 * volume


def test_invalid_dipoles():
    born = born_mapping([0.0, 0.0], False)
    for emitted, emitter, spectator in [
        (1, 2, 3),  # an incoming emitted parton
        (5, 2, 3),  # past the real process
        (4, 4, 3),
        (4, 2, 2),
        (4, 2, 5),
    ]:
        with pytest.raises(ValueError):
            ms.CSMapping(born, emitted, emitter, spectator)
    # an incoming emitter or spectator needs momentum fractions
    for emitter, spectator in [(0, 2), (2, 0), (0, 1)]:
        with pytest.raises(ValueError):
            ms.CSMapping(born_mapping([0.0, 0.0], True), 4, emitter, spectator)
    # cuts on the Born
    cuts = ms.Cuts(
        [
            ms.CutItem(
                observable=ms.Observable([1, -1, 11, -11], "eta", [[11, -11]]), min=0.0
            )
        ]
    )
    with pytest.raises(ValueError):
        ms.CSMapping(
            ms.PhaseSpaceMapping(
                [0.0, 0.0, 0.0, 0.0],
                SQRT_S,
                mode=ms.PhaseSpaceMapping.TChannelMode.rambo,
                cuts=cuts,
            ),
            4,
            2,
            3,
        )
    # the decaying particle emits no parton
    with pytest.raises(ValueError, match="decaying"):
        ms.CSMapping(decay_mapping([TOP, BOTTOM, W_MASS]), 3, 0, 1)
