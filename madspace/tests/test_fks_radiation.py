"""FKSRadiation: the FKS emission off a massless final-state or an incoming
parton as a mapping.

The reference points of test_data/fks (see test_fks_reference.py) give the
real-emission momenta, momentum fractions and radiation Jacobian of the FKS map
of genps_fks.f (the rad_ entries, without the lower bound the cuts put on the
partonic energy) for a Born point and the last three entries of the unit-cube
vector. The mapping must reproduce them, and its inverse the Born and the three
random numbers. The weight of the mapping is the full Jacobian, while
genps_fks.f returns the radiation Jacobian times compute_flux without the
factor xi, in units of xi_max and without the factors 1 - stiny and 1 - ctiny of
the sampling (see test_fks_kernels.py).
"""

import glob
import json
import math
import os

import numpy as np
import pytest
import torch
from madspace_testing import generator

import madspace as ms

STINY, CTINY = 1e-6, 5e-7
SQRT_S = 1000.0
EPS = np.finfo(np.float64).eps
# tau of the Born in the initial-state tests, so that sqrt(s_lab) = SQRT_S / sqrt(TAU)
TAU = 0.01


def conditioning(r, born, momenta, i_fks, j_fks, x_born=None):
    """Precision to which the real momenta determine the random numbers and the
    Jacobian.

    xi_hat = stiny + (1 - stiny) r_xi^2 and y = 1 - 2 (ctiny + (1 - ctiny) r_y^2)
    resolve r_xi only to eps stiny / r_xi and r_y only to eps / r_y absolute
    (y is a double close to 1 near the collinear cutoff), and the Jacobian,
    which is linear in both, to eps (stiny / r_xi^2 + 1 / r_y^2) relative. The
    azimuth is resolved to eps / sin(theta), with theta the angle between i and
    its axis: i + j for a final-state emitter, which i approaches in the
    collinear limits and when j becomes soft (xi_hat -> 1), and the beam axis
    for an incoming one. For a final-state emitter it is also resolved only to
    eps / sin(theta_jbar) through the azimuth of the Born emitter, which sets
    the frame of phi and is ill defined for an emitter close to the beam axis.
    For an incoming emitter xi_max vanishes like 1 - x of the Born when an x of
    the Born approaches 1, which a double resolves only to eps / (1 - x), so
    xi_hat and the Jacobian are resolved to that relative precision too. This
    is the conditioning of the variables (genps_fks.f uses the same), up to
    rounding. Returns the bounds on r_xi and r_y, on r_phi and on
    det det_inv - 1, with a margin of about 10 over what four million random
    points need.
    """
    r_xi, r_y = r[:, 0], r[:, 1]
    p_i = momenta[:, i_fks, 1:]
    if j_fks < 2:
        axis = np.zeros_like(p_i)
        axis[:, 2] = 1.0
        frame = 0.0
    else:
        axis = p_i + momenta[:, j_fks, 1:]
        p_jbar = born[:, j_fks if j_fks < i_fks else j_fks - 1, 1:]
        sin_jbar = np.hypot(p_jbar[:, 0], p_jbar[:, 1]) / np.linalg.norm(p_jbar, axis=1)
        frame = 10 * EPS / sin_jbar
    sin_axis = np.linalg.norm(np.cross(p_i, axis), axis=1) / (
        np.linalg.norm(p_i, axis=1) * np.linalg.norm(axis, axis=1)
    )
    x_edge = 0.0
    if x_born is not None:
        x_edge = 10 * EPS / (1 - np.maximum(x_born[0], x_born[1]))
    bound_r = 10 * EPS * (1 / r_y + STINY / r_xi) + x_edge + 1e-13
    bound_phi = 1000 * EPS / sin_axis + frame + 1e-12
    bound_det = 10 * EPS * (1 / r_y**2 + STINY / r_xi**2) + x_edge + 1e-11
    return bound_r, bound_phi, bound_det


def azimuth_difference(r_phi, r_phi_ref):
    return np.abs((r_phi - r_phi_ref + 0.5) % 1.0 - 0.5)


DATA = sorted(
    glob.glob(
        os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "test_data", "fks", "*.json"
        )
    )
)


def to_torch(x):
    return torch.as_tensor(np.asarray(x), dtype=torch.float64)


def to_numpy(x):
    return torch.as_tensor(x).cpu().numpy()


def flux_factor(shat, nexternal, nin=2):
    """compute_flux of genps_fks.f for massless incoming partons."""
    pi2 = 2 * math.pi
    return 1 / (2 * shat) / pi2 ** (3 * (nexternal - nin) - 4) * pi2**3


def forward(mapping, r, born, x=None):
    """Returns the real momenta, the momentum fractions (None if leptonic) and
    the weight."""
    inputs = [to_torch(r[:, 0]), to_torch(r[:, 1]), to_torch(r[:, 2]), to_torch(born)]
    if x is None:
        momenta, det = mapping.map_forward(inputs)
        return to_numpy(momenta), None, to_numpy(det)
    momenta, x1, x2, det = mapping.map_forward(
        inputs + [to_torch(x[0]), to_torch(x[1])]
    )
    return to_numpy(momenta), (to_numpy(x1), to_numpy(x2)), to_numpy(det)


def inverse(mapping, momenta, x=None):
    """Returns the random numbers, the Born momenta and momentum fractions (None if
    leptonic) and the weight."""
    if x is None:
        r_xi, r_y, r_phi, born, det = mapping.map_inverse([to_torch(momenta)])
        x_born = None
    else:
        r_xi, r_y, r_phi, born, x1, x2, det = mapping.map_inverse(
            [to_torch(momenta), to_torch(x[0]), to_torch(x[1])]
        )
        x_born = (to_numpy(x1), to_numpy(x2))
    r = np.stack([to_numpy(r_xi), to_numpy(r_y), to_numpy(r_phi)], 1)
    return r, to_numpy(born), x_born, to_numpy(det)


def rambo(masses, batch, seed, leptonic=True):
    """Phase space at SQRT_S (leptonic, in the partonic c.m. frame) or at the
    hadronic SQRT_S (momenta in the lab frame). Returns the momenta, the
    momentum fractions (None if leptonic) and the weight."""
    mapping = ms.PhaseSpaceMapping(
        list(masses), SQRT_S, mode="rambo", leptonic=leptonic
    )
    r = torch.rand(
        (batch, mapping.random_dim()), generator=generator(seed), device="cpu"
    )
    out = mapping.map_forward([r.to(torch.get_default_device())])
    x = None if leptonic else (to_numpy(out[1]), to_numpy(out[2]))
    return to_numpy(out[0]), x, to_numpy(out[-1])


def random_born_x(batch, seed):
    """Born momentum fractions with x1 x2 = TAU."""
    u = np.random.default_rng(seed).random(batch)
    return TAU**u, TAU ** (1 - u)


def reference_cases():
    """Every massless emitter of the reference data, final-state and incoming."""
    cases = []
    for path in DATA:
        with open(path) as f:
            data = json.load(f)
        nin = data["nincoming"]
        for conf in data["configs"]:
            j = conf["j_fks"]
            if j > nin and conf["masses"][j - 1] > 0:
                continue
            points = [p for p in conf["points"] if p["jac"] > 0 and "rad_jac" in p]
            if points:
                name = os.path.basename(path)[:-5]
                kind = "initial" if j <= nin else "final"
                cases.append(
                    pytest.param(
                        data, conf, points, id=f"{name}-fks{conf['nfks']}-{kind}"
                    )
                )
    return cases


@pytest.mark.parametrize("data,conf,points", reference_cases())
def test_matches_genps_fks(data, conf, points):
    nex = data["nexternal"]
    i_fks, j_fks = conf["i_fks"] - 1, conf["j_fks"] - 1
    leptonic = data["lpp"] == [0, 0]
    mapping = ms.FKSRadiation(nex - 1, i_fks, j_fks, leptonic=leptonic)
    assert mapping.random_dim() == 3
    assert mapping.initial_state() == (j_fks < 2)

    def field(key):
        return np.array([p[key] for p in points])

    p_born = field("p_born")
    p_ev = field("rad_p_ev")
    r = field("x")[:, -3:]
    x_born = None if leptonic else tuple(field("xbjrk_born").T)
    # momenta in units of the hardest particle of the real emission, which for
    # an incoming emitter is up to 1 / (1 - xi) harder than the Born
    scale = p_ev[..., 0].max(1)[:, None, None]

    momenta, x, det = forward(mapping, r, p_born, x_born)
    np.testing.assert_allclose(momenta / scale, p_ev / scale, rtol=0, atol=1e-13)
    if not leptonic:
        np.testing.assert_allclose(np.stack(x, 1), field("rad_xbjrk_ev"), rtol=1e-13)

    fortran = field("rad_jac") / flux_factor(field("rad_shat_ev"), nex)
    expected = (
        fortran
        * field("rad_xi_ev")
        * field("rad_xiimax_ev")
        * (1 - STINY)
        * (1 - CTINY)
    )
    np.testing.assert_allclose(det, expected, rtol=1e-12)

    r_inv, born_inv, x_inv, det_inv = inverse(mapping, p_ev, x)
    bound_r, bound_phi, bound_det = conditioning(r, p_born, p_ev, i_fks, j_fks, x_born)
    assert (np.abs(r_inv[:, :2] - r[:, :2]).max(1) <= bound_r).all()
    assert (azimuth_difference(r_inv[:, 2], r[:, 2]) <= bound_phi).all()
    np.testing.assert_allclose(born_inv / scale, p_born / scale, rtol=0, atol=1e-13)
    if not leptonic:
        np.testing.assert_allclose(np.stack(x_inv, 1), np.stack(x_born, 1), rtol=1e-12)
    assert (np.abs(det * det_inv - 1) <= bound_det).all()


def test_reference_covers_all_massless_emitters():
    ids = [case.id for case in reference_cases()]
    assert any(i.startswith("epem_") and i.endswith("final") for i in ids)
    assert any(i.startswith("pp_") and i.endswith("final") for i in ids)
    assert any(i.endswith("initial") for i in ids)


def violation(p):
    """Momentum conservation, Euclidean norm of the four-vector, in units of
    SQRT_S."""
    return np.linalg.norm(p[:, 2:].sum(1) - p[:, :2].sum(1), axis=1) / SQRT_S


# (number of Born final-state particles, i_fks, j_fks), with i before or after
# j and spectators before, between and after the FKS pair
LAYOUTS = [(2, 4, 3), (3, 5, 2), (3, 2, 4), (3, 3, 5), (4, 4, 2)]


@pytest.mark.parametrize("n_out,i_fks,j_fks", LAYOUTS)
def test_round_trip(n_out, i_fks, j_fks):
    batch = 2000
    born, _, _ = rambo([0.0] * (n_out + 2), batch, seed=10 + n_out)
    r = np.random.default_rng(11).random((batch, 3))
    mapping = ms.FKSRadiation(n_out + 2, i_fks, j_fks, leptonic=True)
    momenta, _, det = forward(mapping, r, born)

    assert momenta.shape == (batch, n_out + 3, 4)
    np.testing.assert_array_equal(momenta[:, :2], born[:, :2])

    # The emission conserves momentum to rounding. The Born (here from RAMBO,
    # which near the edge of phase space violates conservation by up to 1e-11
    # of sqrt(s)) passes its own violation on through the boost of the
    # spectators, which stretches a four-vector by at most exp(|eta|) in the
    # Euclidean norm, with exp(eta) = s_rec / sqrt(s) and s_rec = E + |p| of
    # the recoil of the real emission.
    k = momenta[:, i_fks] + momenta[:, j_fks]
    s_rec = SQRT_S - k[:, 0] + np.linalg.norm(k[:, 1:], axis=1)
    stretch = np.maximum(SQRT_S / s_rec, s_rec / SQRT_S)
    assert (violation(momenta) <= stretch * violation(born) + 1e-14).all()
    m2 = momenta[..., 0] ** 2 - (momenta[..., 1:] ** 2).sum(-1)
    np.testing.assert_allclose(m2 / SQRT_S**2, 0.0, atol=1e-12)
    assert (det > 0).all()

    r_inv, born_inv, _, det_inv = inverse(mapping, momenta)
    bound_r, bound_phi, bound_det = conditioning(r, born, momenta, i_fks, j_fks)
    assert (np.abs(r_inv[:, :2] - r[:, :2]).max(1) <= bound_r).all()
    assert (azimuth_difference(r_inv[:, 2], r[:, 2]) <= bound_phi).all()
    np.testing.assert_allclose(born_inv / SQRT_S, born / SQRT_S, rtol=0, atol=1e-12)
    assert (np.abs(det * det_inv - 1) <= bound_det).all()


# (Born final-state masses, i_fks, j_fks) with an incoming emitter j
INITIAL_LAYOUTS = [
    ([0.0, 0.0], 4, 0),
    ([0.0, 0.0, 0.0], 2, 1),
    ([0.0, 0.0, 0.0], 5, 0),
    ([173.0, 173.0], 3, 1),
]


@pytest.mark.parametrize("masses,i_fks,j_fks", INITIAL_LAYOUTS)
def test_round_trip_initial(masses, i_fks, j_fks):
    batch = 2000
    n_born = len(masses) + 2
    born, _, _ = rambo([0.0, 0.0] + masses, batch, seed=40 + n_born)
    x_born = random_born_x(batch, seed=41)
    r = np.random.default_rng(42).random((batch, 3))
    mapping = ms.FKSRadiation(n_born, i_fks, j_fks)
    assert mapping.initial_state() and not mapping.leptonic()
    momenta, x, det = forward(mapping, r, born, x_born)
    x1, x2 = x

    # the real emission: x <= 1, its partonic energy and its incoming momenta
    # in the Born c.m. frame, which moves with the Born rapidity in the lab
    assert ((x1 <= 1) & (x2 <= 1)).all()
    s_lab = SQRT_S**2 / TAU
    p_in = momenta[:, 0] + momenta[:, 1]
    s_real = p_in[:, 0] ** 2 - (p_in[:, 1:] ** 2).sum(1)
    np.testing.assert_allclose(s_real, x1 * x2 * s_lab, rtol=1e-12)
    plus1 = momenta[:, 0, 0] + momenta[:, 0, 3]
    np.testing.assert_allclose(
        plus1, x1 * math.sqrt(s_lab) * np.sqrt(x_born[1] / x_born[0]), rtol=1e-12
    )

    # momentum conservation on top of the Born's, which the transverse boost
    # of its final state stretches by at most exp(|eta|) = (E + |p_T|) / sqrt(s)
    # of the recoil
    recoil = momenta[:, 2:].sum(1) - momenta[:, i_fks]
    stretch = (recoil[:, 0] + np.hypot(recoil[:, 1], recoil[:, 2])) / SQRT_S
    scale = momenta[..., 0].max(1)
    assert (
        violation(momenta) * SQRT_S
        <= stretch * violation(born) * SQRT_S + 1e-14 * scale
    ).all()
    # the transverse boost leaves the longitudinal momentum of the final state
    # of the Born unchanged (zero, up to the Born's own violation), so the event
    # stays in the Born c.m. frame
    np.testing.assert_allclose(
        recoil[:, 3] / scale, born[:, 2:, 3].sum(1) / scale, atol=1e-13
    )
    m2 = momenta[..., 0] ** 2 - (momenta[..., 1:] ** 2).sum(-1)
    m2_expected = np.array([0.0] * (n_born + 1))
    for k, m in enumerate(masses):
        m2_expected[k + 2 + (k + 2 >= i_fks)] = m * m
    np.testing.assert_allclose(
        (m2 - m2_expected) / scale[:, None] ** 2, 0.0, atol=1e-12
    )
    assert (det > 0).all()

    r_inv, born_inv, x_inv, det_inv = inverse(mapping, momenta, x)
    bound_r, bound_phi, bound_det = conditioning(r, born, momenta, i_fks, j_fks, x_born)
    assert (np.abs(r_inv[:, :2] - r[:, :2]).max(1) <= bound_r).all()
    assert (azimuth_difference(r_inv[:, 2], r[:, 2]) <= bound_phi).all()
    np.testing.assert_allclose(
        born_inv / scale[:, None, None], born / scale[:, None, None], atol=1e-12
    )
    np.testing.assert_allclose(np.stack(x_inv, 1), np.stack(x_born, 1), rtol=1e-12)
    assert (np.abs(det * det_inv - 1) <= bound_det).all()


@pytest.mark.parametrize("theta", [1e-3, 1e-6, 1e-9, 0.0, math.pi - 1e-6, math.pi])
def test_emitter_near_the_beam_axis(theta):
    """i + j stays along the Born emitter, and the event conserves momentum, for
    emitters close to the z axis (getangles of genps_fks.f takes the sine of the
    polar angle from its cosine and loses half its digits there)."""
    batch = 500
    e = 0.5 * SQRT_S
    p_bar = e * np.array([1.0, math.sin(theta), 0.0, math.cos(theta)])
    born = np.zeros((batch, 4, 4))
    born[:, 0] = [e, 0, 0, e]
    born[:, 1] = [e, 0, 0, -e]
    born[:, 2] = p_bar * [1, -1, -1, -1]
    born[:, 3] = p_bar
    r = np.random.default_rng(31).random((batch, 3))
    mapping = ms.FKSRadiation(4, 4, 3, leptonic=True)
    momenta, _, _ = forward(mapping, r, born)

    k = momenta[:, 3] + momenta[:, 4]
    cross = (
        np.cross(k[:, 1:], born[:, 3, 1:]) / np.linalg.norm(k[:, 1:], axis=1)[:, None]
    )
    np.testing.assert_allclose(cross / SQRT_S, 0.0, atol=1e-15)
    np.testing.assert_allclose(
        momenta[:, 2:].sum(1) / SQRT_S, momenta[:, :2].sum(1) / SQRT_S, atol=1e-14
    )
    _, born_inv, _, _ = inverse(mapping, momenta)
    np.testing.assert_allclose(born_inv / SQRT_S, born / SQRT_S, rtol=0, atol=1e-13)


def assert_same_mean(weight, weight_ref):
    batch = len(weight)
    mean, err = weight.mean(), weight.std() / math.sqrt(batch)
    mean_ref, err_ref = weight_ref.mean(), weight_ref.std() / math.sqrt(len(weight_ref))
    assert abs(mean - mean_ref) < 5 * math.hypot(err, err_ref)


@pytest.mark.parametrize("n_out,i_fks,j_fks", [(2, 4, 2), (3, 2, 4)])
def test_volume(n_out, i_fks, j_fks):
    """A RAMBO Born times the emission covers the (n_out + 1)-body phase space
    once, for any position of i and j."""
    batch = 200000
    born, _, det_born = rambo([0.0] * (n_out + 2), batch, seed=20 + n_out)
    r = np.random.default_rng(21).random((batch, 3))
    mapping = ms.FKSRadiation(n_out + 2, i_fks, j_fks, leptonic=True)
    _, _, det = forward(mapping, r, born)
    _, _, det_real = rambo([0.0] * (n_out + 3), batch, seed=22 + n_out)
    assert_same_mean(det_born * det, det_real)


@pytest.mark.parametrize("j_fks", [0, 1])
def test_volume_initial(j_fks):
    """g g > t t~ at sqrt(s_lab) = SQRT_S times an emission off an incoming
    gluon covers the phase space of g g > t t~ g, momentum fractions included,
    once."""
    batch = 400000
    born, x_born, det_born = rambo([0.0, 0.0, 173.0, 173.0], batch, 50, False)
    r = np.random.default_rng(51).random((batch, 3))
    mapping = ms.FKSRadiation(4, 4, j_fks)
    _, _, det = forward(mapping, r, born, x_born)
    _, _, det_real = rambo([0.0, 0.0, 173.0, 173.0, 0.0], batch, 52, False)
    assert_same_mean(det_born * det, det_real)


@pytest.mark.parametrize(
    "born_count,i_fks,j_fks,leptonic",
    [
        (2, 2, 3, True),
        (4, 4, 4, True),
        (4, 4, 1, True),
        (4, 0, 3, False),
        (4, 5, 3, False),
        (4, 4, 5, False),
    ],
    ids=[
        "no final state",
        "i equals j",
        "leptonic incoming j",
        "initial i",
        "i out",
        "j out",
    ],
)
def test_invalid_layouts(born_count, i_fks, j_fks, leptonic):
    with pytest.raises(ValueError):
        ms.FKSRadiation(born_count, i_fks, j_fks, leptonic=leptonic)
