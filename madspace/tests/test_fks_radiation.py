"""FKSRadiation: the FKS emission off a massless final-state parton as a mapping.

The reference points of test_data/fks (see test_fks_reference.py) give the
real-emission momenta and the radiation Jacobian of genps_fks.f for a Born
point and the last three entries of the unit-cube vector. The mapping must
reproduce them, and its inverse the Born and the three random numbers. The
weight of the mapping is the full Jacobian, while genps_fks.f returns the
radiation Jacobian times compute_flux without the factor xi, in units of
xi_max and without the factors 1 - stiny and 1 - ctiny of the sampling (see
test_fks_kernels.py).
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


def conditioning(r, born, momenta, i_fks, j_fks):
    """Precision to which the real momenta determine the random numbers and the
    Jacobian.

    xi_hat = stiny + (1 - stiny) r_xi^2 and y = 1 - 2 (ctiny + (1 - ctiny) r_y^2)
    resolve r_xi only to eps stiny / r_xi and r_y only to eps / r_y absolute
    (y is a double close to 1 near the collinear cutoff), and the Jacobian,
    which is linear in both, to eps (stiny / r_xi^2 + 1 / r_y^2) relative. The
    azimuth of i about i + j is resolved to eps / sin(theta), with theta the
    angle between i and i + j, which vanishes in the collinear limits and when j
    becomes soft (xi_hat -> 1), and to eps / sin(theta_jbar) through the
    azimuth of the Born emitter, which sets the frame of phi and is ill defined
    for an emitter close to the beam axis. This is the conditioning of the variables
    (genps_fks.f uses the same), up to rounding. Returns the bounds on r_xi and
    r_y, on r_phi and on det det_inv - 1, with a margin of about 10 over what
    four million random points need.
    """
    r_xi, r_y = r[:, 0], r[:, 1]
    p_i = momenta[:, i_fks, 1:]
    k = p_i + momenta[:, j_fks, 1:]
    sin_ik = np.linalg.norm(np.cross(p_i, k), axis=1) / (
        np.linalg.norm(p_i, axis=1) * np.linalg.norm(k, axis=1)
    )
    bound_r = 10 * EPS * (1 / r_y + STINY / r_xi) + 1e-13
    p_jbar = born[:, j_fks if j_fks < i_fks else j_fks - 1, 1:]
    sin_jbar = np.hypot(p_jbar[:, 0], p_jbar[:, 1]) / np.linalg.norm(p_jbar, axis=1)
    bound_phi = 1000 * EPS / sin_ik + 10 * EPS / sin_jbar + 1e-12
    bound_det = 10 * EPS * (1 / r_y**2 + STINY / r_xi**2) + 1e-11
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


def forward(mapping, r, born):
    momenta, det = mapping.map_forward(
        [to_torch(r[:, 0]), to_torch(r[:, 1]), to_torch(r[:, 2]), to_torch(born)]
    )
    return to_numpy(momenta), to_numpy(det)


def inverse(mapping, momenta):
    r_xi, r_y, r_phi, born, det = mapping.map_inverse([to_torch(momenta)])
    r = np.stack([to_numpy(r_xi), to_numpy(r_y), to_numpy(r_phi)], 1)
    return r, to_numpy(born), to_numpy(det)


def rambo(n_out, batch, seed):
    """Massless n_out-body phase space at SQRT_S in the partonic c.m. frame."""
    mapping = ms.PhaseSpaceMapping(
        [0.0] * (n_out + 2), SQRT_S, mode="rambo", leptonic=True
    )
    r = torch.rand(
        (batch, mapping.random_dim()), generator=generator(seed), device="cpu"
    )
    out = mapping.map_forward([r.to(torch.get_default_device())])
    return to_numpy(out[0]), to_numpy(out[-1])


def massless_final_cases():
    cases = []
    for path in DATA:
        with open(path) as f:
            data = json.load(f)
        nin = data["nincoming"]
        for conf in data["configs"]:
            j = conf["j_fks"]
            if j <= nin or conf["masses"][j - 1] > 0:
                continue
            points = [p for p in conf["points"] if p["jac"] > 0 and "rad_jac" in p]
            if points:
                name = os.path.basename(path)[:-5]
                cases.append(
                    pytest.param(data, conf, points, id=f"{name}-fks{conf['nfks']}")
                )
    return cases


@pytest.mark.parametrize("data,conf,points", massless_final_cases())
def test_matches_genps_fks(data, conf, points):
    nex = data["nexternal"]
    mapping = ms.FKSRadiation(nex - 1, conf["i_fks"] - 1, conf["j_fks"] - 1)
    assert mapping.random_dim() == 3
    p_born = np.array([p["p_born"] for p in points])
    p_ev = np.array([p["p_ev"] for p in points])
    r = np.array([p["x"][-3:] for p in points])
    scale = np.sqrt([p["shat_born"] for p in points])[:, None, None]

    momenta, det = forward(mapping, r, p_born)
    np.testing.assert_allclose(momenta / scale, p_ev / scale, rtol=0, atol=1e-12)

    def field(key):
        return np.array([p[key] for p in points])

    fortran = field("rad_jac") / flux_factor(field("shat_ev"), nex)
    expected = fortran * field("xi_ev") * field("xiimax_ev") * (1 - STINY) * (1 - CTINY)
    np.testing.assert_allclose(det, expected, rtol=1e-12)

    r_inv, born_inv, det_inv = inverse(mapping, p_ev)
    i_fks, j_fks = conf["i_fks"] - 1, conf["j_fks"] - 1
    bound_r, bound_phi, bound_det = conditioning(r, p_born, p_ev, i_fks, j_fks)
    assert (np.abs(r_inv[:, :2] - r[:, :2]).max(1) <= bound_r).all()
    assert (azimuth_difference(r_inv[:, 2], r[:, 2]) <= bound_phi).all()
    np.testing.assert_allclose(born_inv / scale, p_born / scale, rtol=0, atol=1e-13)
    assert (np.abs(det * det_inv - 1) <= bound_det).all()


def test_reference_has_hadronic_massless_final_emitters():
    ids = [case.id for case in massless_final_cases()]
    assert any(i.startswith("pp_") for i in ids)
    assert any(i.startswith("epem_") for i in ids)


# (number of Born final-state particles, i_fks, j_fks), with i before or after
# j and spectators before, between and after the FKS pair
LAYOUTS = [(2, 4, 3), (3, 5, 2), (3, 2, 4), (3, 3, 5), (4, 4, 2)]


@pytest.mark.parametrize("n_out,i_fks,j_fks", LAYOUTS)
def test_round_trip(n_out, i_fks, j_fks):
    batch = 2000
    born, _ = rambo(n_out, batch, seed=10 + n_out)
    r = np.random.default_rng(11).random((batch, 3))
    mapping = ms.FKSRadiation(n_out + 2, i_fks, j_fks)
    momenta, det = forward(mapping, r, born)

    assert momenta.shape == (batch, n_out + 3, 4)
    np.testing.assert_array_equal(momenta[:, :2], born[:, :2])

    # The emission conserves momentum to rounding. The Born (here from RAMBO,
    # which near the edge of phase space violates conservation by up to 1e-11
    # of sqrt(s)) passes its own violation on through the boost of the
    # spectators, which stretches a four-vector by at most exp(|eta|) in the
    # Euclidean norm, with exp(eta) = s_rec / sqrt(s) and s_rec = E + |p| of
    # the recoil of the real emission.
    def violation(p):
        return np.linalg.norm(p[:, 2:].sum(1) - p[:, :2].sum(1), axis=1) / SQRT_S

    k = momenta[:, i_fks] + momenta[:, j_fks]
    s_rec = SQRT_S - k[:, 0] + np.linalg.norm(k[:, 1:], axis=1)
    stretch = np.maximum(SQRT_S / s_rec, s_rec / SQRT_S)
    assert (violation(momenta) <= stretch * violation(born) + 1e-14).all()
    m2 = momenta[..., 0] ** 2 - (momenta[..., 1:] ** 2).sum(-1)
    np.testing.assert_allclose(m2 / SQRT_S**2, 0.0, atol=1e-12)
    assert (det > 0).all()

    r_inv, born_inv, det_inv = inverse(mapping, momenta)
    bound_r, bound_phi, bound_det = conditioning(r, born, momenta, i_fks, j_fks)
    assert (np.abs(r_inv[:, :2] - r[:, :2]).max(1) <= bound_r).all()
    assert (azimuth_difference(r_inv[:, 2], r[:, 2]) <= bound_phi).all()
    np.testing.assert_allclose(born_inv / SQRT_S, born / SQRT_S, rtol=0, atol=1e-12)
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
    mapping = ms.FKSRadiation(4, 4, 3)
    momenta, _ = forward(mapping, r, born)

    k = momenta[:, 3] + momenta[:, 4]
    cross = (
        np.cross(k[:, 1:], born[:, 3, 1:]) / np.linalg.norm(k[:, 1:], axis=1)[:, None]
    )
    np.testing.assert_allclose(cross / SQRT_S, 0.0, atol=1e-15)
    np.testing.assert_allclose(
        momenta[:, 2:].sum(1) / SQRT_S, momenta[:, :2].sum(1) / SQRT_S, atol=1e-14
    )
    _, born_inv, _ = inverse(mapping, momenta)
    np.testing.assert_allclose(born_inv / SQRT_S, born / SQRT_S, rtol=0, atol=1e-13)


@pytest.mark.parametrize("n_out,i_fks,j_fks", [(2, 4, 2), (3, 2, 4)])
def test_volume(n_out, i_fks, j_fks):
    """A RAMBO Born times the emission covers the (n_out + 1)-body phase space
    once, for any position of i and j."""
    batch = 200000
    born, det_born = rambo(n_out, batch, seed=20 + n_out)
    r = np.random.default_rng(21).random((batch, 3))
    _, det = forward(ms.FKSRadiation(n_out + 2, i_fks, j_fks), r, born)
    weight = det_born * det

    _, det_real = rambo(n_out + 1, batch, seed=22 + n_out)
    mean, err = weight.mean(), weight.std() / math.sqrt(batch)
    mean_ref, err_ref = det_real.mean(), det_real.std() / math.sqrt(batch)
    assert abs(mean - mean_ref) < 5 * math.hypot(err, err_ref)


@pytest.mark.parametrize(
    "born_count,i_fks,j_fks",
    [(2, 2, 3), (4, 4, 4), (4, 4, 1), (4, 0, 3), (4, 5, 3), (4, 4, 5)],
    ids=["no final state", "i equals j", "initial j", "initial i", "i out", "j out"],
)
def test_invalid_layouts(born_count, i_fks, j_fks):
    with pytest.raises(ValueError):
        ms.FKSRadiation(born_count, i_fks, j_fks)
