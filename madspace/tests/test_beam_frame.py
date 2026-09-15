"""2 -> 1 phase space and asymmetric beams (PhaseSpaceMapping beam_rapidity)."""

import math

import numpy as np
import pytest
from pytest import approx

import madspace as ms

N = 2000


@pytest.fixture
def rng():
    return np.random.default_rng(4321)


def boost_z(p, y):
    q = p.copy()
    q[..., 0] = p[..., 0] * math.cosh(y) + p[..., 3] * math.sinh(y)
    q[..., 3] = p[..., 3] * math.cosh(y) + p[..., 0] * math.sinh(y)
    return q


def two_to_one_topology(mass):
    return ms.Topology(ms.Diagram([0.0, 0.0], [mass], [], [["i0", "i1", "o0"]]))


def test_two_to_one(rng):
    mass, sqrt_s = 91.188, 13000.0
    mapping = ms.PhaseSpaceMapping(two_to_one_topology(mass), sqrt_s)
    assert mapping.random_dim() == 1
    r = rng.random((N, 1))
    p, x1, x2, det = map(np.asarray, mapping.map_forward([r], []))

    tau = mass**2 / sqrt_s**2
    assert x1 * x2 == approx(tau)
    assert p[:, 0] + p[:, 1] == approx(p[:, 2], abs=1e-9)
    assert p[:, 0, 0] == approx(x1 * sqrt_s / 2)
    assert p[:, 1, 0] == approx(x2 * sqrt_s / 2)
    mass2 = p[:, 2, 0] ** 2 - np.sum(p[:, 2, 1:] ** 2, axis=1)
    assert mass2 == approx(mass**2)
    # dPhi_1 = 2 pi delta(s_hat - m^2), dx1 dx2 = |ln tau| / s ds_hat dr
    assert det == approx(2 * math.pi * abs(math.log(tau)) / sqrt_s**2)

    r_inv, det_inv = mapping.map_inverse([p, x1, x2], [])
    assert np.asarray(r_inv) == approx(r)
    assert np.asarray(det_inv) * det == approx(1.0)

    flat = ms.PhaseSpaceMapping([0.0, 0.0, mass], sqrt_s)
    assert np.asarray(flat.map_forward([r], [])[0]) == approx(p)


def test_two_to_one_needs_pdf():
    with pytest.raises(ValueError):
        ms.PhaseSpaceMapping(two_to_one_topology(125.0), 13000.0, leptonic=True)
    with pytest.raises(ValueError):
        ms.PhaseSpaceMapping(two_to_one_topology(0.0), 13000.0)


@pytest.mark.parametrize("leptonic", [False, True], ids=["hadronic", "leptonic"])
@pytest.mark.parametrize("mode", ["rambo", "propagator"])
def test_beam_rapidity(rng, mode, leptonic):
    e1, e2 = 7000.0, 4000.0
    e_cm = 2 * math.sqrt(e1 * e2)
    y0 = 0.5 * math.log(e1 / e2)
    masses = [0.0, 0.0, 0.0, 0.0, 173.0]
    kwargs = dict(leptonic=leptonic, mode=getattr(ms.PhaseSpaceMapping, mode))
    sym = ms.PhaseSpaceMapping(masses, e_cm, **kwargs)
    lab = ms.PhaseSpaceMapping(masses, e_cm, beam_rapidity=y0, **kwargs)
    r = rng.random((N, sym.random_dim()))
    p_sym, x1_sym, x2_sym, det_sym = map(np.asarray, sym.map_forward([r], []))
    p_lab, x1, x2, det = map(np.asarray, lab.map_forward([r], []))
    x1, x2 = np.broadcast_to(x1, (N,)), np.broadcast_to(x2, (N,))
    ok = det_sym > 0

    # same point, same weight: the lab frame is a longitudinal boost away
    assert det == approx(det_sym)
    assert p_lab[ok] == approx(boost_z(p_sym[ok], y0), abs=1e-8)
    # the incoming partons carry the momentum fractions of their own beam
    assert p_lab[ok, 0, 0] == approx(x1[ok] * e1)
    assert p_lab[ok, 1, 0] == approx(x2[ok] * e2)

    r_inv, det_inv = lab.map_inverse([p_lab, x1, x2], [])
    assert np.asarray(r_inv)[ok] == approx(r[ok], abs=1e-6)
    assert np.asarray(det_inv)[ok] * det[ok] == approx(1.0, rel=1e-5)


def test_beam_rapidity_cut_in_lab_frame(rng):
    """An eta cut is applied to the boosted (lab) momenta."""
    e1, e2 = 7000.0, 1000.0
    e_cm = 2 * math.sqrt(e1 * e2)
    y0 = 0.5 * math.log(e1 / e2)
    pids = [2, -2, 11, -11]
    cuts = ms.Cuts([
        ms.CutItem(observable=ms.Observable(pids, "eta", [[11, -11]]), min=0.0)
    ])
    mapping = ms.PhaseSpaceMapping(
        [0.0] * 4, e_cm, cuts=cuts, beam_rapidity=y0, mode=ms.PhaseSpaceMapping.rambo
    )
    free = ms.PhaseSpaceMapping(
        [0.0] * 4, e_cm, beam_rapidity=y0, mode=ms.PhaseSpaceMapping.rambo
    )
    r = rng.random((N, mapping.random_dim()))
    p, _, _, det = map(np.asarray, mapping.map_forward([r], []))
    p_free, _, _, det_free = map(np.asarray, free.map_forward([r], []))
    eta = np.arctanh(p_free[:, 2:, 3] / np.linalg.norm(p_free[:, 2:, 1:], axis=2))
    passed = (eta > 0).all(axis=1) & (det_free > 0)
    assert passed.any() and not passed.all()
    assert ((det > 0) == passed).all()


@pytest.mark.parametrize("pdf_asymmetric_only", [False, True], ids=["energies", "pdfs"])
def test_mirror_beams(rng, pdf_asymmetric_only):
    """mirror_index 1: leg 1 from beam 2, i.e. rotated in the beams' frame"""
    e1, e2 = (6500.0, 6500.0) if pdf_asymmetric_only else (7000.0, 4000.0)
    e_cm = 2 * math.sqrt(e1 * e2)
    y0 = 0.5 * math.log(e1 / e2)
    masses = [0.0, 0.0, 0.0, 0.0]
    plain = ms.PhaseSpaceMapping(masses, e_cm, beam_rapidity=y0)
    mirror = ms.PhaseSpaceMapping(masses, e_cm, beam_rapidity=y0, mirror_beams=True)
    assert mirror.mirror_beams() and not plain.mirror_beams()
    r = rng.random((N, plain.random_dim()))
    p, x1, x2, det = map(np.asarray, plain.map_forward([r], []))
    zeros, ones = np.zeros(N, dtype=np.int32), np.ones(N, dtype=np.int32)
    p0, x1_0, x2_0, det0 = map(np.asarray, mirror.map_forward([r], [zeros]))
    p1, x1_1, x2_1, det1 = map(np.asarray, mirror.map_forward([r], [ones]))
    ok = det > 0
    assert p0[ok] == approx(p[ok], abs=1e-8)
    assert det1 == approx(det)
    assert x1_1 == approx(x1)
    assert x2_1 == approx(x2)
    # leg 1 now moves along -z with its fraction of beam 2, leg 2 along +z
    assert (p1[ok, 0, 3] < 0).all()
    assert p1[ok, 0, 0] == approx(x1[ok] * e2)
    assert (p1[ok, 1, 3] > 0).all()
    assert p1[ok, 1, 0] == approx(x2[ok] * e1)
    # which is the beams'-frame rotation of the unmirrored event
    rot = boost_z(p[ok], -y0)
    rot[..., 2:] *= -1
    assert p1[ok] == approx(boost_z(rot, y0), abs=1e-8)
    r_inv, det_inv = mirror.map_inverse([p1, x1_1, x2_1], [ones])
    assert np.asarray(r_inv)[ok] == approx(r[ok], abs=1e-6)
