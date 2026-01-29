import math

import numpy as np
import pytest
from pytest import approx

import madspace as ms


@pytest.fixture
def rng():
    return np.random.default_rng(1234)


@pytest.fixture(
    params=[
        [0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [173.0, 173.0],
        [173.0, 173.0, 0.0],
        [173.0, 173.0, 0.0, 0.0],
        [173.0, 173.0, 0.0, 0.0, 0.0],
        [80.0, 80.0],
        [80.0, 80.0, 80.0],
        [80.0, 80.0, 80.0, 80.0],
        [80.0, 80.0, 80.0, 80.0, 80.0],
    ],
    ids=[
        "2 particles, massless",
        "3 particles, massless",
        "4 particles, massless",
        "5 particles, massless",
        "2 particles, t tbar",
        "3 particles, t tbar",
        "4 particles, t tbar",
        "5 particles, t tbar",
        "2 particles, W",
        "3 particles, W",
        "4 particles, W",
        "5 particles, W",
    ],
)
def masses(request):
    return [0.0, 0.0, *request.param]


@pytest.fixture(
    params=[
        ms.PhaseSpaceMapping.propagator,
        ms.PhaseSpaceMapping.rambo,  # , ms.PhaseSpaceMapping.chili
    ],
    ids=["propagator", "rambo"],  # , "chili"]
)
def mode(request):
    return request.param


BATCH_SIZE = 1000
CM_ENERGY = 13000.0


def test_t_channel_masses(masses, rng, mode):
    mapping = ms.PhaseSpaceMapping(masses, CM_ENERGY, mode=mode)
    r = rng.random((BATCH_SIZE, mapping.random_dim()))
    (p_ext, x1, x2), det = mapping.map_forward([r])

    m_ext_true = np.full((BATCH_SIZE, len(masses)), masses)
    m_ext = np.sqrt(
        np.maximum(0, p_ext[:, :, 0] ** 2 - np.sum(p_ext[:, :, 1:] ** 2, axis=2))
    )
    assert m_ext == approx(m_ext_true, abs=1e-3, rel=1e-3)


def test_t_channel_incoming(masses, rng, mode):
    mapping = ms.PhaseSpaceMapping(masses, CM_ENERGY, mode=mode)
    r = rng.random((BATCH_SIZE, mapping.random_dim()))
    (p_ext, x1, x2), det = mapping.map_forward([r])
    zeros = np.zeros(BATCH_SIZE)
    p_a = p_ext[:, 0]
    p_b = p_ext[:, 1]
    e_beam = 0.5 * CM_ENERGY

    assert p_a[:, 0] == approx(p_a[:, 3]) and p_b[:, 0] == approx(-p_b[:, 3])
    assert p_a[:, 1] == approx(zeros) and p_a[:, 2] == approx(zeros)
    assert p_b[:, 1] == approx(zeros) and p_b[:, 2] == approx(zeros)
    assert np.all(x1 >= 0) and np.all(x1 <= 1)
    assert np.all(x2 >= 0) and np.all(x2 <= 1)
    assert p_a[:, 0] == approx(e_beam * x1)
    assert p_b[:, 0] == approx(e_beam * x2)


def test_t_channel_momentum_conservation(masses, rng, mode):
    mapping = ms.PhaseSpaceMapping(masses, CM_ENERGY, mode=mode)
    r = rng.random((BATCH_SIZE, mapping.random_dim()))
    (p_ext, x1, x2), det = mapping.map_forward([r])
    p_in = np.sum(p_ext[:, :2], axis=1)
    p_out = np.sum(p_ext[:, 2:], axis=1)

    assert p_out == approx(p_in, rel=1e-6, abs=1e-9)


def test_t_channel_inverse(masses, rng, mode):
    mapping = ms.PhaseSpaceMapping(masses, CM_ENERGY, mode=mode, invariant_power=0.3)
    r = rng.random((BATCH_SIZE, mapping.random_dim()))
    map_out, det = mapping.map_forward([r])
    (r_inv,), det_inv = mapping.map_inverse(map_out)
    one_batch = np.ones_like(det)
    assert r_inv == approx(r, abs=1e-3, rel=1e-3)
    assert det * det_inv == approx(one_batch, rel=1e-5)


@pytest.mark.parametrize(
    "particle_count",
    [2, 3, 4, 5],
    ids=["2 particles", "3 particles", "4 particles", "5 particles"],
)
@pytest.mark.parametrize(
    "energy", [10.0, 100.0, 1000.0], ids=["10GeV", "100GeV", "1TeV"]
)
def test_t_channel_phase_space_volume(particle_count, energy, rng, mode):
    if mode == ms.PhaseSpaceMapping.chili:
        return
    mapping = ms.PhaseSpaceMapping(
        [0.0] * (particle_count + 2), energy, mode=mode, leptonic=True
    )
    sample_count = 100000
    r = rng.random((sample_count, mapping.random_dim()))
    _, det = mapping.map_forward([r])
    ps_volume = (
        (2 * math.pi) ** (4 - 3 * particle_count)
        * (math.pi / 2.0) ** (particle_count - 1)
        * energy ** (2 * particle_count - 4)
        / (math.gamma(particle_count) * math.gamma(particle_count - 1))
    )
    std_error = np.std(det) / np.sqrt(sample_count)
    assert np.mean(det) == approx(ps_volume, abs=3 * std_error, rel=1e-6)
