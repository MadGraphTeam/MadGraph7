"""Phase-space mappings for a decay, i.e. a single incoming particle.

A 1 -> n topology has no t-channel chain and a root virtuality fixed by the
decaying particle's mass, so it reuses the s-channel cascade the 2 -> n mappings
already build. These tests pin that down against results that are known
analytically:

  * the total phase-space volume,
  * the external masses and momentum conservation,
  * the decaying particle at rest, p_in = (M, 0, 0, 0),
  * an exact forward/inverse round trip.
"""

import math

import numpy as np
import pytest
from pytest import approx

import madspace as ms

BATCH_SIZE = 100000
M_TOP = 173.0
M_B = 4.7
M_W = 80.379
W_W = 2.085

rng = np.random.default_rng(20260811)


def massless_volume(n, mass):
    """Volume of the n-body massless phase space of a particle of mass ``mass``.

    Phi_n = (2 pi)^(4-3n) (pi/2)^(n-1) M^(2n-4) / ((n-1)! (n-2)!)
    """
    return (
        (2 * math.pi) ** (4 - 3 * n)
        * (math.pi / 2) ** (n - 1)
        * mass ** (2 * n - 4)
        / (math.factorial(n - 1) * math.factorial(n - 2))
    )


def two_body_volume(mass, m1, m2):
    """Phi_2 = (1/8pi) * sqrt(lambda(M^2, m1^2, m2^2)) / M^2."""
    a, b, c = mass**2, m1**2, m2**2
    lam = a * a + b * b + c * c - 2 * (a * b + a * c + b * c)
    return math.sqrt(lam) / (8 * math.pi * a)


def decay_mapping(outgoing_masses, propagators, vertices, mass=M_TOP):
    diagram = ms.Diagram([mass], outgoing_masses, propagators, vertices)
    return ms.PhaseSpaceMapping(ms.Topology(diagram), mass)


def sample(mapping):
    r = rng.random((BATCH_SIZE, mapping.random_dim()))
    p_ext, x1, x2, det = mapping.map_forward([r], [])
    return r, p_ext, x1, x2, det


def volume(det):
    """Monte-Carlo estimate of the volume and its standard error."""
    return np.mean(det), np.std(det) / math.sqrt(len(det))


# --------------------------------------------------------------------------
# Topology
# --------------------------------------------------------------------------

def test_single_incoming_has_no_t_channel():
    diagram = ms.Diagram([M_TOP], [0.0, 0.0], [], [["i0", "o0", "o1"]])
    topology = ms.Topology(diagram)
    assert topology.t_propagator_count == 0
    assert topology.incoming_masses == [M_TOP]
    assert len(topology.outgoing_masses) == 2


def test_three_or_more_incoming_rejected():
    with pytest.raises(ValueError):
        ms.Diagram([1.0, 1.0, 1.0], [0.0, 0.0], [], [["i0", "o0", "o1"]])


# --------------------------------------------------------------------------
# Kinematics
# --------------------------------------------------------------------------

def test_decaying_particle_is_at_rest():
    mapping = decay_mapping([0.0, 0.0], [], [["i0", "o0", "o1"]])
    _, p_ext, _, _, _ = sample(mapping)
    # One incoming + two outgoing, not the 2 + n of a collision.
    assert p_ext.shape == (BATCH_SIZE, 3, 4)
    assert p_ext[:, 0, 0] == approx(M_TOP)
    assert p_ext[:, 0, 1:] == approx(0.0, abs=1e-10)


def test_momentum_conservation_and_masses():
    mapping = decay_mapping(
        [M_B, 0.0, 0.0],
        [ms.Propagator(mass=M_W, width=W_W, pdg_id=24)],
        [["i0", "o0", "p0"], ["p0", "o1", "o2"]],
    )
    _, p_ext, _, _, _ = sample(mapping)
    assert p_ext.shape == (BATCH_SIZE, 4, 4)
    assert np.sum(p_ext[:, 1:], axis=1) == approx(p_ext[:, 0], abs=1e-8)

    def mass(p):
        m2 = p[:, 0] ** 2 - np.sum(p[:, 1:] ** 2, axis=1)
        return np.sqrt(np.abs(m2))

    assert mass(p_ext[:, 0]) == approx(M_TOP, rel=1e-10)
    assert mass(p_ext[:, 1]) == approx(M_B, rel=1e-6)
    assert mass(p_ext[:, 2]) == approx(0.0, abs=1e-5)
    assert mass(p_ext[:, 3]) == approx(0.0, abs=1e-5)


def test_no_beam_momentum_fractions():
    mapping = decay_mapping([0.0, 0.0], [], [["i0", "o0", "o1"]])
    _, _, x1, x2, _ = sample(mapping)
    assert x1 == approx(1.0)
    assert x2 == approx(1.0)


def test_random_dim_is_3n_minus_4():
    for n in (2, 3):
        mapping = decay_mapping(
            [0.0] * n,
            [] if n == 2 else [ms.Propagator(mass=M_W, width=W_W, pdg_id=24)],
            [["i0", "o0", "o1"]]
            if n == 2
            else [["i0", "o0", "p0"], ["p0", "o1", "o2"]],
        )
        assert mapping.random_dim() == 3 * n - 4
        assert mapping.particle_count() == n + 1


# --------------------------------------------------------------------------
# Phase-space volume
# --------------------------------------------------------------------------

def test_two_body_massless_volume():
    mapping = decay_mapping([0.0, 0.0], [], [["i0", "o0", "o1"]])
    _, _, _, _, det = sample(mapping)
    mean, err = volume(det)
    expected = massless_volume(2, M_TOP)
    # Two-body is sampled exactly: every weight is the same number.
    assert err == approx(0.0, abs=1e-12)
    assert mean == approx(expected, rel=1e-10)


def test_two_body_massive_volume():
    mapping = decay_mapping([M_B, M_W], [], [["i0", "o0", "o1"]])
    _, _, _, _, det = sample(mapping)
    mean, _ = volume(det)
    assert mean == approx(two_body_volume(M_TOP, M_B, M_W), rel=1e-10)


@pytest.mark.parametrize(
    "prop_mass,prop_width,rel_tol",
    [
        # A broad propagator makes the Breit-Wigner sampling nearly flat, so the
        # estimator has little variance and this is a sharp check of the volume.
        (80.0, 60.0, 3e-3),
        # The physical W is a narrow resonance: importance sampling it costs
        # variance on a flat integrand, so only a loose check is possible here.
        (M_W, W_W, 3e-2),
    ],
    ids=["broad", "narrow-W"],
)
def test_three_body_massless_volume(prop_mass, prop_width, rel_tol):
    """t -> b f f' through a Breit-Wigner propagator. The sampled invariant
    covers the full range, so whatever the propagator's width, the integral is
    the plain massless three-body volume."""
    mapping = decay_mapping(
        [0.0, 0.0, 0.0],
        [ms.Propagator(mass=prop_mass, width=prop_width, pdg_id=24)],
        [["i0", "o0", "p0"], ["p0", "o1", "o2"]],
    )
    _, _, _, _, det = sample(mapping)
    mean, _ = volume(det)
    assert mean == approx(massless_volume(3, M_TOP), rel=rel_tol)


# --------------------------------------------------------------------------
# Invertibility
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "outgoing,propagators,vertices",
    [
        ([0.0, 0.0], [], [["i0", "o0", "o1"]]),
        ([M_B, M_W], [], [["i0", "o0", "o1"]]),
        (
            [M_B, 0.0, 0.0],
            [ms.Propagator(mass=M_W, width=W_W, pdg_id=24)],
            [["i0", "o0", "p0"], ["p0", "o1", "o2"]],
        ),
    ],
    ids=["2body-massless", "2body-massive", "3body"],
)
def test_forward_inverse_round_trip(outgoing, propagators, vertices):
    mapping = decay_mapping(outgoing, propagators, vertices)
    r, p_ext, x1, x2, det = sample(mapping)
    r_back, det_back = mapping.map_inverse([p_ext, x1, x2], [])
    assert r_back == approx(r, abs=1e-8)
    assert det * det_back == approx(1.0, rel=1e-8)
