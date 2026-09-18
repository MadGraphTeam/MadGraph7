"""The 2->3 scattering block when one of its particles is soft.

TwoToThreeParticleScattering samples s23 inside its kinematic range and weights
the point with the Byckling-Kajantie Jacobian 1 / (8 sqrt(-G4)). The block once
formed the 4x4 Gram determinant G4 (and cos(phi)) directly from invariants of
order s, cancelling terms of order s^4 down to a result proportional to the
square of the s23 range. Once a particle is soft that range is tiny, the
difference drowned in rounding, and the block returned Jacobians of up to
1e14 in place of O(1). One such point in a few million could throw a whole
ColorOrderedMapping integration off by a factor of up to 1e12.

Both are now computed from where s23 lies in its range. These tests pin that:
the Jacobian follows its smooth soft limit, and ColorOrderedMapping integrates
the massless n-body phase space to its analytic value.
"""

import math

import numpy as np
import pytest

import madspace as ms

E_BEAM = 500.0
PA = np.array([E_BEAM, 0.0, 0.0, E_BEAM])
PB = np.array([E_BEAM, 0.0, 0.0, -E_BEAM])
# deep enough into the soft limit to reach where the old Gram determinant
# broke down (about 1e-8)
DELTAS = 10.0 ** -np.arange(2, 15)


def massless(energy, theta, phi):
    return energy * np.array(
        [
            1.0,
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta),
        ]
    )


def lsquare(p):
    return p[..., 0] ** 2 - np.sum(p[..., 1:] ** 2, axis=-1)


def scatter(p3, m1, m2):
    mapping = ms.TwoToThreeParticleScattering(0.8, 0.0, 0.0, 0.8, 0.0, 0.0)
    n = len(m1)
    inputs = [
        np.zeros(n, dtype=np.int32),
        np.full(n, 0.37),
        np.full(n, 0.61),
        m1,
        m2,
    ]
    conditions = [np.tile(PA, (n, 1)), np.tile(PB, (n, 1)), p3]
    _k1, _k2, det = mapping.map_forward(inputs, conditions)
    return np.asarray(det)


def assert_smooth(values):
    """Finite values stay within a factor 2 of the first one. A NaN point has
    zero weight, which in this corner of measure ~1e-8 costs nothing."""
    finite = np.isfinite(values)
    assert np.mean(finite) > 0.5
    ratio = values[finite] / values[0]
    assert np.all((ratio > 0.5) & (ratio < 2.0)), ratio


def test_jacobian_with_soft_peeled_particle():
    """m1 -> sqrt(s12) leaves particle 2 soft. Its phase space closes
    linearly, so det / delta must tend to a constant."""
    p3 = np.tile(massless(350.0, 0.7, 0.3), (len(DELTAS), 1))
    sqrt_s12 = math.sqrt(lsquare(PA + PB - p3[0]))
    det = scatter(p3, sqrt_s12 * (1.0 - DELTAS), np.zeros(len(DELTAS)))
    assert_smooth(det / DELTAS)


def test_jacobian_with_soft_third_particle():
    """A soft third momentum (the particle peeled in the step before) leaves
    the rest of the kinematics regular, so the Jacobian must tend to a
    constant."""
    p3 = np.stack([massless(350.0 * delta, 0.7, 0.3) for delta in DELTAS])
    sqrt_s12 = np.sqrt(lsquare(PA + PB - p3))
    det = scatter(p3, 0.4 * sqrt_s12, np.zeros(len(DELTAS)))
    assert_smooth(det)


@pytest.mark.parametrize(
    "color_order",
    [[0, 2, 3, 4, 5, 1], [0, 2, 1, 3, 4, 5]],
    ids=["single chain", "1+3 split"],
)
def test_color_ordered_massless_volume(color_order):
    """The volume of massless 4-body phase space, (pi/2)^3 s^2 / 12. The
    broken Jacobian showed up only in a few points per million, so this
    needs more than a handful of batches to catch."""
    cm_energy = 1000.0
    n_out = len(color_order) - 2
    volume = (
        (math.pi / 2) ** (n_out - 1)
        * cm_energy ** (2 * (n_out - 2))
        / (math.factorial(n_out - 1) * math.factorial(n_out - 2))
    )
    mapping = ms.ColorOrderedMapping(color_order)
    n = 2_000_000
    means = []
    for seed in range(100, 108):
        rng = np.random.default_rng(seed)
        r = rng.random((n, mapping.random_dim()))
        d = rng.integers(0, 2, size=(n, mapping.discrete_dim())).astype(np.int32)
        inputs = [r[:, i].copy() for i in range(r.shape[1])]
        inputs += [d[:, j].copy() for j in range(d.shape[1])]
        conditions = [np.full(n, cm_energy)] + [np.zeros(n)] * n_out
        *momenta, det = mapping.map_forward(inputs, conditions)
        p = np.stack([np.asarray(k) for k in momenta], axis=1)
        det = np.asarray(det) * 2.0 ** mapping.discrete_dim()
        det = np.where(np.all(np.isfinite(p), axis=(1, 2)), det, 0.0)
        means.append(np.nan_to_num(det).mean())
    # The weights keep an integrable 1/sin(phi) peak at the edges of the s23
    # range, so the error estimate itself is noisy; a fixed tolerance well
    # above the spread (about 1%) is the robust check.
    assert np.mean(means) / volume == pytest.approx(1.0, abs=0.05)
