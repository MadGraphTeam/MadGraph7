"""The 2->3 scattering block when one of its particles is soft.

TwoToThreeParticleScattering samples s23 inside its kinematic range and weights
the point with the Byckling-Kajantie Jacobian 1 / (8 sqrt(-G4)). The block once
formed the 4x4 Gram determinant G4 (and cos(phi)) directly from invariants of
order s, cancelling terms of order s^4 down to a result proportional to the
square of the s23 range. Once a particle is soft that range is tiny, the
difference drowned in rounding, and the block returned Jacobians of up to
1e14 in place of O(1). One such point in a few million could throw a whole
ColorOrderedMapping integration off by a factor of up to 1e12.

Both are now computed from where s23 lies in its range, and that range from
momenta in the p12 rest frame: its Byckling-Kajantie form through 3x3 Gram
determinants cancelled the same way one level down, and gave NaN or zero
Jacobians below a softness of about 1e-8. These tests pin that:
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
    """Every value is finite and within a factor 2 of the first one. (The s23
    sampling moves the limit by a few percent once s23 drops below the 1e-2
    GeV^2 offset of its power map, so this is not a tighter check.)"""
    assert np.all(np.isfinite(values)), values
    ratio = values / values[0]
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


def extreme_points():
    """Soft and collinear limits down to 1e-16, at the corners of the random
    numbers: a soft peeled particle, a soft or vanishing p3, p3 along or
    against pa, a vanishing m1, and s12 -> 0."""
    deltas = [1e-8, 1e-12, 1e-15, 1e-16]
    corners = [0.0, 1e-16, 1e-8, 0.37, 1.0 - 1e-8, 1.0]
    points = []
    for delta in deltas:
        configs = [
            (massless(350.0, 0.7, 0.3), lambda s: math.sqrt(s) * (1.0 - delta)),
            (massless(350.0 * delta, 0.7, 0.3), lambda s: 0.4 * math.sqrt(s)),
            (massless(350.0, delta, 0.3), lambda s: 0.4 * math.sqrt(s)),
            (massless(350.0, math.pi - delta, 0.3), lambda s: 0.4 * math.sqrt(s)),
            (massless(350.0, 0.7, 0.3), lambda s: delta * math.sqrt(s)),
            (massless(500.0 * (1.0 - delta), 0.7, 0.3), lambda s: 0.0),
        ]
        for p3, m1 in configs:
            for r_s23 in corners:
                for r_t1 in corners:
                    for index in (0, 1):
                        points.append((p3, m1(lsquare(PA + PB - p3)), r_s23, r_t1, index))
    return points


@pytest.mark.parametrize("arcsine", [True, False], ids=["arcsine", "flat"])
@pytest.mark.parametrize("s_power", [0.0, 0.8])
def test_finite_at_extreme_points(arcsine, s_power):
    """Forward and inverse stay finite in all these limits. Where p1 or p3 is
    along pa the azimuth is undefined and the weight is zero; there the s23
    range has zero width, and the inverse used to divide 0 / 0. A p3 below
    ~1e-12 GeV (or a p_12 that close to rest) also hit absolute floors in the
    normalisation of the frame and the boost axis and came out far off shell."""
    points = extreme_points()
    n = len(points)
    mapping = ms.TwoToThreeParticleScattering(
        0.8, 0.0, 0.0, s_power, 0.0, 0.0, arcsine_s23=arcsine
    )
    inputs = [
        np.array([pt[4] for pt in points], dtype=np.int32),
        np.array([pt[2] for pt in points]),
        np.array([pt[3] for pt in points]),
        np.array([pt[1] for pt in points]),
        np.zeros(n),
    ]
    conditions = [np.tile(PA, (n, 1)), np.tile(PB, (n, 1)), np.stack([pt[0] for pt in points])]
    p1, p2, det = mapping.map_forward(inputs, conditions)
    p1, p2, det = np.asarray(p1), np.asarray(p2), np.asarray(det)
    assert np.all(np.isfinite(p1)) and np.all(np.isfinite(p2))
    assert np.all(np.isfinite(det)) and np.all(det >= 0)
    # On shell where the weight is not zero: the soft-p3 points used to be off
    # by ~1e4 GeV^2, |p^2 - m^2| / E^2 ~ 1e-2. What is left comes from s12 -> 0,
    # a pair collinear to ~1e-6 whose mass pa + pb - p3 fixes only to ~1e-4,
    # and stays below ~1e-9. (At zero weight the pair mass can be below
    # sqrt(eps) of its energy, which momenta cannot carry at all.)
    live = det > 0
    m1 = inputs[3][live]
    off1 = np.abs(lsquare(p1[live]) - m1**2) / p1[live, 0] ** 2
    off2 = np.abs(lsquare(p2[live])) / p2[live, 0] ** 2
    assert np.max(off1) < 1e-8 and np.max(off2) < 1e-8
    for out in mapping.map_inverse([p1, p2], conditions):
        assert np.all(np.isfinite(np.asarray(out)))


def test_arcsine_rejects_breit_wigner():
    with pytest.raises(ValueError):
        ms.TwoToThreeParticleScattering(0.8, 0.0, 0.0, 0.8, 80.0, 2.0)
    ms.TwoToThreeParticleScattering(0.8, 0.0, 0.0, 0.8, 80.0, 2.0, arcsine_s23=False)
