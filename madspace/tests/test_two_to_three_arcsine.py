"""The arcsine map of the s23 random number in the 2->3 scattering block.

On its kinematic range s23 is linear in cos(phi), and the Byckling-Kajantie
Jacobian is 1 / (sqrt(lambda) (s23_max - s23_min) |sin(phi)|): the measure is
flat in phi. Sampling s23 with any density that is finite at the kinematic
limits leaves an integrable 1/|sin(phi)| in the weight, whose variance diverges
logarithmically. With arcsine_s23 the random number is first mapped flat in
phi / 2 and only then handed to the s23 importance sampling. These tests pin
that
- with flat s23 sampling the block weight becomes exactly independent of r_s23,
- without the map the edge peak is still there,
- forward and inverse agree, with and without cuts,
- ColorOrderedMapping integrates to the same volume either way, with cuts.
"""

import math

import numpy as np
import pytest

import madspace as ms

E_BEAM = 500.0
PA = np.array([E_BEAM, 0.0, 0.0, E_BEAM])
PB = np.array([E_BEAM, 0.0, 0.0, -E_BEAM])
# r_s23 from the middle out to the edges; below ~1e-5 the point sits within
# double precision of the edge of the s23 range and u = (s23 - s23_min) /
# (s23_max - s23_min) can no longer be resolved from s23.
EDGE = 10.0 ** -np.arange(1, 6)
R_S23 = np.concatenate([EDGE, [0.5], 1.0 - EDGE])


def massless(energy, theta, phi):
    return energy * np.array(
        [
            1.0,
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta),
        ]
    )


def block_weights(r_s23, arcsine, s_power=0.0, m1=40.0, m2=0.0):
    mapping = ms.TwoToThreeParticleScattering(
        0.8, 0.0, 0.0, s_power, 0.0, 0.0, arcsine_s23=arcsine
    )
    n = len(r_s23)
    inputs = [
        np.zeros(n, dtype=np.int32),
        np.asarray(r_s23, dtype=np.float64),
        np.full(n, 0.61),
        np.full(n, m1),
        np.full(n, m2),
    ]
    conditions = [
        np.tile(PA, (n, 1)),
        np.tile(PB, (n, 1)),
        np.tile(massless(350.0, 0.7, 0.3), (n, 1)),
    ]
    p1, p2, det = mapping.map_forward(inputs, conditions)
    return mapping, inputs, conditions, np.asarray(det)


def test_flat_weight_is_constant():
    """Flat s23 plus the arcsine map is flat in phi, which is exactly the
    measure: at fixed t1 the weight must not depend on r_s23 at all."""
    *_, det = block_weights(R_S23, arcsine=True)
    assert np.all(np.isfinite(det)) and np.all(det > 0)
    assert det / det[len(EDGE)] == pytest.approx(1.0, rel=1e-5)


@pytest.mark.parametrize("s_power", [0.8, 1.0])
def test_weight_bounded_with_importance_sampling(s_power):
    """With 1/s23^nu importance sampling the weight is no longer constant,
    but it must stay finite and bounded out to the edges of the range."""
    *_, det = block_weights(R_S23, arcsine=True, s_power=s_power)
    assert np.all(np.isfinite(det)) and np.all(det > 0)
    edges = np.concatenate([det[: len(EDGE)], det[len(EDGE) + 1 :]])
    # converged to a finite limit: the last two decades agree
    assert det[len(EDGE) - 1] == pytest.approx(det[len(EDGE) - 2], rel=1e-3)
    assert det[-1] == pytest.approx(det[-2], rel=1e-3)
    assert np.max(edges) / np.min(edges) < 1e3


def test_edge_peak_without_arcsine():
    """Without the map the weight keeps its 1/|sin(phi)| = 1/(2 sqrt(u (1 - u)))
    peak, with u = r_s23 for flat sampling."""
    *_, det = block_weights(R_S23, arcsine=False)
    ratio = det * np.sqrt(R_S23 * (1.0 - R_S23))
    assert ratio == pytest.approx(ratio[len(EDGE)], rel=1e-5)
    assert det[len(EDGE) - 1] / det[len(EDGE)] > 100.0


@pytest.mark.parametrize("arcsine", [True, False], ids=["arcsine", "flat"])
@pytest.mark.parametrize("s_power", [0.0, 0.8, 1.0])
def test_round_trip(arcsine, s_power):
    rng = np.random.default_rng(7)
    r_s23 = np.concatenate([rng.random(10_000), R_S23])
    mapping, inputs, conditions, det = block_weights(
        r_s23, arcsine=arcsine, s_power=s_power
    )
    p1, p2, _ = mapping.map_forward(inputs, conditions)
    *inv_inputs, det_inv = mapping.map_inverse([p1, p2], conditions)
    # the random number of s23 is conditioned like s23 itself: an absolute
    # error of 1e-16 s / (s23_max - s23_min) in u
    assert np.asarray(inv_inputs[1]) == pytest.approx(r_s23, abs=1e-8)
    assert np.asarray(inv_inputs[2]) == pytest.approx(inputs[2], abs=1e-8)
    rt = np.abs(det * np.asarray(det_inv) - 1.0)
    assert np.quantile(rt, 0.999) < 1e-6


CUTS = dict(
    pt_min=[20.0] * 4,
    m_inv_min=[[0.0 if i == j else 10.0 for j in range(4)] for i in range(4)],
    dr_min=[[0.0 if i == j else 0.4 for j in range(4)] for i in range(4)],
)


def sample(color_order, arcsine, seed, n, cuts=None):
    """Weights of ColorOrderedMapping for the massless n-body phase space at
    1 TeV, zero outside the cuts if given."""
    cuts = cuts or {}
    cm_energy = 1000.0
    n_out = len(color_order) - 2
    mapping = ms.ColorOrderedMapping(color_order, arcsine_s23=arcsine, **cuts)
    rng = np.random.default_rng(seed)
    r = rng.random((n, mapping.random_dim()))
    d = rng.integers(0, 2, size=(n, mapping.discrete_dim())).astype(np.int32)
    inputs = [r[:, i].copy() for i in range(r.shape[1])]
    inputs += [d[:, j].copy() for j in range(d.shape[1])]
    conditions = [np.full(n, cm_energy)] + [np.zeros(n)] * n_out
    *momenta, det = mapping.map_forward(inputs, conditions)
    # momenta of the two beams, then of the outgoing particles
    p = np.stack([np.asarray(k) for k in momenta], axis=1)[:, 2:]
    det = np.asarray(det) * 2.0 ** mapping.discrete_dim()
    ok = np.all(np.isfinite(p), axis=(1, 2)) & np.isfinite(det)
    if cuts:
        pt = np.hypot(p[..., 1], p[..., 2])
        ok &= np.all(pt > cuts["pt_min"][0], axis=1)
        eta = np.arcsinh(p[..., 3] / np.maximum(pt, 1e-300))
        phi = np.arctan2(p[..., 2], p[..., 1])
        for i in range(n_out):
            for j in range(i + 1, n_out):
                q = p[:, i] + p[:, j]
                m2 = q[:, 0] ** 2 - np.sum(q[:, 1:] ** 2, axis=1)
                dphi = np.angle(np.exp(1j * (phi[:, i] - phi[:, j])))
                dr = np.hypot(eta[:, i] - eta[:, j], dphi)
                ok &= m2 > cuts["m_inv_min"][i][j] ** 2
                ok &= dr > cuts["dr_min"][i][j]
    return np.where(ok, det, 0.0)


@pytest.mark.parametrize(
    "color_order",
    [[0, 2, 3, 4, 5, 1], [0, 2, 1, 3, 4, 5]],
    ids=["single chain", "1+3 split"],
)
def test_color_ordered_massless_volume(color_order):
    """The analytic massless 4-body volume, (pi/2)^3 s^2 / 12, with the map."""
    volume = (math.pi / 2) ** 3 * 1000.0**4 / 12
    means = [
        sample(color_order, True, seed, 1_000_000).mean() for seed in range(200, 204)
    ]
    assert np.mean(means) / volume == pytest.approx(1.0, abs=0.02)


def test_color_ordered_cut_volume_unchanged():
    """With cuts the sampled s23 range is a part of the kinematic one and the
    map only covers that part. The fiducial volume must not change."""
    order = [0, 2, 3, 4, 5, 1]
    n = 1_000_000
    on = np.concatenate([sample(order, True, s, n, CUTS) for s in range(300, 304)])
    off = np.concatenate([sample(order, False, s, n, CUTS) for s in range(310, 314)])
    err = math.hypot(on.std() / math.sqrt(len(on)), off.std() / math.sqrt(len(off)))
    assert abs(on.mean() - off.mean()) < 4.0 * err
