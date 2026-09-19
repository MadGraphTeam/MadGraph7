"""Numerical precision of ColorOrderedMapping in long chains.

- The forward/inverse round trip holds with cuts: the azimuthal frame of the
  2->2 block no longer jumps with the rounding of the incoming momentum, and
  the massless branch of the block-B s23 bound no longer switches off when the
  inverse reads a massless particle's mass off its momentum.
"""

import numpy as np

import madspace as ms

N = 200_000
CUTS = dict(
    pt_min=[20.0] * 4,
    m_inv_min=[[0.0 if i == j else 10.0 for j in range(4)] for i in range(4)],
    dr_min=[[0.0 if i == j else 0.4 for j in range(4)] for i in range(4)],
)


def run(color_order, cm_energy, seed, cuts=None):
    cuts = cuts or {}
    n_out = len(color_order) - 2
    mapping = ms.ColorOrderedMapping(color_order, **cuts)
    rng = np.random.default_rng(seed)
    r = rng.random((N, mapping.random_dim()))
    d = rng.integers(0, 2, size=(N, mapping.discrete_dim())).astype(np.int32)
    inputs = [r[:, i].copy() for i in range(r.shape[1])]
    inputs += [d[:, j].copy() for j in range(d.shape[1])]
    conditions = [np.full(N, cm_energy)] + [np.zeros(N)] * n_out
    *momenta, det = mapping.map_forward(inputs, conditions)
    return mapping, inputs, conditions, momenta, r, np.asarray(det)


def test_round_trip_with_cuts():
    order = [0, 2, 3, 4, 5, 1]
    mapping, inputs, conditions, momenta, r, det = run(order, 1000.0, 11, CUTS)
    out = mapping.map_inverse(momenta, conditions)
    r_inv = np.stack([np.asarray(o) for o in out[: mapping.random_dim()]], axis=1)
    det_inv = np.asarray(out[-1])
    ok = det > 0
    rt = np.abs(det * det_inv - 1.0)[ok]
    # about 3% of the points failed before, carrying ~4% of the fiducial volume
    assert np.mean(rt > 1e-2) < 1e-4
    assert np.quantile(rt, 0.999) < 1e-6
    assert np.mean(np.max(np.abs(r_inv - r), axis=1)[ok] > 1e-6) < 1e-4
