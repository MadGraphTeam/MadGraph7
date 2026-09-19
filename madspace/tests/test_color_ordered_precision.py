"""Numerical precision of ColorOrderedMapping in long chains.

- Outgoing momenta stay on their mass shell, and still add up to the incoming
  ones by construction: they used to come out off shell by up to a few 1e-5 of
  their energy, from a frame that was not orthonormal and from soft momenta
  formed as differences of hard ones.
- The forward/inverse round trip holds with cuts: the azimuthal frame of the
  2->2 block no longer jumps with the rounding of the incoming momentum, and
  the massless branch of the block-B s23 bound no longer switches off when the
  inverse reads a massless particle's mass off its momentum.
"""

import numpy as np
import pytest

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


@pytest.mark.parametrize(
    "color_order",
    [[0, 2, 3, 4, 5, 1], [0, 2, 3, 4, 5, 6, 1], [0, 2, 3, 1, 4, 5, 6]],
    ids=["4 chain", "5 chain", "5 split"],
)
def test_massless_on_shell(color_order):
    cm_energy = 13000.0
    *_, momenta, _, det = run(color_order, cm_energy, 5)
    p = np.stack([np.asarray(k) for k in momenta], axis=1)[:, 2:]
    p = p[det > 0]
    m2 = p[..., 0] ** 2 - np.sum(p[..., 1:] ** 2, axis=-1)
    # sqrt(|m^2|) / E is ~1e-8 from rounding alone; it reached 4e-5 before
    assert np.max(np.sqrt(np.abs(m2)) / p[..., 0]) < 1e-6
    # momentum conservation to a few ulps of the total energy
    p_sum = np.sum(p, axis=1)
    p_sum[:, 0] -= cm_energy
    assert np.max(np.abs(p_sum)) < 1e-14 * cm_energy


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


@pytest.mark.parametrize(
    "masses",
    [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 173.0, 173.0, 0.0, 0.0]],
    ids=["massless", "t tbar g g"],
)
def test_lab_frame_on_shell(masses):
    """The boost to the hadronic frame keeps every momentum on its mass shell
    and momentum conserved: written as E cosh y + p_z sinh y, a particle moving
    against the boost lost a factor e^{2|y|} of relative precision, up to
    ~3e-5 in sqrt|p^2 - m^2| / E."""
    cm_energy = 13000.0
    mapping = ms.PhaseSpaceMapping(
        masses,
        cm_energy,
        mode=ms.PhaseSpaceMapping.color_ordered,
        color_order=[0, 2, 3, 4, 5, 1],
    )
    rng = np.random.default_rng(3)
    inputs = [rng.random((N, mapping.random_dim()))]
    if mapping.discrete_dim():
        inputs.append(
            rng.integers(0, 2, size=(N, mapping.discrete_dim())).astype(np.int32)
        )
    p, x1, x2, det = mapping.map_forward(inputs)
    # zero-weight points at x1 or x2 ~ 1e-13 have no meaningful momenta
    p = p[det > 1e-20]
    m2 = p[..., 0] ** 2 - np.sum(p[..., 1:] ** 2, axis=-1)
    # |p^2 - m^2| / E^2: a few 1e-15 for massless momenta, ~1e-12 for the top
    # that absorbs the rounding of the others; it reached ~1e-9 before
    off_shell = np.abs(m2 - np.square(masses)) / p[..., 0] ** 2
    assert np.max(off_shell) < 1e-11
    violation = np.abs(np.sum(p[:, 2:], axis=1) - np.sum(p[:, :2], axis=1))
    assert np.max(violation) < 1e-14 * cm_energy
