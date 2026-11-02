"""The sqrt_s_hat_min condition of PhaseSpaceMapping: a floor on sqrt(s_hat) for
each event.

For every event it acts as a cut on sqrt(s_hat) at its value would in a
mapping of its own: it raises the lower end of the s_hat the luminosity mapping
samples, and zeroes the weight below it where there is no luminosity mapping
(leptonic). So the events, momentum fractions and weights are those of the
mapping with that cut, and those of the mapping without any for a floor of 0.
The FKS configurations of a Born use it to give the events of final-state
emitters the floor the cuts imply, and those of incoming emitters none, in one
mapping.
"""

import numpy as np
import pytest
import torch
from madspace_testing import to_numpy

import madspace as ms

SQRT_S = 2000.0
BATCH = 3000
FLOORS = [0.0, 300.0, 700.0]
CASES = {
    "jets": ([0.0] * 5, ms.PhaseSpaceMapping.rambo),
    "jets-propagator": ([0.0] * 5, ms.PhaseSpaceMapping.propagator),
    "tops": ([0.0, 0.0, 173.0, 173.0], ms.PhaseSpaceMapping.rambo),
}


def floor_cut(masses, value):
    O = ms.Observable
    pids = [21] * len(masses)
    return ms.Cuts([ms.CutItem(O(pids, O.obs_sqrt_s, []), min=value)])


def mapping(masses, mode, floor=None, condition=False, leptonic=False):
    return ms.PhaseSpaceMapping(
        masses,
        SQRT_S,
        leptonic=leptonic,
        mode=mode,
        cuts=None if not floor else floor_cut(masses, floor),
        sqrt_s_hat_min_condition=condition,
    )


def as_torch(array):
    return torch.as_tensor(array).to(torch.get_default_device())


def forward(m, r, conditions=()):
    return [to_numpy(t) for t in m.map_forward([as_torch(r)], list(conditions))]


@pytest.mark.parametrize("case", CASES)
def test_events_of_the_floor_cut(case):
    masses, mode = CASES[case]
    with_condition = mapping(masses, mode, condition=True)
    assert with_condition.sqrt_s_hat_min_condition()
    rng = np.random.default_rng(5)
    # away from the edges of the unit cube for the inverse, as in
    # test_fks_mapping.py
    r = 0.02 + 0.96 * rng.random((BATCH, with_condition.random_dim()))
    choice = rng.integers(len(FLOORS), size=BATCH)
    floor = np.array(FLOORS)[choice]
    p, x1, x2, det = forward(with_condition, r, [as_torch(floor)])
    for k, value in enumerate(FLOORS):
        # the reference on the whole batch, so that every event is computed in
        # the same SIMD lane, compared where the floor is its own
        part = choice == k
        reference = forward(mapping(masses, mode, floor=value), r)
        for ours, theirs in zip([p, x1, x2, det], reference):
            np.testing.assert_array_equal(ours[part], theirs[part])
        q = p[part, 0] + p[part, 1]
        sqrt_s_hat = np.sqrt(q[:, 0] ** 2 - (q[:, 1:] ** 2).sum(-1))
        assert (sqrt_s_hat[det[part] > 0] >= value * (1 - 1e-12)).all()

    # the inverse takes the same condition
    r_inv, det_inv = with_condition.map_inverse(
        [as_torch(p), as_torch(x1), as_torch(x2)], [as_torch(floor)]
    )
    # to the conditioning of the mapping without the condition (1e-9 for the
    # propagator mode)
    ok = det > 0
    assert ok.mean() > 0.9
    np.testing.assert_allclose(to_numpy(r_inv)[ok], r[ok], rtol=0, atol=1e-8)
    np.testing.assert_allclose(det[ok] * to_numpy(det_inv)[ok], 1.0, rtol=1e-8)


def test_leptonic_floor_is_a_cut():
    """Without a luminosity mapping s_hat is s_lab: a floor below it changes
    nothing, one above it zeroes every weight, as the cut would."""
    masses = [0.0] * 5
    mode = ms.PhaseSpaceMapping.rambo
    with_condition = mapping(masses, mode, condition=True, leptonic=True)
    r = np.random.default_rng(6).random((BATCH, with_condition.random_dim()))
    floor = np.where(np.arange(BATCH) % 2 == 0, 0.5 * SQRT_S, 1.5 * SQRT_S)
    p, x1, x2, det = forward(with_condition, r, [as_torch(floor)])
    p_ref, _, _, det_ref = forward(mapping(masses, mode, leptonic=True), r)
    below = floor < SQRT_S
    np.testing.assert_array_equal(p, p_ref)
    np.testing.assert_array_equal(det[below], det_ref[below])
    np.testing.assert_array_equal(det[~below], 0.0)


@pytest.mark.parametrize("case", CASES)
def test_floor_above_the_beam_energy(case):
    """A floor at or above sqrt(s_lab) leaves nothing to sample: a finite zero
    weight, as a cut on sqrt(s_hat) there gives, and the other events as
    without it."""
    masses, mode = CASES[case]
    with_condition = mapping(masses, mode, condition=True)
    r = 0.02 + 0.96 * np.random.default_rng(7).random(
        (BATCH, with_condition.random_dim())
    )
    above = np.arange(BATCH) % 3 == 0
    floor = np.where(above, np.where(np.arange(BATCH) % 2 == 0, SQRT_S, 2 * SQRT_S), 0)
    p, x1, x2, det = forward(with_condition, r, [as_torch(floor)])
    reference = forward(mapping(masses, mode), r)
    assert np.isfinite(det).all()
    np.testing.assert_array_equal(det[above], 0.0)
    for ours, theirs in zip([p, x1, x2, det], reference):
        np.testing.assert_array_equal(ours[~above], theirs[~above])
