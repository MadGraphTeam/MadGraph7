"""LeptonLuminosity: the momentum fractions of lepton beams.

Each fraction is drawn from its own endpoint, 1 - x = (1 - x_min) r^(1 / (1 -
p)), with x_min = tau_min for the first beam and tau_min / x1 for the second.
With a reduced density the Jacobian leaves the factors (1 - x)^p to the
density, without it the Jacobian is the full one, so that its mean over the
random numbers is the area x1 x2 >= tau_min of the unit square. The lower
bound on s_hat can come per event from a condition, clipped to [0, s_lab],
and gives the mapping of a fixed bound to the last digit. x is x_min to the
last digit at the end point of the sampling, also for a small x_min, the
inverse holds as x1 approaches tau_min, an empty range of x has a zero
Jacobian, a point outside the range has a zero Jacobian of the inverse, and
1 - x is kept a normal double.
LeptonLuminosityInvariant takes the same options.
"""

import math

import numpy as np
import pytest
import torch
from madspace_testing import to_numpy

import madspace as ms

S_LAB = 1.0e6
POWER = 0.96
N = 200000


def forward(mapping, r1, r2, s_hat_min=None):
    conditions = [] if s_hat_min is None else [torch.as_tensor(s_hat_min)]
    return [
        to_numpy(out)
        for out in mapping.map_forward(
            [torch.as_tensor(r1), torch.as_tensor(r2)], conditions
        )
    ]


def inverse(mapping, outputs, s_hat_min=None):
    conditions = [] if s_hat_min is None else [torch.as_tensor(s_hat_min)]
    return [
        to_numpy(out)
        for out in mapping.map_inverse(
            [torch.as_tensor(o) for o in outputs], conditions
        )
    ]


def random_numbers(n, seed):
    rng = np.random.default_rng(seed)
    return rng.random(n), rng.random(n)


@pytest.mark.parametrize("reduced", [True, False])
def test_fractions_and_jacobian(reduced):
    """x and 1 - x of both beams, s_hat, and the Jacobian per beam: (1 -
    x_min)^(1 - p) / (1 - p), times (1 - x)^p without a reduced density."""
    tau_min = 0.04
    mapping = ms.LeptonLuminosity(
        S_LAB, tau_min * S_LAB, POWER, reduced_density=reduced
    )
    r1, r2 = random_numbers(1000, 1)
    x1, omx1, x2, omx2, s_hat, det = forward(mapping, r1, r2)
    a = 1.0 / (1.0 - POWER)
    span1 = 1.0 - tau_min
    span2 = 1.0 - tau_min / x1
    np.testing.assert_allclose(omx1, span1 * r1**a, rtol=1e-14)
    np.testing.assert_allclose(omx2, span2 * r2**a, rtol=1e-14)
    np.testing.assert_allclose(x1, 1.0 - omx1, rtol=1e-14)
    np.testing.assert_allclose(x2, 1.0 - omx2, rtol=1e-14)
    np.testing.assert_allclose(s_hat, x1 * x2 * S_LAB, rtol=1e-15)
    assert np.all(x1 * x2 >= tau_min * (1 - 1e-14))
    expected = span1 ** (1 - POWER) * span2 ** (1 - POWER) * a * a
    if not reduced:
        expected = expected * (omx1 * omx2) ** POWER
    np.testing.assert_allclose(det, expected, rtol=1e-13)


def test_full_jacobian_integrates_the_area():
    """Without a reduced density the mean of the Jacobian is the area of x1 x2
    >= tau_min in the unit square, 1 - tau_min + tau_min log(tau_min)."""
    tau_min = 0.3
    mapping = ms.LeptonLuminosity(S_LAB, tau_min * S_LAB, POWER, reduced_density=False)
    r1, r2 = random_numbers(N, 2)
    det = forward(mapping, r1, r2)[-1]
    area = 1 - tau_min + tau_min * math.log(tau_min)
    assert abs(det.mean() - area) < 5 * det.std() / math.sqrt(N)


def test_s_hat_min_condition():
    """The lower bound per event gives the mapping with that bound, to the
    last digit, also the inverse. Both run on the same batch, so that every
    event lands in the same SIMD lane, or in the scalar tail of the batch, in
    both."""
    r1, r2 = random_numbers(1000, 3)
    tau = np.where(np.arange(1000) % 2 == 0, 0.01, 0.25)
    per_event = ms.LeptonLuminosity(
        S_LAB, 0.0, POWER, reduced_density=False, s_hat_min_condition=True
    )
    outputs = forward(per_event, r1, r2, tau * S_LAB)
    back = inverse(per_event, outputs[:-1], tau * S_LAB)
    for value in (0.01, 0.25):
        rows = tau == value
        fixed = ms.LeptonLuminosity(S_LAB, value * S_LAB, POWER, reduced_density=False)
        expected = forward(fixed, r1, r2)
        for got, want in zip(outputs, expected):
            np.testing.assert_array_equal(got[rows], want[rows])
        expected = inverse(fixed, outputs[:-1])
        for got, want in zip(back, expected):
            np.testing.assert_array_equal(got[rows], want[rows])


def test_s_hat_min_condition_out_of_range():
    """A bound below 0 is none, a bound above s_lab leaves no range: x1 = x2 =
    1, s_hat = s_lab and a zero Jacobian, all finite."""
    r1, r2 = random_numbers(100, 5)
    mapping = ms.LeptonLuminosity(
        S_LAB, 0.0, POWER, reduced_density=False, s_hat_min_condition=True
    )
    below = forward(mapping, r1, r2, np.full(100, -0.3 * S_LAB))
    zero = forward(mapping, r1, r2, np.zeros(100))
    for got, want in zip(below, zero):
        np.testing.assert_array_equal(got, want)
    x1, omx1, x2, omx2, s_hat, det = forward(mapping, r1, r2, np.full(100, 2 * S_LAB))
    assert np.all(x1 == 1.0) and np.all(x2 == 1.0) and np.all(s_hat == S_LAB)
    assert np.all(omx1 > 0.0) and np.all(omx2 > 0.0)
    assert np.all(det == 0.0)


def test_round_trip_close_to_tau_min():
    """As x1 approaches tau_min the second beam has a narrow range, whose
    width the forward and the inverse map build from x1 in the same way, so
    that the inverse still gives back r2 and the reciprocal Jacobian."""
    tau_min = 0.04
    mapping = ms.LeptonLuminosity(S_LAB, tau_min * S_LAB, POWER, reduced_density=False)
    r1 = 1.0 - np.array([1e-6, 1e-9, 1e-12, 1e-14])
    r2 = np.array([0.1, 0.5, 0.9, 0.5])
    *outputs, det = forward(mapping, r1, r2)
    assert np.all(det > 0.0)
    r1_back, r2_back, det_back = inverse(mapping, outputs)
    np.testing.assert_allclose(r2_back, r2, rtol=1e-14)
    np.testing.assert_allclose(det * det_back, 1.0, rtol=1e-14)


@pytest.mark.parametrize("reduced", [True, False])
@pytest.mark.parametrize("condition", [False, True])
def test_round_trip(reduced, condition):
    """The inverse gives back the random numbers, with the reciprocal
    Jacobian."""
    tau_min = 0.04
    r1, r2 = random_numbers(1000, 4)
    s_hat_min = np.full(1000, tau_min * S_LAB) if condition else None
    mapping = ms.LeptonLuminosity(
        S_LAB,
        0.0 if condition else tau_min * S_LAB,
        POWER,
        reduced_density=reduced,
        s_hat_min_condition=condition,
    )
    *outputs, det = forward(mapping, r1, r2, s_hat_min)
    r1_back, r2_back, det_back = inverse(mapping, outputs, s_hat_min)
    np.testing.assert_allclose(r1_back, r1, rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(r2_back, r2, rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(det * det_back, 1.0, rtol=1e-12)


def test_precision_at_x_min():
    """x is x_min + (1 - x_min) (1 - r^a), so that at r = 1 it is x_min to the
    last digit, also for a small x_min, where 1 - (1 - x_min) r^a would keep
    only the digits of 1 - x_min."""
    tau_min = 1e-10
    mapping = ms.LeptonLuminosity(S_LAB, tau_min * S_LAB, POWER)
    x1, _, x2, _, s_hat, _ = forward(mapping, np.ones(3), np.ones(3))
    assert np.all(x1 == tau_min)
    assert np.all(x2 == tau_min / tau_min)
    x1, _, x2, _, s_hat, _ = forward(mapping, np.full(3, 0.5), np.ones(3))
    assert np.all(x2 == tau_min / x1)


@pytest.mark.parametrize("reduced", [True, False])
def test_empty_range(reduced):
    """At r1 = 1, x1 = tau_min leaves x2 no range: x2 is 1 and the Jacobian
    is zero, also that of the inverse."""
    tau_min = 0.04
    mapping = ms.LeptonLuminosity(
        S_LAB, tau_min * S_LAB, POWER, reduced_density=reduced
    )
    *outputs, det = forward(mapping, np.ones(3), np.array([0.2, 0.7, 1.0]))
    x1, _, x2, _, _ = outputs
    assert np.all(x1 == tau_min)
    assert np.all(x2 == 1.0)
    assert np.all(det == 0.0)
    det_inverse = inverse(mapping, outputs)[-1]
    assert np.all(det_inverse == 0.0)


@pytest.mark.parametrize("reduced", [True, False])
def test_inverse_outside_the_range(reduced):
    """A point with x1 < tau_min, from another channel, leaves x2 no range:
    r2 = 0 and a zero Jacobian, not NaN. The same for x1 = 0 at tau_min = 0,
    where the width of the second beam is 0 / 0."""
    tau_min = 0.04
    mapping = ms.LeptonLuminosity(
        S_LAB, tau_min * S_LAB, POWER, reduced_density=reduced
    )
    x1 = np.array([0.01, 0.03, tau_min * (1 - 1e-15)])
    x2 = np.full(3, 0.5)
    outputs = [x1, 1 - x1, x2, 1 - x2, x1 * x2 * S_LAB]
    r1, r2, det = inverse(mapping, outputs)
    assert np.all(np.isfinite(r1))
    assert np.all(r2 == 0.0)
    assert np.all(det == 0.0)
    mapping = ms.LeptonLuminosity(S_LAB, 0.0, POWER, reduced_density=reduced)
    outputs = [np.zeros(2), np.ones(2), np.full(2, 0.5), np.full(2, 0.5), np.zeros(2)]
    r1, r2, det = inverse(mapping, outputs)
    assert np.all(r2 == 0.0)
    assert np.all(det == 0.0)


def test_one_minus_x_stays_normal():
    """1 - x is at least the smallest normal double, also where r^a
    underflows."""
    mapping = ms.LeptonLuminosity(S_LAB, 0.0, POWER, reduced_density=False)
    r = np.array([1e-300, 1e-20, 0.0])
    _, omx1, x2, omx2, _, det = forward(mapping, r, r)
    assert np.all(omx1 >= np.finfo(float).tiny)
    assert np.all(omx2 >= np.finfo(float).tiny)
    assert np.all(np.isfinite(det))


def test_invalid_parameters():
    for bad in [
        dict(s_lab=0.0),
        dict(s_hat_min=-1.0),
        dict(s_hat_min=2 * S_LAB),
        dict(sampling_power=1.0),
        dict(sampling_power=-0.1),
    ]:
        with pytest.raises(ValueError, match="LeptonLuminosity"):
            ms.LeptonLuminosity(**{"s_lab": S_LAB, "s_hat_min": 0.0, **bad})


def test_accessors():
    mapping = ms.LeptonLuminosity(
        S_LAB, 100.0, 0.975, reduced_density=False, s_hat_min_condition=True
    )
    assert mapping.s_lab() == S_LAB
    assert mapping.s_hat_min() == 100.0
    assert mapping.sampling_power() == 0.975
    assert not mapping.reduced_density()
    assert mapping.s_hat_min_condition()
    assert ms.LeptonLuminosity(S_LAB, 0.0).reduced_density()


def invariant_forward(mapping, n, seed, s_hat_min=None):
    rng = np.random.default_rng(seed)
    r = [torch.as_tensor(rng.random(n)) for _ in range(3)]
    conditions = [] if s_hat_min is None else [torch.as_tensor(s_hat_min)]
    return [to_numpy(out) for out in mapping.map_forward(r, conditions)]


@pytest.mark.parametrize("invariant_power", [0.0, 1.0])
def test_invariant_full_density(invariant_power):
    """LeptonLuminosityInvariant without a reduced density: the full Jacobian
    is the reduced one times (1 - x1)^p (1 - x2)^p, as for LeptonLuminosity,
    so that the two channels can be mixed for a full density."""
    args = (S_LAB, 0.01 * S_LAB, S_LAB, POWER, 0.0, 0.0, invariant_power)
    reduced = invariant_forward(ms.LeptonLuminosityInvariant(*args), 1000, 6)
    full = invariant_forward(
        ms.LeptonLuminosityInvariant(*args, reduced_density=False), 1000, 6
    )
    for got, want in zip(full[:-1], reduced[:-1]):
        np.testing.assert_array_equal(got, want)
    _, omx1, _, omx2, _, det = reduced
    np.testing.assert_allclose(full[-1], det * (omx1 * omx2) ** POWER, rtol=1e-13)


def test_invariant_s_hat_min_condition():
    """The lower bound per event gives the mapping with that bound, to the
    last digit."""
    tau = np.where(np.arange(1000) % 2 == 0, 0.01, 0.25)
    per_event = ms.LeptonLuminosityInvariant(
        S_LAB, 0.0, reduced_density=False, s_hat_min_condition=True
    )
    outputs = invariant_forward(per_event, 1000, 7, tau * S_LAB)
    for value in (0.01, 0.25):
        rows = tau == value
        fixed = ms.LeptonLuminosityInvariant(
            S_LAB, value * S_LAB, reduced_density=False
        )
        expected = invariant_forward(fixed, 1000, 7)
        for got, want in zip(outputs, expected):
            np.testing.assert_array_equal(got[rows], want[rows])


def test_invariant_accessors():
    mapping = ms.LeptonLuminosityInvariant(
        S_LAB, 100.0, reduced_density=False, s_hat_min_condition=True
    )
    assert not mapping.reduced_density()
    assert mapping.s_hat_min_condition()
    assert ms.LeptonLuminosityInvariant(S_LAB, 0.0).reduced_density()
