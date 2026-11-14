"""LeptonLuminosity: the momentum fractions of lepton beams.

Each fraction is drawn from its own endpoint, 1 - x = (1 - x_min) r^(1 / (1 -
p)), with x_min = tau_min for the first beam and tau_min / x1 for the second.
With a reduced density the Jacobian leaves the factors (1 - x)^p to the
density, without it the Jacobian is the full one, so that its mean over the
random numbers is the area x1 x2 >= tau_min of the unit square. The lower
bound on s_hat can come per event from a condition. x is x_min to the last
digit at the end point of the sampling, also for a small x_min, an empty
range of x has a zero Jacobian, and 1 - x is kept a normal double.
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
    """The lower bound per event gives the mapping with that bound."""
    r1, r2 = random_numbers(1000, 3)
    tau = np.where(np.arange(1000) % 2 == 0, 0.01, 0.25)
    per_event = ms.LeptonLuminosity(
        S_LAB, 0.0, POWER, reduced_density=False, s_hat_min_condition=True
    )
    outputs = forward(per_event, r1, r2, tau * S_LAB)
    for value in (0.01, 0.25):
        rows = tau == value
        fixed = ms.LeptonLuminosity(S_LAB, value * S_LAB, POWER, reduced_density=False)
        expected = forward(fixed, r1[rows], r2[rows])
        for got, want in zip(outputs, expected):
            np.testing.assert_allclose(got[rows], want, rtol=1e-15)


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
