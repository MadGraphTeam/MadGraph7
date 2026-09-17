"""EnergyScale: the dynamical scale choices and the run card's scalefact.

scalefact multiplies the DYNAMICAL scale only, exactly like the LO run card's
`scalefact` (Template/LO/SubProcesses/setscales.f, "scale factor for
event-by-event scales"): a fixed scale is an absolute value and is left alone.
"""

import numpy as np
import pytest
from pytest import approx

import madspace as ms

M_TOP = 173.0


def _momenta():
    """A few p p -> t t~ events, (batch, n, 4) with (E, px, py, pz)."""
    rng = np.random.default_rng(20260917)
    batch = 32
    pt = rng.uniform(10.0, 400.0, size=(batch, 2))
    phi = rng.uniform(0.0, 2 * np.pi, size=(batch, 2))
    pz = rng.uniform(-600.0, 600.0, size=(batch, 2))
    e = np.sqrt(M_TOP**2 + pt**2 + pz**2)
    out = np.stack(
        [e, pt * np.cos(phi), pt * np.sin(phi), pz], axis=-1
    )  # (batch, 2, 4)
    e_in = out[:, :, 0].sum(axis=1) / 2
    incoming = np.zeros((batch, 2, 4))
    incoming[:, 0, 0] = e_in
    incoming[:, 0, 3] = e_in
    incoming[:, 1, 0] = e_in
    incoming[:, 1, 3] = -e_in
    return np.concatenate([incoming, out], axis=1)


def _ht(momenta):
    """sum_i sqrt(m_i^2 + pT_i^2) over the final state."""
    final = momenta[:, 2:, :]
    return np.sqrt(
        np.maximum(0.0, final[..., 0] ** 2 - final[..., 3] ** 2)
    ).sum(axis=1)


def _call(scale, momenta):
    ren, fact1, fact2 = scale(momenta)
    return np.asarray(ren), np.asarray(fact1), np.asarray(fact2)


@pytest.mark.parametrize(
    "dyn_type, expected",
    [
        (ms.EnergyScale.transverse_mass, 1.0),
        (ms.EnergyScale.half_transverse_mass, 0.5),
    ],
)
def test_dynamical_scale_definitions(dyn_type, expected):
    momenta = _momenta()
    scale = ms.EnergyScale(momenta.shape[1], dyn_type, False, False, 0.0, 0.0, 0.0)
    ren, fact1, fact2 = _call(scale, momenta)
    reference = expected * _ht(momenta)
    assert ren == approx(reference)
    assert fact1 == approx(reference)
    assert fact2 == approx(reference)


@pytest.mark.parametrize("factor", [0.5, 1.0, 2.0])
def test_scalefact_multiplies_the_dynamical_scale(factor):
    momenta = _momenta()
    scale = ms.EnergyScale(
        momenta.shape[1],
        ms.EnergyScale.half_transverse_mass,
        False,
        False,
        0.0,
        0.0,
        0.0,
        factor,
    )
    ren, fact1, fact2 = _call(scale, momenta)
    reference = factor * 0.5 * _ht(momenta)
    assert ren == approx(reference)
    assert fact1 == approx(reference)
    assert fact2 == approx(reference)


def test_scalefact_default_is_one():
    momenta = _momenta()
    args = (
        momenta.shape[1],
        ms.EnergyScale.half_transverse_mass,
        False,
        False,
        0.0,
        0.0,
        0.0,
    )
    without = _call(ms.EnergyScale(*args), momenta)
    with_one = _call(ms.EnergyScale(*args, 1.0), momenta)
    for a, b in zip(without, with_one):
        assert a == approx(b)


def test_scalefact_leaves_a_fixed_scale_alone():
    momenta = _momenta()
    fixed = 345.0
    scale = ms.EnergyScale(
        momenta.shape[1],
        ms.EnergyScale.half_transverse_mass,
        True,
        True,
        fixed,
        fixed,
        fixed,
        0.5,
    )
    ren, fact1, fact2 = _call(scale, momenta)
    assert ren == approx(np.full(momenta.shape[0], fixed))
    assert fact1 == approx(np.full(momenta.shape[0], fixed))
    assert fact2 == approx(np.full(momenta.shape[0], fixed))


def test_scalefact_with_a_fixed_renormalisation_scale_only():
    """mu_R fixed, mu_F dynamical: only mu_F picks the factor up."""
    momenta = _momenta()
    fixed = 91.188
    scale = ms.EnergyScale(
        momenta.shape[1],
        ms.EnergyScale.half_transverse_mass,
        True,
        False,
        fixed,
        0.0,
        0.0,
        0.25,
    )
    ren, fact1, fact2 = _call(scale, momenta)
    assert ren == approx(np.full(momenta.shape[0], fixed))
    assert fact1 == approx(0.25 * 0.5 * _ht(momenta))
    assert fact2 == approx(0.25 * 0.5 * _ht(momenta))
