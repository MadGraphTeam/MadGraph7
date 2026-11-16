"""Beams of elastic photons from protons at NLO (lpp 2 of the run card): the
equivalent-photon density of a proton that stays intact
(ElasticPhotonDensity), the luminosity of the FKS integrand with it
(FKSLuminosity with photon_beams, and fks_luminosities for lpp 2 and -2),
and the beams the NLO drivers take.

The density must be epa_proton of PhotonFlux.f of MadGraph (the improved
Weizsaecker-Williams formula of Budnev et al. with the dipole form factors
of the proton), here transcribed from the Fortran, at momentum fractions
from 1e-6 to close to 1 and scales from 1 to 1e4 GeV, and zero where the
virtuality of the photon has no range. The Fortran writes its constants
(alpha, the proton mass, the parameters of the form factors) in single
precision, which moves its density by up to a few 1e-7, the transcription
with them as gfortran rounds them.
"""

import math

import numpy as np
import pytest
from madgraph.mg7.nlo.core import check_run_card
from madgraph.mg7.nlo.grids import write_lhapdf_grid
from madspace_testing import default_context

import madspace as ms
from madspace.nlo.fks import fks_luminosities


def epa_proton(x, q2max, single=False):
    """epa_proton and phi_f of PhotonFlux.f, line by line, with its constants
    rounded to single precision as gfortran does with single."""
    rounded = np.float32 if single else float
    alpha, xin, qz = (float(rounded(v)) for v in (0.0072992701, 0.938, 0.71))

    def phi_f(x, qq):
        a, b, c = (float(rounded(v)) for v in (7.16, -3.96, 0.028))
        qq1 = 1 + qq
        y = x * x / (1 - x)
        f = (1 + a * y) * (
            -math.log(qq1 / qq) + 1 / qq1 + 1 / (2 * qq1 * qq1) + 1 / (3 * qq1**3)
        )
        f = f + (1 - b) * y / (4 * qq * qq1**3)
        f = f + c * (1 + y / 4) * (
            math.log((qq1 - b) / qq1)
            + b / qq1
            + b * b / (2 * qq1 * qq1)
            + b**3 / (3 * qq1**3)
        )
        return f

    if x >= 1:
        return 0.0
    qmi = xin * xin * x * x / (1 - x)
    if qmi >= q2max:
        return 0.0
    f = alpha / math.pi * (phi_f(x, q2max / qz) - phi_f(x, qmi / qz)) * (1 - x) / x
    return max(f, 0.0)


def call(function, *args, context=None):
    runtime = ms.FunctionRuntime(function.function(), context or default_context())
    tensors = [np.ascontiguousarray(a) for a in args]
    outputs = [np.array(np.from_dlpack(t)) for t in runtime.call(tensors)]
    return dict(zip(function.return_types().keys(), outputs))


def points(count=500, seed=1):
    """Momentum fractions from 1e-6 up and close to 1, and scales from 1 to
    1e4 GeV."""
    rng = np.random.default_rng(seed)
    half = count // 2
    x = np.concatenate(
        [10.0 ** rng.uniform(-6, 0, half), 1 - 10.0 ** rng.uniform(-6, -0.5, half)]
    )
    q = 10.0 ** rng.uniform(0, 4, 2 * half)
    return x, q


def test_density():
    """x f_gamma of ElasticPhotonDensity is x epa_proton(x, q^2), zero where
    m_p^2 x^2 / (1 - x) is above q^2. Close to that bound the two terms of
    epa_proton cancel, which the kernel takes with log1p and the Fortran with
    log, so there they agree to 1e-12 of the largest density."""
    x, q = points()
    ours = call(ms.ElasticPhotonDensity(), x, q)["pdf"]
    theirs = np.array([xi * epa_proton(xi, qi * qi) for xi, qi in zip(x, q)])
    empty = 0.938**2 * x * x / (1 - x) >= q * q
    assert empty.any() and not empty.all()
    np.testing.assert_array_equal(ours[empty], 0.0)
    atol = 1e-12 * np.abs(theirs).max()
    np.testing.assert_allclose(ours, theirs, rtol=1e-11, atol=atol)
    fortran = np.array(
        [xi * epa_proton(xi, qi * qi, single=True) for xi, qi in zip(x, q)]
    )
    np.testing.assert_allclose(ours, fortran, rtol=1e-6, atol=atol)


def test_fks_luminosity():
    """FKSLuminosity of beams of elastic photons: the product of the
    densities of the photons for a flavor of two photons, with its factor,
    none for the flavors with another parton."""
    x, q = points(count=100, seed=2)
    x1, x2 = x, x[::-1].copy()
    q1, q2 = q, q[::-1].copy()
    density = ms.ElasticPhotonDensity()
    f1 = call(density, x1, q1)["pdf"] / x1
    f2 = call(density, x2, q2)["pdf"] / x2
    flavors = [[[22, 22], [22, 1]], [[1, 22], [22, 22]]]
    factors = [[2.0, 1.0], [1.0, 0.5]]
    luminosity = ms.FKSLuminosity(
        flavors, sum_flavors=True, factors=factors, photon_beams=[True, True]
    )
    assert not luminosity.has_pdf()
    for configuration, factor in ((0, 2.0), (1, 0.5)):
        index = np.full(len(x), configuration, dtype=np.int32)
        out = call(luminosity, x1, x2, q1, q2, index)["luminosity"]
        np.testing.assert_allclose(out, factor * f1 * f2, rtol=1e-15)


def test_fks_luminosities():
    """fks_luminosities of lpp 2 and -2: the photons of both beams, for the
    real emission and the n-body part."""
    configurations = {
        "nincoming": 2,
        "born_pdgs": [22, 22, 6, -6],
        "born_flavors": [[22, 22, 6, -6]],
        "real_flavors": [[[22, 22, 6, -6, 21]]],
        "configurations": [
            dict(i_fks=5, j_fks=3, real_me=1, splitting_type=["QCD"]),
        ],
    }
    real, born = fks_luminosities(configurations, beams=(2, -2), sum_flavors=True)
    x, q = points(count=50, seed=3)
    density = ms.ElasticPhotonDensity()
    f = call(density, x, q)["pdf"] / x
    index = np.zeros(len(x), dtype=np.int32)
    for luminosity in (real, born):
        out = call(luminosity, x, x, q, q, index)["luminosity"]
        np.testing.assert_allclose(out, f * f, rtol=1e-15)


def test_fks_luminosity_invalid(tmp_path):
    """A beam of elastic photons takes no densities of a grid."""
    path = str(tmp_path / "NNPDF23nlo_as_0119_qed_0000.dat")
    write_lhapdf_grid(path)
    grid_less = dict(photon_beams=[True, False], densities=[True, True])
    with pytest.raises(ValueError, match="elastic photons"):
        ms.FKSLuminosity([[[22, 1]]], ms.PdfGrid(path), **grid_less)


def test_run_card():
    """The NLO drivers take elastic photons on both beams, not those of
    gamma-UPC, and no photons with hadrons."""
    base = dict(ickkw=0, dynamical_scale_choice=-1, pdfscheme=0, pdlabel="nn23nlo")
    for lpp in ((2, 2), (2, -2), (-2, -2)):
        check_run_card({**base, "lpp1": lpp[0], "lpp2": lpp[1]})
    for lpp in ((2, 1), (1, 2), (2, 0), (2, 3)):
        with pytest.raises(NotImplementedError):
            check_run_card({**base, "lpp1": lpp[0], "lpp2": lpp[1]})
    with pytest.raises(NotImplementedError, match="gamma-UPC"):
        check_run_card({**base, "lpp1": 2, "lpp2": 2, "pdlabel": "edff"})
