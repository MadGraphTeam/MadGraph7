"""The built-in PDF sets of MadFKS in the NLO drivers (madgraph/mg7/nlo/grids.py):
the NNPDF 2.3 sets on the nodes of their grids, the CTEQ6 and CT14qed sets as
tables of fdist of pdf.f, which pdf_table.f of the MG7 templates writes with
the libraries of an aMC@NLO output, and alpha_s(m_Z) and the loops of its
running from pdfwrap.f.

At random momentum fractions and scales within the range of their tables,
PartonDensity of the tables is fdist itself (pdf_table.f at those points)
within the cubic interpolation of madspace, that of the NNPDF grids within
the difference of the interpolations of madspace and of NNPDFDriver.f (see
test_fks_couplings.py). The tables need the libraries of a built output with
hadron beams, those of MADSPACE_FKS_OUTPUTS (pp_epem), and are skipped
otherwise.
"""

import math
import os

import numpy as np
import pytest
from fks_testing import evaluate
from madgraph.mg7.nlo import grids
from madgraph.mg7.nlo.core import Couplings, check_run_card
from madspace_testing import default_context

import madspace as ms

OUTPUTS = os.environ.get("MADSPACE_FKS_OUTPUTS")
OUTPUT = os.path.join(OUTPUTS or "", "pp_epem")
needs_output = pytest.mark.skipif(
    not os.path.isfile(os.path.join(OUTPUT, "lib", "libpdf.a")),
    reason="MADSPACE_FKS_OUTPUTS names no built output pp_epem",
)
# the cubic interpolation of madspace on the nodes of the tables, and its
# difference to the polynomial one of NNPDFDriver.f on the NNPDF grids (3e-3
# at large x and low scales)
TABLE_RTOL = 5e-4
NNPDF_RTOL = 5e-3
GLOBALS = ("pdf_logx", "pdf_logq2", "pdf_coefficients")


def points(pdlabel, count=400, seed=5):
    """Momentum fractions from the lower end of the table to 0.9 and scales
    within its range."""
    xmin, qmin, qmax = grids.TABLE_RANGES.get(pdlabel[:5], (1e-6, 2.0, 1e4))
    rng = np.random.default_rng(seed)
    half = count // 2
    x = np.concatenate(
        [
            10 ** rng.uniform(math.log10(xmin) + 0.1, -1, half),
            rng.uniform(0.1, 0.9, count - half),
        ]
    )
    q = 10 ** rng.uniform(math.log10(qmin) + 0.05, math.log10(qmax) - 0.05, count)
    return x, q


def fdist(pdlabel, directory, x, q):
    """x f of fdist at each point (x[k], q[k]), (count, flavors of PIDS),
    the diagonal of its table on the nodes x and q."""
    table = grids.tabulate(pdlabel, OUTPUT, directory, list(x), list(q))
    return table[np.arange(len(x)), np.arange(len(x))]


@needs_output
@pytest.mark.parametrize(
    "pdlabel",
    [
        "nn23lo",
        "nn23lo1",
        "cteq6_m",
        "cteq6_d",
        "cteq6_l",
        "cteq6l1",
        "ct14q00",
        "ct14q21",
    ],
)
def test_density(pdlabel, tmp_path):
    """PartonDensity of the grid of a built-in set is fdist of MadFKS within
    the interpolation, for every flavor, relative to the largest density at
    the point."""
    path = str(tmp_path / f"{pdlabel}_0000.dat")
    grids.write_builtin_grid(path, pdlabel, OUTPUT, str(tmp_path))
    pdf = ms.PdfGrid(path)
    context = default_context()
    for name in GLOBALS:
        if context.global_exists(name):
            context.delete_global(name)
    pdf.initialize_globals(context)
    x, q = points(pdlabel, count=60)
    (ours,) = evaluate(ms.PartonDensity(pdf, grids.PIDS), x, q)
    theirs = fdist(pdlabel, str(tmp_path), x, q)
    scale = np.abs(theirs).max(axis=1, keepdims=True)
    rtol = NNPDF_RTOL if pdlabel in grids.NNPDF_GRIDS else TABLE_RTOL
    assert (np.abs(ours - theirs) <= rtol * scale).all(), np.abs(
        (ours - theirs) / scale
    ).max()
    if pdlabel.startswith("ct14q"):
        # the photon of CT14qed
        assert (theirs[:, grids.PIDS.index(22)] > 0).all()


def test_alpha_s_one_loop():
    """The 1-loop running of alfas_functions.f: a / (1 + a b0 t) from
    alpha_s(m_Z) through the b and c thresholds, as for cteq6l1."""
    b0 = {3: 0.716197243913527, 4: 0.66314559621623, 5: 0.61009394851893}

    def run(a, t, nf):
        return a / (1 + a * b0[nf] * t)

    asmz = 0.130
    amb = run(asmz, 2 * math.log(4.7 / 91.188), 5)
    amc = run(amb, 2 * math.log(1.42 / 4.7), 4)
    for q, expected in (
        (1000.0, run(asmz, 2 * math.log(1000 / 91.188), 5)),
        (3.0, run(amb, 2 * math.log(3 / 4.7), 4)),
        (1.2, run(amc, 2 * math.log(1.2 / 1.42), 3)),
    ):
        assert grids.alpha_s_2loop(asmz, q, loops=1) == pytest.approx(
            expected, rel=1e-15
        )
    assert grids.alpha_s_2loop(0.118, 91.188) == pytest.approx(0.118, rel=1e-15)


@needs_output
def test_couplings(tmp_path):
    """Couplings of a run with a built-in set take its grid and the running
    of alpha_s of pdfwrap.f, one loop for cteq6l1, and refuse the sets
    pdfwrap.f has no grid for here."""
    directory = os.path.join(OUTPUT, "SubProcesses", "madspace_pdlabels")
    run_card = dict(lpp1=1, lpp2=1, pdlabel="cteq6l1", lhaid=0)
    context = default_context()
    couplings = Couplings(directory, run_card, {}, context)
    assert couplings.asmz == 0.130
    for q in (5.0, 91.188, 500.0):
        (ours,) = evaluate(couplings.running, np.array([q]))
        expected = grids.alpha_s_2loop(0.130, q, loops=1)
        assert ours[0] == pytest.approx(expected, rel=1e-6)
    with pytest.raises(NotImplementedError, match="pdlabel mrs02nl"):
        Couplings(directory, dict(run_card, pdlabel="mrs02nl"), {}, context)


def test_user_scales_in_run_card():
    """The NLO drivers take the scale of the user (dynamical_scale_choice 0
    and 10)."""
    base = dict(ickkw=0, pdfscheme=0, lpp1=1, lpp2=1, pdlabel="nn23nlo")
    for choice in (0, 10, -1):
        check_run_card({**base, "dynamical_scale_choice": choice})
