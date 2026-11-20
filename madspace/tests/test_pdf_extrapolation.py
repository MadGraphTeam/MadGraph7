"""PartonDensity and RunningCoupling outside of their grids against LHAPDF.

Outside of the grid the densities continue as with the continuation
extrapolator of LHAPDF and alpha_s as the interpolated alpha_s of LHAPDF
(AlphaS_Ipol). The tests write a small set in the LHAPDF format (write_set:
two subgrids in q, a flavor that vanishes below the threshold between them,
one that stays below 1e-3 and one that changes sign) and compare at points in
and around the grid with the values LHAPDF gives for it, stored in
test_data/pdf/extrapolation.json. With LHAPDF installed they also compare with
LHAPDF directly, and

    python madspace/tests/test_pdf_extrapolation.py

rewrites the reference.
"""

import json
import math
import os
import tempfile

import numpy as np
import pytest
import torch
from madspace_testing import default_context

import madspace as ms

SET_NAME = "MadSpaceExtrapolationTest"
# in a folder of its own, since test_processes.py reads every JSON file
# directly in test_data as a list of diagrams
REFERENCE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "test_data", "pdf", "extrapolation.json"
)
# the globals of the set in the default context, under their own prefix
PREFIX = "extrapolation_test"
PIDS = [-1, 1, 2, 3, 4, 21]
# x below the first knot (1e-7), in the grid and at its edges; q below the
# first knot (1.3), at the threshold between the subgrids (4.75), in the grid
# and above its last knot (1e5)
XS = [1e-10, 3e-9, 5e-8, 1e-7, 2e-7, 1e-4, 0.03, 0.3, 0.75, 1.0]
QS = [0.3, 0.8, 1.2, 1.3, 1.31, 2.5, 4.75, 30.0, 9e4, 1e5, 2e5, 1e6]
# (at the threshold 4.75 itself LHAPDF takes alpha_s above it and madspace the
# one below it)
ALPHA_S_QS = [0.2, 0.5, 1.0, 1.29, 1.3, 2.0, 4.7, 4.8, 91.1876, 1e5, 1.5e5, 1e7]
# the extrapolation of log(x f) at x = 1e-10 differs by 2e-13, and the
# densities vanish at x = 1 to 6e-20 in LHAPDF
RTOL = 1e-12
ATOL = 1e-15

try:
    import lhapdf

    lhapdf.setVerbosity(0)
except ImportError:
    lhapdf = None


def x_knots():
    small = np.geomspace(1e-7, 0.1, 25)
    return np.concatenate([small, np.linspace(0.1, 1.0, 16)[1:]])


def q_subgrids():
    return [
        np.array([1.3, 1.5, 1.8, 2.2, 2.7, 3.3, 4.0, 4.75]),
        np.array([4.75, 6.0, 8.0, 12.0, 20.0, 40.0, 100.0, 300.0, 1e3, 5e3, 2e4, 1e5]),
    ]


def xf(pid, x, q):
    """x f of the test set: smooth in log x and log q, zero at x = 1."""
    log_q2 = math.log(q * q / 1.69)
    if pid == 21:
        return 3.0 * x**-0.2 * (1 - x) ** 5 * (1 + 0.1 * log_q2)
    if pid == 2:
        return (
            0.6 * x**0.5 * (1 - x) ** 3 * (1 + 0.05 * log_q2)
            + 0.1 * x**-0.1 * (1 - x) ** 7
        )
    if pid == 1:
        return (
            0.3 * x**0.6 * (1 - x) ** 4 * (1 + 0.04 * log_q2)
            + 0.1 * x**-0.1 * (1 - x) ** 7
        )
    if pid == -1:
        return 0.1 * x**-0.1 * (1 - x) ** 7 * (1 + 0.08 * log_q2)
    if pid == 3:
        # below 1e-3 everywhere, extrapolated in x f rather than log(x f)
        return 5e-4 * x**0.2 * (1 - x) ** 6 * (1 + 0.02 * log_q2)
    if pid == 4:
        # zero below the threshold of the second subgrid, changing sign at
        # small x above it
        if q < 4.75:
            return 0.0
        return 0.02 * (x**0.1 - 0.2) * (1 - x) ** 5 * math.log(q / 4.7)
    raise ValueError(pid)


def alpha_s_knots():
    qs = [1.3, 1.5, 2.0, 3.0, 4.75, 4.75, 10.0, 50.0, 91.1876, 500.0, 1e4, 1e5]
    values = [
        0.42,
        0.37,
        0.31,
        0.26,
        0.22,
        0.221,
        0.18,
        0.135,
        0.118,
        0.095,
        0.072,
        0.06,
    ]
    return qs, values


def write_set(directory):
    """The test set in LHAPDF format in directory/SET_NAME, its path."""
    path = os.path.join(directory, SET_NAME)
    os.makedirs(path, exist_ok=True)
    xs = x_knots()
    qs, values = alpha_s_knots()
    with open(os.path.join(path, f"{SET_NAME}.info"), "w") as f:
        f.write(
            "SetDesc: madspace extrapolation test\n"
            "Format: lhagrid1\n"
            "DataVersion: 1\n"
            "NumMembers: 1\n"
            "Particle: 2212\n"
            f"Flavors: [{', '.join(str(pid) for pid in PIDS)}]\n"
            "OrderQCD: 1\n"
            "ErrorType: replicas\n"
            f"XMin: {xs[0]:.17e}\nXMax: 1.0\nQMin: 1.3\nQMax: 1e5\n"
            "AlphaS_Type: ipol\n"
            f"AlphaS_Qs: [{', '.join(repr(q) for q in qs)}]\n"
            f"AlphaS_Vals: [{', '.join(repr(v) for v in values)}]\n"
        )
    with open(os.path.join(path, f"{SET_NAME}_0000.dat"), "w") as f:
        f.write("PdfType: central\nFormat: lhagrid1\n---\n")
        for q_knots in q_subgrids():
            f.write(" ".join(f"{x:.17e}" for x in xs) + "\n")
            f.write(" ".join(f"{q:.17e}" for q in q_knots) + "\n")
            f.write(" ".join(str(pid) for pid in PIDS) + "\n")
            for x in xs:
                for q in q_knots:
                    f.write(" ".join(f"{xf(pid, x, q):.17e}" for pid in PIDS) + "\n")
            f.write("---\n")
    return path


def points():
    x, q = np.meshgrid(XS, QS, indexing="ij")
    return x.flatten(), q.flatten()


def madspace_values(path):
    context = default_context()
    pdf_grid = ms.PdfGrid(os.path.join(path, f"{SET_NAME}_0000.dat"))
    alpha_s_grid = ms.AlphaSGrid(os.path.join(path, f"{SET_NAME}.info"))
    names = [
        f"{PREFIX}.{name}"
        for name in (
            "pdf_logx",
            "pdf_logq2",
            "pdf_coefficients",
            "alpha_s_logq2",
            "alpha_s_coefficients",
        )
    ]
    pdf_grid.initialize_globals(context, PREFIX)
    alpha_s_grid.initialize_globals(context, PREFIX)
    try:
        x, q = points()
        pdf = ms.PartonDensity(pdf_grid, PIDS, False, PREFIX)
        densities = pdf(
            torch.tensor(x, dtype=torch.float64), torch.tensor(q, dtype=torch.float64)
        )
        alpha_s = ms.RunningCoupling(alpha_s_grid, PREFIX)(
            torch.tensor(ALPHA_S_QS, dtype=torch.float64)
        )
        return densities.cpu().numpy(), alpha_s.cpu().numpy()
    finally:
        for name in names:
            if context.global_exists(name):
                context.delete_global(name)


def lhapdf_values(directory):
    """The densities and alpha_s of LHAPDF for the set in directory."""
    lhapdf.pathsPrepend(directory)
    pdf = lhapdf.mkPDF(SET_NAME, 0)
    x, q = points()
    densities = [[pdf.xfxQ(pid, xi, qi) for pid in PIDS] for xi, qi in zip(x, q)]
    alpha_s = [pdf.alphasQ(qi) for qi in ALPHA_S_QS]
    return np.array(densities), np.array(alpha_s)


@pytest.fixture(scope="module")
def test_set():
    with tempfile.TemporaryDirectory() as directory:
        yield directory, write_set(directory)


def assert_close(ours, theirs):
    excess = np.abs(ours - theirs) - RTOL * np.abs(theirs) - ATOL
    worst = np.unravel_index(np.argmax(excess), ours.shape)
    assert np.all(excess <= 0), (
        worst,
        ours[worst],
        theirs[worst],
    )


def test_against_reference(test_set):
    """The densities and alpha_s in and around the grid are those LHAPDF gave
    for the test set (the reference)."""
    _, path = test_set
    with open(REFERENCE) as f:
        reference = json.load(f)
    densities, alpha_s = madspace_values(path)
    assert_close(densities, np.array(reference["xf"]))
    assert_close(alpha_s, np.array(reference["alpha_s"]))


@pytest.mark.skipif(lhapdf is None, reason="lhapdf required to run this test")
def test_against_lhapdf(test_set):
    """The same with LHAPDF itself."""
    directory, path = test_set
    densities, alpha_s = madspace_values(path)
    theirs_densities, theirs_alpha_s = lhapdf_values(directory)
    assert_close(densities, theirs_densities)
    assert_close(alpha_s, theirs_alpha_s)


def test_above_the_last_x_knot(test_set):
    """Above the last x knot, where LHAPDF stops with an error, the densities
    are zero."""
    _, path = test_set
    context = default_context()
    grid = ms.PdfGrid(os.path.join(path, f"{SET_NAME}_0000.dat"))
    grid.initialize_globals(context, PREFIX)
    try:
        x = torch.tensor([1.5, 1.5, 1.5], dtype=torch.float64)
        q = torch.tensor([0.5, 10.0, 1e6], dtype=torch.float64)
        result = ms.PartonDensity(grid, PIDS, False, PREFIX)(x, q)
        assert np.all(result.cpu().numpy() == 0.0)
    finally:
        for name in ("pdf_logx", "pdf_logq2", "pdf_coefficients"):
            if context.global_exists(f"{PREFIX}.{name}"):
                context.delete_global(f"{PREFIX}.{name}")


if __name__ == "__main__":
    if lhapdf is None:
        raise SystemExit("the reference needs LHAPDF")
    with tempfile.TemporaryDirectory() as directory:
        write_set(directory)
        densities, alpha_s = lhapdf_values(directory)
    with open(REFERENCE, "w") as f:
        json.dump(
            dict(
                lhapdf=lhapdf.version(),
                pids=PIDS,
                x=XS,
                q=QS,
                alpha_s_q=ALPHA_S_QS,
                xf=densities.tolist(),
                alpha_s=alpha_s.tolist(),
            ),
            f,
            indent=1,
        )
        f.write("\n")
    print(f"wrote {REFERENCE}")
