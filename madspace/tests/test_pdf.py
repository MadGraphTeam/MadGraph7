import math
import os

import pytest
import torch
from madspace_testing import default_context, to_numpy

import madspace as ms

PDF_SET = "NNPDF40_nlo_as_01180"

try:
    import lhapdf

    lhapdf.setVerbosity(0)
    reference_pdf = lhapdf.mkPDF(PDF_SET, 0)
except ImportError:
    lhapdf = None
    reference_pdf = None
except RuntimeError:
    reference_pdf = None

pytestmark = [
    pytest.mark.skipif(lhapdf is None, reason="lhapdf required to run this test"),
    pytest.mark.skipif(
        reference_pdf is None, reason=f"pdf set {PDF_SET} required to run this test"
    ),
]


def test_pdf():
    ctx = default_context()
    grid = ms.PdfGrid(os.path.join(lhapdf.paths()[0], PDF_SET, f"{PDF_SET}_0000.dat"))
    grid.initialize_globals(ctx)
    pids = [-5, -4, -3, -2, -1, 21, 1, 2, 3, 4, 5]
    pdf = ms.PartonDensity(grid, pids)

    xs = torch.logspace(
        math.log10(reference_pdf.xMin) + 1e-6, math.log10(reference_pdf.xMax), 100
    )
    q2s = torch.logspace(
        math.log10(reference_pdf.q2Min) + 1e-6, math.log10(reference_pdf.q2Max), 100
    )
    x_grid, q2_grid = torch.meshgrid(xs, q2s, indexing="xy")
    x, q2 = x_grid.flatten(), q2_grid.flatten()

    result = pdf(x, torch.sqrt(q2))
    reference = torch.tensor(
        reference_pdf.xfxQ2(pids, to_numpy(x).tolist(), to_numpy(q2).tolist())
    )
    torch.testing.assert_close(result, reference, rtol=1e-6, atol=1e-12)


def test_alpha_s():
    ctx = default_context()
    grid = ms.AlphaSGrid(os.path.join(lhapdf.paths()[0], PDF_SET, f"{PDF_SET}.info"))
    grid.initialize_globals(ctx)
    alpha_s = ms.RunningCoupling(grid)

    q2 = torch.logspace(
        math.log10(reference_pdf.q2Min) + 1e-6, math.log10(reference_pdf.q2Max), 1000
    )

    result = alpha_s(torch.sqrt(q2))
    reference = torch.tensor(
        [reference_pdf.alphasQ2(q2_item) for q2_item in to_numpy(q2).tolist()]
    )
    torch.testing.assert_close(result, reference, rtol=1e-6, atol=1e-12)


def test_pdf_outside_grid():
    """Outside of the grid the densities continue as those of LHAPDF (its
    continuation extrapolator): below the first x knot, below the first and
    above the last q knot, and both at once."""
    ctx = default_context()
    grid = ms.PdfGrid(os.path.join(lhapdf.paths()[0], PDF_SET, f"{PDF_SET}_0000.dat"))
    grid.initialize_globals(ctx, "outside")
    try:
        pids = [-5, -4, -3, -2, -1, 21, 1, 2, 3, 4, 5]
        pdf = ms.PartonDensity(grid, pids, False, "outside")
        x_min, q2_min, q2_max = (
            reference_pdf.xMin,
            reference_pdf.q2Min,
            reference_pdf.q2Max,
        )
        xs = torch.cat(
            [
                torch.logspace(math.log10(x_min) - 3, math.log10(x_min), 5)[:-1],
                torch.logspace(math.log10(x_min), -0.01, 20),
            ]
        )
        q2s = torch.cat(
            [
                torch.logspace(-1, math.log10(q2_min), 6)[:-1],
                torch.logspace(math.log10(q2_min), math.log10(q2_max), 5),
                torch.logspace(math.log10(q2_max), math.log10(q2_max) + 2, 4)[1:],
            ]
        )
        x_grid, q2_grid = torch.meshgrid(xs, q2s, indexing="xy")
        x, q2 = x_grid.flatten(), q2_grid.flatten()
        result = pdf(x, torch.sqrt(q2))
        reference = torch.tensor(
            reference_pdf.xfxQ2(pids, to_numpy(x).tolist(), to_numpy(q2).tolist())
        )
        torch.testing.assert_close(result, reference, rtol=1e-6, atol=1e-12)
    finally:
        for name in ("pdf_logx", "pdf_logq2", "pdf_coefficients"):
            ctx.delete_global(f"outside.{name}")


def test_alpha_s_outside_grid():
    """Below the first and above the last knot alpha_s is that of LHAPDF."""
    ctx = default_context()
    grid = ms.AlphaSGrid(os.path.join(lhapdf.paths()[0], PDF_SET, f"{PDF_SET}.info"))
    grid.initialize_globals(ctx, "outside")
    try:
        alpha_s = ms.RunningCoupling(grid, "outside")
        q2_min, q2_max = reference_pdf.q2Min, reference_pdf.q2Max
        q2 = torch.cat(
            [
                torch.logspace(-1, math.log10(q2_min), 20)[:-1],
                torch.logspace(math.log10(q2_max), math.log10(q2_max) + 2, 5)[1:],
            ]
        )
        result = alpha_s(torch.sqrt(q2))
        reference = torch.tensor(
            [reference_pdf.alphasQ2(q2_item) for q2_item in to_numpy(q2).tolist()]
        )
        torch.testing.assert_close(result, reference, rtol=1e-6, atol=1e-12)
    finally:
        for name in ("alpha_s_logq2", "alpha_s_coefficients"):
            ctx.delete_global(f"outside.{name}")
