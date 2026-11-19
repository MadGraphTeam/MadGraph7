# SPDX-License-Identifier: 0BSD
"""Graded colour correlations: SU(N_c) identities hold degree by degree.

With every U(1)-subtraction piece graded, each correlation <T_i . T_j> is a
Laurent polynomial in N_c, so the colour identities hold for every power of
N_c separately (all-outgoing charges, ``docs/correlator-conventions.md``):

* colour conservation: sum_j <T_i . T_j>^(d) = 0 for every leg i and degree d;
* Casimirs: <T_i . T_i> = C_i B with C_A = N_c and C_F = N_c/2 - 1/(2 N_c),
  B the graded Born, so with every term evaluated at N_c = 3:
  <T_i . T_i>^(d) = 3 B^(d-1) for a gluon and
  (3/2) B^(d-1) - (1/6) B^(d+1) for a quark.

The inherited LC/NLC matrices satisfy these only through the retained order;
the strict truncations satisfy them exactly, because every dipole of one
process shares the reference degree P + 1.
"""

from __future__ import annotations

from decimal import Decimal, localcontext

import pytest

from pyamplicol import (
    ColorCorrelator,
    CorrelatorConfig,
    Generator,
    ModelSource,
    Runtime,
)
from pyamplicol.config import (
    ColorConfig,
    EvaluatorConfig,
    EvaluatorOptimizationConfig,
    GenerationConfig,
    GenerationValidationConfig,
    JITConfig,
    ProcessConfig,
    RunConfig,
)

_POINT = (
    (500, 0, 0, 500),
    (500, 0, 0, -500),
    (500, 300, 0, 400),
    (500, -300, 0, -400),
)
_QUARK_LEGS = {"u d > u d": (1, 2, 3, 4), "u u~ > g g": (1, 2), "g g > g g": ()}


def _configuration(accuracy: str, *, graded: bool) -> RunConfig:
    return RunConfig(
        action="generate",
        process=ProcessConfig(
            coupling_order_policy="explicit",
            max_coupling_orders={"QCD": 2, "QED": 0},
        ),
        color=ColorConfig(accuracy=accuracy, graded=graded),
        generation=GenerationConfig(
            workers=1,
            emit_api_bundle=False,
            validation=GenerationValidationConfig(
                enabled=False, post_build_validation=False
            ),
        ),
        evaluator=EvaluatorConfig(
            execution_mode="compiled",
            optimization=EvaluatorOptimizationConfig(cores=1, horner_iterations=1),
            jit=JITConfig(optimization_level=0),
        ),
    )


def _dipoles() -> tuple[ColorCorrelator, ...]:
    return tuple(
        ColorCorrelator.dipole(f"T{i}{j}", i, j)
        for i in range(1, 5)
        for j in range(1, 5)
    )


@pytest.fixture(scope="module")
def runtimes(tmp_path_factory):
    pytest.importorskip("pyamplicol._rusticol")
    model = ModelSource.built_in_sm()
    root = tmp_path_factory.mktemp("strict-colour-correlated")
    result = {}
    for process in _QUARK_LEGS:
        for accuracy, graded in (("full", True), ("strict-nlc", False)):
            path = root / (
                f"{process.replace(' ', '_').replace('~', 'x')}-{accuracy}"
                f"{'-graded' if graded else ''}"
            )
            Generator(_configuration(accuracy, graded=graded)).generate(
                process,
                path,
                model=model,
                correlators=CorrelatorConfig(color_correlations=_dipoles()),
            )
            result[process, accuracy] = Runtime.load(path)
    return result


def _by_degree(evaluation) -> dict[int, Decimal]:
    return {
        degree: value.real
        for degree, value in zip(evaluation.degrees, evaluation.values[0], strict=True)
    }


@pytest.mark.parametrize("process", tuple(_QUARK_LEGS))
@pytest.mark.parametrize("accuracy", ("full", "strict-nlc"))
def test_colour_conservation_holds_degree_by_degree(runtimes, process, accuracy):
    runtime = runtimes[process, accuracy]
    born = runtime.evaluate_correlated_graded((_POINT,), precision=32)
    scale = abs(born.total[0].real)
    assert scale > 0
    with localcontext() as context:
        context.prec = 40
        for i in range(1, 5):
            rows = [
                runtime.evaluate_correlated_graded(
                    (_POINT,), color_correlation=f"T{i}{j}", precision=32
                )
                for j in range(1, 5)
            ]
            # Every dipole of a leg shares the reference degree P + 1; the
            # graded full colour may reach different lowest degrees.
            assert len({row.degrees[0] for row in rows}) == 1
            graded = [_by_degree(row) for row in rows]
            for degree in set().union(*graded):
                total = sum(
                    (values.get(degree, Decimal(0)) for values in graded), Decimal(0)
                )
                assert abs(total) < Decimal("1e-25") * scale, (i, degree, total)
            for row in rows:
                total = sum((value.real for value in row.values[0]), Decimal(0))
                assert abs(total - row.total[0].real) < Decimal("1e-28") * scale


@pytest.mark.parametrize("process", tuple(_QUARK_LEGS))
def test_casimirs_hold_degree_by_degree(runtimes, process):
    runtime = runtimes[process, "full"]
    born = _by_degree(runtime.evaluate_correlated_graded((_POINT,), precision=32))
    scale = max(abs(value) for value in born.values())
    with localcontext() as context:
        context.prec = 40
        for leg in range(1, 5):
            casimir = _by_degree(
                runtime.evaluate_correlated_graded(
                    (_POINT,), color_correlation=f"T{leg}{leg}", precision=32
                )
            )
            for degree, value in casimir.items():
                if leg in _QUARK_LEGS[process]:
                    expected = born.get(degree - 1, Decimal(0)) * 3 / 2 - born.get(
                        degree + 1, Decimal(0)
                    ) / Decimal(6)
                else:
                    expected = born.get(degree - 1, Decimal(0)) * 3
                assert abs(value - expected) < Decimal("1e-25") * scale, (
                    leg,
                    degree,
                    value,
                    expected,
                )


@pytest.mark.parametrize("process", tuple(_QUARK_LEGS))
def test_strict_nlc_correlations_are_the_window_of_the_graded_full_colour(
    runtimes, process
):
    full = runtimes[process, "full"]
    strict = runtimes[process, "strict-nlc"]
    for request in ("born", "T12", "T13", "T22"):
        graded = full.evaluate_correlated_graded(
            (_POINT,), color_correlation=request, precision=32
        )
        window = strict.evaluate_correlated_graded(
            (_POINT,), color_correlation=request, precision=32
        )
        reference = graded.degrees[0]
        assert window.degrees == (reference, reference - 1, reference - 2)
        kept = _by_degree(graded)
        scale = max(abs(value) for value in kept.values())
        for degree, value in _by_degree(window).items():
            assert abs(value - kept.get(degree, Decimal(0))) < Decimal("1e-28") * scale
        ordinary = strict.evaluate_correlated(
            (_POINT,), color_correlation=request, precision=32
        )[0]
        assert abs(ordinary.real - window.total[0].real) < Decimal("1e-28") * scale
