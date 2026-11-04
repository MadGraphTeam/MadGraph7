# SPDX-License-Identifier: 0BSD
"""Strict and graded tree colour against NJet's N_c scan (independent oracle).

``tests/fixtures/loop/nc/njet_nc_scan.json`` (MQ-NC's oracle) holds NJet 2.1.1's
colour-summed Born ``B(N)`` at N_c = 2, 3, 4, 5 and 6 for fixed points, with
the N_c = 3 colour average at every N. A pure-QCD tree is a Laurent
polynomial in N_c whose degrees step by two from the reference degree
P = n_g + n_qqbar; fitting ``B(N) = sum_d b_d N^d`` over the scan (exact
least squares in rationals, one or more spare scan points as a residual
check) gives the formal 1/N_c expansion without any pyAmpliCol colour code.
Its shares ``b_d 3^d / B(3)`` must equal pyAmpliCol's graded full colour
degree by degree, and the strict-lc (degree P) and strict-nlc (P, P-1, P-2)
totals over full colour. Ratios cancel the couplings and averages.
"""

from __future__ import annotations

import json
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

import pytest

from pyamplicol import Generator, ModelSource, Runtime
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

_SCAN = Path(__file__).resolve().parents[1] / "fixtures/loop/nc/njet_nc_scan.json"
# Scan entry -> (reference degree P, number of fitted degrees P, P-2, ...).
_FITS = {
    "gg_gg": (4, 2),
    "gg_ggg": (5, 2),
    "uux_ggg": (4, 4),
    "uux_ddxg": (3, 3),
}
_DEFAULT = ("gg_ggg", "uux_ddxg")
_NIGHTLY = ("gg_gg", "uux_ggg")


def _scan() -> dict:
    return json.loads(_SCAN.read_text())["processes"]


def _solve(matrix: list[list[Fraction]], vector: list[Fraction]) -> list[Fraction]:
    size = len(vector)
    rows = [[*row, value] for row, value in zip(matrix, vector, strict=True)]
    for column in range(size):
        pivot = next(r for r in range(column, size) if rows[r][column] != 0)
        rows[column], rows[pivot] = rows[pivot], rows[column]
        for r in range(size):
            if r != column and rows[r][column] != 0:
                factor = rows[r][column] / rows[column][column]
                rows[r] = [a - factor * b for a, b in zip(rows[r], rows[column])]
    return [rows[r][size] / rows[r][r] for r in range(size)]


def _fit(entry: dict, point: int, degrees: tuple[int, ...]):
    """Relative least squares ``B(N) ~ sum_d b_d N^d``; returns ``{d: b_d}``
    and the largest relative residual over the scan."""

    data = [
        (Fraction(int(n)), Fraction(Decimal(records[point]["born"])))
        for n, records in entry["nc"].items()
    ]
    design = [[n**d / born for d in degrees] for n, born in data]
    target = [Fraction(1)] * len(data)
    normal = [
        [sum(row[i] * row[j] for row in design) for j in range(len(degrees))]
        for i in range(len(degrees))
    ]
    rhs = [
        sum(row[i] * t for row, t in zip(design, target, strict=True))
        for i in range(len(degrees))
    ]
    coefficients = dict(zip(degrees, _solve(normal, rhs), strict=True))
    residual = max(
        abs(sum(b * n**d for d, b in coefficients.items()) / born - 1)
        for n, born in data
    )
    return coefficients, float(residual)


def _configuration(orders: dict[str, int], accuracy: str, graded: bool) -> RunConfig:
    return RunConfig(
        action="generate",
        process=ProcessConfig(
            coupling_order_policy="explicit", max_coupling_orders=orders
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


@pytest.fixture(scope="module")
def runtimes(tmp_path_factory):
    pytest.importorskip("pyamplicol._rusticol")
    return {}, tmp_path_factory.mktemp("strict-colour-nc-scan")


def _runtime(runtimes, name: str, accuracy: str, graded: bool) -> Runtime:
    cache, root = runtimes
    key = (name, accuracy)
    if key not in cache:
        entry = _scan()[name]
        path = root / f"{name}-{accuracy}"
        Generator(_configuration(dict(entry["born_orders"]), accuracy, graded)).generate(
            entry["process"], path, model=ModelSource.built_in_sm()
        )
        cache[key] = Runtime.load(path)
    return cache[key]


def _check(runtimes, name: str) -> None:
    entry = _scan()[name]
    reference, count = _FITS[name]
    degrees = tuple(range(reference, reference - 2 * count, -2))
    points = tuple(
        tuple(tuple(float(component) for component in leg) for leg in point)
        for point in entry["points"]
    )
    graded_full = _runtime(runtimes, name, "full", True).evaluate_graded(points)
    assert graded_full.reference_degree == reference
    totals = {
        accuracy: _runtime(runtimes, name, accuracy, False).evaluate(points)
        for accuracy in ("strict-lc", "strict-nlc")
    }
    for index in range(len(points)):
        coefficients, residual = _fit(entry, index, degrees)
        assert residual < 1e-10, (name, index, residual)
        born3 = sum(b * 3**d for d, b in coefficients.items())
        expected = {d: float(b * 3**d / born3) for d, b in coefficients.items()}
        total = complex(graded_full.total[index]).real
        shares = {
            degree: complex(value).real / total
            for degree, value in zip(
                graded_full.degrees, graded_full.values[index], strict=True
            )
        }
        # Every degree of either side: a degree the colour weights reach but
        # the amplitudes cancel (or one the scan fit finds zero) is zero.
        for degree in set(expected) | set(shares):
            assert shares.get(degree, 0.0) == pytest.approx(
                expected.get(degree, 0.0), rel=1e-9, abs=1e-9
            ), (name, index, degree, shares, expected)
        lc = complex(totals["strict-lc"][index]).real / total
        nlc = complex(totals["strict-nlc"][index]).real / total
        assert lc == pytest.approx(expected[reference], rel=1e-9)
        assert nlc == pytest.approx(
            sum(expected.get(d, 0.0) for d in range(reference, reference - 3, -1)),
            rel=1e-9,
        )


@pytest.mark.parametrize("name", _DEFAULT)
def test_strict_and_graded_tree_colour_match_the_njet_nc_scan(runtimes, name):
    _check(runtimes, name)


@pytest.mark.loop_nightly
@pytest.mark.parametrize("name", _NIGHTLY)
def test_strict_and_graded_tree_colour_match_the_njet_nc_scan_nightly(
    runtimes, name
):
    _check(runtimes, name)
