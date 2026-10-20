# SPDX-License-Identifier: 0BSD
"""Complete six-fermion QCD source control with genuine six-propagator families."""

from __future__ import annotations

import json
from decimal import Decimal, localcontext
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.loop_nightly


@pytest.fixture(scope="module")
def control(loop_sm_ufo, loop_native):
    symbolica = pytest.importorskip("symbolica")
    if not symbolica.is_licensed():
        pytest.skip("licensed native generation required")
    path = str(loop_sm_ufo)
    from pyamplicol.loops.compiler import prepare_loop_request_model
    from pyamplicol.loops.model_compatibility import prepare_loop_sm_scalar_declarations
    from pyamplicol.models.loop_import import load_raw_loop_model

    fixture = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "fixtures/loop/multiquark-six-propagator.json"
        ).read_text()
    )
    view = prepare_loop_sm_scalar_declarations(load_raw_loop_model(path))
    prepared = prepare_loop_request_model(
        view.derived,
        born_order_bounds={"QCD": 4, "QED": 0},
        loop_order_bounds={"QCD": 6, "QED": 0},
        fixed_parameters=fixture["fixed_parameters"],
    )
    program = prepared.compile(
        (2, -2),
        (1, -1, 3, -3),
        selected_corrections=["QCD"],
        parameters=fixture["parameters"],
        arithmetic="double",
        reduction_engine="fast",
        coefficient_engine="jit",
    )
    return program, fixture


def _components(result):
    return tuple(
        getattr(result, name)
        for name in (
            "bare",
            "uv",
            "r2",
            "virtual",
            "uv_local",
            "uv_external_fields",
        )
    )


def _check_statuses(result, bare):
    channels = sum(len(family.sector_indices) for family in bare.families)
    assert channels > 0
    assert len(result.statuses) == len(result.born)
    for point, row in enumerate(result.statuses):
        assert len(row) == channels
        assert all(lane == point and status == "success" for lane, status in row)


def _check_reference(result, rows):
    for index, row in enumerate(rows):
        for reference in (row, *row["providers"].values()):
            np.testing.assert_allclose(
                float(result.born[index, 0]),
                reference["born"],
                rtol=2e-10,
                atol=1e-22,
            )
            np.testing.assert_allclose(
                result.virtual[index, :, 0].astype(float),
                [reference[key] for key in ("finite", "single_pole", "double_pole")],
                rtol=2e-8,
                atol=1e-21,
            )
        for actual, key in zip(
            _components(result),
            ("bare", "uv", "r2", "full", "uv_vertex", "wf"),
            strict=True,
        ):
            np.testing.assert_allclose(
                actual[index, :, 0].astype(float),
                [
                    row["components"][key][pole]
                    for pole in ("finite", "single_pole", "double_pole")
                ],
                rtol=2e-8,
                atol=abs(row["born"]) * 2e-12,
                err_msg=key,
            )


@pytest.mark.parametrize(
    "arithmetic,precision,bit,tolerance",
    [
        ("double", None, 0, "2e-9"),
        ("double-double", None, 70, "3e-25"),
        ("arbitrary", 80, 180, "1e-60"),
    ],
)
def test_complete_source_six_propagator_poles_and_exact_coupling(
    control, arithmetic, precision, bit, tolerance
):
    from symbolica import E, Expression

    original, fixture = control
    program = (
        original
        if arithmetic == "double"
        else original.rebind(arithmetic=arithmetic, precision=precision)
    )
    bare = program.source_bare.bare
    assert str(bare.initial_average) == "36"
    assert len(bare.process.born) == 7
    assert len(bare.loop_graphs) == 275
    assert max(len(f.denominators) for f in bare.process.families) == 6
    assert dict(bare.order_layout.born[0].order) == {"QCD": 8}
    assert dict(bare.order_layout.virtual[0].order) == {"QCD": 10}
    scales = [str(int(value) ** 2) for value in fixture["mu_r"]]
    # Exact terminating decimal inputs eliminate rounding-induced off-shell
    # invariants without numerical snapping. Rounded-input rescue remains a
    # separate unresolved structural contract, not certified by this fixture.
    points = np.asarray(fixture["points"], dtype=str)
    for point in points:
        vectors = [[Fraction(value) for value in leg] for leg in point]
        assert all(v[0] ** 2 - sum(x**2 for x in v[1:]) == 0 for v in vectors)
        assert all(
            sum(vectors[leg][mu] for leg in (0, 1))
            - sum(vectors[leg][mu] for leg in range(2, 6))
            == 0
            for mu in range(4)
        )
    result = program.evaluate_decimal(points, scales)
    _check_statuses(result, bare)
    _check_reference(result, fixture["rows"])
    factor = (
        Expression.evaluator_multiple(
            [-E("8") * E("118/1000") / (2 * E("pi"))],
            [],
            n_cores=1,
            jit_compile=False,
        )
        .evaluate_complex_with_prec([], 110)[0]
        .to_decimal_tuple()[0]
    )
    with localcontext() as context:
        context.prec = 110
        threshold = Decimal(tolerance)
        for index in range(3):
            born = Decimal(result.born[index, 0])
            expected = born * factor
            assert (
                abs(Decimal(result.virtual[index, 2, 0]) - expected)
                < abs(expected) * threshold
            )
            for pole in range(3):
                total = sum(
                    Decimal(getattr(result, name)[index, pole, 0])
                    for name in ("bare", "uv", "r2")
                )
                assert (
                    abs(Decimal(result.virtual[index, pole, 0]) - total)
                    < abs(born) * threshold
                )
                for value in _components(result):
                    assert abs(Decimal(value[index, pole, 1])) < abs(born) * threshold
        alpha_factor = Decimal(2) if not bit else 1 + Decimal(2) ** -bit
        program.update_parameters({"aS": str(Decimal("0.118") * alpha_factor)})
        changed = program.evaluate_decimal(points[:1], scales[:1])
        _check_statuses(changed, bare)
        born = Decimal(result.born[0, 0])
        assert Decimal(changed.born[0, 0]) != born
        assert (
            abs(Decimal(changed.born[0, 0]) - born * alpha_factor**4)
            < abs(born) * threshold
        )
        for before, after in zip(
            _components(result), _components(changed), strict=True
        ):
            for pole in range(3):
                expected = Decimal(before[0, pole, 0]) * alpha_factor**5
                assert (
                    abs(Decimal(after[0, pole, 0]) - expected) < abs(born) * threshold
                )
    program.update_parameters({"aS": "0.118"})
    empty = program.evaluate_decimal(points[:0], [])
    assert empty.virtual.shape == (0, 3, 2)
    _check_statuses(empty, bare)
    permuted = program.evaluate_decimal(points[[2, 0]], [scales[2], scales[0]])
    _check_statuses(permuted, bare)
    with localcontext() as context:
        context.prec = 110
        for expected, actual in zip(
            _components(result), _components(permuted), strict=True
        ):
            for index, source_index in enumerate((2, 0)):
                scale = abs(Decimal(result.born[source_index, 0])) * Decimal(tolerance)
                assert all(
                    abs(Decimal(x) - Decimal(y)) < scale
                    for x, y in zip(
                        expected[source_index].flat, actual[index].flat, strict=True
                    )
                )


def test_source_symmetric_gram_matches_collier_and_recola(control):
    program, fixture = control
    stress = fixture["symmetric_stress"]
    result = program.evaluate_decimal(
        np.asarray(stress["points"], dtype=str),
        [str(int(value) ** 2) for value in stress["mu_r"]],
    )
    _check_statuses(result, program.source_bare.bare)
    for index, row in enumerate(stress["rows"]):
        for reference in row.values():
            np.testing.assert_allclose(
                float(result.born[index, 0]), reference["born"], rtol=2e-10, atol=0
            )
            np.testing.assert_allclose(
                result.virtual[index, :, 0].astype(float),
                [reference[key] for key in ("finite", "single_pole", "double_pole")],
                rtol=2e-8,
                atol=abs(reference["born"]) * 2e-12,
            )
