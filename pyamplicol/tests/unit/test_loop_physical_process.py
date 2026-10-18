# SPDX-License-Identifier: 0BSD
"""Independent physical controls for multi-graph native bare assembly."""

from __future__ import annotations

import json
import os
from decimal import Decimal, localcontext
from pathlib import Path

import numpy as np
import pytest

from pyamplicol.loops.physical_process import distinct_flavour_qcd_program


@pytest.fixture(scope="module")
def native():
    symbolica = pytest.importorskip("symbolica")
    if not symbolica.is_licensed():
        pytest.skip("native generation requires a licensed Symbolica runtime")
    return pytest.importorskip("_loop_reduction")


@pytest.fixture(scope="module")
def points():
    path = Path(__file__).resolve().parents[2] / "tools/loop_reference/uudd_qcd.json"
    return np.asarray(json.loads(path.read_text())["points"], dtype=str)


def _exact_born_and_double(point, alpha_s="0.118"):
    from symbolica import E, Expression

    from pyamplicol.loops.model_expressions import parse_exact_ufo

    p = [[parse_exact_ufo(value) for value in leg] for leg in point]

    def square(vector):
        return vector[0] ** 2 - sum((x**2 for x in vector[1:]), E("0"))

    s = square([a + b for a, b in zip(p[0], p[1], strict=True)])
    t = square([a - b for a, b in zip(p[0], p[2], strict=True)])
    u = square([a - b for a, b in zip(p[0], p[3], strict=True)])
    alpha = parse_exact_ufo(alpha_s)
    born = E("4/9") * (4 * E("pi") * alpha) ** 2 * (t * t + u * u) / s**2
    double = -4 * E("4/3") * alpha / (2 * E("pi")) * born
    evaluator = Expression.evaluator_multiple(
        [born, double], [], n_cores=1, jit_compile=False
    )
    return [
        value.to_decimal_tuple()[0]
        for value in evaluator.evaluate_complex_with_prec([], 110)
    ]


def _check_born_and_double(result, points, tolerance, alpha_s="0.118"):
    with localcontext() as context:
        context.prec = 120
        for point, born, poles in zip(points, result.born, result.virtual, strict=True):
            targets = _exact_born_and_double(point, alpha_s)
            for actual, target in zip([born[0], poles[2, 0]], targets, strict=True):
                assert abs(Decimal(actual) - target) < abs(target) * tolerance
            assert all(Decimal(pair[1]) == 0 for pair in [born, *poles])
    assert all(status == "success" for row in result.statuses for _, status in row)


@pytest.mark.parametrize(
    "arithmetic,precision,engine,tolerance",
    [
        ("double", None, "fast", Decimal("2e-11")),
        ("double", None, "generic", Decimal("2e-11")),
        ("double-double", None, "generic", Decimal("2e-26")),
        ("arbitrary", 50, "generic", Decimal("2e-43")),
    ],
)
def test_multi_graph_bare_process_born_ir_poles_and_parameter_rebinding(
    native, points, arithmetic, precision, engine, tolerance
):
    program = distinct_flavour_qcd_program(
        backend=native,
        arithmetic=arithmetic,
        precision=precision,
        reduction_engine=engine,
    )
    assert len(program.recurrences) == 14 and len(program.families) == 8
    # Two nonabelian vertex graphs carry imaginary Born/color weights. They
    # cannot be projected to their real parts before the loop integrals.
    assert any(value != value.conj() for row in program.loop_metric for value in row)
    result = program.evaluate_decimal(
        points, ["1000000"] * 3, model_values=["0.118", "173"]
    )
    _check_born_and_double(result, points, tolerance)
    assert len(result.statuses) == len(points)
    assert all(len(row) == len(program.families) for row in result.statuses)
    changed_mass = program.evaluate_decimal(
        points[1:2], ["1000000"], model_values=["0.118", "190"]
    )
    assert abs(
        Decimal(changed_mass.virtual[0, 0, 0]) - Decimal(result.virtual[1, 0, 0])
    ) > Decimal("1e-10")
    rebound = program.evaluate_decimal(
        points[1:2], ["900000"], model_values=["0.1", "190"]
    )
    _check_born_and_double(rebound, points[1:2], tolerance, "0.1")
    empty = program.evaluate_decimal(points[:0], [], model_values=["0.118", "173"])
    assert empty.born.shape == (0, 2) and empty.virtual.shape == (0, 3, 2)
    assert empty.statuses == ()
    with pytest.raises(TypeError, match="decimal strings"):
        program.evaluate_decimal(
            points.astype(float), ["1000000"] * 3, model_values=["0.118", "173"]
        )


@pytest.mark.parametrize("engine", ["fast", "generic"])
def test_bare_process_agrees_with_independent_madloop_components(
    native, points, engine
):
    path = os.environ.get("PYAMPLICOL_UUDD_COMPONENT_REFERENCE")
    if path is None:
        pytest.skip(
            "set PYAMPLICOL_UUDD_COMPONENT_REFERENCE to the MadLoop component report"
        )
    reference = json.loads(Path(path).read_text())
    assert reference["fixture"]["loop_quark_flavours"] == 6
    assert reference["fixture"]["renormalization_flavours"] == 5
    result = distinct_flavour_qcd_program(
        backend=native, reduction_engine=engine
    ).evaluate_decimal(points, ["1000000"] * 3, model_values=["0.118", "173"])
    for index, row in enumerate(reference["rows"]):
        assert float(result.born[index, 0]) == pytest.approx(row["born"], rel=2e-14)
        np.testing.assert_allclose(
            result.virtual[index, :, 0].astype(float),
            [row["bare"][key] for key in ("finite", "single_pole", "double_pole")],
            rtol=3e-12,
            atol=2e-14,
        )


@pytest.mark.parametrize(
    "attribute,stage",
    [("born_projector", "Born projection"), ("postprocess", "bare virtual assembly")],
)
def test_nonfinite_final_physical_projection_is_not_a_success(
    native, points, attribute, stage
):
    from types import SimpleNamespace

    program = distinct_flavour_qcd_program(backend=native, reduction_engine="fast")

    def failed_projection(values):
        return native.NativeBatch.from_decimal(
            [["NaN", "0"]] * values.rows, values.rows, 1, row_ids=values.row_ids
        )

    setattr(program, attribute, SimpleNamespace(evaluate_batch=failed_projection))
    with pytest.raises(ArithmeticError, match=f"nonfinite {stage}"):
        program.evaluate_decimal(points[:1], ["1000000"], model_values=["0.118", "173"])


def test_native_jit_coefficients_match_multi_family_interpreter(native, points):
    results = []
    for engine in ("interpreted", "jit"):
        program = distinct_flavour_qcd_program(
            backend=native, coefficient_engine=engine, reduction_engine="fast"
        )
        result = program.evaluate_decimal(
            points, ["1000000"] * 3, model_values=["0.118", "173"]
        )
        assert result.coefficient_engine == engine
        assert all(
            executor.coefficient_engine == engine
            for executor in (*program.born_executors, *program.loop_executors)
        )
        results.append(result)
    np.testing.assert_allclose(
        results[0].virtual.astype(float),
        results[1].virtual.astype(float),
        rtol=2e-11,
        atol=1e-15,
    )
    with pytest.raises(ValueError, match="double"):
        distinct_flavour_qcd_program(
            backend=native, coefficient_engine="jit", arithmetic="double-double"
        )


def test_high_precision_convergence_and_original_momentum_bits(native, points):
    # Convergence is an arithmetic control, not an independent 80-digit finite
    # reference. Born and the universal double pole have exact controls above.
    lower = distinct_flavour_qcd_program(
        backend=native, arithmetic="arbitrary", precision=50
    )
    upper = distinct_flavour_qcd_program(
        backend=native, arithmetic="arbitrary", precision=80
    )
    with localcontext() as context:
        context.prec = 220
        factor = 1 + Decimal(2) ** -150
        original = points[:1]
        changed = np.asarray(
            [[[str(Decimal(x) * factor) for x in leg] for leg in original[0]]],
            dtype=object,
        )
        np.testing.assert_array_equal(original.astype(float), changed.astype(float))
        low = lower.evaluate_decimal(
            original, ["1000000"], model_values=["0.118", "173"]
        )
        high = upper.evaluate_decimal(
            np.concatenate([original, changed]),
            ["1000000"] * 2,
            model_values=["0.118", "173"],
        )
        _check_born_and_double(
            high, np.concatenate([original, changed]), Decimal("2e-72")
        )
        for a, b in zip(low.virtual[0, :, 0], high.virtual[0, :, 0], strict=True):
            assert abs(Decimal(a) / Decimal(b) - 1) < Decimal("2e-42")
        assert Decimal(high.virtual[1, 1, 0]) != Decimal(high.virtual[0, 1, 0])


def test_same_process_assembler_reproduces_drell_yan_oracle(native, points):
    from symbolica import E, S
    from symbolica.community import hepkit as h

    from pyamplicol.loops.graphs import build_loop_process
    from pyamplicol.loops.physical import DrellYanBareProgram
    from pyamplicol.loops.physical_process import NativeBareProcessProgram

    model = h.Model.qcd_qed()
    process = build_loop_process(
        model.process(["u", "u~"], ["e-", "e+"]),
        selected_corrections=["QCD"],
        perturbative_orders={"QCD": 1},
    )
    alpha_s, inverse_alpha = S("process_test::alpha_s", "process_test::inverse_alpha")
    expressions = {}
    for name in ("GC_2", "GC_3", "GC_11"):
        coupling = model.coupling(name)
        expressions[coupling.symbol.get_name()] = coupling.expression.replace(
            S("UFO::G"), (4 * E("pi") * alpha_s) ** E("1/2")
        ).replace(S("UFO::ee"), (4 * E("pi") / inverse_alpha) ** E("1/2"))
    generic = NativeBareProcessProgram(
        process,
        state_kinds=("u", "v_bar", "u_bar", "v"),
        initial_average=E("36"),
        model_expressions=expressions,
        model_inputs=(alpha_s, inverse_alpha),
        backend=native,
        reduction_engine="fast",
    )
    actual = generic.evaluate_decimal(
        points, ["1000000"] * 3, model_values=["0.118", "132.507"]
    )
    oracle = DrellYanBareProgram(
        backend=native, reduction_engine="fast"
    ).evaluate_decimal(points, ["1000000"] * 3)
    np.testing.assert_allclose(
        actual.born.astype(float), oracle.born.astype(float), rtol=2e-13
    )
    np.testing.assert_allclose(
        actual.virtual.astype(float),
        oracle.virtual.astype(float),
        rtol=2e-11,
        atol=1e-16,
    )
