# SPDX-License-Identifier: 0BSD
"""Native output transport, including sparse failed rows and exact carriers."""

import json
from dataclasses import replace
from decimal import Decimal, localcontext
from pathlib import Path

import pytest

from pyamplicol.config import R2Strategy
from pyamplicol.loops.orders import BornLoopPair, VirtualOrderSector
from pyamplicol.loops.physical_orders import BornSquaredSector, PhysicalOrderLayout
from pyamplicol.loops.result_bridge import stable_model_evaluation
from pyamplicol.loops.stability import (
    ArithmeticSpec,
    NativeStabilityDiagnostic,
    NativeStableModelResult,
    NumericalVerificationError,
)
from pyamplicol.loops.strategy import R2Selection

pytestmark = pytest.mark.loop_fast

Q2, Q4 = (("QCD", 2),), (("QCD", 4),)
E2 = (("QED", 2),)
ORDERS = PhysicalOrderLayout(
    (BornSquaredSector(Q4, ((Q2, Q2),)),),
    (
        VirtualOrderSector((("QCD", 6),), (BornLoopPair(Q2, Q4),)),
        VirtualOrderSector(Q4 + E2, (BornLoopPair(E2, Q4),)),
    ),
    ("QCD",),
)
R2 = R2Selection(R2Strategy.AUTO, R2Strategy.MODEL, "only eligible branch")
DEFAULT_SPEC = ArithmeticSpec("double")


def packet(spec=DEFAULT_SPEC):
    from pyamplicol.loops.native import load_loop_native

    backend = load_loop_native()

    # Successful rows are deliberately ordered differently from the request.
    def batch(values, columns):
        return backend.NativeBatch.from_decimal(
            [(str(value), "0") for value in values],
            2,
            columns,
            row_ids=[6, 91],
            **spec.options,
        )

    born = batch([6, 91], 1)
    virtual = batch([1, -2, 3, 4, -5, 6, 7, -8, 9, 10, -11, 12], 6)
    diagnostics = tuple(
        NativeStabilityDiagnostic(
            point,
            status,
            evidence,
            spec,
            ArithmeticSpec("arbitrary", 100)
            if status == "rescued"
            else spec
            if status == "success"
            else None,
            None,
            (),
            (),
            (),
        )
        for point, status, evidence in (
            (91, "rescued", "empirical-cross-precision-agreement"),
            (3, "failed", "not-established"),
            (6, "success", "not-established"),
        )
    )
    return NativeStableModelResult(
        (91, 3, 6),
        (6, 91),
        born,
        virtual,
        virtual,
        virtual,
        virtual,
        virtual,
        virtual,
        virtual,
        ORDERS,
        (),
        r2_strategy="model",
        diagnostics=diagnostics,
    )


@pytest.mark.parametrize(
    "spec",
    [
        ArithmeticSpec("double"),
        ArithmeticSpec("double-double"),
        ArithmeticSpec("arbitrary", 80),
    ],
)
def test_sparse_transport_preserves_orders_all_poles_and_requested_row_order(
    loop_native, spec
):
    native = packet(spec)
    with pytest.raises(NumericalVerificationError):
        stable_model_evaluation(native, r2=R2)
    output = stable_model_evaluation(native, r2=R2, allow_partial=True)
    assert output.born_values == ((91,), None, (6,))
    assert output.values == (
        ((7, -8, 9), (10, -11, 12)),
        None,
        ((1, -2, 3), (4, -5, 6)),
    )
    assert output.sum_orders() == ((17, -19, 21), None, (5, -7, 9))
    assert output.for_order(Q4 + E2).single_pole == ((-11,), None, (-5,))
    assert output.valid == (True, False, True)
    assert output.diagnostics[0].status == "rescued"
    assert output.diagnostics[0].accuracy_established
    assert "arbitrary at 100 decimal digits" in output.diagnostics[0].detail
    assert not output.diagnostics[2].accuracy_established
    expected_type = float if spec.arithmetic == "double" else Decimal
    assert type(output.values[0][0][0]) is expected_type
    assert output.r2 is R2


@pytest.mark.parametrize(
    "spec,literal",
    [
        (ArithmeticSpec("double-double"), "1.0000000000000000000000001"),
        (
            ArithmeticSpec("arbitrary", 80),
            "1.0000000000000000000000000000000000000000000000000000000000001",
        ),
    ],
)
def test_precise_transport_never_passes_through_float_or_ambient_decimal_context(
    loop_native, spec, literal
):
    native = packet(spec)
    born = type(native.born).from_decimal(
        [(literal, "0"), (literal, "0")],
        2,
        1,
        row_ids=[6, 91],
        **spec.options,
    )
    expected = Decimal(born.to_decimal()[0][0])
    assert expected != 1 and float(expected) == 1
    with localcontext() as context:
        context.prec = 4
        output = stable_model_evaluation(
            replace(native, born=born), r2=R2, allow_partial=True
        )
    assert output.born_values == ((expected,), None, (expected,))


def test_native_carrier_and_row_metadata_must_match_the_result(loop_native):
    native = packet()
    assert (
        stable_model_evaluation(native, r2=R2, allow_partial=True).color_accuracy
        == "full"
    )
    # A native AP buffer cannot inherit a binary64 result's output metadata.
    arbitrary = native.born.convert_arithmetic(arithmetic="arbitrary", precision=16)
    with pytest.raises(ValueError, match="arithmetic or precision"):
        stable_model_evaluation(
            replace(native, born=arbitrary), r2=R2, allow_partial=True
        )
    with pytest.raises(ValueError, match="point/order layout"):
        stable_model_evaluation(
            replace(native, virtual=native.virtual.take_rows([1, 0])),
            r2=R2,
            allow_partial=True,
        )
    with pytest.raises(ValueError, match="executed native strategy"):
        stable_model_evaluation(
            native,
            r2=replace(R2, effective=R2Strategy.DIRECT),
            allow_partial=True,
        )


def test_direct_output_keeps_executed_strategy_and_rejects_relabeling(loop_native):
    native = replace(packet(), r2_strategy="direct")
    direct = replace(R2, effective=R2Strategy.DIRECT)
    output = stable_model_evaluation(native, r2=direct, allow_partial=True)
    assert output.r2 is direct
    assert output.single_pole == ((-8, -11), None, (-2, -5))
    with pytest.raises(ValueError, match="executed native strategy"):
        stable_model_evaluation(native, r2=R2, allow_partial=True)
    with pytest.raises(ValueError, match="executed native strategy"):
        stable_model_evaluation(
            replace(native, r2_strategy="auto"), r2=direct, allow_partial=True
        )


def test_ordinary_output_rejects_tiny_imaginary_values_and_empty_output_keeps_axes(
    loop_native,
):
    spec = ArithmeticSpec("arbitrary", 80)
    native = packet(spec)
    contaminated = type(native.born).from_decimal(
        [("6", "1e-400"), ("91", "0")],
        2,
        1,
        row_ids=[6, 91],
        **spec.options,
    )
    with pytest.raises(ValueError, match="must be real"):
        stable_model_evaluation(
            replace(native, born=contaminated), r2=R2, allow_partial=True
        )
    empty = replace(
        native,
        requested_point_ids=(),
        point_ids=(),
        diagnostics=(),
        born=native.born.take_rows([]),
        virtual=native.virtual.take_rows([]),
    )
    output = stable_model_evaluation(empty, r2=R2)
    assert output.shape == (0, 2, 3) and output.born_shape == (0, 1)
    assert output.values == output.born_values == ()


def test_complete_physical_drell_yan_result_uses_same_output_boundary(
    loop_sm_ufo, loop_native
):
    symbolica = pytest.importorskip("symbolica")
    path = str(loop_sm_ufo)
    if not symbolica.is_licensed():
        pytest.skip("licensed Symbolica and loop_sm reference required")
    from pyamplicol.loops.model import build_loop_model
    from pyamplicol.loops.physical_counterterms import drell_yan_model_virtual_program
    from pyamplicol.loops.stability import NativeModelStabilityProgram
    from pyamplicol.models.loop_import import load_raw_loop_model

    source = build_loop_model(load_raw_loop_model(path))
    model = drell_yan_model_virtual_program(source, arithmetic="double-double")
    program = NativeModelStabilityProgram(
        model,
        (),
        relative_tolerance="1e-24",
        absolute_tolerance="1e-40",
    )
    fixture = (
        Path(__file__).resolve().parents[2] / "tools/loop_reference/drell_yan_qcd.json"
    )
    points = [
        [[str(component) for component in momentum] for momentum in point]
        for point in json.loads(fixture.read_text())["points"][:1]
    ]
    result = program.evaluate_native(
        points,
        ["1000000"],
        model_values=["0.118", "132.507"],
        point_ids=[71],
    )
    output = stable_model_evaluation(result, r2=R2)
    assert output.values[0][0] == tuple(
        Decimal(real) for real, _ in result.virtual.to_decimal()
    )
    assert output.born_values[0] == tuple(
        Decimal(real) for real, _ in result.born.to_decimal()
    )
    assert output.shape == (1, 1, 3)
    assert not output.diagnostics[0].accuracy_established
