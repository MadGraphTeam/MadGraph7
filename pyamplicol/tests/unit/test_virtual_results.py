# SPDX-License-Identifier: 0BSD
from dataclasses import FrozenInstanceError, replace
from decimal import Decimal, localcontext

import pytest

from pyamplicol.api.virtual_results import (
    LaurentCoefficients,
    ResolvedVirtualEvaluation,
    VirtualEvaluation,
    VirtualFailureDiagnostic,
    VirtualLaneDiagnostic,
)
from pyamplicol.config import R2Strategy
from pyamplicol.correlators import CorrelatedValue
from pyamplicol.loops.orders import BornLoopPair, VirtualOrderSector
from pyamplicol.loops.strategy import R2Selection

QCD2 = (("QCD", 2),)
QED2 = (("QED", 2),)
SECTORS = (
    VirtualOrderSector((("QCD", 6),), (BornLoopPair(QCD2, (("QCD", 4),)),)),
    VirtualOrderSector(
        (("QCD", 4), ("QED", 2)),
        (BornLoopPair(QED2, (("QCD", 4),)), BornLoopPair(QCD2, QCD2 + QED2)),
    ),
)
STRATEGY = R2Selection(R2Strategy.AUTO, R2Strategy.MODEL, "only complete strategy")
SUCCESS = VirtualLaneDiagnostic("success", "double", 53, False)


def result():
    return VirtualEvaluation(
        ((10.0, 20.0),),
        (((1.0, 2.0, 3.0), (-4.0, 5.0, -6.0)),),
        ((("QCD", 4),), (("QED", 4),)),
        SECTORS,
        (SUCCESS,),
        STRATEGY,
    )


def test_orders_and_all_laurent_coefficients_remain_individually_accessible():
    value = result()
    assert value.shape == (1, 2, 3)
    assert value.epsilon_powers == (0, -1, -2)
    assert value.finite == ((1.0, -4.0),)
    assert value.single_pole == ((2.0, 5.0),)
    assert value.double_pole == ((3.0, -6.0),)
    selected = value.for_order((("QED", 2), ("QCD", 4)))
    assert selected.shape == (1, 1, 3)
    assert len(selected.sectors[0].pairs) == 2
    assert value.sum_orders() == (LaurentCoefficients(-3.0, 7.0, -3.0),)
    assert selected.born_values == value.born_values
    with pytest.raises(ValueError, match="absent"):
        value.for_order((("QCD", 8),))
    with pytest.raises(TypeError):
        float(value)


def test_result_freezes_inputs_and_never_confuses_three_born_orders_with_epsilon():
    raw = [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]
    value = replace(
        result(),
        values=raw,
        born_values=[[10.0, 20.0, 30.0]],
        born_orders=((), QCD2, QED2),
    )
    raw[0][0][0] = 999.0
    assert value.finite == ((1.0, 4.0),)
    assert type(value.born_values[0]) is tuple
    assert type(value.values[0][0]) is LaurentCoefficients
    with pytest.raises(FrozenInstanceError):
        value.values = ()


def test_failed_lane_does_not_publish_stale_values_and_rescue_is_per_lane():
    value = result()
    failed = VirtualLaneDiagnostic(
        "failed", "arbitrary", 256, False, "reduction failed"
    )
    rescued = VirtualLaneDiagnostic("rescued", "double-double", 106, True)
    value = replace(
        value,
        born_values=(value.born_values[0], None, (Decimal(1), Decimal(2))),
        values=(value.values[0], None, ((Decimal(1),) * 3, (Decimal(2),) * 3)),
        diagnostics=(SUCCESS, failed, rescued),
    )
    assert value.valid == (True, False, True)
    assert value.finite == ((1.0, -4.0), None, (Decimal(1), Decimal(2)))
    assert value.sum_orders()[1] is None
    assert value.r2 is STRATEGY
    assert value.diagnostics[0].accuracy_established is False
    with pytest.raises(ValueError, match="stale"):
        replace(value, diagnostics=(failed, failed, rescued))
    with pytest.raises(TypeError, match="precision-preserving"):
        replace(result(), diagnostics=(rescued,))


def test_resolved_total_keeps_orders_epsilon_and_independent_colour_axes():
    # Two helicities, two Born components, three signed virtual components.
    born = ((((1.0, 2.0), (3.0, 4.0)), ((5.0, 6.0), (7.0, 8.0))),)
    sectors = tuple(
        tuple(
            tuple(
                (float(100 * order + 10 * helicity + colour), -1.0, -2.0)
                for colour in range(3)
            )
            for helicity in range(2)
        )
        for order in range(2)
    )
    value = ResolvedVirtualEvaluation(
        born,
        (sectors,),
        result().born_orders,
        SECTORS,
        (SUCCESS,),
        STRATEGY,
        ("h0", "h1"),
        ("b0", "b1"),
        ("v0", "v1", "v2"),
        "lc",
    )
    assert value.shape == (1, 2, 2, 3, 3)
    assert value.born_shape == (1, 2, 2, 2)
    assert value.finite[0][1][1] == (110.0, 111.0, 112.0)
    assert value.for_order(SECTORS[1].order).total().finite == ((636.0,),)
    total = value.total()
    assert total.born_values == ((10.0, 26.0),)
    assert total.values == (((36.0, -6.0, -12.0), (636.0, -6.0, -12.0)),)
    assert total.sectors == SECTORS


@pytest.mark.parametrize("correlated", [False, True])
def test_high_precision_output_sum_ignores_ambient_decimal_rounding(correlated):
    first = Decimal("100000000000000000000000000000000000000000000000000000000000001")
    second = Decimal("-100000000000000000000000000000000000000000000000000000000000000")
    values = tuple(CorrelatedValue(x, x) if correlated else x for x in (first, second))
    precise = VirtualLaneDiagnostic("success", "arbitrary", 256, True)
    with localcontext() as context:
        context.prec = 6
        value = replace(
            result(),
            born_values=((values[0], values[1]),),
            values=((tuple([values[0]] * 3), tuple([values[1]] * 3)),),
            diagnostics=(precise,),
            correlated=correlated,
        )
        expected = CorrelatedValue(Decimal(1), Decimal(1)) if correlated else Decimal(1)
        assert value.sum_orders() == (
            LaurentCoefficients(expected, expected, expected),
        )


def test_directed_complex_values_are_preserved_and_ordinary_output_rejects_them():
    complex_values = (((1 + 2j, -3 + 4j, 5 - 6j), (0j, 0j, 0j)),)
    with pytest.raises(TypeError, match="must be real"):
        replace(result(), values=complex_values)
    value = replace(result(), values=complex_values, correlated=True)
    assert value.sum_orders() == (LaurentCoefficients(1 + 2j, -3 + 4j, 5 - 6j),)


def test_empty_batch_and_malformed_shapes():
    empty = replace(result(), born_values=(), values=(), diagnostics=())
    assert empty.shape == (0, 2, 3)
    assert empty.finite == empty.sum_orders() == ()
    with pytest.raises(ValueError, match="dimensions"):
        replace(result(), values=(((1.0, 2.0), (3.0, 4.0)),))
    with pytest.raises(ValueError, match="same lane count"):
        replace(result(), diagnostics=())
    with pytest.raises(ValueError, match="nonfinite"):
        replace(result(), values=(((float("nan"), 1.0, 2.0), (3.0, 4.0, 5.0)),))


def test_invalid_arithmetic_metadata_and_mixed_containers_fail_at_construction():
    with pytest.raises(ValueError, match="53 precision bits"):
        VirtualLaneDiagnostic("success", "double", 256, True)
    with pytest.raises(ValueError, match="106 precision bits"):
        VirtualLaneDiagnostic("success", "double-double", 256, True)
    with pytest.raises(TypeError, match="consistent"):
        replace(result(), values=(((Decimal(1),) * 3, (2.0,) * 3),))


def test_failure_channels_are_immutable_and_keep_physical_identity():
    from pyamplicol import EvaluationError

    failure = VirtualFailureDiagnostic(91, 3, 2, 7, "reduction", "provider detail")
    error = EvaluationError("failed", failures=(failure,))
    assert error.failures == (failure,) and str(error) == "failed"
    assert EvaluationError("ordinary failure").failures == ()
    with pytest.raises(FrozenInstanceError):
        failure.point_id = 2
    for field in ("point_id", "helicity", "order_index", "family_index"):
        with pytest.raises(ValueError):
            replace(failure, **{field: -1})
        with pytest.raises(ValueError):
            replace(failure, **{field: True})
    with pytest.raises(ValueError):
        replace(failure, point_id=None)
    assert replace(failure, helicity=None).helicity is None
