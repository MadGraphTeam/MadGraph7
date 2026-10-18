# SPDX-License-Identifier: 0BSD
"""Literal vector-source ingress and compact native spectator layouts."""

from decimal import Decimal, localcontext

import pytest

from pyamplicol.loops.physical_process import quark_gluon_qcd_program
from pyamplicol.loops.spin_sources import NativeSpinSourcePlan, replace_vector_sources


@pytest.fixture(scope="module", params=("double", "double-double", "arbitrary"))
def bare(request):
    symbolica = pytest.importorskip("symbolica")
    if not symbolica.is_licensed():
        pytest.skip("licensed dependency generation required")
    backend = pytest.importorskip("_loop_reduction")
    return quark_gluon_qcd_program(
        backend=backend,
        arithmetic=request.param,
        precision=80 if request.param == "arbitrary" else None,
        reduction_engine="fast" if request.param == "double" else "generic",
    )


def test_native_literal_components_joint_layout_and_original_bits(bare):
    plan = NativeSpinSourcePlan(bare, ((3,), (3, 4)))
    assert plan.source_ir.crossing.phase == (1, 0)
    assert len(plan.layouts[()].helicities) == 16
    assert len(plan.layouts[(3,)].helicities) == 8
    assert len(plan.layouts[(3, 4)].helicities) == 4
    assert bare.helicities == plan.layouts[()].helicities
    assert all(h[2:] == (-1, -1) for h in plan.layouts[(3, 4)].helicities)
    exponent = {"double": -40, "double-double": -90, "arbitrary": -200}[bare.arithmetic]
    with localcontext() as context:
        context.prec = 110
        tiny = Decimal(1) + Decimal(2) ** exponent
        raw = {
            3: (((tiny, Decimal(3)), 2, -4, 5), ((tiny, Decimal(-7)), 9, 11, 13)),
            4: ((Decimal(2), Decimal(-5)), 17, 19, 23),
        }
        prepared = plan.prepare_decimal(raw, [41, 99])
        assert prepared[3].row_ids == prepared[4].row_ids == [41, 99]
        first, second = prepared[3].to_decimal()[0], prepared[3].to_decimal()[4]
        assert abs(Decimal(first[0]) - tiny) < (tiny - 1) / 1000
        assert abs(Decimal(second[0]) - tiny) < (tiny - 1) / 1000
        assert Decimal(first[1]) == 3 and Decimal(second[1]) == -7
        assert tuple(map(Decimal, prepared[4].to_decimal()[0])) == (2, -5)
        assert prepared[4].to_decimal()[:4] == prepared[4].to_decimal()[4:]
        # Every component, including the temporal and imaginary components,
        # enters the actual epsilon_bar slot unchanged by conjugation.
        rows, ids = 8, list(range(80, 88))
        markers = bare.backend.NativeBatch.from_decimal(
            [(str(i), "0") for i in range(rows * 16)],
            rows,
            16,
            row_ids=ids,
            **bare.options,
        )
        values = replace_vector_sources(
            markers,
            prepared,
            point_rows=[0] * 4 + [1] * 4,
            lane_ids=ids,
            state_kinds=bare.state_kinds,
            backend=bare.backend,
        )
        assert values.row_ids == ids
        original, replaced = markers.to_decimal(), values.to_decimal()
        for row in range(rows):
            assert (
                replaced[16 * row : 16 * row + 8] == original[16 * row : 16 * row + 8]
            )
            assert (
                replaced[16 * row + 8 : 16 * row + 12]
                == prepared[3].to_decimal()[4 * (row // 4) : 4 * (row // 4 + 1)]
            )
            assert (
                replaced[16 * row + 12 : 16 * row + 16] == prepared[4].to_decimal()[:4]
            )
    assert plan.layout(None) == plan.layout({}) == plan.layouts[()]
    empty = plan.prepare_decimal({3: ((1, 2), 3, 4, 5)}, [])
    assert (empty[3].rows, empty[3].columns, empty[3].row_ids) == (0, 4, [])


def test_spin_source_declarations_shape_precision_and_failure_ids(bare):
    plan = NativeSpinSourcePlan(bare, ((3,), (3, 4)))
    native = bare.backend.NativeBatch
    vector = native.from_decimal([("1", "0")] * 4, 1, 4, **bare.options)
    with pytest.raises(ValueError, match="joint"):
        plan.prepare_native({4: vector}, [0])
    with pytest.raises(ValueError, match="one-based"):
        plan.prepare_native({True: vector}, [0])
    with pytest.raises(ValueError, match="one row"):
        plan.prepare_native({3: vector.take_columns([0, 1])}, [0])
    rows = vector.take_rows([0, 0], row_ids=[1, 2])
    with pytest.raises(ValueError, match="identities"):
        plan.prepare_native({3: rows}, [9, 10])
    foreign = native.from_decimal(
        [("1", "0")] * 4,
        1,
        4,
        arithmetic="double-double" if bare.arithmetic == "double" else "double",
    )
    with pytest.raises(ValueError):
        plan.prepare_native({3: foreign}, [0])
    with pytest.raises(ValueError):
        plan.prepare_native({3: foreign}, [])
    with pytest.raises(NotImplementedError, match="massless external gluon"):
        NativeSpinSourcePlan(bare, ((1,),))
    # Create nonfinite values through a native dependency evaluator so every
    # arithmetic carrier exercises the same checked source boundary.
    from symbolica import E, S

    from pyamplicol.loops.physical import _saved

    x = S("spin_source_test::x")
    evaluator = bare.backend.CoefficientEvaluator(
        _saved([1 / x, E("0"), E("0"), E("0")], [x]), **bare.options
    )
    bad = evaluator.evaluate_batch(
        native.from_decimal(
            [("1", "0"), ("0", "0")], 2, 1, row_ids=[51, 74], **bare.options
        )
    )
    with pytest.raises(ArithmeticError, match="74"):
        plan.prepare_native({3: bad}, [51, 74])
