# SPDX-License-Identifier: 0BSD
"""Saved family bilinears keep native layouts, precision and fresh workspaces."""

from decimal import Decimal

import pytest

from pyamplicol.loops.family_projectors import bind_family_projector


@pytest.fixture(scope="module")
def native():
    pytest.importorskip("symbolica")
    return pytest.importorskip("_loop_reduction")


@pytest.fixture(scope="module")
def bilinear_state():
    from symbolica import E, S

    from pyamplicol.loops.physical import _saved

    a, b, c = S("projector_test::a", "projector_test::b", "projector_test::c")
    # Complex directed weights and two ordered outputs, not a premature 2Re.
    return _saved([a.conj() * b + E("1i") * c, a.conj() * c - b, a], [a, b, c])


@pytest.mark.parametrize(
    "arithmetic,precision,engine",
    [
        ("double", None, "interpreted"),
        ("double", None, "jit"),
        ("double-double", None, "interpreted"),
        ("arbitrary", 80, "interpreted"),
    ],
)
def test_native_saved_bilinears_keep_order_ids_tails_and_returned_buffers(
    native, bilinear_state, arithmetic, precision, engine
):
    options = dict(arithmetic=arithmetic, precision=precision)
    actual = bind_family_projector(
        bilinear_state, 3, 3, backend=native, options=options, engine=engine
    )
    reference = native.CoefficientEvaluator(bilinear_state, **options)
    retained = []
    for size in (1, 17, 0, 3, 33):
        ids = [71 + 13 * i for i in range(size)]
        batch = native.NativeBatch.from_decimal(
            [
                value
                for i in range(size)
                for value in ([str(i + 1), "0.25"], ["-2.5", str(i)], ["3", "-0.5"])
            ],
            size,
            3,
            arithmetic,
            precision,
            ids,
        )
        output = actual.evaluate_batch(batch)
        assert output.row_ids == ids and output.columns == 3
        # Dyadic inputs keep these small operations exact in every carrier.
        assert output.to_decimal() == reference.evaluate_batch(batch).to_decimal()
        retained.append((output, output.to_decimal()))
    assert all(output.to_decimal() == values for output, values in retained)
    if engine == "jit":
        assert actual.engine == "jit" and actual.tile_rows == 16


@pytest.mark.parametrize(
    "arithmetic,precision,digits", [("double-double", None, 25), ("arbitrary", 80, 60)]
)
def test_interpreter_preserves_sub_binary64_input(
    native, bilinear_state, arithmetic, precision, digits
):
    options = dict(arithmetic=arithmetic, precision=precision)
    projector = bind_family_projector(
        bilinear_state, 3, 3, backend=native, options=options
    )
    value = "1." + "0" * (digits - 1) + "1"
    batch = native.NativeBatch.from_decimal(
        [[value, "0"], ["0", "0"], ["0", "0"]], 1, 3, arithmetic, precision, [5]
    )
    result = projector.evaluate_batch(batch).to_decimal()
    assert Decimal(result[2][0]) != 1
    assert result[2] == batch.to_decimal()[0]


def test_projector_binding_rejects_unsupported_engine_and_carrier(
    native, bilinear_state
):
    for options, engine, message in (
        ({"arithmetic": "double", "precision": None}, "auto", "interpreted or jit"),
        ({"arithmetic": "double-double", "precision": None}, "jit", "requires double"),
        ({"arithmetic": "arbitrary", "precision": 80}, "jit", "requires double"),
    ):
        with pytest.raises(ValueError, match=message):
            bind_family_projector(
                bilinear_state, 3, 3, backend=native, options=options, engine=engine
            )
    projector = bind_family_projector(
        bilinear_state,
        3,
        3,
        backend=native,
        options={"arithmetic": "double", "precision": None},
        engine="jit",
    )
    wrong_carrier = native.NativeBatch.from_decimal(
        [["0", "0"]] * 3, 1, 3, "arbitrary", 80, [3]
    )
    with pytest.raises(ValueError, match=r"arithmetic|precision"):
        projector.evaluate_batch(wrong_carrier)
