# SPDX-License-Identifier: 0BSD
"""Projected d/s transport against independently prepared complete members."""

import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.loop_physics


@pytest.fixture(scope="module")
def saved_pair(saved_loop_fixture):
    location = str(saved_loop_fixture("projected-flavor-v43ff063"))
    original = str(saved_loop_fixture("fixed-full-flavor"))
    from pyamplicol import _rusticol
    from pyamplicol._internal.versions import verify_native_module
    from pyamplicol.loops.native import load_loop_native

    assert load_loop_native() is _rusticol._loop
    verify_native_module(_rusticol)
    point_file = (
        Path(__file__).resolve().parents[2] / "tools/loop_reference/uudd_qcd.json"
    )
    points = json.loads(point_file.read_text(), parse_float=str, parse_int=str)[
        "points"
    ]
    return Path(location), Path(original), _rusticol._loop, points


def _blocks(result):
    return result.born, *result.component_batches, result.totals


def _compare(left, right, tolerance, ids):
    assert left.point_ids == right.point_ids == ids
    assert not left.failures and not right.failures
    for a, b in zip(_blocks(left), _blocks(right), strict=True):
        assert a.rows == b.rows and a.columns == b.columns and a.row_ids == b.row_ids
        assert all(ok for _, ok in a.agreement_rows(b, tolerance, tolerance))


@pytest.mark.parametrize(
    "arithmetic,precision,engine,tolerance",
    [
        ("double", None, "fast", "3e-9"),
        ("double-double", None, "generic", "3e-24"),
        ("arbitrary", 70, "generic", "3e-54"),
    ],
)
def test_formal_full_equals_independent_original_model(
    saved_pair, arithmetic, precision, engine, tolerance
):
    root, original, native, points = saved_pair
    options = dict(arithmetic=arithmetic, precision=precision, reduction_engine=engine)
    parameters = {"aS": ("0.118", "0"), "MT": ("173", "0"), "MU_R": ("1000", "0")}
    for name in ("ddgg", "ssgg"):
        projected = native.NativeLoopProgram(
            str(root / f"{name}-full-owned-ordinary.pacbin"),
            parameters=parameters | {"MB": ("0", "0"), "ymb": ("0", "0")},
            **options,
        )
        fixed = native.NativeLoopProgram(
            str(original / f"independent-{name}.pacbin"),
            parameters=parameters,
            **options,
        )
        _compare(
            projected.evaluate(points[:1], ["1000000"], point_ids=[91]),
            fixed.evaluate(points[:1], ["1000000"], point_ids=[91]),
            tolerance,
            [91],
        )


@pytest.mark.parametrize("accuracy", ["lc", "nlc"])
@pytest.mark.parametrize(
    "arithmetic,precision,engine,tolerance",
    [
        ("double", None, "fast", "3e-9"),
        ("double-double", None, "generic", "3e-24"),
        ("arbitrary", 70, "generic", "3e-54"),
    ],
)
def test_independent_projected_members_keep_all_signed_axes(
    saved_pair, accuracy, arithmetic, precision, engine, tolerance
):
    root, _, native, points = saved_pair
    options = dict(arithmetic=arithmetic, precision=precision, reduction_engine=engine)
    left, right = (
        native.NativeLoopProgram(
            str(root / f"{name}-{accuracy}-owned-resolved.pacbin"), **options
        )
        for name in ("ddgg", "ssgg")
    )
    for size in (0, 1, 3) if arithmetic == "double" else (1,):
        ids, scales = [91, 7, 2**64 - 1][:size], ["1000000", "900000", "800000"][:size]
        chosen = [points[1], points[0], points[1]][:size]
        for resolved in (False, True):
            _compare(
                left.evaluate(chosen, scales, point_ids=ids, resolved=resolved),
                right.evaluate(chosen, scales, point_ids=ids, resolved=resolved),
                tolerance,
                ids,
            )


@pytest.mark.parametrize("accuracy", ["lc", "nlc"])
@pytest.mark.parametrize(
    "arithmetic,precision,engine,tolerance",
    [
        ("double", None, "fast", "3e-9"),
        ("double-double", None, "generic", "3e-24"),
        ("arbitrary", 70, "generic", "3e-54"),
    ],
)
def test_independent_projected_identity_sources_preserve_both_terms(
    saved_pair, accuracy, arithmetic, precision, engine, tolerance
):
    from .test_loop_generation_spin_flavor_runtime import _vectors

    root, _, native, points = saved_pair
    options = dict(arithmetic=arithmetic, precision=precision, reduction_engine=engine)
    left = native.NativeVirtualRuntime(
        str(root / f"{accuracy}-spin-mapped-artifact"),
        process="ssgg",
        arithmetic=arithmetic,
        precision=precision,
    )
    right = native.NativeLoopProgram(
        str(root / f"ssgg-{accuracy}-current-spin.pacbin"), **options
    )
    first, second = _vectors()
    for selected in ((), (3,), (4,), (3, 4)):
        vectors = {leg: [first if leg == 3 else second] for leg in selected}
        a, b = (
            program.evaluate_directed(
                points[:1], ["1000000"], point_ids=[91], spin_vectors=vectors
            )
            for program in (left, right)
        )
        assert a.spin_legs == b.spin_legs == list(selected)
        _compare(a, b, tolerance, [91])
        for x, y in (
            (a.directed_first, b.directed_first),
            (a.directed_second, b.directed_second),
        ):
            assert (
                x.rows == y.rows and x.columns == y.columns and x.row_ids == y.row_ids
            )
            assert all(ok for _, ok in x.agreement_rows(y, tolerance, tolerance))
