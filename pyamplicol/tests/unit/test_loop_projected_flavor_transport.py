# SPDX-License-Identifier: 0BSD
"""Projected d/s transport against independently prepared complete members."""

import json
import os
from copy import copy
from dataclasses import replace
from pathlib import Path

import pytest

_LIVE_PROJECTED_PAIR = None
_LIVE_PROJECTED_CATALOGUES = None


@pytest.fixture(scope="module")
def owned_pair():
    if _LIVE_PROJECTED_PAIR is None:
        pytest.skip("retained independently generated formal source owners required")
    return _LIVE_PROJECTED_PAIR


@pytest.fixture(scope="module")
def owned_catalogues(owned_pair):
    if _LIVE_PROJECTED_CATALOGUES is None:
        pytest.skip("retained paired projected source catalogues required")
    return *owned_pair, _LIVE_PROJECTED_CATALOGUES


@pytest.mark.parametrize("accuracy", ["lc", "nlc", "full"])
def test_actual_identity_source_catalogue_admission(owned_catalogues, accuracy):
    from pyamplicol.generation.process_grouping import LegMapping
    from pyamplicol.loops.flavor_transport import IdentityFlavorTransport

    proof, views, catalogues = owned_catalogues
    left, right = (views[name, accuracy] for name in ("ddgg", "ssgg"))
    a, b = (catalogues[name, accuracy] for name in ("ddgg", "ssgg"))
    relation = IdentityFlavorTransport.admit(
        proof,
        left,
        right,
        LegMapping((0, 1, 2, 3), False, "exact-loop-species"),
        representative_directed=a[0],
        member_directed=b[0],
        representative_spin=a[1:],
        member_spin=b[1:],
    )
    assert {item.bare.spin_layout.legs for item in relation.representative_spin} == {
        (3,),
        (4,),
        (3, 4),
    }
    assert relation.representative_directed is a[0]
    assert relation.member_directed is b[0]


@pytest.mark.parametrize("accuracy", ["lc", "nlc", "full"])
def test_projected_transport_does_not_admit_unproved_ordered_histories(
    owned_catalogues, accuracy
):
    from pyamplicol.color.connections import ColorConnection, EmitGluon
    from pyamplicol.generation.process_grouping import LegMapping
    from pyamplicol.loops.flavor_transport import IdentityFlavorTransport

    proof, views, catalogues = owned_catalogues
    left, right = (views[name, accuracy] for name in ("ddgg", "ssgg"))
    plans = []
    for name in ("ddgg", "ssgg"):
        actual = catalogues[name, accuracy][0]
        connection = ColorConnection(actual.bare.color_legs, (EmitGluon(0, -1),))
        plans.append(
            replace(
                actual, bare=replace(actual.bare, histories=((connection, connection),))
            )
        )
    with pytest.raises(NotImplementedError, match="identity colour histories"):
        IdentityFlavorTransport.admit(
            proof,
            left,
            right,
            LegMapping((0, 1, 2, 3), False, "exact-loop-species"),
            representative_directed=plans[0],
            member_directed=plans[1],
        )


def _copy_source(source):
    model, bare = copy(source.virtual), copy(source.virtual.bare)
    model.bare = bare
    return replace(
        source, virtual=model, source_bare=replace(source.source_bare, bare=bare)
    )


@pytest.mark.parametrize("accuracy", ["lc", "nlc", "full"])
def test_actual_formal_transport_admission_and_retained_axis(owned_pair, accuracy):
    from pyamplicol.generation.process_grouping import LegMapping
    from pyamplicol.loops.flavor_transport import IdentityFlavorTransport

    proof, views = owned_pair
    left, right = (views[name, accuracy] for name in ("ddgg", "ssgg"))
    relation = IdentityFlavorTransport.admit(
        proof, left, right, LegMapping((0, 1, 2, 3), False, "exact-loop-species")
    )
    assert relation.representative_plan is left.prepared
    assert relation.member_plan is right.prepared
    assert left.prepared is not right.prepared
    assert relation.bind(left).representative is left
    for source in (left, right):
        assert source.virtual.source_color is proof.plan.source_color
        assert dict(source.prepared.metadata)["source_color"] is proof.plan.source_color
        assert (
            source.rebind(arithmetic="double").virtual.source_color
            is proof.plan.source_color
        )
    if accuracy == "lc":
        a, b = (
            left.virtual.bare.born_color_components,
            right.virtual.bare.born_color_components,
        )
        assert a.ids == b.ids and a.members != b.members


@pytest.mark.parametrize("accuracy", ["lc", "nlc", "full"])
@pytest.mark.parametrize("replacement", ["absent", "foreign", "paired-substitution"])
def test_actual_formal_transport_rejects_foreign_declaration(
    owned_pair, accuracy, replacement
):
    from pyamplicol.generation.process_grouping import LegMapping
    from pyamplicol.loops.flavor_transport import IdentityFlavorTransport

    proof, views = owned_pair
    left, right = (_copy_source(views[name, accuracy]) for name in ("ddgg", "ssgg"))
    changed = None if replacement == "absent" else replace(proof.plan.source_color)
    left.virtual.source_color = changed
    if replacement == "paired-substitution":
        right.virtual.source_color = changed
        proof = replace(proof, plan=replace(proof.plan, source_color=changed))
    with pytest.raises(ValueError, match="source colour owner"):
        IdentityFlavorTransport.admit(
            proof, left, right, LegMapping((0, 1, 2, 3), False, "exact-loop-species")
        )


@pytest.mark.parametrize(
    "replacement", ["ids", "members", "typed-ports", "degree", "formal-owner"]
)
def test_actual_formal_transport_rejects_changed_component_authority(
    owned_pair, replacement
):
    from pyamplicol.generation.process_grouping import LegMapping
    from pyamplicol.loops.flavor_transport import IdentityFlavorTransport

    proof, views = owned_pair
    left, right = (_copy_source(views[name, "lc"]) for name in ("ddgg", "ssgg"))
    bare = left.virtual.bare
    if replacement in {"ids", "members"}:
        components = bare.born_color_components
        bare.born_color_components = replace(
            components,
            **{replacement: tuple(reversed(getattr(components, replacement)))},
        )
    elif replacement == "typed-ports":
        from symbolica import S

        tensor = bare.born_color_tensors[0]
        axis = next(iter(tensor.axes))
        bare.born_color_tensors = (
            tensor.rename_indices({axis.index: S("flavor_test::foreign_port")}),
            *bare.born_color_tensors[1:],
        )
    elif replacement == "degree":
        bare.virtual_color_degrees = tuple(
            value + 1 for value in bare.virtual_color_degrees
        )
    else:
        bare.formal_color_metrics = replace(bare.formal_color_metrics)
    with pytest.raises(
        ValueError, match=r"formal colour|formal projection|colour components"
    ):
        IdentityFlavorTransport.admit(
            proof, left, right, LegMapping((0, 1, 2, 3), False, "exact-loop-species")
        )


@pytest.fixture(scope="module")
def saved_pair():
    location = os.environ.get("PYAMPLICOL_PROJECTED_FLAVOR_FIXTURES")
    original = os.environ.get("PYAMPLICOL_FIXED_FULL_FLAVOR_FIXTURES")
    if not location or not original:
        pytest.skip("retained independent projected and fixed-full sources required")
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
