# SPDX-License-Identifier: 0BSD
"""Saved complete-family descriptors preserve their actual producer boundaries."""

import json
import os
from pathlib import Path

import pytest

from .test_loop_portable import _rewrite


@pytest.fixture(scope="module")
def ordered_runtime():
    root = os.environ.get("PYAMPLICOL_ORDERED_COLOR_FIXTURES")
    if not root:
        pytest.skip("actual retained ordered-family artifacts required")
    from pyamplicol import _rusticol
    from pyamplicol._internal.versions import verify_native_module
    from pyamplicol.loops.native import load_loop_native

    verify_native_module(_rusticol)
    assert load_loop_native() is _rusticol._loop
    return Path(root), _rusticol._loop


def _projections(data):
    return (
        data["directed_projection"],
        *(item["directed"] for item in data["spin_projections"]),
    )


def _change(data, operation):
    for projection in _projections(data):
        operation(projection)


def _operations(projection, operations):
    projection["histories"][1].update(bra=operations, ket=operations)
    bank = projection["color_family_projection"]
    bank["families"][bank["history_families"][1]]["order"] = len(operations)


def test_native_complete_family_descriptor_and_positive_fresh_labels(
    ordered_runtime, tmp_path
):
    root, native = ordered_runtime
    source = root / "w-lc-ordered.pacbin"
    plan = native.PortableLoopPlan(str(source))
    data = json.loads(plan.metadata_json())
    bank = data["directed_projection"]["color_family_projection"]
    assert bank["kind"] == "complete-typed-history-families-v1"
    assert [item["order"] for item in bank["families"]] == [0, 1, 1]
    assert [item["virtual_degrees"] for item in bank["families"]] == [[3], [4], [3]]
    assert all(p["color_family_projection"] == bank for p in _projections(data))
    assert data["color_projection"]["virtual_degrees"] == [3]
    assert data.get("born_color_components") is None

    def rename(data):
        for projection in _projections(data):
            for history in projection["histories"]:
                for step in (*history["bra"], *history["ket"]):
                    for name in step.keys() - {"kind"}:
                        if step[name] < 0:
                            step[name] += 2**64

    changed = tmp_path / "positive-fresh.pacbin"
    _rewrite(source, changed, rename)
    native.PortableLoopPlan(str(changed))


@pytest.mark.parametrize(
    "operation,message",
    (
        (lambda p: p.update(color_family_projection=None), "convention"),
        (
            lambda p: p.update(convention="sun-graded-identity-all-outgoing-v1"),
            "convention",
        ),
        (
            lambda p: p["color_family_projection"].update(accuracy="nlc"),
            "family descriptor",
        ),
        (
            lambda p: p["color_family_projection"]["families"][1].update(
                virtual_degrees=[]
            ),
            "degree axes",
        ),
        (
            lambda p: p["color_family_projection"]["families"][0].update(
                virtual_degrees=[99]
            ),
            "identity family degrees",
        ),
        (
            lambda p: p["color_family_projection"]["history_families"].__setitem__(
                1, 99
            ),
            "family descriptor",
        ),
        (
            lambda p: p["color_family_projection"]["families"].append(
                p["color_family_projection"]["families"][0]
            ),
            "unique and referenced",
        ),
        (
            lambda p: p["color_family_projection"]["families"][1].update(
                output_legs=[[0, 3]]
            ),
            "output roles",
        ),
        (
            lambda p: _operations(
                p, [{"kind": "emit-gluon", "emitter": 0, "emitted": 1.5}]
            ),
            "labels must be integers",
        ),
        (
            lambda p: _operations(
                p, [{"kind": "emit-gluon", "emitter": 0, "emitted": True}]
            ),
            "invalid directed colour operation",
        ),
        (
            lambda p: _operations(
                p, [{"kind": "split-gluon", "parent": 0, "quark": -1, "antiquark": -2}]
            ),
            "active gluon",
        ),
        (
            lambda p: _operations(
                p, [{"kind": "split-gluon", "parent": 3, "quark": 3, "antiquark": -2}]
            ),
            "remain fresh",
        ),
        (
            lambda p: _operations(
                p,
                [
                    {"kind": "split-gluon", "parent": 3, "quark": -1, "antiquark": -2},
                    {"kind": "emit-gluon", "emitter": -1, "emitted": -2},
                ],
            ),
            "remain fresh",
        ),
        (
            lambda p: _operations(
                p,
                [
                    {"kind": "emit-gluon", "emitter": 3, "emitted": -1},
                    {"kind": "split-gluon", "parent": -1, "quark": -1, "antiquark": -2},
                ],
            ),
            "remain fresh",
        ),
        (
            lambda p: p["histories"][1].update(
                ket=[{"kind": "emit-gluon", "emitter": 0, "emitted": -2}]
            ),
            "output roles",
        ),
    ),
)
def test_native_rejects_unproved_family_or_typed_operation(
    ordered_runtime, tmp_path, operation, message
):
    root, native = ordered_runtime
    destination = tmp_path / "bad.pacbin"
    _rewrite(root / "w-lc-ordered.pacbin", destination, lambda d: _change(d, operation))
    with pytest.raises(ValueError, match=message):
        native.PortableLoopPlan(str(destination))


def test_native_rejects_matching_changed_spin_metadata(ordered_runtime, tmp_path):
    root, native = ordered_runtime

    def change(data):
        for item in data["spin_projections"]:
            item["directed"]["color_family_projection"]["families"][1][
                "virtual_degrees"
            ] = [99]

    destination = tmp_path / "foreign-spin-family.pacbin"
    _rewrite(root / "w-lc-ordered.pacbin", destination, change)
    with pytest.raises(ValueError, match="history columns differ"):
        native.PortableLoopPlan(str(destination))
