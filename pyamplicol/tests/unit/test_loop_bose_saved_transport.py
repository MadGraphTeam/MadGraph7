# SPDX-License-Identifier: 0BSD
"""Actual admitted Bose members through the current saved execution boundary."""

import json
from decimal import localcontext
from pathlib import Path

import pytest

from .test_loop_portable import _rewrite
from .test_loop_projected_flavor_runtime import _close_values
from .test_loop_projected_flavor_transport import _compare

pytestmark = pytest.mark.loop_fast


@pytest.fixture(scope="module")
def saved_bose(saved_loop_fixture):
    location = str(saved_loop_fixture("bose-permutation"))
    from pyamplicol import _rusticol
    from pyamplicol._internal.versions import verify_native_module
    from pyamplicol.loops.native import load_loop_native

    assert load_loop_native() is _rusticol._loop
    verify_native_module(_rusticol)
    reference = (
        Path(__file__).resolve().parents[2] / "tools/loop_reference/uudd_qcd.json"
    )
    points = json.loads(reference.read_text(), parse_float=str, parse_int=str)["points"]
    return Path(location), _rusticol._loop, points


@pytest.mark.parametrize(
    "arithmetic,precision,engine,tolerance",
    [
        ("double", None, "fast", "3e-9"),
        ("double-double", None, "generic", "3e-24"),
        ("arbitrary", 70, "generic", "3e-54"),
    ],
)
def test_saved_bose_member_all_complete_components(
    saved_bose, arithmetic, precision, engine, tolerance
):
    root, native, points = saved_bose
    parameters = {"aS": ("0.118", "0"), "MT": ("173", "0"), "MU_R": ("1000", "0")}
    mapped = native.NativeVirtualRuntime(
        str(root / "ordinary-artifact"),
        process="ssgg-swap",
        parameters=parameters,
        arithmetic=arithmetic,
        precision=precision,
    )
    independent = native.NativeLoopProgram(
        str(root / "ssgg-independent.pacbin"),
        parameters=parameters,
        arithmetic=arithmetic,
        precision=precision,
        reduction_engine=engine,
    )
    metadata = json.loads(mapped.metadata_json())
    assert metadata["process_id"] == "ssgg-swap"
    assert metadata["representative_process_id"] == "ddgg"
    assert metadata["external_permutation"] == [0, 1, 3, 2]
    assert metadata["physics_scope"] == "selected-process"
    assert not metadata["physics"]["resolved"]
    assert metadata["representative_physics"]["resolved"]
    for size in (0, 1, 3) if arithmetic == "double" else (1,):
        batch = [points[1], points[0], points[1]][:size]
        ids = [91, 7, 2**64 - 1][:size]
        scales = ["900000", "1000000", "900000"][:size]
        _compare(
            mapped.evaluate(batch, scales, point_ids=ids),
            independent.evaluate(batch, scales, point_ids=ids),
            tolerance,
            ids,
        )
    with pytest.raises(ValueError, match=r"resolved.*transport"):
        mapped.evaluate_resolved(points[:1], ["1000000"])


def test_selected_bose_capabilities_keep_representative_plans(saved_bose):
    from pyamplicol import Runtime
    from pyamplicol.api.errors import CompatibilityError

    root, native, points = saved_bose
    for mode in ("ordinary", "spin"):
        path = root / f"{mode}-artifact"
        selected = Runtime.load(path, process="ssgg-swap")
        representative = Runtime.load(path, process="ddgg")
        assert selected.external_permutation == (0, 1, 3, 2)
        assert representative.external_permutation == (0, 1, 2, 3)
        assert selected.representative_process_key == "ddgg"
        assert tuple(p.pdg_id for p in selected.physics.external_particles) == (
            3,
            -3,
            21,
            21,
        )
        info, original = selected.inspect(), representative.inspect()
        assert info["external_permutation"] == [0, 1, 3, 2]
        assert info["physics_scope"] == "selected-process"
        assert info["resolved"] is False and info["correlations"] is False
        assert info["color_correlations"] == []
        assert info["spin_correlations"] == ()
        assert info["representative_physics"]["resolved"] == (mode == "ordinary")
        assert info["representative_physics"]["correlations"] == (mode == "spin")
        assert original["resolved"] == (mode == "ordinary")
        assert original["correlations"] == (mode == "spin")
        # A directed representative still admits the member's ordinary result.
        actual = selected.evaluate(points[:1], mu_r2="1000000")
        assert actual.shape == (1, 1, 3)
        selected.clear()
        assert selected.evaluate(points[:1], mu_r2="1000000").values == actual.values
        with pytest.raises(CompatibilityError, match="resolved"):
            selected.evaluate_resolved(points[:1], mu_r2="1000000")
        with pytest.raises(CompatibilityError, match=r"directed|correlation|histories"):
            selected.evaluate_correlated(points[:1], mu_r2="1000000")
        raw = native.NativeVirtualRuntime(str(path), process="ssgg-swap")
        with pytest.raises(ValueError, match=r"directed.*transport"):
            raw.evaluate_directed(points[:1], ["1000000"])
        if mode == "ordinary":
            resolved = representative.evaluate_resolved(points[:1], mu_r2="1000000")
            _close_values(
                resolved.total().values,
                representative.evaluate(points[:1], mu_r2="1000000").values,
            )
        else:
            assert original["color_correlations"] and original["spin_correlations"]
            correlated = representative.evaluate_correlated(points[:1], mu_r2="1000000")
            assert correlated.shape == (1, 1, 3)


def test_bose_subset_rescue_keeps_exact_input_ids(saved_bose):
    from pyamplicol import Runtime
    from pyamplicol.api import VirtualArithmeticSpec, VirtualStabilityPolicy

    root, _, points = saved_bose
    runtime = Runtime.load(root / "ordinary-artifact", process="ssgg-swap")
    with localcontext() as context:
        context.prec = 100
        result = runtime.evaluate(
            [points[0]] * 3,
            mu_r2=["1000000", "1e-400", "1000000"],
            stability=VirtualStabilityPolicy(
                (
                    VirtualArithmeticSpec("arbitrary", 60),
                    VirtualArithmeticSpec("arbitrary", 80),
                ),
                "1e-8",
                "1e-9",
            ),
        )
    assert [row.status for row in result.diagnostics] == [
        "success",
        "rescued",
        "success",
    ]
    assert result.values[0] == result.values[2]
    assert result.diagnostics[1].verification.point_id == 1
    assert {
        failure.point_id
        for attempt in result.diagnostics[1].verification.attempts
        for failure in attempt.failures
    } == {1}
    assert [
        attempt.arithmetic.arithmetic
        for attempt in result.diagnostics[1].verification.attempts
    ][-2:] == [
        "arbitrary",
        "arbitrary",
    ]


@pytest.mark.parametrize("permutation", [[0, 1, 2, 2], [2, 1, 0, 3], [1, 0, 2, 3]])
def test_saved_member_rejects_invalid_or_unproved_permutation(
    saved_bose, tmp_path, permutation
):
    root, native, _ = saved_bose

    def change(plan):
        assert len(plan["admitted_flavor_members"]) == 1
        plan["admitted_flavor_members"][0]["permutation"] = permutation

    path = tmp_path / "unproved.pacbin"
    _rewrite(root / "admitted-ordinary.pacbin", path, change)
    with pytest.raises(ValueError, match=r"permutation|boson|external"):
        native.PortableLoopPlan(str(path))
