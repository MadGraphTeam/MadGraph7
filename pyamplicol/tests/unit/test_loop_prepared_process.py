# SPDX-License-Identifier: 0BSD
"""Complete arithmetic views reuse cold physics and own numerical workspaces."""

import json
import os
from dataclasses import FrozenInstanceError
from decimal import Decimal, localcontext
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(scope="module")
def controls():
    symbolica = pytest.importorskip("symbolica")
    if not symbolica.is_licensed():
        pytest.skip("licensed dependency generation required")
    pytest.importorskip("_loop_reduction")
    model_path = os.environ.get("PYAMPLICOL_NLO_UFO_REFERENCE")
    if not model_path:
        pytest.skip("set PYAMPLICOL_NLO_UFO_REFERENCE")
    from pyamplicol.loops.model import build_loop_model
    from pyamplicol.loops.physical_counterterms import (
        distinct_flavour_model_virtual_program,
        drell_yan_model_virtual_program,
    )
    from pyamplicol.models.loop_import import load_raw_loop_model

    source = build_loop_model(load_raw_loop_model(model_path))
    root = Path(__file__).resolve().parents[2] / "tools/loop_reference"
    return tuple(
        (
            factory(source, reduction_engine="fast", coefficient_engine="jit"),
            np.asarray(json.loads((root / fixture).read_text())["points"], dtype=str),
            model_values,
        )
        for factory, fixture, model_values in (
            (
                drell_yan_model_virtual_program,
                "drell_yan_qcd.json",
                ["0.118", "132.507"],
            ),
            (distinct_flavour_model_virtual_program, "uudd_qcd.json", ["0.118", "173"]),
        )
    )


def forbid_cold_work(monkeypatch):
    from pyamplicol.loops import physical_counterterms, physical_process
    from pyamplicol.loops.color_bilinears import PhysicalColorSpace
    from pyamplicol.loops.kernels import CoefficientKernel, PreparedCoefficientDAG
    from pyamplicol.loops.kinematics import LoopKinematicPlan

    def forbidden(*args, **kwargs):
        raise AssertionError("arithmetic view repeated cold physics preparation")

    for owner, name in (
        (physical_process, "build_color_loop_recurrence"),
        (physical_process, "build_color_tree_recurrence"),
        (physical_process, "build_loop_kinematics"),
        (physical_counterterms, "build_color_tree_recurrence"),
        (physical_process, "_saved"),
        (physical_counterterms, "_saved"),
        (physical_counterterms, "_saved_scalar_pairs"),
        (PhysicalColorSpace, "overlap"),
        (CoefficientKernel, "evaluator_bytes"),
        (PreparedCoefficientDAG, "from_dag"),
        (LoopKinematicPlan, "evaluator_bytes"),
    ):
        monkeypatch.setattr(owner, name, forbidden)


def batches(result):
    return (result.born, *result.component_batches, result.totals)


@pytest.mark.parametrize("case", [0, 1])
def test_complete_views_retain_exact_physics_and_independent_buffers(
    controls, monkeypatch, case
):
    program, points, values = controls[case]
    prepared = program.prepared
    baseline = program.evaluate_ordered_native(
        points, ["1000000"] * 3, model_values=values
    )
    reference = tuple(batch.to_decimal() for batch in batches(baseline))
    forbid_cold_work(monkeypatch)
    views = [
        program.rebind(
            arithmetic=mode,
            precision=precision,
            reduction_engine=engine,
            coefficient_engine=compiler,
        )
        for mode, precision, engine, compiler in (
            ("double", None, "fast", "jit"),
            ("double-double", None, "generic", "interpreted"),
            ("arbitrary", 80, "generic", "interpreted"),
        )
    ]
    assert all(view.prepared is prepared for view in views)
    assert all(view.bare.prepared is prepared.bare for view in views)
    for view in views:
        assert view.bare.model_evaluator is not program.bare.model_evaluator
        assert view.bare.families[0].reducer is not program.bare.families[0].reducer
        assert view.bare.loop_executors[0].plan.workspace_capacity == (0, 0, 0)
        assert (
            view.bare.loop_executors[0].prepared
            is program.bare.loop_executors[0].prepared
        )
        result = view.evaluate_ordered_native(
            points, ["1000000"] * 3, model_values=values
        )
        assert result.statuses == baseline.statuses
        for actual, expected in zip(batches(result), batches(baseline), strict=True):
            if view.bare.arithmetic == "double":
                assert actual.to_decimal() == expected.to_decimal()
            else:
                assert all(
                    good for _, good in actual.agreement_rows(expected, "1e-8", "1e-12")
                )
        changed = view.evaluate_ordered_native(
            points[:1], ["700000"], model_values=["0.12", values[1]]
        )
        assert (
            changed.virtual.to_decimal() != result.virtual.take_rows([0]).to_decimal()
        )
        empty = view.evaluate_ordered_native(points[:0], [], model_values=values)
        assert empty.born.rows == 0 and empty.virtual.rows == 0
        restored = view.evaluate_ordered_native(
            points, ["1000000"] * 3, model_values=values
        )
        assert tuple(batch.to_decimal() for batch in batches(restored)) == tuple(
            batch.to_decimal() for batch in batches(result)
        )
    assert tuple(batch.to_decimal() for batch in batches(baseline)) == reference
    assert (
        program.bare.coefficient_kernel_cache
        is not views[0].bare.coefficient_kernel_cache
    )
    with pytest.raises(FrozenInstanceError):
        prepared.bare.color_metrics_revision = 99


def test_retained_precision_view_uses_original_small_coupling_bits(
    controls, monkeypatch
):
    program, points, values = controls[0]
    from pyamplicol.loops.stability import ArithmeticSpec, NativeModelStabilityProgram

    wrapper = NativeModelStabilityProgram(
        program,
        [ArithmeticSpec("arbitrary", 80), ArithmeticSpec("arbitrary", 100)],
        relative_tolerance="1e-10",
        absolute_tolerance="1e-15",
    )
    forbid_cold_work(monkeypatch)
    view = wrapper._view(ArithmeticSpec("arbitrary", 80))
    with localcontext() as context:
        context.prec = 110
        changed_alpha = Decimal(values[0]) + Decimal(2) ** -180
        left = view.evaluate_ordered_native(
            points[:1], ["1000000"], model_values=values
        )
        right = view.evaluate_ordered_native(
            points[:1], ["1000000"], model_values=[str(changed_alpha), values[1]]
        )
        a = Decimal(left.virtual.to_decimal()[2][0])
        b = Decimal(right.virtual.to_decimal()[2][0])
        assert a != b
        assert abs(b - a * changed_alpha / Decimal(values[0])) < abs(a) * Decimal(
            "1e-75"
        )


def test_color_projection_replaces_retained_weight_instructions(controls):
    program, points, values = controls[0]
    bare = program.bare.rebind(arithmetic="double", reduction_engine="fast")
    old = bare.prepared
    before = bare.evaluate_ordered_native(points[:1], ["1000000"], model_values=values)
    bare.set_color_metrics(
        tuple(tuple(2 * x for x in row) for row in bare.born_metric),
        tuple(tuple(2 * x for x in row) for row in bare.loop_metric),
    )
    updated = bare.prepared
    assert (
        updated is not old
        and updated.color_metrics_revision == old.color_metrics_revision + 1
    )
    new_view = bare.rebind(arithmetic="double", reduction_engine="fast")
    after = new_view.evaluate_ordered_native(
        points[:1], ["1000000"], model_values=values
    )
    with localcontext() as context:
        context.prec = 50
        for a, b in zip(
            after.virtual.to_decimal(), before.virtual.to_decimal(), strict=True
        ):
            assert abs(Decimal(a[0]) - 2 * Decimal(b[0])) < Decimal("1e-15")
    # The earlier frozen plan retains its own weights, with fresh numeric state.
    previous = old.bind(
        backend=bare.backend, arithmetic="double", reduction_engine="fast"
    )
    replay = previous.evaluate_ordered_native(
        points[:1], ["1000000"], model_values=values
    )
    assert replay.virtual.to_decimal() == before.virtual.to_decimal()


def test_retained_external_fields_have_fresh_scalar_workspaces(monkeypatch, controls):
    from pyamplicol.loops.model import build_loop_model
    from pyamplicol.loops.physical_counterterms import top_pair_model_virtual_program
    from pyamplicol.models.loop_import import load_raw_loop_model

    source = build_loop_model(
        load_raw_loop_model(os.environ["PYAMPLICOL_NLO_UFO_REFERENCE"])
    )
    program = top_pair_model_virtual_program(source, reduction_engine="fast")
    path = (
        Path(__file__).resolve().parents[2] / "tools/loop_reference/top_pair_qcd.json"
    )
    points = np.asarray(json.loads(path.read_text())["points"], dtype=str)[:1]
    expected = program.evaluate_ordered_native(
        points, ["1000000"], model_values=["0.118", "173"]
    )
    assert program.field_currents
    assert any(
        Decimal(pair[0]) != 0 for pair in expected.uv_external_fields.to_decimal()
    )
    forbid_cold_work(monkeypatch)
    view = program.rebind(arithmetic="double-double")
    assert all(
        left.coefficient_evaluator is not right.coefficient_evaluator
        for left, right in zip(program.field_currents, view.field_currents, strict=True)
    )
    actual = view.evaluate_ordered_native(
        points, ["1000000"], model_values=["0.118", "173"]
    )
    for left, right in zip(batches(actual), batches(expected), strict=True):
        assert all(good for _, good in left.agreement_rows(right, "1e-8", "1e-12"))


@pytest.mark.parametrize("accuracy", ["full", "lc", "nlc"])
def test_standalone_formal_projection_cannot_lose_its_scope_in_a_view(
    controls, accuracy
):
    from symbolica import S

    from pyamplicol.loops.physical_color import project_physical_color

    bare = controls[0][0].bare.rebind(arithmetic="double", reduction_engine="fast")
    assert bare.prepared is controls[0][0].prepared.bare
    project_physical_color(bare, S("prepared_scope::Nc"), [], accuracy)
    assert bare.born_color_degree is not None
    with pytest.raises(NotImplementedError, match="unprojected"):
        bare.rebind(arithmetic="double-double")


@pytest.fixture(scope="module")
def direct_control():
    symbolica = pytest.importorskip("symbolica")
    pytest.importorskip("_loop_reduction")
    source_path = os.environ.get("PYAMPLICOL_NLO_UFO_REFERENCE")
    reference_path = os.environ.get("PYAMPLICOL_GG_COMPONENT_REFERENCE")
    if not symbolica.is_licensed() or not source_path or not reference_path:
        pytest.skip("licensed loop_sm and independent gg component reference required")
    from pyamplicol.loops.model import build_loop_model
    from pyamplicol.loops.physical_counterterms import (
        NativeModelVirtualProgram,
        gluon_scattering_model_virtual_program,
    )
    from pyamplicol.models.loop_import import load_raw_loop_model

    source = build_loop_model(load_raw_loop_model(source_path))
    direct = gluon_scattering_model_virtual_program(
        source, r2_strategy="direct", reduction_engine="fast", coefficient_engine="jit"
    )
    # Independently compiled effective R2 trees: the model wrapper never reads
    # the dimensional remainder. Both wrappers share the original N4 numerator.
    model = NativeModelVirtualProgram(
        direct.bare,
        direct.matched_insertions,
        source_bindings=direct.source_bindings,
        external_fields=direct.external_fields,
    )
    root = Path(__file__).resolve().parents[2] / "tools/loop_reference"
    points = np.asarray(
        json.loads((root / "gluon_scattering_qcd.json").read_text())["points"],
        dtype=str,
    )
    return direct, model, points, json.loads(Path(reference_path).read_text())


def test_direct_views_reuse_dimensional_instructions_and_fixed_strategy(
    direct_control, monkeypatch
):
    from pyamplicol.loops import direct_kernels, physical_counterterms, physical_process
    from pyamplicol.loops.stability import ArithmeticSpec, NativeModelStabilityProgram

    program, model, points, reference = direct_control
    prepared, model_prepared = program.prepared, model.prepared
    assert prepared.bare is model_prepared.bare
    assert any(f.rational_numerators != ("0", "0", "0") for f in prepared.bare.families)
    assert any(f.rational_numerators[1] != "0" for f in prepared.bare.families)
    baseline = program.evaluate_ordered_native(
        points[:2], ["1000000"] * 2, model_values=["0.118", "173"]
    )
    baseline_values = tuple(value.to_decimal() for value in batches(baseline))
    for name, key in (
        ("virtual", "full"),
        ("bare", "bare"),
        ("uv", "uv"),
        ("r2", "r2"),
        ("uv_local", "uv_vertex"),
        ("uv_external_fields", "wf"),
    ):
        np.testing.assert_allclose(
            np.asarray(getattr(baseline, name).to_decimal(), dtype=float)[:, 0].reshape(
                2, 3
            ),
            [
                [row[key][pole] for pole in ("finite", "single_pole", "double_pole")]
                for row in reference["rows"][:2]
            ],
            rtol=5e-11,
            atol=1e-10,
        )
    wrapper = NativeModelStabilityProgram(
        program,
        [ArithmeticSpec("double-double"), ArithmeticSpec("arbitrary", 50)],
        relative_tolerance="1e-8",
        absolute_tolerance="1e-10",
    )
    forbid_cold_work(monkeypatch)

    def forbidden(*args, **kwargs):
        raise AssertionError("direct view repeated cold physics preparation")

    for owner, name in (
        (physical_process.NativeBareProcessProgram, "__init__"),
        (physical_counterterms.NativeModelVirtualProgram, "__init__"),
        (physical_process, "_reduction_numerators"),
        (direct_kernels, "dimensional_kernels"),
        (direct_kernels, "closed_reduction_powers"),
    ):
        monkeypatch.setattr(owner, name, forbidden)
    for arithmetic, precision, engine, bits, tolerance in (
        ("double", None, "fast", None, "1e-10"),
        ("double-double", None, "generic", 80, "1e-27"),
        ("arbitrary", 50, "generic", 140, "1e-45"),
    ):
        view = (
            program.rebind(arithmetic=arithmetic, reduction_engine=engine)
            if arithmetic == "double"
            else wrapper._view(ArithmeticSpec(arithmetic, precision))
        )
        assert view.prepared is prepared
        assert view.r2_strategy == "direct" and view.bare.dimensional_numerators
        assert all(item.rule.kind != "R2" for item in view.insertions)
        assert all(current.insertion.rule.kind != "R2" for current in view.currents)
        assert view.bare.loop_reduction_powers == program.bare.loop_reduction_powers
        assert view.bare.loop_executors[0].plan.workspace_capacity == (0, 0, 0)
        for old, new in zip(program.bare.families, view.bare.families, strict=True):
            assert new.reducer is not old.reducer
            if old.rational_reducer is not None:
                assert new.rational_reducer is not old.rational_reducer
                assert new.rational_reducer is not None
        actual = view.evaluate_ordered_partial_native(
            points[:2],
            ["1000000"] * 2,
            model_values=["0.118", "173"],
            point_ids=[71, 9],
        )
        assert actual.r2_strategy == "direct"
        assert actual.requested_point_ids == (71, 9) and not actual.failures
        assert all(value.row_ids == [71, 9] for value in batches(actual))
        for value, expected in zip(batches(actual), batches(baseline), strict=True):
            assert all(
                ok
                for _, ok in value.take_rows(
                    list(range(value.rows)), expected.row_ids
                ).agreement_rows(expected, "1e-8", "1e-9")
            )
        if arithmetic == "arbitrary":
            # No new loop/CT graph generation: bind the independent effective
            # R2 instructions in the same true arithmetic as the direct result.
            oracle = model_prepared.bind(
                backend=view.bare.backend, arithmetic=arithmetic, precision=precision
            )
            assert oracle.r2_strategy == "model"
            assert any(c.insertion.rule.kind == "R2" for c in oracle.currents)
            source_result = oracle.evaluate_ordered_native(
                points[:2], ["1000000"] * 2, model_values=["0.118", "173"]
            )
            for value, expected in zip(
                batches(actual), batches(source_result), strict=True
            ):
                assert all(
                    ok
                    for _, ok in value.take_rows(
                        list(range(value.rows)), expected.row_ids
                    ).agreement_rows(expected, "3e-40", "1e-38")
                )
        saved = tuple(value.to_decimal() for value in batches(actual))
        changed = view.evaluate_ordered_native(
            points[:1], ["700000"], model_values=["0.12", "190"]
        )
        assert (
            changed.virtual.to_decimal() != actual.virtual.take_rows([0]).to_decimal()
        )
        empty = view.evaluate_ordered_partial_native(
            points[:0], [], model_values=["0.118", "173"], point_ids=[]
        )
        assert empty.r2_strategy == "direct" and empty.virtual.rows == 0
        restored = view.evaluate_ordered_partial_native(
            points[:2],
            ["1000000"] * 2,
            model_values=["0.118", "173"],
            point_ids=[71, 9],
        )
        assert tuple(value.to_decimal() for value in batches(restored)) == saved
        if bits is not None:
            with localcontext() as context:
                context.prec = 100
                alpha = Decimal("0.118") + Decimal(2) ** -bits
                shifted = view.evaluate_ordered_native(
                    points[:1], ["1000000"], model_values=[str(alpha), "173"]
                )
                a = Decimal(actual.virtual.to_decimal()[2][0])
                b = Decimal(shifted.virtual.to_decimal()[2][0])
                assert a != b
                assert abs(b - a * (alpha / Decimal("0.118")) ** 3) < abs(a) * Decimal(
                    tolerance
                )
        assert tuple(value.to_decimal() for value in batches(actual)) == saved
    assert tuple(value.to_decimal() for value in batches(baseline)) == baseline_values


def test_direct_retained_scope_rejects_incomplete_or_mixed_completion(direct_control):
    from types import SimpleNamespace

    from pyamplicol.loops.prepared_process import (
        PreparedBareProcess,
        PreparedModelProcess,
    )

    program, model, _, _ = direct_control
    for override in ({"model_state": object()}, {"_formal_color_projected": True}):
        copy = SimpleNamespace(**(vars(program.bare) | override))
        with pytest.raises(NotImplementedError, match="unprojected"):
            PreparedBareProcess.from_program(copy)
    bare = SimpleNamespace(**(vars(program.bare) | {"dimensional_numerators": False}))
    incomplete = SimpleNamespace(**(vars(program) | {"bare": bare}))
    with pytest.raises(ValueError, match="dimensional"):
        PreparedModelProcess.from_program(incomplete)
    for key, value in (("insertions", model.insertions), ("currents", model.currents)):
        mixed = SimpleNamespace(**(vars(program) | {key: value}))
        with pytest.raises(ValueError, match="model R2"):
            PreparedModelProcess.from_program(mixed)
