# SPDX-License-Identifier: 0BSD
"""Actual declared-colour W source composition; no assumed SM continuation."""

import json
from decimal import Decimal, localcontext
from pathlib import Path

import pytest

from .test_loop_massive_vector_sources import (
    test_charged_three_state_sources_and_exact_replay as _check_three_states,
)

pytestmark = pytest.mark.loop_physics


def assert_retained_formal_completion(original, projected):
    """Check all actual tensor/scalar grades before the numerical Nc binding.

    The independent arithmetic checks use this same exact preparation. This is
    a formal projection control, not an additional independent loop provider.
    """
    from symbolica import E

    from pyamplicol.loops.color_bilinears import SUNColorSpace
    from pyamplicol.loops.color_projection import prepare_color_projection
    from pyamplicol.loops.scalar_incidence import scalar_grade_term_mask

    full = projected["full"].virtual
    formal = full.bare.formal_color_metrics
    formal.require_owner(full.bare)
    assert len(full.bare.order_layout.virtual) == 1
    space = SUNColorSpace(formal.nc)
    rows = [[value] for row in formal.loop for value in row]
    bare_count = len(rows)
    for current in full.currents:
        weights = space.rectangular(full.bare.born_color_tensors, current.color_tensors)
        active = scalar_grade_term_mask(
            current.dag,
            current.executor.prepared.output_nodes,
            current.insertion.rule.active_coupling_symbols,
            current.coefficient_expressions,
        )
        rows.extend(
            [formal.nc**current.color_grade * value if selected else E("0")]
            for row in weights
            for value, selected in zip(row, active, strict=True)
        )
    for current in full.field_currents:
        weights = space.rectangular(
            full.bare.born_color_tensors,
            tuple(full.bare.born_color_tensors[i] for i in current.born_indices),
        )
        rows.extend(
            [
                formal.nc**current.color_grade * value
                if current.coefficient_expression != 0
                else E("0")
            ]
            for row in weights
            for value in row
        )
    plan = prepare_color_projection(formal.nc, rows)
    assert formal.virtual_degrees == (plan.leading_degree,) == (3,)
    assert formal.born_degrees == (2,)
    # Fixed-flavour terms can be odd powers below the common leading degree.
    odd_rows = [
        i
        for i, row in enumerate(plan.rows)
        if row and max(degree for degree, _ in row) == plan.leading_degree - 1
    ]
    assert odd_rows
    for accuracy, source in projected.items():
        model = source.virtual
        model.bare.formal_color_metrics.require_owner(model.bare)
        assert model.bare.virtual_color_degrees == (plan.leading_degree,)
        expected = plan.project(accuracy)
        actual = [
            value for row in model.bare.formal_color_metrics.loop for value in row
        ]
        assert all(
            (a - b).together() == 0
            for a, b in zip(actual, expected[:bare_count], strict=True)
        )
        full_currents = (*full.currents, *full.field_currents)
        actual_currents = (*model.currents, *model.field_currents)
        for left, right in zip(full_currents, actual_currents, strict=True):
            assert left.insertion is right.insertion
            assert (left.epsilon_power, left.color_grade) == (
                right.epsilon_power,
                right.color_grade,
            )
        actual_weights = [
            value
            for current in actual_currents
            for row in current.color_weights
            for value in row
        ]
        assert all(
            (a - b.replace(formal.nc, 3) / model.bare.initial_average).together() == 0
            for a, b in zip(actual_weights, expected[bare_count:], strict=True)
        )
        if accuracy == "lc":
            assert all(expected[i] == 0 for i in odd_rows)
        if accuracy == "nlc":
            assert all(expected[i] != 0 for i in odd_rows)
    assert original.virtual.bare.formal_color_metrics is None
    return {
        "born_degree": 2,
        "virtual_degree": 3,
        "odd_rows": len(odd_rows),
        "rows": len(rows),
    }


def assert_retained_incomplete_declaration_rejected(
    original,
    source_color,
    *,
    tree_key=("vertices", 34, 0, 0, None),
    ct_key=("counterterm_vertices", 27, 0, 0, 0),
):
    """Deleting an actual selected charged entry fails the existing scope gate."""
    from dataclasses import replace

    model = original.virtual
    bare = model.bare
    plan = bare.model_state.plan
    ordinary = {
        (vertex.interaction, vertex.order)
        for graph in (*bare.process.born, *bare.loop_graphs)
        for vertex in graph.vertices
    } | {
        (vertex.interaction, vertex.order)
        for item in model.insertions
        for vertex in item.graph.vertices
        if vertex.id != item.vertex_id
    }
    missing_tree = replace(
        source_color, kernel_keys=source_color.kernel_keys - {tree_key}
    )
    with pytest.raises(
        NotImplementedError, match="no declared formal-colour continuation"
    ):
        missing_tree.require_ordinary_kernels(
            ordinary, {name for name, _ in plan.outputs}, scalar_plan=plan
        )
    missing_ct = replace(source_color, kernel_keys=source_color.kernel_keys - {ct_key})
    with pytest.raises(
        NotImplementedError, match="no declared formal-colour continuation"
    ):
        missing_ct.require_kernels(
            entry.key for item in model.insertions for entry in item.rule.entries
        )


@pytest.fixture(scope="module")
def projected_massive(saved_loop_fixture):
    root = str(saved_loop_fixture("massive-projected-sources-v43ff063"))
    from pyamplicol import _rusticol
    from pyamplicol._internal.versions import verify_native_module
    from pyamplicol.loops.native import load_loop_native

    verify_native_module(_rusticol)
    assert load_loop_native() is _rusticol._loop
    return Path(root), _rusticol._loop


@pytest.mark.parametrize("accuracy", ["lc", "nlc"])
@pytest.mark.parametrize(
    "arithmetic,precision,engine,tolerance,bit",
    [
        ("double", None, "fast", "8e-8", 20),
        ("double-double", None, "generic", "8e-22", 70),
        ("arbitrary", 70, "generic", "8e-54", 170),
    ],
)
def test_projected_massive_three_state_sources(
    projected_massive, accuracy, arithmetic, precision, engine, tolerance, bit
):
    root, native = projected_massive
    descriptor = json.loads(
        native.PortableLoopPlan(
            str(root / accuracy / "outgoing-spin.pacbin")
        ).metadata_json()
    )
    assert descriptor["color_projection"]["accuracy"] == accuracy
    assert (
        descriptor["directed_projection"]["convention"]
        == "sun-graded-identity-all-outgoing-v1"
    )
    assert descriptor.get("born_color_components") is None
    _check_three_states(
        (root / accuracy, native),
        "outgoing",
        3,
        arithmetic,
        precision,
        engine,
        tolerance,
        bit,
    )


def assert_retained_full_projection(original, projected, points):
    """Grade decomposition must reconstruct every unchanged complete component."""
    for arithmetic, precision, tolerance in (
        ("double", None, "1e-8"),
        ("double-double", None, "1e-24"),
        ("arbitrary", 70, "1e-60"),
    ):
        options = dict(
            arithmetic=arithmetic,
            precision=precision,
            reduction_engine="fast" if arithmetic == "double" else "generic",
            coefficient_engine="interpreted",
        )
        reference = original.rebind(**options).virtual.evaluate_ordered_native(
            points, ["1000000"]
        )
        result = (
            projected["full"]
            .rebind(**options)
            .virtual.evaluate_ordered_native(points, ["1000000"])
        )
        for actual, expected in zip(
            (result.born, *result.component_batches),
            (reference.born, *reference.component_batches),
            strict=True,
        ):
            with localcontext() as context:
                context.prec = 110
                pairs = list(
                    zip(actual.to_decimal(), expected.to_decimal(), strict=True)
                )
                scale = max(
                    Decimal("1e-100"),
                    *(abs(Decimal(x)) for a, b in pairs for row in (a, b) for x in row),
                )
                error = max(
                    abs(Decimal(a) - Decimal(b))
                    for left, right in pairs
                    for a, b in zip(left, right, strict=True)
                )
                # A finite UV coefficient can vanish by cancellation while its
                # pole remains nonzero. Normalize within this same component,
                # never by Born or by an unrelated observable.
                assert error <= Decimal(tolerance) * scale
                print(
                    arithmetic,
                    "component normalized error",
                    str(error / scale),
                    flush=True,
                )


def test_actual_charged_declaration_keys_and_negative_scope(loop_sm_colour_model):
    from pyamplicol.models.loading import load_compiled_model
    from pyamplicol.models.loop_color_declaration import (
        LoopColorDeclaration,
        resolve_loop_color_declaration,
    )

    # Prepared now from the committed example, not a saved compiled snapshot
    # whose compiler fingerprint goes stale.
    compiled = load_compiled_model(loop_sm_colour_model)
    raw = compiled.raw_loop_model
    declaration = resolve_loop_color_declaration(raw)
    stored = LoopColorDeclaration.from_dict(
        json.loads(
            (
                Path(__file__).resolve().parents[2]
                / "examples/loop/source_colour/loop_sm_qcd_subset.json"
            ).read_text()
        )
    )
    assert declaration == stored
    # Actual source collection IDs and group labels are part of the explicit
    # provider input. Both charged entries are independently retained source data.
    expected = {
        ("vertices", 34, 0, 0, None): ("V_31", "Identity(1,2)", "GC_12"),
        ("vertices", 43, 0, 0, None): ("V_40", "Identity(1,2)", "GC_39"),
        ("counterterm_vertices", 36, 0, 0, 0): ("V_R2uxdW", "Identity(1,2)", "R2_uxdW"),
        ("counterterm_vertices", 27, 0, 0, 0): ("V_R2dxuW", "Identity(1,2)", "R2_dxuW"),
        ("counterterm_vertices", 2, 0, 0, 0): ("V_R2GDD", "T(3,2,1)", "R2_GQQ"),
        ("counterterm_vertices", 3, 0, 0, 0): ("V_R2GUU", "T(3,2,1)", "R2_GQQ"),
        ("counterterm_vertices", 9, 0, 0, 0): ("V_R2DD", "Identity(1,2)", "R2_QQq"),
        ("counterterm_vertices", 10, 0, 0, 0): ("V_R2UU", "Identity(1,2)", "R2_QQq"),
    }
    for vertex_id, name in ((59, "V_UVGDD"), (60, "V_UVGUU")):
        for group, suffix in enumerate(("q", "c", "b", "t", "g")):
            expected["counterterm_vertices", vertex_id, 0, 0, group] = (
                name,
                "T(3,2,1)",
                f"UV_GQQ{suffix}",
            )
    assert set(expected) <= set(declaration.kernel_keys)
    for (collection, index, color, lorentz, group), (
        name,
        tensor,
        coupling,
    ) in expected.items():
        record = raw.collections[collection][index]
        assert record["name"] == name
        assert record["fields"]["color"][color] == tensor
        entries = record["fields"]["couplings"]["entries"]
        key = [color, lorentz] if group is None else [color, lorentz, group]
        assert (
            next(
                value
                for indices, value in entries
                if tuple(indices["items"]) == tuple(key)
            )["name"]
            == coupling
        )
    assert declaration.parameter_identities == {
        "CA": "Nc",
        "CF": "(Nc**2-1)/(2*Nc)",
        "TF": "0.5",
        "Ncol": "Nc",
    }
    assert declaration.scheme_parameter_values == {"lhv": "1"}


@pytest.mark.parametrize("accuracy", ["lc", "nlc"])
@pytest.mark.parametrize(
    "arithmetic,precision,tolerance",
    [
        ("double", None, "1e-8"),
        ("double-double", None, "1e-24"),
        ("arbitrary", 70, "1e-60"),
    ],
)
def test_public_massive_projected_resolved_reconstruction(
    projected_massive, accuracy, arithmetic, precision, tolerance, direction="outgoing"
):
    from pyamplicol import Runtime

    root, _ = projected_massive
    runtime = Runtime.load(
        root / accuracy / "public-ordinary", arithmetic=arithmetic, precision=precision
    )
    point = json.loads((root / accuracy / f"{direction}-ordinary.json").read_text())[
        "point"
    ]
    inclusive = runtime.evaluate([point], mu_r2="1000000")
    resolved = runtime.evaluate_resolved([point], mu_r2="1000000")
    assert resolved.color_accuracy == accuracy
    assert resolved.shape[-1] == 3 and resolved.shape[2] == 24
    assert runtime.physics.reduction.kind == (
        "born-anchored-interference" if accuracy == "lc" else "contracted-color"
    )
    assert bool(runtime.physics.born_anchored_color_components) == (accuracy == "lc")
    total = resolved.total()
    with localcontext() as context:
        context.prec = 110
        for actual, expected in zip(
            (*total.born_values[0], *total.values[0][0]),
            (*inclusive.born_values[0], *inclusive.values[0][0]),
            strict=True,
        ):
            assert abs(Decimal(actual) - Decimal(expected)) <= Decimal(tolerance) * max(
                abs(Decimal(expected)), Decimal(1)
            )
    assert runtime.evaluate_resolved([], mu_r2=[]).total().values == ()


@pytest.mark.parametrize("accuracy", ["lc", "nlc"])
def test_public_massive_projected_identity_source_rescue(
    projected_massive, accuracy, direction="outgoing", leg=3
):
    from pyamplicol import CorrelatedRequest, Runtime
    from pyamplicol.api import VirtualArithmeticSpec, VirtualStabilityPolicy
    from pyamplicol.api.errors import EvaluationError

    root, _ = projected_massive
    runtime = Runtime.load(root / accuracy / "public-spin")
    assert runtime.physics.reduction.kind == "contracted-color"
    assert not runtime.physics.born_anchored_color_components
    assert runtime.inspect()["spin_correlations"] == ((leg,), (4,), (leg, 4))
    point = json.loads((root / accuracy / f"{direction}-ordinary.json").read_text())[
        "point"
    ]
    w = [("2", "3"), ("5", "-7"), ("11", "13"), ("-17", "19")]
    g = [("0", "0"), ("0.8", "0"), ("0", "1"), ("-0.6", "0")]
    points, scales = [point] * 3, ["1000000", "1e-400", "1000000"]
    request = {"joint": CorrelatedRequest("born", {leg: w, 4: g})}
    with pytest.raises(EvaluationError) as failure:
        runtime.evaluate_correlated_many(points, request, mu_r2=scales)
    assert {item.point_id for item in failure.value.failures} == {1}
    rescued = runtime.evaluate_correlated_many(
        points,
        request,
        mu_r2=scales,
        stability=VirtualStabilityPolicy(
            (
                VirtualArithmeticSpec("arbitrary", 60),
                VirtualArithmeticSpec("arbitrary", 80),
            ),
            "1e-8",
            "1e-9",
        ),
    )["joint"]
    assert [row.status for row in rescued.diagnostics] == [
        "success",
        "rescued",
        "success",
    ]
    assert rescued.values[0] == rescued.values[2]
    runtime.set_spin_correlation_vectors({leg: w, 4: g})
    first = runtime.evaluate_correlated(points[:1], mu_r2=scales[:1])
    runtime.clear()
    assert (
        runtime.evaluate_correlated(points[:1], mu_r2=scales[:1]).values == first.values
    )
    assert runtime.evaluate_correlated([], mu_r2=[]).shape == (0, 1, 3)
