# SPDX-License-Identifier: 0BSD
"""Signed virtual colour presentation is separate from positive tree flows."""

import io
import json
from dataclasses import replace
from decimal import Decimal

import pytest

from pyamplicol.api import BornAnchoredColorComponent
from pyamplicol.api.results import (
    ColorFlow,
    ContractedColorComponent,
    ExternalParticle,
    HelicityConfiguration,
    PhysicsReduction,
    ProcessPhysics,
)
from pyamplicol.api.virtual_results import (
    ResolvedVirtualEvaluation,
    VirtualLaneDiagnostic,
)
from pyamplicol.cli.main import write_result
from pyamplicol.config import R2Strategy
from pyamplicol.loops.orders import BornLoopPair, VirtualOrderSector
from pyamplicol.loops.strategy import R2Selection


def _physics():
    return ProcessPhysics(
        process_id="p",
        process="g g > g g",
        color_accuracy="lc",
        helicity_coverage="all",
        color_coverage="all",
        color_kind="physical-lc-flows",
        structural_zero_helicity_count=0,
        external_particles=(ExternalParticle(0, 1, "g", 21, "incoming", 0),),
        helicities=(HelicityConfiguration("h0", 0, (1,), True, False, "h0", 1.0),),
        color_flows=(ColorFlow("flow0", 0, (1,), True, "flow0", 1.0),),
        contracted_color_components=(),
        reduction=PhysicsReduction("lc-diagonal", ()),
        model_parameters=(),
        selector_capabilities=(),
    )


def _anchors():
    return (
        BornAnchoredColorComponent("b0", 0, (0, 2), "signed Born tensor group 0"),
        BornAnchoredColorComponent("b1", 1, (1,), "signed Born tensor group 1"),
    )


def test_tree_and_virtual_colour_metadata_keep_distinct_semantics():
    import pyamplicol

    assert pyamplicol.BornAnchoredColorComponent is BornAnchoredColorComponent
    tree = _physics()
    assert tree.contribution_kind == "tree"
    assert tree.color_ids == tree.color_flow_ids == ("flow0",)
    virtual = replace(
        tree,
        contribution_kind="virtual",
        color_kind="born-anchored-interference",
        color_flows=(),
        born_anchored_color_components=_anchors(),
        reduction=PhysicsReduction("born-anchored-interference", ()),
    )
    assert virtual.color_ids == ("b0", "b1")
    assert virtual.color_flow_ids == ()
    assert virtual.contracted_color_components == ()
    for accuracy in ("nlc", "full"):
        contracted = replace(
            virtual,
            color_accuracy=accuracy,
            color_kind="contracted-color",
            born_anchored_color_components=(),
            contracted_color_components=(
                ContractedColorComponent(accuracy, 0, accuracy),
            ),
            reduction=PhysicsReduction("contracted-color", ()),
        )
        assert contracted.color_ids == (accuracy,)
        assert not contracted.color_flows
    with pytest.raises(ValueError, match="Born anchors apply only"):
        replace(virtual, contribution_kind="tree")
    with pytest.raises(ValueError, match="signed Born anchors"):
        replace(virtual, born_anchored_color_components=())
    with pytest.raises(ValueError, match="signed Born anchors"):
        replace(virtual, color_flows=tree.color_flows)
    with pytest.raises(ValueError, match="reduction kind"):
        replace(virtual, reduction=PhysicsReduction("lc-diagonal", ()))
    with pytest.raises(ValueError, match="physical color flows"):
        replace(tree, color_flows=())
    with pytest.raises(ValueError, match="reduction kind"):
        replace(tree, reduction=PhysicsReduction("born-anchored-interference", ()))


def test_virtual_lc_contracted_identity_observable_has_no_born_anchors():
    contracted = replace(
        _physics(),
        contribution_kind="virtual",
        color_kind="contracted-color",
        color_flows=(),
        contracted_color_components=(ContractedColorComponent("lc", 0, "lc"),),
        reduction=PhysicsReduction("contracted-color", ()),
    )
    assert contracted.color_ids == ("lc",)
    assert not contracted.color_flows
    assert not contracted.born_anchored_color_components
    for changes in (
        {"contracted_color_components": ()},
        {
            "contracted_color_components": (
                ContractedColorComponent("lc", 0, "lc"),
                ContractedColorComponent("extra", 1, "extra"),
            )
        },
        {"born_anchored_color_components": _anchors()},
        {"color_flows": _physics().color_flows},
    ):
        with pytest.raises(ValueError, match="one inclusive color component"):
            replace(contracted, **changes)
    with pytest.raises(ValueError, match="reduction kind"):
        replace(
            contracted, reduction=PhysicsReduction("born-anchored-interference", ())
        )


@pytest.mark.parametrize("members", [(), (0, 0), (-1,), (True,)])
def test_anchor_members_are_exact_indices(members):
    with pytest.raises(ValueError):
        BornAnchoredColorComponent("b0", 0, members, "signed anchor")


def _resolved():
    qcd2, qcd4 = (("QCD", 2),), (("QCD", 4),)
    exact = Decimal("-1.123456789012345678901234567890123456789")
    return ResolvedVirtualEvaluation(
        ((((exact, Decimal(2)), (Decimal(3), Decimal(-4))),),),
        (
            (
                (
                    (
                        (exact, Decimal(-2), Decimal(3)),
                        (Decimal(4), Decimal(5), Decimal(-6)),
                    ),
                    (
                        (Decimal(7), Decimal(-8), Decimal(9)),
                        (Decimal(-10), Decimal(11), Decimal(12)),
                    ),
                ),
            ),
        ),
        (qcd4,),
        (VirtualOrderSector((("QCD", 6),), (BornLoopPair(qcd2, qcd4),)),),
        (VirtualLaneDiagnostic("success", "arbitrary", 233, False),),
        R2Selection(R2Strategy.AUTO, R2Strategy.MODEL, "only complete strategy"),
        ("h0", "h1"),
        ("b0", "b1"),
        ("b0", "b1"),
        "lc",
    )


def test_signed_resolved_json_and_human_preserve_axes_precision_and_accuracy():
    value = _resolved()
    assert value.total().color_accuracy == "lc"
    assert value.total().for_order((("QCD", 6),)).color_accuracy == "lc"
    for format in ("json", "human"):
        stream = io.StringIO()
        write_result(value, format=format, stream=stream)
        text = stream.getvalue()
        assert "-1.123456789012345678901234567890123456789" in text
        assert "h0" in text and "h1" in text and "b0" in text and "b1" in text
        if format == "json":
            data = json.loads(text)
            assert data["color_accuracy"] == "lc"
            assert data["shape"] == [1, 1, 2, 2, 3]
            assert data["born_shape"] == [1, 1, 2, 2]
            assert data["epsilon_powers"] == [0, -1, -2]
        else:
            assert "Born anchor" in text and "colour accuracy=LC" in text
            assert "signed, potentially overcomplete" in text
            assert "single pole [-1]" in text and "double pole [-2]" in text
    total = replace(value.total(), color_accuracy="nlc")
    stream = io.StringIO()
    write_result(total, format="human", stream=stream)
    assert "colour accuracy=NLC" in stream.getvalue()
    with pytest.raises(ValueError, match="colour accuracy"):
        replace(total, color_accuracy="unknown")


def _transport_backend(monkeypatch, accuracy="lc"):
    """Synthetic transport, deliberately unequal Born and virtual order counts."""
    from pathlib import Path
    from types import SimpleNamespace

    from pyamplicol.runtime import virtual

    component_ids = ["b0", "b1", "b2"] if accuracy == "lc" else [accuracy]
    partition = (
        {
            "kind": "canonical-born-tensor-partition",
            "ids": component_ids,
            "members": [[2], [0, 3], [1]],
        }
        if accuracy == "lc"
        else None
    )
    data = {
        "artifact_id": "fixture-only",
        "process_id": "p",
        "expression": "transport fixture",
        "external_pdgs": [21],
        "initial_average": 1,
        "physics": {
            "color_accuracy": accuracy,
            "resolved": True,
            "arithmetic": ["double", "double-double", "arbitrary"],
            "orders": {
                "born": [{"order": [["QCD", p]]} for p in (2, 4)],
                "virtual": [
                    {
                        "order": [["QCD", p + 3]],
                        "pairs": [{"born": [["QCD", p]], "loop": [["QCD", 3]]}],
                    }
                    for p in (1, 2, 3)
                ],
            },
        },
        "plan": {
            "external_pdgs": [21],
            "state_legs": [{"incoming": True}],
            "helicities": [[1], [-1]],
            "born_color_components": partition,
            "model_binding": {
                "owner": {
                    "external_names": [],
                    "external_types": [],
                    "fixed_parameters": {},
                }
            },
        },
    }
    operation = dict(
        requested="direct",
        algorithm="direct",
        reason="exact direct weight schedule",
        regular_kernels=0,
        direct_residual_entries=0,
    )
    handle = SimpleNamespace(
        arithmetic="arbitrary",
        precision=70,
        precision_bits=233,
        r2_strategy="model",
        source_revision=0,
        parameter_overrides={},
        metadata_json=lambda: json.dumps(data),
        color_contraction_json=json.dumps(
            {"ordinary": operation, "resolved": operation}
        ),
        weight_operation_json=json.dumps(operation),
        failures=[],
    )

    def batch(rows, width):
        return SimpleNamespace(
            rows=rows,
            columns=width,
            row_ids=list(range(rows)),
            to_decimal=lambda: [
                (str(-(r * 100 + column + 1)), "0")
                for r in range(rows)
                for column in range(width)
            ],
        )

    def evaluate(points, scales, *, resolved, r2_strategy):
        rows = len(points) * (2 * len(component_ids) if resolved else 1)
        return SimpleNamespace(
            requested_point_ids=list(range(len(points))),
            point_ids=list(range(len(points))),
            r2_strategy="model",
            primary_color_contraction_json=handle.weight_operation_json,
            failures=handle.failures,
            statuses=[[] for _ in range(rows)],
            born=batch(rows, 2),
            component_batches=[batch(rows, 9)] * 6,
        )

    handle.evaluate_partial = evaluate
    monkeypatch.setattr(
        virtual,
        "load_loop_native",
        lambda: SimpleNamespace(NativeVirtualRuntime=lambda *a, **kw: handle),
    )
    return virtual.VirtualRuntimeBackend(
        Path("fixture"), arithmetic="arbitrary", precision=70
    ), handle


@pytest.mark.parametrize("accuracy", ["lc", "nlc", "full"])
def test_native_component_transport_preserves_all_axes_and_empty(monkeypatch, accuracy):
    backend, _ = _transport_backend(monkeypatch, accuracy)
    components = 3 if accuracy == "lc" else 1
    points = [[[1, 0, 0, 1]]] * 3
    result = backend.evaluate_resolved(points, mu_r2=[1, 2, 3])
    assert result.born_shape == (3, 2, 2, components)
    assert result.shape == (3, 3, 2, components, 3)
    assert result.color_accuracy == accuracy
    assert backend.physics.contribution_kind == "virtual"
    assert backend.physics.color_ids == result.virtual_color_ids
    assert backend.physics.color_flow_ids == ()
    assert backend.inspect()["resolved_color_components"] == list(
        result.virtual_color_ids
    )
    for p in range(3):
        for h in range(2):
            for c in range(components):
                row = (p * 2 + h) * components + c
                for order in range(2):
                    assert result.born_values[p][order][h][c] == Decimal(
                        -(row * 100 + order + 1)
                    )
                for order in range(3):
                    assert tuple(result.values[p][order][h][c]) == tuple(
                        Decimal(-(row * 100 + order * 3 + e + 1)) for e in range(3)
                    )
    assert backend.evaluate(points, mu_r2=1).color_accuracy == accuracy
    empty = backend.evaluate_resolved([], mu_r2=[])
    assert empty.shape == (0, 3, 2, components, 3)
    assert empty.total().shape == (0, 3, 3)


def test_native_component_failure_retains_physical_identity(monkeypatch):
    from types import SimpleNamespace

    from pyamplicol.api.errors import EvaluationError

    backend, native = _transport_backend(monkeypatch)
    native.failures = [
        SimpleNamespace(
            point_id=1,
            helicity=1,
            order_index=2,
            family_index=4,
            stage="reduction",
            detail="injected transport failure",
            external_leg=None,
            color_component_index=2,
            history_index=None,
            direction=None,
        )
    ]
    with pytest.raises(EvaluationError) as caught:
        backend.evaluate_resolved([[[1, 0, 0, 1]]] * 3, mu_r2=1)
    failure = caught.value.failures[0]
    assert (
        failure.point_id,
        failure.helicity,
        failure.color_component_index,
        failure.order_index,
    ) == (1, 1, 2, 2)


def test_artifact_colour_declaration_requires_matching_native_partition():
    from dataclasses import asdict

    from pyamplicol.generation.loop_artifact import (
        LoopExecutionMetadata,
        LoopSchemeDeclaration,
    )
    from pyamplicol.loops.physical_orders import BornSquaredSector, PhysicalOrderLayout

    order = (("QCD", 2),)
    layout = PhysicalOrderLayout(
        (BornSquaredSector((("QCD", 4),), ((order, order),)),),
        _resolved().sectors,
        ("QCD",),
    )
    scheme = LoopSchemeDeclaration(
        dimensional_regularization="declared test scheme",
        external_states="four-dimensional",
        gamma5="none",
        renormalization="fixture",
        mass_scheme="fixture",
        finite_restoration="fixture",
        model_convention="transport test only",
    )
    metadata = LoopExecutionMetadata(
        scheme,
        layout,
        "model",
        ("model",),
        ("double", "double-double", "arbitrary"),
        True,
        resolved=True,
        color_accuracy="lc",
    )
    plan = {
        "color_projection": {"accuracy": "lc"},
        "helicity_projection": {},
        "born_color_components": {
            "kind": "canonical-born-tensor-partition",
            "ids": ["b0"],
            "members": [[0]],
        },
        "order_layout": json.loads(json.dumps(asdict(layout))),
        "r2_strategy": "model",
        "model_binding": {"kind": "source"},
    }
    metadata.validate_plan(plan)
    for bad in (
        {**plan, "born_color_components": None},
        {**plan, "color_projection": {"accuracy": "full"}},
        {**plan, "helicity_projection": None},
    ):
        with pytest.raises(ValueError):
            metadata.validate_plan(bad)
    with pytest.raises(ValueError, match="resolved canonical Born"):
        replace(metadata, resolved=False).validate_plan(plan)
    from pyamplicol.loops.physical_correlations import (
        PROJECTED_FAMILY_CONVENTION,
        PROJECTED_IDENTITY_CONVENTION,
    )

    directed = {
        "kind": "directed-complex",
        "convention": PROJECTED_IDENTITY_CONVENTION,
        "histories": [{"id": "born", "bra": [], "ket": []}],
    }
    for accuracy in ("lc", "nlc", "full"):
        inclusive = replace(
            metadata,
            color_accuracy=accuracy,
            resolved=False,
            correlations=True,
            interference=None,
        )
        projected = {
            **plan,
            "color_projection": {"accuracy": accuracy},
            "helicity_projection": None,
            "born_color_components": None,
            "directed_projection": directed,
        }
        inclusive.validate_plan(projected)
        family = {
            "kind": "complete-typed-history-families-v1",
            "accuracy": accuracy,
            "families": [],  # Native validation owns actual family axes/operations.
            "history_families": [],
        }
        family_directed = {
            **directed,
            "convention": PROJECTED_FAMILY_CONVENTION,
            "color_family_projection": family,
        }
        inclusive.validate_plan({**projected, "directed_projection": family_directed})
        for invalid in (
            {**projected, "born_color_components": plan["born_color_components"]},
            {**projected, "helicity_projection": {}},
            {
                **projected,
                "directed_projection": {
                    **directed,
                    "convention": "su3-literal-tau-physical-charge-all-outgoing-v1",
                },
            },
            {
                **projected,
                "directed_projection": {
                    **directed,
                    "histories": [
                        {"id": "T", "bra": [{"emitter": 1}], "ket": [{"emitter": 1}]}
                    ],
                },
            },
            {
                **projected,
                "directed_projection": {**directed, "color_family_projection": family},
            },
            {
                **projected,
                "directed_projection": {
                    **family_directed,
                    "color_family_projection": None,
                },
            },
            {
                **projected,
                "directed_projection": {
                    **family_directed,
                    "color_family_projection": {**family, "accuracy": "unknown"},
                },
            },
        ):
            with pytest.raises(ValueError):
                inclusive.validate_plan(invalid)
    for accuracy in ("nlc", "full"):
        adjusted = {**plan, "color_projection": {"accuracy": accuracy}}
        with pytest.raises(ValueError, match="require LC"):
            replace(metadata, color_accuracy=accuracy).validate_plan(adjusted)
        replace(metadata, color_accuracy=accuracy).validate_plan(
            {**adjusted, "born_color_components": None}
        )
