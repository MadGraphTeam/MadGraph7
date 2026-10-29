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

pytestmark = pytest.mark.loop_fast


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
