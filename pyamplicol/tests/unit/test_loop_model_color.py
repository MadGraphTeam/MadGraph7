# SPDX-License-Identifier: 0BSD
"""Exact public tensor trace algebra, including noncommuting order controls."""

import json

import pytest

from pyamplicol.loops.model_color import prepare_color_expression

pytestmark = pytest.mark.loop_fast


@pytest.fixture(scope="module")
def hep():
    symbolica = pytest.importorskip("symbolica")
    if not symbolica.is_licensed():
        pytest.skip("native local Feynman rules require licensed Symbolica")
    return pytest.importorskip("symbolica.community.hepkit")


def _native_color(hep, source, n):
    payload = json.loads(hep.Model.qcd_qed().to_json())
    payload["vertex_rules"] = [
        {
            "name": "TRACE_CONTROL",
            "particles": ["g"] * n,
            "color_structures": [
                prepare_color_expression(source, [8] * n).to_canonical_string()
            ],
            "lorentz_structures": ["TRACE_UNIT"],
            "couplings": [["TRACE_ONE"]],
        }
    ]
    payload["lorentz_structures"] = [
        {"name": "TRACE_UNIT", "spins": [3] * n, "structure": "1"}
    ]
    payload["couplings"] = [
        {"name": "TRACE_ONE", "expression": "1", "orders": [], "value": None}
    ]
    model = hep.Model.from_json(json.dumps(payload))
    graph = hep.FeynmanDiagram.from_dot(
        model,
        'digraph trace { ext [style=invis]; v [int_id="TRACE_CONTROL"];'
        + 'ext -> v [particle="g"];' * n
        + "}",
    ).apply_feynman_rules()
    return model.expand_couplings(graph.vertices[0].numerator_expression())


def test_trace_two_has_real_adjoint_ports_and_native_tr_normalization(hep):
    from symbolica import E

    trace = _native_color(hep, "Tr(1,2)", 2)
    identity = _native_color(hep, "Identity(1,2)/2", 2)
    assert trace.rank == 2
    assert trace.axes == identity.axes
    residual = (trace - identity).simplify_algebra(
        gamma=False, color=True, color_substitute_cof_dimension_invariants=True
    )
    assert residual.to_expression() == 0
    actual = trace.to_tensor().to_dense()
    assert tuple(actual.shape) == (8, 8)
    assert all(
        actual[a, b] == E("1/2") * int(a == b) for a in range(8) for b in range(8)
    )


def test_noncommuting_trace_order_is_retained_by_native_algebra(hep):
    forward = _native_color(hep, "Tr(1,2,3)", 3)
    cyclic = _native_color(hep, "Tr(2,3,1)", 3)
    reverse = _native_color(hep, "Tr(2,1,3)", 3)
    commutator = _native_color(hep, "complex(0,1)*f(1,2,3)/2", 3)
    assert forward.rank == reverse.rank == 3
    settings = dict(
        gamma=False, color=True, color_substitute_cof_dimension_invariants=True
    )
    assert (forward - cyclic).simplify_algebra(**settings).to_expression() == 0
    assert (forward - reverse - commutator).simplify_algebra(
        **settings
    ).expand().to_expression() == 0
    # Their nonzero imaginary component rules out accidentally symmetrizing Tr.
    aligned = reverse.permute_axes([reverse.axes.index(axis) for axis in forward.axes])
    assert forward.to_tensor()[[0, 1, 2]] != aligned.to_tensor()[[0, 1, 2]]


@pytest.mark.parametrize(
    "source, colors",
    [
        ("Tr(-1,2)", [8, 8]),
        ("Tr(1,2)", [3, -3]),
        ("Tr(1,3)", [8, 8]),
        ("Tr(1,1)", [8, 8]),
        ("Tr(1/2,2)", [8, 8]),
        ("Tr(1)", [8]),
        ("UnknownColor(1,2)", [8, 8]),
        ("T6(1,2,3)", [8, 6, -6]),
    ],
)
def test_unsupported_trace_or_color_head_fails_before_native_scalar(
    hep, source, colors
):
    with pytest.raises(NotImplementedError):
        prepare_color_expression(source, colors)
