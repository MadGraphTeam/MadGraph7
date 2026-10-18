# SPDX-License-Identifier: 0BSD
"""Typed source endpoints leave scalar momentum labels and Lorentz ports intact."""

from types import SimpleNamespace

import pytest


def test_source_endpoint_map_is_typed_simultaneous_and_involutive():
    pytest.importorskip("symbolica")
    from symbolica import S
    from symbolica.community import hepkit as h
    from symbolica.community import tensor as t

    from pyamplicol.loops.vertex_frames import native_physical_vertex

    a, b = (h.Symbols.half_edge()(index, 1) for index in (3, 8))
    middle = S("vertex_frame_test::dummy")
    momentum = S("vertex_frame_test::momentum")(a)
    # Reuse a's index label for a DIFFERENT representation and inside an opaque
    # scalar function. Neither occurrence may move with its bispinor Slot.
    local = (
        momentum
        * t.TensorExpression.gamma5(4)(a, middle)
        * t.TensorExpression.dirac_gamma(4)(middle, b, a)
    )
    graph = SimpleNamespace(
        edges=(SimpleNamespace(id=10, spin=2), SimpleNamespace(id=11, spin=2)),
        half_edges=((3, 4, 10, "incoming"), (8, 4, 11, "outgoing")),
    )
    mapped = native_physical_vertex(graph, 4, local)
    expected = (
        momentum
        * t.TensorExpression.gamma5(4)(b, middle)
        * t.TensorExpression.dirac_gamma(4)(middle, a, a)
    )
    assert mapped == expected and mapped != local
    assert set(mapped.axes) == set(local.axes)
    assert native_physical_vertex(graph, 4, mapped) == local
    # The same operation cannot silently accept a missing port or a four-fermion
    # endpoint layout outside the certified model envelope.
    with pytest.raises(ValueError, match="bispinor ports"):
        native_physical_vertex(graph, 4, t.TensorExpression(momentum))
    graph.half_edges += ((9, 4, 10, "incoming"), (12, 4, 11, "outgoing"))
    with pytest.raises(NotImplementedError, match="two Dirac"):
        native_physical_vertex(graph, 4, local)


def test_source_ffv_gate_checks_expression_and_slots_not_names():
    pytest.importorskip("symbolica")
    from ufo_model_loader.symbolica_processing import wrap_indices

    from pyamplicol.loops.model_expressions import parse_exact_ufo
    from pyamplicol.loops.vertex_frames import certified_ffv_source

    def source(value, spins=(2, 2, 3)):
        return SimpleNamespace(
            spins=spins, structure=wrap_indices(parse_exact_ufo(value))
        )

    assert certified_ffv_source(source("Gamma(3,2,1)"))
    assert certified_ffv_source(
        source("Gamma(3,2,-1)*ProjM(-1,1)+4*Gamma(3,2,-1)*ProjP(-1,1)")
    )
    assert not certified_ffv_source(source("Gamma(3,1,2)"))
    assert not certified_ffv_source(source("P(-1,1)*Gamma(-1,2,1)", (2, 2)))


@pytest.mark.parametrize(
    "expression",
    (
        "ProjM(2,1)",
        "ProjM(2,1)-ProjP(2,1)",
        "ProjP(2,1)",
        "ProjM(2,1)+ProjP(2,1)",
        "Identity(2,1)",
    ),
)
def test_local_ffs_basis_gate_does_not_extend_process_admission(expression):
    pytest.importorskip("symbolica")
    from ufo_model_loader.symbolica_processing import wrap_indices

    from pyamplicol.loops.model_expressions import parse_exact_ufo
    from pyamplicol.loops.vertex_frames import (
        certified_ffs_source,
        certified_ffv_source,
    )

    structure = SimpleNamespace(
        name="untrusted-name",
        spins=(2, 2, 1),
        structure=wrap_indices(parse_exact_ufo(expression)),
    )
    assert certified_ffs_source(structure)
    assert not certified_ffv_source(structure)
    structure.spins = (2, 2, 3)
    assert not certified_ffs_source(structure)


@pytest.mark.parametrize(
    "expression,spins",
    (
        ("ProjM(1,2)", (2, 2, 1)),
        ("2*ProjM(2,1)", (2, 2, 1)),
        ("Gamma5(2,1)", (2, 2, 1)),
        ("P(-1,3)*Gamma(-1,2,1)", (2, 2, 1)),
        ("Identity(2,1)", (2, 2, 2, 2)),
    ),
)
def test_local_ffs_basis_rejects_untested_source_forms(expression, spins):
    pytest.importorskip("symbolica")
    from ufo_model_loader.symbolica_processing import wrap_indices

    from pyamplicol.loops.model_expressions import parse_exact_ufo
    from pyamplicol.loops.vertex_frames import certified_ffs_source

    assert not certified_ffs_source(
        SimpleNamespace(
            spins=spins, structure=wrap_indices(parse_exact_ufo(expression))
        )
    )
