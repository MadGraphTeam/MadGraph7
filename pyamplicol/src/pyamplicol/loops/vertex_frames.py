# SPDX-License-Identifier: 0BSD
"""Explicit source vertex endpoints, separate from propagator conventions.

The standard UFO Dirac source slots and physical HELAS current slots have
opposite endpoint order. Scalar Dirac currents additionally exchange the native
chiral projectors, as in the existing tree compiler. Both operations act on
localized dependency tensors; propagators and momenta retain their own frame.
Physical callers must establish the source tensor's conformance domain. This
four-dimensional boundary is not a dimensional gamma5 prescription.
"""

from __future__ import annotations


def validate_vertex_spinor_frame(frame):
    if frame not in (None, "ufo", "native-physical"):
        raise ValueError("vertex spinor frame must be ufo or native-physical")


def native_physical_vertex(graph, vertex_id, lorentz):
    """Translate certified local Dirac vertices into the physical current frame."""
    from symbolica.community import hepkit as h
    from symbolica.community import tensor as t

    fermion_edges = {edge.id for edge in graph.edges if edge.spin == 2}
    labels = tuple(
        h.Symbols.half_edge()(half, 1)
        for half, vertex, edge, _flow in graph.half_edges
        if vertex == vertex_id and edge in fermion_edges
    )
    if not labels:
        return lorentz
    if len(labels) != 2:
        raise NotImplementedError("source frame requires two Dirac vertex endpoints")
    axes = tuple(lorentz.axes)
    slots = tuple(
        slot
        for label in labels
        for slot in axes
        if slot.index == label and slot.representation == t.Representation.bis(4)
    )
    if len(slots) != 2 or slots[0].index == slots[1].index:
        raise ValueError("source vertex lacks its two distinct bispinor ports")
    # Do not intern: renaming native Slot keys already preserves compound
    # half-edge labels, while interning would replace them with cooked labels.
    result = lorentz.rename_indices(
        {slots[0]: slots[1].index, slots[1]: slots[0].index}
    )
    edge_spins = {edge.id: edge.spin for edge in graph.edges}
    spins = sorted(
        edge_spins[edge]
        for _half, vertex, edge, _flow in graph.half_edges
        if vertex == vertex_id
    )
    if spins == [1, 2, 2]:
        # Whole-vertex endpoint exchange suffices for the admitted FFV bases.
        # A diagonal FFS source matrix cannot detect the additional physical
        # current convention: UFO ProjM/ProjP become native projp/projm. This is
        # the same mapping as models.tensors._LorentzContext, independently
        # checked on directed FFS -> propagator -> chiral FFV ALOHA currents.
        # Simultaneous native replacements retain all scalar coefficients and
        # leave the propagator (including its mass term) untouched.
        from symbolica import Replacement, S

        left, right = S("loop_vertex_frame::left_", "loop_vertex_frame::right_")
        minus = t.TensorExpression.projm(4)(left, right).to_expression()
        plus = t.TensorExpression.projp(4)(left, right).to_expression()
        result = result.replace_multiple(
            [Replacement(minus, plus), Replacement(plus, minus)]
        )
    if set(result.axes) != set(axes):
        raise ValueError("source endpoint mapping changed the native external frame")
    return result


def certified_ffv_source(structure):
    """Certified FFV bases and their exact source chiral linear combinations."""
    from ufo_model_loader.symbolica_processing import wrap_indices

    from .model_expressions import parse_exact_ufo

    sources = (
        "Gamma(3,2,1)",
        "Gamma(3,2,-1)*ProjM(-1,1)",
        "Gamma(3,2,-1)*ProjM(-1,1)+2*Gamma(3,2,-1)*ProjP(-1,1)",
        "Gamma(3,2,-1)*ProjP(-1,1)",
        # FFV3/5 are exact linear combinations of independently checked FFV2/6.
        "Gamma(3,2,-1)*ProjM(-1,1)-2*Gamma(3,2,-1)*ProjP(-1,1)",
        "Gamma(3,2,-1)*ProjM(-1,1)+4*Gamma(3,2,-1)*ProjP(-1,1)",
    )
    return tuple(structure.spins) == (2, 2, 3) and any(
        (structure.structure - wrap_indices(parse_exact_ufo(source))).expand() == 0
        for source in sources
    )


def certified_ffs_source(structure):
    """Locally checked four-dimensional Dirac/scalar source bases.

    This identifies the exact local source expressions, including Yukawa and
    Goldstone chiral bases. It does not certify their scalar couplings, an
    internal propagator, a complete source process or a dimensional prescription.
    Physical source dispatch keeps its separate process-admission gate.
    """
    from ufo_model_loader.symbolica_processing import wrap_indices

    from .model_expressions import parse_exact_ufo

    sources = (
        "ProjM(2,1)",
        "ProjM(2,1)-ProjP(2,1)",
        "ProjP(2,1)",
        "ProjM(2,1)+ProjP(2,1)",
        "Identity(2,1)",
    )
    return tuple(structure.spins) == (2, 2, 1) and any(
        (structure.structure - wrap_indices(parse_exact_ufo(source))).expand() == 0
        for source in sources
    )


def certified_source_dirac_frame(structure):
    """Locally checked four-dimensional Dirac vertices, with exact source ports."""
    return certified_ffv_source(structure) or certified_ffs_source(structure)
