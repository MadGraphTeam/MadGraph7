# SPDX-License-Identifier: 0BSD
"""Explicit source vertex endpoints, separate from propagator conventions.

The standard UFO Dirac FFV source slots and the physical HELAS current slots
have opposite endpoint order. This adapter uses dependency-owned typed index
renaming on a localized vertex. It does not transpose gamma matrices, change
momenta, or apply a dimensional gamma5 prescription. Physical callers still
have to establish the source tensor's conformance domain.
"""

from __future__ import annotations


def validate_vertex_spinor_frame(frame):
    if frame not in (None, "ufo", "native-physical"):
        raise ValueError("vertex spinor frame must be ufo or native-physical")


def native_physical_vertex(graph, vertex_id, lorentz):
    """Swap only the two local Dirac endpoint Slot keys in the native frame."""
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
