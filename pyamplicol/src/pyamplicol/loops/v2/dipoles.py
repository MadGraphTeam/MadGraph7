# SPDX-License-Identifier: 0BSD
"""IR dipole operators per relabelling orbit (WP3.1 package C4).

Catani's pole prediction needs ``C_ij[b, b'] = <b|T_i.T_j|b'>`` at N_c = 3
over the Born colour basis for every pair of coloured legs i != j
(``assemble.LoopSlice.dipoles``, the Python oracle). Placement (a) of
``RUST-MIGRATION.md`` §10 item 6: the operators are built **at generation**
in rusticol-core (``loop_v2::dipole_orbits::DipoleOrbits::build``: one exact
entry per orbit of (i, j, b, b') under the relabellings of identical
external adjoint legs that map the basis onto itself, colour conservation
checked exactly), **stored** in a portable form (``FORMAT``: JSON with no
paths, user names or host state; 133 kB at g g > g g g g g, 538 kB at
u u~ > g g g g g) and **expanded at load** by orbit traversal with index
arithmetic only: no Symbolica, no licence, no basis tensors.

This module is the thin Python binding of that Rust API: the Python
runtime's path (``LoopSlice.dipoles`` under the "dipoles" stage of
``native_cells.NATIVE_STAGES``) and the artifact writer's. The native
runtime (generation-scale-2's I5c) calls the same Rust API in-crate; DESIGN
§12.4 documents the API and the stored form.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Any

from . import native_cells

__all__ = [
    "FORMAT",
    "build",
    "load",
    "matrices",
    "operators",
    "slice_operators",
    "stored",
]

# ``loop_v2::dipole_orbits::FORMAT``: the version of the stored form.
FORMAT = "pyamplicol-dipole-orbits-1"

Operators = dict[tuple[int, int], tuple[tuple[Fraction, ...], ...]]


def build(slice_: Any, threads: int | None = None) -> Any:
    """The orbit representatives of a slice's Born basis, built and checked
    exactly in rusticol-core (a ``NativeDipoleOrbits``; at generation). The
    candidate relabellings are the external adjoint legs of one particle."""

    return native_cells.dipole_orbits(slice_, threads)


def stored(orbits: Any) -> str:
    """The portable stored form of ``orbits`` (``FORMAT``)."""

    text: str = orbits.to_json()
    return text


def load(text: str) -> Any:
    """``NativeDipoleOrbits`` from its stored form, validated before any
    table of its index space is allocated (``DipoleOrbits::validate``:
    format; coloured legs with distinct labels in leg order and
    representations 3, 3bar or 8; an index space c^2 B^2 that fits; scales,
    generator permutations, index ranges and representatives off the
    diagonal; orbits large enough to cover every entry). A crafted member
    raises ``ValueError``; allocations of the expansions are fallible too.
    ``expand`` checks the exact orbit cover and ``check_conservation`` the
    values. No Symbolica state and no licence are touched."""

    return native_cells._native().NativeDipoleOrbits.from_json(text)


def operators(orbits: Any) -> Operators:
    """Every exact entry, as ``LoopSlice.dipoles`` returns them:
    ``{(i, j): rows}`` over coloured leg labels i != j in leg order, rows b
    and columns b' of the Born basis."""

    return native_cells.dipoles(None, orbits)


def matrices(orbits: Any) -> dict[tuple[int, int], tuple[tuple[float, ...], ...]]:
    """Every entry in binary64 (one correct rounding of each exact value),
    keyed and ordered as ``operators``."""

    labels = [label for label, _representation in orbits.coloured]
    count = len(labels)
    dense = orbits.expand_f64()
    size = len(dense) // (count * count) if count else 0
    width = round(size**0.5)
    out: dict[tuple[int, int], tuple[tuple[float, ...], ...]] = {}
    for i, a in enumerate(labels):
        for j, b in enumerate(labels):
            if i == j:
                continue
            start = (i * count + j) * size
            out[a, b] = tuple(
                tuple(dense[start + row * width : start + (row + 1) * width])
                for row in range(width)
            )
    return out


def slice_operators(slice_: Any) -> Operators | None:
    """``LoopSlice.dipoles()`` under the "dipoles" stage (built natively and
    expanded), or None when the stage is off (the switch in
    ``LoopSlice.dipoles`` caches it)."""

    if "dipoles" not in native_cells.NATIVE_STAGES:
        return None
    return operators(build(slice_))
