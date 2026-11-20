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
    "catani_record",
    "correlations",
    "correlator",
    "load",
    "matrices",
    "operators",
    "record_applies",
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


# -- the native Catani operator (M-FINAL MF-8G-3) ---------------------------


def correlator(orbits: Any, trees: Any) -> Any:
    """The executor's colour input of Catani's operator
    (``NativeDipoleOrbits.correlator``): the stored orbits with the tree
    program's Born components (``NativeTrees.components``: basis element,
    Born order, N_c grade, folded at N_c = 3 as ``assemble``'s Born vectors)
    and every (bra, ket) pair of Born orders. ``V2Executor.evaluate(...,
    dipoles=)`` then returns per point sum_rows Re <A_a|T_i.T_j|A_b> for each
    pair, contracted over the orbits (no dense operator)."""

    orders = {order: k for k, order in enumerate(trees.born_orders)}
    components = [
        (int(b), orders[order], str(Fraction(3) ** int(grade)))
        for b, order, grade in trees.components
    ]
    sandwiches = [(a, b) for a in range(len(orders)) for b in range(len(orders))]
    return orbits.correlator(components, len(orders), sandwiches)


def correlations(
    values: Any, labels: Any, born_orders: Any, real: Any
) -> dict[tuple[Any, Any], dict[tuple[int, int], Any]]:
    """One point's ``V2Executor.evaluate(..., dipoles=)`` output as
    {(bra order, ket order): {(label i, label j): value}} over the coloured
    leg ``labels`` (the correlator's order) and ``born_orders`` (the
    correlator's sandwiches: every pair); ``real`` reads a carrier value."""

    c = len(labels)
    out: dict[tuple[Any, Any], dict[tuple[int, int], Any]] = {}
    k = 0
    for a in born_orders:
        for b in born_orders:
            block = {}
            for i in range(c):
                for j in range(c):
                    if i != j:
                        block[labels[i], labels[j]] = real(values[k + i * c + j][0])
            out[a, b] = block
            k += c * c
    if k != len(values):
        raise ValueError("colour correlations of another layout")
    return out


def catani_record(slice_: Any) -> dict[str, Any] | None:
    """What Catani's QCD operator reads of a slice, for a stored process
    (``native_process.NativeProcess.export``; ``assemble._NativePoint.
    _stored_qcd_poles``): per coloured leg (label, representation, gamma_j
    as a fraction text), the strong order, the predicted sectors, the
    default of alpha_s and the conditions under which the record holds
    (``record_applies``): every adjustable mass parameter the leg masses and
    beta_0's light species read, with whether it was zero. None where the
    stored route makes no prediction: a request other than full colour
    QCD-only, a massive coloured leg, or an operator the slice does not
    cover (the live slice's ``_PointEvaluation._qcd_poles`` serves those)."""

    from . import assemble
    from .families import ZERO, mass_name

    request = slice_.request
    strong = assemble._correction_orders(slice_).strong
    if (
        request.accuracy != "full"
        or strong is None
        or slice_.alpha_s_parameter is None
        or tuple(request.corrections) != (strong,)
    ):
        return None
    compiled = slice_.compiled
    values = assemble._kernel_parameters(compiled, dict(request.fixed_parameters))
    fixed = set(request.fixed_parameters)
    conditions: dict[str, bool] = {}

    def condition(mass: Any) -> None:
        mass = mass_name(mass)
        if mass != ZERO and mass not in fixed:
            conditions[mass] = values.get(mass) == 0

    for record in slice_.data["particles"]:
        if abs(int(record["color"])) != 1:
            condition(record["mass"])
    beta0 = assemble._strong_beta0(slice_, strong, values)
    representations = {leg.label: leg.representation for leg in slice_.basis.legs}
    legs = []
    for leg in slice_.families.legs:
        label = leg.index + 1
        representation = representations[label]
        if representation == 1:
            continue
        mass = mass_name(leg.mass.mass)
        if not (mass == ZERO or values.get(mass) == 0):
            return None
        gamma = assemble._strong_leg_gamma(slice_, leg, beta0)
        if gamma is None or assemble.colour_casimir(representation) is None:
            return None
        legs.append([label, int(representation), str(Fraction(gamma))])
    predicted = sorted(assemble._predicted_sectors(slice_))
    return {
        "strong": strong,
        "legs": sorted(legs),
        "predicted": [[list(pair) for pair in sector] for sector in predicted],
        "alpha_s": str(values[slice_.alpha_s_parameter]),
        "conditions": sorted([name, zero] for name, zero in conditions.items()),
    }


def record_applies(record: Any, parameters: Any) -> bool:
    """Whether a stored Catani record (``catani_record``) holds at a point's
    parameters (its caller and fixed values; absent ones keep their
    defaults): every recorded mass parameter is zero exactly where it was."""

    if record is None:
        return False
    for name, zero in record["conditions"]:
        value = parameters.get(name)
        if value is not None and (Fraction(value) == 0) != zero:
            return False
    return True
