# SPDX-License-Identifier: 0BSD
"""Fail-closed guards of electroweak loop requests (WP3.2 EWI-0b).

``LoopSlice.generate`` calls these before any counterterm lowering, so the
public generator, the tools, the tests and the artifacts all see them:

* ``electroweak`` decides whether a request is electroweak: a correction
  other than ``QCD``, or a colour-singlet species (photon, Z, W, H,
  Goldstones, leptons, electroweak ghosts) that is perturbative in a
  requested correction and not vetoed.
* ``require_no_external_photons`` refuses an electroweak request with an
  on-shell external photon (``ExternalPhotonError``). Such a photon couples
  with alpha(0), not with the loops' coupling, plus the charge
  renormalisation term of that change (MadLoop's and OpenLoops' default,
  MASTER_PLAN §4.9 "external-photon conventions"); v2 does not implement it
  yet (WP3.2 EWI-6b; the user's decision [U9] is pending), and a virtual
  without it is wrong at O(alpha) by the photon's light-fermion logarithms.
* ``require_zero_widths`` refuses an electroweak request whose tree currents
  (the Born and the loop-side blocks) propagate a particle with a width that
  is not zero, coloured or not (``ElectroweakWidthError``). v2 virtuals are
  real on-shell: loop propagators carry no width and the model counterterms
  renormalise the real masses, so a Z, W, H or top width in the trees mixes
  two schemes (DESIGN §6.1's top-width case: a 1.8e-5 double-pole mismatch
  and a 2.8 % finite shift). The fix is to fix the widths to zero, or the
  complex-mass scheme, which v2 does not have yet (WP3.2 EWI-10): every v2
  request is a non-CMS request. This is the recommended default of the
  user's decision [U8] (refuse rather than zero the widths silently).

QCD-only requests are not electroweak and pass unchanged (their coloured
widths keep ``artifact.require_zero_coloured_widths``). The width query
mirrors ``artifact.coloured_widths`` for every species; the two are to share
one propagator query once A4 (which edits ``coloured_widths``) integrates
(wp32/PLAN.md §3.2 EWI-0b).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from decimal import Decimal
from typing import Any

from pyamplicol.api.errors import GenerationError

# UFO spins are 2s + 1.
VECTOR_SPIN = 3


class ElectroweakWidthError(GenerationError):
    """An electroweak request whose tree currents propagate a particle with
    a non-zero width outside the complex-mass scheme."""


class ExternalPhotonError(GenerationError):
    """An electroweak request with an on-shell external photon, whose
    external-photon convention v2 does not implement yet."""


def electroweak(
    compiled: Any, request: Any, species_orders: Mapping[int, Iterable[str]]
) -> bool:
    """Whether ``request`` is an electroweak loop request: a correction
    other than QCD, or a non-vetoed species that is perturbative in a
    requested correction (``species_orders``: |PDG| -> coupling orders, the
    prepared structure's ``perturbative_species_orders``) and a colour
    singlet."""

    corrections = set(request.corrections)
    if corrections - {"QCD"}:
        return True
    veto = set(request.particle_veto)
    colour = {abs(int(p.pdg_code)): abs(int(p.color)) for p in compiled.ir.particles}
    return any(
        colour.get(abs(int(pdg))) == 1 and corrections & set(orders)
        for pdg, orders in species_orders.items()
        if abs(int(pdg)) not in veto
    )


def _zero(name: str, parameters: Mapping[str, Any], fixed: Mapping[str, Any]) -> bool:
    """Whether the mass or width parameter ``name`` is zero: ``ZERO``, a
    compile-time zero or fixed to zero."""

    from pyamplicol.models.compiler_kernels import _is_compile_time_zero_parameter

    if _is_compile_time_zero_parameter(name, parameters):
        return True
    value = fixed.get(name)
    return value is not None and Decimal(str(value)) == 0


def is_photon(record: Any, parameters: Mapping[str, Any], fixed) -> bool:
    """A massless, neutral, colourless physical vector (the photon)."""

    return (
        int(record.spin) == VECTOR_SPIN
        and abs(int(record.color)) == 1
        and float(record.charge) == 0
        and int(record.ghost_number) == 0
        and not record.goldstoneboson
        and _zero(str(record.mass), parameters, fixed)
    )


def external_photons(
    compiled: Any, process: Any, fixed: Mapping[str, Any]
) -> tuple[int, ...]:
    """The 1-based labels of the external photons of ``process`` (a model
    process IR)."""

    particles = {int(record.pdg_code): record for record in compiled.ir.particles}
    parameters = {record.name: record for record in compiled.ir.parameters}
    return tuple(
        int(leg.label)
        for leg in process.legs
        if (record := particles.get(int(leg.pdg))) is not None
        and is_photon(record, parameters, fixed)
    )


def require_no_external_photons(
    compiled: Any, request: Any, process: Any, fixed: Mapping[str, Any]
) -> None:
    """Refuse an electroweak request with an on-shell external photon
    (``ExternalPhotonError``; module docstring)."""

    photons = external_photons(compiled, process, fixed)
    if not photons:
        return
    raise ExternalPhotonError(
        f"{request.process!r}: on-shell external photons (legs "
        f"{list(photons)}) in a request with corrections "
        f"{list(request.corrections)}. Electroweak corrections need the "
        "external-photon convention: alpha(0) for every on-shell external "
        "photon instead of the loops' coupling, with the matching charge "
        "renormalisation term (MadLoop's and OpenLoops' default), which the v2 "
        "engine does not implement yet; without it the virtual is wrong at "
        "O(alpha). QCD corrections to such processes are available, and "
        "loop.engine='legacy' covers electroweak 2 -> 2 controls"
    )


def propagating_widths(
    dags: Sequence[Any], compiled: Any, fixed: Mapping[str, Any]
) -> dict[str, str]:
    """``{particle: width parameter}`` of the particles that a tree current
    of ``dags`` propagates with a width that is not zero: neither ``ZERO``,
    a compile-time zero nor fixed to zero. Source currents (external legs),
    auxiliary currents and currents of all legs but one (amputated: a Born
    root contracts them with the remaining leg) do not count, as in
    ``artifact.coloured_widths``, for coloured and colourless particles."""

    particles = {int(record.pdg_code): record for record in compiled.ir.particles}
    parameters = {record.name: record for record in compiled.ir.parameters}
    out: dict[str, str] = {}
    for dag in dags:
        if dag is None:
            continue
        sources = set(dag.sources)
        legs = (1 << len(dag.process.legs)) - 1
        for current_id, current in enumerate(dag.currents):
            index = current.index
            if (
                current_id in sources
                or index.auxiliary_kind is not None
                or (legs & ~index.momentum_mask).bit_count() == 1
            ):
                continue
            record = particles.get(int(index.particle_id))
            if record is None:
                continue
            width = str(record.width)
            if not _zero(width, parameters, fixed):
                out[str(record.name)] = width
    return out


def require_zero_widths(
    compiled: Any, request: Any, dags: Sequence[Any], fixed: Mapping[str, Any]
) -> None:
    """Refuse an electroweak request whose tree currents propagate a
    particle with a non-zero width (``ElectroweakWidthError``; module
    docstring)."""

    widths = propagating_widths(dags, compiled, fixed)
    if not widths:
        return
    named = ", ".join(f"{name} ({width})" for name, width in sorted(widths.items()))
    hint = ", ".join(f'{width} = "0"' for width in sorted(set(widths.values())))
    raise ElectroweakWidthError(
        f"{request.process!r}: an electroweak request (corrections "
        f"{list(request.corrections)}) whose tree currents propagate particles "
        f"with a width that is not fixed to zero: {named}. v2 virtuals are "
        "real on-shell (loop propagators carry no width and the model "
        "counterterms renormalise the real masses), so a width in the Born or "
        "the loop-side trees mixes two schemes. Fix the widths to zero in "
        f"[loop.preparation] fixed_parameters ({hint}); resonant electroweak "
        "bosons need the complex-mass scheme, which the v2 engine does not "
        "implement yet"
    )


__all__ = [
    "ElectroweakWidthError",
    "ExternalPhotonError",
    "electroweak",
    "external_photons",
    "is_photon",
    "propagating_widths",
    "require_no_external_photons",
    "require_zero_widths",
]
