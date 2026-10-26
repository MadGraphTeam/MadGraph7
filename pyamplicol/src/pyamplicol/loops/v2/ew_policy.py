# SPDX-License-Identifier: 0BSD
"""Fail-closed guards of electroweak loop requests (WP3.2 EWI-0b).

``LoopSlice.generate`` calls these before any counterterm lowering, so the
public generator, the tools, the tests and the artifacts all see them:

* ``electroweak`` decides whether a request is electroweak: a correction
  other than the model's strong one, or a colour-singlet species (photon, Z,
  W, H, Goldstones, leptons, electroweak ghosts) that is perturbative in a
  requested correction and not vetoed. QCD-only requests are not
  electroweak and pass every guard unchanged (their coloured widths keep
  ``artifact.require_zero_coloured_widths``). The strong and the
  electromagnetic orders are identified by the vertices of the gluon and of
  the photon (``correction_orders``), never by their names (genericity audit
  m1): a model whose orders are spelt otherwise takes the same decisions.
* ``require_no_external_photons`` refuses an electroweak request with an
  on-shell external photon (``ExternalPhotonError``). The user's decision
  [U9] (2026-10-25): every on-shell external photon couples with alpha(0),
  not with the loops' coupling (alpha_Gmu in a G_mu model), together with
  the charge-renormalisation term of that change (MadLoop's and OpenLoops'
  default; MASTER_PLAN §4.9 "external-photon conventions"). v2 implements
  it in WP3.2 EWI-6b; until then a virtual with such a photon would be wrong
  at O(alpha) by the photon's light-fermion logarithms, so it is refused.
* ``require_zero_widths`` refuses an electroweak request whose tree currents
  propagate a particle with a width that is not zero, coloured or not
  (``ElectroweakWidthError``). v2 virtuals are real on-shell: loop
  propagators carry no width and the model counterterms renormalise the
  real masses, so a Z, W, H or top width in the trees mixes two schemes
  (DESIGN §6.1's top-width case: a 1.8e-5 double-pole mismatch and a 2.8 %
  finite shift). The user's decision [U8] (2026-10-25): refuse, with the fix
  (the widths fixed to zero, or the complex-mass scheme, WP3.2 EWI-10, which
  v2 does not have yet: every v2 request is a non-CMS request). The trees
  scanned are the Born and the loop-side blocks, before their kernels and
  the counterterm lowering. A counterterm tree (one insertion of a local
  vertex into the blocks of the loop it renormalises) propagates what those
  blocks propagate. Its program's DAG is not scanned: it also holds
  currents no tree of the request completes (g g > g g on loop_qcd_qed_sm:
  a Higgs from the R2 g g H vertex), whose widths enter no amplitude.

* ``feynman_gauge_model`` is the gauge scope of an electroweak request
  (WP3.2 EWI-4b, DESIGN §6.8): Born, blocks, loop lines and counterterms
  all take the 't Hooft-Feynman gauge (xi = 1) with explicit Goldstones and
  ghosts, the gauge of the model's UV and R2 rules, in which a massive
  vector's loop line has rank <= 1 (the unitary k k / M^2 term would break
  rank <= N). ``require_feynman_gauge`` refuses an electroweak request on a
  model that cannot be taken in that gauge (``ElectroweakGaugeError``: its
  source declares no gauge 1, which the loop preparation refuses for every
  loop request already, or a bundle prepared in unitary gauge);
  ``prepare_request_model`` runs it, so the dry run reports it. QCD-only
  requests keep the compiled model's gauge (unitary by default, their
  validated trees); a model compiled in 't Hooft-Feynman gauge keeps it for
  every request. The gauge of a generated request is its compiled model's
  (``LoopSlice.compiled.ir.gauge``).

The width query (``propagated_species``) is the current scan of
``artifact.coloured_widths``, for every species; a DAG that scans itself
natively (A4's ``dag_input.NativeDag``) is asked directly. The two share
``blocks.propagated_species`` once A4 integrates (wp32/PLAN.md §3.2 EWI-0b).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from decimal import Decimal
from typing import Any, NamedTuple

from pyamplicol.api.errors import GenerationError

# UFO spins are 2s + 1.
VECTOR_SPIN = 3


class ElectroweakWidthError(GenerationError):
    """An electroweak request whose tree currents propagate a particle with
    a non-zero width outside the complex-mass scheme."""


class ExternalPhotonError(GenerationError):
    """An electroweak request with an on-shell external photon, whose
    alpha(0) convention v2 does not implement yet."""


class ElectroweakGaugeError(GenerationError):
    """An electroweak request on a model that cannot be taken in
    't Hooft-Feynman gauge (its source declares no gauge 1, or a prepared
    bundle carries another gauge)."""


class CorrectionOrders(NamedTuple):
    """The coupling orders of a model's strong and electromagnetic
    corrections, identified structurally (``correction_orders``); None where
    the model has no such order."""

    strong: str | None
    abelian: str | None
    # The names of the massless adjoint gauge vectors (the gluon) and of
    # the photons whose vertices identify them.
    gluons: frozenset[str] = frozenset()
    photons: frozenset[str] = frozenset()


# UFO colour of the adjoint representation.
ADJOINT = 8


def _gauge_order(terms: Iterable[Any]) -> str | None:
    """The one coupling order every term of ``terms`` carries (a positive
    power), or None if there is none, several, or no term."""

    common: set[str] | None = None
    for term in terms:
        carried = {name for name, power in term.coupling_orders if int(power) > 0}
        common = carried if common is None else common & carried
    if not common or len(common) != 1:
        return None
    (name,) = common
    return name


def correction_orders(compiled: Any, fixed: Mapping[str, Any]) -> CorrectionOrders:
    """The strong and electromagnetic coupling orders of ``compiled``, from
    the vertices of its gauge bosons, never from the orders' names
    (genericity audit m1):

    * strong: the order carried by every minimal gauge vertex of the
      massless physical adjoint vector (the gluon): its three-point vertices
      with a coloured particle-antiparticle pair or with two more of itself;
    * abelian: the order carried by every three-point vertex of the photon
      (``is_photon``; ``fixed`` may bind its mass to zero) with an
      electrically charged particle-antiparticle pair.

    Either is None when the model has no such boson or its vertices carry no
    single common order; a correction that is neither then has no Catani
    operator, and an electroweak classification treats it as electroweak."""

    particles = {str(record.name): record for record in compiled.ir.particles}
    parameters = {record.name: record for record in compiled.ir.parameters}

    def conjugates(a: Any, b: Any) -> bool:
        return str(a.antiname) == str(b.name)

    def massless(record: Any) -> bool:
        return is_zero(str(record.mass), parameters, fixed)

    gluons = {
        name
        for name, record in particles.items()
        if record.auxiliary_kind is None
        and int(record.spin) == VECTOR_SPIN
        and abs(int(record.color)) == ADJOINT
        and int(record.ghost_number) == 0
        and not record.goldstoneboson
        and massless(record)
    }
    photons = {
        name
        for name, record in particles.items()
        if record.auxiliary_kind is None and is_photon(record, parameters, fixed)
    }
    strong_terms, abelian_terms = [], []
    for term in compiled.ir.vertex_terms:
        if len(term.particles) != 3:
            continue
        records = [particles.get(name) for name in term.particles]
        if any(record is None for record in records):
            continue
        for index, name in enumerate(term.particles):
            others = [records[k] for k in range(3) if k != index]
            pair = conjugates(*others)
            if name in gluons and (
                all(str(r.name) in gluons for r in others)
                or (pair and abs(int(others[0].color)) != 1)
            ):
                strong_terms.append(term)
                break
            if name in photons and pair and float(others[0].charge) != 0:
                abelian_terms.append(term)
                break
    return CorrectionOrders(
        _gauge_order(strong_terms),
        _gauge_order(abelian_terms),
        frozenset(gluons),
        frozenset(photons),
    )


def electroweak(
    compiled: Any, request: Any, species_orders: Mapping[int, Iterable[str]]
) -> bool:
    """Whether ``request`` is an electroweak loop request: a correction
    other than the model's strong one (``correction_orders``: identified by
    the gluon's vertices, not by its name), or a non-vetoed colour-singlet
    species that is perturbative in a requested correction
    (``species_orders``: |PDG| -> coupling orders, the prepared structure's
    ``perturbative_species_orders``)."""

    corrections = set(request.corrections)
    strong = correction_orders(compiled, request.fixed_parameters).strong
    if corrections - {strong}:
        return True
    veto = set(request.particle_veto)
    colour = {abs(int(p.pdg_code)): abs(int(p.color)) for p in compiled.ir.particles}
    return any(
        colour.get(abs(int(pdg))) == 1 and corrections & set(orders)
        for pdg, orders in species_orders.items()
        if abs(int(pdg)) not in veto
    )


def require_feynman_gauge(compiled: Any, request: Any) -> None:
    """Refuse an electroweak ``request`` on a model that cannot be taken in
    't Hooft-Feynman gauge (``ElectroweakGaugeError``): its source declares
    no gauge 1 (``models.loading.declared_model_gauges``; the UFO's rules
    are written for the gauges it declares), or it is a prepared bundle in
    another gauge."""

    from pyamplicol.models.loading import declared_model_gauges

    if compiled.ir.gauge == "feynman":
        return
    declared = declared_model_gauges(compiled)
    reason = None
    if declared is None or 1 not in declared:
        reason = (
            "its source declares no gauge"
            if declared is None
            else f"its source declares only the gauges {list(declared)}"
        )
    elif compiled.prepared_bundle is not None:
        reason = f"it is a model bundle prepared in {compiled.ir.gauge} gauge"
    if reason is None:
        return
    raise ElectroweakGaugeError(
        f"{request.process!r}: an electroweak request (corrections "
        f"{list(request.corrections)}) needs the 't Hooft-Feynman gauge (gauge 1) "
        f"of its loop model throughout (Born, blocks, loop lines and "
        f"counterterms; the gauge of the model's UV and R2 rules), and "
        f"{compiled.name!r} cannot be taken in it: {reason}. Use a UFO model "
        "that declares gauge 1 (e.g. loop_qcd_qed_sm), or restrict the request "
        "to QCD corrections"
    )


def feynman_gauge_model(compiled: Any, request: Any) -> Any:
    """``compiled`` in 't Hooft-Feynman gauge for the electroweak
    ``request`` (``models.loading.with_model_gauge``; module docstring),
    refused by ``require_feynman_gauge``."""

    from pyamplicol.models.loading import with_model_gauge

    require_feynman_gauge(compiled, request)
    return with_model_gauge(compiled, "feynman")


def is_zero(name: str, parameters: Mapping[str, Any], fixed: Mapping[str, Any]) -> bool:
    """Whether the mass or width parameter ``name`` is zero: ``ZERO``, a
    compile-time zero or fixed to zero by ``fixed``."""

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
        and is_zero(str(record.mass), parameters, fixed)
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
        f"{list(request.corrections)}. Electroweak corrections take the "
        "external-photon convention alpha(0): every on-shell external photon "
        "couples with alpha(0) instead of the loops' coupling, with the "
        "matching charge-renormalisation term (MadLoop's and OpenLoops' "
        "default; the user's decision of 2026-10-25), which the v2 engine does "
        "not implement yet (WP3.2 EWI-6b); without it the virtual is wrong at "
        "O(alpha). QCD corrections to such processes are available"
    )


def propagated_species(dag: Any) -> tuple[tuple[int, bool], ...]:
    """Distinct ``(particle id, amputated)`` of the non-source,
    non-auxiliary currents of ``dag``, sorted; ``amputated``: the current's
    momentum leaves exactly one leg out (a Born root contracts it with the
    remaining leg; a loop attached to it would be an external self-energy).
    A DAG with its own scan (``propagated_species``) is asked for it."""

    scan = getattr(dag, "propagated_species", None)
    if scan is not None:
        return tuple(scan())
    sources = set(dag.sources)
    legs = (1 << len(dag.process.legs)) - 1
    return tuple(
        sorted(
            {
                (
                    int(current.index.particle_id),
                    (legs & ~current.index.momentum_mask).bit_count() == 1,
                )
                for current_id, current in enumerate(dag.currents)
                if current_id not in sources and current.index.auxiliary_kind is None
            }
        )
    )


def propagating_widths(
    dags: Sequence[Any], compiled: Any, fixed: Mapping[str, Any]
) -> dict[str, str]:
    """``{particle: width parameter}`` of the particles, coloured or not,
    that a non-amputated tree current of ``dags`` (``propagated_species``)
    propagates with a width that is not zero: neither ``ZERO``, a
    compile-time zero nor fixed to zero."""

    particles = {int(record.pdg_code): record for record in compiled.ir.particles}
    parameters = {record.name: record for record in compiled.ir.parameters}
    out: dict[str, str] = {}
    for dag in dags:
        if dag is None:
            continue
        for particle_id, amputated in propagated_species(dag):
            record = particles.get(particle_id)
            if amputated or record is None:
                continue
            width = str(record.width)
            if not is_zero(width, parameters, fixed):
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
        "counterterms renormalise the real masses), so a width in the Born, "
        "the loop-side or the counterterm trees mixes two schemes. Fix the "
        f"widths to zero in [loop.preparation] fixed_parameters ({hint}); "
        "resonant electroweak bosons need the complex-mass scheme, which the "
        "v2 engine does not implement yet (WP3.2 EWI-10)"
    )


__all__ = [
    "ADJOINT",
    "CorrectionOrders",
    "ElectroweakGaugeError",
    "ElectroweakWidthError",
    "ExternalPhotonError",
    "correction_orders",
    "electroweak",
    "external_photons",
    "feynman_gauge_model",
    "is_photon",
    "is_zero",
    "propagated_species",
    "propagating_widths",
    "require_feynman_gauge",
    "require_no_external_photons",
    "require_zero_widths",
]
