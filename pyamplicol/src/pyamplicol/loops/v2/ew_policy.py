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
* ``external_photon_scheme`` is the external-photon convention of an
  electroweak request with on-shell external photons (WP3.2 EWI-6b). The
  user's decision [U9] (2026-10-25): a tagged on-shell external photon
  couples with alpha(0), not with the loops' coupling (alpha_Gmu in a G_mu
  model), together with the charge-renormalisation term of that change
  (MadLoop's and OpenLoops' default; MASTER_PLAN §4.9 "external-photon
  conventions"). The tags are per leg since MS-PHOTON (m-sm/PLAN.md Q1 (a),
  MG5_aMC's ``!a!``; ``photon_tags``, ``SliceRequest.external_photons``,
  "final" by default); an untagged photon keeps the loops' coupling and its
  gamma_gamma pole. A model that cannot take the convention for a tagged
  photon (no conversion counterterm, no adjustable 1/alpha(0), loops that
  couple with alpha(0)) is refused (``ExternalPhotonError``,
  ``external-photon-scheme``): without it the virtual would be wrong at
  O(alpha) by the photon's light-fermion logarithms.
* ``require_zero_widths`` refuses a real on-shell electroweak request whose
  tree currents propagate a particle with a width that is not zero, coloured
  or not (``ElectroweakWidthError``, listed key ``electroweak-width``). Real
  on-shell virtuals: loop propagators carry no width and the model
  counterterms renormalise the real masses, so a Z, W, H or top width in the
  trees mixes two schemes (DESIGN §6.1's top-width case: a 1.8e-5
  double-pole mismatch and a 2.8 % finite shift). The user's decision [U8]
  (2026-10-25): refuse, with the fix (the widths fixed to zero, or the
  complex-mass scheme, ``loops.model_cms``, whose model propagates every
  particle of a width with its complex pole mass and width zero, so this
  check passes on it). The trees
  scanned are the Born and the loop-side blocks, before their kernels and
  the counterterm lowering. A counterterm tree (one insertion of a local
  vertex into the blocks of the loop it renormalises) propagates what those
  blocks propagate. Its program's DAG is not scanned: it also holds
  currents no tree of the request completes (g g > g g on loop_qcd_qed_sm:
  a Higgs from the R2 g g H vertex), whose widths enter no amplitude.

* ``gauge_closed_veto`` adds to a ``particle_veto`` the Faddeev-Popov
  ghosts of the vector bosons it removes (``resolve_request``; any
  correction): a ghost loop without its boson breaks the gauge
  cancellations it exists for.
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

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
from typing import Any, NamedTuple

from .refusals import LoopRequestUnsupported

# UFO spins are 2s + 1.
FERMION_SPIN = 2
VECTOR_SPIN = 3


class ElectroweakWidthError(LoopRequestUnsupported):
    """An electroweak request whose tree currents propagate a particle with
    a non-zero width outside the complex-mass scheme (listed key
    ``electroweak-width``)."""

    def __init__(self, message: str) -> None:
        super().__init__("electroweak-width", message)


class ExternalPhotonError(LoopRequestUnsupported):
    """A listed external-photon refusal (``refusals``): tagged on-shell
    external photons on a model or request that cannot take the alpha(0)
    convention (``external-photon-scheme``, ``external_photon_scheme``), or a
    tag list naming a leg that is not an on-shell external photon
    (``external-photon-legs``, ``photon_tags``)."""


class ElectroweakGaugeError(LoopRequestUnsupported):
    """An electroweak request on a model that cannot be taken in
    't Hooft-Feynman gauge (its source declares no gauge 1, or a prepared
    bundle carries another gauge)."""

    # The listed refusal ``feynman-gauge`` (``refusals``).
    def __init__(self, message: str) -> None:
        super().__init__("feynman-gauge", message)


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


def request_corrections(compiled: Any, request: Any) -> tuple[str, ...]:
    """The correction orders of a loop request: its ``corrections``, or,
    when it names none (``SliceRequest.corrections`` is None), the model's
    strong correction (``correction_orders``: the order the gluon's minimal
    vertices carry, never an order's name; genericity audit 2, m7). A
    request naming none of a model without a strong correction is refused."""

    if request.corrections is not None:
        return tuple(request.corrections)
    strong = correction_orders(compiled, request.fixed_parameters).strong
    if strong is None:
        from pyamplicol.api.errors import GenerationError

        raise GenerationError(
            f"{request.process!r}: the request names no correction orders and "
            "the model has no strong correction (no massless adjoint vector "
            "whose minimal vertices carry one coupling order); name the "
            "corrections (SliceRequest.corrections, [process] "
            "loop_coupling_orders)"
        )
    return (strong,)


def electroweak(
    compiled: Any, request: Any, species_orders: Mapping[int, Iterable[str]]
) -> bool:
    """Whether ``request`` is an electroweak loop request: a correction
    other than the model's strong one (``correction_orders``: identified by
    the gluon's vertices, not by its name), or a non-vetoed colour-singlet
    species that is perturbative in a requested correction
    (``species_orders``: |PDG| -> coupling orders, the prepared structure's
    ``perturbative_species_orders``)."""

    corrections = set(request_corrections(compiled, request))
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


def photon_species(compiled: Any, fixed: Mapping[str, Any]) -> frozenset[int]:
    """The |PDG| codes of the model's photon candidates (``is_photon``:
    massless, neutral, colourless physical vectors); more than one makes the
    photon ambiguous (genericity audit 2 G2-m6, ``external_photon_scheme``)."""

    parameters = {record.name: record for record in compiled.ir.parameters}
    return frozenset(
        abs(int(record.pdg_code))
        for record in compiled.ir.particles
        if getattr(record, "auxiliary_kind", None) is None
        and is_photon(record, parameters, fixed)
    )


# ``loop.external_photons`` (``SliceRequest.external_photons``): which
# on-shell external photons are tagged (MG5_aMC's ``!a!``).
PHOTON_TAG_CHOICES = ("final", "all", "none")
PHOTON_TAG_DEFAULT = "final"


def photon_tag_option(value: Any) -> str | tuple[int, ...]:
    """``value`` as a ``SliceRequest.external_photons`` option: one of
    ``PHOTON_TAG_CHOICES`` or the sorted 1-based labels of the tagged legs
    (positive integers, each once); anything else raises ``ValueError``."""

    if isinstance(value, str):
        if value not in PHOTON_TAG_CHOICES:
            raise ValueError(
                "external_photons must be one of "
                f"{', '.join(map(repr, PHOTON_TAG_CHOICES))} or a list of 1-based "
                f"leg labels; got {value!r}"
            )
        return value
    if isinstance(value, bytes | Mapping) or not isinstance(value, Iterable):
        raise ValueError(
            f"external_photons must be a choice or a list of leg labels; got {value!r}"
        )
    labels = tuple(value)
    if any(isinstance(x, bool) or not isinstance(x, int) or x < 1 for x in labels):
        raise ValueError(
            "external_photons lists 1-based leg labels (positive integers); "
            f"got {value!r}"
        )
    if len(set(labels)) != len(labels):
        raise ValueError(f"external_photons repeats a leg: {list(labels)}")
    return tuple(sorted(labels))


def photon_tags(
    compiled: Any, request: Any, process: Any, fixed: Mapping[str, Any]
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """The (tagged, untagged) 1-based labels of the on-shell external photons
    of ``process`` (``external_photons``) under ``request.external_photons``
    (m-sm/PLAN.md Q1 (a), MG5_aMC's ``!a!``): "final" tags the final-state
    ones, "all" every one, "none" none, a list exactly its legs. A listed leg
    that is not an on-shell external photon of the process is refused
    (``ExternalPhotonError`` ``external-photon-legs``). A tagged photon
    couples with alpha(0) when the request takes the electromagnetic
    correction (``external_photon_scheme``); an untagged one keeps the
    loops' coupling and its gamma_gamma pole."""

    photons = external_photons(compiled, process, fixed)
    option = photon_tag_option(getattr(request, "external_photons", PHOTON_TAG_DEFAULT))
    legs = {int(leg.label): leg for leg in process.legs}
    if option == "all":
        tagged = photons
    elif option == "none":
        tagged = ()
    elif option == "final":
        tagged = tuple(label for label in photons if legs[label].side == "final")
    else:
        wrong = [label for label in option if label not in photons]
        if wrong:
            names = {int(r.pdg_code): str(r.name) for r in compiled.ir.particles}

            def describe(label: int) -> str:
                leg = legs.get(label)
                if leg is None:
                    return f"{label} (the process has {len(legs)} legs)"
                name = names.get(int(leg.pdg), str(leg.pdg))
                return f"{label} ({name}, {leg.side})"

            raise ExternalPhotonError(
                "external-photon-legs",
                f"{request.process!r}: loop.external_photons lists the legs "
                f"{', '.join(map(describe, wrong))}, which are not on-shell "
                "external photons (the process's on-shell photons: legs "
                f"{list(photons)})",
            )
        tagged = tuple(option)
    return tagged, tuple(label for label in photons if label not in tagged)


def slice_photon_tags(slice_: Any) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """The (tagged, untagged) on-shell external photons of a generated slice:
    the tagged ones are its ``external_photons`` legs (alpha(0)), the others
    keep the loops' coupling."""

    compiled = slice_.compiled
    parameters = {record.name: record for record in compiled.ir.parameters}
    records = {str(record.name): record for record in compiled.ir.particles}
    fixed = slice_.request.fixed_parameters
    photons = tuple(
        leg.index + 1
        for leg in slice_.families.legs
        if is_photon(records[leg.field], parameters, fixed)
    )
    convention = getattr(slice_, "external_photons", None)
    tagged = set(() if convention is None else convention.legs)
    return (
        tuple(label for label in photons if label in tagged),
        tuple(label for label in photons if label not in tagged),
    )


# MG5_aMC's loop models declare the external-photon conversion
# delta Z_e^{G_mu} - delta Z_e^{alpha(0)} as this counterterm parameter, and
# MG5_aMC applies it by this name outside the UFO (1804.10017 §2;
# ``export_fks.write_rescale_a0gmu_file``): there is no structural marker of
# it in a UFO.
ALPHA0_CONVERSION = "eCoup_DGmua0_UV_EW"
# The UFO/SLHA code of 1/alpha(0) in MG5_aMC's G_mu loop models (aEW0M1), an
# external parameter that enters no coupling of the model.
ALPHA0_INVERSE_CODE = ("SMINPUTS", (1,))


@dataclass(frozen=True)
class ExternalPhotons:
    """The alpha(0) convention of the tagged on-shell external photons of an
    electroweak request (WP3.2 EWI-6b; the user's decision [U9] of
    2026-10-25, per leg since MS-PHOTON: m-sm/PLAN.md Q1 (a), MG5_aMC's
    ``!a!``): a tagged photon couples with alpha(0) instead of the loops'
    alpha_Gmu.

    The Born and every virtual component are multiplied by
    (alpha(0) / alpha_Gmu)^n for n tagged photons (``scale``; the coupling e
    of each tagged photon), and each tagged photon adds the field-like term
    -2 (delta Z_e^{G_mu} - delta Z_e^{alpha(0)}) Born to the virtual of the
    electromagnetic correction (the model's ``conversion`` parameter as an
    external-field factor of the photon leg, ``assemble``), as MadLoop's
    aMC@NLO interface does for tagged photons. Untagged photons are not
    listed: they keep the loops' coupling (``slice_photon_tags``)."""

    # 1-based labels of the tagged on-shell external photons (never empty).
    legs: tuple[int, ...]
    # The external parameter holding 1/alpha(0).
    alpha0_inverse: str
    # The model's counterterm parameter delta Z_e^{G_mu} - delta Z_e^{alpha(0)};
    # None where the loops already couple with alpha(0) (a declared
    # ``alpha0_inverse`` the loops' alpha is the inverse of: no rescaling and
    # no conversion term; genericity audit 2 G2-m5).
    conversion: str | None
    # The electromagnetic correction (coupling order) the term belongs to.
    correction: str
    convention: str = "alpha0"


def external_photon_scheme(
    compiled: Any,
    request: Any,
    process: Any,
    fixed: Mapping[str, Any],
    species_orders: Mapping[int, Iterable[str]],
) -> ExternalPhotons | None:
    """The alpha(0) convention of the tagged on-shell external photons of an
    electroweak ``request`` (``ExternalPhotons``; ``photon_tags``), None
    without a tagged photon (untagged ones need no model input).

    Refused (``ExternalPhotonError`` ``external-photon-scheme``) where the
    convention cannot be applied to a tagged photon: the electromagnetic
    correction is not requested (or the model has none), the model declares
    no conversion parameter (``ALPHA0_CONVERSION``) or no unique adjustable
    1/alpha(0) at ``ALPHA0_INVERSE_CODE`` that the loops' coupling
    (``poles.qed_coupling``) does not depend on (a model whose loops couple
    with alpha(0) or alpha(M_Z) needs another conversion), its value is fixed
    to zero, or ``particle_veto`` removes a species perturbative in the
    electromagnetic correction (the conversion is one counterterm parameter
    over the whole electroweak content). The message names the remedy for
    the model at hand: untagged photons, or a model (restriction) that sets
    1/alpha(0)."""

    from .poles import qed_coupling

    photons, untagged = photon_tags(compiled, request, process, fixed)
    candidates = photon_species(compiled, fixed)
    if (photons or untagged) and len(candidates) > 1:
        # Genericity audit 2 G2-m6: which massless neutral colourless vector
        # is the photon is ambiguous (a dark photon, a massless Z'), so
        # neither the alpha(0) convention nor an untagged photon's
        # gamma_gamma can be assigned to the legs: fail closed.
        raise ExternalPhotonError(
            "external-photon-scheme",
            f"{request.process!r}: the external legs {list(photons + untagged)} "
            "are massless, neutral, colourless vectors, and the model has "
            f"several such species (|PDG| {sorted(candidates)}): which one is the "
            "photon is ambiguous, so neither the alpha(0) convention of a tagged "
            "photon nor the gamma_gamma pole of an untagged one can be assigned. "
            "Use a model with one photon",
        )
    if not photons:
        return None

    def refuse(reason: str, remedy: str) -> ExternalPhotonError:
        return ExternalPhotonError(
            "external-photon-scheme",
            f"{request.process!r}: tagged on-shell external photons (legs "
            f"{list(photons)}; loop.external_photons = "
            f"{_option_text(request.external_photons)}) in a request with "
            f"corrections {list(request.corrections)}. A tagged photon couples "
            "with alpha(0) instead of the loops' coupling, with the matching "
            "charge-renormalisation term (MG5_aMC's alpha(0) formula over the "
            f"tagged legs), which cannot be applied here: {reason}. {remedy}",
        )

    untag = (
        'leave them untagged (loop.external_photons = "none"; they then keep '
        "the loops' coupling and their gamma_gamma pole)"
    )
    abelian = correction_orders(compiled, fixed).abelian
    if abelian is None or abelian not in request.corrections:
        raise refuse(
            "the request does not take the electromagnetic correction"
            if abelian is not None
            else "the model has no electromagnetic coupling order",
            f"Instead {untag}; QCD corrections to such processes are available",
        )
    parameters = {record.name: record for record in compiled.ir.parameters}
    coupling = qed_coupling(compiled, lambda r: is_photon(r, parameters, fixed))
    conversion = getattr(request, "alpha0_conversion", None) or ALPHA0_CONVERSION
    declared = getattr(request, "alpha0_inverse", None)
    if declared is not None:
        # Genericity audit 2 G2-m5: the model's 1/alpha(0) named by the
        # request (``loop.alpha0_inverse_parameter``).
        record = parameters.get(declared)
        if record is None or record.nature != "external":
            raise refuse(
                f"loop.alpha0_inverse_parameter names {declared!r}, which is no "
                "adjustable external parameter of the model"
                + (" (fixed by its restriction)" if record is not None else ""),
                "Instead name the model's adjustable 1/alpha(0), or " + untag,
            )
        value = fixed.get(declared)
        if value is not None and Decimal(str(value)) == 0:
            raise refuse(
                f"{declared} (1/alpha(0)) is fixed to zero",
                f"Instead set {declared} (e.g. 137.035999), or {untag}",
            )
        if coupling is not None and _is_inverse_of(coupling, declared):
            # The loops already couple with alpha(0) (alpha = 1/declared):
            # a tagged photon needs no rescaling and no conversion term.
            return ExternalPhotons(photons, declared, None, abelian)
    raw = compiled.raw_loop_model
    names = {record["name"] for record in raw.collections["counterterm_parameters"]}
    if conversion not in names:
        raise refuse(
            f"the model declares no conversion counterterm {conversion} "
            "(delta Z_e^Gmu - delta Z_e^alpha(0), as MG5_aMC's G_mu loop models"
            + (
                "; MG5_aMC refuses tagged photons on such a model too)"
                if conversion == ALPHA0_CONVERSION
                else "; named by loop.alpha0_conversion_parameter)"
            ),
            f"Instead {untag}, or use a model that declares the conversion with "
            "an adjustable 1/alpha(0) (MG5_aMC's loop_qcd_qed_sm_Gmu under its "
            "restriction a0; another model's names: loop.alpha0_conversion_parameter "
            "and loop.alpha0_inverse_parameter)",
        )
    if declared is not None:
        inverse = parameters[declared]
    else:
        block, code = ALPHA0_INVERSE_CODE
        coded = [
            record
            for record in compiled.ir.parameters
            if (record.lhablock or "").upper() == block
            and tuple(record.lhacode) == code
        ]
        found = [record for record in coded if record.nature == "external"]
        if len(found) != 1:
            raise refuse(
                f"the model has no adjustable 1/alpha(0) at {block} {list(code)} "
                + (
                    f"({', '.join(r.name for r in coded)} fixed by its restriction, "
                    "zero in MG5_aMC's default card of a G_mu loop model)"
                    if coded
                    else "(MG5_aMC's G_mu loop models: aEW0M1)"
                ),
                "Instead compile the model with a restriction that sets it, as "
                "MG5_aMC requires (e.g. its restrict_a0: model.restriction = "
                f'"a0"), or {untag}'
                if coded
                else f"Instead name it (loop.alpha0_inverse_parameter), or {untag}",
            )
        (inverse,) = found
    if coupling is None or inverse.name in {name for _symbol, name in coupling[1]}:
        raise refuse(
            "the loops' electromagnetic coupling is not a G_mu-scheme one that "
            f"leaves {inverse.name} free",
            f"Instead {untag}"
            + (
                "; if the loops couple with alpha(0), name its parameter "
                "(loop.alpha0_inverse_parameter) and no conversion applies"
                if declared is None
                else ""
            ),
        )
    value = fixed.get(inverse.name)
    if value is not None and Decimal(str(value)) == 0:
        raise refuse(
            f"{inverse.name} (1/alpha(0)) is fixed to zero",
            f"Instead set {inverse.name} (e.g. 137.035999), or {untag}",
        )
    veto = set(request.particle_veto)
    vetoed = sorted(
        abs(int(pdg))
        for pdg, orders in species_orders.items()
        if abs(int(pdg)) in veto and abelian in set(orders)
    )
    if vetoed:
        raise refuse(
            f"particle_veto removes the species {vetoed}, perturbative in the "
            f"{abelian} correction, from the conversion's one counterterm "
            "parameter",
            f"Instead {untag}, or drop the veto",
        )
    return ExternalPhotons(photons, inverse.name, conversion, abelian)


def _is_inverse_of(coupling: tuple[str, Any], name: str) -> bool:
    """Whether the loops' alpha (``poles.qed_coupling``: canonical text and
    its (symbol, parameter) pairs) is exactly 1/``name``."""

    from symbolica import E

    symbols = {parameter: symbol for symbol, parameter in coupling[1]}
    symbol = symbols.get(name)
    if symbol is None or len(symbols) != 1:
        return False
    product = (E(coupling[0]) * E(symbol)).expand()
    return product.to_canonical_string() == E("1").to_canonical_string()


def _option_text(option: Any) -> str:
    return f'"{option}"' if isinstance(option, str) else str(list(option))


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
        "resonant particles need the complex-mass scheme ([loop.model.scheme] "
        'mass_scheme = "complex-mass")'
    )


# The regularisation parameter of MG5_aMC's NLOCT loop models (``lhv``):
# lambda_HV, recognised by its declared TeX name; the model's R2 rules depend
# on it (lambda_HV = 1: 't Hooft-Veltman, 0: four-dimensional helicity).
HV_TEXNAME = r"\lambda_{HV}"


def gauge_ghost_partners(compiled: Any) -> dict[int, frozenset[int]]:
    """|PDG| of every Faddeev-Popov ghost -> the |PDG| codes of the vector
    bosons it can be the ghost of: the vectors (spin 3, no ghost number) of
    the ghost's colour, |charge| and mass parameter, the ghost's mass being
    its boson's in the 't Hooft-Feynman gauge of electroweak loops (MG5_aMC's
    loop models: ghWp, ghWm with MW, ghZ with MZ, ghA massless and colourless,
    ghG massless with colour 8). A structural reading, never names."""

    vectors: dict[tuple[object, ...], set[int]] = defaultdict(set)
    ghosts: dict[int, tuple[object, ...]] = {}
    for record in compiled.model["particles"]:
        spin = int(record["spin"])
        number = int(record.get("ghost_number", 0) or 0)
        key = (
            int(record["color"]),
            abs(Fraction(str(record.get("charge", 0) or 0))),
            str(record["mass"]),
        )
        if spin == 3 and not number:
            vectors[key].add(abs(int(record["pdg_code"])))
        elif number:
            ghosts[abs(int(record["pdg_code"]))] = key
    return {
        ghost: frozenset(vectors[key]) for ghost, key in ghosts.items() if vectors[key]
    }


def gauge_closed_veto(compiled: Any, veto: Iterable[int]) -> tuple[int, ...]:
    """``veto`` (|PDG| codes, ``particle_veto``) with the Faddeev-Popov
    ghosts of its vector bosons (``gauge_ghost_partners``; a ghost that could
    belong to several vectors only when all of them are vetoed). A ghost
    compensates the unphysical polarisations of its boson in the loops where
    that boson runs; with the boson vetoed its ghost loops are left alone and
    break the gauge cancellations: vetoing the W keeps W ghost loops in the
    photon self-energy, ghWp ghWp~ A, which MadLoop's photon-only control
    excludes (``mixed-uudd-bare-control.json``: its QED^6 bare sector of
    u u~ > d d~ misses by 1.2 % of the single pole with them)."""

    vetoed = {abs(int(pdg)) for pdg in veto}
    added = {
        ghost
        for ghost, partners in gauge_ghost_partners(compiled).items()
        if partners <= vetoed
    }
    return tuple(sorted(vetoed | added))


# SLHA codes of the electroweak inputs the loops' coupling may depend on.
SMINPUTS_ALPHA_INVERSE = ("SMINPUTS", (1,))
SMINPUTS_FERMI = ("SMINPUTS", (2,))


def scheme_conventions(slice_: Any) -> dict[str, object] | None:
    """The conventions an electroweak request's virtual is defined in (WP3.2
    EWI-7; requirements EW-14 and EW-18), None for a request that is not
    electroweak (``electroweak``). v2 makes no private gamma-five or scheme
    choice: four-dimensional numerators with d-dimensional denominators plus
    the model's R2 and UV rules define the result, so the record states the
    model's conventions as the generation found them:

    * ``regularisation`` "HV" ('t Hooft-Veltman: external states and the
      reduced numerators four-dimensional, the rational terms the model's
      R2), with ``lambda_HV`` the value of the model's HV parameter (the one
      declared with TeX name ``HV_TEXNAME``, MG5_aMC's ``lhv``; None if the
      model declares none);
    * ``gamma5``: the model's R2 rules at their own reading point (NLOCT,
      GMP-type), with four-dimensional Dirac traces in the loops;
      "generation-complete": a ``particle_veto`` that breaks the
      cancellation of the Levi-Civita counterterm cells is refused unless
      overridden ([U2], ``ct_recursion``), every other cell is complete;
    * ``gauge``: the gauge of Born, loops and counterterms (``LoopSlice.
      compiled``; 't Hooft-Feynman for every electroweak request, DESIGN
      §6.8);
    * ``uv_scheme``: the model's UV counterterm vertices and external-field
      factors, inserted once (REQUIREMENTS_ADDENDUM J);
    * ``mass_scheme`` "real-on-shell" (no width in any propagator,
      ``require_zero_widths``) or "complex-mass" (MG5_aMC's complex-mass
      model, ``loops.model_cms``), the latter with ``complex_mass``: its
      pole masses, widths, Yukawa ties, reglog convention and the model's
      CMSParam, LogSwitch and GfRedefinitionChoice
      (``model_cms.convention_record``); a complex-mass request that is not
      electroweak (QCD corrections with complex pole masses) records these
      two fields only;
    * ``ew_input_scheme``: "G_mu" when the loops' electromagnetic coupling
      (``poles.qed_coupling``) depends on the Fermi constant (SLHA SMINPUTS
      2), "alpha" when it depends on an input 1/alpha (SMINPUTS 1: alpha(M_Z)
      in MG5_aMC's ``loop_qcd_qed_sm``), else "model"; ``alpha_inputs`` the
      external parameters it depends on, with their SLHA codes;
    * ``external_photons``: the coupling of the on-shell external photons
      (MS-PHOTON, per leg: ``photon_tags``): "alpha(0)" when every one is
      tagged (``external_photon_scheme``, [U9]), "loop-coupling" when none
      is, "mixed" otherwise, "none" without such photons; with photons the
      1-based ``tagged_photon_legs`` (alpha(0) and the conversion term) and
      ``untagged_photon_legs`` (the loops' coupling, gamma_gamma in the IR
      poles) and the request's ``external_photons_option``;
    * ``pole_split``: how the "uv" and "ir" selections split the
      counterterms' poles by origin (``poles.split_sector``): "regulators"
      with the model's pole regulators ``pole_regulators`` (UV and IR names,
      ``LoopSlice.pole_regulators``: the counterterms' UV parts at the IR
      regulator 0, bare + counterterm UV = 0 a check), else "remainder"
      (the field factors' UV part is what makes the renormalised virtual UV
      finite: a model without them, or with one in ``fixed_parameters``).
    """

    from pyamplicol.loops.model_cms import COMPLEX_MASS, convention_record

    from .poles import qed_coupling

    compiled, request = slice_.compiled, slice_.request
    complex_mass = convention_record(compiled)
    mass_scheme = (
        {}
        if complex_mass is None
        else {
            "mass_scheme": COMPLEX_MASS,
            "complex_mass": complex_mass,
        }
    )
    species = slice_.prepared.structure.perturbative_species_orders
    if not electroweak(compiled, request, species):
        return mass_scheme or None
    fixed = request.fixed_parameters
    parameters = {record.name: record for record in compiled.ir.parameters}
    lambda_hv = None
    raw = compiled.raw_loop_model
    if raw is not None:
        for record in raw.collections["parameters"]:
            fields = record["fields"]
            if str(fields.get("texname", "")).strip() == HV_TEXNAME:
                value = fields.get("value")
                if isinstance(value, Mapping):
                    value = value.get("expression")
                lambda_hv = None if value is None else str(value)
                break
    coupling = qed_coupling(compiled, lambda r: is_photon(r, parameters, fixed))
    inputs = sorted({name for _symbol, name in coupling[1]}) if coupling else []

    def code(name: str) -> tuple[str, tuple[int, ...]]:
        record = parameters[name]
        return (record.lhablock or "").upper(), tuple(record.lhacode)

    codes = {name: code(name) for name in inputs}
    scheme = (
        "G_mu"
        if SMINPUTS_FERMI in codes.values()
        else "alpha"
        if SMINPUTS_ALPHA_INVERSE in codes.values()
        else "model"
    )
    return {
        "regularisation": "HV",
        "lambda_HV": lambda_hv,
        "gamma5": (
            "the model's R2 rules at their reading point (NLOCT, GMP-type), "
            "four-dimensional Dirac traces in the loops; generation-complete"
        ),
        "gauge": str(compiled.ir.gauge),
        "uv_scheme": ("the model's UV counterterm vertices and external-field factors"),
        "mass_scheme": "real-on-shell",
        **mass_scheme,
        "ew_input_scheme": scheme,
        "alpha_inputs": {
            name: f"{block} {' '.join(map(str, lha))}".strip()
            for name, (block, lha) in codes.items()
        },
        **_photon_conventions(slice_),
        **_pole_split(slice_),
    }


def _pole_split(slice_: Any) -> dict[str, object]:
    """``scheme_conventions``' pole-split entries (MS-POLES)."""

    regulators = slice_.pole_regulators
    if regulators is None:
        return {"pole_split": "remainder"}
    return {"pole_split": "regulators", "pole_regulators": regulators.record()}


def _photon_conventions(slice_: Any) -> dict[str, object]:
    """``scheme_conventions``' external-photon entries (MS-PHOTON): without
    on-shell external photons only ``external_photons`` "none" (the records
    of such requests keep their bytes)."""

    tagged, untagged = slice_photon_tags(slice_)
    if not tagged and not untagged:
        return {"external_photons": "none"}
    return {
        "external_photons": (
            "alpha(0)" if not untagged else "loop-coupling" if not tagged else "mixed"
        ),
        "tagged_photon_legs": list(tagged),
        "untagged_photon_legs": list(untagged),
        "external_photons_option": (
            option
            if isinstance(option := slice_.request.external_photons, str)
            else list(option)
        ),
    }


__all__ = [
    "ADJOINT",
    "ALPHA0_CONVERSION",
    "ALPHA0_INVERSE_CODE",
    "HV_TEXNAME",
    "PHOTON_TAG_CHOICES",
    "PHOTON_TAG_DEFAULT",
    "CorrectionOrders",
    "ElectroweakGaugeError",
    "ElectroweakWidthError",
    "ExternalPhotonError",
    "ExternalPhotons",
    "correction_orders",
    "electroweak",
    "external_photon_scheme",
    "external_photons",
    "feynman_gauge_model",
    "gauge_closed_veto",
    "gauge_ghost_partners",
    "is_photon",
    "is_zero",
    "photon_species",
    "photon_tag_option",
    "photon_tags",
    "propagated_species",
    "propagating_widths",
    "require_feynman_gauge",
    "require_zero_widths",
    "scheme_conventions",
    "slice_photon_tags",
]
