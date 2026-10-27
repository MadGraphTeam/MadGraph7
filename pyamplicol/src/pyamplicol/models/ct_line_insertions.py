# SPDX-License-Identifier: 0BSD
"""Two-point counterterm insertions as clone lines of the tree model (design §6.1).

A two-point counterterm (CT) ``Pi`` on an internal line of a tree replaces the
line's propagator ``P`` by ``P Pi P``, or ``P_Y Pi P_X`` when it mixes two
fields. In a Berends-Giele recursion every internal line is one current, so
the insertion is a unary step on a current, which the tree engine expresses
without a new node kind:

* **Line classes.** A two-point vertex ``Pi(X, Y)`` (UFO legs, both fields
  incoming) sits on a line whose current enters it at its leg of ``X`` and
  leaves it at its leg of ``Y``: the current is produced at a vertex with a
  leg of the antiparticle of ``X`` and read at a vertex with a leg of the
  antiparticle of ``Y`` (``_orientation`` fixes which of the two legs is
  ``X``; a diagonal term has ``Y`` the antiparticle of ``X``). Each class
  (``X``, ``Y``, physical orders of ``Pi``, slot) gets a clone pair: ``C``,
  read as the antiparticle of ``Y`` (a copy of its record), and its
  antiparticle ``C~``, read as the antiparticle of ``X``. A self-conjugate
  field's diagonal class has one self-conjugate clone.
* **Propagators.** A clone's propagator is model-supplied: ``N_C(1, 2) =
  N_out(1, B) Pi(in -> A, out -> B) N_in(A, 2) / (D_in D_out)``, with the
  propagators of the field entering (``X`` for ``C``) and leaving (the
  antiparticle of ``Y``) the two-point vertex; ``C~`` is the reversed line.
  ``Pi`` is the sum of the class's CT terms, each its exact derived coupling
  (``S_slot * sum M_rule * c_rule``) times the number of its colour
  (``c * delta``, ``ct_color_flow.two_point_colour_factor``) times its
  Lorentz structure; the denominators ``D`` are those the engine applies,
  including the width of its standard propagators.
* **Weyl lines.** The tree engine propagates massless fermions as Weyl
  currents with its own propagator, which a model-supplied propagator would
  turn into Dirac currents that no Weyl source closes. When every field of a
  class is a Weyl field and ``Pi`` is a multiple of the slashed momentum
  (``P_Y Pi P_X = kappa P``, ``_line_factor``), the clones keep the
  standard propagator and ``kappa`` multiplies the couplings of the copies
  on the leg of ``C~``; a clone line joins a leg of ``C`` and one of
  ``C~``, so it carries ``kappa`` once. Otherwise (a chiral or
  chirality-flipping ``Pi``, or a Weyl field mixing with a Dirac field) the
  class's Weyl fields are propagated as Dirac fields of the lowered model
  (``dirac_fields``: the caller makes their standard propagators
  model-supplied, so that their sources, currents and clones are Dirac
  spinors), and the clones are model-supplied.
* **Vertex copies.** Every ordinary vertex with a leg of the field a clone
  is read as is copied once per clone, with that leg on the clone; the
  copy's couplings carry the original orders, half of the orders of ``Pi``,
  the insertion order once and the slot order once.

A clone current is produced by one vertex copy and consumed by another, so a
line with an insertion carries exactly two copies (the orders of ``Pi``, the
insertion order twice and the slot order twice; a CT vertex carries the
latter two twice itself, so one insertion of either kind has the same
pseudo-orders). Every internal line is covered exactly once, and external
legs never are: the clone is no external particle, and the root closure pairs
a current only with the sink of its own particle.

Index conventions of the composition, with the UFO propagator ``N(1, 2)``
(leg 2 joins the vertex that produces the current, leg 1 the one that
consumes it; the momentum ``p`` flows along the current) and the two-point
vertex with legs ``r`` (the field the current enters, incoming momentum
``p``) and ``o`` (incoming momentum ``-p``)::

    N_clone(1, 2) = N_out(1, B) Pi(r -> A, o -> B) N_in(A, 2) / (D_in D_out)

* **Scalar-vector mixing** (a Goldstone and its vector in 't Hooft-Feynman
  gauge, WP3.2 EWI-4b: G0-Z, G0-A, G+-W-) is a **vertex insertion**. By
  Lorentz covariance ``Pi^mu = kappa p^mu`` with ``p`` the line momentum, so
  the momentum moves into the vertex copies on the scalar side: their clone
  leg becomes a vector leg whose Lorentz structure is the original one times
  ``P(mu_k, k)``, the tree momentum of that leg (minus the momentum flowing
  into it). ``Pi`` is read as the tree engine reads every vertex, on the
  tree momenta of its legs: its scalar leg takes ``-P(k)``, its vector leg
  ``+P(k)``, whichever way the current flows (``_momentum_ratio``). Both
  clones are then vector lines, the clone of the scalar side a
  copy of the vector's antiparticle, with the propagator ``kappa N_S N_V(1,
  2) / (D_S D_V)`` (``N_S`` and ``D_S`` the scalar's, ``N_V`` and ``D_V``
  the vector's): the current a vector-side vertex consumes is the vector's
  own (``P_V Pi P_S J``), the one a scalar-side copy consumes is contracted
  with its momentum there. A clone of the scalar side is therefore not a
  current of the scalar it is read as (``reads_as``): it can be consumed by
  a vertex copy only, never by an ordinary vertex (the counterterm closure
  refuses such a root).
* **Gauge.** The propagators composed into a clone line are those the tree
  engine applies: a vector with a standard propagator in a model compiled
  in 't Hooft-Feynman gauge (``gauge``) propagates with ``-i g`` whatever its
  record says (the UFO record of a massive vector is the unitary one). A
  term on a Goldstone the tree model absorbs (``absorbed``: unitary gauge)
  is refused: no tree vertex holds the Goldstone, so its counterterm
  program has no line to insert it on (the caller records such a rule as
  inert, REQUIREMENTS_ADDENDUM J), and a clone would bring its vertices back.

Refused (``refused``, the caller keeps such rules as site closures): mixing
between fields of non-conjugate charges or of spins other than equal ones
or a scalar and a vector, terms on absorbed Goldstones, mixing structures
that are not a multiple of the line momentum, orders of ``Pi`` that are not
even, spins other than 0, 1/2 and 1, fields without a propagator record,
Lorentz structures with momentum heads other than ``P``, and vertices with
two legs of the same fermion species of a clone.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any

from .ct_color_flow import UnloweredCountertermColour, two_point_colour_factor

_SPINS = (1, 2, 3)
_DUMMY_OFFSET = (1000, 2000, 3000)  # dummies of the two propagators and Pi
_INNER = (9001, 9002)  # the indices A and B of the composition


@dataclass
class LineInsertions:
    """Payload records of the clone lines of a set of two-point CT terms.

    ``clones`` maps every lowered term to its clone particle ``C``,
    ``members`` every vertex copy to the terms of its clone line, ``refused``
    every term that cannot be lowered to the reason and ``reads_as`` every
    clone particle to the field whose current it is read as (``C`` the
    antiparticle of ``Y``, ``C~`` that of ``X``). ``factors`` maps the clone
    ``C~`` of every Weyl line, whose copies carry ``kappa``
    (``_line_factor``), to ``kappa`` in the payload's syntax.
    ``dirac_fields`` are the Weyl fields (both charge states) the lowered
    model must propagate as Dirac fields: the caller makes their standard
    propagators model-supplied. ``lorentz_structures`` are the Lorentz
    records of the scalar-side copies of mixing lines (the original
    structure times the leg's momentum, module docstring).
    """

    parameters: list[dict[str, Any]] = field(default_factory=list)
    particles: list[dict[str, Any]] = field(default_factory=list)
    propagators: list[dict[str, Any]] = field(default_factory=list)
    couplings: list[dict[str, Any]] = field(default_factory=list)
    vertex_rules: list[dict[str, Any]] = field(default_factory=list)
    clones: dict[Any, str] = field(default_factory=dict)
    members: dict[str, tuple[Any, ...]] = field(default_factory=dict)
    refused: dict[Any, str] = field(default_factory=dict)
    factors: dict[str, str] = field(default_factory=dict)
    reads_as: dict[str, str] = field(default_factory=dict)
    dirac_fields: frozenset[str] = frozenset()
    lorentz_structures: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class _Clone:
    """One clone particle of a line class.

    ``name`` reads as ``field`` (whose record it copies) and is the
    antiparticle of ``partner``; its current enters the two-point vertex at
    the leg of ``into`` and leaves it towards a leg of ``field``.
    """

    name: str
    field: str
    partner: str
    into: str
    sign: int


def lower_line_insertions(
    payload: Mapping[str, Any],
    insertions: Mapping[Any, str],
    *,
    prefix: str,
    insertion_order: str,
    slot_order: Callable[[Any], str],
    weyl_fields: frozenset[str] = frozenset(),
    dirac_fields: frozenset[str] = frozenset(),
    gauge: str = "unitary",
    absorbed: frozenset[str] = frozenset(),
) -> LineInsertions:
    """Clone lines of the two-point CT terms ``insertions`` (term -> coupling).

    ``payload`` is the compiled model payload whose ordinary vertices are
    copied (it is not modified); ``prefix`` prefixes every new record name,
    ``insertion_order`` and ``slot_order(term)`` name the pseudo-orders.
    ``weyl_fields`` are the fermions whose currents the tree engine keeps as
    Weyl currents (massless, standard propagator); a class of Weyl fields
    whose ``Pi`` is a multiple of the slashed momentum keeps the standard
    propagator and carries ``P_Y Pi P_X = kappa P`` as the factor ``kappa``
    (``_line_factor``). ``dirac_fields`` are Weyl fields to propagate as
    Dirac fields regardless (with the fields the classes require, in
    ``LineInsertions.dirac_fields``). ``gauge`` is the compiled model's
    gauge and ``absorbed`` the Goldstones its tree model absorbs (both
    charge states; module docstring).
    """

    from symbolica import E

    particles = {str(item["name"]): item for item in payload["particles"]}
    propagators = {str(item["name"]): item for item in payload["propagators"]}
    lorentz = {str(item["name"]): item for item in payload["lorentz_structures"]}
    out = LineInsertions()

    def numerator(name: str) -> str:
        return _engine_numerator(
            propagators[str(particles[name]["propagator"])], particles[name], gauge
        )

    classes: dict[tuple[object, ...], list[Any]] = defaultdict(list)
    for term in sorted(insertions):
        reason = _refusal(term, particles, propagators, lorentz, absorbed)
        if reason is not None:
            out.refused[term] = reason
            continue
        into, out_of = _orientation(term, particles)
        classes[
            into, out_of, term.orders, term.component, term.power, term.grade
        ].append(term)
    used = {abs(int(item["pdg_code"])) for item in payload["particles"]}
    pdg = max(9_900_000, max(used, default=0) + 1)
    couplings = {str(item["name"]): item for item in payload["couplings"]}
    copyable = [
        rule
        for rule in payload["vertex_rules"]
        if not str(rule["name"]).startswith(prefix)
        and not any(
            int(particles[str(name)].get("ghost_number", 0))
            for name in rule["particles"]
        )
    ]
    ordered = sorted(classes.items())
    lines = {
        key: _clones(f"{prefix}_LINE{number}", key[0], key[1], particles)
        for number, (key, _terms) in enumerate(ordered)
    }

    def conjugates(names):
        return {*names, *(str(particles[name]["antiname"]) for name in names)}

    # Weyl classes need a multiple of the slashed momentum on Weyl fields
    # only; their other Weyl fields are propagated as Dirac fields, which
    # can turn further classes into Dirac classes.
    forced = conjugates(dirac_fields) & set(particles)
    factors: dict[tuple[object, ...], Any] = {}
    while True:
        weyl = set(weyl_fields) - forced
        grown = set(forced)
        factors.clear()
        for key, terms in ordered:
            names = {clone.field for clone in lines[key]}
            names |= {clone.into for clone in lines[key]}
            names = conjugates(names)
            if not names & weyl:
                continue
            factor = None
            if names <= weyl:
                factor = _line_factor(
                    str(payload["name"]),
                    propagators,
                    particles,
                    lines[key],
                    [(insertions[term], term) for term in terms],
                    lorentz,
                )
            if factor is None or isinstance(factor, str):
                grown |= names & weyl
            else:
                factors[key] = factor
        if grown == forced:
            break
        forced = grown
    out.dirac_fields = frozenset(forced)
    for key, terms in ordered:
        orders = key[2]
        clones = lines[key]
        species = {clone.field: clone.name for clone in clones}
        reason = _copy_refusal(copyable, species, particles)
        if reason is not None:
            out.refused.update(dict.fromkeys(terms, reason))
            continue
        if _mixing(clones, particles):
            reason = _lower_mixing(
                out,
                clones,
                [(insertions[term], term) for term in terms],
                pdg,
                particles,
                propagators,
                lorentz,
                copyable,
                couplings,
                numerator,
                {
                    **{name: power // 2 for name, power in orders},
                    insertion_order: 1,
                    slot_order(terms[0]): 1,
                },
            )
            if reason is not None:
                out.refused.update(dict.fromkeys(terms, reason))
                continue
            out.clones.update(dict.fromkeys(terms, clones[0].name))
            pdg += 1
            continue
        factor = factors.get(key)
        scales: list[str] = []
        if factor is None:
            # The scales (coupling times colour, the same in both directions)
            # are derived model parameters, so that the propagator kernel
            # stays free of constants such as pi, which compiled kernels
            # round to binary64.
            for index, (value, _structure) in enumerate(
                _insertion_terms(
                    [(insertions[term], term) for term in terms],
                    lorentz,
                    particles,
                    clones[0].into,
                )
            ):
                scales.append(f"{clones[0].name}_X{index}")
                out.parameters.append(
                    {
                        "expression": value.to_canonical_string(),
                        "lhablock": None,
                        "lhacode": None,
                        "name": scales[-1],
                        "nature": "internal",
                        "parameter_type": "complex",
                        "value": None,
                    }
                )
        for clone in clones:
            # A clone propagates only through its own propagator record (P Pi
            # P, or a Weyl field's massless one): its mass and width are not
            # those of a particle of the model (a massive vector's clone would
            # otherwise be a second Goldstone partner).
            record = _clone_record(particles[clone.field], clone, pdg)
            out.particles.append(record)
            out.reads_as[clone.name] = clone.field
            standard = propagators[str(particles[clone.field]["propagator"])]
            line = {**standard, "name": f"{clone.name}_prop"}
            if factor is None:
                entering = propagators[str(particles[clone.into]["propagator"])]
                line.update(
                    numerator=_line_numerator(
                        numerator(clone.into),
                        numerator(clone.field),
                        [
                            (E(f"UFO::{{}}::{scale}"), structure)
                            for scale, (_value, structure) in zip(
                                scales,
                                _insertion_terms(
                                    [(insertions[term], term) for term in terms],
                                    lorentz,
                                    particles,
                                    clone.into,
                                ),
                                strict=True,
                            )
                        ],
                    ),
                    denominator=(
                        f"({_engine_denominator(entering, particles[clone.into])})"
                        f"*({_engine_denominator(standard, particles[clone.field])})"
                    ),
                    pyamplicol_source="model-supplied",
                )
            out.propagators.append({**line, "particle": clone.name})
        pdg += 1
        extra = {
            **{name: power // 2 for name, power in orders},
            insertion_order: 1,
            slot_order(terms[0]): 1,
        }
        out.clones.update(dict.fromkeys(terms, clones[0].name))
        # A clone line joins a leg of C and a leg of C~, so the factor of a
        # Weyl line sits on the copies of the latter.
        scaled = set()
        if factor is not None:
            scaled = {clones[-1].field}
            out.factors[clones[-1].name] = factor.to_canonical_string()
        copied: dict[tuple[str, bool], str] = {}
        prefix_name = clones[0].name
        for rule in copyable:
            legs = [str(name) for name in rule["particles"]]
            for name, clone_name in species.items():
                if name not in legs:
                    continue
                position = legs.index(name)
                scale = name in scaled
                rows = []
                for row in rule["couplings"]:
                    for coupling in row:
                        copy_key = (coupling, scale)
                        if coupling is not None and copy_key not in copied:
                            copied[copy_key] = f"{prefix_name}_C{len(copied)}"
                            original = couplings[coupling]
                            coupling_record = {
                                **original,
                                "name": copied[copy_key],
                                "orders": _added_orders(original["orders"], extra),
                            }
                            if scale:
                                coupling_record.update(
                                    expression=_scaled(original["expression"], factor),
                                    value=[1.0, 0.0],
                                )
                            out.couplings.append(coupling_record)
                    rows.append([copied.get((coupling, scale)) for coupling in row])
                copy = f"{clone_name}__{rule['name']}"
                out.vertex_rules.append(
                    {
                        **rule,
                        "name": copy,
                        "particles": [
                            *legs[:position],
                            clone_name,
                            *legs[position + 1 :],
                        ],
                        "couplings": rows,
                    }
                )
                out.members[copy] = tuple(terms)
    return out


def _orientation(
    term: Any, particles: Mapping[str, Mapping[str, Any]]
) -> tuple[str, str]:
    """``(X, Y)``: the leg a class's current enters and the one it leaves.

    ``C`` is read as the antiparticle of ``Y``; the orientation is the one
    whose ``C`` has a positive PDG code (the field, not the antifield, of a
    diagonal fermion term), then the larger one, so that ``Pi(A, Z)`` and
    ``Pi(Z, A)`` share a class.
    """

    first, second = (str(name) for name in term.particles)

    def key(pair: tuple[str, str]) -> tuple[bool, int, str]:
        read = particles[str(particles[pair[1]]["antiname"])]
        code = int(read["pdg_code"])
        return code > 0, abs(code), str(read["name"])

    return max(((first, second), (second, first)), key=key)


def _clones(
    name: str, into: str, out_of: str, particles: Mapping[str, Mapping[str, Any]]
) -> tuple[_Clone, ...]:
    """``(C, C~)`` of the class whose current enters at ``into`` and leaves at
    ``out_of`` (one self-conjugate clone for a self-conjugate diagonal class)."""

    read = str(particles[out_of]["antiname"])
    other = str(particles[into]["antiname"])
    if read == other:
        return (_Clone(name, read, name, into, 1),)
    anti = f"{name}~"
    return (
        _Clone(name, read, anti, into, 1),
        _Clone(anti, other, name, out_of, -1),
    )


def _added_orders(
    orders: Sequence[Sequence[Any]], extra: Mapping[str, int]
) -> list[list[Any]]:
    total = dict(extra)
    for name, power in orders:
        total[str(name)] = total.get(str(name), 0) + int(power)
    return [[name, power] for name, power in sorted(total.items())]


def _copy_refusal(
    rules: Sequence[Mapping[str, Any]],
    species: Mapping[str, str],
    particles: Mapping[str, Mapping[str, Any]],
) -> str | None:
    """Why the vertices of a field cannot be copied onto its clone line.

    A vertex with two legs of one fermion species is antisymmetric in them;
    which leg goes onto the clone would change its sign.
    """

    for rule in rules:
        legs = [str(name) for name in rule["particles"]]
        for name in species:
            if legs.count(name) > 1 and int(particles[name]["spin"]) % 2 == 0:
                return f"vertex {rule['name']} has two legs of the fermion {name}"
    return None


def _refusal(
    term: Any,
    particles: Mapping[str, Mapping[str, Any]],
    propagators: Mapping[str, Mapping[str, Any]],
    lorentz: Mapping[str, Mapping[str, Any]],
    absorbed: frozenset[str] = frozenset(),
) -> str | None:
    first, second = (str(name) for name in term.particles)
    records = (particles[first], particles[second])
    for record in records:
        if int(record["spin"]) not in _SPINS:
            return f"two-point insertion on a spin-{record['spin']} line"
    for name in (first, second):
        if name in absorbed:
            return (
                f"two-point insertion on the Goldstone {name}, which the "
                "unitary-gauge tree model absorbs"
            )
    if str(records[0]["antiname"]) != second:
        conjugate = particles[str(records[1]["antiname"])]
        spins = {int(records[0]["spin"]), int(records[1]["spin"])}
        if len(spins) > 1 and spins != {1, 3}:
            return f"two-point mixing insertion {first}-{second} between spins"
        if _quantum_numbers(records[0]) != _quantum_numbers(conjugate):
            return (
                f"two-point mixing insertion {first}-{second} between "
                "non-conjugate charges"
            )
    if any(power % 2 for _name, power in term.orders):
        return f"two-point insertion with odd orders {term.orders}"
    for name in term.particles:
        if particles[name].get("propagator") not in propagators:
            return f"two-point insertion on {name}, which has no propagator record"
    if "PSlash" in str(lorentz[term.lorentz]["structure"]):
        return f"two-point Lorentz structure {term.lorentz} with PSlash"
    try:
        two_point_colour_factor(
            term.colour, [int(particles[name]["color"]) for name in term.particles]
        )
    except UnloweredCountertermColour as error:
        return str(error)
    return None


def _clone_record(source: Mapping[str, Any], clone: _Clone, pdg: int) -> dict[str, Any]:
    """The particle record of ``clone``, a copy of ``source``.

    A clone propagates only through its own propagator record (P Pi P, or
    a Weyl field's massless one): its mass and width are not those of a
    particle of the model (a massive vector's clone would otherwise be a
    second Goldstone partner), and it is no Goldstone (a Goldstone's clone
    has no vector partner).
    """

    record = dict(source)
    record.update(
        name=clone.name,
        antiname=clone.partner,
        pdg_code=clone.sign * pdg,
        propagator=f"{clone.name}_prop",
        mass="ZERO",
        width="ZERO",
    )
    if "goldstoneboson" in record:
        record["goldstoneboson"] = False
    return record


def _engine_numerator(
    propagator: Mapping[str, Any], particle: Mapping[str, Any], gauge: str
) -> str:
    """The numerator the tree engine applies for ``propagator``: the record's,
    except for a vector with a standard propagator in 't Hooft-Feynman
    gauge, which propagates with ``-i g`` (``models.external_evaluation``;
    the record of a massive vector holds the unitary numerator)."""

    if (
        gauge == "feynman"
        and int(particle["spin"]) == 3
        and propagator.get("pyamplicol_source") == "default-feynman"
    ):
        return _FEYNMAN_VECTOR_NUMERATOR
    return str(propagator["numerator"])


# The 't Hooft-Feynman vector numerator -i g(1, 2) in the payload's syntax.
_FEYNMAN_VECTOR_NUMERATOR = "(-𝑖)*UFO::{}::Metric(UFO::{}::idx(1,1),UFO::{}::idx(1,2))"  # noqa: RUF001


def _mixing(
    clones: Sequence[_Clone], particles: Mapping[str, Mapping[str, Any]]
) -> bool:
    """Whether the line of ``clones`` mixes a scalar and a vector."""

    return len({int(particles[clone.field]["spin"]) for clone in clones}) > 1


def _lower_mixing(
    out: LineInsertions,
    clones: Sequence[_Clone],
    terms: Sequence[tuple[str, Any]],
    pdg: int,
    particles: Mapping[str, Mapping[str, Any]],
    propagators: Mapping[str, Mapping[str, Any]],
    lorentz: Mapping[str, Mapping[str, Any]],
    rules: Sequence[Mapping[str, Any]],
    couplings: Mapping[str, Mapping[str, Any]],
    numerator: Callable[[str], str],
    extra: Mapping[str, int],
) -> str | None:
    """Lower a scalar-vector mixing line as a vertex insertion into ``out``
    (module docstring), or return why it cannot be.

    ``clones`` are the line's two clones, ``terms`` its ``(coupling, term)``
    pairs, ``pdg`` the clones' PDG code, ``rules`` the copyable vertex rules
    and ``couplings`` the payload's couplings by name, ``numerator(field)``
    the propagator numerator the engine applies and ``extra`` the orders the
    copies add.
    """

    from symbolica import E

    scalar_side = next(c for c in clones if int(particles[c.field]["spin"]) == 1)
    vector_side = next(c for c in clones if c is not scalar_side)
    scalar, vector = scalar_side.field, vector_side.field
    kappa = None
    for coupling, term in terms:
        ratio = _momentum_ratio(term, lorentz, particles)
        if ratio is None:
            return (
                f"two-point mixing structure {term.lorentz} of {term.particles} "
                "is not a multiple of the line momentum"
            )
        colour = two_point_colour_factor(
            term.colour, [int(particles[name]["color"]) for name in term.particles]
        )
        value = E(coupling) * ratio * E(f"{colour.numerator}/{colour.denominator}")
        kappa = value if kappa is None else kappa + value
    assert kappa is not None
    scale = f"{clones[0].name}_X0"
    out.parameters.append(
        {
            "expression": kappa.expand().to_canonical_string(),
            "lhablock": None,
            "lhacode": None,
            "name": scale,
            "nature": "internal",
            "parameter_type": "complex",
            "value": None,
        }
    )
    # Both clones are vector lines; the scalar side's is a copy of the
    # vector's antiparticle (the charges of the scalar it is read as).
    vector_record = particles[vector]
    composite = (
        (E(f"UFO::{{}}::{scale}") * E(numerator(scalar)) * E(numerator(vector)))
        .expand()
        .to_canonical_string()
    )
    denominator = "*".join(
        f"({_engine_denominator(propagators[str(record['propagator'])], record)})"
        for record in (particles[scalar], vector_record)
    )
    for clone in clones:
        source = (
            vector_record
            if clone is vector_side
            else particles[str(vector_record["antiname"])]
        )
        out.particles.append(_clone_record(source, clone, pdg))
        out.reads_as[clone.name] = clone.field
        standard = propagators[str(vector_record["propagator"])]
        out.propagators.append(
            {
                **standard,
                "name": f"{clone.name}_prop",
                "particle": clone.name,
                "numerator": composite,
                "denominator": denominator,
                "pyamplicol_source": "model-supplied",
            }
        )
    copied: dict[str, str] = {}
    known = {str(name) for name in lorentz}
    prefix_name = clones[0].name
    for rule in rules:
        legs = [str(name) for name in rule["particles"]]
        for clone in clones:
            if clone.field not in legs:
                continue
            position = legs.index(clone.field)
            rows = []
            for row in rule["couplings"]:
                for coupling in row:
                    if coupling is not None and coupling not in copied:
                        copied[coupling] = f"{prefix_name}_C{len(copied)}"
                        original = couplings[coupling]
                        out.couplings.append(
                            {
                                **original,
                                "name": copied[coupling],
                                "orders": _added_orders(original["orders"], extra),
                            }
                        )
                rows.append([copied.get(coupling) for coupling in row])
            structures = list(rule["lorentz_structures"])
            if clone is scalar_side:
                structures = [
                    _promoted_lorentz(out, lorentz, known, name, position)
                    for name in structures
                ]
            copy = f"{clone.name}__{rule['name']}"
            out.vertex_rules.append(
                {
                    **rule,
                    "name": copy,
                    "particles": [*legs[:position], clone.name, *legs[position + 1 :]],
                    "lorentz_structures": structures,
                    "couplings": rows,
                }
            )
            out.members[copy] = tuple(term for _coupling, term in terms)
    return None


def _momentum_ratio(
    term: Any,
    lorentz: Mapping[str, Mapping[str, Any]],
    particles: Mapping[str, Mapping[str, Any]],
) -> Any:
    """The number ``r`` with ``Pi_t^mu = r P(mu, k)`` for a scalar-vector
    two-point term, ``P(k)`` the momentum the tree engine gives the leg ``k``
    of the scalar-side vertex copy that the line joins, or ``None`` if the
    structure is no such multiple.

    The tree engine evaluates every vertex on the tree momenta of its
    currents (the leg-mask momentum, initial legs negated: minus the
    momentum flowing into the vertex), so ``P(k)`` is minus the momentum
    flowing into leg ``k``, i.e. the momentum flowing into ``Pi``'s scalar
    leg. ``Pi`` is read the same way, on the tree momenta of its legs (minus
    the momenta flowing into them): its scalar leg takes ``-P(k)`` and its
    vector leg ``+P(k)``, whichever way the current flows. The reading with
    the momenta flowing in (``+P(k)`` on the scalar leg) gives every mixing
    the opposite sign (WP3.2 EWI-7b): MadLoop's Goldstone-vector mixing
    counterterms of g t > t t t~ (``tests/fixtures/loop/goldstone-mixing.json``)
    and the site oracle (O6, which reads the mixings as the tree engine does,
    ``ct_sites.SiteClosure._reading_sign``) decide it.
    """

    from symbolica import E

    legs = list(term.particles)
    spins = [int(particles[name]["spin"]) for name in legs]
    scalar, vector = 1 + spins.index(1), 1 + spins.index(3)
    momentum, index = _placeholder(), E("UFO::{}::dummy(9101)")
    structure = E(str(lorentz[term.lorentz]["structure"]))
    structure = structure.replace(
        E(f"UFO::{{}}::P(a_,UFO::{{}}::idx(1,{scalar}))"), -momentum(E("a_"))
    ).replace(E(f"UFO::{{}}::P(a_,UFO::{{}}::idx(1,{vector}))"), momentum(E("a_")))
    structure = structure.replace(E(f"UFO::{{}}::idx(1,{vector})"), index).expand()
    ratio = structure.replace(momentum(index), E("1"))
    remainder = (structure - ratio * momentum(index)).expand()
    if ratio.get_all_symbols(False) or remainder != E("0"):
        return None
    return ratio


def _promoted_lorentz(
    out: LineInsertions,
    lorentz: Mapping[str, Mapping[str, Any]],
    known: set[str],
    name: str,
    position: int,
) -> str:
    """The Lorentz structure ``name`` with its scalar leg ``position``
    (0-based) promoted to a vector leg carrying its incoming momentum,
    ``L * P(mu_k, k)``, added to ``out`` once."""

    record = lorentz[name]
    leg = position + 1
    promoted = f"{name}__P{leg}"
    if promoted not in known:
        spins = [int(spin) for spin in record["spins"]]
        if spins[position] != 1:
            raise ValueError(f"leg {leg} of {name} is not a scalar leg")
        spins[position] = 3
        out.lorentz_structures.append(
            {
                **record,
                "name": promoted,
                "spins": spins,
                "structure": (
                    f"({record['structure']})*UFO::{{}}::P("
                    f"UFO::{{}}::idx(1,{leg}),UFO::{{}}::idx(1,{leg}))"
                ),
            }
        )
        known.add(promoted)
    return promoted


def _quantum_numbers(record: Mapping[str, Any]) -> tuple[object, ...]:
    """The conserved numbers a clone copies from the field it is read as."""

    return (
        int(record["color"]),
        Fraction(str(record.get("charge", 0) or 0)),
        int(record.get("lepton_number", 0) or 0),
        int(record.get("ghost_number", 0) or 0),
    )


def _insertion_terms(
    terms: Sequence[tuple[str, Any]],
    lorentz: Mapping[str, Mapping[str, Any]],
    particles: Mapping[str, Mapping[str, Any]],
    current: str,
) -> list[tuple[Any, Any]]:
    """``(scale, Pi_t(r -> A, o -> B))`` of every term for a ``current`` line.

    The scale is the term's coupling times the number of its colour; momenta
    are the placeholder ``q`` of ``current`` (``P`` of the receiving leg ``r``
    is ``q``, of the other leg ``-q``).
    """

    from symbolica import E

    inner = [E(f"UFO::{{}}::dummy({index})") for index in _INNER]
    out = []
    for coupling, term in terms:
        legs = list(term.particles)
        receiving = len(legs) - legs[::-1].index(current)  # 1-based UFO leg
        other = 3 - receiving
        structure = _renamed_dummies(
            E(str(lorentz[term.lorentz]["structure"])), _DUMMY_OFFSET[2]
        )
        momentum = _placeholder()
        structure = structure.replace(
            E(f"UFO::{{}}::P(a_,UFO::{{}}::idx(1,{receiving}))"), momentum(E("a_"))
        ).replace(E(f"UFO::{{}}::P(a_,UFO::{{}}::idx(1,{other}))"), -momentum(E("a_")))
        if int(particles[legs[0]]["spin"]) == 2:
            # The propagator reading of a clone line takes a spinor chain of
            # Pi from its receiving end in reverse (``_reversed_chain``).
            structure = _reversed_chain(structure)
            receiving, other = other, receiving
        structure = structure.replace(
            E(f"UFO::{{}}::idx(1,{receiving})"), inner[0]
        ).replace(E(f"UFO::{{}}::idx(1,{other})"), inner[1])
        colour = two_point_colour_factor(
            term.colour, [int(particles[name]["color"]) for name in legs]
        )
        scale = E(coupling) * E(f"{colour.numerator}/{colour.denominator}")
        out.append((scale, structure))
    return out


# Spinor tensors of a UFO Lorentz structure and their spinor arguments
# (0-based positions of the row and the column index).
_SPINOR_TENSORS = (
    ("Gamma", 3, (1, 2)),
    ("Gamma5", 2, (0, 1)),
    ("ProjM", 2, (0, 1)),
    ("ProjP", 2, (0, 1)),
    ("Identity", 2, (0, 1)),
    ("PSlash", 2, (0, 1)),
    ("Sigma", 4, (2, 3)),
)


def _reversed_chain(structure: Any) -> Any:
    """``structure`` with every spinor tensor's row and column exchanged.

    With the two legs of a fermion two-point exchanged as well, a spinor
    chain ``X1 ... Xn`` from one leg to the other becomes ``Xn ... X1``
    (a single factor is unchanged). A clone line's propagator reads a chain
    of ``Pi`` in reverse (``_composed`` puts ``Pi`` between the propagators
    with its receiving leg on the side of the entering one), so ``Pi`` is
    composed reversed and read in order: ``P_Y Pi P_X`` with the chirality
    of every lone or chained projector (WP3.2 EWI-4b; the propagator reading
    used to exchange ``ProjM`` and ``ProjP``, right for chains and wrong for
    lone projectors).
    """

    from symbolica import E

    names = ("a_", "b_", "c_", "d_")
    for head, arity, (row, column) in _SPINOR_TENSORS:
        arguments = list(names[:arity])
        swapped = list(arguments)
        swapped[row], swapped[column] = arguments[column], arguments[row]
        structure = structure.replace(
            E(f"UFO::{{}}::{head}({','.join(arguments)})"),
            E(f"UFO::{{}}::{head}({','.join(swapped)})"),
        )
    return structure


def _line_numerator(
    entering: str, leaving: str, terms: Sequence[tuple[Any, Any]]
) -> str:
    """``N_out(1, B) Pi(r -> A, o -> B) N_in(A, 2)`` in the payload's syntax."""

    from symbolica import E

    insertion = E("0")
    for scale, structure in terms:
        insertion += scale * structure
    return _composed(entering, leaving, insertion)


def _composed(entering: str, leaving: str, insertion: Any) -> str:
    from symbolica import E

    inner = [E(f"UFO::{{}}::dummy({index})") for index in _INNER]
    first = _with_index(
        _momenta_as_placeholder(E(leaving)), 2, inner[1], _DUMMY_OFFSET[0]
    )
    second = _with_index(
        _momenta_as_placeholder(E(entering)), 1, inner[0], _DUMMY_OFFSET[1]
    )
    # Expanded, one spinor chain or tensor per term. (The unexpanded product
    # once seemed not to finish in ``custom_propagator_components``: that was
    # Spenso's Rayon workers waiting for the GIL under the package-scoped
    # Symbolica key, which ``pyamplicol/__init__`` now prevents.)
    composed = (first * insertion * second).expand()
    return _momenta_restored(composed).to_canonical_string()


def _engine_denominator(
    propagator: Mapping[str, Any], particle: Mapping[str, Any]
) -> str:
    """The denominator the tree engine applies for ``propagator``.

    A model-supplied propagator is applied as written; the record of a
    default propagator mirrors the engine's standard one, which adds the
    width, ``p^2 - M^2 + i M W`` (``models.expressions``).
    """

    denominator = str(propagator["denominator"])
    if propagator.get("pyamplicol_source") != "default-feynman":
        return denominator
    mass, width = str(particle["mass"]), str(particle["width"])
    if mass.upper() == "ZERO" or width.upper() == "ZERO":
        return denominator
    # Symbolica's imaginary unit.
    return f"({denominator})+𝑖*UFO::{{}}::{mass}*UFO::{{}}::{width}"  # noqa: RUF001


def _line_factor(
    model_name: str,
    propagators: Mapping[str, Mapping[str, Any]],
    particles: Mapping[str, Mapping[str, Any]],
    clones: Sequence[_Clone],
    terms: Sequence[tuple[str, Any]],
    lorentz: Mapping[str, Mapping[str, Any]],
) -> Any:
    """The scalar ``kappa`` with ``P_Y Pi P_X = kappa P`` on a Weyl line.

    Each term's Lorentz structure must give a constant multiple ``r_t`` of
    the clone's (massless) propagator, the same in both directions of the
    line; ``kappa`` is then ``sum_t r_t * scale_t``. Returns the reason as a
    string otherwise (a chiral or chirality-flipping ``Pi``).
    """

    from .._internal.physics.symbols import symbols
    from .external_evaluation import custom_propagator_components
    from .tensors import normalize_lorentz_expression

    model_symbols = symbols.model(model_name)

    def components(numerator: str, denominator: str, spin: int) -> tuple[Any, ...]:
        normalized = [
            model_symbols.expression(
                normalize_lorentz_expression(
                    text, (spin, spin), model_symbols=model_symbols, vertex=False
                ).expression
            )
            for text in (numerator, denominator)
        ]
        return custom_propagator_components(
            model_symbols, model_name, spin, 4, *normalized
        )

    def massless(name: str) -> tuple[str, str]:
        # A Weyl current propagates massless whatever the mass parameter.
        propagator = propagators[str(particles[name]["propagator"])]
        return tuple(
            _massless(str(propagator[part]), str(particles[name]["mass"]))
            for part in ("numerator", "denominator")
        )

    ratios: dict[int, list[Any]] = {}
    for clone in clones:
        spin = int(particles[clone.field]["spin"])
        leaving, leaving_denominator = massless(clone.field)
        entering, entering_denominator = massless(clone.into)
        line = components(leaving, leaving_denominator, spin)
        denominator = f"({entering_denominator})*({leaving_denominator})"
        for index, (_scale, structure) in enumerate(
            _insertion_terms(terms, lorentz, particles, clone.into)
        ):
            ratio = _constant_ratio(
                components(_composed(entering, leaving, structure), denominator, spin),
                line,
            )
            if ratio is None:
                return (
                    f"two-point Lorentz structure {terms[index][1].lorentz} on the "
                    f"Weyl line of {clone.field} is not a multiple of its propagator"
                )
            ratios.setdefault(index, []).append(ratio)
    if any(len(set(map(str, values))) != 1 for values in ratios.values()):
        return "two-point insertion differs between the directions of a Weyl line"
    total = None
    for index, (scale, _structure) in enumerate(
        _insertion_terms(terms, lorentz, particles, clones[0].into)
    ):
        term = ratios[index][0] * scale
        total = term if total is None else total + term
    return total


def _massless(expression: str, mass: str) -> str:
    from symbolica import E

    if mass.upper() == "ZERO":
        return expression
    return E(expression).replace(E(f"UFO::{{}}::{mass}"), E("0")).to_canonical_string()


def _constant_ratio(numerators: Sequence[Any], denominators: Sequence[Any]) -> Any:
    """The number ``r`` with ``numerators == r * denominators``, or ``None``."""

    from symbolica import E

    ratio = None
    for numerator, denominator in zip(numerators, denominators, strict=True):
        if denominator.expand() == E("0"):
            if numerator.expand() != E("0"):
                return None
            continue
        value = (numerator / denominator).cancel()
        if value.get_all_symbols(False) or (ratio is not None and value != ratio):
            return None
        ratio = value
    return ratio


def _scaled(expression: str, factor: Any) -> str:
    from symbolica import E

    return (E(expression) * factor).expand().to_canonical_string()


def _placeholder() -> Any:
    from .._internal.physics.symbols import symbols

    return symbols.symbol("ct_line_momentum")


def _momenta_as_placeholder(expression: Any) -> Any:
    """Single-index propagator momenta ``P(mu)`` as the placeholder ``q(mu)``."""

    from symbolica import E

    return expression.replace(E("UFO::{}::P(a_)"), _placeholder()(E("a_")))


def _momenta_restored(expression: Any) -> Any:
    """The placeholder ``q(mu)`` as ``P(mu, leg 1)``, the propagator momentum."""

    from symbolica import E

    return expression.replace(
        _placeholder()(E("a_")), E("UFO::{}::P(a_,UFO::{}::idx(1,1))")
    )


def _with_index(expression: Any, leg: int, index: Any, offset: int) -> Any:
    from symbolica import E

    return _renamed_dummies(expression, offset).replace(
        E(f"UFO::{{}}::idx(1,{leg})"), index
    )


def _renamed_dummies(expression: Any, offset: int) -> Any:
    from symbolica import E

    return expression.replace(
        E("UFO::{}::dummy(n_)"), E(f"UFO::{{}}::dummy(n_+{offset})")
    )


__all__ = ["LineInsertions", "lower_line_insertions"]
