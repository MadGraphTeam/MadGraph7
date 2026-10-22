# SPDX-License-Identifier: 0BSD
"""Two-point counterterm insertions as clone lines of the tree model (design §6.1).

A two-point counterterm (CT) ``Pi`` on an internal line of a tree replaces the
line's propagator ``P`` by ``P Pi P``. In a Berends-Giele recursion every
internal line is one current, so the insertion is a unary step on a current,
which the tree engine expresses without a new node kind:

* **Clone particles.** Each line insertion class (field, physical orders of
  ``Pi``, slot) gets a clone of the field: the same quantum numbers, colour,
  spin and mass, and a model-supplied propagator ``P Pi P``, with the
  squared denominator of ``P``. ``Pi`` is the sum of the class's CT terms,
  each its exact derived coupling (``S_slot * sum M_rule * c_rule``) times the
  number of its colour (``c * delta``, ``ct_color_flow.two_point_colour_factor``)
  times its Lorentz structure. A fermion field and its antifield get a clone
  pair; a self-conjugate field one self-conjugate clone.
* **Weyl lines.** The tree engine propagates massless fermions as Weyl
  currents with its own propagator, which a model-supplied propagator would
  turn into Dirac currents that no external source closes. Such a clone
  keeps the field's propagator instead: on a Weyl line ``Pi`` must be a
  multiple of the slashed momentum, so ``P Pi P = kappa P`` with an exact
  scalar ``kappa`` (``_line_factor``), which multiplies the couplings of the
  copies on the clone's antiparticle leg. A clone line joins one leg of the
  clone and one of its antiparticle, so it carries ``kappa`` once.
* **Vertex copies.** Every ordinary vertex with a leg of the field (or of its
  antifield) is copied once per such species, with that leg on the clone;
  the copy's couplings carry the original orders, half of the orders of
  ``Pi``, the insertion order once and the slot order once.

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
vertex with legs ``r`` (the current's particle, incoming momentum ``p``) and
``o`` (incoming momentum ``-p``)::

    N_clone(1, 2) = N(1, B) Pi(r -> A, o -> B) N(A, 2) / D^2

Refused (``refused``, the caller keeps such rules as site closures): mixing
insertions between different fields, orders of ``Pi`` that are not even,
spins other than 0, 1/2 and 1, fields without a propagator record,
Lorentz structures with momentum heads other than ``P``, vertices with two
legs of the same fermion species of the field, and on Weyl lines terms that
are no multiple of the slashed momentum (they flip the chirality).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .ct_color_flow import UnloweredCountertermColour, two_point_colour_factor

_SPINS = (1, 2, 3)
_DUMMY_OFFSET = (1000, 2000, 3000)  # dummies of the two propagators and Pi
_INNER = (9001, 9002)  # the indices A and B of the composition


@dataclass
class LineInsertions:
    """Payload records of the clone lines of a set of two-point CT terms.

    ``clones`` maps every lowered term to its clone particle, ``members``
    every vertex copy to the terms of its clone line and ``refused`` every
    term that cannot be lowered to the reason.
    """

    parameters: list[dict[str, Any]] = field(default_factory=list)
    particles: list[dict[str, Any]] = field(default_factory=list)
    propagators: list[dict[str, Any]] = field(default_factory=list)
    couplings: list[dict[str, Any]] = field(default_factory=list)
    vertex_rules: list[dict[str, Any]] = field(default_factory=list)
    clones: dict[Any, str] = field(default_factory=dict)
    members: dict[str, tuple[Any, ...]] = field(default_factory=dict)
    refused: dict[Any, str] = field(default_factory=dict)


def lower_line_insertions(
    payload: Mapping[str, Any],
    insertions: Mapping[Any, str],
    *,
    prefix: str,
    insertion_order: str,
    slot_order: Callable[[Any], str],
    weyl_fields: frozenset[str] = frozenset(),
) -> LineInsertions:
    """Clone lines of the two-point CT terms ``insertions`` (term -> coupling).

    ``payload`` is the compiled model payload whose ordinary vertices are
    copied (it is not modified); ``prefix`` prefixes every new record name,
    ``insertion_order`` and ``slot_order(term)`` name the pseudo-orders.
    ``weyl_fields`` are the fermions whose currents the tree engine keeps as
    Weyl currents (massless, standard propagator); their clone lines keep the
    standard propagator and carry ``P Pi P = kappa P`` as the factor
    ``kappa`` (``_line_factor``).
    """

    from symbolica import E

    particles = {str(item["name"]): item for item in payload["particles"]}
    propagators = {str(item["name"]): item for item in payload["propagators"]}
    lorentz = {str(item["name"]): item for item in payload["lorentz_structures"]}
    out = LineInsertions()
    classes: dict[tuple[object, ...], list[Any]] = defaultdict(list)
    for term in sorted(insertions):
        reason = _refusal(term, particles, propagators, lorentz)
        if reason is not None:
            out.refused[term] = reason
            continue
        first, second = term.particles
        canonical = first if int(particles[first]["pdg_code"]) > 0 else second
        classes[canonical, term.orders, term.component, term.power].append(term)
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
    for number, ((source, orders, _component, _power), terms) in enumerate(
        sorted(classes.items())
    ):
        clone = f"{prefix}_LINE{number}"
        species = {source: clone}
        anti = str(particles[source]["antiname"])
        if anti != source:
            species[anti] = f"{clone}~"
        reason = _copy_refusal(copyable, species, particles)
        if reason is not None:
            out.refused.update(dict.fromkeys(terms, reason))
            continue
        factor = None
        if source in weyl_fields:
            factor = _line_factor(
                str(payload["name"]),
                propagators,
                particles,
                species,
                [(insertions[term], term) for term in terms],
                lorentz,
            )
            if isinstance(factor, str):
                out.refused.update(dict.fromkeys(terms, factor))
                continue
        scales: list[str] = []
        if factor is None:
            # The scales (coupling times colour) are derived model parameters,
            # so that the propagator kernel stays free of constants such as
            # pi, which compiled kernels round to binary64.
            for index, (value, _structure) in enumerate(
                _insertion_terms(
                    [(insertions[term], term) for term in terms],
                    lorentz,
                    particles,
                    source,
                )
            ):
                scales.append(f"{clone}_X{index}")
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
        for name, clone_name in species.items():
            record = dict(particles[name])
            record.update(
                name=clone_name,
                antiname=species[str(record["antiname"])],
                pdg_code=pdg if int(record["pdg_code"]) > 0 else -pdg,
                propagator=f"{clone_name}_prop",
            )
            out.particles.append(record)
            source_propagator = propagators[str(particles[name]["propagator"])]
            line = {**source_propagator, "name": f"{clone_name}_prop"}
            if factor is None:
                line.update(
                    numerator=_line_numerator(
                        source_propagator["numerator"],
                        [
                            (E(f"UFO::{{}}::{scale}"), structure)
                            for scale, (_value, structure) in zip(
                                scales,
                                _insertion_terms(
                                    [(insertions[term], term) for term in terms],
                                    lorentz,
                                    particles,
                                    name,
                                ),
                                strict=True,
                            )
                        ],
                    ),
                    denominator=f"({source_propagator['denominator']})^2",
                    pyamplicol_source="model-supplied",
                )
            out.propagators.append({**line, "particle": clone_name})
        pdg += 1
        extra = {
            **{name: power // 2 for name, power in orders},
            insertion_order: 1,
            slot_order(terms[0]): 1,
        }
        out.clones.update(dict.fromkeys(terms, clone))
        # A clone line joins a leg of the clone and a leg of its antiparticle,
        # so the factor of a Weyl line sits on the copies of the latter.
        scaled = {anti} if factor is not None else set()
        copied: dict[tuple[str, bool], str] = {}
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
                        key = (coupling, scale)
                        if coupling is not None and key not in copied:
                            copied[key] = f"{clone}_C{len(copied)}"
                            original = couplings[coupling]
                            record = {
                                **original,
                                "name": copied[key],
                                "orders": _added_orders(original["orders"], extra),
                            }
                            if scale:
                                record.update(
                                    expression=_scaled(original["expression"], factor),
                                    value=[1.0, 0.0],
                                )
                            out.couplings.append(record)
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
) -> str | None:
    first, second = term.particles
    if str(particles[first]["antiname"]) != second:
        return f"two-point mixing insertion {first}-{second}"
    record = particles[first]
    if int(record["spin"]) not in _SPINS:
        return f"two-point insertion on a spin-{record['spin']} line"
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
        structure = structure.replace(
            E(f"UFO::{{}}::idx(1,{receiving})"), inner[0]
        ).replace(E(f"UFO::{{}}::idx(1,{other})"), inner[1])
        colour = two_point_colour_factor(
            term.colour, [int(particles[name]["color"]) for name in legs]
        )
        scale = E(coupling) * E(f"{colour.numerator}/{colour.denominator}")
        out.append((scale, structure))
    return out


def _line_numerator(numerator: str, terms: Sequence[tuple[Any, Any]]) -> str:
    """``N(1, B) Pi(r -> A, o -> B) N(A, 2)`` in the payload's syntax."""

    from symbolica import E

    insertion = E("0")
    for scale, structure in terms:
        insertion += scale * structure
    return _composed(numerator, insertion)


def _composed(numerator: str, insertion: Any) -> str:
    from symbolica import E

    inner = [E(f"UFO::{{}}::dummy({index})") for index in _INNER]
    first, second = (
        _with_index(_momenta_as_placeholder(E(numerator)), leg, inner[slot], offset)
        for leg, slot, offset in ((2, 1, _DUMMY_OFFSET[0]), (1, 0, _DUMMY_OFFSET[1]))
    )
    return _momenta_restored(first * insertion * second).to_canonical_string()


def _line_factor(
    model_name: str,
    propagators: Mapping[str, Mapping[str, Any]],
    particles: Mapping[str, Mapping[str, Any]],
    species: Mapping[str, str],
    terms: Sequence[tuple[str, Any]],
    lorentz: Mapping[str, Mapping[str, Any]],
) -> Any:
    """The scalar ``kappa`` with ``P Pi P = kappa P`` on a Weyl fermion line.

    Each term's Lorentz structure must give a constant multiple ``r_t`` of
    the field's propagator (``Pi_t`` proportional to the slashed momentum),
    the same for the field and its antifield; ``kappa`` is then
    ``sum_t r_t * scale_t``. Returns the reason as a string otherwise.
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

    ratios: dict[int, list[Any]] = {}
    for name in species:
        propagator = propagators[str(particles[name]["propagator"])]
        spin = int(particles[name]["spin"])
        # A Weyl current propagates massless whatever the mass parameter.
        numerator, denominator = (
            _massless(str(propagator[part]), str(particles[name]["mass"]))
            for part in ("numerator", "denominator")
        )
        line = components(numerator, denominator, spin)
        squared = f"({denominator})^2"
        for index, (_scale, structure) in enumerate(
            _insertion_terms(terms, lorentz, particles, name)
        ):
            ratio = _constant_ratio(
                components(_composed(numerator, structure), squared, spin), line
            )
            if ratio is None:
                return (
                    f"two-point Lorentz structure {terms[index][1].lorentz} on the "
                    f"Weyl line of {name} is not a multiple of its propagator"
                )
            ratios.setdefault(index, []).append(ratio)
    if any(len(set(map(str, values))) != 1 for values in ratios.values()):
        return "two-point insertion differs between a Weyl field and its antifield"
    total = None
    for index, (scale, _structure) in enumerate(
        _insertion_terms(terms, lorentz, particles, next(iter(species)))
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
