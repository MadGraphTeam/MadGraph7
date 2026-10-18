# SPDX-License-Identifier: 0BSD
"""Exact external-field counterterms multiplying dependency-owned Born graphs.

These are the UFO particle.counterterm factors, not generated two-point
insertions. The model supplies any delta-Z/2 normalization. Each external leg
and each allowed loop-species alternative contributes once, as in MadLoop.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

from .graphs import GraphIR, LoopProcessIR
from .model import LoopModelIR
from .model_expressions import LaurentExpression, epsilon_symbol, parse_exact_ufo
from .orders import canonical_order, squared_order


def _entries(value, context):
    if not isinstance(value, Mapping) or value.get("kind") != "mapping":
        raise ValueError(f"{context} must be a raw UFO mapping")
    return value["entries"]


def _laurent(value, source):
    from symbolica import E, S

    expression, seen = E("0"), set()
    for power, coefficient in _entries(value, "external-field Laurent value"):
        if isinstance(power, bool) or not isinstance(power, int) or power in seen:
            raise ValueError("external-field epsilon powers must be distinct integers")
        seen.add(power)
        if isinstance(coefficient, bool) or not isinstance(coefficient, (str, int)):
            # Unlike Parameter.value, arbitrary particle attributes have no
            # observed source-literal contract. Never invent exactness from a
            # rounded float/complex value or a serialized numeric wrapper.
            raise ValueError(
                "external-field values require exact expression strings or integers"
            )
        expression += parse_exact_ufo(str(coefficient)) * epsilon_symbol() ** power
    # Existing scalar lowering has already resolved every CT-dependent alias.
    # Reuse its exact results, leaving ordinary model inputs symbolic.
    for definition in source.scalars.values():
        if (
            definition.collection == "counterterm_parameters"
            or definition.source != definition.value.expression
        ):
            expression = expression.replace(
                S(f"UFO::{definition.name}"), definition.value.expression
            )
    return LaurentExpression(expression)


@dataclass(frozen=True, slots=True)
class ExternalFieldRule:
    particle_id: int
    pdg: int
    spin: int
    color: int
    correction: str
    loop_group_index: int
    alternatives: tuple[tuple[int, ...], ...]
    coefficient: LaurentExpression

    @property
    def order(self):
        return ((self.correction, 2),)


def prepare_external_field_rules(source: LoopModelIR) -> tuple[ExternalFieldRule, ...]:
    """Read one-loop particle factors without fabricating a local interaction.

    Keys contain one power per source coupling order, in its original registry
    order, followed by a loop-group index. Exactly one unit power is a one-loop
    field correction; higher orders are outside this stream. Laurent values
    initially accept exact strings/integers, including CT-parameter aliases.
    """
    orders = tuple(
        record["name"] for record in source.raw.collections["coupling_orders"]
    )
    known_pdgs = {
        abs(record["fields"]["pdg_code"])
        for record in source.raw.collections["particles"]
    }
    rules, seen_particles = [], set()
    for particle in source.raw.collections["particles"]:
        fields = particle["fields"]
        values = fields.get("counterterm")
        if values is None:
            continue
        entries = _entries(values, "particle counterterms")
        if not entries:
            continue
        pdg = fields["pdg_code"]
        if pdg <= 0 or pdg in seen_particles:
            raise NotImplementedError(
                "external-field definitions require one positive-PDG particle record"
            )
        seen_particles.add(pdg)
        groups = fields.get("loop_particles")
        if not isinstance(groups, tuple):
            # RawLoopModel freezes source lists as tuples.
            raise ValueError("external-field loop groups must be explicitly declared")
        seen_keys = set()
        for encoded_key, value in entries:
            if (
                not isinstance(encoded_key, Mapping)
                or encoded_key.get("kind") != "tuple"
            ):
                raise ValueError("external-field order key must be a tuple")
            key = tuple(encoded_key["items"])
            if (
                len(key) != len(orders) + 1
                or any(
                    isinstance(item, bool) or not isinstance(item, int) or item < 0
                    for item in key
                )
                or key in seen_keys
            ):
                raise ValueError("invalid or duplicate external-field order key")
            seen_keys.add(key)
            powers, group_index = key[:-1], key[-1]
            if sum(powers) != 1:
                continue
            if group_index >= len(groups):
                raise ValueError("external-field loop-group index is absent")
            alternatives = tuple(
                tuple(alternative) for alternative in groups[group_index]
            )
            if not alternatives or any(
                isinstance(species, bool)
                or not isinstance(species, int)
                or species <= 0
                or species not in known_pdgs
                for alternative in alternatives
                for species in alternative
            ):
                raise ValueError(
                    "external-field alternatives require known positive PDGs"
                )
            rules.append(
                ExternalFieldRule(
                    particle["id"],
                    pdg,
                    fields["spin"],
                    fields["color"],
                    orders[powers.index(1)],
                    group_index,
                    alternatives,
                    _laurent(value, source),
                )
            )
    return tuple(rules)


@dataclass(frozen=True, slots=True)
class ExternalFieldInsertion:
    born: GraphIR = field(repr=False)
    external_index: int
    rule: ExternalFieldRule
    alternative_index: int

    @property
    def order(self):
        return squared_order(self.born.order, self.rule.order)

    def coefficient(self, epsilon_power: int):
        return self.rule.coefficient.coefficient(epsilon_power)


@dataclass(frozen=True, slots=True)
class ExternalFieldProcessIR:
    insertions: tuple[ExternalFieldInsertion, ...]
    stage: str = field(default="external-field-born-factor-oracle", init=False)
    scheme_validated: bool = field(default=False, init=False)


def match_external_field_counterterms(
    source: LoopModelIR,
    loops: LoopProcessIR,
    *,
    selected_corrections: Iterable[str],
    forbidden_particles: Iterable[int] = (),
    allowed_amplitude_orders: Iterable[tuple[tuple[str, int], ...]] | None = None,
) -> ExternalFieldProcessIR:
    """Attach model factors to admitted Borns, retaining leg/alternative identity.

    ``forbidden_particles`` is the caller's complete resolved process restriction,
    including native particle-selection/veto filters. ``allowed_amplitude_orders``
    optionally applies the caller's generated amplitude-order envelope. Neither
    is inferred from a numerical Born or from a loop-order difference.
    """
    selected = frozenset(selected_corrections)
    if selected != frozenset(loops.orders.correction_names):
        raise ValueError(
            "external fields and loop catalogue must share correction selection"
        )
    forbidden_values = tuple(forbidden_particles)
    if any(
        isinstance(pdg, bool) or not isinstance(pdg, int) for pdg in forbidden_values
    ):
        raise ValueError("forbidden external-field species must be integer PDGs")
    forbidden = frozenset(abs(pdg) for pdg in forbidden_values)
    allowed = (
        None
        if allowed_amplitude_orders is None
        else frozenset(canonical_order(order) for order in allowed_amplitude_orders)
    )
    rules = prepare_external_field_rules(source)
    insertions = []
    for born in loops.born:
        for leg in born.edges:
            if leg.external_index is None:
                continue
            for rule in rules:
                if abs(leg.pdg) != rule.pdg or rule.correction not in selected:
                    continue
                if leg.spin != rule.spin or abs(leg.color) != abs(rule.color):
                    raise ValueError(
                        "external-field source/native particle representations differ"
                    )
                order = squared_order(born.order, rule.order)
                if allowed is not None and order not in allowed:
                    continue
                for index, alternative in enumerate(rule.alternatives):
                    if not forbidden.intersection(alternative):
                        insertions.append(
                            ExternalFieldInsertion(
                                born, leg.external_index, rule, index
                            )
                        )
    return ExternalFieldProcessIR(tuple(insertions))
