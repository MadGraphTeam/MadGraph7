# SPDX-License-Identifier: 0BSD
"""Exact external-field counterterms multiplying dependency-owned Born graphs.

These are the UFO particle.counterterm factors, not generated two-point
insertions. The model supplies any delta-Z/2 normalization. Each external leg
and each allowed loop-species alternative contributes once, as in MadLoop.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from .graphs import GraphIR, LoopProcessIR
from .model import LoopModelIR
from .model_expressions import LaurentExpression, epsilon_symbol, parse_exact_ufo
from .orders import canonical_order, squared_order
from .scalar_frontend import (
    ScalarExpression,
    require_laurent_scalar,
    require_scalar_value,
)


def _entries(value, context):
    if not isinstance(value, Mapping) or value.get("kind") != "mapping":
        raise ValueError(f"{context} must be a raw UFO mapping")
    return value["entries"]


def _laurent(value, source):
    from symbolica import E, S

    if source.scalar_program is not None:
        library = source.scalar_program.library
        bindings = {}
        for definition in source.scalars.values():
            original = library.lower_value(
                source.raw.exact_value(definition.collection, definition.id)
            )
            if (
                definition.collection == "counterterm_parameters"
                or original.value != definition.value.expression
                or original.diagnostic != definition.diagnostic
            ):
                bindings[definition.name] = ScalarExpression(
                    definition.value.expression, definition.diagnostic
                )
        parts, seen = [], set()
        for power, coefficient in _entries(value, "external-field Laurent value"):
            if type(power) is not int or power in seen:
                raise ValueError(
                    "external-field epsilon powers must be distinct integers"
                )
            if isinstance(coefficient, bool) or not isinstance(coefficient, (str, int)):
                raise ValueError(
                    "external-field values require exact expression strings or integers"
                )
            seen.add(power)
            parts.append((power, library.lower(str(coefficient), locals=bindings)))
        scalar = ScalarExpression(
            sum(
                (part.value * epsilon_symbol() ** power for power, part in parts),
                E("0"),
            ),
            sum((part.diagnostic for _, part in parts), E("0")),
        )
        require_laurent_scalar(scalar)
        return LaurentExpression(scalar.value), scalar.diagnostic

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
    return LaurentExpression(expression), E("0")


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
    diagnostic: Any = field(default=None, repr=False)
    scalar_program: Any = field(default=None, repr=False)
    source_terms: tuple[tuple[int, str], ...] | None = field(default=None, repr=False)

    def coefficient_expression(self, power: int) -> ScalarExpression:
        from symbolica import E

        return ScalarExpression(
            self.coefficient.coefficient(power),
            E("0") if self.diagnostic is None else self.diagnostic,
        )

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
            coefficient, diagnostic = _laurent(value, source)
            rules.append(
                ExternalFieldRule(
                    particle["id"],
                    pdg,
                    fields["spin"],
                    fields["color"],
                    orders[powers.index(1)],
                    group_index,
                    alternatives,
                    coefficient,
                    diagnostic,
                    source.scalar_program,
                    tuple(
                        (power, str(coefficient))
                        for power, coefficient in _entries(
                            value, "external-field Laurent value"
                        )
                    ),
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

    def coefficient_expression(self, epsilon_power: int) -> ScalarExpression:
        return self.rule.coefficient_expression(epsilon_power)

    def coefficient(self, epsilon_power: int):
        return require_scalar_value(
            self.coefficient_expression(epsilon_power),
            "external-field coefficient",
            self.rule.scalar_program,
        )


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
