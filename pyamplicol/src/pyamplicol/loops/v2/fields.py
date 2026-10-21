# SPDX-License-Identifier: 0BSD
"""Scalar external-field counterterms of the v2 engine (design §6.5).

External self-energies are never sites (LSZ, ``loops.v2.ct_sites``); the model
supplies them as UFO ``particle.counterterm`` factors, delta-Z/2 per external
leg. Each external leg, each field rule of its species and correction, and each
loop-species alternative of the rule's loop group that the process does not
forbid contributes once, as in MadLoop and the legacy
``loops.wavefunction.match_external_field_counterterms``.

v2 has no Born graphs, so an insertion is keyed by the Born amplitude order b
it multiplies instead: its amplitude order is b plus the rule's correction
order, and it is kept only when that order is an admitted loop amplitude
order. The field component of a squared-order sector is

    sum_insertions 2 Re( c_eps * <A0_a | A0_b> ) / average

over the sector's (Born order a, loop order b + delta) pairs, where c_eps is the
rule's Laurent coefficient (the model includes the 1/2) and <A0_a|A0_b> the
helicity- and colour-summed Born interference of the two amplitude orders.
Chiral and mixing delta-Z are CT-tagged sources in the tree recursion (wave 2).

Field coefficients stay symbolic: ``counterterm_scalar_plan`` binds them,
together with the site couplings, through the source scalar program, with the
legacy's anonymous Laurent definitions of the particle factors.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..scalar_bindings import ScalarBindingPlan, prepare_scalar_bindings
from ..wavefunction import ExternalFieldRule
from .ct_sites import LAURENT_POWERS, CountertermSites
from .families import ExternalLeg, LoopOrderSelection, Orders, _model_data, add_orders

FIELD_COMPONENT = "uv_external_fields"


@dataclass(frozen=True, slots=True)
class FieldInsertion:
    """One external-field factor: a leg, a rule alternative, a Born order."""

    leg: int  # 0-based process leg (bit of the block masks)
    rule: ExternalFieldRule = field(repr=False)
    alternative: int
    born_order: Orders  # amplitude order of the Born the factor multiplies

    @property
    def orders(self) -> Orders:
        """Amplitude order of the insertion (the loop amplitude order)."""
        return add_orders(self.born_order, self.rule.order)

    @property
    def identity(self) -> tuple[int, str, int]:
        """The rule's source identity (particle record, correction, loop group)."""
        return (self.rule.particle_id, self.rule.correction, self.rule.loop_group_index)

    @property
    def insertion_key(self) -> str:
        """Legacy insertion-multiset key."""
        return f"leg{self.leg}:{self.rule.pdg}"


def external_field_insertions(
    legs: Sequence[ExternalLeg],
    rules: Iterable[ExternalFieldRule],
    native_model: Any,
    selection: LoopOrderSelection,
    born_orders: Iterable[Orders],
    *,
    forbidden: Iterable[int] = (),
) -> tuple[FieldInsertion, ...]:
    """Field insertions of a process (``legs`` carry incoming field names).

    ``rules`` are ``loops.wavefunction.prepare_external_field_rules`` of the
    prepared source model, ``born_orders`` the exact Born amplitude orders and
    ``forbidden`` the |PDG| codes the process excludes.
    """

    particles = {
        entry["name"]: entry for entry in _model_data(native_model)["particles"]
    }
    forbidden = frozenset(abs(int(pdg)) for pdg in forbidden)
    rules = tuple(rules)
    born_orders = tuple(sorted({add_orders(order) for order in born_orders}))
    out = []
    for leg in legs:
        particle = particles[leg.field]
        for rule in rules:
            if abs(int(particle["pdg_code"])) != rule.pdg:
                continue
            if rule.correction not in selection.corrections:
                continue
            representation = (particle["spin"], abs(particle["color"]))
            if representation != (rule.spin, abs(rule.color)):
                raise ValueError(
                    "external-field source and native particle representations differ"
                )
            for born_order in born_orders:
                if add_orders(born_order, rule.order) not in selection.loop_orders:
                    continue
                for index, alternative in enumerate(rule.alternatives):
                    if not forbidden.intersection(alternative):
                        out.append(FieldInsertion(leg.index, rule, index, born_order))
    return tuple(out)


def field_definitions(
    insertions: Iterable[FieldInsertion],
) -> dict[tuple[int, str, int], tuple[str, Mapping[str, Any]]]:
    """Anonymous scalar name and Laurent source of each field rule identity."""

    out: dict[tuple[int, str, int], tuple[str, Mapping[str, Any]]] = {}
    for insertion in insertions:
        terms = insertion.rule.source_terms
        if terms is None:
            raise ValueError("external-field rule has no source Laurent terms")
        known = out.get(insertion.identity)
        if known is None:
            name = f"loop_v2_particle_field_{len(out)}"
            out[insertion.identity] = (name, {"kind": "laurent", "terms": terms})
        elif known[1]["terms"] != terms:
            raise ValueError("external-field identity has conflicting definitions")
    return out


def counterterm_scalar_plan(
    structure: Any,
    sites: CountertermSites,
    insertions: Iterable[FieldInsertion],
) -> tuple[ScalarBindingPlan, dict[tuple[int, str, int], str]]:
    """One source scalar plan for every site coupling and field coefficient.

    Outputs are ``(name, epsilon power)`` for powers 0, -1, -2; the second
    value maps each field rule identity to its output name. ``structure`` is
    the prepared covariant structure (``prepare_loop_request_model(...).structure``).
    """

    definitions = field_definitions(insertions)
    outputs = sorted(
        {
            *sites.coupling_outputs(),
            *(
                (name, power)
                for name, _source in definitions.values()
                for power in LAURENT_POWERS
            ),
        }
    )
    plan = prepare_scalar_bindings(
        structure.raw,
        outputs,
        program=structure.scalars,
        fixed_parameters=structure.fixed_parameters,
        anonymous_definitions={name: source for name, source in definitions.values()},
    )
    return plan, {identity: name for identity, (name, _) in definitions.items()}


__all__ = [
    "FIELD_COMPONENT",
    "FieldInsertion",
    "counterterm_scalar_plan",
    "external_field_insertions",
    "field_definitions",
]
