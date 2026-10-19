# SPDX-License-Identifier: 0BSD
"""Exact species symmetries of a retained ordinary/UV/R2 source plan.

This cold prerequisite proves no external-leg, spin-source or colour transport.
It does not establish the plan's physics completeness or correction selection.
Tree flavour profiles may propose a permutation; only this exact source check
can certify the retained loop plan. Charge conjugation and direct R2 are outside
the present domain.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import cache
from typing import Any

from .graphs import _rule_orders, _species_orders
from .model_expressions import LaurentExpression, epsilon_symbol, parse_exact_ufo
from .scalar_bindings import prepare_scalar_bindings
from .source_color import SourceColorContinuation
from .wavefunction import prepare_external_field_rules


@dataclass(frozen=True, slots=True)
class RetainedFlavorPlan:
    """Existing source owners and the actual retained ordinary-rule allow-list.

    ``ordinary_rules`` must be the complete retained generation allow-list, not
    the vertices observed in one representative's generated diagrams. Every rule
    in ``counterterms`` is checked, including all of its local basis components.
    The caller retains responsibility for choosing a physically complete plan.
    """

    structure: Any = field(repr=False)
    counterterms: Any = field(repr=False)
    ordinary_rules: tuple[str, ...]
    correction_names: tuple[str, ...]
    source_color: SourceColorContinuation | None = field(default=None, repr=False)

    def __post_init__(self):
        source = self.counterterms.source
        if (
            self.counterterms.ordinary_model is not self.structure.native
            or source.raw is not self.structure.raw
            or source.scalar_program is not self.structure.scalars
        ):
            raise ValueError("flavour proof requires one retained source/scalar owner")
        if self.source_color is not None and (
            not isinstance(self.source_color, SourceColorContinuation)
            or self.source_color.model is not source
            or self.source_color.model.scalar_program is not self.structure.scalars
        ):
            raise ValueError(
                "flavour colour continuation requires the retained source owner"
            )
        rules, corrections = tuple(self.ordinary_rules), tuple(self.correction_names)
        available = {rule.name for rule in self.structure.native.vertex_rules}
        if not rules or len(set(rules)) != len(rules) or not set(rules) <= available:
            raise ValueError("invalid retained ordinary-rule allow-list")
        orders = {
            record["name"]: record["fields"].get("perturbative_expansion", 0)
            for record in source.raw.collections["coupling_orders"]
        }
        if (
            not corrections
            or len(set(corrections)) != len(corrections)
            or any(orders.get(name, 0) < 1 for name in corrections)
        ):
            raise ValueError("invalid retained loop correction names")
        object.__setattr__(self, "ordinary_rules", rules)
        object.__setattr__(self, "correction_names", corrections)


@dataclass(frozen=True, slots=True)
class FlavorSymmetry:
    """A species-only proof, tied to the exact retained source/plan owners."""

    plan: RetainedFlavorPlan = field(repr=False)
    species: tuple[tuple[int, int], ...]
    ordinary_rule_count: int
    counterterm_rule_count: int
    external_field_rule_count: int
    stage: str = field(default="exact-retained-species-symmetry", init=False)
    external_transport_validated: bool = field(default=False, init=False)
    direct_r2_validated: bool = field(default=False, init=False)


def _powers(expression):
    """Use the dependency's finite Laurent decomposition without truncation."""
    epsilon = epsilon_symbol()
    result = []
    for monomial, coefficient in expression.together().coefficient_list(epsilon):
        if epsilon in coefficient.get_all_symbols():
            raise NotImplementedError("flavour proof requires finite Laurent scalars")
        rational = monomial.to_rational_polynomial(vars=[epsilon])
        degree = rational.numerator().degree(epsilon) - rational.denominator().degree(
            epsilon
        )
        if monomial != epsilon**degree:
            raise NotImplementedError("unsupported epsilon power in flavour proof")
        result.append(degree)
    # A zero value can still carry an active source diagnostic.
    return tuple(sorted(result)) or (0,)


def _canonical(expression):
    return expression.together().expand().to_canonical_string()


def _alternatives(alternatives, relabel):
    # Alternatives are a multiset, not a set. Repeated UFO entries retain their
    # multiplicity; within an alternative only its listed ordering is immaterial.
    return tuple(
        sorted(
            tuple(sorted(relabel.get(pdg, pdg) for pdg in alt)) for alt in alternatives
        )
    )


def _whole_rule_aliases(rule, model):
    """Check the factory's whole replacement, without an entrywise UFO pairing."""
    from symbolica.community import hepkit as hep

    from .counterterms import _TAG

    continuation = rule.whole_rule_continuation
    if continuation is None:
        if rule.native_coefficients:
            raise ValueError("native coefficients lack their whole-rule owner")
        return {}
    if Counter(map(id, rule.entries)) != Counter(map(id, continuation.source_entries)):
        raise ValueError("whole-rule source entries or multiplicities differ")
    native = model.vertex_rule(rule.name)
    terms, coefficients = continuation.terms, rule.native_coefficients
    if (
        len(coefficients) != len(terms)
        or len({name for name, _ in coefficients}) != len(coefficients)
        or len(native.color_structures) != len(terms)
        or len(native.lorentz_structures) != len(terms)
        or len(native.couplings) != len(terms)
    ):
        raise ValueError("whole-rule native coefficient/tensor coverage differs")
    for index, ((alias, value), term) in enumerate(
        zip(coefficients, terms, strict=True)
    ):
        if (
            value != term.coefficient
            or tuple(native.particles) != term.kernel.particle_names
            or len(term.kernel.leg_half_edges) != len(native.particles)
        ):
            raise ValueError("whole-rule coefficient or ordered ports differ")
        if tuple(native.couplings[index]) != tuple(
            alias if column == index else None for column in range(len(terms))
        ):
            raise ValueError("whole-rule native alias slots differ")
        coupling = model.coupling(alias)
        if coupling.expression != value or tuple(
            sorted(coupling.orders.items())
        ) != tuple(sorted((*rule.order, (_TAG, 1)))):
            raise ValueError("whole-rule native coefficient or coupling orders differ")
        color, lorentz = (
            term.kernel.color.to_expression(),
            term.kernel.lorentz.to_expression(),
        )
        # The same dependency-owned port renaming used by prepare_counterterm_model.
        for leg, half in enumerate(term.kernel.leg_half_edges, 1):
            port = hep.Symbols.half_edge()(half, 1)
            local = hep.Symbols.ufo_index()(1, leg)
            color, lorentz = color.replace(port, local), lorentz.replace(port, local)
        actual_lorentz = model.lorentz_structure(native.lorentz_structures[index])
        if (
            _canonical(native.color_structures[index]) != _canonical(color)
            or _canonical(actual_lorentz.structure) != _canonical(lorentz)
            or tuple(actual_lorentz.spins)
            != tuple(model.particle(name).spin for name in native.particles)
        ):
            raise ValueError(
                "whole-rule native tensors differ from the owned continuation"
            )
    return dict(coefficients)


def prove_flavor_relabeling(
    plan: RetainedFlavorPlan,
    species_map: Mapping[int, int],
    *,
    charge_conjugated: bool = False,
    r2_strategy: str = "model",
) -> FlavorSymmetry:
    """Prove an explicit signed-PDG permutation for every retained local rule.

    Independent external parameters remain distinct symbolic inputs, regardless
    of their defaults. Only the structure's existing fixed-parameter bindings
    can collapse them. Both source entries and actual prepared native tensors
    are compared in their exact ordered frames; no tensor identities or local
    port permutations are invented here. Scalar algebra belongs to Symbolica.
    """
    if not isinstance(plan, RetainedFlavorPlan):
        raise TypeError("expected a retained flavour plan")
    if charge_conjugated or r2_strategy != "model":
        raise NotImplementedError(
            "flavour proof supports model R2 without charge conjugation"
        )
    if not isinstance(species_map, Mapping) or any(
        type(a) is not int or type(b) is not int for a, b in species_map.items()
    ):
        raise TypeError("species relabeling requires integer PDGs")
    source, structure = plan.counterterms.source, plan.structure
    from symbolica import S

    raw = source.raw
    particles = {
        record["fields"]["pdg_code"]: record for record in raw.collections["particles"]
    }
    if len(particles) != len(raw.collections["particles"]):
        raise ValueError("ambiguous source particle PDGs")
    if (
        not set(species_map) <= particles.keys()
        or not set(species_map.values()) <= particles.keys()
    ):
        raise ValueError("flavour permutation contains unknown species")
    relabel = {pdg: species_map.get(pdg, pdg) for pdg in particles}
    if set(relabel.values()) != set(particles):
        raise ValueError("flavour relabeling must be a complete bijection")
    by_name = {record["name"]: pdg for pdg, record in particles.items()}
    anti = {
        pdg: by_name[record["fields"]["antiname"]] for pdg, record in particles.items()
    }
    for pdg, target in relabel.items():
        if relabel[anti[pdg]] != anti[target]:
            raise ValueError("flavour permutation must commute with antiparticles")
        if pdg != target and (
            particles[pdg]["fields"]["spin"] != 2
            or anti[pdg] == pdg
            or (pdg > 0) != (target > 0)
        ):
            raise NotImplementedError(
                "only orientation-preserving Dirac flavour permutations are certified"
            )

    # Native whole-model metadata determines perturbative species membership;
    # the retained allow-list does not reconstruct it from an amplitude order.
    eligibility = _species_orders(structure.native, _rule_orders(structure.native))
    if any(
        eligibility.get(abs(pdg)) != eligibility.get(abs(target))
        for pdg, target in relabel.items()
    ):
        raise ValueError("flavour permutation changes correction eligibility")

    native_models = (structure.native, plan.counterterms.native)
    native_payloads = tuple(json.loads(model.to_json()) for model in native_models)
    for payload in native_payloads:
        multiplets = payload.get("covariant_cut_multiplets", {})
        partners = tuple(
            sorted(
                (int(pdg), tuple(sorted(values))) for pdg, values in multiplets.items()
            )
        )
        mapped_partners = tuple(
            sorted(
                (relabel[pdg], tuple(sorted(relabel[value] for value in values)))
                for pdg, values in partners
            )
        )
        if partners != mapped_partners:
            raise ValueError("flavour permutation changes covariant multiplet partners")

    ordinary = {}
    for entry in source.kernels:
        if entry.kind == "tree" and entry.vertex_name in plan.ordinary_rules:
            ordinary.setdefault(entry.vertex_name, []).append(entry)
    if set(ordinary) != set(plan.ordinary_rules):
        raise ValueError("retained ordinary rules lack exact source entries")
    counterterms = tuple(plan.counterterms.rules.values())
    fields = tuple(
        rule
        for rule in prepare_external_field_rules(source)
        if rule.correction in plan.correction_names
    )
    entries = [entry for values in ordinary.values() for entry in values]
    entries += [entry for rule in counterterms for entry in rule.entries]
    whole_aliases = {
        rule.name: _whole_rule_aliases(rule, plan.counterterms.native)
        for rule in counterterms
    }
    if any(whole_aliases.values()) and plan.source_color is None:
        raise ValueError(
            "whole-rule flavour proof requires its formal source continuation"
        )

    outputs = {}
    for entry in entries:
        for power in _powers(entry.coupling.value.expression):
            outputs[entry.coupling.name, power] = None
    parameter_names = {
        record["fields"][key]["name"]
        for record in particles.values()
        for key in ("mass", "width")
    }
    for payload in native_payloads:
        parameter_names.update(
            particle[key]
            for particle in payload["particles"]
            for key in ("mass", "width")
        )
    ordinary_parameters = set(parameter_names)
    propagator_symbols = {
        symbol
        for model in native_models
        for propagator in model.propagators
        for value in (propagator.numerator, propagator.denominator)
        for symbol in value.get_all_symbols()
    }
    field_symbols = {
        symbol
        for rule in fields
        for value in (rule.coefficient.expression, rule.diagnostic)
        for symbol in value.get_all_symbols()
    }
    field_symbols.update(propagator_symbols)
    for record in raw.collections["parameters"]:
        if S(f"UFO::{record['name']}") in propagator_symbols:
            ordinary_parameters.add(record["name"])
        if S(f"UFO::{record['name']}") in field_symbols:
            parameter_names.add(record["name"])
    for name in sorted(parameter_names):
        outputs[name, 0] = None
    binding = prepare_scalar_bindings(
        raw,
        tuple(outputs),
        program=structure.scalars,
        fixed_parameters=structure.fixed_parameters,
    )
    field_names = {}
    if plan.source_color is not None:
        color = plan.source_color
        color.require_kernels(entry.key for entry in entries)
        color.require_fields(
            (rule.particle_id, rule.correction, rule.loop_group_index)
            for rule in fields
        )
        color.require_whole_rules(counterterms, binding)
        ordinary_outputs = tuple(
            dict.fromkeys(
                [
                    (entry.coupling.name, power)
                    for values in ordinary.values()
                    for entry in values
                    for power in _powers(entry.coupling.value.expression)
                ]
                + [(name, 0) for name in sorted(ordinary_parameters)]
            )
        )
        color.require_ordinary_scalars(
            prepare_scalar_bindings(
                raw,
                ordinary_outputs,
                program=structure.scalars,
                fixed_parameters=structure.fixed_parameters,
            )
        )
        anonymous = {}
        for index, rule in enumerate(fields):
            if rule.source_terms is None:
                raise ValueError(
                    "formal field symmetry requires the owned source terms"
                )
            name = f"loop_flavour_field_{index}"
            field_names[id(rule)] = name
            anonymous[name] = {"kind": "laurent", "terms": rule.source_terms}
            for power in _powers(rule.coefficient.expression):
                outputs[name, power] = None
        for name in color.scheme_parameter_values:
            outputs[name, 0] = None
        binding = color.formal_scalars(
            tuple(outputs),
            program=structure.scalars,
            fixed_parameters=structure.fixed_parameters,
            anonymous_definitions=anonymous,
            exact_outputs={
                (name, power): LaurentExpression(expression).coefficient(power)
                for aliases in whole_aliases.values()
                for name, expression in aliases.items()
                for power in _powers(expression)
            },
        )
        # Scheme equality alone must not discard an active source diagnostic.
        if color.scheme_parameter_values:
            schemes = prepare_scalar_bindings(
                raw,
                tuple((name, 0) for name in color.scheme_parameter_values),
                program=structure.scalars,
                fixed_parameters=structure.fixed_parameters,
            )
            if any(value.diagnostic != 0 for value in schemes.expressions):
                raise ValueError(
                    "flavour scheme ties require inactive source diagnostics"
                )
    resolved = dict(zip(binding.outputs, binding.expressions, strict=True))

    @cache
    def scalar(name, powers):
        return tuple(
            (
                power,
                _canonical(resolved[name, power].value),
                _canonical(resolved[name, power].diagnostic),
            )
            for power in powers
        )

    @cache
    def resolve_native(expression):
        for name in parameter_names:
            symbol = S(f"UFO::{name}")
            if symbol not in expression.get_all_symbols():
                continue
            replacement = resolved[name, 0]
            if replacement.diagnostic != 0:
                raise NotImplementedError(
                    "native tensor symmetry needs paired parameter diagnostics"
                )
            expression = expression.replace(symbol, replacement.value)
        return _canonical(expression)

    def particle_signature(record):
        data = record["fields"]
        if not data.get("charge_source", {}).get("exact", False):
            raise ValueError("particle charge lacks exact source metadata")
        charge = parse_exact_ufo(data["charge_exact"]["expression"])
        descriptive = {
            "pdg_code",
            "name",
            "antiname",
            "texname",
            "antitexname",
            "line",
            "spin",
            "color",
            "mass",
            "width",
            "charge",
            "charge_exact",
            "charge_source",
            "LeptonNumber",
            "GhostNumber",
            "propagating",
            "goldstoneboson",
            "GoldstoneBoson",
            "selfconjugate",
            "loop_particles",
            "counterterm",
        }
        # Unknown quantum/model annotations are never silently flavour-blind.
        extra = tuple(
            sorted(
                (key, repr(value))
                for key, value in data.items()
                if key not in descriptive
            )
        )
        return (
            data["spin"],
            data["color"],
            _canonical(charge),
            scalar(data["mass"]["name"], (0,)),
            scalar(data["width"]["name"], (0,)),
            data.get("LeptonNumber", 0),
            data.get("GhostNumber", 0),
            data.get("propagating", True),
            data.get("goldstoneboson", data.get("GoldstoneBoson", False)),
            anti[data["pdg_code"]] == data["pdg_code"],
            extra,
        )

    for pdg, target in relabel.items():
        if pdg != target and particle_signature(particles[pdg]) != particle_signature(
            particles[target]
        ):
            raise ValueError(
                "flavour permutation changes mass, width, charge or particle metadata"
            )

    # The retained native models are public immutable producer objects, but may
    # have been constructed with explicit local representatives. Compare their
    # actual propagators and particle metadata, not an assumed uniform lowering.
    for model, payload in zip(native_models, native_payloads, strict=True):
        native_particles = {item["pdg_code"]: item for item in payload["particles"]}
        if set(native_particles) != set(particles):
            raise ValueError("native/source species sets differ")

        def native_particle_signature(
            pdg, model=model, native_particles=native_particles
        ):
            item = native_particles[pdg]
            native = model.particle(item["name"])
            prop = model.propagator(item["propagator"])
            metadata = tuple(
                sorted(
                    (key, repr(value))
                    for key, value in item.items()
                    if key
                    not in {
                        "pdg_code",
                        "name",
                        "antiname",
                        "texname",
                        "antitexname",
                        "mass",
                        "width",
                        "propagator",
                    }
                )
            )
            return (
                metadata,
                native.is_massless,
                scalar(item["mass"], (0,)),
                scalar(item["width"], (0,)),
                resolve_native(prop.numerator),
                resolve_native(prop.denominator),
            )

        if any(
            native_particle_signature(pdg) != native_particle_signature(target)
            for pdg, target in relabel.items()
            if pdg != target
        ):
            raise ValueError(
                "flavour permutation changes prepared native particles or propagators"
            )

    def basis(entry):
        return (
            entry.color_index,
            entry.lorentz_index,
            entry.color,
            entry.lorentz,
            entry.order,
            scalar(entry.coupling.name, _powers(entry.coupling.value.expression)),
        )

    def ordinary_signature(values, permutation):
        first = values[0]
        pdgs = tuple(
            raw.collections["particles"][index]["fields"]["pdg_code"]
            for index in first.particle_ids
        )
        return tuple(permutation.get(pdg, pdg) for pdg in pdgs), tuple(
            sorted(basis(entry) for entry in values)
        )

    identity = {}
    if Counter(
        ordinary_signature(values, identity) for values in ordinary.values()
    ) != Counter(ordinary_signature(values, relabel) for values in ordinary.values()):
        raise ValueError(
            "flavour permutation changes retained ordinary tensors or couplings"
        )

    def native_rule_signature(model, name, aliases, permutation, whole=None):
        rule = model.vertex_rule(name)
        pdgs = tuple(model.particle(name).pdg_code for name in rule.particles)
        slots = []
        for color_index, row in enumerate(rule.couplings):
            for lorentz_index, alias in enumerate(row):
                if alias is None:
                    continue
                coupling = model.coupling(alias)
                if whole:
                    if alias not in whole:
                        raise ValueError(
                            "whole-rule coupling lacks its owned coefficient"
                        )
                    value_signature = scalar(alias, _powers(whole[alias]))
                else:
                    entry = aliases.get(alias)
                    if entry is None:
                        raise ValueError(
                            "prepared native coupling lacks its source binding"
                        )
                    if coupling.expression != entry.coupling.value.expression:
                        raise ValueError(
                            "prepared native coupling overrides its source binding"
                        )
                    value_signature = scalar(
                        entry.coupling.name, _powers(entry.coupling.value.expression)
                    )
                lorentz = model.lorentz_structure(
                    rule.lorentz_structures[lorentz_index]
                )
                slots.append(
                    (
                        color_index,
                        lorentz_index,
                        resolve_native(rule.color_structures[color_index]),
                        tuple(lorentz.spins),
                        resolve_native(lorentz.structure),
                        tuple(sorted(coupling.orders.items())),
                        value_signature,
                    )
                )
        return tuple(permutation.get(pdg, pdg) for pdg in pdgs), tuple(slots)

    def ordinary_native(model, permutation):
        return Counter(
            native_rule_signature(
                model,
                name,
                {entry.coupling.name: entry for entry in values},
                permutation,
            )
            for name, values in ordinary.items()
        )

    for model in native_models:
        if ordinary_native(model, identity) != ordinary_native(model, relabel):
            raise ValueError(
                "flavour permutation changes prepared native ordinary tensors"
            )

    def counterterm_signature(rule, permutation):
        continuation = rule.whole_rule_continuation
        return (
            rule.kind,
            tuple(permutation.get(pdg, pdg) for pdg in rule.particle_pdgs),
            rule.order,
            rule.perturbation_type,
            rule.vertex_spinor_frame,
            _alternatives(rule.alternatives, permutation),
            None
            if continuation is None
            else (
                continuation.scheme,
                _canonical(continuation.nc),
                _canonical(continuation.gauge_coupling),
                continuation.contribution,
            ),
            tuple(sorted(basis(entry) for entry in rule.entries)),
            native_rule_signature(
                plan.counterterms.native,
                rule.name,
                dict(rule.couplings),
                permutation,
                whole_aliases[rule.name],
            ),
        )

    if Counter(
        counterterm_signature(rule, identity) for rule in counterterms
    ) != Counter(counterterm_signature(rule, relabel) for rule in counterterms):
        raise ValueError(
            "flavour permutation changes retained UV/R2 tensors, loop groups "
            "or Laurent coefficients"
        )

    # Field rules already retain full Laurent values and source diagnostics. A
    # field expression with a diagnostic-bearing parameter needs a future paired
    # lowering entry point; do not activate that diagnostic unconditionally.
    parameter_values = []
    for record in raw.collections["parameters"]:
        if record["name"] not in parameter_names:
            continue
        value = resolved[record["name"], 0]
        parameter_values.append((S(f"UFO::{record['name']}"), value))

    def field_signature(rule, permutation):
        if plan.source_color is not None:
            return (
                permutation.get(rule.pdg, rule.pdg),
                rule.spin,
                rule.color,
                rule.correction,
                rule.order,
                _alternatives(rule.alternatives, permutation),
                scalar(field_names[id(rule)], _powers(rule.coefficient.expression)),
            )
        value, diagnostic = rule.coefficient.expression, rule.diagnostic
        for symbol, replacement in parameter_values:
            if (
                symbol in value.get_all_symbols()
                or symbol in diagnostic.get_all_symbols()
            ):
                if replacement.diagnostic != 0:
                    raise NotImplementedError(
                        "external-field symmetry needs paired parameter diagnostics"
                    )
                value = value.replace(symbol, replacement.value)
                diagnostic = diagnostic.replace(symbol, replacement.value)
        return (
            permutation.get(rule.pdg, rule.pdg),
            rule.spin,
            rule.color,
            rule.correction,
            rule.order,
            _alternatives(rule.alternatives, permutation),
            _canonical(value),
            _canonical(diagnostic),
        )

    if Counter(field_signature(rule, identity) for rule in fields) != Counter(
        field_signature(rule, relabel) for rule in fields
    ):
        raise ValueError("flavour permutation changes external-field counterterms")
    return FlavorSymmetry(
        plan,
        tuple(sorted((a, b) for a, b in relabel.items() if a != b)),
        len(ordinary),
        len(counterterms),
        len(fields),
    )
