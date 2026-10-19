# SPDX-License-Identifier: 0BSD
"""Structural covariant model ingress, separate from numerical scalar binding."""

from __future__ import annotations

import json
from dataclasses import dataclass
from types import MappingProxyType, SimpleNamespace
from typing import Any

from .model_color import prepare_color_expression
from .model_expressions import epsilon_symbol, parse_exact_ufo
from .scalar_bindings import prepare_scalar_bindings
from .scalar_frontend import ScalarProgram, lower_scalar_program


@dataclass(frozen=True, slots=True)
class StructuralCovariantModel:
    raw: Any
    scalars: ScalarProgram
    native: Any
    required_bindings: tuple[str, ...]
    diagnostics: tuple[Any, ...]
    fixed_parameters: Any
    request_scope: Any = None
    perturbative_species_orders: Any = None

    def scalar_bindings(self, outputs):
        """Prepare numerical bindings with this structure's exact fixed values."""
        return prepare_scalar_bindings(
            self.raw,
            outputs,
            program=self.scalars,
            fixed_parameters=self.fixed_parameters,
        )


def _flag(fields, name):
    return bool(fields.get(name[0], fields.get(name[1], name[2])))


def _reference(raw, ref, collection):
    if ref.get("kind") != "reference" or ref.get("collection") != collection:
        raise ValueError(f"expected a source {collection} reference")
    index = ref.get("id")
    if type(index) is not int or not 0 <= index < len(raw.collections[collection]):
        raise ValueError("invalid source reference index")
    record = raw.collections[collection][index]
    if record["name"] != ref["name"]:
        raise ValueError("source record identity differs from reference")
    return record


def _charge(record):
    from symbolica import AtomType

    fields = record["fields"]
    if not fields.get("charge_source", {}).get("exact", False):
        raise ValueError(f"particle {record['name']!r} lacks exact charge source")
    value = fields["charge_exact"]
    if value is None or value["kind"] != "expression":
        raise ValueError("particle charge requires one exact scalar expression")
    charge = parse_exact_ufo(value["expression"])
    if charge.get_type() != AtomType.Num or charge != charge.conj():
        raise NotImplementedError(
            "structural quantum charges must be rational real constants"
        )
    # The existing dependency serializer accepts exact rational strings.
    return charge


def _repair(raw, particles, compatibility):
    if compatibility is None:
        return ()
    if compatibility != "gmu_particle_anti":
        raise ValueError("unknown structural source compatibility profile")
    positive = raw.record("particles", "G+")
    negative = raw.record("particles", "G-")
    a, b = positive["fields"], negative["fields"]
    source = b.get("charge_source", {})
    call = source.get("anti_constructor", {})
    expected_parameters = (
        "pdg_code",
        "name",
        "antiname",
        "spin",
        "color",
        "mass",
        "width",
        "texname",
        "antitexname",
        "charge",
        "loop_particles",
        "counterterm",
        "line",
        "propagating",
        "goldstoneboson",
    )
    expected_arguments = (
        "-self.pdg_code",
        "self.antiname",
        "self.name",
        "self.spin",
        "newcolor",
        "self.mass",
        "self.width",
        "self.antitexname",
        "self.texname",
        "-self.charge",
        "self.line",
        "self.propagating",
        "self.goldstoneboson",
    )
    if (
        tuple(raw.metadata.get("gauge", ())) != (1,)
        or source.get("kind") != "source-antiparticle-charge"
        or tuple(call.get("parameters", ())) != expected_parameters
        or tuple(call.get("arguments", ())) != expected_arguments
        or (a["name"], a["antiname"], b["name"], b["antiname"])
        != ("G+", "G-", "G-", "G+")
        or (a["pdg_code"], b["pdg_code"]) != (251, -251)
        or (a["spin"], b["spin"], a["color"], b["color"]) != (1, 1, 1, 1)
        or (
            a["mass"]["name"],
            b["mass"]["name"],
            a["width"]["name"],
            b["width"]["name"],
        )
        != ("MW", "MW", "WW", "WW")
        or not _flag(a, ("GoldstoneBoson", "goldstoneboson", False))
        or _flag(b, ("GoldstoneBoson", "goldstoneboson", False))
        or not _flag(a, ("Propagating", "propagating", True))
        or not _flag(b, ("Propagating", "propagating", True))
        or _charge(positive) != 1
        or _charge(negative) != -1
    ):
        raise ValueError(
            "source does not match the recognized Gmu anti metadata mismatch"
        )
    particles[negative["id"]]["goldstoneboson"] = True
    return (
        MappingProxyType(
            {
                "profile": compatibility,
                "particle_id": negative["id"],
                "particle": "G-",
                "field": "goldstoneboson",
                "raw": False,
                "derived": True,
                "reason": (
                    "legacy anti positional arguments bind Goldstone flag to line; "
                    "constructor defaults lose the flag"
                ),
            }
        ),
    )


def _validate_pairs(particles):
    by_name = {p["name"]: p for p in particles}
    if len(by_name) != len(particles):
        raise ValueError("ambiguous particle names in structural model")
    for particle in particles:
        try:
            anti = by_name[particle["antiname"]]
        except KeyError as exc:
            raise ValueError("missing source antiparticle") from exc
        if anti["antiname"] != particle["name"]:
            raise ValueError("antiparticle names are not reciprocal")
        for key in ("spin", "mass", "width", "propagating", "goldstoneboson"):
            if particle[key] != anti[key]:
                raise ValueError(
                    f"antiparticle metadata mismatch for {particle['name']}: {key}"
                )
        self_conjugate = particle is anti
        if not self_conjugate and particle["pdg_code"] != -anti["pdg_code"]:
            raise ValueError("antiparticle PDG signs differ")
        color = particle["color"]
        if anti["color"] != (color if color in (1, 8) else -color):
            raise ValueError("antiparticle colour representations differ")
        for key in ("charge", "ghost_number", "lepton_number"):
            left, right = (
                parse_exact_ufo(str(particle[key])),
                parse_exact_ufo(str(anti[key])),
            )
            if left + right != 0:
                raise ValueError(
                    f"antiparticle metadata mismatch for {particle['name']}: {key}"
                )
        for key in ("y_charge", "y_charge_right"):
            if particle[key] is not None or anti[key] is not None:
                raise NotImplementedError(
                    "chiral hypercharge conjugation requires a certified "
                    "source convention"
                )


def build_covariant_structure(
    raw: Any,
    *,
    scalars: ScalarProgram | None = None,
    covariant_multiplets: Any = None,
    compatibility: str | None = None,
    fixed_parameters: Any = None,
    request_scope: Any = None,
) -> StructuralCovariantModel:
    """Preserve raw species/tree vertices through existing public ModelJSON APIs.

    External parameter defaults never enter numerical caches. Source-resolved
    mass expressions preserve the dependency's structural-zero classification. This
    object is for graph/kernel construction; use its scalar program for numeric
    binding, including diagnostics. Custom propagators and unspecified gauges
    require separate certification and are rejected by this bounded adapter.
    """
    from symbolica import S
    from symbolica.community import hepkit as hep
    from ufo_model_loader.model import Propagator, SerializableParticle
    from ufo_model_loader.symbolica_processing import wrap_indices

    if request_scope is not None:
        request_scope.require_raw(raw)
    if 1 not in raw.metadata.get("gauge", ()):
        raise ValueError("source model does not declare Feynman gauge")
    if raw.collections["propagators"]:
        raise NotImplementedError(
            "custom source propagators require covariant certification"
        )
    restriction = raw.metadata.get("restriction", {})
    if restriction.get("card") is not None and not restriction.get("applied", False):
        raise NotImplementedError(
            "unapplied source restrictions require explicit exact lowering"
        )
    if restriction.get("applied", False):
        from .model_restrictions import restriction_fixed_parameters

        fixed_parameters = restriction_fixed_parameters(raw, fixed_parameters)
    program = scalars if scalars is not None else lower_scalar_program(raw)
    if program.library.raw is not raw:
        raise ValueError("scalar program belongs to another raw model")
    mass_names = tuple(
        dict.fromkeys(
            _reference(raw, record["fields"]["mass"], "parameters")["name"]
            for record in raw.collections["particles"]
        )
    )
    # Resolve mass aliases with adjustable externals still independent. Explicit
    # generation constants are shared with subsequent numerical scalar bindings.
    fixed_names = tuple(fixed_parameters) if fixed_parameters is not None else ()
    coupling_names = (
        tuple(
            dict.fromkeys(
                _reference(raw, ref, "couplings")["name"]
                for record in raw.collections["vertices"]
                for _key, ref in record["fields"]["couplings"]["entries"]
            )
        )
        if fixed_names
        else ()
    )
    names = tuple(dict.fromkeys((*mass_names, *fixed_names, *coupling_names)))
    if names:
        specialization = prepare_scalar_bindings(
            raw,
            [(name, 0) for name in names],
            program=program,
            fixed_parameters=fixed_parameters,
        )
        resolved = dict(zip(names, specialization.expressions, strict=True))
        fixed = specialization.fixed_parameters
    else:
        resolved, fixed = {}, MappingProxyType({})
    payload = {
        key: []
        for key in (
            "orders",
            "parameters",
            "particles",
            "propagators",
            "couplings",
            "lorentz_structures",
            "vertex_rules",
        )
    }
    payload.update(name=raw.name + "_covariant_structure", restriction=None)
    required = []
    for record in raw.collections["coupling_orders"]:
        fields = record["fields"]
        payload["orders"].append(
            {key: fields[key] for key in ("name", "expansion_order", "hierarchy")}
        )
    for record in raw.collections["parameters"]:
        fields = record["fields"]
        definition = program.definitions[("parameters", record["id"])]
        expression = definition.value.expression
        if epsilon_symbol() in expression.get_all_symbols():
            raise NotImplementedError(
                "Laurent parameters cannot define structural species"
            )
        if record["name"] == "ZERO" and expression != 0:
            raise ValueError("reserved ZERO parameter is not exactly zero")
        if fields["nature"] == "external":
            if record["name"] in fixed:
                expression = resolved[record["name"]].value
            else:
                required.append(record["name"])
                expression = S(
                    f"pyamplicol_structural::required_parameter_{record['id']}"
                )
        elif fields["nature"] != "internal":
            raise ValueError("unknown source parameter nature")
        payload["parameters"].append(
            dict(
                name=record["name"],
                nature="internal",
                parameter_type=fields["type"],
                expression=expression.to_canonical_string(),
                value=None,
                lhablock=None,
                lhacode=None,
            )
        )
    for record in raw.collections["particles"]:
        fields = record["fields"]
        if fields["spin"] not in (-1, 1, 2, 3):
            raise NotImplementedError("unsupported covariant particle spin")
        if fields["spin"] == 2 and fields["name"] == fields["antiname"]:
            raise NotImplementedError(
                "Majorana species are outside the current loop domain"
            )
        if fields.get("propagator") is not None:
            raise NotImplementedError("custom source propagator metadata")
        particle = {
            key: fields[key]
            for key in (
                "pdg_code",
                "name",
                "antiname",
                "spin",
                "color",
                "texname",
                "antitexname",
            )
        }
        particle.update(
            mass=_reference(raw, fields["mass"], "parameters")["name"],
            width=_reference(raw, fields["width"], "parameters")["name"],
            charge=_charge(record).to_canonical_string(),
            ghost_number=fields.get("GhostNumber", 0),
            lepton_number=fields.get("LeptonNumber", 0),
            y_charge=fields.get("Y"),
            y_charge_right=fields.get("YRight"),
            propagating=_flag(fields, ("Propagating", "propagating", True)),
            goldstoneboson=_flag(fields, ("GoldstoneBoson", "goldstoneboson", False)),
        )
        if not particle["propagating"]:
            raise NotImplementedError(
                "auxiliary nonpropagating species require certified propagators"
            )
        payload["particles"].append(vars(SerializableParticle(**particle)))
    diagnostics = _repair(raw, payload["particles"], compatibility)
    zero_couplings = tuple(
        name
        for name in coupling_names
        if resolved[name].value == 0 and resolved[name].diagnostic == 0
    )
    if zero_couplings:
        diagnostics += (
            MappingProxyType(
                {
                    "kind": "fixed-parameter-zero-couplings",
                    "couplings": zero_couplings,
                }
            ),
        )
    species_orders = None
    if request_scope is not None:
        from .graphs import _species_orders_from_rules
        from .orders import canonical_order

        complete_rules = []
        for record in raw.collections["vertices"]:
            orders = {
                canonical_order(tuple(coupling["fields"]["order"]["entries"]))
                for _key, reference in record["fields"]["couplings"]["entries"]
                for coupling in (_reference(raw, reference, "couplings"),)
                if coupling["name"] not in zero_couplings
            }
            if not orders:
                continue
            if len(orders) != 1:
                raise NotImplementedError(
                    "source ordinary vertices require homogeneous coupling orders"
                )
            pdgs = tuple(
                abs(_reference(raw, reference, "particles")["fields"]["pdg_code"])
                for reference in record["fields"]["particles"]
            )
            complete_rules.append((pdgs, next(iter(orders))))
        species_orders = MappingProxyType(_species_orders_from_rules(complete_rules))
    _validate_pairs(payload["particles"])
    # The currently installed Community classifier recognizes a literal exact
    # zero, but does not resolve source parameter aliases. Use the same scalar
    # binding preparation as numerical consumers, leaving externals independent.
    for parameter in payload["parameters"]:
        expression = (
            resolved.get(parameter["name"]) if parameter["name"] in mass_names else None
        )
        if expression is None:
            continue
        if expression.value == 0 and expression.diagnostic != 0:
            raise NotImplementedError(
                "conditional mass validity cannot certify a structural zero"
            )
        parameter["expression"] = expression.value.to_canonical_string()
    # Classification itself belongs to the native model. Numeric defaults have
    # not entered either the resolved expressions or the structural payload.
    classification = hep.Model.from_json(json.dumps(payload))
    for particle in payload["particles"]:
        native = classification.particle(particle["name"])
        proxy = SimpleNamespace(**particle)
        proxy.mass = SimpleNamespace(name=particle["mass"])
        proxy.is_massive = lambda native=native: not native.is_massless
        propagator = Propagator.from_particle(proxy, "Feynman")
        if particle["spin"] == 3:
            # Existing public custom-propagator API. This is the numerator of
            # native Particle.spin_sum(..., covariant=True), including masses.
            propagator = Propagator(
                propagator.name,
                proxy,
                "-complex(0,1)*Metric(1,2)",
                str(propagator.denominator),
            )
        particle["propagator"] = propagator.name
        # Native family extraction retains the source mass symbol (apart from
        # ZERO), including aliases proved structurally massless above. Keep that
        # same symbolic denominator; the exact source binding supplies its zero.
        denominator = propagator.denominator
        if native.is_massless and native.mass != 0:
            denominator_proxy = SimpleNamespace(**vars(proxy))
            denominator_proxy.is_massive = lambda: True
            denominator = Propagator.from_particle(
                denominator_proxy, "Feynman"
            ).denominator
        payload["propagators"].append(
            dict(
                name=propagator.name,
                particle=particle["name"],
                numerator=wrap_indices(propagator.numerator).to_canonical_string(),
                denominator=wrap_indices(denominator).to_canonical_string(),
            )
        )
    used_couplings, lorentz_names = set(), {}
    for record in raw.collections["vertices"]:
        fields = record["fields"]
        active = []
        for key, reference in fields["couplings"]["entries"]:
            color, tensor = key["items"]
            if (
                request_scope is not None
                and ("vertices", record["id"], color, tensor, None)
                not in request_scope.admitted_keys
            ):
                continue
            coupling = _reference(raw, reference, "couplings")
            if coupling["name"] not in zero_couplings:
                active.append((color, tensor, coupling))
        if not active:
            continue
        color_indices = sorted({color for color, _tensor, _coupling in active})
        tensor_indices = sorted({tensor for _color, tensor, _coupling in active})
        source_particles = [
            _reference(raw, ref, "particles") for ref in fields["particles"]
        ]
        spins = tuple(p["fields"]["spin"] for p in source_particles)
        if spins.count(2) > 2:
            raise NotImplementedError(
                "four-fermion vertices are outside the current loop domain"
            )
        lorentz = []
        for tensor_index in tensor_indices:
            ref = fields["lorentz"][tensor_index]
            source = _reference(raw, ref, "lorentz_structures")
            key = (source["id"], spins)
            if key not in lorentz_names:
                if tuple(map(abs, source["fields"]["spins"])) != tuple(map(abs, spins)):
                    raise ValueError(
                        f"source vertex {record['name']!r} (ID {record['id']}) "
                        f"uses Lorentz tensor {source['name']!r} with spin slots "
                        f"{tuple(source['fields']['spins'])}, but its particles "
                        f"require {spins}; prepare a corrected model declaration "
                        "before loop generation"
                    )
                name = f"source_lorentz_{source['id']}_{len(lorentz_names)}"
                lorentz_names[key] = name
                payload["lorentz_structures"].append(
                    dict(
                        name=name,
                        spins=spins,
                        structure=wrap_indices(
                            parse_exact_ufo(source["fields"]["structure"])
                        ).to_canonical_string(),
                    )
                )
            lorentz.append(lorentz_names[key])
        couplings = [[None] * len(lorentz) for _ in color_indices]
        for source_color, source_tensor, coupling in active:
            color = color_indices.index(source_color)
            tensor = tensor_indices.index(source_tensor)
            if couplings[color][tensor] is not None:
                raise ValueError("duplicate source colour/Lorentz coupling slot")
            couplings[color][tensor] = coupling["name"]
            used_couplings.add(coupling["id"])
        if all(coupling is None for row in couplings for coupling in row):
            continue
        payload["vertex_rules"].append(
            dict(
                name=record["name"],
                particles=[p["name"] for p in source_particles],
                color_structures=[
                    prepare_color_expression(
                        source, [p["fields"]["color"] for p in source_particles]
                    ).to_canonical_string()
                    for source in (fields["color"][index] for index in color_indices)
                ],
                lorentz_structures=lorentz,
                couplings=couplings,
            )
        )
    for index in sorted(used_couplings):
        record = raw.collections["couplings"][index]
        expression = program.definitions[("couplings", index)].value.expression
        if epsilon_symbol() in expression.get_all_symbols():
            raise NotImplementedError(
                "ordinary structural vertices cannot contain Laurent poles"
            )
        payload["couplings"].append(
            dict(
                name=record["name"],
                expression=expression.to_canonical_string(),
                orders=[list(pair) for pair in record["fields"]["order"]["entries"]],
                value=None,
            )
        )
    if covariant_multiplets is not None:
        payload["covariant_cut_multiplets"] = {
            str(pdg): list(partners) for pdg, partners in covariant_multiplets.items()
        }
    native = hep.Model.from_json(json.dumps(payload))
    return StructuralCovariantModel(
        raw,
        program,
        native,
        tuple(required),
        diagnostics,
        fixed,
        request_scope,
        species_orders,
    )
