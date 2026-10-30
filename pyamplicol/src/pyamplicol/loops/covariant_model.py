# SPDX-License-Identifier: 0BSD
"""Structural covariant model ingress, separate from numerical scalar binding."""

from __future__ import annotations

import json
from dataclasses import dataclass
from types import MappingProxyType, SimpleNamespace
from typing import Any

from .._internal.physics.symbols import symbols
from ..processes.model import ufo_goldstone_flag
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


# The legacy UFO object library's ``Particle.anti`` (MG5_aMC's loop_qcd_qed_sm
# and its G_mu variants): positional arguments bind ``self.goldstoneboson`` to
# the ``line`` parameter, so the constructor default drops the antiparticle's
# Goldstone flag. A fingerprint of the library code, not of any particle.
_LEGACY_ANTI_PARAMETERS = (
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
_LEGACY_ANTI_ARGUMENTS = (
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


def _lost_goldstone_flag(raw, record) -> bool:
    """Whether ``record`` is an antiparticle built by the legacy
    ``Particle.anti`` from a declared Goldstone whose flag it lost, its other
    conjugation-invariant metadata agreeing with the particle's."""

    b = record["fields"]
    source = b.get("charge_source", {})
    call = source.get("anti_constructor", {})
    if (
        source.get("kind") != "source-antiparticle-charge"
        or tuple(call.get("parameters", ())) != _LEGACY_ANTI_PARAMETERS
        or tuple(call.get("arguments", ())) != _LEGACY_ANTI_ARGUMENTS
        or ufo_goldstone_flag(b)
        or b["antiname"] == b["name"]
    ):
        return False
    try:
        partner = raw.record("particles", b["antiname"])
    except (KeyError, ValueError):
        return False
    a = partner["fields"]
    return (
        ufo_goldstone_flag(a)
        and a["antiname"] == b["name"]
        and a["pdg_code"] == -b["pdg_code"]
        and (a["spin"], a["color"]) == (b["spin"], b["color"])
        and (a["mass"]["name"], a["width"]["name"])
        == (b["mass"]["name"], b["width"]["name"])
        and _flag(a, ("Propagating", "propagating", True))
        == _flag(b, ("Propagating", "propagating", True))
        and _charge(partner) == -_charge(record)
    )


def _repair(raw, particles, compatibility):
    """The explicit source-compatibility profile ``gmu_particle_anti``: every
    antiparticle that lost its Goldstone flag to the legacy ``Particle.anti``
    (``_lost_goldstone_flag``) gets it back, one diagnostic each. The profile
    on a source whose flags the structural source repair
    (``repair_source_declarations``, rule ``GOLDSTONE_FLAG``) has already
    restored is satisfied as it is; on a source without that defect it is
    refused."""

    if compatibility is None:
        return ()
    if compatibility != "gmu_particle_anti":
        raise ValueError("unknown structural source compatibility profile")
    repaired = [
        record
        for record in raw.collections["particles"]
        if _lost_goldstone_flag(raw, record)
    ]
    if not repaired:
        if any(
            item.get("rule") == GOLDSTONE_FLAG
            for item in raw.metadata.get(SOURCE_REPAIRS_KEY, ())
        ):
            return ()
        raise ValueError(
            "source does not match the recognized Gmu anti metadata mismatch "
            "(an antiparticle of a declared Goldstone that the legacy "
            "Particle.anti built without the flag)"
        )
    diagnostics = []
    for record in repaired:
        particles[record["id"]]["goldstoneboson"] = True
        diagnostics.append(
            MappingProxyType(
                {
                    "profile": compatibility,
                    "particle_id": record["id"],
                    "particle": record["name"],
                    "field": "goldstoneboson",
                    "raw": False,
                    "derived": True,
                    "reason": _LOST_FLAG_REASON,
                }
            )
        )
    return tuple(diagnostics)


_LOST_FLAG_REASON = (
    "legacy anti positional arguments bind Goldstone flag to line; constructor "
    "defaults lose the flag"
)


# Source declarations repaired by structure (never by a particle, vertex or
# tensor name), recorded in the derived source's metadata and reported with
# the structure's diagnostics.
SOURCE_REPAIRS_KEY = "pyamplicol_source_repairs"
CONSTANT_TENSOR_ARITY = "constant-tensor-slot-arity"
GOLDSTONE_FLAG = "antiparticle-goldstone-flag"


def _slot_spins(record) -> tuple[int, ...]:
    return tuple(abs(int(spin)) for spin in record["fields"]["spins"])


def _constant_structure(record) -> Any:
    """The exact value of a slot-independent (constant) Lorentz tensor, or
    None when the structure depends on a slot."""

    from symbolica import AtomType

    value = parse_exact_ufo(record["fields"]["structure"])
    return value if value.get_type() == AtomType.Num else None


def repair_source_declarations(raw: Any, within: Any = None) -> Any:
    """``raw`` with its structural source defects repaired, or ``raw`` itself
    when it has none (the rules are idempotent). Two rules, each found by
    structure (never by a particle, vertex or tensor name):

    * ``GOLDSTONE_FLAG``: an antiparticle that the legacy UFO object
      library's ``Particle.anti`` built from a declared Goldstone boson
      without the flag (``_lost_goldstone_flag``: the library's positional
      arguments bind ``goldstoneboson`` to ``line``; MG5_aMC's
      loop_qcd_qed_sm and loop_qcd_qed_sm_Gmu) gets the flag back. Particles
      belong to every request, so ``within`` does not restrict this rule.
    * ``CONSTANT_TENSOR_ARITY``: a vertex (tree or counterterm) whose
      particles are all scalars and that references a constant Lorentz
      tensor with a different number of (scalar) slots means that constant
      on its own particles: the reference is moved to the model's unique
      tensor with the vertex's spin signature and the same exact constant.
      MadLoop 3.7.0's loop_sm declares a four-scalar vertex with the
      three-slot constant SSS1 next to the four-slot SSSS1 of the same value.
      Any other slot mismatch is left as declared: a request that reaches it
      is refused at model preparation with the listed key
      ``lorentz-slot-arity`` (``build_covariant_structure``), an unreachable
      one stays inert. ``within``, if given, restricts this rule to those
      ``(collection, vertex id)`` pairs (the vertices a request reaches).

    Each repair is recorded under ``SOURCE_REPAIRS_KEY`` in the derived
    source's metadata (the compiled model and the artifact carry it) and in
    the covariant structure's diagnostics."""

    flags = [
        record
        for record in raw.collections["particles"]
        if _lost_goldstone_flag(raw, record)
    ]
    tensors = raw.collections["lorentz_structures"]
    repairs = []
    for collection in ("vertices", "counterterm_vertices"):
        for vertex in raw.collections.get(collection, ()):
            if within is not None and (collection, vertex["id"]) not in within:
                continue
            fields = vertex["fields"]
            spins = tuple(
                abs(int(_reference(raw, ref, "particles")["fields"]["spin"]))
                for ref in fields["particles"]
            )
            if set(spins) != {1}:
                continue
            for position, ref in enumerate(fields["lorentz"]):
                old = _reference(raw, ref, "lorentz_structures")
                if _slot_spins(old) == spins or set(_slot_spins(old)) != {1}:
                    continue
                constant = _constant_structure(old)
                if constant is None:
                    continue
                candidates = [
                    record
                    for record in tensors
                    if _slot_spins(record) == spins
                    and _constant_structure(record) == constant
                ]
                if len(candidates) == 1:
                    repairs.append((collection, vertex, position, old, candidates[0]))
    if not repairs and not flags:
        return raw
    payload = raw.to_dict()
    diagnostics = list(raw.metadata.get(SOURCE_REPAIRS_KEY, ()))
    for record in flags:
        payload["collections"]["particles"][record["id"]]["fields"][
            "goldstoneboson"
        ] = True
        diagnostics.append(
            {
                "rule": GOLDSTONE_FLAG,
                "collection": "particles",
                "particle": [record["id"], record["name"]],
                "field": "goldstoneboson",
                "original": False,
                "derived": True,
                "reason": _LOST_FLAG_REASON,
            }
        )
    for collection, vertex, position, old, new in repairs:
        target = payload["collections"][collection][vertex["id"]]["fields"]
        target["lorentz"][position] = {
            "kind": "reference",
            "collection": "lorentz_structures",
            "id": new["id"],
            "name": new["name"],
        }
        diagnostics.append(
            {
                "rule": CONSTANT_TENSOR_ARITY,
                "collection": collection,
                "vertex": [vertex["id"], vertex["name"]],
                "lorentz_position": position,
                "original_lorentz": [old["id"], old["name"]],
                "derived_lorentz": [new["id"], new["name"]],
                "reason": (
                    f"{len(_slot_spins(new))} scalar particles take the model's "
                    f"{len(_slot_spins(new))}-slot tensor of the same exact "
                    f"constant, not the {len(_slot_spins(old))}-slot one"
                ),
            }
        )
    payload["metadata"][SOURCE_REPAIRS_KEY] = diagnostics
    return type(raw).from_dict(payload)


def with_source_repairs(model: Any) -> Any:
    """A compiled model whose raw NLO sidecar carries its source repairs
    (``repair_source_declarations``); the model itself when it has none."""

    from dataclasses import replace

    from pyamplicol.models.loop_import import RawLoopModel

    raw = model.raw_loop_model
    if not isinstance(raw, RawLoopModel):
        return model
    repaired = repair_source_declarations(raw)
    if repaired is raw:
        return model
    return replace(
        model, raw_loop_model=repaired, _serialized_path=None, _prepared_bundle=None
    )


def _validate_pairs(particles):
    """Particle-antiparticle pairs must be reciprocal and agree in their
    metadata (the listed refusal ``antiparticle-metadata``; the Goldstone
    flag MG5_aMC's legacy object library drops is repaired first,
    ``repair_source_declarations``)."""

    from pyamplicol.loops.v2.refusals import LoopRequestUnsupportedValue

    def refuse(message: str) -> LoopRequestUnsupportedValue:
        return LoopRequestUnsupportedValue("antiparticle-metadata", message)

    by_name = {p["name"]: p for p in particles}
    if len(by_name) != len(particles):
        raise ValueError("ambiguous particle names in structural model")
    for particle in particles:
        try:
            anti = by_name[particle["antiname"]]
        except KeyError as exc:
            raise refuse(f"missing source antiparticle of {particle['name']}") from exc
        if anti["antiname"] != particle["name"]:
            raise refuse(f"antiparticle names of {particle['name']} are not reciprocal")
        for key in ("spin", "mass", "width", "propagating", "goldstoneboson"):
            if particle[key] != anti[key]:
                raise refuse(
                    f"antiparticle metadata mismatch for {particle['name']}: {key}"
                )
        self_conjugate = particle is anti
        if not self_conjugate and particle["pdg_code"] != -anti["pdg_code"]:
            raise refuse(f"antiparticle PDG signs of {particle['name']} differ")
        color = particle["color"]
        if anti["color"] != (color if color in (1, 8) else -color):
            raise refuse(
                f"antiparticle colour representations of {particle['name']} differ"
            )
        for key in ("charge", "ghost_number", "lepton_number"):
            left, right = (
                parse_exact_ufo(str(particle[key])),
                parse_exact_ufo(str(anti[key])),
            )
            if left + right != 0:
                raise refuse(
                    f"antiparticle metadata mismatch for {particle['name']}: {key}"
                )
        for key in ("y_charge", "y_charge_right"):
            if any(
                value is not None and parse_exact_ufo(str(value)) != 0
                for value in (particle[key], anti[key])
            ):
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
    from symbolica.community import hepkit as hep
    from ufo_model_loader.model import Propagator, SerializableParticle
    from ufo_model_loader.symbolica_processing import wrap_indices

    if request_scope is not None:
        request_scope.require_raw(raw)
    if 1 not in raw.metadata.get("gauge", ()):
        from pyamplicol.loops.v2.refusals import LoopRequestUnsupportedValue

        raise LoopRequestUnsupportedValue(
            "feynman-gauge",
            "the source model does not declare Feynman gauge (gauge = "
            f"{list(raw.metadata.get('gauge', ()))})",
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
                expression = symbols.qualified(
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
            from .v2.refusals import LoopRequestUnsupported

            raise LoopRequestUnsupported(
                "majorana-fermion",
                f"the model declares the self-conjugate fermion {fields['name']!r}",
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
            goldstoneboson=ufo_goldstone_flag(fields),
        )
        if not particle["propagating"]:
            raise NotImplementedError(
                "auxiliary nonpropagating species require certified propagators"
            )
        payload["particles"].append(vars(SerializableParticle(**particle)))
    diagnostics = _repair(raw, payload["particles"], compatibility)
    diagnostics += tuple(
        MappingProxyType(dict(item))
        for item in raw.metadata.get(SOURCE_REPAIRS_KEY, ())
    )
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
    zero_parameter = any(p["name"] == "ZERO" for p in payload["parameters"])
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
        denominator = propagator.denominator
        if native.is_massless and native.mass != 0:
            # A mass alias proved structurally massless above. The Community
            # HepKit of the integration kernel (9cdc2949) gives it mass 0 and
            # exports it as ZERO; the published Community 3.0.2 distribution
            # keeps the source symbol, which the v2 families would read as a
            # derived mass of value zero (refused, `_model_masses`). Write ZERO,
            # so that the native model is the same under both.
            if zero_parameter:
                particle["mass"] = "ZERO"
            else:
                # No ZERO parameter to name: native family extraction retains
                # the source symbol; keep that same symbolic denominator (the
                # exact source binding supplies its zero).
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
            from .v2.refusals import LoopRequestUnsupported

            raise LoopRequestUnsupported(
                "multi-fermion-contact",
                f"source vertex {record['name']!r} (ID {record['id']}) joins "
                f"{spins.count(2)} fermions",
            )
        lorentz = []
        for tensor_index in tensor_indices:
            ref = fields["lorentz"][tensor_index]
            source = _reference(raw, ref, "lorentz_structures")
            key = (source["id"], spins)
            if key not in lorentz_names:
                if tuple(map(abs, source["fields"]["spins"])) != tuple(map(abs, spins)):
                    from .v2.refusals import LoopRequestUnsupportedValue

                    # Constant scalar tensors were repaired by the source
                    # rule (``repair_source_declarations``); this one is not
                    # (a ``ValueError`` as the untyped refusal it replaces).
                    raise LoopRequestUnsupportedValue(
                        "lorentz-slot-arity",
                        f"source vertex {record['name']!r} (ID {record['id']}) "
                        f"uses Lorentz tensor {source['name']!r} with spin slots "
                        f"{tuple(source['fields']['spins'])}, but its particles "
                        f"require {spins}",
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
