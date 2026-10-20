# SPDX-License-Identifier: 0BSD
"""Native counterterm trees matched to the currents of admitted loop graphs.

This is a graph and exact-source boundary, not a tensor-convention or complete
renormalization certificate. A prepared model must be used for *both* ordinary
graphs and counterterm trees: interaction IDs and slots belong to that model.
HepKit owns generation, connected components, excision and tree signs; public
Symbolica.Graph owns rooted-current canonicalization. No loop symmetry/sign is
copied to a counterterm tree.

Matching follows MadLoop 3.7 ``set_LoopCT_vertices``. R2/UVmass match the actual
loop species; UVloop uses its alternatives only for forbidden-species filtering.
Empty alternatives are wildcards. Alternative indices preserve multiplicities.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any

from .graphs import (
    GraphIR,
    LoopProcessIR,
    _propagator_denominators,
    _rule_orders,
    _snapshot,
    _species_orders,
    _summed_orders,
)
from .model import KernelBasisEntry, LoopModelIR
from .model_color import prepare_color_expression
from .model_correspondence import (
    CountertermBasisCorrespondence,
    CountertermPrimitiveValidation,
    CountertermTensorRepresentative,
    OrdinaryBasisCatalogue,
    OrdinaryBasisRepresentative,
    validate_counterterm_primitives,
)
from .model_expressions import parse_exact_ufo
from .qcd_r2_color import QCDR2Continuation

_TAG = "PYAMPLICOL_CT_INSERTION"
_PREFIX = "PYAMPLICOL_CT_"


@dataclass(frozen=True, slots=True)
class CountertermRule:
    name: str
    kind: str
    entries: tuple[KernelBasisEntry, ...]
    particle_pdgs: tuple[int, ...]
    order: tuple[tuple[str, int], ...]
    alternatives: tuple[tuple[int, ...], ...]
    perturbation_type: str | None
    # Native coupling name -> exact model basis entry. Full Laurent expressions
    # stay in the model; consumers select epsilon sectors after substitution.
    couplings: tuple[tuple[str, KernelBasisEntry], ...]
    coupling_symbols: tuple[Any, ...] = field(default=(), repr=False)
    scalar_program: Any = field(default=None, repr=False)
    basis_correspondences: tuple[CountertermBasisCorrespondence | None, ...] = ()
    primitive_validations: tuple[CountertermPrimitiveValidation | None, ...] = ()
    tensor_representatives: tuple[CountertermTensorRepresentative | None, ...] = ()
    vertex_spinor_frame: str | None = None
    # A checked whole group is a separate native representation. Its original
    # entries, aliases and local evidence remain available for source diagnostics
    # and perturbative matching; no entrywise correspondence is implied.
    whole_rule_continuation: QCDR2Continuation | None = None
    native_coefficients: tuple[tuple[str, Any], ...] = ()
    native_coupling_symbols: tuple[Any, ...] = field(default=(), repr=False)

    @property
    def active_coupling_symbols(self) -> tuple[Any, ...]:
        return (
            self.native_coupling_symbols
            if self.whole_rule_continuation is not None
            else self.coupling_symbols
        )

    @property
    def has_ordinary_basis_correspondence(self) -> bool:
        return bool(self.basis_correspondences) and all(
            item is not None for item in self.basis_correspondences
        )

    @property
    def has_validated_tensor_basis(self) -> bool:
        """Every basis has explicit ordinary or independent local conformance."""
        if self.whole_rule_continuation is not None:
            return True
        return (
            len(self.entries)
            == len(self.basis_correspondences)
            == len(self.primitive_validations)
            and bool(self.entries)
            and all(
                ordinary is not None or primitive is not None
                for ordinary, primitive in zip(
                    self.basis_correspondences, self.primitive_validations, strict=True
                )
            )
        )


@dataclass(frozen=True, slots=True)
class PreparedCountertermModel:
    ordinary_model: Any = field(repr=False)
    native: Any = field(repr=False)
    source: LoopModelIR = field(repr=False)
    rules: Mapping[str, CountertermRule]
    unavailable_entries: tuple[tuple[Any, ...], ...]
    tag: str = _TAG
    vertex_spinor_frame: str | None = None
    tensor_convention_validated: bool = field(default=False, init=False)
    source_entries_complete: bool = False

    def process(self, incoming, outgoing, *, counterterms=False, **filters):
        """Construct a process sharing the prepared model and explicit rules.

        ``vertex_allow`` and ``vertex_veto`` restrict ordinary rules. Candidate
        counterterm rules are appended only when ``counterterms=True``. Particle
        restrictions are forwarded unchanged to HepKit.
        """
        ordinary = {rule.name for rule in self.ordinary_model.vertex_rules}
        allow = filters.pop("vertex_allow", None)
        veto = filters.pop("vertex_veto", ()) or ()
        names = (
            ordinary
            if allow is None
            else {item if isinstance(item, str) else item.name for item in allow}
        )
        names = names - {item if isinstance(item, str) else item.name for item in veto}
        if not names <= ordinary:
            raise ValueError("counterterm process filters must name ordinary rules")
        if counterterms:
            names |= self.rules.keys()
        return self.native.process(
            incoming, outgoing, vertex_allow=sorted(names), **filters
        )


def _effective_kind(entry: KernelBasisEntry) -> str:
    # MadLoop import_ufo.py:1571-1583 resolves the legacy type 'UV'. Preserve
    # its exact rule (identical particle names, not merely antiparticle pairs).
    if entry.kind == "UV":
        return (
            "UVmass"
            if len(entry.particle_names) == 2
            and entry.particle_names[0] == entry.particle_names[1]
            else "UVloop"
        )
    if entry.kind not in {"R2", "UVmass", "UVloop", "UVtree"}:
        raise NotImplementedError(f"unsupported CT insertion type {entry.kind!r}")
    return entry.kind


def prepare_counterterm_model(
    source: LoopModelIR,
    ordinary_model: Any,
    *,
    entries: Iterable[KernelBasisEntry] | None = None,
    ordinary_basis_representatives: Iterable[OrdinaryBasisRepresentative] = (),
    primitive_validations: Iterable[CountertermPrimitiveValidation] = (),
    tensor_representatives: Iterable[CountertermTensorRepresentative] = (),
    whole_rule_continuations: Iterable[QCDR2Continuation] = (),
    vertex_spinor_frame: str | None = None,
) -> PreparedCountertermModel:
    """Append homogeneous, tagged CT rules through native model interchange.

    No parameter is evaluated and no source tensor is silently transposed. An
    explicit tensor representative preserves its exact source evidence and
    replaces only that entry's native Lorentz tensor. The
    checked whole-rule continuation instead supplies the complete native tensor
    for one source loop group. Original entries and their aliases stay intact;
    replacement coefficients are exposed separately. This certifies local tensor
    conformance, not the physical scheme of the complete process. The
    native model retains exact Laurent coupling expressions. Missing external
    species are recorded explicitly for a deliberately restricted tree model;
    this does not certify the omitted sector or the model's physical scheme.
    """
    from symbolica.community import hepkit as hep
    from ufo_model_loader.symbolica_processing import wrap_indices

    if vertex_spinor_frame not in (None, "ufo"):
        raise ValueError("stored source counterterm frame must be ufo")
    ordinary_basis_representatives = tuple(ordinary_basis_representatives)
    tensor_representatives = tuple(tensor_representatives)
    continuations = {}
    for continuation in whole_rule_continuations:
        if not isinstance(continuation, QCDR2Continuation):
            raise TypeError("whole-rule continuation must be a checked QCD R2 rule")
        if not continuation.source_entries or not continuation.terms:
            raise ValueError("whole-rule continuation must contain its complete group")
        if continuation.key in continuations:
            raise ValueError("duplicate whole-rule continuation")
        original = tuple(
            entry
            for entry in source.kernels
            if (entry.collection, entry.vertex_id, entry.loop_group_index, entry.order)
            == continuation.key
        )
        if len(original) != len(continuation.source_entries) or {
            id(entry) for entry in original
        } != {id(entry) for entry in continuation.source_entries}:
            raise ValueError(
                "whole-rule continuation requires its complete source group"
            )
        continuations[continuation.key] = continuation
    if vertex_spinor_frame == "ufo" and (
        ordinary_basis_representatives or tensor_representatives
    ):
        raise ValueError("raw source counterterms cannot mix native representatives")

    payload = json.loads(ordinary_model.to_json())
    representatives = OrdinaryBasisCatalogue(
        source, ordinary_model, ordinary_basis_representatives
    )
    tensor_representatives = tuple(tensor_representatives)
    primitive_evidence = validate_counterterm_primitives(
        source,
        (
            *primitive_validations,
            *(item.source_validation for item in tensor_representatives),
        ),
    )
    tensor_evidence = {
        item.source_validation.source_key: item for item in tensor_representatives
    }
    for collection in ("orders", "couplings", "lorentz_structures", "vertex_rules"):
        if any(item["name"].startswith(_PREFIX) for item in payload[collection]):
            raise ValueError("ordinary model uses the reserved counterterm namespace")
    payload["orders"].append({"name": _TAG, "expansion_order": 1, "hierarchy": 0})
    species = {item["pdg_code"]: item for item in payload["particles"]}
    groups = defaultdict(list)
    unavailable = []
    seen = set()
    selected = (
        (entry for entry in source.kernels if entry.kind != "tree")
        if entries is None
        else entries
    )
    for entry in selected:
        if not any(entry is existing for existing in source.kernels):
            raise ValueError("counterterm entry does not belong to the source model")
        if entry.key in seen:
            raise ValueError("duplicate counterterm basis entry")
        seen.add(entry.key)
        _effective_kind(entry)
        particles = [
            source.raw.collections["particles"][i]["fields"] for i in entry.particle_ids
        ]
        if any(p["pdg_code"] not in species for p in particles):
            unavailable.append(entry.key)
            continue
        if len(particles) < 2:
            raise NotImplementedError("one-point counterterms require a tadpole policy")
        for particle in particles:
            native = species[particle["pdg_code"]]
            if any(native[key] != particle[key] for key in ("spin", "color")):
                raise ValueError(
                    "source/native counterterm species representations differ"
                )
        groups[entry.vertex_id, entry.loop_group_index, entry.order].append(entry)

    rules = {}
    used_continuations = set()
    for number, members in enumerate(groups.values()):
        first = members[0]
        continuation_key = (
            first.collection,
            first.vertex_id,
            first.loop_group_index,
            first.order,
        )
        continuation = continuations.get(continuation_key)
        if continuation is not None:
            if len(members) != len(continuation.source_entries) or any(
                not any(entry is item for item in members)
                for entry in continuation.source_entries
            ):
                raise ValueError(
                    "whole-rule continuation cannot replace a partial group"
                )
            used_continuations.add(continuation_key)
        name = f"{_PREFIX}V{number}"
        colors = sorted({entry.color_index for entry in members})
        lorentz = sorted({entry.lorentz_index for entry in members})
        particle_pdgs = tuple(
            source.raw.collections["particles"][i]["fields"]["pdg_code"]
            for i in first.particle_ids
        )
        matrix = [[None for _ in lorentz] for _ in colors]
        aliases = []
        native_coefficients = []
        correspondences = tuple(representatives.counterterm(entry) for entry in members)
        local_colors, local_lorentz = {}, {}
        for entry, correspondence in zip(members, correspondences, strict=True):
            tensor_representative = tensor_evidence.get(entry.key)
            if continuation is not None and (
                correspondence is not None or tensor_representative is not None
            ):
                raise ValueError(
                    "whole-rule continuation conflicts with entry representatives"
                )
            if correspondence is not None and tensor_representative is not None:
                raise ValueError(
                    "counterterm has conflicting ordinary and tensor representatives"
                )
            source_color = prepare_color_expression(
                entry.color, [species[pdg]["color"] for pdg in particle_pdgs]
            ).to_canonical_string()
            color = correspondence.color if correspondence else source_color
            lorentz_value = (
                correspondence.lorentz
                if correspondence
                else wrap_indices(
                    parse_exact_ufo(
                        tensor_representative.native_lorentz
                        if tensor_representative
                        else entry.lorentz
                    )
                ).to_canonical_string()
            )
            for mapping, index, value in (
                (local_colors, entry.color_index, color),
                (local_lorentz, entry.lorentz_index, lorentz_value),
            ):
                if index in mapping and mapping[index] != value:
                    raise ValueError(
                        "inconsistent ordinary representatives for a shared "
                        "CT component"
                    )
                mapping[index] = value
        for index, entry in enumerate(members):
            alias = f"{name}_C{index}"
            matrix[colors.index(entry.color_index)][
                lorentz.index(entry.lorentz_index)
            ] = alias
            payload["couplings"].append(
                {
                    "name": alias,
                    # Structural metadata only: graph tensors retain the alias.
                    # Its numerical value and diagnostic are bound together from
                    # the source-owned rule in the physical CT consumer.
                    "expression": entry.coupling.value.expression.to_canonical_string(),
                    "orders": [list(order) for order in entry.order] + [[_TAG, 1]],
                    "value": None,
                }
            )
            aliases.append((alias, entry))
        lorentz_names = []
        for index in lorentz:
            entry = next(e for e in members if e.lorentz_index == index)
            tensor_name = f"{name}_L{index}"
            spins = [species[pdg]["spin"] for pdg in particle_pdgs]
            declared = source.raw.collections["lorentz_structures"][entry.lorentz_id][
                "fields"
            ]["spins"]
            if tuple(map(abs, spins)) != tuple(map(abs, declared)):
                raise NotImplementedError(
                    "inconsistent counterterm Lorentz spin signature"
                )
            payload["lorentz_structures"].append(
                {
                    "name": tensor_name,
                    "spins": spins,
                    "structure": local_lorentz[index],
                }
            )
            lorentz_names.append(tensor_name)
        if continuation is not None:
            # These local four-vector kernels have no spinor orientation. Their
            # declared ports are relabelled through HepKit's existing UFO index
            # boundary; the dependency owns all tensor algebra and localization.
            if any(species[pdg]["spin"] != 3 for pdg in particle_pdgs):
                raise ValueError("QCD R2 whole-rule continuation requires vector ports")
            local_colors = {}
            lorentz_names = []
            count = len(continuation.terms)
            matrix = [[None for _ in range(count)] for _ in range(count)]
            colors = list(range(count))
            for index, term in enumerate(continuation.terms):
                if term.kernel.particle_names != first.particle_names or len(
                    term.kernel.leg_half_edges
                ) != len(particle_pdgs):
                    raise ValueError("whole-rule continuation ordered ports differ")
                color = term.kernel.color.to_expression()
                lorentz_value = term.kernel.lorentz.to_expression()
                for leg, half in enumerate(term.kernel.leg_half_edges, 1):
                    port = hep.Symbols.half_edge()(half, 1)
                    local = hep.Symbols.ufo_index()(1, leg)
                    color = color.replace(port, local)
                    lorentz_value = lorentz_value.replace(port, local)
                alias = f"{name}_R{index}"
                native_coefficients.append((alias, term.coefficient))
                payload["couplings"].append(
                    {
                        "name": alias,
                        "expression": term.coefficient.to_canonical_string(),
                        "orders": [list(order) for order in first.order] + [[_TAG, 1]],
                        "value": None,
                    }
                )
                tensor_name = f"{name}_RL{index}"
                payload["lorentz_structures"].append(
                    {
                        "name": tensor_name,
                        "spins": [species[pdg]["spin"] for pdg in particle_pdgs],
                        "structure": lorentz_value.to_canonical_string(),
                    }
                )
                lorentz_names.append(tensor_name)
                local_colors[index] = color.to_canonical_string()
                matrix[index][index] = alias
        payload["vertex_rules"].append(
            {
                "name": name,
                "particles": [species[pdg]["name"] for pdg in particle_pdgs],
                "color_structures": [local_colors[i] for i in colors],
                "lorentz_structures": lorentz_names,
                "couplings": matrix,
            }
        )
        alternatives = tuple(
            tuple(
                sorted(
                    source.raw.collections["particles"][i]["fields"]["pdg_code"]
                    for i in alternative
                )
            )
            for alternative in first.loop_particles or ((),)
        )
        if any(pdg < 0 for alternative in alternatives for pdg in alternative):
            raise ValueError("CT loop alternatives must use positive particle PDGs")
        fields = source.raw.collections["counterterm_vertices"][first.vertex_id][
            "fields"
        ]
        rules[name] = CountertermRule(
            name,
            _effective_kind(first),
            tuple(members),
            particle_pdgs,
            first.order,
            alternatives,
            fields.get("perturbation_type"),
            tuple(aliases),
            scalar_program=source.scalar_program,
            basis_correspondences=correspondences,
            primitive_validations=tuple(
                primitive_evidence.get(entry.key) for entry in members
            ),
            tensor_representatives=tuple(
                tensor_evidence.get(entry.key) for entry in members
            ),
            vertex_spinor_frame=vertex_spinor_frame,
            whole_rule_continuation=continuation,
            native_coefficients=tuple(native_coefficients),
        )
    if used_continuations != continuations.keys():
        raise ValueError(
            "whole-rule continuation was not selected in the prepared model"
        )
    native_model = hep.Model.from_json(json.dumps(payload))
    rules = {
        name: replace(
            rule,
            coupling_symbols=tuple(
                native_model.coupling(alias).symbol for alias, _ in rule.couplings
            ),
            native_coupling_symbols=tuple(
                native_model.coupling(alias).symbol
                for alias, _ in rule.native_coefficients
            ),
        )
        for name, rule in rules.items()
    }
    return PreparedCountertermModel(
        ordinary_model,
        native_model,
        source,
        MappingProxyType(rules),
        tuple(unavailable),
        vertex_spinor_frame=vertex_spinor_frame,
        source_entries_complete=seen
        == {
            ("counterterm_vertices", record["id"], *key["items"])
            for record in source.raw.collections["counterterm_vertices"]
            for key, _coupling in record["fields"]["couplings"]["entries"]
        },
    )


@dataclass(frozen=True, slots=True, order=True)
class AttachedCurrent:
    binding_pdg: int
    kind: str
    identity: str


def _rooted_current_key(current, physical_external_indices, symmetric_slots=None):
    """Canonicalize native current metadata with public Symbolica.Graph.

    Only the excision-created boundary loses its artificial incoming/outgoing
    state. Physical labels/states, interaction slots and particle flow survive.
    Port nodes retain each native interaction slot, except for full Bose
    permutation classes proved on that complete model vertex.
    """
    from symbolica import Graph, S

    vertex_color, port_color, external_color, root_color, particle_color = S(
        "loop_current::vertex",
        "loop_current::port",
        "loop_current::external",
        "loop_current::root",
        "loop_current::particle",
    )
    payload = json.loads(current.to_json())
    names = {vertex.id: vertex.interaction for vertex in current.vertices}
    symmetric_slots = symmetric_slots or {}
    graph = Graph()
    vertices = {
        index: graph.add_node(vertex_color(vertex["interaction"]))
        for index, vertex in enumerate(payload["vertices"])
    }
    roots = 0
    for edge, (endpoints, data) in zip(current.edges, payload["edges"], strict=True):
        if edge.is_dummy:
            raise NotImplementedError("dummy current ports are unsupported")
        ports = []
        for endpoint in ("source", "target"):
            vertex = endpoints[endpoint]
            if vertex is not None:
                slot = data[f"{endpoint}_slot"]
                if slot is None:
                    raise NotImplementedError(
                        "native current has an unassigned interaction slot"
                    )
                labels = symmetric_slots.get(names[vertex])
                port = graph.add_node(
                    port_color(slot if labels is None else labels[slot])
                )
                graph.add_edge(vertices[vertex], port)
            elif edge.external_index in physical_external_indices:
                port = graph.add_node(
                    external_color(
                        edge.external_index,
                        1 if edge.external_state == "incoming" else 2,
                    )
                )
            else:
                port = graph.add_node(root_color())
                roots += 1
            ports.append(port)
        if edge.directed and edge.particle_pdg < 0:
            ports.reverse()
        graph.add_edge(*ports, edge.directed, particle_color(abs(edge.particle_pdg)))
    if roots != 1:
        raise ValueError("a rooted current requires exactly one artificial boundary")
    return graph.canonize()[0].to_dot()


def attached_currents(
    diagram: Any, region_vertices: Iterable[int], *, symmetric_slots=None
) -> tuple[AttachedCurrent, ...]:
    """Native canonical identities for currents incident to a collapsed region.

    Direct external legs use their physical leg identity. Composite tree currents
    use native connected-components/excision and Symbolica graph canonicalization,
    preserving external labels, unproved interaction slots and particle identities.
    """
    region = frozenset(region_vertices)
    if not region or not region <= {v.id for v in diagram.vertices}:
        raise ValueError("collapsed region must contain diagram vertices")
    boundary = [
        edge
        for edge in diagram.edges
        if (edge.source in region) != (edge.target in region)
    ]

    def binding(edge):
        return (
            edge.particle.antiparticle.pdg_code
            if edge.source in region
            else edge.particle_pdg
        )

    currents = [
        AttachedCurrent(
            binding(edge), "external", f"{edge.external_index}:{edge.external_state}"
        )
        for edge in boundary
        if edge.is_external
    ]
    outside = diagram.filter(node=lambda vertex: vertex.id not in region)
    for component in outside.connected_components():
        nodes = {v.id for v in component.vertices}
        attachments = [
            edge for edge in boundary if edge.source in nodes or edge.target in nodes
        ]
        if len(attachments) != 1:
            raise ValueError("counterterm attachment must be one rooted tree current")
        current = component.excise()
        if current.loop_count:
            raise ValueError("counterterm attachment contains a loop")
        currents.append(
            AttachedCurrent(
                binding(attachments[0]),
                "tree",
                _rooted_current_key(
                    current,
                    {edge.external_index for edge in diagram.external_edges},
                    symmetric_slots,
                ),
            )
        )
    if len(currents) != len(boundary):
        raise ValueError("native current partition did not cover the region boundary")
    return tuple(sorted(currents))


@dataclass(frozen=True, slots=True)
class CountertermInsertion:
    graph: GraphIR
    rule: CountertermRule
    vertex_id: int
    alternative_index: int
    witness_loop_ids: tuple[str, ...]

    def coefficient_expressions(self, epsilon_power: int):
        """Native aliases paired with exact values and active diagnostics."""
        if self.rule.whole_rule_continuation is not None:
            from symbolica import E

            from .scalar_frontend import ScalarExpression

            # Source diagnostics are still mandatory even when a checked tensor
            # replaces the source values. The scalar frontend's diagnostic
            # composition is additive; zero is the successful value.
            diagnostic = sum(
                (
                    value.diagnostic
                    for _, value in self.source_coefficient_expressions(epsilon_power)
                ),
                E("0"),
            )
            return tuple(
                (
                    symbol,
                    ScalarExpression(
                        value if epsilon_power == 0 else E("0"), diagnostic
                    ),
                )
                for symbol, (_, value) in zip(
                    self.rule.native_coupling_symbols,
                    self.rule.native_coefficients,
                    strict=True,
                )
            )
        return self.source_coefficient_expressions(epsilon_power)

    def source_coefficient_expressions(self, epsilon_power: int):
        """Original source aliases, values and diagnostics, including replaced rules."""
        return tuple(
            (symbol, entry.scalar_expression(epsilon_power))
            for symbol, (_, entry) in zip(
                self.rule.coupling_symbols, self.rule.couplings, strict=True
            )
        )

    def coefficient_substitutions(self, epsilon_power: int):
        """Legacy value-only access rejects diagnostic-bearing source scalars."""
        from .scalar_frontend import require_scalar_value

        return tuple(
            (
                symbol,
                require_scalar_value(
                    value, "counterterm substitution", self.rule.scalar_program
                ),
            )
            for symbol, value in self.coefficient_expressions(epsilon_power)
        )


@dataclass(frozen=True, slots=True)
class CountertermProcessIR:
    insertions: tuple[CountertermInsertion, ...]
    unmatched_candidates: tuple[str, ...]
    stage: str = field(default="native-counterterm-attachment-oracle", init=False)
    scheme_validated: bool = field(default=False, init=False)
    external_wavefunction_counterterms_included: bool = field(default=False, init=False)


def _fingerprint(diagram):
    return tuple(json.loads(diagram.to_json())["model"])


def _ordinary_bose_slots(prepared, interactions):
    """Prove ordinary port classes with their actual owned scalar definitions."""
    from .scalar_bindings import prepare_scalar_bindings
    from .vertex_symmetry import symmetric_boson_slots

    names = tuple(sorted(set(interactions)))
    source = prepared.source
    if source.scalar_program is None:
        # A CT-only source over a separate ordinary model supplies no paired
        # ordinary scalar owner. Retain its literal native tensor proof.
        return symmetric_boson_slots(prepared.native, names)
    aliases = {
        alias
        for name in names
        for row in prepared.native.vertex_rule(name).couplings
        for alias in row
        if alias is not None
    }
    ordinary = {
        entry.coupling.name: entry.coupling
        for entry in source.kernels
        if entry.collection == "vertices"
    }
    for alias in aliases:
        binding = ordinary.get(alias)
        if binding is None:
            raise ValueError("ordinary native coupling lacks its source binding")
        if (
            source.scalar_program.definitions.get((binding.collection, binding.id))
            is not binding
        ):
            raise ValueError("ordinary coupling belongs to a different scalar owner")
        if prepared.native.coupling(alias).expression != binding.value.expression:
            raise ValueError("ordinary native coupling overrides its source binding")
    bindings = prepare_scalar_bindings(
        source.raw,
        tuple((alias, 0) for alias in sorted(aliases)),
        program=source.scalar_program,
    )
    # Exact symbolic values establish only local Bose equality. Their paired
    # diagnostics and helper library remain owned by the retained source; the
    # physical scalar binding evaluates them without this proof discarding any.
    return symmetric_boson_slots(
        prepared.native,
        names,
        coupling_expressions={
            name: expression.value
            for (name, _), expression in zip(
                bindings.outputs, bindings.expressions, strict=True
            )
        },
    )


def match_counterterms(
    prepared: PreparedCountertermModel,
    process: Any,
    loops: LoopProcessIR,
    *,
    selected_corrections: Iterable[str],
    forbidden_particles: Iterable[int] = (),
    order_bounds: Mapping[str, int | tuple[int, int | None]] | None = None,
) -> CountertermProcessIR:
    """Generate exactly-one-CT trees, then prove actual loop attachments.

    The process must include candidate CT rules, with the same external states
    and ordinary-rule/particle restrictions used for the supplied loop catalogue.
    UVtree insertions use ordinary tree generation and forbidden-species filtering;
    R2, UVmass and UVloop additionally require a matching admitted loop subgraph.
    External-leg wavefunction factors remain a separate, explicit prerequisite.
    """
    selected = frozenset(selected_corrections)
    if selected != frozenset(loops.orders.correction_names):
        raise ValueError(
            "counterterms and loop catalogue must share correction selection"
        )
    if process.model.to_json() != prepared.native.to_json():
        raise ValueError("counterterm process must use the prepared native model")
    forbidden = {abs(pdg) for pdg in forbidden_particles}

    def selector_pdg(selector):
        return abs(
            process.model.particle(selector.name).pdg_code
            if selector.is_name
            else selector.pdg
        )

    forbidden.update(map(selector_pdg, process.particle_veto))
    if process.particle_selection is not None:
        allowed = set(map(selector_pdg, process.particle_selection))
        forbidden.update(
            abs(p.pdg_code)
            for p in process.model.particles
            if abs(p.pdg_code) not in allowed
        )
    bounds = dict(order_bounds or {})
    if prepared.tag in bounds:
        raise ValueError("the counterterm insertion tag is internal")
    bounds[prepared.tag] = 1
    if any(len(rule.particles) < 3 for rule in prepared.ordinary_model.vertex_rules):
        raise NotImplementedError(
            "ordinary bivalent rules require a bounded insertion policy"
        )
    # With one CT of valence >=2 and otherwise valence >=3, a tree has at most
    # n_external-1 vertices. Bound native topology enumeration before its order
    # filter is applied to the bivalent counterterm rules.
    max_vertices = (
        len(process.incoming) + max(map(len, process.outgoing_alternatives)) - 1
    )
    result = process.generate_diagrams(
        loops=0,
        max_vertices=max_vertices,
        coupling_orders=bounds,
        maximum_bridges=None,
        self_energy=None,
        tadpoles=None,
        zero_snails=None,
        allow_self_loops=True,
        allow_zero_flow_edges=True,
        factorized_loop_topologies_count_range=None,
        symmetrize_initial=False,
        symmetrize_final=False,
        numerator_grouping=None,
        filter_zero_color=False,
        threads=1,
        progress=None,
    )
    if not result.report.completed:
        raise RuntimeError("native counterterm tree generation did not complete")
    rule_orders = _rule_orders(prepared.native)
    species_orders = _species_orders(
        prepared.ordinary_model, _rule_orders(prepared.ordinary_model)
    )
    denominators = _propagator_denominators(prepared.native)
    symmetric_slots = _ordinary_bose_slots(
        prepared,
        {
            vertex.interaction
            for graph in loops.loops
            for vertex in graph.native.vertices
        },
    )
    witnesses = defaultdict(list)
    fingerprints = {_fingerprint(g.native) for g in (*loops.born, *loops.loops)}
    for graph in loops.loops:
        cycles, _ = graph.native.cycle_basis()
        vertices = {v.id for v in cycles[0].vertices}
        local_order = _summed_orders(
            v.order for v in graph.vertices if v.id in vertices
        )
        key = (
            attached_currents(graph.native, vertices, symmetric_slots=symmetric_slots),
            local_order,
        )
        witnesses[key].append(graph)
    insertions, unmatched = [], []
    for diagram in sorted(result.diagrams, key=lambda item: item.id):
        if fingerprints and fingerprints != {_fingerprint(diagram)}:
            raise ValueError(
                "loop and CT trees must share the prepared native model fingerprint"
            )
        sites = [v for v in diagram.vertices if v.interaction in prepared.rules]
        if len(sites) != 1:
            raise ValueError("native CT tag did not select exactly one insertion")
        site = sites[0]
        rule = prepared.rules[site.interaction]
        if (
            rule.perturbation_type is not None
            and rule.perturbation_type not in selected
        ):
            unmatched.append(diagram.id)
            continue
        snapshot = _snapshot(diagram, rule_orders, species_orders, denominators)
        snapshot = replace(
            snapshot,
            order=tuple(item for item in snapshot.order if item[0] != prepared.tag),
            vertices=tuple(
                replace(
                    v, order=tuple(item for item in v.order if item[0] != prepared.tag)
                )
                for v in snapshot.vertices
            ),
        )
        currents = attached_currents(
            diagram, (site.id,), symmetric_slots=symmetric_slots
        )
        matches = witnesses.get((currents, rule.order), ())
        count = len(insertions)
        for index, alternative in enumerate(rule.alternatives):
            if rule.kind in {"UVtree", "UVloop"} and forbidden.intersection(
                alternative
            ):
                continue
            if rule.kind == "UVtree":
                if not selected.intersection(name for name, _ in rule.order):
                    continue
                matching_ids = ()
            else:
                matching_ids = tuple(
                    graph.id
                    for graph in matches
                    if rule.kind == "UVloop"
                    or not alternative
                    or graph.loop_species == alternative
                )
                if not matching_ids:
                    continue
            insertions.append(
                CountertermInsertion(snapshot, rule, site.id, index, matching_ids)
            )
        if len(insertions) == count:
            unmatched.append(diagram.id)
    return CountertermProcessIR(tuple(insertions), tuple(unmatched))
