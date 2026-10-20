# SPDX-License-Identifier: 0BSD
"""Exact species transport and narrowly proved external Bose permutations.

This is separate from the species-only proof. Admission uses two actual
source-generated programs and never trusts numerical tree profiles or mutable
process-group records. Literal vector-source composition retains both actual
prepared catalogs. Projected colour also retains its actual formal owner and
proves the semantic signed component axes. Crossing needs a separate proof.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pyamplicol.generation.process_grouping import GroupMember, LegMapping

from .flavor_mappings import FlavorSymmetry
from .physical_source import NativeSourceModelProgram
from .request import LoopExternalRequest, prepare_loop_external_request


def _color_owner(proof, model):
    """Retain the actual projection and declaration, never an Nc=3 inference."""
    bare = model.bare
    captured = model.prepared
    metadata, bare_metadata = dict(captured.metadata), dict(captured.bare.metadata)
    continuation = proof.plan.source_color
    if (
        getattr(model, "source_color", None) is not continuation
        or metadata.get("source_color") is not continuation
    ):
        raise ValueError("flavour transport has a different source colour owner")
    projection = model.color_projection
    if projection is None:
        if (
            continuation is not None
            or model.color_accuracy != "full"
            or getattr(bare, "_formal_color_projected", False)
        ):
            raise ValueError("flavour transport requires its actual formal projection")
        return None
    formal = getattr(bare, "formal_color_metrics", None)
    if (
        continuation is None
        or formal is None
        or projection.formal_metrics is not formal
        or metadata.get("color_projection") is not projection
        or bare_metadata.get("formal_color_metrics") is not formal
        or model.color_accuracy != projection.accuracy
        or projection.accuracy != formal.accuracy
        or projection.born_degrees != formal.born_degrees
        or projection.virtual_degrees != formal.virtual_degrees
        or formal.nc != continuation.nc
    ):
        raise ValueError("flavour transport requires its actual formal projection")
    formal.require_owner(bare)
    return formal


def _component_axes(source):
    """Recompute the signed anchors from exact tensors and absolute orders."""
    from collections import Counter

    from .color_components import prepare_born_color_components
    from .color_orbits import _canonical_with_ports

    bare = source.virtual.bare
    components = bare.born_color_components
    captured = dict(source.prepared.bare.metadata).get("born_color_components")
    if source.virtual.color_accuracy != "lc":
        if components is not None or captured is not None:
            raise ValueError("contracted flavour colour axis has unexpected anchors")
        return None
    expected = prepare_born_color_components(bare.born_color_tensors)
    if components != expected or captured != expected:
        raise ValueError("flavour signed colour components differ from actual tensors")
    canonical = []
    for tensor in bare.born_color_tensors:
        expression, reason = _canonical_with_ports(tensor)
        if reason is not None:
            raise NotImplementedError(f"flavour colour component is unproved: {reason}")
        canonical.append((expression, frozenset(tensor.axes)))
    # Raw columns may be ordered differently in independently generated graphs.
    # Keep multiplicities and absolute orders within each public signed anchor.
    return components.ids, tuple(
        Counter((canonical[index], bare.born_term_orders[index]) for index in members)
        for members in components.members
    )


def _paired_color(representative, member, left, right):
    if (left is None) != (right is None):
        raise ValueError("flavour transport changes formal colour convention")
    if left is not None and (
        left.accuracy != right.accuracy
        or left.nc != right.nc
        or left.initial_average != right.initial_average
        or left.born_degrees != right.born_degrees
        or left.virtual_degrees != right.virtual_degrees
        or left.born_pairs != right.born_pairs
        or left.virtual_pairs != right.virtual_pairs
    ):
        raise ValueError("flavour transport changes formal colour sectors or degrees")
    if _component_axes(representative) != _component_axes(member):
        raise ValueError("flavour transport changes signed colour component axes")


def _request(proof, source):
    """Read the actual source/current owner before retaining its cold identity."""
    if not isinstance(source, NativeSourceModelProgram):
        raise TypeError("flavour transport requires source-generated model programs")
    plan, bare = proof.plan, source.virtual.bare
    structure = source.source_bare.structure
    scalar = source.source_bare.state
    if (
        structure is not plan.structure
        or source.source_bare.bare is not bare
        or scalar is not bare.model_state
        or scalar.plan is None
        or scalar.plan.program is not structure.scalars
        or scalar.plan.fixed_parameters != structure.fixed_parameters
    ):
        raise ValueError("flavour transport requires the retained source/scalar owner")
    if source.virtual.r2_strategy != "model" or bare.dimensional_numerators:
        raise NotImplementedError("flavour transport requires model R2")
    # Prove the full local generation allow-list, never the vertices observed in
    # these diagrams. The existing request scope admits the retained CT entries.
    if set(plan.ordinary_rules) != {
        rule.name for rule in structure.native.vertex_rules
    }:
        raise ValueError("flavour transport needs the complete generation allow-list")
    if structure.request_scope is not None:
        structure.request_scope.require_local_counterterms(plan.counterterms)
    elif not plan.counterterms.source_entries_complete:
        raise ValueError("flavour transport needs complete source counterterms")
    if bare.order_layout.correction_names != plan.correction_names:
        raise ValueError(
            "flavour transport correction selection differs from its proof"
        )
    for insertion in source.matched.insertions:
        if plan.counterterms.rules.get(insertion.rule.name) is not insertion.rule:
            raise ValueError("flavour transport counterterms have a different owner")
    if tuple(source.virtual.matched_insertions) != tuple(source.matched.insertions):
        raise ValueError("flavour transport current and matched counterterms differ")
    request = source.source_bare.external_request
    expected = prepare_loop_external_request(
        structure.native, request.incoming, request.outgoing
    )
    external = tuple(
        edge.particle.pdg_code
        for edge in sorted(
            bare.process.born[0].native.external_edges,
            key=lambda edge: edge.external_index,
        )
    )
    if (
        request != expected
        or external != request.external_pdgs
        or bare.state_layout != request.states
        or bare.initial_average != request.initial_average
    ):
        raise ValueError("flavour transport request differs from its actual currents")
    return request, _color_owner(proof, source.virtual)


def _directed_pair(representative, member, left, right):
    """Check actual preparations before comparing their observable catalogs."""
    from .prepared_correlations import PreparedDirectedModel

    if not isinstance(left, PreparedDirectedModel) or not isinstance(
        right, PreparedDirectedModel
    ):
        raise TypeError(
            "flavour source admission requires paired directed preparations"
        )
    left.require_owner(representative.virtual)
    right.require_owner(member.virtual)
    a, b = left.bare, right.bare
    if a.histories != b.histories or a.color_legs != b.color_legs:
        raise ValueError("flavour transport changes directed history or colour ports")
    if a.helicities != b.helicities:
        raise ValueError("flavour transport changes spectator helicity layouts")
    if representative.virtual.color_projection is not None and any(
        connection.emissions for pair in a.histories for connection in pair
    ):
        raise NotImplementedError(
            "projected flavour transport requires identity colour histories"
        )


def _spin_catalog(source, ordinary, prepared):
    """Retain dynamic class keys and the model's actual SourceIR authority."""
    from .spin_sources import NativeSpinSourcePlan

    result = {}
    bare = source.virtual.bare
    external = source.source_bare.external_request.external_pdgs
    for item in prepared:
        item.require_owner(source.virtual)
        layout = item.bare.spin_layout
        if layout is None:
            raise ValueError("spin flavour catalog contains an ordinary preparation")
        layout.require_owner(bare)
        if layout.legs in result:
            raise ValueError("spin flavour catalog repeats a joint source class")
        if (
            item.bare.histories != ordinary.bare.histories
            or item.bare.color_legs != ordinary.bare.color_legs
        ):
            raise ValueError("spin flavour catalog changes directed histories or ports")
        if item.bare.helicities != layout.helicities:
            raise ValueError("spin flavour catalog differs from its spectator layout")
        if layout.contracts != tuple(
            bare.source_contracts[external[leg - 1]] for leg in layout.legs
        ):
            raise ValueError("spin flavour catalog differs from model-owned SourceIR")
        result[layout.legs] = item
    # A shared owner pointer and pairwise equality do not establish that phases
    # and spectator rows were derived from that owner. Reuse the same source
    # factory once here; its inputs are structural SourceIR and bare helicities.
    expected = NativeSpinSourcePlan(bare, tuple(result)).layouts
    for key, item in result.items():
        layout = item.bare.spin_layout
        if layout.phase_states != expected[key].phase_states:
            raise ValueError("spin flavour catalog changes model-owned crossing phase")
        if layout.helicities != expected[key].helicities:
            raise ValueError("spin flavour catalog changes owner spectator helicities")
    return result


def _admit_sources(representative, member, species, left, right, left_spin, right_spin):
    if left is None and right is None and not left_spin and not right_spin:
        return (), ()
    _directed_pair(representative, member, left, right)
    if left.bare.spin_layout is not None or right.bare.spin_layout is not None:
        raise ValueError("flavour admission requires paired ordinary directed plans")
    if any(
        item.bare.helicities != source.virtual.bare.helicities
        for source, item in ((representative, left), (member, right))
    ):
        raise ValueError("ordinary flavour catalog changes owner helicities")
    a, b = (
        _spin_catalog(representative, left, left_spin),
        _spin_catalog(member, right, right_spin),
    )
    if a.keys() != b.keys():
        raise ValueError("flavour transport changes declared joint source classes")
    external = representative.source_bare.external_request.external_pdgs
    for key in a:
        _directed_pair(representative, member, a[key], b[key])
        source, target = a[key].bare.spin_layout, b[key].bare.spin_layout
        if any(
            species.get(external[leg - 1], external[leg - 1]) != external[leg - 1]
            for leg in key
        ):
            raise NotImplementedError(
                "flavour transport must preserve selected vector species"
            )
        if (
            source.contracts != target.contracts
            or source.phase_states != target.phase_states
        ):
            raise ValueError("flavour transport changes SourceIR or crossing phase")
    return tuple(a[key] for key in sorted(a)), tuple(b[key] for key in sorted(b))


def _bose_local_rules(plan):
    """Apply the existing local proof to each complete retained eligibility group."""
    from symbolica import E

    from .flavor_mappings import _powers, _whole_rule_aliases
    from .model_expressions import LaurentExpression, epsilon_symbol
    from .scalar_bindings import prepare_scalar_bindings
    from .vertex_symmetry import symmetric_boson_slots

    model = plan.counterterms.native
    outputs, exact, aliases = {}, {}, {}
    for name in plan.ordinary_rules:
        for row in model.vertex_rule(name).couplings:
            for alias in row:
                if alias is not None:
                    outputs[alias, 0] = None
                    aliases[alias] = ((alias, 0),)
    for rule in plan.counterterms.rules.values():
        whole = _whole_rule_aliases(rule, model)
        if whole:
            if plan.source_color is None:
                raise ValueError(
                    "whole-rule Bose proof requires its formal source owner"
                )
            for alias, expression in whole.items():
                aliases[alias] = tuple((alias, power) for power in _powers(expression))
                for output in aliases[alias]:
                    exact[output] = LaurentExpression(expression).coefficient(output[1])
        else:
            for alias, entry in rule.couplings:
                terms = tuple(
                    (entry.coupling.name, power)
                    for power in _powers(entry.coupling.value.expression)
                )
                aliases[alias] = terms
                outputs.update(dict.fromkeys(terms))
    if plan.source_color is None:
        binding = prepare_scalar_bindings(
            plan.counterterms.source.raw,
            tuple(outputs),
            program=plan.structure.scalars,
            fixed_parameters=plan.structure.fixed_parameters,
        )
    else:
        binding = plan.source_color.formal_scalars(
            tuple(outputs),
            program=plan.structure.scalars,
            fixed_parameters=plan.structure.fixed_parameters,
            exact_outputs=exact,
        )
    resolved = dict(zip(binding.outputs, binding.expressions, strict=True))
    epsilon = epsilon_symbol()
    expansions = {
        alias: sum(
            (epsilon**power * resolved[name, power].value for name, power in terms),
            E("0"),
        )
        for alias, terms in aliases.items()
    }
    names = (*plan.ordinary_rules, *plan.counterterms.rules)
    classes = symmetric_boson_slots(
        model, names, coupling_expressions=expansions, conserve_momentum=True
    )
    for name in names:
        rule = model.vertex_rule(name)
        groups = {}
        for index, species in enumerate(rule.particles):
            particle = model.particle(species)
            if (
                particle.spin > 0
                and particle.spin % 2 == 1
                and particle.is_self_antiparticle
            ):
                groups.setdefault(species, []).append(index)
        labels = classes.get(name, tuple(range(len(rule.particles))))
        if any(
            len({labels[index] for index in slots}) != 1 for slots in groups.values()
        ):
            raise NotImplementedError(
                f"complete local Bose symmetry is unproved for {name}"
            )
    # Keep the paired source preparation: scalar diagnostics are never erased
    # merely because its value disappeared in the exact tensor comparison.
    return binding


def _bose_external_fields(plan, source, permutation):
    """Check the actual complete field multiset and its physical-port image."""
    from collections import Counter

    from .wavefunction import match_external_field_counterterms

    model, process = source.virtual, source.virtual.bare.process
    actual = model.external_fields
    if dict(source.prepared.metadata).get("external_fields") is not actual:
        raise ValueError("Bose field factors differ from the captured model owner")
    expected = match_external_field_counterterms(
        plan.counterterms.source,
        process,
        selected_corrections=plan.correction_names,
        allowed_amplitude_orders={
            pair.loop for sector in process.orders.sectors for pair in sector.pairs
        },
    ).insertions
    born = {id(graph): graph for graph in process.born}
    rules = {
        (
            item.rule.particle_id,
            item.rule.correction,
            item.rule.loop_group_index,
        ): item.rule
        for item in expected
    }

    def entries(insertions, remap):
        result = []
        for item in insertions:
            rule = item.rule
            key = (rule.particle_id, rule.correction, rule.loop_group_index)
            if (
                born.get(id(item.born)) is not item.born
                or rules.get(key) != rule
                or type(item.external_index) is not int
                or not 0 <= item.external_index < len(permutation)
            ):
                raise ValueError(
                    "Bose field insertion has a different source or Born owner"
                )
            index = permutation[item.external_index] if remap else item.external_index
            result.append((id(item.born), index, key, item.alternative_index))
        return Counter(result)

    original = entries(actual, False)
    if original != entries(expected, False):
        raise ValueError(
            "Bose field insertions differ from the complete selected source rules"
        )
    if original != entries(actual, True):
        raise ValueError("Bose permutation changes the indexed external-field multiset")


def _admit_bose_permutation(symmetry, representative, member, mapping):
    """Same-side identical self-conjugate bosons, ordinary full colour only."""
    permutation = mapping.permutation
    request = representative.source_bare.external_request
    legs = len(request.external_pdgs)
    if (
        len(permutation) != legs
        or any(type(index) is not int for index in permutation)
        or sorted(permutation) != list(range(legs))
    ):
        raise ValueError("flavour external permutation must be a complete bijection")
    for source in (representative, member):
        if source.virtual.color_accuracy != "full":
            raise NotImplementedError("Bose permutation transport requires full colour")
        particles = source.source_bare.structure.native
        external = source.source_bare.external_request.external_pdgs
        states = source.source_bare.external_request.states.legs
        for index, target in enumerate(permutation):
            if index == target:
                continue
            particle = particles.particle_by_pdg(external[index])
            if (
                external[index] != external[target]
                or states[index].incoming != states[target].incoming
                or particle.spin <= 0
                or particle.spin % 2 != 1
                or not particle.is_self_antiparticle
                or states[index].kind != states[target].kind
                or states[index].massless != states[target].massless
                or states[index].helicities != states[target].helicities
            ):
                raise NotImplementedError(
                    "external permutation requires identical self-conjugate bosons "
                    "on the same process side"
                )
    binding = _bose_local_rules(symmetry.plan)
    for source in (representative, member):
        _bose_external_fields(symmetry.plan, source, permutation)
    return binding


@dataclass(frozen=True, slots=True)
class IdentityFlavorTransport:
    """Admitted unit-factor relation between two prepared source programs.

    ``species`` maps representative PDGs to member PDGs. The existing LegMapping
    maps member momenta to representative ports. Nonidentity maps additionally
    require exact local Bose covariance and the complete indexed field multiset;
    they admit only ordinary inclusive full colour. No relation is inferred from
    numerical model defaults or equality of a few amplitudes.
    """

    symmetry: FlavorSymmetry = field(repr=False)
    mapping: LegMapping
    representative_request: LoopExternalRequest
    member_request: LoopExternalRequest
    representative_plan: Any = field(repr=False)
    member_plan: Any = field(repr=False)
    orders: Any
    representative_directed: Any = field(default=None, repr=False)
    member_directed: Any = field(default=None, repr=False)
    representative_spin: tuple = field(default=(), repr=False)
    member_spin: tuple = field(default=(), repr=False)
    bose_scalar_bindings: Any = field(default=None, repr=False)

    @property
    def permutes_external_legs(self):
        return self.mapping.permutation != tuple(
            range(len(self.representative_request.external_pdgs))
        )

    @classmethod
    def admit(
        cls,
        symmetry,
        representative,
        member,
        mapping,
        *,
        representative_directed=None,
        member_directed=None,
        representative_spin=(),
        member_spin=(),
    ):
        if not isinstance(symmetry, FlavorSymmetry):
            raise TypeError(
                "flavour transport requires an exact retained species proof"
            )
        if not isinstance(mapping, LegMapping):
            raise TypeError("flavour transport requires an existing LegMapping")
        left, left_color = _request(symmetry, representative)
        right, right_color = _request(symmetry, member)
        if mapping.conjugated:
            raise NotImplementedError("charge-conjugated flavour transport is unproved")
        nonidentity = mapping.permutation != tuple(range(len(left.external_pdgs)))
        bose_scalars = None
        if nonidentity:
            if (
                representative_directed is not None
                or member_directed is not None
                or representative_spin
                or member_spin
            ):
                raise NotImplementedError(
                    "Bose permutation transport has no directed or vector-source "
                    "axis proof"
                )
            bose_scalars = _admit_bose_permutation(
                symmetry, representative, member, mapping
            )
        species = dict(symmetry.species)
        if tuple(
            species.get(pdg, pdg) for pdg in left.external_pdgs
        ) != right.external_pdgs or len(left.incoming) != len(right.incoming):
            raise ValueError("member request is not the proved species image")
        if left.states != right.states or left.initial_average != right.initial_average:
            raise ValueError(
                "flavour transport changes external states or normalization"
            )
        a, b = representative.virtual.bare, member.virtual.bare
        if (
            a.order_layout != b.order_layout
            or a.process.orders.born_orders != b.process.orders.born_orders
            or a.process.convention != b.process.convention
            or a.vertex_spinor_frame != b.vertex_spinor_frame
            or a.helicities != b.helicities
        ):
            raise ValueError("flavour transport changes orders, pairing or conventions")
        _paired_color(
            representative,
            member,
            left_color,
            right_color,
        )
        left_spin, right_spin = _admit_sources(
            representative,
            member,
            species,
            representative_directed,
            member_directed,
            tuple(representative_spin),
            tuple(member_spin),
        )
        return cls(
            symmetry,
            mapping,
            left,
            right,
            representative.prepared,
            member.prepared,
            a.order_layout,
            representative_directed,
            member_directed,
            left_spin,
            right_spin,
            bose_scalars,
        )

    def _require_binding(self, representative, directed):
        """Use the same owner/capability boundary for every wrapper construction."""
        if (
            not isinstance(representative, NativeSourceModelProgram)
            or representative.prepared is not self.representative_plan
        ):
            raise ValueError(
                "flavour transport belongs to another prepared representative"
            )
        if directed is not None:
            if self.permutes_external_legs:
                raise NotImplementedError(
                    "Bose permutation transport has no directed axis proof"
                )
            from .prepared_correlations import PreparedDirectedModel

            if not isinstance(directed, PreparedDirectedModel):
                raise TypeError("expected retained directed model instructions")
            directed.require_owner(representative.virtual)
            if directed.bare.spin_layout is not None:
                if not any(directed is item for item in self.representative_spin):
                    raise ValueError("mapped spin class has no paired source admission")
            elif (
                self.representative_directed is not None
                and directed is not self.representative_directed
            ):
                raise ValueError("mapped histories differ from their paired admission")

    def bind(self, representative, *, directed=None):
        """Reuse admitted currents and their exact mutable global parameter owner."""
        return NativeIdentityFlavorProgram(self, representative, directed)


@dataclass(frozen=True, slots=True)
class NativeIdentityFlavorProgram:
    """A member view of existing batched native executors, with no scalar factor.

    Whole result objects pass through unchanged, including complex cross terms,
    pole/order axes, component diagnostics and original native lane identities.
    Parameter updates use the same atomic source owner as the representative.
    Fixed generation ties therefore cannot be broken by a member update.
    """

    transport: IdentityFlavorTransport
    representative: NativeSourceModelProgram = field(repr=False)
    directed_plan: Any = field(default=None, repr=False)
    _directed: Any = field(default=None, init=False, repr=False)

    def __post_init__(self):
        self.transport._require_binding(self.representative, self.directed_plan)
        if self.directed_plan is not None:
            object.__setattr__(
                self, "_directed", self.directed_plan.bind(self.representative)
            )

    @property
    def external_request(self):
        return self.transport.member_request

    @property
    def state(self):
        return self.representative.source_bare.state

    def rebind(self, **options):
        return self.transport.bind(
            self.representative.rebind(**options), directed=self.directed_plan
        )

    def update_parameters(self, changes):
        return self.representative.update_parameters(changes)

    def _momenta(self, momenta):
        if not self.transport.permutes_external_legs:
            return momenta
        import numpy as np

        values = np.asarray(momenta, dtype=object)
        permutation = self.transport.mapping.permutation
        if values.ndim != 3 or values.shape[1:] != (len(permutation), 4):
            raise ValueError("expected (batch, leg, four components) momenta")
        return values[:, permutation, :]

    def evaluate_ordered_native(self, momenta, scales_squared):
        return self.representative.virtual.evaluate_ordered_native(
            self._momenta(momenta), scales_squared
        )

    def evaluate_ordered_decimal(self, momenta, scales_squared):
        return self.representative.evaluate_ordered_decimal(
            self._momenta(momenta), scales_squared
        )

    def evaluate_directed_native(
        self, momenta, scales_squared, *, point_ids=None, spin_sources=None
    ):
        if self._directed is None:
            raise ValueError("mapped evaluation has no retained directed histories")
        return self._directed.evaluate_native(
            momenta, scales_squared, point_ids=point_ids, spin_sources=spin_sources
        )

    def evaluate_directed_decimal(
        self, momenta, scales_squared, *, point_ids=None, spin_sources=None
    ):
        if self._directed is None:
            raise ValueError("mapped evaluation has no retained directed histories")
        return self._directed.evaluate_decimal(
            momenta, scales_squared, point_ids=point_ids, spin_sources=spin_sources
        )


def _member_expression_identity(structure, expression):
    """Resolve public spelling using actual retained source/native identities."""
    import json

    from pyamplicol.api.requests import ProcessRequest
    from pyamplicol.processes.model import (
        is_external_model_particle,
        resolve_model_particle_name,
    )

    from .covariant_model import _flag

    expression = ProcessRequest.parse(expression).expression
    initial, separator, final = expression.partition(">")
    if not separator or ">" in final:
        raise ValueError("flavour member requires one concrete collision")
    names = {}
    metadata = {
        particle["pdg_code"]: particle
        for particle in json.loads(structure.native.to_json())["particles"]
    }
    for record in structure.raw.collections["particles"]:
        fields = record["fields"]
        pdg = fields["pdg_code"]
        actual_fields = metadata[pdg]
        if not is_external_model_particle(
            spin=actual_fields["spin"],
            ghost_number=actual_fields.get("ghost_number", 0),
            goldstoneboson=actual_fields.get("goldstoneboson", False),
            propagating=actual_fields.get("propagating", True),
            auxiliary_kind=actual_fields.get("auxiliary_kind"),
        ):
            continue
        if not is_external_model_particle(
            spin=fields["spin"],
            ghost_number=fields.get("GhostNumber", 0),
            goldstoneboson=_flag(fields, ("GoldstoneBoson", "goldstoneboson", False)),
            propagating=_flag(fields, ("Propagating", "propagating", True)),
            auxiliary_kind=fields.get("auxiliary_kind"),
        ):
            raise ValueError(
                "source/native external eligibility differs for flavour member"
            )
        native = structure.native.particle_by_pdg(pdg)
        for name, actual in ((record["name"], pdg), (native.name, native.pdg_code)):
            if name in names and names[name] != actual:
                raise ValueError(
                    "ambiguous source/native particle name in flavour member"
                )
            names[name] = actual
    incoming, outgoing = initial.split(), final.split()
    resolved = tuple(
        names[
            resolve_model_particle_name(
                name, tuple(names), model_name=structure.raw.name
            )
        ]
        for name in (*incoming, *outgoing)
    )
    return resolved, len(incoming)


@dataclass(frozen=True, slots=True)
class AdmittedFlavorMember:
    """An existing public group member backed by an actual source admission."""

    transport: IdentityFlavorTransport = field(repr=False)
    member: GroupMember

    def __post_init__(self):
        from pyamplicol.api.requests import ProcessRequest

        request = self.transport.member_request
        ProcessRequest.parse(self.member.expression, name=self.member.process_id)
        pdgs, initial_count = _member_expression_identity(
            self.transport.symmetry.plan.structure, self.member.expression
        )
        if (
            self.member.pdgs != request.external_pdgs
            or self.member.initial_count != len(request.incoming)
            or pdgs != request.external_pdgs
            or initial_count != len(request.incoming)
            or self.member.permutation != self.transport.mapping.permutation
            or self.member.factor != 1
            or self.member.conjugated
            or self.member.generated
        ):
            raise ValueError("public flavour member differs from the admitted process")

    def for_program(self, program, *, spin_directed=()):
        if program.prepared is not self.transport.representative_plan:
            raise ValueError(
                "flavour admission belongs to another saved representative"
            )
        _color_owner(self.transport.symmetry, program)
        if self.transport.permutes_external_legs:
            binding = self.transport.bose_scalar_bindings
            structure = self.transport.symmetry.plan.structure
            if (
                binding is None
                or binding.program is not structure.scalars
                or binding.fixed_parameters != structure.fixed_parameters
            ):
                raise ValueError(
                    "Bose transport lacks its retained paired source scalars"
                )
        expected = {
            item.bare.spin_layout.legs: item
            for item in self.transport.representative_spin
        }
        actual = tuple(spin_directed)
        if not self.transport.permutes_external_legs and (
            len(actual) != len(expected)
            or any(
                item.bare.spin_layout is None
                or expected.get(item.bare.spin_layout.legs) is not item
                for item in actual
            )
            or len({item.bare.spin_layout.legs for item in actual}) != len(expected)
        ):
            raise ValueError(
                "saved spin classes differ from their paired flavour admission"
            )
        result = {
            "process_id": self.member.process_id,
            "expression": self.member.expression,
            "external_pdgs": self.member.pdgs,
            "species": self.transport.symmetry.species,
        }
        if self.transport.permutes_external_legs:
            result["permutation"] = self.transport.mapping.permutation
        return result


def identity_species_candidate(representative, member):
    """Complete the port-imposed signed substitution as a candidate permutation.

    This is only request bookkeeping: open chains close into cycles and all
    untouched species stay fixed. Exact local-rule proof still decides whether
    this candidate is a symmetry. Failure never removes an independent process.
    """
    if len(representative) != len(member):
        return None
    mapping = {}
    inverse = {}
    for source, target in zip(representative, member, strict=True):
        for a, b in ((source, target), (-source, -target)):
            if (a in mapping and mapping[a] != b) or (b in inverse and inverse[b] != a):
                return None
            mapping[a], inverse[b] = b, a
    for start in mapping.keys() - inverse.keys():
        end = mapping[start]
        while end in mapping:
            end = mapping[end]
        mapping[end] = start
    return {a: b for a, b in mapping.items() if a != b}
