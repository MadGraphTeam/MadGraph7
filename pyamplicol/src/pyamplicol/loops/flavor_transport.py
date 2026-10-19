# SPDX-License-Identifier: 0BSD
"""Internal transport for exact species symmetries with unchanged external ports.

This is separate from the species-only proof and from public artifact mappings.
Admission uses two actual source-generated programs. It does not certify a tree
profile, a process-group manifest, crossing, resolved colour or spin transport.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pyamplicol.generation.process_grouping import GroupMember, LegMapping

from .flavor_mappings import FlavorSymmetry
from .physical_source import NativeSourceModelProgram
from .request import LoopExternalRequest, prepare_loop_external_request


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
    if (
        source.virtual.r2_strategy != "model"
        or source.virtual.color_accuracy != "full"
        or source.virtual.color_projection is not None
        or bare.dimensional_numerators
        or getattr(bare, "_formal_color_projected", False)
    ):
        raise NotImplementedError("flavour transport requires full colour and model R2")
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
    return request


@dataclass(frozen=True, slots=True)
class IdentityFlavorTransport:
    """Admitted unit-factor relation between two prepared source programs.

    ``species`` maps representative PDGs to member PDGs. The existing LegMapping
    still maps member momenta to representative ports; those ports are unchanged
    in this first supported domain. No relation is inferred from numerical model
    defaults or equality of a few amplitudes.
    """

    symmetry: FlavorSymmetry = field(repr=False)
    mapping: LegMapping
    representative_request: LoopExternalRequest
    member_request: LoopExternalRequest
    representative_plan: Any = field(repr=False)
    member_plan: Any = field(repr=False)
    orders: Any

    @classmethod
    def admit(cls, symmetry, representative, member, mapping):
        if not isinstance(symmetry, FlavorSymmetry):
            raise TypeError(
                "flavour transport requires an exact retained species proof"
            )
        if not isinstance(mapping, LegMapping):
            raise TypeError("flavour transport requires an existing LegMapping")
        left, right = _request(symmetry, representative), _request(symmetry, member)
        if mapping.conjugated or mapping.permutation != tuple(
            range(len(left.external_pdgs))
        ):
            raise NotImplementedError(
                "flavour transport currently requires unchanged external port order"
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
        return cls(
            symmetry,
            mapping,
            left,
            right,
            representative.prepared,
            member.prepared,
            a.order_layout,
        )

    def bind(self, representative, *, directed=None):
        """Reuse admitted currents and their exact mutable global parameter owner."""
        if (
            not isinstance(representative, NativeSourceModelProgram)
            or representative.prepared is not self.representative_plan
        ):
            raise ValueError(
                "flavour transport belongs to another prepared representative"
            )
        if directed is not None:
            from .prepared_correlations import PreparedDirectedModel

            if not isinstance(directed, PreparedDirectedModel):
                raise TypeError("expected retained directed model instructions")
            directed.require_owner(representative.virtual)
            if directed.bare.spin_layout is not None:
                raise NotImplementedError(
                    "mapped spin-source transport is not yet certified"
                )
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

    def evaluate_ordered_native(self, momenta, scales_squared):
        return self.representative.virtual.evaluate_ordered_native(
            momenta, scales_squared
        )

    def evaluate_ordered_decimal(self, momenta, scales_squared):
        return self.representative.evaluate_ordered_decimal(momenta, scales_squared)

    def evaluate_directed_native(self, momenta, scales_squared, *, point_ids=None):
        if self._directed is None:
            raise ValueError("mapped evaluation has no retained directed histories")
        return self._directed.evaluate_native(
            momenta, scales_squared, point_ids=point_ids
        )

    def evaluate_directed_decimal(self, momenta, scales_squared, *, point_ids=None):
        if self._directed is None:
            raise ValueError("mapped evaluation has no retained directed histories")
        return self._directed.evaluate_decimal(
            momenta, scales_squared, point_ids=point_ids
        )


@dataclass(frozen=True, slots=True)
class AdmittedFlavorMember:
    """An existing public group member backed by an actual source admission."""

    transport: IdentityFlavorTransport = field(repr=False)
    member: GroupMember

    def __post_init__(self):
        from pyamplicol.api.requests import ProcessRequest

        request = self.transport.member_request
        expression = " ".join((*request.incoming, ">", *request.outgoing))
        ProcessRequest.parse(self.member.expression, name=self.member.process_id)
        if (
            self.member.pdgs != request.external_pdgs
            or self.member.initial_count != len(request.incoming)
            or self.member.expression != expression
            or self.member.permutation != self.transport.mapping.permutation
            or self.member.factor != 1
            or self.member.conjugated
            or self.member.generated
        ):
            raise ValueError("public flavour member differs from the admitted process")

    def for_program(self, program):
        if program.prepared is not self.transport.representative_plan:
            raise ValueError(
                "flavour admission belongs to another saved representative"
            )
        return {
            "process_id": self.member.process_id,
            "expression": self.member.expression,
            "external_pdgs": self.member.pdgs,
            "species": self.transport.symmetry.species,
        }


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
