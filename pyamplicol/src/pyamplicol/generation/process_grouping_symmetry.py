# SPDX-License-Identifier: 0BSD
"""Whether a flavour-blind relation is a symmetry of the allowed model vertices.

``--flavor-blind`` assumes that processes related by a type-preserving flavour
relabelling, optionally combined with charge conjugation, have equal matrix
elements. That holds when the relation maps the vertices the representative is
generated with onto themselves, which this module checks locally:

* The relabelling of the member's flavours onto the representative's is
  completed to a permutation of the model's particles (flavours present on
  one side only are paired by particle type) and applied to every allowed
  vertex. Each moved vertex must have an allowed image with the same particles
  in the same order and the same structure and coupling: Lorentz and colour
  expressions and resolved coupling for compiled models, kernel kind and
  coupling for the built-in SM.
* Charge conjugation with unchanged helicities is a symmetry of vector-like
  interactions only. It is accepted when the conjugate of every allowed
  vertex is allowed and no allowed vertex couples fermions chirally: a
  Lorentz structure that changes when its left- and right-handed projectors
  are exchanged (or contains ``gamma5``), or a built-in electroweak current
  with different left- and right-handed couplings.

Allowed vertices are those whose coupling orders fit ``process.max_coupling_orders``
and, under the default ``minimal`` coupling-order policy, the representative's
minimal coupling-order envelope. Masses and widths already belong to the
particle type of a flavour class. The check inspects vertices, not diagrams,
so it may warn about a relation that happens to hold for the processes at hand.
"""

from __future__ import annotations

import functools
import re
from collections import Counter
from collections.abc import Collection, Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field

from ..models.base import Model
from ..models.contracts import CompiledModelIR
from .process_grouping import GroupingProcess, LegMapping, ParticleTable


@dataclass(frozen=True, slots=True)
class SymmetryVertex:
    particles: tuple[int, ...]
    orders: tuple[tuple[str, int], ...]
    signature: Hashable
    """Structure and coupling that a symmetry must preserve."""
    chiral: bool = False


@dataclass(frozen=True, slots=True)
class _AllowedVertices:
    by_particles: Mapping[tuple[int, ...], Counter[Hashable]]
    multisets: frozenset[tuple[int, ...]]
    chiral: tuple[tuple[int, ...], ...]


def _orders_allowed(
    orders: Iterable[tuple[str, int]],
    limits: Mapping[str, int],
) -> bool:
    return all(
        int(value) <= limits.get(str(name).upper(), int(value))
        for name, value in orders
    )


_PROJECTOR = re.compile(r"(proj)([mp])", re.IGNORECASE)
_EXCHANGED_PROJECTOR = {"m": "p", "p": "m", "M": "P", "P": "M"}


@functools.cache
def _lorentz_is_chiral(expression: str) -> bool:
    if "gamma5" in expression.casefold():
        return True
    if _PROJECTOR.search(expression) is None:
        return False
    from ..models import compiler_symbolica as _sym

    _sym._ensure_symbolica()
    exchanged = _PROJECTOR.sub(
        lambda match: match.group(1) + _EXCHANGED_PROJECTOR[match.group(2)],
        expression,
    )
    try:
        return bool(
            _sym.E(expression).expand().to_canonical_string()
            != _sym.E(exchanged).expand().to_canonical_string()
        )
    except Exception:  # an unparsable structure is not assumed vector-like
        return True


@dataclass(slots=True)
class VertexSymmetryTable:
    """Model vertices and antiparticles, indexed per set of coupling limits."""

    vertices: Sequence[SymmetryVertex]
    antiparticles: Mapping[int, int]
    _allowed: dict[tuple[tuple[str, int], ...], _AllowedVertices] = field(
        default_factory=dict
    )

    @classmethod
    def from_compiled_ir(cls, model: CompiledModelIR) -> VertexSymmetryTable:
        by_name = {particle.name: particle for particle in model.particles}
        couplings = {
            coupling.name: coupling.resolved_expression for coupling in model.couplings
        }
        vertices = [
            SymmetryVertex(
                particles=tuple(by_name[name].pdg_code for name in term.particles),
                orders=tuple(term.coupling_orders),
                signature=(
                    term.lorentz_expression,
                    term.color_expression,
                    couplings.get(term.coupling, term.coupling_expression),
                ),
                chiral=_lorentz_is_chiral(term.lorentz_expression),
            )
            for term in model.vertex_terms
        ]
        antiparticles = {
            particle.pdg_code: by_name[particle.antiname].pdg_code
            for particle in model.particles
            if particle.antiname in by_name
        }
        return cls(vertices, antiparticles)

    @classmethod
    def from_builtin_model(cls, model: Model) -> VertexSymmetryTable:
        vertices = []
        for vertex in model.vertices:
            rule = model.vertex_lowering_rule(vertex.kind)
            vertices.append(
                SymmetryVertex(
                    particles=tuple(vertex.particles),
                    orders=tuple(model.vertex_coupling_orders(vertex)),
                    signature=(vertex.kind, tuple(vertex.coupling)),
                    # Built-in electroweak fermion currents carry (left, right)
                    # couplings.
                    chiral=rule.expression_head == "fermion_gauge_weyl_current"
                    and vertex.coupling[0] != vertex.coupling[1],
                )
            )
        antiparticles = {
            pdg: particle.anti_pdg for pdg, particle in model.particles.items()
        }
        return cls(vertices, antiparticles)

    def _antiparticle(self, pdg: int) -> int:
        return self.antiparticles.get(pdg, -pdg)

    def allowed(self, limits: Mapping[str, int]) -> _AllowedVertices:
        key = tuple(
            sorted((str(name).upper(), int(value)) for name, value in limits.items())
        )
        cached = self._allowed.get(key)
        if cached is not None:
            return cached
        upper = dict(key)
        by_particles: dict[tuple[int, ...], Counter[Hashable]] = {}
        chiral: dict[tuple[int, ...], None] = {}
        for vertex in self.vertices:
            if not _orders_allowed(vertex.orders, upper):
                continue
            by_particles.setdefault(vertex.particles, Counter())[vertex.signature] += 1
            if vertex.chiral:
                chiral.setdefault(vertex.particles, None)
        result = _AllowedVertices(
            by_particles=by_particles,
            multisets=frozenset(tuple(sorted(particles)) for particles in by_particles),
            chiral=tuple(chiral),
        )
        self._allowed[key] = result
        return result

    def relabelling(
        self,
        member: GroupingProcess,
        representative: GroupingProcess,
        mapping: LegMapping,
        table: ParticleTable,
    ) -> dict[int, int]:
        """The particle permutation induced by ``mapping``, conjugation aside."""

        partial: dict[int, int] = {}
        for representative_leg, member_leg in enumerate(mapping.permutation):
            source = member.pdgs[member_leg]
            if mapping.conjugated:
                source = self._antiparticle(source)
            target = representative.pdgs[representative_leg]
            for one, other in (
                (source, target),
                (self._antiparticle(source), self._antiparticle(target)),
            ):
                if partial.setdefault(one, other) != other:
                    raise ValueError("leg mapping is not a flavour relabelling")
        if len(set(partial.values())) != len(partial):
            raise ValueError("leg mapping is not a flavour relabelling")
        # Complete the injection to a permutation: particles that are only
        # images receive the particles that are only sources, type by type.
        sources = sorted(
            (pdg for pdg in partial if pdg not in partial.values()),
            key=lambda pdg: (table[pdg].type, abs(pdg), pdg),
        )
        images = sorted(
            (pdg for pdg in partial.values() if pdg not in partial),
            key=lambda pdg: (table[pdg].type, abs(pdg), pdg),
        )
        partial.update(zip(images, sources, strict=True))
        return {
            source: target for source, target in partial.items() if source != target
        }

    def _permutation_reason(
        self,
        permutation: Mapping[int, int],
        allowed: _AllowedVertices,
        table: ParticleTable,
    ) -> str | None:
        for particles, signatures in allowed.by_particles.items():
            image = tuple(permutation.get(pdg, pdg) for pdg in particles)
            if image != particles and allowed.by_particles.get(image) != signatures:
                moved = ", ".join(
                    f"{_name(table, source)}->{_name(table, target)}"
                    for source, target in sorted(permutation.items())
                    if source > 0 or table[source].type.self_conjugate
                )
                return (
                    f"relabelling {moved} changes the allowed vertex "
                    f"{_vertex_text(table, particles)}"
                )
        return None

    def _conjugation_reason(
        self,
        allowed: _AllowedVertices,
        flavours: Iterable[int],
        table: ParticleTable,
    ) -> str | None:
        if allowed.chiral:
            # Prefer an example that involves the flavours of the process.
            present = set(flavours)
            chiral = next(
                (
                    particles
                    for particles in allowed.chiral
                    if present & {abs(pdg) for pdg in particles}
                ),
                allowed.chiral[0],
            )
            return (
                "charge conjugation is broken by the chiral allowed vertex "
                f"{_vertex_text(table, chiral)}"
            )
        missing = next(
            (
                particles
                for particles in allowed.multisets
                if tuple(sorted(self._antiparticle(pdg) for pdg in particles))
                not in allowed.multisets
            ),
            None,
        )
        if missing is not None:
            return (
                "charge conjugation has no allowed image of the vertex "
                f"{_vertex_text(table, missing)}"
            )
        return None

    def relation_warning(
        self,
        member: GroupingProcess,
        representative: GroupingProcess,
        mapping: LegMapping,
        table: ParticleTable,
        *,
        limits: Mapping[str, int],
    ) -> str | None:
        """Why ``mapping`` may not be a symmetry, or ``None``."""

        allowed = self.allowed(limits)
        relation = (
            f"{member.expression!r} assumed equal to {representative.expression!r}"
        )
        permutation = self.relabelling(member, representative, mapping, table)
        reasons = [self._permutation_reason(permutation, allowed, table)]
        if mapping.conjugated:
            reasons.append(
                self._conjugation_reason(
                    allowed, {abs(pdg) for pdg in representative.pdgs}, table
                )
            )
        return _warning(relation, reasons, limits)

    def flavour_union_warning(
        self,
        template: GroupingProcess,
        flavours: Collection[int],
        table: ParticleTable,
        *,
        limits: Mapping[str, int],
        conjugated: bool = False,
    ) -> str | None:
        """Why merging ``flavours`` into one may change some pairing pattern of
        ``template``, or ``None``.

        Every allowed vertex must hold each union flavour an even number of
        times (vertex PDG signs follow model conventions), and exchanging
        any union flavour with the smallest one must map the allowed vertices
        onto themselves; ``conjugated`` also requires charge conjugation.
        """

        allowed = self.allowed(limits)
        lowest = min(flavours)
        reasons: list[str | None] = []
        changing = next(
            (
                particles
                for particles in allowed.by_particles
                if any(
                    sum(abs(pdg) == flavour for pdg in particles) % 2
                    for flavour in flavours
                )
            ),
            None,
        )
        if changing is not None:
            reasons.append(
                "the allowed vertex "
                f"{_vertex_text(table, changing)} changes a union quark flavour"
            )
        for flavour in sorted(set(flavours) - {lowest}):
            exchange = {flavour: lowest, lowest: flavour}
            exchange[self._antiparticle(flavour)] = self._antiparticle(lowest)
            exchange[self._antiparticle(lowest)] = self._antiparticle(flavour)
            reason = self._permutation_reason(exchange, allowed, table)
            if reason is not None:
                reasons.append(reason)
                break
        if conjugated:
            reasons.append(self._conjugation_reason(allowed, flavours, table))
        names = " ".join(_name(table, flavour) for flavour in sorted(flavours))
        relation = f"flavour union {template.expression!r} of {names}"
        return _warning(relation, reasons, limits)


def _name(table: ParticleTable, pdg: int) -> str:
    particle = table.get(pdg)
    return particle.name if particle is not None else str(pdg)


def _vertex_text(table: ParticleTable, particles: Sequence[int]) -> str:
    return "[" + " ".join(_name(table, pdg) for pdg in particles) + "]"


def _warning(
    relation: str,
    reasons: Sequence[str | None],
    limits: Mapping[str, int],
) -> str | None:
    found = [reason for reason in reasons if reason is not None]
    if not found:
        return None
    limit_text = ", ".join(f"{key}<={value}" for key, value in sorted(limits.items()))
    return (
        f"{relation}, but {' and '.join(found)} "
        f"(coupling orders: {limit_text or 'unrestricted'})"
    )


__all__ = ["SymmetryVertex", "VertexSymmetryTable"]
