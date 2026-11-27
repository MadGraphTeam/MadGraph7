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
* Charge conjugation with unchanged momenta and helicities (C, not CP) is
  accepted when no allowed vertex couples fermions chirally (a Lorentz
  structure that changes when its left- and right-handed projectors are
  exchanged, or contains ``gamma5``; a built-in electroweak current with
  different left- and right-handed couplings) and C maps the allowed vertices
  onto themselves. For compiled models every allowed vertex, self-conjugate
  ones such as ``(q~, q, g)`` included, is transformed explicitly: particles
  go to their antiparticles (legs relabelled into the model's ordering of the
  image), a fermion pair at legs 1 and 2 has its chain transposed (C M^T C^-1:
  ``Gamma`` and ``Sigma`` odd, ``ProjM``/``ProjP``/``Gamma5``/``Identity``
  even), each ``f`` is odd (``d`` and ``Identity`` even), and self-conjugate
  fields carry one fixed C parity: -1 for vectors and ghosts, +1 for scalars.
  The image vertex must then have the identical Lorentz x colour x coupling
  sum. Any other structure (Levi-Civita, ``T`` without a fermion pair at legs
  1 and 2, spin 3/2 or 2, Majorana fermions) is refused. The fixed parities
  make the gate sufficient, not necessary: a model that is C symmetric only
  with other parities is warned about. The built-in SM compares conjugate
  vertices' couplings and accepts its self-conjugate vertices by kind (its
  vector-like gauge, gluon and scalar couplings). Hermiticity is not C: the
  conjugate coupling is not compared with the complex conjugate (UFO
  couplings carry the Feynman factor ``i``).

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
from itertools import permutations

from .._internal.physics.symbols import symbols as symbol_registry
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
    c_term: tuple[str, str, str] | None = None
    """Compiled IR: UFO Lorentz and colour structure and coupling symbol."""
    c_signature: Hashable = None
    """Built-in SM: what the C image must repeat; ``None`` means ``signature``."""
    c_even: bool = True
    """Built-in SM: a known C-even kind, if the particles are self-conjugate."""


@dataclass(frozen=True, slots=True)
class _AllowedVertices:
    by_particles: Mapping[tuple[int, ...], Counter[Hashable]]
    multisets: frozenset[tuple[int, ...]]
    chiral: tuple[tuple[int, ...], ...]
    c_signatures: Mapping[tuple[int, ...], Counter[Hashable]] = field(
        default_factory=dict
    )
    c_unknown: frozenset[tuple[int, ...]] = frozenset()
    c_terms: Mapping[tuple[int, ...], tuple[tuple[str, str, str], ...]] = field(
        default_factory=dict
    )
    c_reason: dict[str, tuple[str, tuple[int, ...]] | None] = field(
        default_factory=dict, compare=False
    )


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


# Spinor matrices and their sign under C M^T C^-1 (``_c_structure``).
_SPINOR_C_SIGN = {"Gamma": -1, "Sigma": -1, "Gamma5": 1, "ProjM": 1, "ProjP": 1}
_TENSOR_CALL = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\(([^()]*)\)")
_INTEGERS = re.compile(r"-?\d+(,-?\d+)*")
_BUILTIN_C_EVEN_HEADS = frozenset(
    {
        # Self-conjugate vertex kinds of the built-in SM: vector-like gauge
        # currents (C-odd vectors), gluon self-couplings and scalar couplings.
        "fermion_gauge_weyl_current",
        "quark_gluon_weyl_current",
        "qcd_u1_subtraction_current",
        "three_gluon_current",
        symbol_registry.qualified_name("two_gluon_to_tensor"),
        symbol_registry.qualified_name("tensor_gluon_to_gluon"),
        symbol_registry.qualified_name("gluon_tensor_to_gluon"),
        "three_vector_current",
        "two_vector_to_tensor",
        "two_vector_to_scalar",
        "scalar_vector_to_vector",
        "vector_scalar_to_vector",
        "two_scalar_to_scalar",
    }
)


def _ufo_structure(source: str) -> str:
    """A compiled UFO structure source in UFO's own integer-index notation."""

    text = source.replace("UFO::{}::", "").replace("UFO::", "")
    text = re.sub(r"\bidx\(1,(\d+)\)", r"\1", text)
    return re.sub(r"\bdummy\((\d+)\)", r"-\1", text).replace(" ", "")


def _sorted_arguments(
    arguments: list[int], antisymmetric: bool
) -> tuple[list[int], int]:
    """Sorted arguments and the sort's sign (0: repeated antisymmetric index)."""

    sign = 1
    if antisymmetric:
        if len(set(arguments)) != len(arguments):
            return arguments, 0
        for left in range(len(arguments)):
            for right in range(left + 1, len(arguments)):
                if arguments[left] > arguments[right]:
                    sign = -sign
    return sorted(arguments), sign


@functools.cache
def _c_structure(
    text: str, color: bool, legs: tuple[int, ...], mode: int
) -> str | None:
    """A UFO structure, leg ``i`` relabelled ``legs[i-1]``, symmetric tensors sorted;
    ``mode`` 1/2 applies C to a bosonic/fermion-pair term; ``None`` if unknown."""

    failed = False

    def replace(match: re.Match[str]) -> str:
        nonlocal failed
        name, raw = match.group(1), match.group(2)
        if _INTEGERS.fullmatch(raw) is None:
            failed = True
            return "0"
        arguments = [int(value) for value in raw.split(",")]
        if name == "complex" and not color:
            return match.group(0)
        if any(value > len(legs) for value in arguments):
            failed = True
            return "0"
        sign = 1
        if not color and (name in _SPINOR_C_SIGN or name == "Identity"):
            if mode == 1 or len(arguments) < 2:
                failed = True
            elif mode == 2:
                arguments[-2], arguments[-1] = arguments[-1], arguments[-2]
                sign = _SPINOR_C_SIGN.get(name, 1)
        elif color and name == "T":
            if mode == 1 or len(arguments) != 3:
                failed = True
            elif mode == 2:
                arguments[1], arguments[2] = arguments[2], arguments[1]
        elif (not color and name not in {"Metric", "P"}) or (
            color and name not in {"f", "d", "Identity"}
        ):
            failed = True
        arguments = [legs[value - 1] if value > 0 else value for value in arguments]
        if name in {"Metric", "Identity", "d", "f"} and (color or name == "Metric"):
            arguments, parity = _sorted_arguments(arguments, antisymmetric=name == "f")
            sign *= parity * (-1 if name == "f" and mode else 1)
        family = "color" if color else "lorentz"
        call = f"cg_{family}_{name}({','.join(map(str, arguments))})"
        return call if sign == 1 else f"({sign}*{call})"

    result = _TENSOR_CALL.sub(replace, text)
    if failed or re.search(r"[A-Za-z_]\w*\(", _TENSOR_CALL.sub("", result)):
        return None
    return result


@functools.cache
def _canonical(expression: str) -> str:
    from ..models import compiler_symbolica as _sym

    _sym._ensure_symbolica()
    return str(_sym.E(expression).expand().to_canonical_string())


def _vertex_sum(
    terms: Iterable[tuple[str, str, str]], legs: tuple[int, ...], mode: int, sign: int
) -> str | None:
    """Canonical sum of Lorentz x colour x coupling over a vertex's terms."""

    parts = []
    for lorentz, color, coupling in terms:
        lorentz_text = _c_structure(lorentz, False, legs, mode)
        color_text = _c_structure(color, True, legs, mode)
        if lorentz_text is None or color_text is None:
            return None
        parts.append(f"({lorentz_text})*({color_text})*{coupling}")
    try:
        return _canonical(f"({sign})*(" + "+".join(parts) + ")")
    except Exception:  # an unparsable structure is not assumed C symmetric
        return None


@dataclass(slots=True)
class VertexSymmetryTable:
    """Model vertices and antiparticles, indexed per set of coupling limits."""

    vertices: Sequence[SymmetryVertex]
    antiparticles: Mapping[int, int]
    c_fields: Mapping[int, tuple[int, bool]] | None = None
    """Compiled IR: UFO spin code and ghost flag per PDG (C transform)."""
    _allowed: dict[tuple[tuple[str, int], ...], _AllowedVertices] = field(
        default_factory=dict
    )

    @classmethod
    def from_compiled_ir(cls, model: CompiledModelIR) -> VertexSymmetryTable:
        by_name = {particle.name: particle for particle in model.particles}
        couplings = {
            coupling.name: coupling.resolved_expression for coupling in model.couplings
        }
        # Couplings enter the C comparison as opaque symbols, one per expression.
        symbols: dict[str, str] = {}
        vertices = []
        for term in model.vertex_terms:
            coupling = couplings.get(term.coupling, term.coupling_expression)
            symbol = symbols.setdefault(coupling, f"cg_coupling_{len(symbols)}")
            vertices.append(
                SymmetryVertex(
                    particles=tuple(by_name[name].pdg_code for name in term.particles),
                    orders=tuple(term.coupling_orders),
                    signature=(
                        term.lorentz_expression,
                        term.color_expression,
                        coupling,
                    ),
                    chiral=_lorentz_is_chiral(term.lorentz_expression),
                    c_term=(
                        _ufo_structure(term.lorentz_source),
                        _ufo_structure(term.color_source),
                        symbol,
                    ),
                )
            )
        antiparticles = {
            particle.pdg_code: by_name[particle.antiname].pdg_code
            for particle in model.particles
            if particle.antiname in by_name
        }
        fields = {
            particle.pdg_code: (
                int(particle.spin),
                int(getattr(particle, "ghost_number", 0)) != 0
                or int(particle.spin) < 0,
            )
            for particle in model.particles
        }
        return cls(vertices, antiparticles, fields)

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
                    # Kinds are oriented current rules whose conjugates the model
                    # pairs by construction: C compares the coupling only.
                    c_signature=tuple(vertex.coupling),
                    c_even=rule.expression_head in _BUILTIN_C_EVEN_HEADS,
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
        c_signatures: dict[tuple[int, ...], Counter[Hashable]] = {}
        c_unknown: set[tuple[int, ...]] = set()
        c_terms: dict[tuple[int, ...], list[tuple[str, str, str]]] = {}
        for vertex in self.vertices:
            if not _orders_allowed(vertex.orders, upper):
                continue
            by_particles.setdefault(vertex.particles, Counter())[vertex.signature] += 1
            c_signatures.setdefault(vertex.particles, Counter())[
                vertex.signature if vertex.c_signature is None else vertex.c_signature
            ] += 1
            if not vertex.c_even:
                c_unknown.add(vertex.particles)
            if vertex.c_term is not None:
                c_terms.setdefault(vertex.particles, []).append(vertex.c_term)
            if vertex.chiral:
                chiral.setdefault(vertex.particles, None)
        result = _AllowedVertices(
            by_particles=by_particles,
            multisets=frozenset(tuple(sorted(particles)) for particles in by_particles),
            chiral=tuple(chiral),
            c_signatures=c_signatures,
            c_unknown=frozenset(c_unknown),
            c_terms={particles: tuple(terms) for particles, terms in c_terms.items()},
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
        return self._c_image_reason(allowed, table)

    def _c_image_reason(
        self, allowed: _AllowedVertices, table: ParticleTable
    ) -> str | None:
        """The C condition per allowed vertex (see the module docstring)."""

        if "c" not in allowed.c_reason:
            allowed.c_reason["c"] = (
                self._builtin_c_failure(allowed)
                if self.c_fields is None
                else next(
                    (
                        (failure, particles)
                        for particles in allowed.c_terms
                        if (failure := self._compiled_c_failure(particles, allowed))
                    ),
                    None,
                )
            )
        found = allowed.c_reason["c"]
        if found is None:
            return None
        failure, particles = found
        vertex = _vertex_text(table, particles)
        return {
            "levi-civita": "charge conjugation is not checked for the Levi-Civita "
            f"structure of the allowed vertex {vertex}",
            "unchecked": "charge conjugation cannot be checked for the structure "
            f"of the allowed vertex {vertex}",
            "changed": "charge conjugation changes the coupling or structure of "
            f"the allowed vertex {vertex}",
        }[failure]

    def _builtin_c_failure(
        self, allowed: _AllowedVertices
    ) -> tuple[str, tuple[int, ...]] | None:
        for particles, signatures in allowed.c_signatures.items():
            image = tuple(self._antiparticle(pdg) for pdg in particles)
            if sorted(image) == sorted(particles):
                if particles in allowed.c_unknown:
                    return "unchecked", particles
                continue
            # Some allowed ordering of the image repeats the coupling signatures.
            if not any(
                allowed.c_signatures[other] == signatures
                for other in allowed.c_signatures
                if sorted(other) == sorted(image)
            ):
                return "changed", particles
        return None

    def _compiled_c_failure(
        self, particles: tuple[int, ...], allowed: _AllowedVertices
    ) -> str | None:
        fields = self.c_fields or {}
        terms = allowed.c_terms[particles]
        if any("Epsilon(" in lorentz for lorentz, _, _ in terms):
            return "levi-civita"
        if any(pdg not in fields for pdg in particles):
            return "unchecked"
        spins = [fields[pdg][0] for pdg in particles]
        ghosts = [fields[pdg][1] for pdg in particles]
        # Ghosts are fixed by C; physical particles go to their antiparticles.
        image = [
            pdg if ghost else self._antiparticle(pdg)
            for pdg, ghost in zip(particles, ghosts, strict=True)
        ]
        fermions = [
            index
            for index, (spin, ghost) in enumerate(zip(spins, ghosts, strict=True))
            if not ghost and spin % 2 == 0
        ]
        if fermions not in ([], [0, 1]) or any(
            not ghost and spin not in (1, 2, 3)
            for spin, ghost in zip(spins, ghosts, strict=True)
        ):
            return "unchecked"
        sign = 1
        for pdg, target, spin, ghost in zip(
            particles, image, spins, ghosts, strict=True
        ):
            if target == pdg:
                if spin == 2:  # a Majorana fermion
                    return "unchecked"
                sign *= -1 if ghost or spin == 3 else 1
        # The C image of the fermion bilinear swaps the pair's legs.
        moved = list(range(len(particles)))
        if fermions:
            image[0], image[1] = image[1], image[0]
            moved[0], moved[1] = 1, 0
        for other, other_terms in allowed.c_terms.items():
            if sorted(other) != sorted(image):
                continue
            target_sum = _vertex_sum(other_terms, tuple(range(1, len(other) + 1)), 0, 1)
            # Every placement among identical particles must agree, so a vertex
            # that is not Bose symmetric is not accepted by a lucky relabelling.
            conjugated_sums = {
                _vertex_sum(
                    terms,
                    tuple(order[moved[leg]] + 1 for leg in range(len(particles))),
                    2 if fermions else 1,
                    sign,
                )
                for order in permutations(range(len(other)))
                if all(other[order[slot]] == image[slot] for slot in range(len(image)))
                and (not fermions or order[:2] == (0, 1))
            }
            if None in conjugated_sums or target_sum is None:
                return "unchecked"
            if conjugated_sums == {target_sum}:
                return None
        return "changed"

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
