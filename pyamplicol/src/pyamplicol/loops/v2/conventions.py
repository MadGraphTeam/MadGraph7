# SPDX-License-Identifier: 0BSD
"""Born-loop conventions of the v2 engine (design §3.3, §6.4, §11.1, WP1.8).

Everything here fixes how the separately built pieces (tree blocks, loop
segments, seeds, counterterm sites, NinjaRS integrals) combine into one
physical virtual. The rules were derived for the tree engine's conventions and
are checked against the frozen legacy per-family numerators (O2) at 200 bits,
family by family, for gg->gg, uu~->dd~, uu~->dd~g, uu~->ggg, uu~->uu~ and
uu~->uu~g.

Amplitudes. Block rows are the tree runtime's currents; ``Block.wick_sign``
turns each into its Feynman-rule current, signed as FeynKit signs diagrams
(coloured fermion lines listed (fundamental, antifundamental) against ascending
labels, the open end last). The Born is closed with the same rows at the model
vertex that contains leg 1 (``ct_sites.SiteClosure``), so Born, counterterm
sites and loop skeletons share one phase: the tree engine's i-factors (vertex
and propagator factors of the UFO rules) are in the rows and kernels, and the
loop integral needs no further phase.

Fermion legs of loop segments are read as the tree engine reads a vertex: the
current of tree particle p fills the UFO leg named p (``tree_reading_legs``).
The families and the colour seeds read the conjugate ("incoming") legs; for
spinor legs the two readings transpose the Dirac chain, and only the tree
reading matches the tree runtime's fermion rows (bosons are self-conjugate
and unaffected).

Leaf weight of a skeleton (``LeafWeights``), design §3.3:

    w_s = (automorphism weight) x (merged-flavour multiplicity) x eps_s,

with eps_s the product of

* the parity of the full coloured fermion-line listing of the loop diagram over
  the product of its blocks' own listings (the blocks already carry theirs):
  lines through the loop join the open ends of the two blocks where the loop
  line turns fermionic and back; the identity pairing inside each block
  represents its trees, as for counterterm sites;
* -1 per closed fermion or ghost loop (Wick sewing of the cut pair);
* -1 per fermion propagator on the loop. The tree reading of a fermion line
  is its transposed Dirac chain: every vertex tensor is contracted on its
  swapped spinor slots and every propagator is the antifermion propagator
  S(-k)^T of the momentum k the line carries (``models.expressions``), so the
  chain equals the families' reading with S(-k) = -(k-slash - m) in place of
  S(k) = k-slash + m. No property of the vertex tensors enters (no
  charge-conjugation matrix, unlike Furry's theorem), so the sign holds for
  every Lorentz structure. Vector and chiral currents are checked against
  MadLoop with the Z on open quark lines (Drell-Yan equals the universal
  massless form factor) and on closed quark loops (u u~ > e- e+ g g; reversing
  the top's axial coupling moves the finite part by 6e-4 to 7e-3); scalar and
  tensor couplings follow from the same identity, without a process oracle
  yet (DESIGN WP3.1). The tree reading therefore also evaluates the line's
  propagator with the opposite mass sign (``TreeReading.mass_signs``). The
  flip acts on the mass parameter in the whole segment kernel (vertex and
  propagator), so a vertex whose couplings depend on the mass of the loop
  line it emits is refused (a Yukawa written as MT/v rather than through its
  own parameter).

The tree engine's ``merge_sign`` (``generation/fermion_ordering.py``) is +1
for coloured lines, whose ordering the colour flow owns, so for coloured
fermions the rule above is the design §3.3 ring sign. It is refused
(``FermionSignError``, at generation) wherever it is not established:
colour-singlet fermion lines through the loop (they need the Koszul signs of
``fermion_ordering.py``), vertices with more than one fermion line, and loop
fermion vertices coupling through the mass of the line they emit (above).

Normalisation. NinjaRS returns mu^(2 eps) / (i pi^(2-eps) r_Gamma) times the
integral of N / prod D (design §11.1), so with Born-weighted family numerators
the bare virtual of a squared-order sector is

    bare = 2 Re( i / (16 pi^2) sum_f Omega_f ) / average,

``average`` being the initial spin and colour average times the identical
final-state factorials (``loops.request``), in the MadLoop/BLHA C_eps
convention of the providers. Counterterm couplings already carry the loop
factor (design §6.4).
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

from pyamplicol.generation.fermion_ordering import _fermion_rep_requires_exterior_sign

from .blocks import _open_last, _parity, identity_pairing_sign
from .families import ZERO, mass_name, model_data
from .skeletons import Content, Skeleton

GHOST_SPIN = -1
FERMION_SPIN = 2

_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

# Loop factor of the reducer's integrals: d^4q/(2 pi)^4 = i/(16 pi^2) x Omega.
LOOP_FACTOR_DENOMINATOR = 16


class FermionSignError(NotImplementedError):
    """A fermion line the slice's sign rule does not cover."""


@dataclass(frozen=True, slots=True)
class Normalisation:
    """Divisor of every squared quantity: initial averages times final factorials."""

    average: int

    @classmethod
    def of(
        cls, native_model: Any, incoming: Sequence[Any], outgoing: Sequence[Any]
    ) -> Normalisation:
        from pyamplicol.loops.request import prepare_loop_external_request

        request = prepare_loop_external_request(native_model, incoming, outgoing)
        average = Fraction(str(request.initial_average))
        if average.denominator != 1 or average <= 0:
            raise ValueError(f"unexpected normalisation divisor {average}")
        return cls(int(average))


def tree_reading_legs(
    particles: Sequence[str],
    legs: Sequence[int],
    spins: Mapping[str, int],
    antinames: Mapping[str, str],
) -> tuple[int, ...]:
    """The UFO legs of one vertex reading with fermion legs read as the tree
    engine reads them (module docstring).

    ``legs`` are 1-based UFO legs in the incoming reading (loop in, loop out,
    blocks); each fermion leg is exchanged with the other leg of its fermion
    line. A vertex with two fermion legs has one line, whatever their
    flavours (a W vertex joins t and b): its two legs are exchanged, as the
    tree engine reads the vertex through its conjugate rule. With more
    fermion legs each pairs with the leg of its conjugate particle.
    """

    fermions = [leg for leg in legs if spins[particles[leg - 1]] == FERMION_SPIN]
    if len(fermions) == 2:
        first, second = fermions
        swap = {first: second, second: first}
        return tuple(swap.get(leg, leg) for leg in legs)
    swap = {}
    for leg in fermions:
        partners = [
            other
            for other in fermions
            if other != leg and particles[other - 1] == antinames[particles[leg - 1]]
        ]
        if len(partners) != 1:
            raise FermionSignError(
                f"vertex {tuple(particles)} has more than one fermion line"
            )
        swap[leg] = partners[0]
    return tuple(swap.get(leg, leg) for leg in legs)


class TreeReading:
    """``reference_executor.CatalogueKinematics`` leg hook for tree rows."""

    def __init__(self, native_model: Any):
        data = model_data(native_model)
        self._particles = {
            rule["name"]: rule["particles"] for rule in data["vertex_rules"]
        }
        self._spins = {p["name"]: int(p["spin"]) for p in data["particles"]}
        self._antinames = {p["name"]: p["antiname"] for p in data["particles"]}
        self._masses = {p["name"]: mass_name(p["mass"]) for p in data["particles"]}

    def __call__(self, rule: str, legs: tuple[int, ...]) -> tuple[int, ...]:
        return tree_reading_legs(
            self._particles[rule], legs, self._spins, self._antinames
        )

    def mass_signs(self, loop_out: str) -> Mapping[str, int]:
        """Mass parameters whose sign the tree reading flips in a segment
        whose loop line leaves as ``loop_out`` (module docstring)."""

        mass = self._masses[loop_out]
        if self._spins[loop_out] != FERMION_SPIN or mass == ZERO:
            return {}
        return {mass: -1}


class LeafWeights:
    """Leaf weights w_s of skeletons (module docstring), a ``LeafWeight``.

    ``ends`` maps the label of every coloured external fermion to its
    all-outgoing representation (+3 or -3), as ``SiteClosure`` records them.
    """

    def __init__(self, native_model: Any, ends: Mapping[int, int]):
        data = model_data(native_model)
        self._particles = {p["name"]: p for p in data["particles"]}
        self._rules = {rule["name"]: rule for rule in data["vertex_rules"]}
        self._expressions = {
            entry["name"]: entry.get("expression") or ""
            for entry in (*data["parameters"], *data["couplings"])
        }
        self._ends = dict(ends)
        self._signs: dict[tuple[Content, ...], int] = {}
        self._depends: dict[str, frozenset[str]] = {}

    def __call__(self, skeleton: Skeleton, contents: Sequence[Content]) -> Fraction:
        sign = self._signs.get(skeleton.contents)
        if sign is None:
            sign = self._signs[skeleton.contents] = self.fermion_sign(skeleton)
        return skeleton.weight * skeleton.multiplicity * sign

    def fermion_sign(self, skeleton: Skeleton) -> int:
        """eps_s of a skeleton (reading-independent)."""

        self._require_mass_free_couplings(skeleton.contents)
        spins = [int(self._particles[name]["spin"]) for name in skeleton.species]
        sign = -1 if all(spin == GHOST_SPIN for spin in spins) else 1
        for content in skeleton.contents:
            sign *= self.propagator_sign(content)
        return sign * self._line_parity(skeleton.contents)

    def propagator_sign(self, content: Content) -> int:
        """The factor of eps_s owned by the loop propagator after a position:
        -1 for a fermion (the transposed tree reading, module docstring).
        Schedule M carries it on the position's dressing step."""

        return -1 if self._fermion(content.loop_out) else 1

    def _fermion(self, name: str) -> bool:
        return int(self._particles[name]["spin"]) == FERMION_SPIN

    def _require_mass_free_couplings(self, contents: Sequence[Content]) -> None:
        """Refuse a loop fermion vertex coupling through the mass of the loop
        line it emits (module docstring: the tree reading flips that mass)."""

        for content in contents:
            if not self._fermion(content.loop_out):
                continue
            mass = mass_name(self._particles[content.loop_out]["mass"])
            if mass == ZERO:
                continue
            for term in content.terms:
                rule = term.vertex.rule
                if any(
                    mass in self._dependencies(coupling)
                    for row in self._rules[rule]["couplings"]
                    for coupling in row
                ):
                    raise FermionSignError(
                        f"loop fermion vertex {rule} couples through the mass "
                        f"{mass} of the loop line it emits; the tree reading "
                        "flips that mass in the whole segment kernel"
                    )

    def _dependencies(self, name: str) -> frozenset[str]:
        """Every model parameter a coupling or parameter depends on."""

        found = self._depends.get(name)
        if found is None:
            self._depends[name] = frozenset()  # model parameters are acyclic
            direct = set(_IDENTIFIER.findall(self._expressions.get(name, "")))
            found = frozenset(direct & self._expressions.keys()).union(
                *(
                    self._dependencies(other)
                    for other in direct & self._expressions.keys()
                )
            )
            self._depends[name] = found
        return found

    def _line_parity(self, contents: Sequence[Content]) -> int:
        """Pairing parity of the coloured lines and closed-loop sewing signs."""

        sign = 1
        pairs: list[tuple[int, int]] = []
        links: dict[tuple, list[tuple]] = {}
        open_ends: dict[tuple, tuple[int, int]] = {}

        def link(a: tuple, b: tuple) -> None:
            links.setdefault(a, []).append(b)
            links.setdefault(b, []).append(a)

        count = len(contents)
        for position, content in enumerate(contents):
            if len(content.terms) != 1:
                raise ValueError("leaf weights need one routing term per content")
            ports = []
            if self._fermion(content.loop_in):
                ports.append(("in", position))
            if self._fermion(content.loop_out):
                ports.append(("out", position))
            for slot, block in enumerate(content.terms[0].blocks):
                labels = [
                    label for label in self._ends if block.mask >> (label - 1) & 1
                ]
                starts = [label for label in labels if self._ends[label] == 3]
                stops = [label for label in labels if self._ends[label] == -3]
                entry = self._particles[block.field]
                if int(entry["spin"]) == FERMION_SPIN:
                    if _fermion_rep_requires_exterior_sign(int(entry["color"])):
                        raise FermionSignError(
                            "a colour-singlet fermion line through the loop needs "
                            "the Koszul signs of generation/fermion_ordering.py"
                        )
                    (starts if int(entry["color"]) == 3 else stops).append(None)
                sign *= identity_pairing_sign(starts, stops)
                for start, stop in zip(
                    sorted(starts, key=_open_last),
                    sorted(stops, key=_open_last),
                    strict=True,
                ):
                    port = ("block", position, slot)
                    if start is None:
                        open_ends[port] = (3, stop)
                        ports.append(port)
                    elif stop is None:
                        open_ends[port] = (-3, start)
                        ports.append(port)
                    else:
                        pairs.append((start, stop))
            if len(ports) not in (0, 2):
                raise FermionSignError("a loop vertex joins more than one fermion line")
            if ports:
                link(*ports)
            if self._fermion(content.loop_out):
                link(("out", position), ("in", (position + 1) % count))
        seen: set[tuple] = set()
        for start in links:
            if start in seen:
                continue
            component, stack = [], [start]
            while stack:
                port = stack.pop()
                if port not in seen:
                    seen.add(port)
                    component.append(port)
                    stack.extend(links[port])
            ends = [open_ends[port] for port in component if port in open_ends]
            if not ends:
                sign = -sign  # a closed fermion loop
                continue
            partners = dict(ends)
            if len(ends) != 2 or set(partners) != {3, -3}:
                raise FermionSignError("an open fermion line without two ends")
            # The open end of a fundamental-type block faces an antifundamental
            # endpoint: the joined line runs from partners[-3] to partners[3].
            pairs.append((partners[-3], partners[3]))
        return sign * _parity([label for pair in pairs for label in pair])


__all__ = [
    "LOOP_FACTOR_DENOMINATOR",
    "FermionSignError",
    "LeafWeights",
    "Normalisation",
    "TreeReading",
    "tree_reading_legs",
]
