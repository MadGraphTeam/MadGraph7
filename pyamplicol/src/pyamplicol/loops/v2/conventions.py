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

Legs of loop segments are read as the tree engine reads a vertex: the
current of tree particle p fills the UFO leg named p (``TreeReading.vertex``).
The families and the colour seeds read the conjugate ("incoming") legs; for
spinor legs the two readings transpose the Dirac chain, and only the tree
reading matches the tree runtime's fermion rows (self-conjugate bosons are
unaffected). A vertex whose particles are not closed under conjugation (a W
on a loop quark line) has no leg for the tree particles of its currents: the
tree engine reads them through the model's conjugate rule, whose couplings
and Lorentz structures are the rule's own complex-conjugated with the chiral
projectors exchanged, so the segment takes the conjugate rule's kernels
(WP3.2 EWI-2; the root closure does the same, ``ct_sites.SiteClosure``).
The tree engine also reads every momentum and colour flow all-outgoing. The
segment kernel carries the difference, one rule for every vertex: the sign of
its Lorentz structure under the momentum reversal (with each boson leg on its
antiparticle's) times its colour tensor's under the transposition of the
adjoint generators (``colour_transposition_sign``). It cancels on the SM's
coloured vertices and is -1 on the momentum-odd Goldstone vertices (Z G0 H,
W G0 G+-, W G+- H) and, by the field signs of the closed vertices (G0
negated), on the ghost-G0 vertices: the loops then share the tree engine's
readings, to which the Goldstone Yukawas are matched
(``catalogue._CatalogueBuilder.reading_sign``, WP3.2 EWI-5; G2-M6).

Leaf weight of a skeleton (``LeafWeights``), design §3.3:

    w_s = (automorphism weight) x (merged-flavour multiplicity) x eps_s,

with eps_s the product of

* the parity of the full coloured fermion-line listing of the loop diagram over
  the product of its blocks' own listings (the blocks already carry theirs):
  lines through the loop join the open ends of the two blocks where the loop
  line turns fermionic and back; the identity pairing inside each block
  represents its trees, as for counterterm sites;
* the Koszul sign of the colour-singlet fermions (``colour_singlet_sign``).
  A block row carries the tree engine's Koszul signs of colour-singlet
  fermions (``generation/fermion_ordering.py``): it is the block current's
  coefficient of the product of its singlet sources in source order. The
  loop diagram is the coefficient of all of them, so eps_s carries the sign
  of concatenating the blocks' singlet supports in the order of the contents,
  times the sign that brings the open singlet ends the loop joins into
  (antiparticle, particle) pairs, the input order of the tree engine's
  two-fermion merge with sign +1 (the block of incoming field f holds the
  external end of anti(f)). Every singlet line is then listed (antiparticle,
  particle), inside the blocks and through the loop alike, so the Wick sign
  of every diagram is kept up to one sign per process, which the Born shares
  (its root closes singlet lines with the same sign, ``ct_sites.
  SiteClosure``). This is the tree engine's ``merge_sign`` composed over the
  blocks; no property of the vertices enters;
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
(``FermionSignError``, at generation) wherever it is not established: a
line joining a coloured and a colour-singlet end (the tree engine refuses
such mixed sign ownership too), vertices with more than one fermion line, and
loop fermion vertices coupling through the mass of the line they emit
(above).

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
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from types import MappingProxyType
from typing import Any

from pyamplicol.generation.fermion_ordering import (
    _external_fermion_support_sign,
    _fermion_input_orientation_sign,
)

from .blocks import _open_last, _parity, identity_pairing_sign
from .families import ZERO, add_orders, mass_name, model_data
from .skeletons import Content, Skeleton

GHOST_SPIN = -1
FERMION_SPIN = 2

_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

# Loop factor of the reducer's integrals: d^4q/(2 pi)^4 = i/(16 pi^2) x Omega.
LOOP_FACTOR_DENOMINATOR = 16


class FermionSignError(NotImplementedError):
    """A fermion line the slice's sign rule does not cover."""

    # The listed internal refusal (``refusals.LOOP_INTERNAL_REFUSALS``).
    refusal_key = "fermion-sign"


class ReadingSignError(NotImplementedError):
    """A vertex without a sign between the tree engine's reading and a loop
    segment's or root closure's: its colour tensor has no sign under the
    transposition of its adjoint generators (``colour_transposition_sign``:
    a trace of three or more adjoint indices, a sextet or epsilon tensor, an
    unknown tensor, or terms of both signs), or its Lorentz structure (the
    root closure's kernel, ``ct_sites._momentum_parity``) none under the
    momentum reversal. Generation refuses the request naming the vertex."""

    # The listed internal refusal (``refusals.LOOP_INTERNAL_REFUSALS``).
    refusal_key = "reading-sign"


# The namespace qualifier of a UFO symbol as a compiled model prints it
# (``UFO::{}::f(1,2,3)``; ``CompiledVertexTerm.color_source``).
_NAMESPACE = re.compile(r"\b[A-Za-z_]\w*::(?:\{[^{}]*\}::)?")


def colour_transposition_sign(colour: str, what: str = "a vertex") -> int:
    """The sign a UFO colour tensor takes when every adjoint generator is
    transposed (the order of every trace reversed, the ends of every
    fundamental line swapped): -1 per ``f``; 1 per ``d``, generator ``T``
    (transposed between swapped ends), ``Identity`` or two-index trace; the
    terms of a sum must share it.

    The tree engine reads every vertex with its outgoing particles' colour
    flows, loop segments and root closures with the incoming ones; the two
    differ by this sign times the Lorentz structure's under the momentum
    reversal (``catalogue._CatalogueBuilder.reading_sign``). ``colour`` is
    the UFO string, bare or namespace-qualified as a compiled model prints
    it. Other tensors (traces of three or more adjoint indices, which are
    mixtures of ``d`` and ``f``; sextet and epsilon tensors) are refused with
    ``ReadingSignError`` naming ``what``."""

    import ast

    def terms(node: ast.AST) -> list[ast.AST]:
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
            return terms(node.left) + terms(node.right)
        if isinstance(node, ast.UnaryOp):
            return terms(node.operand)
        return [node]

    try:
        tree = ast.parse(_NAMESPACE.sub("", colour), mode="eval")
    except SyntaxError as error:
        raise ReadingSignError(
            f"{what}: its colour tensor {colour!r} is no UFO colour expression"
        ) from error
    signs = set()
    for term in terms(tree.body):
        sign = 1
        for call in (n for n in ast.walk(term) if isinstance(n, ast.Call)):
            name = call.func.id if isinstance(call.func, ast.Name) else None
            arity = len(call.args)
            if name == "complex":
                continue
            if name == "f" and arity == 3:
                sign = -sign
            elif not (
                (name in ("d", "T") and arity == 3)
                or (name in ("Identity", "IdentityB", "Tr") and arity == 2)
            ):
                raise ReadingSignError(
                    f"{what}: the colour tensor {name} of {arity} indices has no "
                    "sign under the transposition of its adjoint generators (only "
                    "f, d, T, Identity and two-index traces have one), so it "
                    "cannot be read as the tree engine reads it"
                )
        signs.add(sign)
    if len(signs) != 1:
        raise ReadingSignError(
            f"{what}: the terms of its colour change sign differently under the "
            "transposition of the adjoint generators"
        )
    return signs.pop()


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
    """The UFO legs of one reading of a vertex closed under conjugation, with
    fermion legs read as the tree engine reads them (module docstring).

    ``legs`` are 1-based UFO legs in the incoming reading (loop in, loop out,
    blocks); each fermion leg is exchanged with the leg of its conjugate (the
    other leg of its fermion line). Bosons and ghosts keep the incoming
    reading: only spinor legs transpose, and ghost loop lines meet no tree
    row (the readings are checked against legacy, O2). Vertices not closed
    under conjugation are read through their conjugate rule
    (``TreeReading.vertex``), ghost vertices on their own legs
    (``conjugate_vertex_rules``).
    """

    fermions = [leg for leg in legs if spins[particles[leg - 1]] == FERMION_SPIN]
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


def _antiparticle_legs(
    items: Sequence[str], particles: Sequence[str], antinames: Mapping[str, str]
) -> tuple[int, ...]:
    """1-based legs of ``particles`` holding the antiparticle of each item,
    each leg used once (identical legs in order: a UFO vertex is symmetric
    under the exchange of identical legs)."""

    free = list(range(1, len(particles) + 1))
    out = []
    for item in items:
        wanted = antinames[item]
        leg = next((leg for leg in free if particles[leg - 1] == wanted), None)
        if leg is None:
            raise FermionSignError(
                f"vertex {tuple(particles)} has no leg for the antiparticle of {item}"
            )
        free.remove(leg)
        out.append(leg)
    return tuple(out)


def closed_under_conjugation(
    particles: Sequence[str], antinames: Mapping[str, str]
) -> bool:
    """Whether a vertex's particle multiset is its own conjugate: the tree
    engine then reads the vertex on its own legs, else through its conjugate
    rule (``conjugate_vertex_rules``, ``ct_sites.ConjugateVertices``)."""

    return sorted(particles) == sorted(antinames[name] for name in particles)


def conjugate_vertex_rules(data: Mapping[str, Any]) -> dict[str, str | None]:
    """The conjugate rule of every vertex rule of a native model (``data``).

    A rule whose particles are closed under conjugation is its own conjugate;
    otherwise the conjugate is the rule of the antiparticles at the same
    coupling orders (a Hermitian Lagrangian holds it: the W couplings of
    [d~ u W-] and [u~ d W+]). None marks a rule without a unique conjugate
    or with several colour structures (their pairing with the conjugate's is
    not established), which the readings refuse.

    A rule with a ghost leg is read on its own legs, closed under
    conjugation or not (WP3.2 EWI-7: the W ghost vertices [ghA ghWm~ W-]):
    ghost loop lines meet no tree row and keep the incoming reading, and the
    ghost Lagrangian pairs no vertex with a Hermitian conjugate (the "rule
    of the antiparticles" [ghWm ghA~ W+] differentiates the other ghost leg,
    so reading through it would move the momentum factor); its boson legs
    take the tree engine's basis through the field signs of the catalogue
    (``catalogue._CatalogueBuilder.reading_sign``).
    """

    antinames = {p["name"]: p["antiname"] for p in data["particles"]}
    ghosts = {p["name"] for p in data["particles"] if int(p["spin"]) == GHOST_SPIN}
    orders_of = {
        entry["name"]: add_orders(
            tuple((name, power) for name, power in entry["orders"])
        )
        for entry in data["couplings"]
    }

    def orders(rule) -> frozenset:
        return frozenset(
            orders_of[name] for row in rule["couplings"] for name in row if name
        )

    by_fields: dict[tuple, list[Mapping[str, Any]]] = {}
    for rule in data["vertex_rules"]:
        key = (tuple(sorted(rule["particles"])), orders(rule))
        by_fields.setdefault(key, []).append(rule)
    out: dict[str, str | None] = {}
    for rule in data["vertex_rules"]:
        if closed_under_conjugation(rule["particles"], antinames) or (
            ghosts.intersection(rule["particles"])
        ):
            out[rule["name"]] = rule["name"]
            continue
        conjugate = tuple(sorted(antinames[name] for name in rule["particles"]))
        found = by_fields.get((conjugate, orders(rule)), [])
        out[rule["name"]] = (
            found[0]["name"]
            if len(found) == 1
            and len(rule["color_structures"]) == 1
            and len(found[0]["color_structures"]) == 1
            else None
        )
    return out


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
        self._conjugates = conjugate_vertex_rules(data)

    def __call__(self, rule: str, legs: tuple[int, ...]) -> tuple[int, ...]:
        return tree_reading_legs(
            self._particles[rule], legs, self._spins, self._antinames
        )

    def conjugate(self, rule: str) -> str:
        """The rule the tree engine reads for the families' ``rule``: itself
        when closed under conjugation, else its conjugate rule."""

        found = self._conjugates.get(rule)
        if found is None:
            raise FermionSignError(
                f"vertex {rule} {tuple(self._particles[rule])} is not closed under "
                "conjugation and has no unique conjugate rule with one colour "
                "structure; its tree reading is not established"
            )
        return found

    def vertex(
        self, rule: str, legs: tuple[int, ...]
    ) -> tuple[str, tuple[int, ...], Mapping[int, int] | None]:
        """(rule read, its legs, colour index of the read rule per colour index
        of ``rule``, None for the identity) for a reading of the families'
        ``rule``.

        ``legs`` are the 1-based UFO legs of ``rule`` in the incoming reading
        (loop in, loop out, blocks). The tree engine reads the current of tree
        particle p on the UFO leg of p (``models.compiler_entry.
        _compile_oriented_kernels``); the incoming reading puts the current of
        field f, the tree particle anti(f), on the leg of f. A rule closed
        under conjugation is read on its own legs (``tree_reading_legs``:
        spinor legs transposed). A rule that is not (a W coupling of a loop
        quark line) has no leg for the tree particle anti(f) of its fermion
        currents: the tree engine reads them through the conjugate rule, with
        its couplings and Lorentz structures, which differ from the rule's own
        by complex conjugation and the exchange of chiral projectors (the CKM
        phase of [u~ b W+]; the Yukawas of [t~ b G+]); each current takes the
        conjugate's leg of its tree particle, a boson the slot of its own
        Lorentz type. The colour stays the incoming reading of ``rule`` (the
        colour seeds).
        """

        read = self.conjugate(rule)
        if read == rule:
            return rule, self(rule, legs), None
        own = self._particles[rule]
        return (
            read,
            _antiparticle_legs(
                [own[leg - 1] for leg in legs], self._particles[read], self._antinames
            ),
            _FIRST,
        )

    def mass_signs(self, loop_out: str) -> Mapping[str, int]:
        """Mass parameters whose sign the tree reading flips in a segment
        whose loop line leaves as ``loop_out`` (module docstring)."""

        mass = self._masses[loop_out]
        if self._spins[loop_out] != FERMION_SPIN or mass == ZERO:
            return {}
        return {mass: -1}


# Open rules have one colour structure (``conjugate_vertex_rules``).
_FIRST: Mapping[int, int] = MappingProxyType({0: 0})


# Source orientations of ``fermion_ordering._fermion_input_orientation_sign``.
PARTICLE, ANTIPARTICLE = "particle", "antiparticle"
# The value of a colour-singlet external fermion in the fermion ends of
# ``LeafWeights`` and ``ct_sites.SiteClosure`` (its colour representation;
# coloured ends hold their all-outgoing representation, 3 or -3).
SINGLET_END = 1


def fermion_end(representation: int, what: str = "a fermion") -> int:
    """The fermion end of a fermion of colour representation
    ``representation``, the one classifier of ``LeafWeights`` and
    ``ct_sites.SiteClosure`` (external ends, blocks): the tree engine's
    sign ownership (``fermion_ordering._fermion_rep_requires_exterior_sign``,
    the classifier the DAG admission and ``refusals`` use): ``SINGLET_END``
    for a fermion the Wick order signs (a colour singlet), its
    representation (3 or -3) for one the colour flow signs (a (anti)triplet).
    A representation the tree engine refuses (sextets, octets) has no sign
    rule and is refused (``FermionSignError`` naming ``what``)."""

    from pyamplicol.generation.fermion_ordering import (
        _fermion_rep_requires_exterior_sign,
    )

    representation = int(representation)
    try:
        singlet = _fermion_rep_requires_exterior_sign(representation, False)
    except ValueError as error:
        raise FermionSignError(
            f"{what}: a fermion of colour representation {representation} has no "
            f"sign rule ({error})"
        ) from None
    return SINGLET_END if singlet else representation


def fermion_ends(fermions: Iterable[tuple[int, int]]) -> dict[int, int]:
    """label -> ``fermion_end`` of the external fermions given as (label,
    all-outgoing colour representation)."""

    return {
        label: fermion_end(representation, f"external fermion {label}")
        for label, representation in fermions
    }


def open_end_orientation(field: Mapping[str, Any]) -> str:
    """Source orientation of the external end of the open fermion line of a
    block whose incoming field is the native particle entry ``field``: the
    block of incoming field f holds the external end of anti(f) (an incoming
    e- enters as the field e- and is the outgoing source e+), so its
    orientation is the opposite of f's. f's is the tree engine's
    ``CompiledParticleRecord.source_orientation`` (``models.contracts``:
    self-conjugate, else particle or antiparticle by the sign of the PDG
    code), read from the same fields of the native entry
    (``test_loop_v2_ew_leptons``: equal on every fermion of the G_mu UFO)."""

    if field["name"] == field["antiname"]:
        raise FermionSignError(
            f"self-conjugate fermion {field['name']}: Majorana lines have no "
            "source orientation (the tree engine refuses them too)"
        )
    if int(field["pdg_code"]) == 0:
        raise FermionSignError(
            f"fermion {field['name']}: a non-self-conjugate particle cannot use "
            "PDG code zero (the tree engine refuses it too)"
        )
    return ANTIPARTICLE if int(field["pdg_code"]) > 0 else PARTICLE


def colour_singlet_sign(
    supports: Sequence[Sequence[int]], lines: Sequence[tuple[int, str, int, str]]
) -> int:
    """The Koszul sign that joins blocks carrying colour-singlet fermions
    (module docstring; design §3.3).

    ``supports`` lists, per block in the order the blocks are read (loop
    contents, or the legs of a root vertex), the labels of the colour-singlet
    external fermions it holds; labels rank the sources as the tree engine
    ranks them (source slots, in label order). ``lines`` lists every
    singlet line the blocks' open ends join outside them, as (index into
    ``supports``, source orientation of that open end) for its two ends.

    A block row is its current's coefficient of the product of its singlet
    sources in source order, so the product of the rows in block order is the
    diagram's coefficient of all sources times the Koszul sign of
    concatenating the supports (the tree engine's support sign, merged block
    by block); the open ends, odd elements in block order, then pair as the
    tree engine's two-fermion merge pairs its inputs (support sign of the two
    ends by block order, times their orientation sign), and the closed pairs,
    even, join in any order (support sign of each line against the lines
    before it). The product does not depend on the block order.
    """

    ranks = {label: label for support in supports for label in support}
    sign = 1
    joined: tuple[int, ...] = ()
    for support in supports:
        labels = tuple(sorted(support))
        sign *= _external_fermion_support_sign(joined, labels, ranks)
        joined = (*joined, *labels)
    ends = {index: index for line in lines for index in line[0::2]}
    if len(ends) != 2 * len(lines):
        raise FermionSignError(
            "a colour-singlet fermion line with two ends in one block"
        )
    joined = ()
    for first, first_orientation, second, second_orientation in lines:
        if not supports[first] or not supports[second]:
            raise FermionSignError(
                "an open colour-singlet fermion line without a singlet source"
            )
        try:
            sign *= _external_fermion_support_sign(
                (first,), (second,), ends
            ) * _fermion_input_orientation_sign(
                True, first_orientation, True, True, second_orientation, True
            )
        except ValueError as error:
            raise FermionSignError(str(error)) from error
        sign *= _external_fermion_support_sign(joined, (first, second), ends)
        joined = (*joined, first, second)
    return sign


# The kind of a content for ``LeafWeights.fermion_sign``: no fermion line
# (its loop line a ghost, or another boson), or a fermion line or anything
# else the full sign must read.
_BOSONIC, _GHOST, _FERMIONIC = 0, 1, 2


class LeafWeights:
    """Leaf weights w_s of skeletons (module docstring), a ``LeafWeight``.

    ``ends`` maps the label of every coloured external fermion to its
    all-outgoing representation (+3 or -3) and of every colour-singlet one to
    ``SINGLET_END``, as ``SiteClosure`` records them.

    eps_s depends on the skeleton's ``contents`` alone, and every caller asks
    for it per reading: generation checks every reading of every skeleton
    (F and B, ``fermion_sign``, which refuses fermion lines it does not
    cover), the plan export every F leaf (``__call__``). ``fermion_sign``
    therefore computes it once per distinct ``contents`` tuple
    (``_fermion_sign``, the full rule; a refusal is raised again at every
    call, nothing being kept for it), and a tuple whose contents carry no
    fermion line takes a fast path: per content, once, its kind
    (``_content_kind``), and the tuple's sign is -1 for a closed ghost loop,
    else +1 (WP3.1 D1, RUST-MIGRATION.md §3.D; equal to ``_fermion_sign``
    on every reading of the controls).
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
        # The coloured fermion ends as a leg mask (``_content_kind``).
        self._end_mask = sum(1 << (label - 1) for label in self._ends)
        self._signs: dict[tuple[Content, ...], int] = {}
        self._kinds: dict[Content, int] = {}
        self._depends: dict[str, frozenset[str]] = {}

    def __call__(self, skeleton: Skeleton, contents: Sequence[Content]) -> Fraction:
        return skeleton.weight * skeleton.multiplicity * self.fermion_sign(skeleton)

    def fermion_sign(self, skeleton: Skeleton) -> int:
        """eps_s of a skeleton (reading-independent), once per distinct
        ``contents`` tuple (class docstring)."""

        contents = skeleton.contents
        kinds = self._kinds
        ghost = True
        for content in contents:
            kind = kinds.get(content)
            if kind is None:
                kind = kinds[content] = self._content_kind(content)
            if kind == _FERMIONIC:
                break
            ghost = ghost and kind == _GHOST
        else:
            # No fermion line: no propagator, closed-loop or line-parity sign
            # and no mass-free coupling to require (``_fermion_sign``).
            return -1 if ghost else 1
        sign = self._signs.get(contents)
        if sign is None:
            sign = self._signs[contents] = self._fermion_sign(skeleton)
        return sign

    def _content_kind(self, content: Content) -> int:
        """Whether ``_fermion_sign`` reads anything of ``content`` but its
        loop line's ghost spin: a fermion loop line, a fermion block, a
        block holding a coloured fermion end (the inputs of
        ``_line_parity`` and ``_require_mass_free_couplings``), or another
        than one routing term (refused there) make it fermionic."""

        if (
            self._fermion(content.loop_in)
            or self._fermion(content.loop_out)
            or len(content.terms) != 1
        ):
            return _FERMIONIC
        for block in content.terms[0].blocks:
            if block.mask & self._end_mask or self._fermion(block.field):
                return _FERMIONIC
        spin = int(self._particles[content.loop_out]["spin"])
        return _GHOST if spin == GHOST_SPIN else _BOSONIC

    def _fermion_sign(self, skeleton: Skeleton) -> int:
        """eps_s of a skeleton by the full rule (the oracle of
        ``fermion_sign``'s cache and fast path)."""

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
        line it emits (module docstring: the tree reading flips that mass).

        An input of eps_s: a content whose fermion loop line this reads makes
        ``_content_kind`` return ``_FERMIONIC`` (``fermion_sign``'s fast path
        skips this check). Any new input read here (WP3.2 EWI-5) must make
        ``_content_kind`` fermionic too; ``check_leaf_weights`` in
        ``tests/unit/test_loop_v2_native_lowering.py`` is the guard."""

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
        """Pairing parity of the coloured lines, the Koszul sign of the
        colour-singlet fermions (``colour_singlet_sign``) and closed-loop
        sewing signs.

        An input of eps_s: every content whose loop line, block field or
        external fermion end (``_ends``: coloured ends +3/-3, colour-singlet
        ends 1) this reads makes ``_content_kind`` return ``_FERMIONIC``, and
        ``fermion_sign``'s fast path returns +1 or the ghost sign without
        calling it. A block holding a colour-singlet fermion is read even
        when its own field is a boson (its support enters the Koszul sign);
        its labels are in ``_ends``, so ``_content_kind`` reads it too.
        ``check_leaf_weights`` in ``tests/unit/test_loop_v2_native_lowering.py``
        is the guard (on the controls with such content)."""

        sign = 1
        pairs: list[tuple[int, int]] = []
        links: dict[tuple, list[tuple]] = {}
        open_ends: dict[tuple, tuple[int, int]] = {}
        # Colour-singlet fermions: each block's singlet labels in content
        # order, and per open singlet end (port) its block's index there and
        # the source orientation of its external end.
        supports: list[tuple[int, ...]] = []
        singlet_ends: dict[tuple, tuple[int, str]] = {}

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
                singlets = tuple(
                    label for label in labels if self._ends[label] == SINGLET_END
                )
                if singlets:
                    supports.append(singlets)
                port = ("block", position, slot)
                entry = self._particles[block.field]
                if int(entry["spin"]) == FERMION_SPIN:
                    end = fermion_end(entry["color"], f"fermion block {block.field}")
                    if end == SINGLET_END:
                        if not singlets:
                            raise FermionSignError(
                                f"a colour-singlet fermion block {block.field} "
                                "holds no colour-singlet fermion end"
                            )
                        singlet_ends[port] = (
                            len(supports) - 1,
                            open_end_orientation(entry),
                        )
                        ports.append(port)
                    else:
                        (starts if end == 3 else stops).append(None)
                sign *= identity_pairing_sign(starts, stops)
                for start, stop in zip(
                    sorted(starts, key=_open_last),
                    sorted(stops, key=_open_last),
                    strict=True,
                ):
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
        lines: list[tuple[int, str, int, str]] = []
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
            singlet = [singlet_ends[port] for port in component if port in singlet_ends]
            if not ends and not singlet:
                sign = -sign  # a closed fermion loop
                continue
            if ends and singlet:
                raise FermionSignError(
                    "a fermion line through the loop joins a coloured and a "
                    "colour-singlet end"
                )
            if singlet:
                if len(singlet) != 2:
                    raise FermionSignError("an open fermion line without two ends")
                lines.append((*singlet[0], *singlet[1]))
                continue
            partners = dict(ends)
            if len(ends) != 2 or set(partners) != {3, -3}:
                raise FermionSignError("an open fermion line without two ends")
            # The open end of a fundamental-type block faces an antifundamental
            # endpoint: the joined line runs from partners[-3] to partners[3].
            pairs.append((partners[-3], partners[3]))
        if supports:
            sign *= colour_singlet_sign(supports, lines)
        return sign * _parity([label for pair in pairs for label in pair])


__all__ = [
    "LOOP_FACTOR_DENOMINATOR",
    "SINGLET_END",
    "FermionSignError",
    "LeafWeights",
    "Normalisation",
    "ReadingSignError",
    "TreeReading",
    "closed_under_conjugation",
    "colour_singlet_sign",
    "colour_transposition_sign",
    "conjugate_vertex_rules",
    "fermion_end",
    "fermion_ends",
    "open_end_orientation",
    "tree_reading_legs",
]
