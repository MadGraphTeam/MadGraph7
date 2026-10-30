# SPDX-License-Identifier: 0BSD
"""Counterterm sites: the test oracle of design §6.6 and the root closure.

Every tree with exactly one counterterm (CT) vertex can be rooted at that
vertex. A *site* is a model CT rule applied at the root to m block currents
whose leg sets partition the external legs, so the sum over sites is the
complete CT=1 amplitude without listing a single tree (design §6.6):

* the arity m runs over the rules' own particle counts (k_max is the largest);
* partitions are unordered and their block currents are assigned to the rule's
  field multiset; the UFO vertex is fully symmetrised, so each assignment
  counts once (the Berends-Giele rule);
* a two-point site needs two blocks of at least two legs each: insertions on
  external legs are LSZ field factors (``loops.v2.fields``), never sites.

Production admits counterterms by the plain tree-level rule
(``ct_recursion.admitted_alternatives``: coupling orders, one insertion and
``particle_veto``), passed as ``admission`` to ``generate_counterterm_sites``
for the CT vertices at the root of the closure (``ct_closure``). The site
closure of every rule, the O6 oracle, keeps MadLoop's matching
(``set_LoopCT_vertices``, as the removed legacy ``loops.counterterms.
match_counterterms`` did) as an independent check of that rule, with loop
graphs replaced by *witnesses*: admitted family routings whose structural
rank satisfies R >= 2N - 4. A witness signature is the multiset of its attached
block currents (leg mask, field, orders) plus the summed orders of its loop
vertices; it is realised by one or more loop-species sets (sorted |PDG| of the
loop lines, merged closed-loop flavours expanded to their members):

* R2 and UVmass rules need a witness of the site's signature whose species set
  is one of the rule's alternatives (an empty alternative is a wildcard);
* UVloop rules need any witness, and their alternatives only filter forbidden
  species;
* UVtree rules need no witness;
* each admitted alternative is one insertion, as in MadLoop.

Completeness (design §6.3): every divergent witness must be covered, for each
of its loop-species sets, by a species-matched (R2 or UVmass) rule of its
signature, or by a certified zero: supplied by a certifier, or certified here
for closed fermion loops whose orientations cancel by charge conjugation
(Furry) or whose colour trace vanishes (``closed_fermion_zeros``). Faddeev-Popov ghost
loops count in the loop group of their gauge boson, as in UFO loop
declarations. Otherwise generation fails and names the uncovered signature.
UV poles that are renormalised through couplings rather than per 1PI function
(e.g. internal gluon self-energies) are not part of the coverage rule; the
IR-pole check covers them at runtime.

Colour, kinematics and signs of a site term (design §6.4, §6.6):

* the vertex colour is the rule entry's UFO colour lowered to tau = sqrt(2) t
  strings and traces (``loops.v2.colour`` conventions), closed exactly with the
  colour-flow strings of the attached block words; a block attaches to the UFO
  leg of its incoming field, as in the family recursion;
* the Lorentz part is the catalogue ``ct-site`` kernel (``loops.v2.catalogue``)
  on the tree engine's block rows. The tree engine reads a vertex with the
  current of tree particle p on the UFO leg of particle p
  (``models.compiler_entry._compile_oriented_kernels``), the conjugate of the
  incoming-field reading; for spinor legs this transposes the Dirac chain. The
  kernel therefore takes each row on the leg of its tree particle, while the
  momenta keep the physical reading: UFO leg i carries the momentum flowing
  into the vertex from the block of incoming field i, i.e. minus its leg-mask
  momentum (initial legs negated). A vertex whose particle multiset is not
  closed under conjugation (a W coupling) has no leg for the tree particles
  of its blocks: the tree engine reads them through the model's conjugate
  rule, so its kernels and couplings are the conjugate rule's, which differ
  from the vertex's own by complex conjugation and the exchange of chiral
  projectors (``ConjugateVertices``), while its colour stays the incoming
  reading of the vertex (WP3.2 EWI-2). The tree engine reads every vertex
  with the rows and tree momenta of its particles' currents and their
  outgoing colour flows; the closure gives each kernel leg the same row and
  the same block's momentum, but flowing in (the tree momentum negated), and
  closes the incoming colour. The two readings differ by (-1)^(momentum
  degree of the kernel) times the transposition sign of the colour, which
  every term carries (``SiteClosure._reading_sign``), on every vertex, at
  Born roots, counterterm roots and two-point sites alike (WP3.2 EWI-7b,
  genericity audit 2 G2-M7). It is 1 on every QCD vertex (``f`` with one
  momentum; ``d``, generators and metrics without; a quark two-point odd
  in the momentum and diagonal in colour, whose two block momenta are
  opposite, as in the earlier reading of its legs) and -1 on every
  colourless vertex odd in the momenta: W+ W- A/Z, Z H G0, W G H, W G G0,
  their R2 and UV counterterms, the Goldstone-vector mixings G0-Z, G0-A
  and G-W at two-point sites (the clone lines read them the same way,
  ``models.ct_line_insertions._momentum_ratio``), the Levi-Civita R2 of a
  Z with gluons and the d*f cells of the four-gluon R2 (which sum to
  zero). The earlier reading kept sign 1 and gave the momenta in UFO-leg
  order: w+ w- > u u~ had a Born 177 times its time-reversed process's and
  z h > t t~ a Feynman-gauge Born 20 times its unitary-gauge one. A kernel
  neither even nor odd in the momenta or a colour without a transposition
  sign is refused (``ReadingSignError``);
* ``Block.wick_sign`` turns each block into its Feynman-rule current and the
  root sign completes FeynKit's diagram sign: the parity of the full coloured
  fermion line listing over the product of the blocks' listings
  (``loops.v2.blocks``). One coloured or colour-singlet fermion line may
  close at the root. The blocks carry the tree engine's Koszul signs of
  colour-singlet fermions (``generation/fermion_ordering.py``), so the root
  completes them as the loops do (``conventions.colour_singlet_sign``): the
  Koszul sign of the attached blocks' singlet supports in leg order, times
  the merge sign of the singlet line's two open ends if one closes at the
  root (WP3.2 EWI-1 for one singlet line, EWI-5 for any number).

Entries whose Lorentz structure the catalogue cannot normalise (a
Levi-Civita tensor) have no ``ct-site`` kernel. Where the same UFO colour and
Lorentz cell of several rules (the loop groups of one vertex) is closed on the
same blocks and the coefficients, weighted by the admitted alternatives,
cancel identically (``vanishes_identically``: the anomaly-free sum of a Z's
axial couplings over the quark doublets), the cell is dropped
(``SiteClosure.cancelled_cells``); otherwise the site fails for want of a
kernel.

Laurent couplings stay symbolic (``CountertermSites.coupling_outputs``) and are
bound by the source scalar program (``loops.v2.fields.counterterm_scalar_plan``),
like the legacy counterterm scale view. Components split by kind: R2 is
``r2``; UVloop, UVtree and UVmass are ``uv_local``; external fields are
``uv_external_fields`` (``loops.v2.fields``).

The reference values (``Carrier``, ``SiteClosure.value``, ``add_colour``,
``ColourOverlaps``) evaluate sites in any carrier with exact colour at
N_c = 3; the production form puts the CT vertices into the tree recursion
(design §6.1, wave 2) and keeps these sites as its oracle (O6).
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, fields
from fractions import Fraction
from functools import reduce
from itertools import product
from operator import add
from types import MappingProxyType
from typing import Any

from pyamplicol.color.connections import (
    ColorTensor,
    ConnectedColorTensor,
    ExactColorCoefficient,
    TensorTerm,
    contract_connected_tensor_nc_terms,
)
from pyamplicol.color.contraction_types import NC
from pyamplicol.color.correlator_matrices import process_color_legs
from pyamplicol.loops.model import KernelBasisEntry

from .blocks import Block, TreeBlocks, _NativeFields, _parity, identity_pairing_sign
from .catalogue import (
    CountertermSiteBinding,
    CountertermSiteKey,
    LoopKernelCatalogue,
)
from .colour import (
    ColourNetwork,
    ColourProduct,
    ColourString,
    _exact_number,
    _head,
    _structure_constant,
    block_colour_network,
    multiply,
    reduce_network,
)
from .conventions import (
    GHOST_SPIN,
    SINGLET_END,
    FermionSignError,
    ReadingSignError,
    closed_under_conjugation,
    colour_singlet_sign,
    fermion_end,
    fermion_ends,
    open_end_orientation,
)
from .conventions import (
    colour_transposition_sign as _colour_transposition_sign,
)
from .families import (
    BlockCurrent,
    BlockTable,
    FamilySet,
    LoopOrderSelection,
    Orders,
    _set_partitions,
    add_orders,
    model_data,
)

# Kinds whose alternatives must equal a witness's loop species (MadLoop).
SPECIES_MATCHED_KINDS = frozenset({"R2", "UVmass"})
COUNTERTERM_KINDS = frozenset({"R2", "UVmass", "UVloop", "UVtree"})
COMPONENT_BY_KIND: Mapping[str, str] = {
    "R2": "r2",
    "UVloop": "uv_local",
    "UVtree": "uv_local",
    "UVmass": "uv_local",
}
LAURENT_POWERS = (0, -1, -2)

Species = tuple[int, ...]
# (UFO collection, vertex id, colour index, Lorentz index): one coupling cell of
# a UFO vertex, shared by the loop groups (rules) of that vertex.
Cell = tuple[str, int, int, int]


class CountertermCompletenessError(ValueError):
    """A divergent witness is covered neither by a rule nor by a certified zero."""

    # The listed internal refusal (``refusals.LOOP_INTERNAL_REFUSALS``).
    refusal_key = "counterterm-completeness"


# --------------------------------------------------------------------------- records


@dataclass(frozen=True, slots=True, order=True)
class Signature:
    """Attached block currents (sorted) and the orders of the local vertex/loop."""

    attachments: tuple[BlockCurrent, ...]
    orders: Orders

    @property
    def fields(self) -> tuple[str, ...]:
        return tuple(sorted(block.field for block in self.attachments))

    def describe(self) -> str:
        blocks = ", ".join(
            f"{block.field}{{{_legs(block.mask)}}}" for block in self.attachments
        )
        return f"[{blocks}] at {dict(self.orders)}"


@dataclass(frozen=True, slots=True)
class CertifiedZero:
    """A divergent witness whose rational and UV parts vanish, with its proof."""

    signature: Signature
    species: Species
    certificate: str


@dataclass(frozen=True, slots=True)
class Coverage:
    """How one divergent witness (signature, loop species) is covered."""

    signature: Signature
    species: Species
    rules: tuple[str, ...]  # covering species-matched rules
    certificate: str | None = None  # set for a certified zero


@dataclass(frozen=True, slots=True)
class CountertermSite:
    """One rule closed at the root on block currents (one per UFO leg)."""

    rule: Any = field(repr=False, compare=False)  # loops.counterterms.CountertermRule
    rule_name: str
    blocks: tuple[BlockCurrent, ...]  # block attached to UFO leg i + 1
    alternatives: tuple[int, ...]  # admitted alternative indices of the rule

    @property
    def kind(self) -> str:
        return self.rule.kind

    @property
    def component(self) -> str:
        return COMPONENT_BY_KIND[self.kind]

    @property
    def orders(self) -> Orders:
        """Amplitude orders of the CT=1 trees of this site (the loop orders)."""
        return add_orders(self.rule.order, *(block.orders for block in self.blocks))

    @property
    def signature(self) -> Signature:
        return Signature(tuple(sorted(self.blocks)), add_orders(self.rule.order))

    @property
    def insertion_key(self) -> str:
        """Legacy insertion-multiset key: kind, UFO vertex names, particle PDGs."""
        names = "+".join(sorted({entry.vertex_name for entry in self.rule.entries}))
        pdgs = ",".join(map(str, self.rule.particle_pdgs))
        return f"{self.kind}:{names}:{pdgs}"


@dataclass(frozen=True)
class CountertermSites:
    """Admitted sites, the divergent witness table and its coverage log."""

    sites: tuple[CountertermSite, ...]
    witnesses: Mapping[Signature, frozenset[Species]]
    coverage: tuple[Coverage, ...]
    k_max: int

    def coupling_outputs(self) -> tuple[tuple[str, int], ...]:
        """Source scalar outputs (coupling name, epsilon power) of every
        site's own rule. A closure reads a vertex not closed under
        conjugation through its conjugate rule and binds that rule's
        couplings (``SiteClosure.read_entries``, which the slice's scalar
        plan takes, ``assemble.LoopSlice._scalar_plan``)."""
        return tuple(
            sorted(
                {
                    (entry.coupling.name, power)
                    for site in self.sites
                    for entry in site.rule.entries
                    for power in LAURENT_POWERS
                }
            )
        )


# --------------------------------------------------------------------------- helpers


def _legs(mask: int) -> str:
    return ",".join(
        str(index) for index in range(mask.bit_length()) if mask >> index & 1
    )


def _particles(data: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {entry["name"]: entry for entry in data["particles"]}


def ghost_partners(native_model: Any) -> dict[int, int]:
    """|PDG| of each ghost -> |PDG| of the vector boson of its diagonal vertex.

    A ghost loop belongs to the loop group of its gauge boson. The partner is
    the vector of the model's ghost-antighost-vector vertex; a ghost with no
    such unique vertex has no partner and its witnesses need their own rules.
    """

    data = model_data(native_model)
    by_name = _particles(data)
    found: dict[int, set[int]] = defaultdict(set)
    for rule in data["vertex_rules"]:
        particles = [by_name[name] for name in rule["particles"]]
        ghosts = [p for p in particles if int(p.get("ghost_number", 0))]
        vectors = [p for p in particles if p["spin"] == 3]
        if (
            len(particles) == 3
            and len(ghosts) == 2
            and len(vectors) == 1
            and abs(ghosts[0]["pdg_code"]) == abs(ghosts[1]["pdg_code"])
        ):
            found[abs(int(ghosts[0]["pdg_code"]))].add(abs(int(vectors[0]["pdg_code"])))
    return {
        ghost: vectors.pop() for ghost, vectors in found.items() if len(vectors) == 1
    }


# --------------------------------------------------------------------------- witnesses


def divergent_witnesses(
    families: FamilySet, native_model: Any
) -> dict[Signature, frozenset[Species]]:
    """Signatures of every admitted routing with R >= 2N - 4 (design §6.2).

    The rank of a routing is the sum of its positions' q-degrees (vertex plus
    outgoing propagator numerator), the structural rank of design §3.3. A
    position of q-degree >= 2 makes the divergent set unbounded by power
    counting (effective theories); such models are refused here.
    """

    by_name = _particles(model_data(native_model))
    members: dict[str, set[str]] = defaultdict(set)
    for member, representative in families.merged.items():
        members[representative].add(member)

    def species_class(name: str) -> tuple[int, ...]:
        return tuple(
            sorted(
                {abs(int(by_name[m]["pdg_code"])) for m in members.get(name, {name})}
            )
        )

    found: dict[Signature, set[Species]] = defaultdict(set)
    for family in families.families.values():
        bound = 2 * family.size - 4
        for closure in family.closures:
            cut, positions = families.routing(closure)
            degrees = [position.vertex.q_degree for position in positions]
            if max(degrees) >= 2:
                raise NotImplementedError(
                    "a loop vertex with q-degree >= 2 makes the superficially "
                    "divergent set unbounded; counterterm completeness needs a "
                    "model declaration (design §6.3)"
                )
            if sum(degrees) < bound:
                continue
            signature = Signature(
                tuple(sorted(block for p in positions for block in p.blocks)),
                add_orders(*(position.vertex.orders for position in positions)),
            )
            lines = {cut, *(position.vertex.loop_out for position in positions)}
            classes = {species_class(name) for name in lines}
            # Members of a merged class keep their own identity (one |PDG| each).
            for choice in product(*sorted(classes)):
                found[signature].add(tuple(sorted(set(choice))))
    return {signature: frozenset(rows) for signature, rows in found.items()}


# The pure vector current Gamma(mu of the boson leg, spinor, spinor) in the
# native model's Lorentz structure syntax.
_VECTOR_CURRENT = re.compile(r"Gamma\(idx\(1,(\d+)\),idx\(1,(\d+)\),idx\(1,(\d+)\)\)")
FERMION_SPIN = 2


def closed_fermion_zeros(
    witnesses: Mapping[Signature, frozenset[Species]], native_model: Any
) -> tuple[CertifiedZero, ...]:
    """Certified zeros of divergent closed fermion loops (design §6.3).

    A witness whose loop species is one fermion f is a closed f loop with its
    attachments in every cyclic order and both orientations. Its UV and R2
    parts vanish, so no rule needs to cover it, when

    * the colour factor is Tr_R(T^a) = 0: exactly one adjoint attachment,
      every other one a colour singlet; or
    * charge conjugation cancels the orientations (Furry): the colour factor is
      unchanged by reversing the order of the attachments (at most two
      adjoint attachments, the rest singlets), and every vertex joining f, its
      antiparticle and an attached field is a pure vector current
      (C gamma^mu^T C^-1 = -gamma^mu) with an odd number of attachments. The
      reversed orientation is then -1 times the other in d dimensions, cyclic
      order by cyclic order.

    Both need every model vertex with f and its antiparticle to be a
    three-point vertex with one other field (renormalisable couplings).
    E.g. one photon and two gluons on a quark loop (Furry), one photon and
    one gluon (traceless colour). A Z keeps its axial part, which R2 rules
    such as R2_GGZ cover.
    """

    data = model_data(native_model)
    particles = _particles(data)
    by_pdg = {int(entry["pdg_code"]): entry for entry in data["particles"]}
    lorentz = {
        entry["name"]: "".join(str(entry["structure"]).split())
        for entry in data["lorentz_structures"]
    }
    zeros = []
    for signature, rows in sorted(witnesses.items()):
        for species in sorted(rows):
            fermion = by_pdg.get(species[0]) if len(species) == 1 else None
            if fermion is None or int(fermion["spin"]) != FERMION_SPIN:
                continue
            certificate = _closed_fermion_certificate(
                signature.fields, fermion, particles, data["vertex_rules"], lorentz
            )
            if certificate is not None:
                zeros.append(CertifiedZero(signature, species, certificate))
    return tuple(zeros)


def _closed_fermion_certificate(
    fields: Sequence[str],
    fermion: Mapping[str, Any],
    particles: Mapping[str, Mapping[str, Any]],
    rules: Sequence[Mapping[str, Any]],
    lorentz: Mapping[str, str],
) -> str | None:
    line = sorted((fermion["name"], fermion["antiname"]))
    couplings: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for rule in rules:
        names = list(rule["particles"])
        if not (fermion["name"] in names or fermion["antiname"] in names):
            continue
        rest = list(names)
        for name in line:
            if name not in rest:
                rest = []
                break
            rest.remove(name)
        if not rest:
            if sorted(names) != line:
                continue  # f couples to another flavour: not on this loop
            return None
        if len(rest) != 1:
            return None  # a contact or multi-fermion vertex
        couplings[rest[0]].append(rule)
    if any(name not in couplings for name in fields):
        return None
    colours = [int(particles[name]["color"]) for name in fields]
    if any(colour not in (1, 8) for colour in colours):
        return None
    adjoint = colours.count(8)
    if adjoint == 1:
        return "colour: one adjoint on a closed fermion loop gives Tr(T^a) = 0"
    if (
        len(fields) % 2 == 1
        and adjoint <= 2
        and all(
            _pure_vector_current(rule, particles, lorentz)
            for name in fields
            for rule in couplings[name]
        )
    ):
        return (
            "Furry: an odd number of vector couplings on a closed fermion loop "
            "with a reversal-symmetric colour trace"
        )
    return None


def _pure_vector_current(
    rule: Mapping[str, Any],
    particles: Mapping[str, Mapping[str, Any]],
    lorentz: Mapping[str, str],
) -> bool:
    """Whether every Lorentz structure of ``rule`` is Gamma(boson, f, f)."""

    fermions = {
        leg
        for leg, name in enumerate(rule["particles"], start=1)
        if int(particles[name]["spin"]) == FERMION_SPIN
    }
    for name in rule["lorentz_structures"]:
        match = _VECTOR_CURRENT.fullmatch(lorentz[name])
        if match is None:
            return False
        boson, *spinors = (int(leg) for leg in match.groups())
        if boson in fermions or set(spinors) != fermions:
            return False
    return True


def _coverage(
    witnesses: Mapping[Signature, frozenset[Species]],
    rules: Mapping[str, Any],
    names: Mapping[int, str],
    corrections: frozenset[str],
    partners: Mapping[int, int],
    zeros: Iterable[CertifiedZero],
) -> tuple[Coverage, ...]:
    covering: dict[tuple[tuple[str, ...], Orders], dict[Species, set[str]]] = (
        defaultdict(lambda: defaultdict(set))
    )
    for name, rule in rules.items():
        if rule.kind not in SPECIES_MATCHED_KINDS or not _selected(rule, corrections):
            continue
        key = (
            tuple(sorted(names[pdg] for pdg in rule.particle_pdgs)),
            add_orders(rule.order),
        )
        for alternative in rule.alternatives:
            covering[key][tuple(alternative)].add(name)
    certified = {(zero.signature, zero.species): zero for zero in zeros}
    log, missing = [], []
    for signature in sorted(witnesses):
        options = covering.get((signature.fields, signature.orders), {})
        for species in sorted(witnesses[signature]):
            group = tuple(sorted({partners.get(pdg, pdg) for pdg in species}))
            matched = options.get(group, set()) | options.get((), set())
            zero = certified.get((signature, species))
            if matched:
                log.append(Coverage(signature, species, tuple(sorted(matched))))
            elif zero is not None:
                log.append(Coverage(signature, species, (), zero.certificate))
            else:
                missing.append(f"{signature.describe()} with loop species {species}")
    if missing:
        raise CountertermCompletenessError(
            "divergent witnesses without a counterterm rule or certified zero "
            "(design §6.3): " + "; ".join(missing)
        )
    return tuple(log)


def _selected(rule: Any, corrections: frozenset[str]) -> bool:
    return rule.perturbation_type is None or rule.perturbation_type in corrections


# --------------------------------------------------------------------------- sites


def generate_counterterm_sites(
    families: FamilySet | None,
    blocks: BlockTable,
    rules: Mapping[str, Any],
    native_model: Any,
    selection: LoopOrderSelection,
    *,
    forbidden: Iterable[int] = (),
    zeros: Iterable[CertifiedZero] = (),
    whole_rules: bool = False,
    closed_rules: Iterable[str] | None = None,
    root_mask: int | None = None,
    admission: Mapping[str, tuple[int, ...]] | None = None,
) -> CountertermSites:
    """Admitted CT sites of a process, after the completeness check.

    ``rules`` are the prepared model's counterterm rules
    (``PreparedCountertermModel.rules``); ``blocks`` is the family recursion's
    block table and ``native_model`` the native model (or its JSON mapping)
    whose names both use. ``forbidden`` are the |PDG| codes the process
    excludes; ``zeros`` are certified zeros of divergent witnesses, besides
    those of ``closed_fermion_zeros``.
    ``whole_rules`` admits rules carrying a whole-rule colour continuation;
    the caller then closes their replacement terms (``SiteClosure.close``)
    instead of ``SiteClosure.terms``. ``closed_rules`` restricts the sites to
    the named rules (the rules the CT recursion leaves to site closure,
    ``loops.v2.ct_recursion``); completeness is still checked on all rules.
    ``root_mask`` keeps only the sites with that leg set as one block (the
    CT vertices at the root of the counterterm closure, ``ct_closure``).
    ``admission`` (rule name ->
    admitted alternative indices) replaces the witness admission and the
    completeness check: production's plain rule (``ct_recursion``); then
    ``families`` is not read.
    """

    data = model_data(native_model)
    names = {int(entry["pdg_code"]): entry["name"] for entry in data["particles"]}
    closed = None if closed_rules is None else frozenset(closed_rules)
    for name, rule in rules.items():
        if rule.kind not in COUNTERTERM_KINDS:
            raise ValueError(f"counterterm rule {name} has unknown kind {rule.kind!r}")
        if (
            rule.whole_rule_continuation is not None
            and not whole_rules
            and (closed is None or name in closed)
        ):
            raise NotImplementedError(
                f"counterterm rule {name} carries a whole-rule colour continuation; "
                "site closure needs its UFO entries"
            )
    corrections = frozenset(selection.corrections)
    forbidden = frozenset(abs(int(pdg)) for pdg in forbidden)
    witnesses = {} if admission is not None else divergent_witnesses(families, data)
    coverage = (
        _coverage(
            witnesses,
            rules,
            names,
            corrections,
            ghost_partners(data),
            (*zeros, *closed_fermion_zeros(witnesses, data)),
        )
        if admission is None
        else ()
    )
    full = (1 << len(blocks.legs)) - 1
    sites = []
    for name in sorted(rules):
        rule = rules[name]
        if not _selected(rule, corrections):
            continue
        if closed is not None and name not in closed:
            continue
        fields = tuple(names[pdg] for pdg in rule.particle_pdgs)
        if root_mask is None:
            partitions = _set_partitions(full, len(fields), len(fields))
        else:
            partitions = (
                (root_mask, *rest)
                for rest in _set_partitions(
                    full & ~root_mask, len(fields) - 1, len(fields) - 1
                )
            )
        for parts in partitions:
            if len(parts) == 2 and any(part & (part - 1) == 0 for part in parts):
                continue  # an external-leg insertion: an LSZ field factor
            for choice in product(*(blocks.currents(part) for part in parts)):
                if sorted(block.field for block in choice) != sorted(fields):
                    continue
                site = _admit(
                    name,
                    rule,
                    _assign(fields, choice),
                    witnesses,
                    selection,
                    forbidden,
                    None if admission is None else admission.get(name, ()),
                )
                if site is not None:
                    sites.append(site)
    k_max = max((len(rule.particle_pdgs) for rule in rules.values()), default=0)
    return CountertermSites(tuple(sites), witnesses, coverage, k_max)


def _assign(
    fields: Sequence[str], choice: Sequence[BlockCurrent]
) -> tuple[BlockCurrent, ...]:
    """Blocks per UFO leg; identical fields take blocks in mask order."""

    free = sorted(choice)
    out = []
    for name in fields:
        block = next(block for block in free if block.field == name)
        free.remove(block)
        out.append(block)
    return tuple(out)


def witness_alternatives(
    rule: Any,
    signature: Signature,
    witnesses: Mapping[Signature, frozenset[Species]],
    selection: LoopOrderSelection,
    forbidden: frozenset[int] = frozenset(),
) -> tuple[int, ...]:
    """Alternatives of ``rule`` admitted on a site of ``signature`` (§6.2).

    The admission predicate a CT recursion consults per vertex instance: it
    depends on the attached currents and the witness table only, never on a
    tree. ``signature.orders`` must be the rule's own orders.
    """

    species = witnesses.get(signature, frozenset())
    admitted = []
    for index, alternative in enumerate(rule.alternatives):
        alternative = tuple(alternative)
        if rule.kind in SPECIES_MATCHED_KINDS:
            ok = bool(species) if not alternative else alternative in species
        elif forbidden.intersection(alternative):
            ok = False
        elif rule.kind == "UVloop":
            ok = bool(species)
        else:  # UVtree: eligible through its own correction orders
            ok = bool(selection.corrections.intersection(n for n, _ in rule.order))
        if ok:
            admitted.append(index)
    return tuple(admitted)


def _admit(
    name: str,
    rule: Any,
    blocks: tuple[BlockCurrent, ...],
    witnesses: Mapping[Signature, frozenset[Species]],
    selection: LoopOrderSelection,
    forbidden: frozenset[int],
    admitted: tuple[int, ...] | None = None,
) -> CountertermSite | None:
    site = CountertermSite(rule, name, blocks, ())
    if site.orders not in selection.loop_orders:
        return None
    if admitted is None:
        admitted = witness_alternatives(
            rule, site.signature, witnesses, selection, forbidden
        )
    return CountertermSite(rule, name, blocks, admitted) if admitted else None


# --------------------------------------------------------------------------- colour


def _ufo_index(atom: Any) -> tuple[str, int]:
    value = int(atom.to_canonical_string())
    return ("leg", value) if value > 0 else ("dummy", value)


def _ufo_factor(atom: Any, representations: Sequence[int]) -> ColourNetwork:
    from symbolica import AtomType

    kind = atom.get_type()
    if kind == AtomType.Num:
        return (ColourProduct(_exact_number(atom)),)
    if kind != AtomType.Fn:
        raise NotImplementedError(f"unsupported UFO colour factor {atom}")
    head = _head(atom)
    labels = [_ufo_index(argument) for argument in atom]
    if head == "f":
        return _structure_constant(*labels)
    if head == "d":
        # d^abc = 2 tr(t^a {t^b, t^c}) = (tr(abc) + tr(acb)) / sqrt(2) with tau.
        a, b, c = labels
        return (
            ColourProduct(traces=((a, b, c),), root2_power=1),
            ColourProduct(traces=((a, c, b),), root2_power=1),
        )
    if head == "Tr":
        return (ColourProduct(traces=(tuple(labels),), root2_power=len(labels)),)
    if head == "T":
        adjoint, row, col = labels
        return (
            ColourProduct(strings=(ColourString(row, (adjoint,), col),), root2_power=1),
        )
    if head == "Identity":
        reps = [
            representations[label[1] - 1] if label[0] == "leg" else None
            for label in labels
        ]
        if 8 in reps:
            return (ColourProduct(traces=(tuple(labels),)),)  # delta^ab
        if 3 in reps or -3 in reps:
            row, col = labels if reps[0] == 3 or reps[1] == -3 else labels[::-1]
            return (ColourProduct(strings=(ColourString(row, (), col),)),)
        raise NotImplementedError(f"colour Identity between dummy indices: {atom}")
    raise NotImplementedError(
        f"UFO colour tensor {head!r} does not lower to colour flow"
    )


def ufo_colour_network(source: str, representations: Sequence[int]) -> ColourNetwork:
    """Lower a UFO colour string to tau strings and traces (``loops.v2.colour``).

    Handles Identity, T, f, d and Tr with positive leg and negative dummy
    indices; labels are ``("leg", i)`` and ``("dummy", -k)``. The tree
    normaliser (``models.tensors.normalize_color_expression``) has no ``Tr``,
    which UFO counterterm vertices use.
    """

    from symbolica import AtomType

    from ..model_expressions import parse_exact_ufo

    expression = parse_exact_ufo(source).expand()
    terms = list(expression) if expression.get_type() == AtomType.Add else [expression]
    out: list[ColourProduct] = []
    for term in terms:
        factors = list(term) if term.get_type() == AtomType.Mul else [term]
        out.extend(
            multiply(*(_ufo_factor(factor, representations) for factor in factors))
        )
    return tuple(out)


# --------------------------------------------------------------------------- terms


@dataclass(frozen=True, slots=True)
class SiteTerm:
    """One rule entry on one tree block per UFO leg (a word and chirality each).

    ``blocks[i]`` attaches to UFO leg i + 1 with its incoming field, the
    reading of the colour and of the family recursion. The kernel reads the
    same vertex as the tree engine does: the row and the momentum of the
    block ``blocks[kernel_legs[i]]``, whose tree particle is UFO particle
    i + 1, fill kernel leg i + 1 (module docstring). The term's value is
    ``sign * coupling(entry) * kernel(rows, momenta)`` times ``colour``.
    """

    entry: Any = field(repr=False)  # loops.model.KernelBasisEntry
    binding: CountertermSiteBinding = field(repr=False)
    blocks: tuple[Block, ...]
    kernel_legs: tuple[int, ...]
    sign: int
    colour: ConnectedColorTensor = field(repr=False)


@dataclass(frozen=True, slots=True)
class PromotedRootEntry(KernelBasisEntry):
    """A root vertex entry read on a scalar-vector mixing line (WP3.2
    EWI-7, routed item (v)): its scalar leg ``promoted_leg`` (0-based) takes
    the line's vector current, contracted with the leg's momentum, exactly as
    the scalar-side vertex copies of ``models.ct_line_insertions`` do inside
    the blocks (Lorentz structure ``L * P(mu_k, k)``,
    ``_promoted_lorentz``). Its key (``collection``) and Lorentz name differ
    from the entry's; colour, coupling and orders are the entry's. The kernel
    catalogue builds it from the base structure ``base_lorentz_name``
    (``catalogue._CatalogueBuilder.add_counterterm_sites``)."""

    promoted_leg: int = -1
    base_lorentz_name: str = ""


def promoted_root_entry(entry: Any, leg: int) -> PromotedRootEntry:
    """``entry`` with its scalar leg ``leg`` (0-based) promoted to the vector
    leg of a scalar-vector mixing line (``PromotedRootEntry``)."""

    values = {item.name: getattr(entry, item.name) for item in fields(KernelBasisEntry)}
    values["collection"] = f"{entry.collection}/promoted-leg-{leg + 1}"
    values["lorentz_name"] = f"{entry.lorentz_name}__P{leg + 1}"
    return PromotedRootEntry(
        **values, promoted_leg=leg, base_lorentz_name=entry.lorentz_name
    )


def _loop_content(entries: Sequence[Any]) -> frozenset | None:
    """The loop content of a rule's entries, invariant under conjugation:
    each loop group alternative as its sorted |PDG| codes (None without
    declared loop particles, e.g. an ordinary vertex)."""

    found = {
        tuple(sorted(abs(int(pdg)) for pdg in alternative))
        for entry in entries
        for alternative in (getattr(entry, "loop_particles", None) or ())
    }
    return frozenset(found) if found else None


class ConjugateVertices:
    """The model's conjugate rule of each vertex not closed under conjugation.

    The root closure attaches each block to the UFO leg of its incoming field
    f (the families' reading) and reads it, as the tree engine does, as the
    current of tree particle anti(f) on the leg of that particle. A vertex
    whose particles are not closed under conjugation has no such leg; the
    tree engine reads those blocks through the vertex of the antiparticles
    at the same coupling orders, the conjugate rule (a Hermitian Lagrangian
    holds it): [u~ d W+] for the blocks of [d~ u W-]. Its couplings and
    Lorentz structures are the vertex's own complex-conjugated with the chiral
    projectors exchanged (the CKM phase of [u~ b W+], the Yukawas of
    [t~ b G+]), so the root takes the conjugate rule's entries.

    ``groups`` are (name, entries) of one ordinary vertex or counterterm rule
    each (``KernelBasisEntry``, one collection and kind per group). A group's
    conjugate is the group of the same collection, kind and coupling orders on
    the antiparticles (among several, the one with the same loop content:
    a counterterm vertex split by loop group); it must be unique, and both
    must have one colour
    structure (the closure keeps the vertex's own colour, the incoming
    reading of the colour seeds, and the pairing of several structures with
    the conjugate's is not established), and neither may have a ghost leg
    (an electroweak ghost line, whose reading through the conjugate is not
    established); the same conditions as ``conventions.
    conjugate_vertex_rules`` for the loop segments. Otherwise ``of`` refuses.
    """

    def __init__(self, groups: Iterable[tuple[str, Sequence[Any]]], native_model: Any):
        data = model_data(native_model)
        self._anti = {p["name"]: p["antiname"] for p in data["particles"]}
        self._ghosts = frozenset(
            p["name"] for p in data["particles"] if int(p["spin"]) == GHOST_SPIN
        )
        self._groups: dict[tuple, list[tuple[str, tuple[Any, ...]]]] = defaultdict(list)
        for name, entries in groups:
            entries = tuple(entries)
            if entries:
                self._groups[self._key(entries, entries[0].particle_names)].append(
                    (name, entries)
                )

    @staticmethod
    def _key(entries: Sequence[Any], names: Sequence[str]) -> tuple:
        first = entries[0]
        return (
            first.collection,
            first.kind,
            tuple(sorted(names)),
            frozenset(add_orders(tuple(entry.order)) for entry in entries),
        )

    def named(self, entries: Sequence[Any]) -> tuple[str, tuple[Any, ...]]:
        """(name, entries) of the conjugate rule of a vertex's ``entries``."""

        first = entries[0]
        names = [self._anti[name] for name in first.particle_names]
        found = self._groups.get(self._key(entries, names), [])
        if len(found) > 1:
            # A counterterm vertex split by loop content (one rule per UFO
            # loop group, e.g. the R2 [W- A G+] of MG5_aMC's NLOCT UFOs, one
            # per quark pair): Hermiticity pairs each rule with the
            # conjugate's rule of the same loop content (|PDG| per group).
            content = _loop_content(entries)
            found = [item for item in found if _loop_content(item[1]) == content]
        if len(found) != 1:
            raise NotImplementedError(
                f"{first.vertex_name} [{' '.join(first.particle_names)}] is not "
                f"closed under conjugation and has {len(found)} conjugate rules "
                f"[{' '.join(names)}] at its coupling orders; its tree reading "
                "needs exactly one"
            )
        name, conjugate = found[0]
        if self._ghosts.intersection(first.particle_names):
            raise NotImplementedError(
                f"{first.vertex_name} [{' '.join(first.particle_names)}] is not "
                "closed under conjugation and has a ghost leg, whose reading "
                "through the conjugate rule is not established"
            )
        for group in (entries, conjugate):
            if len({entry.color_index for entry in group}) != 1:
                raise NotImplementedError(
                    f"{group[0].vertex_name} is read through its conjugate rule "
                    "and has several colour structures, whose pairing with the "
                    "conjugate's is not established"
                )
        return name, conjugate

    def of(self, entries: Sequence[Any]) -> tuple[Any, ...]:
        """The entries of the conjugate rule of a vertex's ``entries``."""

        return self.named(entries)[1]

    def closed(self, entries: Sequence[Any]) -> bool:
        """Whether the vertex of ``entries`` is closed under conjugation."""

        return not entries or closed_under_conjugation(
            entries[0].particle_names, self._anti
        )


class SiteClosure:
    """Kinematic and colour terms of vertices closed at the root of tree blocks.

    ``blocks`` are the loop request's tree blocks, ``native_model`` the native
    model of the block currents' field names and ``catalogue`` a kernel
    catalogue with ``ct-site`` bindings for every entry that is closed (the
    conjugate rules of the vertices not closed under conjugation included).
    ``conjugates`` finds those conjugate rules; without it such vertices are
    refused.
    """

    def __init__(
        self,
        blocks: TreeBlocks,
        native_model: Any,
        catalogue: LoopKernelCatalogue,
        conjugates: ConjugateVertices | None = None,
    ):
        data = model_data(native_model)
        self.tree = blocks
        self.catalogue = catalogue
        self._fields = _NativeFields.of(data)
        self._by_name = {entry["name"]: entry for entry in data["particles"]}
        self._anti = {name: entry["antiname"] for name, entry in self._by_name.items()}
        self.legs = process_color_legs(blocks.process)
        self._representations = {leg.label: leg.representation for leg in self.legs}
        # External fermion endpoints: label -> all-outgoing representation
        # (3 or -3) of a coloured one, ``SINGLET_END`` of a colour-singlet one
        # (``conventions.fermion_ends``, shared with ``LeafWeights`` and
        # ``_root_sign``; other representations are refused). Labels rank the
        # singlet sources as the tree engine's source slots do
        # (``colour_singlet_sign``).
        labels = [leg.label for leg in blocks.process.legs]
        if labels != sorted(labels):
            raise ValueError("process legs out of label order")
        self._ends = fermion_ends(
            (leg.label, self._representations[leg.label])
            for leg in blocks.process.legs
            if self._fields.by_pdg[int(leg.outgoing_pdg)]["spin"] == 2
        )
        self._bindings = {
            binding.key: binding for binding in catalogue.counterterm_sites
        }
        # (entry key, kernel id) -> ``_reading_sign``.
        self._reading: dict[tuple[Any, int], int] = {}
        self._unsupported = frozenset(
            item.entry_key for item in catalogue.unsupported_counterterms
        )
        self._blocks: dict[BlockCurrent, list[Block]] = defaultdict(list)
        for block in blocks.blocks:
            key = block.key
            if key.auxiliary_kind is None:
                field_name = self._fields.incoming(key.particle_id)["name"]
                self._blocks[
                    BlockCurrent(key.external_mask, field_name, key.coupling_orders)
                ].append(block)
        self._colours: dict[tuple, ConnectedColorTensor] = {}
        self._conjugates = conjugates

    def share_colours(self, other: SiteClosure) -> None:
        """Share ``other``'s closed vertex colours (one cache for both) and,
        unless this closure has its own, its conjugate rules.

        A closed colour depends on the vertex colour, the attached fields and
        the blocks' colour flows (its key) and on the process legs and the
        native fields, which must agree: the CT program's closure and its
        slice's (the Born closure) close the same vertices on blocks of the
        same words, so each colour is reduced once.
        """

        if (
            self.legs != other.legs
            or self._representations != other._representations
            or self._fields.by_pdg != other._fields.by_pdg
        ):
            raise ValueError("closures of different processes or models")
        self._colours = other._colours
        if self._conjugates is None:
            self._conjugates = other._conjugates

    @property
    def fermion_ends(self) -> Mapping[int, int]:
        """External fermions: label -> all-outgoing representation (3, -3)
        of a coloured one, ``SINGLET_END`` of a colour-singlet one."""
        return MappingProxyType(self._ends)

    def terms(
        self, site: CountertermSite, cancelled: Iterable[Cell] = ()
    ) -> tuple[SiteTerm, ...]:
        """Terms of ``site``, without the entries of its ``cancelled`` cells."""

        cancelled = frozenset(cancelled)
        entries = tuple(
            entry for entry in site.rule.entries if entry_cell(entry) not in cancelled
        )
        if len(entries) != len(site.rule.entries) and not self._self_conjugate(
            site.rule.entries
        ):
            raise NotImplementedError(
                f"{site.rule_name}: cancelled cells of a vertex read through its "
                "conjugate rule"
            )
        return self.vertex_terms(entries, site.blocks)

    def cancelled_cells(
        self,
        sites: Iterable[CountertermSite],
        coefficient: Callable[[Any, Any, int], Any],
        vanishes: Callable[[Any], bool],
    ) -> dict[CountertermSite, frozenset[Cell]]:
        """Cells without a ``ct-site`` kernel that cancel between rules.

        An entry the catalogue records as unsupported (its Lorentz structure
        does not normalise, e.g. a Levi-Civita tensor) has no kernel. The same
        cell of the other rules of its UFO vertex closed on the same blocks
        has the same kernel, colour and signs, so the cell contributes the
        kernel times the sum over those rules of the number of admitted
        alternatives times ``coefficient(rule, entry, power)`` (exact). Where
        ``vanishes`` holds for that sum at every Laurent power, the cell is
        cancelled on those sites (``terms(site, cancelled)``); otherwise its
        entries stay and ``terms`` refuses them for want of a kernel.
        """

        groups: dict[tuple, list[tuple[CountertermSite, Any]]] = defaultdict(list)
        for site in sites:
            if site.rule.whole_rule_continuation is not None:
                continue
            for entry in site.rule.entries:
                if _entry_key(entry) in self._unsupported:
                    groups[site.blocks, entry_cell(entry)].append((site, entry))
        out: dict[CountertermSite, set[Cell]] = defaultdict(set)
        for (_blocks, cell), members in groups.items():
            if all(
                vanishes(
                    reduce(
                        add,
                        (
                            len(site.alternatives)
                            * coefficient(site.rule, entry, power)
                            for site, entry in members
                        ),
                    )
                )
                for power in LAURENT_POWERS
            ):
                for site, _entry in members:
                    out[site].add(cell)
        return {site: frozenset(cells) for site, cells in out.items()}

    def vertex_terms(
        self,
        entries: Iterable[Any],
        attachments: Sequence[BlockCurrent],
        promoted: frozenset[int] = frozenset(),
    ) -> tuple[SiteTerm, ...]:
        """Terms of the vertex ``entries`` with ``attachments`` on its UFO legs.

        ``promoted`` are the ids of blocks that carry the vector current of a
        scalar-vector mixing line read as its scalar (``ct_closure.CloneLine.
        promoted``): the kernel leg such a block fills reads it through the
        entry's promoted copy (``promoted_root_entry``)."""

        options = []
        for current in attachments:
            rows = self._blocks.get(current)
            if not rows:
                raise ValueError(f"no tree block for {current}")
            options.append(rows)
        sign = self._root_sign(attachments)
        entries = tuple(entries)
        # A vertex read through its conjugate rule keeps its own single
        # colour structure (``read_entries``).
        colour_entry = None if self._self_conjugate(entries) else entries[0]
        entries = self.read_entries(entries)
        out = []
        for entry in entries:
            legs = self._kernel_legs(entry, attachments)
            for choice in product(*options):
                read = entry
                if promoted:
                    hits = [
                        k
                        for k, index in enumerate(legs)
                        if choice[index].id in promoted
                    ]
                    if len(hits) > 1:
                        raise ValueError(
                            f"{entry.vertex_name}: several mixing lines at one root"
                        )
                    if hits:
                        read = promoted_root_entry(entry, hits[0])
                chiralities = tuple(choice[index].key.chirality for index in legs)
                key = CountertermSiteKey(*_entry_key(read), chiralities)
                binding = self._bindings.get(key)
                if binding is None:
                    if any(chiralities):
                        continue  # a chirality selection rule of this entry
                    raise ValueError(f"catalogue has no ct-site kernel for {key}")
                wick = 1
                for block in choice:
                    wick *= block.wick_sign
                out.append(
                    SiteTerm(
                        read,
                        binding,
                        choice,
                        legs,
                        sign * wick * self._reading_sign(read, binding),
                        self._colour(colour_entry or entry, attachments, choice),
                    )
                )
        return tuple(out)

    def _reading_sign(self, entry: Any, binding: Any) -> int:
        """The sign between the tree engine's reading of a vertex and the
        root closure's (module docstring, "Colour, kinematics and signs").

        Both readings fill each kernel leg with the block row of its tree
        particle (``_kernel_legs``; a vertex not closed under conjugation
        through its conjugate rule, ``read_entries``) and with that block's
        momentum (``value``, the native closure terms alike), but the tree
        engine takes the tree momenta and the outgoing colour flows, the
        closure the momenta flowing in and the incoming colour: every
        momentum is reversed and every adjoint colour flow transposed (an
        ``f`` changes sign; ``d``, generators between the swapped ends of a
        fundamental line, two-index traces and identities do not). The
        closure value is therefore the tree engine's, the reading of every
        other vertex of the amplitude, times (-1)^(momentum degree of the
        kernel) times the colour's transposition sign, on every vertex. A
        kernel neither even nor odd in the momenta or a colour without a
        transposition sign is refused (``ReadingSignError`` naming the
        vertex).
        """

        key = (_entry_key(entry), binding.kernel_id)
        found = self._reading.get(key)
        if found is None:
            kernel = self.catalogue.kernels[binding.kernel_id]
            what = (
                f"vertex {entry.vertex_name} [{' '.join(entry.particle_names)}] "
                f"(Lorentz structure {entry.lorentz_name}, colour {entry.color})"
            )
            found = _momentum_parity(kernel, what) * _colour_transposition_sign(
                entry.color, what
            )
            self._reading[key] = found
        return found

    def read_entries(self, entries: Sequence[Any]) -> tuple[Any, ...]:
        """The entries the root closure reads for a vertex's ``entries``
        (``vertex_terms``), so their couplings are the ones its terms bind:
        ``entries`` for a vertex closed under conjugation, else its conjugate
        rule's. The tree engine reads the blocks of an open vertex through the
        conjugate rule (module docstring): that rule's entries give kernels
        and couplings, the vertex's single colour structure the colour."""

        entries = tuple(entries)
        if self._self_conjugate(entries):
            return entries
        if self._conjugates is None:
            raise NotImplementedError(
                f"{entries[0].vertex_name} is not closed under conjugation; "
                "the tree reading of its legs needs its conjugate rule"
            )
        return self._conjugates.of(entries)

    def _self_conjugate(self, entries: Sequence[Any]) -> bool:
        return not entries or closed_under_conjugation(
            entries[0].particle_names, self._anti
        )

    def value(
        self,
        term: SiteTerm,
        rows: Sequence[Sequence[Any]],
        momenta: Sequence[Sequence[Any]],
        carrier: Carrier,
    ) -> Any:
        """Kinematic value of one term: its ``ct-site`` kernel on tree rows.

        ``rows[i]`` is the tree row of ``term.blocks[i]`` and ``momenta[i]``
        its tree momentum (the leg-mask momentum, initial legs negated). The
        kernel takes the rows and the momenta flowing into the vertex in its
        own leg order (``term.kernel_legs``). Coupling, sign and colour are
        not applied.
        """

        kernel = self.catalogue.kernels[term.binding.kernel_id]
        currents = [value for index in term.kernel_legs for value in rows[index]]
        # Kernel leg i + 1 takes the row and the momentum of the block of its
        # tree particle, the momentum flowing in (``_reading_sign``).
        vectors = [-value for index in term.kernel_legs for value in momenta[index]]
        sources = {"block-current": currents, "block-momentum": vectors}
        try:
            values = [sources[item.role][item.component] for item in kernel.inputs]
        except KeyError as error:
            raise NotImplementedError(
                f"ct-site kernel input role {error} needs a model-parameter binding"
            ) from error
        (value,) = carrier.kernel(kernel, values)
        return value

    def _kernel_legs(
        self, entry: Any, attachments: Sequence[BlockCurrent]
    ) -> tuple[int, ...]:
        """Attachment index per kernel leg: the block whose tree particle is the
        leg's particle (the tree engine's reading of a vertex). ``entry`` is
        an entry of the attached vertex, or of its conjugate rule when the
        vertex is not closed under conjugation (``vertex_terms``)."""

        free = list(range(len(attachments)))
        legs = []
        for name in entry.particle_names:
            wanted = self._by_name[name]["antiname"]
            index = next((i for i in free if attachments[i].field == wanted), None)
            if index is None:
                raise ValueError(
                    f"{entry.vertex_name}: no attachment of tree particle {name} "
                    f"among {[a.field for a in attachments]}"
                )
            free.remove(index)
            legs.append(index)
        return tuple(legs)

    def close(
        self, network: ColourNetwork, attachments: Sequence[BlockCurrent]
    ) -> tuple[tuple[tuple[Block, ...], int, ConnectedColorTensor], ...]:
        """A vertex colour ``network`` closed on every block choice.

        ``network`` labels the vertex's UFO legs ``("leg", i)`` (1-based),
        with ``attachments[i - 1]`` on leg i. Returns (blocks, sign, colour)
        per choice, the sign being the root sign times the blocks' Wick
        signs, as for ``terms``; whole-rule colour continuations use it.
        """

        options = []
        for current in attachments:
            rows = self._blocks.get(current)
            if not rows:
                raise ValueError(f"no tree block for {current}")
            options.append(rows)
        sign = self._root_sign(attachments)
        out = []
        for choice in product(*options):
            wick = 1
            for block in choice:
                wick *= block.wick_sign
            out.append((choice, sign * wick, self._closed(network, network, choice)))
        return tuple(out)

    def _colour(
        self, entry: Any, attachments: Sequence[BlockCurrent], choice: Sequence[Block]
    ) -> ConnectedColorTensor:
        key = (entry.color, tuple(a.field for a in attachments))
        colours = [int(self._by_name[a.field]["color"]) for a in attachments]
        return self._closed(
            key, lambda: ufo_colour_network(entry.color, colours), choice
        )

    def _closed(
        self, key: Any, network: Any, choice: Sequence[Block]
    ) -> ConnectedColorTensor:
        """The vertex colour (a network, or a factory of one) closed on blocks."""

        key = (key, *(b.colour_flow for b in choice))
        cached = self._colours.get(key)
        if cached is not None:
            return cached
        if callable(network):
            network = network()

        def rename(label):
            kind, value = label
            return ("block", value - 1) if kind == "leg" else label

        factors = [tuple(p.relabel(rename) for p in network)]
        for index, block in enumerate(choice):
            factors.append(
                block_colour_network(
                    block.colour_flow,
                    self._representations,
                    int(self._fields.by_pdg[block.key.particle_id]["color"]),
                    ("block", index),
                )
            )
        external = {label: label for label in self._representations}
        tensor = reduce_network(multiply(*factors), self.legs, external)
        self._colours[key] = tensor
        return tensor

    def _root_sign(self, attachments: Sequence[BlockCurrent]) -> int:
        """FeynKit diagram parity over the product of the blocks' parities,
        and the Koszul sign of the colour-singlet fermions.

        Each block's diagrams are signed by the parity of their coloured
        fermion lines listed (fundamental, antifundamental) with the open end
        last; the ratio to the parity of the full trees depends only on the
        labels, so any pairing (here the identity pairing) represents it.
        Colour-singlet fermions carry the tree engine's Koszul signs inside
        the block rows, so the root completes them as the loops do
        (``conventions.colour_singlet_sign``, design §3.3): the blocks'
        singlet supports in the order of the root's legs, and the two open
        ends of a singlet line that closes at the root.
        """

        pairs: list[tuple[int, int]] = []
        opened: list[tuple[int, int]] = []  # (open-end representation, partner)
        supports: list[tuple[int, ...]] = []
        singlet_ends: list[tuple[int, str]] = []
        sign = 1
        for current in attachments:
            labels = [label for label in self._ends if current.mask >> (label - 1) & 1]
            starts = [label for label in labels if self._ends[label] == 3]
            stops = [label for label in labels if self._ends[label] == -3]
            singlets = tuple(
                label for label in labels if self._ends[label] == SINGLET_END
            )
            if singlets:
                supports.append(singlets)
            entry = self._by_name[current.field]
            end = (
                fermion_end(entry["color"], f"fermion block {current.field}")
                if entry["spin"] == 2
                else None
            )
            if end == SINGLET_END:
                if not singlets:
                    raise FermionSignError(
                        f"a colour-singlet fermion block {current.field} holds no "
                        "colour-singlet fermion end"
                    )
                singlet_ends.append((len(supports) - 1, open_end_orientation(entry)))
            elif end is not None:
                (starts if end == 3 else stops).append(None)
            sign *= identity_pairing_sign(starts, stops)
            for start, stop in zip(
                sorted(starts, key=_open_last),
                sorted(stops, key=_open_last),
                strict=True,
            ):
                if start is None:
                    opened.append((3, stop))
                elif stop is None:
                    opened.append((-3, start))
                else:
                    pairs.append((start, stop))
        if opened:
            ends = dict(opened)
            if len(opened) != 2 or set(ends) != {3, -3}:
                raise NotImplementedError(
                    "site closure joins one coloured fermion line at the root; "
                    "vertices with more fermion lines need their spinor flow"
                )
            pairs.append((ends[-3], ends[3]))
        if singlet_ends and len(singlet_ends) != 2:
            raise NotImplementedError(
                "site closure joins one colour-singlet fermion line at the root; "
                "vertices with more fermion lines need their spinor flow"
            )
        if supports:
            lines = [(*singlet_ends[0], *singlet_ends[1])] if singlet_ends else []
            sign *= colour_singlet_sign(supports, lines)
        return sign * _parity([label for pair in pairs for label in pair])


def _open_last(label: int | None) -> tuple[bool, int]:
    return (label is None, 0 if label is None else label)


def _entry_key(entry: Any) -> tuple[str, int, int, int, int]:
    return (
        entry.collection,
        entry.vertex_id,
        entry.color_index,
        entry.lorentz_index,
        -1 if entry.loop_group_index is None else entry.loop_group_index,
    )


def _momentum_parity(kernel: Any, what: str = "a ct-site kernel") -> int:
    """(-1)^(momentum degree) of a ``ct-site`` kernel: 1 when its components
    are even in its block momenta, -1 when odd; mixed parity is refused
    (``ReadingSignError`` naming ``what``)."""

    from symbolica import E, Replacement

    flipped = [
        Replacement(E(item.symbol), -E(item.symbol))
        for item in kernel.inputs
        if item.role == "block-momentum"
    ]
    if not flipped:
        return 1
    parities = set()
    for expression in kernel.exact_expressions:
        value = E(expression)
        mirrored = value.replace_multiple(flipped)
        if (mirrored - value).expand() == E("0"):
            parities.add(1)
        elif (mirrored + value).expand() == E("0"):
            parities.add(-1)
        else:
            parities.add(0)
    if len(parities) != 1 or 0 in parities:
        raise ReadingSignError(
            f"{what}: its kernel is neither even nor odd in the block momenta, "
            "so the root closure has no sign between the tree engine's reading "
            "of its legs and its own"
        )
    return parities.pop()


def entry_cell(entry: Any) -> Cell:
    """The UFO coupling cell of a rule entry (its key without the loop group)."""

    return _entry_key(entry)[:4]


def vanishes_identically(compiled: Any, expression: Any) -> bool:
    """Whether ``expression`` vanishes for every value of the model's inputs.

    ``expression`` is in the compiled payload's parameter symbols
    (``UFO::{}::name``); internal parameters are replaced by their exact
    resolved definitions (in the model's external parameters) before the
    expansion is compared with zero. A cancellation that needs particular
    external values is not found (the caller then keeps the terms).
    """

    from symbolica import E

    from pyamplicol._internal.physics.symbols import symbols

    zero = E("0")
    value = expression.expand()
    if value == zero:
        return True
    model_symbols = symbols.model(compiled.name)
    value = model_symbols.expression(value.to_canonical_string())
    return resolved_internal_parameters(compiled, value).expand() == zero


def restricted_to_zero(compiled: Any, entries: Sequence[Any]) -> bool:
    """Whether every coupling of a vertex's loop-model ``entries``
    (``KernelBasisEntry`` rows of ``LoopModelIR.kernels``: one vertex, or one
    vertex at one coupling order) vanishes identically in the restricted
    model ``compiled``, so that its tree model holds no term of them: each
    coupling with the source helpers it calls replaced by their exact bodies
    (``ScalarFunctionLibrary.inline``; e.g. UFO's ``complexconjugate`` of a
    CKM element that the restriction sets to zero) and its internal
    parameters resolved to the restriction's external ones
    (``vanishes_identically``). The decision reads the coupling values, never
    vertex, particle or parameter names. False when it cannot be shown (no
    entries, an internal parameter left unresolved, a helper body that keeps
    a conditional of the parameters).

    The one rule for vertices a model's restriction removes (fail-closed-2):
    the loop catalogue leaves them out (``vanishing_loop_vertices``,
    ``families.ModelLoopCatalogue``) and the counterterm root closure skips
    them (``ct_closure``, WP3.2 EWI-5)."""

    if not entries:
        return False
    try:
        return all(
            vanishes_identically(compiled, _inlined_coupling(entry))
            for entry in entries
        )
    except ValueError:  # an internal parameter left unresolved
        return False


def _inlined_coupling(entry: Any) -> Any:
    """The coupling of a loop-model entry with every source helper call
    replaced by its exact body (the entry's scalar program's library)."""

    value = entry.scalar_expression().value
    program = getattr(entry, "scalar_program", None)
    if program is None:
        return value
    return program.library.inline(value, lambda _arguments: True)


def vanishing_loop_vertices(
    compiled: Any, kernels: Iterable[Any], names: Iterable[str] | None = None
) -> frozenset[tuple[str, Orders]]:
    """The (vertex name, coupling orders) of the tree vertices of a loop
    model's ``kernels`` (``LoopModelIR.kernels``; of the vertex names
    ``names`` only, if given) every coupling of which at those orders
    vanishes identically at the restriction of ``compiled``
    (``restricted_to_zero``): such a vertex contributes nothing, and the
    restricted tree model holds no colour or Lorentz term of it."""

    wanted = None if names is None else frozenset(names)
    groups: dict[tuple[str, Orders], list[Any]] = defaultdict(list)
    for entry in kernels:
        if entry.kind != "tree":
            continue
        if wanted is not None and entry.vertex_name not in wanted:
            continue
        groups[entry.vertex_name, add_orders(tuple(entry.order))].append(entry)
    return frozenset(
        key for key, entries in groups.items() if restricted_to_zero(compiled, entries)
    )


def resolved_internal_parameters(compiled: Any, expression: Any) -> Any:
    """``expression`` (in the model's symbols) with every internal parameter
    replaced by its resolved definition, which the model compiler writes in
    the external parameters alone; fails if one is left."""

    from symbolica import Replacement

    definitions = _internal_definitions(compiled)
    read = set(expression.get_all_symbols(False))
    present = [
        Replacement(symbol, definition)
        for symbol, definition in definitions
        if symbol in read
    ]
    if present:
        expression = expression.replace_multiple(present)
        read = set(expression.get_all_symbols(False))
    left = sorted(str(symbol) for symbol, _ in definitions if symbol in read)
    if left:
        raise ValueError(f"internal parameters {left} are not resolved")
    return expression


def _internal_definitions(compiled: Any) -> tuple[tuple[Any, Any], ...]:
    """(symbol, parsed resolved definition) of every internal parameter of
    ``compiled``, in record order, formed once per parameter table."""

    from symbolica import E

    from pyamplicol._internal.physics.symbols import symbols

    records = compiled.ir.parameters
    found = _INTERNAL_DEFINITIONS.get(id(records))
    # The table itself is kept, so its id is not reused while cached.
    if found is not None and found[0] is records:
        return found[1]
    model_symbols = symbols.model(compiled.name)
    definitions = tuple(
        (model_symbols.symbol(record.name), E(f"({record.resolved_expression})"))
        for record in records
        if record.nature == "internal"
    )
    if len(_INTERNAL_DEFINITIONS) >= 64:
        _INTERNAL_DEFINITIONS.clear()
    _INTERNAL_DEFINITIONS[id(records)] = (records, definitions)
    return definitions


_INTERNAL_DEFINITIONS: dict[int, tuple[Any, tuple[tuple[Any, Any], ...]]] = {}


# ----------------------------------------------------------------- reference values


ColourVector = dict[ColorTensor, Any]


@dataclass(frozen=True)
class Carrier:
    """Arithmetic of a reference evaluation (binary64 or arbitrary precision).

    ``number(c, p)`` converts the exact complex rational ``c`` times
    ``2^(-p/2)``; ``kernel(spec, values)`` evaluates a catalogue kernel on its
    input values, in the kernel's input order. Values only need ``+``, ``*``
    (also by int), unary ``-`` and ``conjugate()``.
    """

    number: Callable[[ExactColorCoefficient, int], Any]
    kernel: Callable[[Any, Sequence[Any]], Sequence[Any]]
    zero: Any = 0


def add_colour(
    vector: ColourVector, tensor: ConnectedColorTensor, value: Any, carrier: Carrier
) -> None:
    """``vector += value * tensor`` at N_c = 3, keyed by literal structures."""

    for term in tensor.terms:
        weight = carrier.number(
            term.coefficient.scaled(Fraction(NC) ** term.nc_power),
            tensor.normalization_power,
        )
        vector[term.tensor] = vector.get(term.tensor, carrier.zero) + weight * value


class ColourOverlaps:
    """Exact <t|t'> of literal colour structures at N_c = 3, cached."""

    def __init__(self, legs: Iterable[Any]):
        self.legs = tuple(legs)
        self._cache: dict[tuple[ColorTensor, ColorTensor], Fraction] = {}

    def __call__(self, left: ColorTensor, right: ColorTensor) -> Fraction:
        key = (left, right)
        cached = self._cache.get(key)
        if cached is None:
            one = ExactColorCoefficient(1)
            cells = contract_connected_tensor_nc_terms(
                ConnectedColorTensor(self.legs, (TensorTerm(one, left),)),
                ConnectedColorTensor(self.legs, (TensorTerm(one, right),)),
            )
            total = ExactColorCoefficient()
            for power, value in cells.items():
                total = total + value.scaled(Fraction(NC) ** power)
            if total.imag:
                raise ValueError("a literal colour overlap is not real")
            cached = self._cache[key] = total.real
        return cached

    def interfere(self, bra: ColourVector, ket: ColourVector, carrier: Carrier) -> Any:
        """sum_{t,t'} conj(bra[t]) ket[t'] <t|t'>."""

        total = carrier.zero
        for left, a in bra.items():
            conjugate = a.conjugate()
            for right, b in ket.items():
                overlap = self(left, right)
                if overlap:
                    weight = carrier.number(ExactColorCoefficient(overlap), 0)
                    total = total + conjugate * b * weight
        return total


__all__ = [
    "COMPONENT_BY_KIND",
    "COUNTERTERM_KINDS",
    "LAURENT_POWERS",
    "SPECIES_MATCHED_KINDS",
    "Carrier",
    "Cell",
    "CertifiedZero",
    "ColourOverlaps",
    "ColourVector",
    "CountertermCompletenessError",
    "CountertermSite",
    "CountertermSites",
    "Coverage",
    "PromotedRootEntry",
    "ReadingSignError",
    "Signature",
    "SiteClosure",
    "SiteTerm",
    "add_colour",
    "closed_fermion_zeros",
    "divergent_witnesses",
    "entry_cell",
    "generate_counterterm_sites",
    "ghost_partners",
    "promoted_root_entry",
    "restricted_to_zero",
    "ufo_colour_network",
    "vanishes_identically",
    "vanishing_loop_vertices",
    "witness_alternatives",
]
