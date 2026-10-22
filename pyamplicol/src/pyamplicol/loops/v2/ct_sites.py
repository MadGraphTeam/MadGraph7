# SPDX-License-Identifier: 0BSD
"""Counterterm sites: the slice form and permanent oracle of design §6.2-§6.6.

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

Admission (design §6.2) follows MadLoop's ``set_LoopCT_vertices`` as the
legacy ``loops.counterterms.match_counterterms`` does, with loop graphs
replaced by *witnesses*: admitted family routings whose structural rank
satisfies R >= 2N - 4. A witness signature is the multiset of its attached
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
signature, or by a certified zero supplied by a certifier. Faddeev-Popov ghost
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
  momentum (initial legs negated). Vertices whose particle multiset is not
  closed under conjugation (e.g. W couplings) need their conjugate rule for
  that reading and are refused;
* ``Block.wick_sign`` turns each block into its Feynman-rule current and the
  root sign completes FeynKit's diagram sign: the parity of the full coloured
  fermion line listing over the product of the blocks' listings
  (``loops.v2.blocks``). One coloured fermion line may close at the root.

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

from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from itertools import product
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


class CountertermCompletenessError(ValueError):
    """A divergent witness is covered neither by a rule nor by a certified zero."""


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
        """Source scalar outputs (coupling name, epsilon power) of every site."""
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
    families: FamilySet,
    blocks: BlockTable,
    rules: Mapping[str, Any],
    native_model: Any,
    selection: LoopOrderSelection,
    *,
    forbidden: Iterable[int] = (),
    zeros: Iterable[CertifiedZero] = (),
    whole_rules: bool = False,
) -> CountertermSites:
    """Admitted CT sites of a process, after the completeness check.

    ``rules`` are the prepared model's counterterm rules
    (``PreparedCountertermModel.rules``); ``blocks`` is the family recursion's
    block table and ``native_model`` the native model (or its JSON mapping)
    whose names both use. ``forbidden`` are the |PDG| codes the process
    excludes; ``zeros`` are certified zeros of divergent witnesses.
    ``whole_rules`` admits rules carrying a whole-rule colour continuation;
    the caller then closes their replacement terms (``SiteClosure.close``)
    instead of ``SiteClosure.terms``.
    """

    data = model_data(native_model)
    names = {int(entry["pdg_code"]): entry["name"] for entry in data["particles"]}
    for name, rule in rules.items():
        if rule.kind not in COUNTERTERM_KINDS:
            raise ValueError(f"counterterm rule {name} has unknown kind {rule.kind!r}")
        if rule.whole_rule_continuation is not None and not whole_rules:
            raise NotImplementedError(
                f"counterterm rule {name} carries a whole-rule colour continuation; "
                "site closure needs its UFO entries"
            )
    corrections = frozenset(selection.corrections)
    forbidden = frozenset(abs(int(pdg)) for pdg in forbidden)
    witnesses = divergent_witnesses(families, data)
    coverage = _coverage(
        witnesses, rules, names, corrections, ghost_partners(data), zeros
    )
    full = (1 << len(blocks.legs)) - 1
    sites = []
    for name in sorted(rules):
        rule = rules[name]
        if not _selected(rule, corrections):
            continue
        fields = tuple(names[pdg] for pdg in rule.particle_pdgs)
        for parts in _set_partitions(full, len(fields), len(fields)):
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


def admitted_alternatives(
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
) -> CountertermSite | None:
    site = CountertermSite(rule, name, blocks, ())
    if site.orders not in selection.loop_orders:
        return None
    admitted = admitted_alternatives(
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
    same vertex as the tree engine does: the row of the block
    ``blocks[kernel_legs[i]]``, whose tree particle is UFO particle i + 1,
    fills kernel leg i + 1 (module docstring). The term's value is
    ``sign * coupling(entry) * kernel(rows, momenta)`` times ``colour``.
    """

    entry: Any = field(repr=False)  # loops.model.KernelBasisEntry
    binding: CountertermSiteBinding = field(repr=False)
    blocks: tuple[Block, ...]
    kernel_legs: tuple[int, ...]
    sign: int
    colour: ConnectedColorTensor = field(repr=False)


class SiteClosure:
    """Kinematic and colour terms of vertices closed at the root of tree blocks.

    ``blocks`` are the loop request's tree blocks, ``native_model`` the native
    model of the block currents' field names and ``catalogue`` a kernel
    catalogue with ``ct-site`` bindings for every entry that is closed.
    """

    def __init__(
        self, blocks: TreeBlocks, native_model: Any, catalogue: LoopKernelCatalogue
    ):
        data = model_data(native_model)
        self.tree = blocks
        self.catalogue = catalogue
        self._fields = _NativeFields.of(data)
        self._by_name = {entry["name"]: entry for entry in data["particles"]}
        self.legs = process_color_legs(blocks.process)
        self._representations = {leg.label: leg.representation for leg in self.legs}
        # Coloured fermion endpoints (label -> all-outgoing representation).
        self._ends = {
            leg.label: self._representations[leg.label]
            for leg in blocks.process.legs
            if abs(self._representations[leg.label]) == 3
            and self._fields.by_pdg[int(leg.outgoing_pdg)]["spin"] == 2
        }
        self._bindings = {
            binding.key: binding for binding in catalogue.counterterm_sites
        }
        self._blocks: dict[BlockCurrent, list[Block]] = defaultdict(list)
        for block in blocks.blocks:
            key = block.key
            if key.auxiliary_kind is None:
                field_name = self._fields.incoming(key.particle_id)["name"]
                self._blocks[
                    BlockCurrent(key.external_mask, field_name, key.coupling_orders)
                ].append(block)
        self._colours: dict[tuple, ConnectedColorTensor] = {}

    @property
    def fermion_ends(self) -> Mapping[int, int]:
        """Coloured external fermions: label -> all-outgoing representation."""
        return MappingProxyType(self._ends)

    def terms(self, site: CountertermSite) -> tuple[SiteTerm, ...]:
        return self.vertex_terms(site.rule.entries, site.blocks)

    def vertex_terms(
        self, entries: Iterable[Any], attachments: Sequence[BlockCurrent]
    ) -> tuple[SiteTerm, ...]:
        """Terms of the vertex ``entries`` with ``attachments`` on its UFO legs."""

        options = []
        for current in attachments:
            rows = self._blocks.get(current)
            if not rows:
                raise ValueError(f"no tree block for {current}")
            options.append(rows)
        sign = self._root_sign(attachments)
        out = []
        for entry in entries:
            legs = self._kernel_legs(entry, attachments)
            for choice in product(*options):
                chiralities = tuple(choice[index].key.chirality for index in legs)
                key = CountertermSiteKey(*_entry_key(entry), chiralities)
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
                        entry,
                        binding,
                        choice,
                        legs,
                        sign * wick,
                        self._colour(entry, attachments, choice),
                    )
                )
        return tuple(out)

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
        kernel takes the rows in its own leg order and the momenta flowing
        into the vertex. Coupling, sign and colour are not applied.
        """

        kernel = self.catalogue.kernels[term.binding.kernel_id]
        currents = [value for index in term.kernel_legs for value in rows[index]]
        # Momenta keep the physical reading: UFO leg i + 1 carries the momentum
        # flowing in from the block whose incoming field is its particle.
        vectors = [-value for momentum in momenta for value in momentum]
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
        leg's particle (the tree engine's reading of a vertex)."""

        free = list(range(len(attachments)))
        legs = []
        for name in entry.particle_names:
            wanted = self._by_name[name]["antiname"]
            index = next((i for i in free if attachments[i].field == wanted), None)
            if index is None:
                raise NotImplementedError(
                    f"{entry.vertex_name} is not closed under conjugation; the tree "
                    "reading of its legs needs its conjugate rule"
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

        key = (key, *(b.key.word for b in choice))
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
        """FeynKit diagram parity over the product of the blocks' parities.

        Each block's diagrams are signed by the parity of their coloured
        fermion lines listed (fundamental, antifundamental) with the open end
        last; the ratio to the parity of the full trees depends only on the
        labels, so any pairing (here the identity pairing) represents it.
        """

        pairs: list[tuple[int, int]] = []
        opened: list[tuple[int, int]] = []  # (open-end representation, partner)
        sign = 1
        for current in attachments:
            labels = [label for label in self._ends if current.mask >> (label - 1) & 1]
            starts = [label for label in labels if self._ends[label] == 3]
            stops = [label for label in labels if self._ends[label] == -3]
            entry = self._by_name[current.field]
            if entry["spin"] == 2:
                if abs(int(entry["color"])) != 3:
                    raise NotImplementedError(
                        "a colour-singlet fermion line closing at the root needs "
                        "the Koszul sign rule of design §3.3"
                    )
                (starts if int(entry["color"]) == 3 else stops).append(None)
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
    "CertifiedZero",
    "ColourOverlaps",
    "ColourVector",
    "CountertermCompletenessError",
    "CountertermSite",
    "CountertermSites",
    "Coverage",
    "Signature",
    "SiteClosure",
    "SiteTerm",
    "add_colour",
    "admitted_alternatives",
    "divergent_witnesses",
    "generate_counterterm_sites",
    "ghost_partners",
    "ufo_colour_network",
]
