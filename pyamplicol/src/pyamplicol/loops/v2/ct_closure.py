# SPDX-License-Identifier: 0BSD
"""Counterterm amplitudes closed at leg 1 like the Born (design §6.1, §6.4).

The production form of the counterterms (CT) of the v2 engine. The tree
recursion over the CT-lowered model (``loops.v2.ct_recursion``) builds, in
loop-request mode, the colour-flow currents without and with one CT insertion
over every leg set not containing leg 1. The CT amplitude is then closed at
the vertex holding leg 1, exactly as the slice closes its Born
(``loops.v2.assemble._born_terms``): no CT tree and no CT placement is
listed, only the last step of the recursion, which every tree shares.

A one-CT tree rooted at leg 1 has its insertion

* **inside a block** (``ROOT_VERTEX``): an ordinary vertex at the root with
  exactly one attached CT = 1 current, whose rows already carry the lowered
  coupling ``S_slot * sum_rules M_rule * c_rule``;
* **on the line between the root and a block** (``ROOT_LINE``): a clone line
  (``models.ct_line_insertions``) whose consuming vertex copy is the root. The
  root reads the ordinary vertex on the clone current (insertion order 1);
  the copy's other half of the orders of ``Pi`` is added, and on a Weyl line
  the copy on the leg of the clone ``C~`` carries ``kappa``
  (``CountertermLoweredModel.line_factors``);
* **at the root** (``ROOT_COUNTERTERM``): a CT vertex holding leg 1 on CT = 0
  currents: the site closure of the rule with leg 1 as one block
  (``loops.v2.ct_sites``), i.e. the recursion's last step for that vertex.

Every term is a ``ct_sites.SiteTerm`` on the blocks of the CT program, so
colour (exact tensors at any N_c grade), signs and kinematics follow the
site and Born closures. Clone currents are read as the field the vertex
copies consuming them replace (``CountertermLoweredModel.line_fields``: the
field itself on a diagonal line, the field the line leaves the two-point
vertex into on a mixing line): their block keys carry that field's particle
id, and the insertion order of their orders marks them.
"""

from __future__ import annotations

import logging
from collections import OrderedDict, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from itertools import permutations, product
from typing import Any, Literal, cast

from pyamplicol.models.loading import (
    COUNTERTERM_INSERTION_ORDER,
    counterterm_slot_order,
)

from .blocks import TreeBlocks, _blocks, vetoed_particle_ids
from .ct_recursion import (
    AnomalousVetoError,
    CountertermRecursion,
    UnloweredCountertermError,
    _physical,
    rule_key,
)
from .ct_sites import (
    COMPONENT_BY_KIND,
    CountertermSite,
    SiteClosure,
    generate_counterterm_sites,
    resolved_internal_parameters,
    restricted_to_zero,
)
from .families import (
    LoopOrderSelection,
    Orders,
    _set_partitions,
    add_orders,
    model_data,
)

_LOG = logging.getLogger(__name__)

RootKind = Literal["vertex", "line", "counterterm"]
ROOT_VERTEX: RootKind = "vertex"
ROOT_LINE: RootKind = "line"
ROOT_COUNTERTERM: RootKind = "counterterm"
# The leg set of leg 1 alone (bit 0), where a CT amplitude closes.
_ROOT_LEG = 1


@dataclass(frozen=True, slots=True)
class RootReachability:
    """Loop-request reachability of the currents a leg-1 closure attaches:
    every leg set of the process that holds neither leg 1 (bit 0) nor all the
    other legs. A root attaches leg 1 and at least two currents (a two-point
    insertion next to leg 1 sits on an external line: field terms), so no
    root term reads the set of all legs but leg 1; that set would hold most
    of the program (80 % of its interactions at g g > g g g)."""

    full_mask: int

    def admits_mask(self, mask: int) -> bool:
        return (
            mask > 0
            and not mask & 1
            and mask & self.full_mask == mask
            and mask != self.full_mask & ~1
        )

    def admits_block(self, index: Any) -> bool:
        return self.admits_mask(index.external_mask)


@dataclass(frozen=True)
class InsertionReachability:
    """Whether a CT vertex can enter a tree of a request's CT program.

    REQUIREMENTS_ADDENDUM J: a vertex that cannot is inert (it contributes
    nothing and needs no lowering). ``currents`` maps every leg set of the
    process (the full set included) to the tree-engine particle ids of the
    currents without a CT insertion over it, each with its Pareto-minimal
    coupling orders. They come from the tree engine's own forward species and
    coupling-order sweep (``generation.dag_reachability.
    _species_coupling_orders_by_mask``, which the DAG compiler's closure-order
    reachability runs) over the program's model, order limits and vetoed
    particles, as ``CountertermBlocks.build`` compiles the program, without
    the compiler's colour-engine and leg-set filters. The sweep ignores
    colour, helicity and flavour-flow labels, so it over-approximates the
    currents: a vertex it finds in no tree enters none (sound); one it finds
    may still vanish.

    A vertex of UFO fields ``f_1..f_n`` and orders ``o`` enters a tree iff the
    legs split into ``n`` blocks whose currents enter its legs with orders
    that sum with ``o`` to within the physical ``limits``. The CT program
    reads a vertex in one of two ways, told apart by leg 1's block:

    * leg 1 alone is a block: a CT-root site (``CountertermClosure.
      root_sites``), which reads each block by its incoming field like the
      Born closure (``blocks.TreeBlockTable``): the current entering a leg of
      field ``f`` is the tree-engine particle ``anti(f)``;
    * otherwise: an insertion inside a current (the program's DAG, whose
      currents hold neither leg 1 nor all the other legs,
      ``RootReachability``), which the tree engine reads with the current of
      tree particle ``p`` on the UFO leg of ``p``
      (``models.compiler_entry._compile_oriented_kernels``), so the current
      entering a leg of field ``f`` is ``f``. The block holding leg 1 is the
      rest of the tree re-rooted at the vertex: the path from the vertex's
      result to the root's ordinary vertex, read through the conjugate of
      that vertex (an ordinary vertex set closed under conjugation, as a
      Hermitian Lagrangian gives), whose currents the sweep over leg sets
      with leg 1 holds.

    The two readings differ only for vertices whose fields are not closed
    under conjugation (e.g. W couplings): in u d~ > e+ ve g the R2 of
    [u~ d W+] sits inside the current of legs 1 and 5 and that of [d~ u W-]
    at the root. A block of one leg is the external particle itself; a
    two-point vertex needs two blocks of at least two legs, as the CT program
    puts two-point insertions on internal lines only
    (``models.ct_line_insertions``: a clone line is never external; an
    external line takes the particle's field factor). A one-point (tadpole)
    vertex is the root of a current over every leg. No loop, site or witness
    is consulted.
    """

    full_mask: int
    currents: Mapping[int, Mapping[int, tuple[Orders, ...]]]
    limits: Mapping[str, int]
    antiparticles: Mapping[int, int]

    @classmethod
    def build(
        cls,
        model: Any,
        process: Any,
        order_limits: Mapping[str, int],
        *,
        ignored_particle_ids: Iterable[int] = (),
    ) -> InsertionReachability:
        """Sweep the tree-engine ``model`` over ``process`` (a model process
        IR) within ``order_limits`` (a CT program's
        ``CountertermRecursion.coupling_order_limits``; its insertion order
        is set to 0 for the currents), without the ``ignored_particle_ids``
        (signed tree-engine ids, ``vetoed_particle_ids``)."""

        from pyamplicol.generation.dag_ordering import _labels_mask
        from pyamplicol.generation.dag_reachability import (
            _normalize_coupling_order_limits,
            _species_coupling_orders_by_mask,
        )

        limits = _normalize_coupling_order_limits(order_limits)
        insertion = COUNTERTERM_INSERTION_ORDER.upper()
        swept = _species_coupling_orders_by_mask(
            process,
            model,
            max_coupling_orders={**limits, insertion: 0},
            ignored_particle_ids=frozenset(int(p) for p in ignored_particle_ids),
            include_full_mask=True,
        )
        return cls(
            _labels_mask(leg.label for leg in process.legs),
            {
                mask: {pid: tuple(sorted(o)) for pid, o in species.items() if o}
                for mask, species in swept.items()
            },
            {k: v for k, v in limits.items() if k != insertion},
            {
                int(pdg): int(particle.anti_pdg)
                for pdg, particle in model.particles.items()
            },
        )

    def admits(self, rule: Any) -> bool:
        """Whether the vertex of ``rule`` (a ``CountertermRule``) can enter a
        tree of the program, at any of its entries' coupling orders."""

        from pyamplicol.generation.dag_reachability import (
            _combine_coupling_order_tuples,
            _coupling_orders_within_limits,
        )

        fields = [int(pdg) for pdg in rule.particle_pdgs]
        if not fields or any(f not in self.antiparticles for f in fields):
            return False  # a field the tree model does not hold (a ghost)
        own = {
            _combine_coupling_order_tuples(tuple(e.order)) for e in rule.entries
        } or {_combine_coupling_order_tuples(tuple(rule.order))}
        # Leg 1 alone: the closure's reading (a CT root); otherwise the tree
        # engine's (an insertion inside a current).
        readings = {
            True: set(permutations(self.antiparticles[f] for f in fields)),
            False: set(permutations(fields)),
        }
        for parts in _set_partitions(self.full_mask, len(fields), len(fields)):
            if len(parts) == 2 and any(not part & (part - 1) for part in parts):
                continue  # a two-point vertex on an external line
            for arrangement in readings[_ROOT_LEG in parts]:
                options = [
                    self.currents.get(part, {}).get(target, ())
                    for part, target in zip(parts, arrangement, strict=True)
                ]
                if not all(options):
                    continue
                for orders in own:
                    for choice in product(*options):
                        if _coupling_orders_within_limits(
                            _combine_coupling_order_tuples(orders, *choice),
                            self.limits,
                        ):
                            return True
        return False


@dataclass(frozen=True, slots=True)
class CloneLine:
    """A clone particle of the CT program read as the field ``field_id``.

    ``half`` are the physical orders of the vertex copy at either end of the
    line (half of those of ``Pi``), ``slot`` its (component, power, grade)
    and ``factor`` the Weyl factor ``kappa`` that the copies of this clone's
    leg carry, or ``None``. ``promoted``: the scalar side of a scalar-vector
    mixing line (``models.ct_line_insertions``), a vector current that only
    the vertex copies carrying the line momentum consume, never an ordinary
    vertex of the field it is read as.
    """

    field_id: int
    half: Orders
    slot: tuple[str, int, int]
    factor: str | None = None
    promoted: bool = False


@dataclass(frozen=True, slots=True)
class ClosureTerm:
    """One root term of a CT amplitude.

    The value is ``multiplicity * scale * coupling * term.sign * kernel``
    times ``term.colour`` times ``N_c^grade``: ``coupling`` is the source
    scalar ``(coupling name, epsilon power)`` of the root vertex entry (power
    0 for an ordinary root, ``power`` for a CT root; its own N_c grades come
    with it), ``scale`` a model expression (the Weyl factor) or ``None`` and
    ``grade`` the N_c grade of the inserted current's coupling (design
    §6.7). ``component``, ``power`` and ``orders`` (the physical amplitude
    orders) name the CT amplitude the term adds to.
    """

    term: Any  # ct_sites.SiteTerm or assemble.WholeRuleTerm
    root: RootKind
    component: str
    power: int
    orders: Orders
    coupling: tuple[str, int] | None
    multiplicity: int = 1
    scale: str | None = None
    grade: int = 0


@dataclass(frozen=True)
class CountertermBlocks:
    """The block currents of a CT program and its clone lines.

    ``blocks`` are the loop-request blocks of the CT-lowered model over the
    leg sets without leg 1, with clone currents keyed by their field;
    ``clones`` maps the id of every clone block to its ``CloneLine``.
    """

    recursion: CountertermRecursion
    blocks: TreeBlocks
    clones: Mapping[int, CloneLine]

    @classmethod
    def build(
        cls,
        recursion: CountertermRecursion,
        process: Any,
        *,
        ignored_particle_ids: Iterable[int] = (),
        progress_callback: Any = None,
        u1_grades: bool = False,
    ) -> CountertermBlocks:
        """Compile the CT program's blocks for ``process`` (a model process IR
        of the lowered model at full colour); ``u1_grades`` (leading and
        next-to-leading colour) makes every block exact in N_c
        (``generation.dag_grades``).

        Under ``dag_input.NATIVE_DAG`` the native builder makes the program
        (``dag_input.native_counterterm_blocks``, WP3.1 A4): its DAG is a
        ``dag_input.NativeDag`` handle, decoded only by the per-point Python
        rows (``CountertermProgram.rows``); ``progress_callback`` gets the
        native stage's start and finish. The tree engine's loop-request
        mode below is the oracle, and the stage for a request the native
        builder does not cover (``TreeBlocks.dag_fallback`` names it)."""

        from . import dag_input

        fallback = None
        if dag_input.NATIVE_DAG:
            stage = "counterterm-program"
            dag_input.native_progress(progress_callback, stage)
            try:
                native = dag_input.native_counterterm_blocks(
                    recursion,
                    process,
                    ignored_particle_ids=ignored_particle_ids,
                    u1_grades=u1_grades,
                )
            except dag_input.NativeDagUnsupported as error:
                # Only a listed feature falls back (rule 2).
                fallback = dag_input.listed_fallback(error)
                dag_input.native_progress(progress_callback, stage, fallback=fallback)
                _LOG.warning(
                    "%s: the tree engine builds the counterterm program "
                    "(the native builder does not cover %s)",
                    process.process,
                    fallback,
                )
            else:
                program = cls(
                    recursion,
                    TreeBlocks(
                        process=process,
                        blocks_dag=native.dag,
                        born_dag=native.dag,
                        blocks=native.blocks,
                        born_roots=(),
                        kernel_variant=native.kernel_variant,
                        graded=u1_grades,
                    ),
                    native.clones,
                )
                # The U(1) leakage scan is answered now (the handle keeps
                # it), so the native columns are not held to the end of
                # generation; the per-point rows and the export rebuild and
                # decode them.
                u1_leaking_structures(program)
                dag_input.native_progress(
                    progress_callback, stage, (native.dag, len(native.blocks))
                )
                native.dag.release()
                return program

        from pyamplicol.generation.dag_compiler import GenericDAGCompiler
        from pyamplicol.models.prepared_catalog_builder import (
            model_tree_kernel_variant,
        )

        # The lowered model's Weyl-projection support comes decided from
        # the lowering cache (``CountertermLoweredModel.tree_model``).
        model = recursion.model.lowered.tree_model()
        full = (1 << len(process.legs)) - 1
        dag = GenericDAGCompiler(
            model=model,
            max_coupling_orders=recursion.coupling_order_limits(),
            ignored_particle_ids=tuple(ignored_particle_ids),
            progress_callback=progress_callback,
            loop_blocks=True,
            loop_block_reachability=RootReachability(full),
            u1_grades=u1_grades,
        ).compile(replace(process, color_accuracy="full"))
        variant = model_tree_kernel_variant(model)
        lines = _clone_lines(recursion)
        blocks, clones = [], {}
        for block in _blocks(dag, model, variant):
            line = lines.get(block.key.particle_id)
            if line is not None:
                clones[block.id] = line
                block = replace(
                    block, key=replace(block.key, particle_id=line.field_id)
                )
            blocks.append(block)
        tree = TreeBlocks(
            process=process,
            blocks_dag=dag,
            born_dag=dag,
            blocks=tuple(blocks),
            born_roots=(),
            kernel_variant=variant,
            dag_fallback=fallback,
            graded=u1_grades,
        )
        return cls(recursion, tree, clones)


def _clone_lines(recursion: CountertermRecursion) -> dict[int, CloneLine]:
    """Clone particle id -> ``CloneLine`` (both clones of every line)."""

    lowered = recursion.model.lowered
    particles = {
        str(item["name"]): item for item in lowered.compiled.model["particles"]
    }
    out: dict[int, CloneLine] = {}
    for structure, clone in lowered.insertions.items():
        half = tuple((name, power // 2) for name, power in structure.orders)
        slot = (structure.component, structure.power, structure.grade)
        for clone_name in {clone, str(particles[clone]["antiname"])}:
            read = particles[lowered.line_fields[clone_name]]
            out[int(particles[clone_name]["pdg_code"])] = CloneLine(
                int(read["pdg_code"]),
                add_orders(half),
                slot,
                lowered.line_factors.get(clone_name),
                int(particles[clone_name]["spin"]) != int(read["spin"]),
            )
    return out


def _insertions(orders: Orders) -> int:
    return dict(orders).get(COUNTERTERM_INSERTION_ORDER, 0)


def _slot(
    orders: Orders, slots: Mapping[str, tuple[str, int, int]]
) -> tuple[str, int, int] | None:
    """The slot of a CT = 1 current, ``None`` if no rule of the request has
    it (a lowered rule of other requests: multiplicity 0)."""

    found = [slots[name] for name, _power in orders if name in slots]
    if len(found) > 1:
        raise ValueError(f"CT current orders {orders!r} hold several slots")
    return found[0] if found else None


@dataclass(frozen=True)
class CountertermClosure:
    """The root terms of a process's CT amplitudes (module docstring).

    ``terms`` hold every root term; ``sites`` the CT-root sites (CT vertices
    holding leg 1), whose rule entries the kernel catalogue must bind.
    """

    program: CountertermBlocks
    closure: SiteClosure
    sites: tuple[CountertermSite, ...]
    terms: tuple[ClosureTerm, ...]

    @staticmethod
    def root_sites(
        program: CountertermBlocks,
        native_model: Any,
        selection: LoopOrderSelection,
    ) -> tuple[CountertermSite, ...]:
        """Sites of the recursion's rules with leg 1 as one block (CT roots),
        with the recursion's own admission."""

        recursion = program.recursion
        table = _PlainTable(program.blocks.block_table(native_model))
        sites = generate_counterterm_sites(
            None,
            table,
            recursion.rules,
            native_model,
            selection,
            whole_rules=True,
            root_mask=1,
            admission=recursion.admission,
        )
        return sites.sites

    @classmethod
    def build(
        cls,
        program: CountertermBlocks,
        sites: Sequence[CountertermSite],
        ordinary: Sequence[Sequence[Any]],
        native_model: Any,
        catalogue: Any,
        *,
        colours: SiteClosure | None = None,
    ) -> CountertermClosure:
        """Root terms of ``program``.

        ``sites`` are its ``root_sites``, ``ordinary`` the model's ordinary
        tree vertex entries by vertex (the Born closure's) and ``catalogue`` a
        loop kernel catalogue binding all of their entries. ``colours`` (the
        slice's Born closure) shares its closed colours with the program's
        closure (``SiteClosure.share_colours``).
        """

        recursion = program.recursion
        closure = SiteClosure(program.blocks, native_model, catalogue)
        if colours is not None:
            closure.share_colours(colours)
        table = program.blocks.block_table(native_model)
        legs = table.legs
        full = (1 << len(legs)) - 1
        loop = set(recursion.loop_orders)
        slots = {
            name: slot for slot in recursion.slots for name in (_slot_order(slot),)
        }
        # The vertex rules of the lowered tree model (its own and the copies).
        lowered = recursion.model.lowered.compiled
        tree_vertices = frozenset(
            str(rule["name"]) for rule in lowered.model["vertex_rules"]
        )
        terms: list[ClosureTerm] = []
        # The clone blocks of the scalar sides of scalar-vector mixing lines.
        promoted = frozenset(
            block_id for block_id, line in program.clones.items() if line.promoted
        )
        for entries in ordinary:
            fields = entries[0].particle_names
            wanted, need = frozenset(fields), sorted(fields)
            order = add_orders(tuple(entries[0].order))
            # Leg 1 alone is one block, the rest is partitioned directly; only
            # currents of the vertex's fields are combined.
            for rest in _set_partitions(full & ~1, len(fields) - 1, len(fields) - 1):
                parts = (1, *rest)
                pools = [
                    [b for b in table.currents(part) if b.field in wanted]
                    for part in parts
                ]
                for choice in product(*pools):
                    if sorted(b.field for b in choice) != need:
                        continue
                    inserted = [b for b in choice if _insertions(b.orders)]
                    if len(inserted) != 1:
                        continue
                    (current,) = inserted
                    clone = _insertions(current.orders) == 1
                    physical = add_orders(order, *(_physical(b.orders) for b in choice))
                    if not clone and physical not in loop:
                        continue
                    slot = _slot(current.orders, slots)
                    if slot is None:
                        continue
                    component, power, grade = slot
                    lines = closure._blocks.get(current) if clone else None
                    if lines and all(
                        add_orders(physical, program.clones[block.id].half) not in loop
                        for block in lines
                    ):
                        # No consuming copy reaches a loop order: the vertex's
                        # terms are not built (a vertex the closure refuses,
                        # e.g. a W vertex not closed under conjugation at
                        # u u~ > e- e+ d d~ g, may sit at such orders).
                        continue
                    if lines and any(
                        program.clones[b.id].promoted
                        and add_orders(physical, program.clones[b.id].half) in loop
                        for b in lines
                    ):
                        # A Goldstone-vector mixing line ends at the root on
                        # its Goldstone side (WP3.2 EWI-7, routed item (v)):
                        # its current is the vector's, which only the
                        # promoted copies of a vertex consume
                        # (``models.ct_line_insertions``). A root vertex the
                        # request's tree model does not hold because the
                        # restriction makes every coupling of it vanish (a
                        # Yukawa of a massless fermion) has no copy and
                        # contributes nothing (WP3.2 EWI-5); any other closes
                        # through its promoted entries
                        # (``ct_sites.promoted_root_entry``, bound by the
                        # catalogue, ``promoted_root_entries``).
                        absent = entries[0].vertex_name not in tree_vertices
                        if absent and restricted_to_zero(lowered, entries):
                            continue
                    attachments = _assigned(fields, choice)
                    index = attachments.index(current)
                    for term in closure.vertex_terms(
                        entries, attachments, promoted if clone else frozenset()
                    ):
                        line = program.clones.get(term.blocks[index].id)
                        if clone:
                            # The consuming copy (the root) carries the other
                            # half of the orders of Pi and, on a Weyl line,
                            # kappa if this is the clone of the antiparticle.
                            total = add_orders(physical, line.half)
                            if total not in loop:
                                continue
                        else:
                            total = physical
                        terms.append(
                            ClosureTerm(
                                term,
                                ROOT_LINE if clone else ROOT_VERTEX,
                                component,
                                power,
                                total,
                                (term.entry.coupling.name, 0),
                                scale=None if line is None else line.factor,
                                grade=grade,
                            )
                        )
        for site in sites:
            if site.rule.whole_rule_continuation is not None:
                # The replacement terms of a whole-rule colour continuation.
                from .assemble import _whole_rule_terms

                terms.extend(
                    ClosureTerm(
                        term,
                        ROOT_COUNTERTERM,
                        COMPONENT_BY_KIND[site.kind],
                        0,
                        site.orders,
                        None,
                        multiplicity=len(site.alternatives),
                    )
                    for term in _whole_rule_terms(closure, site)
                )
                continue
            # Cells the lowered model leaves out cancel here too.
            cancelled = recursion.model.unlowered.get(rule_key(site.rule), {})
            for term in closure.terms(site, cancelled):
                for power in _powers(recursion, site):
                    terms.append(
                        ClosureTerm(
                            term,
                            ROOT_COUNTERTERM,
                            COMPONENT_BY_KIND[site.kind],
                            power,
                            site.orders,
                            (term.entry.coupling.name, power),
                            multiplicity=len(site.alternatives),
                        )
                    )
        return cls(program, closure, tuple(sites), tuple(terms))


def _u1_coupling_kinds(compiled: Any) -> frozenset[int]:
    """Kernels of CT colours that couple to the U(1) part of a gluon line.

    d-symbols and identities between adjoint legs do (structure constants
    and fundamental generators do not, or have their U(1)-subtraction
    auxiliaries, ``compiler_color_flow``).
    """

    out = set()
    for kernel in compiled.ir.oriented_kernels:
        rule = kernel.lc_recurrence_color_rule_kind or ""
        source = kernel.color_source or ""
        if not kernel.vertex.startswith("PYAMPLICOL_CT_V"):
            continue
        if "d(" in source or rule in {"adjoint-symmetric-invariant", "color-identity"}:
            out.add(int(kernel.kind))
    return frozenset(out)


def u1_projected(compiled: Any) -> bool:
    """Whether the lowered model projects the gluon legs of its counterterm
    vertices onto SU(N) (``models.compiler_color_flow.
    synthesize_u1_projections``, all or nothing per model)."""

    from pyamplicol.models.contracts import U1_PROJECTION_SUFFIX

    return any(
        kernel.vertex.endswith(U1_PROJECTION_SUFFIX)
        for kernel in compiled.ir.oriented_kernels
    )


class UnprojectedU1Error(NotImplementedError):
    """A CT vertex would take the U(1) part of a fermion line's gluon."""


def u1_leaking_structures(program: CountertermBlocks) -> frozenset[Any]:
    """Lowered CT terms whose value would take the U(1) part of a gluon.

    The tree engine subtracts the U(1) part of a gluon between fermion lines
    with auxiliaries that only fermion vertices produce and absorb. A CT
    vertex whose colour couples to that part (``_u1_coupling_kinds``) inside
    a block therefore must neither absorb a gluon current of a fermion line
    nor feed its gluon into a fermion line, unless the lowered model projects
    its gluon legs onto SU(N) (``u1_projected``); at the root the exact
    colour closure projects every block onto SU(N). Returns the lowered terms
    (``CountertermStructure``) of every unprojected CT vertex of the program
    that does either.
    """

    return _u1_leaking(program.recursion, program.blocks.blocks_dag)


def _u1_leaking(recursion: CountertermRecursion, dag: Any) -> frozenset[Any]:
    """``u1_leaking_structures`` of the program DAG ``dag`` of ``recursion``:
    the scan is native for a ``dag_input.NativeDag`` (``NativeDag.u1_leaking``,
    no decode) and over the currents of a ``GenericDAG`` (the oracle)."""

    from .dag_input import NativeDag

    lowered = recursion.model.lowered
    compiled = lowered.compiled
    if u1_projected(compiled):
        return frozenset()
    kinds = _u1_coupling_kinds(compiled)
    if not kinds:
        return frozenset()
    colours = {int(p.pdg_code): int(p.color) for p in compiled.ir.particles}
    fundamental = {
        leg.label
        for leg in dag.process.legs
        if leg.outgoing_pdg is not None and abs(colours[int(leg.outgoing_pdg)]) == 3
    }
    generator_kinds = {
        int(kernel.kind)
        for kernel in compiled.ir.oriented_kernels
        if kernel.color_projection_structure == "fundamental-generator"
    }
    # A root attaches at least two blocks to leg 1, so the currents of all
    # legs but leg 1 are never read (nor what only they derive from).
    rest = frozenset(leg.label for leg in dag.process.legs) - {1}
    leaking: set[int] = set()
    if isinstance(dag, NativeDag):
        leaking |= dag.u1_leaking(
            kinds,
            generator_kinds,
            (pdg for pdg, colour in colours.items() if colour == 8),
            sum(1 << (label - 1) for label in fundamental),
            sum(1 << (label - 1) for label in rest),
        )
    else:
        currents = dag.currents
        interactions = [
            interaction
            for interaction in dag.interactions
            if frozenset(currents[interaction.result_id].index.external_labels) != rest
        ]
        producers: dict[int, set[int]] = defaultdict(set)
        for interaction in interactions:
            kind = int(interaction.vertex_kind)
            if kind not in kinds:
                continue
            for input_id in (interaction.left_id, interaction.right_id):
                index = currents[input_id].index
                if colours.get(int(index.particle_id)) == 8 and (
                    fundamental & set(index.external_labels)
                ):
                    leaking.add(kind)
            if colours.get(int(currents[interaction.result_id].index.particle_id)) == 8:
                producers[interaction.result_id].add(kind)
        for interaction in interactions:
            if int(interaction.vertex_kind) in generator_kinds:
                for input_id in (interaction.left_id, interaction.right_id):
                    leaking |= producers.get(input_id, set())
    if not leaking:
        return frozenset()
    by_coupling: dict[str, set[Any]] = defaultdict(set)
    for structure, coupling in lowered.couplings.items():
        by_coupling[coupling].add(structure)
    coupling_of = {int(term.id): term.coupling for term in compiled.ir.vertex_terms}
    out: set[Any] = set()
    for kernel in compiled.ir.oriented_kernels:
        if int(kernel.kind) in leaking:
            for term_id in kernel.term_ids or (kernel.term_id,):
                out |= by_coupling.get(coupling_of.get(int(term_id), ""), set())
    return frozenset(out)


def _slot_order(slot: tuple[str, int, int]) -> str:
    return counterterm_slot_order(*slot)


def _powers(recursion: CountertermRecursion, site: CountertermSite) -> tuple[int, ...]:
    """Epsilon powers of a CT-root site: those of its rule's lowered terms."""

    return tuple(
        sorted(
            {structure.power for _entry, structure in recursion.terms[site.rule_name]}
        )
    )


def _assigned(fields: Sequence[str], choice: Sequence[Any]) -> list[Any]:
    free, out = sorted(choice), []
    for name in fields:
        block = next(b for b in free if b.field == name)
        free.remove(block)
        out.append(block)
    return out


@dataclass(frozen=True)
class _PlainTable:
    """A block table restricted to currents without a CT insertion."""

    table: Any
    _cache: dict = field(default_factory=dict, compare=False)

    @property
    def legs(self):
        return self.table.legs

    def currents(self, mask: int):
        found = self._cache.get(mask)
        if found is None:
            found = self._cache[mask] = tuple(
                b for b in self.table.currents(mask) if not _insertions(b.orders)
            )
        return found


def promoted_root_entries(
    program: CountertermBlocks | None,
    ordinary: Iterable[Sequence[Any]],
    read: Callable[[Sequence[Any]], Sequence[Any]],
    native_model: Any,
) -> list[Any]:
    """The promoted root entries (``ct_sites.promoted_root_entry``) the
    closure of ``program`` may read: every scalar leg, of the field a
    scalar-vector mixing line's scalar side is read as, of the entries the
    root reads (``read``: the vertex's own entries or its conjugate rule's,
    ``SiteClosure.read_entries``) for each ordinary vertex that holds leg
    1's current (``CountertermClosure.build``'s root vertices). The kernel
    catalogue binds them with the closure's other entries."""

    from .ct_sites import promoted_root_entry

    if program is None:
        return []
    fields = {abs(line.field_id) for line in program.clones.values() if line.promoted}
    if not fields:
        return []
    root = {
        current.field
        for current in program.blocks.block_table(native_model).currents(_ROOT_LEG)
    }
    by_name = {
        str(record["name"]): record for record in model_data(native_model)["particles"]
    }
    out = []
    for rows in ordinary:
        names = rows[0].particle_names
        if root.isdisjoint(names) or not any(
            abs(int(by_name[n]["pdg_code"])) in fields for n in names
        ):
            continue
        try:
            entries = read(rows)
        except NotImplementedError:
            continue  # the closure refuses such a vertex itself
        for entry in entries:
            for leg, name in enumerate(entry.particle_names):
                record = by_name[name]
                if abs(int(record["pdg_code"])) in fields and int(record["spin"]) == 1:
                    out.append(promoted_root_entry(entry, leg))
    return out


def counterterm_closure_entries(
    sites: Iterable[CountertermSite], ordinary: Iterable[Sequence[Any]]
) -> list[Any]:
    """The kernel entries a closure binds (catalogue ``counterterm_entries``)."""

    entries = [e for site in sites for e in site.rule.entries]
    entries += [e for rows in ordinary for e in rows]
    return list({id(e): e for e in entries}.values())


def model_value(compiled: Any, expression: str, parameters: Mapping[str, Any], digits):
    """``expression`` (payload syntax) at the model ``parameters``.

    Internal parameters resolve through their exact definitions, external
    ones take ``parameters`` or their defaults; ``digits`` decimal digits
    (Symbolica's arbitrary-precision evaluation), returned as a pair of
    decimal strings.
    """

    from decimal import Decimal

    from symbolica import E

    from pyamplicol._internal.physics.symbols import symbols
    from pyamplicol.runtime.symbolica_exact import _upcast_complex_inputs

    model_symbols = symbols.model(compiled.name)
    value = E(expression)
    internal = {
        record.name: record.resolved_expression
        for record in compiled.ir.parameters
        if record.nature == "internal"
    }
    # Payload symbols (``UFO::{}::name``) are the model's parameters.
    for name in {*internal, *compiled.parameter_defaults}:
        payload = E(f"UFO::{{}}::{name}")
        if value.contains(payload):
            value = value.replace(
                payload, E(model_symbols.symbol(name).to_canonical_string())
            )
    value = resolved_internal_parameters(compiled, value)
    external = {
        name: parameters.get(name, pair[0])
        for name, pair in compiled.parameter_defaults.items()
        if name not in internal
    }
    names = [
        name
        for name in external
        if value.contains(E(model_symbols.symbol(name).to_canonical_string()))
    ]
    substitutions = dict(
        zip(
            (E(model_symbols.symbol(name).to_canonical_string()) for name in names),
            _upcast_complex_inputs(
                [(Decimal(str(external[name])), Decimal(0)) for name in names],
                digits,
            ),
            strict=True,
        )
    )
    return value.evaluate(
        substitutions, decimal_digit_precision=digits
    ).to_decimal_tuple()


def program_propagated_currents(dag: Any) -> tuple[int, ...]:
    """The currents of a CT program whose propagated value the closure reads:
    every current but the sources (the runtime schema's
    ``propagated_currents``)."""

    return tuple(current.id for current in dag.currents if not current.is_source)


# ------------------------------------------------------------------ production

# The lowered models of the most recent (model, fixed parameters, veto,
# declaration, request scope) keys; a request of another key evicts the oldest.
_LOWERED: OrderedDict[tuple[object, ...], tuple[Any, Any]] = OrderedDict()
_LOWERED_KEPT = 4


def program_order_limits(
    compiled: Any, born_orders: Iterable[Orders], loop_orders: Iterable[Orders]
) -> dict[str, int]:
    """Order limits of a request's CT program: every physical order of the
    model bounded by its largest Born or loop power (0 when none carries it;
    the tree generator leaves an unlisted order unbounded)."""

    limits = {
        str(item["name"]): 0
        for item in compiled.model["orders"]
        if not str(item["name"]).startswith("PYAMPLICOL_")
    }
    for orders in (*born_orders, *loop_orders):
        for name, power in orders:
            limits[name] = max(limits.get(name, 0), int(power))
    return limits


def lowered_counterterm_model(
    compiled: Any,
    fixed_parameters: Mapping[str, Any],
    *,
    veto: Iterable[int] = (),
    corrections: Iterable[str] | None = None,
    declaration: Any = None,
    request: tuple[Any, Any, Any] | None = None,
    order_limits: Mapping[str, int] | None = None,
    allow_anomalous_veto: bool = False,
) -> Any:
    """The CT-lowered model of ``compiled`` for the closure, cached.

    The rules of a preparation with the request's ``fixed_parameters`` are
    lowered once per model, fixed parameters, vetoed |PDG| codes
    (``particle_veto``), corrections, colour ``declaration`` and request
    scope, with the multi-trace colours. A
    declaration (``LoopColorDeclaration``) grades the couplings by N_c and
    replaces fixed-SU(3) rules by their whole-rule continuations (design
    §6.7), as ``LoopSlice.generate`` prepares a declared request.

    ``request`` is ``(scope, prepared, continuation)``: the request's own
    loop-request model (``prepared``, after the declaration's
    ``continuation``) and a hashable ``scope`` naming what determines it
    besides the arguments above (its order bounds and compatibility profile).
    Its rules are then lowered, into a tree model kept within the program's
    ``order_limits`` (``program_order_limits``): at g g > g g g g, 9 of
    loop_sm's 96 vertices and only its QCD counterterms. Without ``request``
    the whole model is prepared without bounds and lowered.
    ``allow_anomalous_veto`` is the [U2] override (``ct_recursion``, "Anomalous
    vetoes").
    """

    from .ct_recursion import lower_model_counterterms

    key = (
        id(compiled),
        tuple(sorted((str(k), str(v)) for k, v in fixed_parameters.items())),
        tuple(sorted({abs(int(pdg)) for pdg in veto})),
        None if corrections is None else tuple(sorted(corrections)),
        None if declaration is None else repr(declaration),
        None if request is None else request[0],
        None if order_limits is None else tuple(sorted(order_limits.items())),
        bool(allow_anomalous_veto),
    )
    found = _LOWERED.get(key)
    if found is None or found[0] is not compiled:
        if request is not None:
            _scope, prepared, continuation = request
        else:
            from pyamplicol.loops.compiler import prepare_loop_request_model

            prepared = prepare_loop_request_model(
                compiled,
                fixed_parameters=dict(fixed_parameters),
            )
            continuation = None
            if declaration is not None:
                from pyamplicol.loops.source_color_declaration import (
                    prepare_declared_source_color,
                )

                prepared, continuation = prepare_declared_source_color(
                    prepared, declaration, scheme=declaration.scheme
                )
        lowered = lower_model_counterterms(
            compiled,
            prepared.counterterms.rules,
            fixed_parameters=prepared.structure.fixed_parameters,
            veto=key[2],
            corrections=key[3],
            multi_trace=True,
            continuation=continuation,
            order_limits=order_limits,
            species_orders=prepared.structure.perturbative_species_orders,
            allow_anomalous_veto=key[-1],
        )
        found = _LOWERED[key] = (compiled, lowered)
        while len(_LOWERED) > _LOWERED_KEPT:
            _LOWERED.popitem(last=False)
    _LOWERED.move_to_end(key)
    return found[1]


def counterterm_program_blocks(
    compiled: Any,
    prepared: Any,
    process: str,
    native_model: Any,
    selection: LoopOrderSelection,
    born_orders: Iterable[Orders],
    *,
    veto: Iterable[int] = (),
    fixed_parameters: Mapping[str, Any] | None = None,
    declaration: Any = None,
    continuation: Any = None,
    scope: Any = None,
    stage: Callable[[str, Callable[[], Any]], Any] | None = None,
    u1_grades: bool = False,
    allow_anomalous_veto: bool = False,
) -> CountertermBlocks:
    """The CT program (``CountertermBlocks``) of a request (design §6.1).

    ``prepared`` is the request's loop-request model, ``process`` its process
    text, ``native_model`` its native model and ``veto`` its vetoed |PDG|
    codes; ``fixed_parameters`` the request's fixed parameters (those of
    ``prepared``) and ``declaration`` its colour declaration, if any, with the
    ``continuation`` it gave ``prepared``. ``scope`` (hashable: the request's
    order bounds and compatibility profile) lets the lowered model of this
    request's rules, within its order limits, be reused by requests of the
    same scope; without it every rule of the model is lowered. ``stage``
    (``name, work -> work()``) runs the lowering, the recursion and the
    program's DAG, e.g. to time them (``tools/loop_v2_dag.py``).
    ``u1_grades`` (leading and next-to-leading colour) makes every block of
    the program exact in N_c (``generation.dag_grades``) and refuses a
    reachable counterterm vertex the ``continuation`` does not cover
    (``refusals.require_declared_colour_rules``).

    A rule the lowering cannot lower is inert when its vertex enters no tree
    of the program (``InsertionReachability``, REQUIREMENTS_ADDENDUM J); the
    request is refused (``UnloweredCountertermError``) when one can enter,
    with ``AnomalousVetoError`` when one of them holds a Levi-Civita cell
    the veto keeps from cancelling and ``allow_anomalous_veto`` (the [U2]
    override, ``ct_recursion``) is not set.
    """

    from pyamplicol.processes.model import build_model_process_ir

    def run(name: str, work: Callable[[], Any]) -> Any:
        return work() if stage is None else stage(name, work)

    veto = tuple(sorted({abs(int(pdg)) for pdg in veto}))
    born_orders = tuple(born_orders)
    lowered = run(
        "CT lowering",
        lambda: lowered_counterterm_model(
            compiled,
            dict(fixed_parameters or {}),
            veto=veto,
            corrections=selection.corrections,
            declaration=declaration,
            request=None if scope is None else (scope, prepared, continuation),
            order_limits=None
            if scope is None
            else program_order_limits(compiled, born_orders, selection.loop_orders),
            allow_anomalous_veto=allow_anomalous_veto,
        ),
    )
    rules = prepared.counterterms.rules

    def recursion_of() -> tuple[CountertermRecursion, Any]:
        recursion = CountertermRecursion.build(lowered, rules, selection, born_orders)
        ir = build_model_process_ir(
            process, lowered.lowered.compiled.ir, color_accuracy="full"
        )
        if recursion.residual:
            # A rule that cannot be lowered is inert when its vertex enters
            # no tree of the program (REQUIREMENTS_ADDENDUM J), and refused
            # otherwise; there is no site fallback.
            reachability = InsertionReachability.build(
                lowered.lowered.tree_model(),
                ir,
                recursion.coupling_order_limits(),
                ignored_particle_ids=vetoed_particle_ids(veto),
            )
            recursion = recursion.with_inert(rules, reachability.admits)
        return recursion, ir

    recursion, ir = run("CT recursion", recursion_of)
    if recursion.residual:
        particles = {
            int(item["pdg_code"]): str(item["name"])
            for item in model_data(native_model)["particles"]
        }

        def fields(name: str) -> str:
            return " ".join(particles[p] for p in rules[name].particle_pdgs)

        anomalous = any(
            rule_key(rules[name]) in lowered.anomalous_veto
            for name in recursion.residual
        )
        raise (AnomalousVetoError if anomalous else UnloweredCountertermError)(
            "counterterm rules the tree recursion cannot lower, whose "
            "vertices can enter a tree of the request: "
            + "; ".join(
                f"{name} [{fields(name)}] ({reason})"
                for name, reason in sorted(recursion.residual.items())
            )
        )
    if continuation is not None and u1_grades:
        # LC and NLC (the grade split) grade the CT couplings by the
        # declaration's N_c continuation: a vertex it does not cover is
        # refused when a tree of the request can hold it (``refusals``,
        # listed; MQ-NC). Full colour reads the source couplings at N_c = 3
        # and needs no coverage.
        from .refusals import require_declared_colour_rules

        def reachability() -> Callable[[Any], bool]:
            return InsertionReachability.build(
                lowered.lowered.tree_model(),
                ir,
                recursion.coupling_order_limits(),
                ignored_particle_ids=vetoed_particle_ids(veto),
            ).admits

        require_declared_colour_rules(
            continuation,
            {name: rules[name] for name in recursion.rules},
            reachability,
            process,
        )
    program = run(
        "CT program",
        lambda: CountertermBlocks.build(
            recursion,
            ir,
            ignored_particle_ids=vetoed_particle_ids(veto),
            u1_grades=u1_grades,
        ),
    )
    # The lowered model projects the gluon legs of its CT vertices onto
    # SU(N) unless a tree vertex of the model couples to the U(1) part of
    # a gluon itself; the request's blocks are then not exact either.
    leaking = u1_leaking_structures(program)
    if leaking:
        raise UnprojectedU1Error(
            "counterterm vertices "
            + ", ".join(sorted({str(s.particles) for s in leaking}))
            + " would take the U(1) part of a fermion line's gluon: the "
            "model's tree vertices couple to it, so its counterterm "
            "vertices are not projected onto SU(N) (design §6.1)"
        )
    return cast(CountertermBlocks, program)


@dataclass
class CountertermProgram:
    """The CT amplitudes of one loop request in production (design §6.1).

    ``blocks`` is the CT program (``CountertermBlocks``), ``root_sites`` its
    CT-root sites. ``bind`` attaches the root terms (``closure``); ``point``
    evaluates the program's tree rows (``rows``) and Weyl factors at one
    point.
    """

    blocks: CountertermBlocks
    root_sites: tuple[CountertermSite, ...]
    closure: CountertermClosure | None = None
    _rows: Any = field(default=None, repr=False, compare=False)
    # The parameter-only work of the last point (checked assumptions and the
    # Weyl factors per carrier precision), reused while the parameters stay.
    _parameter_work: tuple[Any, dict] = field(
        default_factory=lambda: (None, {}), repr=False, compare=False
    )

    @property
    def recursion(self) -> CountertermRecursion:
        return self.blocks.recursion

    @property
    def rows(self) -> Any:
        """The program's tree rows (``assemble.TreeRows``), compiled on first
        use: generation and a native evaluation of the program do not need
        this Python evaluator (at gg -> gggg it takes most of the bind's
        6 minutes and 11.8 GB). Its stage evaluators stay interpreted: one
        point costs the same as with Symbolica's JIT, whose compile took
        69 s of a 115 s point at gg -> ggg and over an hour at gg -> gggg."""

        if self._rows is None:
            from .assemble import TreeRows

            dag = self.blocks.blocks.blocks_dag
            self._rows = TreeRows(
                dag,
                self.recursion.model.lowered.compiled,
                jit=False,
                # The largest leg sets feed no interaction of the program
                # (``RootReachability``) but the closure reads them propagated.
                propagated_currents=program_propagated_currents(dag),
            )
        return self._rows

    @classmethod
    def build(
        cls,
        compiled: Any,
        prepared: Any,
        process: str,
        native_model: Any,
        selection: LoopOrderSelection,
        born_orders: Iterable[Orders],
        **options: Any,
    ) -> CountertermProgram:
        """The CT program of a request (``prepared`` its loop-request model):
        ``counterterm_program_blocks`` with its root sites."""

        program = counterterm_program_blocks(
            compiled, prepared, process, native_model, selection, born_orders, **options
        )
        return cls(
            program, CountertermClosure.root_sites(program, native_model, selection)
        )

    def bind(
        self,
        ordinary: Sequence[Sequence[Any]],
        native_model: Any,
        catalogue: Any,
        *,
        colours: SiteClosure | None = None,
    ) -> CountertermProgram:
        """Attach the root terms (``catalogue`` binds the
        ``counterterm_closure_entries``; ``colours``, the slice's Born
        closure, shares its closed colours)."""

        self.closure = CountertermClosure.build(
            self.blocks,
            self.root_sites,
            ordinary,
            native_model,
            catalogue,
            colours=colours,
        )
        return self

    def point(
        self,
        point: Any,
        parameters: Mapping[str, str],
        numbers: Any,
        poles: bool,
        *,
        sources: Mapping[int, Any] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        """Rows of the program and Weyl factors at one point.

        ``parameters`` are the model parameters of the point (decimal
        strings); the CT multiplicities and slot switches (pole slots off
        without ``poles``) are added. ``sources`` replaces the wavefunctions
        of vector legs as in ``TreeRows.evaluate``: the spin and Ward sources
        of the loop side (CT amplitudes are loop side).
        """

        values = dict(parameters)
        recursion = self.recursion
        values.update(
            {
                name: str(v)
                for name, v in recursion.parameter_values(
                    poles=poles, card=parameters, numbers=numbers
                ).items()
            }
        )
        key = tuple(sorted(values.items()))
        if self._parameter_work[0] != key:
            recursion.model.check_parameters(
                {name: complex(float(v)) for name, v in parameters.items()}
            )
            self._parameter_work = (key, {})
        rows = self.rows.evaluate(point, values, numbers, sources=sources)
        digits = numbers.precision.working_digits
        factors = self._parameter_work[1].setdefault(digits, {})
        scales = {}
        for term in self.closure.terms:
            if term.scale is not None and term.scale not in scales:
                if term.scale not in factors:
                    factors[term.scale] = model_value(
                        recursion.model.lowered.compiled, term.scale, values, digits
                    )
                scales[term.scale] = numbers.pair(*factors[term.scale])
        return rows, scales


__all__ = [
    "ROOT_COUNTERTERM",
    "ROOT_LINE",
    "ROOT_VERTEX",
    "CloneLine",
    "ClosureTerm",
    "CountertermBlocks",
    "CountertermClosure",
    "CountertermProgram",
    "InsertionReachability",
    "RootReachability",
    "UnprojectedU1Error",
    "counterterm_closure_entries",
    "counterterm_program_blocks",
    "lowered_counterterm_model",
    "model_value",
    "program_propagated_currents",
    "promoted_root_entries",
    "u1_leaking_structures",
    "u1_projected",
]
