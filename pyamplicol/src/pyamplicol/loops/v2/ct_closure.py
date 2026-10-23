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

from collections import OrderedDict, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from itertools import product
from typing import Any, Literal

from pyamplicol.models.loading import (
    COUNTERTERM_INSERTION_ORDER,
    counterterm_slot_order,
)

from .blocks import TreeBlocks, _blocks
from .ct_recursion import CountertermRecursion, _physical
from .ct_sites import (
    COMPONENT_BY_KIND,
    CountertermSite,
    SiteClosure,
    generate_counterterm_sites,
    resolved_internal_parameters,
)
from .families import (
    FamilySet,
    LoopOrderSelection,
    Orders,
    _set_partitions,
    add_orders,
)

RootKind = Literal["vertex", "line", "counterterm"]
ROOT_VERTEX: RootKind = "vertex"
ROOT_LINE: RootKind = "line"
ROOT_COUNTERTERM: RootKind = "counterterm"


@dataclass(frozen=True, slots=True)
class RootReachability:
    """Loop-request reachability of the currents a leg-1 closure attaches:
    every leg set of the process that does not hold leg 1 (bit 0)."""

    full_mask: int

    def admits_mask(self, mask: int) -> bool:
        return mask > 0 and not mask & 1 and mask & self.full_mask == mask

    def admits_block(self, index: Any) -> bool:
        return self.admits_mask(index.external_mask)


@dataclass(frozen=True, slots=True)
class CloneLine:
    """A clone particle of the CT program read as the field ``field_id``.

    ``half`` are the physical orders of the vertex copy at either end of the
    line (half of those of ``Pi``), ``slot`` its (component, power, grade)
    and ``factor`` the Weyl factor ``kappa`` that the copies of this clone's
    leg carry, or ``None``.
    """

    field_id: int
    half: Orders
    slot: tuple[str, int, int]
    factor: str | None = None


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
    ) -> CountertermBlocks:
        """Compile the CT program's blocks for ``process`` (a model process IR
        of the lowered model at full colour)."""

        from pyamplicol.generation.dag_compiler import GenericDAGCompiler
        from pyamplicol.models.external import CompiledUFOModel
        from pyamplicol.models.prepared_catalog_builder import (
            model_tree_kernel_variant,
        )

        lowered = recursion.model.lowered
        model = CompiledUFOModel(lowered.compiled)
        full = (1 << len(process.legs)) - 1
        dag = GenericDAGCompiler(
            model=model,
            max_coupling_orders=recursion.coupling_order_limits(),
            ignored_particle_ids=tuple(ignored_particle_ids),
            progress_callback=progress_callback,
            loop_blocks=True,
            loop_block_reachability=RootReachability(full),
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
            out[int(particles[clone_name]["pdg_code"])] = CloneLine(
                int(particles[lowered.line_fields[clone_name]]["pdg_code"]),
                add_orders(half),
                slot,
                lowered.line_factors.get(clone_name),
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
        families: FamilySet,
        rules: Mapping[str, Any],
        native_model: Any,
        selection: LoopOrderSelection,
        *,
        forbidden: Iterable[int] = (),
    ) -> tuple[CountertermSite, ...]:
        """Sites of the recursion's rules with leg 1 as one block (CT roots).

        ``rules`` are all rules of the request, for the completeness check.
        """

        recursion = program.recursion
        table = _PlainTable(program.blocks.block_table(native_model))
        sites = generate_counterterm_sites(
            families,
            table,
            rules,
            native_model,
            selection,
            forbidden=forbidden,
            whole_rules=True,
            closed_rules=recursion.rules,
            root_mask=1,
            # The request's own site generation checks completeness.
            check_coverage=False,
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
    ) -> CountertermClosure:
        """Root terms of ``program``.

        ``sites`` are its ``root_sites``, ``ordinary`` the model's ordinary
        tree vertex entries by vertex (the Born closure's) and ``catalogue`` a
        loop kernel catalogue binding all of their entries.
        """

        recursion = program.recursion
        closure = SiteClosure(program.blocks, native_model, catalogue)
        table = program.blocks.block_table(native_model)
        legs = table.legs
        full = (1 << len(legs)) - 1
        loop = set(recursion.loop_orders)
        slots = {
            name: slot for slot in recursion.slots for name in (_slot_order(slot),)
        }
        terms: list[ClosureTerm] = []
        for entries in ordinary:
            fields = entries[0].particle_names
            order = add_orders(tuple(entries[0].order))
            # Leg 1 alone is one block, the rest is partitioned directly.
            for rest in _set_partitions(full & ~1, len(fields) - 1, len(fields) - 1):
                parts = (1, *rest)
                for choice in product(*(table.currents(part) for part in parts)):
                    if sorted(b.field for b in choice) != sorted(fields):
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
                    attachments = _assigned(fields, choice)
                    index = attachments.index(current)
                    for term in closure.vertex_terms(entries, attachments):
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
            cancelled = recursion.cancelled.get(site.rule_name, ())
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

    from pyamplicol.models.compiler_color_flow import U1_PROJECTION_SUFFIX

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

    dag = program.blocks.blocks_dag
    lowered = program.recursion.model.lowered
    compiled = lowered.compiled
    if u1_projected(compiled):
        return frozenset()
    kinds = _u1_coupling_kinds(compiled)
    if not kinds:
        return frozenset()
    currents = dag.currents
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
    interactions = [
        interaction
        for interaction in dag.interactions
        if frozenset(currents[interaction.result_id].index.external_labels) != rest
    ]
    leaking: set[int] = set()
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


# ------------------------------------------------------------------ production

# The lowered models of the most recent (model, fixed parameters, veto,
# declaration) keys; a request of another key evicts the oldest.
_LOWERED: OrderedDict[tuple[object, ...], tuple[Any, Any]] = OrderedDict()
_LOWERED_KEPT = 4


def lowered_counterterm_model(
    compiled: Any,
    fixed_parameters: Mapping[str, Any],
    *,
    forbidden: Iterable[int] = (),
    declaration: Any = None,
) -> Any:
    """The CT-lowered model of ``compiled`` for the closure, cached.

    Every rule of the model (an unbounded preparation with the request's
    ``fixed_parameters``) is lowered once per model, fixed parameters, vetoed
    |PDG| codes and colour ``declaration``, with the multi-trace colours. A
    declaration (``LoopColorDeclaration``) grades the couplings by N_c and
    replaces fixed-SU(3) rules by their whole-rule continuations (design
    §6.7), as ``LoopSlice.generate`` prepares a declared request.
    """

    from pyamplicol.loops.compiler import prepare_loop_request_model

    from .ct_recursion import lower_model_counterterms

    key = (
        id(compiled),
        tuple(sorted((str(k), str(v)) for k, v in fixed_parameters.items())),
        tuple(sorted({abs(int(pdg)) for pdg in forbidden})),
        None if declaration is None else repr(declaration),
    )
    found = _LOWERED.get(key)
    if found is None or found[0] is not compiled:
        prepared = prepare_loop_request_model(
            compiled, fixed_parameters=dict(fixed_parameters)
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
            forbidden=key[2],
            multi_trace=True,
            continuation=continuation,
        )
        found = _LOWERED[key] = (compiled, lowered)
        while len(_LOWERED) > _LOWERED_KEPT:
            _LOWERED.popitem(last=False)
    _LOWERED.move_to_end(key)
    return found[1]


@dataclass
class CountertermProgram:
    """The CT amplitudes of one loop request in production (design §6.1).

    ``blocks`` is the CT program (``CountertermBlocks``), ``root_sites`` its
    CT-root sites; ``residual`` names the request's rules left to site
    closure. ``bind`` attaches the root terms (``closure``); ``point``
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

            self._rows = TreeRows(
                self.blocks.blocks.blocks_dag,
                self.recursion.model.lowered.compiled,
                jit=False,
            )
        return self._rows

    @property
    def residual(self) -> tuple[str, ...]:
        return tuple(sorted(self.recursion.residual))

    @classmethod
    def build(
        cls,
        compiled: Any,
        prepared: Any,
        process: str,
        families: FamilySet,
        native_model: Any,
        selection: LoopOrderSelection,
        born_orders: Iterable[Orders],
        *,
        forbidden: Iterable[int] = (),
        fixed_parameters: Mapping[str, Any] | None = None,
        declaration: Any = None,
    ) -> CountertermProgram:
        """The CT program of a request (``prepared`` its loop-request model).

        ``process`` is the request's process text, ``families`` and
        ``native_model`` its family set and native model; ``fixed_parameters``
        the request's fixed parameters (those of ``prepared``) and
        ``declaration`` its colour declaration, if any.
        """

        from pyamplicol.processes.model import build_model_process_ir

        forbidden = tuple(forbidden)
        lowered = lowered_counterterm_model(
            compiled,
            dict(fixed_parameters or {}),
            forbidden=forbidden,
            declaration=declaration,
        )
        rules = prepared.counterterms.rules
        recursion = CountertermRecursion.build(
            lowered,
            rules,
            families,
            native_model,
            selection,
            born_orders,
            forbidden=forbidden,
        )
        ir = build_model_process_ir(
            process, lowered.lowered.compiled.ir, color_accuracy="full"
        )
        program = CountertermBlocks.build(recursion, ir, ignored_particle_ids=forbidden)
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
        sites = CountertermClosure.root_sites(
            program, families, rules, native_model, selection, forbidden=forbidden
        )
        return cls(program, sites)

    def bind(
        self, ordinary: Sequence[Sequence[Any]], native_model: Any, catalogue: Any
    ) -> CountertermProgram:
        """Attach the root terms (``catalogue`` binds the
        ``counterterm_closure_entries``)."""

        self.closure = CountertermClosure.build(
            self.blocks, self.root_sites, ordinary, native_model, catalogue
        )
        return self

    def point(
        self, point: Any, parameters: Mapping[str, str], numbers: Any, poles: bool
    ) -> tuple[Any, dict[str, Any]]:
        """Rows of the program and Weyl factors at one point.

        ``parameters`` are the model parameters of the point (decimal
        strings); the CT multiplicities and slot switches (pole slots off
        without ``poles``) are added.
        """

        values = dict(parameters)
        recursion = self.recursion
        values.update(
            {
                name: str(v)
                for name, v in recursion.parameter_values(poles=poles).items()
            }
        )
        key = tuple(sorted(values.items()))
        if self._parameter_work[0] != key:
            recursion.model.check_parameters(
                {name: complex(float(v)) for name, v in parameters.items()}
            )
            self._parameter_work = (key, {})
        rows = self.rows.evaluate(point, values, numbers)
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
    "RootReachability",
    "UnprojectedU1Error",
    "counterterm_closure_entries",
    "lowered_counterterm_model",
    "model_value",
    "u1_leaking_structures",
    "u1_projected",
]
