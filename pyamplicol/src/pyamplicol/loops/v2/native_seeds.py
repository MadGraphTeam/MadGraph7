# SPDX-License-Identifier: 0BSD
"""Native seed combinations and seed classes (WP3.1 I10, package native-seeds).

``build_seed_table`` and ``extend_seed_table`` here are the native twins of
the functions of ``loops.v2.colour``, which stay their oracles
(``tests/unit/test_loop_v2_native_seeds.py``): equal tables (classes,
orders, cells, certificates) and equal ``class_of``. The per-skeleton work
runs in ``rusticol-core`` ``loop_v2::generate::seeds`` (binding
``SeedBuilder``):

* **B2** the combinations (``CompressedColour.combination``,
  ``_ring_combination`` with its per-label-renaming memo, ``_combination``,
  ``canonical_primitive``), the certificate bookkeeping and
  ``Skeleton.orders``;
* **B3** the class keys (``_ClassKeys``, ``_ring_relabelling``,
  ``_proportional_key``), ratios and the ``row`` / ``class_of``
  accumulation of ``_extended``, the class orders and the LC/NLC prune
  liveness of ``LoopSlice._prune_colour`` (``NativeSeedStage.keep``).

Python keeps what is per distinct object and needs Spenso/Idenso: each
distinct content's compression (``CompressedColour._content``, vertex
decompositions), the certificate check on the smallest instance, the colour
networks of the exact descriptors and contents that the in-crate
reductions read, and the cells (``seed_cells``, package native-cells). The
exact reduction of each distinct canonical quark ring
(``_reduced_primitive``) and generic seed (``SeedColour.seed``) runs
in-crate (``NATIVE_REDUCTIONS``; off, in Python between the native steps).

The stage input is Θ(contents): per distinct content its position kind,
single-leg flag, terms (coefficient, descriptor) and coupling orders, and
per skeleton its content ids as a little-endian binary table. An exact
descriptor's network is handed over as its ``repr`` with every external
adjoint label replaced by a slot (``_template``), so the native code
reproduces ``_ring_relabelling`` and the ``repr`` orders of
``canonical_primitive`` without the network.

**The switch** (phase 1): ``colour.build_seed_table`` and
``colour.extend_seed_table`` dispatch here while ``NATIVE`` is on (the
default); ``python_oracle()`` runs the Python stages. **B1:** the readings
of a retained native schedule build (``generator_input.NativeBuild``) feed
the stage in-crate (``SeedBuilder.from_generation``) instead of bytes
tables, once ``NativeBuild.readings_of`` has checked that the skeletons are
exactly those readings; ``NativeSeedStage.reading_keep`` (``prune_keep``
for a slice) gives the prune liveness as flags per reading, from which
``NativeGeneration.schedules`` rebuilds the pruned plan natively. The exact
reductions (quark rings, generic seeds) run in-crate too
(``NATIVE_REDUCTIONS``, ``networks.rs``).

**Design note: a lazy ``seed_fourier``** (RUST-MIGRATION.md §10 item 7, for
phase 1). ``LoopSlice._plan_contraction`` certifies the seed FFT eagerly
under ``colour_contraction="auto"`` (6-9 s at seven legs). Its only reader
is the per-point Python path (``_Evaluation`` builds ``SectorSeeds`` with
``fourier=slice_.fourier`` in binary64), whose ``algorithm`` is the
reported ``colour_contraction``; the native executor and the observables
refuse ``"fft"`` and never read it. Every assignment of ``LoopSlice.table``
(``_prune_colour``) is followed by ``_plan_contraction`` (``generate`` and
``ObservableSlice``). So computing ``seed_fourier(self.table)`` on the first
read, with ``_plan_contraction`` resetting the cache, returns the object
the eager form would have held at that read. Under ``"fft"`` the
certification stays eager, because its refusal ("no certified
symmetric-group orbit pair") must be raised at generation. ``LazyFourier``
below is that cache; the test proves that per-point seeds, the reported
``colour_contraction`` and the certified orbits are identical (phase 1 moves
it into ``assemble.LoopSlice``).
"""

from __future__ import annotations

import json
import sys
import time
from array import array
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import replace
from fractions import Fraction
from typing import Any, cast

from pyamplicol.color.connections import (
    ColorTensor,
    ConnectedColorTensor,
    ExactColorCoefficient,
    OpenColorString,
    TensorTerm,
)

from . import generator_input
from .colour import (
    _NATIVE_REDUCED,
    COMPRESSION_THEOREMS,
    BornBasis,
    CompressedColour,
    CompressionCertificate,
    PositionColour,
    Primitive,
    PrimitiveKind,
    SeedColour,
    SeedCombination,
    SeedFourier,
    SeedTable,
    _combination_tensor,
    _lie_vertices,
    _reduced_primitive,
    block_colour_network,
    born_basis,
    canonical_seed,
    relabel,
    seed_cells,
    seed_fourier,
)
from .families import Orders
from .skeletons import Content, Skeleton

__all__ = [
    "LazyFourier",
    "NativeSeedStage",
    "build_seed_table",
    "extend_seed_table",
]

_POSITIONS = {"adjoint": 0, "fundamental": 1, "exact": 2}

#: The switch (RUST-MIGRATION rule 1): ``colour.build_seed_table`` and
#: ``colour.extend_seed_table`` dispatch to this module's native stages, so
#: the main slice, the poles companion, the LC/NLC prune and the
#: observables' prune all take them. Off, the Python stages run: they are
#: the oracle (``python_oracle``). The native stage covers every input the
#: Python stage covers (Python keeps, per distinct object, each content's
#: compression, the certificate check and the cells; the exact reductions
#: run in-crate, ``NATIVE_REDUCTIONS``), so no feature has a per-request
#: Python fallback; any native failure fails closed.
NATIVE = True


#: Quark-ring reductions and generic seeds in-crate (``networks.rs``, the
#: twin of ``colour.reduce_network``); off, the stage calls the Python
#: ``_reduced_primitive`` and ``SeedColour.seed`` for them.
NATIVE_REDUCTIONS = True


@contextmanager
def python_oracle() -> Iterator[None]:
    """Run the Python seed stages (the oracle) inside the block, with the
    exact reductions of ``_reduced_primitive`` recomputed in Python (its
    cache may hold natively reduced rings)."""

    global NATIVE
    saved, NATIVE = NATIVE, False
    _reduced_primitive.cache_clear()
    _NATIVE_REDUCED.clear()
    try:
        yield
    finally:
        NATIVE = saved


def prune_keep(
    slice_: Any, observables: Iterable[tuple[SeedTable, Mapping[Orders, Any]]] = ()
) -> generator_input.ReadingKeep | None:
    """``LoopSlice._prune_colour``'s ``keep`` as native prune flags (B1):
    the native liveness (``NativeSeedStage.reading_keep``) of the seed stage
    that read the slice's complete schedule build in-crate, when the slice's
    table is still that stage's; None otherwise (the Python callback then
    applies). ``NativeGeneration.schedules(..., keep=prune_keep(...))``
    rebuilds the pruned schedules without a Python callback; the switch of
    ``_prune_colour`` to it is its owner's one line."""

    generation = getattr(slice_, "generation", None)
    base = getattr(generation, "base", None)
    stage: NativeSeedStage | None = getattr(base, "seed_stage", None)
    if stage is None or stage.table is not slice_.table:
        return None
    return stage.reading_keep(slice_, observables)


class _Networks:
    """Colour networks as the native reducer reads them (``NetworkSpec``):
    external legs keep their labels, "in"/"out" are the loop lines (-1,
    -2), every other label a local one (-3 - k) of its network."""

    def encode(self, items: Iterable[tuple[int, Any]]) -> str:
        return json.dumps(
            [[key, self.network(network)] for key, network in items],
            separators=(",", ":"),
        )

    def factored(self, colour: SeedColour, items: Iterable[tuple[int, Content]]) -> str:
        """The exact networks of contents (``CompressedColour._exact_network``)
        as products of distinct factors, which the native side expands
        (``multiply``): each vertex structure relabelled to its slots
        ("in", "out", ("block", j)) and each attached block's word
        (``block_colour_network`` on ("block", j)) is encoded once; a content
        lists its factor ids in ``_exact_network``'s order. One local
        numbering over every factor keeps a label shared by name (the
        reducer is invariant under any injective renaming of local labels)."""

        local: dict[Any, int] = {}
        factors: list[list[Any]] = []
        vertex_ids: dict[tuple[Any, ...], int] = {}
        block_ids: dict[tuple[int, int], int] = {}
        blocks_alive: list[Any] = []
        out: list[list[Any]] = []
        for key, content in items:
            state = content.colour
            if not isinstance(state, PositionColour):
                raise TypeError("seed classes need PositionColour contents")
            in_leg, out_leg, *attached = state.legs
            vertex = (state.rule, state.colour_index, tuple(state.legs))
            f = vertex_ids.get(vertex)
            if f is None:
                slots: dict[int, Any] = {in_leg: "in", out_leg: "out"}
                slots.update((leg, ("block", j)) for j, leg in enumerate(attached))

                def rename(label: Any, slots: dict[int, Any] = slots) -> Any:
                    kind, value = label
                    return slots[value] if kind == "leg" else ("dummy", value)

                f = vertex_ids[vertex] = len(factors)
                factors.append(
                    self.network(relabel(colour.vertex_network(state), rename), local)
                )
            ids = [f]
            for j, block in enumerate(colour.attached_blocks(content)):
                b = block_ids.get((id(block), j))
                if b is None:
                    blocks_alive.append(block)
                    b = block_ids[id(block), j] = len(factors)
                    factors.append(
                        self.network(
                            block_colour_network(
                                block.colour_flow,
                                colour._representations,
                                colour.field_colour(block),
                                ("block", j),
                            ),
                            local,
                        )
                    )
                ids.append(b)
            out.append([key, ids])
        return json.dumps({"factors": factors, "contents": out}, separators=(",", ":"))

    @staticmethod
    def network(network: Any, local: dict[Any, int] | None = None) -> list[Any]:
        if local is None:
            local = {}

        def label(value: Any) -> int:
            if isinstance(value, int):
                if value < 0:
                    raise ValueError(f"a negative external colour label {value}")
                return value
            if value == "in":
                return -1
            if value == "out":
                return -2
            return -3 - local.setdefault(value, len(local))

        return [
            [
                _text(term.coefficient.real),
                _text(term.coefficient.imag),
                term.nc_power,
                term.root2_power,
                [
                    [label(s.row), [label(a) for a in s.word], label(s.col)]
                    for s in term.strings
                ],
                [[label(a) for a in trace] for trace in term.traces],
            ]
            for term in network
        ]


def _decoded(raw: Sequence[Any], legs: Sequence[Any]) -> ConnectedColorTensor:
    """A natively reduced tensor (``(normalisation, terms)``)."""

    normalization, terms = raw
    return ConnectedColorTensor(
        tuple(legs),
        tuple(
            TensorTerm(
                ExactColorCoefficient(Fraction(re), Fraction(im)),
                ColorTensor(
                    tuple(OpenColorString(f, tuple(a), af) for f, a, af in strings),
                    tuple(tuple(trace) for trace in traces),
                ),
                power,
            )
            for re, im, power, strings, traces in terms
        ),
        normalization,
    )


def _prime_reduction(ring: Primitive, legs: Sequence[Any], raw: Sequence[Any]) -> None:
    """``_reduced_primitive(ring, legs)`` cached with the native reduction."""

    key = (ring, tuple(legs))
    _NATIVE_REDUCED[key] = _decoded(raw, legs)
    try:
        _reduced_primitive(*key)
    finally:
        _NATIVE_REDUCED.pop(key, None)


def _native(native: Any) -> Any:
    if native is None:
        from pyamplicol.loops.native import load_loop_native

        native = load_loop_native()
    return native


def build_seed_table(
    colour: SeedColour,
    skeletons: Iterable[Skeleton],
    basis: BornBasis | None = None,
    *,
    compress: bool = False,
    native: Any = None,
) -> tuple[SeedTable, dict[Skeleton, SeedCombination]]:
    """``colour.build_seed_table`` with its per-skeleton work native."""

    return NativeSeedStage(colour, native).build(skeletons, basis, compress=compress)


def extend_seed_table(
    table: SeedTable,
    class_of: Mapping[Skeleton, SeedCombination],
    colour: SeedColour,
    skeletons: Iterable[Skeleton],
    *,
    compress: bool = False,
    native: Any = None,
) -> tuple[SeedTable, dict[Skeleton, SeedCombination]]:
    """``colour.extend_seed_table`` with its per-skeleton work native."""

    return NativeSeedStage(colour, native).extend(
        table, class_of, skeletons, compress=compress
    )


class _Slot:
    """An external adjoint label in a network ``repr`` template."""

    __slots__ = ("label",)

    def __init__(self, label: int):
        self.label = label

    def __repr__(self) -> str:
        return f"\x00{self.label}\x00"


def _template(network: Any, adjoint: frozenset[int]) -> tuple[list[str], list[int]]:
    """``repr(network)`` split at its external adjoint labels: the parts
    between them and the labels, in the order ``_ring_relabelling`` visits
    them (strings' row, word, column, then traces, term by term)."""

    def slot(label: Any) -> Any:
        return _Slot(label) if isinstance(label, int) and label in adjoint else label

    pieces = repr(relabel(network, slot)).split("\x00")
    return pieces[0::2], [int(label) for label in pieces[1::2]]


def _text(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


def _tensor(tensor: ConnectedColorTensor) -> list[Any]:
    return [
        tensor.normalization_power,
        [
            [
                _text(term.coefficient.real),
                _text(term.coefficient.imag),
                term.nc_power,
                [
                    [
                        s.fundamental_label,
                        list(s.adjoint_labels),
                        s.antifundamental_label,
                    ]
                    for s in term.tensor.open_strings
                ],
                [list(trace) for trace in term.tensor.traces],
            ]
            for term in tensor.terms
        ],
    ]


def _skeleton_tables(
    skeletons: Sequence[Skeleton],
) -> tuple[list[Content], bytes, bytes]:
    """The distinct contents (first seen, by identity: the generator shares
    content objects) and the skeletons as content-id spans."""

    index: dict[int, int] = {}
    contents: list[Content] = []
    ids = array("I")
    offsets = array("Q", [0])
    for skeleton in skeletons:
        for content in skeleton.contents:
            key = id(content)
            found = index.get(key)
            if found is None:
                found = index[key] = len(contents)
                contents.append(content)
            ids.append(found)
        offsets.append(len(ids))
    if ids.itemsize != 4 or offsets.itemsize != 8:
        raise RuntimeError("native seeds need 32-bit and 64-bit array words")
    if sys.byteorder != "little":
        ids.byteswap()
        offsets.byteswap()
    return contents, offsets.tobytes(), ids.tobytes()


class NativeSeedStage:
    """One native seed-table stage (``build`` or ``extend``).

    After a stage, ``keep`` evaluates the LC/NLC prune's liveness of
    ``LoopSlice._prune_colour`` natively on the stage's skeletons,
    ``timings`` holds the wall time of its steps (seconds), ``counts`` its
    sizes (skeletons, distinct contents, reductions, classes, rows) and
    ``table`` the table it returned.
    """

    def __init__(self, colour: SeedColour, native: Any = None):
        self.colour = colour
        self.native = _native(native)
        self.adjoint = frozenset(
            leg.label for leg in colour.legs if leg.representation == 8
        )
        self._builder: Any = None
        # The retained schedule build whose readings fed the stage in-crate
        # (``generator_input.build_of``), or None (bytes tables).
        self.build_source: Any = None
        self._skeletons: list[Skeleton] = []
        self._orders: list[Orders] = []
        self._names: list[str] = []
        self._class_orders: list[list[int]] = []
        self.timings: dict[str, float] = {}
        self.counts: dict[str, int] = {}
        # The stage's native inputs (JSON), to replay it elsewhere (kept only
        # with ``generator_input.KEEP_REPLAY_INPUTS``): the content tables,
        # and the inputs of the in-crate reductions (legs, exact descriptor
        # and content networks) or the reduced tensors.
        self.spec_text = ""
        self.reduce_inputs: tuple[list[int], list[int], str, str] | None = None
        self.reductions_text = ("[]", "[]")
        self.kind = ""
        # The stage's needs (ring needs, generic skeletons) and its encoder.
        self.needs: tuple[list[Any], list[int]] = ([], [])
        self.encoder: _Encoder | None = None
        self.table: SeedTable | None = None
        self._clock = 0.0

    def _step(self, name: str) -> None:
        now = time.perf_counter()
        if name:
            self.timings[name] = round(now - self._clock, 4)
        self._clock = now

    # -- public stages -----------------------------------------------------

    def build(
        self,
        skeletons: Iterable[Skeleton],
        basis: BornBasis | None = None,
        *,
        compress: bool = False,
    ) -> tuple[SeedTable, dict[Skeleton, SeedCombination]]:
        if basis is None:
            basis = born_basis(self.colour.blocks)
        table = SeedTable(basis, (), (), (), self.colour.formal_nc)
        return self._run(table, {}, list(skeletons), compress, build=True)

    def extend(
        self,
        table: SeedTable,
        class_of: Mapping[Skeleton, SeedCombination],
        skeletons: Iterable[Skeleton],
        *,
        compress: bool = False,
    ) -> tuple[SeedTable, dict[Skeleton, SeedCombination]]:
        new = [s for s in skeletons if s not in class_of]
        return self._run(table, class_of, new, compress, build=False)

    # -- the stage ---------------------------------------------------------

    def _run(
        self,
        table: SeedTable,
        class_of: Mapping[Skeleton, SeedCombination],
        skeletons: list[Skeleton],
        compress: bool,
        *,
        build: bool,
    ) -> tuple[SeedTable, dict[Skeleton, SeedCombination]]:
        colour = self.colour
        self.kind = "build" if build else "extend"
        self.timings = {}
        self._step("")
        compressed = CompressedColour(colour) if compress else None
        # B1: the readings of a retained schedule build feed the stage
        # in-crate; any other skeleton list goes as bytes tables.
        found = generator_input.build_of(skeletons)
        if found is not None:
            source, used = found
            contents = [source.contents[c] for c in used]
            offsets = ids = b""
        else:
            source = None
            contents, offsets, ids = _skeleton_tables(skeletons)
        self.build_source = source
        if source is not None and source.complete and build:
            # The prune's native liveness reads this stage (``prune_keep``).
            source.seed_stage = self
        self._step("skeleton_tables")
        encoder = _Encoder(self.adjoint, compressed)
        names = sorted(
            {name for content in contents for name, _power in content.orders}
            | {name for orders in table.orders for o in orders for name, _ in o}
        )
        name_id = {name: i for i, name in enumerate(names)}

        def order_vector(orders: Orders) -> list[list[int]]:
            return [[name_id[name], power] for name, power in orders]

        content_specs = [
            [*encoder.content(content), order_vector(content.orders)]
            for content in contents
        ]
        existing = [
            [*encoder.existing(seed), [order_vector(o) for o in sorted(orders)]]
            for seed, orders in zip(table.classes, table.orders, strict=True)
        ]
        spec = {
            "adjoint": sorted(self.adjoint),
            "compress": compress,
            "coefficients": encoder.coefficients,
            "templates": encoder.templates,
            "descriptors": encoder.descriptors,
            "contents": content_specs,
            "existing": existing,
            "tensors": encoder.tensors,
        }
        text = json.dumps(spec, separators=(",", ":"))
        replay = generator_input.KEEP_REPLAY_INPUTS
        if replay:
            self.spec_text = text
        self._step("content_tables")
        if source is not None:
            builder = self.native.SeedBuilder.from_generation(
                source.native, source.handle, text
            )
        else:
            builder = self.native.SeedBuilder(text, offsets, ids)
        self._step("native_input")
        candidates = builder.analyse()
        self._step("native_combinations")
        if compressed is None:
            certified: list[str] = []
        elif build:
            certificates = self._certificates(
                builder, candidates, compressed, skeletons, encoder
            )
            table = replace(table, certificates=certificates)
            certified = list(certificates)
        else:
            certified = list(table.certificates)
        self._step("certificates")
        rings, generics = builder.needs(certified)
        self.needs = (rings, generics)
        self.encoder = encoder
        self._step("native_needs")
        legs = table.basis.legs
        generic_seeds: Callable[[int], ConnectedColorTensor]
        if NATIVE_REDUCTIONS:
            # The quark rings and generic seeds reduced in-crate; the ring
            # tensors also fill ``_reduced_primitive``'s cache (the cells).
            networks = _Networks()
            reduce_inputs = (
                [leg.label for leg in legs],
                [leg.label for leg in colour.legs],
                networks.encode(
                    (d, encoder._objects[d][1])
                    for d in builder.ring_exact_descriptors()
                ),
                networks.factored(
                    colour, ((c, contents[c]) for c in builder.generic_contents())
                ),
            )
            if replay:
                self.reduce_inputs = reduce_inputs
            builder.reduce(*reduce_inputs)
            del reduce_inputs
            self._step("reductions")
            for k, need in enumerate(rings):
                _prime_reduction(encoder.ring(need), legs, builder.ring_tensor(k))

            def generic_seeds(g: int) -> ConnectedColorTensor:
                return _decoded(builder.generic_tensor(g), colour.legs)

            new_classes, class_orders, raw_rows, skeleton_rows = (
                builder.complete_reduced()
            )
        else:
            ring_tensors = [
                _tensor(canonical_seed(_reduced_primitive(encoder.ring(need), legs)))
                for need in rings
            ]
            python_seeds = [colour.seed(skeletons[s]) for s in generics]
            generic_seeds = python_seeds.__getitem__
            self._step("reductions")
            reductions_text = (
                json.dumps(ring_tensors, separators=(",", ":")),
                json.dumps(
                    [encoder.key_tensor(seed) for seed in python_seeds],
                    separators=(",", ":"),
                ),
            )
            if replay:
                self.reductions_text = reductions_text
            new_classes, class_orders, raw_rows, skeleton_rows = builder.complete(
                *reductions_text
            )
        self._step("native_classes")
        coefficients = [
            ExactColorCoefficient(Fraction(re), Fraction(im))
            for re, im in builder.coefficients()
        ]
        order_tuples = [
            tuple((names[name], power) for name, power in vector)
            for vector in builder.order_vectors()
        ]
        classes = list(table.classes)
        for tag, primitive, generic in new_classes:
            if tag == "G":
                classes.append(generic_seeds(generic))
            else:
                classes.append(encoder.primitive(primitive))
        rows = [tuple((c, coefficients[x]) for c, x in row) for row in raw_rows]
        if len(skeleton_rows) != len(skeletons):
            raise RuntimeError("the native seed rows do not match the skeletons")
        out = dict(class_of)
        out.update(zip(skeletons, map(rows.__getitem__, skeleton_rows), strict=True))
        self._step("decode")
        self.counts = {
            "skeletons": len(skeletons),
            "contents": len(contents),
            "ring_reductions": len(rings),
            "generic_seeds": len(generics),
            "new_classes": len(new_classes),
            "rows": len(raw_rows),
        }
        cells = list(table.cells)
        cells.extend(seed_cells(table.basis, classes[len(cells) :]))
        self._step("cells")
        self._builder = builder
        self._skeletons = skeletons
        self._orders = order_tuples
        self._names = names
        self._class_orders = class_orders
        self.table = replace(
            table,
            classes=tuple(classes),
            orders=tuple(
                frozenset(order_tuples[o] for o in row) for row in class_orders
            ),
            cells=tuple(cells),
        )
        return self.table, out

    def _certificates(
        self,
        builder: Any,
        candidates: Sequence[tuple[str, Sequence[int]]],
        compressed: CompressedColour,
        skeletons: Sequence[Skeleton],
        encoder: _Encoder,
    ) -> dict[PrimitiveKind, CompressionCertificate]:
        """``CompressedColour.certificate`` on the native candidates: the
        smallest single-leg instance of each kind (ties by the ``repr`` of
        its contents) has its generic seed compared exactly."""

        colour = self.colour
        trees = _lie_vertices(colour, colour.all_adjoint_rules())
        if trees is None:
            return {}  # pure-gluon blocks are not certified Lie elements
        out: dict[PrimitiveKind, CompressionCertificate] = {}
        for name, indices in candidates:
            kind = cast(PrimitiveKind, name)
            best = min(indices, key=lambda s: repr(skeletons[s].contents))
            instance = skeletons[best]
            raw = builder.combination(best)
            combination = tuple(
                (encoder.primitive(primitive), encoder.coefficient(builder, x))
                for primitive, x in raw
            )
            generic = colour.seed(instance)
            if canonical_seed(_combination_tensor(combination, colour.legs)) != generic:
                raise ArithmeticError(
                    f"{kind} compression differs from the generic seed of "
                    f"{instance.family}"
                )
            out[kind] = CompressionCertificate(
                kind,
                tuple(
                    key
                    for key, value in compressed._vertices.items()
                    if value is not None and kind in (key[-1], "quark-ring")
                ),
                trees,
                instance,
                COMPRESSION_THEOREMS[kind],
            )
        return out

    # -- prune liveness ------------------------------------------------------

    def keep(
        self,
        slice_: Any,
        observables: Iterable[tuple[SeedTable, Mapping[Orders, Any]]] = (),
        *,
        table: SeedTable | None = None,
    ) -> tuple[dict[tuple[int, Orders], bool], list[bool]]:
        """``_prune_colour``'s ``class_live`` of every (class, loop order) of
        the stage's table and ``keep`` of each of its skeletons, natively.

        ``slice_`` gives the accuracy, sectors, Born basis and grades and the
        virtual sector degrees (``LoopSlice``); ``table`` defaults to its
        seed table.
        """

        if self._builder is None:
            raise RuntimeError("keep needs a completed stage")
        table = slice_.table if table is None else table
        accuracy = slice_.request.accuracy
        name_id = {name: i for i, name in enumerate(self._names)}

        def vector(orders: Orders) -> list[list[int]]:
            return [[name_id[name], power] for name, power in orders]

        sectors = [sector for sector, _pairs in slice_.sectors]
        criteria = [(table, slice_.degrees()["virtual"]), *observables]
        # The powers of each cell's nonzero coefficients (``class_live``'s
        # ``value and ...``), in any order: the liveness tests "any". The
        # components are tested directly, cheaper than
        # ``ExactColorCoefficient.__bool__`` (3.65 M coefficients in 0.78 M
        # cells at g g > g g g g g LC: the flags take 4.9 s instead of 5.3 s).
        spec = {
            "accuracy": accuracy,
            "tables": [
                {
                    "powers": [
                        [
                            [p for p, value in cell.items() if value.real or value.imag]
                            for cell in row
                        ]
                        for row in criterion.cells
                    ],
                    "degrees": [degrees[sector] for sector in sectors],
                }
                for criterion, degrees in criteria
            ],
            "sectors": [
                [[vector(a), vector(loop)] for a, loop in sorted(pairs)]
                for _sector, pairs in slice_.sectors
            ],
            "basis_orders": [
                [vector(o) for o in sorted(orders)] for orders in slice_.basis.orders
            ],
            "grades": [
                [b, vector(a), list(grades)]
                for (b, a), grades in slice_._born_grades().items()
            ],
        }
        live, keep = self._builder.liveness(json.dumps(spec, separators=(",", ":")))
        flags = {
            (c, self._orders[o]): flag
            for c, (orders, row) in enumerate(
                zip(self._class_orders, live, strict=True)
            )
            for o, flag in zip(orders, row, strict=True)
        }
        return flags, list(keep)

    def reading_keep(
        self,
        slice_: Any,
        observables: Iterable[tuple[SeedTable, Mapping[Orders, Any]]] = (),
    ) -> generator_input.ReadingKeep:
        """``_prune_colour``'s ``keep`` as prune flags per reading of the
        retained build that fed this stage in-crate (B1): the native LC/NLC
        prune liveness (B3) with the slice's degrees, Born grades and
        sectors. ``NativeGeneration.schedules`` rebuilds the pruned plan from
        them without a Python callback."""

        source = self.build_source
        if source is None or not source.complete:
            raise RuntimeError(
                "reading flags need a stage fed by a complete retained build"
            )
        _flags, keep = self.keep(slice_, observables)
        return generator_input.ReadingKeep(source, bytes(map(bool, keep)))


class _Encoder:
    """The Θ(contents) tables of a stage: coefficients, network templates,
    descriptors (each distinct one once), existing classes' tensors."""

    def __init__(self, adjoint: frozenset[int], compressed: CompressedColour | None):
        self.adjoint = adjoint
        self.compressed = compressed
        self.coefficients: list[list[str]] = []
        self._coefficient_id: dict[ExactColorCoefficient, int] = {}
        self.templates: list[list[str]] = []
        self._template_id: dict[tuple[str, ...], int] = {}
        self.descriptors: list[list[Any]] = []
        self._descriptor_id: dict[Any, int] = {}
        self._objects: list[Any] = []
        self._exact: dict[int, tuple[Any, list[int]]] = {}
        self.tensors: list[Any] = []
        self._decompositions: dict[int, Any] = {}
        self._decoded: dict[tuple[str, str], ExactColorCoefficient] = {}

    def _coefficient(self, value: ExactColorCoefficient) -> int:
        found = self._coefficient_id.get(value)
        if found is None:
            found = self._coefficient_id[value] = len(self.coefficients)
            self.coefficients.append([_text(value.real), _text(value.imag)])
        return found

    def _descriptor(self, descriptor: tuple[Any, ...]) -> int:
        kind = descriptor[0]
        if kind == "exact":
            parts, labels = _template(descriptor[1], self.adjoint)
            text = tuple(parts)
            template = self._template_id.get(text)
            if template is None:
                template = self._template_id[text] = len(self.templates)
                self.templates.append(parts)
            payload: tuple[Any, ...] = (2, tuple(labels), False, template)
        elif kind == "adjoint":
            payload = (0, tuple(descriptor[1]), False, 0)
        else:
            payload = (1, tuple(descriptor[1]), bool(descriptor[2]), 0)
        found = self._descriptor_id.get(payload)
        if found is None:
            found = self._descriptor_id[payload] = len(self.descriptors)
            self.descriptors.append([payload[0], list(payload[1]), *payload[2:]])
            self._objects.append(descriptor)
        return found

    def content(self, content: Content) -> list[Any]:
        """(position kind, single-leg, terms) of ``CompressedColour._content``."""

        compressed = self.compressed
        if compressed is None:
            return [_POSITIONS["exact"], True, []]
        position, single, terms = compressed._content(content)
        out = []
        for (vertex, t), descriptor in terms:
            if vertex is None:
                value = ExactColorCoefficient(Fraction(1))
            else:
                decomposition = self._decompositions.get(vertex)
                if decomposition is None:
                    self._decompositions = {
                        id(d): d for d in compressed._vertices.values() if d is not None
                    }
                    decomposition = self._decompositions[vertex]
                value = decomposition.terms[t][0]
            out.append([self._coefficient(value), self._descriptor(descriptor)])
        return [_POSITIONS[position], bool(single), out]

    def existing(self, seed: Any) -> list[Any]:
        """An existing class: (tag, items)."""

        if isinstance(seed, Primitive):
            if seed.kind == "quark-ring":
                ring = cast(tuple[tuple[Any, ...], ...], seed.word)
                return [2, [self._descriptor(d) for d in ring]]
            return [0 if seed.kind == "adjoint-ring" else 1, list(seed.word)]
        self.tensors.append(self.key_tensor(seed))
        return [3, [len(self.tensors) - 1]]

    def key_tensor(self, seed: ConnectedColorTensor) -> list[Any]:
        """A generic seed as its class key reads it: ``canonical_seed`` of it
        when classes merge by proportionality (``_ClassKeys(merge=True)``),
        itself otherwise."""

        merge = self.compressed is not None
        return _tensor(canonical_seed(seed) if merge else seed)

    def primitive(self, raw: Sequence[Any]) -> Primitive:
        kind, letters, descriptors = raw
        if kind == "quark-ring":
            return Primitive(kind, tuple(self._objects[d] for d in descriptors))
        return Primitive(kind, tuple(letters))

    def coefficient(self, builder: Any, x: int) -> ExactColorCoefficient:
        if not self._decoded:
            self._decoded = {
                (re, im): ExactColorCoefficient(Fraction(re), Fraction(im))
                for re, im in builder.coefficients()
            }
        re, im = builder.coefficients()[x]
        return self._decoded[re, im]

    def ring(self, need: Sequence[Any]) -> Primitive:
        """The canonical relabelled quark ring of a ring need
        (``_ring_relabelling``)."""

        descriptors, forward = need
        rename = dict(forward)

        def label(value: Any) -> Any:
            return rename.get(value, value) if isinstance(value, int) else value

        word = []
        for d in descriptors:
            descriptor = self._objects[d]
            if not rename:
                word.append(descriptor)
            elif descriptor[0] == "exact":
                word.append(("exact", relabel(descriptor[1], label)))
            else:
                kind, letters, *rest = descriptor
                word.append((kind, tuple(map(label, letters)), *rest))
        return Primitive("quark-ring", tuple(word))


class LazyFourier:
    """``seed_fourier`` of a seed table, certified on first use (design note
    in the module docstring; phase 1 moves it into ``LoopSlice``)."""

    def __init__(self, table: Callable[[], SeedTable]):
        self._table = table
        self._done = False
        self._value: SeedFourier | None = None

    def reset(self) -> None:
        self._done = False
        self._value = None

    def get(self) -> SeedFourier | None:
        if not self._done:
            self._value = seed_fourier(self._table())
            self._done = True
        return self._value
