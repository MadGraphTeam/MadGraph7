# SPDX-License-Identifier: 0BSD
"""Tree blocks and Born export for loop requests (design §3.2).

A *block* is a colour-ordered off-shell tree current over a proper subset B of
the external legs: the sum of all trees on B that a loop vertex can absorb.
Blocks come from the tree engine in loop-request mode
(``GenericDAGCompiler(loop_blocks=True)``): every U(N) colour-flow ordering of
every admitted leg set, with the Born-closure pruning replaced by a loop-aware
reachability test. The Born is the ordinary tree compilation of the same
process, unchanged, with its roots bucketed by exact coupling order.

Conventions a consumer relies on:

* **Identity.** A block is keyed by its full ``CurrentIndex`` minus the
  helicity ancestry (``BlockKey``); its rows are the helicity assignments of
  its legs. Massless fermion blocks split by chirality, as in the tree engine.
* **Values.** A row is the tree runtime's value of that current: a source
  wavefunction for one leg, otherwise the current including the propagator of
  its own field (the value a loop vertex attaches).
* **Colour.** ``BlockColourFlow`` lists the U(N) colour-flow strings of the
  block's word. A string runs from a fundamental leg (or the incoming open end)
  through adjoint legs to an antifundamental leg (or the outgoing open end), in
  the orientation of ``LCOpenColorLine`` (fundamental label first). With
  generators normalised as ``tr(t^a t^b) = delta^{ab}/2`` and ``tau = sqrt(2) t``
  each string ``s -> a_1 ... a_k -> e`` contributes
  ``(tau^{a_k} ... tau^{a_1})_{e s}`` and an adjoint field closes
  ``out -> in`` with ``tau^X_{in, out}``.
  U(1)-subtraction auxiliaries are separate colour-singlet blocks.
* **Signs.** ``BlockColourFlow.sign`` extends the tree engine's Born colour
  metric (``color/correlator_matrices.py``) to blocks: the parity of the
  antifundamental string ends listed in the order of their fundamental
  starts, both ranked by label. The block's own open end ranks after every
  leg, and the two open ends of an adjoint block join through it into one
  string. ``sum_words colour_flow.sign * colour * row`` is the current in the
  colour-flow convention of loop-request blocks (the tree Born programs sign
  coloured fermions by their Wick order instead,
  ``generation/fermion_ordering.py``; the two agree on gluon-joined Born
  roots). Against Feynman rules the colour-flow convention carries ``-1``
  per colour-octet connection (a gluon, contact or U(1)-subtraction current
  joining two coloured fermion lines) and nothing for colour-singlet
  connections (photon, Z, W), so
  ``Block.wick_sign`` folds in ``Block.connection_sign`` and the parity of
  the identity pairing of the current's coloured fermion endpoints (each
  line listed (fundamental, antifundamental) against ascending labels, open
  end last). Then ``sum_words wick_sign * colour * row`` is the Feynman-rule
  current with each diagram signed by the parity of its fermion lines
  listed that way, FeynKit's convention, for flavour-changing lines as well.
  The connection parity is the same along every tree of a block; generation
  fails otherwise. Colour-singlet fermions carry the tree engine's Koszul
  signs inside the rows (``generation/fermion_ordering.py``). The loop sign
  rule of design §3.3 composes these signs.
* **Kernel variants.** ``KERNEL_VARIANT_BY_BLOCK_CLASS`` is the tree kernel
  variant a block class requires; a block records the variant of the model's
  compiled kernels (``compile_tree_kernel_variant``), which must satisfy it.
* **Families.** ``TreeBlocks.block_table`` is the WP1.3 ``BlockTable`` view of
  the blocks in a native model's incoming field names, and
  ``requested_loop_blocks`` turns ``FamilySet.block_requests()`` back into the
  compiler's loop-aware reachability.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from itertools import combinations, pairwise
from types import MappingProxyType
from typing import Any, Literal

from pyamplicol.generation.dag_compiler import (
    DAGProgressCallback,
    GenericDAGCompiler,
    LoopBlockReachability,
    compile_generic_dag,
)
from pyamplicol.generation.dag_types import (
    AmplitudeRoot,
    ColorState,
    CurrentIndex,
    GenericDAG,
)
from pyamplicol.loops.v2.families import (
    BlockCurrent,
    ExternalLeg,
    MassId,
    _mass_name,
    _model_data,
    add_orders,
)
from pyamplicol.models.base import CouplingOrders, Model, QuantumNumberFlow
from pyamplicol.models.contracts import U1_SUBTRACTION_AUXILIARY_KIND, TreeKernelVariant
from pyamplicol.models.prepared_catalog_builder import _tree_kernel_variant
from pyamplicol.processes.ir import CanonicalProcessIR

BlockClass = Literal["plain", "ward", "spin-source", "counterterm"]

# The transverse massless Yang-Mills kernel drops only terms proportional to
# an input current's self-contraction p.J, so it is exact for blocks built
# from transverse on-shell sources ("plain"). Every other class has a
# non-conserved input and needs the full Feynman rule (design §3.2 item 2).
KERNEL_VARIANT_BY_BLOCK_CLASS: Mapping[BlockClass, TreeKernelVariant] = (
    MappingProxyType(
        {
            "plain": "transverse",
            "ward": "full",
            "spin-source": "full",
            "counterterm": "full",
        }
    )
)


@dataclass(frozen=True, slots=True)
class RequestedLoopBlocks:
    """Loop-aware reachability fed by the family recursion's position table.

    ``requests`` holds the (leg mask, particle id, coupling orders) of every
    block current an admitted position consumes. Particle ids follow the tree
    engine: the field produced by the tree and entering the loop vertex, in
    the all-outgoing convention (the conjugate of the incoming field name);
    ``requested_loop_blocks`` converts family requests.
    """

    requests: frozenset[tuple[int, int, CouplingOrders]]
    _masks: frozenset[int] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        requests = frozenset(
            (int(mask), int(particle_id), add_orders(orders))
            for mask, particle_id, orders in self.requests
        )
        object.__setattr__(self, "requests", requests)
        object.__setattr__(self, "_masks", frozenset(row[0] for row in requests))

    def admits_mask(self, mask: int) -> bool:
        return mask in self._masks

    def admits_block(self, index: CurrentIndex) -> bool:
        return (
            index.external_mask,
            index.particle_id,
            index.coupling_orders,
        ) in self.requests


@dataclass(frozen=True, slots=True)
class _NativeFields:
    """Incoming field names of a native model and their tree-engine ids."""

    by_name: Mapping[str, Mapping[str, Any]]
    by_pdg: Mapping[int, Mapping[str, Any]]

    @classmethod
    def of(cls, model: Any) -> _NativeFields:
        particles = _model_data(model)["particles"]
        return cls(
            {entry["name"]: entry for entry in particles},
            {int(entry["pdg_code"]): entry for entry in particles},
        )

    def incoming(self, particle_id: int) -> Mapping[str, Any]:
        """Incoming field of an all-outgoing tree-engine particle id."""

        return self.by_name[self.by_pdg[int(particle_id)]["antiname"]]

    def particle_id(self, incoming_name: str) -> int:
        return int(self.by_name[self.by_name[incoming_name]["antiname"]]["pdg_code"])


def requested_loop_blocks(
    requests: Iterable[BlockCurrent], native_model: Any
) -> RequestedLoopBlocks:
    """Reachability of the block currents a family set consumes.

    ``requests`` is ``FamilySet.block_requests()``; ``native_model`` is the
    native model (or its JSON mapping) whose names the families use.
    """

    fields = _NativeFields.of(native_model)
    return RequestedLoopBlocks(
        frozenset(
            (request.mask, fields.particle_id(request.field), request.orders)
            for request in requests
        )
    )


@dataclass(frozen=True, slots=True)
class BlockColourFlow:
    """U(N) colour-flow strings and sign of a block word (module docstring)."""

    strings: tuple[tuple[int, ...], ...]
    incoming_open: bool
    outgoing_open: bool
    singlet_labels: tuple[int, ...] = ()
    sign: int = 1


def block_colour_flow(
    word: Iterable[int],
    representations: Mapping[int, int],
    field_representation: int,
) -> BlockColourFlow:
    """Split a block word into its colour-flow strings and pairing sign.

    ``representations`` maps leg labels to the colour representation of the
    outgoing particle (1, 3, -3 or 8); ``field_representation`` is that of
    the block's field. A new string starts at every fundamental leg that
    follows an antifundamental one.
    """

    word = tuple(word)
    coloured: list[int] = []
    singlets: list[int] = []
    for label in word:
        representation = int(representations[label])
        if representation == 1:
            singlets.append(label)
        elif representation in (3, -3, 8):
            coloured.append(label)
        else:
            raise ValueError(f"leg {label} is not colour-flow lowerable")
    strings: list[list[int]] = []
    for label in coloured:
        if strings and representations[label] == 3:
            strings.append([label])
        elif strings:
            strings[-1].append(label)
        else:
            strings.append([label])
    for left, right in pairwise(coloured):
        if (representations[left] == -3) != (representations[right] == 3):
            raise ValueError(f"word {word} is not a colour-flow block word")
    incoming_open = bool(coloured) and representations[coloured[0]] != 3
    outgoing_open = bool(coloured) and representations[coloured[-1]] != -3
    expected = {1: (False, False), 3: (False, True), -3: (True, False)}.get(
        int(field_representation), (True, True)
    )
    if coloured and (incoming_open, outgoing_open) != expected:
        raise ValueError(
            f"word {word} does not match a field in representation "
            f"{field_representation}"
        )
    return BlockColourFlow(
        strings=tuple(tuple(string) for string in strings),
        incoming_open=incoming_open,
        outgoing_open=outgoing_open,
        singlet_labels=tuple(singlets),
        sign=_pairing_sign(strings, representations),
    )


def _pairing_sign(strings: list[list[int]], representations: Mapping[int, int]) -> int:
    """Parity of the string pairing; ``None`` is the open end, ranked last."""

    end_of: dict[int | None, int | None] = {}
    for string in strings:
        start = string[0] if representations[string[0]] == 3 else None
        end = string[-1] if representations[string[-1]] == -3 else None
        end_of[start] = end
    if None in end_of and None in end_of.values():
        # An adjoint field joins its open ends: the string ending at it
        # continues through it into the string starting at it. A single
        # adjoint string closes into a trace without endpoints.
        end = end_of.pop(None)
        if end is not None:
            (start,) = (start for start, other in end_of.items() if other is None)
            end_of[start] = end
    return _parity([end_of[start] for start in sorted(end_of, key=_open_last)])


def _open_last(label: int | None) -> tuple[bool, int]:
    return (label is None, 0 if label is None else label)


def _parity(sequence: list[int | None]) -> int:
    ranks = [_open_last(item) for item in sequence]
    inversions = sum(left > right for left, right in combinations(ranks, 2))
    return -1 if inversions % 2 else 1


def identity_pairing_sign(
    fundamentals: Iterable[int | None], antifundamentals: Iterable[int | None]
) -> int:
    """Parity of the identity pairing of coloured fermion endpoints.

    Lines pair the i-th smallest fundamental with the i-th smallest
    antifundamental endpoint (``None`` is the open end, ranked last), each
    listed (fundamental, antifundamental), relative to ascending order.
    """

    starts = sorted(fundamentals, key=_open_last)
    ends = sorted(antifundamentals, key=_open_last)
    if len(starts) != len(ends):
        raise ValueError("unpaired fermion endpoints")
    return _parity([item for pair in zip(starts, ends, strict=True) for item in pair])


@dataclass(frozen=True, slots=True)
class BlockKey:
    """Full ``CurrentIndex`` of a block without its helicity ancestry."""

    external_mask: int
    particle_id: int
    word: tuple[int, ...]
    coupling_orders: CouplingOrders
    color_state: ColorState
    chirality: int
    spin_state: int | tuple[int, ...]
    flavour_flow: tuple[int, ...]
    quantum_number_flow: QuantumNumberFlow
    auxiliary_kind: str | None

    @classmethod
    def from_index(cls, index: CurrentIndex) -> BlockKey:
        return cls(
            external_mask=index.external_mask,
            particle_id=index.particle_id,
            word=index.ordered_external_labels,
            coupling_orders=index.coupling_orders,
            color_state=index.color_state,
            chirality=index.chirality,
            spin_state=index.spin_state,
            flavour_flow=index.flavour_flow,
            quantum_number_flow=index.quantum_number_flow,
            auxiliary_kind=index.auxiliary_kind,
        )

    @property
    def labels(self) -> tuple[int, ...]:
        return tuple(sorted(self.word))


@dataclass(frozen=True, slots=True)
class BlockRow:
    """One helicity assignment of a block: ``((label, helicity), ...)``."""

    helicities: tuple[tuple[int, int], ...]
    current_id: int


@dataclass(frozen=True, slots=True)
class Block:
    """One block; ``wick_sign`` is its Feynman-rule sign (module docstring)."""

    id: int
    key: BlockKey
    colour_flow: BlockColourFlow
    block_class: BlockClass
    kernel_variant: TreeKernelVariant
    rows: tuple[BlockRow, ...]
    connection_sign: int = 1
    pairing_sign: int = 1

    @property
    def is_source(self) -> bool:
        return len(self.key.word) == 1

    @property
    def wick_sign(self) -> int:
        return self.colour_flow.sign * self.connection_sign * self.pairing_sign


@dataclass(frozen=True, slots=True)
class BornRoot:
    """A Born amplitude root with its exact coupling order and helicities."""

    root_id: int
    coupling_orders: CouplingOrders
    color_sector_id: int | None
    helicities: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class TreeBlocks:
    """Blocks and Born of one loop request (design §3.2 exports)."""

    process: CanonicalProcessIR
    blocks_dag: GenericDAG
    born_dag: GenericDAG
    blocks: tuple[Block, ...]
    born_roots: tuple[BornRoot, ...]
    kernel_variant: TreeKernelVariant

    def blocks_on(self, mask: int) -> tuple[Block, ...]:
        return tuple(block for block in self.blocks if block.key.external_mask == mask)

    def field_blocks(self, mask: int) -> tuple[tuple[int, CouplingOrders], ...]:
        """Distinct (particle id, coupling orders) of the field blocks on ``mask``.

        Auxiliary blocks (contact and U(1)-subtraction pieces) are part of
        their field's colour decomposition, not separate fields.
        """

        return tuple(
            sorted(
                {
                    (block.key.particle_id, block.key.coupling_orders)
                    for block in self.blocks_on(mask)
                    if block.key.auxiliary_kind is None
                }
            )
        )

    def block_table(self, native_model: Any) -> TreeBlockTable:
        """The ``BlockTable`` (WP1.3) view in ``native_model``'s field names."""

        return TreeBlockTable(self, _NativeFields.of(native_model))

    @property
    def born_orders(self) -> tuple[CouplingOrders, ...]:
        return tuple(sorted({root.coupling_orders for root in self.born_roots}))

    def born_roots_by_order(self) -> Mapping[CouplingOrders, tuple[BornRoot, ...]]:
        buckets: dict[CouplingOrders, list[BornRoot]] = {}
        for root in self.born_roots:
            buckets.setdefault(root.coupling_orders, []).append(root)
        return MappingProxyType(
            {orders: tuple(roots) for orders, roots in sorted(buckets.items())}
        )

    def squared_born_orders(
        self,
    ) -> Mapping[CouplingOrders, tuple[tuple[CouplingOrders, CouplingOrders], ...]]:
        """Amplitude-order pairs (a, b) of each squared order a + b."""

        orders = self.born_orders
        pairs: dict[CouplingOrders, list[tuple[CouplingOrders, CouplingOrders]]] = {}
        for left in orders:
            for right in orders:
                pairs.setdefault(add_orders(left, right), []).append((left, right))
        return MappingProxyType(
            {squared: tuple(rows) for squared, rows in sorted(pairs.items())}
        )


class TreeBlockTable:
    """Field-level ``BlockTable`` view of loop-request tree blocks.

    Legs are 0-based (bit ``i`` of a mask is tree label ``i + 1``) and carry
    the incoming field, so outgoing legs are crossed; a block current's field
    is the incoming name of the field entering the loop vertex.
    """

    def __init__(self, blocks: TreeBlocks, fields: _NativeFields):
        self._blocks = blocks
        self._fields = fields
        legs = []
        for index, leg in enumerate(
            sorted(blocks.process.legs, key=lambda leg: leg.label)
        ):
            if leg.label != index + 1 or leg.outgoing_pdg is None:
                raise ValueError("tree blocks need legs labelled 1..n")
            incoming = fields.incoming(leg.outgoing_pdg)
            legs.append(
                ExternalLeg(
                    index,
                    incoming["name"],
                    MassId(_mass_name(incoming["mass"]), _mass_name(incoming["width"])),
                    incoming["color"],
                )
            )
        self._legs = tuple(legs)
        found: dict[int, set[BlockCurrent]] = {}
        for block in blocks.blocks:
            key = block.key
            if key.auxiliary_kind is None:
                found.setdefault(key.external_mask, set()).add(
                    BlockCurrent(
                        key.external_mask,
                        fields.incoming(key.particle_id)["name"],
                        key.coupling_orders,
                    )
                )
        self._currents = {mask: tuple(sorted(rows)) for mask, rows in found.items()}

    @property
    def legs(self) -> tuple[ExternalLeg, ...]:
        return self._legs

    def currents(self, mask: int) -> tuple[BlockCurrent, ...]:
        return self._currents.get(mask, ())


def build_tree_blocks(
    process: CanonicalProcessIR,
    model: Model,
    *,
    born_coupling_orders: Mapping[str, int] | None,
    block_coupling_orders: Mapping[str, int] | None,
    reachability: LoopBlockReachability | None = None,
    ignored_particle_ids: Iterable[int] = (),
    ignored_vertex_kinds: Iterable[int] = (),
    progress_callback: DAGProgressCallback | None = None,
) -> TreeBlocks:
    """Compile the blocks and the Born of one loop request.

    The Born is ``compile_generic_dag`` of ``process`` with
    ``born_coupling_orders``, exactly as a tree request compiles it. Blocks are
    exact colour-flow currents, so they are compiled at full colour whatever
    the requested accuracy; ``block_coupling_orders`` bounds the orders a
    block may carry (the loop-amplitude orders bound it without loss).
    """

    ignored_particles = tuple(ignored_particle_ids)
    ignored_kinds = tuple(ignored_vertex_kinds)
    born_dag = compile_generic_dag(
        process,
        model=model,
        max_coupling_orders=born_coupling_orders,
        ignored_particle_ids=ignored_particles,
        ignored_vertex_kinds=ignored_kinds,
        progress_callback=progress_callback,
    )
    blocks_dag = GenericDAGCompiler(
        model=model,
        max_coupling_orders=block_coupling_orders,
        ignored_particle_ids=ignored_particles,
        ignored_vertex_kinds=ignored_kinds,
        progress_callback=progress_callback,
        loop_blocks=True,
        loop_block_reachability=reachability,
    ).compile(replace(process, color_accuracy="full"))
    kernel_variant = _tree_kernel_variant(model)
    return TreeBlocks(
        process=process,
        blocks_dag=blocks_dag,
        born_dag=born_dag,
        blocks=_blocks(blocks_dag, model, kernel_variant),
        born_roots=_born_roots(born_dag, model),
        kernel_variant=kernel_variant,
    )


def _helicity_decoder(dag: GenericDAG):
    bits: list[tuple[int, int, int]] = []
    for source_id in dag.sources:
        source = dag.currents[source_id]
        if source.source_leg_label is None or source.source_helicity is None:
            raise ValueError("block source has no leg or helicity")
        bits.append(
            (
                source.index.helicity_ancestry,
                source.source_leg_label,
                source.source_helicity,
            )
        )

    def decode(ancestry: int) -> tuple[tuple[int, int], ...]:
        return tuple(
            sorted((label, helicity) for bit, label, helicity in bits if ancestry & bit)
        )

    return decode


def _checked_kernel_variant(
    block_class: BlockClass, kernel_variant: TreeKernelVariant
) -> TreeKernelVariant:
    required = KERNEL_VARIANT_BY_BLOCK_CLASS[block_class]
    if required == "full" and kernel_variant != "full":
        raise ValueError(
            f"{block_class!r} blocks need the full tree kernels; compile the "
            "model with compile_tree_kernel_variant(compiled, 'full')"
        )
    return kernel_variant


def _blocks(
    dag: GenericDAG, model: Model, kernel_variant: TreeKernelVariant
) -> tuple[Block, ...]:
    decode = _helicity_decoder(dag)
    representations = {
        leg.label: int(model.color_rep(int(leg.outgoing_pdg)))
        for leg in dag.process.legs
        if leg.outgoing_pdg is not None
    }
    endpoints = {
        leg.label: representations[leg.label]
        for leg in dag.process.legs
        if leg.outgoing_pdg is not None
        and abs(representations[leg.label]) == 3
        and model.is_fermion(int(leg.outgoing_pdg))
    }
    connections = _connection_parities(dag, model, endpoints)
    rows_by_key: dict[BlockKey, list[BlockRow]] = {}
    for current in dag.currents:
        rows_by_key.setdefault(BlockKey.from_index(current.index), []).append(
            BlockRow(decode(current.index.helicity_ancestry), current.id)
        )
    blocks = []
    for key, rows in rows_by_key.items():
        field_representation = model.color_rep(key.particle_id)
        open_end = (
            {None: -int(field_representation)}
            if model.is_fermion(key.particle_id) and abs(field_representation) == 3
            else {}
        )
        ends = {
            **{label: endpoints[label] for label in key.word if label in endpoints},
            **open_end,
        }
        parities = {connections[row.current_id] for row in rows}
        if len(parities) != 1:
            raise ValueError(f"block {key.word} mixes colour-connection signs")
        blocks.append(
            Block(
                id=len(blocks),
                key=key,
                colour_flow=block_colour_flow(
                    key.word, representations, field_representation
                ),
                block_class="plain",
                kernel_variant=_checked_kernel_variant("plain", kernel_variant),
                rows=tuple(sorted(rows, key=lambda row: row.helicities)),
                connection_sign=-1 if parities.pop() else 1,
                pairing_sign=identity_pairing_sign(
                    [label for label, rep in ends.items() if rep == 3],
                    [label for label, rep in ends.items() if rep == -3],
                ),
            )
        )
    return tuple(blocks)


def _connection_parities(
    dag: GenericDAG, model: Model, endpoints: Mapping[int, int]
) -> list[int]:
    """Parity of the colour-octet connections between fermion lines.

    A merge connects two lines when both inputs hold coloured fermion
    endpoints and one of them is a colour-octet current (a gluon, contact or
    U(1)-subtraction auxiliary). Every derivation of a current must agree.
    """

    def octet(index: CurrentIndex) -> bool:
        return (
            model.color_rep(index.particle_id) == 8
            or index.auxiliary_kind == U1_SUBTRACTION_AUXILIARY_KIND
        )

    def has_endpoints(index: CurrentIndex) -> bool:
        return any(label in endpoints for label in index.external_labels)

    currents = dag.currents
    parity: list[int | None] = [
        0 if current.is_source else None for current in currents
    ]
    for interaction in sorted(
        dag.interactions,
        key=lambda row: currents[row.result_id].index.external_mask.bit_count(),
    ):
        left = currents[interaction.left_id].index
        right = currents[interaction.right_id].index
        connects = (
            has_endpoints(left)
            and has_endpoints(right)
            and (octet(left) or octet(right))
        )
        value = parity[interaction.left_id] ^ parity[interaction.right_id] ^ connects
        known = parity[interaction.result_id]
        if known is None:
            parity[interaction.result_id] = value
        elif known != value:
            raise ValueError(
                f"current {interaction.result_id} has derivations with different "
                "colour-connection signs"
            )
    return parity


def _born_root_orders(
    dag: GenericDAG,
    model: Model,
    root: AmplitudeRoot,
    vertices: Mapping[tuple[int, tuple[int, int, int]], Any],
) -> CouplingOrders:
    left = dag.currents[root.left_id].index
    right = dag.currents[root.right_id].index
    if root.kind == "vertex-closure":
        if root.vertex_kind is None or root.vertex_particles is None:
            raise ValueError("vertex-closure root has no vertex")
        vertex = vertices.get((root.vertex_kind, root.vertex_particles))
        if vertex is None:
            raise ValueError(
                f"vertex-closure root {root.id} names no model vertex "
                f"{root.vertex_kind} {root.vertex_particles}"
            )
        return add_orders(model.combine_coupling_orders(left, right, vertex))
    return add_orders(left.coupling_orders, right.coupling_orders)


def _born_roots(dag: GenericDAG, model: Model) -> tuple[BornRoot, ...]:
    decode = _helicity_decoder(dag)
    vertices: dict[tuple[int, tuple[int, int, int]], Any] = {}
    for vertex in model.vertices:
        vertices.setdefault((vertex.kind, vertex.particles), vertex)
    return tuple(
        BornRoot(
            root_id=root.id,
            coupling_orders=_born_root_orders(dag, model, root, vertices),
            color_sector_id=root.color_sector_id,
            helicities=decode(
                dag.currents[root.left_id].index.helicity_ancestry
                | dag.currents[root.right_id].index.helicity_ancestry
            ),
        )
        for root in dag.amplitude_roots
    )


__all__ = [
    "KERNEL_VARIANT_BY_BLOCK_CLASS",
    "Block",
    "BlockClass",
    "BlockColourFlow",
    "BlockKey",
    "BlockRow",
    "BornRoot",
    "RequestedLoopBlocks",
    "TreeBlockTable",
    "TreeBlocks",
    "block_colour_flow",
    "build_tree_blocks",
    "identity_pairing_sign",
    "requested_loop_blocks",
]
