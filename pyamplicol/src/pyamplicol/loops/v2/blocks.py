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
  the full-colour current is the sum over blocks of the string product, each
  string ``s -> a_1 ... a_k -> e`` contributing ``(tau^{a_k} ... tau^{a_1})_{e s}``
  and an adjoint field closing ``out -> in`` with ``tau^X_{in, out}``.
  U(1)-subtraction auxiliaries are separate colour-singlet blocks.
* **Signs.** Block currents carry the tree engine's fermion-ordering signs
  (Koszul signs for colour-singlet fermions, colour-flow pairing for coloured
  lines); the loop sign rule of design §3.3 composes them.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from itertools import pairwise
from types import MappingProxyType
from typing import Literal

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
from pyamplicol.models.base import CouplingOrders, Model, QuantumNumberFlow
from pyamplicol.processes.ir import CanonicalProcessIR

BlockClass = Literal["plain", "ward", "spin-source", "counterterm"]
TreeKernelVariant = Literal["transverse", "full"]

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


def add_coupling_orders(*terms: CouplingOrders) -> CouplingOrders:
    totals: dict[str, int] = {}
    for orders in terms:
        for name, power in orders:
            totals[str(name).upper()] = totals.get(str(name).upper(), 0) + int(power)
    return tuple(sorted((name, power) for name, power in totals.items() if power))


@dataclass(frozen=True, slots=True)
class RequestedLoopBlocks:
    """Loop-aware reachability fed by the family recursion's position table.

    ``requests`` holds the (leg mask, particle id, coupling orders) of every
    block current an admitted position consumes. Particle ids follow the tree
    engine: the field produced by the tree and entering the loop vertex, in
    the all-outgoing convention (the conjugate of the incoming field name).
    """

    requests: frozenset[tuple[int, int, CouplingOrders]]
    _masks: frozenset[int] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        requests = frozenset(
            (int(mask), int(particle_id), add_coupling_orders(orders))
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
class BlockColourFlow:
    """U(N) colour-flow strings of a block word (module docstring)."""

    strings: tuple[tuple[int, ...], ...]
    incoming_open: bool
    outgoing_open: bool
    singlet_labels: tuple[int, ...] = ()


def block_colour_flow(
    word: Iterable[int],
    representations: Mapping[int, int],
) -> BlockColourFlow:
    """Split a block word into its colour-flow strings.

    ``representations`` maps leg labels to the colour representation of the
    outgoing particle (1, 3, -3 or 8). A new string starts at every
    fundamental leg that follows an antifundamental one.
    """

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
            raise ValueError(f"word {tuple(word)} is not a colour-flow block word")
    return BlockColourFlow(
        strings=tuple(tuple(string) for string in strings),
        incoming_open=bool(coloured) and representations[coloured[0]] != 3,
        outgoing_open=bool(coloured) and representations[coloured[-1]] != -3,
        singlet_labels=tuple(singlets),
    )


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
    id: int
    key: BlockKey
    colour_flow: BlockColourFlow
    block_class: BlockClass
    kernel_variant: TreeKernelVariant
    rows: tuple[BlockRow, ...]

    @property
    def is_source(self) -> bool:
        return len(self.key.word) == 1


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

    def blocks_on(self, mask: int) -> tuple[Block, ...]:
        return tuple(block for block in self.blocks if block.key.external_mask == mask)

    def field_blocks(self, mask: int) -> tuple[tuple[int, CouplingOrders], ...]:
        """Distinct (particle id, coupling orders) of the blocks on ``mask``."""

        return tuple(
            sorted(
                {
                    (block.key.particle_id, block.key.coupling_orders)
                    for block in self.blocks_on(mask)
                }
            )
        )

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
                pairs.setdefault(add_coupling_orders(left, right), []).append(
                    (left, right)
                )
        return MappingProxyType(
            {squared: tuple(rows) for squared, rows in sorted(pairs.items())}
        )


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
    representations = {
        leg.label: int(model.color_rep(int(leg.outgoing_pdg)))
        for leg in process.legs
        if leg.outgoing_pdg is not None
    }
    return TreeBlocks(
        process=process,
        blocks_dag=blocks_dag,
        born_dag=born_dag,
        blocks=_blocks(blocks_dag, representations),
        born_roots=_born_roots(born_dag, model),
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


def _blocks(
    dag: GenericDAG,
    representations: Mapping[int, int],
) -> tuple[Block, ...]:
    decode = _helicity_decoder(dag)
    rows_by_key: dict[BlockKey, list[BlockRow]] = {}
    for current in dag.currents:
        rows_by_key.setdefault(BlockKey.from_index(current.index), []).append(
            BlockRow(decode(current.index.helicity_ancestry), current.id)
        )
    blocks = []
    for key, rows in rows_by_key.items():
        blocks.append(
            Block(
                id=len(blocks),
                key=key,
                colour_flow=block_colour_flow(key.word, representations),
                block_class="plain",
                kernel_variant=KERNEL_VARIANT_BY_BLOCK_CLASS["plain"],
                rows=tuple(sorted(rows, key=lambda row: row.helicities)),
            )
        )
    return tuple(blocks)


def _born_root_orders(dag: GenericDAG, model: Model, root: AmplitudeRoot):
    left = dag.currents[root.left_id].index
    right = dag.currents[root.right_id].index
    if root.kind == "vertex-closure":
        if root.vertex_kind is None or root.vertex_particles is None:
            raise ValueError("vertex-closure root has no vertex")
        vertex = next(
            vertex
            for vertex in model.vertices
            if vertex.kind == root.vertex_kind
            and vertex.particles == root.vertex_particles
        )
        return add_coupling_orders(model.combine_coupling_orders(left, right, vertex))
    return add_coupling_orders(left.coupling_orders, right.coupling_orders)


def _born_roots(dag: GenericDAG, model: Model) -> tuple[BornRoot, ...]:
    decode = _helicity_decoder(dag)
    return tuple(
        BornRoot(
            root_id=root.id,
            coupling_orders=_born_root_orders(dag, model, root),
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
    "TreeBlocks",
    "TreeKernelVariant",
    "add_coupling_orders",
    "block_colour_flow",
    "build_tree_blocks",
]
