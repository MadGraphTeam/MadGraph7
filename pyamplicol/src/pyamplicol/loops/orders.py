# SPDX-License-Identifier: 0BSD
"""Perturbative bookkeeping after dependency-backed graph eligibility.

Correction eligibility is supplied by graph construction, never reconstructed
from the difference between a loop order and a Born order. All powers refer to
fundamental couplings (not alpha_s or alpha).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from pyamplicol.models.base import CouplingOrders


def canonical_order(order: CouplingOrders) -> CouplingOrders:
    values: dict[str, int] = {}
    for name, power in order:
        if not isinstance(name, str) or not name or name in values:
            raise ValueError("coupling order names must be nonempty and unique")
        if isinstance(power, bool) or not isinstance(power, int) or power < 0:
            raise ValueError("coupling powers must be nonnegative integers")
        values[name] = power
    return tuple(sorted((name, power) for name, power in values.items() if power))


def squared_order(born: CouplingOrders, loop: CouplingOrders) -> CouplingOrders:
    values = dict(born)
    for name, power in loop:
        values[name] = values.get(name, 0) + power
    return tuple(sorted(values.items()))


@dataclass(frozen=True, slots=True)
class LoopEligibility:
    """Physical-origin facts for MadLoop's joint correction-set selection.

    Every circulating species must be perturbative in at least one selected
    correction, and a selected order must occur at a loop vertex. Consequently
    a mixed gluon/photon box can require QCD and QED together. These facts come
    from the model and graph, not from absolute amplitude-order differences.
    Additional graph vetoes belong to graph construction.
    """

    species_orders: tuple[frozenset[str], ...]
    vertex_orders: frozenset[str]

    def __post_init__(self) -> None:
        species = tuple(frozenset(orders) for orders in self.species_orders)
        vertices = frozenset(self.vertex_orders)
        if any(
            not isinstance(name, str) or not name
            for orders in (*species, vertices)
            for name in orders
        ):
            raise ValueError("eligibility requires nonempty physical correction names")
        object.__setattr__(self, "species_orders", species)
        object.__setattr__(self, "vertex_orders", vertices)

    def admits(self, selected: frozenset[str]) -> bool:
        return bool(self.vertex_orders & selected) and all(
            bool(orders & selected) for orders in self.species_orders
        )


@dataclass(frozen=True, slots=True)
class LoopContribution:
    """An eligible physical graph or insertion with dependency-owned identity.

    Auxiliary evanescent states inherit their physical origin's eligibility.
    Different completions have different IDs; choosing an R2 branch happens after
    the catalogue is built and does not change the squared-order axis.
    """

    id: str
    order: CouplingOrders
    eligibility: LoopEligibility
    kind: Literal["loop", "uv", "model-r2", "evanescent"] = "loop"

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id:
            raise ValueError("loop contribution requires a graph/insertion ID")
        object.__setattr__(self, "order", canonical_order(self.order))
        if not isinstance(self.eligibility, LoopEligibility):
            raise ValueError("loop contribution requires physical eligibility facts")
        if self.kind not in {"loop", "uv", "model-r2", "evanescent"}:
            raise ValueError(f"unknown loop contribution kind {self.kind!r}")


@dataclass(frozen=True, slots=True)
class BornLoopPair:
    born: CouplingOrders
    loop: CouplingOrders

    def __post_init__(self) -> None:
        object.__setattr__(self, "born", canonical_order(self.born))
        object.__setattr__(self, "loop", canonical_order(self.loop))

    @property
    def squared(self) -> CouplingOrders:
        return squared_order(self.born, self.loop)


@dataclass(frozen=True, slots=True)
class VirtualOrderSector:
    order: CouplingOrders
    pairs: tuple[BornLoopPair, ...]


@dataclass(frozen=True, slots=True)
class LoopOrderPlan:
    correction_names: tuple[str, ...]
    born_orders: tuple[CouplingOrders, ...]
    contributions: tuple[LoopContribution, ...]
    sectors: tuple[VirtualOrderSector, ...]

    def sector(self, order: CouplingOrders) -> VirtualOrderSector:
        wanted = canonical_order(order)
        for sector in self.sectors:
            if sector.order == wanted:
                return sector
        raise ValueError(
            f"squared coupling order {wanted!r} is absent from the catalogue"
        )


def compile_order_plan(
    *,
    selected_corrections: Iterable[str],
    perturbative_orders: Mapping[str, int],
    born_orders: Iterable[CouplingOrders],
    contributions: Iterable[LoopContribution],
    allowed_pairs: Iterable[BornLoopPair],
) -> LoopOrderPlan:
    """Collect graph-selected contributions and explicit allowed interferences.

    The caller must retain structurally generated zero sectors. Numerical values,
    helicities and event momenta never enter this function. A missing Born catalogue
    means a loop-induced process; a vanishing Born evaluation does not.
    """
    names = tuple(sorted(set(selected_corrections)))
    if not names:
        raise ValueError("virtual generation requires at least one correction name")
    for name in names:
        if name not in perturbative_orders:
            raise ValueError(f"unknown loop correction {name!r}")
        if perturbative_orders[name] < 1:
            raise ValueError(f"model does not support one-loop correction {name!r}")
    born = tuple(sorted({canonical_order(order) for order in born_orders}))
    if not born:
        raise ValueError(
            "loop-induced processes without a Born amplitude are unsupported"
        )
    by_id: dict[str, LoopContribution] = {}
    for item in contributions:
        if not item.eligibility.admits(frozenset(names)):
            continue
        previous = by_id.get(item.id)
        if previous is not None:
            if previous != item:
                raise ValueError(f"conflicting loop contribution identity {item.id!r}")
        by_id[item.id] = item
    retained = tuple(by_id[key] for key in sorted(by_id))
    loop_orders = {item.order for item in retained}
    pairs = set(allowed_pairs)
    for pair in pairs:
        if pair.born not in born or pair.loop not in loop_orders:
            raise ValueError("allowed Born/loop pair references an unselected order")
    grouped: dict[CouplingOrders, list[BornLoopPair]] = {}
    for pair in sorted(pairs, key=lambda pair: (pair.born, pair.loop)):
        grouped.setdefault(pair.squared, []).append(pair)
    if not grouped:
        raise ValueError(
            "selected corrections generate no Born-loop interference sectors"
        )
    return LoopOrderPlan(
        names,
        born,
        retained,
        tuple(
            VirtualOrderSector(order, tuple(grouped[order]))
            for order in sorted(grouped)
        ),
    )
