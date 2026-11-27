# SPDX-License-Identifier: 0BSD
"""Coupling-order names between the tree engine and the loop engines.

The tree engine and its native DAG key every coupling order by
``models.order_names.order_key``; the loop engines read orders from the
model's NLO source with the declared spelling. ``CouplingOrderNames``
connects the two:

* user input (``max_coupling_orders``, ``loop_coupling_orders``,
  ``loop.model.loop_coupling_orders``) resolves once against the declared
  names, in any case (``resolve``);
* orders the tree engine returns (current and Born-root orders, minimal
  limits) translate back to the declared spelling (``orders``,
  ``mapping``) where the loop side takes them. Names the model does not
  declare (the engine's synthetic counterterm orders) pass unchanged.

Declared names one key apart are refused (``CouplingOrderNameError``, the
model compiler's rule). For a model that spells every order as its key
(every shipped model) every translation is the identity.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from pyamplicol.models.order_names import (
    CouplingOrderNameError,
    order_key,
    require_distinct_order_keys,
)


@dataclass(frozen=True)
class CouplingOrderNames:
    """The declared coupling-order names of one model (module docstring)."""

    declared: tuple[str, ...]
    by_key: Mapping[str, str]

    @property
    def identity(self) -> bool:
        """Every declared name is its own key (translation is the identity)."""

        return all(key == name for key, name in self.by_key.items())

    @classmethod
    def of(cls, names: Iterable[str]) -> CouplingOrderNames:
        declared = tuple(str(name) for name in names)
        return cls(declared, MappingProxyType(require_distinct_order_keys(declared)))

    @classmethod
    def of_model(cls, model: Any) -> CouplingOrderNames:
        """The names of a ``CompiledModel`` or of a tree-engine model holding
        one (``CompiledUFOModel``, ``Model.compiled``). Any other model is
        refused: its order spelling is unknown, and an identity translation
        would hand the tree keys to the loop side."""

        ir = getattr(model, "ir", None)
        if ir is None:
            ir = getattr(getattr(model, "compiled", None), "ir", None)
        if ir is None:
            raise TypeError(
                "coupling-order names need a compiled model (a CompiledModel or "
                f"a tree model holding one); got {type(model).__name__}"
            )
        return cls.of(order.name for order in ir.orders)

    def resolve(self, name: str, *, what: str = "coupling order") -> str:
        """The declared spelling of a user-supplied name (any case); a model
        that declares no coupling order leaves it as given."""

        if name in self.declared or not self.declared:
            return name
        found = self.by_key.get(order_key(name))
        if found is None:
            raise CouplingOrderNameError(
                f"{what} {name!r} is not a coupling order of the model "
                f"(declared: {list(self.declared)})"
            )
        return found

    def resolve_all(
        self, names: Iterable[str], *, what: str = "coupling order"
    ) -> tuple[str, ...]:
        return tuple(self.resolve(name, what=what) for name in names)

    def declared_name(self, name: str) -> str:
        """The declared spelling of a tree-engine key; other names unchanged."""

        return self.by_key.get(order_key(name), name)

    def orders(self, orders: Iterable[tuple[str, int]]) -> tuple[tuple[str, int], ...]:
        """Tree-engine orders (sorted, no zero power) with declared names,
        sorted; the same tuple when the translation is the identity."""

        if self.identity and isinstance(orders, tuple):
            return orders
        totals: dict[str, int] = {}
        for name, power in orders:
            declared = self.declared_name(name)
            totals[declared] = totals.get(declared, 0) + int(power)
        return tuple(sorted((name, power) for name, power in totals.items() if power))

    def mapping(self, values: Mapping[str, Any]) -> dict[str, Any]:
        """A name-keyed mapping (e.g. order limits) with declared names,
        sorted by name."""

        out: dict[str, Any] = {}
        for name, value in values.items():
            declared = self.declared_name(name)
            if declared in out:
                raise CouplingOrderNameError(
                    f"coupling order {declared!r} is given twice ({name!r})"
                )
            out[declared] = value
        return dict(sorted(out.items()))


__all__ = [
    "CouplingOrderNameError",
    "CouplingOrderNames",
    "order_key",
]
