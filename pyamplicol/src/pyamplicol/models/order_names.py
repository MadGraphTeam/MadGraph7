# SPDX-License-Identifier: 0BSD
"""Coupling-order names: the model's declared spellings and the tree key.

A UFO declares its coupling orders with any spelling (``QCD``, ``Qe``,
``NPprop``). The tree engine and its native DAG key every order by
``order_key`` (``str.upper``, MG5_aMC's convention): limits, vertex
increments and current orders alike. The loop engines read orders from the
model's NLO source with the declared spelling. ``CouplingOrderNames``
connects the two:

* a model whose declared names collide under ``order_key`` is refused
  (``CouplingOrderNameError``): the tree engine would merge them silently;
* user input (``max_coupling_orders``, ``loop_coupling_orders``,
  ``loop.model.loop_coupling_orders``) resolves once against the declared
  names, in any case (``resolve``);
* orders the tree engine returns (current and Born-root orders, minimal
  limits) translate back to the declared spelling (``orders``,
  ``mapping``) where the loop side takes them. Names the model does not
  declare (the engine's synthetic counterterm orders) pass unchanged.

For a model that spells every order as its key (every shipped model) every
translation is the identity.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any


class CouplingOrderNameError(ValueError):
    """Ambiguous or unknown coupling-order names."""


def order_key(name: object) -> str:
    """The tree engine's key of a coupling-order name."""

    return str(name).upper()


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
        by_key: dict[str, str] = {}
        for name in declared:
            other = by_key.setdefault(order_key(name), name)
            if other != name:
                raise CouplingOrderNameError(
                    f"the model declares the coupling orders {other!r} and "
                    f"{name!r}, which differ only in case: coupling-order names "
                    "are case-insensitive identifiers (the tree engine keys them "
                    f"as {order_key(name)!r}); rename one of them"
                )
        return cls(declared, MappingProxyType(by_key))

    @classmethod
    def of_model(cls, model: Any) -> CouplingOrderNames:
        """The names of a ``CompiledModel``, its ``CompiledModelIR`` or a
        tree-engine model holding one (``CompiledUFOModel``); identity
        translation for a model without declared orders."""

        ir = getattr(model, "ir", None)
        if ir is None:
            compiled = getattr(model, "compiled", None)
            ir = getattr(compiled, "ir", None)
        if ir is None and hasattr(model, "orders"):
            ir = model
        orders = () if ir is None else getattr(ir, "orders", ())
        return cls.of(order.name for order in orders)

    def resolve(self, name: str, *, what: str = "coupling order") -> str:
        """The declared spelling of a user-supplied name (any case)."""

        if name in self.declared:
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
