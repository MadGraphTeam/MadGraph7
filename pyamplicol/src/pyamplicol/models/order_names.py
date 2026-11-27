# SPDX-License-Identifier: 0BSD
"""Coupling-order names: the tree engine's key and the model's collision rule.

A UFO declares its coupling orders with any spelling (``QCD``, ``Qe``,
``NPprop``). The tree engine and its native DAG key every order by
``order_key`` (``str.upper``, MG5_aMC's convention): limits, vertex
increments and current orders alike. A model whose declared names collide
under ``order_key`` is refused when its IR is built
(``require_distinct_order_keys``): the tree engine would merge them silently.

This module is a model-compiler source (it decides which models compile),
so it holds only that rule. The translation between the declared spelling,
which the loop engines read, and the tree key lives with the loop side
(``pyamplicol.loops.order_names``), outside the compiled-model identity.
"""

from __future__ import annotations

from collections.abc import Iterable


class CouplingOrderNameError(ValueError):
    """Ambiguous or unknown coupling-order names."""


def order_key(name: object) -> str:
    """The tree engine's key of a coupling-order name."""

    return str(name).upper()


def require_distinct_order_keys(names: Iterable[str]) -> dict[str, str]:
    """``order_key`` -> declared name of ``names``; refused
    (``CouplingOrderNameError``) when two declared names share a key."""

    by_key: dict[str, str] = {}
    for name in (str(name) for name in names):
        other = by_key.setdefault(order_key(name), name)
        if other != name:
            raise CouplingOrderNameError(
                f"the model declares the coupling orders {other!r} and "
                f"{name!r}, which differ only in case: coupling-order names "
                "are case-insensitive identifiers (the tree engine keys them "
                f"as {order_key(name)!r}); rename one of them"
            )
    return by_key


__all__ = [
    "CouplingOrderNameError",
    "order_key",
    "require_distinct_order_keys",
]
