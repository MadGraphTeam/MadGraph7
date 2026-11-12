# SPDX-License-Identifier: 0BSD
"""Conservative request-order bounds before local source tensor lowering.

This only proves that an individual nonnegative coupling order cannot occur in
an admitted amplitude. Physical loop correction eligibility and CT/external-field
matching retain their existing graph/source rules.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from .orders import canonical_order


def _bounds(values, names):
    if values is None:
        return ()
    if not isinstance(values, Mapping):
        raise TypeError("coupling order bounds must be a mapping")
    result = []
    for name, value in values.items():
        if name not in names:
            raise ValueError(f"unknown request coupling order {name!r}")
        if type(value) is int:
            lower, upper = value, value
        elif isinstance(value, tuple) and len(value) == 2:
            lower, upper = value
        else:
            raise TypeError(
                "coupling bounds require integers or inclusive range tuples"
            )
        if type(lower) is not int or lower < 0:
            raise ValueError("coupling lower bounds must be nonnegative integers")
        if upper is not None and (type(upper) is not int or upper < lower):
            raise ValueError("coupling upper bounds must be integers >= lower or None")
        result.append((name, lower, upper))
    return tuple(sorted(result))


@dataclass(frozen=True, slots=True)
class ExcludedSourceBasis:
    key: tuple[Any, ...]
    order: tuple[tuple[str, int], ...]
    exceeding_order: str
    power: int
    ceiling: int


@dataclass(frozen=True, slots=True)
class LoopRequestScope:
    raw: Any = field(repr=False, compare=False)
    order_names: tuple[str, ...]
    born_bounds: tuple[tuple[str, int, int | None], ...]
    loop_bounds: tuple[tuple[str, int, int | None], ...]
    ordinary_ceilings: tuple[tuple[str, int], ...]
    counterterm_ceilings: tuple[tuple[str, int], ...]
    admitted_keys: frozenset[tuple[Any, ...]]
    excluded: tuple[ExcludedSourceBasis, ...]

    @property
    def born_order_bounds(self):
        return {name: (lower, upper) for name, lower, upper in self.born_bounds}

    @property
    def loop_order_bounds(self):
        return {name: (lower, upper) for name, lower, upper in self.loop_bounds}

    def require_raw(self, raw):
        if raw is not self.raw:
            raise ValueError("request scope belongs to a different raw model instance")

    def require_bounds(self, born, loop):
        if (
            _bounds(born, self.order_names) != self.born_bounds
            or _bounds(loop, self.order_names) != self.loop_bounds
        ):
            raise ValueError("process bounds differ from the prepared source request")

    def require_local_counterterms(self, prepared):
        """Prove selected local-source coverage, separately from particle fields."""
        self.require_raw(prepared.source.raw)
        expected = {
            key for key in self.admitted_keys if key[0] == "counterterm_vertices"
        }
        actual = [
            entry.key for rule in prepared.rules.values() for entry in rule.entries
        ]
        if (
            len(actual) != len(set(actual))
            or set(actual) != expected
            or prepared.unavailable_entries
        ):
            raise ValueError("request is missing or has extra source counterterm bases")


def prepare_loop_request_scope(raw, *, born_order_bounds=None, loop_order_bounds=None):
    """Retain every unproved entry and every unbounded order coordinate.

    Integer bounds mean exact powers and tuples mean inclusive ranges, matching
    HepKit. Lower bounds cannot exclude a local factor. Ordinary factors use the
    union ceiling of Born and loop requests; a coordinate absent/unbounded in
    either request is unbounded in that union. Local CTs use loop ceilings only.
    Every source coupling must have declared nonnegative integral orders before
    any exclusion is permitted, including couplings unused by a current vertex.
    """
    from .model import _reference

    names = tuple(record["name"] for record in raw.collections["coupling_orders"])
    if len(names) != len(set(names)) or any(
        not isinstance(n, str) or not n for n in names
    ):
        raise ValueError("source coupling order names must be nonempty and unique")
    born, loop = _bounds(born_order_bounds, names), _bounds(loop_order_bounds, names)
    born_upper = {name: upper for name, _, upper in born if upper is not None}
    loop_upper = {name: upper for name, _, upper in loop if upper is not None}
    ordinary = {
        name: max(born_upper[name], loop_upper[name])
        for name in born_upper.keys() & loop_upper.keys()
    }
    coupling_orders = {}
    for record in raw.collections["couplings"]:
        value = record["fields"].get("order")
        if not isinstance(value, Mapping) or value.get("kind") != "mapping":
            raise ValueError("source coupling is missing explicit order data")
        order = canonical_order(value["entries"])
        if any(name not in names for name, _ in value["entries"]):
            raise ValueError("source coupling refers to an undeclared order")
        coupling_orders[record["id"]] = order
    admitted, excluded, seen = set(), [], set()
    for collection in ("vertices", "counterterm_vertices"):
        ceilings = ordinary if collection == "vertices" else loop_upper
        for record in raw.collections[collection]:
            for source_key, reference in record["fields"]["couplings"]["entries"]:
                indices = tuple(source_key["items"])
                size = 2 if collection == "vertices" else 3
                if len(indices) != size or any(
                    type(i) is not int or i < 0 for i in indices
                ):
                    raise ValueError("source coupling has invalid local basis indices")
                key = (collection, record["id"], *indices)
                if collection == "vertices":
                    key += (None,)
                if key in seen:
                    raise ValueError("source contains duplicate local basis keys")
                seen.add(key)
                coupling = _reference(raw, reference, "couplings")
                order = coupling_orders[coupling["id"]]
                violations = [
                    (name, power, ceilings[name])
                    for name, power in order
                    if name in ceilings and power > ceilings[name]
                ]
                if violations:
                    name, power, ceiling = violations[0]
                    excluded.append(
                        ExcludedSourceBasis(key, order, name, power, ceiling)
                    )
                else:
                    admitted.add(key)
    return LoopRequestScope(
        raw,
        names,
        born,
        loop,
        tuple(sorted(ordinary.items())),
        tuple(sorted(loop_upper.items())),
        frozenset(admitted),
        tuple(excluded),
    )
