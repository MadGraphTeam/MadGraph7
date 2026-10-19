# SPDX-License-Identifier: 0BSD
"""Cold regular-orbit certificates for exact rectangular colour operators.

The tensor dependency proves relabellings. Existing permutation helpers provide
coordinates. This module only partitions current-column identities and exact
metric entries; it executes neither amplitudes nor Fourier transforms.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from itertools import permutations
from math import factorial
from typing import Any

from pyamplicol.color.symmetric_group import _relative_permutation
from pyamplicol.models.base import CouplingOrders

from .exact import require_exact_numbers
from .orders import canonical_order


@dataclass(frozen=True, slots=True)
class ColorTensorAlias:
    """Exactly equal tensors at one absolute order; all physical columns survive."""

    columns: tuple[int, ...]
    order: CouplingOrders
    tensor: Any = field(repr=False, compare=False)
    canonical: Any = field(repr=False, compare=False)
    unproved_reason: str | None = None


@dataclass(frozen=True, slots=True)
class RegularColorOrbit:
    """Alias IDs in lexicographic image-permutation order, identity first."""

    aliases: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class ColorOrbitPartition:
    aliases: tuple[ColorTensorAlias, ...]
    orbits: tuple[RegularColorOrbit, ...]
    residual_aliases: tuple[int, ...]

    @property
    def column_count(self) -> int:
        return sum(len(alias.columns) for alias in self.aliases)


@dataclass(frozen=True, slots=True)
class RectangularOrbitKernel:
    bank: str
    born_orbit: int
    loop_orbit: int
    values: tuple[Any, ...] = field(repr=False)


@dataclass(frozen=True, slots=True)
class RectangularDirectEntry:
    bank: str
    born_alias: int
    loop_alias: int
    value: Any = field(repr=False)


@dataclass(frozen=True, slots=True)
class RectangularColorCertificate:
    """Independent orbit maps and complete one-time metric coverage.

    Metric-bank names retain the caller's sector/operator/source signature.
    Their physical admission remains with that prepared observable owner. A
    non-equivariant orbit pair is retained entirely in the direct entries.
    """

    labels: tuple[Any, ...]
    born: ColorOrbitPartition
    loop: ColorOrbitPartition
    banks: tuple[str, ...]
    kernels: tuple[RectangularOrbitKernel, ...]
    residual_entries: tuple[RectangularDirectEntry, ...]
    nc: Any = field(default=None, repr=False)

    @property
    def degree(self) -> int:
        return len(self.labels)


def _same(left: Any, right: Any) -> bool:
    return (left - right).together() == 0


def _canonical_with_ports(tensor):
    try:
        canonical = tensor.canonize()
    except (ValueError, RuntimeError):
        return None, "dependency tensor canonicalization failed"
    if frozenset(canonical.axes) != frozenset(tensor.axes):
        return None, "dependency canonicalization changed typed external ports"
    expression = canonical.to_expression()
    require_exact_numbers(expression)
    return expression, None


def _partition(tensors, orders, labels):
    groups: dict[tuple[Any, CouplingOrders], int] = {}
    columns, representatives, canonicals, absolute_orders, reasons = [], [], [], [], []
    for column, (tensor, order) in enumerate(zip(tensors, orders, strict=True)):
        # Every open port, including fixed fundamental/dual ports, is checked.
        # The known dependency duality defect must not certify a false action.
        canonical, reason = _canonical_with_ports(tensor)
        key = (canonical, order)
        alias = groups.get(key) if canonical is not None else None
        if alias is None:
            alias = len(columns)
            columns.append([])
            representatives.append(tensor)
            canonicals.append(canonical)
            absolute_orders.append(order)
            reasons.append(reason)
            if canonical is not None:
                groups[key] = alias
        columns[alias].append(column)
    aliases = tuple(
        ColorTensorAlias(tuple(row), order, tensor, canonical, reason)
        for row, order, tensor, canonical, reason in zip(
            columns, absolute_orders, representatives, canonicals, reasons, strict=True
        )
    )
    group_order = factorial(len(labels))
    # A regular orbit needs this many distinct actual classes. This necessary
    # condition avoids enumerating a large group absent from the input domain.
    if group_order > len(aliases):
        return ColorOrbitPartition(aliases, (), tuple(range(len(aliases))))
    elements = tuple(permutations(range(len(labels))))
    covered: set[int] = set()
    orbits = []
    for identity, alias in enumerate(aliases):
        if identity in covered or alias.canonical is None:
            continue
        images = []
        for permutation in elements:
            rename = {labels[i]: labels[j] for i, j in enumerate(permutation)}
            try:
                expression, _ = _canonical_with_ports(
                    alias.tensor.rename_indices(rename)
                )
            except (ValueError, RuntimeError):
                break
            if expression is None:
                break
            image = groups.get((expression, alias.order))
            if image is None or image in covered:
                break
            images.append(image)
        if len(images) != group_order or len(set(images)) != group_order:
            continue
        if images[0] != identity:
            raise ValueError(
                "dependency tensor canonicalization changed under identity action"
            )
        orbits.append(RegularColorOrbit(tuple(images)))
        covered.update(images)
    return ColorOrbitPartition(
        aliases,
        tuple(orbits),
        tuple(i for i in range(len(aliases)) if i not in covered),
    )


def prepare_rectangular_color_certificate(
    born_tensors: Sequence[Any],
    loop_tensors: Sequence[Any],
    *,
    born_orders: Sequence[CouplingOrders],
    loop_orders: Sequence[CouplingOrders],
    labels: Sequence[Any],
    metrics: Mapping[str, Sequence[Sequence[Any]]],
    nc: Any = None,
) -> RectangularColorCertificate:
    """Certify actual tensors and ordered exact matrices under a common action.

    ``labels`` identifies dependency external indices, not guessed particle
    positions. Each selected port must have the same adjoint representation and
    direction. Every metric bank has the complete Born-by-loop original-column
    shape and must preserve exact aliases. Its entries may be complex rational
    functions of the explicitly supplied formal Nc variable. No equality is
    inferred from a substitution at Nc=3, nor are amplitudes summed here.
    """
    from symbolica import AtomType, Expression
    from symbolica.community import tensor as t

    born_tensors, loop_tensors = tuple(born_tensors), tuple(loop_tensors)
    born_orders = tuple(canonical_order(order) for order in born_orders)
    loop_orders = tuple(canonical_order(order) for order in loop_orders)
    labels = tuple(labels)
    if len(born_orders) != len(born_tensors) or len(loop_orders) != len(loop_tensors):
        raise ValueError(
            "absolute coupling orders must label every original colour column"
        )
    if len(set(labels)) != len(labels):
        raise ValueError("external action labels must be distinct")
    if nc is not None and (
        not isinstance(nc, Expression) or nc.get_type() != AtomType.Var
    ):
        raise TypeError("formal Nc must be a symbolic variable")
    tensors = (*born_tensors, *loop_tensors)
    interface = None
    representation = None
    for tensor in tensors:
        if not isinstance(tensor, t.TensorExpression):
            raise TypeError("colour orbit preparation requires dependency tensors")
        require_exact_numbers(tensor.to_expression())
        ports = tuple(tensor.axes)
        if interface is None:
            interface = frozenset(ports)
        elif frozenset(ports) != interface:
            raise ValueError("Born and loop tensor port types or directions differ")
        for label in labels:
            selected = [axis.representation for axis in ports if axis.index == label]
            if len(selected) != 1:
                raise ValueError(
                    "external action label is absent or ambiguous in a tensor"
                )
            port = selected[0]
            if port.name != t.Representation.coad(8).name:
                raise ValueError("external action requires adjoint ports")
            if representation is None:
                representation = port
            elif port != representation:
                raise ValueError(
                    "external action port representations or directions differ"
                )
    if labels and not tensors:
        raise ValueError("external action labels require an actual tensor interface")
    born = _partition(born_tensors, born_orders, labels)
    loop = _partition(loop_tensors, loop_orders, labels)
    banks, kernels, residual = [], [], []
    elements = None
    for bank, raw in metrics.items():
        if not isinstance(bank, str) or not bank:
            raise ValueError("metric banks require nonempty signature names")
        matrix = tuple(tuple(row) for row in raw)
        if len(matrix) != len(born_tensors) or any(
            len(row) != len(loop_tensors) for row in matrix
        ):
            raise ValueError("metric bank must have complete Born-by-loop column shape")
        allowed_symbols = set() if nc is None else {nc}
        for row in matrix:
            for value in row:
                require_exact_numbers(value)
                if set(value.get_all_symbols()) - allowed_symbols:
                    raise ValueError("metric bank has undeclared formal symbols")
        compressed = []
        for bra in born.aliases:
            row = []
            for ket in loop.aliases:
                value = matrix[bra.columns[0]][ket.columns[0]]
                if any(
                    not _same(matrix[i][j], value)
                    for i in bra.columns
                    for j in ket.columns
                ):
                    raise ValueError(
                        "metric bank distinguishes exact tensor/order aliases"
                    )
                row.append(value)
            compressed.append(row)
        banks.append(bank)
        covered: set[tuple[int, int]] = set()
        for a, left in enumerate(born.orbits):
            for b, right in enumerate(loop.orbits):
                if elements is None:
                    elements = tuple(permutations(range(len(labels))))
                coordinates = {value: index for index, value in enumerate(elements)}
                values = tuple(compressed[left.aliases[0]][j] for j in right.aliases)
                if not all(
                    _same(
                        compressed[i][j],
                        values[coordinates[_relative_permutation(g, h)]],
                    )
                    for g, i in zip(elements, left.aliases, strict=True)
                    for h, j in zip(elements, right.aliases, strict=True)
                ):
                    continue
                kernels.append(RectangularOrbitKernel(bank, a, b, values))
                for i in left.aliases:
                    for j in right.aliases:
                        if (i, j) in covered:
                            raise ValueError("regular colour orbit pairs overlap")
                        covered.add((i, j))
        residual.extend(
            RectangularDirectEntry(bank, i, j, value)
            for i, row in enumerate(compressed)
            for j, value in enumerate(row)
            if (i, j) not in covered
        )
    return RectangularColorCertificate(
        labels, born, loop, tuple(banks), tuple(kernels), tuple(residual), nc
    )
