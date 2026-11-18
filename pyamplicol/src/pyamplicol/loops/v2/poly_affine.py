# SPDX-License-Identifier: 0BSD
"""Loop-momentum polynomials and their exact affine frame maps (design §3.3, §10.2).

Open-loop payloads, segment kernels and family numerators are polynomials in
the loop momentum q = (q0, q1, q2, q3) with raw monomial coefficients and no
symmetry factors (MASTER_PLAN §2.12). They are stored densely in the NinjaRS
tensor layout: rank r holds C(r+4, 4) coefficients graded by degree, in the
exponent order of ``ninjars::fast::tensor_layout`` (``MonomialLayout.native``,
the same order the v2 reducer ingests), so a lower rank is a prefix of a
higher one and a family numerator is handed to NinjaRS without reordering.

Frames (design §3.3). A reading of a family ring puts the cut before its first
position; propagator i follows position i and D_i = (q + V_i)^2 - m_i^2 with
V_i the summed incoming momenta of positions 1..i. Two readings of one ring are
related by q = sign * q' - V(shift_mask) (``families.Relabelling``). ``relabel``
re-expands a numerator under that map in the coefficients' own arithmetic
(exact rationals, Decimal or binary64): N'(q') = N(sign * q' - V), with one
univariate Taylor shift per component, O(4 R C(R+4, 4)) multiply-adds.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import cached_property
from math import comb
from typing import Any

from .families import FamilyFrame, FamilyKey, Relabelling, relabelling, ring_readings

Exponents = tuple[int, int, int, int]

DIMENSION = 4  # components of the loop momentum


def size(rank: int) -> int:
    """Number of monomials of degree <= ``rank`` in four variables."""

    return comb(rank + DIMENSION, DIMENSION)


@dataclass(frozen=True)
class MonomialLayout:
    """Dense monomial order up to ``rank`` (graded; lower ranks are prefixes)."""

    exponents: tuple[Exponents, ...]

    def __post_init__(self) -> None:
        exponents = tuple(tuple(int(p) for p in item) for item in self.exponents)
        object.__setattr__(self, "exponents", exponents)
        degrees = [sum(item) for item in exponents]
        if (
            any(len(item) != DIMENSION or min(item) < 0 for item in exponents)
            or degrees != sorted(degrees)
            or len(set(exponents)) != len(exponents)
            or len(exponents) != size(degrees[-1] if degrees else 0)
        ):
            raise ValueError("a monomial layout lists every monomial once, by degree")

    @classmethod
    def native(cls, native: Any, rank: int) -> MonomialLayout:
        """The NinjaRS tensor order (``loop_native.v2_tensor_monomials``)."""

        return cls(tuple(map(tuple, native.v2_tensor_monomials(rank))))

    @property
    def rank(self) -> int:
        return sum(self.exponents[-1])

    @cached_property
    def index(self) -> dict[Exponents, int]:
        return {exponents: i for i, exponents in enumerate(self.exponents)}

    def check(self, rank: int) -> None:
        if rank > self.rank:
            raise ValueError(f"rank {rank} exceeds the layout's rank {self.rank}")

    def product_table(self, rank: int, degree: int) -> tuple[tuple[int, ...], ...]:
        """``table[m][n]``: index of monomial m times monomial n (the raise table)."""

        self.check(rank + degree)
        return self._products(rank, degree)

    def _products(self, rank: int, degree: int) -> tuple[tuple[int, ...], ...]:
        cache = self.__dict__.setdefault("_product_cache", {})
        table = cache.get((rank, degree))
        if table is None:
            index, exponents = self.index, self.exponents
            table = cache[(rank, degree)] = tuple(
                tuple(
                    index[tuple(a + b for a, b in zip(left, right, strict=True))]
                    for right in exponents[: size(degree)]
                )
                for left in exponents[: size(rank)]
            )
        return table


def monomial(q: Sequence[Any], exponents: Exponents, one: Any = 1) -> Any:
    value = one
    for component, power in zip(q, exponents, strict=True):
        for _ in range(power):
            value = value * component
    return value


def evaluate(
    coefficients: Sequence[Any], layout: MonomialLayout, q: Sequence[Any]
) -> Any:
    """sum_alpha c_alpha q^alpha (the dense prefix of ``layout``)."""

    total = 0
    for coefficient, exponents in zip(coefficients, layout.exponents, strict=False):
        total = total + coefficient * monomial(q, exponents)
    return total


Counter = Callable[[int], None]


def substitute(
    coefficients: Sequence[Any],
    layout: MonomialLayout,
    *,
    sign: int = 1,
    shift: Sequence[Any] | None = None,
    count: Counter | None = None,
) -> list[Any]:
    """Coefficients of N(sign * q + shift), exact in the coefficients' arithmetic.

    One univariate Taylor shift per component, then the sign of odd degrees.
    ``count`` receives the multiply-adds performed (shift powers excluded).
    """

    if sign not in (1, -1):
        raise ValueError("an affine frame map has sign +1 or -1")
    out = list(coefficients)
    rank = sum(layout.exponents[len(out) - 1]) if out else 0
    if size(rank) != len(out):
        raise ValueError("coefficients must fill whole ranks of the layout")
    if shift is not None:
        index, exponents = layout.index, layout.exponents
        for axis, value in enumerate(shift):
            powers = [1]
            for _ in range(rank):
                powers.append(powers[-1] * value)
            shifted = [0] * len(out)
            work = 0
            for position, coefficient in enumerate(out):
                source = exponents[position]
                top = source[axis]
                for power in range(top + 1):
                    target = (*source[:axis], power, *source[axis + 1 :])
                    factor = comb(top, power) * powers[top - power]
                    slot = index[target]
                    shifted[slot] = shifted[slot] + factor * coefficient
                    work += 1
            out = shifted
            if count is not None:
                count(work)
    if sign == -1:
        out = [
            -value if sum(exponents) % 2 else value
            for value, exponents in zip(out, layout.exponents, strict=False)
        ]
    return out


def leg_momentum(mask: int, momenta: Sequence[Sequence[Any]]) -> tuple[Any, ...]:
    """Summed incoming momentum of the legs in ``mask``."""

    total = [0] * DIMENSION
    for leg, momentum in enumerate(momenta):
        if mask >> leg & 1:
            total = [a + b for a, b in zip(total, momentum, strict=True)]
    return tuple(total)


def relabel(
    coefficients: Sequence[Any],
    layout: MonomialLayout,
    relabelled: Relabelling,
    momenta: Sequence[Sequence[Any]],
    *,
    count: Counter | None = None,
) -> list[Any]:
    """N'(q') = N(sign * q' - V(shift_mask)): the numerator in the new frame."""

    shift = tuple(-value for value in leg_momentum(relabelled.shift_mask, momenta))
    return substitute(
        coefficients, layout, sign=relabelled.sign, shift=shift, count=count
    )


def sequence_relabelling(source: FamilyKey, target: FamilyKey) -> Relabelling:
    """The map from the ring ``source`` (a (leg mask, mass id) sequence) to
    its reading ``target``.

    A ring with N >= 3 positions has distinct leg masks, so the reading is
    unique. A symmetric ring of N <= 2 positions has several readings with
    the same sequence; the first (identity first) is returned, which is the
    right one only when the numerator, not just the family, is symmetric.
    """

    for start, reflected in ring_readings(len(source)):
        frame = FamilyFrame(start, reflected)
        if frame.apply(source) == target:
            return relabelling(source, frame)
    raise ValueError(f"{target} is not a reading of the ring {source}")


def substitute_work(rank: int) -> int:
    """Multiply-adds of ``substitute`` with a shift, for a full rank."""

    # One axis visits alpha_mu + 1 targets per monomial alpha.
    return sum(
        (size(degree) - (size(degree - 1) if degree else 0)) * (degree + DIMENSION)
        for degree in range(rank + 1)
    )
