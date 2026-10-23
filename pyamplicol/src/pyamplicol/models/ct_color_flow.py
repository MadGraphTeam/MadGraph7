# SPDX-License-Identifier: 0BSD
"""Colour of counterterm vertices lowered into the tree recursion (design §6.1).

A counterterm (CT) vertex enters the tree recursion as an ordinary vertex
(``models.loading.lower_counterterm_vertices``), so its UFO colour must be one
the tree compiler lowers to colour flow: the singlet, ``Identity``, ``T``,
``f`` and the quartic contact ``f(-1,a,b)*f(-1,c,d)`` (or with the summed
index in any slot). Two UFO forms that only CT vertices use are rewritten
exactly into that vocabulary:

* ``Tr(a,b)`` between two adjoint legs is ``Identity(a,b)/2`` (generators
  normalised as ``tr(t^a t^b) = delta^{ab}/2``, the UFO convention);
* a numeric factor stays a numeric factor of the colour structure.

With ``multi_trace`` (the loop-request blocks of the counterterm closure,
``loops.v2.ct_closure``, which carry closed traces) three more forms lower:

* the d-symbol ``d(a,b,c)`` on three adjoint legs, whose colour flow is both
  orders of its legs with one sign;
* a product of two adjoint trilinear tensors sharing one summed index on
  four adjoint legs (``d*d``, ``d*f``, ``f*f``): the tree compiler's contact
  split carries the summed index on a U(N) auxiliary line, so ``d*d`` gains
  the U(1) term ``(2/N_c) delta delta`` (``d_product_u1_term``), which the
  caller subtracts;
* ``Identity(a,b)*Identity(c,d)`` (or ``Tr``) on four adjoint legs: the pair
  closes into a colour singlet carrying the closed trace ``tr(tau^a tau^b)``
  and the other two legs share one colour line.

Every other form raises ``UnloweredCountertermColour``: without
``multi_trace`` the forms above (the fixed-SU(3) four-gluon R2), a d-symbol
on a vertex with colour-singlet legs (the gluon-gluon-gluon-vector R2), ``Tr``
with three or more indices and any unknown tensor.

Callers close such rules as sites (``loops.v2.ct_sites``) instead; the
decision is per rule, never per entry, so a rule is either wholly in the
recursion or wholly a site.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction

_UFO_NAMESPACE = "UFO::{}::"
_FACTOR = re.compile(
    r"\s*(?:(?P<number>-?\d+(?:/\d+)?)|(?P<head>[A-Za-z]+)\((?P<args>[-\d,\s]*)\))\s*"
)


class UnloweredCountertermColour(NotImplementedError):
    """A CT colour structure the tree recursion cannot carry yet."""


@dataclass(frozen=True, slots=True)
class _Factor:
    head: str | None  # None for a numeric factor
    args: tuple[int, ...] = ()
    number: str = ""


def _factors(source: str) -> tuple[_Factor, ...]:
    factors = []
    for text in source.split("*"):
        match = _FACTOR.fullmatch(text)
        if match is None:
            raise UnloweredCountertermColour(
                f"counterterm colour {source!r} is not a product of UFO tensors"
            )
        if match["number"] is not None:
            factors.append(_Factor(None, number=match["number"]))
            continue
        args = tuple(int(value) for value in match["args"].split(",") if value.strip())
        factors.append(_Factor(match["head"], args))
    return tuple(factors)


def tree_colour_source(
    source: str, representations: Sequence[int], *, multi_trace: bool = False
) -> str:
    """The tree-model colour string of one CT entry colour ``source``.

    ``representations`` are the colour representations of the vertex legs
    (UFO leg i is ``representations[i - 1]``). The result is in the compiled
    payload's namespaced form (``UFO::{}::f(1,2,3)``). ``multi_trace`` admits
    the colours of the module docstring that need closed traces.
    """

    representations = tuple(int(value) for value in representations)
    source = source.strip()
    if source == "1":
        return "1"
    tensors = []
    numbers = []
    for factor in _factors(source):
        if factor.head is None:
            numbers.append(factor.number)
            continue
        legs = [index for index in factor.args if index > 0]
        if any(index > len(representations) for index in legs):
            raise UnloweredCountertermColour(
                f"counterterm colour {source!r} names a leg beyond the vertex"
            )
        if factor.head in ("Identity", "T", "f"):
            tensors.append(factor)
        elif factor.head == "d" and multi_trace:
            if any(representations[index - 1] != 8 for index in legs):
                raise UnloweredCountertermColour(
                    f"counterterm colour {source!r}: d-symbol needs adjoint legs"
                )
            tensors.append(factor)
        elif factor.head == "Tr" and len(factor.args) == 2:
            if any(representations[index - 1] != 8 for index in legs) or len(
                legs
            ) != len(factor.args):
                raise UnloweredCountertermColour(
                    f"counterterm colour {source!r}: Tr(a,b) needs two adjoint legs"
                )
            numbers.append("1/2")
            tensors.append(_Factor("Identity", factor.args))
        else:
            raise UnloweredCountertermColour(
                f"counterterm colour {source!r} needs the multi-trace colour state "
                "of design §6.1 (d-symbol or a trace of three or more generators)"
            )
    if multi_trace and any(factor.head == "d" for factor in tensors):
        adjoint = all(value == 8 for value in representations)
        # A d-symbol on three adjoint legs, alone (a trilinear vertex) or with
        # one colour-singlet leg (a contact split by ``compiler_contacts``).
        trilinear = len(representations) == 3 and adjoint
        singlet_contact = len(representations) == 4 and sorted(
            abs(value) for value in representations
        ) == [1, 8, 8, 8]
        if len(tensors) == 1 and not (trilinear or singlet_contact):
            raise UnloweredCountertermColour(
                f"counterterm colour {source!r}: no colour-flow form of a "
                "d-symbol on these legs"
            )
        if len(tensors) == 2 and not (adjoint and _adjoint_trilinears(tensors)):
            raise UnloweredCountertermColour(
                f"counterterm colour {source!r} is no product of two adjoint "
                "trilinear tensors sharing one summed index"
            )
    elif len(tensors) > 2 or (
        len(tensors) == 2
        and not _quartic_structure_constants(tensors)
        and not (multi_trace and _adjoint_identities(tensors, representations))
    ):
        raise UnloweredCountertermColour(
            f"counterterm colour {source!r} is a product the tree colour flow "
            "carries only as the multi-trace colour state of design §6.1"
        )
    parts = [*numbers]
    parts.extend(
        f"{_UFO_NAMESPACE}{factor.head}({','.join(map(str, factor.args))})"
        for factor in tensors
    )
    return "*".join(parts) if parts else "1"


def two_point_colour_factor(source: str, representations: Sequence[int]) -> Fraction:
    """The number ``c`` of a two-point CT colour ``c * delta(1,2)``.

    A two-point insertion keeps the colour of the line it sits on, so its
    colour must be proportional to the identity between its two legs:
    ``Identity(1,2)`` or, between adjoint legs, ``Tr(1,2) = delta/2``.
    """

    lowered = tree_colour_source(source, representations)
    identities = (f"{_UFO_NAMESPACE}Identity(1,2)", f"{_UFO_NAMESPACE}Identity(2,1)")
    factor = Fraction(1)
    identity = 0
    for text in lowered.split("*"):
        if text in identities:
            identity += 1
        elif text.lstrip("-").replace("/", "").isdigit():
            factor *= Fraction(text)
        else:
            identity = -1
            break
    if identity != int(any(abs(int(value)) != 1 for value in representations)):
        raise UnloweredCountertermColour(
            f"two-point counterterm colour {source!r} is not proportional to the "
            "identity of its line"
        )
    return factor


def d_product_u1_term(source: str) -> str | None:
    """``Identity(a,b)*Identity(c,d)`` of ``d(e,a,b)*d(e,c,d)``, else ``None``.

    The tree compiler's contact split of a product of two d-symbols sums the
    shared index over U(N) (``ct_color_flow`` docstring), which adds
    ``(2/N_c) delta^{ab} delta^{cd}`` to the SU(N) product; callers subtract
    it as this colour with ``-2/N_c`` times the product's coefficient. The
    pairs are listed in ascending order, as UFO models write them.
    """

    try:
        factors = _factors(source.strip())
    except UnloweredCountertermColour:
        return None
    if any(factor.head is None for factor in factors):
        return None
    if len(factors) != 2 or any(factor.head != "d" for factor in factors):
        return None
    if not _adjoint_trilinears(factors):
        return None
    pairs = sorted(
        tuple(sorted(index for index in factor.args if index > 0)) for factor in factors
    )
    return "*".join(f"Identity({a},{b})" for a, b in pairs)


def _adjoint_trilinears(tensors: Sequence[_Factor]) -> bool:
    """``f``/``d`` products sharing exactly one summed (negative) index."""

    if any(factor.head not in ("f", "d") for factor in tensors):
        return False
    left, right = ({index for index in factor.args if index < 0} for factor in tensors)
    return len(left) == len(right) == 1 and left == right


def _adjoint_identities(
    tensors: Sequence[_Factor], representations: Sequence[int]
) -> bool:
    """Two ``Identity`` factors pairing four adjoint legs."""

    if any(factor.head != "Identity" for factor in tensors):
        return False
    legs = sorted(index for factor in tensors for index in factor.args)
    return legs == [1, 2, 3, 4] and all(value == 8 for value in representations)


def _quartic_structure_constants(tensors: Sequence[_Factor]) -> bool:
    """``f*f`` sharing exactly one summed (negative) index: a quartic contact."""

    if any(factor.head != "f" for factor in tensors):
        return False
    left, right = ({index for index in factor.args if index < 0} for factor in tensors)
    return len(left) == len(right) == 1 and left == right


__all__ = [
    "UnloweredCountertermColour",
    "d_product_u1_term",
    "tree_colour_source",
    "two_point_colour_factor",
]
