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

Every other form raises ``UnloweredCountertermColour``:

* the d-symbol (``d(a,b,c)``, ``d*d``, ``d*f``) and products of two
  ``Identity``/``Tr`` factors on four coloured legs, i.e. the fixed-SU(3)
  four-gluon R2 and the gluon-gluon-gluon-vector R2. Their colour-flow form
  has a closed trace beside the open string of a current: the multi-trace
  colour state of design §6.1, which the tree recursion does not carry yet;
* ``Tr`` with three or more indices, and any unknown tensor.

Callers close such rules as sites (``loops.v2.ct_sites``) instead; the
decision is per rule, never per entry, so a rule is either wholly in the
recursion or wholly a site.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

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


def tree_colour_source(source: str, representations: Sequence[int]) -> str:
    """The tree-model colour string of one CT entry colour ``source``.

    ``representations`` are the colour representations of the vertex legs
    (UFO leg i is ``representations[i - 1]``). The result is in the compiled
    payload's namespaced form (``UFO::{}::f(1,2,3)``).
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
    if len(tensors) > 2 or (
        len(tensors) == 2 and not _quartic_structure_constants(tensors)
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


def _quartic_structure_constants(tensors: Sequence[_Factor]) -> bool:
    """``f*f`` sharing exactly one summed (negative) index: a quartic contact."""

    if any(factor.head != "f" for factor in tensors):
        return False
    left, right = ({index for index in factor.args if index < 0} for factor in tensors)
    return len(left) == len(right) == 1 and left == right


__all__ = ["UnloweredCountertermColour", "tree_colour_source"]
