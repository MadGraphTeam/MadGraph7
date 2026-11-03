# SPDX-License-Identifier: 0BSD
"""Counterterm colours projected exactly onto the tree colour vocabulary.

A counterterm vertex enters the tree recursion as an ordinary vertex, so its
UFO colour must be one ``models.ct_color_flow.tree_colour_source`` lowers. A
UFO may write a CT colour outside that vocabulary although the colour is a
combination of vocabulary colours, e.g. the trace of four adjoint generators
``f(1,-1,-2)*f(2,-2,-3)*f(3,-3,-4)*f(4,-4,-1)`` (HC_NLO_X0's five-point
[G G G G X0] R2): at the fixed SU(3) of UFO colours it is a sum of
``f*f`` and ``Identity*Identity`` pairings of its four adjoint legs.

``vocabulary_colours`` finds that combination by exact orthogonal projection
onto candidate colours built from the vocabulary's primitive adjoint tensors
(``Identity``, ``f``, ``d``): every product of at most two of them (the
vocabulary's colours have at most two factors) whose open indices are the
adjoint legs, each once, with at most one summed index -- one primitive on
all legs (``f(a,b,c)``), two sharing a summed index (``f(-1,a,b)*d(-1,c,d)``)
or two disjoint ones (``Identity(a,b)*Identity(c,d)``) -- that the tree
vocabulary admits on the vertex's legs (``tree_colour_source`` decides, so the
candidates follow the primitives' arities and the tree compiler, never a leg
count): every colour is
reduced by the engine's colour algebra to tau traces of its legs
(``ct_sites.ufo_colour_network``, SU(N_c) Fierz in ``colour.reduce_network``)
and every overlap is their full contraction (``colour.contract_colour``) at
N_c = 3, exact in complex rationals; a maximal independent set of candidates
is kept (the Jacobi identity, and the SU(3) relation of the d*d pairings,
make the candidates dependent), the Gram system is solved exactly and the
residual norm must vanish. Otherwise the colour has no vocabulary form and
``None`` is returned (the caller keeps it unlowered).

Only colours whose legs are adjoint or colour singlets are projected (the
candidates are adjoint tensors). The projection is at N_c = 3: a request that
grades counterterms by a formal N_c (a source colour continuation) does not
use it.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterator, Sequence
from fractions import Fraction
from functools import lru_cache

from pyamplicol.color.connections import (
    ColorLeg,
    ConnectedColorTensor,
    ExactColorCoefficient,
)
from pyamplicol.color.contraction_types import NC

from .colour import ColourNetwork, contract_colour, reduce_network

# The vocabulary's primitive adjoint tensors and their numbers of indices.
_ARITY = {"f": 3, "Identity": 2, "d": 3}
# Products of two primitives, in the order the independent set is kept: f*f
# first, then the identities, then the d-symbol forms (the kept set prefers
# the forms every contact lowering carries). The first factor holds the first
# adjoint leg, so (d, f) and (f, d) are different candidates.
_PRODUCTS = (
    ("f", "f"),
    ("Identity", "Identity"),
    ("d", "d"),
    ("d", "f"),
    ("f", "d"),
    ("Identity", "f"),
    ("f", "Identity"),
    ("Identity", "d"),
    ("d", "Identity"),
)


def _network(source: str, representations: tuple[int, ...]) -> ColourNetwork:
    from .ct_sites import ufo_colour_network

    return ufo_colour_network(source, representations)


@lru_cache(maxsize=1 << 12)
def _reduced(source: str, representations: tuple[int, ...]) -> ConnectedColorTensor:
    """``source`` on its adjoint legs as literal tau traces (exact in N_c)."""

    adjoint = [leg for leg, value in enumerate(representations, 1) if value == 8]
    return reduce_network(
        _network(source, representations),
        tuple(ColorLeg(leg, 8) for leg in adjoint),
        {("leg", leg): leg for leg in adjoint},
    )


@lru_cache(maxsize=1 << 12)
def _overlap(
    left: str, right: str, representations: tuple[int, ...]
) -> tuple[Fraction, Fraction]:
    """``<left|right>``: the full contraction of the conjugate of ``left``
    with ``right`` over every leg (``colour.contract_colour``) at N_c = 3."""

    laurent = contract_colour(
        _reduced(left, representations), _reduced(right, representations)
    )
    total = ExactColorCoefficient(0)
    for power, value in laurent.items():
        total = total + value.scaled(Fraction(NC) ** power)
    return total.real, total.imag


def _forms(adjoint: Sequence[int]) -> Iterator[str]:
    """The candidate colours on the adjoint legs (module docstring): one
    primitive on every leg, or two primitives sharing one summed index or
    none. ``Identity`` never shares a summed index (``Identity(-1,a)*X(-1,..)``
    is ``X(a,..)``), and no pair shares two (a multiple of an ``Identity``)."""

    if not adjoint:
        return
    first, rest = adjoint[0], tuple(adjoint[1:])
    for head, arity in _ARITY.items():
        if arity == len(adjoint):
            yield f"{head}({','.join(map(str, adjoint))})"
    for left, right in _PRODUCTS:
        for summed in (1, 0):
            if summed and "Identity" in (left, right):
                continue
            size = _ARITY[left] - summed
            if size + _ARITY[right] - summed != len(adjoint):
                continue
            prefix = ("-1",) if summed else ()
            for others in itertools.combinations(rest, size - 1):
                lhs = (*prefix, *map(str, (first, *others)))
                rhs = (*prefix, *(str(leg) for leg in rest if leg not in others))
                yield f"{left}({','.join(lhs)})*{right}({','.join(rhs)})"


def _candidates(
    representations: tuple[int, ...], multi_trace: bool
) -> tuple[str, ...]:
    """The candidate colours (``_forms``) the tree vocabulary admits on
    these legs."""

    from pyamplicol.models.ct_color_flow import (
        UnloweredCountertermColour,
        tree_colour_source,
    )

    adjoint = [leg for leg, value in enumerate(representations, 1) if value == 8]
    out = []
    for text in _forms(adjoint):
        try:
            tree_colour_source(text, representations, multi_trace=multi_trace)
        except UnloweredCountertermColour:
            continue
        out.append(text)
    return tuple(out)


def _solve(
    gram: list[list[Fraction]], rhs: list[Fraction]
) -> list[Fraction]:
    size = len(rhs)
    rows = [list(gram[i]) + [rhs[i]] for i in range(size)]
    for column in range(size):
        pivot = next(r for r in range(column, size) if rows[r][column] != 0)
        rows[column], rows[pivot] = rows[pivot], rows[column]
        head = rows[column][column]
        rows[column] = [value / head for value in rows[column]]
        for r in range(size):
            if r != column and rows[r][column] != 0:
                factor = rows[r][column]
                rows[r] = [a - factor * b for a, b in zip(rows[r], rows[column])]
    return [rows[i][size] for i in range(size)]


def _independent(
    candidates: Sequence[str], representations: tuple[int, ...]
) -> tuple[str, ...]:
    """A maximal linearly independent subset, greedily in candidate order
    (exact rank test of the Gram matrix)."""

    kept: list[str] = []
    for candidate in candidates:
        trial = [*kept, candidate]
        gram = [[_overlap(a, b, representations)[0] for b in trial] for a in trial]
        if _rank(gram) == len(trial):
            kept.append(candidate)
    return tuple(kept)


def _rank(matrix: list[list[Fraction]]) -> int:
    rows = [list(row) for row in matrix]
    rank = 0
    columns = len(rows[0]) if rows else 0
    for column in range(columns):
        pivot = next(
            (r for r in range(rank, len(rows)) if rows[r][column] != 0), None
        )
        if pivot is None:
            continue
        rows[rank], rows[pivot] = rows[pivot], rows[rank]
        for r in range(len(rows)):
            if r != rank and rows[r][column] != 0:
                factor = rows[r][column] / rows[rank][column]
                rows[r] = [a - factor * b for a, b in zip(rows[r], rows[rank])]
        rank += 1
    return rank


@lru_cache(maxsize=1 << 10)
def vocabulary_colours(
    source: str, representations: tuple[int, ...], *, multi_trace: bool = True
) -> tuple[tuple[str, ExactColorCoefficient], ...] | None:
    """``source`` as ``((vocabulary colour, exact factor), ...)`` at N_c = 3,
    or ``None`` when it is no combination of the vocabulary's adjoint pairings
    on these legs (module docstring). ``representations`` are the colour
    representations of the vertex legs (UFO leg i is entry i - 1)."""

    representations = tuple(int(value) for value in representations)
    if any(value not in (1, 8) for value in representations):
        return None
    candidates = _candidates(representations, multi_trace)
    if not candidates:
        return None
    try:
        norm = _overlap(source, source, representations)
    except (NotImplementedError, ValueError):
        return None  # not a colour tensor of these legs (open indices, ...)
    basis = _independent(candidates, representations)
    gram = [[_overlap(a, b, representations)[0] for b in basis] for a in basis]
    projections = [_overlap(b, source, representations) for b in basis]
    real = _solve(gram, [value[0] for value in projections])
    imag = _solve(gram, [value[1] for value in projections])
    # || source - P source ||^2 = <s|s> - sum_i conj(v_i) x_i (Gram real).
    residual_real = norm[0] - sum(
        v[0] * x + v[1] * y for v, x, y in zip(projections, real, imag)
    )
    residual_imag = norm[1] - sum(
        v[0] * y - v[1] * x for v, x, y in zip(projections, real, imag)
    )
    if residual_real != 0 or residual_imag != 0:
        return None
    return tuple(
        (colour, ExactColorCoefficient(x, y))
        for colour, x, y in zip(basis, real, imag)
        if x or y
    )


__all__ = ["vocabulary_colours"]
