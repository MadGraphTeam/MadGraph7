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
onto candidate products of two adjoint tensors (``f*f``, ``Identity*Identity``,
``d*d``, ``d*f``, ``f*d`` over every pairing of the adjoint legs) that the tree
vocabulary admits on the vertex's legs (``tree_colour_source`` decides, so the
candidates follow the tree compiler, never a leg count): every overlap is a
full colour contraction by the engine's colour reduction (tau strings and
traces, ``ct_sites.ufo_colour_network``; SU(N_c) Fierz,
``colour.reduce_network``) at N_c = 3, exact in complex rationals; a maximal
independent set of candidates is kept (the Jacobi identity, and the SU(3)
relation of the d*d pairings, make the candidates dependent), the Gram system
is solved exactly and the residual norm must vanish. Otherwise the colour has
no vocabulary form and ``None`` is returned (the caller keeps it unlowered).

Only colours whose legs are adjoint or colour singlets are projected (the
candidates are adjoint tensors). The projection is at N_c = 3: a request that
grades counterterms by a formal N_c (a source colour continuation) does not
use it.
"""

from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction
from functools import lru_cache

from pyamplicol.color.connections import ExactColorCoefficient
from pyamplicol.color.contraction_types import NC

from .colour import (
    ColourNetwork,
    ColourProduct,
    ColourString,
    multiply,
    reduce_network,
    relabel,
)

# The candidate products, in the order the independent set is kept.
_PRODUCTS = (
    ("f", "f"),
    ("Identity", "Identity"),
    ("d", "d"),
    ("d", "f"),
    ("f", "d"),
)


def _conjugate(network: ColourNetwork) -> ColourNetwork:
    """The complex conjugate: tau is hermitian, so a trace or string reverses."""

    return tuple(
        ColourProduct(
            term.coefficient.conjugate(),
            tuple(
                ColourString(string.col, tuple(reversed(string.word)), string.row)
                for string in term.strings
            ),
            tuple(tuple(reversed(trace)) for trace in term.traces),
            term.nc_power,
            term.root2_power,
        )
        for term in network
    )


def _network(source: str, representations: tuple[int, ...]) -> ColourNetwork:
    from .ct_sites import ufo_colour_network

    return ufo_colour_network(source, representations)


@lru_cache(maxsize=1 << 12)
def _overlap(
    left: str, right: str, representations: tuple[int, ...]
) -> tuple[Fraction, Fraction]:
    """``<left|right>``: the full contraction of the conjugate of ``left``
    with ``right`` over every leg and summed index, at N_c = 3."""

    bra = _conjugate(_network(left, representations))
    ket = relabel(
        _network(right, representations),
        lambda label: ("ket", label) if label[0] != "leg" else label,
    )
    tensor = reduce_network(multiply(bra, ket), (), {})
    if any(term.tensor.open_strings or term.tensor.traces for term in tensor.terms):
        raise ValueError("a full colour contraction left an open index")
    total = ExactColorCoefficient(0)
    for term in tensor.terms:
        total = total + term.coefficient.scaled(Fraction(NC) ** term.nc_power)
    if tensor.normalization_power % 2:
        if total:
            raise ValueError("a colour overlap is irrational (odd power of sqrt 2)")
        return Fraction(0), Fraction(0)
    scale = Fraction(1, 2 ** (tensor.normalization_power // 2))
    return total.real * scale, total.imag * scale


def _candidates(
    representations: tuple[int, ...], multi_trace: bool
) -> tuple[str, ...]:
    """Products of two adjoint tensors over the pairings of the adjoint legs
    that the tree vocabulary admits on these legs."""

    from pyamplicol.models.ct_color_flow import (
        UnloweredCountertermColour,
        tree_colour_source,
    )

    adjoint = [leg for leg, value in enumerate(representations, 1) if value == 8]
    if len(adjoint) != 4:
        return ()
    first = adjoint[0]
    pairings = []
    for partner in adjoint[1:]:
        pair = (first, partner)
        other = tuple(leg for leg in adjoint if leg not in pair)
        pairings.append((pair, other))
    out = []
    # f*f first, then the identities, then the d-symbol forms: the kept
    # independent set prefers the forms every contact lowering carries.
    for left_head, right_head in _PRODUCTS:
        for (a, b), (c, d) in pairings:
            if left_head == "Identity":
                text = f"Identity({a},{b})*Identity({c},{d})"
            else:
                text = f"{left_head}(-1,{a},{b})*{right_head}(-1,{c},{d})"
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
        return None
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
