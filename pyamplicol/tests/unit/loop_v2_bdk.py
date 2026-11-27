# SPDX-License-Identifier: 0BSD
"""BDK's five-parton colour data for oracle O14 (hep-ph/9409393).

The tree and one-loop colour structures of qbar q g g g as v2 literal
tensors and BDK's tree-loop colour brackets as Laurent polynomials in N_c.
Shared by ``test_loop_v2_colour.py`` (the exact colour sum) and
``test_loop_v2_observables.py`` (resolved seed columns).
"""

from __future__ import annotations

import itertools
import re
from fractions import Fraction

from pyamplicol.color.connections import ColorLeg
from pyamplicol.loops.v2.colour import (
    ColourProduct,
    reduce_network,
    spenso_colour_network,
)


def _spenso(expression: str) -> str:
    return re.sub(r"\b(f|t|g|chain|coad|cof|dind|in|out)\b", r"spenso::\1", expression)


def bdk_structures():
    """BDK's tree and one-loop colour structures of qbar(1) q(2) g3 g4 g5
    (hep-ph/9409393 eq. fiveqqdecomp), as v2 literal tensors.

    BDK's T^a is tau^a. Tree: (T^s3 T^s4 T^s5)_{i2}^{ibar1} for s in S_3.
    Loop: N_c (T T T) (A_{5;1}), Tr(T^x T^y) T^z (A_{5;3}, keyed by the
    unordered pair and z), Tr(T3 T4 T5) and Tr(T5 T4 T3) delta (A_{5;4},
    keyed by their cyclic class).
    """

    legs = [ColorLeg(1, 3), ColorLeg(2, -3), *(ColorLeg(a, 8) for a in (3, 4, 5))]
    external = {
        f"e::{name}": label for name, label in zip(("i1", "i2"), (1, 2), strict=True)
    }
    external.update({f"e::a{a}": a for a in (3, 4, 5)})

    def t(a):
        return f"t(coad(8,e::a{a}),in,out)"

    def tensor(expression, nc=0):
        # BDK normalise tr(T^a T^b) = delta^ab: T = tau, no 1/sqrt2 factors.
        network = tuple(
            ColourProduct(p.coefficient, p.strings, p.traces, p.nc_power + nc)
            for p in spenso_colour_network(_spenso(expression))
        )
        return reduce_network(network, legs, external)

    def chain(*gluons):
        body = ",".join(t(a) for a in gluons)
        return f"chain(cof(3,e::i2),dind(cof(3,e::i1)){',' if body else ''}{body})"

    def trace(*gluons):
        return "trace(cof(3),cyclic(" + ",".join(t(a) for a in gluons) + "))"

    orders = list(itertools.permutations((3, 4, 5)))
    tree = {s: tensor(chain(*s)) for s in orders}
    loop = {("1", s): tensor(chain(*s), nc=1) for s in orders}
    for z in (3, 4, 5):
        x, y = (a for a in (3, 4, 5) if a != z)
        loop["3", frozenset((x, y)), z] = tensor(trace(x, y) + "*" + chain(z))
    for s in ((3, 4, 5), (5, 4, 3)):
        loop["4", s] = tensor(trace(*s) + "*" + chain())
    return tree, loop


def _cyclic(s):
    return min(s[i:] + s[:i] for i in range(3))


def laurent_times(cell, factor):
    out = {}
    for p, a in cell.items():
        for q, b in factor.items():
            out[p + q] = out.get(p + q, 0) + a * b
    return {p: v for p, v in out.items() if v}


def bdk_bracket(s):
    """BDK's bracket of tree ordering ``s`` (hep-ph/9409393 eq.
    NLOCrossSection): one-loop structure key -> Laurent polynomial in N_c;
    sum_colours A_tree^* A_loop = (N^2-1)/N sum_s A_tree(s)^* sum_key
    bracket(s)[key] A_key."""

    def laurent(*pairs):
        return {power: Fraction(value) for power, value in pairs}

    leading = laurent((4, 1), (2, -2), (0, 1))  # (N^2-1)^2
    first = laurent((2, -1), (0, 1))  # -(N^2-1)
    reversed_ = laurent((2, 1), (0, 1))  # N^2+1
    trace = laurent((2, 1), (0, -2))  # N^2-2
    s3, s4, s5 = s
    bracket: dict = {}

    def add(key, cell):
        total = dict(bracket.get(key, {}))
        for p, v in cell.items():
            total[p] = total.get(p, 0) + v
        bracket[key] = {p: v for p, v in total.items() if v}

    add(("1", (s3, s4, s5)), leading)
    add(("1", (s4, s3, s5)), first)
    add(("1", (s3, s5, s4)), first)
    add(("3", frozenset((s4, s5)), s3), laurent_times(first, {0: -1}))
    add(("3", frozenset((s3, s4)), s5), laurent_times(first, {0: -1}))
    add(("1", (s5, s4, s3)), reversed_)
    add(("4", (3, 4, 5) if _cyclic(s) == (3, 4, 5) else (5, 4, 3)), trace)
    add(("1", (s4, s5, s3)), {0: 1})
    add(("1", (s5, s3, s4)), {0: 1})
    add(("3", frozenset((s5, s3)), s4), {0: -1})
    rev = (s5, s4, s3)
    add(("4", (3, 4, 5) if _cyclic(rev) == (3, 4, 5) else (5, 4, 3)), {0: -2})
    return bracket


# (N^2-1)/N, BDK's overall colour factor of the tree-loop interference.
BDK_PREFACTOR = {1: Fraction(1), -1: Fraction(-1)}
