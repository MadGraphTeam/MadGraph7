# SPDX-License-Identifier: 0BSD
"""Counterterm colours outside the tree vocabulary, projected exactly onto
the vocabulary's adjoint pairings at N_c = 3 (``loops.v2.ct_colour_projection``;
MF-HC05: HC_NLO_X0's five-point [G G G G X0] R2 carries the trace of four
adjoint generators, ``f*f*f*f``).

The projection uses the engine's colour reduction; these tests check it
against explicit SU(3) tensors (numpy, independent of that reduction) on
the HC ring, on other multi-trace forms (fundamental traces, a d-symbol sum)
and on colours without a vocabulary form, and check how the counterterm
lowering consumes it (``TreeVocabulary.colour_terms``, ``rule_rows``).
"""

from __future__ import annotations

import ast
import itertools
from fractions import Fraction
from types import SimpleNamespace

import numpy as np
import pytest

RING = "f(1,-1,-2)*f(2,-2,-3)*f(3,-3,-4)*f(4,-4,-1)"
FIVE = (8, 8, 8, 8, 1)


def _su3():
    """SU(3) f^abc, d^abc and T^a = lambda^a / 2 (Gell-Mann)."""

    lam = np.zeros((8, 3, 3), dtype=complex)
    lam[0][0, 1] = lam[0][1, 0] = 1
    lam[1][0, 1], lam[1][1, 0] = -1j, 1j
    lam[2][0, 0], lam[2][1, 1] = 1, -1
    lam[3][0, 2] = lam[3][2, 0] = 1
    lam[4][0, 2], lam[4][2, 0] = -1j, 1j
    lam[5][1, 2] = lam[5][2, 1] = 1
    lam[6][1, 2], lam[6][2, 1] = -1j, 1j
    lam[7] = np.diag([1, 1, -2]) / np.sqrt(3)
    t = lam / 2
    products = np.einsum("aij,bjk->abik", t, t)
    commutator = products - products.transpose(1, 0, 2, 3)
    anticommutator = products + products.transpose(1, 0, 2, 3)
    f = (-2j * np.einsum("abij,cji->abc", commutator, t)).real
    d = (2 * np.einsum("abij,cji->abc", anticommutator, t)).real
    return f, d, t


F, D, T = _su3()
_EYE = np.eye(8)


def _terms(node) -> list[tuple[Fraction, list[tuple[str, list[int]]]]]:
    """A UFO colour expression as a sum of (coefficient, tensor factors)."""

    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
        sign = 1 if isinstance(node.op, ast.Add) else -1
        right = [(sign * c, f) for c, f in _terms(node.right)]
        return _terms(node.left) + right
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
        return [
            (a * b, f + g) for a, f in _terms(node.left) for b, g in _terms(node.right)
        ]
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        ((denominator, none),) = _terms(node.right)
        assert not none
        return [(c / denominator, f) for c, f in _terms(node.left)]
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return [(-c, f) for c, f in _terms(node.operand)]
    if isinstance(node, ast.Constant):
        return [(Fraction(node.value), [])]
    assert isinstance(node, ast.Call)
    return [(Fraction(1), [(node.func.id, [ast.literal_eval(a) for a in node.args])])]


def _factor_tensor(head: str, arity: int) -> np.ndarray:
    if head == "Tr":
        letters = "abcdefgh"[:arity]
        rows = "ijklmnop"
        spec = ",".join(
            f"{letters[k]}{rows[k]}{rows[(k + 1) % arity]}" for k in range(arity)
        )
        return np.einsum(f"{spec}->{letters}", *([T] * arity))
    return {"f": F, "d": D, "Identity": _EYE}[head]


def _tensor(source: str, legs: int = 4) -> np.ndarray:
    """A UFO colour string over adjoint legs 1..legs (summed indices < 0) as
    an explicit SU(3) tensor."""

    total = np.zeros((8,) * legs, dtype=complex)
    for coefficient, factors in _terms(ast.parse(source, mode="eval").body):
        labels: dict[int, str] = {}
        for _head, args in factors:
            for value in args:
                labels.setdefault(value, chr(65 + len(labels)))
        out = "".join(labels[leg] for leg in range(1, legs + 1))
        spec = ",".join("".join(labels[v] for v in args) for _h, args in factors)
        operands = [_factor_tensor(head, len(args)) for head, args in factors]
        total += float(coefficient) * np.einsum(f"{spec}->{out}", *operands)
    return total


def _recombined(found, legs: int = 4) -> np.ndarray:
    return sum(
        (complex(float(x.real), float(x.imag)) * _tensor(colour, legs) for colour, x in found),
        np.zeros((8,) * legs, dtype=complex),
    )


def _ring(a, b, c, d) -> str:
    return f"f({a},-1,-2)*f({b},-2,-3)*f({c},-3,-4)*f({d},-4,-1)"


@pytest.fixture(scope="module")
def projection():
    pytest.importorskip("symbolica")
    from pyamplicol.loops.v2 import ct_colour_projection

    return ct_colour_projection


@pytest.mark.loop_fast
def test_the_hc_ring_is_a_combination_of_f_f_and_identity_pairings(projection):
    """Tr_adj(F1 F2 F3 F4) at SU(3) on four adjoint legs and a singlet: the
    exact factors, and their sum is the ring as an explicit SU(3) tensor."""

    found = projection.vocabulary_colours(RING, FIVE)
    assert [(c, x.real, x.imag) for c, x in found] == [
        ("f(-1,1,2)*f(-1,3,4)", Fraction(-1), 0),
        ("f(-1,1,3)*f(-1,2,4)", Fraction(1, 2), 0),
        ("Identity(1,2)*Identity(3,4)", Fraction(3, 4), 0),
        ("Identity(1,3)*Identity(2,4)", Fraction(3, 4), 0),
        ("Identity(1,4)*Identity(2,3)", Fraction(3, 4), 0),
    ]
    for legs in itertools.permutations((1, 2, 3, 4)):
        found = projection.vocabulary_colours(_ring(*legs), FIVE)
        assert found is not None
        assert np.abs(_recombined(found) - _tensor(_ring(*legs))).max() < 1e-12


@pytest.mark.loop_fast
@pytest.mark.parametrize(
    "source",
    [
        # The ring in fundamental traces: N [tr(1234) + tr(4321)] + 2 tr tr.
        "3*Tr(1,2,3,4) + 3*Tr(4,3,2,1) + 2*Tr(1,2)*Tr(3,4) + 2*Tr(1,3)*Tr(2,4)"
        " + 2*Tr(1,4)*Tr(2,3)",
        # A d-symbol product: outside the five-point vocabulary, inside its
        # span at SU(3).
        "d(-1,1,2)*d(-1,3,4)",
        "Identity(1,2)*Identity(3,4) + 1/2*Identity(1,4)*Identity(2,3)"
        " + 1/2*Identity(1,3)*Identity(2,4) + 3/4*d(-1,1,2)*d(-1,3,4)"
        " - 3/4*f(-1,1,2)*f(-1,3,4)",
        "f(-1,1,-2)*f(-2,2,-3)*f(-3,3,-4)*f(-4,4,-1)",
    ],
)
def test_other_multi_trace_colours_project_exactly(projection, source):
    found = projection.vocabulary_colours(source, FIVE)
    assert found is not None
    assert all(not colour.startswith("d(") for colour, _ in found)
    assert np.abs(_recombined(found) - _tensor(source)).max() < 1e-12


@pytest.mark.loop_fast
def test_colours_without_a_vocabulary_form_are_not_projected(projection):
    """The C-odd part of a single trace has no f*f/Identity*Identity form on
    five legs (the d*f pairings it needs are not lowered there); on four
    adjoint legs the vocabulary admits d*f and it projects. Fundamental legs
    are never projected."""

    single = "Tr(1,2,3,4)"
    assert projection.vocabulary_colours(single, FIVE) is None
    found = projection.vocabulary_colours(single, (8, 8, 8, 8))
    assert any(colour.startswith("d(-1,1,2)*f(") for colour, _ in found)
    assert np.abs(_recombined(found) - _tensor(single)).max() < 1e-12
    assert projection.vocabulary_colours("T(3,2,1)*T(4,1,2)", (3, -3, 8, 8, 1)) is None
    assert projection.vocabulary_colours(RING, (8, 8, 8, 8, 8)) is None


@pytest.mark.loop_fast
def test_identity_pairs_with_singlet_legs_are_in_the_multi_trace_vocabulary():
    from pyamplicol.models.ct_color_flow import (
        UnloweredCountertermColour,
        tree_colour_source,
    )

    pair = "Identity(1,2)*Identity(3,4)"
    assert tree_colour_source(pair, FIVE, multi_trace=True) == (
        "UFO::{}::Identity(1,2)*UFO::{}::Identity(3,4)"
    )
    assert tree_colour_source("Identity(1,3)*Identity(2,5)", (8, 8, 8, 1, 8), multi_trace=True)
    for source, representations in (
        (pair, (8, 8, 8, 8, 8)),  # an adjoint leg left out
        (pair, (8, 8, 8, 8, 3)),
        ("Identity(1,2)*Identity(3,5)", FIVE),  # a singlet leg paired
    ):
        with pytest.raises(UnloweredCountertermColour):
            tree_colour_source(source, representations, multi_trace=True)
    with pytest.raises(UnloweredCountertermColour):
        tree_colour_source(pair, FIVE)
    with pytest.raises(UnloweredCountertermColour, match="multi-trace"):
        tree_colour_source(RING, FIVE, multi_trace=True)


def _vocabulary(projected: bool):
    from pyamplicol.loops.v2.ct_recursion import TreeVocabulary

    particles = [
        {"name": "G", "color": 8},
        {"name": "X0", "color": 1},
    ]
    compiled = SimpleNamespace(model={"particles": particles})
    return TreeVocabulary(compiled, multi_trace=True, projected=projected)


@pytest.mark.loop_fast
def test_counterterm_rows_take_the_projected_colours_with_their_factors():
    """``TreeVocabulary.colour_terms`` keeps a vocabulary colour as it is
    (factor None: the rows stay byte-identical), projects the ring only when
    ``projected`` (no formal-N_c continuation), and ``rule_rows`` gives one
    vertex term per projected colour with its exact factor."""

    pytest.importorskip("symbolica")
    from symbolica import E

    from pyamplicol.loops.v2.ct_recursion import rule_rows, scaled_by_colour

    legs = ("G", "G", "G", "G", "X0")
    ring = SimpleNamespace(color=RING, particle_names=legs, lorentz_name="L5", order=(("QCD", 6),))
    pair = SimpleNamespace(
        color="f(-1,1,2)*f(3,4,-1)", particle_names=legs, lorentz_name="L5", order=(("QCD", 6),)
    )
    vocabulary = _vocabulary(projected=True)
    assert vocabulary.colour_terms(pair) == (("f(-1,1,2)*f(3,4,-1)", None),)
    terms = vocabulary.colour_terms(ring)
    assert [colour for colour, _ in terms][:2] == ["f(-1,1,2)*f(-1,3,4)", "f(-1,1,3)*f(-1,2,4)"]
    # The Lorentz half of the cell (a normalised metric structure, here
    # stubbed) is unchanged; the colour half no longer refuses.
    vocabulary._lorentz["L5"], vocabulary._levi_civita["L5"] = None, False
    assert vocabulary.entry_reason(ring) is None
    unprojected = _vocabulary(projected=False)
    assert unprojected.colour_terms(ring) is None
    assert "multi-trace" in unprojected._cell(ring)[0]

    zero = E("0")
    coupling = SimpleNamespace(value=SimpleNamespace(coefficient=lambda power: E("1") if power == 0 else zero))
    rule = SimpleNamespace(kind="R2", entries=(ring,))
    ring.coupling = coupling
    rows = rule_rows(rule, colours=vocabulary.colour_terms)
    assert [(s.colour, s.power) for _e, s, _f in rows] == [(colour, 0) for colour, _ in terms]
    assert [factor for _e, _s, factor in rows] == [factor for _, factor in terms]
    assert scaled_by_colour(E("g"), None) == E("g")
    assert scaled_by_colour(E("g"), terms[1][1]) == E("g") / 2
    plain = rule_rows(rule)
    assert [(s.colour, f) for _e, s, f in plain] == [(RING, None)]
