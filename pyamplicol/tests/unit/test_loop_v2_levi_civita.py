# SPDX-License-Identifier: 0BSD
"""Levi-Civita counterterm kernels of the v2 catalogue against ALOHA.

``R2_GGZ`` and ``R2_GGGVa`` (the axial parts of the Z-gluon R2 rules) carry
``Epsilon``, which the tree compiler normalises to the dense
``models.tensors.LEVI_CIVITA`` (ALOHA's components). The catalogue's
``ct-site`` contraction of each, on exact off-shell complex currents and
momenta, equals the unchanged compiled ALOHA routine of MG5_aMC 3.7.0
``loop_sm`` (values frozen below, produced by
``tools/loop_reference/levi_civita_aloha.py``) to binary64 rounding; the tree
kernels do too (``test_model_levi_civita_kernels``). The root closure reads
these vertices with the momenta flowing in and the incoming colour, the tree
engine with its outgoing particles': ``ct_sites.SiteClosure._reading_sign``
(WP3.2 EWI-7b, ``test_loop_v2_epsilon_sign``).
"""

from __future__ import annotations

from fractions import Fraction

import pytest

from tools.loop_reference.levi_civita_aloha import INPUTS

# ``tools/loop_reference/levi_civita_aloha.py``: (real, imaginary) per sample.
ALOHA = {
    "R2_GGZ": (
        (-2.180348408548779, 0.5712062435793254),
        (-6.406222222222222, 15.696296296296293),
    ),
    "R2_GGGVa": (
        (1.246004938271605, 0.314074074074074),
        (-0.10745432289285177, -0.1733110425240054),
    ),
}
# The UFO structures (MG5_aMC 3.7.0 ``loop_sm/lorentz.py``), wrapped.
STRUCTURES = {
    "R2_GGZ": (
        "-1*UFO::Epsilon(UFO::idx(1,3),UFO::idx(1,1),UFO::idx(1,2),UFO::dummy(1))"
        "*UFO::P(UFO::dummy(1),UFO::idx(1,1))"
        "+UFO::Epsilon(UFO::idx(1,3),UFO::idx(1,1),UFO::idx(1,2),UFO::dummy(1))"
        "*UFO::P(UFO::dummy(1),UFO::idx(1,2))"
    ),
    "R2_GGGVa": (
        "UFO::Epsilon(UFO::idx(1,4),UFO::idx(1,1),UFO::idx(1,2),UFO::idx(1,3))"
    ),
}


def _complex(value) -> complex:
    from symbolica import Expression

    real = ((value + value.conj()) / 2).expand()
    imaginary = ((value - value.conj()) / (2 * Expression.I)).expand()
    return complex(
        float(Fraction(real.to_canonical_string())),
        float(Fraction(imaginary.to_canonical_string())),
    )


@pytest.mark.loop_fast
@pytest.mark.parametrize("name", tuple(STRUCTURES))
def test_levi_civita_ct_kernels_equal_aloha(name):
    from pyamplicol._internal.physics.symbols import symbols
    from pyamplicol.loops.v2.catalogue import _contract_lorentz
    from pyamplicol.models import compiler_symbolica as sym
    from pyamplicol.models.tensors import normalize_lorentz_expression

    sym._ensure_symbolica()
    registry = symbols.model("loop_v2_levi_civita")
    legs = len(INPUTS[name][0][0])
    spins = (3,) * legs
    expression = normalize_lorentz_expression(
        STRUCTURES[name], spins, model_symbols=registry
    ).expression
    for (momenta, polarisations), (real, imaginary) in zip(
        INPUTS[name], ALOHA[name], strict=True
    ):
        currents = {
            leg: tuple(
                sym.E(str(Fraction(re))) + sym.Expression.I * sym.E(str(Fraction(im)))
                for re, im in polarisation
            )
            for leg, polarisation in enumerate(polarisations)
        }
        flows = {
            leg: tuple(sym.E(str(Fraction(x))) for x in momentum)
            for leg, momentum in enumerate(momenta)
        }
        (value,) = _contract_lorentz(
            registry, expression, spins, currents, flows, output_leg=None, tag="lc"
        )
        expected = complex(real, imaginary)
        assert abs(_complex(value) - expected) <= 1e-14 * abs(expected), (
            name,
            value,
            expected,
        )


@pytest.mark.loop_fast
def test_the_tree_compiler_normalises_epsilon():
    """The tree compiler admits Epsilon (WP3.2 EWI-7b): the normalised
    structure contracts ``LEVI_CIVITA`` and no UFO tensor is left."""

    from pyamplicol.models.tensors import LEVI_CIVITA, normalize_lorentz_expression

    normalized = normalize_lorentz_expression(STRUCTURES["R2_GGGVa"], (3, 3, 3, 3))
    assert LEVI_CIVITA.rsplit("::", 1)[-1] in normalized.tensor_heads
    assert "UFO::" not in normalized.expression


@pytest.mark.loop_fast
def test_reading_signs_of_root_vertices():
    """The parts of the root closure's reading sign, applied to every vertex
    closed at a root (WP3.2 EWI-7b, genericity audit 2 G2-M7): the colour's
    sign under the transposition of its adjoint generators (-1 per ``f``;
    ``d``, fundamental generators between swapped ends, ``Identity``,
    two-index traces and singlets keep it; products multiply) and the
    kernel's momentum parity. A colour without one (a trace of three octets,
    a sextet generator, an epsilon tensor, a mixture of ``d`` and ``f``
    terms) or a kernel neither even nor odd in the momenta is refused with the
    typed ``ReadingSignError`` naming the vertex."""

    from types import SimpleNamespace

    from pyamplicol.loops.v2.ct_sites import (
        ReadingSignError,
        _colour_transposition_sign,
        _momentum_parity,
    )

    for colour, sign in (
        ("1", 1),
        ("f(1,2,3)", -1),
        ("d(1,2,3)", 1),
        ("T(3,2,1)", 1),
        ("Identity(1,2)", 1),
        ("Tr(1,2)", 1),
        ("complex(0,1)*f(1,2,3)", -1),
        ("f(-1,1,2)*f(3,4,-1)", 1),
        ("d(-1,1,2)*f(3,4,-1)", -1),
        ("f(-1,1,2)*T(-1,4,3)", -1),
        ("f(1,2,3) - f(1,3,2)", -1),
    ):
        assert _colour_transposition_sign(colour) == sign, colour
    for colour, message in (
        ("Tr(1,2,3)", "colour tensor Tr of 3 indices"),
        ("T6(1,2,3)", "colour tensor T6 of 3 indices"),
        ("Epsilon(1,2,3)", "colour tensor Epsilon of 3 indices"),
        ("f(1,2,3) + d(1,2,3)", "change sign differently"),
    ):
        with pytest.raises(ReadingSignError, match=rf"vertex V_X: .*{message}"):
            _colour_transposition_sign(colour, "vertex V_X")

    def kernel(*expressions):
        inputs = (
            SimpleNamespace(symbol="p0", role="block-momentum"),
            SimpleNamespace(symbol="p1", role="block-momentum"),
            SimpleNamespace(symbol="j0", role="block-current"),
        )
        return SimpleNamespace(inputs=inputs, exact_expressions=expressions)

    assert _momentum_parity(kernel("j0*p0 - j0*p1", "p0")) == -1
    assert _momentum_parity(kernel("j0", "p0*p1")) == 1
    with pytest.raises(ReadingSignError, match=r"vertex V_X: .*neither even nor odd"):
        _momentum_parity(kernel("j0*p0 + j0"), "vertex V_X")
    with pytest.raises(ReadingSignError, match="neither even nor odd"):
        _momentum_parity(kernel("j0*p0", "j0"))
