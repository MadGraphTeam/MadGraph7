# SPDX-License-Identifier: 0BSD
"""Levi-Civita counterterm kernels of the v2 catalogue against ALOHA.

``R2_GGZ`` and ``R2_GGGVa`` (the axial parts of the Z-gluon R2 rules) carry
``Epsilon``. The catalogue's ``ct-site`` contraction of each, on exact
off-shell complex currents and conserved momenta flowing into the vertex,
equals the unchanged compiled ALOHA routine of MG5_aMC 3.7.0 ``loop_sm``
(values frozen below, produced by ``tools/loop_reference/levi_civita_aloha.py``)
to binary64 rounding.
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
    from pyamplicol.loops.v2.catalogue import _contract_lorentz, _with_levi_civita
    from pyamplicol.models import compiler_symbolica as sym
    from pyamplicol.models.tensors import normalize_lorentz_expression

    sym._ensure_symbolica()
    registry = symbols.model("loop_v2_levi_civita")
    legs = len(INPUTS[name][0][0])
    spins = (3,) * legs
    expression = normalize_lorentz_expression(
        _with_levi_civita(STRUCTURES[name]), spins, model_symbols=registry
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
def test_the_tree_compiler_still_refuses_epsilon():
    """Only the loop catalogue's counterterm sites admit Epsilon."""

    from pyamplicol.models.tensors import normalize_lorentz_expression

    with pytest.raises(ValueError, match="unsupported UFO tensors: Epsilon"):
        normalize_lorentz_expression(STRUCTURES["R2_GGGVa"], (3, 3, 3, 3))
