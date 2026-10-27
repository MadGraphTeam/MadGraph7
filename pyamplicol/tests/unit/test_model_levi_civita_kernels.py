# SPDX-License-Identifier: 0BSD
"""Tree kernels of UFO ``Epsilon`` (WP3.2 EWI-7b) against ALOHA.

The tree compiler lowers the Levi-Civita tensor of UFO Lorentz structures
(``models.tensors.LEVI_CIVITA``, ALOHA's components), so the R2 of a Z with
gluons (``R2_GGZ``, ``R2_GGGVa``) becomes ordinary tree kernels where its
cells do not cancel between the quark doublets (``loops.v2.ct_recursion``).
On the exact off-shell inputs of ``tools/loop_reference/levi_civita_aloha.py``
the kernels equal the unchanged compiled ALOHA routines of MG5_aMC 3.7.0
``loop_sm`` (values frozen in ``test_loop_v2_levi_civita``) exactly:

* ``R2_GGZ``: every oriented three-point kernel, its output current contracted
  with the third leg's polarisation;
* ``R2_GGGVa``: the four-point contact (a two-gluon partial and a final
  kernel): with a gluon result, the partial of the other two gluons and the
  final with the Z; with the Z result, the colour-weighted sum over the three
  gluon pairs (the ``f`` colour is antisymmetric, so each pair enters with the
  sign of its permutation; each carries a third of the vertex).

The engine's own Dirac algebra reads the tensor as ALOHA does:
Tr(gamma5 a b c d) = 4 i epsilon(a, b, c, d) with ALOHA's components. The
physical sign (bare and R2 parts against MadLoop, the gluon Ward identity of
an anomalous triangle) is ``test_loop_v2_epsilon_sign``'s.
"""

from __future__ import annotations

import json
from fractions import Fraction
from itertools import permutations

import pytest

from tools.loop_reference.levi_civita_aloha import INPUTS

from .test_loop_v2_levi_civita import ALOHA

METRIC = (1, -1, -1, -1)


@pytest.fixture(scope="module")
def levi_civita_ir(loop_sm_compiled_model):
    """``name -> compiled IR`` of a model with the gluon, the Z and one
    vertex carrying loop_sm's Levi-Civita structure ``name``."""

    from pyamplicol.models.compiler_entry import compile_ufo_model_ir
    from pyamplicol.models.loading import load_compiled_model

    payload = json.loads(json.dumps(load_compiled_model(loop_sm_compiled_model).model))
    fields = {"G", "Z"}
    payload["particles"] = [p for p in payload["particles"] if p["name"] in fields]
    payload["propagators"] = [
        p for p in payload["propagators"] if p["particle"] in fields
    ]
    payload["lorentz_structures"] = [
        item for item in payload["lorentz_structures"] if item["name"] in ALOHA
    ]
    payload["couplings"] = [
        {
            "expression": "UFO::{}::MZ",
            "name": "PYAMPLICOL_TEST_C",
            "orders": [["QCD", 2], ["QED", 1]],
            "value": [91.188, 0.0],
        }
    ]
    vertices = {
        "R2_GGZ": (["G", "G", "Z"], "UFO::{}::Identity(1,2)"),
        "R2_GGGVa": (["G", "G", "G", "Z"], "UFO::{}::f(1,2,3)"),
    }

    def compile_(name: str):
        particles, colour = vertices[name]
        model = dict(payload)
        model["vertex_rules"] = [
            {
                "name": "V_PYAMPLICOL_TEST",
                "particles": particles,
                "color_structures": [colour],
                "lorentz_structures": [name],
                "couplings": [["PYAMPLICOL_TEST_C"]],
            }
        ]
        return compile_ufo_model_ir(model, tree_kernel_variant="full")

    return compile_


def _number(value: str):
    from pyamplicol.models import compiler_symbolica as sym

    return sym.E(str(Fraction(value)))


def _complex(value) -> complex:
    """The coefficient of the kernels' coupling symbol, as a complex number."""

    from symbolica import Expression

    from pyamplicol.models import compiler_symbolica as sym

    (coupling,) = (
        symbol
        for symbol in value.get_all_symbols(False)
        if "derived_coupling" in symbol.get_name()
    )
    coefficient = value.replace(coupling, sym.E("1")).expand()
    real = ((coefficient + coefficient.conj()) / 2).expand()
    imaginary = ((coefficient - coefficient.conj()) / (2 * Expression.I)).expand()
    return complex(
        float(Fraction(real.to_canonical_string())),
        float(Fraction(imaginary.to_canonical_string())),
    )


def _evaluate(ir, kernel, left, right, left_momentum=None, right_momentum=None):
    """The output components of one oriented kernel on exact inputs."""

    from pyamplicol.models import compiler_symbolica as sym

    space = f"model_{ir.name}::compiler::{{}}::" if ir.name else ""
    out = []
    for component in kernel.component_expressions:
        value = sym.E(component)
        for side, vector, momentum in (
            ("left", left, left_momentum),
            ("right", right, right_momentum),
        ):
            for index, entry in enumerate(vector):
                value = value.replace(
                    sym.E(f"{space}kernel_{kernel.kind}_{side}_{index}"), entry
                )
            for index, entry in enumerate(momentum or ()):
                value = value.replace(
                    sym.E(f"{space}kernel_{kernel.kind}_{side}_momentum_{index}"),
                    entry,
                )
        out.append(value.expand())
    return out


def _contract(current, polarisation):
    from pyamplicol.models import compiler_symbolica as sym

    total = sym.E("0")
    for mu in range(4):
        total = total + current[mu] * METRIC[mu] * polarisation[mu]
    return total.expand()


def _inputs(name: str):
    from pyamplicol.models import compiler_symbolica as sym

    sym._ensure_symbolica()
    for (momenta, polarisations), expected in zip(
        INPUTS[name], ALOHA[name], strict=True
    ):
        vectors = [
            [_number(re) + sym.Expression.I * _number(im) for re, im in leg]
            for leg in polarisations
        ]
        flows = [[_number(x) for x in leg] for leg in momenta]
        yield vectors, flows, complex(*expected)


def _assert_aloha(value, expected: complex, context) -> None:
    assert abs(_complex(value) - expected) <= 1e-14 * abs(expected), (
        context,
        value,
        expected,
    )


def test_three_point_levi_civita_tree_kernels_equal_aloha(levi_civita_ir):
    ir = levi_civita_ir("R2_GGZ")
    kernels = [k for k in ir.oriented_kernels if k.vertex == "V_PYAMPLICOL_TEST"]
    # The gluon results share their kernels (equal under the exchange of the
    # two gluons); the Z result has its own.
    assert {k.source_particle_legs[2] for k in kernels} == {0, 2}
    for vectors, flows, expected in _inputs("R2_GGZ"):
        for kernel in kernels:
            left, right, result = kernel.source_particle_legs
            current = _evaluate(
                ir, kernel, vectors[left], vectors[right], flows[left], flows[right]
            )
            value = _contract(current, vectors[result])
            _assert_aloha(value, expected, kernel.source_particle_legs)


def test_four_point_levi_civita_contact_kernels_equal_aloha(levi_civita_ir):
    ir = levi_civita_ir("R2_GGGVa")
    kernels = {k.kind: k for k in ir.oriented_kernels}
    (partial,) = (k for k in kernels.values() if k.vertex.endswith("contact-partial"))
    finals = [k for k in kernels.values() if "contact-final" in k.vertex]
    gluons, z = (0, 1, 2), 3
    for vectors, _flows, expected in _inputs("R2_GGGVa"):
        for final in finals:
            left, right, result = final.source_particle_legs
            physical = right if left < 0 else left
            if result == z:
                # The colour-weighted sum over the gluon pairs: f's sign.
                total = None
                for pair in ((0, 1), (0, 2), (1, 2)):
                    third = next(g for g in gluons if g not in pair)
                    sign = _sign((*pair, third))
                    aux = _evaluate(ir, partial, vectors[pair[0]], vectors[pair[1]])
                    current = (
                        _evaluate(ir, final, aux, vectors[third])
                        if left < 0
                        else _evaluate(ir, final, vectors[third], aux)
                    )
                    value = sign * _contract(current, vectors[z])
                    total = value if total is None else total + value
                _assert_aloha(total.expand(), expected, final.source_particle_legs)
                continue
            pair = tuple(g for g in gluons if g != result)
            assert physical == z
            aux = _evaluate(ir, partial, vectors[pair[0]], vectors[pair[1]])
            current = (
                _evaluate(ir, final, aux, vectors[z])
                if left < 0
                else _evaluate(ir, final, vectors[z], aux)
            )
            _assert_aloha(
                _contract(current, vectors[result]),
                expected,
                final.source_particle_legs,
            )


def _sign(order) -> int:
    inversions = sum(
        1
        for i in range(len(order))
        for j in range(i + 1, len(order))
        if order[i] > order[j]
    )
    return -1 if inversions % 2 else 1


@pytest.mark.loop_fast
def test_the_gamma5_trace_reads_the_levi_civita_tensor_as_aloha():
    """Tr(gamma5 a b c d) = 4 i epsilon(a, b, c, d) in the engine's networks
    (spenso's gamma matrices, gamma5 = i gamma^0 gamma^1 gamma^2 gamma^3,
    Minkowski slots contracted with the metric), as with ALOHA's matrices
    and its ``Epsilon`` routines (epsilon^{0123} = -1)."""

    from symbolica.community.tensor import TensorName

    from pyamplicol.models import compiler_symbolica as sym
    from pyamplicol.models import tensors
    from pyamplicol.models.compiler_kernels import DenseTensors, _execute_dense_network

    tensors._ensure_symbolica()
    sym._ensure_symbolica()
    minkowski = tensors.Representation.mink(4)
    bispinor = tensors.Representation.bis(4)
    gamma = tensors.TensorExpression.dirac_gamma(4)
    gamma5 = tensors.TensorExpression.gamma5(4)
    vectors = ((3, 1, -2, 5), (7, -4, 2, 1), (-1, 6, 3, -2), (2, 2, -5, 9))
    names = [f"pyamplicol::test_levi_civita_vector_{i}" for i in range(4)]
    dense = DenseTensors()
    for name, vector in zip(names, vectors, strict=True):
        dense.register_dense(name, (minkowski,), tuple(sym.E(str(x)) for x in vector))

    def vector(index: int, label: str):
        return TensorName(names[index])(minkowski(label)).to_expression()

    spinors = [bispinor(f"s{i}") for i in range(5)]
    trace = tensors._as_expression(gamma5(spinors[0], spinors[1]))
    for i in range(4):
        following = spinors[i + 2] if i < 3 else spinors[0]
        trace *= tensors._as_expression(
            gamma(spinors[i + 1], following, minkowski(f"x{i}"))
        ) * vector(i, f"x{i}")
    epsilon = TensorName(tensors.LEVI_CIVITA)(
        *[minkowski(f"y{i}") for i in range(4)]
    ).to_expression()
    for i in range(4):
        epsilon *= vector(i, f"y{i}")
    (traced,) = _execute_dense_network(trace, dense, ()).values
    (contracted,) = _execute_dense_network(epsilon, dense, ()).values
    assert (traced - 4 * sym.Expression.I * contracted).expand() == sym.E("0")
    # ALOHA's Epsilon(1,2,3,4) on the same contravariant components: the
    # metric-contracted determinant with epsilon^{0123} = -1.
    determinant = sum(
        _sign(order)
        * vectors[0][order[0]]
        * vectors[1][order[1]]
        * vectors[2][order[2]]
        * vectors[3][order[3]]
        for order in permutations(range(4))
    )
    assert contracted == sym.E(str(determinant))
