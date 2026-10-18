# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import json
import os

import numpy as np
import pytest

from pyamplicol.loops.color_terms import (
    ColorTermDAG,
    LocalColorTerm,
    model_basis_terms,
    split_local_color_terms,
)
from pyamplicol.loops.kernels import CoefficientDAG
from pyamplicol.loops.model import CovariantKernelCatalogue, build_loop_model
from pyamplicol.models.loop_import import load_raw_loop_model


@pytest.fixture(scope="module")
def native():
    symbolica = pytest.importorskip("symbolica")
    if not symbolica.is_licensed():
        pytest.skip("native local Feynman rules require licensed Symbolica")
    from symbolica.community import hepkit, tensor

    return symbolica, hepkit, tensor


@pytest.fixture(scope="module")
def quartic(native):
    _, hep, _ = native
    diagrams = (
        hep.Model.qcd()
        .process(["g", "g"], ["g", "g"])
        .generate_diagrams(loops=0, progress=None, threads=1)
        .diagrams
    )
    graph = next(diagram for diagram in diagrams if len(diagram.vertices) == 1)
    return graph.vertices[0].numerator_expression()


def _project(expression, native, library):
    """Exact arbitrary external-component controls, not physical polarizations."""
    symbolica, _, tensor = native
    for axis in tuple(expression.axes):
        edge = int(next(iter(axis.index)))
        dimension = int(axis.representation.dimension)
        name = tensor.TensorName(f"color_test::state_{dimension}_{edge}")
        vector = [
            symbolica.E(str(((edge + 1) * (j + 3) + j * j) % 11 - 5))
            for j in range(dimension)
        ]
        library.register(tensor.Tensor.dense(name(axis.representation.dual()), vector))
        expression *= name(axis.dual())
    return expression


def _evaluate(current, points, parameters, native, library):
    symbolica, _, _ = native
    values = np.asarray(parameters, dtype=complex)
    outputs = current.dag.evaluate_batch(values)
    result = np.zeros(len(points), dtype=complex)
    for term in current.terms:
        node = current.dag.nodes[term.node]
        assert not node.axes
        powers = np.asarray([label.powers for label in node.output_labels])
        monomials = np.prod(points[:, None, :] ** powers[None, :, :], axis=2)
        color = term.color.to_tensor(library)
        assert not color.axes
        color_value = symbolica.Expression.evaluator_multiple(
            [color[0]], [], n_cores=1, jit_compile=False
        ).evaluate_complex(np.empty((len(points), 0), dtype=complex))[:, 0]
        result += np.sum(outputs[term.node] * monomials, axis=1) * color_value
    return result


def test_four_gluon_basis_local_branching_matches_native_contraction(native, quartic):
    symbolica, _, tensor = native
    q = tuple(symbolica.S(f"color_test::q{i}") for i in range(4))
    dag = CoefficientDAG(q)
    colored = ColorTermDAG(dag)
    library = tensor.TensorLibrary.hep_lib_atom()
    terms = split_local_color_terms(quartic)
    assert len(terms) == 3
    projected = tuple(
        LocalColorTerm(
            _project(term.lorentz, native, library),
            # The second test site is an independent scalar local factor. Close
            # these test-only SU(3) components before reusing them, so dummy
            # indices from the first site's native graph frame are not reused.
            tensor.TensorExpression(
                _project(term.color, native, library).to_tensor(library)[0]
            ),
        )
        for term in terms
    )
    first = colored.add("first-quartic", projected, library=library)
    second_terms = tuple(
        LocalColorTerm((1 + q[0]) * term.lorentz, term.color) for term in projected
    )
    second = colored.add("second-quartic", second_terms, (first,), library=library)
    assert first.closed and second.closed and not second.is_zero
    assert len(first.terms) == 3 and len(second.terms) == 9
    # Independent native local contraction; only this small test squares the
    # contracted local value. Production keeps node references for both sites.
    reference = _project(quartic, native, library).to_tensor(library)[0]
    oracle = symbolica.Expression.evaluator_multiple(
        [(1 + q[0]) * reference**2], [*dag.parameters, *q], n_cores=1, jit_compile=False
    )
    rng = np.random.default_rng(37419)
    parameters = rng.normal(size=(4, len(dag.parameters))).astype(complex)
    points = rng.normal(size=(4, 4)).astype(complex)
    actual = _evaluate(second, points, parameters, native, library)
    expected = oracle.evaluate_complex(np.concatenate([parameters, points], axis=1))[
        :, 0
    ]
    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=1e-9)
    assert all(
        not expression.contains(variable)
        for node in dag.nodes
        for expression in node.expressions
        for variable in q
    )


def test_equal_lorentz_instructions_share_nodes_and_keep_color_multiplicity(native):
    symbolica, _, tensor = native
    q = tuple(symbolica.S(f"color_merge_test::q{i}") for i in range(4))
    dag = CoefficientDAG(q)
    colored = ColorTermDAG(dag)
    factor = tensor.TensorExpression(q[0] + 1)
    current = colored.add(
        "same-local-factor",
        (
            LocalColorTerm(factor, tensor.TensorExpression(symbolica.E("2"))),
            LocalColorTerm(factor, tensor.TensorExpression(symbolica.E("3"))),
        ),
    )
    assert len(dag.nodes) == len(current.terms) == 1
    assert current.terms[0].color.to_expression() == 5
    zero = colored.add("zero", ())
    assert zero.is_zero and zero.closed
    assert colored.add("propagate-zero", (), (zero,)).is_zero
    other = ColorTermDAG(CoefficientDAG(q))
    with pytest.raises(ValueError, match="belong"):
        other.add("foreign", (), (current,))


def test_local_zero_and_component_axis_order_preserved(native, quartic):
    symbolica, _, tensor = native
    assert split_local_color_terms(tensor.TensorExpression(symbolica.E("0"))) == ()
    terms = split_local_color_terms(quartic)
    assert all(term.color.rank == 4 and term.lorentz.rank == 4 for term in terms)
    # Consumers receive native axes; local splitting cannot silently align by
    # ordinal position and thereby permute Lorentz or colour labels.
    assert (
        sum((term.lorentz * term.color for term in terms), 0) - quartic
    ).expand().to_expression() == 0


def test_r2_model_basis_keeps_selected_loop_group_and_exact_laurent_power(native):
    symbolica, _, tensor = native
    source = os.environ.get("PYAMPLICOL_NLO_UFO_REFERENCE")
    if source is None:
        pytest.skip("set PYAMPLICOL_NLO_UFO_REFERENCE to the local loop_sm model")
    from ufo_model_loader.model import Model

    ir = build_loop_model(load_raw_loop_model(source))
    structural = Model.from_ufo_model(source)
    structural.wrap_indices_in_lorentz_structures()
    catalogue = CovariantKernelCatalogue(ir, json.loads(structural.to_json()))
    selected = tuple(
        entry
        for entry in ir.kernels
        if entry.vertex_name == "V_R2RGA" and entry.loop_group_index == 1
    )
    terms = model_basis_terms(catalogue, selected)
    assert len(terms) == len(selected) == 29
    assert tuple(term.source for term in terms) == tuple(
        entry.key for entry in selected
    )
    library = tensor.TensorLibrary.hep_lib_atom()
    dag = CoefficientDAG(tuple(symbolica.S(f"r2_color_test::q{i}") for i in range(4)))
    current = ColorTermDAG(dag).add(
        "four-gluon-r2",
        tuple(
            LocalColorTerm(
                _project(term.lorentz, native, library), term.color, term.source
            )
            for term in terms
        ),
        library=library,
    )
    assert current.closed and not current.is_zero
    assert all(term.color.rank == 4 for term in current.terms)
    assert all(node.rank == 0 for node in dag.nodes)
    assert len(dag.nodes) <= len(terms)
    assert model_basis_terms(catalogue, selected, epsilon_power=-1) == ()
    mass = next(entry for entry in ir.kernels if entry.vertex_name == "V_UVtMass")
    assert len(model_basis_terms(catalogue, (mass,), epsilon_power=-1)) == 1
    with pytest.raises(ValueError, match="one vertex"):
        model_basis_terms(catalogue, (*selected, mass))
