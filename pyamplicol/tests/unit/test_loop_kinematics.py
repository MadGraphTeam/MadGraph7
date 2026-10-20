# SPDX-License-Identifier: 0BSD
"""Kinematic bindings checked against independent native denominator routing."""

from __future__ import annotations

import pytest

from pyamplicol.loops.graphs import build_loop_process
from pyamplicol.loops.kinematics import build_loop_kinematics
from pyamplicol.loops.recurrence import build_open_loop_recurrence

pytestmark = pytest.mark.loop_fast


@pytest.fixture(scope="module")
def native():
    symbolica = pytest.importorskip("symbolica")
    if not symbolica.is_licensed():
        pytest.skip("native graph construction requires a licensed Symbolica")
    from symbolica.community import hepkit, tensor

    return symbolica, hepkit, tensor


@pytest.fixture(scope="module")
def processes(native):
    _, h, _ = native
    return {
        name: build_loop_process(
            h.Model.qcd_qed().process(["u", "u~"], final),
            selected_corrections=["QCD"],
            perturbative_orders={"QCD": 1},
        )
        for name, final in [("dy", ["e-", "e+"]), ("quarks", ["d", "d~"])]
    }


def _native_denominators(recurrence, native):
    s, h, t = native
    graph = recurrence.graph.native
    library = t.TensorLibrary.hep_lib_atom()
    loop = t.TensorName.vector("kinematic_oracle::q")
    external = [
        t.TensorName.vector(f"kinematic_oracle::p{i}")
        for i in range(len(recurrence.momentum_parameters))
    ]
    library.register(
        t.Tensor.dense(loop(t.Representation.mink(4)), recurrence.loop_components)
    )
    for name, components in zip(external, recurrence.momentum_parameters, strict=True):
        library.register(t.Tensor.dense(name(t.Representation.mink(4)), components))
    wild = s.S(
        "kinematic_oracle::e_",
        "kinematic_oracle::p_",
        "kinematic_oracle::m_",
        "kinematic_oracle::v_",
    )
    basis = graph.loop_momentum_basis
    result = {}
    for edge in graph.internal_edges:
        if edge.id not in recurrence.graph.loop_edges:
            continue
        expression = (
            edge.denominator_expression(dimension=4)
            .to_expression()
            .replace(h.Symbols.denominator()(*wild), wild[-1])
        )
        routed = basis.route_expression(
            t.TensorExpression(expression),
            loop_momenta=[loop],
            external_momenta=[
                external[i]
                for i, edge_id in enumerate(basis.external_edges)
                if edge_id not in basis.dependent_externals
            ],
        )
        result[edge.id] = routed.to_tensor(library)[0].expand()
    return result


@pytest.mark.parametrize("massive", [False, True])
def test_routed_kinematics_match_native_denominators_and_family_map(
    native, processes, massive
):
    s, _, t = native
    graph = (
        next(g for g in processes["quarks"].loops if g.loop_species == (6,))
        if massive
        else processes["dy"].loops[0]
    )
    plans = []
    for routing in ("graph", "family"):
        recurrence = build_open_loop_recurrence(graph, routing=routing)
        plan = build_loop_kinematics(recurrence)
        expected = _native_denominators(recurrence, native)
        for edge, shift, mass in zip(
            plan.edge_ids, plan.shifts, plan.masses_squared, strict=True
        ):
            vector = t.Tensor.dense(
                t.TensorName.vector("kinematic_test::q_plus_v")(
                    t.Representation.mink(4)
                ),
                [
                    q + value
                    for q, value in zip(recurrence.dag.variables, shift, strict=True)
                ],
            )
            actual = vector.dot(vector).to_tensor()[0] - mass
            assert (actual - expected[edge]).expand() == 0
        assert plan.evaluator_bytes()
        assert plan.propagators == len(graph.loop_edges)
        for i in range(plan.propagators):
            assert plan.s_matrix[i * plan.propagators + i] == 0
            for j in range(plan.propagators):
                assert (
                    plan.s_matrix[i * plan.propagators + j]
                    == plan.s_matrix[j * plan.propagators + i]
                )
        plans.append(plan)
    # A shared affine shift or reversal changes neither these pair invariants
    # nor the graph edge ordering. This must hold symbolically, before sampling.
    assert plans[0].edge_ids == plans[1].edge_ids
    assert all(
        (a - b).expand() == 0
        for a, b in zip(plans[0].s_matrix, plans[1].s_matrix, strict=True)
    )
    if massive:
        assert all(mass != s.E("0") for mass in plans[0].masses_squared)
    else:
        # The dependency's qcd_qed model gives the u quark a symbolic mass; only
        # the gluon propagator of this vertex correction is massless.
        u_mass = native[1].Model.qcd_qed().particle("u").mass ** 2
        assert sorted(map(str, plans[0].masses_squared)) == sorted(
            map(str, (u_mass, u_mass, s.E("0")))
        )


def test_exact_invariant_input_plan_preserves_lightlike_relations(
    loop_native, native, processes
):
    s, _, _ = native
    recurrence = build_open_loop_recurrence(processes["dy"].loops[0])
    plan = build_loop_kinematics(recurrence)
    backend = loop_native
    if not hasattr(backend, "ExactPolynomialEvaluator"):
        pytest.skip("exact native kinematics evaluator not built")
    from decimal import Decimal, localcontext

    with localcontext() as context:
        context.prec = 220
        scale = 1 + Decimal(2) ** -180
        values = [
            [str(Decimal(x) * scale), "0"]
            for leg in [
                [500, 0, 0, 500],
                [500, 0, 0, -500],
                [500, 300, 0, 400],
                [500, -300, 0, -400],
            ]
            for x in leg
        ]
        evaluator = backend.ExactPolynomialEvaluator(plan.evaluator_bytes())
        result = evaluator.evaluate_decimal(values, 1, "arbitrary", 80, [61])
        assert result.row_ids == [61]
        matrix = [
            Decimal(pair[0]) for pair in result.to_decimal()[4 * plan.propagators :]
        ]
        assert len(matrix) == 9
        assert sum(value != 0 for value in matrix) == 2
        for value in matrix:
            if value:
                assert abs(value / (Decimal(1000000) * scale**2) - 1) < Decimal("1e-78")
        # Symbolica independently substitutes original exact rationals, without
        # passing through any precision-selected momentum buffer.
        # Use the existing exact UFO scalar input boundary for decimal literals.
        from pyamplicol.loops.model_expressions import parse_exact_ufo

        replacements = [
            s.Replacement(p, parse_exact_ufo(x[0]))
            for p, x in zip(plan.momentum_parameters, values, strict=True)
        ]
        for expression, value in zip(plan.s_matrix, matrix, strict=True):
            exact = expression.replace_multiple(replacements).expand()
            if value == 0:
                assert exact == 0


def test_family_order_binds_identical_propagators_before_combining_numerators(
    native, processes
):
    by_family, boxes = {}, 0
    for graph in processes["quarks"].loops:
        recurrence = build_open_loop_recurrence(graph, routing="family")
        plan = build_loop_kinematics(recurrence, family_order=True)
        assert plan.denominator_order == "family"
        signature = (plan.shifts, plan.masses_squared, plan.s_matrix)
        previous = by_family.setdefault(graph.family.family_index, signature)
        assert signature == previous
        # Family denominators follow the loop cycle: neighbours share a vertex.
        ends = {
            edge.id: {edge.source, edge.target} for edge in graph.native.internal_edges
        }
        ids = plan.edge_ids
        assert all(
            ends[a] & ends[b] for a, b in zip(ids, ids[1:] + ids[:1], strict=True)
        )
        boxes += len(ids) == 4
    assert boxes
    recurrence = build_open_loop_recurrence(processes["dy"].loops[0])
    with pytest.raises(ValueError, match="family momentum routing"):
        build_loop_kinematics(recurrence, family_order=True)
