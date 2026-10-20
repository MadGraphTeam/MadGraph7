# SPDX-License-Identifier: 0BSD
"""Retained exact instructions, with independent native precision workspaces."""

from dataclasses import FrozenInstanceError
from decimal import Decimal, localcontext

import pytest

from pyamplicol.loops.kernels import (
    CoefficientDAG,
    CoefficientKernel,
    NativeCoefficientDAG,
    PreparedCoefficientDAG,
)

pytestmark = pytest.mark.loop_fast


@pytest.fixture(scope="module")
def native(loop_native):
    symbolica = pytest.importorskip("symbolica")
    if not symbolica.is_licensed():
        pytest.skip("licensed exact tensor generation required")
    return loop_native


def make_dag():
    from symbolica import E, S
    from symbolica.community import tensor as t

    q = tuple(S(*(f"prepared_coefficient::q{i}" for i in range(4))))
    mass, external = S("prepared_coefficient::mass", "prepared_coefficient::external")
    dag = CoefficientDAG(q)
    first = dag.add("local", t.TensorExpression(mass * q[0] + external).to_tensor())
    dag.add("close", t.TensorExpression(E("2")).to_tensor(), (first,))
    return dag


@pytest.mark.parametrize(
    "arithmetic,precision,engine",
    [
        ("double", None, "interpreted"),
        ("double", None, "jit"),
        ("double-double", None, "interpreted"),
        ("arbitrary", 80, "interpreted"),
    ],
)
def test_rebinding_retains_instructions_and_independent_exact_inputs(
    native, monkeypatch, arithmetic, precision, engine
):
    dag = make_dag()
    original = NativeCoefficientDAG(dag, backend=native, output_nodes=(1,))
    prepared = original.prepared
    identity = prepared.identity
    # The producer graph is deliberately no longer usable. Rebinding must not
    # revisit it, optimize symbolic expressions, or save evaluator instructions.
    dag.nodes.clear()
    dag.domain = "changed-after-preparation"

    def forbidden(*args, **kwargs):
        raise AssertionError("retained instruction rebinding revisited the producer")

    monkeypatch.setattr(CoefficientDAG, "add", forbidden)
    monkeypatch.setattr(CoefficientKernel, "evaluator_bytes", forbidden)
    monkeypatch.setattr(PreparedCoefficientDAG, "from_dag", forbidden)
    options = dict(arithmetic=arithmetic, precision=precision)
    first = original.rebind(**options, coefficient_engine=engine)
    second = NativeCoefficientDAG(
        prepared, backend=native, coefficient_engine=engine, **options
    )
    assert first.prepared is second.prepared is prepared
    assert first.schedule_identity[:-3] == second.schedule_identity[:-3] == identity
    assert first.plan is not second.plan and second.plan.workspace_capacity == (0, 0, 0)
    assert first.dag is second.dag is None

    with localcontext() as context:
        context.prec = 110
        precise_mass = str(Decimal(1) + Decimal(2) ** -180)
        masses = [precise_mass, "1.25", "-3.75"]
        ids = [42, 7, 91]
        values = [
            [mass if p.get_name().endswith("::mass") else "7", "0"]
            for mass in masses
            for p in prepared.parameters
        ]
        inputs = native.NativeBatch.from_decimal(
            values, len(ids), len(prepared.parameters), row_ids=ids, **options
        )
        result = first.evaluate_batch(inputs)
        retained = result.to_decimal()
        assert first.plan.workspace_capacity != (0, 0, 0)
        assert second.plan.workspace_capacity == (0, 0, 0)
        labels = prepared.output_metadata[0][1]
        for index, mass in enumerate(masses):
            row = retained[index * len(labels) : (index + 1) * len(labels)]
            coefficients = {
                label.powers[0]: Decimal(pair[0])
                for label, pair in zip(labels, row, strict=True)
            }
            assert coefficients[0] == 14
            if index or arithmetic == "arbitrary":
                assert abs(coefficients[1] - 2 * Decimal(mass)) < Decimal("1e-75")
            if index == 0 and arithmetic == "arbitrary":
                assert coefficients[1] != 2  # Original bits beyond binary64.
        assert result.row_ids == ids
        other = second.evaluate_batch(inputs.take_rows([2, 0]))
        assert other.row_ids == [91, 42]
        assert other.to_decimal() == result.take_rows([2, 0]).to_decimal()
        empty = second.evaluate_batch(inputs.take_rows([]))
        assert empty.rows == 0 and empty.columns == result.columns
        assert result.to_decimal() == retained
    assert prepared.identity == identity
    with pytest.raises(FrozenInstanceError):
        prepared.instructions.parameter_count = 999
    with pytest.raises(ValueError, match="already selected"):
        NativeCoefficientDAG(prepared, backend=native, output_nodes=(0,))


def test_prepared_output_selection_keeps_descriptor_identity(native):
    dag = make_dag()
    all_nodes = PreparedCoefficientDAG.from_dag(dag)
    duplicated = PreparedCoefficientDAG.from_dag(dag, output_nodes=(1, 1))
    assert all_nodes.instructions.states == duplicated.instructions.states
    assert all_nodes.identity != duplicated.identity
    assert duplicated.output_widths[0] == duplicated.output_widths[1]
    with pytest.raises(ValueError, match="existing coefficient"):
        PreparedCoefficientDAG.from_dag(dag, output_nodes=(2,))
