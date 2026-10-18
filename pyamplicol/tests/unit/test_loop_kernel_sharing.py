# SPDX-License-Identifier: 0BSD
"""Exact schedule binding identity, independent of graph/flavour guesses."""

from dataclasses import replace
from decimal import Decimal

import pytest

from pyamplicol.loops.kernels import (
    CoefficientDAG,
    NativeCoefficientDAG,
    coefficient_sharing_groups,
)


@pytest.fixture(scope="module")
def native():
    symbolica = pytest.importorskip("symbolica")
    if not symbolica.is_licensed():
        pytest.skip("licensed exact tensor generation required")
    return pytest.importorskip("_loop_reduction")


def make_dag(mass_name, *, external_name="sharing_external", axis=0, auxiliary=False):
    from symbolica import E, S
    from symbolica.community import tensor as t

    q = tuple(S("sharing_q0", "sharing_q1", "sharing_q2", "sharing_q3"))
    mass, external = S(mass_name, external_name)
    options = (
        {
            "extra_variables": tuple(S("sharing_mu", "sharing_eps")),
            "extra_weights": (1, 0),
        }
        if auxiliary
        else {}
    )
    dag = CoefficientDAG(q, **options)
    node = dag.add("local", t.TensorExpression(mass * q[axis] + external).to_tensor())
    dag.add("close", t.TensorExpression(E("2")).to_tensor(), (node,))
    return dag


def compile_dag(native, dag, *, output_nodes=(1,), **options):
    return NativeCoefficientDAG(
        dag, backend=native, output_nodes=output_nodes, **options
    )


def test_exact_model_bindings_group_without_sampled_mass_or_flavour_inference(native):
    from symbolica import E, S

    a, b, heavy = [
        compile_dag(native, make_dag(name))
        for name in ("mass_a", "mass_b", "mass_heavy")
    ]
    raw = coefficient_sharing_groups((a, b, heavy), model_bindings={})
    assert raw.representative_for == (0, 1, 2)
    parameter = S("shared_model_parameter")
    bindings = {
        "python::mass_a": 2 * parameter,
        "python::mass_b": 2 * parameter,
        "python::mass_heavy": S("independent_heavy"),
    }
    groups = coefficient_sharing_groups((a, b, heavy), model_bindings=bindings)
    assert groups.representatives == (0, 2) and groups.representative_for == (0, 0, 2)
    # Equal numbers at a point are irrelevant: these are independent symbols.
    changed = {**bindings, "python::mass_b": parameter}
    assert coefficient_sharing_groups(
        (a, b, heavy), model_bindings=changed
    ).representative_for == (0, 1, 2)
    assert coefficient_sharing_groups(
        (a, b), model_bindings={"python::mass_a": E("0"), "python::mass_b": E("0")}
    ).representative_for == (0, 0)
    with pytest.raises((TypeError, ValueError), match=r"exact|float"):
        coefficient_sharing_groups((a,), model_bindings={"python::mass_a": E("0.1")})


@pytest.mark.parametrize(
    "arithmetic,precision,engine",
    [
        ("double", None, "interpreted"),
        ("double", None, "jit"),
        ("double-double", None, "interpreted"),
        ("arbitrary", 80, "interpreted"),
    ],
)
def test_shared_outputs_recompute_changed_model_inputs_and_preserve_each_graph_position(
    native, arithmetic, precision, engine
):
    from symbolica import S

    cache = native.CoefficientKernelCache(arithmetic, precision, engine)
    executors = tuple(
        compile_dag(
            native,
            make_dag(name),
            arithmetic=arithmetic,
            precision=precision,
            coefficient_engine=engine,
            kernel_cache=cache,
        )
        for name in ("mass_a", "mass_b", "mass_heavy")
    )
    parameter, heavy = S("shared_model_parameter", "independent_heavy")
    groups = coefficient_sharing_groups(
        executors,
        model_bindings={
            "python::mass_a": 2 * parameter,
            "python::mass_b": 2 * parameter,
            "python::mass_heavy": heavy,
        },
    )
    retained = []
    for value in ("0", "1.25", "-3.75"):
        outputs = {}
        for index in groups.representatives:
            executor = executors[index]
            actual = {
                "python::sharing_external": "7",
                "python::mass_a": str(2 * Decimal(value)),
                "python::mass_b": str(2 * Decimal(value)),
                "python::mass_heavy": "13",
            }
            inputs = native.NativeBatch.from_decimal(
                [[actual[p.get_name()], "0"] for p in executor.parameters],
                1,
                len(executor.parameters),
                arithmetic,
                precision,
                [42],
            )
            outputs[index] = executor.evaluate_batch(inputs)
        original = tuple(outputs[i] for i in groups.representative_for)
        assert original[0] is original[1] and original[0] is not original[2]
        coefficients = {
            label.powers[0]: Decimal(pair[0])
            for label, pair in zip(
                executors[0].dag.nodes[1].output_labels,
                original[0].to_decimal(),
                strict=True,
            )
        }
        assert coefficients == {0: Decimal(14), 1: 4 * Decimal(value)}
        assert original[0].row_ids == [42]
        retained.append((original[0], original[0].to_decimal()))
    assert all(output.to_decimal() == snapshot for output, snapshot in retained)
    assert cache.hits > 0


def test_packed_layout_domain_and_precision_prevent_false_sharing(native):
    from symbolica import E

    bindings = {"python::mass_a": E("0")}
    base = compile_dag(native, make_dag("mass_a"))
    variants = [
        compile_dag(native, make_dag("mass_a", external_name="different_external")),
        compile_dag(native, make_dag("mass_a", axis=1)),
        compile_dag(native, make_dag("mass_a", auxiliary=True)),
        compile_dag(native, make_dag("mass_a"), output_nodes=(0,)),
        compile_dag(native, make_dag("mass_a"), output_nodes=(1, 1)),
        compile_dag(native, make_dag("mass_a"), arithmetic="double-double"),
        compile_dag(native, make_dag("mass_a"), coefficient_engine="jit"),
    ]
    altered = make_dag("mass_a")
    altered.domain = "different-certified-domain"
    variants.append(compile_dag(native, altered))
    reordered = make_dag("mass_a")
    final = reordered.nodes[-1]
    reordered.nodes[-1] = replace(
        final, output_labels=tuple(reversed(final.output_labels))
    )
    variants.append(compile_dag(native, reordered))
    for other in variants:
        assert coefficient_sharing_groups(
            (base, other), model_bindings=bindings
        ).representative_for == (0, 1)
    before = base.sharing_key(bindings)
    base.dag.nodes.append(base.dag.nodes[-1])
    assert base.sharing_key(bindings) == before
    p60 = compile_dag(native, make_dag("mass_a"), arithmetic="arbitrary", precision=60)
    p80 = compile_dag(native, make_dag("mass_a"), arithmetic="arbitrary", precision=80)
    assert coefficient_sharing_groups(
        (p60, p80), model_bindings=bindings
    ).representative_for == (0, 1)
