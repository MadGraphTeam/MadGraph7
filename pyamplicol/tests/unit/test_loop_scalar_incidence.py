# SPDX-License-Identifier: 0BSD
"""Local dependency algebra proves scalar alias routes without graph expansion."""

from __future__ import annotations

import pytest

from pyamplicol.loops.kernels import CoefficientDAG
from pyamplicol.loops.scalar_incidence import (
    scalar_alias_incidence,
    scalar_grade_term_mask,
)

pytestmark = pytest.mark.loop_fast


@pytest.fixture
def primitives():
    from symbolica import S
    from symbolica.community import tensor as t

    aliases = S("incidence_test::g1", "incidence_test::g2")
    dag = CoefficientDAG(tuple(S(f"incidence_test::q{i}") for i in range(4)))

    def add(name, expression, dependencies=()):
        return dag.add(name, t.TensorExpression(expression).to_tensor(), dependencies)

    return dag, aliases, add


def test_two_aliases_one_term_and_shared_ordinary_current(primitives):
    from symbolica import S

    dag, (g1, g2), add = primitives
    x = S("incidence_test::x")
    ordinary = add("shared ordinary current", x)
    mixed = add("two aliases", 3 * g1 + 2 * g2, (ordinary,))
    single = add("second colour term", g2, (ordinary,))
    carried = add("independent bridge", x + 1, (mixed,))
    assert scalar_alias_incidence(dag, (mixed, single, carried, mixed), (g1, g2)) == (
        frozenset((0, 1)),
        frozenset((1,)),
        frozenset((0, 1)),
        frozenset((0, 1)),
    )


def test_exact_local_cancellations_and_zero_outputs(primitives):
    from symbolica import E

    dag, (g1, g2), add = primitives
    cancelled = add("cancelled quadratic terms", g1 * g2 - g2 * g1 + g2)
    nonlinear = add("discarded nonlinear branch", g1**2)
    zero = add("zero local contraction", E("0"), (nonlinear,))
    assert scalar_alias_incidence(dag, (cancelled, zero), (g1, g2)) == (
        frozenset((1,)),
        frozenset(),
    )


@pytest.mark.parametrize("kind", ["constant", "square", "product", "propagated"])
def test_contamination_requires_a_constructive_linearity_proof(primitives, kind):
    dag, (g1, g2), add = primitives
    if kind == "propagated":
        first = add("first insertion", g1)
        output = add("second insertion", g2, (first,))
    else:
        output = add(
            "contamination",
            {"constant": g1 + 1, "square": g1**2, "product": g1 * g2}[kind],
        )
    with pytest.raises(NotImplementedError, match="degree-one alias proof"):
        scalar_alias_incidence(dag, (output,), (g1, g2))


@pytest.mark.parametrize("kind", ["inverse", "function"])
def test_nonpolynomial_alias_dependence_is_explicit(primitives, kind):
    dag, (g1, g2), add = primitives
    output = add("nonpolynomial source", 1 / g1 if kind == "inverse" else g1.log())
    with pytest.raises(
        NotImplementedError, match=r"nonpolynomial source.*not polynomial"
    ):
        scalar_alias_incidence(dag, (output,), (g1, g2))


def test_grade_mask_never_drops_other_alias_in_same_tensor_term(primitives):
    from symbolica import S

    from pyamplicol.loops.color_projection import prepare_color_projection

    dag, (g1, g2), add = primitives
    first = add("mixed", g1 + g2)
    second = add("only g2", g2)
    scalar_alias_incidence(dag, (first, second), (g1, g2))
    nc = S("incidence_test::Nc")
    # A degree-one scalar is present only in g1. The second term's high-degree
    # overlap must not enter the structural leading degree of this grade.
    from symbolica import E

    active = scalar_grade_term_mask(dag, (first, second), (g1, g2), (E("1"), E("0")))
    rows = [
        [nc * weight if keep else E("0")]
        for weight, keep in zip((nc**2 - 1, nc**4), active, strict=True)
    ]
    projection = prepare_color_projection(nc, rows)
    assert projection.leading_degree == 3
    assert projection.project("lc") == (nc**3, E("0"))
    assert scalar_grade_term_mask(dag, (first, second), (g1, g2), (E("0"), E("1"))) == (
        True,
        True,
    )


def test_exact_grade_cancellation_cannot_set_leading_colour_degree(primitives):
    from symbolica import E, S

    from pyamplicol.loops.color_projection import prepare_color_projection

    dag, (g1, g2), add = primitives
    current = add("ordinary shared current", S("incidence_test::J"))
    cancelled = add("two cancelling aliases", g1 + g2, (current,))
    propagated = add("bridge", E("2"), (cancelled,))
    retained = add("lower-colour contribution", g1, (current,))
    scalar_alias_incidence(dag, (propagated, retained), (g1, g2))
    mask = scalar_grade_term_mask(
        dag, (propagated, retained), (g1, g2), (E("1"), E("-1"))
    )
    assert mask == (False, True)
    nc = S("incidence_test::Nc")
    projection = prepare_color_projection(
        nc,
        [
            [weight if active else E("0")]
            for weight, active in zip((nc**5, nc**2), mask, strict=True)
        ],
    )
    assert projection.leading_degree == 2
    assert projection.project("lc") == projection.project("nlc") == (E("0"), nc**2)


def test_equal_local_components_share_reference_for_downstream_cancellation(primitives):
    from symbolica import E, S
    from symbolica.community import tensor as t

    dag, (g1, g2), _ = primitives
    x = S("incidence_test::J")
    slot = t.Representation.mink(4)(S("incidence_test::mu"))
    vector = dag.add(
        "aliased vector",
        t.Tensor.dense(
            t.TensorName("incidence_test::vector")(slot),
            [g1 * x, g2 * x, E("0"), E("0")],
        ),
    )
    output = dag.add(
        "Minkowski difference",
        t.Tensor.dense(
            t.TensorName("incidence_test::projector")(slot.dual()),
            [E("1"), E("1"), E("0"), E("0")],
        ),
        (vector,),
    )
    scalar_alias_incidence(dag, (output,), (g1, g2))
    assert scalar_grade_term_mask(dag, (output,), (g1, g2), (E("1"), E("1"))) == (
        False,
    )
    assert scalar_grade_term_mask(dag, (output,), (g1, g2), (E("1"), E("2"))) == (True,)


@pytest.mark.parametrize("kind", ["complex_constants", "reference_collision"])
def test_local_grade_references_preserve_constants_and_input_identity(primitives, kind):
    from symbolica import AtomType, E, S
    from symbolica.community import tensor as t

    dag, (g1, g2), _ = primitives
    slot = t.Representation.mink(4)(S("incidence_test::mu"))
    if kind == "complex_constants":
        assert E("1i").get_type() == E("1+1i").get_type() == AtomType.Num
        components, covector = (g1, g2), (E("1"), E("1+1i") / E("1i"))
        coefficients, expected = (E("1+1i"), E("1i")), False
    else:
        x = S("loop_grade_incidence::current_0")
        components, covector = (g1 * (x + 1), g1 * x), (E("1"), E("1"))
        coefficients, expected = (E("1"), E("1")), True
    vector = dag.add(
        "phase-related components",
        t.Tensor.dense(
            t.TensorName("incidence_test::vector")(slot), [*components, E("0"), E("0")]
        ),
    )
    output = dag.add(
        "constant local contraction",
        t.Tensor.dense(
            t.TensorName("incidence_test::projector")(slot.dual()),
            [*covector, E("0"), E("0")],
        ),
        (vector,),
    )
    scalar_alias_incidence(dag, (output,), (g1, g2))
    assert scalar_grade_term_mask(dag, (output,), (g1, g2), coefficients) == (expected,)
