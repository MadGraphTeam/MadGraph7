# SPDX-License-Identifier: 0BSD
"""Independent formal-Nc controls; no production model certification implied."""

import pytest

from pyamplicol.loops.color_projection import prepare_color_projection

pytestmark = pytest.mark.loop_fast


@pytest.fixture(scope="module")
def symbols():
    symbolic = pytest.importorskip("symbolica")
    return symbolic.S("Nc", "Nf", "A", "B", "C", "D")


def test_fixed_nf_and_odd_suppression_powers(symbols):
    nc, nf, a, b, c, d = symbols
    full = a * nc**5 + b * nf * nc**4 + c * nc**3 + d * nc**2
    plan = prepare_color_projection(nc, [[full]])
    assert plan.leading_degree == 5
    assert plan.project("lc") == (a * nc**5,)
    assert plan.project("nlc") == (a * nc**5 + b * nf * nc**4 + c * nc**3,)
    assert plan.project("full") == (full,)


def test_common_degree_across_helicity_and_laurent_components(symbols):
    nc, _, a, b, c, _ = symbols
    plan = prepare_color_projection(nc, [[a * nc**5], [b * nc**3], [c * nc], []])
    assert plan.project("lc") == (a * nc**5, 0, 0, 0)
    assert plan.project("nlc") == (a * nc**5, b * nc**3, 0, 0)
    # A numerical Born/helicity zero cannot select a new leading degree.
    assert tuple(value.replace(a, 0) for value in plan.project("lc")) == (0, 0, 0, 0)


def test_exact_cancellations_and_hidden_counterterm_invariants(symbols):
    from symbolica import S

    nc, _, a, b, _, _ = symbols
    cf, ca, ncol = S("CF", "CA", "Ncol")
    plan = prepare_color_projection(
        nc,
        [[a * ncol**4, -a * nc**4, b * cf * ca]],
        substitutions=((cf, (nc**2 - 1) / (2 * nc)), (ca, nc), (ncol, nc)),
    )
    assert plan.leading_degree == 2
    assert plan.project("lc") == (b * nc**2 / 2,)
    assert (plan.project("nlc")[0] - b * (nc**2 - 1) / 2).expand() == 0
    assert plan.project("nlc")[0].replace(nc, 3) == 4 * b


def test_negative_degrees_and_structural_zeros(symbols):
    nc, _, a, b, c, _ = symbols
    plan = prepare_color_projection(nc, [[a / nc + b / nc**2 + c / nc**3]])
    assert plan.leading_degree == -1
    assert plan.project("lc") == (a / nc,)
    assert plan.project("nlc") == plan.project("full")
    zero = prepare_color_projection(nc, [[], [a, -a]])
    assert zero.leading_degree is None
    assert zero.project("lc") == zero.project("nlc") == (0, 0)


def test_semantic_alias_order_does_not_change_projection(symbols):
    from graphlib import CycleError

    from symbolica import S

    nc = symbols[0]
    ca, cf = S("CA", "CF")
    identities = ((ca, nc), (cf, (ca**2 - 1) / (2 * ca)))
    forward = prepare_color_projection(nc, [[ca * cf]], substitutions=identities)
    reverse = prepare_color_projection(nc, [[ca * cf]], substitutions=identities[::-1])
    assert forward == reverse
    assert forward.leading_degree == 2
    assert forward.project("lc") == (nc**2 / 2,)
    with pytest.raises(CycleError):
        prepare_color_projection(nc, [[ca]], substitutions=((ca, cf), (cf, ca)))
    with pytest.raises(ValueError, match="conflicting"):
        prepare_color_projection(nc, [[ca]], substitutions=((ca, nc), (ca, 2 * nc)))


def test_non_laurent_dependence_and_numeric_invariant_guessing_fail(symbols):
    from symbolica import E, Expression

    nc = symbols[0]
    for expression in [1 / (nc + 1), nc ** E("1/2"), nc.sin()]:
        with pytest.raises(ValueError, match="Laurent"):
            prepare_color_projection(nc, [[expression]])
    with pytest.raises(TypeError, match="semantic"):
        prepare_color_projection(nc, [[nc]], substitutions=((E("3"), nc),))
    with pytest.raises(ValueError, match="before projection"):
        prepare_color_projection(nc, [[nc]], substitutions=((nc, E("3")),))
    with pytest.raises(TypeError, match="symbolic"):
        prepare_color_projection(nc, [[1.0]])
    with pytest.raises(ValueError, match="floating"):
        prepare_color_projection(nc, [[Expression.num(0.1) * nc]])
    with pytest.raises(ValueError):
        prepare_color_projection(nc, [[nc]]).project("tree-lc")
