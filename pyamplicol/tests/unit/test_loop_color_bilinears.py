# SPDX-License-Identifier: 0BSD
"""Formal rectangular weights checked against native explicit SU(3) components."""

from itertools import product

import pytest

from pyamplicol.loops.color_bilinears import (
    PhysicalColorSpace,
    SUNColorSpace,
    external_color_tensor,
)


@pytest.fixture(scope="module")
def algebra():
    symbolica = pytest.importorskip("symbolica")
    from symbolica.community import tensor

    return symbolica, tensor


def _dense_overlap(left, right):
    left, right = left.to_tensor().to_dense(), right.to_tensor().to_dense()
    right = right.permute_axes([right.axes.index(axis) for axis in left.axes])
    return sum(left[i].conj() * right[i] for i in range(len(left)))


def test_physical_overlap_exact_complex_entries_and_native_dummy_scoping(algebra):
    s, t = algebra
    space = PhysicalColorSpace()
    generator = t.TensorExpression.color_t(8, 3)
    left = generator("a", "i", "k") * generator("b", "k", "j")
    right = generator("b", "i", "k") * generator("a", "k", "j")
    matrix = space.rectangular([left, right], [left, right, s.E("3i") * right])
    for i, j in product(range(2), range(3)):
        assert matrix[i][j] == _dense_overlap(
            (left, right)[i], (left, right, s.E("3i") * right)[j]
        )
    assert matrix[0][1] == s.E("-2/3")
    assert matrix[0][2] == s.E("-2i")
    relabeled = generator("a", "i", "x") * generator("b", "x", "j")
    assert space.overlap(relabeled, right) == matrix[0][1]
    # The literal coefficient stays three; no model invariant is inferred.
    assert space.overlap(left, 3 * right) == s.E("-2")
    with pytest.raises(ValueError, match="ports differ"):
        space.overlap(left, right.rename_indices({"a": "other"}))
    foreign = t.TensorName("unresolved_physical_color")(t.Representation.coad(8))
    with pytest.raises(NotImplementedError, match="unsupported"):
        space.overlap(foreign("a"), foreign("a"))
    assert space.overlap(left, t.TensorExpression(s.E("0"))) == 0


def test_physical_symmetric_trace_quartic_matches_dense_without_lifting(algebra):
    s, t = algebra
    generator = t.TensorExpression.color_t(8, 3)

    def symmetric(a, b, c):
        return 4 * t.trace(
            t.Representation.cof(3),
            t.FactorProjector.symmetric(
                *(generator(label, t.AUTO, t.AUTO) for label in (a, b, c))
            ),
        )

    left = symmetric("a", "c", "x") * symmetric("b", "d", "x")
    right = symmetric("a", "d", "x") * symmetric("b", "c", "x")
    space = PhysicalColorSpace()
    assert space.overlap(left, right) == s.E("-20/3") == _dense_overlap(left, right)
    assert space.overlap(left, s.E("2i") * right) == s.E("-40i/3")


def test_rectangular_complex_weights_dummy_scoping_and_semantic_dimensions(algebra):
    s, t = algebra
    nc = s.S("color_bilinear_test::Nc", is_real=True)
    space = SUNColorSpace(nc)
    generator = t.TensorExpression.color_t(8, 3)
    # Both sides deliberately reuse the same explicit contracted label. Native
    # dummy scoping must keep each color sum independent before multiplication.
    left = generator("a", "i", "k") * generator("b", "k", "j")
    right = generator("b", "i", "k") * generator("a", "k", "j")
    born = [left, right]
    loops = [left, right, s.E("2i") * right]
    matrix = space.rectangular(born, loops)
    assert len(matrix) == 2 and all(len(row) == 3 for row in matrix)
    for i, j in product(range(2), range(3)):
        assert matrix[i][j].replace(nc, 3) == _dense_overlap(born[i], loops[j])
    assert matrix[0][1] == matrix[1][0].conj()
    assert (matrix[0][2] - s.E("2i") * matrix[0][1]).together() == 0
    assert matrix[0][1] == -(nc**2 - 1) / (4 * nc)
    # An ordinary number is never guessed to mean CA, CF or an external average.
    assert space.overlap(left, s.E("3") * right) == 3 * matrix[0][1]


def test_external_ports_zero_and_unsupported_residuals(algebra):
    s, t = algebra
    nc = s.S("color_bilinear_guard::Nc")
    space = SUNColorSpace(nc)
    generator = t.TensorExpression.color_t(8, 3)
    left = generator("a", "i", "j")
    with pytest.raises(ValueError, match="ports differ"):
        space.overlap(left, generator("b", "i", "j"))
    assert space.overlap(left, t.TensorExpression(s.E("0"))) == 0
    assert space.overlap(
        t.TensorExpression(s.E("1")), t.TensorExpression(s.E("2i"))
    ) == s.E("2i")
    foreign = t.TensorName("unsupported_color_tensor")(t.Representation.coad(8))
    with pytest.raises(NotImplementedError, match="unsupported"):
        space.overlap(foreign("a"), foreign("a"))
    with pytest.raises(TypeError, match="symbolic"):
        SUNColorSpace(s.E("3"))


def test_compact_trace_channels_lift_like_explicit_generator_loops(algebra):
    s, t = algebra
    nc = s.S("compact_color_trace_test::Nc", is_real=True)
    space = SUNColorSpace(nc)
    generator = t.TensorExpression.color_t(8, 3)
    for degree in (2, 3, 4):
        labels = [f"trace_a{i}" for i in range(degree)]
        compact = t.trace(
            t.Representation.cof(3),
            *(generator(label, t.AUTO, t.AUTO) for label in labels),
        )
        explicit = t.TensorExpression(s.E("1"))
        for i, label in enumerate(labels):
            explicit *= generator(label, f"trace_i{i}", f"trace_i{(i + 1) % degree}")
        actual = space.overlap(compact, compact)
        expected = space.overlap(explicit, explicit)
        assert (actual - expected).together() == 0
        assert (space.overlap(compact, explicit) - expected).together() == 0
        if degree == 3:
            assert (actual - (nc**2 - 1) * (nc**2 - 2) / (8 * nc)).together() == 0


def test_crossed_symmetric_trace_products_use_native_projector_expansion(algebra):
    s, t = algebra
    nc = s.S("symmetric_color_trace_test::Nc", is_real=True)
    generator = t.TensorExpression.color_t(8, 3)

    def symmetric(a, b, c):
        # UFO d^{abc} = 4 Tr(T^{(a} T^b T^{c)}), with Tr(Ta Tb)=delta/2.
        return 4 * t.trace(
            t.Representation.cof(3),
            t.FactorProjector.symmetric(
                *(generator(label, t.AUTO, t.AUTO) for label in (a, b, c))
            ),
        )

    left = symmetric("a", "c", "x") * symmetric("b", "d", "x")
    right = symmetric("a", "d", "x") * symmetric("b", "c", "x")
    space = SUNColorSpace(nc)
    value = space.overlap(left, right)
    expected = 32 - s.E("17/2") * nc**2 + nc**4 / 2 - 24 / nc**2
    assert (value - expected).together() == 0
    assert value.replace(nc, 3) == s.E("-20/3") == _dense_overlap(left, right)
    assert space.overlap(right, left) == value
    assert (space.overlap(left, s.E("2i") * right) - s.E("2i") * value).together() == 0


def test_closed_quartic_network_signed_canonicalization_and_complex_factors(algebra):
    s, t = algebra
    nc = s.S("quartic_color_network_test::Nc", is_real=True)
    space = SUNColorSpace(nc)
    a, b, c, d, e, f, g, h = s.S(*(f"quartic_test::{x}" for x in "abcdefgh"))
    ports = s.S(*(f"quartic_test::fund{i}" for i in range(7)))
    generator = t.TensorExpression.color_t(8, 3)
    structure = t.TensorExpression.color_f(8)
    network = s.E("1i") * structure(b, e, h) * structure(f, c, d) * structure(h, c, g)
    for index, label in enumerate((a, b, d, g, a, e, f)):
        network *= generator(label, ports[index], ports[(index + 1) % 7])
    assert not network.axes
    # Independent exact SU(3) matrix contraction, without colour identities or
    # signed canonicalization, fixes the full network's sign and normalization.
    component = network.to_tensor(t.TensorLibrary.hep_lib_atom())[0]
    assert component == s.E("-7/4")
    unit = t.TensorExpression(s.E("1"))
    physical = PhysicalColorSpace()
    assert physical.overlap(unit, network) == component
    assert physical.overlap(s.E("2+3i") * unit, s.E("4-5i") * network) == (
        s.E("2+3i").conj() * s.E("4-5i") * component
    )
    expected = -(nc**2 - 1) * (nc**2 - 2) / 32
    value = space.overlap(unit, network)
    assert (value - expected).together() == 0
    assert value.replace(nc, 3) == component
    left, right = s.E("2+3i"), s.E("4-5i")
    weighted = space.overlap(left * unit, right * network)
    assert (weighted - left.conj() * right * expected).together() == 0
    assert space.overlap(right * network, left * unit) == weighted.conj()


def test_closed_seven_leg_color_network_exact_components_and_conjugation(algebra):
    s, t = algebra
    generator = t.TensorExpression.color_t(8, 3)

    def trace(labels):
        return t.trace(
            t.Representation.cof(3),
            *(generator(label, t.AUTO, t.AUTO) for label in labels),
        )

    # The independent Born-loop color contraction from uubar->ddbarssbarg.
    # Native structural contraction yields these three ordered traces; ordinary
    # symbolic reduction retains higher symmetric invariants. Exact SU3 matrix
    # components supply an oracle without a private trace/Fierz identity.
    network = trace("abc") * trace("ecdf") * trace("eabdf")
    assert not network.axes
    expected = network.to_tensor(t.TensorLibrary.hep_lib_atom())[0]
    assert expected == s.E("-1/9")
    unit = t.TensorExpression(s.E("1"))
    physical = PhysicalColorSpace()
    assert physical.overlap(unit, network) == expected
    left, right = s.E("2+3i"), s.E("4-5i")
    weighted = physical.overlap(left * unit, right * network)
    assert weighted == left.conj() * right * expected
    assert physical.overlap(right * network, left * unit) == weighted.conj()


@pytest.mark.parametrize(
    "incoming,outgoing,born_count",
    [
        (("u", "u~"), ("d", "d~"), 1),
        (("u", "u~"), ("u", "u~"), 2),
        (("u", "d"), ("u", "d"), 1),
    ],
)
def test_native_four_quark_graph_rectangle_and_common_projection(
    algebra, incoming, outgoing, born_count
):
    s, _t = algebra
    if not s.is_licensed():
        pytest.skip("native graph generation requires licensed Symbolica")
    from symbolica.community import hepkit as h

    from pyamplicol.loops.color_projection import prepare_color_projection
    from pyamplicol.loops.graphs import build_loop_process
    from pyamplicol.loops.recurrence import separate_local_color

    process = build_loop_process(
        h.Model.qcd_qed().process(incoming, outgoing),
        selected_corrections=["QCD"],
        perturbative_orders={"QCD": 1},
        born_order_bounds={"QCD": 2, "QED": 0},
        loop_order_bounds={"QCD": 4, "QED": 0},
    )

    def color(graph):
        expression = s.E("1")
        for item in [*graph.native.vertices, *graph.native.internal_edges]:
            _, factor = separate_local_color(item.numerator_expression())
            expression *= factor
        return external_color_tensor(graph, _t.TensorExpression(expression))

    born, loops = tuple(map(color, process.born)), tuple(map(color, process.loops))
    nc = s.S("physical_color_test::Nc", is_real=True)
    matrix = SUNColorSpace(nc).rectangular(born, loops)
    assert len(matrix) == born_count and len(matrix[0]) == len(process.loops)
    for left, row in zip(born, matrix, strict=True):
        for weight, loop in zip(row, loops, strict=True):
            assert weight.replace(nc, 3) == _dense_overlap(left, loop)
    weights = [weight for row in matrix for weight in row]
    assert any(weight != weight.conj() for weight in weights)
    # These are unaveraged bare graph weights. Their common structural degree
    # is 3; fermion-loop degree-2 terms must remain at NLC, with fixed Nf.
    projection = prepare_color_projection(nc, [[value] for value in weights])
    assert projection.leading_degree == 3
    lc, nlc = projection.project("lc"), projection.project("nlc")
    assert any(a == 0 and b != 0 for a, b in zip(lc, nlc, strict=True))
    assert projection.project("full") == tuple(value.expand() for value in weights)
