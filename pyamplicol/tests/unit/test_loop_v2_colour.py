# SPDX-License-Identifier: 0BSD
"""Generic colour seeds (design §3.4, §7.1-§7.5, WP1.4).

Oracles: exact colour algebra, the seed FFT against the direct contraction
and the removed legacy engine's frozen leading-colour degrees; the seed
colour weights are checked at the value level against that engine's frozen
family numerators (``test_loop_v2_slice``).
"""

from __future__ import annotations

import dataclasses
import itertools
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

import numpy as np
import pytest

from pyamplicol.color.connections import (
    ColorLeg,
    ColorTensor,
    ConnectedColorTensor,
    ExactColorCoefficient,
    TensorTerm,
    contract_connected_tensor_nc_terms,
)
from pyamplicol.loops.v2.blocks import block_colour_flow, build_tree_blocks
from pyamplicol.loops.v2.colour import (
    BornBasis,
    ColourProduct,
    ColourString,
    Primitive,
    SeedColour,
    _adjoint_chain,
    _exact_number,
    _PrimitiveCells,
    block_colour_network,
    born_basis,
    build_seed_table,
    canonical_primitive,
    certify_leading_colour,
    decompose_vertex,
    face_count,
    multiply,
    project,
    reduce_network,
    seed_fourier,
    spenso_colour_network,
)
from pyamplicol.loops.v2.families import (
    LoopOrderSelection,
    ModelLoopCatalogue,
    add_orders,
    generate_families,
)
from pyamplicol.loops.v2.skeletons import DressedColour, family_skeletons
from tools.loop_reference import legacy_capture

from .loop_v2_bdk import BDK_PREFACTOR, bdk_bracket, bdk_structures, laurent_times

ZERO = ExactColorCoefficient()
# Design §16.3, strong per-family oracle O2.
LEG = re.compile(r"physical_color_leg_(\d+)$")


# ------------------------------------------------------------------ dense oracle


@pytest.fixture(scope="module")
def spenso():
    """Spenso's tensor library and its dense SU(3) generators t^a_ij."""

    from symbolica import E
    from symbolica.community import tensor as t

    library = t.TensorLibrary.hep_lib_atom()
    generators = _numeric(
        t.TensorExpression(
            E(
                "spenso::t(spenso::coad(8,x::a),spenso::cof(3,x::i),"
                "spenso::dind(spenso::cof(3,x::j)))"
            )
        )
        .to_tensor(library)
        .to_dense()
    )
    return library, generators


def _numeric(tensor):
    return tensor.map_components(
        lambda value: (
            complex(value.evaluate({}))
            if hasattr(value, "evaluate")
            else complex(value)
        ),
        dtype=complex,
    ).to_numpy()


def _dense(tensor: ConnectedColorTensor, axes, generators) -> np.ndarray:
    """Components of literal structures: OpenColorString(q, w, qbar) is
    (tau^w_k .. tau^w_1)[qbar, q] and a trace (w) is tr(tau^w_k .. tau^w_1)."""

    tau = np.sqrt(2) * generators
    dimensions = {leg.label: 8 if leg.representation == 8 else 3 for leg in tensor.legs}
    out = np.zeros([dimensions[axis] for axis in axes], dtype=complex)
    for index in itertools.product(*(range(dimensions[axis]) for axis in axes)):
        value = dict(zip(axes, index, strict=True))
        total = 0j
        for term in tensor.terms:
            product = complex(term.coefficient) * 3.0**term.nc_power
            for string in term.tensor.open_strings:
                matrix = np.eye(3)
                for label in reversed(string.adjoint_labels):
                    matrix = matrix @ tau[value[label]]
                product *= matrix[
                    value[string.antifundamental_label], value[string.fundamental_label]
                ]
            for trace in term.tensor.traces:
                matrix = np.eye(3)
                for label in reversed(trace):
                    matrix = matrix @ tau[value[label]]
                product *= np.trace(matrix)
            total += product
        out[index] = total * 2.0 ** (-tensor.normalization_power / 2)
    return out


SPENSO_CASES = [
    ("f(coad(8,e::a),coad(8,e::b),coad(8,e::c))", {"a": 8, "b": 8, "c": 8}),
    ("t(coad(8,e::a),cof(3,e::i),dind(cof(3,e::j)))", {"a": 8, "i": -3, "j": 3}),
    (
        "chain(cof(3,e::i),dind(cof(3,e::j)),t(coad(8,e::a),in,out),"
        "t(coad(8,e::b),in,out))",
        {"a": 8, "b": 8, "i": -3, "j": 3},
    ),
    # internal adjoint and fundamental contractions, metrics, a closed loop
    (
        "f(coad(8,e::a),coad(8,e::b),coad(8,e::x))"
        "*t(coad(8,e::y),cof(3,e::i),dind(cof(3,e::k)))"
        "*g(dind(cof(3,e::m)),cof(3,e::k))"
        "*t(coad(8,e::z),cof(3,e::m),dind(cof(3,e::j)))"
        "*g(coad(8,e::x),coad(8,e::y))*f(coad(8,e::z),coad(8,e::c),coad(8,e::d))",
        {"a": 8, "b": 8, "c": 8, "d": 8, "i": -3, "j": 3},
    ),
    (
        "t(coad(8,e::a),cof(3,e::k),dind(cof(3,e::l)))"
        "*t(coad(8,e::b),cof(3,e::l),dind(cof(3,e::k)))*1/3*𝑖",  # noqa: RUF001
        {"a": 8, "b": 8},
    ),
]


def _spenso(expression: str) -> str:
    return re.sub(r"\b(f|t|g|chain|coad|cof|dind|in|out)\b", r"spenso::\1", expression)


@pytest.mark.loop_fast
@pytest.mark.parametrize(
    ("expression", "representations"),
    SPENSO_CASES,
    ids=["f", "t", "chain", "contracted", "closed-loop"],
)
def test_spenso_colour_lowering_matches_dense_components(
    spenso, expression, representations
):
    from symbolica import E
    from symbolica.community import tensor as t

    library, generators = spenso
    expression = _spenso(expression)
    labels = {name: index + 1 for index, name in enumerate(representations)}
    legs = [ColorLeg(labels[name], rep) for name, rep in representations.items()]
    network = spenso_colour_network(expression)
    external = {f"e::{name}": label for name, label in labels.items()}
    reduced = reduce_network(network, legs, external)
    tensor = t.TensorExpression(E(expression)).to_tensor(library)
    axes = [labels[re.search(r"(\w+)\)+$", str(axis)).group(1)] for axis in tensor.axes]
    expected = _numeric(tensor.to_dense())
    assert np.abs(_dense(reduced, axes, generators) - expected).max() < 1e-14


@pytest.mark.loop_fast
@pytest.mark.parametrize(
    ("word", "field"),
    [
        ((3, 1, 2, 5, 6), 3),
        ((4, 3, 5), 8),
        ((1, 3, 4, 2), 8),
        ((1, 2, 4, 3), 8),
        ((2, 1), 1),
        ((5,), 8),
        ((1,), -3),
    ],
)
def test_block_words_lower_in_the_blocks_convention(spenso, word, field):
    """A string s -> a_1..a_k -> e is (tau^a_k .. tau^a_1)[e, s] and an adjoint
    field closes with tau^X[in, out] (``loops.v2.blocks``), as checked against
    FeynKit tree sums in ``test_loop_v2_blocks``."""

    _library, generators = spenso
    representations = {1: -3, 2: 3, 3: 3, 4: -3, 5: 8, 6: 1}
    flow = block_colour_flow(word, representations, field)
    x = 9
    network = block_colour_network(flow, representations, field, x)
    used = {label for label in word if representations[label] != 1}
    legs = [ColorLeg(label, representations[label]) for label in sorted(used)]
    if field != 1:
        legs.append(ColorLeg(x, -field if abs(field) == 3 else 8))
    reduced = reduce_network(network, legs, {leg.label: leg.label for leg in legs})
    axes = [leg.label for leg in legs]

    # The convention restated with dense generators.
    tau = np.sqrt(2) * generators
    dims = [
        8 if (representations.get(a) == 8 or (a == x and field == 8)) else 3
        for a in axes
    ]
    expected = np.zeros(dims, dtype=complex)
    for index in itertools.product(*(range(d) for d in dims)):
        value = dict(zip(axes, index, strict=True))
        total = 0j
        for opens in itertools.product(range(3), repeat=2):
            product = 1.0 + 0j
            for labels in flow.strings:
                start = (
                    value[labels[0]]
                    if representations[labels[0]] == 3
                    else (value[x] if field == -3 else opens[0])
                )
                end = (
                    value[labels[-1]]
                    if representations[labels[-1]] == -3
                    else (value[x] if field == 3 else opens[1])
                )
                matrix = np.eye(3)
                for label in reversed(
                    [label for label in labels if representations[label] == 8]
                ):
                    matrix = matrix @ tau[value[label]]
                product *= matrix[end, start]
            if field == 8:
                product *= tau[value[x]][opens[0], opens[1]]
            elif opens != (0, 0):
                continue  # no open adjoint ends to sum over
            total += product
        expected[index] = total
    assert np.abs(_dense(reduced, axes, generators) - expected).max() < 1e-14


@pytest.mark.loop_fast
def test_fierz_reduction_and_face_counts():
    legs = [ColorLeg(1, 3), ColorLeg(2, -3)]
    # sum_a (tau^a tau^a)^2_1 = (N - 1/N) delta
    casimir = reduce_network(
        (ColourProduct(strings=(ColourString(2, ("a", "a"), 1),)),), legs, {1: 1, 2: 2}
    )
    assert {term.nc_power: term.coefficient for term in casimir.terms} == {
        1: ExactColorCoefficient(1),
        -1: ExactColorCoefficient(-1),
    }
    # sum_ab tr(tau^a tau^b) tr(tau^a tau^b) = N^2 - 1, a closed network
    closed = reduce_network((ColourProduct(traces=(("a", "b"), ("a", "b"))),), [], {})
    assert {term.nc_power: term.coefficient.real for term in closed.terms} == {
        2: 1,
        0: -1,
    }
    single = ColorTensor(traces=((1, 2, 3, 4),))
    assert face_count(single, single) == 4
    assert face_count(single, ColorTensor(traces=((1, 2, 4, 3),))) == 2
    assert face_count(single, ColorTensor(traces=((1, 2), (3, 4)))) == 3


# ------------------------------------------------- certified compressions


def _ring_network(kind: str, word) -> tuple:
    """Tr_adj(F^w_1 ... F^w_n) as a chain of F's, or tr(tau^w)."""

    if kind == "fundamental-trace":
        return (ColourProduct(strings=(ColourString("x", tuple(word), "x"),)),)
    chain = _adjoint_chain(word, "x", "y", "ring")
    return multiply(chain, (ColourProduct(traces=(("x", "y"),)),))  # delta_xy


@pytest.mark.loop_fast
@pytest.mark.parametrize("kind", ["adjoint-ring", "fundamental-trace"])
@pytest.mark.parametrize("word", [(1, 2), (1, 2, 3), (2, 1, 3), (1, 2, 3, 4)])
def test_primitive_expansions_equal_their_networks(kind, word):
    """Primitive.un_terms (the gl(N) trace of ad products) and the cycle-count
    cells equal the exact SU(N_c) reduction of the generator networks."""

    legs = [ColorLeg(label, 8) for label in sorted(word)]
    external = {leg.label: leg.label for leg in legs}
    primitive = Primitive(kind, word)
    expected = reduce_network(_ring_network(kind, word), legs, external)
    assert primitive.tensor(legs) == expected
    canonical, sign = canonical_primitive(kind, word)
    assert canonical.tensor(legs).terms == tuple(
        TensorTerm(t.coefficient.scaled(sign), t.tensor, t.nc_power)
        for t in expected.terms
    )
    # Cells by cycle counting (adjoint) or exact contraction (fundamental),
    # relabelling cache included, against exact SU(N_c) contractions.
    bras = [
        ConnectedColorTensor(legs, (TensorTerm(ExactColorCoefficient(1, 1), tensor),))
        for tensor in (
            ColorTensor(traces=(tuple(sorted(word)),)),
            ColorTensor(traces=(tuple(reversed(word)),)),
            *(
                [ColorTensor(traces=(tuple(sorted(word))[:2], tuple(sorted(word))[2:]))]
                if len(word) == 4
                else []
            ),
        )
    ]
    basis = BornBasis(
        tuple(legs), tuple(range(len(bras))), tuple(bras), (frozenset(),) * len(bras)
    )
    cells = _PrimitiveCells(basis)
    for bra, cell in zip(bras, cells(primitive), strict=True):
        assert cell == contract_connected_tensor_nc_terms(bra, expected)


def _rational(expression: str) -> tuple:
    """A Spenso colour network with its 1/sqrt2 factors removed (the seeds'
    normalisation: vertex networks absorb 2^(p/2))."""

    return tuple(
        ColourProduct(p.coefficient, p.strings, p.traces, p.nc_power)
        for p in spenso_colour_network(_spenso(expression))
    )


@pytest.mark.loop_fast
def test_vertex_decompositions_are_exact_identities():
    """f^{x b y} is (F^b)_xy up to a constant, products of two f's are F
    products, T^b is a tau string, and a symmetric d-like structure is no
    Lie-type product (refused)."""

    f = _rational("f(coad(8,e::x),coad(8,e::b),coad(8,e::y))")
    found = decompose_vertex(f, "e::x", "e::y", ["e::b"], "adjoint-ring")
    assert found is not None
    ((c, order),) = found.terms
    assert order == (0,) and c == ExactColorCoefficient(0, -1)

    ff = _rational(
        "f(coad(8,e::x),coad(8,e::a),coad(8,e::z))"
        "*f(coad(8,e::z),coad(8,e::b),coad(8,e::y))"
    )
    found = decompose_vertex(ff, "e::x", "e::y", ["e::a", "e::b"], "adjoint-ring")
    assert found is not None and [order for _c, order in found.terms] == [(0, 1)]
    commutator = _rational(
        "f(coad(8,e::x),coad(8,e::y),coad(8,e::z))"
        "*f(coad(8,e::z),coad(8,e::a),coad(8,e::b))"
    )
    found = decompose_vertex(
        commutator, "e::x", "e::y", ["e::a", "e::b"], "adjoint-ring"
    )
    assert found is not None
    (c01, o01), (c10, o10) = found.terms
    assert (o01, o10) == ((0, 1), (1, 0)) and c01 == c10.scaled(-1)

    t = _rational("t(coad(8,e::b),cof(3,e::i),dind(cof(3,e::j)))")
    found = decompose_vertex(t, "e::j", "e::i", ["e::b"], "fundamental-trace")
    assert found is not None and found.forward and len(found.terms) == 1
    reversed_ = decompose_vertex(t, "e::i", "e::j", ["e::b"], "fundamental-trace")
    assert reversed_ is not None and not reversed_.forward

    symmetric = (
        ColourProduct(traces=(("x", "b", "y"),)),
        ColourProduct(traces=(("x", "y", "b"),)),
    )
    assert decompose_vertex(symmetric, "x", "y", ["b"], "adjoint-ring") is None


@pytest.mark.loop_fast
def test_bdk_five_point_colour_sum_o14():
    """O14 (design §7.6, §14): the exact-N_c colour sum of qbar q g g g,
    sum_colours A_tree^* A_loop, from v2's exact contractions of the BDK
    colour structures equals hep-ph/9409393 eq. NLOCrossSection as Laurent
    polynomials in N_c, entry by entry: for tree ordering s and every
    one-loop partial amplitude, the coefficient is (N^2-1)/N times BDK's
    bracket. It also fixes the NLC classification: relative to the leading
    (N^2-1)^2 A_{5;1}(s), the coefficients of degree P-2 and P-4 are
    exactly those of the bracket's N^2 and N^0 parts."""

    tree, loop = bdk_structures()
    for s in tree:
        bracket = bdk_bracket(s)
        for key, structure in loop.items():
            mine = contract_connected_tensor_nc_terms(tree[s], structure)
            expected = laurent_times(bracket.get(key, {}), BDK_PREFACTOR)
            assert {p: v.real for p, v in mine.items()} == expected, (s, key)
            assert all(not v.imag for v in mine.values())


# ------------------------------------------------------------------ slice setup


@dataclass
class V2Slice:
    """The v2 generation objects of one control process."""

    control: Any
    process: Any
    data: Any
    families: Any
    skeletons: dict
    colour: SeedColour
    table: Any
    class_of: dict
    born_order: tuple
    loop_order: tuple
    catalogue: Any
    vertex_terms: tuple

    @property
    def pairs(self):
        return frozenset({(self.born_order, self.loop_order)})


@pytest.fixture(scope="module")
def v2_slice(loop_sm_compiled_model):
    from pyamplicol.loops.compiler import prepare_loop_request_model
    from pyamplicol.models.external import CompiledUFOModel
    from pyamplicol.models.loading import load_compiled_model
    from pyamplicol.processes.model import build_model_process_ir

    compiled = load_compiled_model(loop_sm_compiled_model)
    built = {}

    def build(name: str) -> V2Slice:
        if name in built:
            return built[name]
        control = legacy_capture.CONTROLS[name]
        structure = prepare_loop_request_model(
            compiled,
            born_order_bounds=control.born_order_bounds,
            loop_order_bounds=control.loop_order_bounds,
            fixed_parameters=legacy_capture.FIXED_PARAMETERS,
        ).structure
        data = json.loads(structure.native.to_json())
        catalogue = ModelLoopCatalogue(
            data, species_orders=structure.perturbative_species_orders
        )
        born = add_orders(tuple(control.born_order_bounds.items()))
        loop = add_orders(tuple(control.loop_order_bounds.items()))
        process = build_model_process_ir(
            control.process, compiled.ir, color_accuracy="full"
        )
        blocks = build_tree_blocks(
            process,
            CompiledUFOModel(compiled),
            born_coupling_orders=control.born_order_bounds,
            block_coupling_orders=control.loop_order_bounds,
        )
        families = generate_families(
            blocks.block_table(data),
            catalogue,
            LoopOrderSelection(frozenset({"QCD"}), frozenset({loop})),
        )
        colour = SeedColour(blocks, data, compiled.ir.vertex_terms)
        skeletons = {
            key: family_skeletons(families, family, colour, catalogue)
            for key, family in families.families.items()
        }
        table, class_of = build_seed_table(
            colour, [s for rows in skeletons.values() for s in rows]
        )
        built[name] = V2Slice(
            control,
            process,
            data,
            families,
            skeletons,
            colour,
            table,
            class_of,
            born,
            loop,
            catalogue,
            tuple(compiled.ir.vertex_terms),
        )
        return built[name]

    return build


def _skeleton_list(v2: V2Slice) -> list:
    return [s for rows in v2.skeletons.values() for s in rows]


@pytest.mark.loop_physics
@pytest.mark.parametrize(
    ("name", "kinds"),
    [
        ("gg_gg", {"adjoint-ring": 3, "fundamental-trace": 6}),
        ("uux_ggg", {"quark-ring": None}),
    ],
)
def test_compressions_are_certified(v2_slice, name, kinds):
    """Certified compressions (design §3.4, §7.2): every compressed class is
    certified by exact vertex decompositions, Lie-type tree vertices and the
    smallest single-leg instance, whose generic seed equals its compressed
    one; gg -> gg collapses to the (n-1)!/2 adjoint rings and (n-1)! closed
    fundamental traces. The physics (family numerators, components) is
    checked against the legacy engine with compression on (slice O1/O2)."""

    from pyamplicol.loops.v2.colour import COMPRESSION_THEOREMS, Primitive

    v2 = v2_slice(name)
    table, class_of = build_seed_table(v2.colour, _skeleton_list(v2), compress=True)
    found = {}
    for seed in table.classes:
        kind = seed.kind if isinstance(seed, Primitive) else "generic"
        found[kind] = found.get(kind, 0) + 1
    assert set(table.certificates) == set(kinds)
    for kind, count in kinds.items():
        if count is not None:
            assert found[kind] == count
        certificate = table.certificates[kind]
        assert certificate.theorem == COMPRESSION_THEOREMS[kind]
        assert certificate.vertices and certificate.tree_vertices
        assert {rule for rule, _index in certificate.tree_vertices} >= {"V_3", "V_4"}
    # Every skeleton is seeded: by its compressed combination or generically.
    assert set(class_of) == set(_skeleton_list(v2))
    assert all(class_of.values())


@pytest.mark.loop_physics
def test_quark_rings_reduce_once_per_relabelling(v2_slice):
    """Quark-ring tensors and cells come from one reduction per relabelling
    class of their external adjoint labels; every class of uu~ -> ggg equals
    its direct SU(N_c) reduction and contraction, and the 93 rings need
    fewer reductions than classes."""

    from pyamplicol.loops.v2.colour import Primitive, _ring_relabelling

    v2 = v2_slice("uux_ggg")
    table, _class_of = build_seed_table(v2.colour, _skeleton_list(v2), compress=True)
    legs = table.basis.legs
    adjoint = frozenset(leg.label for leg in legs if leg.representation == 8)
    external = {leg.label: leg.label for leg in legs}
    rings = [
        (c, seed)
        for c, seed in enumerate(table.classes)
        if isinstance(seed, Primitive) and seed.kind == "quark-ring"
    ]
    assert len(rings) == 93
    canonical = set()
    for c, seed in rings:
        direct = reduce_network(seed.network(), legs, external)
        assert seed.tensor(legs) == direct
        assert table.cells[c] == tuple(
            contract_connected_tensor_nc_terms(bra, direct)
            for bra in table.basis.tensors
        )
        canonical.add(_ring_relabelling(seed, adjoint)[0])
    assert len(canonical) < len(rings)


def _born_rows(table, count: int, grades=(0,)) -> list[dict]:
    """Deterministic, irregular binary64 Born rows over the whole basis."""

    rows = []
    for r in range(count):
        row = {}
        for b, orders in enumerate(table.basis.orders):
            for order in orders:
                row[b, order] = {
                    grade: complex(
                        math.sin(1.7 * b + 0.3 * r + grade + 0.1),
                        math.cos(2.3 * b - 0.7 * r + 0.5 * grade),
                    )
                    for grade in grades
                }
        rows.append(row)
    return rows


def _assert_fft_equals_direct(table, fourier, rows, accuracy, native) -> None:
    """The FFT seeds equal the direct contraction to binary64 rounding, per
    (Born order, loop order) pair."""

    born_orders = {a for orders in table.basis.orders for a in orders}
    loop_orders = {o for orders in table.orders for o in orders}
    for pairs in (frozenset({(a, o)}) for a in born_orders for o in loop_orders):
        degree = None
        if accuracy != "full":
            grades = {key: tuple(value) for key, value in rows[0].items()}
            degree = table.degree(pairs, grades)
        fast = fourier.seeds(table, pairs, rows, accuracy, native, degree=degree)
        for row, seeds in zip(rows, fast, strict=True):
            direct = table.seeds(pairs, row, accuracy, cast=complex, degree=degree)
            scale = max(map(abs, direct.values()))
            assert scale > 0
            for key in set(direct) | set(seeds):
                error = abs(seeds.get(key, 0) - direct.get(key, 0))
                assert error <= 1e-13 * scale, (key, accuracy, error / scale)


@pytest.mark.loop_physics
@pytest.mark.parametrize(
    ("name", "orbits"),
    [("gg_gg", (1, 2, 3)), ("uux_ggg", (1, 19, 3))],
)
def test_seed_fft_equals_direct_contraction(v2_slice, loop_native, name, orbits):
    """Seed FFT (design §7.3, WP2.4 acceptance "FFT = direct"): the
    certified symmetric-group orbits of the compressed seed table (adjoint
    rings, closed fundamental traces, quark rings) contracted by the native
    S_m FFT equal the direct contraction at full colour and NLC, Born
    grades included. Each orbit pair is certified on every exact cell."""

    v2 = v2_slice(name)
    table, _class_of = build_seed_table(v2.colour, _skeleton_list(v2), compress=True)
    fourier = seed_fourier(table)
    assert fourier is not None
    assert (
        len(fourier.born_orbits),
        len(fourier.loop_orbits),
        fourier.degree,
    ) == orbits
    assert len(fourier.kernels) == len(fourier.born_orbits) * len(fourier.loop_orbits)
    for a, j in fourier.kernels:
        assert all(
            fourier.covers(c, b)
            for b in fourier.born_orbits[a]
            for c, _sign in fourier.loop_orbits[j]
        )
    for accuracy in ("full", "nlc"):
        _assert_fft_equals_direct(
            table, fourier, _born_rows(table, 3, (0, -1)), accuracy, loop_native
        )


@pytest.mark.loop_nightly
def test_seven_gluon_seed_tables_within_five_minutes(
    loop_sm_compiled_model, loop_native
):
    """WP2.4 acceptance: the full-colour seed table of gg -> ggggg (every
    skeleton, certified compressions, exact cells against the Born basis)
    and its certified S_6 FFT orbits build in under five minutes, and the
    FFT seeds equal the direct ones. Blocks, families and skeletons are
    generated beforehand and not timed, as ``LoopSlice.generate`` makes them
    (loop-aware reachability of the tree blocks, the native families and
    schedules; RUST-MIGRATION §3.B cleanup): the skeletons are every skeleton
    of every family, as ``family_skeletons`` lists them."""

    import time

    from pyamplicol.loops.compiler import prepare_loop_request_model
    from pyamplicol.loops.v2 import assemble, generator_input
    from pyamplicol.loops.v2.blocks import requested_loop_blocks
    from pyamplicol.loops.v2.colour import Primitive
    from pyamplicol.loops.v2.generator_input import NativeGeneration
    from pyamplicol.models.external import CompiledUFOModel
    from pyamplicol.models.loading import load_compiled_model
    from pyamplicol.processes.model import build_model_process_ir

    compiled = load_compiled_model(loop_sm_compiled_model)
    born, loop = {"QCD": 5, "QED": 0}, {"QCD": 7, "QED": 0}
    request = assemble.SliceRequest(
        "g g > g g g g g",
        born,
        loop,
        fixed_parameters=legacy_capture.FIXED_PARAMETERS,
    )
    structure = prepare_loop_request_model(
        compiled,
        born_order_bounds=born,
        loop_order_bounds=loop,
        fixed_parameters=legacy_capture.FIXED_PARAMETERS,
    ).structure
    data = json.loads(structure.native.to_json())
    catalogue = ModelLoopCatalogue(
        data, species_orders=structure.perturbative_species_orders
    )
    process = build_model_process_ir(
        "g g > g g g g g", compiled.ir, color_accuracy="full"
    )
    selection = LoopOrderSelection(
        frozenset({"QCD"}), frozenset({add_orders(tuple(loop.items()))})
    )
    field_table = assemble._field_level_table(data, process, selection, request)
    reachable = assemble._field_level_families(
        data, process, catalogue, selection, request, loop_native, table=field_table
    )
    born_requests = assemble._born_block_requests(
        assemble._born_table(data, process, selection, request, field_table),
        data,
        request.born_orders,
    )
    blocks = build_tree_blocks(
        process,
        CompiledUFOModel(compiled),
        born_coupling_orders=born,
        block_coupling_orders=loop,
        reachability=requested_loop_blocks(
            reachable.block_requests() | born_requests, data
        ),
    )
    generation = NativeGeneration(
        blocks.block_table(data), catalogue, selection, loop_native
    )
    assert generation.families().families == reachable.families
    colour = SeedColour(blocks, data, compiled.ir.vertex_terms)
    generation.schedules(colour, leg_rows=generator_input.leg_rows(blocks))
    assert generation.base is not None
    skeletons = [s for family in generation.base.skeletons for s in family]
    start = time.perf_counter()
    table, class_of = build_seed_table(colour, skeletons, compress=True)
    fourier = seed_fourier(table)
    seconds = time.perf_counter() - start
    assert seconds < 300, seconds
    kinds = [seed.kind for seed in table.classes if isinstance(seed, Primitive)]
    assert len(kinds) == len(table.classes)  # no generic class left
    assert (
        kinds.count("adjoint-ring") == 360 and kinds.count("fundamental-trace") == 720
    )
    assert len(class_of) == len(skeletons)
    assert fourier is not None and fourier.degree == 6
    assert (len(fourier.born_orbits), len(fourier.loop_orbits)) == (1, 2)
    assert len(fourier.kernels) == 2
    _assert_fft_equals_direct(table, fourier, _born_rows(table, 2), "full", loop_native)


# The legacy engine's per-diagram colour oracle (its FeynKit diagrams' colour
# tensors against the seed predictions) was removed with that engine (user
# ruling 2026-10-28, M-QCD Q3); the seed colour weights stay checked against
# the legacy engine's frozen Born-weighted family numerators
# (``test_loop_v2_slice.test_family_numerators_equal_o2``) and LC/NLC
# projections (``test_loop_v2_slice.test_colour_projections_equal_legacy``).


@pytest.mark.loop_fast
@pytest.mark.parametrize(
    ("process", "qcd"),
    [("u u~ > g g", 2), ("u u~ > g g g", 3), ("u u~ > d d~ g", 3), ("g g > g g", 2)],
)
def test_born_sectors_close_block_words_in_their_orientation(
    loop_sm_compiled_model, process, qcd
):
    """A0[b] multiplies the Born basis tensor b in the block-word orientation.

    Every Born root contracts two tree currents; their block words closed with
    each other (U(N) part) give the root's sector tensor, pairing phase
    included. Gluons on one quark line (uu~ -> gg, ggg) fix the orientation.
    The root's weight is its Wick ordering sign (the Born sign convention of
    ``generation/fermion_ordering.py``), pinned here as a table: +1 on one
    quark line; for u u~ -> d d~ g it is -1 exactly where the sector's colour
    pairing (fundamental, antifundamental) is the fermion pairing (2,1)(3,4),
    the sign the colour metric's pairing parity used to supply.
    """

    from pyamplicol.models.external import CompiledUFOModel
    from pyamplicol.models.loading import load_compiled_model
    from pyamplicol.processes.model import build_model_process_ir

    compiled = load_compiled_model(loop_sm_compiled_model)
    model = CompiledUFOModel(compiled)
    orders = {"QCD": qcd, "QED": 0}
    blocks = build_tree_blocks(
        build_model_process_ir(process, compiled.ir, color_accuracy="full"),
        model,
        born_coupling_orders=orders,
        block_coupling_orders=orders,
    )
    basis = born_basis(blocks)
    sectors = dict(zip(basis.sector_ids, basis.tensors, strict=True))
    representations = {leg.label: leg.representation for leg in basis.legs}
    dag = blocks.born_dag
    negative_pairings = {"u u~ > d d~ g": {((2, 1), (3, 4))}}.get(process, set())
    checked = 0
    for root in dag.amplitude_roots:
        ends = [dag.currents[root.left_id].index, dag.currents[root.right_id].index]
        if any(index.auxiliary_kind is not None for index in ends):
            continue  # numeric U(1) subtraction
        fields = [int(model.color_rep(index.particle_id)) for index in ends]
        # Quark ends share their open label; gluon ends close with delta^XY.
        opens = ("x", "y") if fields[0] == 8 else ("x", "x")
        flows = [
            block_colour_flow(index.ordered_external_labels, representations, field)
            for index, field in zip(ends, fields, strict=True)
        ]
        networks = [
            block_colour_network(flow, representations, field, label)
            for flow, field, label in zip(flows, fields, opens, strict=True)
        ]
        if fields[0] == 8:
            networks.append((ColourProduct(traces=(opens,)),))
        closed = reduce_network(
            multiply(*networks),
            basis.legs,
            {leg.label: leg.label for leg in basis.legs},
        )
        (leading,) = [term for term in closed.terms if term.nc_power == 0]
        (expected,) = sectors[root.color_sector_id].terms
        sign = flows[0].sign * flows[1].sign
        pairing = tuple(
            (line.fundamental_label, line.antifundamental_label)
            for line in dag.color_plan.sector(root.color_sector_id).open_color_lines
        )
        assert root.color_weight == (-1.0 if pairing in negative_pairings else 1.0, 0.0)
        assert (leading.tensor, leading.coefficient.scaled(sign)) == (
            expected.tensor,
            expected.coefficient,
        ), root.id
        checked += 1
    assert checked


# --------------------------------------------------------------- leading colour


@pytest.mark.loop_physics
def test_leading_colour_degrees_equal_frozen_legacy_degrees(
    v2_slice, loop_native, loop_reference
):
    """LC degrees against the frozen legacy capture, the structural projector
    and the face rule. The former live comparison of the LC seeds with the
    legacy LC family numerators went with the legacy engine; the per-family O2
    against the frozen legacy numerators (``test_loop_v2_slice.py::
    test_family_numerators_equal_o2``) and the NJet N_c scans cover it."""
    from symbolica import E, Expression, S

    from pyamplicol.loops.color_projection import prepare_color_projection

    v2 = v2_slice("gg_gg")
    table = v2.table
    degree = table.degree(v2.pairs)
    born_degree = table.basis.degree({(v2.born_order, v2.born_order)})
    frozen = loop_reference("legacy/gg_gg")["colour_projections"]["lc"]
    assert (born_degree, degree) == (frozen["born_degree"], frozen["virtual_degree"])

    # The legacy structural projector on the seed cells agrees with ``project``.
    nc = S("v2_colour_test::nc")

    def laurent(cell):
        return sum(
            (
                (E(str(value.real)) + E(str(value.imag)) * Expression.I) * nc**power
                for power, value in cell.items()
            ),
            E("0"),
        )

    rows = [(laurent(cell),) for row in table.cells for cell in row]
    projection = prepare_color_projection(nc, rows)
    assert projection.leading_degree == degree
    cells = [cell for row in table.cells for cell in row]
    for accuracy in ("lc", "nlc", "full"):
        for value, cell in zip(projection.project(accuracy), cells, strict=True):
            assert _exact_number(value.replace(nc, E("3"))) == project(
                cell, degree, accuracy
            )

    # The face rule keeps every gluon and ghost loop class and no quark loop.
    certificate = certify_leading_colour(table, v2.pairs)
    assert certificate.degree == degree
    loop_kind = {}
    for skeleton, ((c, _one),) in v2.class_of.items():
        species = {content.loop_in.rstrip("~") for content in skeleton.contents}
        loop_kind.setdefault(c, set()).update(species)
    for c, kinds in loop_kind.items():
        quark = bool(kinds - {"G", "gh"})
        assert (c in certificate.leading) == (not quark), kinds
        assert certificate.exact_degrees[c] == (degree - 1 if quark else degree)


@pytest.mark.loop_physics
def test_lc_needs_nc_exact_tree_currents(v2_slice):
    """U(1)-subtraction pieces enter the blocks with their N_c grade (design
    §3.2 item 4), so uu~ -> dd~g is formal in N_c; a table whose blocks mix
    grades only allows full colour."""

    v2 = v2_slice("uux_ddxg")
    grades = {
        block.colour_flow.nc_grade
        for block in v2.colour.blocks.blocks
        if block.key.auxiliary_kind is None
    }
    assert grades == {0, -1} and v2.table.formal_nc
    assert v2.table.seeds(v2.pairs, {}, "lc") == {}
    mixed = dataclasses.replace(v2.table, formal_nc=False)
    with pytest.raises(ValueError, match="N_c-exact"):
        mixed.seeds(v2.pairs, {}, "lc")
    assert mixed.seeds(v2.pairs, {}, "full") == {}


@pytest.mark.loop_physics
def test_colour_states_keep_skeleton_weights(v2_slice):
    """Reflection is an involution and generic colour states change no
    automorphism weight of the routings (design §3.3)."""

    v2 = v2_slice("gg_gg")
    dressed = {}
    for key, family in v2.families.families.items():
        for skeleton in family_skeletons(
            v2.families, family, DressedColour(), v2.catalogue
        ):
            dressed[(key, tuple(c.terms for c in skeleton.contents))] = skeleton.weight
    weights = Counter()
    for key, skeletons in v2.skeletons.items():
        for skeleton in skeletons:
            for content in skeleton.contents:
                state = content.colour
                assert v2.colour.reflected(v2.colour.reflected(state)) == state
            terms = tuple(c.terms for c in skeleton.contents)
            assert skeleton.weight == dressed[(key, terms)]
            weights[skeleton.weight] += 1
    assert weights[Fraction(1, 2)] > 0
