# SPDX-License-Identifier: 0BSD
"""Generic colour seeds (design §3.4, §7.1-§7.5, WP1.4).

Oracles: the legacy engine and its FeynKit loop diagrams (tests only). Each
diagram is read as v2 contents and, for every choice of vertex colour
structures, its colour is predicted from the seed of the skeleton it forms;
each legacy loop colour term is one such choice. The predicted Born x loop
colour weights equal the legacy contractions exactly (``Fraction``
arithmetic), and with the legacy loop coefficients they give the O2 family
coefficients of the legacy capture.
"""

from __future__ import annotations

import dataclasses
import gzip
import itertools
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
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
    evaluate_color_nc_terms,
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
    relabel,
    seed_fourier,
    spenso_colour_network,
)
from pyamplicol.loops.v2.families import (
    LoopOrderSelection,
    MassId,
    ModelLoopCatalogue,
    add_orders,
    canonical_family_key,
    generate_families,
)
from pyamplicol.loops.v2.skeletons import DressedColour, family_skeletons
from tools.loop_reference import legacy_capture

from .loop_v2_bdk import BDK_PREFACTOR, bdk_bracket, bdk_structures, laurent_times

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "loop" / "legacy"
ZERO = ExactColorCoefficient()
# Design §16.3, strong per-family oracle O2.
O2_TOLERANCE = {"double": 1e-13, "arbitrary-200": 1e-50}
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
    from pyamplicol.loops.v2 import assemble
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
    generation.schedules(colour)
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


@pytest.fixture(scope="module")
def legacy_program(loop_sm_ufo):
    built = {}

    def build(name: str, accuracy: str = "full"):
        key = (name, accuracy)
        if key not in built:
            program, _, _ = legacy_capture._program(
                loop_sm_ufo, legacy_capture.CONTROLS[name], color_accuracy=accuracy
            )
            built[key] = program
        return built[key]

    return build


# --------------------------------------------------------------- diagram oracle
#
# The legacy loop graphs are FeynKit one-loop diagrams (oracle only). Each is
# read around its loop as v2 contents, and its closed colour, for every choice
# of vertex colour structures, is predicted from the seeds of the skeletons
# those contents form. Nothing is fitted to the seeds.

OPEN = 1000  # colour label of a tree's open leg (process legs are 1..n)
UFO_SLOT = re.compile(r"ufo_c_(\d+)$")
ONE = ExactColorCoefficient(1)


def _raw(tensor: ConnectedColorTensor, formal: bool) -> dict:
    """Coordinates over (structure, N_c power), or over structures at N_c = 3.

    The tensor is 2^(-normalization_power/2) times this vector.
    """

    out = {}
    for term in tensor.terms:
        key = (term.tensor, term.nc_power) if formal else term.tensor
        value = term.coefficient.scaled(
            Fraction(1) if formal else Fraction(3) ** term.nc_power
        )
        out[key] = out.get(key, ZERO) + value
    return {key: value for key, value in out.items() if value}


def _rescaled(vector: dict, power: int, target: int) -> dict:
    """2^(-power/2) vector, written as 2^(-target/2) times the result."""

    assert (target - power) % 2 == 0, (power, target)
    factor = Fraction(2) ** ((target - power) // 2)
    return {key: value.scaled(factor) for key, value in vector.items()}


def _negated(vector: dict) -> dict:
    return {key: value.scaled(-1) for key, value in vector.items()}


def _complex_div(a: ExactColorCoefficient, b: ExactColorCoefficient):
    norm = b.real * b.real + b.imag * b.imag
    return (a * b.conjugate()).scaled(1 / norm)


def _solve(columns, target):
    """The unique exact r with sum_c r_c columns[c] == target, or None."""

    keys = sorted({key for column in columns for key in column} | set(target), key=repr)
    rows = [
        [column.get(key, ZERO) for column in columns] + [target.get(key, ZERO)]
        for key in keys
    ]
    rank = 0
    for col in range(len(columns)):
        pivot = next((i for i in range(rank, len(rows)) if rows[i][col]), None)
        assert pivot is not None, "linearly dependent columns"
        rows[rank], rows[pivot] = rows[pivot], rows[rank]
        head = rows[rank][col]
        rows[rank] = [_complex_div(value, head) for value in rows[rank]]
        for i, row in enumerate(rows):
            if i != rank and row[col]:
                factor = row[col]
                rows[i] = [
                    x + (factor * y).scaled(-1)
                    for x, y in zip(row, rows[rank], strict=True)
                ]
        rank += 1
    if any(row[-1] for row in rows[rank:]):
        return None
    return [row[-1] for row in rows[:rank]]


def _rotation(ring: tuple) -> tuple:
    return min(ring[i:] + ring[:i] for i in range(len(ring)))


def _first_free(particles, wanted) -> tuple[int, ...]:
    """The first free UFO leg of each wanted incoming particle, in order."""

    free = list(range(1, len(particles) + 1))
    legs = []
    for name in wanted:
        leg = next(leg for leg in free if particles[leg - 1] == name)
        free.remove(leg)
        legs.append(leg)
    assert not free, (particles, wanted)
    return tuple(legs)


@dataclass(frozen=True)
class Attachment:
    """A tree (or external leg) entering a loop vertex as a block current."""

    mask: int
    field: str  # incoming field at the loop vertex
    orders: tuple
    edge: int
    tree: tuple[int, ...]  # tree vertex ids, empty for an external leg


@dataclass
class DiagramPosition:
    """A loop vertex read as a v2 content: rule, content legs and blocks."""

    vertex: int
    rule: str
    legs: tuple[int, ...]  # UFO legs of loop in, loop out and attachments
    labels: dict  # UFO leg -> colour label of its edge
    loop_in: str
    loop_out: str
    orders: tuple
    attachments: tuple[Attachment, ...]


class Diagram:
    """One FeynKit loop diagram: its edges, vertex legs and both readings."""

    def __init__(self, graph, oracle: DiagramOracle):
        self.graph = graph
        self.oracle = oracle
        self.edges = {edge.id: edge for edge in graph.edges}
        self.vertices = {vertex.id: vertex for vertex in graph.vertices}
        self.ends: dict[int, list[int]] = {}
        for _half, vertex, edge, _flow in graph.half_edges:
            self.ends.setdefault(vertex, []).append(edge)
        self.loop = set(graph.loop_edges)
        assert len(self.loop) >= 2
        start = min(v for e in self.loop for v in self._sides(e))
        walk, vertex = [], start
        edge = min(e for e in self.ends[start] if e in self.loop)
        while True:
            walk.append((vertex, edge))
            vertex = self._other(edge, vertex)
            if vertex == start:
                break
            edge = next(e for e in self.ends[vertex] if e in self.loop and e != edge)
        forward = [(v, walk[i - 1][1], e) for i, (v, e) in enumerate(walk)]
        backward = [(v, out, into) for v, into, out in reversed(forward)]
        self.readings = [self._positions(reading) for reading in (forward, backward)]
        positions = self.readings[0]
        masses = [
            MassId(re.sub(r".*::", "", self.edges[edge].mass))
            for _v, _into, edge in forward
        ]
        key = canonical_family_key(
            [sum(a.mask for a in p.attachments) for p in positions], masses
        )
        self.family = tuple((mask, mass.mass) for mask, mass in key)
        self.tree_vertices = sorted(set(self.vertices) - {p.vertex for p in positions})
        self.tree_labels = {}
        for vertex in self.tree_vertices:
            particles = oracle.rules[self.vertices[vertex].interaction]
            ends = self.ends[vertex]
            legs = _first_free(particles, [self.incoming(e, vertex) for e in ends])
            self.tree_labels[vertex] = dict(
                zip(legs, map(self.label, ends), strict=True)
            )

    def _sides(self, edge):
        return (self.edges[edge].source, self.edges[edge].target)

    def _other(self, edge, vertex):
        source, target = self._sides(edge)
        return target if source == vertex else source

    def incoming(self, edge, vertex) -> str:
        """The particle entering ``vertex`` along ``edge``."""

        particle = self.edges[edge].particle
        return (
            particle
            if self.edges[edge].target == vertex
            else self.oracle.anti(particle)
        )

    def label(self, edge):
        index = self.edges[edge].external_index
        return ("edge", edge) if index is None else index + 1

    def _attachment(self, vertex, edge) -> Attachment:
        if self.edges[edge].external_index is not None:
            return Attachment(
                1 << self.edges[edge].external_index,
                self.incoming(edge, vertex),
                (),
                edge,
                (),
            )
        root = self._other(edge, vertex)
        tree, stack, seen, mask = {root}, [root], {edge}, 0
        while stack:
            current = stack.pop()
            for other in self.ends[current]:
                if other in seen:
                    continue
                seen.add(other)
                index = self.edges[other].external_index
                if index is not None:
                    mask |= 1 << index
                elif (far := self._other(other, current)) not in tree:
                    tree.add(far)
                    stack.append(far)
        orders = add_orders(*(self.vertices[v].order for v in tree))
        return Attachment(
            mask, self.incoming(edge, vertex), orders, edge, tuple(sorted(tree))
        )

    def _positions(self, reading) -> list[DiagramPosition]:
        positions = []
        for vertex, into, out in reading:
            rule = self.vertices[vertex].interaction
            attachments = tuple(
                sorted(
                    (
                        self._attachment(vertex, edge)
                        for edge in self.ends[vertex]
                        if edge not in self.loop
                    ),
                    key=lambda attachment: attachment.mask,
                )
            )
            loop_in = self.incoming(into, vertex)
            loop_out = self.oracle.anti(self.incoming(out, vertex))
            # Content legs (``colour`` module docstring): the first free UFO
            # leg of loop in, loop out, then of each block by mask.
            legs = _first_free(
                self.oracle.rules[rule],
                [
                    loop_in,
                    self.oracle.anti(loop_out),
                    *(a.field for a in attachments),
                ],
            )
            edges = [into, out, *(a.edge for a in attachments)]
            positions.append(
                DiagramPosition(
                    vertex,
                    rule,
                    legs,
                    dict(zip(legs, map(self.label, edges), strict=True)),
                    loop_in,
                    loop_out,
                    add_orders(
                        self.vertices[vertex].order, *(a.orders for a in attachments)
                    ),
                    attachments,
                )
            )
        return positions

    def choices(self):
        """Every choice of colour structure at every vertex."""

        vertices = sorted(self.vertices)
        options = [
            sorted(self.oracle.vertex[self.vertices[v].interaction]) for v in vertices
        ]
        for choice in itertools.product(*options):
            yield tuple(zip(vertices, choice, strict=True))


class DiagramOracle:
    """Colour of FeynKit loop diagrams predicted from v2 seed classes.

    A diagram's loop vertices take the content legs, its trees any legs, and
    each choice of UFO colour structures closes into a colour tensor. Its trees
    decompose exactly onto the words of their block (the block currents are
    the FeynKit tree sums per word, ``test_loop_v2_blocks``), which turns the
    diagram into v2 contents. The prediction is the seed of the skeleton those
    contents form, times the tree coefficients and 2^(-p/2) per loop vertex
    (``colour`` module docstring).
    """

    def __init__(self, v2: V2Slice):
        self.v2 = v2
        # FeynKit's trees are compared formally in N_c where no block carries
        # a U(1)-subtraction piece (whose N_c grade v2 records), else at 3.
        self.formal = all(
            block.colour_flow.nc_grade == 0
            for block in v2.colour.blocks.blocks
            if block.key.auxiliary_kind is None
        )
        self.particles = {p["name"]: p for p in v2.data["particles"]}
        self.by_pdg = {int(p["pdg_code"]): p for p in v2.data["particles"]}
        self.rules = {r["name"]: tuple(r["particles"]) for r in v2.data["vertex_rules"]}
        # rule -> colour index -> (colour network, colour-flow normalisation p)
        self.vertex: dict[str, dict[int, tuple]] = {}
        for term in v2.vertex_terms:
            if term.vertex in self.rules:
                self.vertex.setdefault(term.vertex, {}).setdefault(
                    term.color_index,
                    (
                        spenso_colour_network(term.color_expression),
                        int(term.lc_color_normalization_power),
                    ),
                )
        self.legs = v2.colour.legs
        self.representations = {leg.label: leg.representation for leg in self.legs}
        self.words: dict[tuple, dict] = {}
        for block in v2.colour.blocks.blocks:
            key = block.key
            if key.auxiliary_kind is None:
                field = self.anti(self.by_pdg[key.particle_id]["name"])
                self.words.setdefault(
                    (key.external_mask, field, key.coupling_orders), {}
                ).setdefault(key.word, block)
        self.rings: dict[tuple, list] = {}
        for family, skeletons in v2.skeletons.items():
            for skeleton in skeletons:
                ring = tuple(map(self._content, skeleton.contents))
                self.rings.setdefault((family, _rotation(ring)), []).append(skeleton)

    def anti(self, name: str) -> str:
        return self.particles[name]["antiname"]

    def merged(self, name: str) -> str:
        return self.v2.families.merged.get(name, name)

    def _content(self, content) -> tuple:
        state = content.colour
        return (
            tuple(map(self.merged, self.rules[state.rule])),
            state.colour_index,
            state.legs,
            state.words,
            content.loop_in,
            content.loop_out,
            content.mask,
            content.orders,
        )

    def _vertex(self, diagram: Diagram, vertex: int, labels, index: int):
        network, _power = self.vertex[diagram.vertices[vertex].interaction][index]

        def rename(name):
            match = UFO_SLOT.search(str(name))
            return labels[int(match.group(1))] if match else ("dummy", vertex, name)

        return relabel(network, rename)

    def assemble(self, diagram: Diagram, positions, choice) -> ConnectedColorTensor:
        """The diagram's closed colour for one choice of colour structures."""

        choice = dict(choice)
        labels = {p.vertex: p.labels for p in positions} | diagram.tree_labels
        network = multiply(
            *(self._vertex(diagram, v, labels[v], choice[v]) for v in sorted(labels))
        )
        return reduce_network(
            network, self.legs, {leg.label: leg.label for leg in self.legs}
        )

    def tree_words(self, diagram: Diagram, attachment: Attachment, choice):
        """([(word, c_w)], power): the tree is 2^(-power/2) sum_w c_w |word>."""

        words = self.words[(attachment.mask, attachment.field, attachment.orders)]
        if not attachment.tree:
            (word,) = words
            return [(word, ONE)], 0
        block = next(iter(words.values()))
        field = int(self.by_pdg[block.key.particle_id]["color"])
        legs = [
            leg
            for leg in self.legs
            if attachment.mask >> (leg.label - 1) & 1 and leg.representation != 1
        ]
        if field != 1:
            legs.append(ColorLeg(OPEN, -field if abs(field) == 3 else 8))
        external = {leg.label: leg.label for leg in legs}
        tree = reduce_network(
            multiply(
                *(
                    self._vertex(diagram, v, diagram.tree_labels[v], choice[v])
                    for v in attachment.tree
                )
            ),
            legs,
            external | {("edge", attachment.edge): OPEN},
        )
        ordered = sorted(words)
        columns = []
        for word in ordered:
            string = reduce_network(
                block_colour_network(
                    words[word].colour_flow, self.representations, field, OPEN
                ),
                legs,
                external,
            )
            assert string.normalization_power == 0
            columns.append(_raw(string, self.formal))
        solution = _solve(columns, _raw(tree, self.formal))
        assert solution is not None, attachment
        pairs = [(word, x) for word, x in zip(ordered, solution, strict=True) if x]
        return pairs, tree.normalization_power

    def predict(self, diagram: Diagram, family, positions, choice):
        """({seed class: coefficient}, power, skeletons): the diagram colour is
        2^(-power/2) sum_c coefficient kappa_c; None if a content reading has
        no skeleton."""

        choice = dict(choice)
        power, options = 0, []
        for position in positions:
            power += self.vertex[position.rule][choice[position.vertex]][1]
            per_block = []
            for attachment in position.attachments:
                pairs, tree_power = self.tree_words(diagram, attachment, choice)
                power += tree_power
                per_block.append(pairs)
            options.append(list(itertools.product(*per_block)))
        coefficients, skeletons = {}, set()
        for words in itertools.product(*options):
            ring, coefficient = [], ONE
            for position, chosen in zip(positions, words, strict=True):
                for _word, x in chosen:
                    coefficient = coefficient * x
                ring.append(
                    (
                        tuple(map(self.merged, self.rules[position.rule])),
                        choice[position.vertex],
                        position.legs,
                        tuple(word for word, _x in chosen),
                        self.merged(position.loop_in),
                        self.merged(position.loop_out),
                        sum(a.mask for a in position.attachments),
                        position.orders,
                    )
                )
            found = self.rings.get((family, _rotation(tuple(ring))))
            if not found:
                return None
            (skeleton,) = found
            ((seed, _one),) = self.v2.class_of[skeleton]
            assert self.v2.table.classes[seed].normalization_power == 0
            coefficients[seed] = coefficients.get(seed, ZERO) + coefficient
            skeletons.add(skeleton)
        return coefficients, power, skeletons


def _legacy_tensor(expression, legs) -> ConnectedColorTensor:
    network = spenso_colour_network(expression.to_expression())
    labels = {
        label
        for product in network
        for label in (
            *(end for s in product.strings for end in (s.row, s.col, *s.word)),
            *(label for trace in product.traces for label in trace),
        )
    }
    external = {
        label: int(LEG.search(str(label)).group(1)) + 1
        for label in labels
        if LEG.search(str(label))
    }
    return reduce_network(network, legs, external)


def _number(pair) -> ExactColorCoefficient:
    return ExactColorCoefficient(Fraction(pair[0]), Fraction(pair[1]))


@dataclass
class LegacyAnalysis:
    born: list  # literal legacy Born colour tensors
    loop: list  # literal legacy loop colour tensors
    families: dict  # legacy family index -> v2 family key (None: certified zero)
    transport: list  # R[legacy Born term][basis element]
    decomposition: dict  # loop term -> {seed class: exact coefficient}
    root2: int  # 1/sqrt(2) power of the legacy Born colour tensors
    skeletons: set  # skeletons predicted by some diagram
    certified_terms: set  # loop terms of certified-zero families


def _analyse(v2: V2Slice, bare) -> LegacyAnalysis:
    """Predict every legacy loop colour term from the v2 seeds (no fitting)."""

    oracle = DiagramOracle(v2)
    legs = oracle.legs
    born = [_legacy_tensor(tensor, legs) for tensor in bare.born_color_tensors]
    loop = [_legacy_tensor(tensor, legs) for tensor in bare.loop_color_tensors]
    table = v2.table
    basis = table.basis
    # Exact colour overlaps: the legacy metrics are <born|ket> / average
    # (projected by the legacy LC program, so checked at full colour only).
    average = _exact_number(bare.initial_average).real
    metrics = (
        ((bare.born_metric, born), (bare.loop_metric, loop))
        if getattr(bare, "color_accuracy", "full") == "full"
        else ()
    )
    for metric, kets in metrics:
        for bra, row in zip(born, metric, strict=True):
            for ket, value in zip(kets, row, strict=True):
                mine = evaluate_color_nc_terms(
                    contract_connected_tensor_nc_terms(bra, ket)
                )
                assert mine.scaled(1 / average) == _exact_number(value)
    # Legacy Born colour vectors in the tree engine's Born basis.
    transport = []
    for tensor in born:
        row = [ZERO] * len(basis)
        for term in tensor.terms:
            matches = [
                b
                for b, element in enumerate(basis.tensors)
                if element.terms[0].tensor == term.tensor
                and v2.born_order in basis.orders[b]
            ]
            assert matches, term
            sign = basis.tensors[matches[0]].terms[0].coefficient
            row[matches[0]] = row[matches[0]] + _complex_div(
                term.coefficient.scaled(Fraction(3) ** term.nc_power), sign
            )
        transport.append(row)
    (root2,) = {tensor.normalization_power for tensor in born}
    known = {
        tuple((mask, mass.mass) for mask, mass in key): key
        for key in v2.families.families
    }
    scaleless = {
        tuple((mask, mass.mass) for mask, mass in log.key)
        for log in v2.families.scaleless
    }
    families: dict[int, set] = {}
    decomposition, predicted_skeletons, certified_terms = {}, set(), set()
    for index, graph in enumerate(bare.loop_graphs):
        diagram = Diagram(graph, oracle)
        key = known.get(diagram.family)
        families.setdefault(graph.family.family_index, set()).add(key)
        offset = bare.loop_term_offsets[index]
        terms = range(offset, offset + len(bare.recurrences[index].terms))
        if key is None:
            assert diagram.family in scaleless, diagram.family
            certified_terms.update(terms)
            continue
        # The reading(s) that are v2 skeletons: one, or both for N <= 2.
        readings = []
        for positions in diagram.readings:
            predictions = {}
            for choice in diagram.choices():
                prediction = oracle.predict(diagram, key, positions, choice)
                if prediction is None:
                    break
                predictions[choice] = prediction
            else:
                readings.append((positions, predictions))
        assert readings, (index, diagram.family)
        assert len(readings) == 1 or len(diagram.readings[0]) <= 2, index
        positions, predictions = readings[0]
        assembled = {}
        for choice, (coefficients, power, skeletons) in predictions.items():
            tensor = oracle.assemble(diagram, positions, choice)
            vector = _raw(tensor, oracle.formal)
            seeded = {}
            for seed, coefficient in coefficients.items():
                for structure, value in _raw(
                    table.classes[seed], oracle.formal
                ).items():
                    seeded[structure] = (
                        seeded.get(structure, ZERO) + coefficient * value
                    )
            seeded = {key: value for key, value in seeded.items() if value}
            assert vector == _rescaled(seeded, power, tensor.normalization_power), (
                index,
                choice,
            )
            assembled[choice] = (vector, tensor.normalization_power)
            predicted_skeletons |= skeletons
        # Every legacy colour term is (up to its vertex leg order) one choice.
        for term in terms:
            target = _raw(loop[term], oracle.formal)
            power = loop[term].normalization_power
            matches = [
                (sign, choice)
                for choice, (vector, own) in assembled.items()
                for sign, candidate in (
                    (1, _rescaled(vector, own, power)),
                    (-1, _negated(_rescaled(vector, own, power))),
                )
                if candidate == target
            ]
            # Up to a sign: the legacy term's vertex leg order may differ from
            # the choice, and its capture has flipped the sign of some closed
            # quark-loop T^a boxes between sessions. The seeds are matched
            # sign-exactly to the diagram closure above, so this cannot hide a
            # wrong kappa; the sign of seed x kinematics against legacy is O2's
            # (test_loop_v2_slice.py::test_family_numerators_equal_o2).
            assert matches, (index, term)
            sign, choice = matches[0]
            coefficients, own, _skeletons = predictions[choice]
            decomposition[term] = {
                seed: value.scaled(sign)
                for seed, value in _rescaled(coefficients, own, power).items()
                if value
            }
    assert all(len(keys) == 1 for keys in families.values())
    return LegacyAnalysis(
        born,
        loop,
        {index: next(iter(keys)) for index, keys in families.items()},
        transport,
        decomposition,
        root2,
        predicted_skeletons,
        certified_terms,
    )


def _seeded_family_numerators(
    v2: V2Slice, analysis: LegacyAnalysis, bare, point, scale, accuracy="full"
):
    """Legacy loop coefficients weighted by v2 seeds, per legacy family index.

    At full colour each seeded Born x loop colour weight must equal the legacy
    contraction exactly.
    """

    currents = bare.prepare_currents_native(
        np.asarray([point], dtype=str), [scale], model_values=None
    )
    helicities = currents.batch.helicity_count
    count = len(bare.born_colors)
    values = [_number(pair) for pair in currents.born.to_decimal()]
    legacy_born = [values[h * count : (h + 1) * count] for h in range(helicities)]
    average = _exact_number(bare.initial_average).real
    metric = [[_exact_number(value) for value in row] for row in bare.loop_metric]
    table = v2.table
    weights = {}
    for h, amplitudes in enumerate(legacy_born):
        born = {
            (b, v2.born_order): {
                0: sum(
                    (
                        a * row[b]
                        for a, row in zip(amplitudes, analysis.transport, strict=True)
                    ),
                    ZERO,
                )
            }
            for b in range(len(table.basis))
        }
        seeds = table.seeds(
            v2.pairs,
            born,
            accuracy,
            cast=lambda x: x,
            conjugate=lambda z: z.conjugate(),
        )
        for term, coefficients in analysis.decomposition.items():
            root2 = analysis.root2 + analysis.loop[term].normalization_power
            weight = sum(
                (
                    x * seeds.get((c, v2.loop_order), ZERO)
                    for c, x in coefficients.items()
                ),
                ZERO,
            ).scaled(Fraction(1, 2 ** (root2 // 2)) / average)
            if accuracy == "full":
                legacy = sum(
                    (
                        a.conjugate() * row[term]
                        for a, row in zip(amplitudes, metric, strict=True)
                    ),
                    ZERO,
                )
                assert weight == legacy, (term, h)
            weights[(term, h)] = weight
    loops = [[_number(pair) for pair in batch.to_decimal()] for batch in currents.loops]
    out = {}
    for family in bare.families:
        if analysis.families[family.index] is None:
            continue
        position = {power: i for i, power in enumerate(family.powers)}
        rows = [[ZERO] * len(family.powers) for _ in family.sector_indices]
        for index in family.graph_indices:
            width = sum(map(len, bare.loop_reduction_powers[index]))
            offset = 0
            for local, labels in enumerate(bare.loop_reduction_powers[index]):
                term = bare.loop_term_offsets[index] + local
                for column, power in enumerate(labels):
                    if power is None:
                        continue
                    total = sum(
                        (
                            weights[(term, h)]
                            * loops[index][h * width + offset + column]
                            for h in range(helicities)
                        ),
                        ZERO,
                    )
                    for row in rows:
                        row[position[power]] = row[position[power]] + total
                offset += len(labels)
        out[family.index] = rows
    return out


def _magnitude(value: ExactColorCoefficient) -> float:
    return abs(complex(value))


def _assert_close(mine, reference, tolerance, context):
    """Every coefficient within ``tolerance`` of the family's largest one."""

    for index, rows in mine.items():
        expected = [[_number(pair) for pair in row] for row in reference[index]]
        scale = max(map(_magnitude, itertools.chain(*expected)))  # 0: LC zero
        error = max(
            _magnitude(value + reference_value.scaled(-1))
            for row, reference_row in zip(rows, expected, strict=True)
            for value, reference_value in zip(row, reference_row, strict=True)
        )
        assert error <= tolerance * scale, (context, index, error, scale)


def _legacy_numerators(virtual, control, point, scale) -> dict:
    """The O2 capture: legacy Born-weighted family coefficients, same frames."""

    rows = legacy_capture._family_coefficients(virtual, control, point, scale)
    return {
        family.index: row
        for family, row in zip(virtual.bare.families, rows, strict=True)
    }


@pytest.fixture(scope="module")
def legacy_analysis(v2_slice, legacy_program):
    built = {}

    def build(name: str, accuracy: str = "full"):
        if (name, accuracy) not in built:
            bare = legacy_capture._virtual(legacy_program(name, accuracy)).bare
            built[(name, accuracy)] = _analyse(v2_slice(name), bare)
        return built[(name, accuracy)]

    return build


# ------------------------------------------------------------------- O2 oracle


@pytest.mark.loop_physics
@pytest.mark.parametrize("name", ["gg_gg", "uux_ddxg"])
def test_feynkit_diagram_colours_equal_seed_predictions(
    v2_slice, legacy_analysis, loop_native, name
):
    """Every legacy (FeynKit) loop diagram, under every choice of vertex colour
    structures, has the colour its v2 skeletons' seeds predict (asserted in
    ``_analyse``), and every skeleton is predicted by some diagram."""

    v2 = v2_slice(name)
    analysis = legacy_analysis(name)
    assert analysis.skeletons == {s for rows in v2.skeletons.values() for s in rows}
    assert set(analysis.decomposition) | analysis.certified_terms == set(
        range(len(analysis.loop))
    )
    assert v2.colour.formal_nc


@pytest.mark.loop_physics
@pytest.mark.parametrize("name", ["gg_gg", "uux_ddxg"])
def test_seed_sums_reproduce_legacy_contractions_via_o2(
    v2_slice, legacy_program, legacy_analysis, loop_native, name
):
    """Seed-weighted Born x loop colour weights equal the legacy contractions
    exactly; summed with the legacy loop coefficients they give the O2 family
    coefficients of the legacy capture in its own frames (the frozen O2 files
    against the live capture are WP0.1's check)."""

    v2 = v2_slice(name)
    o2 = json.loads(
        gzip.decompress((FIXTURES / f"{name}.families.json.gz").read_bytes())
    )
    program = legacy_program(name)
    analysis = legacy_analysis(name)
    certified = [index for index, key in analysis.families.items() if key is None]
    assert len(analysis.families) == len(o2["families"])
    assert sorted(
        key for key in analysis.families.values() if key is not None
    ) == sorted(v2.families.families)
    assert len(certified) == len(o2["families"]) - len(v2.families.families)
    point, scale = o2["points"][0], o2["scales_squared"][0]
    for carrier in legacy_capture.CARRIERS:
        virtual = legacy_capture._virtual(legacy_capture._rebind(program, carrier))
        mine = _seeded_family_numerators(v2, analysis, virtual.bare, point, scale)
        reference = _legacy_numerators(virtual, v2.control, point, scale)
        _assert_close(mine, reference, O2_TOLERANCE[carrier], carrier)


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
def test_leading_colour_degrees_equal_legacy_projection(
    v2_slice, legacy_program, legacy_analysis, loop_native, loop_reference
):
    from symbolica import E, Expression, S

    from pyamplicol.loops.color_projection import prepare_color_projection

    v2 = v2_slice("gg_gg")
    table = v2.table
    degree = table.degree(v2.pairs)
    born_degree = table.basis.degree({(v2.born_order, v2.born_order)})
    frozen = loop_reference("legacy/gg_gg")["colour_projections"]["lc"]
    program = legacy_program("gg_gg", "lc")
    bare = legacy_capture._virtual(program).bare
    assert (born_degree, degree) == (frozen["born_degree"], frozen["virtual_degree"])
    assert (born_degree, degree) == (bare.born_color_degree, bare.virtual_color_degree)

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

    # LC seeds reproduce the legacy LC family numerators.
    analysis = legacy_analysis("gg_gg", "lc")
    o2 = json.loads(gzip.decompress((FIXTURES / "gg_gg.families.json.gz").read_bytes()))
    point, scale = o2["points"][0], o2["scales_squared"][0]
    mine = _seeded_family_numerators(v2, analysis, bare, point, scale, "lc")
    legacy = legacy_capture._family_coefficients(
        legacy_capture._virtual(program), v2.control, point, scale
    )
    reference = {
        family.index: rows for family, rows in zip(bare.families, legacy, strict=True)
    }
    _assert_close(mine, reference, O2_TOLERANCE["double"], "lc")


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
