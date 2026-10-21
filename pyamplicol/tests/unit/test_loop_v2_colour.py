# SPDX-License-Identifier: 0BSD
"""Generic colour seeds (design §3.4, §7.1-§7.5, WP1.4).

The legacy engine is the oracle. Its Spenso colour tensors are lowered with
the code that lowers the vertex colours, its Born x loop colour overlaps are
compared cell by cell, every legacy loop colour term is decomposed into the
seed classes of the v2 skeletons of its family, and the seed-weighted Born
contractions, summed with the legacy loop coefficients, give the frozen O2
family numerators. Arithmetic on the legacy numbers is exact (``Fraction``),
so the seed-weighted contraction equals the legacy one identically.
"""

from __future__ import annotations

import gzip
import itertools
import json
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
    contract_connected_tensor_nc_terms,
    evaluate_color_nc_terms,
)
from pyamplicol.loops.v2.blocks import block_colour_flow, build_tree_blocks
from pyamplicol.loops.v2.colour import (
    ColourProduct,
    ColourString,
    SeedColour,
    _exact_number,
    block_colour_network,
    build_seed_table,
    certify_leading_colour,
    face_count,
    project,
    reduce_network,
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
        )
        return built[name]

    return build


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


# ------------------------------------------------------------- legacy analysis


def _complex_div(a: ExactColorCoefficient, b: ExactColorCoefficient):
    norm = b.real * b.real + b.imag * b.imag
    return (a * b.conjugate()).scaled(1 / norm)


def _solve(columns, target):
    """Exact r with sum_c r_c columns[c] == target, or None."""

    keys = sorted({key for column in columns for key in column} | set(target), key=repr)
    rows = [
        [column.get(key, ZERO) for column in columns] + [target.get(key, ZERO)]
        for key in keys
    ]
    pivots, rank = [], 0
    for col in range(len(columns)):
        pivot = next((i for i in range(rank, len(rows)) if rows[i][col]), None)
        if pivot is None:
            continue
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
        pivots.append(col)
        rank += 1
    if any(row[-1] for row in rows[rank:]):
        return None
    solution = [ZERO] * len(columns)
    for i, col in enumerate(pivots):
        solution[col] = rows[i][-1]
    return solution


def _vector(tensor: ConnectedColorTensor, formal: bool):
    """Coordinates over (structure, N_c power), or over structures at N_c = 3."""

    out = {}
    for term in tensor.terms:
        key = (term.tensor, term.nc_power) if formal else term.tensor
        value = term.coefficient.scaled(
            Fraction(1) if formal else Fraction(3) ** term.nc_power
        )
        out[key] = out.get(key, ZERO) + value
    return out


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


def _family_masks(shifts, incoming: int) -> list[int]:
    """Position leg masks from the O2 frame shifts (incoming momenta as given)."""

    legs = len(shifts[0])
    full = (1 << legs) - 1
    outgoing = [[-x if j < incoming else x for j, x in enumerate(s)] for s in shifts]
    masks = []
    for i, shift in enumerate(outgoing):
        step = [a - b for a, b in zip(shift, outgoing[i - 1], strict=True)]
        bits = [value - min(step) for value in step]
        assert set(bits) <= {0, 1}, shifts
        masks.append(sum(1 << j for j, bit in enumerate(bits) if bit))
    if sum(masks) != full or any(a & b for a, b in itertools.combinations(masks, 2)):
        masks = [full ^ mask for mask in masks]
    assert sum(masks) == full
    return masks


def _number(pair) -> ExactColorCoefficient:
    return ExactColorCoefficient(Fraction(pair[0]), Fraction(pair[1]))


@dataclass
class LegacyAnalysis:
    bare: Any
    legs: tuple
    born: list  # literal legacy Born colour tensors
    loop: list  # literal legacy loop colour tensors
    families: dict  # legacy family index -> v2 family key (None: certified zero)
    transport: list  # R[legacy Born term][basis element]
    decomposition: dict  # loop term -> {seed class: exact coefficient}
    root2: int  # 1/sqrt(2) power of every Born x loop product


def _analyse(v2: V2Slice, bare) -> LegacyAnalysis:
    """Map legacy colour terms onto the v2 Born basis and seed classes."""

    from pyamplicol.color.correlator_matrices import process_color_legs

    legs = process_color_legs(v2.process)
    born = [_legacy_tensor(tensor, legs) for tensor in bare.born_color_tensors]
    loop = [_legacy_tensor(tensor, legs) for tensor in bare.loop_color_tensors]
    table, colour = v2.table, v2.colour
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
    # Legacy families and loop terms onto v2 families and seed classes.
    by_key = {
        tuple((mask, mass.mass) for mask, mass in key): key
        for key in v2.families.families
    }
    scaleless = {
        tuple((mask, mass.mass) for mask, mass in log.key)
        for log in v2.families.scaleless
    }
    families = {}
    for record in legacy_capture._family_layout(bare):
        key = canonical_family_key(
            _family_masks(record["shifts"], len(v2.control.incoming)),
            [MassId(mass.split("^")[0]) for mass in record["masses_squared"]],
        )
        names = tuple((mask, mass.mass) for mask, mass in key)
        assert names in by_key or names in scaleless, names
        families[record["index"]] = by_key.get(names)
    pdg = {entry["name"]: abs(entry["pdg_code"]) for entry in v2.data["particles"]}
    name = {abs(entry["pdg_code"]): entry["name"] for entry in v2.data["particles"]}
    merged = v2.families.merged

    def species(names):
        return frozenset(pdg[merged.get(n, n)] for n in names)

    graph_of = [g for g, rec in enumerate(bare.recurrences) for _ in rec.terms]
    decomposition = {}
    for term, tensor in enumerate(loop):
        graph = bare.loop_graphs[graph_of[term]]
        key = families[graph.family.family_index]
        if key is None:
            continue
        wanted = species(name[p] for p in graph.loop_species)
        candidates = sorted(
            {
                v2.class_of[skeleton]
                for skeleton in v2.skeletons[key]
                if species(
                    n for c in skeleton.contents for n in (c.loop_in, c.loop_out)
                )
                == wanted
            }
        )
        solution = _solve(
            [_vector(table.classes[c], colour.formal_nc) for c in candidates],
            _vector(tensor, colour.formal_nc),
        )
        assert solution is not None, (term, wanted)
        assert (root2 + tensor.normalization_power) % 2 == 0
        decomposition[term] = {
            c: value for c, value in zip(candidates, solution, strict=True) if value
        }
    return LegacyAnalysis(
        bare, legs, born, loop, families, transport, decomposition, root2
    )


def _seeded_family_numerators(
    v2: V2Slice, analysis: LegacyAnalysis, bare, point, scale, accuracy="full"
):
    """Legacy loop coefficients weighted by v2 seeds, and the legacy contraction.

    Returns (per family index: [sector][power] coefficients, weights checked).
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
            (b, v2.born_order): sum(
                (
                    a * row[b]
                    for a, row in zip(amplitudes, analysis.transport, strict=True)
                ),
                ZERO,
            )
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


# Exact loop momenta at which two frames of a family numerator are compared.
SAMPLE_MOMENTA = (
    (Fraction(3, 7), Fraction(-5, 11), Fraction(2, 13), Fraction(7, 17)),
    (Fraction(-9, 5), Fraction(1, 3), Fraction(4, 9), Fraction(-2, 7)),
    (Fraction(1, 2), Fraction(6, 5), Fraction(-3, 4), Fraction(5, 3)),
)


def _frame_maps(source, target, incoming):
    """Every (sigma, offset) with q_target = sigma q_source + offset that maps
    the denominators of one O2 frame onto the other's; ``offset`` holds
    coefficients of the all-outgoing external momenta. A symmetric family
    (e.g. a bubble) has several."""

    def outgoing(shift):
        return [-x if j < incoming else x for j, x in enumerate(shift)]

    def equal(left, right):  # equal momenta up to momentum conservation
        return len({x - y for x, y in zip(left, right, strict=True)}) == 1

    a = [outgoing(shift) for shift in source["shifts"]]
    b = [outgoing(shift) for shift in target["shifts"]]
    maps = []
    for sigma, first in itertools.product((1, -1), range(len(b))):
        offset = [sigma * x - y for x, y in zip(a[0], b[first], strict=True)]
        image = []
        for i, shift in enumerate(a):
            want = [sigma * x - o for x, o in zip(shift, offset, strict=True)]
            image.extend(
                k
                for k, other in enumerate(b)
                if equal(other, want)
                and source["masses_squared"][i] == target["masses_squared"][k]
            )
        if sorted(image) == list(range(len(b))):
            maps.append((sigma, offset))
    assert maps, "O2 frames are not related by an affine map"
    return maps


def _evaluate(rows, powers, q):
    """The family numerator rows at exact loop momentum ``q``."""

    out = []
    for row in rows:
        total, scale = ZERO, 0.0
        for coefficient, power in zip(row, powers, strict=True):
            assert power[4:] == [0, 0]  # four-dimensional numerators
            monomial = Fraction(1)
            for component, exponent in zip(q, power[:4], strict=True):
                monomial *= component**exponent
            total = total + coefficient.scaled(monomial)
            scale += _magnitude(coefficient) * abs(float(monomial))
        out.append((total, scale))
    return out


def _assert_same_numerators(mine, layout, o2, carrier, incoming):
    """Seeded numerators equal the frozen O2 ones under each family's frame map."""

    tolerance = O2_TOLERANCE[carrier]
    frozen = {
        canonical: (record, rows)
        for record, rows in zip(
            o2["families"], o2["coefficients"][carrier], strict=True
        )
        for canonical in [_canonical_record(record, incoming)]
    }
    momenta = [[Fraction(component) for component in leg] for leg in o2["points"][0]]
    outgoing = [
        [-x for x in leg] if j < incoming else leg for j, leg in enumerate(momenta)
    ]
    for record in layout:
        if record["index"] not in mine:
            continue
        target, rows = frozen[_canonical_record(record, incoming)]
        assert target["powers"] == record["powers"]
        assert target["sectors"] == record["sectors"]
        reference = [[_number(pair) for pair in row] for row in rows]
        errors = []
        for sigma, offset in _frame_maps(record, target, incoming):
            shift = [
                sum(c * leg[mu] for c, leg in zip(offset, outgoing, strict=True))
                for mu in range(4)
            ]
            worst = 0.0
            for q in SAMPLE_MOMENTA:
                moved = [
                    sigma * component + s for component, s in zip(q, shift, strict=True)
                ]
                left = _evaluate(mine[record["index"]], record["powers"], q)
                right = _evaluate(reference, target["powers"], moved)
                for (value, own), (expected, scale) in zip(left, right, strict=True):
                    # Each side is rounded relative to its own monomials.
                    error = _magnitude(value + expected.scaled(-1))
                    worst = max(worst, error / max(scale, own))
            errors.append(worst)
        assert min(errors) <= tolerance, (carrier, record["index"], errors)


def _canonical_record(record, incoming):
    key = canonical_family_key(
        _family_masks(record["shifts"], incoming),
        [MassId(mass.split("^")[0]) for mass in record["masses_squared"]],
    )
    return tuple((mask, mass.mass) for mask, mass in key)


# ------------------------------------------------------------------- O2 oracle


@pytest.mark.loop_physics
@pytest.mark.parametrize("name", ["gg_gg", "uux_ddxg"])
def test_seed_sums_reproduce_legacy_contractions_via_o2(
    v2_slice, legacy_program, loop_native, name
):
    v2 = v2_slice(name)
    o2 = json.loads(
        gzip.decompress((FIXTURES / f"{name}.families.json.gz").read_bytes())
    )
    program = legacy_program(name)
    bare = legacy_capture._virtual(program).bare
    # The legacy family order and frames depend on Symbolica's symbol order,
    # so families are matched on their ring and compared under the frame map.
    incoming = len(v2.control.incoming)
    layout = legacy_capture._family_layout(bare)
    assert sorted(_canonical_record(r, incoming) for r in layout) == sorted(
        _canonical_record(r, incoming) for r in o2["families"]
    )
    analysis = _analyse(v2, bare)
    certified = [index for index, key in analysis.families.items() if key is None]
    assert len(certified) == len(o2["families"]) - len(v2.families.families)
    assert v2.colour.formal_nc == (name == "gg_gg")
    for carrier in legacy_capture.CARRIERS:
        rebound = legacy_capture._virtual(legacy_capture._rebind(program, carrier)).bare
        mine = _seeded_family_numerators(
            v2, analysis, rebound, o2["points"][0], o2["scales_squared"][0]
        )
        _assert_same_numerators(mine, layout, o2, carrier, incoming)


# --------------------------------------------------------------- leading colour


@pytest.mark.loop_physics
def test_leading_colour_degrees_equal_legacy_projection(
    v2_slice, legacy_program, loop_native, loop_reference
):
    from symbolica import E, Expression, S

    from pyamplicol.loops.color_projection import prepare_color_projection

    v2 = v2_slice("gg_gg")
    table = v2.table
    degree = table.degree(v2.pairs)
    born_degree = table.basis.degree({(v2.born_order, v2.born_order)})
    frozen = loop_reference("legacy/gg_gg")["leading_colour"]
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
    for skeleton, c in v2.class_of.items():
        species = {content.loop_in.rstrip("~") for content in skeleton.contents}
        loop_kind.setdefault(c, set()).update(species)
    for c, kinds in loop_kind.items():
        quark = bool(kinds - {"G", "gh"})
        assert (c in certificate.leading) == (not quark), kinds
        assert certificate.exact_degrees[c] == (degree - 1 if quark else degree)

    # LC seeds reproduce the legacy LC family numerators.
    analysis = _analyse(v2, bare)
    o2 = json.loads(gzip.decompress((FIXTURES / "gg_gg.families.json.gz").read_bytes()))
    point, scale = o2["points"][0], o2["scales_squared"][0]
    mine = _seeded_family_numerators(v2, analysis, bare, point, scale, "lc")
    legacy = legacy_capture._family_coefficients(
        legacy_capture._virtual(program), v2.control, point, scale
    )
    reference = {
        family.index: rows for family, rows in zip(bare.families, legacy, strict=True)
    }
    _assert_close(mine, reference, 1e-12, "lc")


@pytest.mark.loop_physics
def test_lc_needs_nc_exact_tree_currents(v2_slice):
    v2 = v2_slice("uux_ddxg")
    assert not v2.table.formal_nc
    with pytest.raises(ValueError, match="N_c-exact"):
        v2.table.seeds(v2.pairs, {}, "lc")
    assert v2.table.seeds(v2.pairs, {}, "full") == {}


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
