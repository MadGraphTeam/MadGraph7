# SPDX-License-Identifier: 0BSD
"""Python reference executor of the v2 schedules (design WP1.5, O4, O5).

O4: schedules F, B and M dress one skeleton set into identical family
numerators, exactly, after exact affine frame maps; fused and split kernel
forms agree exactly. O5: each B family numerator equals the skeleton-by-
skeleton product of kernels evaluated at a random exact loop momentum. The
block currents, Born amplitudes and seeds are exact random inputs, which every
one of these identities must hold for; kernels and colour are the real WP1.1
catalogue and WP1.4 seed classes.
"""

from __future__ import annotations

import collections
import json
import random
from fractions import Fraction

import pytest

from pyamplicol.loops.v2 import reference_executor as rx
from pyamplicol.loops.v2.families import (
    FamilyFrame,
    LoopOrderSelection,
    ModelBlockTable,
    ModelLoopCatalogue,
    add_orders,
    generate_families,
    shifts,
)
from pyamplicol.loops.v2.poly_affine import (
    MonomialLayout,
    evaluate,
    leg_momentum,
    relabel,
    sequence_relabelling,
    size,
    substitute,
    substitute_work,
)
from pyamplicol.loops.v2.schedules import (
    SkeletonView,
    build_schedules,
    cheapest,
    merged_graph,
)
from pyamplicol.loops.v2.skeletons import PlanarWords, family_skeletons

GLUONS = ("g g > g g", {"QCD": 2, "QED": 0}, {"QCD": 4, "QED": 0})
QUARKS = ("u u~ > d d~ g", {"QCD": 3, "QED": 0}, {"QCD": 5, "QED": 0})
# An electroweak boson on a loop quark line (couplings with nested roots).
DRELL_YAN = ("u u~ > e- e+ g", {"QCD": 1, "QED": 2}, {"QCD": 3, "QED": 2})


def _rational(generator: random.Random) -> Fraction:
    return Fraction(generator.randint(-50, 50), generator.randint(1, 9))


def _momenta(legs: int, seed: int) -> list[list[Fraction]]:
    """Exact momenta, all incoming, summing to zero (no on-shell condition is
    needed by any identity checked here)."""

    generator = random.Random(seed)
    momenta = [[_rational(generator) for _ in range(4)] for _ in range(legs - 1)]
    momenta.append([-sum(p[i] for p in momenta) for i in range(4)])
    return momenta


def _rows(mask: int) -> int:
    return 2 ** mask.bit_count()


@pytest.fixture(scope="module")
def layout(loop_native):
    return MonomialLayout.native(loop_native, 8)


# ------------------------------------------------------------------ poly_affine


@pytest.mark.loop_fast
def test_layout_is_the_ninjars_tensor_order(loop_native, layout):
    assert (
        layout.exponents[: size(2)] == MonomialLayout.native(loop_native, 2).exponents
    )
    degrees = [sum(e) for e in layout.exponents]
    assert degrees == sorted(degrees) and len(layout.exponents) == size(8)
    table = layout.product_table(3, 2)
    for m, left in enumerate(layout.exponents[: size(3)]):
        for n, right in enumerate(layout.exponents[: size(2)]):
            assert layout.exponents[table[m][n]] == tuple(
                a + b for a, b in zip(left, right, strict=True)
            )
    with pytest.raises(ValueError, match="layout"):
        MonomialLayout(((0, 0, 0, 0), (0, 1, 0, 0)))


@pytest.mark.loop_fast
@pytest.mark.parametrize("sign", [1, -1])
def test_affine_substitution_is_exact(layout, sign):
    generator = random.Random(sign)
    rank = 4
    coefficients = [rx.exact(_rational(generator)) for _ in range(size(rank))]
    shift = [rx.exact(_rational(generator)) for _ in range(4)]
    work = []
    mapped = substitute(coefficients, layout, sign=sign, shift=shift, count=work.append)
    assert sum(work) == substitute_work(rank)
    for _ in range(3):
        q = [rx.exact(_rational(generator)) for _ in range(4)]
        moved = [sign * x + s for x, s in zip(q, shift, strict=True)]
        assert evaluate(mapped, layout, q) == evaluate(coefficients, layout, moved)


def _times(left: dict, right: dict) -> dict:
    out: dict = collections.defaultdict(Fraction)
    for a, x in left.items():
        for b, y in right.items():
            out[tuple(i + j for i, j in zip(a, b, strict=True))] += x * y
    return out


def _denominator(shift) -> dict:
    """(q + V)^2 in the metric (+,-,-,-) as an exponent -> coefficient map."""

    out: dict = collections.defaultdict(Fraction)
    metric = (1, -1, -1, -1)
    for mu in range(4):
        unit = tuple(int(nu == mu) for nu in range(4))
        twice = tuple(2 * u for u in unit)
        out[twice] += metric[mu]
        out[unit] += 2 * metric[mu] * shift[mu]
        out[(0, 0, 0, 0)] += metric[mu] * shift[mu] ** 2
    return out


@pytest.mark.loop_fast
def test_relabelled_frames_permute_the_denominators(layout):
    """Re-expanding prod_i D_i in another reading gives that reading's
    product: every frame map of a ring is an exact affine map of q."""

    momenta = _momenta(4, 11)
    ring = ((0b0001, "m"), (0b0110, "0"), (0b1000, "0"))  # leg masks, mass ids
    old = shifts(ring)
    product: dict = {(0, 0, 0, 0): Fraction(1)}
    for mask in old:
        product = _times(product, _denominator(leg_momentum(mask, momenta)))
    dense = [rx.exact(product.get(e, 0)) for e in layout.exponents[: size(6)]]
    for start in range(3):
        for reflected in (False, True):
            target = FamilyFrame(start, reflected).apply(ring)
            relabelled = sequence_relabelling(ring, target)
            assert relabelled.apply(ring) == target
            mapped = relabel(dense, layout, relabelled, momenta)
            expected: dict = {(0, 0, 0, 0): Fraction(1)}
            for mask in shifts(target):
                expected = _times(expected, _denominator(leg_momentum(mask, momenta)))
            assert mapped == [
                rx.exact(expected.get(e, 0)) for e in layout.exponents[: size(6)]
            ]


# ------------------------------------------------------------------ kernels


@pytest.fixture(scope="module")
def compiled(loop_sm_compiled_model):
    from pyamplicol.models.loading import load_compiled_model

    return load_compiled_model(loop_sm_compiled_model)


@pytest.fixture(scope="module")
def kernels(compiled):
    from pyamplicol.loops.v2.catalogue import build_loop_kernel_catalogue

    return build_loop_kernel_catalogue(compiled)


def _parameters(compiled) -> dict[str, Fraction]:
    values = {
        name: Fraction(str(value[0]))
        for name, value in compiled.parameter_defaults.items()
    }
    values["MB"] = Fraction(0)
    return values


@pytest.mark.loop_fast
def test_exact_kernels_equal_the_symbolica_evaluators(compiled, kernels):
    """The exact polynomial reading of every catalogue kernel against its
    Symbolica evaluator: binary64 (SymJIT) and 60 digits (exact stand-ins)."""

    from decimal import Decimal

    from pyamplicol.loops.v2.catalogue import compile_loop_kernel

    parameters = _parameters(compiled)
    f64 = rx.KernelConstants(parameters, rx.Carrier.f64())
    exact = rx.KernelConstants(parameters, rx.Carrier.exact(60))
    generator = random.Random(5)
    assert len(kernels.kernels) > 200
    for spec in kernels.kernels:
        # Dyadic inputs are exact in the decimal rows of the 60-digit run.
        values = [
            rx.Complex(*(Fraction(generator.randint(-99, 99), 8) for _ in "ri"))
            for _ in spec.inputs
        ]
        precise = [
            parameters[item.model_parameter_name]
            if item.role == "model-parameter"
            else value
            for item, value in zip(spec.inputs, values, strict=True)
        ]
        evaluator = compile_loop_kernel(spec)
        kernel = rx.ExactKernel(spec)

        reference = evaluator.evaluate_complex(
            [[complex(rx.exact(x)) for x in precise]]
        )
        mine = kernel.evaluate([complex(x) for x in values], f64)
        scale = max(map(abs, reference[0])) or 1
        assert (
            max(abs(a - b) for a, b in zip(mine, reference[0], strict=True))
            <= 1e-13 * scale
        ), spec

        # Symbolica derives the working precision from each input's digits.
        rows = [
            tuple(
                format(Decimal(x.numerator) / x.denominator, ".70e")
                for x in (z.real, z.imag)
            )
            for z in map(rx.exact, precise)
        ]
        reference = [
            rx.exact(tuple(map(Fraction, value.to_decimal_tuple())))
            for value in evaluator.evaluate_with_precision(rows, 60)
        ]
        inputs = [exact.carrier.number(x) for x in values]
        mine = [rx.exact(x) for x in kernel.evaluate(inputs, exact)]
        scale = max(map(_magnitude, reference)) or 1
        assert (
            max(_magnitude(a - b) for a, b in zip(mine, reference, strict=True))
            <= 1e-55 * scale
        ), spec


# ------------------------------------------------------------------ schedules


@pytest.mark.loop_fast
@pytest.mark.parametrize(
    ("incoming", "outgoing", "born"),
    [
        (["g", "g"], ["g", "g", "g"], {"QCD": 3}),
        (["u", "u~"], ["d", "d~", "g"], {"QCD": 3}),
        (["u", "u~"], ["u", "u~", "g"], {"QCD": 3}),
    ],
)
def test_merged_paths_are_exactly_the_skeletons(incoming, outgoing, born):
    """Every cut-to-closure path of schedule M is one skeleton: merging a
    longer ring's prefix with a bubble's remainder would count the mirror
    reading of another bubble twice."""

    from pyamplicol.loops.control_models import qcd_qed_control_model

    model = qcd_qed_control_model()
    catalogue = ModelLoopCatalogue(model)
    loop = {**born, "QCD": born["QCD"] + 2}
    selection = LoopOrderSelection(
        frozenset({"QCD"}), frozenset({tuple(sorted(loop.items()))})
    )
    table = ModelBlockTable(model, incoming, outgoing, cap=selection.cap)
    families = generate_families(table, catalogue, selection)
    colour = PlanarWords(len(families.legs))
    skeletons = [
        skeleton
        for family in families.families.values()
        for skeleton in family_skeletons(families, family, colour, catalogue)
    ]
    graph = merged_graph(skeletons)
    children = collections.defaultdict(list)
    for parent, child, label in graph.edges:
        children[parent].append((child, label))
    real = {tuple(c.label for c in s.contents) for s in skeletons}
    paths = []
    stack = [(root, ()) for root in graph.entries]
    while stack:
        state, labels = stack.pop()
        if state in graph.closures:
            paths.append(labels)
        stack.extend((child, (*labels, label)) for child, label in children[state])
    assert sorted(paths) == sorted(real)
    assert all(graph.starts[s] in graph.entries for s in skeletons)


class _Case:
    """One loop request lowered to a B plan, kernels, seeds and an executor."""

    def __init__(
        self,
        compiled,
        kernels,
        layout,
        process,
        colour_model,
        *,
        lc=False,
        forward=False,
    ):
        from pyamplicol.loops.compiler import prepare_loop_request_model
        from pyamplicol.loops.v2.blocks import build_tree_blocks
        from pyamplicol.loops.v2.colour import SeedColour, born_basis
        from pyamplicol.models.external import CompiledUFOModel
        from pyamplicol.processes.model import build_model_process_ir

        text, born_bounds, loop_bounds = process
        structure = prepare_loop_request_model(
            compiled,
            born_order_bounds=born_bounds,
            loop_order_bounds=loop_bounds,
            fixed_parameters={"MB": "0"},
        ).structure
        self.data = json.loads(structure.native.to_json())
        catalogue = ModelLoopCatalogue(
            self.data, species_orders=structure.perturbative_species_orders
        )
        born = add_orders(tuple(born_bounds.items()))
        loop = add_orders(tuple(loop_bounds.items()))
        process_ir = build_model_process_ir(text, compiled.ir, color_accuracy="full")
        blocks = build_tree_blocks(
            process_ir,
            CompiledUFOModel(compiled),
            born_coupling_orders=born_bounds,
            block_coupling_orders=loop_bounds,
        )
        families = generate_families(
            blocks.block_table(self.data),
            catalogue,
            LoopOrderSelection(frozenset({"QCD"}), frozenset({loop})),
        )
        legs = len(families.legs)
        self.momenta = _momenta(legs, 7)
        generator = random.Random(3)
        self.block_rows = rx.SyntheticBlockRows(
            [(-1, 1)] * legs, chiralities=rx.block_chiralities(kernels), seed=3
        )
        full = self.block_rows.row_set((1 << legs) - 1)
        if colour_model == "seed":
            colour = SeedColour(blocks, self.data, compiled.ir.vertex_terms)
            self.plan = build_schedules(
                families, catalogue, colour, policy=cheapest(_rows)
            )
            basis = born_basis(blocks)
            amplitudes = {
                row: {
                    (b, born): rx.Complex(_rational(generator), _rational(generator))
                    for b in range(len(basis))
                }
                for row in full
            }
            self.seeds = rx.ColourSeeds.build(
                colour, rx.reading_skeletons(self.plan), amplitudes, {(born, loop)}
            )
        else:
            colour = PlanarWords(legs)
            keep = None
            if lc:
                # All-gluon LC: the face rule drops exactly the quark loops.
                def keep(skeleton):
                    return not any(
                        abs(catalogue.species(name).colour) == 3
                        for name in skeleton.species
                    )

            view = SkeletonView(catalogue, colour, keep=keep)
            policy, choose = cheapest(_rows), None
            if forward:
                # Word seeds belong to the F reading; read every skeleton in
                # its own direction (a reversed word needs its own seed).
                def is_forward(trie):
                    return all(
                        ring in view.view(skeleton).readings[: len(ring)]
                        for skeleton, ring in trie.leaves
                    )

                cost = cheapest(_rows)

                def policy(family, candidates):
                    return cost(family, [c for c in candidates if is_forward(c[1])])

                def choose(rings):
                    return rings[0]

            options = {"view": view} | ({"choose": choose} if choose else {})
            self.plan = build_schedules(
                families, catalogue, colour, policy=policy, **options
            )
            self.seeds = rx.SyntheticSeeds(full, seed=5)
        self.compiled = compiled
        self.kernels = kernels
        self.layout = layout
        self.masses = lambda name: catalogue.species(name).mass

    def executor(self, carrier=None, form="fused", numeric=True):
        carrier = carrier or rx.Carrier.exact()
        constants = rx.KernelConstants(_parameters(self.compiled), carrier)
        kinematics = rx.CatalogueKinematics(
            self.kernels,
            self.compiled.ir.vertex_terms,
            self.data,
            self.block_rows,
            self.momenta,
            constants,
        )
        seeds = self.seeds
        if not numeric:
            seeds = rx.SyntheticSeeds(
                self.block_rows.row_set((1 << len(self.momenta)) - 1), values=False
            )
        return rx.ReferenceExecutor(
            kinematics,
            seeds,
            momenta=self.momenta,
            layout=self.layout,
            carrier=carrier,
            form=form,
            numeric=numeric,
        )


def _assert_equal(left: rx.Execution, right: rx.Execution) -> None:
    """Family numerators equal exactly (frames, then every coefficient)."""

    assert left.numerators.keys() == right.numerators.keys()
    for key, mine in left.numerators.items():
        other = right.numerators[key]
        assert mine.frame == other.frame, key
        rank = max(mine.rank, other.rank)
        for a, b in zip(mine.columns, other.columns, strict=True):
            a = list(a) + [0] * (size(rank) - len(a))
            b = list(b) + [0] * (size(rank) - len(b))
            assert a == b, key


def _assert_nonzero(execution: rx.Execution) -> None:
    """Every compared family numerator is nonzero (no vacuous identity)."""

    assert execution.numerators
    for key, numerator in execution.numerators.items():
        assert any(any(column) for column in numerator.columns), key


def _assert_o5(case: _Case, executor, execution: rx.Execution) -> None:
    generator = random.Random(17)
    q = [rx.exact(_rational(generator)) for _ in range(4)]
    with executor.carrier.scope():
        for (family, orders), numerator in execution.numerators.items():
            direct = rx.direct_family_value(executor, case.plan, family, orders, q)
            point = [executor.carrier.number(x) for x in q]
            value = [
                evaluate(column, case.layout, point) for column in numerator.columns
            ]
            assert direct == value, family


@pytest.mark.loop_physics
@pytest.mark.parametrize(
    "process", [GLUONS, QUARKS, DRELL_YAN], ids=["gg_gg", "uux_ddxg", "uux_emepg"]
)
def test_generic_colour_schedules_agree_exactly(
    compiled, kernels, layout, loop_native, process
):
    """O4 and O5 in full colour with WP1.4 seed classes: B in its cost-optimal
    frames (some read against the loop direction), F mapped skeleton by
    skeleton onto them, fused and split forms; M needs colour words and
    falls back to B."""

    case = _Case(compiled, kernels, layout, process, "seed")
    fused = case.executor()
    b = fused.run_b(case.plan)
    _assert_nonzero(b)
    _assert_equal(b, case.executor(form="split").run_b(case.plan))
    skeletons = rx.plan_skeletons(case.plan)
    _assert_equal(b, fused.run_f(skeletons, rx.trie_targets(case.plan)))
    with pytest.raises(rx.MScheduleUnavailable, match="colour-ordered"):
        fused.run_m(skeletons, rx.trie_targets(case.plan))
    _assert_o5(case, fused, b)


@pytest.mark.loop_physics
@pytest.mark.parametrize(
    ("process", "lc"),
    [(GLUONS, True), (GLUONS, False), (QUARKS, False)],
    ids=["gg_gg-lc", "gg_gg-fc", "uux_ddxg-fc"],
)
def test_colour_word_schedules_agree_exactly(
    compiled, kernels, layout, loop_native, process, lc
):
    """O4 with colour-ordered contents (seeds per cut word): F = B = M, also
    on a quark line."""

    case = _Case(compiled, kernels, layout, process, "planar", lc=lc, forward=True)
    executor = case.executor()
    skeletons = rx.plan_skeletons(case.plan)
    b = executor.run_b(case.plan)
    _assert_nonzero(b)
    _assert_equal(b, executor.run_f(skeletons, rx.trie_targets(case.plan)))
    _assert_equal(b, executor.run_m(skeletons, rx.trie_targets(case.plan)))
    accuracy = "lc" if lc else "full"
    if not lc:  # no FC certificate exists yet (WP2.4): M falls back to B
        gated = executor.run(case.plan, "M", accuracy)
        assert "certificates" in gated.fallback
        _assert_equal(gated, b)
    # The word identities are algebraic: run M as if every class were
    # certified, in the F frames of the families.
    m = executor.run(case.plan, "M", accuracy, certified=lambda skeleton: True)
    assert m.fallback is None
    forward = rx.forward_targets(skeletons, case.masses)
    _assert_equal(m, case.executor(form="split").run_m(skeletons, forward))
    _assert_o5(case, executor, b)


@pytest.mark.loop_physics
def test_carriers_agree_with_exact_arithmetic(compiled, kernels, layout, loop_native):
    case = _Case(compiled, kernels, layout, GLUONS, "seed")
    exact = case.executor(rx.Carrier.exact(60)).run_b(case.plan)
    for carrier, tolerance in ((rx.Carrier.f64(), 1e-13), (rx.Carrier.ap(50), 1e-45)):
        other = case.executor(carrier).run_b(case.plan)
        assert other.counts == exact.counts
        for key, mine in exact.numerators.items():
            for a, b in zip(mine.columns, other.numerators[key].columns, strict=True):
                values = [rx.exact(x) for x in a]
                scale = max(abs(complex(x)) for x in values)
                assert scale > 0
                error = max(
                    abs(complex(x - rx.exact(y)))
                    if carrier.kind == "f64"
                    else _magnitude(x - rx.exact(y))
                    for x, y in zip(values, b, strict=True)
                )
                assert error <= tolerance * scale, (carrier, key)


def _magnitude(value) -> float:
    """|value| for exact values far below binary64 resolution."""

    norm = value.real * value.real + value.imag * value.imag
    return float(norm) ** 0.5 if norm else 0.0


@pytest.mark.loop_physics
def test_counts_need_no_numerics(compiled, kernels, layout, loop_native):
    """A count-only walk records the operations of the numeric run exactly."""

    case = _Case(compiled, kernels, layout, GLUONS, "planar", lc=True, forward=True)
    numeric, counting = case.executor(), case.executor(numeric=False)
    skeletons = rx.plan_skeletons(case.plan)
    targets = rx.trie_targets(case.plan)
    for run in (
        lambda e: e.run_b(case.plan),
        lambda e: e.run_f(skeletons, targets),
        lambda e: e.run_m(skeletons, targets),
    ):
        counted = run(counting)
        assert counted.numerators == {}
        assert counted.counts == run(numeric).counts
    counts = numeric.run_b(case.plan).counts
    assert counts["best/step"] <= min(
        counts["fused/step"],
        counts["split-accumulate/step"] + counts["split-vertex/step"],
    )


@pytest.mark.loop_physics
def test_operation_counts_regenerate(compiled):
    """The G1 operation counts (gg -> gggg LC and FC, uu~ -> e-e+dd~g and
    uu~ -> e-e+ggg) regenerate exactly. B and F always run; M runs at LC and
    falls back to B in full colour until WP2.4 certifies the class."""

    from pathlib import Path

    from tools import loop_v2_operation_counts as counts

    fixture = (
        Path(__file__).resolve().parents[1] / "fixtures/loop/v2/operation_counts.json"
    )
    expected = json.loads(fixture.read_text())
    assert counts.operation_counts(counts.prepare(compiled)) == expected
    pending = set()
    for entry in expected["processes"]:
        for accuracy, record in entry["counts"].items():
            assert "fallback" not in record["B"] and "fallback" not in record["F"]
            assert ("fallback" in record["M"]) == (accuracy == "FC")
            assert record["F"]["best/step"] > record["B"]["best/step"]
            if "M-uncertified" in record:
                assert "fallback" not in record["M-uncertified"]
                pending.add(entry["process"])
    assert pending == {"g g > g g g g", "u u~ > e- e+ g g g"}
