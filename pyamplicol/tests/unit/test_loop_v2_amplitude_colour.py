# SPDX-License-Identifier: 0BSD
"""The amplitude colour basis of loop-induced requests (M-LIT LIT-COL).

``colour.amplitude_basis`` forms, for a request without a Born, the literal
tensors T of the seed classes' expansions and of the counterterm kets, the
coefficient cells c_{c,t} read off each class's own expansion and the metric
G_{tt'} = <t|t'>, all exact Laurent polynomials in N_c (full colour only).
Its native stage (``native_cells.amplitude_basis``, rusticol-core
``cells::amplitude_basis``) must equal the Python oracle exactly
(``native_cells.native_stages(())``), independent of the thread count.

Default tier (``loop_fast``): synthetic tables of every class kind the
engine forms without generation (adjoint rings, fundamental traces, generic
classes with quark strings, U(1) pieces and 1/sqrt(2) powers) against the
oracle and closed forms; the exact colour zero; the amplitude-mode seed
contraction; g g > h, g g > h g and u u~ > h g generated with their
counterterm kets (quark rings). ``loop_physics``: g g > g g h and
g g > z z g.
"""

from __future__ import annotations

import random
from fractions import Fraction
from itertools import permutations
from typing import Any

import pytest

from pyamplicol.color.connections import (
    ColorLeg,
    ColorTensor,
    ConnectedColorTensor,
    ExactColorCoefficient,
    OpenColorString,
    TensorTerm,
)
from pyamplicol.loops.v2 import colour, native_cells
from pyamplicol.loops.v2.colour import BornBasis, Primitive, SeedTable

E = ExactColorCoefficient


def _table(legs, classes) -> SeedTable:
    legs = tuple(legs)
    return SeedTable(
        BornBasis(legs, (), (), ()),
        tuple(classes),
        tuple(frozenset() for _ in classes),
        (),
    )


def _lanes(table: SeedTable, kets=()) -> colour.AmplitudeBasis:
    """The native basis, asserted equal to the oracle's and to the native
    one on one thread."""

    with native_cells.native_stages(()):
        oracle = colour.amplitude_basis(table, kets)
    native = colour.amplitude_basis(table, kets)
    assert native == oracle
    one = native_cells.amplitude_basis(
        table, colour.amplitude_kets(table.basis.legs, kets), threads=1
    )
    assert one == native
    return native


def _trace(*labels: int) -> ColorTensor:
    return ColorTensor((), (labels,))


def _assert_expansions(basis: colour.AmplitudeBasis, table: SeedTable) -> None:
    """Each class's cells are its own expansion's coefficients on T (its
    1/sqrt(2) power folded): nothing overlapped."""

    for c, row in enumerate(basis.cells):
        kappa = table.tensor(c)
        scale = Fraction(1, 2 ** (kappa.normalization_power // 2))
        expected = {
            (basis.tensors.index(term.tensor), term.nc_power): term.coefficient.scaled(
                scale
            )
            for term in kappa.terms
        }
        assert {(t, p): v for t, cell in row for p, v in cell.items()} == expected


def _random_tensor(rng: random.Random, legs, labels, quarks) -> ConnectedColorTensor:
    """A random literal tensor on ``legs`` (open strings pairing each quark
    with an antiquark, adjoint labels spread over strings and traces), with
    complex coefficients, N_c powers and an even 1/sqrt(2) power."""

    terms = []
    while not terms:
        for _ in range(rng.randint(1, 4)):
            adjoint = list(labels)
            rng.shuffle(adjoint)
            strings = []
            for q, qbar in quarks:
                cut = rng.randint(0, len(adjoint))
                strings.append(OpenColorString(q, tuple(adjoint[:cut]), qbar))
                adjoint = adjoint[cut:]
            traces = []
            while adjoint:
                cut = rng.randint(2, max(2, len(adjoint)))
                traces.append(tuple(adjoint[:cut]))
                adjoint = adjoint[cut:]
            if any(len(t) == 1 for t in traces):
                continue
            terms.append(
                TensorTerm(
                    E(
                        Fraction(rng.randint(-3, 3), rng.randint(1, 4)),
                        Fraction(rng.randint(-2, 2), rng.randint(1, 3)),
                    ),
                    ColorTensor(tuple(strings), tuple(traces)),
                    rng.randint(-1, 1),
                )
            )
    return ConnectedColorTensor(legs, tuple(terms), 2 * rng.randint(0, 1))


# ------------------------------------------------------------ synthetic tables


@pytest.mark.loop_fast
def test_primitive_classes_equal_the_oracle_and_closed_forms():
    # Two gluons: Tr_adj(F^1 F^2) = 2 N tr(12), tr(tau^1 tau^2) = tr(12).
    two = (ColorLeg(1, 8), ColorLeg(2, 8), ColorLeg(3, 1))
    basis = _lanes(
        _table(
            two,
            [Primitive("adjoint-ring", (1, 2)), Primitive("fundamental-trace", (1, 2))],
        )
    )
    assert basis.tensors == (_trace(1, 2),)
    assert basis.cells == (((0, {1: E(2)}),), ((0, {0: E(1)}),))
    assert basis.metric == (({0: E(-1), 2: E(1)},),)
    assert basis.metric_at_three() == ((E(8),),)
    assert basis.counterterms == ()
    assert not basis.colour_zero
    # Three gluons: tr(tau^1 tau^2 tau^3) is the literal trace (3 2 1);
    # Tr_adj(F^1 F^2 F^3) = N tr(3 2 1) - N tr(1 2 3).
    three = tuple(ColorLeg(a, 8) for a in (1, 2, 3))
    basis = _lanes(
        _table(
            three,
            [
                Primitive("fundamental-trace", (1, 2, 3)),
                Primitive("fundamental-trace", (1, 3, 2)),
                Primitive("adjoint-ring", (1, 2, 3)),
            ],
        ),
        [_trace(1, 2, 3)],
    )
    assert basis.tensors == (_trace(1, 3, 2), _trace(1, 2, 3))
    assert basis.cells == (
        ((0, {0: E(1)}),),
        ((1, {0: E(1)}),),
        ((0, {1: E(1)}), (1, {1: E(-1)})),
    )
    assert basis.counterterms == (1,)
    diagonal = {-1: E(2), 1: E(-3), 3: E(1)}
    off = {-1: E(2), 1: E(-2)}
    assert basis.metric == ((diagonal, off), (off, diagonal))
    assert basis.metric_at_three() == (
        (E(Fraction(56, 3)), E(Fraction(-16, 3))),
        (E(Fraction(-16, 3)), E(Fraction(56, 3))),
    )
    # Projected once per basis (an in-memory slice squares every point with
    # it): the same exact values, the same object on every call.
    assert basis.metric_at_three() is basis.metric_at_three()
    # Five gluons: adjoint rings and fundamental traces of a sample of
    # words (single and double traces of their SU(N_c) expansions) and a
    # double-trace ket.
    five = tuple(ColorLeg(a, 8) for a in range(1, 6))
    words = list(permutations(range(1, 6)))[::17]
    kinds = ("adjoint-ring", "fundamental-trace")
    table = _table(five, [Primitive(kind, w) for w in words for kind in kinds])
    ket = ColorTensor((), ((1, 2), (3, 4, 5)))
    basis = _lanes(table, [ket])
    _assert_expansions(basis, table)
    assert basis.tensors[basis.counterterms[0]] == ket
    assert len(basis.metric) == len(basis.tensors) > len(words)


@pytest.mark.loop_fast
def test_generic_classes_equal_the_oracle():
    rng = random.Random(20261106)
    for quarks, gluons in (((), 4), (((1, 2),), 3), (((1, 2), (3, 4)), 2)):
        legs = [ColorLeg(q, 3) for q, _ in quarks] + [
            ColorLeg(b, -3) for _, b in quarks
        ]
        labels = list(range(10, 10 + gluons))
        legs += [ColorLeg(a, 8) for a in labels]
        classes = [_random_tensor(rng, legs, labels, quarks) for _ in range(12)]
        kets = [
            term.tensor
            for ket in (_random_tensor(rng, legs, labels, quarks) for _ in range(4))
            for term in ket.terms
        ]
        table = _table(legs, classes)
        basis = _lanes(table, kets)
        _assert_expansions(basis, table)
        assert {basis.tensors[t] for t in basis.counterterms} == set(kets)
        # The metric is <t|t'> of the elements.
        singles = basis.seed_basis.tensors
        assert basis.metric == tuple(
            tuple(colour.contract_colour(a, b) for b in singles) for a in singles
        )


@pytest.mark.loop_fast
def test_sqrt2_powers_fold_or_are_refused():
    legs = (ColorLeg(1, 8), ColorLeg(2, 8))

    def single(power: int) -> ConnectedColorTensor:
        return ConnectedColorTensor(legs, (TensorTerm(E(3), _trace(1, 2), -1),), power)

    basis = _lanes(_table(legs, [single(2)]))
    assert basis.cells == (((0, {-1: E(Fraction(3, 2))}),),)
    for lane in (native_cells.STAGES, ()):
        with (
            native_cells.native_stages(lane),
            pytest.raises(ValueError, match="sqrt\\(2\\)"),
        ):
            colour.amplitude_basis(_table(legs, [single(1)]))


@pytest.mark.loop_fast
def test_colour_zero_is_exact_and_needs_every_cell_and_ket_to_vanish():
    # h > g a: tr(tau^a) = 0, and a ket on one adjoint label vanishes too.
    one = (ColorLeg(1, 8), ColorLeg(2, 1), ColorLeg(3, 1))
    basis = _lanes(
        _table(one, [Primitive("fundamental-trace", (1,))]), [ColorTensor((), ((1,),))]
    )
    assert basis.tensors == () and basis.cells == ((),) and basis.colour_zero
    # A cell that vanishes at N_c = 3 only, (N_c^2 - 9) tr(12): zero by
    # colour unless a counterterm ket carries a tensor.
    legs = (ColorLeg(1, 8), ColorLeg(2, 8))
    vanishing = ConnectedColorTensor(
        legs, (TensorTerm(E(1), _trace(1, 2), 2), TensorTerm(E(-9), _trace(1, 2), 0))
    )
    basis = _lanes(_table(legs, [vanishing]))
    assert basis.cells == (((0, {0: E(-9), 2: E(1)}),),)
    assert basis.cells_at_three() == ((),) and basis.colour_zero
    assert not _lanes(_table(legs, [vanishing]), [_trace(1, 2)]).colour_zero
    assert not _lanes(_table(legs, [Primitive("adjoint-ring", (1, 2))])).colour_zero


@pytest.mark.loop_fast
def test_amplitude_mode_contracts_seeds_directly():
    from pyamplicol.loops.v2.refusals import LoopRequestUnsupported

    assert colour.amplitude_contraction("auto", "g g > h") == "direct"
    assert colour.amplitude_contraction("direct", "g g > h") == "direct"
    with pytest.raises(LoopRequestUnsupported) as refused:
        colour.amplitude_contraction("fft", "g g > h")
    assert refused.value.key == "forced-seed-fft-unserved"
    with pytest.raises(ValueError, match="unknown colour contraction"):
        colour.amplitude_contraction("planar", "g g > h")


# ------------------------------------------------------------ generated slices

# name: (process, loop amplitude orders, request options).
PROCESSES: dict[str, tuple[str, dict[str, int], dict[str, Any]]] = {
    "gg_h": ("g g > h", {"QCD": 2, "QED": 1}, {}),
    "gg_hg": ("g g > h g", {"QCD": 3, "QED": 1}, {"fixed_parameters": {"MB": "0"}}),
    "uux_hg": ("u u~ > h g", {"QCD": 3, "QED": 1}, {"fixed_parameters": {"MB": "0"}}),
    "gg_ggh": ("g g > g g h", {"QCD": 4, "QED": 1}, {"fixed_parameters": {"MB": "0"}}),
    "gg_zzg": ("g g > z z g", {"QCD": 3, "QED": 2}, {"fixed_parameters": {"MB": "0"}}),
}
# |T| and the metric's diagonal / off-diagonal values at N_c = 3.
EXPECTED = {
    "gg_h": (1, {Fraction(8)}, set()),
    "gg_hg": (2, {Fraction(56, 3)}, {Fraction(-16, 3)}),
    "uux_hg": (1, {Fraction(8)}, set()),
    "gg_ggh": (6, {Fraction(152, 3)}, {Fraction(32, 3), Fraction(-16, 3)}),
    "gg_zzg": (2, {Fraction(56, 3)}, {Fraction(-16, 3)}),
}


def _request(name: str, **options: Any):
    from pyamplicol.loops.v2.assemble import SliceRequest

    process, orders, base = PROCESSES[name]
    # A loop-induced request: no tree exists at the loop amplitude orders,
    # so the Born is empty (``blocks.born_orders == ()``).
    return SliceRequest(process, dict(orders), dict(orders), **{**base, **options})


@pytest.fixture(scope="module")
def compiled(loop_sm_compiled_model):
    from pyamplicol.models.loading import load_compiled_model

    return load_compiled_model(loop_sm_compiled_model)


@pytest.fixture(scope="module")
def generated(compiled, loop_native):
    from pyamplicol.loops.v2.assemble import LoopSlice

    built: dict[str, Any] = {}

    def get(name: str):
        if name not in built:
            built[name] = LoopSlice.generate(compiled, _request(name), loop_native)
        return built[name]

    return get


def _check_generated(slice_, name: str) -> None:
    assert slice_.blocks.born_orders == ()
    assert slice_.fourier is None
    kets = native_cells.counterterm_tensors(slice_)
    assert kets, "the R2 kets carry colour"
    basis = _lanes(slice_.table, kets)
    size, diagonal, off = EXPECTED[name]
    assert len(basis) == size
    assert not basis.colour_zero
    metric = basis.metric_at_three()
    assert {metric[t][t].real for t in range(size)} == diagonal
    assert {
        metric[t][u].real for t in range(size) for u in range(size) if t != u
    } == off
    assert all(not cell.imag for row in metric for cell in row)
    _assert_expansions(basis, slice_.table)


@pytest.mark.loop_fast
@pytest.mark.parametrize("name", ["gg_h", "gg_hg", "uux_hg"])
def test_generated_bases_equal_the_oracle(generated, name):
    _check_generated(generated(name), name)


@pytest.mark.loop_physics
@pytest.mark.parametrize("name", ["gg_ggh", "gg_zzg"])
def test_generated_bases_equal_the_oracle_beyond_two_to_two(generated, name):
    _check_generated(generated(name), name)


@pytest.mark.loop_fast
def test_forced_seed_fft_is_refused_without_a_born(compiled, loop_native):
    from pyamplicol.loops.v2.assemble import LoopSlice
    from pyamplicol.loops.v2.refusals import LoopRequestUnsupported

    with pytest.raises(LoopRequestUnsupported) as refused:
        LoopSlice.generate(
            compiled, _request("gg_h", colour_contraction="fft"), loop_native
        )
    assert refused.value.key == "forced-seed-fft-unserved"
    assert "no Born" in str(refused.value)
