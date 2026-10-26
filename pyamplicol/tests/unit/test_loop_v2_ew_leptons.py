# SPDX-License-Identifier: 0BSD
"""Colour-singlet fermion lines through the loop and at the root (WP3.2 EWI-5).

The blocks carry the tree engine's Koszul signs of colour-singlet fermions
(``generation/fermion_ordering.py``): a block row is its current's
coefficient of the product of its singlet sources in source order. The loop
leaf weights and the root closure complete them with one rule,
``conventions.colour_singlet_sign``: the Koszul sign of the blocks' singlet
supports in the order they are read, times the merge sign of the open ends
each line joins outside the blocks, listed (antiparticle, particle) as the
tree engine's two-fermion merge takes them. Checks:

* the rule's defining identity, against an independent count of inversions:
  the product of the blocks' own Koszul signs and the rule is the sign of the
  whole diagram, every line listed (antiparticle, particle), in every block
  order; with every block one source, that sign is the tree engine's
  ``FermionOrderingSign.tree_sign``;
* processes with two colour-singlet lines, refused before EWI-5 (Born closure
  / tree runtime 0.53 and 0.34): the Born closed at leg 1 is the tree
  runtime's amplitude up to a real sign, with identical leptons too, and Born
  and virtual (QCD corrections) are invariant under a relabelling of the
  final state that changes the interleaving of the lines' supports (without
  the loop's Koszul sign the virtual moves by O(1)); the leaf weights' fast
  path equals the full rule there (``check_leaf_weights``, WP3.1 D1).
"""

from __future__ import annotations

import itertools
import random
from pathlib import Path
from types import SimpleNamespace

import pytest

from pyamplicol.loops.v2.conventions import (
    ANTIPARTICLE,
    PARTICLE,
    FermionSignError,
    colour_singlet_sign,
)

pytestmark = pytest.mark.loop_physics

TOLERANCE = 1e-12
_SIBLINGS: dict[str, object] = {}


def _sibling(name: str):
    """A helper module of this directory (pytest imports test modules by
    path, ``--import-mode=importlib``), loaded once under a private name."""

    import importlib.util

    if name not in _SIBLINGS:
        path = Path(__file__).with_name(f"{name}.py")
        spec = importlib.util.spec_from_file_location(f"_ew_leptons_{name}", path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _SIBLINGS[name] = module
    return _SIBLINGS[name]


# ---------------------------------------------------------------- the rule


def _parity(sequence) -> int:
    """Sign of the permutation sorting ``sequence`` (an independent count)."""

    inversions = sum(a > b for i, a in enumerate(sequence) for b in sequence[i + 1 :])
    return -1 if inversions % 2 else 1


def _diagram(rng: random.Random):
    """A random diagram's singlet lines and blocks: each line is closed
    inside one block or joins the open ends of two blocks; a block holds
    closed lines and at most one open end. Returns (lines as (antiparticle,
    particle) labels, blocks as (closed lines, open end or None))."""

    count = rng.randint(1, 5)
    labels = list(range(1, 2 * count + 1))
    rng.shuffle(labels)
    lines = [tuple(labels[2 * i : 2 * i + 2]) for i in range(count)]
    blocks: list[tuple[list, int | None]] = []
    for line in lines:
        if rng.random() < 0.5:
            blocks.append(([line], None))
        else:
            blocks.extend(([], end) for end in line)
    rng.shuffle(blocks)
    merged: list[tuple[list, int | None]] = []
    for closed, end in blocks:
        if merged and rng.random() < 0.4 and (end is None or merged[-1][1] is None):
            previous, other = merged[-1]
            merged[-1] = (previous + closed, end if other is None else other)
        else:
            merged.append((list(closed), end))
    return lines, merged


def _block_sign(closed, end) -> int:
    """A block row's Koszul sign: its sources in order against its closed
    lines listed (antiparticle, particle), the open end last."""

    listing = [label for line in closed for label in line]
    if end is not None:
        listing.append(end)
    position = {label: rank for rank, label in enumerate(sorted(listing))}
    return _parity([position[label] for label in listing])


def _rule(lines, blocks) -> int:
    supports = [
        tuple(label for line in closed for label in line)
        + (() if end is None else (end,))
        for closed, end in blocks
    ]
    index = {end: i for i, (_closed, end) in enumerate(blocks) if end is not None}
    joined = [
        (index[anti], ANTIPARTICLE, index[part], PARTICLE)
        for anti, part in lines
        if anti in index
    ]
    return colour_singlet_sign(supports, joined)


@pytest.mark.loop_fast
def test_singlet_sign_completes_the_blocks_to_the_diagram_sign():
    """Blocks' Koszul signs times the rule = the diagram's sign with every
    line listed (antiparticle, particle), in every order of the blocks."""

    rng = random.Random(20261026)
    for _case in range(400):
        lines, blocks = _diagram(rng)
        diagram = _parity([label for line in lines for label in line])
        for order in itertools.islice(itertools.permutations(blocks), 24):
            product = 1
            for closed, end in order:
                product *= _block_sign(closed, end)
            assert product * _rule(lines, list(order)) == diagram, (lines, order)


class _Model:
    """The particle facts ``FermionOrderingSign`` reads: colour-singlet
    Dirac fermions, positive codes particles."""

    def is_fermion(self, particle_id: int) -> bool:
        return True

    def color_rep(self, particle_id: int) -> int:
        return 1

    def source_orientation(self, particle_id: int) -> str:
        return PARTICLE if particle_id > 0 else ANTIPARTICLE


@pytest.mark.loop_fast
def test_singlet_sign_of_single_sources_is_the_tree_engines_tree_sign():
    """With every block one source (a tree's external legs), the rule is
    the tree engine's sign of a tree with these lines."""

    from pyamplicol.generation.fermion_ordering import FermionOrderingSign

    rng = random.Random(7)
    for _case in range(200):
        count = rng.randint(1, 4)
        labels = list(range(1, 2 * count + 1))
        rng.shuffle(labels)
        lines = [tuple(labels[2 * i : 2 * i + 2]) for i in range(count)]
        codes = {anti: -11 for anti, _part in lines}
        codes.update({part: 11 for _anti, part in lines})
        legs = {
            label: SimpleNamespace(
                label=label,
                statistics="fermion",
                color_role="singlet",
                outgoing_pdg=codes[label],
            )
            for label in sorted(codes)
        }
        process = SimpleNamespace(legs=tuple(legs.values()))
        engine = FermionOrderingSign(process, _Model())
        pairs = [(legs[a], legs[p]) for a, p in lines]
        rng.shuffle(pairs)
        expected = engine.tree_sign([tuple(rng.sample(pair, 2)) for pair in pairs])
        order = sorted(codes, key=lambda _label: rng.random())
        index = {label: i for i, label in enumerate(order)}
        assert (
            colour_singlet_sign(
                [(label,) for label in order],
                [(index[a], ANTIPARTICLE, index[p], PARTICLE) for a, p in lines],
            )
            == expected
        )


@pytest.mark.loop_fast
def test_singlet_sign_refuses_lines_it_cannot_pair():
    """Two ends of one orientation (Majorana-like), or a line whose two ends
    are one block's, are refused."""

    with pytest.raises(FermionSignError):
        colour_singlet_sign([(1,), (2,)], [(0, PARTICLE, 1, PARTICLE)])
    with pytest.raises(FermionSignError):
        colour_singlet_sign([(1, 2)], [(0, ANTIPARTICLE, 0, PARTICLE)])


# ------------------------------------------- two singlet lines (QCD, loop_sm)

ZERO_WIDTHS = {"MB": "0", "WT": "0", "WZ": "0", "WW": "0", "WH": "0"}
PARAMETERS = {"aS": "0.118", "MT": "173"}
# An exact massless 2 -> 4 point at sqrt(s) = 1000 (rational directions).
FOUR_LEPTON_POINT = [
    ["500", "0", "0", "500"],
    ["500", "0", "0", "-500"],
    ["8245250/27301", "-7611000/27301", "0", "-3171250/27301"],
    ["883500/27301", "72000/27301", "756000/27301", "451500/27301"],
    ["11308500/27301", "7539000/27301", "-7539000/27301", "3769500/27301"],
    ["6863750/27301", "0", "6783000/27301", "-1049750/27301"],
]
# (process, relabelled process, the relabelled legs' indices in the first):
# the first two interleave the supports of the lines' Z/photon blocks
# ({3, 5} and {4, 6}) where the relabelled ones do not; e- e- e+ e+ has two
# pairings.
RELABELLINGS = [
    ("u u~ > e- m- e+ m+", "u u~ > e- e+ m- m+", (0, 1, 2, 4, 3, 5)),
    ("u u~ > e- e- e+ e+", "u u~ > e- e+ e- e+", (0, 1, 2, 4, 3, 5)),
]


@pytest.fixture(scope="module")
def four_leptons(loop_sm_compiled_model, loop_native):
    from pyamplicol.loops.v2.assemble import LoopSlice, SliceRequest
    from pyamplicol.models.loading import load_compiled_model

    compiled = load_compiled_model(loop_sm_compiled_model)
    built = {}

    def build(process: str):
        if process not in built:
            built[process] = LoopSlice.generate(
                compiled,
                SliceRequest(
                    process,
                    {"QCD": 0, "QED": 4},
                    {"QCD": 2, "QED": 4},
                    fixed_parameters=ZERO_WIDTHS,
                ),
                loop_native,
            )
        return built[process]

    return build


def _totals(result) -> tuple[float, list[float]]:
    born = sum(float(value) for value in result.born.values())
    virtual = [0.0, 0.0, 0.0]
    for laurent in result.components["virtual"].values():
        virtual = [
            a + float(b) for a, b in zip(virtual, laurent.coefficients, strict=True)
        ]
    return born, virtual


@pytest.mark.parametrize(("process", "relabelled", "legs"), RELABELLINGS)
def test_two_singlet_lines_are_invariant_under_relabelling(
    four_leptons, process, relabelled, legs
):
    """Two colour-singlet lines (refused before EWI-5): the Born closed at
    leg 1 is the tree runtime's amplitude up to a real sign, and Born and
    virtual (QCD corrections: the quark line through the loop, the lepton
    lines in Z/photon blocks) equal the relabelled process's at the
    relabelled momenta. The relabelling changes the Koszul sign of the
    blocks' supports; without it in the loop the virtual moves by O(1)
    (0.8-1.6 relative at this point), without it at the root the Born
    closure did (0.53, 0.34). The leaf weights' fast path equals the full
    rule on every reading (WP3.1 D1's guard)."""

    _born_closure_ratio = _sibling("test_loop_v2_ew_lines")._born_closure_ratio
    check_leaf_weights = _sibling("test_loop_v2_native_lowering").check_leaf_weights
    first, second = four_leptons(process), four_leptons(relabelled)
    ratio, residual = _born_closure_ratio(
        first, FOUR_LEPTON_POINT, "1000000", PARAMETERS
    )
    assert abs(abs(ratio) - 1) <= TOLERANCE and abs(ratio.imag) <= TOLERANCE, ratio
    assert residual <= TOLERANCE, residual
    check_leaf_weights(first)
    born, virtual = _totals(
        first.evaluate(FOUR_LEPTON_POINT, "1000000", parameters=PARAMETERS)
    )
    other_born, other_virtual = _totals(
        second.evaluate(
            [FOUR_LEPTON_POINT[i] for i in legs], "1000000", parameters=PARAMETERS
        )
    )
    assert abs(born - other_born) <= TOLERANCE * abs(born), (born, other_born)
    scale = max(map(abs, virtual))
    for mine, theirs in zip(virtual, other_virtual, strict=True):
        assert abs(mine - theirs) <= 1e-12 * scale, (virtual, other_virtual)


# ------------------------------- the legacy loop path as an EW oracle (L1/L2)


def _undo_projector_exchange(original):
    """``loops.vertex_frames.native_physical_vertex`` with its FFS
    ProjM/ProjP exchange undone (the same simultaneous replacement applied
    once more): the negative control."""

    def vertex(graph, vertex_id, lorentz):
        from symbolica import Replacement
        from symbolica.community import tensor as t

        from pyamplicol._internal.physics.symbols import symbols

        result = original(graph, vertex_id, lorentz)
        spins = {edge.id: edge.spin for edge in graph.edges}
        attached = sorted(
            spins[edge]
            for _half, vertex_, edge, _flow in graph.half_edges
            if vertex_ == vertex_id
        )
        if attached != [1, 2, 2]:
            return result
        left, right = symbols.qualified(
            "loop_vertex_frame::left_", "loop_vertex_frame::right_"
        )
        minus = t.TensorExpression.projm(4)(left, right).to_expression()
        plus = t.TensorExpression.projp(4)(left, right).to_expression()
        return result.replace_multiple(
            [Replacement(minus, plus), Replacement(plus, minus)]
        )

    return vertex


@pytest.mark.loop_nightly
def test_legacy_ffs_projector_exchange_is_checked_by_the_massive_muon_control(
    loop_gmu_ufo, loop_native, monkeypatch
):
    """The legacy loop path (``loops.vertex_frames``) keeps its own ProjM/
    ProjP exchange of FFS vertices, which ``models.tensors`` dropped
    (EWI-4a). Before it serves as the second engine (L1/L2, O2) of the
    electroweak controls it is re-checked with a Goldstone-sign-sensitive
    oracle: MadLoop's isolated FFS loops of e- e+ > mu- mu+ with a massive
    muon (the G0 and G+- Yukawas of the muon and the top;
    ``tools/loop_reference/massive_muon_ew.json``). The exchange gives
    MadLoop's FFS subset (2e-8, the legacy test's tolerance); undone, the
    subset misses by O(1) and the bare virtual by 1e-7 (the 2e-9 component
    check of ``test_loop_physical_massive_source`` fails)."""

    import numpy as np

    import pyamplicol.loops.graph_basis as graph_basis

    legacy = _sibling("test_loop_physical_massive_source")
    reference = legacy.reference.__wrapped__()
    points = np.asarray(reference["points"], dtype=str)
    scales = reference["scales_squared"]
    expected = np.asarray(reference["ffs_bare"])

    def ffs_subset():
        structure, prepared = legacy.source_model.__wrapped__(
            loop_gmu_ufo, loop_native
        )
        from symbolica import E

        from pyamplicol.loops.physical_source import source_model_program

        program = source_model_program(
            structure,
            prepared,
            ("e-", "e+"),
            ("m-", "m+"),
            selected_corrections=["QED"],
            state_kinds=("u", "v_bar", "u_bar", "v"),
            initial_average=E("4"),
            born_order_bounds={"QCD": 0, "QED": 2},
            loop_order_bounds={"QCD": 0, "QED": 4},
            parameters=reference["parameters"],
            arithmetic="double",
            reduction_engine="fast",
            coefficient_engine="jit",
        )
        bare = program.source_bare.bare
        filtered = legacy._ffs_only(bare, prepared.native)
        values = filtered.evaluate_decimal(points, scales).virtual[:, :, 0]
        return values.astype(float)

    np.testing.assert_allclose(ffs_subset(), expected, rtol=2e-8, atol=1e-19)
    monkeypatch.setattr(
        graph_basis,
        "native_physical_vertex",
        _undo_projector_exchange(graph_basis.native_physical_vertex),
    )
    undone = ffs_subset()
    miss = np.max(np.abs(undone - expected)) / np.max(np.abs(expected))
    assert miss > 1e-2, miss
