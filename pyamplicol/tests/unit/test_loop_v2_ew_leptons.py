# SPDX-License-Identifier: 0BSD
"""Colour-singlet fermion lines through the loop and at the root (WP3.2 EWI-5).

The blocks carry the tree engine's Koszul signs of colour-singlet fermions
(``generation/fermion_ordering.py``): a block row is its current's
coefficient of the product of its singlet sources in source order. The loop
leaf weights and the root closure complete them with one rule,
``conventions.colour_singlet_sign``: the Koszul sign of the blocks' singlet
supports in the order they are read, times the merge sign of the open ends
each line joins outside the blocks, listed (antiparticle, particle) as the
tree engine's two-fermion merge takes them. Any number of singlet lines,
mixed with any number of coloured ones (whose ordering the colour flow
owns). Checks:

* the rule's defining identity, against an independent count of inversions:
  the product of the blocks' own Koszul signs and the rule is the sign of the
  whole diagram, every line listed (antiparticle, particle), in every block
  order; with every block one source, that sign is the tree engine's
  ``FermionOrderingSign.tree_sign``;
* two colour-singlet lines and a quark line (QCD corrections, loop_sm),
  refused before EWI-5 (Born closure / tree runtime 0.53 and 0.34): the Born
  closed at leg 1 is the tree runtime's amplitude up to a real sign, with
  identical leptons too; Born and virtual are invariant under a relabelling
  of the final state that changes the interleaving of the lines' supports
  (without the loop's Koszul sign the virtual moves by O(1)); the leaf
  weights' fast path equals the full rule (``check_leaf_weights``, WP3.1 D1);
  u u~ > e- e+ mu- mu+ equals MadLoop (``required/uux_emepmummp``);
* nightly, full electroweak corrections with lepton lines through the loop:
  the four MadLoop G_mu controls (``electroweak-*-control.json``, Feynman
  gauge, lhv = 1) per Laurent coefficient and component (bare, R2, UV,
  field and counterterm parts), Bhabha on loop_qcd_qed_sm against MadLoop
  (``required/emep_emep_ew``: the s- and t-channel lines through the boxes),
  O2 per family against the legacy engine (L1/L2) on e- e+ > mu- mu+ with
  massless and massive muons, the poles of e- e+ > mu- mu+ on
  loop_qcd_qed_sm against the QED Catani operator (full EW; photon-only a
  strict xfail: the veto splits counterterm couplings), the legacy engine's
  FFS projector exchange checked against MadLoop before the legacy engine
  serves as the second engine, and the native point inputs of
  e- e+ > mu- mu+ (a strict xfail until EWI-N);
* the tree engine's reading sign of colour-singlet loop segments
  (``catalogue._CatalogueBuilder.reading_sign``): the tree engine reads
  every vertex all-outgoing, so a colour-singlet vertex odd in the momenta of
  its legs read on their own legs (Z G0 H, W G0 G+-, W G+- H) is the
  segment's reading times -1, and the ghost-G0 vertices take the Goldstones'
  field sign. Without it the Z-G0-H triangles on an open fermion line had the
  opposite sign of legacy and MadLoop (O2 rel 2.0 on those families of the
  massive-muon control, which missed MadLoop by 1e-7); checked by the
  massive-muon control, its O2 and, nightly, g g > t t~ e- e+ against
  MadLoop, Recola and OpenLoops (``required/gg_ttxemep_ew``: off by O(1) in
  the finite part without the sign).
"""

from __future__ import annotations

import itertools
import json
import math
import random
import shutil
from fractions import Fraction
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

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests/fixtures/loop"
TOLERANCE = 1e-12
LAURENT = ("finite", "single_pole", "double_pole")
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


def _metric(got: float, want: float, floor: float) -> float:
    """The fixtures' metric: |a - b| / max(|a|, |b|, floor), the floor
    |Born| times the loop coupling over 2 pi for Laurent coefficients."""

    scale = max(abs(got), abs(want), floor)
    return abs(got - want) / scale if scale else 0.0


def _totals(result) -> tuple[float, list[float]]:
    """(Born, the virtual's finite part, single and double pole), summed
    over the squared-order sectors."""

    born = sum(float(value) for value in result.born.values())
    return born, _component(result, "virtual")


def _component(result, name: str) -> list[float]:
    total = [0.0, 0.0, 0.0]
    for laurent in result.components[name].values():
        total = [a + float(b) for a, b in zip(total, laurent.coefficients, strict=True)]
    return total


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


# ------------------------- the tree reading of colour-singlet loop segments

# The colour-singlet vertices of the G_mu UFO whose tree reading is minus the
# segments' (sorted particles): odd in the momenta of legs read on their own
# legs (Z G0 H) or through the conjugate rule (W G0 G+-, W G+- H), and the
# ghost vertices of the negated G0.
READING_FLIPS = {
    ("G0", "H", "Z"),
    ("G+", "G0", "W-"),
    ("G-", "G0", "W+"),
    ("G+", "H", "W-"),
    ("G-", "H", "W+"),
    ("G0", "ghWm", "ghWm~"),
    ("G0", "ghWp", "ghWp~"),
}


@pytest.mark.loop_fast
def test_colour_singlet_segments_take_the_tree_engines_reading_sign(loop_gmu_ufo):
    """The tree engine reads a vertex all-outgoing (outgoing tree particles on
    their own legs, outgoing momenta); the loop segments read boson legs by
    their incoming fields with incoming momenta. On every colour-singlet
    vertex of the G_mu UFO in 't Hooft-Feynman gauge the two readings differ
    by a sign, -1 exactly on the momentum-odd Goldstone vertices and the
    ghost-G0 vertices (``READING_FLIPS``); coloured vertices keep their
    reading (their colour reading compensates). The signs of the closed
    vertices are field signs: G0 negated, s(G+) s(G-) = 1, Z, photon, H and
    W kept (W G0 G+- and W G+- H, read through their conjugate rules, are no
    field signs), so the Goldstone Yukawas' lone chiral projectors (matched
    to the tree engine's trees, ``models.tensors``) and the loops share one
    reading."""

    from pyamplicol.loops.v2.catalogue import _CatalogueBuilder
    from pyamplicol.models.external import CompiledUFOModel
    from pyamplicol.models.loading import compile_model_source, with_model_gauge

    compiled = with_model_gauge(
        compile_model_source(loop_gmu_ufo, require_supported=False), "feynman"
    )
    builder = _CatalogueBuilder(compiled, CompiledUFOModel(compiled))
    records = {record.name: record for record in compiled.ir.particles}
    flips, singlets = set(), 0
    for term in compiled.ir.vertex_terms:
        if any(records[name].auxiliary_kind is not None for name in term.particles):
            continue
        sign = builder.reading_sign(term)
        assert sign in (1, -1)
        if any(abs(records[name].color) != 1 for name in term.particles):
            assert sign == 1, term.vertex
            continue
        singlets += 1
        if sign < 0:
            flips.add(tuple(sorted(term.particles)))
    assert flips == READING_FLIPS
    assert singlets > 3 * len(READING_FLIPS)
    signs = builder._signs()
    fields = (["G0"], ["G+", "G-"], ["H"], ["Z"], ["A"], ["W+", "W-"], ["G"])
    assert [signs.product(names) for names in fields] == [-1, 1, 1, 1, 1, 1, 1]


# A toy extension of the G_mu UFO (genericity audit 2, G2-M6): a colour-
# triplet scalar pair (sq, sd) coupled to the W, the photon and the gluon as
# the Goldstones and the Higgs are, and a dipole Lorentz structure
# sigma^{mu nu} k_nu on a quark-gluon and an electron-photon vertex. Each
# coloured vertex is a copy of a colour-singlet one: (toy vertex, its
# particles, the singlet copy's particles, Lorentz structure).
TOY_PARTICLES = """
SQ = Particle(pdg_code = 9000101, name = 'sq', antiname = 'sq~', spin = 1,
              color = 3, mass = Param.ZERO, width = Param.ZERO,
              texname = 'sq', antitexname = 'sq~', charge = 2/3,
              GhostNumber = 0, LeptonNumber = 0)
SQ__tilde__ = SQ.anti()
SD = Particle(pdg_code = 9000102, name = 'sd', antiname = 'sd~', spin = 1,
              color = 3, mass = Param.ZERO, width = Param.ZERO,
              texname = 'sd', antitexname = 'sd~', charge = -1/3,
              GhostNumber = 0, LeptonNumber = 0)
SD__tilde__ = SD.anti()
"""
TOY_LORENTZ = """
FFVdipole = Lorentz(name = 'FFVdipole', spins = [ 2, 2, 3 ],
                    structure = 'P(-1,3)*Gamma(3,2,-2)*Gamma(-1,-2,1) - '
                                'P(-1,3)*Gamma(-1,2,-2)*Gamma(3,-2,1)')
"""
TOY_VERTICES = [
    # (name, particles, colour, Lorentz, coupling, singlet copy's particles)
    ("V_toy_wp", "W__plus__, P.SQ__tilde__, P.SD", "Identity(2,3)", "VSS1",
     "GC_55", ("W+", "G-", "H")),
    ("V_toy_wm", "W__minus__, P.SQ, P.SD__tilde__", "Identity(2,3)", "VSS1",
     "GC_55", ("W-", "G+", "H")),
    ("V_toy_a", "A, P.SQ__tilde__, P.SQ", "Identity(2,3)", "VSS1", "GC_3",
     ("A", "G-", "G+")),
    ("V_toy_g", "G, P.SQ__tilde__, P.SQ", "T(1,3,2)", "VSS1", "GC_11",
     ("A", "G-", "G+")),
    ("V_toy_gdip", "u__tilde__, P.u, P.G", "T(3,2,1)", "FFVdipole", "GC_11",
     ("e+", "e-", "A")),
    ("V_toy_adip", "e__plus__, P.e__minus__, P.A", "1", "FFVdipole", "GC_3",
     ("e+", "e-", "A")),
]  # fmt: skip


def _toy_coloured_ufo(loop_gmu_ufo, directory: Path) -> Path:
    target = directory / "loop_qcd_qed_sm_Gmu_toy_colour"
    shutil.copytree(
        loop_gmu_ufo, target, ignore=shutil.ignore_patterns("__pycache__", "*.pkl")
    )
    vertices = "".join(
        f"\n{name} = Vertex(name = '{name}', particles = [ P.{particles} ],\n"
        f"    color = [ '{colour}' ], lorentz = [ L.{lorentz} ],\n"
        f"    couplings = {{(0,0):C.{coupling}}})\n"
        for name, particles, colour, lorentz, coupling, _ in TOY_VERTICES
    )
    for file, text in (
        ("particles.py", TOY_PARTICLES),
        ("lorentz.py", TOY_LORENTZ),
        ("vertices.py", vertices),
    ):
        with (target / file).open("a") as handle:
            handle.write(text)
    return target


@pytest.mark.loop_fast
def test_coloured_segments_take_the_sign_of_their_colour_singlet_copy(
    loop_gmu_ufo, tmp_path
):
    """One reading-sign rule for every vertex (genericity audit 2, G2-M6):
    the Lorentz ratio of the two readings times the colour tensor's sign
    under the transposition of its adjoint generators. A coloured vertex
    whose colour does not compensate its momenta takes the sign of its
    colour-singlet copy: W sq~ sd as W G- H (-1, where a rule keeping every
    coloured vertex at 1 gives 1), the gluon's and the photon's scalar
    vertices as A G- G+, the chromomagnetic dipole as the photon's (-1). A
    colour tensor without a transposition sign is refused, typed."""

    from dataclasses import replace

    from pyamplicol.loops.v2.catalogue import (
        LoopKernelCatalogueError,
        _CatalogueBuilder,
    )
    from pyamplicol.models.external import CompiledUFOModel
    from pyamplicol.models.loading import compile_model_source, with_model_gauge

    compiled = with_model_gauge(
        compile_model_source(
            _toy_coloured_ufo(loop_gmu_ufo, tmp_path), require_supported=False
        ),
        "feynman",
    )
    builder = _CatalogueBuilder(compiled, CompiledUFOModel(compiled))
    records = {record.name: record for record in compiled.ir.particles}
    terms = {
        (term.vertex, term.lorentz_name): term for term in compiled.ir.vertex_terms
    }
    singlets = {
        (tuple(sorted(term.particles)), term.lorentz_name): term
        for term in compiled.ir.vertex_terms
        if all(abs(records[name].color) == 1 for name in term.particles)
    }
    signs = {}
    for name, _, _, lorentz, _, copy in TOY_VERTICES:
        signs[name] = builder.reading_sign(terms[name, lorentz])
        singlet = singlets[tuple(sorted(copy)), lorentz]
        assert signs[name] == builder.reading_sign(singlet), name
    assert signs == {
        "V_toy_wp": -1,
        "V_toy_wm": -1,
        "V_toy_a": 1,
        "V_toy_g": 1,
        "V_toy_gdip": -1,
        "V_toy_adip": -1,
    }
    coloured = terms["V_toy_g", "VSS1"]
    for colour, match in (
        ("UFO::{}::Epsilon(1,2,3)", "Epsilon of 3 indices has no sign"),
        ("UFO::{}::f(1,2,3) + UFO::{}::d(1,2,3)", "change sign differently"),
    ):
        with pytest.raises(LoopKernelCatalogueError, match=match):
            builder.reading_sign(replace(coloured, id=-1, color_source=colour))


@pytest.mark.loop_fast
def test_fermion_ends_refuse_sextets_and_octets():
    """One fermion classifier (genericity audit 2, G2-m2) for the external
    ends, the root closure and the leaf weights: colour singlets are singlet
    ends, (anti)triplets keep their representation, sextet and octet
    fermions are refused, typed, at each of the three."""

    from pyamplicol.loops.v2.conventions import (
        SINGLET_END,
        LeafWeights,
        fermion_end,
        fermion_ends,
    )
    from pyamplicol.loops.v2.ct_sites import SiteClosure

    assert fermion_ends([(1, 1), (2, -3), (3, 3)]) == {1: SINGLET_END, 2: -3, 3: 3}
    assert [fermion_end(rep) for rep in (1, 3, -3)] == [SINGLET_END, 3, -3]
    for representation in (6, -6, 8):
        match = f"representation {representation} has no sign rule"
        with pytest.raises(FermionSignError, match=match):
            fermion_ends([(2, representation)])
        particle = {"name": "x", "antiname": "x~", "spin": 2, "mass": "ZERO"}
        model = {
            "particles": [
                {**particle, "color": representation},
                {"name": "g", "antiname": "g", "spin": 3, "color": 8, "mass": "ZERO"},
            ],
            "parameters": [{"name": "ZERO", "expression": "0"}],
            "couplings": [],
            "vertex_rules": [],
        }
        block = SimpleNamespace(mask=1, field="x")
        content = SimpleNamespace(
            loop_in="g", loop_out="g", terms=(SimpleNamespace(blocks=(block,)),)
        )
        with pytest.raises(FermionSignError, match=match):
            LeafWeights(model, {})._line_parity([content])
        closure = object.__new__(SiteClosure)
        closure._ends = {}
        closure._by_name = {"x": model["particles"][0]}
        with pytest.raises(FermionSignError, match=match):
            closure._root_sign([block])


# --------------------------- two singlet lines and a quark line (QCD, loop_sm)

# MadLoop's u u~ > e- e+ mu- mu+ [virt=QCD] (loop_sm-no_b_mass), whose fixed
# parameters every slice of this section takes.
FOUR_LEPTONS = "required/uux_emepmummp"
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
def four_leptons(loop_sm_compiled_model, loop_native, loop_reference):
    """(slice factory, MadLoop reference): two lepton lines and a quark line
    on loop_sm with QCD corrections, each process generated once."""

    from pyamplicol.loops.v2.assemble import LoopSlice, SliceRequest
    from pyamplicol.models.loading import load_compiled_model

    reference = loop_reference(FOUR_LEPTONS)
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
                    fixed_parameters=dict(reference["fixed_parameters"]),
                ),
                loop_native,
            )
        return built[process]

    return build, reference


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
    build, reference = four_leptons
    parameters = dict(reference["parameters"])
    first, second = build(process), build(relabelled)
    ratio, residual = _born_closure_ratio(
        first, FOUR_LEPTON_POINT, "1000000", parameters
    )
    assert abs(abs(ratio) - 1) <= TOLERANCE and abs(ratio.imag) <= TOLERANCE, ratio
    assert residual <= TOLERANCE, residual
    check_leaf_weights(first)
    born, virtual = _totals(
        first.evaluate(FOUR_LEPTON_POINT, "1000000", parameters=parameters)
    )
    other_born, other_virtual = _totals(
        second.evaluate(
            [FOUR_LEPTON_POINT[i] for i in legs], "1000000", parameters=parameters
        )
    )
    assert abs(born - other_born) <= TOLERANCE * abs(born), (born, other_born)
    scale = max(map(abs, virtual))
    for mine, theirs in zip(virtual, other_virtual, strict=True):
        assert abs(mine - theirs) <= TOLERANCE * scale, (virtual, other_virtual)


def test_two_lepton_lines_and_a_quark_line_equal_madloop(four_leptons):
    """u u~ > e- e+ mu- mu+ with QCD corrections equals MadLoop at the
    reference's three points and scales, Born and every Laurent coefficient
    to 1e-12 in the fixtures' metric (measured: 3.9e-14 for the finite part,
    MadLoop's own estimate 5e-14 to 9e-14). The lepton lines' Z/photon blocks
    interleave at the root and in every loop (B4)."""

    build, reference = four_leptons
    slice_ = build("u u~ > e- e+ m- m+")
    alpha_s = float(reference["parameters"]["aS"])
    for point, mu, row in zip(
        reference["points"], reference["mu_r"], reference["rows"], strict=True
    ):
        result = slice_.evaluate(
            point, str(Fraction(mu) ** 2), parameters=dict(reference["parameters"])
        )
        born, virtual = _totals(result)
        madloop = row["providers"]["madloop"]
        floor = abs(madloop["born"]) * alpha_s / (2 * math.pi)
        assert _metric(born, madloop["born"], 0.0) <= TOLERANCE
        for value, key in zip(virtual, LAURENT, strict=True):
            assert _metric(value, madloop[key], floor) <= TOLERANCE, (key, value)


# ------------------------------------- full electroweak corrections (nightly)

# The G_mu inputs of the MadLoop controls (their ``parameters``).
CONTROL_PARAMETERS = {
    "Gf": "0.0000116639",
    "MW": "80.419",
    "MZ": "91.188",
    "MH": "125",
    "MT": "173.3",
    "ymt": "173.3",
    "aS": "0.118",
}
# Electroweak requests fix every width to zero outside the complex-mass
# scheme (``ew_policy``, [U8]); the controls' massless b and tau.
ZERO_WIDTHS = {"WT": "0", "WW": "0", "WZ": "0", "WH": "0"}
G_MU_FIXED = {"MB": "0", "ymb": "0", "MTA": "0", "ymtau": "0", **ZERO_WIDTHS}
# alpha_Gmu at the controls' inputs (the loop coupling of the metric's floor).
ALPHA_GMU = 0.0075467722613403515
# The controls' phase-space points (``phase_space_fixture``; the massive
# control states its own ``exact_points``).
CONTROL_POINTS = ROOT / "tools/loop_reference/uudd_qcd.json"
# control -> (process, slice); the components MadLoop instruments: the full
# virtual, the bare loops, R2, the UV vertex counterterms, the field
# counterterms (no separate field amplitudes: zero) and all counterterms.
CONTROLS = {
    "electroweak-mumu-control": ("e- e+ > m- m+", "massless"),
    "electroweak-mumu-scale-control": ("e- e+ > m- m+", "massless"),
    "electroweak-eeuu-control": ("e- e+ > u u~", "massless"),
    "electroweak-massive-mumu-control": ("e- e+ > m- m+", "massive"),
}
COMPONENTS = {
    "full": ("virtual",),
    "bare": ("bare",),
    "r2": ("r2",),
    "uv_local": ("uv",),
    "uv": ("uv",),
    "fields": ("uv_external_fields",),
    "ct": ("r2", "uv", "uv_external_fields"),
}


def _massive_muon_ufo(loop_gmu_ufo, directory: Path, control) -> Path:
    """The G_mu UFO with the massive-muon control's restriction card (its
    exact text: MM = ymm = 0.105658 GeV, the rest the default restriction):
    the model MadLoop generated the control with, which keeps the massless
    leptons' counterterms."""

    import hashlib

    card = control["restriction_card_text"]
    assert (
        hashlib.sha256(card.encode()).hexdigest() == control["restriction_card_sha256"]
    )
    target = directory / "loop_qcd_qed_sm_Gmu_massive_muon"
    shutil.copytree(
        loop_gmu_ufo, target, ignore=shutil.ignore_patterns("__pycache__", "*.pkl")
    )
    (target / "restrict_default.dat").write_text(card)
    return target


@pytest.fixture(scope="module")
def gmu_slices(loop_gmu_ufo, loop_native, tmp_path_factory):
    """``slice(process, muons)``: full electroweak corrections (QED^4 loops
    over QED^2 Borns, 't Hooft-Feynman gauge) on the G_mu UFO, with
    massless or massive muons, each generated once (about 5 min)."""

    from pyamplicol.loops.v2.assemble import LoopSlice, SliceRequest
    from pyamplicol.models.loading import compile_model_source

    built = {}

    def build(process: str, muons: str):
        if (process, muons) not in built:
            if muons == "massive":
                control = json.loads(
                    (FIXTURES / "electroweak-massive-mumu-control.json").read_text()
                )
                ufo = _massive_muon_ufo(
                    loop_gmu_ufo, tmp_path_factory.mktemp("massive-muon"), control
                )
                fixed = ZERO_WIDTHS
            else:
                ufo, fixed = loop_gmu_ufo, G_MU_FIXED
            compiled = compile_model_source(ufo, require_supported=False)
            request = SliceRequest(
                process,
                {"QED": 2, "QCD": 0},
                {"QED": 4, "QCD": 0},
                corrections=("QED",),
                fixed_parameters=fixed,
                compatibility="gmu_particle_anti",
                scale_parameter="MU_R",
            )
            built[process, muons] = LoopSlice.generate(compiled, request, loop_native)
        return built[process, muons]

    return build


def _control_rows(control: dict):
    """(point, mu_R^2, row) of every row of a MadLoop control."""

    if control.get("exact_points"):
        points = control["exact_points"]
    else:
        points = json.loads(CONTROL_POINTS.read_text())["points"]
        points = [[[str(x) for x in leg] for leg in point] for point in points]
    scales = control.get("mu_r2") or [
        row.get("mu_r2", "1000000") for row in control["rows"]
    ]
    for index, row in enumerate(control["rows"]):
        yield points[index % len(points)], row.get("mu_r2") or scales[index], row


def _control_parameters(muons: str) -> dict[str, str]:
    if muons == "massive":
        return {**CONTROL_PARAMETERS, "MM": "0.105658", "ymm": "0.105658"}
    return dict(CONTROL_PARAMETERS)


@pytest.mark.loop_nightly
@pytest.mark.parametrize("name", list(CONTROLS))
def test_electroweak_controls_equal_madloop(gmu_slices, name):
    """The four MadLoop G_mu controls (full electroweak corrections, Feynman
    gauge, lhv = 1; two colour-singlet lines, the lepton lines through the
    boxes and vertex loops): Born and every Laurent coefficient of the full
    virtual and of each instrumented component (bare loops, R2, UV vertex
    counterterms, field counterterms, all counterterms) at 1e-12 in the
    fixtures' metric, at every row's point and scale. The massive control
    carries the muon's Yukawa couplings: G0, H and G+- loops on the muon
    line, the Z-G0-H triangles included."""

    control = json.loads((FIXTURES / f"{name}.json").read_text())
    process, muons = CONTROLS[name]
    slice_ = gmu_slices(process, muons)
    worst: dict[str, float] = {}
    for point, scale, row in _control_rows(control):
        result = slice_.evaluate(point, scale, parameters=_control_parameters(muons))
        born = sum(float(value) for value in result.born.values())
        theirs = row["born"]["1"]
        assert _metric(born, theirs, 0.0) <= TOLERANCE, (born, theirs)
        floor = abs(theirs) * ALPHA_GMU / (2 * math.pi)
        for key, parts in COMPONENTS.items():
            if key not in row:
                continue
            mine = [0.0, 0.0, 0.0]
            for part in parts:
                mine = [
                    a + b for a, b in zip(mine, _component(result, part), strict=True)
                ]
            for value, laurent in zip(mine, LAURENT, strict=True):
                error = _metric(value, row[key]["1"][laurent], floor)
                worst[key] = max(worst.get(key, 0.0), error)
                assert error <= TOLERANCE, (key, laurent, value, row[key]["1"])
    print(name, {key: f"{error:.1e}" for key, error in worst.items()})


@pytest.mark.loop_nightly
@pytest.mark.xfail(
    strict=True,
    reason=(
        "WP3.2 EWI-N: the native point inputs of electroweak requests are not "
        "exported yet"
    ),
)
def test_mumu_control_validates_with_native_inputs(gmu_slices):
    """e- e+ > mu- mu+ (full EW) through the native executor with native
    point inputs: a strict xfail until EWI-N exports them, so that a native
    gap shows here (WP3.2 M8), never as a silent fallback."""

    from pyamplicol.loops.v2.plan_export import NativeExecutor

    control = json.loads((FIXTURES / "electroweak-mumu-control.json").read_text())
    slice_ = gmu_slices("e- e+ > m- m+", "massless")
    executor = NativeExecutor(slice_, forward=False, inputs="native")
    point, scale, row = next(_control_rows(control))
    (result,) = executor.evaluate([point], [scale], parameters=CONTROL_PARAMETERS)
    mine = [float(x) for x in result.totals[:4]]
    theirs = [row["born"]["1"], *(row["full"]["1"][key] for key in LAURENT)]
    floor = abs(theirs[0]) * ALPHA_GMU / (2 * math.pi)
    for index, (value, want) in enumerate(zip(mine, theirs, strict=True)):
        assert _metric(value, want, 0.0 if index == 0 else floor) <= TOLERANCE


# -------------------------------------------------- O2 against the legacy engine

# Legacy (``loops.physical_source``) inputs of the G_mu controls: the
# restriction "full" with the default restriction's zeros fixed.
_LEGACY_ZEROS = (
    "aEW0M1",
    "MD",
    "MU",
    "MS",
    "MC",
    "MB",
    "Me",
    "MTA",
    "lamWS",
    "AWS",
    "rhoWS",
    "etaWS",
    "ymdo",
    "ymup",
    "yms",
    "ymc",
    "ymb",
    "yme",
    "ymtau",
)
_LEGACY_ONES = ("epsUV", "epsIR", "LogSwitch", "GfRedefinitionChoice")
# 60 digits (O2's arbitrary carrier, ``test_loop_v2_slice.O2_TOLERANCE``).
O2_DIGITS = 60
O2_TOLERANCE = 1e-50


def _legacy_families(loop_gmu_ufo, process: str, muons: str, point, scale) -> dict:
    """The legacy engine's per-family numerators of a 2 -> 2 lepton process
    (full EW, G_mu) at one point, in ``tools/loop_reference/legacy_capture``'s
    layout (60 digits). Its families are FeynKit's loop diagrams, each with
    its FeynKit overall factor (closed-loop and external-fermion signs)."""

    import numpy as np

    from pyamplicol.loops.covariant_model import build_covariant_structure
    from pyamplicol.loops.physical_source import source_bare_program
    from pyamplicol.models.loop_import import load_raw_loop_model
    from tools.loop_reference.legacy_capture import _family_layout

    zeros = _LEGACY_ZEROS + (("MM", "ymm") if muons == "massless" else ())
    source = build_covariant_structure(
        load_raw_loop_model(str(loop_gmu_ufo), restriction="full"),
        compatibility="gmu_particle_anti",
        fixed_parameters={
            **dict.fromkeys(zeros, "0"),
            **dict.fromkeys(_LEGACY_ONES, "1"),
        },
        covariant_multiplets={
            23: [23, 250, 9000002, -9000002],
            24: [24, 251, 9000003, -9000004],
            -24: [-24, -251, 9000004, -9000003],
        },
    )
    parameters = {**_control_parameters(muons), **ZERO_WIDTHS, "MU_R": "1000"}
    initial, final = (tuple(side.split()) for side in process.split(">"))
    program = source_bare_program(
        source,
        initial,
        final,
        selected_corrections=["QED"],
        born_order_bounds={"QCD": 0, "QED": 2},
        loop_order_bounds={"QCD": 0, "QED": 4},
        parameters=parameters,
        arithmetic="arbitrary",
        precision=O2_DIGITS,
        reduction_engine="generic",
        coefficient_engine="interpreted",
    )
    bare = program.bare
    currents = bare.prepare_currents_native(np.asarray([point], dtype=str), [scale])
    weights = getattr(bare, "dual_born_weights", None)
    dual = None if weights is None else weights.evaluate_batch(currents.born)
    rows = []
    for family in bare.families:
        values = family.projector.evaluate_batch(
            bare.family_projection_arguments(currents, family, weights=dual)
        ).to_decimal()
        width = len(family.powers)
        rows.append([values[i : i + width] for i in range(0, len(values), width)])
    return {"families": _family_layout(bare), "rows": rows}


def _assert_families_equal_legacy(slice_, legacy: dict, point, scale, parameters):
    """O2 per family: every legacy family's numerator equals the sum of the
    v2 families on its ring (v2 keeps the widths' labels apart: the W,
    Goldstone and ghost rings at M_W, equal at zero widths) after an exact
    affine frame map, symmetrised over the ring's automorphisms,
    coefficients weighted by sqrt(s)^rank (``test_loop_v2_slice``), at 60
    digits; the two family sets are equal (no scaleless family in these
    2 -> 2 processes). Returns the number of families compared."""

    from pyamplicol.loops.v2 import reference_executor as rx
    from pyamplicol.loops.v2.assemble import Precision
    from pyamplicol.loops.v2.poly_affine import size, substitute

    slice_o2 = _sibling("test_loop_v2_slice")
    execution, momenta = slice_.family_numerators(
        point,
        scale,
        parameters=parameters,
        precision=Precision.arbitrary(O2_DIGITS),
    )
    legs = len(point)
    signs = [1, 1] + [-1] * (legs - 2)  # given momenta -> all incoming
    rings = [
        (
            [
                slice_o2._normal([c * s for c, s in zip(v, signs, strict=True)])
                for v in family["shifts"]
            ],
            family["masses_squared"],
        )
        for family in legacy["families"]
    ]
    groups: dict[int, list] = {}
    for (key, _orders), numerator in execution.numerators.items():
        vectors, masses = slice_o2._ring(numerator.sequence, legs)
        found = [
            index
            for index, (shifts, masses_) in enumerate(rings)
            if slice_o2._maps(vectors, masses, shifts, masses_)
        ]
        assert len(found) == 1, key
        groups.setdefault(found[0], []).append((numerator, vectors, masses))
    assert sorted(groups) == list(range(len(rings)))  # no scaleless family here
    layout, average = slice_.layout, slice_.normalisation.average
    for index, members in groups.items():
        family, (row,) = legacy["families"][index], legacy["rows"][index]
        _first, vectors, masses = members[0]
        rank = max(
            [sum(p[:4]) for p in family["powers"]] + [n.rank for n, _v, _m in members]
        )
        dense = [rx.Complex(0, 0)] * size(rank)
        for powers, (re, im) in zip(family["powers"], row, strict=True):
            assert powers[4:] == [0, 0]
            dense[layout.index[tuple(powers[:4])]] = rx.Complex(
                Fraction(re), Fraction(im)
            )

        def frame(sign_shift):
            sign, shift = sign_shift
            return sign, [slice_o2._momentum(shift, momenta, mu) for mu in range(4)]

        sign, shift = frame(slice_o2._maps(vectors, masses, *rings[index])[0])
        theirs = substitute(dense, layout, sign=sign, shift=shift)
        mine = [0] * len(theirs)
        for numerator, other_vectors, other_masses in members:
            (column,) = numerator.columns
            part = [rx.exact(x) * Fraction(1, average) for x in column]
            part += [0] * (len(theirs) - len(part))
            if (other_vectors, other_masses) != (vectors, masses):
                # Into the first member's frame.
                sign, shift = frame(
                    slice_o2._maps(vectors, masses, other_vectors, other_masses)[0]
                )
                part = substitute(part, layout, sign=sign, shift=shift)
            mine = [a + b for a, b in zip(mine, part, strict=True)]
        autos = slice_o2._maps(vectors, masses, vectors, masses)
        if len(autos) > 1:
            mine = slice_o2._symmetrised(mine, layout, momenta, autos)
            theirs = slice_o2._symmetrised(theirs, layout, momenta, autos)
        weights = slice_o2._weights(layout, len(theirs), momenta)
        largest = max(
            slice_o2._magnitude(b) * w for b, w in zip(theirs, weights, strict=True)
        )
        error = max(
            slice_o2._magnitude(a - b) * w
            for a, b, w in zip(mine, theirs, weights, strict=True)
        )
        assert error <= O2_TOLERANCE * largest, (masses, error / largest)
    return len(rings)


@pytest.mark.loop_nightly
@pytest.mark.parametrize("muons", ["massless", "massive"])
def test_family_numerators_equal_the_legacy_engine(gmu_slices, loop_gmu_ufo, muons):
    """O2 per family (DESIGN risk 2) on e- e+ > mu- mu+ with full EW
    corrections (``_assert_families_equal_legacy``). This localises a sign
    of a Goldstone, ghost or lepton line to a family: the families of the
    lepton lines through the loop (Koszul signs) and, with massive muons, of
    the Yukawa couplings (G0, H and G+- on the muon line, the Z G0 H
    triangles)."""

    point = [["500", "0", "0", "500"], ["500", "0", "0", "-500"]]
    point += [["500", "300", "0", "400"], ["500", "-300", "0", "-400"]]
    if muons == "massive":
        control = json.loads(
            (FIXTURES / "electroweak-massive-mumu-control.json").read_text()
        )
        point = control["exact_points"][0]
    scale = "1000000"
    process = "e- e+ > m- m+"
    legacy = _legacy_families(loop_gmu_ufo, process, muons, point, scale)
    _assert_families_equal_legacy(
        gmu_slices(process, muons), legacy, point, scale, _control_parameters(muons)
    )


@pytest.mark.loop_nightly
def test_bhabha_families_carry_feynkit_overall_factors(gmu_slices, loop_gmu_ufo):
    """O13 EW (DESIGN §14, WP3.2 EWI-5): e- e+ > e- e+ with full EW
    corrections, per family against the legacy engine, whose families are
    FeynKit's loop diagrams with FeynKit's overall factors (the closed-loop
    sign and the relative sign of the two ways the identical external
    leptons pair into lines). Bhabha is the process where that pairing
    matters: the s- and t-channel lines run through different boxes and
    vertex loops, so a Koszul sign of a lepton line through the loop that
    were wrong on one pairing flips whole families, which the summed
    virtual against MadLoop (``test_bhabha_equals_madloop``) would only show
    as a total."""

    point = [["500", "0", "0", "500"], ["500", "0", "0", "-500"]]
    point += [["500", "300", "0", "400"], ["500", "-300", "0", "-400"]]
    scale = "1000000"
    process = "e- e+ > e- e+"
    legacy = _legacy_families(loop_gmu_ufo, process, "massless", point, scale)
    compared = _assert_families_equal_legacy(
        gmu_slices(process, "massless"),
        legacy,
        point,
        scale,
        _control_parameters("massless"),
    )
    print("families", compared)


# ------------------------------------- g g > t t~ e- e+ (EW, three providers)

TOP_PAIR_LEPTONS = "required/gg_ttxemep_ew"


@pytest.mark.loop_nightly
def test_top_pair_and_leptons_equal_three_providers(
    loop_gmu_ufo, loop_native, loop_reference
):
    """g g > t t~ e- e+ with QCD and QED corrections on the G_mu UFO equals
    MadLoop, Recola and OpenLoops: Born and every Laurent coefficient of the
    virtual at the reference's three points, against MadLoop to 1e-11 and the
    other two to the fixture's cross-provider tolerance, in the fixtures'
    metric. The Goldstone sector with top Yukawas: Z* -> G0 H triangles and W
    G+- H loops on the top line, which the colour-singlet segments' reading
    sign fixes (O(1) in the finite part without it). Evaluated by the native
    executor in double-double: binary64 loses 2e-10 of the finite part to
    cancellations at point 0. Generation about 10 min, 8 GB."""

    from pyamplicol.loops.v2.assemble import LoopSlice, Precision, SliceRequest
    from pyamplicol.loops.v2.plan_export import NativeExecutor, export_plan
    from pyamplicol.models.loading import compile_model_source

    reference = loop_reference(TOP_PAIR_LEPTONS)
    compiled = compile_model_source(loop_gmu_ufo, require_supported=False)
    externals = {record.name for record in compiled.ir.parameters}
    fixed = {k: v for k, v in reference["fixed_parameters"].items() if k in externals}
    request = SliceRequest(
        "g g > t t~ e- e+",
        dict(reference["born_orders"]),
        dict(reference["loop_orders"]),
        corrections=("QCD", "QED"),
        fixed_parameters=fixed,
        compatibility="gmu_particle_anti",
        scale_parameter="MU_R",
    )
    slice_ = LoopSlice.generate(compiled, request, loop_native)
    executor = NativeExecutor(
        slice_, forward=False, exported=export_plan(slice_, forward=False)
    )
    alpha = float(reference["derived_parameters"]["alpha"])
    tolerances = {"madloop": 1e-11, "recola": reference["tolerance"]}
    tolerances["openloops"] = reference["tolerance"]
    worst = dict.fromkeys(tolerances, 0.0)
    for point, mu, row in zip(
        reference["points"], reference["mu_r"], reference["rows"], strict=True
    ):
        (result,) = executor.evaluate(
            [point],
            [str(Fraction(mu) ** 2)],
            parameters=dict(reference["parameters"]),
            precision=Precision.double_double(),
        )
        born, *virtual = (float(x) for x in result.totals[:4])
        for provider, tolerance in tolerances.items():
            values = row["providers"][provider]
            floor = abs(values["born"]) * alpha / (2 * math.pi)
            assert _metric(born, values["born"], 0.0) <= tolerance, provider
            for value, key in zip(virtual, LAURENT, strict=True):
                error = _metric(value, values[key], floor)
                worst[provider] = max(worst[provider], error)
                assert error <= tolerance, (provider, key, value, values[key])
    print({provider: f"{error:.1e}" for provider, error in worst.items()})


# ------------------------------------------------ Bhabha (loop_qcd_qed_sm)

BHABHA = "required/emep_emep_ew"


@pytest.mark.loop_nightly
def test_bhabha_equals_madloop(loop_qcd_qed_sm_ufo, loop_native, loop_reference):
    """e- e+ > e- e+ with full electroweak corrections on loop_qcd_qed_sm
    (alpha(M_Z) scheme, Feynman gauge) equals MadLoop at the reference's three
    points and scales, Born and every Laurent coefficient to 1e-12 in the
    fixtures' metric (measured: 5.8e-15). Two identical lepton lines in the
    s- and t-channel Borns, both through the boxes: their relative sign is
    the Koszul sign. Generation about 2 min."""

    from pyamplicol.loops.v2.assemble import LoopSlice, SliceRequest
    from pyamplicol.models.loading import compile_model_source

    reference = loop_reference(BHABHA)
    compiled = compile_model_source(loop_qcd_qed_sm_ufo, require_supported=False)
    request = SliceRequest(
        "e- e+ > e- e+",
        {"QCD": 0, "QED": 2},
        {"QCD": 0, "QED": 4},
        corrections=("QED",),
        fixed_parameters=dict(reference["fixed_parameters"]),
        compatibility="gmu_particle_anti",
        scale_parameter="MU_R",
    )
    slice_ = LoopSlice.generate(compiled, request, loop_native)
    alpha = float(reference["derived_parameters"]["alpha"])
    for point, mu, row in zip(
        reference["points"], reference["mu_r"], reference["rows"], strict=True
    ):
        result = slice_.evaluate(
            point, str(Fraction(mu) ** 2), parameters=dict(reference["parameters"])
        )
        born, virtual = _totals(result)
        madloop = row["providers"]["madloop"]
        floor = abs(madloop["born"]) * alpha / (2 * math.pi)
        assert _metric(born, madloop["born"], 0.0) <= TOLERANCE
        for value, key in zip(virtual, LAURENT, strict=True):
            assert _metric(value, madloop[key], floor) <= TOLERANCE, (key, value)


# ------------------------------------- e- e+ > mu- mu+ on loop_qcd_qed_sm

# Photon-only QED corrections: Z, W, H and the Goldstones vetoed.
PHOTON_ONLY = (23, 24, 25, 250, 251)


def _qed_catani(loop_qcd_qed_sm_ufo, loop_native, veto) -> None:
    """e- e+ > mu- mu+ with QED corrections on loop_qcd_qed_sm (alpha(M_Z)
    scheme, widths zero), ``veto`` the vetoed particles: the renormalised
    virtual's single and double poles equal the QED Catani operator's
    prediction (EWI-6) per sector to 1e-12. This is the UV check too: the
    external-field factors' UV part is the remainder that makes the
    renormalised virtual UV finite by construction (``poles``), so only the
    total poles test the bare loops and the counterterms together."""

    from pyamplicol.loops.v2.assemble import LoopSlice, SliceRequest
    from pyamplicol.models.loading import compile_model_source

    compiled = compile_model_source(loop_qcd_qed_sm_ufo, require_supported=False)
    request = SliceRequest(
        "e- e+ > m- m+",
        {"QED": 2, "QCD": 0},
        {"QED": 4, "QCD": 0},
        corrections=("QED",),
        particle_veto=veto,
        fixed_parameters=ZERO_WIDTHS,
        compatibility="gmu_particle_anti",
        scale_parameter="MU_R",
    )
    slice_ = LoopSlice.generate(compiled, request, loop_native)
    point = [["500", "0", "0", "500"], ["500", "0", "0", "-500"]]
    point += [["500", "240", "320", "300"], ["500", "-240", "-320", "-300"]]
    result = slice_.evaluate(point, "1000000", parameters={"aS": "0.118", "MT": "173"})
    assert result.ir_poles is not None
    for sector, virtual in result.components["virtual"].items():
        single, double = (complex(x) for x in result.ir_poles[sector])
        scale = max(abs(single), abs(double))
        assert scale > 0
        assert abs(complex(virtual.double_pole) - double) <= TOLERANCE * scale
        assert abs(complex(virtual.single_pole) - single) <= TOLERANCE * scale


@pytest.mark.loop_nightly
def test_qed_poles_of_mumu_on_loop_qcd_qed_sm_are_catani(
    loop_qcd_qed_sm_ufo, loop_native
):
    """Full electroweak corrections (measured: single pole 4.5e-15 of the
    operator's). Generation about 1 min."""

    _qed_catani(loop_qcd_qed_sm_ufo, loop_native, ())


@pytest.mark.loop_nightly
@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason=(
        "photon-only veto on loop_qcd_qed_sm: the single pole is -3.487e-4 "
        "against Catani's -1.841e-4 (double pole equal); the veto drops "
        "loop_qcd_qed_sm's counterterm couplings that mix photon and Z parts "
        "whole, which the field factors' remainder absorbs in the UV only "
        "([U2] veto guard, WP3.2 EWI-7)"
    ),
)
def test_qed_poles_of_photon_only_mumu_on_loop_qcd_qed_sm_are_catani(
    loop_qcd_qed_sm_ufo, loop_native
):
    """Photon-only QED corrections (Z, W, H and the Goldstones vetoed): a
    lepton line through every box and vertex loop. Generation about 1 min."""

    _qed_catani(loop_qcd_qed_sm_ufo, loop_native, PHOTON_ONLY)


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
        structure, prepared = legacy.source_model.__wrapped__(loop_gmu_ufo, loop_native)
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
