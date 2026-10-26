# SPDX-License-Identifier: 0BSD
"""Native exact colour cells against their Python oracles (WP3.1 package C).

``loops/v2/native_cells`` (rusticol-core ``loop_v2::generate::cells`` and
``loop_v2::dipole_orbits``) must equal the Python stages exactly. Phase 1
switched them on inside the Python stages (``native_cells.NATIVE_STAGES``),
so every control is generated in two lanes: the production lane (every
stage native) and the oracle lane (``native_cells.native_stages(())``):

* C1: ``colour.seed_cells`` cell by cell (adjoint rings, fundamental traces,
  quark rings and generic classes; the slice's, the companion's and the
  prune's tables), ``contract_colour``, ``face_count`` and
  ``BornBasis.gram``; outputs independent of the thread count;
* C2: ``plan_export._Exporter.seed_records``, byte-identical plan JSON and
  the same Born components;
* C3: ``LoopSlice.degrees``, ``_AdjointOrbits`` and the liveness of
  ``_prune_colour``; S1, S4, S5 and S8 equal between the lanes;
* C4: ``LoopSlice.dipoles`` from the orbit representatives built in
  rusticol-core, which needs no Symbolica licence (``dipoles``);
* C5: the directed tables (``DirectedColour._table``) and the complete-family
  degrees (``ObservableSlice._family_degrees``), with the pruned slices;
* rule 9: generation decodes each table's cells once, the degrees none.

Default tier (``loop_fast``, about 27 s at host load 80, of which the two
generations and the session's compiled model take about 21 s): every stage
against its Python oracle on the production lane's slices of g g > g g (FC,
recursion counterterms; adjoint rings and fundamental traces) and
u u~ > g g (LC, site closure; quark rings, generic classes and the prune of
the slice and its companion), the directed tables and the dipole family of
g g > g g, native operator images on random tensors. ``loop_physics``
generates every control in both lanes (the default ones, u u~ > g g g at LC,
g g > g g g and u u~ > g g g for the dipoles as WP3.1 I5c asks, LC and NLC
controls; end to end and stage by stage), the
directed tables and families of WP2.5's controls and contractions beyond the
subset-sum label limit; six legs are ``loop_nightly``.
"""

from __future__ import annotations

import json
import os
import random
import re
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

import pytest

from pyamplicol.color.connections import (
    ColorLeg,
    ColorTensor,
    ConnectedColorTensor,
    EmitGluon,
    ExactColorCoefficient,
    OpenColorString,
    TensorTerm,
)
from pyamplicol.color.correlator_matrices import ColorCorrelator
from pyamplicol.loops.v2 import colour, dipoles, native_cells, snapshot
from tools.loop_reference import legacy_capture

_ROOT = Path(__file__).resolve().parents[2]
_SOURCE = _ROOT / "src"
# The package's Rust sources (the no-licence rule covers them).
_RUST = (
    "rust/crates/rusticol-core/src/loop_v2/generate/cells.rs",
    "rust/crates/rusticol-core/src/loop_v2/dipole_orbits.rs",
    "rust/loop-reduction/python/src/v2_colour.rs",
)

# (process, Born QCD order, accuracy[, counterterm form]). The default tier
# covers both forms of the counterterm keys of the degrees: the recursion's
# closure terms (g g > g g) and site closure (u u~ > g g, the cheaper form),
# every class kind (adjoint rings and fundamental traces; quark rings and
# generic classes) and the LC prune of a slice and its companion.
DEFAULT = {
    "gg_gg": ("g g > g g", 2, "full"),
    "uux_gg_lc": ("u u~ > g g", 2, "lc", "sites"),
}
PHYSICS = {
    "uux_ggg_lc": ("u u~ > g g g", 3, "lc", "sites"),
    "gg_ggg": ("g g > g g g", 3, "full"),
    "uux_ggg": ("u u~ > g g g", 3, "full"),
    "gg_gg_lc": ("g g > g g", 2, "lc"),
    "gg_gg_nlc": ("g g > g g", 2, "nlc"),
    "uux_ddxg": ("u u~ > d d~ g", 3, "full"),
    "uux_ddxg_nlc": ("u u~ > d d~ g", 3, "nlc"),
    "uux_uuxg": ("u u~ > u u~ g", 3, "full"),
    "uux_ggg_nlc": ("u u~ > g g g", 3, "nlc"),
}
# Six legs with site closure (rule 6: the recursion's CT program costs
# minutes there and enters only the counterterm keys, covered above).
NIGHTLY = {
    "uux_gggg": ("u u~ > g g g g", 4, "full", "sites"),
    "gg_gggg_lc": ("g g > g g g g", 4, "lc", "sites"),
}
CONTROLS = {**DEFAULT, **PHYSICS, **NIGHTLY}
LANES = {"native": native_cells.STAGES, "python": frozenset()}


@dataclass
class Generated:
    """One control generated in one lane, with the ``keep`` closures that
    ``_prune_colour`` handed to the native schedules (the slice's, then the
    companion's) and the cells the generation decoded (rule 9)."""

    slice: Any
    kept: list
    decoded: int


def python(function, *args, **kwargs):
    """``function`` in the oracle lane (every stage of package C Python)."""

    with native_cells.native_stages(()):
        return function(*args, **kwargs)


def uncached(slice_, key: str, method):
    """A cached ``LoopSlice`` method (``degrees``, ``dipoles``) evaluated
    again in the oracle lane; the slice's cached value is restored."""

    saved = slice_._cache.pop(key, None)
    try:
        return python(method)
    finally:
        slice_._cache.pop(key, None)
        if saved is not None:
            slice_._cache[key] = saved


def _request(name: str):
    from pyamplicol.loops.v2.assemble import SliceRequest
    from pyamplicol.models.loop_color_declaration import LoopColorDeclaration

    process, born, accuracy, *form = CONTROLS[name]
    options: dict[str, Any] = {"counterterms": form[0]} if form else {}
    if accuracy != "full":
        options.update(
            accuracy=accuracy,
            colour_declaration=LoopColorDeclaration.from_dict(
                json.loads(legacy_capture.COLOUR_DECLARATION.read_text())
            ),
        )
    return SliceRequest(
        process,
        {"QCD": born, "QED": 0},
        {"QCD": born + 2, "QED": 0},
        fixed_parameters=legacy_capture.FIXED_PARAMETERS,
        **options,
    )


@pytest.fixture(scope="module")
def compiled(loop_sm_compiled_model):
    from pyamplicol.models.loading import load_compiled_model

    return load_compiled_model(loop_sm_compiled_model)


@pytest.fixture(scope="module")
def slices(compiled, loop_native):
    """Generated slices per (control, lane), cached."""

    from pyamplicol.loops.v2 import generator_input
    from pyamplicol.loops.v2.assemble import LoopSlice

    built: dict[tuple[str, str], Generated] = {}

    def get(name: str, lane: str = "native") -> Generated:
        if (name, lane) not in built:
            kept: list = []
            original = generator_input.NativeGeneration.schedules

            def spy(self, colour_model, **kwargs):
                if kwargs.get("keep") is not None:
                    kept.append(kwargs["keep"])
                return original(self, colour_model, **kwargs)

            generator_input.NativeGeneration.schedules = spy
            try:
                with native_cells.native_stages(LANES[lane]):
                    before = native_cells.decoded_cells()
                    slice_ = LoopSlice.generate(compiled, _request(name), loop_native)
                    decoded = native_cells.decoded_cells() - before
            finally:
                generator_input.NativeGeneration.schedules = original
            built[name, lane] = Generated(slice_, kept, decoded)
        return built[name, lane]

    return get


# --------------------------------------------------------------- C1 cells


def _random_tensor(rng: random.Random, legs, labels, quarks) -> ConnectedColorTensor:
    """A random literal tensor on ``legs``: open strings pairing each quark
    with an antiquark, the adjoint labels spread over strings and traces."""

    terms = []
    for _ in range(rng.randint(1, 3)):
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
        coefficient = ExactColorCoefficient(
            Fraction(rng.randint(-3, 3), rng.randint(1, 4)),
            Fraction(rng.randint(-2, 2), rng.randint(1, 3)),
        )
        terms.append(
            TensorTerm(
                coefficient,
                ColorTensor(tuple(strings), tuple(traces)),
                rng.randint(-1, 1),
            )
        )
    return ConnectedColorTensor(legs, tuple(terms), 2 * rng.randint(0, 1))


@pytest.mark.loop_fast
def test_contractions_and_faces_equal_python():
    rng = random.Random(20261023)
    for quarks, gluons in (((), 5), (((1, 2),), 3), (((1, 2), (3, 4)), 2)):
        legs = [ColorLeg(q, 3) for q, _ in quarks] + [
            ColorLeg(b, -3) for _, b in quarks
        ]
        labels = list(range(10, 10 + gluons))
        legs += [ColorLeg(a, 8) for a in labels]
        pairs = [
            (
                _random_tensor(rng, legs, labels, quarks),
                _random_tensor(rng, legs, labels, quarks),
            )
            for _ in range(40)
        ]
        assert native_cells.contract_many(pairs) == [
            colour.contract_colour(a, b) for a, b in pairs
        ]
        for a, b in pairs[:10]:
            for left in a.terms:
                for right in b.terms:
                    assert native_cells.face_count(
                        left.tensor, right.tensor
                    ) == colour.face_count(left.tensor, right.tensor)


_QUARK_LEGS = (ColorLeg(1, 3), ColorLeg(2, -3), ColorLeg(3, 8), ColorLeg(4, 8))


def _quark_basis(*structures) -> colour.BornBasis:
    tensors = tuple(
        ConnectedColorTensor(
            _QUARK_LEGS,
            (TensorTerm(ExactColorCoefficient(1), ColorTensor(strings, traces)),),
        )
        for strings, traces in structures
    )
    return colour.BornBasis(
        _QUARK_LEGS, tuple(range(len(tensors))), tensors, (frozenset(),) * len(tensors)
    )


# A quark string apart from the ring's labels, and one holding them.
_SPLIT = ((OpenColorString(1, (), 2),), ((3, 4),))
_JOINED = ((OpenColorString(1, (3, 4), 2),), ())


@pytest.mark.loop_physics
def test_contractions_beyond_the_subset_sum_limit_equal_python():
    """Above ``cycles.rs``'s label limit (24) the native overlaps take the
    exact Fierz recursion; Python's ``contract_colour`` falls back to
    ``contraction_trace`` (about 10 s per single-term pair at 25 labels, so
    one pair with and one without a quark string)."""

    rng = random.Random(20261024)
    for quarks in ((), ((1, 2),)):
        legs = [ColorLeg(q, 3) for q, _ in quarks] + [
            ColorLeg(b, -3) for _, b in quarks
        ]
        labels = list(range(10, 35))
        legs += [ColorLeg(a, 8) for a in labels]
        pair = []
        while len(pair) < 2:
            tensor = _random_tensor(rng, legs, labels, quarks)
            if len(tensor.terms) == 1:
                pair.append(tensor)
        assert native_cells.contract_colour(*pair) == colour.contract_colour(*pair)


# Native refusal of each primitive kind against a bra with an open quark
# string, and Python's refusal of the joined bra (the split one: below).
_REFUSALS = {
    "adjoint-ring": ("adjoint-ring class against a Born", "not contracted exactly"),
    "fundamental-trace": ("bra string is not glued", "do not match the typed"),
}


@pytest.mark.loop_fast
@pytest.mark.parametrize("kind", sorted(_REFUSALS))
def test_primitives_against_quark_bras_are_refused(kind):
    """Primitives against bras the relabelling cache cannot key (a quark
    leg) take the direct path of ``_PrimitiveCells._cell``. A primitive
    carries no fundamental index, so its overlap with a bra holding an open
    quark string is not defined: refused natively (rule 2), as Python
    refuses a fundamental trace there; Python refuses an adjoint ring only
    where the string holds its labels (oracle defect 3, below)."""

    primitive = colour.Primitive(kind, (3, 4))
    refused, oracle_refused = _REFUSALS[kind]
    for structure in (_SPLIT, _JOINED):
        basis = _quark_basis(structure)
        with pytest.raises(ValueError, match=refused):
            native_cells.seed_cells(basis, [primitive])
        with pytest.raises(ValueError, match=refused):
            native_cells.NativeCells(basis)(primitive)
        # The switched stage fails closed (rule 2: no Python fallback).
        with pytest.raises(ValueError, match=refused):
            colour.seed_cells(basis, [primitive])
    structures = (_JOINED,) if kind == "adjoint-ring" else (_SPLIT, _JOINED)
    for structure in structures:
        with pytest.raises(ValueError, match=oracle_refused):
            python(colour.seed_cells, _quark_basis(structure), [primitive])


@pytest.mark.loop_fast
@pytest.mark.xfail(
    strict=True,
    raises=pytest.fail.Exception,
    reason="oracle defect 3 (read-only, rule 3): _PrimitiveCells._overlap "
    "drops the bra's quark index loop and returns a value for an adjoint "
    "ring against a bra with an open string",
)
def test_python_adjoint_ring_against_a_quark_bra_is_refused():
    ring = colour.Primitive("adjoint-ring", (3, 4))
    with pytest.raises(ValueError):
        python(colour.seed_cells, _quark_basis(_SPLIT), [ring])


@pytest.mark.loop_fast
def test_stored_dipole_orbits_are_validated(slices):
    slice_ = slices("gg_gg").slice
    stored = json.loads(native_cells.dipole_orbits(slice_).to_json())
    reloaded = type(native_cells.dipole_orbits(slice_)).from_json
    duplicated = dict(stored, representatives=stored["representatives"] * 2)
    with pytest.raises(ValueError, match="overlapping"):
        reloaded(json.dumps(duplicated)).expand()
    missing = dict(stored, representatives=stored["representatives"][1:])
    with pytest.raises(ValueError, match="do not cover"):
        reloaded(json.dumps(missing)).check_conservation()
    generator = dict(stored["generators"][0], legs=[0] * len(stored["coloured"]))
    broken = dict(stored, generators=[generator])
    with pytest.raises(ValueError, match="not a permutation"):
        reloaded(json.dumps(broken))
    # Crafted members are refused at load with a typed error, before any
    # table of their index space is allocated (an abort before): a basis
    # the stored data cannot cover, an index space past the address space,
    # coloured legs with a repeated label or out of leg order.
    for basis, message in ((2**20, "do not cover"), (2**32, "overflow")):
        crafted = dict(stored, basis=basis, generators=[], representatives=[], gram=[])
        with pytest.raises(ValueError, match=message):
            dipoles.load(json.dumps(crafted)).check_conservation()
    coloured = stored["coloured"]
    for legs in (
        [coloured[0], *coloured[:-1]],
        [*coloured[1:], coloured[0]],
        [[coloured[0][0], 6], *coloured[1:]],
    ):
        with pytest.raises(ValueError, match=r"leg order|representation 6"):
            dipoles.operators(dipoles.load(json.dumps(dict(stored, coloured=legs))))


# ------------------------------------------------- switched generation


def _switched_equal(native: Generated, oracle: Generated) -> None:
    """The production lane equals the oracle lane: S1, S4, S5 and S8 (the
    plan bytes), the companion's S5, the degrees and (FC) the dipoles; rule
    9 on both lanes."""

    a, b = native.slice, oracle.slice
    assert snapshot.stage_digests(a) == python(snapshot.stage_digests, b)
    assert a.table.cells == b.table.cells
    assert (a.scaleless is None) == (b.scaleless is None)
    if a.scaleless is not None:
        assert snapshot.digest(snapshot.seeds_records(a.scaleless)) == (
            snapshot.digest(snapshot.seeds_records(b.scaleless))
        )
    assert a.degrees() == python(b.degrees)
    _decodes(native)
    assert oracle.decoded == 0
    if a.request.accuracy == "full":
        assert a.dipoles() == python(b.dipoles)


def _decodes(native: Generated) -> None:
    """Rule 9: each table built during generation decoded its own cells
    once (an extension decodes only its new classes); the degrees and the
    plan export decode none."""

    from pyamplicol.loops.v2.plan_export import export_plan

    tables = [native.slice]
    if native.slice.scaleless is not None:
        tables.append(native.slice.scaleless)
    assert native.decoded == sum(len(x.table.classes) * len(x.basis) for x in tables)
    before = native_cells.decoded_cells()
    native_cells.degrees(native.slice)
    export_plan(native.slice, forward=False)
    assert native_cells.decoded_cells() == before


@pytest.mark.loop_fast
@pytest.mark.parametrize("name", sorted(DEFAULT))
def test_generation_decodes_each_table_once(slices, name):
    _decodes(slices(name))


def _seed_cells_equal(native: Generated, oracle: Generated | None = None) -> None:
    table = native.slice.table
    basis = table.basis
    kinds = {
        seed.kind if isinstance(seed, colour.Primitive) else "generic"
        for seed in table.classes
    }
    assert kinds, "no seed class"
    expected = python(colour.seed_cells, basis, table.classes)
    assert table.cells == expected
    if oracle is not None:
        assert oracle.slice.table.cells == expected
    assert native_cells.seed_cells(basis, table.classes) == expected
    assert colour.seed_cells(basis, table.classes) == expected
    twin = native.slice.scaleless
    if twin is not None:
        assert twin.table.cells == python(
            colour.seed_cells, twin.basis, twin.table.classes
        )
    # The Born metric.
    assert native_cells.gram(basis) == basis.gram


@pytest.mark.loop_fast
@pytest.mark.parametrize("name", sorted(DEFAULT))
def test_seed_cells_equal_python(slices, name):
    native = slices(name)
    _seed_cells_equal(native)
    # Rule 5: one thread and several give the same cells (the companion's
    # generic classes share cells per worker).
    tables = [native.slice.table]
    if native.slice.scaleless is not None:
        tables.append(native.slice.scaleless.table)
    for table in tables:
        one = native_cells.seed_cells(table.basis, table.classes, threads=1)
        assert one == native_cells.seed_cells(table.basis, table.classes, threads=4)


@pytest.mark.loop_fast
def test_rebased_cells_share_caches_and_equal_python(slices):
    """A rebased engine (another basis on the same legs, e.g. operator
    images) equals ``seed_cells`` against that basis and shares the caches
    both ways, as ``_PrimitiveCells.rebased``; so does the native twin of a
    rebased ``_PrimitiveCells`` (the switched path of the directed tables)."""

    table = slices("uux_gg_lc").slice.table
    basis = table.basis
    other = colour.BornBasis(
        basis.legs,
        basis.sector_ids[::-1],
        basis.tensors[::-1],
        basis.orders[::-1],
    )
    expected = python(colour.seed_cells, other, table.classes)
    engine = native_cells.NativeCells(basis)
    rebased = engine.rebased(other)
    assert native_cells.seed_cells(other, table.classes, rebased) == expected
    # The parent sees the rebased engine's relabelled bras, rings and ring
    # overlaps; its own table (the same tensors reordered) is all cache hits.
    shared = rebased.cache_sizes
    assert shared[0] and shared[1] and shared[3], shared
    assert engine.cache_sizes == shared
    assert native_cells.seed_cells(basis, table.classes, engine) == table.cells
    assert engine.cache_sizes == rebased.cache_sizes == shared
    with pytest.raises(ValueError, match="another basis"):
        native_cells.seed_cells(basis, table.classes, rebased)
    # The switched seed_cells through _PrimitiveCells and its rebased copy.
    cells = colour._PrimitiveCells(basis)
    copy = cells.rebased(other)
    assert colour.seed_cells(other, table.classes, copy) == expected
    assert cells.native().cache_sizes == copy.native().cache_sizes
    assert colour.seed_cells(basis, table.classes, cells) == table.cells
    with pytest.raises(ValueError, match="another basis"):
        colour.seed_cells(basis, table.classes, copy)


# ---------------------------------------------------------- C2 projection


def _records_equal(slice_, monkeypatch, columns=None) -> None:
    from pyamplicol.loops.v2 import plan_export

    captured: dict = {}
    original = plan_export._Exporter.seed_records

    def spy(self, columns):
        seeds = list(self.seeds)
        before = dict(self.born_components)
        result = original(self, columns)
        captured.update(
            seeds=seeds,
            before=before,
            after=list(self.born_components),
            columns=tuple(columns),
            result=result,
        )
        return result

    monkeypatch.setattr(plan_export._Exporter, "seed_records", spy)
    plan_export.export_plan(slice_, columns=columns)
    monkeypatch.undo()
    # Columns that differ only by their helicity rows share one weight
    # column (the third value, the plan's ``column_weights``; None when every
    # column has its own): the class records hold one weight list per weight
    # column, the projection of each weight column's first column.
    python_classes, python_records, shared = captured["result"]
    if shared is not None:
        keys, of_column = plan_export.weight_columns(captured["columns"])
        first: dict[int, int] = {}
        for k, w in enumerate(of_column):
            first.setdefault(w, k)
        captured["columns"] = tuple(
            captured["columns"][first[w]] for w in range(len(keys))
        )
    components = dict(captured["before"])
    class_json, records_json = native_cells.project_records(
        slice_, captured["columns"], captured["seeds"], components
    )
    assert class_json == json.dumps(python_classes, separators=(",", ":")).encode()
    assert records_json == json.dumps(python_records, separators=(",", ":")).encode()
    assert list(components) == captured["after"]
    # Rule 5: one thread or several.
    for threads in (1, 3):
        again = dict(captured["before"])
        assert native_cells.project_records(
            slice_, captured["columns"], captured["seeds"], again, threads=threads
        ) == (class_json, records_json)
        assert list(again) == captured["after"]


@pytest.mark.loop_fast
@pytest.mark.parametrize("name", sorted(DEFAULT))
def test_seed_records_equal_the_exporter(slices, monkeypatch, name):
    _records_equal(slices(name).slice, monkeypatch)


@pytest.mark.loop_fast
def test_seed_records_of_shared_weight_columns_equal_the_exporter(slices, monkeypatch):
    """The helicity-resolved columns of g g > g g share seed-weight columns
    (generation-scale-2's ``column_weights``): the projection of each weight
    column's first column equals the exporter's class and seed records."""

    from pyamplicol.loops.v2 import plan_export
    from pyamplicol.loops.v2.observables import ColumnPlan, ObservableSlice

    slice_ = slices("gg_gg").slice
    observables = ObservableSlice.of(slice_)
    plan = ColumnPlan(observables, observables.resolved_columns("helicity-colour", {}))
    _keys, of_column = plan_export.weight_columns(plan.seed_columns)
    assert len(set(of_column)) < len(of_column)
    _records_equal(slice_, monkeypatch, columns=plan.seed_columns)


# ------------------------------------------------------------ C3 degrees


def _degrees_equal(native: Generated, oracle: Generated | None = None) -> None:
    """Degrees and liveness against the oracle lane's slice, or (without
    it) against the Python stages on the production lane's slice: its
    degrees recomputed in Python and the keep closures of its own prune
    (Python liveness over the native cells and degrees)."""

    slice_ = native.slice
    expected = (
        python(oracle.slice.degrees)
        if oracle is not None
        else uncached(slice_, "degrees", slice_.degrees)
    )
    assert native_cells.degrees(slice_) == expected
    # Rule 5: one thread or several.
    assert native_cells.degrees(slice_, threads=1) == expected
    assert native_cells.degrees(slice_, threads=3) == expected
    assert slice_.degrees() == expected
    table = slice_.table
    for pairs in (pairs for _sector, pairs in slice_.sectors):
        assert native_cells.table_degree(
            table, pairs, slice_._born_grades()
        ) == table.degree(pairs, slice_._born_grades())
        assert native_cells.table_degree(table, pairs) == table.degree(pairs)
    if slice_.request.accuracy == "full":
        return
    # The liveness of _prune_colour: the keep closures on every skeleton,
    # for the slice and then its certified-scaleless companion (pruned
    # after it at LC/NLC, with the slice's degrees).
    reference = oracle or native
    pruned = [reference.slice]
    twin = reference.slice.scaleless
    if twin is not None and twin.colour_pruned:
        pruned.append(twin)
    assert len(reference.kept) == len(pruned) == len(native.kept)
    for target, keep in zip(pruned, reference.kept, strict=True):
        live = native_cells.live_classes(target)
        # Rule 5: one thread or several.
        assert native_cells.live_classes(target, threads=1) == live
        assert native_cells.live_classes(target, threads=3) == live
        checked = 0
        for skeleton, combination in target.class_of.items():
            native_live = any(live[c, skeleton.orders] for c, _x in combination)
            assert native_live == python(keep, skeleton), skeleton
            checked += 1
        assert checked


@pytest.mark.loop_fast
def test_liveness_reads_observable_tables(slices, monkeypatch):
    """Observable criteria of ``_prune_colour`` (directed tables: cells
    against operator images, the Born basis kept) enter the native liveness
    through their own cells. A copy of the slice table with cells kept at
    the projection for some dead classes revives them, as in the oracle's
    ``keep`` closure (captured from ``_prune_colour`` with the same
    criteria, aborted before it changes the slice)."""

    from dataclasses import replace

    from pyamplicol.loops.v2 import assemble, generator_input

    slice_ = slices("uux_gg_lc").slice
    degrees = slice_.degrees()["virtual"]
    grades = slice_._born_grades()
    live = native_cells.live_classes(slice_)
    cells = [list(row) for row in slice_.table.cells]
    revived = []
    for (c, order), alive in sorted(live.items()):
        found = next(
            (
                (degrees[sector] - g, b)
                for sector, pairs in slice_.sectors
                if degrees[sector] is not None
                for a, loop in pairs
                if loop == order
                for b, born in enumerate(slice_.basis.orders)
                if a in born
                for g in grades.get((b, a), ())
            ),
            None,
        )
        if alive or found is None or len(revived) == 3:
            continue
        power, b = found
        cells[c][b] = {power: ExactColorCoefficient(1)}
        revived.append((c, order))
    assert revived
    directed = replace(slice_.table, cells=tuple(map(tuple, cells)))
    native = native_cells.live_classes(slice_, [(directed, degrees)])
    assert all(native[key] for key in revived)
    assert sum(native.values()) == sum(live.values()) + len(revived)

    class Captured(Exception):
        pass

    kept: list = []

    def capture(keep):
        kept.append(keep)
        raise Captured

    monkeypatch.setattr(
        generator_input.NativeGeneration,
        "schedules",
        lambda self, colour_model, **kwargs: capture(kwargs["keep"]),
    )
    monkeypatch.setattr(
        assemble, "SkeletonView", lambda *args, keep, **kwargs: capture(keep)
    )
    with pytest.raises(Captured):
        slice_._prune_colour([(directed, degrees)])
    monkeypatch.undo()
    (keep,) = kept
    for skeleton, combination in slice_.class_of.items():
        assert any(native[c, skeleton.orders] for c, _x in combination) == keep(
            skeleton
        ), skeleton


def _orbits_equal(slice_) -> None:
    from pyamplicol.loops.v2.assemble import _AdjointOrbits

    born_keys = [
        frozenset((i.tensor, i.nc_power) for t in terms for i in t.colour.terms)
        for terms in slice_.born_terms.values()
    ]
    legs, process = slice_.basis.legs, slice_.blocks.process
    orbits = _AdjointOrbits(legs, process, born_keys)
    oracle = python(_AdjointOrbits, legs, process, born_keys)
    # The switch: the native classes and rename under the "orbits" stage.
    assert isinstance(orbits, native_cells._Orbits)
    assert type(oracle) is _AdjointOrbits
    assert orbits.classes == oracle.classes
    for tensor in slice_.basis.tensors:
        for term in tensor.terms:
            assert orbits.rename(term.tensor) == oracle.rename(term.tensor)
    for site in slice_.sites.sites:
        assert orbits.site_key(site) == oracle.site_key(site)


@pytest.mark.loop_fast
@pytest.mark.parametrize("name", sorted(DEFAULT))
def test_degrees_and_liveness_equal_python(slices, name):
    _degrees_equal(slices(name))
    _orbits_equal(slices(name).slice)


# ------------------------------------------------------------ C4 dipoles


def _dipoles_equal(native: Generated, oracle: Generated | None = None) -> None:
    slice_ = native.slice
    expected = (
        python(oracle.slice.dipoles)
        if oracle is not None
        else uncached(slice_, "dipoles", slice_.dipoles)
    )
    orbits = dipoles.build(slice_)
    assert orbits.digest() == native_cells.dipoles_digest(expected)
    # Rule 9: only the expansions decode, one count per entry.
    assert orbits.decodes == 0
    expanded = dipoles.operators(orbits)
    assert expanded == expected == slice_.dipoles()
    assert orbits.decodes == sum(len(rows) * len(rows[0]) for rows in expanded.values())
    # The stored form reloads to the same operators; one thread or several.
    text = dipoles.stored(orbits)
    assert json.loads(text)["format"] == dipoles.FORMAT
    stored = dipoles.load(text)
    assert dipoles.stored(stored) == text
    assert dipoles.operators(stored) == expected
    assert dipoles.build(slice_, threads=1).to_json() == text
    stored.check_conservation()
    floats = dipoles.matrices(stored)
    assert list(floats) == list(expected)
    assert all(
        float(x) == y
        for pair, rows in expected.items()
        for row, values in zip(rows, floats[pair], strict=True)
        for x, y in zip(row, values, strict=True)
    )


@pytest.mark.loop_fast
@pytest.mark.parametrize("name", sorted(DEFAULT))
def test_dipole_orbits_equal_python(slices, name):
    _dipoles_equal(slices(name))


# ---------------------------------------------------- C5 directed colour


# (process, Born QCD order, accuracy, histories).
OBSERVABLES_PHYSICS = {
    "gg_gg_lc_T12": ("g g > g g", 2, "lc", ("T12",)),
    "uux_ggg_directed": ("u u~ > g g g", 3, "full", ("T12", "T35", "chain")),
    "ddx_gg_directed": ("d d~ > g g", 2, "full", ("T12", "T23", "casimir", "ordered")),
    "uux_ggg_nlc_T12": ("u u~ > g g g", 3, "nlc", ("T12",)),
    "gg_ggg_lc_T13": ("g g > g g g", 3, "lc", ("T13",)),
}
OBSERVABLES_NIGHTLY = {
    "gg_gggg_lc_T12": ("g g > g g g g", 4, "lc", ("T12",)),
}
HISTORIES = {
    "T12": ColorCorrelator.dipole("T12", 1, 2),
    "T13": ColorCorrelator.dipole("T13", 1, 3),
    "T23": ColorCorrelator.dipole("T23", 2, 3),
    "T35": ColorCorrelator.dipole("T35", 3, 5),
    "casimir": ColorCorrelator.dipole("casimir", 1, 1),
    "ordered": ColorCorrelator(
        "ordered",
        (EmitGluon(1, -1), EmitGluon(1, -2)),
        (EmitGluon(2, -1), EmitGluon(3, -2)),
    ),
    "chain": ColorCorrelator(
        "chain",
        (EmitGluon(1, -1), EmitGluon(-1, -2)),
        (EmitGluon(5, -1), EmitGluon(4, -2)),
    ),
}


@pytest.fixture(scope="module")
def observables(compiled, loop_native):
    """``ObservableSlice.generate`` per (control, lane), cached."""

    from pyamplicol.correlators import CorrelatorConfig
    from pyamplicol.loops.v2.assemble import SliceRequest
    from pyamplicol.loops.v2.observables import ObservableSlice
    from pyamplicol.models.loop_color_declaration import LoopColorDeclaration

    controls = {**OBSERVABLES_PHYSICS, **OBSERVABLES_NIGHTLY}
    built: dict[tuple[str, str], Any] = {}

    def get(name: str, lane: str = "native") -> Any:
        if (name, lane) not in built:
            process, born, accuracy, histories = controls[name]
            options: dict[str, Any] = {}
            if accuracy != "full":
                options.update(
                    accuracy=accuracy,
                    colour_declaration=LoopColorDeclaration.from_dict(
                        json.loads(legacy_capture.COLOUR_DECLARATION.read_text())
                    ),
                )
            request = SliceRequest(
                process,
                {"QCD": born, "QED": 0},
                {"QCD": born + 2, "QED": 0},
                fixed_parameters=legacy_capture.FIXED_PARAMETERS,
                **options,
            )
            correlators = CorrelatorConfig(
                color_correlations=tuple(HISTORIES[h] for h in histories)
            )
            with native_cells.native_stages(LANES[lane]):
                built[name, lane] = ObservableSlice.generate(
                    compiled, request, loop_native, correlators=correlators
                )
        return built[name, lane]

    return get


def _directed_equal(native: Any, oracle: Any) -> None:
    """Directed tables (both directions), the complete-family degrees and
    the slices they pruned equal the oracle lane's; the native family
    degrees also equal the Python ones on the production lane's tables."""

    from pyamplicol.loops.v2.observables import DirectedColour

    assert snapshot.stage_digests(native.slice) == python(
        snapshot.stage_digests, oracle.slice
    )
    assert native._families == oracle._families
    directed = [h for h in native.histories if h.bra]
    assert directed
    for history in directed:
        mine, theirs = native.directed(history), python(oracle.directed, history)
        assert mine.table.cells == theirs.table.cells, history.id
        assert mine.swapped.cells == python(getattr, theirs, "swapped").cells
        # The switched table against the Python route on the same objects.
        fresh = python(DirectedColour, native.slice.table, history, None, None)
        assert mine.table.cells == fresh.table.cells
        if native.slice.request.accuracy != "full":
            family = native.family(history)
            expected = python(native._family_degrees, family)
            assert native_cells.family_degrees(native, family) == expected
            assert native._families[family] == expected


def _emissions(connection) -> list[tuple[int, int]]:
    return [(e.emitter_label, e.emitted_label) for e in connection.emissions]


@pytest.mark.loop_fast
def test_native_operator_images_equal_python():
    """``cells::operator_image`` (``connect``, then ``absorb`` by the SU(N_c)
    completeness relation) gives the colour vectors of
    ``DirectedColour.operator`` (``observables.connect`` and ``absorb``):
    equal legs and normalisation, and equal exact overlaps with random
    tensors, for dipoles, Casimirs, ordered histories and emitter chains on
    gluon and quark legs."""

    from pyamplicol.color.connections import ColorConnection
    from pyamplicol.loops.v2.observables import absorb, connect

    rng = random.Random(20261025)
    for quarks, gluons in (((), 4), (((1, 2),), 3), (((1, 2), (3, 4)), 2)):
        legs = [ColorLeg(q, 3) for q, _ in quarks] + [
            ColorLeg(b, -3) for _, b in quarks
        ]
        labels = list(range(10, 10 + gluons))
        legs += [ColorLeg(a, 8) for a in labels]
        legs = sorted(legs)
        coloured = [leg.label for leg in legs]
        histories = [
            ColorCorrelator.dipole("d", rng.choice(coloured), rng.choice(coloured))
            for _ in range(4)
        ]
        a, b, c = (rng.choice(coloured) for _ in range(3))
        histories += [
            ColorCorrelator("casimir", (EmitGluon(a, -1),), (EmitGluon(a, -1),)),
            ColorCorrelator(
                "ordered",
                (EmitGluon(a, -1), EmitGluon(a, -2)),
                (EmitGluon(b, -1), EmitGluon(c, -2)),
            ),
            ColorCorrelator(
                "chain",
                (EmitGluon(a, -1), EmitGluon(-1, -2)),
                (EmitGluon(b, -1), EmitGluon(c, -2)),
            ),
        ]
        tensors = [_random_tensor(rng, legs, labels, quarks) for _ in range(5)]
        kets = [_random_tensor(rng, legs, labels, quarks) for _ in range(5)]
        for history in histories:
            bra = ColorConnection(tuple(legs), history.bra)
            ket = ColorConnection(tuple(legs), history.ket)
            native = native_cells.operator_images(
                tensors, _emissions(bra), _emissions(ket)
            )
            for tensor, image in zip(tensors, native, strict=True):
                expected = connect(tensor, bra)
                for emission in reversed(ket.emissions):
                    expected = absorb(expected, emission)
                assert image.legs == expected.legs
                assert image.normalization_power == expected.normalization_power
                for other in [*kets, *tensors]:
                    assert colour.contract_colour(image, other) == (
                        colour.contract_colour(expected, other)
                    ), history


@pytest.mark.loop_fast
def test_directed_tables_and_family_degrees_equal_python(slices):
    """The directed tables of g g > g g (both directions; a dipole, a Casimir
    and an ordered history) against the Python route, and the complete-family degrees
    of the dipole against the Python ones on the same slice (the ordered
    history's 28-member family and the LC/NLC families with their prune
    are ``loop_physics``)."""

    from pyamplicol.correlators import CorrelatorConfig
    from pyamplicol.loops.v2.observables import DirectedColour, ObservableSlice

    correlators = CorrelatorConfig(
        color_correlations=tuple(HISTORIES[h] for h in ("T12", "casimir", "ordered"))
    )
    slice_ = slices("gg_gg").slice
    native = ObservableSlice.of(slice_, correlators=correlators)
    for history in native.histories:
        if not history.bra:
            continue
        mine = native.directed(history)
        theirs = python(DirectedColour, slice_.table, history)
        assert mine.table.cells == theirs.table.cells, history.id
        assert mine.swapped.cells == python(getattr, theirs, "swapped").cells
    # Rule 5: the native images and their cells do not depend on the
    # thread count.
    engine = native_cells.NativeCells(slice_.table.basis)
    one, four = (
        native_cells.decode(
            engine.directed([(1, -1), (1, -2)], [(2, -1), (3, -2)], threads).table(
                slice_.table.classes, threads
            )
        )
        for threads in (1, 4)
    )
    assert one == four
    for history in (HISTORIES["T12"],):
        family = native.family(history)
        assert native_cells.family_degrees(native, family) == python(
            native._family_degrees, family
        )


@pytest.mark.loop_physics
@pytest.mark.parametrize("name", sorted(OBSERVABLES_PHYSICS))
def test_wider_directed_controls_equal_python(observables, name):
    _directed_equal(observables(name), observables(name, "python"))


@pytest.mark.loop_physics
def test_ordered_history_family_equals_python(slices):
    """The 28-member complete family of an ordered two-emission history
    (784 history pairs) on g g > g g."""

    from pyamplicol.loops.v2.observables import ObservableSlice

    native = ObservableSlice.of(slices("gg_gg").slice)
    family = native.family(HISTORIES["ordered"])
    assert len(family) == 28
    assert native_cells.family_degrees(native, family) == python(
        native._family_degrees, family
    )


@pytest.mark.loop_physics
def test_family_with_a_quark_pair_splitting_equals_python(observables):
    """A family holding quark-pair splittings (connected images on both
    sides, ``DirectedColour.generic_table``) takes the native batch with
    the connected ket images."""

    from pyamplicol.color.connections import SplitGluon

    obs = observables("uux_ggg_nlc_T12")
    split = ColorCorrelator("split", (SplitGluon(3, -1, -2),), (SplitGluon(3, -1, -2),))
    family = python(obs.family, split)
    assert any(not isinstance(e, EmitGluon) for f in family for e in f.emissions)
    expected = python(obs._family_degrees, family)
    assert native_cells.family_degrees(obs, family) == expected


@pytest.mark.loop_nightly
@pytest.mark.parametrize("name", sorted(OBSERVABLES_NIGHTLY))
def test_six_leg_directed_controls_equal_python(observables, name):
    _directed_equal(observables(name), observables(name, "python"))


# ------------------------------------------------------------ no licence


@pytest.mark.loop_fast
def test_new_rust_uses_only_exact_number_domains_of_symbolica():
    """No licence call (rule 4): the package's Rust names Symbolica only
    for its exact numbers (``symbolica::domains::{integer, rational}``),
    never its state, atoms or licence. The no-key run below is a smoke test
    only: unlicensed Symbolica runs restricted rather than failing."""

    for path in _RUST:
        text = (_ROOT / path).read_text()
        uses = re.findall(r"\bsymbolica\b[\w:{}, ]*", text)
        assert all(
            re.fullmatch(r"symbolica::domains::(integer|rational)::\w+", use)
            for use in uses
        ), (path, uses)


@pytest.mark.loop_fast
def test_new_bindings_run_without_a_symbolica_key(tmp_path):
    """Every binding of the package (cells, projection, degrees, liveness,
    dipole orbits and their stored form through ``loops/v2/dipoles``, and
    C5's operator images, directed and rebased engines, registered quark
    rings and directed overlap degrees) runs in a process without a
    Symbolica key (smoke; the static check above is the one that fails on a
    Symbolica call)."""

    environment = {
        key: value
        for key, value in os.environ.items()
        if key not in ("SYMBOLICA_LICENSE", "SYMBOLICA_MASTER_LICENSE")
    }
    environment["PYTHONPATH"] = os.pathsep.join(
        entry for entry in (str(_SOURCE), environment.get("PYTHONPATH")) if entry
    )
    environment["SYMBOLICA_HIDE_BANNER"] = "1"
    source = textwrap.dedent(
        """
        from pyamplicol.loops.native import load_loop_native
        native = load_loop_native()
        legs = [(1, 8), (2, 8), (3, 8), (4, 8)]
        def trace(*labels):
            return (legs, [("1", "0", 0, ([], [list(labels)]))], 0)
        basis = [trace(1, *p) for p in ((2, 3, 4), (2, 4, 3), (3, 2, 4),
                                        (3, 4, 2), (4, 2, 3), (4, 3, 2))]
        engine = native.NativeCellEngine(legs, basis)
        table = engine.table([(0, [1, 2, 3, 4], 0, [], None),
                              (1, [2, 1, 4, 3], 0, [], None),
                              (3, [], 0, [], trace(1, 3, 2, 4))])
        values, ids = table.decode()
        assert len(ids) == 3 and table.decodes == 3
        native.colour_gram(basis)
        native.overlap_degrees(legs, [[(([], [[1, 2, 3, 4]]), 0)]], [(0, 0, 0)])
        assert native.tensor_overlap_degrees(
            [[(basis[0], 1)]], [[(basis[1], 0)]], [(0, 0, 0)]) == [3]
        orbits = native.NativeDipoleOrbits.build(legs, basis, [[1, 2, 3, 4]])
        orbits.check_conservation()
        from pyamplicol.loops.v2 import dipoles
        loaded = dipoles.load(dipoles.stored(orbits))
        loaded.expand_f64()
        assert dipoles.operators(loaded) and dipoles.matrices(loaded)
        assert loaded.digest() == orbits.digest() and loaded.expand()
        assert native.contract_colours([(basis[0], basis[1])])
        assert native.face_counts([(([], [[1, 2, 3, 4]]), ([], [[1, 3, 2, 4]]))])
        copy = native.NativeCellTable.from_cells([[v for v in (values[i] for i in row)]
                                                  for row in ids])
        assert copy.concatenated(table).decode()[1][3:] == ids
        assert copy.degree([[0]] * 3, [[0]] * 6, [(0, 0)]) is not None
        keys = native.NativeAdjointOrbits([(1, 21), (2, 21), (3, 21), (4, 21)],
                                         [[(([], [[1, 2, 3, 4]]), 0)]])
        keys.rename(([], [[2, 1, 4, 3]]))
        native.live_classes([(table, [None])], [[(0, 0)]], [[0]] * 6,
                            [((0, 0), [0])], "lc", [(0, 0)])
        native.project_records([(table, [[0]] * 3, [[0]] * 6)],
                               [(0, [(0, 0)], None, None)],
                               [((b, 0), [0]) for b in range(6)], "nlc", [],
                               [([(0, "1", "0")], 0)], 2)
        # C5: operator images, directed and rebased engines, quark rings.
        images = native.operator_images(basis[:2], [(1, -1)], [(2, -1)])
        assert len(images) == 2 and images[0][0] == legs
        directed = engine.directed([(1, -1)], [(2, -1)], 2)
        assert directed.table([(0, [1, 2, 3, 4], 0, [], None)]).decode()[1]
        rebased = engine.rebased(basis[::-1])
        ring = rebased.register_ring(basis[1])
        assert rebased.table([(2, [], ring, [], None)]).decode()[1]
        assert native.directed_overlap_degrees(
            legs, [[(([], [[1, 2, 3, 4]]), 0)]], [[(([], [[1, 3, 2, 4]]), 0)]],
            [([(1, -1)], [(2, -1)])], [(0, 0, 0, 0)], 2) == [3]
        print("no licence needed", orbits.representatives)
        """
    )
    completed = subprocess.run(
        (sys.executable, "-c", source),
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
        cwd=tmp_path,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "no licence needed" in completed.stdout


# ----------------------------------------------------------- wider controls


def _all_equal(name: str, slices, monkeypatch) -> None:
    native, oracle = slices(name), slices(name, "python")
    _switched_equal(native, oracle)
    _seed_cells_equal(native, oracle)
    _records_equal(native.slice, monkeypatch)
    _degrees_equal(native, oracle)
    _orbits_equal(native.slice)
    if native.slice.request.accuracy == "full":
        _dipoles_equal(native, oracle)


@pytest.mark.loop_physics
@pytest.mark.parametrize("name", sorted([*PHYSICS, *DEFAULT]))
def test_wider_controls_equal_python(slices, monkeypatch, name):
    """Every control generated in both lanes: equal end to end (S1, S4, S5,
    S8, the companion, the prune) and stage by stage."""

    _all_equal(name, slices, monkeypatch)


@pytest.mark.loop_nightly
@pytest.mark.parametrize("name", sorted(NIGHTLY))
def test_six_leg_controls_equal_python(slices, monkeypatch, name):
    _all_equal(name, slices, monkeypatch)
