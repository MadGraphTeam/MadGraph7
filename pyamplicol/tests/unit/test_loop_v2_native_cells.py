# SPDX-License-Identifier: 0BSD
"""Native exact colour cells against their Python oracles (WP3.1 package C).

``loops/v2/native_cells`` (rusticol-core ``loop_v2::generate::cells`` and
``loop_v2::dipole_orbits``) must equal the Python stages exactly:

* C1: ``colour.seed_cells`` cell by cell (adjoint rings, fundamental traces,
  quark rings and generic classes), ``contract_colour``, ``face_count`` and
  ``BornBasis.gram``; outputs independent of the thread count;
* C2: ``plan_export._Exporter.seed_records``, byte-identical plan JSON and
  the same Born components;
* C3: ``LoopSlice.degrees`` and the liveness of ``_prune_colour``;
* C4: ``LoopSlice.dipoles`` from the orbit representatives built in
  rusticol-core, which needs no Symbolica licence.

Default tier (``loop_fast``, about 25 s): g g > g g (FC; adjoint rings and
fundamental traces) and u u~ > g g g (LC; quark rings and generic classes).
Wider controls are ``loop_physics`` (g g > g g g and u u~ > g g g for the
dipoles, as WP3.1 I5c asks), six legs ``loop_nightly``.
"""

from __future__ import annotations

import json
import os
import random
import subprocess
import sys
import textwrap
from fractions import Fraction
from pathlib import Path

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
from tools.loop_reference import legacy_capture

_SOURCE = Path(__file__).resolve().parents[2] / "src"

# (process, Born QCD order, accuracy[, counterterm form]). The default tier
# covers both forms of the counterterm keys of the degrees: the recursion's
# closure terms (g g > g g) and site closure (u u~ > g g g, the cheaper form).
DEFAULT = {
    "gg_gg": ("g g > g g", 2, "full"),
    "uux_ggg_lc": ("u u~ > g g g", 3, "lc", "sites"),
}
PHYSICS = {
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


@pytest.fixture(scope="module")
def slices(loop_sm_compiled_model, loop_native):
    """Generated slices per control, with the ``keep`` closures that
    ``_prune_colour`` handed to the native schedules (its liveness oracle)."""

    from pyamplicol.loops.v2 import generator_input
    from pyamplicol.loops.v2.assemble import LoopSlice, SliceRequest
    from pyamplicol.models.loading import load_compiled_model
    from pyamplicol.models.loop_color_declaration import LoopColorDeclaration

    compiled = load_compiled_model(loop_sm_compiled_model)
    declaration = LoopColorDeclaration.from_dict(
        json.loads(legacy_capture.COLOUR_DECLARATION.read_text())
    )
    built: dict[str, tuple] = {}

    def get(name: str):
        if name not in built:
            process, born, accuracy, *form = CONTROLS[name]
            options = {"counterterms": form[0]} if form else {}
            if accuracy != "full":
                options.update(accuracy=accuracy, colour_declaration=declaration)
            kept: list = []
            original = generator_input.NativeGeneration.schedules

            def spy(self, colour_model, **kwargs):
                if kwargs.get("keep") is not None:
                    kept.append(kwargs["keep"])
                return original(self, colour_model, **kwargs)

            generator_input.NativeGeneration.schedules = spy
            try:
                slice_ = LoopSlice.generate(
                    compiled,
                    SliceRequest(
                        process,
                        {"QCD": born, "QED": 0},
                        {"QCD": born + 2, "QED": 0},
                        fixed_parameters=legacy_capture.FIXED_PARAMETERS,
                        **options,
                    ),
                    loop_native,
                )
            finally:
                generator_input.NativeGeneration.schedules = original
            built[name] = (slice_, kept)
        return built[name]

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


# Refusal of each primitive kind against a bra with an open quark string.
_REFUSALS = {
    "adjoint-ring": "adjoint-ring class against a Born",
    "fundamental-trace": "bra string is not glued",
}


@pytest.mark.loop_fast
@pytest.mark.parametrize("kind", sorted(_REFUSALS))
def test_primitives_against_quark_bras_are_refused(kind):
    """Primitives against bras the relabelling cache cannot key (a quark
    leg) take the direct path of ``_PrimitiveCells._cell``. A primitive
    carries no fundamental index, so its overlap with a bra holding an open
    quark string is not defined: refused natively (rule 2), where Python
    refuses only a string holding the primitive's labels (oracle defect 3,
    below)."""

    primitive = colour.Primitive(kind, (3, 4))
    for structure in (_SPLIT, _JOINED):
        basis = _quark_basis(structure)
        with pytest.raises(ValueError, match=_REFUSALS[kind]):
            native_cells.seed_cells(basis, [primitive])
        with pytest.raises(ValueError, match=_REFUSALS[kind]):
            native_cells.NativeCells(basis)(primitive)
    with pytest.raises(ValueError, match="not contracted exactly twice"):
        colour.seed_cells(_quark_basis(_JOINED), [primitive])


@pytest.mark.loop_fast
@pytest.mark.parametrize("kind", sorted(_REFUSALS))
@pytest.mark.xfail(
    strict=True,
    raises=pytest.fail.Exception,
    reason="oracle defect 3 (read-only, rule 3): Python's overlap drops the "
    "bra's quark string (color.connections.overlap_traces) and returns a "
    "value for a primitive against a bra with an open string",
)
def test_python_primitive_against_a_quark_bra_is_refused(kind):
    with pytest.raises(ValueError):
        colour.seed_cells(_quark_basis(_SPLIT), [colour.Primitive(kind, (3, 4))])


@pytest.mark.loop_fast
def test_stored_dipole_orbits_are_validated(slices):
    slice_, _kept = slices("gg_gg")
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


def _seed_cells_equal(slice_) -> None:
    table = slice_.table
    basis = table.basis
    kinds = {
        seed.kind if isinstance(seed, colour.Primitive) else "generic"
        for seed in table.classes
    }
    assert kinds, "no seed class"
    assert native_cells.seed_cells(basis, table.classes) == table.cells
    # The Born metric.
    assert native_cells.gram(basis) == basis.gram


@pytest.mark.loop_fast
@pytest.mark.parametrize("name", sorted(DEFAULT))
def test_seed_cells_equal_python(slices, name):
    slice_, _kept = slices(name)
    _seed_cells_equal(slice_)
    # Rule 5: one thread and several give the same cells.
    table = slice_.table
    one = native_cells.seed_cells(table.basis, table.classes, threads=1)
    assert one == native_cells.seed_cells(table.basis, table.classes, threads=4)


@pytest.mark.loop_fast
def test_rebased_cells_share_caches_and_equal_python(slices):
    """A rebased engine (another basis on the same legs, e.g. operator
    images) equals ``seed_cells`` against that basis and shares the caches
    both ways, as ``_PrimitiveCells.rebased``."""

    slice_, _kept = slices("uux_ggg_lc")
    table = slice_.table
    basis = table.basis
    other = colour.BornBasis(
        basis.legs,
        basis.sector_ids[::-1],
        basis.tensors[::-1],
        basis.orders[::-1],
    )
    engine = native_cells.NativeCells(basis)
    rebased = engine.rebased(other)
    assert native_cells.seed_cells(other, table.classes, rebased) == colour.seed_cells(
        other, table.classes
    )
    # The parent sees the rebased engine's relabelled bras, rings and ring
    # overlaps; its own table (the same tensors reordered) is all cache hits.
    shared = rebased.cache_sizes
    assert shared[0] and shared[1] and shared[3], shared
    assert engine.cache_sizes == shared
    assert native_cells.seed_cells(basis, table.classes, engine) == table.cells
    assert engine.cache_sizes == rebased.cache_sizes == shared
    with pytest.raises(ValueError, match="another basis"):
        native_cells.seed_cells(basis, table.classes, rebased)


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
    components = dict(captured["before"])
    class_json, records_json = native_cells.project_records(
        slice_, captured["columns"], captured["seeds"], components
    )
    python_classes, python_records = captured["result"]
    assert class_json == json.dumps(python_classes, separators=(",", ":")).encode()
    assert records_json == json.dumps(python_records, separators=(",", ":")).encode()
    assert list(components) == captured["after"]


@pytest.mark.loop_fast
@pytest.mark.parametrize("name", sorted(DEFAULT))
def test_seed_records_equal_the_exporter(slices, monkeypatch, name):
    slice_, _kept = slices(name)
    _records_equal(slice_, monkeypatch)


# ------------------------------------------------------------ C3 degrees


def _degrees_equal(slice_, kept) -> None:
    assert native_cells.degrees(slice_) == slice_.degrees()
    table = slice_.table
    for pairs in (pairs for _sector, pairs in slice_.sectors):
        assert native_cells.table_degree(
            table, pairs, slice_._born_grades()
        ) == table.degree(pairs, slice_._born_grades())
        assert native_cells.table_degree(table, pairs) == table.degree(pairs)
    if slice_.request.accuracy == "full":
        return
    # The liveness of _prune_colour: its keep closure on every skeleton.
    (keep,) = kept
    live = native_cells.live_classes(slice_)
    checked = 0
    for skeleton, combination in slice_.class_of.items():
        native = any(live[c, skeleton.orders] for c, _x in combination)
        assert native == keep(skeleton), skeleton
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

    slice_, _kept = slices("uux_ggg_lc")
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


@pytest.mark.loop_fast
@pytest.mark.parametrize("name", sorted(DEFAULT))
def test_degrees_and_liveness_equal_python(slices, name):
    slice_, kept = slices(name)
    _degrees_equal(slice_, kept)
    from pyamplicol.loops.v2.assemble import _AdjointOrbits

    born_keys = [
        frozenset((i.tensor, i.nc_power) for t in terms for i in t.colour.terms)
        for terms in slice_.born_terms.values()
    ]
    legs, process = slice_.basis.legs, slice_.blocks.process
    orbits = native_cells.adjoint_orbits(legs, process, born_keys)
    python = _AdjointOrbits(legs, process, born_keys)
    assert orbits.classes == python.classes
    for tensor in slice_.basis.tensors:
        for term in tensor.terms:
            assert orbits.rename(term.tensor) == python.rename(term.tensor)
    for site in slice_.sites.sites:
        assert orbits.site_key(site) == python.site_key(site)


# ------------------------------------------------------------ C4 dipoles


def _dipoles_equal(slice_) -> None:
    orbits = native_cells.dipole_orbits(slice_)
    assert native_cells.dipoles(slice_, orbits) == slice_.dipoles()
    assert orbits.digest() == native_cells.dipoles_digest(slice_.dipoles())
    # The stored form reloads to the same operators; one thread or several.
    stored = type(orbits).from_json(orbits.to_json())
    assert stored.to_json() == orbits.to_json()
    assert native_cells.dipoles(slice_, stored) == slice_.dipoles()
    assert native_cells.dipole_orbits(slice_, threads=1).to_json() == orbits.to_json()
    stored.check_conservation()


@pytest.mark.loop_fast
@pytest.mark.parametrize("name", sorted(DEFAULT))
def test_dipole_orbits_equal_python(slices, name):
    slice_, _kept = slices(name)
    _dipoles_equal(slice_)


@pytest.mark.loop_fast
def test_new_bindings_need_no_symbolica_licence(tmp_path):
    """The native cells, projection, degrees and dipole orbits run in a
    process without a Symbolica key (no licence call)."""

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
        orbits = native.NativeDipoleOrbits.build(legs, basis, [[1, 2, 3, 4]])
        orbits.check_conservation()
        loaded = native.NativeDipoleOrbits.from_json(orbits.to_json())
        loaded.expand_f64()
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
                               [([(0, "1", "0")], 0)])
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


@pytest.mark.loop_physics
@pytest.mark.parametrize("name", sorted(PHYSICS))
def test_wider_controls_equal_python(slices, monkeypatch, name):
    slice_, kept = slices(name)
    _seed_cells_equal(slice_)
    _records_equal(slice_, monkeypatch)
    _degrees_equal(slice_, kept)
    if slice_.request.accuracy == "full":
        _dipoles_equal(slice_)


@pytest.mark.loop_nightly
@pytest.mark.parametrize("name", sorted(NIGHTLY))
def test_six_leg_controls_equal_python(slices, monkeypatch, name):
    slice_, kept = slices(name)
    _seed_cells_equal(slice_)
    _records_equal(slice_, monkeypatch)
    _degrees_equal(slice_, kept)
    if slice_.request.accuracy == "full":
        _dipoles_equal(slice_)
