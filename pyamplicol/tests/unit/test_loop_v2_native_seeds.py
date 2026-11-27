# SPDX-License-Identifier: 0BSD
"""Native seed combinations and classes equal the Python seed tables (WP3.1
I10, package native-seeds B2/B3).

``loops.v2.colour.build_seed_table`` / ``extend_seed_table`` with the
switch off (``native_seeds.python_oracle``) are the oracles: the native
stages (``loops.v2.native_seeds``, the production default) give equal tables
(classes in the same order, orders, cells, certificates) and an equal
``class_of`` (same skeleton order, same rows), with and without certified
compressions, for builds and extensions; the native prune liveness equals
``LoopSlice._prune_colour``'s; the bindings need no Symbolica key and their
output does not depend on the thread count.

B1 (the retained arena): a slice's seed stage reads its schedule build's
readings in-crate, the LC/NLC prune rebuilds the schedules from the retained
skeletons (with the callback or the native liveness: equal plans) and
decodes nothing else, and the arena answers the rank bound.

Default tier: three 4-5-leg controls (adjoint rings and fundamental traces,
quark rings, two quark lines with generic seeds) and g g > g g at LC.
``loop_physics``: the slice (readings, LC/NLC prune, S5 digests, plan bytes).
``loop_nightly``: six legs.
"""

from __future__ import annotations

import dataclasses
import itertools
import json
import os
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

import pytest

from pyamplicol.loops.v2 import generator_input, native_seeds
from pyamplicol.loops.v2.blocks import build_tree_blocks
from pyamplicol.loops.v2.colour import (
    SeedColour,
    build_seed_table,
    extend_seed_table,
    seed_fourier,
)
from pyamplicol.loops.v2.families import (
    LoopOrderSelection,
    ModelLoopCatalogue,
    add_orders,
    generate_families,
)
from pyamplicol.loops.v2.skeletons import family_skeletons
from tests.unit.test_loop_v2_generate import assert_same_schedules
from tools.loop_reference import legacy_capture

ROOT = Path(__file__).resolve().parents[2]


def python_build(colour, skeletons, basis=None, *, compress=False):
    """The Python oracle of ``build_seed_table``."""

    with native_seeds.python_oracle():
        return build_seed_table(colour, skeletons, basis, compress=compress)


def python_extend(table, class_of, colour, skeletons, *, compress=False):
    """The Python oracle of ``extend_seed_table``."""

    with native_seeds.python_oracle():
        return extend_seed_table(table, class_of, colour, skeletons, compress=compress)


def assert_same_tables(native, python) -> None:
    """Equal seed tables and ``class_of``, including every order."""

    (table, class_of), (expected, expected_class_of) = native, python
    assert table.basis is expected.basis or table.basis == expected.basis
    assert table.classes == expected.classes
    assert table.orders == expected.orders
    assert table.cells == expected.cells
    assert table.formal_nc == expected.formal_nc
    assert list(table.certificates) == list(expected.certificates)
    for kind, certificate in table.certificates.items():
        other = expected.certificates[kind]
        assert certificate.vertices == other.vertices, kind
        assert certificate.tree_vertices == other.tree_vertices, kind
        assert certificate.instance == other.instance, kind
        assert certificate.theorem == other.theorem, kind
    assert list(class_of) == list(expected_class_of)
    assert list(class_of.values()) == list(expected_class_of.values())


@pytest.fixture(scope="module")
def controls(loop_sm_compiled_model):
    """(colour, skeletons) of a control, as ``test_loop_v2_colour.v2_slice``."""

    from pyamplicol.loops.compiler import prepare_loop_request_model
    from pyamplicol.models.external import CompiledUFOModel
    from pyamplicol.models.loading import load_compiled_model
    from pyamplicol.processes.model import build_model_process_ir

    compiled = load_compiled_model(loop_sm_compiled_model)
    built: dict = {}

    def build(name: str):
        if name not in built:
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
            process = build_model_process_ir(
                control.process, compiled.ir, color_accuracy="full"
            )
            blocks = build_tree_blocks(
                process,
                CompiledUFOModel(compiled),
                born_coupling_orders=control.born_order_bounds,
                block_coupling_orders=control.loop_order_bounds,
            )
            loop = add_orders(tuple(control.loop_order_bounds.items()))
            families = generate_families(
                blocks.block_table(data),
                catalogue,
                LoopOrderSelection(frozenset({"QCD"}), frozenset({loop})),
            )
            colour = SeedColour(blocks, data, compiled.ir.vertex_terms)
            skeletons = [
                s
                for family in families.families.values()
                for s in family_skeletons(families, family, colour, catalogue)
            ]
            built[name] = colour, skeletons
            generations[name] = (
                blocks.block_table(data),
                catalogue,
                LoopOrderSelection(frozenset({"QCD"}), frozenset({loop})),
            )
            leg_rows[name] = generator_input.leg_rows(blocks)
        return built[name]

    generations: dict = {}
    leg_rows: dict = {}
    build.generations = generations  # type: ignore[attr-defined]
    build.leg_rows = leg_rows  # type: ignore[attr-defined]
    return build


def _native_generation(controls, loop_native, name: str):
    """(seed colour, native generation, Python families, catalogue) of a
    control (``controls``)."""

    from pyamplicol.loops.v2.generator_input import NativeGeneration

    colour, _skeletons = controls(name)
    table, catalogue, selection = controls.generations[name]
    return (
        colour,
        NativeGeneration(table, catalogue, selection, loop_native),
        generate_families(table, catalogue, selection),
        catalogue,
    )


CONTROLS = ["gg_gg", "uux_ggg", "uux_ddxg"]


@pytest.mark.parametrize(
    ("name", "compress"),
    [
        pytest.param(name, compress, marks=mark)
        for name in CONTROLS
        for compress in (True, False)
        # Generic seeds of u u~ > g g g only (5 s): uux_ddxg has generic ones.
        for mark in [
            pytest.mark.loop_physics
            if (name, compress) == ("uux_ggg", False)
            else pytest.mark.loop_fast
        ]
    ],
)
def test_native_seed_tables_equal_python(controls, loop_native, name, compress):
    """Build on two thirds of the skeletons, then extend with an overlapping
    later part (new skeletons and skeletons already seeded): tables and
    ``class_of`` equal the Python stages', and so do the skeletons' orders.
    The native stages run first on empty reduction caches, so their cells
    read the natively reduced rings; the oracle recomputes its own. The
    stage keeps no replay copy of its inputs by default."""

    colour, skeletons = controls(name)
    first = skeletons[: (2 * len(skeletons)) // 3]
    rest = skeletons[len(first) // 2 :]
    with native_seeds.python_oracle():
        pass  # empty the reduction caches
    stage = native_seeds.NativeSeedStage(colour, loop_native)
    native = stage.build(first, compress=compress)
    assert stage.spec_text == "" and stage.reduce_inputs is None
    extended = native_seeds.extend_seed_table(*native, colour, rest, compress=compress)
    python = python_build(colour, first, compress=compress)
    assert_same_tables(native, python)
    orders = stage._builder.skeleton_orders()
    assert [stage._orders[o] for o in orders] == [s.orders for s in first]
    assert bool(native[0].certificates) == compress
    assert_same_tables(
        extended, python_extend(*python, colour, rest, compress=compress)
    )
    # The switch: the colour functions run the native stages.
    assert_same_tables(build_seed_table(colour, first, compress=compress), python)


@pytest.mark.loop_fast
@pytest.mark.parametrize(
    ("name", "compress"), [("uux_ggg", True), ("uux_ddxg", False), ("gg_gg", False)]
)
def test_native_reductions_equal_python(
    controls, loop_native, monkeypatch, name, compress
):
    """The in-crate reductions (``networks.rs``): every quark-ring need's
    tensor equals ``reduce_network`` of the canonical ring's network and
    every generic need's seed equals ``SeedColour.seed``; the stage with
    the Python reductions (``NATIVE_REDUCTIONS`` off) gives the same table;
    the native ring tensors fill ``_reduced_primitive``'s cache."""

    from pyamplicol.loops.v2 import colour as colour_module
    from pyamplicol.loops.v2.colour import reduce_network

    colour, skeletons = controls(name)
    with native_seeds.python_oracle():
        pass  # empty the reduction caches
    stage = native_seeds.NativeSeedStage(colour, loop_native)
    native = stage.build(skeletons, compress=compress)
    rings, generics = stage.needs
    assert stage.encoder is not None
    assert generics and bool(rings) == compress
    legs = native[0].basis.legs
    external = {leg.label: leg.label for leg in legs}
    for k, need in enumerate(rings):
        ring = stage.encoder.ring(need)
        expected = reduce_network(ring.network(), legs, external)
        decoded = native_seeds._decoded(stage._builder.ring_tensor(k), legs)
        assert decoded == expected
        assert colour_module._reduced_primitive(ring, legs) == expected  # cached
    for g, s in enumerate(generics):
        assert native_seeds._decoded(
            stage._builder.generic_tensor(g), colour.legs
        ) == colour.seed(skeletons[s])
    monkeypatch.setattr(native_seeds, "NATIVE_REDUCTIONS", False)
    assert_same_tables(
        native_seeds.NativeSeedStage(colour, loop_native).build(
            skeletons, compress=compress
        ),
        native,
    )


@pytest.mark.loop_fast
def test_native_extension_of_an_uncompressed_table(controls, loop_native):
    """Extending a compressed table without compression (existing quark
    rings keyed by themselves, generic classes by their tensors) and an
    uncompressed one with it (no certificate: generic seeds only)."""

    colour, skeletons = controls("uux_ddxg")
    half = len(skeletons) // 2
    for built, extended in ((True, False), (False, True)):
        table = python_build(colour, skeletons[:half], compress=built)
        assert_same_tables(
            native_seeds.extend_seed_table(
                *table, colour, skeletons, compress=extended, native=loop_native
            ),
            python_extend(*table, colour, skeletons, compress=extended),
        )


@pytest.mark.loop_fast
def test_native_extension_of_an_empty_table_with_certificates(controls, loop_native):
    """The poles companion's seeds on the slice's certificates (an empty
    table holding another table's certificates, extended with compression;
    generation-scale-2's lean companion): equal to the Python stage."""

    from pyamplicol.loops.v2.colour import SeedTable

    colour, skeletons = controls("uux_ggg")
    half = len(skeletons) // 2
    built, _class_of = python_build(colour, skeletons[:half], compress=True)
    assert built.certificates
    empty = SeedTable(
        built.basis, (), (), (), built.formal_nc, certificates=dict(built.certificates)
    )
    assert_same_tables(
        native_seeds.extend_seed_table(
            empty, {}, colour, skeletons[half:], compress=True, native=loop_native
        ),
        python_extend(empty, {}, colour, skeletons[half:], compress=True),
    )


_NO_KEY_SCRIPT = r"""
import json, sys
from pyamplicol.loops.native import load_loop_native

native = load_loop_native()
spec = {
    "adjoint": [1, 2, 3, 4],
    "compress": True,
    "coefficients": [["1/1", "0/1"], ["-1/2", "1/3"]],
    "templates": [],
    "descriptors": [[0, [k], False, 0] for k in (1, 2, 3, 4)] + [[0, [3, 4], False, 0]],
    "contents": [[0, True, [[0, d]], [[0, 1]]] for d in range(4)]
    + [[0, False, [[1, 4], [0, 4]], [[0, 2]]]],
    "existing": [],
    "tensors": [],
}
rings = [[0, 1, 2, 3], [0, 2, 1, 3], [3, 2, 1, 0], [0, 1, 4], [4, 1, 0]]
offsets = [0]
ids = []
for ring in rings:
    ids += ring
    offsets.append(len(ids))
builder = native.SeedBuilder(
    json.dumps(spec),
    b"".join(o.to_bytes(8, "little") for o in offsets),
    b"".join(i.to_bytes(4, "little") for i in ids),
)
candidates = builder.analyse()
combination = builder.combination(3)
needs = builder.needs(["adjoint-ring"])
out = builder.complete("[]", "[]")
print(json.dumps([candidates, combination, needs, out, builder.coefficients(),
                  builder.order_vectors(), builder.skeleton_orders()]))
"""


def _check_no_key_seeds(one: list) -> None:
    """The phase-0 binding on adjoint rings 1234, 1324, 4321 (the reflection
    of 1234, sign +1) and a position with two decomposition terms on the
    word 34."""

    candidates, combination, needs, out, coefficients, orders, skeleton_orders = one
    assert candidates == [["adjoint-ring", [0, 1, 2]]]
    assert needs == [[], []]
    classes, class_orders, rows, skeleton_rows = out
    words = [primitive[1] for _tag, primitive, _g in classes]
    assert words == [[1, 2, 3, 4], [1, 3, 2, 4], [1, 2, 4, 3]]
    # 4321 is the reflection of 1234 with sign (-1)^4: the same class.
    assert skeleton_rows[0] == skeleton_rows[2]
    assert [rows[r] for r in skeleton_rows[:2]] == [[[0, 0]], [[1, 0]]]
    # (1 + (-1/2 + i/3)) Tr(F^1 F^2 F^3 F^4)-type words 1 2 (34).
    assert combination is not None and len(combination) == 1
    assert coefficients[combination[0][1]] == ["1/2", "1/3"]
    assert [orders[o] for o in skeleton_orders] == [[[0, 4]]] * 5
    assert len(class_orders) == 3


def _no_key_runs(threads: Sequence[str], script: str, *args: str) -> list[list]:
    """The last two JSON lines of ``script`` run without a Symbolica key,
    once per Rayon thread count of ``threads`` (concurrently)."""

    env = {
        key: value
        for key, value in os.environ.items()
        if key not in ("SYMBOLICA_LICENSE", "SYMBOLICA_MASTER_LICENSE")
    }
    env["SYMBOLICA_HIDE_BANNER"] = "1"
    runs = [
        subprocess.Popen(
            [sys.executable, "-c", script, *args],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env={**env, "RAYON_NUM_THREADS": count},
            cwd=ROOT,
        )
        for count in threads
    ]
    out = []
    for run in runs:
        stdout, stderr = run.communicate(timeout=120)
        assert run.returncode == 0, stderr[-4000:]
        out.append([json.loads(line) for line in stdout.strip().splitlines()[-2:]])
    return out


_ARENA_SCRIPT = r"""
import hashlib, json, sys
from pyamplicol.loops.native import load_loop_native

native = load_loop_native()
data = json.loads(open(sys.argv[1]).read())
generation = native.V2Generation(data["families"])
built = generation.schedules(data["schedules"])
rebuilt = generation.rebuild(built[0], data["schedules"], bytes(data["keep"]))
readings = generation.readings(built[0])
seeds = native.SeedBuilder.from_generation(generation, built[0], data["spec"])
candidates = seeds.analyse()
needs = seeds.needs(data["certified"])
seeds.reduce(*data["reduce"])
out = seeds.complete_reduced()
rings = [seeds.ring_tensor(k) for k in range(len(needs[0]))]
generics = [seeds.generic_tensor(g) for g in range(len(needs[1]))]
digest = hashlib.sha256(
    repr((built, rebuilt, readings, candidates, needs, out, rings, generics,
          seeds.coefficients()))
    .encode()
).hexdigest()
print(json.dumps([digest, generation.max_rank(rebuilt[0]), len(out[3])]))
"""


@pytest.mark.loop_fast
def test_retained_arena(controls, loop_native, tmp_path, monkeypatch):
    """B1 on u u~ > g g g (seed colour model): the readings of a retained
    build feed the seed stage in-crate (equal to the Python oracle and to the
    bytes tables); a skeleton list that is not exactly a build's readings
    takes the bytes tables. A filtered build is rebuilt from the retained
    skeletons, with a keep callback or with flags per reading
    (``ReadingKeep``), equal to the Python ``build_schedules``, decoding only
    the build and its rebuilds. The arena's rank bound is ``MonomialLayout``'s.
    Every native seeds binding (phase 0 and B1) runs without a Symbolica
    key, with equal output on one and many threads."""

    from pyamplicol.loops.v2 import generator_input
    from pyamplicol.loops.v2 import reference_executor as rx
    from pyamplicol.loops.v2.schedules import SkeletonView, build_schedules

    # The stages' inputs are replayed in subprocesses below.
    monkeypatch.setattr(generator_input, "KEEP_REPLAY_INPUTS", True)
    colour, generation, families, catalogue = _native_generation(
        controls, loop_native, "uux_ggg"
    )
    plan = generation.schedules(colour, leg_rows=controls.leg_rows["uux_ggg"])
    base = generation.base
    assert base is not None and base.complete and base.plan is plan
    readings = rx.reading_skeletons(plan)
    found = generator_input.build_of(readings)
    assert found is not None and found[0] is base
    assert generator_input.build_of(readings[:-2]) is None
    assert generator_input.build_of(list(reversed(readings))) is None
    swapped = list(readings)
    swapped[1] = dataclasses.replace(
        swapped[1],
        contents=tuple(dataclasses.replace(c) for c in swapped[1].contents),
    )
    assert swapped == readings
    assert generator_input.build_of(swapped) is None
    stage = native_seeds.NativeSeedStage(colour, loop_native)
    in_crate = stage.build(readings, compress=True)
    assert stage.build_source is base
    replay = stage.spec_text, stage.reduce_inputs
    assert stage.counts["ring_reductions"] and stage.counts["generic_seeds"]
    by_bytes = stage.build(swapped, compress=True)
    assert stage.build_source is None
    assert_same_tables(in_crate, by_bytes)
    assert_same_tables(in_crate, python_build(colour, readings, compress=True))

    # The prune: every family's frame, a filter on the F readings.
    frames = {key: schedule.trie.frame for key, schedule in plan.schedules.items()}

    def keep(skeleton):
        return skeleton.contents[0].mask % 3 != 1 or len(skeleton.contents) > 3

    flags = bytes(keep(s) if i % 2 == 0 else i % 3 == 0 for i, s in enumerate(readings))
    by_callback = generation.schedules(colour, frames=frames, keep=keep)
    by_flags = generation.schedules(
        colour, frames=frames, keep=generator_input.ReadingKeep(base, flags)
    )
    python = build_schedules(
        families,
        catalogue,
        colour,
        policy=lambda family, candidates: next(
            trie for frame, trie in candidates if frame == frames[family.key]
        ),
        view=SkeletonView(catalogue, colour, keep=keep),
    )
    assert_same_schedules(by_callback, python)
    assert_same_schedules(by_flags, python)
    assert len(by_callback.view._contents) < len(plan.view._contents)
    assert dict(generation.decodes) == {"build": 1, "rebuild": 2}
    for schedules in (plan, by_callback):
        build = generation.build_of_plan(schedules)
        assert build is not None
        assert build.max_rank() == max(
            node.rank_in + max(segment.q_degree for segment, _ in node.edges)
            for schedule in schedules.schedules.values()
            for node in schedule.trie.nodes
        )

    # No key, one and many threads: the arena and the in-crate seed feed.
    inputs = tmp_path / "arena.json"
    inputs.write_text(
        json.dumps(
            {
                "families": generation.input_json,
                "schedules": json.dumps(
                    {
                        **base.payload,
                        "policy": {
                            "kind": "fixed",
                            "frames": [
                                [frame.start, frame.reflected]
                                for frame in frames.values()
                            ],
                        },
                    }
                ),
                "keep": [
                    int(keep(s)) for s in itertools.chain.from_iterable(base.skeletons)
                ],
                "spec": replay[0],
                "reduce": replay[1],
                "certified": list(in_crate[0].certificates),
            }
        )
    )
    one, many = _no_key_runs(("1", "16"), _NO_KEY_SCRIPT + _ARENA_SCRIPT, str(inputs))
    assert one == many
    _check_no_key_seeds(one[0])
    assert one[1][2] == len(readings)


@pytest.mark.loop_fast
def test_lazy_fourier_is_the_eager_certification(controls):
    """Design note of ``native_seeds`` (RUST-MIGRATION §10 item 7): the
    lazy cache returns the eager ``seed_fourier`` of the table at its first
    read and again after a reset, so per-point seeds and the reported
    contraction cannot change."""

    colour, skeletons = controls("uux_ggg")
    table, _class_of = build_seed_table(colour, skeletons, compress=True)
    current = {"table": table}
    lazy = native_seeds.LazyFourier(lambda: current["table"])
    eager = seed_fourier(table)
    assert eager is not None
    assert lazy.get() == eager
    smaller, _ = build_seed_table(colour, skeletons[:10], compress=True)
    current["table"] = smaller
    assert lazy.get() == eager  # cached until _plan_contraction resets it
    lazy.reset()
    assert lazy.get() == seed_fourier(smaller)


@pytest.mark.loop_physics
def test_lazy_seed_fourier_keeps_points_and_the_reported_contraction(
    loop_sm_compiled_model, loop_native, monkeypatch
):
    """The phase-1 edit of the design note, emulated by a subclass: under
    "auto" ``_plan_contraction`` installs the lazy cache instead of certifying
    the FFT. Generation then makes no ``seed_fourier`` call; the first
    binary64 point makes one, and every point result (values and the
    reported ``colour_contraction``) is identical to the eager slice's, as is
    the certified ``fourier`` and a ``dataclasses.replace`` copy."""

    import dataclasses

    from pyamplicol.loops.v2 import assemble
    from pyamplicol.models.loading import load_compiled_model
    from tests.unit.test_loop_v2_slice import PARAMETERS, _fixture, _request

    calls = []
    eager_fourier = assemble.seed_fourier

    def counted(table):
        calls.append(len(table.classes))
        return eager_fourier(table)

    monkeypatch.setattr(assemble, "seed_fourier", counted)
    monkeypatch.setattr(native_seeds, "seed_fourier", counted)

    class LazySlice(assemble.LoopSlice):
        _lazy: native_seeds.LazyFourier | None = None
        _stored = None

        @property  # type: ignore[override]
        def fourier(self):
            return self._lazy.get() if self._lazy is not None else self._stored

        @fourier.setter
        def fourier(self, value):
            self._stored = value

        def _plan_contraction(self) -> None:
            if self.request.colour_contraction == "auto" and (
                self.request.accuracy != "lc"
            ):
                self._lazy = native_seeds.LazyFourier(lambda: self.table)
                return
            super()._plan_contraction()

    compiled = load_compiled_model(loop_sm_compiled_model)
    request = _request("gg_gg")
    eager = assemble.LoopSlice.generate(compiled, request, loop_native)
    assert len(calls) == 1
    lazy = LazySlice.generate(compiled, request, loop_native)
    assert len(calls) == 1  # generation certifies nothing
    fixture = _fixture("gg_gg")
    for point, scale in zip(fixture["points"], fixture["scales_squared"], strict=True):
        mine = lazy.evaluate(point, scale, parameters=PARAMETERS)
        theirs = eager.evaluate(point, scale, parameters=PARAMETERS)
        assert mine.colour_contraction == theirs.colour_contraction
        assert mine.colour_contraction == "symmetric-group-fft"
        assert repr(mine) == repr(theirs)
    assert len(calls) == 2  # one certification, at the first point
    assert lazy.fourier == eager.fourier
    copy = dataclasses.replace(lazy, request=lazy.request)
    assert copy.fourier == eager.fourier


# ------------------------------------------------------------------ the slice


def _slice_request(process: str, born: dict, loop: dict, **options):
    from pyamplicol.loops.v2.assemble import SliceRequest

    return SliceRequest(
        process, born, loop, fixed_parameters=legacy_capture.FIXED_PARAMETERS, **options
    )


SLICE_CASES = [
    ("g g > g g g", {"QCD": 3, "QED": 0}, {"QCD": 5, "QED": 0}, "full"),
    ("g g > g g g", {"QCD": 3, "QED": 0}, {"QCD": 5, "QED": 0}, "lc"),
    ("u u~ > g g g", {"QCD": 3, "QED": 0}, {"QCD": 5, "QED": 0}, "nlc"),
]


def _colour_options(accuracy: str) -> dict:
    from pyamplicol.models.loop_color_declaration import LoopColorDeclaration

    if accuracy == "full":
        return {}
    return {
        "accuracy": accuracy,
        "colour_declaration": LoopColorDeclaration.from_dict(
            json.loads(legacy_capture.COLOUR_DECLARATION.read_text())
        ),
    }


@pytest.fixture
def seed_stages(monkeypatch):
    """Every native seed stage run (``NativeSeedStage``), in order."""

    stages: list = []
    run = native_seeds.NativeSeedStage._run

    def recorded(self, *args, **kwargs):
        stages.append(self)
        return run(self, *args, **kwargs)

    monkeypatch.setattr(native_seeds.NativeSeedStage, "_run", recorded)
    return stages


@pytest.mark.loop_physics
@pytest.mark.parametrize(("process", "born", "loop", "accuracy"), SLICE_CASES)
def test_native_seeds_in_the_slice(
    loop_sm_compiled_model, loop_native, seed_stages, process, born, loop, accuracy
):
    """``LoopSlice.generate`` with the switch on (the native
    ``build_seed_table`` and ``extend_seed_table``) has the Python run's S5
    digest at FC, LC and NLC (prune included), for the main slice and its
    poles companion (``scaleless``), and with ``colour_contraction="direct"``
    the same plan bytes (S8). B1: each slice's seed table read its schedule
    build in-crate and the prune rebuilt the schedules from the retained
    skeletons (one decode per build and rebuild, no skeleton-list decode);
    the native prune liveness equals ``_prune_colour``'s callback on every
    reading, and the schedules rebuilt from it equal the callback's."""

    from pyamplicol.loops.v2 import assemble, snapshot
    from pyamplicol.loops.v2.plan_export import export_plan
    from pyamplicol.models.loading import load_compiled_model

    compiled = load_compiled_model(loop_sm_compiled_model)
    request = _slice_request(
        process, born, loop, colour_contraction="direct", **_colour_options(accuracy)
    )
    # Native first on empty reduction caches (its cells read the natively
    # reduced rings), then the oracle, which recomputes its own.
    with native_seeds.python_oracle():
        pass
    native = assemble.LoopSlice.generate(compiled, request, loop_native)
    native_stages = list(seed_stages)
    with native_seeds.python_oracle():
        python = assemble.LoopSlice.generate(compiled, request, loop_native)
    assert seed_stages == native_stages  # the oracle ran no native stage
    assert snapshot.digest(snapshot.seeds_records(native)) == snapshot.digest(
        snapshot.seeds_records(python)
    )
    assert python.scaleless is not None and native.scaleless is not None
    assert snapshot.digest(snapshot.seeds_records(native.scaleless)) == (
        snapshot.digest(snapshot.seeds_records(python.scaleless))
    )
    assert (
        export_plan(native, forward=False).json()
        == export_plan(python, forward=False).json()
    )
    # B1: the main slice seeded in-crate from its complete build; its poles
    # companion extends an empty table holding the slice's certificates
    # (``LoopSlice._companion_seeds``, generation-scale-2's lean companion),
    # also in-crate from its own complete build; at LC/NLC each extended
    # after a rebuild from the retained skeletons.
    built = [stage for stage in native_stages if stage.kind == "build"]
    assert len(built) == 1  # the main slice
    assert all(s.build_source is not None and s.build_source.complete for s in built)
    assert any(
        s.kind == "extend" and s.build_source is not None and s.build_source.complete
        for s in native_stages
    )  # the companion
    pruned = accuracy != "full"
    for slice_ in (native, native.scaleless):
        decodes = dict(slice_.generation.decodes)
        assert decodes == ({"build": 1, "rebuild": 1} if pruned else {"build": 1})
        build = slice_.generation.build_of_plan(slice_.plan)
        assert build is not None
        assert build.max_rank() == max(
            node.rank_in + max(segment.q_degree for segment, _ in node.edges)
            for schedule in slice_.plan.schedules.values()
            for node in schedule.trie.nodes
        )
    if not pruned:
        return
    del seed_stages[:]
    unpruned = assemble.LoopSlice.generate(compiled, request, loop_native, prune=False)
    stage = seed_stages[0]
    flags, keep = stage.keep(unpruned)
    assert any(flags.values())
    oracle = _oracle_keep(unpruned, assemble.LoopSlice._prune_colour)
    assert keep == [oracle(s) for s in stage._skeletons]
    # LC prunes skeletons of g g > g g g; NLC keeps every u u~ > g g g one.
    assert any(keep) and all(keep) == (accuracy == "nlc")
    frames = {
        key: schedule.trie.frame for key, schedule in unpruned.plan.schedules.items()
    }
    generation = unpruned.generation
    by_callback = generation.schedules(unpruned.colour, frames=frames, keep=oracle)
    by_liveness = generation.schedules(
        unpruned.colour, frames=frames, keep=stage.reading_keep(unpruned)
    )
    assert_same_schedules(by_liveness, by_callback)
    # The one-line switch offered to ``_prune_colour``'s owner.
    proposed = native_seeds.prune_keep(unpruned)
    assert proposed is not None
    assert proposed.flags == stage.reading_keep(unpruned).flags
    assert_same_schedules(
        generation.schedules(unpruned.colour, frames=frames, keep=proposed),
        by_callback,
    )
    assert_same_schedules(by_callback, native.plan)


class _Captured(Exception):
    pass


def _oracle_keep(slice_, prune, observables=()):
    """``_prune_colour``'s ``keep`` callback (captured where the pruned
    schedules are rebuilt, natively or in Python, before they are)."""

    from pyamplicol.loops.v2 import assemble

    seen = {}

    def native(colour, *, frames=None, keep=None, **kw):
        seen["keep"] = keep
        raise _Captured

    def python(families, catalogue, colour, *, policy=None, view=None, **kw):
        seen["keep"] = view.keep
        raise _Captured

    with pytest.MonkeyPatch.context() as patch:
        if slice_.generation is not None:
            patch.setattr(slice_.generation, "schedules", native)
        patch.setattr(assemble, "build_schedules", python)
        # The callback itself, also once ``_prune_colour`` passes the native
        # liveness (``prune_keep``) when there is one.
        patch.setattr(native_seeds, "prune_keep", lambda *a, **k: None)
        with pytest.raises(_Captured):
            prune(slice_, observables)
    return seen["keep"]


@pytest.mark.loop_physics
def test_native_prune_liveness_with_directed_colour_tables(
    loop_sm_compiled_model, loop_native, monkeypatch
):
    """The observables' LC prune (``ObservableSlice.of`` passes the directed
    colour tables of a dipole as extra criteria): native ``keep`` with those
    criteria equals ``_prune_colour``'s callback on every reading, and the
    schedules rebuilt from the native liveness (B1) equal the pruned plan."""

    from pyamplicol.color.correlator_matrices import ColorCorrelator
    from pyamplicol.correlators import CorrelatorConfig
    from pyamplicol.loops.v2 import assemble
    from pyamplicol.loops.v2.observables import ObservableSlice
    from pyamplicol.models.loading import load_compiled_model
    from pyamplicol.models.loop_color_declaration import LoopColorDeclaration

    stages: list = []

    def build(colour, skeletons, basis=None, *, compress=False):
        stage = native_seeds.NativeSeedStage(colour, loop_native)
        stages.append(stage)
        return stage.build(skeletons, basis, compress=compress)

    checked = []
    prune = assemble.LoopSlice._prune_colour

    def checking(self, observables=()):
        observables = list(observables)
        found = [s for s in stages if s.table is self.table]
        if not found:
            # The poles companion's table extends the slice's certificates
            # (``LoopSlice._companion_seeds``): no build stage, no observables.
            assert not observables
            return prune(self, observables)
        (stage,) = found
        _flags, keep = stage.keep(self, observables)
        oracle = _oracle_keep(self, prune, observables)
        checked.append(
            (bool(observables), keep == [oracle(s) for s in stage._skeletons])
        )
        assert any(keep) or not observables
        frames = {
            key: schedule.trie.frame for key, schedule in self.plan.schedules.items()
        }
        native = self.generation.schedules(
            self.colour, frames=frames, keep=stage.reading_keep(self, observables)
        )
        prune(self, observables)
        assert_same_schedules(native, self.plan)

    monkeypatch.setattr(assemble, "build_seed_table", build)
    monkeypatch.setattr(assemble.LoopSlice, "_prune_colour", checking)
    request = _slice_request(
        "g g > g g g",
        {"QCD": 3, "QED": 0},
        {"QCD": 5, "QED": 0},
        accuracy="lc",
        colour_declaration=LoopColorDeclaration.from_dict(
            json.loads(legacy_capture.COLOUR_DECLARATION.read_text())
        ),
    )
    ObservableSlice.generate(
        load_compiled_model(loop_sm_compiled_model),
        request,
        loop_native,
        correlators=CorrelatorConfig(
            color_correlations=(ColorCorrelator.dipole("T12", 1, 2),)
        ),
    )
    # The slice pruned natively for the dipole's directed tables (the poles
    # companion, on the slice's certificates, by the Python callback).
    assert all(equal for _directed, equal in checked)
    assert any(directed for directed, _equal in checked)


# ------------------------------------------------------------------ six legs


@pytest.mark.loop_nightly
@pytest.mark.parametrize(
    ("process", "born", "loop"),
    [
        ("g g > g g g g", {"QCD": 4, "QED": 0}, {"QCD": 6, "QED": 0}),
        ("u u~ > g g g g", {"QCD": 4, "QED": 0}, {"QCD": 6, "QED": 0}),
    ],
)
def test_native_seeds_at_six_legs(
    loop_sm_compiled_model, loop_native, monkeypatch, process, born, loop
):
    """Six legs, FC: the slice's seed table built natively equals the
    Python one (S5 of the readings)."""

    from pyamplicol.loops.v2 import assemble, snapshot
    from pyamplicol.models.loading import load_compiled_model

    compiled = load_compiled_model(loop_sm_compiled_model)
    request = _slice_request(process, born, loop, counterterms="sites")
    captured = {}

    def build(colour, skeletons, basis=None, *, compress=False):
        skeletons = list(skeletons)
        with native_seeds.python_oracle():
            pass  # native first on empty reduction caches
        captured["native"] = native_seeds.build_seed_table(
            colour, skeletons, basis, compress=compress, native=loop_native
        )
        captured["python"] = python_build(colour, skeletons, basis, compress=compress)
        raise _Captured

    monkeypatch.setattr(assemble, "build_seed_table", build)
    with pytest.raises(_Captured):
        assemble.LoopSlice.generate(compiled, request, loop_native)
    assert_same_tables(captured["native"], captured["python"])

    class _Slice:
        table, class_of = captured["native"]

    class _Oracle:
        table, class_of = captured["python"]

    assert snapshot.digest(snapshot.seeds_records(_Slice)) == snapshot.digest(
        snapshot.seeds_records(_Oracle)
    )
