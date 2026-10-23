# SPDX-License-Identifier: 0BSD
"""Native seed combinations and classes equal the Python seed tables (WP3.1
I10, package native-seeds B2/B3).

``loops.v2.colour.build_seed_table`` / ``extend_seed_table`` are the
oracles: the native stages (``loops.v2.native_seeds``) give equal tables
(classes in the same order, orders, cells, certificates) and an equal
``class_of`` (same skeleton order, same rows), with and without certified
compressions, for builds and extensions; the native prune liveness equals
``LoopSlice._prune_colour``'s; the binding needs no Symbolica key and its
output does not depend on the thread count.

Default tier: three 4-5-leg controls (adjoint rings and fundamental traces,
quark rings, two quark lines with generic seeds). ``loop_physics``: the
slice (readings, LC/NLC prune, S5 digests). ``loop_nightly``: six legs.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from pyamplicol.loops.v2 import native_seeds
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
from tools.loop_reference import legacy_capture

ROOT = Path(__file__).resolve().parents[2]


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
        return built[name]

    return build


CONTROLS = ["gg_gg", "uux_ggg", "uux_ddxg"]


@pytest.mark.loop_fast
@pytest.mark.parametrize("compress", [True, False])
@pytest.mark.parametrize("name", CONTROLS)
def test_native_seed_tables_equal_python(controls, loop_native, name, compress):
    """Build on two thirds of the skeletons, then extend with an overlapping
    later part (new skeletons and skeletons already seeded): tables and
    ``class_of`` equal the Python stages', and so do the skeletons' orders."""

    colour, skeletons = controls(name)
    first = skeletons[: (2 * len(skeletons)) // 3]
    python = build_seed_table(colour, first, compress=compress)
    stage = native_seeds.NativeSeedStage(colour, loop_native)
    native = stage.build(first, compress=compress)
    assert_same_tables(native, python)
    orders = stage._builder.skeleton_orders()
    assert [stage._orders[o] for o in orders] == [s.orders for s in first]
    assert bool(native[0].certificates) == compress
    rest = skeletons[len(first) // 2 :]
    assert_same_tables(
        native_seeds.extend_seed_table(*native, colour, rest, compress=compress),
        extend_seed_table(*python, colour, rest, compress=compress),
    )


@pytest.mark.loop_fast
def test_native_extension_of_an_uncompressed_table(controls, loop_native):
    """Extending a compressed table without compression (existing quark
    rings keyed by themselves, generic classes by their tensors) and an
    uncompressed one with it (no certificate: generic seeds only)."""

    colour, skeletons = controls("uux_ddxg")
    half = len(skeletons) // 2
    for built, extended in ((True, False), (False, True)):
        table = build_seed_table(colour, skeletons[:half], compress=built)
        assert_same_tables(
            native_seeds.extend_seed_table(
                *table, colour, skeletons, compress=extended, native=loop_native
            ),
            extend_seed_table(*table, colour, skeletons, compress=extended),
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


def _no_key_run(threads: str | None) -> list:
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in ("SYMBOLICA_LICENSE", "SYMBOLICA_MASTER_LICENSE")
    }
    env["SYMBOLICA_HIDE_BANNER"] = "1"
    if threads is not None:
        env["RAYON_NUM_THREADS"] = threads
    result = subprocess.run(
        [sys.executable, "-c", _NO_KEY_SCRIPT],
        capture_output=True,
        text=True,
        env=env,
        cwd=ROOT,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stderr[-4000:]
    return json.loads(result.stdout.strip().splitlines()[-1])


@pytest.mark.loop_fast
def test_native_seeds_need_no_key_and_no_thread_count(loop_native):
    """The binding makes no Symbolica licence call (a process without a key
    runs every step) and its output is the same with one and many threads:
    adjoint rings 1234, 1324, 4321 (the reflection of 1234, sign +1) and a
    position with two decomposition terms on the word 34."""

    one, many = _no_key_run("1"), _no_key_run("16")
    assert one == many
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


@pytest.mark.loop_physics
@pytest.mark.parametrize(("process", "born", "loop", "accuracy"), SLICE_CASES)
def test_native_seeds_in_the_slice(
    loop_sm_compiled_model, loop_native, monkeypatch, process, born, loop, accuracy
):
    """``LoopSlice.generate`` with the native stages in place of
    ``build_seed_table`` and ``extend_seed_table`` (the phase-1 switch) has
    the Python run's S5 digest, at FC, LC and NLC (prune included); the
    native prune liveness equals ``_prune_colour``'s on every reading."""

    from pyamplicol.loops.v2 import assemble, snapshot
    from pyamplicol.models.loading import load_compiled_model
    from pyamplicol.models.loop_color_declaration import LoopColorDeclaration

    compiled = load_compiled_model(loop_sm_compiled_model)
    options = {}
    if accuracy != "full":
        options = {
            "accuracy": accuracy,
            "colour_declaration": LoopColorDeclaration.from_dict(
                json.loads(legacy_capture.COLOUR_DECLARATION.read_text())
            ),
        }
    request = _slice_request(process, born, loop, **options)
    python = assemble.LoopSlice.generate(compiled, request, loop_native)
    stages: list = []

    def build(colour, skeletons, basis=None, *, compress=False):
        stage = native_seeds.NativeSeedStage(colour, loop_native)
        stages.append(stage)
        return stage.build(skeletons, basis, compress=compress)

    def extend(table, class_of, colour, skeletons, *, compress=False):
        stage = native_seeds.NativeSeedStage(colour, loop_native)
        stages.append(stage)
        return stage.extend(table, class_of, skeletons, compress=compress)

    with monkeypatch.context() as patch:
        patch.setattr(assemble, "build_seed_table", build)
        patch.setattr(assemble, "extend_seed_table", extend)
        native = assemble.LoopSlice.generate(
            compiled, request, loop_native, prune=False
        )
        built = stages[0]
        if accuracy != "full":
            flags, keep = built.keep(native)
            assert any(flags.values())
            # The oracle's keep: _prune_colour's callback on the readings.
            seen = {}

            def capture(colour, *, frames=None, keep=None, **kw):
                seen["keep"] = keep
                raise _Captured

            patch.setattr(native.generation, "schedules", capture, raising=False)
            with pytest.raises(_Captured):
                native._prune_colour()
            assert keep == [seen["keep"](s) for s in built._skeletons]
            # LC prunes skeletons of g g > g g g; NLC keeps every u u~ > g g g one.
            assert any(keep) and all(keep) == (accuracy == "nlc")
    with monkeypatch.context() as patch:
        patch.setattr(assemble, "build_seed_table", build)
        patch.setattr(assemble, "extend_seed_table", extend)
        native = assemble.LoopSlice.generate(compiled, request, loop_native)
    assert snapshot.digest(snapshot.seeds_records(native)) == snapshot.digest(
        snapshot.seeds_records(python)
    )
    assert len(stages) == 2 + (accuracy != "full")


class _Captured(Exception):
    pass


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
        captured["python"] = build_seed_table(
            colour, skeletons, basis, compress=compress
        )
        captured["native"] = native_seeds.build_seed_table(
            colour, skeletons, basis, compress=compress, native=loop_native
        )
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
