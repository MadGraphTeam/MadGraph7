# SPDX-License-Identifier: 0BSD
"""The binary tree program (M-FINAL MF-8G-1, ``native_trees.NATIVE_SECTIONS``).

The tree program's large arrays are binary sections of a container
(rusticol-core ``loop_v2::trees::binary``) instead of Python lists in one
JSON text: the counterterm program's steps come from its native DAG as
columns and are pruned natively, its overlap cells are formed natively as a
table (``NativeCellTable.section``), and the trees' steps and the closure
terms move to sections after their pruning; the trees' steps come from the native blocks DAG too
(A5's tree part: no Python ``TreeRows``). The JSON program (``NATIVE_SECTIONS
= False``: Python ``TreeRows`` and lists) is the oracle:

* decoded (``NativeTrees.full_program``), the binary program is the
  oracle's JSON program byte for byte, with the same saved kernels, plane
  applications and layout fields; the program Python holds has none of the
  sectioned arrays;
* the native executor reads either form into bitwise equal native inputs
  and evaluations, with and without poles;
* the native twins of the pruning (``tree_steps_prune``,
  ``tree_cells_live_ket``) and of the cells are covered by the decoded
  equality of the pruned programs (the oracle prunes in Python);
* a stored process holds ``tree-program.bin``; Python reads only its head
  at load, and the loaded process evaluates bitwise as generated.

Default tier: g g > g g at leading colour and u u~ > d d~ g at full colour
(both with the counterterm program); nightly: g g > g g g g at leading and
full colour (the 7-leg check is the package's recorded run).
"""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import pytest

from pyamplicol.loops.v2.assemble import LoopSlice, SliceRequest
from pyamplicol.loops.v2.plan_export import NativeExecutor

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "fixtures" / "loop"
PARAMETERS = {"aS": "0.118", "MT": "173"}
FIXED = {"MB": "0"}

PROCESSES = {
    "gg_gg_lc": ("g g > g g", {"QCD": 2, "QED": 0}, "lc", "legacy/gg_gg.json"),
    "uux_ddxg": ("u u~ > d d~ g", {"QCD": 3, "QED": 0}, "full", "required/uux_ddxg.json"),
    "gg_gggg_lc": ("g g > g g g g", {"QCD": 4, "QED": 0}, "lc", "required/gg_gggg.json"),
    "gg_gggg": ("g g > g g g g", {"QCD": 4, "QED": 0}, "full", "required/gg_gggg.json"),
}


def _points(source: str, count: int) -> tuple[list, list[str]]:
    data = json.loads((FIXTURES / source).read_text(), parse_float=str)
    points = data["points"][:count]
    mu = data.get("mu_r", ["1000"] * len(points))
    if isinstance(mu, str):
        mu = [mu] * len(points)
    return points, [format(Decimal(str(m)) ** 2, "f") for m in mu[:count]]


@pytest.fixture(scope="module")
def slices(loop_sm_compiled_model, loop_native):
    from pyamplicol.models.loading import load_compiled_model
    from pyamplicol.models.loop_color_declaration import LoopColorDeclaration
    from tools.loop_reference import legacy_capture

    compiled = load_compiled_model(loop_sm_compiled_model)
    declaration = LoopColorDeclaration.from_dict(
        json.loads(legacy_capture.COLOUR_DECLARATION.read_text())
    )
    built: dict[str, LoopSlice] = {}

    def build(name: str) -> LoopSlice:
        if name not in built:
            process, born, accuracy, _source = PROCESSES[name]
            loop = {k: v + (2 if k == "QCD" else 0) for k, v in born.items()}
            request = SliceRequest(
                process,
                born,
                loop,
                accuracy=accuracy,
                fixed_parameters=FIXED,
                colour_declaration=None if accuracy == "full" else declaration,
            )
            built[name] = LoopSlice.generate(compiled, request, loop_native)
        return built[name]

    return build


def _executors(slice_, monkeypatch) -> tuple[NativeExecutor, NativeExecutor]:
    """(binary, JSON oracle) native executors of one slice. The binary export
    builds no Python ``TreeRows`` (A5's tree part: the trees' program from
    the native blocks DAG); the oracle's does."""

    from pyamplicol.loops.v2 import native_trees

    from pyamplicol.loops.v2 import dag_input

    monkeypatch.setattr(native_trees, "NATIVE_SECTIONS", True)
    slice_._cache.pop("tree-rows", None)
    rebuilds = dag_input.rebuild_count()
    binary = NativeExecutor(slice_, forward=False, inputs="native")
    assert "tree-rows" not in slice_._cache
    # The counterterm program was emitted at its build (item 4): only the
    # loop-block DAG's released columns are rebuilt for the trees' program.
    assert dag_input.rebuild_count() == rebuilds + 1
    assert slice_.counterterm_program.blocks.blocks.blocks_dag.released
    monkeypatch.setattr(native_trees, "NATIVE_SECTIONS", False)
    oracle = NativeExecutor(slice_, forward=False, inputs="native")
    assert "tree-rows" in slice_._cache
    monkeypatch.setattr(native_trees, "NATIVE_SECTIONS", True)
    return binary, oracle


def _check(name, slices, monkeypatch, count) -> None:
    from pyamplicol.loops.native import load_loop_native
    from pyamplicol.loops.v2 import native_trees
    from pyamplicol.loops.v2.native_process import NativeProcess

    slice_ = slices(name)
    binary, oracle = _executors(slice_, monkeypatch)
    trees, reference = binary.exported.trees, oracle.exported.trees
    assert trees.is_binary and not reference.is_binary
    names = dict(trees.sections)
    assert set(names) == {
        native_trees.TREE_STEPS,
        native_trees.COUNTERTERM_STEPS,
        native_trees.CELLS,
        native_trees.BORN_TERMS,
        native_trees.COUNTERTERM_TERMS,
    }
    # The sectioned arrays are never Python lists in the program.
    program = trees.program
    assert "steps" not in program["interactions"]
    assert "steps" not in program["counterterm_trees"]["interactions"]
    assert "cells" not in program["counterterms"]
    assert "terms" not in program["counterterms"] and "terms" not in program["born"]
    # Decoded, the program is the oracle's byte for byte.
    assert trees.json() == reference.json()
    assert binary.exported.json() == oracle.exported.json()
    for field in (
        "kernels",
        "parameters",
        "planes",
        "couplings",
        "born_orders",
        "squared_orders",
        "masses",
        "components",
        "counterterm_outputs",
    ):
        assert getattr(trees, field) == getattr(reference, field), field
    # The executors read either form into bitwise equal values.
    points, scales = _points(PROCESSES[name][3], count)
    assert repr(binary.native_inputs(points, scales, PARAMETERS)) == repr(
        oracle.native_inputs(points, scales, PARAMETERS)
    )
    for poles in ("none", "all"):
        assert repr(
            binary.evaluate(points, scales, parameters=PARAMETERS, poles=poles)
        ) == repr(oracle.evaluate(points, scales, parameters=PARAMETERS, poles=poles))
    # Stored: the binary member; Python reads its head only at load.
    process = NativeProcess.of(binary)
    members = process.members()
    assert "tree-program.bin" in members and "tree-program.json" not in members
    loaded = NativeProcess.from_members(binary._plan_json, members)
    assert loaded.trees.container == members["tree-program.bin"]
    assert set(loaded.trees.program) <= {
        "schema",
        "legs",
        "initial",
        "state_size",
        "model_offset",
        "model_count",
        "momentum_slots",
        "source_fill",
        "parameters",
    }
    assert loaded.trees.json() == reference.json()
    executor = loaded.executor(load_loop_native())
    for poles in ("none", "all"):
        mine = executor.evaluate(points, scales, parameters=PARAMETERS, poles=poles)
        theirs = binary.evaluate(points, scales, parameters=PARAMETERS, poles=poles)
        # Lane labels differ by design (a loaded process labels lanes by
        # family index): the values are compared.
        for a, b in zip(mine, theirs, strict=True):
            assert (a.born, a.components, a.totals) == (b.born, b.components, b.totals)


@pytest.mark.loop_fast
@pytest.mark.parametrize("name", ["gg_gg_lc", "uux_ddxg"])
def test_binary_tree_program_equals_the_json_oracle(slices, monkeypatch, name):
    """g g > g g (LC: the native pruning drops ket components and counterterm
    currents) and u u~ > d d~ g (FC: quark lines, gram and literal cells)."""

    _check(name, slices, monkeypatch, 2)


@pytest.mark.loop_nightly
@pytest.mark.parametrize("name", ["gg_gggg_lc", "gg_gggg"])
def test_six_gluon_binary_tree_program_equals_the_json_oracle(
    slices, monkeypatch, name
):
    """g g > g g g g at leading and full colour (one point)."""

    _check(name, slices, monkeypatch, 1)


@pytest.mark.loop_fast
def test_container_and_sections_refuse_damage(loop_native):
    """The container's reader refuses a repeated or unknown section, a
    truncated payload and trailing bytes; a step section round-trips the
    oracle's JSON steps (weights interned in first use) and its pruning
    keeps what a reader marks."""

    native = loop_native
    steps = [
        {
            "interactions": [
                [0, 0, 1, 1, 1, 4, 8, 0, "1/2", "0"],
                [0, 1, 1, 0, 1, 8, 4, 0, "-1", "0"],
            ],
            "accumulators": [1],
            "outputs": [[0, -1, 12, 2, 1]],
        },
        {
            "interactions": [[0, 2, 1, 0, 1, 12, 4, 0, "1/2", "0"]],
            "accumulators": [1],
            "outputs": [[0, 0, 16, 3, 1]],
        },
    ]
    text = json.dumps(steps, separators=(",", ":"))
    section = bytes(native.tree_steps_from_json(text.encode()))
    assert native.tree_steps_json(section) == text
    assert native.tree_steps_counts(section) == (3, 2, 2)
    # Read only state component 2 (the first step's output): the second
    # step and its propagator go, the vertex kernel stays.
    live = bytearray(24)
    live[2] = 1
    pruned, kernels, propagators = native.tree_steps_prune(section, live, 1, 1)
    assert (kernels, propagators) == ([0], [])
    assert json.loads(native.tree_steps_json(pruned)) == [
        steps[0],
        {"interactions": [], "accumulators": [], "outputs": []},
    ]
    good = bytes(native.tree_program_container([("head", 0, b"{}")]))
    assert native.tree_program_index(good) == [("head", 0)]
    for damaged in (
        bytes(native.tree_program_container([("head", 0, b"{}"), ("head", 0, b"{}")])),
        good[:-1],
        good + b"\0",
    ):
        with pytest.raises(ValueError):
            native.tree_program_index(damaged)


def _metric_oracle(slice_, components, squared_orders):
    """``native_trees._component_metric`` before its cell cache (MF-8G-1):
    every pair of Born components projected and formatted on its own."""

    from fractions import Fraction

    from pyamplicol.color.contraction_types import NC
    from pyamplicol.loops.v2.colour import kept
    from pyamplicol.loops.v2.families import add_orders
    from pyamplicol.loops.v2.native_trees import _fraction_text, _zero_colour

    accuracy = slice_.request.accuracy
    degrees = slice_.degrees()["born"] if accuracy != "full" else {}
    index = {order: s for s, order in enumerate(squared_orders)}
    gram = slice_.basis.gram
    out = []
    for (b, left, p), j in components.items():
        for (b2, right, q), j2 in components.items():
            squared = add_orders(left, right)
            degree = degrees.get(squared)
            value = _zero_colour()
            for power, cell in gram[b][b2].items():
                exponent = power + p + q
                if kept(exponent, degree, accuracy):
                    value = value + cell.scaled(Fraction(NC) ** exponent)
            if value:
                out.append(
                    [index[squared], j, j2, _fraction_text(value.real), _fraction_text(value.imag)]
                )
    return out


@pytest.mark.loop_fast
@pytest.mark.parametrize("name", ["gg_gg_lc", "uux_ddxg"])
def test_component_metric_cache_equals_the_pairwise_oracle(slices, name):
    """The Born metric's cell cache (``_component_metric``) gives the
    pairwise loop's entries, in order."""

    from pyamplicol.loops.v2.native_trees import _component_metric

    slice_ = slices(name)
    trees = NativeExecutor(slice_, forward=False, inputs="native").exported.trees
    components = {key: j for j, key in enumerate(trees.components)}
    expected = _metric_oracle(slice_, components, trees.squared_orders)
    assert _component_metric(slice_, components, trees.squared_orders) == expected
    assert trees.program["born"]["metric"] == expected


@pytest.mark.loop_fast
@pytest.mark.parametrize("name", ["gg_gg_lc", "uux_ddxg"])
def test_chunked_cell_section_is_the_one_table_section(slices, monkeypatch, name):
    """The counterterm overlap cells formed in chunks of ket tensors, each on
    a fresh engine, and joined (``_join_cell_sections``) are the one-table
    section byte for byte."""

    from pyamplicol.loops.v2 import native_trees

    slice_ = slices(name)
    literal = NativeExecutor(slice_, forward=False, inputs="native").exported.trees.ket_tensors
    monkeypatch.setattr(native_trees, "_CELL_CHUNK", 10**9)
    whole = native_trees._cell_section(slice_, literal)
    for chunk in (1, 3, 7):
        monkeypatch.setattr(native_trees, "_CELL_CHUNK", chunk)
        assert native_trees._cell_section(slice_, literal) == whole, chunk
