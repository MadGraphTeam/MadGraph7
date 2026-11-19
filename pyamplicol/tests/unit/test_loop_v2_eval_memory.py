# SPDX-License-Identifier: 0BSD
"""Bounded evaluation memory of the native v2 executor (M-FINAL MF-8G-2).

With native trees ``V2Executor.evaluate`` streams a batch in parts of whole
lane chunks (``part_points``: each part's tree fill, then its dressing and
reduction), fills the trees ``fill_points`` points at a time
(``TreeProgram::inputs_chunked``) and holds at most ``input_budget`` bytes of
chunk inputs per lane chunk (a live input the current bucket chunk does not
read is dropped and rebuilt when read again). Auxiliary carriers and lane
counts share the loaded plan, tree program and seed FFT of an executor built
already (``V2Executor.derive``). Every bound changes memory, not arithmetic:
each output is bitwise that of the unbounded run (defaults: the budgets of
the native executor, which a small process never reaches).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pyamplicol.loops.v2.assemble import LoopSlice, Precision, SliceRequest
from pyamplicol.loops.v2.plan_export import NativeExecutor

ROOT = Path(__file__).resolve().parents[2]
PARAMETERS = {"aS": "0.118", "MT": "173"}
FIXED = {"MB": "0"}
DROPPED = "chunk inputs dropped early"

# process, Born orders, accuracy, G1 stream or required fixture
PROCESSES = {
    "gg_gg_lc": ("g g > g g", {"QCD": 2, "QED": 0}, "lc", "gg_gg"),
    "uux_ddxg": ("u u~ > d d~ g", {"QCD": 3, "QED": 0}, "full", "uux_ddxg"),
    "gg_ttx": ("g g > t t~", {"QCD": 2, "QED": 0}, "full", "required/gg_ttx.json"),
}
# Every bound alone and all of them at their tightest.
BOUNDS = (
    {"part_points": 1},
    {"part_points": 5},
    {"fill_points": 1},
    {"fill_points": 3},
    {"input_budget": 0},
    {"input_budget": 1 << 20},
    {"part_points": 1, "fill_points": 1, "input_budget": 0},
)


@pytest.fixture(scope="module")
def executors(loop_sm_compiled_model, loop_native):
    from pyamplicol.models.loading import load_compiled_model
    from pyamplicol.models.loop_color_declaration import LoopColorDeclaration
    from tools.loop_reference import legacy_capture

    compiled = load_compiled_model(loop_sm_compiled_model)
    declaration = LoopColorDeclaration.from_dict(
        json.loads(legacy_capture.COLOUR_DECLARATION.read_text())
    )
    built: dict[str, NativeExecutor] = {}

    def build(name: str) -> NativeExecutor:
        if name not in built:
            process, born, accuracy, _stream = PROCESSES[name]
            loop = {k: v + (2 if k == "QCD" else 0) for k, v in born.items()}
            request = SliceRequest(
                process,
                born,
                loop,
                accuracy=accuracy,
                fixed_parameters=FIXED,
                colour_declaration=None if accuracy == "full" else declaration,
            )
            slice_ = LoopSlice.generate(compiled, request, loop_native)
            built[name] = NativeExecutor(slice_, forward=False, inputs="native")
        return built[name]

    return build


def _batch(name: str, count: int = 11):
    """``count`` points at three scales: distinct points of the G1 stream,
    or the required fixture's points in turn."""

    from tools.loop_benchmark import streams

    source = PROCESSES[name][3]
    if source.endswith(".json"):
        fixture = json.loads(
            (ROOT / "tests/fixtures/loop" / source).read_text(), parse_float=str
        )["points"]
        points = [fixture[k % len(fixture)] for k in range(count)]
    else:
        frozen = streams.load(source)
        points = [streams.exact(frozen, k) for k in range(count)]
    scales = [("250000", "1000000", "4000000")[k % 3] for k in range(count)]
    return points, scales


def _inputs(native: NativeExecutor, points, scales, precision, wards=None):
    arithmetic, _digits = native._arithmetic(precision, False)
    evaluations = native._native_points(points, scales, PARAMETERS, precision)
    if wards is not None:
        from pyamplicol.loops.v2.assemble import _fraction
        from pyamplicol.loops.v2.ward import loop_vector, ward_leg

        for evaluation, point, ward in zip(evaluations, points, wards, strict=True):
            leg = ward_leg(ward)
            momentum = [_fraction(x) for x in point[leg - 1]]
            evaluation.loop_sources = {leg: loop_vector(ward, momentum)}
    return native._inputs(evaluations, arithmetic)


# Report counts that follow the bucket chunking (inputs read per chunk).
CHUNKED = {DROPPED, "factors", "vertex tensors"}


def _values(output, counts: bool = True) -> str:
    """Every output but the report (wall times), and with ``counts`` the
    report's work counts (steps, lanes, masters, points; not the per-chunk
    input reads, which a chunk-input budget splits)."""

    values = {k: v for k, v in output.items() if k != "report"}
    if not counts:
        return repr(values)
    work = {k: v for k, v in output["report"]["counts"].items() if k not in CHUNKED}
    return repr((values, work))


@pytest.mark.loop_physics
@pytest.mark.parametrize("name", tuple(PROCESSES))
@pytest.mark.parametrize("poles", ["all", "none", "split"])
def test_memory_bounds_are_bitwise_the_unbounded_run(executors, name, poles):
    """Eleven points (lanes of 16: one partial chunk; parts of 1 and 5 points
    round up to one chunk of 16 lanes, so this streams per lane chunk
    only through ``fill_points`` and ``input_budget``) with the re-test pass,
    every bound and their combination: bitwise the default run."""

    native = executors(name)
    points, scales = _batch(name)
    inputs = _inputs(native, points, scales, Precision.double())
    executor = native.native("double", None, len(points))
    options = {"poles": poles, "check": True, "rescale": True}
    reference = _values(executor.evaluate(inputs, **options))
    for bounds in BOUNDS:
        assert _values(executor.evaluate(inputs, **options, **bounds)) == reference, (
            bounds
        )


@pytest.mark.loop_physics
@pytest.mark.parametrize("name", ["uux_ddxg", "gg_ttx"])
def test_streamed_parts_of_lane_chunks_are_bitwise_the_batch(executors, name):
    """Lanes of 2 (a derived double executor): eleven points stream in parts
    of one, two and three lane chunks, with one tree fill per point; bucket
    chunks of one family (``chunk_families=1``) and no chunk-input budget
    drop and rebuild live inputs (counted): bitwise the batch at once."""

    native = executors(name)
    points, scales = _batch(name)
    inputs = _inputs(native, points, scales, Precision.double())
    native.native("double", None, len(points))
    executor = native.native("double", None, len(points)).derive(
        "double", lanes=2, fourier=native.exported.fourier is not None
    )
    options = {"poles": "all", "chunk_families": 1}
    reference = executor.evaluate(inputs, **options)
    assert reference["report"]["counts"].get(DROPPED, 0) == 0
    # Bucket chunks of one family dress and reduce what the default ones do.
    default = executor.evaluate(inputs, poles="all")
    assert _values(default, counts=False) == _values(reference, counts=False)
    for part in (2, 4, 6):
        out = executor.evaluate(
            inputs, **options, part_points=part, fill_points=1, input_budget=0
        )
        assert out["report"]["counts"][DROPPED] > 0
        assert _values(out) == _values(reference), part


@pytest.mark.loop_physics
def test_double_double_bounds_and_derived_executors(executors):
    """Double-double: the executor derived from the binary64 one (shared
    plan, tree program and seed FFT) equals one built from the plan JSON,
    with and without the tightest bounds; an exact executor derives without
    trees or FFT, and a derived executor cannot share what its source lacks."""

    native = executors("uux_ddxg")
    points, scales = _batch("uux_ddxg", 3)
    precision = Precision.double_double()
    inputs = _inputs(native, points, scales, precision)
    double = native.native("double", None, len(points))
    derived = native.native("double-double", None, len(points))
    assert derived.has_trees and native._native[("double", None, 16)] is double
    fresh = native.slice.native.V2Executor(
        native._plan_json,
        "double-double",
        None,
        lanes=derived.lanes,
        forms=native.forms,
        trees=native.exported.trees.executor_argument(),
        fourier=native.exported.fourier,
    )
    reference = _values(fresh.evaluate(inputs, poles="all"))
    assert _values(derived.evaluate(inputs, poles="all")) == reference
    tight = {"part_points": 1, "fill_points": 1, "input_budget": 0}
    assert _values(derived.evaluate(inputs, poles="all", **tight)) == reference
    exact = native.native("exact", None, len(points))
    assert not exact.has_trees and exact.fourier is None
    with pytest.raises(ValueError, match="no tree program to share"):
        exact.derive("double")
    with pytest.raises(ValueError, match="seed"):
        double.derive("exact", trees=False, fourier=True)


@pytest.mark.loop_physics
def test_loop_side_sources_stream_bitwise(loop_sm_compiled_model, loop_native):
    """The asymmetric Ward mode's loop-side sources (``loop_sources``: a
    second tree run per point) under every bound: bitwise the default run."""

    from pyamplicol.loops.v2 import ward
    from pyamplicol.models.loading import load_compiled_model
    from tools.loop_reference import legacy_capture

    compiled = load_compiled_model(loop_sm_compiled_model)
    request = SliceRequest(
        "g g > g g",
        {"QCD": 2, "QED": 0},
        {"QCD": 4, "QED": 0},
        fixed_parameters=legacy_capture.FIXED_PARAMETERS,
    )
    native = NativeExecutor(
        ward.ward_slice(compiled, request, loop_native), inputs="native", forward=False
    )
    data = json.loads((ROOT / "tests/fixtures/loop/legacy/gg_gg.json").read_text())
    point = data["points"][0]
    scale = data["scales_squared"][0]
    # Gauge vectors c p_j / p_j^0 and one literal (non-gauge) vector.
    wards = [ward.gauge_ward(leg, (1, 0), point) for leg in (1, 2, 3, 4)]
    wards.append((3, (("0.1", "0.2"), ("0.8", "-0.3"), ("0.4", "1"), ("-0.6", "0.5"))))
    points, scales = [point] * len(wards), [scale] * len(wards)
    inputs = _inputs(native, points, scales, Precision.double(), wards)
    assert inputs.get("loop_sources")
    executor = native.native("double", None, len(points))
    reference = _values(executor.evaluate(inputs, poles="all"))
    for bounds in BOUNDS:
        assert _values(executor.evaluate(inputs, poles="all", **bounds)) == reference
