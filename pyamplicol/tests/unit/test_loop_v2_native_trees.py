# SPDX-License-Identifier: 0BSD
"""Native per-point trees and Born of the v2 executor (WP3.1 I3/I4 part).

``NativeExecutor(inputs="native")`` exports the slice's tree program (the
interaction program) and its Born root closure (``loops/v2/native_trees.py``)
and forms every point's block rows and Born components in ``rusticol-core``
(``loop_v2::trees``), without Python per point. Checks against the slice's
Python inputs (``inputs="python"``, the same exported plan):

* block rows and Born components per point in binary64;
* the bare virtual per sector and |A0|^2 per squared Born order, in binary64
  and in double-double, on processes with fixed parameters, massive external
  tops, and leading and next-to-leading colour (the Born metric per Born
  component);
* per-point renormalisation scales in one batch, and massive external
  spinors in double-double with a mass that is no binary64 number (the
  executor takes exact masses and rounds them to its carrier).
"""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import pytest

from pyamplicol.loops.v2.assemble import LoopSlice, Precision, SliceRequest
from pyamplicol.loops.v2.plan_export import NativeExecutor

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "fixtures" / "loop"
PARAMETERS = {"aS": "0.118", "MT": "173"}
FIXED = {"MB": "0"}
DOUBLE_DOUBLE = Precision.double_double()


def _points(source: str) -> tuple[list, list[str]]:
    data = json.loads((FIXTURES / source).read_text(), parse_float=str)
    points = data["points"]
    mu = data.get("mu_r", ["1000"] * len(points))
    if isinstance(mu, str):
        mu = [mu] * len(points)
    return points, [format(Decimal(str(m)) ** 2, "f") for m in mu]


# process, Born orders, accuracy, points
PROCESSES = {
    # QED = 0: without the bound the Born has a QCD 1 QED 2 order (photon, Z
    # and W exchange) that the root closure does not build, and generation
    # refuses it (``test_loop_v2_generate``); before that refusal its
    # interference sector (QCD 6, QED 2) was silently zero here.
    "uux_ddxg": (
        "u u~ > d d~ g",
        {"QCD": 3, "QED": 0},
        "full",
        "required/uux_ddxg.json",
    ),
    "gg_gg": ("g g > g g", {"QCD": 2, "QED": 0}, "full", "legacy/gg_gg.json"),
    "gg_gg_lc": ("g g > g g", {"QCD": 2, "QED": 0}, "lc", "legacy/gg_gg.json"),
    "uux_ddxg_nlc": (
        "u u~ > d d~ g",
        {"QCD": 3, "QED": 0},
        "nlc",
        "required/uux_ddxg.json",
    ),
    "gg_ttx": ("g g > t t~", {"QCD": 2, "QED": 0}, "full", "required/gg_ttx.json"),
}


@pytest.fixture(scope="module")
def executors(loop_sm_compiled_model, loop_native):
    from pyamplicol.models.loading import load_compiled_model
    from pyamplicol.models.loop_color_declaration import LoopColorDeclaration
    from tools.loop_reference import legacy_capture

    compiled = load_compiled_model(loop_sm_compiled_model)
    declaration = LoopColorDeclaration.from_dict(
        json.loads(legacy_capture.COLOUR_DECLARATION.read_text())
    )
    built: dict[str, tuple[NativeExecutor, NativeExecutor]] = {}

    def build(name: str) -> tuple[NativeExecutor, NativeExecutor]:
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
            slice_ = LoopSlice.generate(compiled, request, loop_native)
            native = NativeExecutor(slice_, forward=False, inputs="native")
            python = NativeExecutor(slice_, forward=False, exported=native.exported)
            built[name] = native, python
        return built[name]

    return build


def _deviation(values, reference) -> float:
    """Largest deviation of two complex vectors on the reference's scale."""

    pairs = [(complex(*a), complex(*b)) for a, b in zip(values, reference, strict=True)]
    scale = max((abs(b) for _a, b in pairs), default=0.0) or 1.0
    return max((abs(a - b) for a, b in pairs), default=0.0) / scale


@pytest.mark.loop_physics
@pytest.mark.parametrize("name", ["uux_ddxg", "gg_ttx"])
def test_native_rows_and_born_components_equal_the_python_inputs(executors, name):
    """Block rows and the plan's Born components of every point."""

    native, python = executors(name)
    points, scales = _points(PROCESSES[name][3])
    rows, born, _amplitudes, _squared = native.native_inputs(points, scales, PARAMETERS)
    _precision, evaluations = python._points(points, scales, PARAMETERS, None, True)
    reference = python._inputs(evaluations, "double")
    for k in range(len(points)):
        assert len(rows[k]) == len(reference["block_rows"][k])
        assert _deviation(rows[k], reference["block_rows"][k]) <= 1e-14
        assert _deviation(born[k], reference["born"][k]) <= 1e-13


def _assert_same(native, python, tolerance) -> None:
    for a, b in zip(native, python, strict=True):
        for sector, value in b.components["bare"].items():
            mine = a.components["bare"][sector].coefficients
            scale = max(abs(complex(x)) for x in value.coefficients) or 1.0
            for x, y in zip(mine, value.coefficients, strict=True):
                assert abs(complex(x) - complex(y)) <= tolerance * scale, sector
        assert a.born.keys() == b.born.keys()
        for order, value in b.born.items():
            assert abs(float(a.born[order]) - float(value)) <= tolerance * abs(
                float(value)
            )
        assert "virtual" not in a.components


@pytest.mark.loop_physics
@pytest.mark.parametrize(
    "name", ["uux_ddxg", "gg_gg", "gg_ttx", "gg_gg_lc", "uux_ddxg_nlc"]
)
def test_native_bare_and_born_equal_the_python_inputs(executors, name):
    """The bare virtual of every sector and |A0|^2 (at the request's colour
    accuracy) in binary64."""

    native, python = executors(name)
    points, scales = _points(PROCESSES[name][3])
    mine = native.evaluate(points, scales, parameters=PARAMETERS)
    theirs = python.evaluate(points, scales, parameters=PARAMETERS)
    _assert_same(mine, theirs, 1e-12)


@pytest.mark.loop_physics
@pytest.mark.parametrize("name", ["uux_ddxg", "gg_ttx"])
def test_native_double_double_equals_the_python_inputs(executors, name):
    """Double-double: native rows in DoubleFloat against Decimal inputs."""

    native, python = executors(name)
    points, scales = _points(PROCESSES[name][3])
    points, scales = points[:1], scales[:1]
    mine = native.evaluate(
        points, scales, parameters=PARAMETERS, precision=DOUBLE_DOUBLE
    )
    theirs = python.evaluate(
        points, scales, parameters=PARAMETERS, precision=DOUBLE_DOUBLE
    )
    _assert_same(mine, theirs, 1e-26)


@pytest.mark.loop_physics
def test_native_scales_may_change_per_point(executors):
    """One batch at two scales equals the two single-scale batches."""

    native, _python = executors("uux_ddxg")
    points, _scales = _points(PROCESSES["uux_ddxg"][3])
    point = points[0]
    both = native.evaluate([point, point], ["490000", "1960000"], parameters=PARAMETERS)
    for result, scale in zip(both, ["490000", "1960000"], strict=True):
        (alone,) = native.evaluate([point], [scale], parameters=PARAMETERS)
        for sector, value in alone.components["bare"].items():
            assert result.components["bare"][sector].coefficients == value.coefficients
    assert both[0].components != both[1].components


@pytest.mark.loop_physics
def test_native_massive_sources_carry_the_carrier_precision(executors):
    """MT = 173.1 is no binary64 number: in double-double the native massive
    spinors take the mass in the carrier and equal the Decimal inputs."""

    from fractions import Fraction

    from tools.loop_reference.exact_points import generate

    native, python = executors("gg_ttx")
    mass = Fraction(1731, 10)
    (point,) = generate([0, 0, mass, mass], count=1)
    points = [[[str(x) for x in p] for p in point]]
    parameters = {**PARAMETERS, "MT": "173.1"}
    mine = native.evaluate(
        points, ["1000000"], parameters=parameters, precision=DOUBLE_DOUBLE
    )
    theirs = python.evaluate(
        points, ["1000000"], parameters=parameters, precision=DOUBLE_DOUBLE
    )
    _assert_same(mine, theirs, 1e-26)
