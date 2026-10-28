# SPDX-License-Identifier: 0BSD
"""Ward identities on the native per-point input path (M-QCD MQ-WARD;
REQUIREMENTS_ADDENDUM H; user ruling 2026-10-28 ~16:10, Q4).

One external gluon's polarisation vector is replaced by its momentum,
epsilon_j -> c p_j / p_j^0 with c = 1 and c = i, on the loop side of the
interference (``loops.v2.ward``): the native trees run the substituted leg
in their second, loop-side run (``rusticol-core`` ``loop_v2::trees::
TreeProgram::inputs_loop_side``), and the Python lane forms the same inputs
as the oracle. The criterion is ``ward.ward_row``: the Born and the virtual
of every sector and Laurent coefficient vanish to the carrier's rounding
against the physical evaluation of the same point (double-double: 1e-25 of
the physical scale; binary64: within ten times the evaluation's own
reduction-direction estimate, itself within 1e-9 of the scale; the Born
1e-12 / 1e-24).

* Default tier: g g > g g and u u~ > g g g (every gluon leg, both factors,
  binary64 and double-double) on native inputs; native inputs = Python
  inputs on the non-gauge control (a literal vector on the loop side, where
  nothing vanishes) in double-double, and the Python-input Ward evaluation
  meets the criterion; the refusals (transverse kernels, a stored process,
  a quark leg, a rotation estimate).
* Nightly: the ladder ``tools/loop_v2_ward.py`` campaigns of
  ``tools/loop_benchmark`` processes up to seven legs (g g > g g g, quark
  lines with every gluon leg in turn, massive tops, g g > g g g g,
  u u~ > g g g g g), stream point 0.
* Nightly, ADDENDUM H's seven-gluon check: g g > g g g g g, legs 1, 4 and 7
  (the MQ-WARD campaign: generation 168 s, binary64 72 s and double-double
  721 s on 12 threads, 27 GB).
"""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

import pytest

from pyamplicol.loops.v2 import ward
from pyamplicol.loops.v2.assemble import COMPONENTS, Precision, SliceRequest
from pyamplicol.loops.v2.plan_export import NativeExecutor
from pyamplicol.loops.v2.stability import Rotation, StabilityChecks
from tools.loop_reference import legacy_capture

pytestmark = pytest.mark.loop_physics

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "fixtures" / "loop"
PARAMETERS = {"aS": "0.118", "MT": "173"}
CARRIERS = {"double": Precision.double(), "double-double": Precision.double_double()}
# process -> (text, Born orders, loop orders, gluon legs)
DEFAULT = {
    "gg_gg": ("g g > g g", {"QCD": 2, "QED": 0}, {"QCD": 4, "QED": 0}, (1, 2, 3, 4)),
    "uux_ggg": ("u u~ > g g g", {"QCD": 3, "QED": 0}, {"QCD": 5, "QED": 0}, (3, 4, 5)),
}
# A literal complex four-vector (neither transverse nor light-like): the
# non-gauge control, where no part vanishes.
VECTOR = (("0.1", "0.2"), ("0.8", "-0.3"), ("0.4", "1"), ("-0.6", "0.5"))


def _point(name: str) -> tuple[list, str]:
    data = json.loads((FIXTURES / "legacy" / f"{name}.json").read_text())
    return data["points"][0], data["scales_squared"][0]


@pytest.fixture(scope="module")
def slices(loop_sm_compiled_model, loop_native):
    """The Ward slice (full tree kernels) of a default process, generated once;
    ``kernels="transverse"`` the ordinary one."""

    from pyamplicol.loops.v2.assemble import LoopSlice
    from pyamplicol.models.loading import load_compiled_model

    compiled = load_compiled_model(loop_sm_compiled_model)
    built: dict = {}

    def build(name: str, kernels: str = "full"):
        key = (name, kernels)
        if key not in built:
            text, born, loop, _legs = DEFAULT[name]
            request = SliceRequest(
                text, born, loop, fixed_parameters=legacy_capture.FIXED_PARAMETERS
            )
            generate = ward.ward_slice if kernels == "full" else LoopSlice.generate
            built[key] = generate(compiled, request, loop_native)
        return built[key]

    return build


def _evaluate(executor, point, scale, wards, carrier, checks=True):
    """The physical evaluation and one per Ward value, in one batch."""

    stability = ward.WARD_CHECKS if checks and carrier == "double" else None
    results = executor.evaluate(
        [point] * (len(wards) + 1),
        [scale] * (len(wards) + 1),
        parameters=PARAMETERS,
        precision=CARRIERS[carrier],
        ward=[None, *wards],
        stability=stability,
    )
    return results[0], results[1:]


# ------------------------------------------------------------ default tier


@pytest.mark.parametrize("name", tuple(DEFAULT))
def test_ward_identity_on_native_inputs(slices, name):
    """Every gluon leg, c = 1 and i, binary64 and double-double, native
    inputs: the Born and the virtual vanish (``ward.ward_row``). Measured at
    the first legacy point: the binary64 virtual at most 1.6e-15 (g g > g g)
    and 6.0e-13 (u u~ > g g g) of the physical scale, within 1.5 and 0.8
    times its estimate; double-double 1.8e-31 and 1.7e-28; the Born ratio
    below 3e-17 (binary64) and 3e-33 (double-double)."""

    slice_ = slices(name)
    legs = DEFAULT[name][3]
    assert ward.gauge_legs(slice_) == legs
    point, scale = _point(name)
    executor = NativeExecutor(slice_, inputs="native", forward=False)
    configurations = [(leg, factor) for leg in legs for factor in ward.WARD_FACTORS]
    wards = [ward.gauge_ward(leg, factor, point) for leg, factor in configurations]
    for carrier in CARRIERS:
        physical, results = _evaluate(executor, point, scale, wards, carrier)
        assert physical.loop_born is None
        for configuration, result in zip(configurations, results, strict=True):
            row = ward.ward_row(physical, result, carrier)
            assert row["passed"] is True, (carrier, configuration, row)


def test_native_loop_side_inputs_match_python(slices):
    """The native trees' loop-side run against the Python lane's, u u~ > g g g
    (every gluon leg) and g g > g g (an initial leg): with a literal vector on
    the loop side (the non-gauge control) no part vanishes (each above 1e-4
    of the scale) and native = Python inputs to 1e-25 of the scale per part,
    coefficient and sector in double-double (measured: bitwise); the
    Python-input Ward evaluation (c = 1, binary64, the first leg) meets the
    criterion too."""

    for name, legs in (("uux_ggg", (3, 4, 5)), ("gg_gg", (1,))):
        slice_ = slices(name)
        point, scale = _point(name)
        executors = {
            inputs: NativeExecutor(slice_, inputs=inputs, forward=False)
            for inputs in ("native", "python")
        }
        physical, _ = _evaluate(
            executors["native"], point, scale, [], "double-double", checks=False
        )
        size = ward.ward_scales(physical)
        born = abs(sum(complex(v) for v in physical.born.values()))
        controls = [(leg, VECTOR) for leg in legs]
        runs = {
            inputs: _evaluate(
                executor, point, scale, controls, "double-double", checks=False
            )[1]
            for inputs, executor in executors.items()
        }
        for native, python, control in zip(
            runs["native"], runs["python"], controls, strict=True
        ):
            for part in COMPONENTS:
                largest = ward.largest(native, part)
                if part in ("virtual", "r2", "uv_local", "uv_external_fields"):
                    # Every part reads the loop-side vector (smallest
                    # measured: R2, leg 5 of u u~ > g g g, 1.8e-3 of the
                    # scale).
                    ratio = max(x / y for x, y in zip(largest, size, strict=True))
                    assert ratio > 1e-4, (name, control, part)
                for sector, laurent in native.components[part].items():
                    other = python.components[part][sector]
                    for k, value in enumerate(laurent.coefficients):
                        deviation = abs(complex(value) - complex(other.coefficients[k]))
                        assert deviation <= 1e-25 * size[k], (name, control, part, k)
            loop_born = [
                sum(complex(v) for v in run.loop_born.values())
                for run in (native, python)
            ]
            assert abs(loop_born[0]) > 1e-3 * born
            assert abs(loop_born[0] - loop_born[1]) <= 1e-25 * born
        # The Python inputs of a Ward evaluation meet the criterion as well.
        physical, (result,) = _evaluate(
            executors["python"],
            point,
            scale,
            [ward.gauge_ward(legs[0], (1, 0), point)],
            "double",
        )
        row = ward.ward_row(physical, result, "double")
        assert row["passed"] is True, (name, row)


def test_ward_mode_refusals(slices, loop_native):
    """The Ward mode fails closed: a slice on the transverse kernels, a stored
    process (its record carries no kernel variant), a Ward value per point of
    the wrong count, a quark leg (no vector source; both input paths), and a
    criterion read from a rotation check."""

    from pyamplicol.loops.v2.native_process import NativeProcess

    point, scale = _point("uux_ggg")
    one = ward.gauge_ward(3, (1, 0), point)
    transverse = NativeExecutor(
        slices("uux_ggg", "transverse"), inputs="native", forward=False
    )
    assert slices("uux_ggg", "transverse").blocks.kernel_variant == "transverse"
    with pytest.raises(ValueError, match="full tree kernels"):
        transverse.evaluate([point], [scale], parameters=PARAMETERS, ward=one)
    executor = NativeExecutor(slices("uux_ggg"), inputs="native", forward=False)
    stored = NativeProcess.of(executor).executor(loop_native)
    with pytest.raises(ValueError, match="stored process"):
        stored.evaluate([point], [scale], parameters=PARAMETERS, ward=one)
    with pytest.raises(ValueError, match="one Ward value per point"):
        executor.evaluate([point], [scale], parameters=PARAMETERS, ward=[one, one])
    # A quark leg has no vector source slot in the native trees, and the
    # Python lane refuses a vector on a spinor source.
    with pytest.raises(Exception, match="no vector source slot"):
        executor.evaluate([point], [scale], parameters=PARAMETERS, ward=(1, (1, 0)))
    python = NativeExecutor(slices("uux_ggg"), inputs="python", forward=False)
    with pytest.raises(Exception, match="four-component vector source"):
        python.evaluate([point], [scale], parameters=PARAMETERS, ward=(1, (1, 0)))
    rotation = StabilityChecks(
        rotations=(Rotation.about("z", Fraction(3, 5), Fraction(4, 5)),),
        reduction_direction=True,
    )
    physical, (result,) = (
        executor.evaluate([point], [scale], parameters=PARAMETERS)[0],
        executor.evaluate(
            [point], [scale], parameters=PARAMETERS, ward=one, stability=rotation
        ),
    )
    with pytest.raises(ValueError, match="reduction-direction check alone"):
        ward.ward_row(physical, result, "double")


# ------------------------------------------------------------ nightly ladder

LADDER = (
    "gg_ggg",
    "uux_ddxg",
    "uux_uuxg",
    "uux_gggg",
    "gg_ttxgg",
    "gg_gggg",
    "uux_ggggg",
)


def _campaign(arguments: list[str], record: Path) -> dict:
    from tools.loop_v2_ward import main

    status = main([*arguments, "--json", str(record)])
    data = json.loads(record.read_text())
    assert status == 0 and data["passed"], json.dumps(data["points"], indent=1)
    return data


@pytest.mark.loop_nightly
@pytest.mark.parametrize("key", LADDER)
def test_ward_identity_ladder(key, tmp_path, loop_native, loop_sm_ufo):
    """``tools/loop_v2_ward.py`` on a ``tools/loop_benchmark`` process, stream
    point 0: every gluon leg, c = 1 and i, binary64 and double-double, native
    inputs. Quark lines (identical and distinct flavours, massive tops) with
    each gluon leg in turn; g g > g g g g at six legs; u u~ > g g g g g at
    seven (generation about 6 min)."""

    del loop_native, loop_sm_ufo  # the staged native and the UFO must exist
    _campaign([key, "--points", "0", "--scratch", str(tmp_path)], tmp_path / "w.json")


@pytest.mark.loop_nightly
def test_seven_gluon_ward_identity(tmp_path, loop_native, loop_sm_ufo):
    """ADDENDUM H's seven-gluon Ward check: g g > g g g g g (top-free, NJet's
    scheme), legs 1, 4 and 7, c = 1 and i, binary64 and double-double, native
    inputs (the MQ-WARD campaign configuration; about 20 min on 12 threads
    and 27 GB). Measured: the binary64 virtual at most 2.9e-10 of the
    physical scale (0.5 of its estimate, the estimate 6.6e-10 of the scale),
    double-double 2.8e-26; the Born ratio 9.8e-16 and 5.3e-32."""

    del loop_native, loop_sm_ufo
    data = _campaign(
        [
            "gg_ggggg",
            "--points",
            "0",
            "--legs",
            "1",
            "4",
            "7",
            "--scratch",
            str(tmp_path),
        ],
        tmp_path / "w.json",
    )
    assert data["particle_veto"] == [6]
