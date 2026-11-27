# SPDX-License-Identifier: 0BSD
"""Stability estimates and precision carriers of the native v2 path (WP2.3).

* Carriers: the native double-double and arbitrary-precision evaluations
  equal the Python reference at 60 digits (the Python path has no
  double-double arithmetic and refuses it); each carrier's own estimate
  brackets its true error (within the user's one-order-of-magnitude margin,
  ``ESTIMATE_MARGIN``) and stays meaningful.
* Injected instabilities: a reduction lane perturbed in the reference
  evaluation is flagged by the reduction-direction check and by a rotation
  check; a perturbed numerator input (a tree row) is flagged by a rotation
  check, while the reduction-direction check, which re-reduces the same
  coefficients, cannot see it (its documented scope).
* No check, no estimate, in every carrier.
* A failed lane fails its point alone (``FailedPoints``).
* The committed calibration (``tools/loop_v2_stability_calibration.py``,
  ``tests/fixtures/loop/v2/stability_calibration.json``) covers 2000 binary64
  and 400 double-double points per process against Float(256) and records
  each configuration's false-negative rate within its documented bound.
"""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import pytest

from pyamplicol.loops.v2.assemble import LoopSlice, Precision, SliceRequest
from pyamplicol.loops.v2.plan_export import NativeExecutor
from pyamplicol.loops.v2.stability import (
    REDUCTION,
    RESCALED,
    ROTATION,
    Rotation,
    StabilityChecks,
    deviation,
)

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "fixtures" / "loop"
PARAMETERS = {"aS": "0.118", "MT": "173"}
FIXED = {"MB": "0"}
REFERENCE = Precision.arbitrary(60)
CHECKS = StabilityChecks((Rotation.from_axis_angle((1, 1, 0), 0.9),), True)
ONLY_REDUCTION = (REDUCTION,)
ONLY_ROTATION = (ROTATION.format(0),)
RESCALED_CHECK = StabilityChecks(reduction_direction=True, rescaled=True)
# USER DECISION (2026-10-23; DESIGN §10.4, ``test_loop_v2_slice.py``): a
# single-point test of a stability estimate E holds the true error to at most
# 10 E, for every check configuration ("you can easily afford a
# one-order-of-magnitude safety margin in the relative stability thresholds").
ESTIMATE_MARGIN = 10


def _points(name: str) -> tuple[list, list[str]]:
    if name == "gg_gg":
        path = ROOT / "tools" / "loop_reference" / "gluon_scattering_qcd.json"
        data = json.loads(path.read_text(), parse_float=str)
        return data["points"], ["1000000"] * len(data["points"])
    path = FIXTURES / "required" / "uux_ddxg.json"
    data = json.loads(path.read_text(), parse_float=str)
    points = data["points"]
    mu = data.get("mu_r", ["1000"] * len(points))
    if isinstance(mu, str):
        mu = [mu] * len(points)
    return points, [format(Decimal(str(m)) ** 2, "f") for m in mu]


PROCESSES = {
    "gg_gg": ("g g > g g", {"QCD": 2, "QED": 0}),
    "uux_ddxg": ("u u~ > d d~ g", {"QCD": 3, "QED": 0}),
}


@pytest.fixture(scope="module")
def slices(loop_sm_compiled_model, loop_native):
    from pyamplicol.models.loading import load_compiled_model

    compiled = load_compiled_model(loop_sm_compiled_model)
    built: dict[str, tuple[LoopSlice, NativeExecutor]] = {}

    def build(name: str) -> tuple[LoopSlice, NativeExecutor]:
        if name not in built:
            process, born = PROCESSES[name]
            loop = {k: v + (2 if k == "QCD" else 0) for k, v in born.items()}
            slice_ = LoopSlice.generate(
                compiled,
                SliceRequest(process, born, loop, fixed_parameters=FIXED),
                loop_native,
            )
            built[name] = slice_, NativeExecutor(slice_)
        return built[name]

    return build


def _virtual(result) -> dict:
    return result.components["virtual"]


# ----------------------------------------------------------------- carriers


@pytest.mark.loop_fast
def test_one_reduction_check_per_evaluation():
    assert StabilityChecks().reduction is None and StabilityChecks().frame is None
    assert StabilityChecks(reduction_direction=True).frame == "reversed"
    assert RESCALED_CHECK.reduction == RESCALED and RESCALED_CHECK.frame == "rescaled"
    with pytest.raises(ValueError, match="variant of the reduction-direction"):
        StabilityChecks(rescaled=True)


@pytest.mark.loop_fast
def test_recommended_checks_and_their_public_spellings(tmp_path):
    """The recommended configuration (the O1 rule's estimate, package
    mq-stab): one rotation that moves the beam axis and every other
    coordinate axis (a rotation about z re-rounds no beam momentum and
    under-reported u u~ > u u~ e+ e-'s double pole 23-fold:
    ``test_loop_v2_fermion_lines.py``) and the rescaled reduction check, the
    one check that evaluates the master integrals again. The public API, the
    TOML card and the CLI spell it."""

    from pyamplicol.api.virtual_results import VirtualStabilityChecks
    from pyamplicol.cli.parser import build_parser
    from pyamplicol.config import ConfigurationError, resolve_config
    from pyamplicol.loops.v2.stability import RECOMMENDED_CHECKS, default_rotation

    (rotation,) = RECOMMENDED_CHECKS.rotations
    assert rotation == default_rotation(0)
    for axis in range(3):
        image = [rotation.matrix[i][axis] for i in range(3)]
        assert all(image[i] != (i == axis) for i in range(3)), (axis, image)
    assert RECOMMENDED_CHECKS.reduction == RESCALED
    public = VirtualStabilityChecks(1, True, rescaled=True)
    assert public.to_checks() == RECOMMENDED_CHECKS
    assert VirtualStabilityChecks(1, True).to_checks().reduction == REDUCTION
    with pytest.raises(ValueError, match="variant of the reduction-direction"):
        VirtualStabilityChecks(1, rescaled=True)
    with pytest.raises(TypeError, match="rescaled must be a boolean"):
        VirtualStabilityChecks(1, True, rescaled=1)
    card = tmp_path / "card.toml"
    card.write_text(
        'action = "generate"\n[evaluation.stability_checks]\n'
        "rotation_count = 1\nreduction_direction = true\nrescaled = true\n"
    )
    checks = resolve_config(card).effective.evaluation.stability_checks
    assert checks.to_checks() == public
    card.write_text(
        'action = "generate"\n[evaluation.stability_checks]\nrescaled = true\n'
    )
    with pytest.raises(ConfigurationError, match="stability_checks"):
        resolve_config(card)
    arguments = build_parser().parse_args(
        ["evaluate", "--stability-reduction-direction", "--stability-rescaled"]
    )
    assert getattr(arguments, "evaluation.stability_checks.rescaled") is True


@pytest.mark.loop_fast
def test_double_double_is_a_native_carrier():
    carrier = Precision.double_double()
    assert carrier.unit_roundoff == 2.0**-104
    assert carrier.working_digits == 52
    with pytest.raises(ValueError, match="takes no digits"):
        Precision("double-double", 30)


@pytest.mark.loop_physics
@pytest.mark.parametrize("name", ["gg_gg", "uux_ddxg"])
def test_native_carriers_equal_the_reference(slices, name):
    """Native double-double and 60-digit evaluations against the Python
    60-digit reference: the double-double error is bracketed by its own
    estimate (within ``ESTIMATE_MARGIN``), which is below 1e-24; the
    arbitrary-precision run agrees to the reference's digits."""

    slice_, executor = slices(name)
    points, scales = _points(name)
    point, scale = points[0], scales[0]
    reference = slice_.evaluate(
        point, scale, parameters=PARAMETERS, precision=REFERENCE
    )
    with pytest.raises(ValueError, match="native executor"):
        slice_.evaluate(
            point, scale, parameters=PARAMETERS, precision=Precision.double_double()
        )
    (double_double,) = executor.evaluate(
        [point],
        [scale],
        parameters=PARAMETERS,
        precision=Precision.double_double(),
        stability=CHECKS,
    )
    error = deviation(_virtual(reference), _virtual(double_double))
    estimate = double_double.stability
    assert 0 < error <= ESTIMATE_MARGIN * estimate.relative, (error, estimate)
    assert estimate.relative < 1e-24, estimate
    for names in (ONLY_REDUCTION, ONLY_ROTATION):
        assert error <= ESTIMATE_MARGIN * estimate.subset(names).relative, names
    (rescaled,) = executor.evaluate(
        [point],
        [scale],
        parameters=PARAMETERS,
        precision=Precision.double_double(),
        stability=RESCALED_CHECK,
    )
    assert error <= ESTIMATE_MARGIN * rescaled.stability.relative
    assert rescaled.stability.relative < 1e-24
    (arbitrary,) = executor.evaluate(
        [point], [scale], parameters=PARAMETERS, precision=REFERENCE
    )
    assert deviation(_virtual(reference), _virtual(arbitrary)) < 1e-55
    assert arbitrary.stability is None
    (plain,) = executor.evaluate(
        [point], [scale], parameters=PARAMETERS, precision=Precision.double_double()
    )
    assert plain.stability is None


@pytest.mark.loop_physics
def test_rescaled_check_in_the_slice_and_natively(slices):
    """The rescaled reduction check (masters recomputed at scaled inputs)
    gives the same estimate in the Python slice and natively, and brackets
    the binary64 error (within ``ESTIMATE_MARGIN``)."""

    from pyamplicol.loops.v2.assemble import Laurent

    slice_, executor = slices("uux_ddxg")
    points, scales = _points("uux_ddxg")
    point, scale = points[0], scales[0]
    (native,) = executor.evaluate(
        [point], [scale], parameters=PARAMETERS, stability=RESCALED_CHECK
    )
    reference = slice_.evaluate(
        point, scale, parameters=PARAMETERS, stability=RESCALED_CHECK
    )
    assert native.stability.relative == pytest.approx(
        reference.stability.relative, rel=1e-6
    )
    truth = slice_.evaluate(point, scale, parameters=PARAMETERS, precision=REFERENCE)
    floats = {
        sector: Laurent(*[float(x) for x in value.coefficients])
        for sector, value in _virtual(truth).items()
    }
    error = deviation(floats, _virtual(native))
    assert 0 < error <= ESTIMATE_MARGIN * native.stability.relative
    assert native.stability.relative < 1e-10


# ----------------------------------------------------------- injected faults


class _Injected:
    """A native executor whose first evaluation (the reference point of an
    estimate) is perturbed: ``lane`` scales one reduction lane's direct
    integrals, ``row`` one numerator input (a tree row) by ``1 + delta``."""

    def __init__(self, inner, delta: float, *, lane: bool = False, row: bool = False):
        self.inner = inner
        self.delta = delta
        self.lane = lane
        self.row = row
        self.calls = 0

    def evaluate(self, inputs, **options):
        first = self.calls == 0
        self.calls += 1
        if first and self.row:
            inputs = dict(inputs)
            rows = [list(point) for point in inputs["block_rows"]]
            # The largest input of the point, so the perturbation is visible.
            index = max(range(len(rows[0])), key=lambda k: abs(complex(*rows[0][k])))
            re, im = rows[0][index]
            rows[0][index] = [re * (1 + self.delta), im * (1 + self.delta)]
            inputs["block_rows"] = rows
        output = self.inner.evaluate(inputs, **options)
        if first and self.lane:
            lanes = list(output["families"][0])
            # The largest lane of the point, so the perturbation is visible.
            index = max(
                range(len(lanes)),
                key=lambda k: max(abs(complex(*v)) for v in lanes[k][1]),
            )
            key, values = lanes[index]
            lanes[index] = (key, [[x * (1 + self.delta) for x in v] for v in values])
            output = dict(output, families=[lanes, *output["families"][1:]])
        return output

    def __getattr__(self, name):
        return getattr(self.inner, name)


def _evaluate_injected(executor, point, scale, checks, **fault):
    native = executor.native("double")
    injected = _Injected(native, 1e-6, **fault)
    executor._native[("double", None, executor.lanes)] = injected
    try:
        (result,) = executor.evaluate(
            [point], [scale], parameters=PARAMETERS, stability=checks
        )
    finally:
        executor._native[("double", None, executor.lanes)] = native
    return result


@pytest.mark.loop_physics
def test_an_unstable_reduction_lane_is_flagged(slices):
    """A lane whose direct reduction is off by 1e-6 (as an ill-conditioned
    reduction is) makes either check's estimate bracket the error it causes
    (within ``ESTIMATE_MARGIN``)."""

    _slice, executor = slices("uux_ddxg")
    points, scales = _points("uux_ddxg")
    clean = executor.evaluate(
        [points[0]], [scales[0]], parameters=PARAMETERS, stability=CHECKS
    )[0]
    faulty = _evaluate_injected(executor, points[0], scales[0], CHECKS, lane=True)
    error = deviation(_virtual(clean), _virtual(faulty))
    assert error > 1e3 * clean.stability.relative
    for names in (ONLY_REDUCTION, ONLY_ROTATION, (*ONLY_ROTATION, REDUCTION)):
        assert error <= ESTIMATE_MARGIN * faulty.stability.subset(names).relative, names
    rescaled = _evaluate_injected(
        executor, points[0], scales[0], RESCALED_CHECK, lane=True
    ).stability
    assert set(rescaled.checks) == {RESCALED}
    assert error <= ESTIMATE_MARGIN * rescaled.relative


@pytest.mark.loop_physics
def test_an_unstable_numerator_is_flagged_by_rotations(slices):
    """A numerator input off by 1e-6 at the reference point: the rotation
    check brackets the error (within ``ESTIMATE_MARGIN``); the
    reduction-direction check re-reduces the same coefficients and so cannot
    see it (it measures the reduction alone, as documented)."""

    _slice, executor = slices("uux_ddxg")
    points, scales = _points("uux_ddxg")
    clean = executor.evaluate(
        [points[0]], [scales[0]], parameters=PARAMETERS, stability=CHECKS
    )[0]
    faulty = _evaluate_injected(executor, points[0], scales[0], CHECKS, row=True)
    error = deviation(_virtual(clean), _virtual(faulty))
    assert error > 1e3 * clean.stability.relative
    assert error <= ESTIMATE_MARGIN * faulty.stability.subset(ONLY_ROTATION).relative
    assert faulty.stability.subset(ONLY_REDUCTION).relative < 1e-3 * error


class _FailingPoint:
    """A native executor whose evaluations report a failed reduction lane
    for point ``index`` (as an ``ImpreciseResult`` master does)."""

    def __init__(self, inner, index: int, *, after: int = 0):
        self.inner = inner
        self.index = index
        self.after = after
        self.calls = 0

    def evaluate(self, inputs, **options):
        output = self.inner.evaluate(inputs, **options)
        self.calls += 1
        if self.calls > self.after:
            failures = [list(f) for f in output["failures"]]
            failures[self.index] = ["lane 0: ImpreciseResult (injected)"]
            output = dict(output, failures=failures)
        return output

    def __getattr__(self, name):
        return getattr(self.inner, name)


@pytest.mark.loop_physics
@pytest.mark.parametrize("after", [0, 1], ids=["reference", "rotation"])
def test_a_failed_point_fails_alone(slices, after):
    """A failed lane fails its point only, in the reference evaluation or in
    a rotation check's: the other points of the batch keep their complete
    results, and no estimate is invented for the failed one."""

    from pyamplicol.loops.v2.plan_export import FailedPoints

    _slice, executor = slices("uux_ddxg")
    points, scales = _points("uux_ddxg")
    points, scales = points[:2], scales[:2]
    clean = executor.evaluate(points, scales, parameters=PARAMETERS, stability=CHECKS)
    key = ("double", None, executor.lanes)
    native = executor.native("double")
    executor._native[key] = _FailingPoint(native, 1, after=after)
    try:
        with pytest.raises(FailedPoints) as caught:
            executor.evaluate(points, scales, parameters=PARAMETERS, stability=CHECKS)
    finally:
        executor._native[key] = native
    failed = caught.value
    assert failed.results[1] is None and set(failed.failures) == {1}
    reason = failed.failures[1][0]
    assert "ImpreciseResult" in reason
    assert reason.startswith(ROTATION.format(0)) == bool(after)
    assert failed.results[0].totals == clean[0].totals
    assert failed.results[0].stability.relative == clean[0].stability.relative


@pytest.mark.loop_physics
def test_multi_chunk_batches_equal_one_chunk(slices):
    """Three points in chunks of two lanes (one padded lane, the master cache
    cleared per chunk) give the one-chunk values bit for bit, estimates
    included: lanes are independent."""

    slice_, executor = slices("uux_ddxg")
    points, scales = _points("uux_ddxg")
    one = executor.evaluate(points, scales, parameters=PARAMETERS, stability=CHECKS)
    chunked = NativeExecutor(slice_, lanes=2, exported=executor.exported).evaluate(
        points, scales, parameters=PARAMETERS, stability=CHECKS
    )
    assert len(points) == 3
    for a, b in zip(one, chunked, strict=True):
        assert a.totals == b.totals
        assert a.stability.absolute == b.stability.absolute


# ----------------------------------------------------------------- calibration

CALIBRATION = FIXTURES / "v2" / "stability_calibration.json"
CALIBRATED = ("gg_ggg", "uux_emepddxg")
CALIBRATION_POINTS = {"double": 2000, "double-double": 400}
# False-negative rates (true error above the estimate) per carrier and kind
# of check configuration, as DESIGN §10.4 states them: the cached
# reduction-direction check alone misses the masters' own rounding; the
# rescaled check with a rotation is the recommended configuration.
FALSE_NEGATIVE_BOUNDS = {
    "double": {
        "reduction-direction": 0.03,
        "reduction-rescaled": 0.005,
        "rotations": 0.01,
        "reduction-direction + rotations": 0.005,
        "reduction-rescaled + rotations": 0.001,
    },
    "double-double": {
        "reduction-direction": 0.015,
        "reduction-rescaled": 0.005,
        "rotations": 0.02,
        "reduction-direction + rotations": 0.0,
        "reduction-rescaled + rotations": 0.0,
    },
}


def _kind(configuration: str) -> str:
    names = configuration.split(" + ")
    reductions = [name for name in names if name in (REDUCTION, RESCALED)]
    if len(reductions) == len(names):
        return reductions[0]
    return " + ".join([*reductions, "rotations"])


@pytest.mark.loop_fast
def test_calibration_bounds_the_false_negative_rates():
    """The recorded O15 calibration (``tools/loop_v2_stability_calibration.py
    report``): Float(256)-checked points per process and carrier, every check
    configuration's false-negative rate, for the Laurent vector and for the
    finite part, within its documented bound."""

    data = json.loads(CALIBRATION.read_text())
    for name in CALIBRATED:
        for carrier, points in CALIBRATION_POINTS.items():
            record = data["processes"][name][carrier]
            assert record["points"] >= points and record["failed"] == 0, name
            configurations = record["configurations"]
            assert {_kind(c) for c in configurations} == set(
                FALSE_NEGATIVE_BOUNDS[carrier]
            )
            for configuration, rates in configurations.items():
                bound = FALSE_NEGATIVE_BOUNDS[carrier][_kind(configuration)]
                assert rates["points"] >= points, (name, carrier, configuration)
                for key in ("false_negative_rate", "false_negative_rate_finite"):
                    assert rates[key] <= bound, (name, carrier, configuration, key)
