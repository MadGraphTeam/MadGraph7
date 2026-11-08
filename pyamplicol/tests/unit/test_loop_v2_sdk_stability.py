# SPDX-License-Identifier: 0BSD
"""Binary64 stability monitoring in the native SDK (m-lit LIT-SDK, PLAN §3.5;
the revision-2c ruling: one Rust implementation, ``rusticol_loop_reduction::
v2_stability``, called by the Python API's binary64 path and by the SDKs,
the Python assembly kept as the oracle).

* The exact rotations: the Rust default rotations (the C ABI's rotation
  count) and the rotation specifications are the Python ones, Fraction for
  Fraction, and the refused specifications are refused in the same words.
* The Python API's binary64 Born-interfered estimates through the native
  implementation (``stability.NATIVE_BINARY64``) equal the Python assembly
  bitwise (every public estimate field, the values unchanged) on g g > t t~
  and g g > g g, for rotations, the (rescaled) reduction check, rotation
  specifications and the cross-backend check.
* The C ABI's setter (``rusticol_virtual_set_stability_checks``) takes every
  ``VirtualStabilityChecks`` field; the SDK's estimates (``"stability"`` of
  the diagnostics) equal the Python API's bitwise, with the same statuses,
  failures and ``not_run`` records, on NinjaRS and on TIRust; without
  checks there is no estimate; a configuration the Python API refuses is an
  invalid argument with its message; the other evaluations say why they
  form none.
* The C++, Fortran and packaged Rust wrappers' setters give the same
  estimates.
* The SDK side of the loop-induced self-test (PLAN §2.a.10), checked as
  ``lit-tir``'s T8: the shipped artifact evaluates through the C ABI, with
  and without the monitoring, in a process without a licence while another
  holds the restricted Symbolica permit, bitwise as licensed and as the
  Python API.
"""

from __future__ import annotations

import ctypes
import json
import math
from decimal import Decimal
from fractions import Fraction

import pytest

from tests.unit.test_loop_v2_sdk_extras import _error, _Request, _ResultShape, _sdk
from tests.unit.test_loop_v2_sdk_parity import GG_POINT, POINT_173, _generate

pytestmark = pytest.mark.loop_physics

STATUS_INVALID_ARGUMENT = 1
STATUS_RUNTIME_ERROR = 3
SUCCESS_STATUS = 4  # RUSTICOL_VIRTUAL_ACCURACY_NOT_ESTABLISHED
FAILED_STATUS = 3
SCALES = ["1000000", "490000"]
PROCESSES = {"ggtt": ("g g > t t~", POINT_173), "gggg": ("g g > g g", GG_POINT)}
# (rotation count, rotation specifications, reduction_direction, rescaled,
# cross_backend, poles, reduction).
CONFIGURATIONS = [
    (2, None, True, True, False, "all", "auto"),
    (1, None, True, False, False, "none", "auto"),
    (0, ["euler:0.1,0.2,0.3", "z:3/5:4/5"], False, False, False, "all", "auto"),
    (1, None, True, True, True, "all", "auto"),
    (1, None, True, True, False, "all", "tirust"),
    (0, None, True, False, True, "none", "tirust"),
]
SPECIFICATIONS = [
    "z:3/5:4/5",
    "x:-0.6:0.8",
    "1,1,0:0.9",
    "0.3,-0.2,0.9:2.5",
    "euler:0.1,0.2,0.3",
    "euler:-1.5,0.25,3",
]
REFUSED = ["z:1/2:1/2", "euler:0.1,0.2", "q:1:0", "1,0:0.5", "0,0,0:1"]


@pytest.fixture(scope="module")
def artifacts(loop_sm_compiled_model, loop_native, tmp_path_factory):
    return {
        name: _generate(
            loop_sm_compiled_model, tmp_path_factory.mktemp(name), expression, name
        )
        for name, (expression, _point) in PROCESSES.items()
    }


def _exact(matrix) -> tuple:
    return tuple(tuple(Fraction(x) for x in row) for row in matrix)


def test_the_native_rotations_are_the_python_ones(loop_native, monkeypatch):
    """The native rotations equal the Python construction (the oracle), and
    the Python API's production rotations (``default_rotation``,
    ``parse_rotation``: the native ones, m-lit lit-sdk-b) equal both."""

    from pyamplicol.api.virtual_results import parse_rotation
    from pyamplicol.loops.v2 import stability
    from pyamplicol.loops.v2.stability import (
        RESCUE_ROTATION,
        default_rotation,
        python_default_rotation,
    )

    produced = [
        parse_rotation(text).matrix for text in (*SPECIFICATIONS, RESCUE_ROTATION)
    ]
    for k in range(64):
        oracle = python_default_rotation(k).matrix
        assert _exact(loop_native.stability_default_rotation(k)) == oracle, k
        assert default_rotation(k).matrix == oracle, k
    # The Python parser alone (the oracle).
    monkeypatch.setattr(stability, "NATIVE_BINARY64", False)
    for text, found in zip((*SPECIFICATIONS, RESCUE_ROTATION), produced, strict=True):
        oracle = parse_rotation(text).matrix
        assert _exact(loop_native.stability_rotation(text)) == oracle, text
        assert found == oracle, text
    for text in REFUSED:
        with pytest.raises(ValueError) as python:
            parse_rotation(text)
        with pytest.raises(ValueError) as native:
            loop_native.stability_rotation(text)
        assert str(native.value) == str(python.value), text


def _checks(configuration):
    from pyamplicol.api.virtual_results import VirtualStabilityChecks

    count, specifications, direction, rescaled, cross, _poles, _reduction = (
        configuration
    )
    return VirtualStabilityChecks(
        tuple(specifications) if specifications else count,
        direction,
        rescaled=rescaled,
        cross_backend=cross,
    )


def _served(loop_native) -> dict:
    return json.loads(loop_native.reduction_backends())


def _turned(point):
    """A second, distinct point of a centre-of-mass point: its final-state
    momenta rotated by 90 degrees about x, (E, px, py, pz) -> (E, px, -pz,
    py), exact in decimal and binary64 (a new scattering angle: another
    value)."""

    out = [list(momentum) for momentum in point]
    for leg in range(2, len(point)):
        energy, x, y, z = point[leg]
        out[leg] = [energy, x, str(-Decimal(z)), y]
    return out


def _python(runtime, point, configuration, points=None):
    """The Python API's call of one configuration (None: refused, with the
    error) on ``points`` (default: ``point`` twice)."""

    from pyamplicol.api.errors import ReductionUnavailable

    *_, poles, reduction = configuration
    try:
        return runtime.evaluate(
            [point, point] if points is None else points,
            mu_r2=SCALES,
            poles=poles,
            reduction=reduction,
            stability_checks=_checks(configuration),
            on_failure="return",
        ), None
    except ReductionUnavailable as error:
        return None, error


@pytest.mark.parametrize("name", sorted(PROCESSES))
def test_the_native_estimates_equal_the_python_assembly(
    artifacts, loop_native, monkeypatch, name
):
    from pyamplicol import Runtime
    from pyamplicol.loops.v2 import stability

    runtime = Runtime.load(artifacts[name], process=name)
    point = PROCESSES[name][1]
    compared = 0
    for configuration in CONFIGURATIONS:
        monkeypatch.setattr(stability, "NATIVE_BINARY64", True)
        native, refused = _python(runtime, point, configuration)
        monkeypatch.setattr(stability, "NATIVE_BINARY64", False)
        oracle, refused_too = _python(runtime, point, configuration)
        assert (refused is None) == (refused_too is None), configuration
        if native is None:
            assert str(refused) == str(refused_too)
            continue
        assert native.values == oracle.values, configuration
        assert native.stability == oracle.stability, configuration
        compared += sum(estimate is not None for estimate in native.stability)
    assert compared


def _number(value):
    if value is None:
        return None
    return "inf" if value == math.inf else value


def _record(estimate) -> dict:
    """The SDK's JSON form of the Python API's public estimate."""

    def laurent(values):
        return [_number(x) for x in values]

    return {
        "relative": _number(estimate.relative),
        "finite": _number(estimate.finite),
        "single_pole": _number(estimate.single_pole),
        "double_pole": _number(estimate.double_pole),
        "checks": [[name, _number(value)] for name, value in estimate.checks],
        "scale": _number(estimate.scale),
        "absolute": [laurent(row) for row in estimate.absolute],
        "magnitude": [laurent(row) for row in estimate.magnitude],
        "not_run": [[name, reason] for name, reason in estimate.not_run],
    }


def _setter(sdk):
    function = sdk.rusticol_virtual_set_stability_checks
    function.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint64,
        ctypes.c_char_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_uint32,
    ]
    return function


def _set(sdk, handle, count, specifications, direction, rescaled, cross) -> int:
    text = None if specifications is None else json.dumps(specifications).encode()
    return _setter(sdk)(handle, count, text, int(direction), int(rescaled), int(cross))


def _json(function, result) -> dict:
    needed = ctypes.c_size_t()
    assert function(result, None, ctypes.c_size_t(0), ctypes.byref(needed)) == 0
    buffer = ctypes.create_string_buffer(needed.value)
    assert function(result, buffer, needed, ctypes.byref(needed)) == 0
    return json.loads(buffer.value)


def _evaluate(sdk, handle, points, scales):
    """One ordinary C ABI evaluation: its status, values, statuses and
    diagnostics (None on a refused call)."""

    flat = [c for point in points for p in point for c in p]
    momenta = (ctypes.c_double * len(flat))(*flat)
    mu = (ctypes.c_double * len(scales))(*scales)
    request = _Request(momenta, len(flat), len(points), None, mu, len(scales))
    result = ctypes.c_void_p()
    status = sdk.rusticol_virtual_evaluate(
        handle, ctypes.byref(request), ctypes.byref(result)
    )
    if status != 0:
        return status, None
    try:
        shape = _ResultShape()
        assert sdk.rusticol_virtual_result_shape(result, ctypes.byref(shape)) == 0
        loop = (ctypes.c_double * max(shape.virtual_values, 1))()
        statuses = (ctypes.c_uint32 * len(points))()
        assert (
            sdk.rusticol_virtual_result_copy_virtual(result, loop, shape.virtual_values)
            == 0
        )
        assert (
            sdk.rusticol_virtual_result_copy_statuses(result, statuses, len(points))
            == 0
        )
        return status, {
            "virtual": list(loop)[: shape.virtual_values],
            "statuses": list(statuses),
            "diagnostics": _json(sdk.rusticol_virtual_result_diagnostics_json, result),
        }
    finally:
        sdk.rusticol_virtual_result_free(result)


def _flat(result) -> list:
    out = []
    for point in result.values:
        if point is None:
            continue
        for laurent in point:
            out.extend(math.nan if x is None else x for x in laurent)
    return out


def _same(found: list, expected: list) -> bool:
    return len(found) == len(expected) and all(
        (math.isnan(a) and math.isnan(float(b))) or a == b
        for a, b in zip(found, expected, strict=True)
    )


@pytest.mark.parametrize("name", sorted(PROCESSES))
def test_sdk_estimates_equal_the_python_api(artifacts, loop_native, name):
    from pyamplicol import Runtime

    artifact = artifacts[name]
    runtime = Runtime.load(artifact, process=name)
    point = PROCESSES[name][1]
    # Two distinct points (a lane or point order mix-up changes values).
    exact = [point, _turned(point)]
    points = [[[float(Decimal(c)) for c in p] for p in each] for each in exact]
    scales = [float(Decimal(s)) for s in SCALES]
    sdk = _sdk()
    handle = ctypes.c_void_p()
    status = sdk.rusticol_virtual_open(
        str(artifact).encode(), name.encode(), None, None, ctypes.byref(handle)
    )
    assert status == 0, _error(sdk)
    estimated = 0
    try:
        # No checks (the default): no estimate.
        status, found = _evaluate(sdk, handle, points, scales)
        assert status == 0 and found["diagnostics"]["stability"] is None
        for configuration in CONFIGURATIONS:
            count, specifications, direction, rescaled, cross, poles, reduction = (
                configuration
            )
            assert sdk.rusticol_virtual_set_poles(handle, poles.encode()) == 0
            status = sdk.rusticol_virtual_set_reduction(handle, reduction.encode())
            expected, refused = _python(runtime, point, configuration, exact)
            if reduction == "tirust" and "tirust" not in _served(loop_native):
                assert status == STATUS_RUNTIME_ERROR and refused is not None
                continue
            assert status == 0, _error(sdk)
            status = _set(
                sdk, handle, count, specifications, direction, rescaled, cross
            )
            assert status == 0, _error(sdk)
            status, found = _evaluate(sdk, handle, points, scales)
            if refused is not None:
                assert status == STATUS_RUNTIME_ERROR, configuration
                assert str(refused) in _error(sdk)
                continue
            assert status == 0, (configuration, _error(sdk))
            served = [value is not None for value in expected.values]
            assert found["statuses"] == [
                SUCCESS_STATUS if ok else FAILED_STATUS for ok in served
            ], configuration
            assert _same(found["virtual"], _flat(expected)), configuration
            failures = found["diagnostics"]["failures"]
            for index, ok in enumerate(served):
                if not ok:
                    assert "; ".join(failures[str(index)]) == (
                        expected.diagnostics[index].detail
                    ), configuration
            records = found["diagnostics"]["stability"]["estimates"]
            assert records == [
                _record(estimate)
                for estimate in expected.stability
                if estimate is not None
            ], configuration
            estimated += len(records)
        # A configuration the Python API refuses: its message, nothing set.
        from pyamplicol.api.virtual_results import VirtualStabilityChecks

        with pytest.raises(ValueError) as python:
            VirtualStabilityChecks(0, False, rescaled=True)
        assert _set(sdk, handle, 0, None, False, True, False) == (
            STATUS_INVALID_ARGUMENT
        )
        assert str(python.value) in _error(sdk)
        assert _set(sdk, handle, 1, ["z:3/5:4/5"], False, False, False) == (
            STATUS_INVALID_ARGUMENT
        )
        # Another carrier forms no estimate and says why; statuses unchanged.
        assert sdk.rusticol_virtual_set_reduction(handle, b"auto") == 0
        assert _set(sdk, handle, 1, None, True, False, False) == 0
        status = sdk.rusticol_virtual_set_arithmetic(
            handle, b"double-double", ctypes.c_uint32(0)
        )
        assert status == 0, _error(sdk)
        status, found = _evaluate(sdk, handle, points, scales)
        assert status == 0
        assert found["statuses"] == [SUCCESS_STATUS] * 2
        record = found["diagnostics"]["stability"]
        assert record["estimates"] is None and "binary64" in record["reason"]
        # A split pole selection ("uv", "ir") in binary64 with checks set
        # returns its values (the Python API's, bitwise) and the documented
        # null record (the Python API estimates those selections; lit-sdk-b
        # item [5]: before, RUSTICOL_VIRTUAL_RUNTIME_ERROR).
        assert sdk.rusticol_virtual_set_arithmetic(handle, b"double", 0) == 0
        capabilities = _json(sdk.rusticol_virtual_capabilities_json, handle)
        assert "uv" in capabilities["pole_selections"], capabilities["pole_selections"]
        for poles in ("uv", "ir"):
            assert sdk.rusticol_virtual_set_poles(handle, poles.encode()) == 0
            status, found = _evaluate(sdk, handle, points, scales)
            assert status == 0, (poles, _error(sdk))
            assert found["statuses"] == [SUCCESS_STATUS] * 2
            record = found["diagnostics"]["stability"]
            assert record["estimates"] is None, record
            assert '"uv" and "ir"' in record["reason"], record
            expected = runtime.evaluate(exact, mu_r2=SCALES, poles=poles)
            assert _same(found["virtual"], _flat(expected)), poles
        assert sdk.rusticol_virtual_set_poles(handle, b"all") == 0
        # Checks off again: no estimate, no record.
        assert _set(sdk, handle, 0, None, False, False, False) == 0
        status, found = _evaluate(sdk, handle, points, scales)
        assert status == 0 and found["diagnostics"]["stability"] is None
    finally:
        sdk.rusticol_virtual_free(handle)
    assert estimated


# ------------------------------------------------------------ loop-induced

INDUCED_CONFIGURATIONS = [
    (2, None, True, True, False, "all", "auto"),
    (1, None, True, False, False, "none", "auto"),
    (0, ["euler:0.1,0.2,0.3"], False, False, False, "all", "auto"),
    (1, None, True, True, True, "all", "auto"),
    (1, None, True, True, False, "all", "tirust"),
]


@pytest.fixture(scope="module")
def induced_artifacts(loop_sm_ufo, loop_native, tmp_path_factory):
    from tests.unit.loop_v2_induced_observables_rows import ROWS, generate

    return {
        name: generate(tmp_path_factory.mktemp(name), name, expression)
        for name, expression in ROWS.items()
    }


def _induced_python(runtime, points, scales, configuration):
    from pyamplicol.api.errors import ReductionUnavailable

    *_, poles, reduction = configuration
    try:
        return runtime.evaluate(
            points,
            mu_r2=scales,
            poles=poles,
            reduction=reduction,
            stability_checks=_checks(configuration),
            on_failure="return",
        ), None
    except ReductionUnavailable as error:
        return None, error


@pytest.mark.parametrize("name", ["li_gg_hgg", "li_uux_hgg"])
def test_loop_induced_native_estimates_equal_the_python_assembly(
    induced_artifacts, loop_native, monkeypatch, name
):
    """The loop-induced binary64 estimates through the native implementation
    equal the Python assembly (``induced.rotation_budget``, ``budgets``,
    ``cross_budget``, ``scale``) bitwise, the values and the finiteness
    diagnostics unchanged."""

    from pyamplicol import Runtime
    from pyamplicol.loops.v2 import stability
    from tests.unit.test_loop_v2_sdk_induced import _exact, _inputs

    runtime = Runtime.load(induced_artifacts[name], process="p")
    points, scales = _exact(*_inputs(name, 2))
    compared = 0
    for configuration in INDUCED_CONFIGURATIONS:
        monkeypatch.setattr(stability, "NATIVE_BINARY64", True)
        native, refused = _induced_python(runtime, points, scales, configuration)
        monkeypatch.setattr(stability, "NATIVE_BINARY64", False)
        oracle, refused_too = _induced_python(runtime, points, scales, configuration)
        assert (refused is None) == (refused_too is None), configuration
        if native is None:
            assert str(refused) == str(refused_too)
            continue
        assert native.values == oracle.values, configuration
        assert native.stability == oracle.stability, configuration
        assert [d.finiteness for d in native.diagnostics] == [
            d.finiteness for d in oracle.diagnostics
        ]
        compared += sum(estimate is not None for estimate in native.stability)
    assert compared


@pytest.mark.parametrize("name", ["li_gg_hgg", "li_uux_hgg"])
def test_loop_induced_sdk_estimates_equal_the_python_api(
    induced_artifacts, loop_native, name
):
    """The SDK's loop-induced estimates and finiteness ratios equal the
    Python API's bitwise; on TIRust, where it fails points closed, the same
    failed statuses and reasons."""

    from pyamplicol import Runtime
    from tests.unit.test_loop_v2_sdk_induced import _exact, _inputs

    artifact = induced_artifacts[name]
    runtime = Runtime.load(artifact, process="p")
    points, scales = _inputs(name, 2)
    exact_points, exact_scales = _exact(points, scales)
    sdk = _sdk()
    handle = ctypes.c_void_p()
    status = sdk.rusticol_virtual_open(
        str(artifact).encode(), b"p", None, None, ctypes.byref(handle)
    )
    assert status == 0, _error(sdk)
    estimated = 0
    try:
        for configuration in INDUCED_CONFIGURATIONS:
            count, specifications, direction, rescaled, cross, poles, reduction = (
                configuration
            )
            assert sdk.rusticol_virtual_set_poles(handle, poles.encode()) == 0
            status = sdk.rusticol_virtual_set_reduction(handle, reduction.encode())
            expected, refused = _induced_python(
                runtime, exact_points, exact_scales, configuration
            )
            if reduction == "tirust" and "tirust" not in _served(loop_native):
                assert status == STATUS_RUNTIME_ERROR and refused is not None
                continue
            assert status == 0, _error(sdk)
            status = _set(
                sdk, handle, count, specifications, direction, rescaled, cross
            )
            assert status == 0, _error(sdk)
            status, found = _evaluate(sdk, handle, points, scales)
            if refused is not None:
                assert status == STATUS_RUNTIME_ERROR, configuration
                assert str(refused) in _error(sdk)
                continue
            assert status == 0, (configuration, _error(sdk))
            served = [value is not None for value in expected.values]
            assert found["statuses"] == [
                SUCCESS_STATUS if ok else FAILED_STATUS for ok in served
            ], configuration
            assert _same(found["virtual"], _flat(expected)), configuration
            failures = found["diagnostics"]["failures"]
            for index, ok in enumerate(served):
                if not ok:
                    assert "; ".join(failures[str(index)]) == (
                        expected.diagnostics[index].detail
                    ), configuration
            record = found["diagnostics"]["stability"]
            assert record["estimates"] == [
                _record(estimate)
                for estimate in expected.stability
                if estimate is not None
            ], configuration
            ratios = [
                _number(d.finiteness)
                for d, ok in zip(expected.diagnostics, served, strict=True)
                if ok
            ]
            assert record["finiteness"] == (None if poles == "none" else ratios)
            estimated += len(record["estimates"])
    finally:
        sdk.rusticol_virtual_free(handle)
    assert estimated


# ------------------------------------------------------------- wrappers

CXX_CLIENT = r"""
// ./client artifact process input.txt: the stability checks through the C++
// wrapper (one default rotation and the rescaled reduction check, then the
// rotation specification "euler:0.1,0.2,0.3"); per evaluation one line, its
// diagnostics JSON.
#include "rusticol_virtual.hpp"
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

static double hex(std::ifstream &in) {
  std::string token;
  in >> token;
  return std::strtod(token.c_str(), nullptr);
}

int main(int argc, char **argv) {
  if (argc < 4) return 2;
  std::ifstream in(argv[3]);
  std::size_t points = 0, legs = 0;
  in >> points >> legs;
  std::vector<double> momenta(points * legs * 4), scales(points);
  for (std::size_t p = 0; p < points; ++p) {
    for (std::size_t i = 0; i < legs * 4; ++i) momenta[p * legs * 4 + i] = hex(in);
    scales[p] = hex(in);
  }
  try {
    rusticol::VirtualRuntime runtime(argv[1], std::string(argv[2]));
    runtime.set_stability_checks(1, true, true);
    auto first = runtime.evaluate(momenta, points, scales);
    std::printf("%s\n", first.diagnostics_json().c_str());
    const std::string euler = "[\"euler:0.1,0.2,0.3\"]";
    runtime.set_stability_checks(0, false, false, false, euler);
    auto second = runtime.evaluate(momenta, points, scales);
    std::printf("%s\n", second.diagnostics_json().c_str());
  } catch (const rusticol::VirtualError &error) {
    std::printf("error %d %s\n", error.status(), error.what());
    return 1;
  }
  return 0;
}
"""

FORTRAN_CLIENT = r"""
program stability_client
  use, intrinsic :: iso_c_binding
  use rusticol_virtual
  implicit none
  character(len=4096) :: artifact, process, path
  integer :: unit, points, legs, p, l, i
  integer(c_int64_t), allocatable :: bits(:)
  real(c_double), allocatable :: momenta(:,:,:), scales(:)
  type(virtual_runtime) :: runtime
  type(virtual_evaluation) :: result
  integer(c_int) :: ierr
  character(:), allocatable :: message
  call get_command_argument(1, artifact)
  call get_command_argument(2, process)
  call get_command_argument(3, path)
  open(newunit=unit, file=trim(path), status='old', action='read')
  read(unit, *) points, legs
  allocate(momenta(4, legs, points), scales(points), bits(4*legs+1))
  do p = 1, points
    read(unit, *) bits
    do l = 1, legs
      do i = 1, 4
        momenta(i, l, p) = transfer(bits(4*(l-1)+i), 1.0_c_double)
      end do
    end do
    scales(p) = transfer(bits(4*legs+1), 1.0_c_double)
  end do
  close(unit)
  call runtime%load(trim(artifact), process=trim(process), ierr=ierr)
  if (ierr /= VIRTUAL_OK) stop 1
  call runtime%set_stability_checks(1_c_int64_t, .true., .true., .false., ierr=ierr, &
      error_message=message)
  if (ierr /= VIRTUAL_OK) stop 2
  call runtime%evaluate(momenta, scales, result, ierr=ierr, error_message=message)
  if (ierr /= VIRTUAL_OK) stop 3
  write(*, '(a)') result%diagnostics_json
  call runtime%set_stability_checks(0_c_int64_t, .false., .false., .false., &
      rotations='["euler:0.1,0.2,0.3"]', ierr=ierr, error_message=message)
  if (ierr /= VIRTUAL_OK) stop 4
  call runtime%evaluate(momenta, scales, result, ierr=ierr, error_message=message)
  if (ierr /= VIRTUAL_OK) stop 5
  write(*, '(a)') result%diagnostics_json
  call runtime%close()
end program
"""


RUST_CLIENT = """
// ./client artifact process input.txt: the stability checks through the
// packaged single-file Rust wrapper (as the C++ client), the input doubles as
// IEEE bit patterns (signed 64-bit integers, as the Fortran client's).
#[path = "@WRAPPER@"]
#[allow(dead_code)]
mod rusticol;
use rusticol::virtual_amplitudes::Runtime;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let text = std::fs::read_to_string(&args[3])?;
    let mut lines = text.lines();
    let head: Vec<usize> = lines
        .next()
        .ok_or("empty input")?
        .split_whitespace()
        .map(|v| v.parse())
        .collect::<Result<_, _>>()?;
    let (points, legs) = (head[0], head[1]);
    let (mut momenta, mut scales) = (Vec::new(), Vec::new());
    for line in lines.take(points) {
        let values: Vec<f64> = line
            .split_whitespace()
            .map(|v| v.parse::<i64>().map(|bits| f64::from_bits(bits as u64)))
            .collect::<Result<_, _>>()?;
        momenta.extend_from_slice(&values[..4 * legs]);
        scales.push(values[4 * legs]);
    }
    let mut runtime = Runtime::open(&args[1], Some(args[2].as_str()), None, None)?;
    runtime.set_stability_checks(1, None, true, true, false)?;
    let first = runtime.evaluate(&momenta, points, None, &scales, false, None)?;
    println!("{}", first.diagnostics_json);
    runtime.set_stability_checks(0, Some(&["euler:0.1,0.2,0.3"]), false, false, false)?;
    let second = runtime.evaluate(&momenta, points, None, &scales, false, None)?;
    println!("{}", second.diagnostics_json);
    Ok(())
}
"""


def test_the_cxx_fortran_and_rust_wrappers_set_the_checks(
    artifacts, loop_native, tmp_path
):
    """The C++, Fortran and packaged Rust wrappers' ``set_stability_checks``
    (a rotation count with the rescaled check, then a rotation specification)
    give the Python API's estimates (``"stability"`` of the diagnostics) on
    g g > t t~."""

    import subprocess

    from pyamplicol import Runtime
    from pyamplicol.api.virtual_results import VirtualStabilityChecks
    from tests.unit.test_loop_v2_sdk_extras import SDK, WRAPPER, _compiler, _staged_sdk

    build = _staged_sdk()
    link = [f"-L{build}", "-lrusticol_loop_sdk", f"-Wl,-rpath,{build}", "-lm"]
    (tmp_path / "client.cpp").write_text(CXX_CLIENT)
    (tmp_path / "client.f90").write_text(FORTRAN_CLIENT)
    (tmp_path / "client.rs").write_text(RUST_CLIENT.replace("@WRAPPER@", str(WRAPPER)))
    subprocess.run(
        [
            _compiler("RUSTC", "rustc"),
            "--edition=2021",
            "-D",
            "warnings",
            str(tmp_path / "client.rs"),
            "-o",
            str(tmp_path / "rust"),
            "-L",
            f"native={build}",
            "-l",
            "dylib=rusticol_loop_sdk",
            "-C",
            f"link-arg=-Wl,-rpath,{build}",
        ],
        check=True,
        cwd=tmp_path,
    )
    subprocess.run(
        [
            _compiler("CXX", "c++"),
            "-std=c++17",
            "-O2",
            f"-I{SDK / 'include'}",
            str(tmp_path / "client.cpp"),
            *link,
            "-o",
            str(tmp_path / "cpp"),
        ],
        check=True,
        cwd=tmp_path,
    )
    subprocess.run(
        [
            _compiler("FC", "gfortran"),
            "-std=f2018",
            "-O2",
            f"-J{tmp_path}",
            str(SDK / "fortran" / "rusticol_virtual.f90"),
            str(tmp_path / "client.f90"),
            *link,
            "-o",
            str(tmp_path / "fortran"),
        ],
        check=True,
        cwd=tmp_path,
    )
    name = "ggtt"
    point = PROCESSES[name][1]
    floats = [[float(Decimal(c)) for c in p] for p in point]
    scales = [float(Decimal(s)) for s in SCALES]

    def feed(bits: bool) -> str:
        import struct

        def text(value: float) -> str:
            if bits:
                return str(struct.unpack("<q", struct.pack("<d", value))[0])
            return value.hex()

        lines = [f"2 {len(floats)}"]
        for scale in scales:
            lines.append(" ".join([text(c) for p in floats for c in p] + [text(scale)]))
        return "\n".join(lines) + "\n"

    runtime = Runtime.load(artifacts[name], process=name)
    expected = []
    for checks in (
        VirtualStabilityChecks(1, True, rescaled=True),
        VirtualStabilityChecks(("euler:0.1,0.2,0.3",)),
    ):
        result = runtime.evaluate([point, point], mu_r2=SCALES, stability_checks=checks)
        expected.append([_record(estimate) for estimate in result.stability])
    for language in ("cpp", "fortran", "rust"):
        path = tmp_path / f"{language}.txt"
        path.write_text(feed(language != "cpp"))
        run = subprocess.run(
            [str(tmp_path / language), str(artifacts[name]), name, str(path)],
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
        )
        assert run.returncode == 0, run.stdout[-2000:] + run.stderr[-2000:]
        found = [
            json.loads(line)["stability"]["estimates"]
            for line in run.stdout.splitlines()
            if line.strip()
        ]
        assert found == expected, language


# ------------------------------------------------- the SDK's self-test (T8)

SELFTEST_CLIENT = r"""
// ./client artifact process parameters.json input.txt: the SDK side of the
// loop-induced self-test through the C++ wrapper: one binary64 evaluation of
// the shipped artifact on the default backend, then one with the monitoring
// on (two default rotations and the rescaled reduction check). Per
// evaluation: the statuses, the Born value count, the virtual values as bit
// patterns and the diagnostics JSON, one line each.
#include "rusticol_virtual.hpp"
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

int main(int argc, char **argv) {
  if (argc < 5) return 2;
  std::ifstream parameters_file(argv[3]);
  std::stringstream parameters;
  parameters << parameters_file.rdbuf();
  std::ifstream in(argv[4]);
  std::size_t points = 0, legs = 0;
  in >> points >> legs;
  std::vector<double> momenta(points * legs * 4), scales(points);
  std::string token;
  for (std::size_t p = 0; p < points; ++p) {
    for (std::size_t i = 0; i < legs * 4; ++i) {
      in >> token;
      momenta[p * legs * 4 + i] = std::strtod(token.c_str(), nullptr);
    }
    in >> token;
    scales[p] = std::strtod(token.c_str(), nullptr);
  }
  try {
    rusticol::VirtualRuntime runtime(argv[1], std::string(argv[2]), parameters.str());
    for (int pass = 0; pass < 2; ++pass) {
      if (pass == 1) runtime.set_stability_checks(2, true, true);
      auto result = runtime.evaluate(momenta, points, scales);
      std::printf("statuses");
      for (auto status : result.statuses()) std::printf(" %u", status);
      std::printf("\nborn %zu\nvirtual", result.born().size());
      for (double value : result.virtual_values()) {
        std::uint64_t bits;
        std::memcpy(&bits, &value, sizeof bits);
        std::printf(" %" PRIu64, bits);
      }
      std::printf("\ndiagnostics %s\n", result.diagnostics_json().c_str());
    }
    // The selections (m-lit lit-sdk-b): the amplitude-basis anchors and one
    // helicity row, in binary64, or with a fifth argument "extended" in
    // double-double and arbitrary precision (the package key's carriers).
    runtime.set_stability_checks(0);
    const bool extended = argc > 5 && std::strcmp(argv[5], "extended") == 0;
    for (int pass = 0; pass < 2; ++pass) {
      if (extended && pass == 0) runtime.set_arithmetic("double-double");
      if (extended && pass == 1) runtime.set_arithmetic("arbitrary", 32);
      rusticol::VirtualSelection selection;
      if (pass == 0) selection.color_resolution = std::string("amplitude-basis");
      else selection.helicity_ids = {"h1"};
      bool resolved = pass == 0;
      auto result =
          runtime.evaluate_selected(momenta, points, scales, selection, resolved);
      std::printf("statuses");
      for (auto status : result.statuses()) std::printf(" %u", status);
      std::printf("\nvirtual");
      for (double value : result.virtual_values()) {
        std::uint64_t bits;
        std::memcpy(&bits, &value, sizeof bits);
        std::printf(" %" PRIu64, bits);
      }
      std::printf("\naxes %s\ndecimal %s\n", result.axes_json().c_str(),
                  result.decimal_json().c_str());
    }
  } catch (const rusticol::VirtualError &error) {
    std::printf("error %d %s\n", error.status(), error.what());
    return 1;
  }
  return 0;
}
"""


def test_the_sdk_self_test_touches_no_symbolica_state(loop_native, tmp_path):
    """The SDK side of the loop-induced self-test (PLAN §2.a.10 / §3.5),
    checked as ``lit-tir``'s T8: a fresh process with ``SYMBOLICA_LICENSE``
    removed and ``PYAMPLICOL_NATIVE_LIBRARY_KEY=off`` evaluates the shipped
    artifact (``assets/selftest_virtual/li``, g g > h) through the C ABI on
    the default backend, without and with the binary64 monitoring, while
    another process holds the restricted Symbolica permit of a private lock
    directory (any Symbolica state access aborts). Its output equals the
    licensed run's; the values and the estimates equal the Python API's
    bitwise, with no Born values."""

    import importlib.util
    import os
    import struct
    import subprocess
    from pathlib import Path

    from pyamplicol import Runtime
    from pyamplicol._internal import selftest_virtual_li as selftest
    from pyamplicol.api.virtual_results import VirtualStabilityChecks
    from tests.unit.test_loop_v2_sdk_extras import SDK, _compiler, _staged_sdk

    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location(
        "loop_artifact_portability", root / "tools" / "loop_artifact_portability.py"
    )
    assert spec is not None and spec.loader is not None
    portability = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(portability)

    build = _staged_sdk()
    (tmp_path / "client.cpp").write_text(SELFTEST_CLIENT)
    subprocess.run(
        [
            _compiler("CXX", "c++"),
            "-std=c++17",
            "-O2",
            f"-I{SDK / 'include'}",
            str(tmp_path / "client.cpp"),
            f"-L{build}",
            "-lrusticol_loop_sdk",
            f"-Wl,-rpath,{build}",
            "-lm",
            "-o",
            str(tmp_path / "client"),
        ],
        check=True,
        cwd=tmp_path,
    )
    artifact = selftest.fixture_root() / "artifact"
    expected = json.loads((selftest.fixture_root() / "expected.json").read_text())
    # The SDK reads each binary64 input as its exact value
    # (rusticol_virtual.h), so the Python API is given those exact values
    # (the decimal text of the doubles). The Python API reads a float by its
    # shortest decimal (runtime/virtual.py, assemble._fraction), half an ulp
    # from the double where the two differ (here 33.6, 50.592, ...): g g > h's
    # values do not see it (its invariants are exact either way), its
    # rotation check does (estimates that differ in their last digits).
    points = [
        [[float(Decimal(c)) for c in p] for p in point] for point in expected["points"]
    ]
    scales = [float(Decimal(s)) for s in expected["mu_r2"]]
    exact_points = [[[str(Decimal(c)) for c in p] for p in point] for point in points]
    exact_scales = [str(Decimal(s)) for s in scales]
    (tmp_path / "parameters.json").write_text(
        json.dumps({key: [value, "0"] for key, value in expected["parameters"].items()})
    )
    lines = [f"{len(points)} {len(points[0])}"]
    for point, scale in zip(points, scales, strict=True):
        lines.append(" ".join(v.hex() for v in [*(c for p in point for c in p), scale]))
    (tmp_path / "input.txt").write_text("\n".join(lines) + "\n")
    command = [
        str(tmp_path / "client"),
        str(artifact),
        selftest.NAME,
        str(tmp_path / "parameters.json"),
        str(tmp_path / "input.txt"),
    ]

    def run(env: dict[str, str], extra: tuple[str, ...] = ()) -> list[str]:
        done = subprocess.run(
            [*command, *extra],
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
            env=env,
        )
        output = done.stdout[-3000:] + done.stderr[-3000:]
        assert done.returncode == 0, output
        assert "restricted Symbolica" not in output
        return done.stdout.splitlines()

    lock = tmp_path / "symbolica-locks"
    lock.mkdir()
    lock.chmod(0o700)
    base = {k: v for k, v in os.environ.items() if k != "SYMBOLICA_LICENSE"}
    with portability.RestrictedPermit(lock_dir=str(lock)) as permit:
        assert permit.record["verified"], permit.record
        unlicensed = run(
            permit.environment({**base, "PYAMPLICOL_NATIVE_LIBRARY_KEY": "off"})
        )
    licensed = run(dict(os.environ))
    assert unlicensed == licensed
    # The extended carriers of the selections run under pyAmpliCol's package
    # key without a user licence (D7): equal to the licensed run.
    extended = run(base, ("extended",))
    assert extended == run(dict(os.environ), ("extended",))

    runtime = Runtime.load(
        artifact, process=selftest.NAME, model_parameters=expected["parameters"]
    )
    mask = 2**64 - 1
    for index, checks in enumerate(
        (None, VirtualStabilityChecks(2, True, rescaled=True))
    ):
        result = runtime.evaluate(
            exact_points, mu_r2=exact_scales, stability_checks=checks
        )
        assert result.contribution == "loop-induced"
        statuses, born, values, diagnostics = licensed[4 * index : 4 * index + 4]
        assert statuses == "statuses" + " 4" * len(points), statuses
        assert born == "born 0"
        bits = [int(b) for b in values.split()[1:]]
        assert bits == [
            struct.unpack("<q", struct.pack("<d", v))[0] & mask for v in _flat(result)
        ]
        record = json.loads(diagnostics.split(" ", 1)[1])["stability"]
        if checks is None:
            assert record is None
        else:
            assert record["estimates"] == [_record(e) for e in result.stability]
    # The selections: binary64 bitwise the Python API's, axes its IDs; the
    # extended carriers its Decimal values.
    anchors = runtime.evaluate_resolved(
        exact_points, mu_r2=exact_scales, color_resolution="amplitude-basis"
    )
    one = runtime.evaluate(exact_points, mu_r2=exact_scales, helicities=["h1"])
    flat_anchors = [
        math.nan if x is None else x
        for v in anchors.values
        for sector in v
        for row in sector
        for c in row
        for x in (c.finite, c.single_pole, c.double_pole)
    ]
    for index, (expected_values, ids) in enumerate(
        (
            (
                flat_anchors,
                [list(anchors.helicity_ids), list(anchors.virtual_color_ids)],
            ),
            (_flat(one), [[], []]),
        )
    ):
        statuses, values, axes, decimal = licensed[8 + 4 * index : 12 + 4 * index]
        assert statuses == "statuses" + " 4" * len(points), statuses
        bits = [int(b) for b in values.split()[1:]]
        assert bits == [
            struct.unpack("<q", struct.pack("<d", float(v)))[0] & mask
            for v in expected_values
        ], index
        axes = json.loads(axes.split(" ", 1)[1])
        assert [axes["helicity_ids"], axes["color_ids"]] == ids
        assert decimal == "decimal null"
    references = (
        runtime.evaluate_resolved(
            exact_points,
            mu_r2=exact_scales,
            color_resolution="amplitude-basis",
            arithmetic="double-double",
        ),
        runtime.evaluate(
            exact_points,
            mu_r2=exact_scales,
            helicities=["h1"],
            arithmetic="arbitrary",
            precision=32,
        ),
    )
    for index, reference in enumerate(references):
        decimal = json.loads(extended[8 + 4 * index + 3].split(" ", 1)[1])
        cells = (
            [c for v in reference.values for sector in v for row in sector for c in row]
            if index == 0
            else [sector for v in reference.values for sector in v]
        )
        values = [x for c in cells for x in (c.finite, c.single_pole, c.double_pole)]
        assert [None if x is None else Decimal(x) for x in decimal["virtual"]] == [
            None if x is None else Decimal(x) for x in values
        ], index
