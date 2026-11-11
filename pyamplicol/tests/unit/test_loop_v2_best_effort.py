# SPDX-License-Identifier: 0BSD
"""The best available value of a refused point (m-lit ux-besteffort; USER
2026-11-11 items 7-9).

With the reduction backends' own acceptance checks on (the default), an
explicitly selected backend and no configured alternative left, a point the
backend refuses is returned by default (``on_failure="best-effort"``) with
the value the same backend gives WITHOUT its own checks
(``reduction_unchecked=True``, bitwise), lane status ``"best-effort"`` (never
a success), the refusal kept and an uncertainty: the call's own stability
estimate when it ran checks, else a rotation re-test of that point alone; the
result's ``warnings`` metadata lists it. Nothing is printed, logged or warned
by default; ``emit_warnings()`` is the opt-in single Python warning;
``"return"`` keeps the failed lanes without a value; ``"raise"`` raises.
"""

from __future__ import annotations

import json
import math
import warnings
from decimal import Decimal
from fractions import Fraction

import pytest

from tests.unit.loop_v2_induced_observables_rows import generate


def _tirust(native) -> bool:
    return "tirust" in json.loads(native.reduction_backends())


@pytest.fixture(scope="module")
def aag(loop_sm_ufo, loop_native, tmp_path_factory):
    """g g > a a g (LI-33) and its first frozen stream points: they include
    points TIRust refuses structurally at the default acceptance."""

    from tools.loop_benchmark import streams

    if not _tirust(loop_native):
        pytest.skip("the staged native extension is built without TIRust")
    artifact = generate(tmp_path_factory.mktemp("aag"), "li_gg_aag", "g g > a a g")
    stream = streams.load("li_gg_aag")
    count = 6
    points = [
        [[str(Decimal(float(Fraction(c)))) for c in leg] for leg in point]
        for point in stream["points"][:count]
    ]
    scales = [str(Decimal(float(Fraction(mu) ** 2))) for mu in stream["mu_r"][:count]]
    return artifact, points, scales


# ------------------------------------------------------------ the model


def test_the_failure_policy_has_one_definition_and_best_effort_is_the_default():
    from pyamplicol.config.models import (
        DEFAULT_ON_FAILURE,
        ON_FAILURE,
        EvaluationConfig,
    )

    assert ON_FAILURE == ("best-effort", "return", "raise")
    assert DEFAULT_ON_FAILURE == "best-effort"
    assert EvaluationConfig().on_failure is None


def test_a_best_effort_record_states_its_refusal_and_uncertainty():
    from pyamplicol.api.virtual_results import (
        VirtualBestEffort,
        VirtualLaneDiagnostic,
        VirtualReduction,
        best_effort_detail,
    )

    record = VirtualBestEffort("TIRust Unstable: x", 3e-4, "rotation", 2e-3)
    assert "3.00e-04" in best_effort_detail(record)
    none = VirtualBestEffort("TIRust Unstable: x", None, "none")
    assert "no uncertainty could be estimated" in best_effort_detail(none)
    with pytest.raises(ValueError):
        VirtualBestEffort("x", None, "rotation")
    with pytest.raises(ValueError):
        VirtualBestEffort("", 1e-3, "rotation")
    reduction = VirtualReduction(
        "tirust", "tirust", "best effort", unchecked=True, best_effort=record
    )
    lane = VirtualLaneDiagnostic(
        "best-effort", "double", 53, False, "d", reduction=reduction
    )
    assert lane.has_value and not lane.valid and lane.best_effort is record
    with pytest.raises(ValueError):
        VirtualLaneDiagnostic("success", "double", 53, False, "d", reduction=reduction)
    with pytest.raises(ValueError):
        VirtualReduction("tirust", "tirust", "x", unchecked=False, best_effort=record)


def test_numerical_warnings_summarise_the_lanes_and_emit_only_on_request():
    from pyamplicol.api.result_warnings import (
        NumericalAccuracyWarning,
        ResultWarning,
        emit,
        merge_warnings,
        numerical_warnings,
        summary_line,
    )
    from pyamplicol.api.virtual_results import (
        VirtualBestEffort,
        VirtualFailureDiagnostic,
        VirtualLaneDiagnostic,
        VirtualReduction,
    )

    best = VirtualReduction(
        "tirust",
        "tirust",
        "r",
        unchecked=True,
        best_effort=VirtualBestEffort("refused A", 1e-3, "rotation"),
    )
    lanes = (
        VirtualLaneDiagnostic("success", "double", 53, False, "ok"),
        VirtualLaneDiagnostic("best-effort", "double", 53, False, "b", reduction=best),
        VirtualLaneDiagnostic(
            "failed",
            "double",
            53,
            False,
            "f",
            failures=(
                VirtualFailureDiagnostic(2, None, None, None, "v2", "no value B"),
            ),
        ),
        VirtualLaneDiagnostic("best-effort", "double", 53, False, "b", reduction=best),
    )
    found = numerical_warnings(lanes, "evaluate")
    assert [(w.kind, w.count, w.points, w.message) for w in found] == [
        ("best-effort", 2, (1, 3), "refused A"),
        ("failed", 1, (2,), "no value B"),
    ]
    assert ResultWarning.from_json(found[0].to_json()) == found[0]
    merged = merge_warnings(found, found[:1])
    assert merged[0].count == 4 and merged[0].points == (1, 3, 1, 3)
    assert summary_line(found).startswith("3 captured warning(s)")
    assert summary_line(()) is None
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert emit(found)
    assert [w.category for w in caught] == [NumericalAccuracyWarning]
    assert not emit(())


# ------------------------------------------------------------ the engine


def _quiet(capfd, call):
    """``call()`` with every Python warning recorded and the process's
    stdout / stderr captured: the evaluation path emits nothing."""

    capfd.readouterr()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = call()
    out, err = capfd.readouterr()
    assert not caught, [str(w.message) for w in caught]
    assert out == "" and err == "", (out[-500:], err[-500:])
    return result


@pytest.mark.loop_physics
def test_refused_points_carry_the_unchecked_value_by_default(aag, capfd):
    """Default policy on explicit TIRust: accepted points bitwise the
    "return" call's, refused points the unchecked TIRust call's values with
    status "best-effort", the refusal and a rotation re-test uncertainty;
    nothing emitted; ``"raise"`` raises; ``"return"`` keeps failed lanes."""

    from pyamplicol import EvaluationError, Runtime

    artifact, points, scales = aag
    options = {"mu_r2": scales, "poles": "none", "reduction": "tirust"}
    runtime = Runtime.load(artifact, process="p")
    returned = runtime.evaluate(points, on_failure="return", **options)
    refused = [p for p, ok in enumerate(returned.valid) if not ok]
    assert refused, "the fixture points include refused ones"
    unchecked = runtime.evaluate(
        points, on_failure="return", reduction_unchecked=True, **options
    )
    found = _quiet(capfd, lambda: runtime.evaluate(points, **options))
    assert found.has_warnings
    assert found.statuses == tuple(
        "best-effort" if p in refused else "success" for p in range(len(points))
    )
    for p, lane in enumerate(found.diagnostics):
        if p not in refused:
            assert repr(found.values[p]) == repr(returned.values[p]), p
            assert lane.best_effort is None
            continue
        assert repr(found.values[p]) == repr(unchecked.values[p]), p
        record = lane.best_effort
        assert record is not None and not lane.valid and lane.has_value
        assert record.refusal and "TIRust" in record.refusal
        assert lane.reduction.effective == "tirust" and lane.reduction.unchecked
        if record.uncertainty is None:
            assert record.source == "none"
        else:
            assert record.source == "rotation"
            assert math.isfinite(record.uncertainty) and record.uncertainty >= 0
        assert lane.failures and lane.failures[0].detail == record.refusal
    # The call ran no checks: the public result carries no estimate.
    assert found.stability is None
    (warning,) = found.warnings
    assert (warning.source, warning.kind, warning.points) == (
        "numerical",
        "best-effort",
        tuple(refused),
    )
    with pytest.raises(EvaluationError):
        runtime.evaluate(points, on_failure="raise", **options)


@pytest.mark.loop_physics
def test_a_checked_call_gives_its_own_estimate_as_the_uncertainty(aag):
    from pyamplicol import Runtime
    from pyamplicol.api.virtual_results import VirtualStabilityChecks

    artifact, points, scales = aag
    checks = VirtualStabilityChecks(rotations=1)
    options = {"mu_r2": scales, "poles": "none", "reduction": "tirust"}
    runtime = Runtime.load(artifact, process="p")
    unchecked = runtime.evaluate(
        points, on_failure="return", reduction_unchecked=True, **options
    )
    found = runtime.evaluate(points, stability_checks=checks, **options)
    served = [p for p, s in enumerate(found.statuses) if s == "best-effort"]
    assert served
    for p in served:
        record = found.diagnostics[p].best_effort
        assert repr(found.values[p]) == repr(unchecked.values[p]), p
        assert record.source == "stability-checks"
        assert record.uncertainty == found.stability[p].relative


@pytest.mark.loop_physics
def test_the_resolved_call_serves_best_effort_values(aag):
    from pyamplicol import Runtime

    artifact, points, scales = aag
    options = {"mu_r2": scales, "poles": "none", "reduction": "tirust"}
    runtime = Runtime.load(artifact, process="p")
    returned = runtime.evaluate_resolved(points, on_failure="return", **options)
    found = runtime.evaluate_resolved(points, **options)
    refused = [p for p, ok in enumerate(returned.valid) if not ok]
    assert refused
    for p in range(len(points)):
        if p in refused:
            assert found.statuses[p] == "best-effort"
            assert found.values[p] is not None
        else:
            assert repr(found.values[p]) == repr(returned.values[p])


# ------------------------------------------------------------ the SDKs

BEST_EFFORT_STATUS = 5
RUNTIME_ERROR = 3

CLIENT_CPP = r"""
// ./client artifact process input.txt: explicit TIRust under the default
// failure policy through the C++ wrapper: statuses, uncertainties (bits)
// and the diagnostics, one line each.
#include "rusticol_virtual.hpp"
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

int main(int argc, char **argv) {
  if (argc < 4) return 2;
  std::ifstream in(argv[3]);
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
    rusticol::VirtualRuntime runtime(argv[1], std::string(argv[2]));
    runtime.set_poles("none");
    runtime.set_reduction("tirust");
    auto result = runtime.evaluate(momenta, points, scales);
    std::printf("statuses");
    for (auto status : result.statuses()) std::printf(" %u", status);
    std::printf("\nuncertainties");
    for (double value : result.uncertainties()) std::printf(" %a", value);
    std::printf("\ndiagnostics %s\n", result.diagnostics_json().c_str());
    runtime.set_on_failure("raise");
    try {
      runtime.evaluate(momenta, points, scales);
      std::printf("strict no error\n");
    } catch (const rusticol::VirtualError &error) {
      std::printf("strict error %d\n", error.status());
    }
  } catch (const rusticol::VirtualError &error) {
    std::printf("error %d %s\n", error.status(), error.what());
    return 1;
  }
  return 0;
}
"""

CLIENT_FORTRAN = r"""
program best_effort_client
  use, intrinsic :: iso_c_binding
  use rusticol_virtual
  implicit none
  character(len=4096) :: artifact, process, path
  integer :: unit, points, legs, p, l, i
  integer(c_int64_t), allocatable :: bits(:)
  real(c_double), allocatable :: momenta(:,:,:), scales(:)
  type(virtual_runtime) :: runtime
  type(virtual_evaluation) :: result, strict
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
  call runtime%set_poles('none', ierr=ierr)
  if (ierr /= VIRTUAL_OK) stop 2
  call runtime%set_reduction('tirust', ierr=ierr)
  if (ierr /= VIRTUAL_OK) stop 3
  call runtime%evaluate(momenta, scales, result, ierr=ierr, error_message=message)
  if (ierr /= VIRTUAL_OK) stop 4
  write(*, '(a,*(1x,i0))') 'statuses', result%statuses
  write(*, '(a,*(1x,i0))') 'uncertainties', &
    transfer(result%uncertainties, 1_c_int64_t, size(result%uncertainties))
  call runtime%set_on_failure('raise', ierr=ierr)
  if (ierr /= VIRTUAL_OK) stop 5
  call runtime%evaluate(momenta, scales, strict, ierr=ierr)
  write(*, '(a,1x,i0)') 'strict error', ierr
  call runtime%close()
end program
"""

CLIENT_RUST = """
#[path = "@WRAPPER@"]
#[allow(dead_code)]
mod rusticol;
use rusticol::virtual_amplitudes::{PointStatus, Runtime};

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
    runtime.set_poles("none")?;
    runtime.set_reduction("tirust")?;
    let result = runtime.evaluate(&momenta, points, None, &scales, false, None)?;
    let codes: Vec<String> = result
        .statuses
        .iter()
        .map(|s| (if *s == PointStatus::BestEffort { "5" } else { "x" }).to_owned())
        .collect();
    println!("statuses {}", codes.join(" "));
    let bits: Vec<String> = result
        .uncertainties
        .iter()
        .map(|u| (u.to_bits() as i64).to_string())
        .collect();
    println!("uncertainties {}", bits.join(" "));
    runtime.set_on_failure("raise")?;
    let strict = runtime.evaluate(&momenta, points, None, &scales, false, None);
    println!("strict error {}", i32::from(strict.is_err()));
    Ok(())
}
"""


def _sdk_call(sdk, handle, points, scales, *, policy=None, unchecked=None):
    from tests.unit.test_loop_v2_sdk_induced import _evaluate

    if policy is not None:
        assert sdk.rusticol_virtual_set_on_failure(handle, policy) == 0
    if unchecked is not None:
        assert sdk.rusticol_virtual_set_reduction_unchecked(handle, unchecked) == 0
    return _evaluate(sdk, handle, points, scales)


def _uncertainties(sdk, handle, points, scales) -> list[float]:
    import ctypes

    from tests.unit.test_loop_v2_sdk_induced import _Request

    flat = [c for point in points for p in point for c in p]
    momenta = (ctypes.c_double * len(flat))(*flat)
    mu = (ctypes.c_double * len(scales))(*scales)
    request = _Request(momenta, len(flat), len(points), None, mu, len(scales))
    result = ctypes.c_void_p()
    assert (
        sdk.rusticol_virtual_evaluate(
            handle, ctypes.byref(request), ctypes.byref(result)
        )
        == 0
    )
    try:
        out = (ctypes.c_double * len(points))()
        function = sdk.rusticol_virtual_result_copy_uncertainties
        function.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
        assert function(result, out, len(points)) == 0
        return list(out)
    finally:
        sdk.rusticol_virtual_result_free(result)


@pytest.mark.loop_physics
def test_every_sdk_returns_refused_points_with_their_best_effort_value(
    aag, loop_native, tmp_path
):
    """The C ABI and the C++, Fortran and packaged Rust wrappers on explicit
    TIRust: by default the call succeeds, the refused points carry status
    BEST_EFFORT with the SDK's own unchecked TIRust values (bitwise) and the
    Python API's best-effort values, a finite uncertainty (rotation re-test)
    and the refusal in the diagnostics; "return" keeps them FAILED; "raise"
    is a runtime error; the accepted points keep their values."""

    import ctypes
    import subprocess

    from pyamplicol import Runtime
    from tests.unit.test_loop_v2_sdk_extras import (
        SDK,
        WRAPPER,
        _compiler,
        _error,
        _sdk,
        _staged_sdk,
    )
    from tests.unit.test_loop_v2_sdk_induced import (
        FAILED_STATUS,
        _evaluate,
        _open,
        _python_flat,
        _same,
    )
    from tests.unit.test_loop_v2_unchecked import _feed

    artifact, exact_points, exact_scales = aag
    points = [[[float(c) for c in leg] for leg in point] for point in exact_points]
    scales = [float(Fraction(s)) for s in exact_scales]
    python = Runtime.load(artifact, process="p").evaluate(
        exact_points, mu_r2=exact_scales, poles="none", reduction="tirust"
    )
    refused = [p for p, s in enumerate(python.statuses) if s == "best-effort"]
    assert refused
    sdk = _sdk()
    sdk.rusticol_virtual_set_on_failure.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    sdk.rusticol_virtual_set_reduction_unchecked.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
    ]
    handle = _open(sdk, artifact)
    try:
        assert sdk.rusticol_virtual_set_poles(handle, b"none") == 0, _error(sdk)
        assert sdk.rusticol_virtual_set_reduction(handle, b"tirust") == 0
        found = _evaluate(sdk, handle, points, scales)
        assert [p for p, s in enumerate(found["statuses"]) if s == 5] == refused
        assert FAILED_STATUS not in found["statuses"]
        assert _same(found["virtual"], _python_flat(python))
        record = found["diagnostics"]
        assert sorted(int(k) for k in record["best_effort"]) == refused
        for entry in record["best_effort"].values():
            assert "TIRust" in entry["refusal"] and entry["source"] == "rotation"
            assert math.isfinite(entry["uncertainty"])
        (warning,) = record["warnings"]
        assert warning["kind"] == "best-effort" and warning["count"] == len(refused)
        assert record["on_failure"] == "best-effort"
        uncertainties = _uncertainties(sdk, handle, points, scales)
        assert [math.isfinite(u) for u in uncertainties] == [
            p in refused for p in range(len(points))
        ]
        unchecked = _sdk_call(sdk, handle, points, scales, unchecked=1)
        assert _same(found["virtual"], unchecked["virtual"])
        assert sdk.rusticol_virtual_set_reduction_unchecked(handle, 0) == 0
        returned = _sdk_call(sdk, handle, points, scales, policy=b"return")
        assert [p for p, s in enumerate(returned["statuses"]) if s == 3] == refused
        assert sdk.rusticol_virtual_set_on_failure(handle, b"raise") == 0
        request_points = points
        flat = [c for point in request_points for p in point for c in p]
        from tests.unit.test_loop_v2_sdk_induced import _Request

        momenta = (ctypes.c_double * len(flat))(*flat)
        mu = (ctypes.c_double * len(scales))(*scales)
        request = _Request(momenta, len(flat), len(points), None, mu, len(scales))
        result = ctypes.c_void_p()
        status = sdk.rusticol_virtual_evaluate(
            handle, ctypes.byref(request), ctypes.byref(result)
        )
        assert status == RUNTIME_ERROR and not result.value
        assert "on_failure" in _error(sdk)
        assert sdk.rusticol_virtual_set_on_failure(handle, b"warn") != 0
    finally:
        sdk.rusticol_virtual_free(handle)

    build = _staged_sdk()
    link = [f"-L{build}", "-lrusticol_loop_sdk", f"-Wl,-rpath,{build}", "-lm"]
    (tmp_path / "client.cpp").write_text(CLIENT_CPP)
    (tmp_path / "client.f90").write_text(CLIENT_FORTRAN)
    (tmp_path / "client.rs").write_text(CLIENT_RUST.replace("@WRAPPER@", str(WRAPPER)))
    compile_steps = [
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
    ]
    for step in compile_steps:
        subprocess.run(step, check=True, cwd=tmp_path)
    for language in ("cpp", "fortran", "rust"):
        path = tmp_path / f"{language}.txt"
        path.write_text(_feed(points, scales, bits=language != "cpp"))
        run = subprocess.run(
            [str(tmp_path / language), str(artifact), "p", str(path)],
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
        )
        assert run.returncode == 0, run.stdout[-2000:] + run.stderr[-2000:]
        lines = {
            line.split(" ", 1)[0]: line.split(" ", 1)[1]
            for line in run.stdout.splitlines()
            if " " in line
        }
        codes = lines["statuses"].split()
        assert [p for p, c in enumerate(codes) if c == "5"] == refused, language
        assert lines["strict"].split()[-1] != "0", language
        values = lines["uncertainties"].split()
        if language == "cpp":
            found_u = [float.fromhex(v) if "nan" not in v else math.nan for v in values]
        else:
            import struct

            found_u = [
                struct.unpack("<d", struct.pack("<q", int(v)))[0] for v in values
            ]
        assert [math.isfinite(u) for u in found_u] == [
            p in refused for p in range(len(points))
        ], language
        for p in refused:
            assert found_u[p] == uncertainties[p], language
