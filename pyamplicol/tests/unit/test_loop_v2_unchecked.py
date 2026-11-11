# SPDX-License-Identifier: 0BSD
"""The unchecked reduction setting (m-lit lit-unchecked; USER 2026-11-08
~21:14 / ~22:34, DECISIONS 22:34 (1)): a public, documented setting, off by
default, that drops the reduction backends' own acceptance and refusal
checks.

* One definition (Rust ``reduction_policy``: ``DEFAULT_UNCHECKED`` off, one
  validator, one refusal text, the record text), read by the configuration,
  the CLI, the Python API and the SDKs.
* TIRust U-choice: its gates and method choice as method selection, no
  acceptance pass, the gate-0 PV last resort for what the method choice
  refuses (g g > a a g's structurally refused stream points); the values of
  the points the checked default serves bitwise unchanged; ``"auto"`` a pure
  TIRust call where nothing fails; NinjaRS with zero stability thresholds.
* The C ABI setter and the C++ / Fortran / packaged Rust wrappers; no
  Symbolica licence needed.
* pyAmpliCol's own checks (monitoring, rescue) stay their own settings.

The Rust side (no acceptance-frame row requested, the last resort's count,
the extended carriers) is ``rust/loop-reduction/tests/v2_tir.rs``.
"""

from __future__ import annotations

import ctypes
import json
import math
import subprocess
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

import pytest

from tests.unit.loop_v2_induced_observables_rows import generate
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
    STATUS_INVALID_ARGUMENT,
    SUCCESS_STATUS,
    _evaluate,
    _open,
    _python_flat,
    _same,
)

ALL = ("double", "double-double", "arbitrary")
# The TIR counts of the native report (``v2_tir::take_statistics``).
CHECK_ROWS = "TIR method-choice check rows"
UNVERIFIED = "TIR method-choice lanes refused by their check"
LAST_RESORT = "TIR last-resort rows"
STRUCTURAL = "refused by the gates, then by the method choice"


def _tirust(native) -> bool:
    return "tirust" in json.loads(native.reduction_backends())


# ------------------------------------------------------------ the setting


@pytest.mark.loop_fast
def test_the_unchecked_setting_has_one_definition_and_one_validator(loop_native):
    from pyamplicol.api.virtual_results import VirtualReduction
    from pyamplicol.config import ConfigurationError
    from pyamplicol.config.models import EvaluationConfig, reduction_unchecked
    from pyamplicol.loops.v2.reduction import (
        current_unchecked,
        default_unchecked,
        executor_options,
        policy,
        running,
    )

    record = policy()["unchecked"]
    assert record["default"] is False and default_unchecked() is False
    assert "boolean" in record["refusal"]
    assert record["record"].startswith("reduction_unchecked: ")
    assert reduction_unchecked(True) is True
    assert reduction_unchecked(False) is False
    for bad in (1, 0, "true", None, 1.0):
        with pytest.raises(ValueError) as refused:
            reduction_unchecked(bad)
        assert str(refused.value) == record["refusal"]
        with pytest.raises(ValueError, match="boolean"):
            VirtualReduction("auto", "tirust", "x", unchecked=bad)
    assert VirtualReduction("auto", "tirust", "x").unchecked is False
    assert EvaluationConfig().reduction_unchecked is False
    assert EvaluationConfig(reduction_unchecked=True).reduction_unchecked is True
    with pytest.raises(ConfigurationError, match=r"evaluation\.reduction_unchecked"):
        EvaluationConfig(reduction_unchecked="yes")
    # The executors' options: nothing for a checked NinjaRS call (what it
    # passed before the setting existed), the flag otherwise; a nested
    # context (a cross-backend check's) keeps it.
    with running("ninjars", None):
        assert executor_options() == {} and current_unchecked() is False
    with running("ninjars", None, True):
        assert executor_options() == {"unchecked": True}
        with running("tirust"):
            assert executor_options() == {"reduction": "tirust", "unchecked": True}
    with running("tirust", None, False):
        assert executor_options() == {"reduction": "tirust"}


@pytest.mark.loop_fast
def test_the_cli_and_the_run_card_set_the_unchecked_setting():
    from pyamplicol.cli.parser import build_parser

    parser = build_parser()
    options = vars(parser.parse_args(["evaluate", "artifact", "--reduction-unchecked"]))
    assert options["evaluation.reduction_unchecked"] is True
    options = vars(
        parser.parse_args(["evaluate", "artifact", "--no-reduction-unchecked"])
    )
    assert options["evaluation.reduction_unchecked"] is False
    assert "evaluation.reduction_unchecked" not in vars(
        parser.parse_args(["evaluate", "artifact"])
    )


@pytest.mark.loop_fast
def test_auto_keeps_its_rule_under_the_setting(loop_native):
    """The backend choice is ``resolve``'s whatever the setting (each backend
    then in its unchecked form): loop-induced TIRust with the per-point
    NinjaRS fallback, Born-interfered NinjaRS, a TIR-only artifact TIRust
    without a fallback, a native without TIRust NinjaRS with its reason; the
    choice carries the call's setting into its record."""

    from dataclasses import replace

    from pyamplicol.loops.v2.reduction import SUMMED, resolve

    class Native:
        def __init__(self, served):
            self._served = served

        def reduction_backends(self):
            return {name: list(carriers) for name, carriers in self._served.items()}

    both = Native({"ninjars": ALL, "tirust": ALL})
    cases = [
        ("loop-induced", both, None, ("tirust", "ninjars")),
        ("virtual", both, None, ("ninjars", None)),
        ("loop-induced", both, {"tirust"}, ("tirust", None)),
        ("loop-induced", Native({"ninjars": ALL}), None, ("ninjars", None)),
    ]
    for kind, native, admitted, expected in cases:
        choice = resolve(
            "auto",
            contribution=kind,
            carrier="double",
            native=native,
            route=SUMMED,
            admitted=admitted,
        )
        assert (choice.effective, choice.fallback) == expected
        assert choice.unchecked is False
        unchecked = replace(choice, unchecked=True)
        assert (unchecked.effective, unchecked.fallback, unchecked.reason) == (
            choice.effective,
            choice.fallback,
            choice.reason,
        )
    off = resolve(
        "auto",
        contribution="loop-induced",
        carrier="double",
        native=Native({"ninjars": ALL}),
        route=SUMMED,
    )
    assert "TIRust, the loop-induced default, is unavailable" in off.reason


# ------------------------------------------- g g > a a g at its stream points


@pytest.fixture(scope="module")
def aag(loop_sm_ufo, loop_native, tmp_path_factory):
    """g g > a a g (LI-33) through the public Generator, and its first frozen
    stream points (exact binary64 texts) with their mu_R^2: the first points
    include ones TIRust refuses structurally (its gates, then its method
    choice) at every acceptance (lit-tirust-default)."""

    from tools.loop_benchmark import streams

    artifact = generate(tmp_path_factory.mktemp("aag"), "li_gg_aag", "g g > a a g")
    stream = streams.load("li_gg_aag")
    count = 6
    points = [
        [[str(Decimal(float(Fraction(c)))) for c in leg] for leg in point]
        for point in stream["points"][:count]
    ]
    scales = [str(Decimal(float(Fraction(mu) ** 2))) for mu in stream["mu_r"][:count]]
    return artifact, points, scales


def _counts(runtime) -> dict:
    """The native report's counts of the runtime's latest summed call (a
    counters-only read, outside any timed path)."""

    return dict(runtime._backend._executor.last_report["counts"])


@pytest.mark.loop_physics
def test_unchecked_tirust_serves_every_point_and_keeps_the_checked_values(
    aag, loop_native
):
    """Done 2 / 4 of lit-unchecked on g g > a a g: the checked TIRust call
    fails points structurally ("refused by the gates, then by the method
    choice"); with ``reduction_unchecked=True`` (load-time, the runtime's
    default for every call) every point is served, the method choice's
    refusals by the gate-0 last resort (counted), no acceptance pass is
    computed (no check row, no lane refused by its check), and the points the
    checked call serves keep their values bitwise. ``"auto"`` under the
    setting is a pure TIRust call (every lane on TIRust, no fallback, the
    values bitwise), explicit NinjaRS unchecked keeps NinjaRS's values (it
    refuses nothing here), and a call's ``reduction_unchecked=False``
    overrides the runtime's setting (the checked values and failures)."""

    from pyamplicol import Runtime

    if not _tirust(loop_native):
        pytest.skip("the staged native extension is built without TIRust")
    artifact, points, scales = aag
    options = {"mu_r2": scales, "on_failure": "return", "poles": "none"}
    checked_runtime = Runtime.load(artifact, process="p")
    checked = checked_runtime.evaluate(points, reduction="tirust", **options)
    checked_counts = _counts(checked_runtime)
    assert LAST_RESORT not in checked_counts
    refused = [
        p for p, lane in enumerate(checked.diagnostics) if STRUCTURAL in lane.detail
    ]
    assert refused, [lane.detail[:200] for lane in checked.diagnostics]
    ninjars = checked_runtime.evaluate(points, reduction="ninjars", **options)

    runtime = Runtime.load(artifact, process="p", reduction_unchecked=True)
    found = runtime.evaluate(points, reduction="tirust", **options)
    counts = _counts(runtime)
    assert counts.get(CHECK_ROWS, 0) == 0 and counts.get(UNVERIFIED, 0) == 0
    assert counts[LAST_RESORT] > 0, counts
    assert all(value is not None for value in found.values), [
        lane.detail[:200] for lane in found.diagnostics
    ]
    for p, value in enumerate(checked.values):
        lane = found.diagnostics[p]
        assert (lane.reduction.effective, lane.reduction.unchecked) == ("tirust", True)
        assert all(math.isfinite(float(x[0])) for x in found.values[p])
        if value is not None:
            assert repr(found.values[p]) == repr(value), p
    auto = runtime.evaluate(points, **options)
    assert repr(auto.values) == repr(found.values)
    for lane in auto.diagnostics:
        assert lane.reduction.effective == "tirust", lane.reduction
        assert lane.reduction.fallback == "ninjars" and lane.reduction.unchecked
    explicit = runtime.evaluate(points, reduction="ninjars", **options)
    assert repr(explicit.values) == repr(ninjars.values)
    assert all(lane.reduction.unchecked for lane in explicit.diagnostics)
    again = runtime.evaluate(
        points, reduction="tirust", reduction_unchecked=False, **options
    )
    assert repr(again.values) == repr(checked.values)
    assert [lane.detail for lane in again.diagnostics] == [
        lane.detail for lane in checked.diagnostics
    ]
    assert not any(lane.reduction.unchecked for lane in again.diagnostics)
    inspect = runtime._backend._reduction_summary()
    assert inspect["unchecked"] is True
    assert checked_runtime._backend._reduction_summary()["unchecked"] is False
    # The public loader takes the target as a load-time default too (it was
    # refused as an unknown option before lit-unchecked's loader keywords).
    targeted = Runtime.load(artifact, process="p", reduction_target=1e-3)
    assert targeted._backend._reduction_summary()["target"] == 1e-3


@pytest.mark.loop_physics
def test_pyamplicols_checks_stay_their_own_settings_under_the_setting(aag, loop_native):
    """The combination reading of lit-unchecked (1)(c): the setting governs
    the reduction backends' own checks only. With pyAmpliCol's stability
    checks every point is served (unchecked) and estimated; under a rescue
    stack every level reduces unchecked and records it."""

    from pyamplicol import Runtime
    from pyamplicol.api.virtual_results import (
        VirtualRescueLevel,
        VirtualRescuePolicy,
        VirtualStabilityChecks,
    )

    if not _tirust(loop_native):
        pytest.skip("the staged native extension is built without TIRust")
    artifact, points, scales = aag
    runtime = Runtime.load(artifact, process="p")
    checked = runtime.evaluate(
        points[:3],
        mu_r2=scales[:3],
        reduction="tirust",
        reduction_unchecked=True,
        poles="none",
        stability_checks=VirtualStabilityChecks(1, True, rescaled=True),
    )
    assert all(estimate is not None for estimate in checked.stability)
    assert all(lane.reduction.unchecked for lane in checked.diagnostics)
    stack = VirtualRescuePolicy(
        (
            VirtualRescueLevel("double", reduction="tirust"),
            VirtualRescueLevel("double-double", reduction="tirust"),
        )
    )
    rescued = runtime.evaluate(
        points[:2],
        mu_r2=scales[:2],
        reduction_unchecked=True,
        poles="none",
        stability=stack,
        on_failure="return",
    )
    for lane in rescued.diagnostics:
        assert lane.reduction is None or lane.reduction.unchecked, lane
    for record in rescued.rescue or ():
        for attempt in record.attempts:
            assert attempt.reduction is None or attempt.reduction.unchecked


# ------------------------------------------------------------ the SDKs

UNCHECKED_CLIENT = r"""
// ./client artifact process input.txt: one "auto" evaluation through the
// C++ wrapper with the unchecked setting on: the statuses, the virtual values
// as bit patterns and the diagnostics JSON, one line each.
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
    runtime.set_reduction_unchecked(true);
    auto result = runtime.evaluate(momenta, points, scales);
    std::printf("statuses");
    for (auto status : result.statuses()) std::printf(" %u", status);
    std::printf("\nvirtual");
    for (double value : result.virtual_values()) {
      std::uint64_t bits;
      std::memcpy(&bits, &value, sizeof bits);
      std::printf(" %" PRIu64, bits);
    }
    std::printf("\ndiagnostics %s\n", result.diagnostics_json().c_str());
  } catch (const rusticol::VirtualError &error) {
    std::printf("error %d %s\n", error.status(), error.what());
    return 1;
  }
  return 0;
}
"""

UNCHECKED_FORTRAN = r"""
program unchecked_client
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
  call runtime%set_poles('none', ierr=ierr)
  if (ierr /= VIRTUAL_OK) stop 2
  call runtime%set_reduction_unchecked(.true., ierr=ierr, error_message=message)
  if (ierr /= VIRTUAL_OK) stop 3
  call runtime%evaluate(momenta, scales, result, ierr=ierr, error_message=message)
  if (ierr /= VIRTUAL_OK) stop 4
  write(*, '(a)') result%diagnostics_json
  call runtime%close()
end program
"""

UNCHECKED_RUST = """
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
    runtime.set_poles("none")?;
    runtime.set_reduction_unchecked(true)?;
    let result = runtime.evaluate(&momenta, points, None, &scales, false, None)?;
    println!("{}", result.diagnostics_json);
    Ok(())
}
"""


def _feed(points, scales, bits: bool) -> str:
    import struct

    def text(value: float) -> str:
        if bits:
            return str(struct.unpack("<q", struct.pack("<d", value))[0])
        return value.hex()

    lines = [f"{len(points)} {len(points[0])}"]
    for point, scale in zip(points, scales, strict=True):
        lines.append(" ".join(text(v) for v in [*(c for p in point for c in p), scale]))
    return "\n".join(lines) + "\n"


@pytest.mark.loop_physics
def test_the_c_abi_and_the_wrappers_set_the_unchecked_setting(
    aag, loop_native, tmp_path
):
    """``rusticol_virtual_set_reduction_unchecked``: 2 and -1 are invalid
    arguments with the one refusal text and change nothing; 1 serves every
    g g > a a g point under "auto" on TIRust without a fallback, bitwise the
    Python API's unchecked values, with ``"unchecked": true`` in the
    diagnostics' and the capabilities' reduction records; 0 restores the
    checked call (its structural refusals fail again, explicit "tirust"). The
    C++, Fortran and packaged Rust wrappers' ``set_reduction_unchecked``
    record the same."""

    from pyamplicol import Runtime
    from pyamplicol.loops.v2.reduction import policy

    if not _tirust(loop_native):
        pytest.skip("the staged native extension is built without TIRust")
    artifact, exact_points, exact_scales = aag
    points = [[[float(c) for c in leg] for leg in point] for point in exact_points]
    scales = [float(Fraction(s)) for s in exact_scales]
    expected = Runtime.load(artifact, process="p").evaluate(
        exact_points,
        mu_r2=exact_scales,
        reduction_unchecked=True,
        poles="none",
        on_failure="return",
    )
    sdk = _sdk()
    setter = sdk.rusticol_virtual_set_reduction_unchecked
    setter.argtypes = [ctypes.c_void_p, ctypes.c_int]
    handle = _open(sdk, artifact)
    try:
        assert sdk.rusticol_virtual_set_poles(handle, b"none") == 0, _error(sdk)
        for bad in (2, -1):
            assert setter(handle, bad) == STATUS_INVALID_ARGUMENT
            assert _error(sdk) == policy()["unchecked"]["refusal"]
        assert setter(handle, 1) == 0, _error(sdk)
        found = _evaluate(sdk, handle, points, scales)
        assert found["statuses"] == [SUCCESS_STATUS] * len(points)
        record = found["diagnostics"]["reduction"]
        assert record["effective"] == "tirust" and record["unchecked"] is True
        assert record["fallback_points"] == 0
        assert found["diagnostics"]["fallbacks"] == {}
        assert _same(found["virtual"], _python_flat(expected))
        capabilities = json.loads(_capabilities(sdk, handle))
        assert capabilities["reduction"]["unchecked"] is True
        assert capabilities["reduction"]["default_unchecked"] is False
        assert setter(handle, 0) == 0
        assert sdk.rusticol_virtual_set_reduction(handle, b"tirust") == 0
        closed = _evaluate(sdk, handle, points, scales)
        # The structural refusals are served best-effort by the default
        # failure policy (m-lit ux-besteffort): status 5, never a success.
        assert 5 in closed["statuses"] and FAILED_STATUS not in closed["statuses"]
        assert closed["diagnostics"]["reduction"]["unchecked"] is False
    finally:
        sdk.rusticol_virtual_free(handle)

    build = _staged_sdk()
    link = [f"-L{build}", "-lrusticol_loop_sdk", f"-Wl,-rpath,{build}", "-lm"]
    (tmp_path / "client.cpp").write_text(UNCHECKED_CLIENT)
    (tmp_path / "client.f90").write_text(UNCHECKED_FORTRAN)
    (tmp_path / "client.rs").write_text(
        UNCHECKED_RUST.replace("@WRAPPER@", str(WRAPPER))
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
        lines = run.stdout.splitlines()
        diagnostics = next(
            line.split(" ", 1)[1] if line.startswith("diagnostics ") else line
            for line in lines
            if line.startswith(("diagnostics ", "{"))
        )
        record = json.loads(diagnostics)["reduction"]
        assert record["effective"] == "tirust" and record["unchecked"], language
        assert record["fallback_points"] == 0, language
        if language == "cpp":
            assert _same(_doubles(lines[1]), found["virtual"])


def _doubles(line: str) -> list[float]:
    """A client's ``virtual`` line (unsigned 64-bit patterns) as doubles."""

    import struct

    return [struct.unpack("<d", struct.pack("<Q", int(b)))[0] for b in line.split()[1:]]


def _capabilities(sdk, handle) -> str:
    needed = ctypes.c_size_t()
    function = sdk.rusticol_virtual_capabilities_json
    assert function(handle, None, ctypes.c_size_t(0), ctypes.byref(needed)) == 0
    buffer = ctypes.create_string_buffer(needed.value)
    assert function(handle, buffer, needed, ctypes.byref(needed)) == 0
    return buffer.value.decode()


@pytest.mark.loop_physics
def test_the_unchecked_sdk_needs_no_symbolica_licence(loop_native, tmp_path):
    """USER 2026-11-08 ~15:25 / ~20:55: no SDK call may need a user licence.
    A fresh process with ``SYMBOLICA_LICENSE`` removed and
    ``PYAMPLICOL_NATIVE_LIBRARY_KEY=off``, while another process holds the
    restricted Symbolica permit, sets the unchecked setting on the shipped
    loop-induced self-test artifact (g g > h) and evaluates it through the C++
    wrapper: its output equals the licensed run's, its values the Python
    API's unchecked values bitwise."""

    import importlib.util
    import os

    from pyamplicol import Runtime
    from pyamplicol._internal import selftest_virtual_li as selftest

    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location(
        "loop_artifact_portability", root / "tools" / "loop_artifact_portability.py"
    )
    assert spec is not None and spec.loader is not None
    portability = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(portability)
    artifact = selftest.fixture_root() / "artifact"
    expected = json.loads((selftest.fixture_root() / "expected.json").read_text())
    points = [
        [[float(Decimal(c)) for c in p] for p in point] for point in expected["points"]
    ]
    scales = [float(Decimal(s)) for s in expected["mu_r2"]]
    exact_points = [[[str(Decimal(c)) for c in p] for p in point] for point in points]
    exact_scales = [str(Decimal(s)) for s in scales]
    client = UNCHECKED_CLIENT.replace(
        "rusticol::VirtualRuntime runtime(argv[1], std::string(argv[2]));",
        "std::ifstream file(argv[4]);\n"
        "    std::string parameters((std::istreambuf_iterator<char>(file)),\n"
        "                           std::istreambuf_iterator<char>());\n"
        "    rusticol::VirtualRuntime runtime(argv[1], std::string(argv[2]), "
        "parameters);",
    ).replace("#include <fstream>", "#include <fstream>\n#include <iterator>")
    client = client.replace("if (argc < 4) return 2;", "if (argc < 5) return 2;")
    build = _staged_sdk()
    (tmp_path / "client.cpp").write_text(client)
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
    (tmp_path / "parameters.json").write_text(
        json.dumps({key: [value, "0"] for key, value in expected["parameters"].items()})
    )
    (tmp_path / "input.txt").write_text(_feed(points, scales, bits=False))
    command = [
        str(tmp_path / "client"),
        str(artifact),
        selftest.NAME,
        str(tmp_path / "input.txt"),
        str(tmp_path / "parameters.json"),
    ]

    def run(env: dict[str, str]) -> list[str]:
        done = subprocess.run(
            command, capture_output=True, text=True, timeout=600, check=False, env=env
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
    statuses, values, diagnostics = licensed
    assert statuses == "statuses" + " 4" * len(points), statuses
    assert json.loads(diagnostics.split(" ", 1)[1])["reduction"]["unchecked"] is True
    result = Runtime.load(
        artifact, process=selftest.NAME, model_parameters=expected["parameters"]
    ).evaluate(exact_points, mu_r2=exact_scales, poles="none", reduction_unchecked=True)
    assert _same(_doubles(values), _python_flat(result))


# ------------------------------------------------- Born-interfered: NinjaRS


@pytest.mark.loop_physics
def test_born_interfered_auto_is_ninjars_unchecked(loop_native):
    """A Born-interfered process (the shipped self-test, e- e+ > e- e+ at
    O(alpha)): ``"auto"`` with the setting is NinjaRS in its unchecked form
    (zero stability thresholds), recorded; it refuses nothing here, so its
    values are bitwise the checked call's (a threshold only refuses)."""

    from pyamplicol import Runtime
    from pyamplicol._internal import selftest_virtual_ew as selftest

    artifact = selftest.fixture_root() / "artifact"
    expected = json.loads((selftest.fixture_root() / "expected.json").read_text())
    runtime = Runtime.load(artifact, process=selftest.NAME, mute_warnings=True)
    options = {"mu_r2": expected["mu_r2"], "poles": "none"}
    checked = runtime.evaluate(expected["points"], **options)
    found = runtime.evaluate(expected["points"], reduction_unchecked=True, **options)
    assert repr(found.values) == repr(checked.values)
    for lane in found.diagnostics:
        assert (lane.reduction.effective, lane.reduction.unchecked) == ("ninjars", True)
        assert lane.reduction.fallback is None
    assert not any(lane.reduction.unchecked for lane in checked.diagnostics)
