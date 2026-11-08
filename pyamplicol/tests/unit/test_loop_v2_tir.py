# SPDX-License-Identifier: 0BSD
"""The TIR backend through the native executor and the public API (m-lit
lit-tir, PLAN §3.3 and §2.b.2-§2.b.6).

* the native extension's capability query (``_loop.reduction_backends``)
  reports TIRust at every carrier exactly when it links TIRust (m-lit
  lit-tir-mp);
* T2: the raw reduction lanes of generated Born-interfered slices (g g > g g,
  u u~ > d d~ g, a complex-mass b b~ > w+ w-) reduced by TIRust agree with
  NinjaRS's within 10 times the larger of the two backends' re-test
  estimates (PLAN §4 L6), all Laurent orders, plus 1e-12 of the lane's
  largest coefficient (``ROUNDING``: where both re-tests are blind, e.g.
  NinjaRS's reversed frame on a bubble, agreement to rounding passes too;
  inside the precision-lax rule); in the default settings the binary64 rows
  the conditioning gates refuse take TIRust's method choice, every such
  lane accepted in the rescaled frame (m-lit lit-tir-mp, ruling 21:50 R4):
  a point with a lane refused there fails closed with TIRust's reason; a top
  in its own gauge loop (g g > t t~), whose invariants need the supplied
  S-matrix (lit-spike-tir F1), is served on the exact on-shell branch and
  agrees with NinjaRS within L6 in binary64 and to the double-double target
  in double-double (m-lit lit-tir-mp);
* T6: the family threads leave TIRust's values bitwise;
* the executor's TIRust validation settings (``prepare_reduction``) rebuild
  its backend (the method choice off: gate-refused rows fail closed, no
  method-choice row) and reset to the defaults bitwise; NinjaRS refuses
  them;
* the public API serves an explicit ``reduction="tirust"`` on the summed and
  the resolved routes (values reduced by TIRust: within the lax precision
  of NinjaRS's and not bitwise equal to them), records it and keeps
  ``"auto"`` on NinjaRS (with its reason); an explicit ``"tirust"`` at
  double-double and arbitrary precision, and on every level of a rescue
  ladder, stays on TIRust (m-lit lit-tir-mp);
* T8: binary64 TIRust touches no Symbolica state: a fresh interpreter
  without the licence and with pyAmpliCol's native library key disabled
  evaluates the shipped loop-induced and Born-interfered self-test artifacts
  on TIRust while another process holds the restricted Symbolica permit,
  with the licensed run's bits (the same on NinjaRS is the baseline).
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "fixtures" / "loop"
_tool_spec = importlib.util.spec_from_file_location(
    "loop_artifact_portability", ROOT / "tools" / "loop_artifact_portability.py"
)
assert _tool_spec is not None and _tool_spec.loader is not None
portability = importlib.util.module_from_spec(_tool_spec)
_tool_spec.loader.exec_module(portability)

NO_NATIVE_KEY = {"PYAMPLICOL_NATIVE_LIBRARY_KEY": "off"}
PARAMETERS = {"aS": "0.118", "MT": "173"}
FIXED = {"MB": "0"}
# L6 (PLAN §4): |TIR - NinjaRS| <= 10 max(E_N, E_T), E the backend's re-test
# deviation, with lit-spike-tir's floor of 4 units of binary64 roundoff of
# the lane's largest Laurent coefficient; plus ROUNDING of that coefficient
# where both re-tests are blind (module docstring; inside the lax rule).
MARGIN = 10.0
FLOOR = 4 * 2.0**-53
ROUNDING = 1e-12
# The lax precision of the public-API comparison (relative to the largest
# entry; below ~1e-5 is not a finding).
LAX = 1e-6
# The double-double target of the two backends on the same inputs (DESIGN
# §11.1's DD = MPFR target; R5: measured and recorded).
DOUBLE_DOUBLE = 1e-24


def _tirust(native) -> bool:
    return bool(native.tirust_available())


@pytest.mark.loop_fast
def test_the_capability_query_reports_tirust_at_every_carrier(loop_native):
    from pyamplicol.loops.v2.reduction import CARRIERS, native_backends

    found = native_backends(loop_native)
    assert found["ninjars"] == frozenset(CARRIERS)
    if _tirust(loop_native):
        assert found["tirust"] == frozenset(CARRIERS)
        # Which families TIRust serves is its own capability query: beyond
        # its former fixed cap of twelve propagators, refused only past what
        # the query reports (no constant of pyAmpliCol's).
        assert loop_native.reducer_rank_refusal(13, 2, "tirust") is None
        refusal = loop_native.reducer_rank_refusal(65, 0, "tirust")
        assert "TIRust cannot reduce 65 propagators" in refusal, refusal
        # TIRust has no rank <= N + 2 limit (m-lit lit-eft): a family beyond
        # NinjaRS's rank is admitted through TIRust alone (a TIR-only
        # artifact), up to TIRust's own representation.
        assert loop_native.reducer_rank_refusal(4, 7, "tirust") is None
        assert "rank <= N+2" in loop_native.reducer_rank_refusal(4, 7, "ninjars")
        assert loop_native.family_rank_refusal(4, 7) is None
        refused = loop_native.family_rank_refusal(1, 1_000_000)
        assert "NinjaRS" in refused and "TIRust cannot reduce" in refused, refused
    else:
        assert "tirust" not in found
        assert "no TIRust" in loop_native.family_rank_refusal(4, 7)


@pytest.mark.loop_fast
def test_tirust_plans_are_built_before_the_first_point_and_refused_typed():
    """``reduction.prepare`` (called by the executor call sites once per
    batch, before ``evaluate``) builds TIRust's plans when the context runs
    TIRust, nothing for NinjaRS, and turns TIRust's typed plan-construction
    refusal into ``ReductionUnavailable``."""

    from pyamplicol import ReductionUnavailable
    from pyamplicol.loops.v2.reduction import prepare, running

    class Executor:
        def __init__(self, refusal=None):
            self.calls, self.refusal = [], refusal

        def prepare_reduction(self, reduction):
            self.calls.append(reduction)
            if self.refusal:
                raise ValueError(self.refusal)
            return [(4, 4)]

    executor = Executor()
    prepare(executor)
    with running("ninjars"):
        prepare(executor)
    assert executor.calls == []
    with running("tirust"):
        prepare(executor)
    assert executor.calls == ["tirust"]
    refused = Executor("TIRust cannot reduce 13 propagators at rank 0: too many")
    with running("tirust"), pytest.raises(ReductionUnavailable, match="13 propagators"):
        prepare(refused)


# -------------------------------------------------------------------- T2, T6


def _reference_points(name: str):
    data = json.loads(
        (ROOT / "tools" / "loop_reference" / f"{name}.json").read_text(),
        parse_float=str,
    )
    points = data["points"]
    return points, ["1000000"] * len(points)


def _fixture_points(source: str):
    from decimal import Decimal

    data = json.loads((FIXTURES / source).read_text(), parse_float=str)
    points = data["points"]
    mu = data.get("mu_r", ["1000"] * len(points))
    if isinstance(mu, str):
        mu = [mu] * len(points)
    return points, [format(Decimal(str(m)) ** 2, "f") for m in mu]


PROCESSES = {
    "gg_gg": (
        "g g > g g",
        {"QCD": 2, "QED": 0},
        lambda: _reference_points("gluon_scattering_qcd"),
    ),
    "uux_ddxg": (
        "u u~ > d d~ g",
        {"QCD": 3, "QED": 0},
        lambda: _fixture_points("required/uux_ddxg.json"),
    ),
    "gg_ttx": (
        "g g > t t~",
        {"QCD": 2, "QED": 0},
        lambda: _fixture_points("required/gg_ttx.json"),
    ),
}


@pytest.fixture(scope="module")
def slices(loop_sm_compiled_model, loop_native):
    from pyamplicol.loops.v2.assemble import LoopSlice, SliceRequest
    from pyamplicol.loops.v2.plan_export import NativeExecutor
    from pyamplicol.models.loading import load_compiled_model

    if not _tirust(loop_native):
        pytest.skip("the staged native extension is built without TIRust")
    compiled = load_compiled_model(loop_sm_compiled_model)
    built: dict = {}

    def build(name: str, threads=None):
        key = (name, threads)
        if key not in built:
            process, born, _points = PROCESSES[name]
            loop = {k: v + (2 if k == "QCD" else 0) for k, v in born.items()}
            if name not in [k for k, _ in built]:
                request = SliceRequest(process, born, loop, fixed_parameters=FIXED)
                slice_ = LoopSlice.generate(compiled, request, loop_native)
            else:
                slice_ = next(v for (k, _), v in built.items() if k == name).slice
            built[key] = NativeExecutor(slice_, threads=threads)
        return built[key]

    return build


def _lanes(executor, points, scales, backend: str, retest: str = "reversed"):
    """Per point (lanes, re-test lanes) of the executor's native evaluation
    on ``backend``, or the failures."""

    from pyamplicol.loops.v2.reduction import running

    with running(backend):
        _precision, evaluations, failures = executor._evaluated(
            points, scales, PARAMETERS, None, "all", retest
        )
    out = []
    for index, evaluation in enumerate(evaluations):
        if index in failures:
            out.append(failures[index])
            continue
        out.append(
            (
                {k: v.coefficients for k, v in evaluation.families.items()},
                {k: v.coefficients for k, v in evaluation.retest_families.items()},
            )
        )
    return out


def _size(z) -> float:
    return abs(complex(z))


def _agree(name, ninja, tir) -> tuple[int, float, int]:
    """Every lane of both backends within L6 on the points TIRust serves (a
    point it fails closed names TIRust's reason); returns (lanes, worst
    ratio, failed points)."""

    count, worst, failed = 0, 0.0, 0
    for n_point, t_point in zip(ninja, tir, strict=True):
        assert not isinstance(n_point, list), (name, n_point)
        if isinstance(t_point, list):
            failed += 1
            assert any("TIRust" in reason for reason in t_point), (name, t_point)
            continue
        (n_direct, n_check), (t_direct, t_check) = n_point, t_point
        assert n_direct.keys() == t_direct.keys(), name
        for lane, n in n_direct.items():
            t = t_direct[lane]
            scale = max(_size(z) for z in n)
            e_n = max(_size(a - b) for a, b in zip(n, n_check[lane], strict=True))
            e_t = max(_size(a - b) for a, b in zip(t, t_check[lane], strict=True))
            # Both re-tests can be blind on a lane (NinjaRS's reversed frame
            # on a bubble, TIR's on a symmetric one): agreement to rounding
            # (ROUNDING of the lane scale) passes too.
            bound = MARGIN * max(e_n, e_t, FLOOR * scale) + ROUNDING * scale
            deviation = max(_size(a - b) for a, b in zip(t, n, strict=True))
            assert deviation <= bound, (name, lane, deviation, e_n, e_t, scale)
            worst = max(worst, deviation / bound)
            count += 1
    return count, worst, failed


# The reasons of TIRust's fail-closed lanes: the conditioning gates (without
# the method choice), the method choice after them, or its acceptance check.
FAIL_CLOSED = (
    "refused by the conditioning gates",
    "refused by the gates, then by the method choice",
    "method-choice lane refused by its check",
    "method-choice lane unverified",
)


def _no_method_choice(report) -> None:
    """Without the method choice no method-choice row is served."""

    counts = report["counts"]
    for name in (
        "TIR method-choice rows",
        "TIR method-choice rows with expansions",
        "TIR method-choice check rows",
    ):
        assert counts.get(name, 0) == 0, (name, counts)


def _method_choice_accepted(report) -> None:
    """The default settings' statistics (module docstring): the method
    choice serves gate-refused rows only, and its acceptance compares at
    most the rows it served directly."""

    counts = report["counts"]
    refused = counts.get("TIR gate refusals", 0)
    chosen = counts.get("TIR method-choice rows", 0)
    expansions = counts.get("TIR method-choice rows with expansions", 0)
    assert expansions <= chosen <= refused, counts
    assert counts.get("TIR method-choice check rows", 0) <= chosen, counts


@pytest.mark.loop_physics
@pytest.mark.parametrize("name", ["gg_gg", "uux_ddxg"])
def test_raw_lanes_agree_with_ninjars_within_their_estimates(slices, name):
    """T2 / L6 on the raw lanes, all Laurent orders, every fixture point."""

    executor = slices(name)
    points, scales = PROCESSES[name][2]()
    ninja = _lanes(executor, points, scales, "ninjars")
    tir = _lanes(executor, points, scales, "tirust")
    lanes, _worst, failed = _agree(name, ninja, tir)
    assert lanes > 0
    statistics = executor.last_report["counts"]
    assert statistics["contracted lanes"] > 0 and statistics["tensor moment rows"] > 0
    _method_choice_accepted(executor.last_report)
    if failed:
        reasons = [r for point in tir if isinstance(point, list) for r in point]
        assert any(any(why in r for why in FAIL_CLOSED) for r in reasons), reasons


@pytest.mark.loop_physics
def test_a_top_in_its_own_gauge_loop_is_served(slices):
    """g g > t t~: the on-shell top legs next to top propagators. TIRust
    takes the S-matrix the executor forms (the on-shell entries exactly the
    legs' squared masses, DESIGN §11.1), so the lanes are served on the exact
    on-shell branch: in binary64 within L6 of NinjaRS's (a row the gates
    refuse fails closed with the gates' reason, never a capability one), and
    in double-double every lane within the double-double target of
    NinjaRS's generic engine on the same inputs."""

    from pyamplicol.loops.v2.assemble import Precision
    from pyamplicol.loops.v2.reduction import running

    executor = slices("gg_ttx")
    points, scales = PROCESSES["gg_ttx"][2]()
    ninja = _lanes(executor, points, scales, "ninjars")
    tir = _lanes(executor, points, scales, "tirust")
    lanes, _worst, _failed = _agree("gg_ttx", ninja, tir)
    assert lanes > 0
    for point in tir:
        if isinstance(point, list):
            assert not any("capability" in reason for reason in point), point
    worst = 0.0
    for backend in ("ninjars", "tirust"):
        with running(backend):
            _precision, evaluations, failures = executor._evaluated(
                points[:1],
                scales[:1],
                PARAMETERS,
                Precision.double_double(),
                "all",
                None,
            )
        assert not failures, failures
        if backend == "ninjars":
            reference = evaluations[0].families
            continue
        found = evaluations[0].families
        assert found.keys() == reference.keys()
        for lane, value in found.items():
            n = reference[lane].coefficients
            scale = max(_size(z) for z in n)
            deviation = max(
                _size(a - b) for a, b in zip(value.coefficients, n, strict=True)
            )
            worst = max(worst, deviation / scale)
    print(f"g g > t t~ double-double TIRust vs NinjaRS: worst relative {worst:.3e}")
    assert worst < DOUBLE_DOUBLE, worst


@pytest.mark.loop_physics
def test_family_threads_leave_tirust_bitwise(slices):
    """T6: one family thread or four, the same TIRust lanes bit for bit."""

    points, scales = PROCESSES["uux_ddxg"][2]()
    one = _lanes(slices("uux_ddxg", threads=1), points, scales, "tirust")
    four = _lanes(slices("uux_ddxg", threads=4), points, scales, "tirust")
    assert one == four


@pytest.mark.loop_physics
def test_validation_settings_rebuild_and_reset_the_tirust_backend(slices, monkeypatch):
    """``V2Executor.prepare_reduction``'s TIRust settings (the validation
    knobs of the discriminating test, DECISIONS 2026-11-07 15:12): NinjaRS
    refuses them, an unknown limit name or acceptance frame is refused;
    another setting rebuilds TIRust's backend for a validation run (the
    artifact runtime's ``reduction.prepare`` bypassed: the method choice
    off, no method-choice row, the gate-refused rows fail closed with the
    gates' reason), and the artifact runtime's next call restores the
    defaults (``reduction.prepare``: the method choice on) and their lanes
    bit for bit."""

    from pyamplicol.loops.v2 import plan_export

    executor = slices("uux_ddxg")
    points, scales = PROCESSES["uux_ddxg"][2]()
    native = executor.native("double", None, len(points))
    with pytest.raises(Exception, match="TIRust's settings"):
        native.prepare_reduction("ninjars", alternative_first=None)
    with pytest.raises(Exception, match="alternative_first must be"):
        native.prepare_reduction("tirust", alternative_first="fastest")
    with pytest.raises(Exception, match="acceptance_frame must be"):
        native.prepare_reduction("tirust", acceptance_frame="reversed")
    plans = native.prepare_reduction("tirust")
    default = _lanes(executor, points, scales, "tirust")
    _method_choice_accepted(executor.last_report)
    validation = native.prepare_reduction(
        "tirust", method_choice=False, alternative_first="default"
    )
    assert validation == plans
    with monkeypatch.context() as patch:
        patch.setattr(plan_export, "prepare", lambda native: None)
        chosen = _lanes(executor, points, scales, "tirust")
    _no_method_choice(executor.last_report)
    if executor.last_report["counts"].get("TIR gate refusals", 0):
        # Every gate-refused row fails closed with the gates' reason.
        reasons = [r for point in chosen if isinstance(point, list) for r in point]
        assert any("refused by the conditioning gates" in r for r in reasons), reasons
    assert _lanes(executor, points, scales, "tirust") == default
    _method_choice_accepted(executor.last_report)


@pytest.mark.loop_physics
def test_complex_mass_lanes_agree(loop_sm_ufo, loop_native, tmp_path_factory):
    """T2 on the complex-mass fixture (lit-spike-tir's b b~ > w+ w-: the
    top's complex pole mass in the t channel and the loops, Im m^2 < 0;
    ``test_loop_v2_native_complex_masses``'s "plain" model)."""

    import math
    from dataclasses import replace

    from pyamplicol.loops.v2.assemble import LoopSlice, SliceRequest, _model_masses
    from pyamplicol.loops.v2.plan_export import NativeExecutor
    from pyamplicol.models.loading import compile_model_source
    from tools.loop_reference.model_compatibility import (
        prepare_loop_sm_scalar_declarations,
    )

    from .loop_v2_ufo_variants import complex_mass_ufo

    if not _tirust(loop_native):
        pytest.skip("the staged native extension is built without TIRust")
    root = tmp_path_factory.mktemp("complex-top")
    ufo = complex_mass_ufo(loop_sm_ufo, root / "loop_sm_plain", "t", "MT", "WT")
    compiled = compile_model_source(ufo, cache_dir=root / "cache")
    compiled = replace(
        compiled,
        raw_loop_model=prepare_loop_sm_scalar_declarations(
            compiled.raw_loop_model
        ).derived,
        _serialized_path=None,
        _prepared_bundle=None,
    )
    request = SliceRequest(
        "b b~ > w+ w-",
        {"QCD": 0, "QED": 2},
        {"QCD": 2, "QED": 2},
        fixed_parameters={"WZ": "0", "WW": "0", "WH": "0", "MB": "0"},
    )
    slice_ = LoopSlice.generate(compiled, request, loop_native)
    executor = NativeExecutor(slice_)
    mw = float(_model_masses(slice_.compiled, {}, 40)("MW"))
    energy = 250.0
    size = math.sqrt(energy * energy - mw * mw)
    legs = [
        (energy, 0.0, 0.0, energy),
        (energy, 0.0, 0.0, -energy),
        (energy, 0.8 * size, 0.0, 0.6 * size),
        (energy, -0.8 * size, 0.0, -0.6 * size),
    ]
    points = [[[repr(x) for x in leg] for leg in legs]]
    ninja = _lanes(executor, points, ["62500"], "ninjars")
    tir = _lanes(executor, points, ["62500"], "tirust")
    lanes, _worst, _failed = _agree("bbx_wpwm_cms", ninja, tir)
    assert lanes > 0
    _method_choice_accepted(executor.last_report)


# ---------------------------------------------------------------- public API


def _li():
    from pyamplicol._internal import selftest_virtual_li

    return selftest_virtual_li


def _ew():
    from pyamplicol._internal import selftest_virtual_ew

    return selftest_virtual_ew


@pytest.mark.loop_physics
def test_tirust_is_served_on_request_and_auto_keeps_ninjars(loop_native):
    """The summed route of the loop-induced self-test (g g > h) and the
    summed and resolved routes of the Born-interfered one serve an explicit
    ``"tirust"`` (recorded as requested) with NinjaRS's values within their
    re-test estimates; ``"auto"`` keeps NinjaRS with its reason; TIRust at
    double-double and arbitrary precision is served on request, also on
    every level of a rescue ladder."""

    from pyamplicol import Runtime
    from pyamplicol.api.virtual_results import VirtualStabilityChecks

    if not _tirust(loop_native):
        pytest.skip("the staged native extension is built without TIRust")
    li = _li()
    expected = json.loads((li.fixture_root() / "expected.json").read_text())
    runtime = Runtime.load(
        li.fixture_root() / "artifact",
        process=li.NAME,
        model_parameters=expected["parameters"],
    )
    points, scales = expected["points"], expected["mu_r2"]
    checks = VirtualStabilityChecks(0, True)
    values = {}
    for backend in ("ninjars", "tirust"):
        value = runtime.evaluate(
            points, mu_r2=scales, stability_checks=checks, reduction=backend
        )
        assert {lane.reduction.effective for lane in value.diagnostics} == {backend}
        assert {lane.reduction.reason for lane in value.diagnostics} == {"requested"}
        values[backend] = value
    for k in range(len(points)):
        for sector, (n, t) in enumerate(
            zip(values["ninjars"].values[k], values["tirust"].values[k], strict=True)
        ):
            e = max(values[b].stability[k].absolute[sector][0] for b in values)
            assert abs(float(n.finite) - float(t.finite)) <= MARGIN * e + 1e-14 * abs(
                float(n.finite)
            ), (k, sector)
    auto = runtime.evaluate(points[:1], mu_r2=scales[:1])
    (lane,) = auto.diagnostics
    assert lane.reduction.effective == "ninjars"
    assert "agreement with NinjaRS on loop-induced processes" in lane.reduction.reason
    # The extended carriers on TIRust (m-lit lit-tir-mp): explicitly
    # requested, served and recorded at double-double and arbitrary
    # precision, within the double-double target of NinjaRS's values.
    for arithmetic, precision in (("double-double", None), ("arbitrary", 40)):
        found = {
            backend: runtime.evaluate(
                points[:1],
                mu_r2=scales[:1],
                arithmetic=arithmetic,
                precision=precision,
                reduction=backend,
            )
            for backend in ("ninjars", "tirust")
        }
        (lane,) = found["tirust"].diagnostics
        assert (lane.reduction.effective, lane.reduction.reason) == (
            "tirust",
            "requested",
        )
        n_values = _flat(found["ninjars"].values)
        t_values = _flat(found["tirust"].values)
        scale = max(abs(x) for x in n_values)
        deviation = max(abs(a - b) for a, b in zip(n_values, t_values, strict=True))
        assert deviation <= DOUBLE_DOUBLE * scale, (arithmetic, deviation, scale)
    # Every level of a rescue ladder resolves to the explicit request: a
    # level on TIRust stays on TIRust (binary64, then double-double).
    from pyamplicol.api.virtual_results import VirtualRescueLevel, VirtualRescuePolicy

    ladder = VirtualRescuePolicy(
        (
            VirtualRescueLevel("double", rotations=0, target=1e-300),
            VirtualRescueLevel("double-double", rotations=0, reduction_check="none"),
        )
    )
    rescued = runtime.evaluate(
        points[:1], mu_r2=scales[:1], stability=ladder, reduction="tirust"
    )
    assert {lane.reduction.effective for lane in rescued.diagnostics} == {"tirust"}
    assert rescued.rescue[0].level == 1, rescued.rescue
    ew = _ew()
    record = json.loads((ew.fixture_root() / "expected.json").read_text())
    born = Runtime.load(ew.fixture_root() / "artifact", process=ew.NAME)
    points, scales = record["points"][:2], record["mu_r2"][:2]
    for call in ("evaluate", "evaluate_resolved"):
        found = {
            backend: getattr(born, call)(points, mu_r2=scales, reduction=backend)
            for backend in ("ninjars", "tirust")
        }
        lanes = [lane for lane in found["tirust"].diagnostics]
        assert {lane.reduction.effective for lane in lanes} == {"tirust"}, call
        assert repr(found["tirust"].born_values) == repr(found["ninjars"].born_values)
        # The virtual values were reduced by TIRust: within the lax
        # precision of NinjaRS's, and not NinjaRS's bits.
        n_values = _flat(found["ninjars"].values)
        t_values = _flat(found["tirust"].values)
        assert len(n_values) == len(t_values) > 0, call
        scale = max(abs(x) for x in n_values)
        assert scale > 0, call
        deviation = max(abs(a - b) for a, b in zip(n_values, t_values, strict=True))
        assert deviation <= LAX * scale, (call, deviation, scale)
        assert repr(found["tirust"].values) != repr(found["ninjars"].values), call
        # In double-double on both routes (m-lit lit-tir-mp): TIRust served,
        # within the double-double target of NinjaRS's values (the
        # self-test's points are exact).
        found = {
            backend: getattr(born, call)(
                points[:1],
                mu_r2=scales[:1],
                arithmetic="double-double",
                reduction=backend,
            )
            for backend in ("ninjars", "tirust")
        }
        assert {lane.reduction.effective for lane in found["tirust"].diagnostics} == {
            "tirust"
        }, call
        n_values = _flat(found["ninjars"].values)
        t_values = _flat(found["tirust"].values)
        scale = max(abs(x) for x in n_values)
        deviation = max(abs(a - b) for a, b in zip(n_values, t_values, strict=True))
        assert deviation <= DOUBLE_DOUBLE * scale, (call, deviation, scale)
    auto = born.evaluate(points[:1], mu_r2=scales[:1])
    assert {lane.reduction.effective for lane in auto.diagnostics} == {"ninjars"}


def _flat(values) -> list[complex]:
    """The numbers of a nested result (``VirtualValues``,
    ``ResolvedVirtualValues``) in order."""

    if isinstance(values, (str, bytes)):
        raise TypeError(values)
    try:
        return [complex(values)]
    except TypeError:
        return [x for item in values for x in _flat(item)]


# ------------------------------------------------------------------------ T8

_T8 = r"""
import json, sys
from pyamplicol import Runtime
from pyamplicol._internal import selftest_virtual_ew as ew
from pyamplicol._internal import selftest_virtual_li as li
backend = sys.argv[1]
out = {}
for module in (li, ew):
    root = module.fixture_root()
    expected = json.loads((root / "expected.json").read_text())
    runtime = Runtime.load(root / "artifact", process=module.NAME,
                           model_parameters=expected.get("parameters") or None)
    value = runtime.evaluate(expected["points"][:2], mu_r2=expected["mu_r2"][:2],
                             reduction=backend)
    out[module.NAME] = {
        "effective": sorted({lane.reduction.effective for lane in value.diagnostics}),
        "values": [[[float(x).hex() for x in sector] for sector in point]
                   for point in value.values],
    }
print(json.dumps(out))
"""


def _run_t8(backend: str, env: dict[str, str]) -> dict:
    run = subprocess.run(
        [sys.executable, "-c", _T8, backend],
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
        env=env,
    )
    output = run.stdout[-2000:] + run.stderr[-4000:]
    assert run.returncode == 0, output
    assert "restricted Symbolica" not in output
    return json.loads(run.stdout.strip().splitlines()[-1])


@pytest.mark.loop_physics
def test_binary64_tirust_touches_no_symbolica_state(loop_native, tmp_path):
    """T8 (module docstring); NinjaRS in the same harness is the baseline."""

    if not _tirust(loop_native):
        pytest.skip("the staged native extension is built without TIRust")
    lock = tmp_path / "symbolica-locks"
    lock.mkdir()
    lock.chmod(0o700)
    base = {
        key: value for key, value in os.environ.items() if key != "SYMBOLICA_LICENSE"
    }
    with portability.RestrictedPermit(lock_dir=str(lock)) as permit:
        assert permit.record["verified"], permit.record
        env = permit.environment({**base, **NO_NATIVE_KEY})
        unlicensed = {
            backend: _run_t8(backend, env) for backend in ("ninjars", "tirust")
        }
    for backend in ("ninjars", "tirust"):
        licensed = _run_t8(backend, dict(os.environ))
        assert unlicensed[backend] == licensed, backend
        for record in licensed.values():
            assert record["effective"] == [backend]
