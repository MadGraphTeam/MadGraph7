# SPDX-License-Identifier: 0BSD
"""The Python product surface of loop-induced processes (m-lit LIT-API;
PLAN §2.a.9, §2.a.10, §2.b.4, rulings Q36, Q11 (b), Q13, Q17, Q18, Q2 (b)).

* Loop-induced results are ``VirtualEvaluation`` / ``ResolvedVirtualEvaluation``
  with ``contribution = "loop-induced"``: empty Born fields, the pole
  selections "all" and "none" only; a "virtual" result with an empty Born
  catalogue still raises.
* The reduction backend (``[loop] reduction`` generation default recorded in
  the artifact, ``[evaluation] reduction`` / ``reduction=`` overrides, CLI
  ``--loop-reduction``): ``auto`` takes the per-type default (loop-induced
  TIRust, Born-interfered NinjaRS) and falls back to NinjaRS with the reason
  recorded; an explicit backend that cannot serve the call is refused typed;
  requested, effective backend and reason in every lane diagnostic.
* Stability estimates carry the absolute estimate E, the scale S and E / S
  (the opt-in ``"scale"`` rescue metric); a loop-induced result's pole
  estimates are against S, and its lane diagnostics carry the amplitude-level
  finiteness ratio.
* The shipped loop-induced self-test artifact (g g > h) equals MadLoop, runs
  without ``SYMBOLICA_LICENSE`` and touches no Symbolica state.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
_tool_spec = importlib.util.spec_from_file_location(
    "loop_artifact_portability", ROOT / "tools" / "loop_artifact_portability.py"
)
assert _tool_spec is not None and _tool_spec.loader is not None
portability = importlib.util.module_from_spec(_tool_spec)
_tool_spec.loader.exec_module(portability)

V5 = "rusticol.loop-runtime.v5"
# pyAmpliCol's native library key disabled (``pyamplicol::DISABLE_VARIABLE``).
NO_NATIVE_KEY = {"PYAMPLICOL_NATIVE_LIBRARY_KEY": "off"}


def _r2():
    from pyamplicol.config import R2Strategy
    from pyamplicol.loops.strategy import R2Selection

    return R2Selection(R2Strategy.AUTO, R2Strategy.MODEL, "test")


def _sector():
    from pyamplicol.loops.orders import BornLoopPair, VirtualOrderSector

    loop = (("QCD", 2), ("QED", 1))
    return VirtualOrderSector((("QCD", 4), ("QED", 2)), (BornLoopPair(loop, loop),))


def _lane(**fields):
    from pyamplicol.api.virtual_results import VirtualLaneDiagnostic

    return VirtualLaneDiagnostic("success", "double", 53, False, **fields)


# ---------------------------------------------------------------- result types


@pytest.mark.loop_fast
def test_loop_induced_results_reuse_the_virtual_types():
    """Q36: zeroed Born fields, the kind in ``contribution``; a "virtual"
    result with an empty Born catalogue still raises; the poles selection
    is {all, none} for loop-induced; resolved results have no Born colour
    axis; ``total()`` keeps the kind."""

    from pyamplicol.api.virtual_results import (
        LaurentCoefficients,
        ResolvedVirtualEvaluation,
        VirtualEvaluation,
    )

    sector = _sector()
    laurent = LaurentCoefficients(0.25, 1e-17, 0.0)
    value = VirtualEvaluation(
        ((),),
        ((laurent,),),
        (),
        (sector,),
        (_lane(),),
        _r2(),
        contribution="loop-induced",
    )
    assert value.born_shape == (1, 0) and value.shape == (1, 1, 3)
    with pytest.raises(ValueError, match="Born and interference catalogues"):
        VirtualEvaluation(((),), ((laurent,),), (), (sector,), (_lane(),), _r2())
    for poles in ("uv", "ir"):
        with pytest.raises(ValueError, match="'all' or 'none'"):
            VirtualEvaluation(
                ((),),
                ((laurent,),),
                (),
                (sector,),
                (_lane(),),
                _r2(),
                poles=poles,
                contribution="loop-induced",
            )
    finite_only = VirtualEvaluation(
        ((),),
        ((LaurentCoefficients(0.25, None, None),),),
        (),
        (sector,),
        (_lane(),),
        _r2(),
        poles="none",
        contribution="loop-induced",
    )
    assert finite_only.finite == ((0.25,),)

    def resolved(born_colours, contribution):
        return ResolvedVirtualEvaluation(
            ((),),
            ((((laurent,), (laurent,)),),),
            (),
            (sector,),
            (_lane(),),
            _r2(),
            ("h0", "h1"),
            born_colours,
            ("full",),
            "full",
            contribution=contribution,
        )

    found = resolved((), "loop-induced")
    assert found.born_shape == (1, 0, 2, 0)
    total = found.total()
    assert total.contribution == "loop-induced" and total.born_values == ((),)
    assert total.values[0][0].finite == 0.5
    with pytest.raises(ValueError, match="no Born colour IDs"):
        resolved(("c0",), "loop-induced")
    with pytest.raises(ValueError, match="nonempty and unique"):
        resolved((), "virtual")


@pytest.mark.loop_fast
def test_reduction_record_finiteness_and_absolute_estimates_are_validated():
    from pyamplicol.api.virtual_results import (
        LaurentCoefficients,
        VirtualLaneDiagnostic,
        VirtualReduction,
        VirtualRescueLevel,
        VirtualStabilityEstimate,
    )

    record = VirtualReduction("auto", "ninjars", "auto: NinjaRS, the default")
    assert _lane(reduction=record, finiteness=1e-16).reduction == record
    for args in (
        ("tirust", "ninjars", "x"),
        ("fast", "ninjars", "x"),
        ("auto", "auto", "x"),
        ("auto", "ninjars", ""),
    ):
        with pytest.raises(ValueError):
            VirtualReduction(*args)
    with pytest.raises(ValueError, match="finiteness"):
        _lane(finiteness=-1.0)
    with pytest.raises(ValueError, match="finiteness"):
        VirtualLaneDiagnostic("failed", "double", 53, False, finiteness=0.0)
    with pytest.raises(TypeError, match="VirtualReduction"):
        _lane(reduction="ninjars")
    e = (LaurentCoefficients(1e-14, 2e-14, 3e-14),)
    s = (LaurentCoefficients(1.0, 2.0, 3.0),)
    checks = (("rotation-0", 1e-14),)
    estimate = VirtualStabilityEstimate(
        1e-12, 1e-12, 1e-14, 1e-14, checks, scale=1e-14, absolute=e, magnitude=s
    )
    assert estimate.absolute == e and estimate.magnitude == s
    with pytest.raises(ValueError, match="given together"):
        VirtualStabilityEstimate(1e-12, 1e-12, None, None, checks, scale=1e-14)
    with pytest.raises(ValueError, match="no pole entries"):
        VirtualStabilityEstimate(
            1e-12, 1e-12, None, None, checks, scale=1e-14, absolute=e, magnitude=s
        )
    with pytest.raises(ValueError, match="follow the sectors"):
        VirtualStabilityEstimate(
            1e-12,
            1e-12,
            1e-14,
            1e-14,
            checks,
            scale=0.0,
            absolute=e,
            magnitude=s * 2,
        )
    # The opt-in "scale" rescue metric (m-lit PLAN §2.a.8).
    assert VirtualRescueLevel("double", metric="scale").metric == "scale"
    with pytest.raises(ValueError, match="metric"):
        VirtualRescueLevel("double", metric="absolute")


@pytest.mark.loop_fast
def test_public_estimate_reports_e_and_s_and_loop_induced_poles_against_s():
    from fractions import Fraction

    from pyamplicol.loops.v2.stability import Budget, StabilityEstimate
    from pyamplicol.runtime.virtual import v2_public_estimate

    class Laurent:
        def __init__(self, *values):
            self.coefficients = values

    sector = (("QCD", 4), ("QED", 2))
    reference = {sector: Laurent(0.5, 1e-18, 0.0)}
    internal = StabilityEstimate.from_checks(
        reference,
        {"rotation-0": {sector: Budget((1e-16, 1e-16, 1e-16))}},
        scale={sector: Budget((10.0, 20.0, 40.0))},
        roundoff=float(Fraction(1, 2**53)),
    )
    virtual = v2_public_estimate(internal, True)
    induced = v2_public_estimate(internal, True, contribution="loop-induced")
    (e,) = induced.absolute
    (s,) = induced.magnitude
    assert s == (10.0, 20.0, 40.0)
    assert e == tuple(float(x) for x in internal.absolute[sector].coefficients)
    ratios = [a / b for a, b in zip(e, s, strict=True)]
    assert induced.scale == max(ratios)
    # Relative to the vanishing poles themselves (inf on the exact zero) for
    # "virtual"; against S for a loop-induced result.
    assert virtual.double_pole == float("inf") and virtual.single_pole > 1
    assert (induced.single_pole, induced.double_pole) == (ratios[1], ratios[2])
    assert induced.finite == virtual.finite and induced.relative == virtual.relative
    finite_only = v2_public_estimate(
        StabilityEstimate.from_checks(
            {sector: Laurent(0.5)},
            {"rotation-0": {sector: Budget((1e-16,))}},
            scale={sector: Budget((10.0,))},
            roundoff=1e-16,
        ),
        False,
        contribution="loop-induced",
    )
    assert finite_only.single_pole is None
    assert finite_only.absolute[0][1:] == (None, None)


# ------------------------------------------------------- reduction resolution


class _Native:
    """A native extension with (``served``) or without the backend query."""

    def __init__(self, served=None):
        if served is not None:
            self.reduction_backends = lambda: json.dumps(served)


ALL = ["double", "double-double", "arbitrary"]


@pytest.mark.loop_fast
def test_auto_takes_the_per_type_default_and_records_every_fallback():
    """Q17 / Q18: auto -> loop-induced TIRust, Born-interfered NinjaRS;
    TIRust unavailable (not built in, a carrier it does not serve) -> NinjaRS
    with the reason; an explicit backend that cannot serve is refused typed;
    a TIR-only artifact has no NinjaRS fallback."""

    from pyamplicol.api.errors import CompatibilityError
    from pyamplicol.loops.v2.reduction import (
        BACKENDS,
        ReductionUnavailable,
        Route,
        resolve,
    )

    # A route whose executor runs every backend: the capability logic alone
    # (the artifact runtime's routes, ``SUMMED`` / ``RESOLVED``, below).
    every = Route("a test route", frozenset(BACKENDS))

    def choice(requested, kind, carrier="double", native=None, admitted=None):
        return resolve(
            requested,
            contribution=kind,
            carrier=carrier,
            native=_Native() if native is None else native,
            route=every,
            admitted=admitted,
        )

    # Today's extensions: no TIRust (m-lit lit-tir adds it and the query).
    found = choice("auto", "loop-induced")
    assert (found.requested, found.effective) == ("auto", "ninjars")
    assert "TIRust, the loop-induced default, is unavailable" in found.reason
    assert "'tir' feature" in found.reason
    found = choice("auto", "virtual")
    assert found.effective == "ninjars"
    assert found.reason == "auto: NinjaRS, the Born-interfered default"
    assert choice("ninjars", "loop-induced").reason == "requested"
    with pytest.raises(ReductionUnavailable, match="not built into") as refused:
        choice("tirust", "virtual")
    assert isinstance(refused.value, CompatibilityError)
    assert refused.value.key == "reduction-unavailable"
    # An extension with TIRust in binary64 only (before m-lit lit-tir-mp).
    tir = _Native({"ninjars": ALL, "tirust": ["double"]})
    assert choice("auto", "loop-induced", native=tir).effective == "tirust"
    assert choice("auto", "virtual", native=tir).effective == "ninjars"
    assert choice("tirust", "virtual", native=tir).effective == "tirust"
    found = choice("auto", "loop-induced", "double-double", tir)
    assert found.effective == "ninjars"
    assert "does not serve the double-double carrier" in found.reason
    with pytest.raises(ReductionUnavailable, match="double-double"):
        choice("tirust", "loop-induced", "double-double", tir)
    # A TIR-only artifact (rank > N + 2, m-lit lit-eft): auto -> TIRust for
    # both kinds, explicit NinjaRS refused, no fallback.
    only = {"tirust"}
    assert choice("auto", "virtual", native=tir, admitted=only).effective == "tirust"
    with pytest.raises(ReductionUnavailable, match="admits"):
        choice("ninjars", "virtual", native=tir, admitted=only)
    with pytest.raises(ReductionUnavailable, match="no reduction backend serves"):
        choice("auto", "loop-induced", "double-double", tir, admitted=only)
    with pytest.raises(CompatibilityError, match="unknown reduction backend"):
        choice("auto", "virtual", native=_Native({"golem": ALL}))
    with pytest.raises(ValueError, match="unknown reduction backend"):
        choice("fast", "virtual")


@pytest.mark.loop_fast
def test_the_effective_backend_is_one_the_route_runs():
    """The guard of every route: the artifact runtime's summed route and its
    resolved outputs run both backends (m-lit lit-tir), so an explicit
    TIRust is served there; ``auto`` keeps NinjaRS on them for both kinds
    (TIRust still fails closed on points NinjaRS serves: no automatic switch
    in this release, DECISIONS 2026-11-08 11:17), the reason saying so; a
    route whose executor does not run TIRust refuses it
    typed and its ``auto`` names the route. One choice list and one
    validator (``config.ReductionBackend``, ``reduction_request``)."""

    from pyamplicol.api.virtual_results import VirtualReduction
    from pyamplicol.config import ReductionBackend
    from pyamplicol.config.models import reduction_request
    from pyamplicol.loops.v2.reduction import (
        BACKENDS,
        RESOLVED,
        SUMMED,
        ReductionUnavailable,
        Route,
        resolve,
    )

    tir = _Native({"ninjars": ALL, "tirust": ALL})
    for route in (SUMMED, RESOLVED):
        assert route.backends == frozenset(BACKENDS)
        found = resolve(
            "auto", contribution="virtual", carrier="double", native=tir, route=route
        )
        assert found == VirtualReduction(
            "auto", "ninjars", "auto: NinjaRS, the Born-interfered default"
        )
        found = resolve(
            "auto",
            contribution="loop-induced",
            carrier="double",
            native=tir,
            route=route,
        )
        assert found.effective == "ninjars"
        assert "fails closed on points NinjaRS serves" in found.reason
        assert route.serves in found.reason
        for kind in ("virtual", "loop-induced"):
            found = resolve(
                "tirust", contribution=kind, carrier="double", native=tir, route=route
            )
            assert found == VirtualReduction("tirust", "tirust", "requested")
    ninja = Route("a NinjaRS route", frozenset({"ninjars"}))
    found = resolve(
        "auto", contribution="loop-induced", carrier="double", native=tir, route=ninja
    )
    assert found.effective == "ninjars"
    assert "not wired to the executor of a NinjaRS route" in found.reason
    with pytest.raises(ReductionUnavailable, match="not wired to the executor"):
        resolve(
            "tirust", contribution="virtual", carrier="double", native=tir, route=ninja
        )
    wired = Route("a wired route", frozenset(BACKENDS))
    found = resolve(
        "auto", contribution="loop-induced", carrier="double", native=tir, route=wired
    )
    assert found.effective == "tirust"
    assert tuple(item.value for item in ReductionBackend)[1:] == BACKENDS
    assert [reduction_request(item) for item in ReductionBackend] == [
        "auto",
        *BACKENDS,
    ]
    for bad in ("golem", 3, None):
        with pytest.raises(ValueError, match="unknown reduction backend"):
            reduction_request(bad)
        with pytest.raises(ValueError, match="unknown reduction backend"):
            VirtualReduction(bad, "ninjars", "x")  # type: ignore[arg-type]


@pytest.mark.loop_physics
def test_every_route_records_its_backend_and_refuses_one_it_does_not_run(
    monkeypatch,
):
    """With an extension that reports TIRust built in at every carrier and
    routes whose executors run NinjaRS only (the artifact runtime's routes
    replaced by NinjaRS-only ones: the guard itself), an explicit TIRust
    request is refused typed before any point runs on every route of the
    shipped Born-interfered artifact (the summed evaluate and a rescue
    level, evaluate_resolved, evaluate_correlated(_many), the leading-colour
    selection), and a loop-induced auto call falls back to NinjaRS with the
    route in its reason (never a silent switch). Every lane records its
    backend as it is built: one lane diagnostic and one result per output,
    no rebuild of finished results."""

    from pyamplicol import Runtime
    from pyamplicol._internal import selftest_virtual_ew as ew
    from pyamplicol.api import virtual_results
    from pyamplicol.api.virtual_results import (
        VirtualReduction,
        VirtualRescueLevel,
        VirtualRescuePolicy,
    )
    from pyamplicol.correlators import CorrelatedRequest
    from pyamplicol.loops.v2 import reduction
    from pyamplicol.loops.v2.reduction import ReductionUnavailable

    for name in ("SUMMED", "RESOLVED"):
        # The artifact runtime reads its routes from ``reduction`` at call
        # time (lit-sdk-b item [8]: no native load at its import).
        route = getattr(reduction, name)
        monkeypatch.setattr(
            reduction, name, route._replace(backends=frozenset({"ninjars"}), auto=None)
        )
    built = {name: frozenset(reduction.CARRIERS) for name in reduction.BACKENDS}
    monkeypatch.setattr(reduction, "native_backends", lambda native: built)
    expected = json.loads((ew.fixture_root() / "expected.json").read_text())
    points, scales = expected["points"][:2], expected["mu_r2"][:2]
    runtime = Runtime.load(ew.fixture_root() / "artifact", process=ew.NAME)
    born = {"a": CorrelatedRequest("born", {}), "b": CorrelatedRequest("born", {})}
    ladder = VirtualRescuePolicy(
        (VirtualRescueLevel("double"), VirtualRescueLevel("double-double"))
    )
    calls = {
        "evaluate": lambda **kw: runtime.evaluate(points, mu_r2=scales, **kw),
        "rescue": lambda **kw: runtime.evaluate(
            points, mu_r2=scales, stability=ladder, **kw
        ),
        "evaluate_resolved": lambda **kw: runtime.evaluate_resolved(
            points, mu_r2=scales, **kw
        ),
        "evaluate_correlated": lambda **kw: runtime.evaluate_correlated(
            points, mu_r2=scales, **kw
        ),
        "evaluate_correlated_many": lambda **kw: runtime.evaluate_correlated_many(
            points, born, mu_r2=scales, **kw
        ),
    }
    auto = VirtualReduction(
        "auto", "ninjars", "auto: NinjaRS, the Born-interfered default"
    )
    built_lanes: list[object] = []
    built_results: list[object] = []

    def counted(cls, sink):
        original = cls.__post_init__

        def wrapped(self):
            sink.append(self)
            original(self)

        monkeypatch.setattr(cls, "__post_init__", wrapped)

    counted(virtual_results.VirtualLaneDiagnostic, built_lanes)
    for cls in ("VirtualEvaluation", "ResolvedVirtualEvaluation"):
        counted(getattr(virtual_results, cls), built_results)
    for name, call in calls.items():
        with pytest.raises(ReductionUnavailable, match="not wired to the executor"):
            call(reduction="tirust")
        built_lanes.clear()
        built_results.clear()
        found = call()
        results = list(found.values()) if isinstance(found, dict) else [found]
        lanes = [lane for result in results for lane in result.diagnostics]
        assert {lane.reduction for lane in lanes} == {auto}, name
        if name != "rescue":
            # (A rescue merges its levels' results into a new one.)
            assert len(built_lanes) == len(lanes), name
            assert len(built_results) == len(results), name
    with pytest.raises(ReductionUnavailable, match="resolved outputs and selections"):
        runtime.evaluate(points, mu_r2=scales, helicities=["h0"], reduction="tirust")
    selftest = _selftest()
    li = _expected()
    induced = Runtime.load(
        selftest.fixture_root() / "artifact",
        process=selftest.NAME,
        model_parameters=li["parameters"],
    )
    value = induced.evaluate(li["points"][:1], mu_r2=li["mu_r2"][:1])
    (lane,) = value.diagnostics
    assert lane.reduction.effective == "ninjars"
    assert "TIRust, the loop-induced default, is unavailable" in lane.reduction.reason
    assert "not wired to the executor of the summed virtual" in lane.reduction.reason
    with pytest.raises(ReductionUnavailable, match="summed virtual"):
        induced.evaluate(li["points"][:1], mu_r2=li["mu_r2"][:1], reduction="tirust")


@pytest.mark.loop_fast
def test_reduction_option_surfaces(tmp_path):
    """``[loop] reduction`` (CLI ``generate --loop-reduction``) is recorded
    in the request only when not auto (seeded records keep their bytes);
    ``[evaluation] reduction`` (CLI ``evaluate --loop-reduction``) overrides
    it; a tree run refuses a backend; the facade forwards ``reduction``
    only to virtual runtimes."""

    from dataclasses import replace

    from pyamplicol.api.errors import CompatibilityError
    from pyamplicol.cli.parser import parse_cli
    from pyamplicol.config import ConfigurationError
    from pyamplicol.loops.v2.artifact import request_from_record, request_record
    from pyamplicol.loops.v2.assemble import SliceRequest

    def generate(*argv):
        return parse_cli(("generate", *argv)).resolve().effective

    assert generate("--loop-coupling-order", "QCD").loop.reduction == "auto"
    for value in ("auto", "ninjars", "tirust"):
        config = generate("--loop-coupling-order", "QCD", "--loop-reduction", value)
        assert config.loop.reduction == value
    with pytest.raises(ConfigurationError, match="requires one-loop generation"):
        generate("--loop-reduction", "tirust")
    with pytest.raises(SystemExit):
        parse_cli(("generate", "--loop-reduction", "golem"))
    evaluate = parse_cli(("evaluate", "x", "--loop-reduction", "ninjars"))
    assert evaluate.resolve().effective.evaluation.reduction == "ninjars"
    request = SliceRequest("g g > h", {}, {"QCD": 2, "QED": 1})
    assert "reduction" not in request_record(request)
    chosen = replace(request, reduction="tirust")
    record = request_record(chosen)
    assert record["reduction"] == "tirust"
    assert request_from_record(record).reduction == "tirust"
    with pytest.raises(ValueError, match="unknown reduction backend"):
        replace(request, reduction="golem")

    # The configuration keys stay importable without numpy (CI's
    # python-compatibility job; the 936e81b3 lesson).
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from pyamplicol.config import LoopConfig, ReductionBackend; "
            "LoopConfig(reduction=ReductionBackend.TIRUST); "
            "print('numpy' in sys.modules)",
        ],
        capture_output=True,
        text=True,
        timeout=300,
        check=True,
    )
    assert probe.stdout.strip().splitlines()[-1] == "False"

    from pyamplicol import Runtime

    class Tree:
        contribution_kind = "tree"

        def __getattr__(self, name):
            return lambda *args, **kwargs: None

    runtime = object.__new__(Runtime)
    runtime._backend = Tree()
    with pytest.raises(CompatibilityError, match="reduction are virtual options"):
        runtime._v2_options(None, None, "ninjars")
    with pytest.raises(ValueError, match="unknown reduction backend"):
        runtime._v2_options(None, None, "golem")


# ------------------------------------------------------- the shipped artifact


def _selftest():
    from pyamplicol._internal import selftest_virtual_li

    return selftest_virtual_li


def _expected() -> dict:
    root = _selftest().fixture_root()
    return json.loads((root / "expected.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def shipped():
    from pyamplicol import Runtime

    selftest = _selftest()
    expected = _expected()
    return Runtime.load(
        selftest.fixture_root() / "artifact",
        process=selftest.NAME,
        model_parameters=expected["parameters"],
    )


@pytest.mark.loop_physics
def test_shipped_self_test_artifact_equals_madloop_and_is_portable():
    from pyamplicol.diagnostics import _prepared_loop_induced_check, run_self_test

    selftest = _selftest()
    root = selftest.fixture_root()
    manifest = json.loads((root / "artifact" / "artifact.json").read_text())
    assert manifest["runtime"]["required_runtime_capabilities"] == [V5]
    summary, worst = selftest.run_prepared_loop_induced_self_test()
    assert worst <= selftest.RELATIVE_TOLERANCE and "poles vanish" in summary
    check = _prepared_loop_induced_check()
    assert check.status == "pass", check.detail
    names = {item.name for item in run_self_test().checks}
    assert "physics-virtual-loop-induced" in names
    # Portable (§0.2): no absolute path, user or workspace in its bytes.
    scan = portability.scan(root / "artifact")
    assert scan["clean"], scan


@pytest.mark.loop_physics
def test_reduction_finiteness_and_estimates_on_the_public_route(shipped):
    """On the shipped g g > h: every lane names its reduction backend (auto
    falls back to NinjaRS while TIRust is not built in, with the reason);
    an explicit NinjaRS request is bitwise auto's; an explicit TIRust request
    is served (on every level of a rescue ladder too, m-lit lit-tir-mp) when
    the native extension has TIRust, else refused before any point;
    with poles and checks the lanes carry the finiteness ratio, the
    estimates E and S, and the "scale" metric accepts the points at level 0;
    the poles vanish within the absolute estimate (Q11 (b), Q13)."""

    from pyamplicol.api.virtual_results import (
        VirtualRescueLevel,
        VirtualRescuePolicy,
        VirtualStabilityChecks,
    )
    from pyamplicol.loops.v2.reduction import ReductionUnavailable

    expected = _expected()
    points, scales = expected["points"], expected["mu_r2"]
    info = shipped.inspect()
    assert info["exact_colour_zero"] is False
    assert info["colour_zero_loop_orders"] == []
    reduction = info["reduction"]
    assert reduction["requested"] == reduction["artifact_default"] == "auto"
    assert reduction["auto_default"] == "tirust"
    assert "ninjars" in reduction["native"]
    auto = shipped.evaluate(points, mu_r2=scales)
    assert auto.contribution == "loop-induced" and auto.born_orders == ()
    for lane in auto.diagnostics:
        assert lane.reduction.requested == "auto"
        assert lane.reduction.effective == "ninjars"
        assert "loop-induced default" in lane.reduction.reason
        # The finiteness ratio exists exactly with poles and stability checks
        # (its scale, the term floor, needs the per-point tensor moments,
        # which only the checks form; documented in python-api.md and on
        # VirtualLaneDiagnostic.finiteness).
        assert lane.finiteness is None
    named = shipped.evaluate(points, mu_r2=scales, reduction="ninjars")
    assert named.values == auto.values
    assert {lane.reduction.reason for lane in named.diagnostics} == {"requested"}
    from pyamplicol.loops.native import load_loop_native

    if load_loop_native().tirust_available():
        # Served on request in binary64 (m-lit lit-tir); NinjaRS's values to
        # rounding.
        tir = shipped.evaluate(points, mu_r2=scales, reduction="tirust")
        assert {lane.reduction.effective for lane in tir.diagnostics} == {"tirust"}
        for mine, theirs in zip(tir.values, auto.values, strict=True):
            (a,), (b,) = mine, theirs
            assert abs(float(a.finite) - float(b.finite)) <= 1e-12 * abs(
                float(b.finite)
            )
    else:
        with pytest.raises(ReductionUnavailable, match="TIRust is not built into"):
            shipped.evaluate(points, mu_r2=scales, reduction="tirust")
    ladder = VirtualRescuePolicy(
        (
            VirtualRescueLevel("double", metric="scale"),
            VirtualRescueLevel("double-double", metric="scale"),
        )
    )
    if load_loop_native().tirust_available():
        rescued = shipped.evaluate(
            points, mu_r2=scales, stability=ladder, reduction="tirust"
        )
        assert {lane.reduction.effective for lane in rescued.diagnostics} == {"tirust"}
    else:
        with pytest.raises(ReductionUnavailable):
            shipped.evaluate(points, mu_r2=scales, stability=ladder, reduction="tirust")
    checks = VirtualStabilityChecks(1, True, rescaled=True)
    checked = shipped.evaluate(points, mu_r2=scales, stability_checks=checks)
    assert checked.values == auto.values
    for lane, estimate, value in zip(
        checked.diagnostics, checked.stability, checked.values, strict=True
    ):
        # The amplitude-level finiteness ratio (of the order of u).
        assert 0.0 <= lane.finiteness <= 1e-12
        (sector,) = value
        (error,) = estimate.absolute
        (size,) = estimate.magnitude
        # Poles vanish within E (necessary-only cross-check, Q36).
        assert abs(sector.single_pole) <= 10 * error.single_pole
        assert abs(sector.double_pole) <= 10 * error.double_pole
        assert estimate.single_pole == error.single_pole / size.single_pole
        assert estimate.scale <= 1e-13 and size.finite >= abs(sector.finite)
    finite_only = shipped.evaluate(
        points, mu_r2=scales, poles="none", stability_checks=checks
    )
    assert all(lane.finiteness is None for lane in finite_only.diagnostics)
    # Any check forms it, the cheapest included (the re-test alone).
    retest = shipped.evaluate(
        points, mu_r2=scales, stability_checks=VirtualStabilityChecks(0, True)
    )
    assert retest.values == auto.values
    assert [lane.finiteness for lane in retest.diagnostics] == [
        lane.finiteness for lane in checked.diagnostics
    ]
    rescued = shipped.evaluate(points, mu_r2=scales, stability=ladder)
    assert rescued.rescue_counts == {0: len(points), "not met": 0, "failed": 0}
    assert all(lane.reduction is not None for lane in rescued.diagnostics)
    assert all(0.0 <= lane.finiteness <= 1e-12 for lane in rescued.diagnostics)


@pytest.mark.loop_physics
def test_generation_default_is_recorded_and_the_artifact_is_backend_independent(
    tmp_path,
):
    """``[loop] reduction = "ninjars"`` is recorded in the request record and
    read as the runtime's request; the plan is byte for byte the shipped
    (auto) artifact's (artifacts do not depend on the backend); a load-time
    ``reduction`` overrides the recorded default."""

    from pyamplicol import Runtime
    from pyamplicol.config import resolve_config

    selftest = _selftest()
    expected = _expected()
    output = tmp_path / "artifact"
    card = selftest.card(output)
    card["loop"]["reduction"] = "ninjars"
    from pyamplicol.api import Generator, ProcessRequest, ProcessSet

    Generator(resolve_config(card, base_dir=tmp_path)).generate(
        ProcessSet((ProcessRequest.parse(selftest.PROCESS, name=selftest.NAME),)),
        output,
    )

    def execution(root):
        path = root / "processes" / selftest.NAME / "execution.json"
        return json.loads(path.read_text())

    mine = execution(output)
    shipped = execution(selftest.fixture_root() / "artifact")
    assert mine["request"]["reduction"] == "ninjars"
    assert "reduction" not in shipped["request"]
    assert mine["plan_sha256"] == shipped["plan_sha256"]
    runtime = Runtime.load(
        output, process=selftest.NAME, model_parameters=expected["parameters"]
    )
    value = runtime.evaluate(expected["points"], mu_r2=expected["mu_r2"])
    assert {lane.reduction.requested for lane in value.diagnostics} == {"ninjars"}
    runtime = Runtime.load(
        output,
        process=selftest.NAME,
        model_parameters=expected["parameters"],
        reduction="auto",
    )
    value = runtime.evaluate(expected["points"], mu_r2=expected["mu_r2"])
    assert {lane.reduction.requested for lane in value.diagnostics} == {"auto"}


@pytest.mark.loop_physics
def test_an_exact_colour_zero_is_reported_with_its_colour_record(tmp_path):
    """h > g a (every loop has a single coloured attachment, Tr T^a = 0):
    the one exact zero (m-lit Q13) evaluates to exactly 0 and says so in
    its lane diagnostics and ``inspect``; a numerical zero never does."""

    from pyamplicol import Runtime
    from pyamplicol.api import Generator, ProcessRequest, ProcessSet
    from pyamplicol.config import resolve_config

    selftest = _selftest()
    output = tmp_path / "artifact"
    card = selftest.card(output)
    card["process"]["entries"] = [{"expression": "h > g a", "name": "h_ga"}]
    Generator(resolve_config(card, base_dir=tmp_path)).generate(
        ProcessSet((ProcessRequest.parse("h > g a", name="h_ga"),)), output
    )
    runtime = Runtime.load(output, process="h_ga")
    info = runtime.inspect()
    assert info["exact_colour_zero"] is True
    assert info["colour_zero_loop_orders"] == [{"QCD": 1, "QED": 2}]
    point = [["125", "0", "0", "0"], ["62.5", "0", "0", "62.5"]]
    point.append(["62.5", "0", "0", "-62.5"])
    value = runtime.evaluate([point], mu_r2="15625")
    assert value.contribution == "loop-induced"
    assert all(x == 0 for laurent in value.values[0] for x in laurent)
    (lane,) = value.diagnostics
    assert lane.detail.startswith("exact colour zero")
    # The colour-algebra fact, not one mechanism (a vanishing coefficient or
    # no routed family both make it).
    assert "every colour coefficient" in lane.detail
    assert "attachment" not in lane.detail


_SELFTEST = r"""
import json, sys
from pyamplicol._internal import selftest_virtual_li as selftest
root = selftest.fixture_root()
expected = json.loads((root / "expected.json").read_text())
summary, worst = selftest.run_prepared_loop_induced_self_test()
result = selftest.evaluate(root / "artifact", expected)
print(json.dumps({
    "summary": summary,
    "values": [[[float(x).hex() for x in sector] for sector in point]
               for point in result.values],
}))
"""


def _run_selftest(env: dict[str, str]) -> dict:
    run = subprocess.run(
        [sys.executable, "-c", _SELFTEST],
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
def test_self_test_runs_without_a_licence_and_touches_no_symbolica_state(tmp_path):
    """PLAN §2.a.10: a fresh interpreter without ``SYMBOLICA_LICENSE`` and
    with pyAmpliCol's native library key disabled runs the loop-induced
    self-test on the default backend while another process holds the
    restricted Symbolica permit of a private lock directory (any Symbolica
    state access would abort), with the licensed run's bits (as
    ``lit-tir``'s T8 and ``test_binary64_observable_calls_touch_no_``
    ``symbolica_state``)."""

    lock = tmp_path / "symbolica-locks"
    lock.mkdir()
    lock.chmod(0o700)
    base = {
        key: value for key, value in os.environ.items() if key != "SYMBOLICA_LICENSE"
    }
    with portability.RestrictedPermit(lock_dir=str(lock)) as permit:
        assert permit.record["verified"], permit.record
        unlicensed = _run_selftest(permit.environment({**base, **NO_NATIVE_KEY}))
    licensed = _run_selftest(dict(os.environ))
    assert unlicensed["values"] == licensed["values"]
    assert "poles vanish" in unlicensed["summary"]


@pytest.mark.loop_physics
def test_a_relocated_self_test_artifact_evaluates_bitwise(tmp_path):
    """The shipped artifact copied elsewhere evaluates bitwise equal (§0.2)."""

    selftest = _selftest()
    moved = tmp_path / "elsewhere" / "artifact"
    shutil.copytree(selftest.fixture_root() / "artifact", moved)
    expected = _expected()
    here = selftest.evaluate(selftest.fixture_root() / "artifact", expected)
    there = selftest.evaluate(moved, expected)
    assert here.values == there.values
