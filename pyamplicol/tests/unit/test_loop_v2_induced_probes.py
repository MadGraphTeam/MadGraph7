# SPDX-License-Identifier: 0BSD
"""The loop-induced generation-time probes as an opt-in (m-lit Q12 as ruled
on 2026-11-08 22:34 (3), package lit-probe).

* **Default off**: a loop-induced generation reaches none of the probes'
  stages (``induced.generation_probes``, ``ward.ward_slice``,
  ``goldstone.goldstone_slice``, ``induced._goldstone_probes``); the
  helicity-zero filter's binary64 pre-screen still runs through
  ``induced.probe_point`` / ``_probe_runs`` / ``physical_floors`` (m-lit
  Q10, it changes the artifact), and only from ``helicity_filter``; the
  record reads ``{"status": "off"}``; the stored plan, the filter record,
  its dropped rows and the binary64 values equal those of the opt-in
  artifact bitwise (g g > h: counterterm kets, no screening; g g > h z:
  screened).
* **Opt-in** ``generation.loop_induced_probes`` in the run config, the CLI
  and the Python config object; any other value refused with one message.
  Every probe runs first in binary64; only a probe whose binary64 run fails
  its verdict or is refused runs again in double-double (a test seam forces
  either); a probe that fails after the fallback refuses the process unless
  ``generation.allow_failed_probes``, which stores it with the failure.
* **A real gauge check**: g g > a a (massless quarks, nf5) generated from a
  test-only copy of the shipped loop_sm without the R2 vertex whose legs are
  the row's external particles (selected by PDG codes from the model's own
  files): the finite part changes far beyond its estimate, the finiteness
  probe still passes (R2 is finite), the binary64 Ward probe fails by far,
  the double-double fallback fails too (a defect, not rounding), and the
  public generation refuses (``loop-induced-probe``) unless allowed; the
  unmodified copy passes in binary64 with no fallback.
"""

from __future__ import annotations

import json
import shutil
import sys
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import pytest

from .loop_r2_dropped import declaration as _declaration
from .loop_r2_dropped import drop_r2_vertex as _drop_r2_vertex

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "fixtures" / "loop" / "required" / "li"
FILTER = "pyamplicol.loops.v2.helicity_filter"
# The one message of an invalid value (card, CLI and Python config).
MESSAGE = r"^generation\.loop_induced_probes must be true or false$"


def _fixture(name: str) -> dict:
    return json.loads((FIXTURES / f"{name}.json").read_text())


def _card(output: Path, expression: str, fixture: dict, source: str = "loop_sm") -> str:
    fixed = ", ".join(
        f"{key} = {json.dumps(value)}"
        for key, value in fixture["scheme_data"]["fixed_parameters"].items()
    )
    return f"""
action = "generate"
[model]
source = {json.dumps(source)}
restriction = "no_b_mass"
[process]
entries = [{{expression = {json.dumps(expression)}, name = "p"}}]
loop_coupling_orders = ["QCD"]
[loop]
r2_strategy = "model"
{_declaration(source)}[loop.preparation]
fixed_parameters = {{{fixed}}}
[generation]
output = {json.dumps(str(output))}
emit_api_bundle = false
[generation.validation]
post_build_validation = false
"""


def _config(root: Path, expression: str, fixture: dict, **generation):
    """The run config of a one-process loop-induced card, with generation
    settings set on the Python config object (``generation``)."""

    from pyamplicol.config import resolve_config

    root.mkdir(parents=True, exist_ok=True)
    card = root / "card.toml"
    source = generation.pop("source", "loop_sm")
    card.write_text(_card(root / "artifact", expression, fixture, source))
    config = resolve_config(card).effective
    if generation:
        config = replace(config, generation=replace(config.generation, **generation))
    return config


def _generate(config, expression: str) -> Path:
    from pyamplicol.api import Generator, ProcessRequest, ProcessSet

    output = Path(config.generation.output)
    Generator(config).generate(
        ProcessSet((ProcessRequest.parse(expression, name="p"),)), output
    )
    return output


def _filters(artifact: Path) -> dict:
    manifest = json.loads((artifact / "artifact.json").read_text())
    (filters,) = (
        record["filters"]
        for record in manifest["extensions"]["generation"]["concrete_processes"]
        if record["id"] == "p"
    )
    return filters


def _values(artifact: Path, fixture: dict, count: int = 2) -> list:
    """The binary64 values at the fixture's first points, as bit patterns."""

    from pyamplicol import Runtime

    runtime = Runtime.load(artifact, process="p")
    out = []
    for point, mu in list(zip(fixture["points"], fixture["mu_r"], strict=True))[:count]:
        value = runtime.evaluate([point], mu_r2=[str(Fraction(mu) ** 2)])
        out.append([[float(x).hex() for x in laurent] for laurent in value.values[0]])
    return out


# ------------------------------------------------------------------ option


@pytest.mark.loop_fast
def test_the_option_is_off_by_default_and_set_in_config_cli_and_python(tmp_path):
    """``generation.loop_induced_probes``: off by default; enabled by the run
    config (TOML), the CLI (``--loop-induced-probes``, ``--no-...``) and the
    Python config object; any non-boolean value is refused with the one
    message; ``allow_failed_probes`` stays its own setting."""

    from pyamplicol.cli import parse_cli
    from pyamplicol.config import ConfigurationError, GenerationConfig, resolve_config

    fixture = _fixture("li_gg_h")
    default = _config(tmp_path / "default", "g g > h", fixture)
    assert default.generation.loop_induced_probes is False
    card = tmp_path / "on.toml"
    card.write_text(
        _card(tmp_path / "out", "g g > h", fixture).replace(
            "[generation]\n", "[generation]\nloop_induced_probes = true\n"
        )
    )
    assert resolve_config(card).effective.generation.loop_induced_probes is True
    bad = tmp_path / "bad.toml"
    bad.write_text(card.read_text().replace("= true", '= "yes"'))
    with pytest.raises(ConfigurationError, match=MESSAGE):
        resolve_config(bad)
    cli = parse_cli(("generate",)).resolve().effective.generation
    assert cli.loop_induced_probes is False
    on = parse_cli(("generate", "--loop-induced-probes")).resolve().effective
    assert on.generation.loop_induced_probes is True
    off = parse_cli(("generate", "--no-loop-induced-probes")).resolve().effective
    assert off.generation.loop_induced_probes is False
    assert not on.generation.allow_failed_probes
    python = GenerationConfig(loop_induced_probes=True, allow_failed_probes=True)
    assert python.loop_induced_probes and python.allow_failed_probes
    with pytest.raises(ConfigurationError, match=MESSAGE):
        GenerationConfig(loop_induced_probes="true")


# ------------------------------------------------------------ default off


def _caller_modules() -> set[str]:
    frame = sys._getframe(2)
    modules = set()
    while frame is not None:
        modules.add(frame.f_globals.get("__name__", ""))
        frame = frame.f_back
    return modules


@pytest.mark.loop_physics
@pytest.mark.parametrize(
    ("expression", "name", "screened"),
    [("g g > h", "li_gg_h", False), ("g g > h z", "li_gg_hz", True)],
    ids=["gg_h-kets", "gg_hz-screened"],
)
def test_default_generation_runs_no_probe_and_equals_the_opt_in_artifact(
    tmp_path, monkeypatch, expression, name, screened
):
    """DECISIONS 2026-11-08 22:34 (3): the default generation of a
    loop-induced process calls none of the probes' stages and records them
    as not run; the helicity-zero filter's binary64 pre-screen (Q10) runs
    the shared probe machinery only from ``helicity_filter`` (g g > h has
    counterterm kets: nothing is screened); the stored plan, the filter
    record (less its timings), the dropped rows and the binary64 values at
    two points are bitwise those of the opt-in generation, whose probes
    pass in binary64 alone."""

    from pyamplicol.loops.v2 import goldstone, induced, ward

    fixture = _fixture(name)

    def refused(*args, **kwargs):
        raise AssertionError("a probe stage ran in a default generation")

    calls: list[str] = []

    def screened_only(function):
        def wrapper(*args, **kwargs):
            assert FILTER in _caller_modules(), function.__name__
            calls.append(function.__name__)
            return function(*args, **kwargs)

        return wrapper

    with monkeypatch.context() as patch:
        patch.setattr(induced, "generation_probes", refused)
        patch.setattr(induced, "_goldstone_probes", refused)
        patch.setattr(ward, "ward_slice", refused)
        patch.setattr(goldstone, "goldstone_slice", refused)
        for function in ("probe_point", "_probe_runs", "physical_floors"):
            patch.setattr(induced, function, screened_only(getattr(induced, function)))
        default = _generate(
            _config(tmp_path / "default", expression, fixture), expression
        )
    assert bool(calls) is screened
    on = _generate(
        _config(tmp_path / "on", expression, fixture, loop_induced_probes=True),
        expression,
    )
    off_filters, on_filters = _filters(default), _filters(on)
    assert off_filters["loop_induced_probes"] == {"status": "off"}
    probes = on_filters["loop_induced_probes"]
    assert probes["status"] == "run" and "failed" not in probes
    assert probes["fallbacks"] == 0 and probes["finiteness"]["passed"]
    assert probes["ward"]["passed"] and probes["ward"]["legs"] == [1]
    assert {entry["carrier"] for entry in probes["probes"]} == {"double"}
    assert probes["reduction"] == {
        "requested": "ninjars",
        "effective": "ninjars",
        "reason": induced.PROBE_REDUCTION_REASON,
    }
    helicity = {
        key: value
        for key, value in off_filters["loop_induced_helicity_filter"].items()
        if key != "seconds"
    }
    assert helicity == {
        key: value
        for key, value in on_filters["loop_induced_helicity_filter"].items()
        if key != "seconds"
    }
    assert (helicity["counterterms"] == "absent") is screened
    plan = Path("processes") / "p" / "loop-runtime.pacbin"
    assert (default / plan).read_bytes() == (on / plan).read_bytes()
    assert _values(default, fixture) == _values(on, fixture)


# ---------------------------------------------------- binary64 + fallback


@pytest.fixture(scope="module")
def gg_h(tmp_path_factory):
    """g g > h on the shipped loop_sm (no_b_mass, zero widths): compiled
    model, native, slice and candidate points (generated once)."""

    from pyamplicol.loops.native import load_loop_native
    from pyamplicol.loops.v2.assemble import LoopSlice, SliceRequest
    from pyamplicol.models.loading import compile_model_source

    fixture = _fixture("li_gg_h")
    compiled = compile_model_source(
        "loop_sm",
        restriction="no_b_mass",
        cache_dir=tmp_path_factory.mktemp("probes-gg-h") / "model-cache",
    )
    native = load_loop_native()
    orders = dict(fixture["loop_orders"])
    fixed = {
        key: value
        for key, value in fixture["scheme_data"]["fixed_parameters"].items()
        if key in compiled.parameter_defaults
    }
    slice_ = LoopSlice.generate(
        compiled,
        SliceRequest(
            "g g > h",
            orders,
            orders,
            contribution="loop-induced",
            fixed_parameters=fixed,
        ),
        native,
    )
    candidates = [[[float(x) for x in p] for p in point] for point in fixture["points"]]
    return SimpleNamespace(
        compiled=compiled, native=native, slice=slice_, candidates=candidates
    )


def _seam(monkeypatch, force):
    """Wrap ``induced.probe_results``: ``force(carrier, ward, result)``
    returns the result the generation-time probes see (a refusal text or a
    ``ProbeResult``); every pass is recorded as (carrier, probe kinds)."""

    from pyamplicol.loops.v2 import induced

    original = induced.probe_results
    passes: list[tuple[str, list[str]]] = []

    def wrapper(executor, point, scale, wards, **kwargs):
        found = original(executor, point, scale, wards, **kwargs)
        carrier = kwargs["precision"].arithmetic
        passes.append((carrier, ["finiteness" if w is None else "ward" for w in wards]))
        return [
            force(carrier, ward, result)
            for ward, result in zip(wards, found, strict=True)
        ]

    monkeypatch.setattr(induced, "probe_results", wrapper)
    return passes


@pytest.mark.loop_physics
def test_only_the_probe_refused_in_binary64_runs_in_double_double(gg_h, monkeypatch):
    """A Ward run the backend refuses in binary64 (a test seam) is re-run in
    double-double on the same lifted point, and only it: the finiteness probe
    keeps its binary64 verdict; the record names the deciding carriers and
    the refusal."""

    from pyamplicol.loops.v2 import induced

    def force(carrier, ward, result):
        if carrier == "double" and ward is not None:
            return "the probe point failed: forced refusal (test seam)"
        return result

    passes = _seam(monkeypatch, force)
    record = induced.generation_probes(
        gg_h.slice, gg_h.compiled, gg_h.native, gg_h.candidates
    )
    assert passes == [
        ("double", ["finiteness", "ward"]),
        ("double-double", ["ward"]),
    ]
    assert record["finiteness"] == {
        "worst": record["finiteness"]["worst"],
        "passed": True,
        "carrier": "double",
    }
    assert record["ward"]["passed"] and record["ward"]["carriers"] == {
        "1": "double-double"
    }
    finiteness, ward = record["probes"]
    assert "fallback" not in finiteness
    assert ward["fallback"] == {
        "carrier": "double",
        "reason": "refused",
        "detail": "the probe point failed: forced refusal (test seam)",
    }
    assert record["fallbacks"] == 1 and "failed" not in record
    assert set(record["seconds"]) >= {"double", "double-double", "total"}


@pytest.mark.loop_physics
def test_only_the_probe_failing_in_binary64_runs_in_double_double(gg_h, monkeypatch):
    """A finiteness verdict failed in binary64 (a test seam: ratio 1e3) is
    re-run in double-double, where it passes, which decides it; the Ward
    probe keeps its binary64 verdict."""

    from types import MappingProxyType

    from pyamplicol.loops.v2 import induced

    def force(carrier, ward, result):
        if carrier == "double" and ward is None:
            return induced.ProbeResult(
                "finiteness",
                MappingProxyType(dict.fromkeys(result.ratios, 1e3)),
                result.scales,
            )
        return result

    passes = _seam(monkeypatch, force)
    record = induced.generation_probes(
        gg_h.slice, gg_h.compiled, gg_h.native, gg_h.candidates
    )
    assert passes == [
        ("double", ["finiteness", "ward"]),
        ("double-double", ["finiteness"]),
    ]
    finiteness, ward = record["probes"]
    assert finiteness["carrier"] == "double-double" and finiteness["passed"]
    assert finiteness["fallback"] == {
        "carrier": "double",
        "reason": "verdict",
        "worst": 1000.0,
    }
    assert ward["carrier"] == "double" and "fallback" not in ward
    assert record["finiteness"]["carrier"] == "double-double"
    assert record["fallbacks"] == 1 and "failed" not in record


@pytest.mark.loop_physics
def test_a_probe_refused_in_every_carrier_fails_generation_unless_allowed(
    gg_h, monkeypatch
):
    """A probe without a verdict in any carrier is a failure, never a pass:
    ``ProbeFailure`` (``loop-induced-probe``) unless allowed, which records
    the refusal and the carrier that last refused it."""

    from pyamplicol.loops.v2 import induced
    from pyamplicol.loops.v2.refusals import LoopEngineRefusal, keyed_error

    def force(carrier, ward, result):
        if ward is not None:
            return f"the probe point failed: forced refusal in {carrier}"
        return result

    _seam(monkeypatch, force)
    with pytest.raises(induced.ProbeFailure, match="ward probe") as refused:
        induced.generation_probes(
            gg_h.slice, gg_h.compiled, gg_h.native, gg_h.candidates
        )
    typed = keyed_error(refused.value)
    assert isinstance(typed, LoopEngineRefusal) and typed.key == "loop-induced-probe"
    record = induced.generation_probes(
        gg_h.slice, gg_h.compiled, gg_h.native, gg_h.candidates, allow_failure=True
    )
    assert record["allowed_failure"] and not record["ward"]["passed"]
    (ward,) = (entry for entry in record["probes"] if entry["probe"] == "ward")
    assert ward["carrier"] == "double-double" and not ward["passed"]
    assert ward["refused"].endswith("forced refusal in double-double")
    assert ward["fallback"]["reason"] == "refused"
    assert record["finiteness"]["passed"] and record["finiteness"]["carrier"] == (
        "double"
    )


@pytest.mark.loop_fast
def test_a_goldstone_probe_refused_in_binary64_runs_in_double_double(monkeypatch):
    """The Goldstone probe of a species takes the same carriers: refused in
    binary64 (a test seam), it runs in double-double, which decides it; a
    species passing in binary64 runs once; the Goldstone slice's generation
    and each probe are timed in the record."""

    from pyamplicol.loops.v2 import goldstone, induced, plan_export
    from pyamplicol.loops.v2.goldstone import GoldstoneResult

    calls: list[tuple[int, str]] = []

    def probe(vector, partner, point, scale, leg, *, precision, physical):
        calls.append((leg, precision.arithmetic))
        if leg == 3 and precision.arithmetic == "double":
            raise induced.ProbePointRefused("the probe point failed: forced (seam)")
        return GoldstoneResult(leg, {("o", "r"): 0.5}, None, 0.1)

    monkeypatch.setattr(
        goldstone, "massive_vector_legs", lambda target: {3: 250, 4: 251}
    )
    monkeypatch.setattr(goldstone, "goldstone_slice", lambda *args: object())
    monkeypatch.setattr(goldstone, "probe", probe)
    monkeypatch.setattr(plan_export, "NativeExecutor", lambda *a, **k: object())
    target = SimpleNamespace(
        compiled=SimpleNamespace(ir=SimpleNamespace(gauge="feynman"))
    )
    found = induced._goldstone_probes(target, None, None, None, None, None, None, {})
    assert calls == [(3, "double"), (3, "double-double"), (4, "double")]
    record = found["record"]
    assert record["passed"] and record["carriers"] == {
        "3": "double-double",
        "4": "double",
    }
    three, four = found["entries"]
    assert three["fallback"] == {
        "carrier": "double",
        "reason": "refused",
        "detail": "the probe point failed: forced (seam)",
    }
    assert "fallback" not in four
    assert set(found["seconds"]["goldstone_slices"]) == {"3", "4"}
    assert set(found["seconds"]["goldstone"]) == {"3", "4"}


# --------------------------------------------------- a dropped R2 vertex


@pytest.mark.loop_physics
def test_the_binary64_ward_probe_catches_a_dropped_r2_vertex(tmp_path):
    """DECISIONS 2026-11-08 22:34 (3): g g > a a (massless quarks, nf5) from a
    copy of the shipped loop_sm without the R2 vertex whose legs are the
    row's external particles (by PDG code), generated through the production
    path (ADDENDUM J: R2 vertices are plain tree-level vertices). The vertex
    contributes (the finite part at the probe point moves far beyond its
    estimate); the finiteness probe still passes in binary64 (R2 is finite);
    the binary64 Ward probe fails by far, and its double-double fallback on
    the same lifted point too (a defect, not rounding); the public
    generation with the option refuses (``loop-induced-probe``) and with
    ``allow_failed_probes`` stores the artifact with the failure recorded.
    The unmodified copy passes in binary64 with no fallback."""

    from importlib import resources

    from pyamplicol.loops.native import load_loop_native
    from pyamplicol.loops.v2 import induced
    from pyamplicol.loops.v2.assemble import LoopSlice, Precision, SliceRequest
    from pyamplicol.loops.v2.plan_export import NativeExecutor
    from pyamplicol.loops.v2.refusals import LoopEngineRefusal
    from pyamplicol.loops.v2.stability import RECOMMENDED_CHECKS
    from pyamplicol.models.loading import compile_model_source

    fixture = _fixture("li_gg_aa_nf5")
    shipped = Path(
        str(resources.files("pyamplicol").joinpath("assets", "models", "ufo"))
    )
    intact, dropped = tmp_path / "loop_sm", tmp_path / "loop_sm_r2_dropped"
    shutil.copytree(shipped / "loop_sm", intact)
    shutil.copytree(shipped / "loop_sm", dropped)
    assert _drop_r2_vertex(dropped, fixture["pdgs"])
    native = load_loop_native()
    orders = dict(fixture["loop_orders"])
    candidates = [[[float(x) for x in p] for p in point] for point in fixture["points"]]
    slices, records = {}, {}
    for label, ufo in (("intact", intact), ("dropped", dropped)):
        compiled = compile_model_source(
            ufo,
            restriction="no_b_mass",
            require_supported=False,
            cache_dir=tmp_path / f"cache-{label}",
        )
        fixed = {
            key: value
            for key, value in fixture["scheme_data"]["fixed_parameters"].items()
            if key in compiled.parameter_defaults
        }
        slices[label] = slice_ = LoopSlice.generate(
            compiled,
            SliceRequest(
                "g g > a a",
                orders,
                orders,
                contribution="loop-induced",
                fixed_parameters=fixed,
                particle_veto=(6,),
            ),
            native,
        )
        records[label] = induced.generation_probes(
            slice_, compiled, native, candidates, allow_failure=True
        )
    intact_record, dropped_record = records["intact"], records["dropped"]
    # The unmodified model: binary64 alone, every probe passes.
    assert "failed" not in intact_record and intact_record["fallbacks"] == 0
    assert {e["carrier"] for e in intact_record["probes"]} == {"double"}
    assert intact_record["ward"]["legs"] == [1, 3]
    # The vertex contributes: the finite part at the probe point moves far
    # beyond the binary64 estimate.
    digits = intact_record["lift_digits"]
    point, _distance = induced.probe_point(
        slices["intact"], candidates[intact_record["candidate"]], digits
    )
    scale = induced._point_scale(point, 2)
    finite, estimate = {}, {}
    for label, slice_ in slices.items():
        executor = NativeExecutor(slice_, forward=False, inputs="native")
        (result,) = executor.evaluate(
            [point],
            [scale],
            precision=Precision.double(),
            stability=RECOMMENDED_CHECKS,
        )
        (sector,) = result.stability.absolute
        finite[label] = float(result.totals[1])
        estimate[label] = result.stability.absolute[sector].coefficients[0]
    moved = abs(finite["dropped"] - finite["intact"])
    assert moved > 1e3 * (estimate["intact"] + estimate["dropped"])
    assert moved > 1e-3 * abs(finite["intact"])
    # The modified model: finiteness passes in binary64; each Ward probe
    # fails by far in binary64 and again in double-double.
    assert dropped_record["finiteness"] == {
        "worst": dropped_record["finiteness"]["worst"],
        "passed": True,
        "carrier": "double",
    }
    wards = [e for e in dropped_record["probes"] if e["probe"] == "ward"]
    assert [e["leg"] for e in wards] == [1, 3]
    for entry in wards:
        assert entry["fallback"]["carrier"] == "double"
        assert entry["fallback"]["reason"] == "verdict"
        assert entry["fallback"]["worst"] > 1e6
        assert entry["carrier"] == "double-double" and not entry["passed"]
        assert entry["worst"] > 1e6
    assert dropped_record["allowed_failure"] and dropped_record["fallbacks"] == 2
    # The public route with the option: refused, or stored with the failure.
    expression = "g g > a a"
    with pytest.raises(LoopEngineRefusal) as refused:
        _generate(
            _config(
                tmp_path / "refused",
                expression,
                fixture,
                source=str(dropped),
                particle_veto=("t",),
                loop_induced_probes=True,
            ),
            expression,
        )
    assert refused.value.key == "loop-induced-probe"
    stored = _generate(
        _config(
            tmp_path / "allowed",
            expression,
            fixture,
            source=str(dropped),
            particle_veto=("t",),
            loop_induced_probes=True,
            allow_failed_probes=True,
        ),
        expression,
    )
    probes = _filters(stored)["loop_induced_probes"]
    assert probes["allowed_failure"] and not probes["ward"]["passed"]
    assert probes["ward"]["carriers"] == {"1": "double-double", "3": "double-double"}
    assert all("in double-double" in failure for failure in probes["failed"])
