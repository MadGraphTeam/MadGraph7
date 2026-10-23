# SPDX-License-Identifier: 0BSD
"""Plan v4 artifacts and the public API through the v2 engine (WP2.6).

``loop.engine = "v2"`` generates virtual processes through the shared-recursion
engine and publishes them as plan v4 processes (``rusticol.loop-runtime.v4``).
Checks, on the public surface only (TOML card, ``Generator``, ``Runtime``, CLI):

* the inclusive virtual of the legacy O1 controls through the public path
  equals the frozen 200-bit legacy values per squared coupling order and
  Laurent coefficient: gg -> gg, u u~ -> d d~ g and u u~ -> t t~ (an open
  massive quark line through the loop) at full colour in binary64,
  double-double and arbitrary precision, u u~ -> d d~ and u u~ -> u u~
  (identical-flavour signs) likewise, gg -> gg and u u~ -> t t~ at leading
  colour and u u~ -> d d~ g at next-to-leading colour in binary64 and
  double-double, and u u~ -> u u~ g and u u~ -> g g g at full colour
  (nightly); Born values are held to 1e-12 in binary64;
* colour correlations (dipoles, Casimirs, ordered two-emission histories and
  their swaps), complex spin sources and helicity-resolved outputs through
  the public generator and runtime equal the frozen legacy observables (O1)
  at full colour (g g > g g; d d~ > g g with s s~ > g g served as an exact
  flavour member) and at leading colour, in binary64 (virtuals held to
  1e-12 unless the value's own stability estimate E exceeds it, then to
  E <= 1e-9; Borns to 1e-12), in double-double (1e-24) and, for g g > g g,
  in arbitrary precision (60 digits, 1e-40); the NLC, LC/NLC member and
  five-leg records nightly, the full-colour ones also in 60 digits;
* ``poles=False`` returns the finite parts with absent poles, also for
  correlated and resolved outputs, from the same artifact;
* stability checks (rotations by count or exact specification, the
  reduction-direction check) return one estimate per point that bounds the
  true error, and no estimate without checks;
* a fresh interpreter rebuilds the per-point inputs from the artifact, with
  the same plan and bitwise-identical binary64 values also after unrelated
  Symbolica symbols were created (the regeneration does not depend on the
  interpreter's symbol history), and refuses a stored plan that differs
  from the regenerated one;
* the native artifact loader (``load_loop_v2_artifact``) admits each plan v4
  process, cross-checks families, buckets, legs, helicity rows and sectors
  against the declarations, and refuses tampered declarations and digests;
  the legacy native virtual runtime (the SDK's loader) refuses plan v4 with
  its reason;
* ``pyamplicol inspect`` summarizes v2 and legacy loop artifacts (engine,
  sectors, families, buckets, colour degrees, plan digest);
* the artifact holds no absolute path, user or host name (relocatable,
  requirements addendum G) and loads from a copy at another path with
  bitwise-identical binary64 values;
* requests v2 does not cover yet fail with the reason and leave
  ``loop.engine = "legacy"`` (the default) selectable; the slice request
  records every ``SliceRequest`` field; several corrections take one exact
  loop order each;
* a six-gluon plan v4 artifact stays below 20 MB (nightly).
"""

from __future__ import annotations

import io
import json
import subprocess
import sys
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

import pytest

from pyamplicol.api import (
    LaurentCoefficients,
    VirtualStabilityChecks,
    VirtualStabilityEstimate,
)
from pyamplicol.api.errors import (
    CompatibilityError,
    EvaluationError,
    GenerationError,
)
from pyamplicol.api.virtual_results import parse_rotation
from pyamplicol.loops.v2.stability import Rotation

ROOT = Path(__file__).resolve().parents[2]
LEGACY = ROOT / "tests" / "fixtures" / "loop" / "legacy"
DOUBLE_TOLERANCE = Decimal("1e-12")
DOUBLE_DOUBLE_TOLERANCE = Decimal("1e-24")
# Binary64 O1 rule (user, 2026-10-23): a value whose own stability estimate E
# exceeds 1e-12 is held to E, which must not exceed 1e-9 (and the value must
# then agree in double-double, which every comparison here runs anyway).
ESTIMATE_CAP = 1e-9
ARBITRARY_DIGITS = 60
ARBITRARY_TOLERANCE = Decimal("1e-40")
OBSERVABLES = LEGACY / "observables"
BINARY64_CHECKS = VirtualStabilityChecks(("z:3/5:4/5",), True)
GG_PARAMETERS = {"aS": "0.118", "MT": "173"}
POINT = [[500, 0, 0, 500], [500, 0, 0, -500], [500, 300, 0, 400], [500, -300, 0, -400]]

SCHEME = """
[loop.model]
loop_coupling_orders = ["QCD"]
uv_complete = true
model_r2_complete = true
[loop.model.scheme]
dimensional_regularization = "HV"
external_states = "physical-four-dimensional"
gamma5 = "source-UFO-four-dimensional-prescription"
renormalization = "source-UFO mass-dependent decoupling"
mass_scheme = "real-on-shell"
finite_restoration = "source-UFO-local-rules"
model_convention = "loop_sm lhv=1 supplied QCD rules"
"""


def _card(
    model, output, entries, *, accuracy="full", declared=True, extra="", process=""
):
    processes = ", ".join(
        f"{{expression={json.dumps(expression)}, name={json.dumps(name)}}}"
        for name, expression in entries
    )
    return f"""
action = "generate"
[model]
source = {json.dumps(str(model))}
[process]
entries = [{processes}]
loop_coupling_orders = ["QCD"]
coupling_order_policy = "minimal"
{process}
[color]
accuracy = "{accuracy}"
contraction = "direct"
[loop]
r2_strategy = "model"
engine = "v2"
[loop.preparation]
fixed_parameters = {{MB = "0"}}
{SCHEME if declared else ""}
[generation]
output = {json.dumps(str(output))}
emit_api_bundle = false
[generation.validation]
post_build_validation = true
samples = 1
{extra}
"""


def _generate(card_text, tmp: Path, entries, correlators=None):
    from pyamplicol.api import Generator, ProcessRequest, ProcessSet
    from pyamplicol.config import resolve_config

    card = tmp / "card.toml"
    card.write_text(card_text)
    processes = ProcessSet(
        tuple(
            ProcessRequest.parse(expression, name=name) for name, expression in entries
        )
    )
    output = tmp / "artifact"
    Generator(resolve_config(card)).generate(processes, output, correlators=correlators)
    return output


def _fixture(name: str) -> dict:
    return json.loads((LEGACY / f"{name}.json").read_text())


def _parameters(fixture: dict) -> dict[str, str]:
    return {k: v for k, v in fixture["parameters"].items() if k != "MU_R"}


def _assert_legacy(result, results, tolerance, virtual_tolerance=None):
    """Born and the complete virtual of every point against legacy O1
    (``virtual_tolerance(index)``, if given, holds the virtual of a point)."""

    for index in range(len(result.values)):
        born = Decimal(results["born"][index][0][0])
        ((mine,),) = (result.born_values[index],)
        assert abs(Decimal(mine) - born) <= tolerance * abs(born), (index, mine)
        reference = [Decimal(x[0]) for x in results["virtual"][index][0]]
        ((laurent,),) = (result.values[index],)
        scale = max(map(abs, reference))
        held = tolerance if virtual_tolerance is None else virtual_tolerance(index)
        for k, (a, b) in enumerate(zip(laurent, reference, strict=True)):
            assert abs(Decimal(a) - b) <= held * scale, (index, k, a, b)


def _assert_inclusive_o1(artifact, name: str, process: str, arbitrary=True) -> None:
    """The public inclusive virtual of one full-colour legacy control in
    binary64 (1e-12, no checks requested), double-double (1e-24) and, if
    ``arbitrary``, 60-digit arbitrary precision (1e-40)."""

    from pyamplicol import Runtime

    fixture = _fixture(name)
    results = fixture["results"]["arbitrary-200"]
    runtime = Runtime.load(
        artifact, process=process, model_parameters=_parameters(fixture)
    )
    points, scales = fixture["points"], fixture["scales_squared"]
    double = runtime.evaluate(points, mu_r2=scales)
    assert double.shape == (len(points), 1, 3) and double.stability is None
    assert all(item.precision_bits == 53 for item in double.diagnostics)
    _assert_legacy(double, results, DOUBLE_TOLERANCE)
    if arbitrary:
        value = runtime.evaluate(
            points, mu_r2=scales, arithmetic="arbitrary", precision=ARBITRARY_DIGITS
        )
        assert isinstance(value.values[0][0].finite, Decimal)
        _assert_legacy(value, results, ARBITRARY_TOLERANCE)
    double_double = runtime.evaluate(points, mu_r2=scales, arithmetic="double-double")
    assert {item.precision_bits for item in double_double.diagnostics} == {106}
    assert isinstance(double_double.values[0][0].finite, Decimal)
    _assert_legacy(double_double, results, DOUBLE_DOUBLE_TOLERANCE)
    # One squared coupling order per partial result; the sector names its pairs.
    (sector,) = double.sectors
    assert double.for_order(sector.order).values == double.values


# ------------------------------------------------------------------ fast checks


@pytest.mark.loop_fast
def test_stability_checks_take_counts_exact_rotations_and_text():
    none = VirtualStabilityChecks()
    assert not none.enabled and none.rotations == ()
    counted = VirtualStabilityChecks(3)
    assert len(counted.rotations) == 3 and counted.enabled
    assert len({rotation.matrix for rotation in counted.rotations}) == 3
    assert all(not rotation.is_identity for rotation in counted.rotations)
    # Default rotations are deterministic.
    assert VirtualStabilityChecks(3).rotations == counted.rotations
    exact = VirtualStabilityChecks(("z:3/5:4/5", "1,1,0:0.7"), True)
    assert exact.rotations[0] == Rotation.about("z", "3/5", "4/5")
    assert exact.rotations[1] == parse_rotation("1,1,0:0.7")
    assert parse_rotation("x:-4/5:3/5").matrix[1][1] == Fraction(-4, 5)
    assert exact.to_checks().reduction_direction
    assert VirtualStabilityChecks(reduction_direction=True).enabled
    for bad in ("z:1/2:1/2", "w:1:0", "1,0:0.3", "z:1:0"):
        with pytest.raises(ValueError):
            VirtualStabilityChecks((bad,))
    with pytest.raises(ValueError):
        VirtualStabilityChecks(-1)
    with pytest.raises(TypeError):
        VirtualStabilityChecks(True)


@pytest.mark.loop_fast
def test_finite_only_results_have_absent_poles_and_estimates_are_typed():
    from pyamplicol.api.virtual_results import (
        VirtualEvaluation,
        VirtualLaneDiagnostic,
    )
    from pyamplicol.config import R2Strategy
    from pyamplicol.loops.orders import BornLoopPair, VirtualOrderSector
    from pyamplicol.loops.strategy import R2Selection

    sector = VirtualOrderSector(
        (("QCD", 6),), (BornLoopPair((("QCD", 2),), (("QCD", 4),)),)
    )
    lane = VirtualLaneDiagnostic("success", "double", 53, False)
    r2 = R2Selection(R2Strategy.MODEL, R2Strategy.MODEL, "test")
    estimate = VirtualStabilityEstimate(
        1e-13, 2e-13, None, None, (("rotation-0", 1e-14),)
    )
    result = VirtualEvaluation(
        ((1.0,),),
        (((-2.0, None, None),),),
        ((("QCD", 4),),),
        (sector,),
        (lane,),
        r2,
        poles=False,
        stability=(estimate,),
    )
    assert result.values[0][0] == LaurentCoefficients(-2.0, None, None)
    assert result.finite == ((-2.0,),)
    with pytest.raises(ValueError, match="no poles"):
        result.single_pole  # noqa: B018
    assert result.sum_orders() == (LaurentCoefficients(-2.0, None, None),)
    assert result.for_order((("QCD", 6),)).stability == (estimate,)
    with pytest.raises(ValueError, match="finite-only"):
        VirtualEvaluation(
            ((1.0,),),
            (((-2.0, 1.0, None),),),
            ((("QCD", 4),),),
            (sector,),
            (lane,),
            r2,
            poles=False,
        )
    with pytest.raises(ValueError, match="pole estimates"):
        VirtualEvaluation(
            ((1.0,),),
            (((-2.0, 1.0, 3.0),),),
            ((("QCD", 4),),),
            (sector,),
            (lane,),
            r2,
            stability=(estimate,),
        )
    with pytest.raises(ValueError, match="estimate"):
        VirtualEvaluation(
            ((1.0,),),
            (((-2.0, 1.0, 3.0),),),
            ((("QCD", 4),),),
            (sector,),
            (lane,),
            r2,
            stability=(None,),
        )
    with pytest.raises(ValueError):
        VirtualStabilityEstimate(float("nan"), 0.0, None, None, (("x", 0.0),))
    with pytest.raises(ValueError):
        VirtualStabilityEstimate(0.0, 0.0, 1.0, None, (("x", 0.0),))


@pytest.mark.loop_fast
def test_configuration_selects_the_engine_and_the_runtime_options(tmp_path):
    from pyamplicol.config import ConfigurationError, LoopEngine, resolve_config

    card = tmp_path / "card.toml"
    card.write_text(
        'action = "generate"\n[loop]\nengine = "v2"\n[evaluation]\npoles = false\n'
        '[evaluation.stability_checks]\nrotations = ["z:3/5:4/5"]\n'
        "reduction_direction = true\n"
    )
    config = resolve_config(card).effective
    assert config.loop.engine is LoopEngine.V2
    assert config.evaluation.poles is False
    checks = config.evaluation.stability_checks.to_checks()
    assert checks.rotations == (Rotation.about("z", "3/5", "4/5"),)
    assert checks.reduction_direction
    default = resolve_config(None, action="generate").effective
    assert default.loop.engine is LoopEngine.LEGACY
    assert default.evaluation.stability_checks.to_checks() is None
    card.write_text(
        'action = "generate"\n[evaluation.stability_checks]\n'
        'rotations = ["z:3/5:4/5"]\nrotation_count = 2\n'
    )
    with pytest.raises(ConfigurationError):
        resolve_config(card)


@pytest.mark.loop_fast
def test_slice_request_records_round_trip():
    """Every ``SliceRequest`` field is recorded and restored (a field added
    by another package needs no edit of the record); a record with an
    unknown or a missing field is refused."""

    from dataclasses import fields

    from pyamplicol.api.errors import ArtifactError
    from pyamplicol.loops.v2.artifact import request_from_record, request_record
    from pyamplicol.loops.v2.assemble import SliceRequest
    from pyamplicol.models.loop_color_declaration import LoopColorDeclaration

    declaration = LoopColorDeclaration.from_dict(
        json.loads(
            (ROOT / "examples/loop/source_colour/loop_sm_qcd_subset.json").read_text()
        )
    )
    request = SliceRequest(
        "g g > g g",
        {"QCD": 2, "QED": 0},
        ({"QCD": 4, "QED": 0}, {"QCD": 2, "QED": 2}),
        corrections=("QCD", "QED"),
        fixed_parameters={"MB": "0"},
        ignored_particle_ids=(23,),
        accuracy="lc",
        colour_declaration=declaration,
        scale_parameter="MU_R",
        alpha_s_parameter="aS",
        compress_colour=False,
        colour_contraction="direct",
    )
    default = SliceRequest("g g > g g", {"QCD": 2}, {"QCD": 4})
    for item in fields(SliceRequest):
        if item.name not in ("process", "born_orders"):
            assert getattr(request, item.name) != getattr(default, item.name), item
    record = request_record(request)
    assert set(record) == {item.name for item in fields(SliceRequest)} | {"correlators"}
    assert json.loads(json.dumps(record)) == record
    assert request_from_record(record) == request
    single = request_record(default)
    assert single["loop_orders"] == {"QCD": 4}
    assert request_from_record(single) == default
    with pytest.raises(ArtifactError, match=r"unknown.*future_option"):
        request_from_record({**record, "future_option": 1})
    with pytest.raises(ArtifactError, match=r"missing.*compress_colour"):
        request_from_record(
            {key: value for key, value in record.items() if key != "compress_colour"}
        )


@pytest.mark.loop_fast
def test_corrections_take_one_loop_order_each_and_one_born_order():
    """The service derives one exact loop order per correction (the Born
    limits raised by two powers of that correction) and refuses a process
    with several Born amplitude orders with the reason."""

    from types import SimpleNamespace

    from pyamplicol.generation.loop_service import (
        _require_one_born_order,
        v2_loop_orders,
    )

    limits = {"QCD": 2, "QED": 0}
    assert v2_loop_orders(limits, ("QCD",)) == {"QCD": 4, "QED": 0}
    assert v2_loop_orders(limits, ("QCD", "QED")) == (
        {"QCD": 4, "QED": 0},
        {"QCD": 2, "QED": 2},
    )
    with pytest.raises(GenerationError):
        v2_loop_orders(limits, ())

    def generated(*orders):
        blocks = SimpleNamespace(born_orders=orders)
        return SimpleNamespace(slice=SimpleNamespace(blocks=blocks))

    _require_one_born_order("one", generated((("QCD", 2),)))
    with pytest.raises(
        GenerationError,
        match=r"several Born amplitude orders.*Born coupling-order limits.*"
        r"loop\.engine='legacy'",
    ):
        _require_one_born_order(
            "mixed", generated((("QCD", 2),), (("QCD", 1), ("QED", 1)))
        )


# ------------------------------------------------------------------ public path


@pytest.fixture(scope="module")
def full_colour(loop_sm_compiled_model, loop_native, tmp_path_factory):
    tmp = tmp_path_factory.mktemp("v2-artifact")
    entries = (
        ("gg", "g g > g g"),
        ("uux_ddxg", "u u~ > d d~ g"),
        ("tt", "u u~ > t t~"),
    )
    return _generate(
        _card(loop_sm_compiled_model, tmp / "artifact", entries), tmp, entries
    )


@pytest.fixture(scope="module")
def quark_controls(loop_sm_compiled_model, loop_native, tmp_path_factory):
    """u u~ > d d~ and u u~ > u u~ (identical-flavour signs) through v2."""

    tmp = tmp_path_factory.mktemp("v2-quarks")
    entries = (("uudd", "u u~ > d d~"), ("uux_uux", "u u~ > u u~"))
    return _generate(
        _card(loop_sm_compiled_model, tmp / "artifact", entries), tmp, entries
    )


@pytest.mark.loop_physics
def test_public_generation_publishes_plan_v4(full_colour):
    from pyamplicol import Runtime
    from pyamplicol.artifacts import load_manifest

    manifest = load_manifest(full_colour, verify_payloads=True)
    assert list(manifest.runtime["required_runtime_capabilities"]) == [
        "rusticol.loop-runtime.v4"
    ]
    assert {record["id"] for record in manifest.processes} == {
        "gg",
        "uux_ddxg",
        "tt",
    }
    runtime = Runtime.load(full_colour, process="uux_ddxg")
    assert runtime.contribution_kind == "virtual"
    info = runtime.inspect()
    assert info["engine"] == "v2" and info["schedule"] == "backward"
    assert info["families"] == 42
    assert sum(count for _n, _r, count in info["buckets"]) == 42
    assert info["poles"].startswith("computed") and not info["rescue_available"]
    physics = runtime.physics
    assert [p.pdg_id for p in physics.external_particles] == [2, -2, 1, -1, 21]


@pytest.mark.loop_physics
@pytest.mark.parametrize(
    "name,process", [("gg_gg", "gg"), ("uux_ddxg", "uux_ddxg"), ("tt", "tt")]
)
def test_public_evaluation_equals_legacy_o1(full_colour, name, process):
    """u u~ > t t~ has open top lines through the loop, which the tree
    reading evaluates with the flipped mass natively too."""

    _assert_inclusive_o1(full_colour, name, process)


@pytest.mark.loop_physics
@pytest.mark.parametrize("name", ["uudd", "uux_uux"])
def test_public_quark_controls_equal_legacy_o1(quark_controls, name):
    _assert_inclusive_o1(quark_controls, name, name)


@pytest.mark.loop_nightly
def test_public_five_leg_controls_equal_legacy_o1(
    loop_sm_compiled_model, loop_native, tmp_path
):
    """u u~ > u u~ g and u u~ > g g g at full colour: binary64 with one
    rotation and the reduction check (virtuals held to max(1e-12, E), E <=
    1e-9; Borns to 1e-12; the first u u~ > g g g point is ill-conditioned in
    binary64) and double-double (1e-24)."""

    from pyamplicol import Runtime

    entries = (("uux_uuxg", "u u~ > u u~ g"), ("uux_ggg", "u u~ > g g g"))
    output = _generate(
        _card(loop_sm_compiled_model, tmp_path / "artifact", entries),
        tmp_path,
        entries,
    )
    for name, _expression in entries:
        fixture = _fixture(name)
        results = fixture["results"]["arbitrary-200"]
        runtime = Runtime.load(
            output, process=name, model_parameters=_parameters(fixture)
        )
        points, scales = fixture["points"], fixture["scales_squared"]
        double = runtime.evaluate(
            points, mu_r2=scales, stability_checks=BINARY64_CHECKS
        )
        _assert_legacy(
            double,
            results,
            DOUBLE_TOLERANCE,
            lambda index, value=double: _tolerance(value, index, "double"),
        )
        double_double = runtime.evaluate(
            points, mu_r2=scales, arithmetic="double-double"
        )
        _assert_legacy(double_double, results, DOUBLE_DOUBLE_TOLERANCE)


@pytest.mark.loop_physics
def test_finite_only_and_stability_estimates(full_colour):
    from pyamplicol import Runtime

    fixture = _fixture("uux_ddxg")
    results = fixture["results"]["arbitrary-200"]
    runtime = Runtime.load(
        full_colour, process="uux_ddxg", model_parameters=_parameters(fixture)
    )
    points, scales = fixture["points"], fixture["scales_squared"]
    full = runtime.evaluate(points, mu_r2=scales)
    finite = runtime.evaluate(points, mu_r2=scales, poles=False)
    assert finite.poles is False and finite.stability is None
    for a, b in zip(finite.values, full.values, strict=True):
        assert a[0] == LaurentCoefficients(b[0].finite, None, None)
    checked = runtime.evaluate(
        points,
        mu_r2=scales,
        stability_checks=VirtualStabilityChecks(2, reduction_direction=True),
    )
    assert checked.values == full.values
    for index, estimate in enumerate(checked.stability):
        assert dict(estimate.checks).keys() == {
            "rotation-0",
            "rotation-1",
            "reduction-direction",
        }
        reference = [Decimal(x[0]) for x in results["virtual"][index][0]]
        scale = max(map(abs, reference))
        error = max(
            abs(Decimal(a) - b)
            for a, b in zip(full.values[index][0], reference, strict=True)
        )
        assert float(error / scale) <= estimate.relative, (index, estimate)
        assert estimate.relative < 1e-9
    reduction_only = runtime.evaluate(
        points[:1],
        mu_r2=scales[:1],
        poles=False,
        stability_checks=VirtualStabilityChecks(reduction_direction=True),
    )
    (estimate,) = reduction_only.stability
    assert estimate.single_pole is None and dict(estimate.checks).keys() == {
        "reduction-direction"
    }


@pytest.mark.loop_physics
def test_v2_runtime_refuses_what_it_does_not_provide(full_colour):
    """Undeclared colour histories and spin classes, helicity selectors, direct
    R2 and a rescue policy are refused; the Born metric ("born") and resolved
    output need no declaration."""

    from pyamplicol import Runtime
    from pyamplicol.api.virtual_results import (
        VirtualArithmeticSpec,
        VirtualStabilityPolicy,
    )

    runtime = Runtime.load(full_colour, process="gg", model_parameters=GG_PARAMETERS)
    point = POINT
    assert [item.id for item in runtime.available_color_correlations()] == ["born"]
    born = runtime.evaluate_correlated([point], mu_r2="1000000")
    inclusive = runtime.evaluate([point], mu_r2="1000000")
    assert born.correlated and born.shape == inclusive.shape
    for a, b in zip(born.values[0][0], inclusive.values[0][0], strict=True):
        assert abs(complex(a) - b) <= 1e-12 * abs(b)
    resolved = runtime.evaluate_resolved([point], mu_r2="1000000")
    assert resolved.shape == (1, 1, 16, 1, 3) and resolved.stability is None
    total = resolved.total()
    for a, b in zip(total.values[0][0], inclusive.values[0][0], strict=True):
        assert abs(a - b) <= 1e-12 * abs(b)
    with pytest.raises(EvaluationError, match="spin-correlation class was not"):
        runtime.set_spin_correlation_vectors({3: [1, 0, 0, 0]})
    for call in (
        lambda: runtime.evaluate_correlated(
            [point], mu_r2="1000000", color_correlation="T12"
        ),
        lambda: runtime.evaluate([point], mu_r2="1000000", helicities=["h0"]),
        lambda: runtime.evaluate([point], mu_r2="1000000", r2_strategy="direct"),
        lambda: runtime.evaluate(
            [point],
            mu_r2="1000000",
            stability=VirtualStabilityPolicy(
                (VirtualArithmeticSpec("arbitrary", 40),), "1e-8", "1e-30"
            ),
        ),
    ):
        with pytest.raises(CompatibilityError):
            call()


# Unrelated Symbolica symbols created before a load: with 130 or 141 of them
# expand() reordered kernel terms (a refused load) and the Born components'
# summation (other binary64 bits) before the terms were put in a canonical
# order (``reference_executor.ExactKernel``).
FILLER_SYMBOLS = (0, 141)


@pytest.mark.loop_physics
def test_fresh_interpreter_rebuilds_inputs_and_refuses_a_different_plan(
    full_colour, monkeypatch
):
    """A new interpreter regenerates the per-point inputs from the artifact
    alone, with the same plan and bitwise-identical binary64 Born and virtual
    values whatever Symbolica symbols it created before; a stored plan
    unequal to the regenerated one is refused."""

    from pyamplicol import Runtime

    here = Runtime.load(full_colour, process="gg", model_parameters=GG_PARAMETERS)
    value = here.evaluate([POINT], mu_r2="1000000")
    expected = [
        here.inspect()["plan_sha256"],
        [list(point) for point in value.born_values],
        [[list(laurent) for laurent in point] for point in value.values],
    ]
    script = "\n".join(
        (
            "import json, sys",
            "from symbolica import S",
            "fillers = [S(f'filler::f{k}') for k in range(int(sys.argv[2]))]",
            "from pyamplicol import Runtime",
            "r = Runtime.load(sys.argv[1], process='gg', model_parameters=PARAMETERS)",
            "v = r.evaluate([POINT], mu_r2='1000000')",
            "print(json.dumps([r.inspect()['plan_sha256'], v.born_values, v.values]))",
        )
    )
    script = script.replace("PARAMETERS", repr(GG_PARAMETERS))
    script = script.replace("POINT", repr(POINT))
    for fillers in FILLER_SYMBOLS:
        process = subprocess.run(
            [sys.executable, "-c", script, str(full_colour), str(fillers)],
            capture_output=True,
            text=True,
            check=True,
        )
        fresh = json.loads(process.stdout.strip().splitlines()[-1])
        assert fresh == expected, fillers

    from pyamplicol.api.errors import ArtifactError
    from pyamplicol.loops.v2 import artifact

    original = artifact.generate_process

    def drifted(*args, **kwargs):
        generated = original(*args, **kwargs)
        generated.plan_json = generated.plan_json.replace('"schema":4', '"schema": 4')
        return generated

    monkeypatch.setattr(artifact, "_slices", type(artifact._slices)())
    monkeypatch.setattr(artifact, "generate_process", drifted)
    with pytest.raises(ArtifactError, match="differs from the stored one"):
        Runtime.load(full_colour, process="gg")


def _rewrite_payloads(source: Path, destination: Path, edits) -> Path:
    """Copy an artifact and replace JSON payloads under a re-signed manifest."""

    import shutil

    from pyamplicol.artifacts import ArtifactBuilder, load_manifest

    shutil.copytree(source, destination)
    manifest = load_manifest(destination)
    records = {record.path: record for record in manifest.payloads}
    with ArtifactBuilder(destination, mode="append") as builder:
        for path, edit in edits.items():
            value = edit(json.loads((destination / path).read_text()))
            record = records[path]
            builder.discard_payloads(path)
            builder.add_json(
                path,
                value,
                role=record.role,
                process_id=record.process_id,
                compact=True,
            )
        builder.finalize(
            kind=manifest.kind,
            producer=manifest.producer,
            model=manifest.model,
            configuration=manifest.configuration,
            processes=manifest.processes,
            runtime=manifest.runtime,
            dependencies=manifest.dependencies,
            default_process_id=manifest.default_process_id,
            extensions=manifest.extensions,
        )
    return destination


@pytest.mark.loop_physics
def test_native_loader_admits_plan_v4_and_refuses_tampering(
    full_colour, loop_native, tmp_path
):
    from pyamplicol import Runtime
    from pyamplicol.api.errors import ArtifactError
    from pyamplicol.loops.v2.artifact import admit

    for process, families in (("gg", 24), ("uux_ddxg", 42)):
        admitted = admit(loop_native, full_colour, process)
        physics = json.loads(
            (full_colour / f"processes/{process}/physics.json").read_text()
        )
        execution = json.loads(
            (full_colour / f"processes/{process}/execution.json").read_text()
        )
        assert admitted["runtime_capability"] == "rusticol.loop-runtime.v4"
        assert admitted["plan_sha256"] == execution["plan_sha256"]
        assert admitted["families"] == families == physics["metadata"]["families"]
        assert admitted["buckets"] == physics["metadata"]["buckets"]
        assert admitted["metadata"] == physics["metadata"]
        assert admitted["request"] == execution["request"]
        assert admitted["legs"] == len(physics["state_legs"])
        assert admitted["rows"] == len(physics["helicities"])
        assert admitted["columns"] == len(physics["metadata"]["sectors"])
    info = Runtime.load(full_colour, process="gg").inspect()
    assert info["native_admission"]["runtime_capability"] == "rusticol.loop-runtime.v4"
    assert info["plan_bytes"] > 0

    # The legacy virtual runtime, which the standalone SDK wraps, names the
    # plan v4 loader instead of misreading the artifact.
    with pytest.raises(ValueError, match="load_loop_v2_artifact"):
        loop_native.NativeVirtualRuntime(str(full_colour), process="gg")

    def shuffle_buckets(physics):
        buckets = physics["metadata"]["buckets"]
        buckets[0][2] += 1
        buckets[-1][2] -= 1
        return physics

    def foreign_digest(execution):
        execution["plan_sha256"] = "0" * 64
        return execution

    def native_inputs(execution):
        execution["point_inputs"] = "native"
        return execution

    def unresolved(physics):
        physics["metadata"]["resolved"] = False
        return physics

    def foreign_spin_leg(physics):
        physics["metadata"]["spin_correlations"] = [[5]]
        return physics

    for name, edits, reason, loaded in (
        (
            "buckets",
            {"processes/gg/physics.json": shuffle_buckets},
            "buckets",
            "buckets",
        ),
        (
            "digest",
            {"processes/gg/execution.json": foreign_digest},
            "differs from its declared digest",
            "differs from its declared digest",
        ),
        (
            "inputs",
            {"processes/gg/execution.json": native_inputs},
            "unsupported plan v4 execution declaration",
            "not a v2 virtual process",
        ),
        (
            "resolved",
            {"processes/gg/physics.json": unresolved},
            "without the forced seed FFT",
            "without the forced seed FFT",
        ),
        (
            "spin",
            {"processes/gg/physics.json": foreign_spin_leg},
            "disagree with the declarations",
            "disagree with the declarations",
        ),
    ):
        tampered = _rewrite_payloads(full_colour, tmp_path / name, edits)
        with pytest.raises(ArtifactError, match=reason):
            admit(loop_native, tampered, "gg")
        with pytest.raises(ArtifactError, match=loaded):
            Runtime.load(tampered, process="gg")


@pytest.mark.loop_physics
def test_inspect_summarizes_v2_loop_artifacts(full_colour):
    from pyamplicol.artifacts import inspect_artifact
    from pyamplicol.cli.main import main

    inspection = inspect_artifact(full_colour)
    by_id = {process.id: process for process in inspection.processes}
    gg = by_id["gg"]
    assert gg.execution_mode == "loop_recurrence"
    assert gg.physical_helicities == 16 and gg.physical_color_components == 1
    loop = gg.loop
    assert loop is not None and loop.engine == "v2"
    assert loop.runtime_capability == "rusticol.loop-runtime.v4"
    assert loop.born_orders == ("QCD^4",)
    assert loop.sectors == (("QCD^6", (("QCD^2", "QCD^4"),)),)
    assert loop.families == 24 and sum(b[2] for b in loop.buckets) == 24
    assert loop.color_degrees is None and loop.schedule == "backward"
    assert loop.poles == "runtime-optional"
    assert loop.stability_checks == ("rotation", "reduction-direction")
    assert loop.arithmetic == ("double", "double-double", "arbitrary")
    assert loop.resolved and not loop.correlations
    assert loop.color_correlations == () and loop.spin_correlations == ()
    assert loop.point_inputs == "regenerated-at-load"
    execution = json.loads((full_colour / "processes/gg/execution.json").read_text())
    assert loop.plan_sha256 == execution["plan_sha256"]
    assert by_id["uux_ddxg"].loop.families == 42

    stdout = io.StringIO()
    from contextlib import redirect_stdout

    with redirect_stdout(stdout):
        code = main(["inspect", str(full_colour), "--color", "never"])
    assert code == 0
    text = stdout.getvalue()
    assert "families / buckets" in text and "24 families in 3 (N, R) buckets" in text
    assert "rusticol.loop-runtime.v4" in text and "plan v4" in text


@pytest.mark.loop_physics
def test_inspect_summarizes_legacy_loop_artifacts(saved_loop_fixture):
    from pyamplicol.artifacts import inspect_artifact

    inspection = inspect_artifact(saved_loop_fixture("sdk-mixed-source"))
    (process,) = inspection.processes
    loop = process.loop
    assert loop is not None and loop.engine == "legacy"
    assert loop.runtime_capability == "rusticol.loop-runtime.v3"
    assert loop.families is None and loop.plan_sha256 is None
    assert len(loop.sectors) == 3 and len(loop.born_orders) == 3
    assert process.execution_mode == "loop_recurrence"


@pytest.mark.loop_physics
def test_cli_generates_and_evaluates_through_v2(
    loop_sm_compiled_model, loop_native, tmp_path
):
    from pyamplicol.cli import run_cli

    entries = (("gg", "g g > g g"),)
    card = tmp_path / "card.toml"
    output = tmp_path / "artifact"
    card.write_text(_card(loop_sm_compiled_model, output, entries))
    stdout, stderr = io.StringIO(), io.StringIO()
    code = run_cli(
        ("generate", "--card", str(card), "--dry-run", "--json", "--progress", "off"),
        stdout=stdout,
        stderr=stderr,
    )
    assert code == 0, stderr.getvalue()
    plan = json.loads(stdout.getvalue())
    assert plan["unsupported_features"] == []
    coverage = plan["estimated_coverage"]
    assert coverage["engine"] == "v2"
    assert coverage["runtime_capability"] == "rusticol.loop-runtime.v4"
    assert coverage["processes"][0]["loop_orders"] == {"QCD": 4, "QED": 0}
    assert not output.exists()
    code = run_cli(
        ("generate", "--card", str(card), "--progress", "off"),
        stdout=io.StringIO(),
        stderr=stderr,
    )
    assert code == 0, stderr.getvalue()
    point = POINT
    momenta = tmp_path / "momenta.json"
    momenta.write_text(json.dumps([point, point]))
    stdout = io.StringIO()
    code = run_cli(
        (
            "evaluate",
            str(output),
            "--process",
            "gg",
            "--momenta",
            str(momenta),
            "--mu-r2",
            "1000000",
            "--json",
            "--progress",
            "off",
            "--no-poles",
            "--stability-rotation",
            "z:3/5:4/5",
            "--stability-reduction-direction",
        ),
        stdout=stdout,
        stderr=stderr,
    )
    assert code == 0, stderr.getvalue()
    payload = json.loads(stdout.getvalue())
    assert payload["poles"] is False and payload["shape"] == [2, 1, 3]
    assert payload["values"][0][0][1:] == [None, None]
    assert [name for name, _ in payload["stability"][0]["checks"]] == [
        "rotation-0",
        "reduction-direction",
    ]
    stdout = io.StringIO()
    code = run_cli(
        (
            "evaluate",
            str(output),
            "--process",
            "gg",
            "--momenta",
            str(momenta),
            "--mu-r2",
            "1000000",
            "--json",
            "--progress",
            "off",
        ),
        stdout=stdout,
        stderr=stderr,
    )
    assert code == 0, stderr.getvalue()
    payload = json.loads(stdout.getvalue())
    assert payload["poles"] is True and payload["stability"] is None
    stdout = io.StringIO()
    code = run_cli(
        (
            "evaluate",
            str(output),
            "--process",
            "gg",
            "--momenta",
            str(momenta),
            "--mu-r2",
            "1000000",
            "--resolved",
            "--no-poles",
            "--stability-reduction-direction",
            "--color",
            "never",
            "--progress",
            "off",
        ),
        stdout=stdout,
        stderr=stderr,
    )
    assert code == 0, stderr.getvalue()
    text = stdout.getvalue()
    assert "Finite-only evaluation" in text and "Stability estimates" in text
    assert "reduction-direction" in text and "absent" in text


@pytest.mark.loop_physics
@pytest.mark.parametrize(
    "name,expression,accuracy",
    [
        ("gg_gg", "g g > g g", "lc"),
        ("tt", "u u~ > t t~", "lc"),
        ("uux_ddxg", "u u~ > d d~ g", "nlc"),
    ],
)
def test_projected_colour_through_v2_equals_legacy_o1(
    loop_sm_colour_model, loop_native, tmp_path, name, expression, accuracy
):
    """LC and NLC projections in binary64 (1e-12) and double-double (1e-24)."""

    from pyamplicol import Runtime

    entries = ((name, expression),)
    output = _generate(
        _card(
            loop_sm_colour_model,
            tmp_path / "artifact",
            entries,
            accuracy=accuracy,
            declared=False,
        ),
        tmp_path,
        entries,
    )
    fixture = _fixture(name)
    reference = fixture["colour_projections"][accuracy]
    results = reference["results"]["arbitrary-200"]
    runtime = Runtime.load(output, process=name, model_parameters=_parameters(fixture))
    info = runtime.inspect()
    assert info["color_accuracy"] == accuracy
    assert list(info["color_degrees"].values()) == [reference["virtual_degree"]]
    points, scales = fixture["points"], fixture["scales_squared"]
    result = runtime.evaluate(points, mu_r2=scales)
    assert result.color_accuracy == accuracy
    _assert_legacy(result, results, DOUBLE_TOLERANCE)
    double_double = runtime.evaluate(points, mu_r2=scales, arithmetic="double-double")
    _assert_legacy(double_double, results, DOUBLE_DOUBLE_TOLERANCE)


# ------------------------------------------------------------------ observables


def _observables_record(name: str) -> dict:
    return json.loads((OBSERVABLES / f"{name}.json").read_text())


def _parts(value) -> tuple[Decimal, Decimal]:
    """(real, imaginary) of a public value or a legacy ``[re, im]`` pair."""

    if isinstance(value, list | tuple) and len(value) == 2:
        return Decimal(value[0]), Decimal(value[1])
    if isinstance(value, complex):
        return Decimal(repr(value.real)), Decimal(repr(value.imag))
    if isinstance(value, Decimal | float | int):
        return Decimal(repr(value) if isinstance(value, float) else value), Decimal(0)
    return Decimal(value.real), Decimal(value.imag)  # CorrelatedValue


def _deviation(actual, expected) -> Decimal:
    from decimal import localcontext

    with localcontext(prec=80):
        (a, b), (c, d) = _parts(actual), _parts(expected)
        return ((a - c) ** 2 + (b - d) ** 2).sqrt()


def _magnitude(value) -> Decimal:
    return _deviation(value, (0, 0))


CARRIER_TOLERANCES = {
    "double": DOUBLE_TOLERANCE,
    "double-double": DOUBLE_DOUBLE_TOLERANCE,
    "arbitrary": ARBITRARY_TOLERANCE,
}


def _estimated(estimate: float) -> Decimal:
    """A binary64 virtual's tolerance under the user's rule: 1e-12, or its
    own estimate E when E exceeds it, which must not exceed 1e-9."""

    assert estimate <= ESTIMATE_CAP, estimate
    return max(DOUBLE_TOLERANCE, Decimal(repr(estimate)))


def _tolerance(result, index: int, carrier: str) -> Decimal:
    """The O1 tolerance of one evaluated point's virtual (module docstring)."""

    if carrier != "double":
        return CARRIER_TOLERANCES[carrier]
    assert result.stability is not None
    return _estimated(result.stability[index].relative)


def _assert_record(result, expected, index: int, carrier: str, label) -> None:
    """Born per order (held to the carrier's tolerance: the estimate is the
    virtual's) and virtual per sector against one legacy record."""

    borns = expected["born"][index]
    scale = max(_magnitude(x) for x in borns)
    tolerance = CARRIER_TOLERANCES[carrier]
    for actual, value in zip(result.born_values[index], borns, strict=True):
        assert _deviation(actual, value) <= tolerance * scale, (label, index)
    sectors = expected["virtual"][index]
    tolerance = _tolerance(result, index, carrier)
    for laurent, values in zip(result.values[index], sectors, strict=True):
        scale = max(_magnitude(x) for x in values)
        for actual, value in zip(laurent, values, strict=True):
            assert _deviation(actual, value) <= tolerance * scale, (label, index)


def _correlators(record: dict):
    from pyamplicol.correlators import CorrelatorConfig

    return CorrelatorConfig.from_json_dict(
        {
            "color_correlations": record["histories"],
            "spin_correlations": record["spin_classes"],
        }
    )


@pytest.fixture(scope="module")
def observables_artifact(
    loop_sm_compiled_model, loop_sm_colour_model, loop_native, tmp_path_factory
):
    """The public v2 artifact of one legacy observables record (generated with
    its colour histories and spin classes; members by exact flavour
    grouping), built once per module."""

    built: dict[str, Path] = {}

    def build(name: str) -> Path:
        if name not in built:
            record = _observables_record(name)
            accuracy = record.get("accuracy", "full")
            entries = tuple(
                (key, expression) for key, expression in record["processes"]
            )
            tmp = tmp_path_factory.mktemp(name)
            card = _card(
                loop_sm_colour_model if accuracy != "full" else loop_sm_compiled_model,
                tmp / "artifact",
                entries,
                accuracy=accuracy,
                declared=accuracy == "full",
                process="flavor_blind = true" if len(entries) > 1 else "",
            )
            built[name] = _generate(card, tmp, entries, _correlators(record))
        return built[name]

    return build


def _spin_vectors(record: dict, legs: str) -> dict[int, tuple]:
    """The literal (real, imaginary) source vectors of one legacy spin class."""

    return {
        int(leg): tuple(
            (Decimal(re), Decimal(im)) for re, im in record["spin_vectors"][k]
        )
        for k, leg in enumerate(legs.split(","))
    }


def _assert_resolved(result, runtime, resolved: dict, carrier: str) -> None:
    """``_compare_resolved`` with sums at 80 digits (Decimal's default
    context would round 60-digit values to 28)."""

    from decimal import localcontext

    with localcontext(prec=80):
        _compare_resolved(result, runtime, resolved, carrier)


def _compare_resolved(result, runtime, resolved: dict, carrier: str) -> None:
    """Per helicity row, the Born and the virtual summed over the Born colour
    components against the legacy resolved record (whose colour anchors are
    diagram tensors: the colour axis is compared through its sum). A row is
    held relative to its own scale, a structural zero (a vanishing Born row,
    whose entries are rounding noise with large relative estimates) relative
    to the point's largest at the carrier's tolerance; in binary64 the Born
    rows at 1e-12 and a row's virtual at its entries' largest estimate under
    the user's rule (``_estimated``)."""

    rows = [tuple(h.values) for h in runtime.physics.helicities]
    legacy = [tuple(row) for row in resolved["helicities"]]
    floor = ARBITRARY_TOLERANCE
    for index in range(len(result.values)):
        born_scale = max(
            _magnitude(x) for order in resolved["born"][index] for x in order
        )
        virtual_scale = max(
            _magnitude(x)
            for sector in resolved["virtual"][index]
            for row in sector
            for x in row
        )
        for h, row in enumerate(rows):
            target = legacy.index(row)
            tolerance = CARRIER_TOLERANCES[carrier]
            born = [order[target] for order in resolved["born"][index]]
            scale = max(map(_magnitude, born))
            structural_zero = scale <= floor * born_scale
            scale = born_scale if structural_zero else scale
            for order, value in enumerate(born):
                mine = sum(
                    (_parts(x)[0] for x in result.born_values[index][order][h]),
                    Decimal(0),
                )
                assert _deviation(mine, value) <= tolerance * scale, (index, row)
            if carrier == "double" and not structural_zero:
                tolerance = _estimated(
                    max(e.relative for e in result.stability[index][h])
                )
            for sector, values in enumerate(resolved["virtual"][index]):
                targets = values[target]
                scale = max(map(_magnitude, targets))
                scale = virtual_scale if scale <= floor * virtual_scale else scale
                for k, value in enumerate(targets):
                    mine = sum(
                        (
                            _parts(laurent[k])[0]
                            for laurent in result.values[index][sector][h]
                        ),
                        Decimal(0),
                    )
                    assert _deviation(mine, value) <= tolerance * scale, (
                        index,
                        row,
                        k,
                    )
        # Every legacy row v2 does not generate is a structural zero.
        for row in set(legacy) - set(rows):
            target = legacy.index(row)
            assert all(
                _magnitude(order[target]) <= floor * born_scale
                for order in resolved["born"][index]
            )


BINARY64_AND_DD = ("double", "double-double")
ALL_CARRIERS = (*BINARY64_AND_DD, "arbitrary")


def _case_id(value) -> str:
    return value if isinstance(value, str) else "+".join(value)


@pytest.mark.loop_physics
@pytest.mark.parametrize(
    "name,carriers",
    (
        ("gg_gg", ALL_CARRIERS),
        ("ddgg_ssgg", BINARY64_AND_DD),
        ("gg_gg_lc", BINARY64_AND_DD),
    ),
    ids=_case_id,
)
def test_public_correlated_and_resolved_calls_equal_legacy_o1(
    observables_artifact, name, carriers
):
    """O1 through the public surface: every legacy directed colour history,
    every joint complex spin source and (where recorded) the helicity-resolved
    outputs, for the representative and its exact flavour member (served by
    the representative: one artifact process), in binary64 with one rotation
    and the reduction-direction check, in double-double and (g g > g g here,
    the other full-colour records nightly) in 60-digit arbitrary precision."""

    _assert_public_o1(observables_artifact, name, carriers)


@pytest.mark.loop_nightly
@pytest.mark.parametrize(
    "name,carriers",
    (
        ("ddgg_ssgg", ("arbitrary",)),
        ("gg_gg_nlc", BINARY64_AND_DD),
        ("ddgg_ssgg_lc", BINARY64_AND_DD),
        ("ddgg_ssgg_nlc", BINARY64_AND_DD),
        ("uux_ggg", ALL_CARRIERS),
        ("uux_ggg_spin", ALL_CARRIERS),
        ("uux_ggg_lc", BINARY64_AND_DD),
        ("uux_ggg_nlc", BINARY64_AND_DD),
    ),
    ids=_case_id,
)
def test_public_correlated_and_resolved_calls_equal_legacy_o1_nightly(
    observables_artifact, name, carriers
):
    """The same O1 on every other frozen legacy observables record: NLC, LC
    and NLC members, and the five-leg u u~ > g g g (directed and spin) at
    full, leading and next-to-leading colour, whose first point is
    binary64-ill-conditioned (virtuals held to their own estimate there, and
    to 1e-24 in double-double); the full-colour records also in 60 digits."""

    _assert_public_o1(observables_artifact, name, carriers)


CARRIER_OPTIONS = {
    "double": {"stability_checks": BINARY64_CHECKS},
    "double-double": {"arithmetic": "double-double"},
    "arbitrary": {"arithmetic": "arbitrary", "precision": ARBITRARY_DIGITS},
}


def _assert_public_o1(observables_artifact, name: str, carriers) -> None:
    from pyamplicol import Runtime
    from pyamplicol.artifacts import load_manifest
    from pyamplicol.correlators import CorrelatedRequest

    record = _observables_record(name)
    output = observables_artifact(name)
    manifest = load_manifest(output)
    representative = record["processes"][0][0]
    assert [process["id"] for process in manifest.processes] == [representative]
    points, scales = record["points"], record["scales_squared"]
    for process, _expression in record["processes"]:
        expected = record["results"]["arbitrary-200"][process]
        runtime = Runtime.load(
            output, process=process, model_parameters=record["parameters"]
        )
        assert runtime.physics.process_id == process
        assert runtime.representative_process_key == representative
        histories = tuple(expected.get("directed", {}))
        assert [item.id for item in runtime.available_color_correlations()] == [
            "born",
            *(history["id"] for history in record["histories"]),
        ]
        requests = {
            history: CorrelatedRequest(history, {}) for history in histories
        } | {
            f"spin:{legs}": CorrelatedRequest("born", _spin_vectors(record, legs))
            for legs in expected.get("spin", {})
        }
        for carrier in carriers:
            options = CARRIER_OPTIONS[carrier]
            values = runtime.evaluate_correlated_many(
                points, requests, mu_r2=scales, **options
            )
            for label, result in values.items():
                assert result.correlated and result.poles
                assert (result.stability is not None) == (carrier == "double")
                target = (
                    expected["spin"][label.removeprefix("spin:")]
                    if label.startswith("spin:")
                    else expected["directed"][label]
                )
                for index in range(len(points)):
                    _assert_record(result, target, index, carrier, (process, label))
            if "resolved" in expected:
                resolved = runtime.evaluate_resolved(points, mu_r2=scales, **options)
                assert (resolved.stability is not None) == (carrier == "double")
                _assert_resolved(resolved, runtime, expected["resolved"], carrier)


@pytest.mark.loop_physics
def test_finite_only_correlated_and_resolved_from_the_same_artifact(
    observables_artifact,
):
    """``poles=False`` on correlated and resolved calls of the artifact that
    computed them with poles: equal finite parts, absent poles, estimates per
    point (correlated) and per resolved entry only when checks ran."""

    from pyamplicol import Runtime
    from pyamplicol.correlators import CorrelatedRequest

    record = _observables_record("gg_gg")
    runtime = Runtime.load(
        observables_artifact("gg_gg"), model_parameters=record["parameters"]
    )
    points, scales = record["points"], record["scales_squared"]
    requests = {"T12": CorrelatedRequest("T12", {})}
    full = runtime.evaluate_correlated_many(points, requests, mu_r2=scales)["T12"]
    finite = runtime.evaluate_correlated_many(
        points, requests, mu_r2=scales, poles=False
    )["T12"]
    assert finite.poles is False and finite.stability is None
    for a, b in zip(finite.values, full.values, strict=True):
        assert a[0] == LaurentCoefficients(b[0].finite, None, None)
    checked = runtime.evaluate_correlated(
        points,
        color_correlation="T12",
        mu_r2=scales,
        poles=False,
        stability_checks=VirtualStabilityChecks(reduction_direction=True),
    )
    assert [dict(e.checks).keys() for e in checked.stability] == [
        {"reduction-direction"}
    ] * len(points)
    assert all(e.single_pole is None for e in checked.stability)
    resolved = runtime.evaluate_resolved(points, mu_r2=scales)
    finite = runtime.evaluate_resolved(
        points, mu_r2=scales, poles=False, stability_checks=VirtualStabilityChecks(1)
    )
    assert finite.poles is False and resolved.stability is None
    with pytest.raises(ValueError, match="finite-only"):
        finite.single_pole  # noqa: B018
    for a, b in zip(finite.values, resolved.values, strict=True):
        for sector_a, sector_b in zip(a, b, strict=True):
            for row_a, row_b in zip(sector_a, sector_b, strict=True):
                for x, y in zip(row_a, row_b, strict=True):
                    assert x == LaurentCoefficients(y.finite, None, None)
    (point_estimates, *_rest) = finite.stability
    assert len(point_estimates) == len(finite.helicity_ids)
    assert all(
        dict(estimate.checks).keys() == {"rotation-0"}
        for row in point_estimates
        for estimate in row
    )
    total = finite.total()
    assert total.poles is False and total.stability is None

    from pyamplicol.artifacts import inspect_artifact

    (process,) = inspect_artifact(observables_artifact("gg_gg")).processes
    loop = process.loop
    assert loop.correlations and loop.resolved
    assert loop.color_correlations == ("casimir", "T12", "T34")
    assert loop.spin_correlations == ((3,), (3, 4))


# ------------------------------------------------------------------ portability


def _strings(path: Path) -> str:
    """The text of one artifact file; a PACBIN's plan member decompressed."""

    from pyamplicol.loops.v2.artifact import read_plan

    if path.name == "loop-runtime.pacbin":
        return read_plan(path) + path.read_bytes().decode("latin-1")
    return path.read_bytes().decode("latin-1")


@pytest.mark.loop_physics
def test_artifacts_hold_no_host_paths_and_load_relocated(
    full_colour, observables_artifact, tmp_path
):
    """Requirements addendum G within WP2.6: no absolute path of the
    generating host (working directories, the model source, the output, the
    home directory), no user or host name in any file of a plan v4 artifact
    (the plan member decompressed); a copy under another path loads without
    the original and gives bitwise-identical binary64 values."""

    import getpass
    import shutil
    import socket

    from pyamplicol import Runtime

    forbidden = {
        str(Path.home()),
        str(ROOT),
        str(Path.cwd()),
        str(full_colour.parent),
        f"/{getpass.getuser()}/",
        f"{getpass.getuser()}@",
        socket.gethostname(),
    }
    roots = (full_colour, observables_artifact("ddgg_ssgg"))
    for root in roots:
        forbidden.add(str(root.resolve().parent))
    for root in roots:
        for path in sorted(p for p in root.rglob("*") if p.is_file()):
            text = _strings(path)
            for item in forbidden:
                assert item not in text, (path.relative_to(root), item)
            assert "/shared/" not in text and "/home/" not in text, path
    moved = tmp_path / "elsewhere" / "copy"
    shutil.copytree(full_colour, moved)
    points = [
        POINT,
        [[500, 0, 0, 500], [500, 0, 0, -500], [500, 0, 300, 400], [500, 0, -300, -400]],
    ]
    script = "\n".join(
        (
            "import json, sys",
            "from pyamplicol import Runtime",
            "r = Runtime.load(sys.argv[1], process='gg', model_parameters=PARAMETERS)",
            "v = r.evaluate(POINTS, mu_r2='1000000')",
            "print(json.dumps([r.inspect()['plan_sha256'], v.born_values, v.values]))",
        )
    )
    script = script.replace("PARAMETERS", repr(GG_PARAMETERS))
    script = script.replace("POINTS", repr(points))
    # A fresh interpreter in another working directory regenerates the point
    # inputs from the copy alone.
    process = subprocess.run(
        [sys.executable, "-c", script, str(moved)],
        capture_output=True,
        text=True,
        check=True,
        cwd=tmp_path,
    )
    digest, born, values = json.loads(process.stdout.strip().splitlines()[-1])
    here = Runtime.load(full_colour, process="gg", model_parameters=GG_PARAMETERS)
    original = here.evaluate(points, mu_r2="1000000")
    assert digest == here.inspect()["plan_sha256"]
    assert born == [list(point) for point in original.born_values]
    assert values == [[list(laurent) for laurent in point] for point in original.values]


@pytest.mark.loop_physics
def test_unsupported_requests_name_the_reason_and_keep_legacy(
    loop_sm_compiled_model, loop_native, tmp_path
):
    """A correction the source model does not expand perturbatively (QED in
    loop_sm, which the legacy graph generator refuses too; several declared
    corrections take one loop order each) and the forced seed FFT are
    refused with their reasons before any generation; the legacy engine
    stays selectable."""

    from pyamplicol.api import Generator, ProcessRequest, ProcessSet
    from pyamplicol.config import resolve_config

    entries = (("gg", "g g > g g"),)
    card = tmp_path / "card.toml"
    processes = ProcessSet((ProcessRequest.parse("g g > g g", name="gg"),))
    for edit, reason in (
        (
            ('loop_coupling_orders = ["QCD"]', 'loop_coupling_orders = ["QCD", "QED"]'),
            r"require declared NLO model support.*\['QED'\]",
        ),
        (
            ('contraction = "direct"', 'contraction = "symmetric-group-fft"'),
            r"contracts seed columns directly.*loop\.engine='legacy'",
        ),
    ):
        card.write_text(
            _card(loop_sm_compiled_model, tmp_path / "a", entries).replace(*edit)
        )
        generator = Generator(resolve_config(card))
        with pytest.raises(GenerationError, match=reason):
            generator.generate(processes, tmp_path / "a")
        assert not (tmp_path / "a").exists()
        assert generator.plan(processes).unsupported_features
    card.write_text(
        _card(loop_sm_compiled_model, tmp_path / "b", entries).replace(
            'engine = "v2"', 'engine = "legacy"'
        )
    )
    plan = Generator(resolve_config(card)).plan(processes)
    assert plan.estimated_coverage.get("engine") is None  # the legacy plan
    assert plan.estimated_coverage["processes"][0]["loop_graph_count"] > 0


# ------------------------------------------------------------------ size


@pytest.mark.loop_nightly
def test_six_gluon_plan_v4_artifact_is_below_twenty_megabytes(
    loop_sm_compiled_model, loop_native, tmp_path
):
    """gg -> gggg (1100 families, 45 MB of plan JSON without the forward
    oracle) publishes < 20 MB (7.4 MB measured, 5.3 MB of it the compiled
    model)."""

    from pyamplicol.loops.v2.artifact import admit, plan_digest, read_plan

    entries = (("gg4g", "g g > g g g g"),)
    card = _card(loop_sm_compiled_model, tmp_path / "artifact", entries).replace(
        "post_build_validation = true", "post_build_validation = false"
    )
    output = _generate(card, tmp_path, entries)
    size = sum(path.stat().st_size for path in output.rglob("*") if path.is_file())
    print(f"gg > gggg plan v4 artifact: {size} bytes")
    assert size < 20 * 10**6, size
    execution = json.loads((output / "processes/gg4g/execution.json").read_text())
    plan = read_plan(output / "processes/gg4g/loop-runtime.pacbin")
    assert plan_digest(plan) == execution["plan_sha256"]
    assert len(json.loads(plan)["families"]) == 1100
    admitted = admit(loop_native, output, "gg4g")
    assert admitted["families"] == 1100
    assert admitted["plan_sha256"] == execution["plan_sha256"]
