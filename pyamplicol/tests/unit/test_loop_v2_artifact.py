# SPDX-License-Identifier: 0BSD
"""Plan v4 artifacts and the public API through the v2 engine (WP2.6).

``loop.engine = "v2"`` generates virtual processes through the shared-recursion
engine and publishes them as plan v4 processes (``rusticol.loop-runtime.v4``).
Checks, on the public surface only (TOML card, ``Generator``, ``Runtime``, CLI):

* gg -> gg and u u~ -> d d~ g at full colour and gg -> gg at leading colour
  equal the frozen 200-bit legacy values (oracle O1) in binary64 and in
  arbitrary precision, per squared coupling order and Laurent coefficient;
* ``poles=False`` returns the finite parts with absent poles;
* stability checks (rotations by count or exact specification, the
  reduction-direction check) return one estimate per point that bounds the
  true error, and no estimate without checks;
* a fresh interpreter rebuilds the per-point inputs from the artifact and
  refuses a stored plan that differs from the regenerated one;
* the native artifact loader (``load_loop_v2_artifact``) admits each plan v4
  process, cross-checks families, buckets, legs, helicity rows and sectors
  against the declarations, and refuses tampered declarations and digests;
  the legacy native virtual runtime (the SDK's loader) refuses plan v4 with
  its reason;
* ``pyamplicol inspect`` summarizes v2 and legacy loop artifacts (engine,
  sectors, families, buckets, colour degrees, plan digest);
* requests v2 does not cover yet fail with the reason and leave
  ``loop.engine = "legacy"`` (the default) selectable;
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
from pyamplicol.api.errors import CompatibilityError, GenerationError
from pyamplicol.api.virtual_results import parse_rotation
from pyamplicol.loops.v2.stability import Rotation

ROOT = Path(__file__).resolve().parents[2]
LEGACY = ROOT / "tests" / "fixtures" / "loop" / "legacy"
DOUBLE_TOLERANCE = Decimal("1e-12")
ARBITRARY_DIGITS = 60
ARBITRARY_TOLERANCE = Decimal("1e-40")
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


def _card(model, output, entries, *, accuracy="full", declared=True, extra=""):
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


def _generate(card_text, tmp: Path, entries):
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
    Generator(resolve_config(card)).generate(processes, output)
    return output


def _fixture(name: str) -> dict:
    return json.loads((LEGACY / f"{name}.json").read_text())


def _parameters(fixture: dict) -> dict[str, str]:
    return {k: v for k, v in fixture["parameters"].items() if k != "MU_R"}


def _assert_legacy(result, results, tolerance):
    """Born and the complete virtual of every point against legacy O1."""

    for index in range(len(result.values)):
        born = Decimal(results["born"][index][0][0])
        ((mine,),) = (result.born_values[index],)
        assert abs(Decimal(mine) - born) <= tolerance * abs(born), (index, mine)
        reference = [Decimal(x[0]) for x in results["virtual"][index][0]]
        ((laurent,),) = (result.values[index],)
        scale = max(map(abs, reference))
        for k, (a, b) in enumerate(zip(laurent, reference, strict=True)):
            assert abs(Decimal(a) - b) <= tolerance * scale, (index, k, a, b)


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
        {"QCD": 4, "QED": 0},
        fixed_parameters={"MB": "0"},
        accuracy="lc",
        colour_declaration=declaration,
        colour_contraction="direct",
    )
    record = request_record(request)
    assert json.loads(json.dumps(record)) == record
    assert request_from_record(record) == request


# ------------------------------------------------------------------ public path


@pytest.fixture(scope="module")
def full_colour(loop_sm_compiled_model, loop_native, tmp_path_factory):
    tmp = tmp_path_factory.mktemp("v2-artifact")
    entries = (("gg", "g g > g g"), ("uux_ddxg", "u u~ > d d~ g"))
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
    assert {record["id"] for record in manifest.processes} == {"gg", "uux_ddxg"}
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
@pytest.mark.parametrize("name,process", [("gg_gg", "gg"), ("uux_ddxg", "uux_ddxg")])
def test_public_evaluation_equals_legacy_o1(full_colour, name, process):
    from pyamplicol import Runtime

    fixture = _fixture(name)
    results = fixture["results"]["arbitrary-200"]
    runtime = Runtime.load(
        full_colour, process=process, model_parameters=_parameters(fixture)
    )
    points, scales = fixture["points"], fixture["scales_squared"]
    double = runtime.evaluate(points, mu_r2=scales)
    assert double.shape == (len(points), 1, 3) and double.stability is None
    assert all(item.precision_bits == 53 for item in double.diagnostics)
    _assert_legacy(double, results, DOUBLE_TOLERANCE)
    arbitrary = runtime.evaluate(
        points, mu_r2=scales, arithmetic="arbitrary", precision=ARBITRARY_DIGITS
    )
    assert isinstance(arbitrary.values[0][0].finite, Decimal)
    _assert_legacy(arbitrary, results, ARBITRARY_TOLERANCE)
    # One squared coupling order per partial result; the sector names its pairs.
    (sector,) = double.sectors
    assert double.for_order(sector.order).values == double.values


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
    from pyamplicol import Runtime
    from pyamplicol.api.virtual_results import (
        VirtualArithmeticSpec,
        VirtualStabilityPolicy,
    )

    runtime = Runtime.load(full_colour, process="gg")
    point = POINT
    for call in (
        lambda: runtime.evaluate_correlated([point], mu_r2="1000000"),
        lambda: runtime.evaluate_resolved([point], mu_r2="1000000"),
        lambda: runtime.evaluate([point], mu_r2="1000000", arithmetic="double-double"),
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


@pytest.mark.loop_physics
def test_fresh_interpreter_rebuilds_inputs_and_refuses_a_different_plan(
    full_colour, monkeypatch
):
    """A new interpreter regenerates the per-point inputs from the artifact
    alone; a stored plan unequal to the regenerated one is refused."""

    script = "\n".join(
        (
            "import json, sys",
            "from pyamplicol import Runtime",
            "r = Runtime.load(sys.argv[1], process='gg', model_parameters=PARAMETERS)",
            "v = r.evaluate([POINT], mu_r2='1000000', poles=False)",
            "print(json.dumps([r.inspect()['plan_sha256'], v.values[0][0].finite]))",
        )
    )
    script = script.replace("PARAMETERS", repr(GG_PARAMETERS))
    script = script.replace("POINT", repr(POINT))
    process = subprocess.run(
        [sys.executable, "-c", script, str(full_colour)],
        capture_output=True,
        text=True,
        check=True,
    )
    digest, finite = json.loads(process.stdout.strip().splitlines()[-1])
    from pyamplicol import Runtime

    here = Runtime.load(full_colour, process="gg", model_parameters=GG_PARAMETERS)
    assert digest == here.inspect()["plan_sha256"]
    finite_here = here.evaluate([POINT], mu_r2="1000000", poles=False)
    assert finite == finite_here.values[0][0].finite

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

    for name, edits, reason in (
        (
            "buckets",
            {"processes/gg/physics.json": shuffle_buckets},
            "buckets",
        ),
        (
            "digest",
            {"processes/gg/execution.json": foreign_digest},
            "differs from its declared digest",
        ),
    ):
        tampered = _rewrite_payloads(full_colour, tmp_path / name, edits)
        with pytest.raises(ArtifactError, match=reason):
            admit(loop_native, tampered, "gg")
        with pytest.raises(ArtifactError, match=reason):
            Runtime.load(tampered, process="gg")


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


@pytest.mark.loop_physics
def test_leading_colour_through_v2_equals_legacy_o1(
    loop_sm_colour_model, loop_native, tmp_path
):
    from pyamplicol import Runtime

    entries = (("gg", "g g > g g"),)
    output = _generate(
        _card(
            loop_sm_colour_model,
            tmp_path / "artifact",
            entries,
            accuracy="lc",
            declared=False,
        ),
        tmp_path,
        entries,
    )
    fixture = _fixture("gg_gg")
    reference = fixture["colour_projections"]["lc"]
    runtime = Runtime.load(output, process="gg", model_parameters=_parameters(fixture))
    info = runtime.inspect()
    assert info["color_accuracy"] == "lc"
    assert list(info["color_degrees"].values()) == [reference["virtual_degree"]]
    result = runtime.evaluate(fixture["points"], mu_r2=fixture["scales_squared"])
    assert result.color_accuracy == "lc"
    _assert_legacy(result, reference["results"]["arbitrary-200"], DOUBLE_TOLERANCE)


@pytest.mark.loop_physics
def test_unsupported_requests_name_the_reason_and_keep_legacy(
    loop_sm_compiled_model, loop_native, tmp_path
):
    from pyamplicol.api import Generator, ProcessRequest, ProcessSet
    from pyamplicol.config import resolve_config
    from pyamplicol.correlators import CorrelatorConfig

    entries = (("gg", "g g > g g"),)
    card = tmp_path / "card.toml"
    card.write_text(_card(loop_sm_compiled_model, tmp_path / "a", entries))
    generator = Generator(resolve_config(card))
    processes = ProcessSet((ProcessRequest.parse("g g > g g", name="gg"),))
    with pytest.raises(GenerationError, match=r"WP2\.5.*loop\.engine='legacy'"):
        generator.generate(processes, tmp_path / "a", correlators=CorrelatorConfig())
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
    """gg -> gggg (1100 families, 98 MB of plan JSON) publishes < 20 MB."""

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
