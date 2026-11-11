# SPDX-License-Identifier: 0BSD
"""Partial weights per squared coupling order of virtual runtimes and the
public loop profiling route (m-lit lit-examples G4, G5).

* ``Runtime.evaluate_orders`` on a Born-interfered runtime with several
  orders (u u~ > d d~, NLO QCD, gluon and photon Borns) is a view of ONE
  ``evaluate`` call: its sectors and Born orders as they are, the explicit
  sums, and the sum equal to the total of an independent public path (the
  resolved evaluation summed over helicities, colour components and
  sectors) to rounding: within gamma_n S, the summation bound of its n
  resolved entries against the pre-cancellation scale S; a loop-induced
  runtime's view has no Born. The CLI
  ``--orders`` prints the explicit sums apart.
* ``pyamplicol profile`` / ``BenchmarkRunner`` of a loop artifact returns the
  per-point time and its split into trees, dressing, scatter, reduction and
  rest from the executor's stage report of the timed calls, for the checked
  default and the unchecked setting; the CLI on both reduction backends, and
  ``--stability`` the stable-digit statistics over flat points.
"""

from __future__ import annotations

import io
import json
import math
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "fixtures" / "loop" / "required" / "li"


def _artifact(
    root: Path,
    expression: str,
    extra: str = "",
    fixed: str = "",
    generation: str = "",
) -> Path:
    from pyamplicol.api import Generator, ProcessRequest, ProcessSet
    from pyamplicol.config import resolve_config

    root.mkdir(parents=True, exist_ok=True)
    card = root / "card.toml"
    card.write_text(
        f"""action = "generate"
[model]
source = "loop_sm"
restriction = "no_b_mass"
[process]
entries = [{{expression = {json.dumps(expression)}, name = "p"}}]
loop_coupling_orders = ["QCD"]
{extra}
[loop.preparation]
fixed_parameters = {{{fixed or 'WT = "0"'}}}
[loop]
r2_strategy = "model"
[generation]
output = "artifact"
emit_api_bundle = false
{generation}
"""
    )
    config = resolve_config(card)
    output = Path(config.effective.generation.output)
    Generator(config).generate(
        ProcessSet((ProcessRequest.parse(expression, name="p"),)), output
    )
    return output


def _points(artifact: Path) -> list:
    record = json.loads(
        (artifact / "processes" / "p" / "validation-momenta.json").read_text()
    )
    return [[entry["momentum"] for entry in point] for point in record["points"]]


@pytest.fixture(scope="module")
def uux_ddx(tmp_path_factory):
    return _artifact(
        tmp_path_factory.mktemp("orders"),
        "u u~ > d d~",
        'coupling_order_policy = "explicit"\n'
        "max_coupling_orders = { QCD = 2, QED = 2 }",
    )


@pytest.mark.loop_fast
def test_the_stage_groups_split_the_wall_time():
    from pyamplicol.runtime.virtual import PROFILE_GROUPS, profile_stages

    seconds = {
        "trees": 1.0,
        "trees: programs": 0.5,
        "seeds": 0.25,
        "factors": 0.25,
        "dressing": 1.0,
        "scatter": 0.5,
        "reduction": 2.0,
        "reduction: masters": 1.0,
        "seed contraction": 0.2,
    }
    virtual = profile_stages(seconds, 6.0, "virtual")
    assert tuple(virtual) == PROFILE_GROUPS
    assert virtual == {
        "trees": 1.0,
        "dressing": 1.5,
        "scatter": 0.5,
        "reduction": 2.0,
        "rest": 1.0,
    }
    induced = profile_stages(seconds, 6.0, "loop-induced")
    assert induced["dressing"] == 1.0 and induced["rest"] == 1.5
    assert profile_stages(seconds, 1.0, "virtual")["rest"] == 0.0


@pytest.mark.loop_fast
def test_the_profile_counts_every_native_batch_of_a_call():
    from pyamplicol.runtime.virtual import collected_stage_seconds

    class Executor:
        def __init__(self) -> None:
            self.last_report = None

        def evaluate(self, batches: int) -> None:
            # A call of several native batches (a per-point fallback, a
            # rescue level, one batch per scale), each with its report.
            for _ in range(batches):
                self.last_report = {"seconds": {"trees": 1.0, "reduction": 2.0}}

    class Backend:
        def __init__(self) -> None:
            self._executor = Executor()

    class Runtime:
        def __init__(self) -> None:
            self._backend = Backend()

    runtime, other = Runtime(), Runtime()
    with collected_stage_seconds(runtime) as seconds:
        runtime._backend._executor.evaluate(3)
        other._backend._executor.evaluate(1)
        assert runtime._backend._executor.last_report["seconds"]["trees"] == 1.0
    assert seconds == {"trees": 3.0, "reduction": 6.0}
    # Only the profiled runtime's executor recorded, and only while active.
    assert type(runtime._backend._executor) is Executor
    assert type(other._backend._executor) is Executor
    runtime._backend._executor.evaluate(1)
    assert seconds == {"trees": 3.0, "reduction": 6.0}
    assert "_collected_stage_seconds" not in vars(runtime._backend._executor)
    with collected_stage_seconds(object()) as nothing:
        assert nothing is None


@pytest.mark.loop_physics
def test_evaluate_orders_is_a_view_of_one_evaluation(uux_ddx, monkeypatch):
    from pyamplicol import Runtime
    from pyamplicol.api import VirtualOrderEvaluation, VirtualStabilityChecks

    runtime = Runtime.load(uux_ddx)
    points = _points(uux_ddx)
    calls = []
    evaluate = runtime._backend.evaluate

    def counted(*args, **kwargs):
        calls.append(1)
        return evaluate(*args, **kwargs)

    monkeypatch.setattr(runtime._backend, "evaluate", counted)
    view = runtime.evaluate_orders(
        points,
        mu_r2="10000",
        stability_checks=VirtualStabilityChecks(1, True, rescaled=True),
    )
    assert calls == [1]
    assert isinstance(view, VirtualOrderEvaluation)
    assert len(view.orders) == 3 and len(view.born_orders) == 3
    assert view.orders == tuple(sector.order for sector in view.evaluation.sectors)
    for index, order in enumerate(view.orders):
        assert view.for_order(order) == tuple(
            point[index] for point in view.evaluation.values
        )
    for index, order in enumerate(view.born_orders):
        assert view.born_for_order(order) == tuple(
            point[index] for point in view.evaluation.born_values
        )
    with pytest.raises(ValueError, match="absent from the catalogue"):
        view.for_order((("QCD", 99),))
    monkeypatch.undo()
    # The explicit sum against an independent public path: the resolved
    # evaluation summed over helicities and colour components (``total``)
    # and over its sectors. An identity held to rounding: within the
    # summation bound gamma_n S (n the resolved entries summed, u the
    # binary64 unit roundoff, S the pre-cancellation scale summed over the
    # sectors), which is far below 10 x the summed estimate here.
    resolved = runtime.evaluate_resolved(points, mu_r2="10000")
    total = resolved.total()
    entries = (
        len(resolved.helicity_ids)
        * len(resolved.virtual_color_ids)
        * len(resolved.sectors)
    )
    unit = 2.0**-53
    gamma = entries * unit / (1 - entries * unit)
    for point, summed in enumerate(view.sum_orders()):
        assert summed == tuple(
            math.fsum(float(sector[k]) for sector in view.evaluation.values[point])
            for k in range(3)
        )
        independent = [
            math.fsum(float(sector[k]) for sector in total.values[point])
            for k in range(3)
        ]
        estimate = view.evaluation.stability[point]
        for k in range(3):
            scale = math.fsum(float(m[k]) for m in estimate.magnitude)
            deviation = abs(float(summed[k]) - independent[k])
            assert deviation <= gamma * scale, (point, k, deviation / (unit * scale))
        born = math.fsum(float(x) for x in total.born_values[point])
        assert view.sum_born_orders()[point] == pytest.approx(born, rel=1e-13)


@pytest.mark.loop_physics
def test_the_cli_orders_and_profile_of_a_loop_artifact(uux_ddx, tmp_path):
    from pyamplicol.benchmarking import VirtualProfileResult
    from pyamplicol.cli.main import run_cli
    from pyamplicol.runtime.virtual import PROFILE_GROUPS

    momenta = tmp_path / "momenta.json"
    momenta.write_text(json.dumps(_points(uux_ddx)))
    stdout = io.StringIO()
    code = run_cli(
        [
            "evaluate",
            str(uux_ddx),
            "--momenta",
            str(momenta),
            "--mu-r2",
            "10000",
            "--orders",
            "--json",
            "--progress",
            "off",
        ],
        stdout=stdout,
        stderr=io.StringIO(),
    )
    assert code == 0
    payload = json.loads(stdout.getvalue())
    assert len(payload["orders"]) == 3
    assert len(payload["sum_orders"]) == len(_points(uux_ddx))
    assert payload["evaluation"]["contribution_kind"] == "virtual"
    human = io.StringIO()
    assert (
        run_cli(
            [
                "evaluate",
                str(uux_ddx),
                "--momenta",
                str(momenta),
                "--mu-r2",
                "10000",
                "--orders",
                "--progress",
                "off",
                "--color",
                "never",
            ],
            stdout=human,
            stderr=io.StringIO(),
        )
        == 0
    )
    assert "Explicit sums over the squared coupling orders" in human.getvalue()
    for setting in ([], ["--reduction-unchecked"]):
        stdout = io.StringIO()
        code = run_cli(
            [
                "profile",
                str(uux_ddx),
                "--batch-size",
                "4",
                "--minimum-samples",
                "2",
                "--warmup-runs",
                "0",
                "--json",
                "--progress",
                "off",
                *setting,
            ],
            stdout=stdout,
            stderr=io.StringIO(),
        )
        assert code == 0
        report = json.loads(stdout.getvalue())
        # Both reduction backends side by side, the type's default first and
        # profiled as "auto" (the default setup); the stability not asked.
        assert [row["backend"] for row in report["runtime"]] == ["ninjars", "tirust"]
        assert report["runtime"][0]["requested"] == "auto"
        assert report["stability"] is None and report["default_backend"] == "ninjars"
        other = report["runtime"][1]
        assert (other["profile"] is None) is (other["refusal"] is not None)
        profile = report["runtime"][0]["profile"]
        assert set(profile["stages"]) == set(PROFILE_GROUPS)
        assert profile["batch"] == 4 and profile["samples"] == 2
        assert profile["unchecked"] is bool(setting)
        assert profile["reduction"]["effective"] == "ninjars"
        # Every lane of every timed call counts (none falls back here).
        assert set(profile["reduction"]["lanes"]) == {"ninjars"}
        assert profile["reduction"]["lanes"]["ninjars"] >= 2
        assert profile["seconds_per_point"] > 0 and profile["label"] is None
        assert sum(profile["stages"].values()) > 0
    # --stability on both backends over three flat points (binary64 with the
    # default rotation against a double-double re-run); human and JSON.
    for output in (["--json"], ["--color", "never"]):
        stdout = io.StringIO()
        code = run_cli(
            [
                "profile",
                str(uux_ddx),
                "--stability",
                "--stability-points",
                "3",
                "--sqrt-s",
                "500",
                "--progress",
                "off",
                *output,
            ],
            stdout=stdout,
            stderr=io.StringIO(),
        )
        assert code == 0
        if output == ["--json"]:
            report = json.loads(stdout.getvalue())
            assert report["runtime"] is None
            assert report["stability_setup"]["points"] == 3
            assert report["stability_setup"]["sqrt_s"] == 500.0
            first = report["stability"][0]
            assert first["backend"] == "ninjars" and first["refusal"] is None
            assert first["points"] + first["failed_reference"] == 3
            # Every flat point is recorded by index; binary64 failures rank
            # among the measured points at 0 digits.
            assert [p["index"] for p in first["per_point"]] == [0, 1, 2]
            assert first["failed_binary64"] == sum(
                p["status"] == "binary64-failed" for p in first["per_point"]
            )
            assert len(first["percentiles"]) == 4
            assert all(0.0 <= d <= 16.0 for d in first["true_digits"])
        else:
            text = stdout.getvalue()
            assert "true digits" in text and "NinjaRS" in text and "50 %" in text
            assert max(len(line) for line in text.splitlines()) <= 100
    from pyamplicol.api import BenchmarkRunner

    result = BenchmarkRunner().run(uux_ddx)
    assert isinstance(result, VirtualProfileResult)
    assert result.contribution == "virtual" and result.failed_points == 0


@pytest.mark.loop_physics
def test_a_loop_induced_view_has_no_born(tmp_path):
    from pyamplicol import Runtime

    fixture = json.loads((FIXTURES / "li_gg_aa_nf5.json").read_text())
    fixed = ", ".join(
        f"{key} = {json.dumps(value)}"
        for key, value in fixture["scheme_data"]["fixed_parameters"].items()
    )
    artifact = _artifact(
        tmp_path,
        "g g > a a",
        "",
        fixed,
        'particle_veto = ["t"]',
    )
    runtime = Runtime.load(artifact)
    view = runtime.evaluate_orders(_points(artifact)[:1], mu_r2="10000")
    assert view.contribution == "loop-induced"
    assert view.born_orders == () and view.sum_born_orders() == (0.0,)
    assert len(view.orders) == 1
    (only,) = view.orders
    assert view.sum_orders()[0] == view.for_order(only)[0]
