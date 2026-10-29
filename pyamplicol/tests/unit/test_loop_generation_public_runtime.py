# SPDX-License-Identifier: 0BSD
"""Opt-in real public generation, with no mocked artifact/native boundaries."""

import io
import json
from decimal import Decimal, localcontext

import pytest

from pyamplicol.api import Generator, ProcessRequest, ProcessSet, Runtime
from pyamplicol.cli import run_cli
from pyamplicol.config import resolve_config

pytestmark = pytest.mark.loop_physics


def _card(model, output):
    # Explicit declarations of the prepared supplied QCD fixture, not generic
    # certificates inferred from the model name or the presence of CT vertices.
    return f"""
action = "generate"
[model]
source = {json.dumps(str(model))}
[process]
entries = [{{expression="u u~ > e- e+", name="dy"}},
           {{expression="g g > g g", name="gg"}}]
loop_coupling_orders = ["QCD"]
coupling_order_policy = "minimal"
[color]
accuracy = "full"
contraction = "direct"
[loop]
r2_strategy = "model"
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
[generation]
output = {json.dumps(str(output))}
emit_api_bundle = false
[generation.validation]
post_build_validation = true
samples = 2
"""


@pytest.fixture
def public_request(loop_sm_compiled_model, tmp_path):
    model = str(loop_sm_compiled_model)
    card = tmp_path / "virtual.toml"
    output = tmp_path / "generated"
    card.write_text(_card(model, output))
    return card, output


def test_real_public_virtual_cli_plan_enumerates_no_loop_diagram(public_request):
    card, output = public_request
    stdout, stderr = io.StringIO(), io.StringIO()
    code = run_cli(
        ("generate", "--card", str(card), "--dry-run", "--json", "--progress", "off"),
        stdout=stdout,
        stderr=stderr,
    )
    assert code == 0, stderr.getvalue()
    plan = json.loads(stdout.getvalue())
    assert plan["unsupported_features"] == []
    assert plan["estimated_coverage"]["resolved"] is True
    assert not output.exists()
    processes = plan["estimated_coverage"]["processes"]
    assert {row["key"] for row in processes} == {"dy", "gg"}
    # Engine v2 (the only one-loop engine) plans without loop diagrams.
    assert plan["estimated_coverage"]["engine"] == "v2"
    assert all(row["engine"] == "v2" for row in processes)
    assert all(row["loop_graph_count"] == 0 for row in processes)


def test_real_public_generation_roundtrips_two_processes_and_cli(
    loop_native, public_request
):
    # Publication must use an authentic staged extension and build metadata.
    # Deliberately do not replace target_info, identity or the runtime loader.
    card, output = public_request
    processes = ProcessSet(
        (
            ProcessRequest.parse("u u~ > e- e+", name="dy"),
            ProcessRequest.parse("g g > g g", name="gg"),
        )
    )
    result = Generator(resolve_config(card)).generate(processes, output)
    assert result.output == output
    # A nonexceptional return also establishes that every staged source/runtime
    # sample, Born sector and Laurent coefficient passed transactional validation.
    point = [
        [500, 0, 0, 500],
        [500, 0, 0, -500],
        [500, 300, 0, 400],
        [500, -300, 0, -400],
    ]
    momentum_path = output.parent / "momenta.json"
    momentum_path.write_text(json.dumps([point, point]))
    records = {}
    for process_id in ("dy", "gg"):
        runtime = Runtime.load(output, process=process_id)
        values = runtime.evaluate([point, point], mu_r2=["1000000", "900000"])
        assert values.shape == (2, 1, 3)
        assert values.born_shape == (2, 1)
        assert all(row.status == "success" for row in values.diagnostics)
        assert values.values[0] != values.values[1]  # scales bind per lane
        records[process_id] = values
        stdout, stderr = io.StringIO(), io.StringIO()
        code = run_cli(
            (
                "evaluate",
                str(output),
                "--process",
                process_id,
                "--momenta",
                str(momentum_path),
                "--mu-r2",
                "1000000",
                "--arithmetic",
                "double",
                "--json",
                "--progress",
                "off",
            ),
            stdout=stdout,
            stderr=stderr,
        )
        assert code == 0, stderr.getvalue()
        payload = json.loads(stdout.getvalue())
        assert payload["shape"] == [2, 1, 3]
        assert payload["epsilon_powers"] == [0, -1, -2]
        assert payload["values"][0][0] == pytest.approx(values.values[0][0])

    # Exercise load-time arithmetic before binding, then the same exact saved
    # instructions for a nonempty batch. This is a carrier/transport regression,
    # not a claim of reference-level accuracy at a particular digit count.
    for process_id in ("dy", "gg"):
        for arithmetic, precision in (("double-double", 32), ("arbitrary", 70)):
            runtime = Runtime.load(
                output, process=process_id, arithmetic=arithmetic, precision=precision
            )
            values = runtime.evaluate([point], mu_r2="1000000")
            assert values.diagnostics[0].arithmetic == arithmetic
            assert tuple(
                float(value) for value in values.values[0][0]
            ) == pytest.approx(records[process_id].values[0][0], rel=1e-9, abs=1e-14)


def test_real_public_generation_publishes_resolved_drell_yan(public_request):
    from pyamplicol.api.virtual_results import ResolvedVirtualEvaluation

    card, output = public_request
    process = ProcessSet((ProcessRequest.parse("u u~ > e- e+", name="dy"),))
    result = Generator(resolve_config(card)).generate(process, output)
    assert len(result.processes.requests) == 1
    point = [
        [500, 0, 0, 500],
        [500, 0, 0, -500],
        [500, 300, 0, 400],
        [500, -300, 0, -400],
    ]
    for arithmetic, precision, tolerance in (
        ("double", 16, "3e-11"),
        ("double-double", 32, "3e-25"),
        ("arbitrary", 70, "1e-58"),
    ):
        runtime = Runtime.load(
            output, arithmetic=arithmetic, precision=precision, process="dy"
        )
        assert runtime.inspect()["resolved"] is True
        inclusive = runtime.evaluate([point], mu_r2="1000000")
        resolved = runtime.evaluate_resolved([point], mu_r2="1000000")
        assert isinstance(resolved, ResolvedVirtualEvaluation)
        assert resolved.shape == (1, 1, 16, 1, 3)
        assert resolved.born_shape == (1, 1, 16, 1)
        assert resolved.virtual_color_ids == resolved.born_color_ids == ("full",)
        total = resolved.total()
        with localcontext() as context:
            context.prec = 100
            for a, b in zip(
                (total.born_values[0][0], *total.values[0][0]),
                (inclusive.born_values[0][0], *inclusive.values[0][0]),
                strict=True,
            ):
                assert abs(Decimal(a) - Decimal(b)) <= Decimal(tolerance) * max(
                    abs(Decimal(b)), Decimal("1")
                )
