# SPDX-License-Identifier: 0BSD
"""Opt-in genuine public colour publication; no native/artifact identity mocks."""

import io
import json
from decimal import Decimal, localcontext

import pytest

from pyamplicol.api import Generator, ProcessRequest, ProcessSet, Runtime
from pyamplicol.cli import run_cli
from pyamplicol.config import resolve_config

pytestmark = pytest.mark.loop_physics


def _card(model, output, process, name, accuracy):
    return f"""
action = "generate"
[model]
source = {json.dumps(model)}
[process]
entries = [{{expression={json.dumps(process)}, name={json.dumps(name)}}}]
loop_coupling_orders = ["QCD"]
coupling_order_policy = "minimal"
[color]
accuracy = {json.dumps(accuracy)}
contraction = "direct"
[loop]
r2_strategy = "model"
[generation]
output = {json.dumps(str(output))}
emit_api_bundle = false
[generation.validation]
post_build_validation = true
samples = 2
"""


@pytest.mark.parametrize(
    "process,name,accuracy,components",
    [
        ("g g > g g", "gg", "lc", 6),
        ("u u~ > e- e+", "dy", "lc", 1),
        ("g g > g g", "gg_nlc", "nlc", 1),
    ],
)
def test_real_declared_colour_generation_and_saved_components(
    loop_sm_colour_model, tmp_path, process, name, accuracy, components
):
    model = str(loop_sm_colour_model)
    # A caller can select the adjoint installed-package proof independently from
    # the open-fundamental proof, which requires the corrected tensor dependency.
    output = tmp_path / name
    card = tmp_path / f"{name}.toml"
    card.write_text(_card(model, output, process, name, accuracy))
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
    assert coverage["color_accuracy"] == accuracy
    assert coverage["source_colour_declared"] is True
    assert coverage["formal_colour_process_coverage_checked"] is False
    assert coverage["canonical_born_components_checked"] is (
        False if accuracy == "lc" else None
    )
    assert not output.exists()

    request = ProcessSet((ProcessRequest.parse(process, name=name),))
    result = Generator(resolve_config(card)).generate(request, output)
    assert result.output == output
    point = [
        [500, 0, 0, 500],
        [500, 0, 0, -500],
        [500, 300, 0, 400],
        [500, -300, 0, -400],
    ]
    # The normal writer has already compared the actual saved artifact to its
    # source program on two independent generated points before publication.
    runtime = Runtime.load(output, process=name)
    inclusive = runtime.evaluate([point], mu_r2="1000000")
    resolved = runtime.evaluate_resolved([point], mu_r2="1000000")
    assert inclusive.color_accuracy == resolved.color_accuracy == accuracy
    assert resolved.shape == (1, 1, 16, components, 3)
    assert resolved.born_shape == (1, 1, 16, components)
    assert resolved.born_color_ids == resolved.virtual_color_ids
    assert runtime.physics.color_flows == ()
    # LC resolves per seed-basis column, each leading-colour flow anchored to
    # its Born basis element (the flows ``evaluate(color_flows=...)``
    # selects); NLC is one contraction.
    if accuracy == "lc":
        assert all(value.startswith("basis-") for value in resolved.born_color_ids)
        assert runtime.physics.reduction.kind == "born-anchored-interference"
        anchored = {
            index
            for anchor in runtime.physics.born_anchored_color_components
            for index in anchor.born_term_indices
        }
        assert anchored == set(range(components))
    else:
        assert resolved.born_color_ids == ("nlc",)
        assert runtime.physics.reduction.kind == "contracted-color"
        assert runtime.physics.born_anchored_color_components == ()
    total = resolved.total()
    assert total.values[0][0] == pytest.approx(inclusive.values[0][0], rel=3e-10)
    assert total.born_values[0] == pytest.approx(inclusive.born_values[0], rel=3e-10)
    assert all(row.status == "success" for row in inclusive.diagnostics)

    precise = Runtime.load(output, process=name, arithmetic="arbitrary", precision=70)
    ap = precise.evaluate([point], mu_r2="1000000")
    ap_total = precise.evaluate_resolved([point], mu_r2="1000000").total()
    with localcontext() as context:
        context.prec = 100
        for a, b in zip(
            (*ap_total.born_values[0], *ap_total.values[0][0]),
            (*ap.born_values[0], *ap.values[0][0]),
            strict=True,
        ):
            assert abs(Decimal(a) - Decimal(b)) <= Decimal("1e-58") * max(
                abs(Decimal(b)), Decimal(1)
            )
    assert tuple(map(float, ap.values[0][0])) == pytest.approx(
        inclusive.values[0][0], rel=3e-9
    )
