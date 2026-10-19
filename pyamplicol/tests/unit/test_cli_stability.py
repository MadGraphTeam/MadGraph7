# SPDX-License-Identifier: 0BSD
"""CLI/config transport shares the public virtual policy and result contracts."""

import io
import json
import tomllib
from dataclasses import replace
from decimal import Decimal

import pytest

from pyamplicol.api import Runtime, VirtualArithmeticSpec, VirtualStabilityPolicy
from pyamplicol.api.virtual_results import (
    VirtualFailureDiagnostic,
    VirtualLaneDiagnostic,
)
from pyamplicol.cli import parse_cli, run_cli
from pyamplicol.cli.handlers import DefaultCliServices, dispatch
from pyamplicol.cli.main import write_result
from pyamplicol.config import (
    Action,
    ConfigurationError,
    EvaluationConfig,
    EvaluationStabilityConfig,
    RunConfig,
    config_to_toml,
    resolve_config,
)
from pyamplicol.reporting import CallbackProgressSink

from .test_cli_virtual import _result

POLICY_FLAGS = (
    "--stability-higher",
    "arbitrary:60",
    "--stability-higher",
    "arbitrary:80",
    "--stability-relative-tolerance",
    "1e-8",
    "--stability-absolute-tolerance",
    "1e-4000",
)


def test_stability_card_flags_and_canonical_round_trip(tmp_path):
    policy = VirtualStabilityPolicy(
        (
            VirtualArithmeticSpec("arbitrary", 60),
            VirtualArithmeticSpec("arbitrary", 80),
        ),
        "1e-8",
        "1e-4000",
    )
    flags = parse_cli(("evaluate", *POLICY_FLAGS, "--on-failure", "return"))
    evaluation = flags.resolve().effective.evaluation
    assert evaluation.stability.to_policy() == policy
    assert evaluation.on_failure == "return"
    document = config_to_toml(flags.resolve().effective)
    roundtrip = resolve_config(tomllib.loads(document)).effective
    assert roundtrip.evaluation == evaluation
    assert "[evaluation.stability]" in document
    assert 'absolute_tolerance = "1e-4000"' in document
    card = tmp_path / "policy.toml"
    card.write_text(document)
    updated = (
        parse_cli(
            (
                "evaluate",
                "--card",
                str(card),
                "--stability-higher",
                "double-double",
                "--stability-higher",
                "arbitrary:100000",
                "--verify-successful",
                "--set",
                'evaluation.stability.absolute_tolerance="0"',
            )
        )
        .resolve()
        .effective.evaluation.stability.to_policy()
    )
    assert updated.higher == (
        VirtualArithmeticSpec("double-double"),
        VirtualArithmeticSpec("arbitrary", 100000),
    )
    assert updated.absolute_tolerance == "0" and updated.verify_successful
    assert EvaluationStabilityConfig().to_policy() is None
    output = io.StringIO()
    assert run_cli(("config", "resolve", str(card), "--json"), stdout=output) == 0
    assert "1e-4000" in output.getvalue()


@pytest.mark.parametrize(
    "higher",
    [
        ("60",),
        ("quad",),
        ("arbitrary",),
        ("arbitrary:0",),
        ("arbitrary:-1",),
        ("arbitrary:1.2",),
        ("arbitrary:+60",),
        ("double:16",),
        ("double-double:32",),
        ("arbitrary:60:80",),
    ],
)
def test_stability_requires_explicit_carriers(higher):
    with pytest.raises(ConfigurationError, match=r"evaluation\.stability"):
        EvaluationStabilityConfig(higher, "1e-8", "0")


@pytest.mark.parametrize(
    "settings",
    [
        {"relative_tolerance": "1e-8"},
        {"absolute_tolerance": "0"},
        {"verify_successful": True},
        {"verify_successful": 1},
        {"higher": ("arbitrary:80",)},
        {"higher": ("arbitrary:80",), "relative_tolerance": "1e-8"},
        {
            "higher": ("arbitrary:80",),
            "relative_tolerance": "bad",
            "absolute_tolerance": "0",
        },
        {
            "higher": ("arbitrary:80",),
            "relative_tolerance": "NaN",
            "absolute_tolerance": "0",
        },
        {
            "higher": ("arbitrary:80",),
            "relative_tolerance": "-1",
            "absolute_tolerance": "0",
        },
        {
            "higher": ("arbitrary:80",),
            "relative_tolerance": "0",
            "absolute_tolerance": "0",
        },
        {
            "higher": ("arbitrary:80",),
            "relative_tolerance": 1e-8,
            "absolute_tolerance": "0",
        },
    ],
)
def test_stability_rejects_incomplete_or_inexact_settings(settings):
    with pytest.raises(ConfigurationError, match=r"evaluation\.stability"):
        EvaluationStabilityConfig(**settings)


@pytest.mark.parametrize("resolved", [False, True])
def test_cli_policy_precedes_binding_and_failure_choice_is_per_call(
    tmp_path, monkeypatch, resolved
):
    path = tmp_path / "points.json"
    path.write_text("[[[1e-4000,0,0,0]]]")
    calls = []

    class Loaded:
        def evaluate(self, points, **options):
            calls.append(("ordinary", points, options))
            return _result()

        def evaluate_resolved(self, points, **options):
            calls.append(("resolved", points, options))
            return _result()

    def load(artifact, **options):
        assert isinstance(options["stability"], VirtualStabilityPolicy)
        assert options["stability"].absolute_tolerance == "1e-4000"
        assert options["arithmetic"] == "double-double"
        calls.append(("load", artifact, options))
        return Loaded()

    monkeypatch.setattr(Runtime, "load", load)
    monkeypatch.setattr(
        "pyamplicol.cli.handlers._virtual_artifact_hint", lambda _: True
    )
    config = (
        parse_cli(
            (
                "evaluate",
                "artifact",
                "--momenta",
                str(path),
                "--mu-r2",
                "1e-4000",
                "--arithmetic",
                "double-double",
                "--on-failure",
                "return",
                *POLICY_FLAGS,
                *(("--resolved",) if resolved else ()),
            )
        )
        .resolve()
        .effective
    )
    DefaultCliServices().evaluate(config, CallbackProgressSink(lambda _: None))
    assert [call[0] for call in calls] == [
        "load",
        "resolved" if resolved else "ordinary",
    ]
    assert calls[1][1][0][0][0] == Decimal("1e-4000")
    assert calls[1][2]["on_failure"] == "return"
    assert "stability" not in calls[1][2]


@pytest.mark.parametrize("flags", [POLICY_FLAGS, ("--on-failure", "return")])
def test_tree_opt_in_rejected_before_load(tmp_path, monkeypatch, flags):
    path = tmp_path / "points.json"
    path.write_text("[]")
    monkeypatch.setattr(
        "pyamplicol.cli.handlers._virtual_artifact_hint", lambda _: False
    )
    monkeypatch.setattr(
        Runtime, "load", lambda *a, **k: pytest.fail("tree load reached")
    )
    config = (
        parse_cli(("evaluate", "artifact", "--momenta", str(path), *flags))
        .resolve()
        .effective
    )
    with pytest.raises(ConfigurationError, match="require a virtual artifact"):
        DefaultCliServices().evaluate(config, CallbackProgressSink(lambda _: None))


@pytest.mark.parametrize(
    "action", [action for action in Action if action != Action.EVALUATE]
)
@pytest.mark.parametrize(
    "evaluation",
    [
        EvaluationConfig(on_failure="return"),
        EvaluationConfig(
            stability=EvaluationStabilityConfig(("arbitrary:80",), "1e-8", "0")
        ),
    ],
)
def test_non_evaluation_actions_do_not_ignore_policy(action, evaluation):
    with pytest.raises(ConfigurationError, match="require the evaluate action"):
        dispatch(
            RunConfig(action=action, evaluation=evaluation),
            object(),
            CallbackProgressSink(lambda _: None),
        )


@pytest.mark.parametrize("correlated", [False, True])
def test_partial_cli_serialization_preserves_holes_and_exact_complex_values(correlated):
    ordinary = _result(correlated=correlated)
    failure = VirtualFailureDiagnostic(
        0, None, None, None, "external wavefunctions", "singular point"
    )
    result = replace(
        ordinary,
        born_values=(None, ordinary.born_values[0]),
        values=(None, ordinary.values[0]),
        diagnostics=(
            VirtualLaneDiagnostic(
                "failed", "arbitrary", 233, False, failures=(failure,)
            ),
            *ordinary.diagnostics,
        ),
    )
    output, human = io.StringIO(), io.StringIO()
    write_result(result, format="json", stream=output)
    write_result(result, format="human", stream=human)
    payload = json.loads(output.getvalue())
    assert payload["values"][0] is payload["born_values"][0] is None
    assert payload["diagnostics"][0]["failures"][0]["point_id"] == 0
    assert payload["epsilon_powers"] == [0, -1, -2]
    value = payload["values"][1][0][0]
    assert (value["real"] if correlated else value) == str(
        ordinary.values[0][0][0].real if correlated else ordinary.values[0][0][0]
    )
    assert "unavailable" in human.getvalue()
