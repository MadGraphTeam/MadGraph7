# SPDX-License-Identifier: 0BSD
"""CLI/config transport of virtual evaluation options and result contracts."""

import io
import json
import tomllib
from dataclasses import replace
from decimal import Decimal

import pytest

from pyamplicol.api import Runtime
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

REMOVED = r"evaluation\.stability \(the precision-rescue policy.* was removed"
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


@pytest.mark.parametrize(
    "flags",
    [
        POLICY_FLAGS,
        ("--stability-higher", "double-double"),
        ("--stability-relative-tolerance", "1e-8"),
        ("--stability-absolute-tolerance", "0"),
        ("--verify-successful",),
        ("--set", 'evaluation.stability.higher=["arbitrary:60"]'),
    ],
)
def test_removed_rescue_options_are_refused_with_the_reason(flags):
    """The precision rescue left with the legacy engine: every spelling of it
    fails with the removal reason (v2 runtimes refuse a policy at load too)."""
    with pytest.raises(ConfigurationError, match=REMOVED):
        parse_cli(("evaluate", *flags)).resolve()
    assert run_cli(("evaluate", *flags), stdout=io.StringIO()) == 2


def test_removed_rescue_section_keeps_only_its_defaults(tmp_path):
    card = tmp_path / "policy.toml"
    card.write_text(
        'action = "evaluate"\n[evaluation.stability]\nhigher = ["arbitrary:60"]\n'
        'relative_tolerance = "1e-8"\nabsolute_tolerance = "0"\n'
    )
    with pytest.raises(ConfigurationError, match=REMOVED):
        resolve_config(card)
    # Stored configurations carry the defaults; they still resolve.
    stored = (
        'action = "evaluate"\n'
        "[evaluation.stability]\nhigher = []\nverify_successful = false\n"
    )
    effective = resolve_config(tomllib.loads(stored)).effective
    assert effective.evaluation.stability == EvaluationStabilityConfig()
    document = config_to_toml(effective)
    assert resolve_config(tomllib.loads(document)).effective == effective
    defaults = parse_cli(("evaluate", "--no-verify-successful")).resolve()
    assert defaults.effective.evaluation.stability == EvaluationStabilityConfig()


@pytest.mark.parametrize(
    "settings",
    [
        {"higher": ("arbitrary:80",)},
        {"higher": ("60",)},
        {"higher": ("arbitrary:80",), "relative_tolerance": "1e-8"},
        {
            "higher": ("arbitrary:80",),
            "relative_tolerance": "1e-8",
            "absolute_tolerance": "0",
        },
        {"relative_tolerance": "1e-8"},
        {"absolute_tolerance": "0"},
        {"verify_successful": True},
        {"verify_successful": 1},
    ],
)
def test_rescue_settings_are_refused_in_the_python_config(settings):
    with pytest.raises(ConfigurationError, match=REMOVED):
        EvaluationStabilityConfig(**settings)


@pytest.mark.parametrize("resolved", [False, True])
def test_cli_arithmetic_precedes_binding_and_failure_choice_is_per_call(
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
        assert "stability" not in options
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


@pytest.mark.parametrize("flags", [("--on-failure", "return"), ("--poles", "none")])
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
        EvaluationConfig(correlated_requests="requests.json"),
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
