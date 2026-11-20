# SPDX-License-Identifier: 0BSD
"""Numerical request ingress and grouped presentation reuse existing APIs."""

import io
import json
import tomllib
from dataclasses import replace
from decimal import Decimal

import pytest

from pyamplicol import CorrelatedRequest, CorrelatedValue, Runtime
from pyamplicol.api.virtual_results import (
    VirtualFailureDiagnostic,
    VirtualLaneDiagnostic,
)
from pyamplicol.cli import parse_cli
from pyamplicol.cli.correlated_requests import correlated_requests_from_json
from pyamplicol.cli.handlers import DefaultCliServices, _read_json, dispatch
from pyamplicol.cli.main import write_result
from pyamplicol.config import (
    ConfigurationError,
    EvaluationConfig,
    RunConfig,
    config_to_toml,
    resolve_config,
)
from pyamplicol.reporting import CallbackProgressSink

from .test_cli_virtual import _result


def test_request_path_roundtrip_and_incompatible_axes(tmp_path):
    request_file = tmp_path / "requests.json"
    config = (
        parse_cli(("evaluate", "--correlated-requests", str(request_file)))
        .resolve()
        .effective
    )
    assert config.evaluation.correlated_requests == request_file
    document = config_to_toml(config)
    assert (
        resolve_config(tomllib.loads(document)).effective.evaluation
        == config.evaluation
    )
    for options in ({"resolved": True}, {"color_flow_ids": ("flow1",)}):
        with pytest.raises(ConfigurationError, match="cannot combine"):
            EvaluationConfig(correlated_requests=request_file, **options)
    with pytest.raises(ConfigurationError, match="require the evaluate action"):
        dispatch(
            RunConfig(action="benchmark", evaluation=config.evaluation),
            object(),
            CallbackProgressSink(lambda _: None),
        )


def test_exact_request_values_defaults_and_duplicate_detection(tmp_path):
    path = tmp_path / "requests.json"
    path.write_text("""{
      "default": {}, "physical": {"spin_vectors": {}},
      "joint": {"color_correlation": "ordered", "spin_vectors": {
        "03": [[0,0], [1.000000000000000000000000000000001, -1e-4000], 0, 0],
        "4": [0, 0, "1", 0]
      }}
    }""")
    requests = correlated_requests_from_json(
        _read_json(path, exact=True, reject_duplicates=True)
    )
    assert list(requests) == ["default", "physical", "joint"]
    assert requests["default"] == CorrelatedRequest()
    assert requests["physical"].spin_vectors == {}
    assert requests["joint"].color_correlation == "ordered"
    assert requests["joint"].spin_vectors[3][1] == [
        Decimal("1.000000000000000000000000000000001"),
        Decimal("-1e-4000"),
    ]
    for raw in (
        '{"x":{},"x":{}}',
        '{"x":{"spin_vectors":{"3":[],"3":[]}}}',
        '{"x":{"color_correlation":"born","color_correlation":"other"}}',
    ):
        path.write_text(raw)
        with pytest.raises(ConfigurationError, match="duplicate JSON object field"):
            _read_json(path, exact=True, reject_duplicates=True)
        assert isinstance(_read_json(path, exact=True), dict)


@pytest.mark.parametrize(
    "value",
    [
        [],
        {"": {}},
        {"x": []},
        {"x": {"unknown": 1}},
        {"x": {"color_correlation": 2}},
        {"x": {"color_correlation": ""}},
        {"x": {"spin_vectors": []}},
        {"x": {"spin_vectors": {"leg3": []}}},
        {"x": {"spin_vectors": {"3.0": []}}},
        {"x": {"spin_vectors": {"3": [], "03": []}}},
        {"x": {"spin_vectors": {"3": [], "+3": []}}},
    ],
)
def test_request_structure_errors_are_contextual(value):
    with pytest.raises(ConfigurationError, match="correlated request"):
        correlated_requests_from_json(value)


@pytest.mark.parametrize(
    "virtual,arithmetic",
    [(True, "double"), (False, None), (False, "double-double"), (False, "arbitrary")],
)
def test_grouped_cli_calls_one_public_batch_and_preserves_tree_arithmetic(
    tmp_path, monkeypatch, virtual, arithmetic
):
    points, requests, parameters = (
        tmp_path / name for name in ("points.json", "requests.json", "parameters.json")
    )
    points.write_text("[[[1.000000000000000000000000000001,0,0,1]],[[2,0,0,2]]]")
    requests.write_text(
        '{"physical": {}, "joint": {"spin_vectors": {"3": [0,1e-4000,0,0]}}}'
    )
    parameters.write_text('{"aS":0.118000000000000000000000000001}')
    calls = []

    class Loaded:
        def evaluate_correlated_many(self, points, requests, **options):
            calls.append(("evaluate", points, requests, options))
            return {label: _result(correlated=True) for label in requests}

    def load(path, **options):
        calls.append(("load", options))
        assert options["model_parameters"]["aS"] == Decimal(
            "0.118000000000000000000000000001"
        )
        if virtual:
            assert "stability" not in options and options["arithmetic"] == arithmetic
        else:
            assert set(options) == {"process", "model_parameters"}
        return Loaded()

    monkeypatch.setattr(Runtime, "load", load)
    monkeypatch.setattr(
        "pyamplicol.cli.handlers._virtual_artifact_hint", lambda _: virtual
    )
    flags = [
        "evaluate",
        "artifact",
        "--momenta",
        str(points),
        "--correlated-requests",
        str(requests),
        "--model-parameters",
        str(parameters),
    ]
    if arithmetic is not None:
        flags += ["--arithmetic", arithmetic]
    if virtual:
        flags += ["--mu-r2", "1000000", "--on-failure", "return"]
    config = parse_cli(flags).resolve().effective
    DefaultCliServices().evaluate(config, CallbackProgressSink(lambda _: None))
    assert [call[0] for call in calls] == ["load", "evaluate"]
    _, batch, mapped, options = calls[-1]
    assert len(batch) == 2 and list(mapped) == ["physical", "joint"]
    assert batch[0][0][0] == Decimal("1.000000000000000000000000000001")
    assert mapped["joint"].spin_vectors[3][1] == Decimal("1e-4000")
    assert options["precision"] == 16 and options["helicities"] is None
    if virtual:
        assert options["mu_r2"] == "1000000" and options["on_failure"] == "return"
        assert "stability" not in options
    else:
        assert options["arithmetic"] == arithmetic
        assert set(options) == {"arithmetic", "precision", "helicities"}


@pytest.mark.parametrize(
    "flags",
    [
        ("--mu-r2", "1"),
        ("--r2-strategy", "model"),
        ("--on-failure", "return"),
        ("--poles", "none"),
    ],
)
def test_tree_grouped_rejects_virtual_options_before_loading(
    tmp_path, monkeypatch, flags
):
    points, requests = tmp_path / "points.json", tmp_path / "requests.json"
    points.write_text("[]")
    requests.write_text('{"physical": {}}')
    monkeypatch.setattr(
        "pyamplicol.cli.handlers._virtual_artifact_hint", lambda _: False
    )
    monkeypatch.setattr(Runtime, "load", lambda *a, **k: pytest.fail("load reached"))
    config = (
        parse_cli(
            (
                "evaluate",
                "artifact",
                "--momenta",
                str(points),
                "--correlated-requests",
                str(requests),
                *flags,
            )
        )
        .resolve()
        .effective
    )
    with pytest.raises(ConfigurationError, match="require a virtual artifact"):
        DefaultCliServices().evaluate(config, CallbackProgressSink(lambda _: None))


def test_grouped_human_json_preserve_masks_and_exact_complex_values():
    ordinary = _result(correlated=True)
    failed = replace(
        ordinary,
        born_values=(None,),
        values=(None,),
        diagnostics=(
            VirtualLaneDiagnostic(
                "failed",
                "arbitrary",
                233,
                False,
                failures=(
                    VirtualFailureDiagnostic(0, None, None, None, "source", "overflow"),
                ),
            ),
        ),
    )
    results = {"physical": ordinary, "failed-source": failed}
    human, machine = io.StringIO(), io.StringIO()
    write_result(results, format="human", stream=human)
    write_result(results, format="json", stream=machine)
    payload = json.loads(machine.getvalue())
    assert payload["failed-source"]["values"] == [None]
    assert payload["physical"]["epsilon_powers"] == [0, -1, -2]
    assert payload["physical"]["values"][0][1][0] == {"real": "-4", "imag": "4"}
    assert "Correlated request: physical" in human.getvalue()
    assert "Correlated request: failed-source" in human.getvalue()
    assert (
        "unavailable" in human.getvalue() and "single pole" in human.getvalue().lower()
    )
    tree = {
        "source": (CorrelatedValue(Decimal("1e-4000"), Decimal("-3.25")),),
        "empty": (),
    }
    human, machine = io.StringIO(), io.StringIO()
    write_result(tree, format="human", stream=human)
    write_result(tree, format="json", stream=machine)
    assert "1E-4000" in human.getvalue() and "-3.25" in human.getvalue()
    assert json.loads(machine.getvalue())["source"][0] == {
        "real": "1E-4000",
        "imag": "-3.25",
    }
    unrelated = io.StringIO()
    write_result({"warnings": ()}, format="human", stream=unrelated)
    assert "Correlated request" not in unrelated.getvalue()
    empty = io.StringIO()
    write_result({"empty": ()}, format="human", stream=empty)
    assert "empty" in empty.getvalue()
