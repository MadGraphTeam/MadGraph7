# SPDX-License-Identifier: 0BSD
"""Virtual CLI transport uses the native runtime without dropping exact inputs."""

import io
import json
from decimal import Decimal, localcontext

import pytest

from pyamplicol.api.virtual_results import VirtualEvaluation, VirtualLaneDiagnostic
from pyamplicol.cli import parse_cli
from pyamplicol.cli.handlers import DefaultCliServices, _read_json
from pyamplicol.cli.main import write_result
from pyamplicol.config import ConfigurationError, EvaluationConfig, R2Strategy
from pyamplicol.correlators import CorrelatedValue
from pyamplicol.loops.orders import BornLoopPair, VirtualOrderSector
from pyamplicol.loops.strategy import R2Selection
from pyamplicol.reporting import CallbackProgressSink

pytestmark = pytest.mark.loop_fast


def _result(*, correlated=False):
    exact = Decimal("1.234567890123456789012345678901234567890123456789")

    def convert(value):
        value = Decimal(value)
        return CorrelatedValue(value, value.copy_negate()) if correlated else value

    value = convert(exact)
    qcd2, qcd4, qed2 = (("QCD", 2),), (("QCD", 4),), (("QED", 2),)
    return VirtualEvaluation(
        ((value,),),
        (((value, convert(-2), convert(3)), (convert(-4), convert(5), convert(-6))),),
        (qcd4,),
        (
            VirtualOrderSector((("QCD", 6),), (BornLoopPair(qcd2, qcd4),)),
            VirtualOrderSector(
                qcd4 + qed2, (BornLoopPair(qed2, qcd4), BornLoopPair(qcd2, qcd2 + qed2))
            ),
        ),
        (VirtualLaneDiagnostic("success", "arbitrary", 233, False),),
        R2Selection(R2Strategy.AUTO, R2Strategy.MODEL, "only complete strategy"),
        correlated=correlated,
    )


def test_virtual_flags_and_toml_scale_choices(tmp_path):
    config = (
        parse_cli(
            (
                "evaluate",
                "artifact",
                "--arithmetic",
                "double-double",
                "--mu-r2",
                "1e400",
                "--r2-strategy",
                "model",
            )
        )
        .resolve()
        .effective
    )
    assert config.process.loop_coupling_orders is None
    assert config.evaluation.arithmetic == "double-double"
    assert config.evaluation.mu_r2 == "1e400"
    assert config.loop.r2_strategy is R2Strategy.MODEL
    config = (
        parse_cli(("evaluate", "--mu-r2-file", str(tmp_path / "scales.json")))
        .resolve()
        .effective
    )
    assert config.evaluation.mu_r2_file == tmp_path / "scales.json"
    with pytest.raises(ConfigurationError, match="mutually exclusive"):
        EvaluationConfig(mu_r2="1", mu_r2_file=tmp_path / "scales.json")
    for arithmetic in ("quad", "float", True):
        with pytest.raises(ConfigurationError):
            EvaluationConfig(arithmetic=arithmetic)
    card = tmp_path / "virtual.toml"
    card.write_text(
        'schema_version = 1\naction = "evaluate"\n[evaluation]\n'
        'artifact = "artifact"\narithmetic = "arbitrary"\nprecision = 70\n'
        'mu_r2_file = "scales.json"\n[loop]\nr2_strategy = "model"\n'
    )
    from_card = parse_cli((str(card),)).resolve().effective
    assert from_card.evaluation.arithmetic == "arbitrary"
    assert from_card.evaluation.precision == 70
    assert from_card.evaluation.mu_r2_file == tmp_path / "scales.json"
    assert from_card.loop.r2_strategy is R2Strategy.MODEL


def test_virtual_exact_json_and_load_defaults_precede_binding(tmp_path, monkeypatch):
    from pyamplicol.api import Runtime

    exact = "0.1180000000000000000000000000000000000000001"
    parameters = tmp_path / "parameters.json"
    momenta = tmp_path / "momenta.json"
    scales = tmp_path / "scales.json"
    parameters.write_text('{"aS":' + exact + ',"mass":[1e400,"1e-80"]}')
    momenta.write_text("[[[" + exact + ",0,0,1]]]")
    scales.write_text("[1e400]")
    calls = {}

    class FakeRuntime:
        def evaluate(self, points, **options):
            calls["points"], calls["evaluate"] = points, options
            return _result()

    def load(path, **options):
        calls["load"] = options
        assert options["arithmetic"] == "arbitrary"
        assert options["precision"] == 70
        assert options["model_parameters"]["aS"] == Decimal(exact)
        assert options["model_parameters"]["mass"] == [Decimal("1e400"), "1e-80"]
        return FakeRuntime()

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
                str(momenta),
                "--model-parameters",
                str(parameters),
                "--arithmetic",
                "arbitrary",
                "--precision",
                "70",
                "--mu-r2-file",
                str(scales),
            )
        )
        .resolve()
        .effective
    )
    result = DefaultCliServices().evaluate(config, CallbackProgressSink(lambda _: None))
    assert result == _result()
    assert calls["points"][0][0][0] == Decimal(exact)
    assert calls["evaluate"]["mu_r2"] == [Decimal("1e400")]
    assert _read_json(momenta)[0][0][0] == float(exact)
    assert isinstance(_read_json(momenta)[0][0][0], float)


def test_tree_cli_keeps_legacy_load_and_evaluation_defaults(tmp_path, monkeypatch):
    from pyamplicol.api import Runtime

    path = tmp_path / "momenta.json"
    path.write_text("[[[1.25,0,0,1]]]")
    calls = {}

    class FakeRuntime:
        def evaluate(self, points, **options):
            assert isinstance(points[0][0][0], float)
            calls["evaluate"] = options
            return (1.0,)

    def load(path, **options):
        calls["load"] = options
        return FakeRuntime()

    monkeypatch.setattr(Runtime, "load", load)
    monkeypatch.setattr(
        "pyamplicol.cli.handlers._virtual_artifact_hint", lambda _: False
    )
    config = (
        parse_cli(("evaluate", "artifact", "--momenta", str(path), "--precision", "40"))
        .resolve()
        .effective
    )
    assert DefaultCliServices().evaluate(
        config, CallbackProgressSink(lambda _: None)
    ) == (1.0,)
    assert set(calls["load"]) == {"process", "model_parameters"}
    assert calls["evaluate"]["precision"] == 40
    assert set(calls["evaluate"]) == {"precision", "helicities", "color_flows"}


def test_virtual_scale_errors_and_nonfinite_json_are_explicit(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "pyamplicol.cli.handlers._virtual_artifact_hint", lambda _: True
    )
    momenta = tmp_path / "momenta.json"
    momenta.write_text("[]")
    config = (
        parse_cli(("evaluate", "artifact", "--momenta", str(momenta)))
        .resolve()
        .effective
    )
    with pytest.raises(ConfigurationError, match="requires --mu-r2"):
        DefaultCliServices().evaluate(config, CallbackProgressSink(lambda _: None))
    invalid = tmp_path / "invalid.json"
    invalid.write_text("[NaN]")
    with pytest.raises(ConfigurationError, match="nonfinite"):
        _read_json(invalid, exact=True)
    scales = tmp_path / "scales.json"
    scales.write_text('{"scale":1}')
    config = (
        parse_cli(
            (
                "evaluate",
                "artifact",
                "--momenta",
                str(momenta),
                "--mu-r2-file",
                str(scales),
            )
        )
        .resolve()
        .effective
    )
    with pytest.raises(ConfigurationError, match="per-point scale list"):
        DefaultCliServices().evaluate(config, CallbackProgressSink(lambda _: None))


@pytest.mark.parametrize("correlated", [False, True])
def test_cli_virtual_output_preserves_every_order_pole_and_decimal(correlated):
    result = _result(correlated=correlated)
    with localcontext() as context:
        context.prec = 6
        human, machine = io.StringIO(), io.StringIO()
        write_result(result, format="human", stream=human)
        write_result(result, format="json", stream=machine)
    payload = json.loads(machine.getvalue())
    assert payload["epsilon_powers"] == [0, -1, -2]
    assert payload["coefficient_names"] == ["finite", "single_pole", "double_pole"]
    assert payload["shape"] == [1, 2, 3]
    expected = ["-4", "5", "-6"]
    assert payload["values"][0][1] == (
        [{"real": item, "imag": str(Decimal(item).copy_negate())} for item in expected]
        if correlated
        else expected
    )
    assert len(payload["sectors"][1]["pairs"]) == 2
    assert payload["diagnostics"][0]["accuracy_established"] is False
    assert "1.234567890123456789012345678901234567890123456789" in human.getvalue()
    assert "single pole [-1]" in human.getvalue()
    assert "double pole [-2]" in human.getvalue()
    assert "QCD^4, QED^2" in human.getvalue()
    assert "requested=auto; effective=model" in human.getvalue()
    assert "accuracy established" in human.getvalue()
