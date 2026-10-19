# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import tomllib

import pytest

from pyamplicol.cli.parser import parse_cli
from pyamplicol.config import (
    ConfigurationError,
    EvaluatorConfig,
    EvaluatorExecutionMode,
    LoopConfig,
    ProcessConfig,
    R2Strategy,
    RunConfig,
    config_to_toml,
    resolve_config,
)


def test_auto_preserves_intent_and_resolves_contribution() -> None:
    tree = resolve_config(action="generate")
    virtual = resolve_config(
        {"process": {"loop_coupling_orders": ["QED", "QCD", "QCD"]}},
        action="generate",
    )
    assert tree.requested.evaluator.execution_mode is EvaluatorExecutionMode.AUTO
    assert tree.effective.evaluator.execution_mode is EvaluatorExecutionMode.RECURRENCE
    assert virtual.requested.evaluator.execution_mode is EvaluatorExecutionMode.AUTO
    assert (
        virtual.effective.evaluator.execution_mode
        is EvaluatorExecutionMode.LOOP_RECURRENCE
    )
    assert virtual.effective.process.loop_coupling_orders == ("QCD", "QED")
    assert virtual.effective.loop.r2_strategy is R2Strategy.AUTO
    assert resolve_config(tomllib.loads(config_to_toml(virtual.requested))) == virtual


@pytest.mark.parametrize("orders", [[], [""], [" "], "QCD", [1]])
def test_empty_or_malformed_correction_selection_is_not_tree(orders: object) -> None:
    with pytest.raises(ConfigurationError, match="loop_coupling_orders"):
        ProcessConfig(loop_coupling_orders=orders)


@pytest.mark.parametrize("mode", ["compiled", "eager", "recurrence", "on-the-fly"])
def test_virtual_requests_cannot_enter_tree_engines(mode: str) -> None:
    with pytest.raises(ConfigurationError, match="loop_coupling_orders requires"):
        RunConfig(
            action="generate",
            process=ProcessConfig(loop_coupling_orders=("QCD",)),
            evaluator=EvaluatorConfig(execution_mode=mode),
        )


def test_loop_engine_needs_corrections_and_r2_cannot_be_disabled() -> None:
    with pytest.raises(
        ConfigurationError, match=r"requires process\.loop_coupling_orders"
    ):
        RunConfig(
            action="generate",
            evaluator=EvaluatorConfig(execution_mode="loop_recurrence"),
        )
    with pytest.raises(ConfigurationError, match=r"loop\.r2_strategy"):
        LoopConfig(r2_strategy="off")


def test_loop_cli_and_toml_have_identical_semantics() -> None:
    cli = parse_cli(
        (
            "generate",
            "--loop-coupling-order",
            "QCD",
            "--loop-coupling-order",
            "QED",
            "--r2-strategy",
            "model",
        )
    ).resolve()
    card = resolve_config(
        {
            "action": "generate",
            "process": {"loop_coupling_orders": ["QCD", "QED"]},
            "loop": {"r2_strategy": "model"},
        }
    )
    assert cli == card


def test_unsupported_loop_request_does_not_silently_generate_tree(tmp_path) -> None:
    from pyamplicol.api import GenerationError, ProcessRequest, ProcessSet
    from pyamplicol.generation.service import GenerationBackend

    backend = GenerationBackend(
        RunConfig(
            action="generate", process=ProcessConfig(loop_coupling_orders=("QCD",))
        ),
        None,
    )
    processes = ProcessSet((ProcessRequest.parse("u u~ > d d~"),))
    plan = backend.plan(processes)
    assert plan.estimated_coverage["contribution_kind"] == "virtual"
    assert any(
        "color.accuracy='full'" in reason for reason in plan.unsupported_features
    )
    assert any(
        "emit_api_bundle=false" in reason for reason in plan.unsupported_features
    )
    with pytest.raises(GenerationError, match="public loop generation"):
        backend.generate(processes, tmp_path / "unsupported")
    assert not (tmp_path / "unsupported").exists()
