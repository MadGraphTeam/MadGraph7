# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import sys
from pathlib import Path

import pytest

from pyamplicol.cli import build_parser, parse_cli
from pyamplicol.config import ConfigurationError, ProcessEntry


def test_direct_command_flags_then_ordered_set_overrides(tmp_path: Path) -> None:
    invocation = parse_cli(
        (
            "generate",
            "d d~ > z g",
            str(tmp_path / "artifact"),
            "--workers",
            "4",
            "--color-accuracy",
            "nlc",
            "--set",
            "generation.workers=2",
            "--set",
            "generation.workers=1",
        )
    )
    config = invocation.resolve().effective
    assert config.action == "generate"
    assert config.process.entries == (ProcessEntry("d d~ > z g"),)
    assert config.generation.output == (tmp_path / "artifact").resolve()
    assert config.generation.workers == 1
    assert config.color.accuracy == "nlc"


def test_card_can_be_invoked_as_the_first_argument(tmp_path: Path) -> None:
    card = tmp_path / "evaluate.toml"
    card.write_text(
        'action = "evaluate"\n[evaluation]\nartifact = "artifact"\n',
        encoding="utf-8",
    )
    invocation = parse_cli((str(card), "--set", "output.format=json"))
    config = invocation.resolve().effective
    assert invocation.action is None
    assert config.action == "evaluate"
    assert config.evaluation.artifact == (tmp_path / "artifact").resolve()
    assert config.output.format == "json"


def test_all_contract_actions_have_subcommands() -> None:
    arguments = {
        "generate": ("generate",),
        "evaluate": ("evaluate",),
        "benchmark": ("benchmark",),
        "inspect": ("inspect",),
        "model-inspect": ("model", "inspect"),
        "model-compile": ("model", "compile"),
        "model-processes": ("model", "processes"),
    }
    for action, argv in arguments.items():
        assert parse_cli(argv).action == action


@pytest.mark.parametrize(
    "command",
    ("model-inspect", "model-compile", "model-processes"),
)
def test_flat_model_commands_are_not_cli_compatibility_aliases(command: str) -> None:
    with pytest.raises(SystemExit):
        parse_cli((command,))


def test_model_processes_accepts_process_requests() -> None:
    invocation = parse_cli(
        (
            "model",
            "processes",
            "d d~ > z g",
            "--process",
            "u u~ > z g",
        )
    )
    config = invocation.resolve().effective
    assert config.action == "model-processes"
    assert config.process.entries == (
        ProcessEntry("d d~ > z g"),
        ProcessEntry("u u~ > z g"),
    )


def test_direct_process_names_are_assembled_into_typed_entries() -> None:
    config = (
        parse_cli(
            (
                "model",
                "processes",
                "d d~ > z g",
                "--process",
                "u u~ > z g",
                "--name",
                "ddbar_zg",
                "--name",
                "uubar_zg",
            )
        )
        .resolve()
        .effective
    )
    assert config.process.entries == (
        ProcessEntry("d d~ > z g", "ddbar_zg"),
        ProcessEntry("u u~ > z g", "uubar_zg"),
    )


def test_direct_process_names_must_align_with_expressions() -> None:
    with pytest.raises(ConfigurationError, match="aligned with process expressions"):
        parse_cli(
            (
                "model",
                "processes",
                "d d~ > z g",
                "--name",
                "one",
                "--name",
                "two",
            )
        )


def test_model_command_accepts_common_flags_before_the_command_group() -> None:
    invocation = parse_cli(("--format", "json", "model", "inspect"))
    assert invocation.resolve().effective.output.format == "json"


def test_top_level_help_lists_nested_model_and_utility_commands() -> None:
    help_text = build_parser().format_help()
    for command in (
        "model",
        "config",
        "examples",
        "doctor",
        "self-test",
        "request-symbolica-trial-license",
        "request-symbolica-hobbyist-license",
    ):
        assert command in help_text
    assert "model-inspect" not in help_text
    assert "model-compile" not in help_text
    assert "model-processes" not in help_text


def test_model_help_does_not_import_symbolica(
    capsys: pytest.CaptureFixture[str],
) -> None:
    sys.modules.pop("symbolica", None)
    with pytest.raises(SystemExit) as exit_info:
        build_parser().parse_args(("model", "--help"))
    assert exit_info.value.code == 0
    assert "{inspect,compile,processes}" in capsys.readouterr().out
    assert "symbolica" not in sys.modules


def test_global_output_flags_are_accepted_before_subcommand(tmp_path: Path) -> None:
    invocation = parse_cli(
        (
            "--format",
            "json",
            "generate",
            "u u~ > g g",
            str(tmp_path / "artifact"),
        )
    )
    assert invocation.resolve().effective.output.format == "json"


def test_set_order_is_preserved_across_subcommand_position(tmp_path: Path) -> None:
    invocation = parse_cli(
        (
            "--set",
            "generation.workers=5",
            "generate",
            "u u~ > g g",
            str(tmp_path / "artifact"),
            "--set",
            "generation.workers=2",
        )
    )
    assert invocation.overrides == (
        "generation.workers=5",
        "generation.workers=2",
    )
    assert invocation.resolve().effective.generation.workers == 2


@pytest.mark.parametrize(
    "arguments",
    (
        ("generate", "--color-coverage", "all"),
        ("generate", "--color-flow", "flow:2,4,1"),
        ("generate", "--zero-current-filter"),
        ("generate", "--no-zero-current-filter"),
        ("generate", "--current-merging"),
        ("generate", "--no-current-merging"),
    ),
)
def test_removed_generation_flags_are_rejected(arguments: tuple[str, ...]) -> None:
    with pytest.raises(SystemExit):
        parse_cli(arguments)


def test_runtime_color_flow_flags_remain_available() -> None:
    evaluation = parse_cli(("evaluate", "--color-flow", "flow:2,4,1"))
    benchmark = parse_cli(("benchmark", "--color-flow", "flow:2,4,1"))

    assert evaluation.resolve().effective.evaluation.color_flow_ids == ("flow:2,4,1",)
    assert benchmark.resolve().effective.benchmark.color_flow_ids == ("flow:2,4,1",)
