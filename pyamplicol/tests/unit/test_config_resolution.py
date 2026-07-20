# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from pyamplicol.config import (
    ClampRequest,
    ConfigurationError,
    EvaluatorExecutionMode,
    ProcessEntry,
    config_to_dict,
    config_to_toml,
    load_config,
    parse_override,
    resolution_to_dict,
    resolve_config,
)


def test_resolution_precedence_paths_and_clamp_record(tmp_path: Path) -> None:
    card = tmp_path / "run.toml"
    card.write_text(
        """
schema_version = 1
action = "generate"

[model]
source = "model.json"

[process]
entries = [{ expression = "d d~ > z g", name = "ddbar_zg" }]

[generation]
output = "card-output"
workers = 6
""".strip(),
        encoding="utf-8",
    )

    resolution = resolve_config(
        card,
        dedicated={"generation.workers": 4, "generation.output": "cli-output"},
        overrides=("generation.workers=3", "generation.workers=2"),
        clamps=(
            ClampRequest(
                "generation.workers",
                1,
                "unlicensed Symbolica core limit",
            ),
        ),
    )

    assert resolution.requested.generation.workers == 2
    assert resolution.effective.generation.workers == 1
    assert resolution.requested.generation.output == (tmp_path / "cli-output").resolve()
    assert resolution.requested.model.source == str((tmp_path / "model.json").resolve())
    assert resolution.was_clamped
    assert resolution.clamps[0].requested == 2
    assert resolution.clamps[0].effective == 1


def test_dynamic_map_override_is_schema_aware(tmp_path: Path) -> None:
    resolution = resolve_config(
        {
            "action": "generate",
            "process": {
                "entries": [{"expression": "u u~ > g g"}],
                "multiparticles": {"j": ["u", "d"]},
            },
        },
        base_dir=tmp_path,
        overrides=(
            'process.multiparticles.j=["u", "d", "g"]',
            "process.max_coupling_orders.QCD=2",
        ),
    )
    assert resolution.effective.process.multiparticles["j"] == ("u", "d", "g")
    assert resolution.effective.process.max_coupling_orders["QCD"] == 2


def test_selected_source_helicity_overrides_accept_signed_values() -> None:
    resolution = resolve_config(
        {"action": "generate"},
        overrides=(
            "process.selected_source_helicities.1=-1",
            "process.selected_source_helicities.2=1",
        ),
    )
    assert resolution.effective.process.selected_source_helicities == {"1": -1, "2": 1}


def test_nullable_and_bare_enum_overrides() -> None:
    assert parse_override("model.cache_dir=null").value is None
    assert parse_override("color.accuracy=nlc").value == "nlc"
    with pytest.raises(ConfigurationError, match="must be true or false"):
        parse_override("model.cache=1")


def test_eager_evaluator_card_and_dotted_overrides_round_trip() -> None:
    pytest.importorskip("tomli_w")
    config = resolve_config(
        {
            "action": "generate",
            "evaluator": {
                "execution_mode": "eager",
                "eager": {"point_tile_size": 2048, "workspace_mib": 384},
            },
        },
        overrides=("evaluator.eager.workspace_mib=512",),
    ).effective

    assert config.evaluator.execution_mode is EvaluatorExecutionMode.EAGER
    assert config.evaluator.eager.point_tile_size == 2048
    assert config.evaluator.eager.workspace_mib == 512
    plain = config_to_dict(config)
    assert plain["evaluator"]["eager"] == {  # type: ignore[index]
        "point_tile_size": 2048,
        "workspace_mib": 512,
    }
    assert resolve_config(plain).effective == config
    serialized = config_to_toml(config)
    assert 'execution_mode = "eager"' in serialized
    assert "[evaluator.eager]" in serialized
    assert resolve_config(tomllib.loads(serialized)).effective == config


@pytest.mark.parametrize(
    "override",
    (
        "evaluator.execution_mode=streaming",
        "evaluator.eager.point_tile_size=0",
        "evaluator.eager.workspace_mib=-1",
    ),
)
def test_invalid_eager_evaluator_overrides_are_rejected(override: str) -> None:
    with pytest.raises(ConfigurationError, match="evaluator"):
        resolve_config({"action": "generate"}, overrides=(override,))


def test_unknown_card_and_override_fields_are_errors(tmp_path: Path) -> None:
    card = tmp_path / "bad.toml"
    card.write_text('action = "generate"\n[generator]\nworkers = 2\n', encoding="utf-8")
    with pytest.raises(ConfigurationError, match="unknown configuration field"):
        load_config(card)
    with pytest.raises(ConfigurationError, match="optimization_level"):
        parse_override("evaluator.jit.optimisation_level=2")


def test_removed_parallel_process_fields_suggest_typed_entries() -> None:
    with pytest.raises(ConfigurationError, match=r"process\.entries"):
        resolve_config(
            {
                "action": "generate",
                "process": {"requests": ["d d~ > z"], "names": ["ddbar_z"]},
            }
        )


def test_process_entry_override_and_optional_name() -> None:
    config = resolve_config(
        {"action": "generate"},
        overrides=(
            'process.entries=[{ expression = "d d~ > z", name = "ddbar_z" }, '
            '{ expression = "u u~ > z" }]',
        ),
    ).effective
    assert config.process.entries == (
        ProcessEntry("d d~ > z", "ddbar_z"),
        ProcessEntry("u u~ > z"),
    )


def test_card_validation_aggregates_entry_field_and_unknown_path_errors() -> None:
    with pytest.raises(ConfigurationError) as exc_info:
        resolve_config(
            {
                "action": "generate",
                "process": {
                    "entries": [
                        {"expresion": "d d~ > z", "name": "1-invalid"},
                        7,
                    ]
                },
                "generation": {"workers": False},
                "evaluator": {"batch_size": "many"},
                "outpt": {"format": "json"},
            }
        )

    message = str(exc_info.value)
    assert "configuration errors" in message
    assert "process.entries[0].expresion" in message
    assert "process entry expression" in message
    assert "process entry name" in message
    assert "process.entries[1] must be a table" in message
    assert "generation.workers" in message
    assert "evaluator.batch_size" in message
    assert "output" in message


@pytest.mark.parametrize(
    "path",
    (
        "color.coverage",
        "color.flow_ids",
        "generation.validation.zero_current_filter",
        "generation.validation.current_merging",
    ),
)
def test_removed_generation_dotted_paths_are_errors(path: str) -> None:
    with pytest.raises(ConfigurationError, match="unknown configuration field"):
        parse_override(f"{path}=true")


def test_config_serialization_uses_plain_paths_and_lists(tmp_path: Path) -> None:
    config = resolve_config(
        {"action": "evaluate", "evaluation": {"artifact": "artifact"}},
        base_dir=tmp_path,
    ).effective
    plain = config_to_dict(config)
    assert plain["evaluation"]["artifact"] == str((tmp_path / "artifact").resolve())  # type: ignore[index]
    assert plain["benchmark"]["helicity_ids"] == []  # type: ignore[index]


def test_resolution_serialization_records_requested_effective_and_reason() -> None:
    resolution = resolve_config(
        {"action": "generate", "generation": {"workers": 4}},
        clamps=(ClampRequest("generation.workers", 1, "restricted mode"),),
    )
    plain = resolution_to_dict(resolution)
    assert plain["requested"]["generation"]["workers"] == 4  # type: ignore[index]
    assert plain["effective"]["generation"]["workers"] == 1  # type: ignore[index]
    assert plain["adjustments"] == [
        {
            "path": "generation.workers",
            "requested": 4,
            "effective": 1,
            "reason": "restricted mode",
        }
    ]


def test_config_toml_round_trip_omits_nulls_without_changing_defaults() -> None:
    pytest.importorskip("tomli_w")
    original = resolve_config(
        {
            "action": "generate",
            "process": {"entries": [{"expression": "d d~ > z"}]},
            "generation": {"workers": 2},
        }
    ).effective
    serialized = config_to_toml(original)
    assert "null" not in serialized
    restored = resolve_config(tomllib.loads(serialized)).effective
    assert restored == original
