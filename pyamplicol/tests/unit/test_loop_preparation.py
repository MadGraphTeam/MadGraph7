# SPDX-License-Identifier: 0BSD
"""Exact source preparation data and presence-preserving configuration."""

from types import ModuleType

import pytest

from pyamplicol.config import (
    ConfigurationError,
    LoopPreparationConfig,
    config_to_dict,
    config_to_toml,
    resolve_config,
)
from pyamplicol.models.loop_import import capture_raw_loop_model
from pyamplicol.models.loop_preparation import (
    LOOP_PREPARATION_KEY,
    LoopPreparation,
    resolve_loop_preparation,
    with_loop_preparation,
)

pytestmark = pytest.mark.loop_fast


def test_exact_sidecar_is_copy_safe_and_conflicts_do_not_round():
    raw = capture_raw_loop_model(ModuleType("preparation_fixture"))
    source = {"24": [24, 251, 9000003, -9000004]}
    fixed = {"tiny": "1e-1000", "phase": ["1.00000000000000000000000001", "-2e-90"]}
    declaration = LoopPreparation("explicit-profile", source, fixed)
    derived = with_loop_preparation(raw, declaration)
    source["24"].append(77)
    fixed["tiny"] = "0"
    assert LOOP_PREPARATION_KEY not in raw.metadata
    assert derived.collections == raw.collections
    restored = resolve_loop_preparation(derived)
    assert restored == declaration
    assert restored.covariant_multiplets[24] == (24, 251, 9000003, -9000004)
    assert restored.fixed_parameters["tiny"] == ("1e-1000", "0")
    with pytest.raises(TypeError):
        restored.fixed_parameters["tiny"] = ("0", "0")
    assert restored.merged_fixed_parameters({"tiny": ("10e-1001", "0")})["tiny"] == (
        "10e-1001",
        "0",
    )
    with pytest.raises(ValueError, match="contradicts loop preparation"):
        restored.merged_fixed_parameters({"tiny": "0"})


def test_source_inheritance_and_explicit_record_replacement():
    original = with_loop_preparation(
        capture_raw_loop_model(ModuleType("preparation_fixture")),
        LoopPreparation("profile", {24: [24, 251]}, {"mass": "0"}),
    )
    assert resolve_loop_preparation(original).compatibility == "profile"
    assert (
        resolve_loop_preparation(original, LoopPreparationConfig()) == LoopPreparation()
    )
    replacement = resolve_loop_preparation(
        original, LoopPreparationConfig(fixed_parameters={"other": "1e-90"})
    )
    assert replacement.compatibility is None
    assert not replacement.covariant_multiplets
    assert replacement.fixed_parameters == {"other": ("1e-90", "0")}


@pytest.mark.parametrize("bad", [True, 24.0, "24.0", "x"])
def test_multiplet_keys_are_integer_identities(bad):
    with pytest.raises(TypeError, match="integer PDGs"):
        LoopPreparation(covariant_multiplets={bad: [24]})


def test_multiplet_aliases_and_member_types_are_not_silently_coerced():
    with pytest.raises(ValueError, match="duplicate"):
        LoopPreparation(covariant_multiplets={24: [24], "+24": [24]})
    for members in ([True], [24.0], ["24"], "24"):
        with pytest.raises(TypeError, match="members must be integer"):
            LoopPreparation(covariant_multiplets={24: members})


@pytest.mark.parametrize(
    "value", [1e-30, 1, True, "NaN", "Infinity", ["1"], ["0", "NaN"]]
)
def test_fixed_inputs_use_existing_exact_ingress(value):
    with pytest.raises((TypeError, ValueError)):
        LoopPreparation(fixed_parameters={"mass": value})


def test_optional_section_presence_survives_serialization(tmp_path):
    absent = resolve_config({"action": "generate"}).requested
    empty = resolve_config(
        {"action": "generate", "loop": {"preparation": {}}}
    ).requested
    assert absent.loop.preparation is None
    assert empty.loop.preparation == LoopPreparationConfig()
    assert config_to_dict(absent)["loop"]["preparation"] is None
    assert config_to_dict(empty)["loop"]["preparation"] is not None
    for index, config in enumerate((absent, empty)):
        card = tmp_path / f"card{index}.toml"
        card.write_text(config_to_toml(config))
        assert resolve_config(card).requested == config


def test_optional_section_overrides_and_leaf_activation():
    card = {
        "action": "generate",
        "loop": {
            "preparation": {
                "compatibility": "profile",
                "fixed_parameters": {"mass": "0"},
            }
        },
    }
    assert (
        resolve_config(
            card, overrides=["loop.preparation=null"]
        ).requested.loop.preparation
        is None
    )
    assert (
        resolve_config(
            card, overrides=["loop.preparation={}"]
        ).requested.loop.preparation
        == LoopPreparationConfig()
    )
    result = resolve_config(
        card, overrides=['loop.preparation={fixed_parameters={tiny="1e-1000"}}']
    ).requested.loop.preparation
    assert result.compatibility is None
    assert result.fixed_parameters == {"tiny": ("1e-1000", "0")}
    result = resolve_config(
        {"action": "generate"},
        overrides=[
            'loop.preparation.fixed_parameters.tiny="1e-1000"',
            "loop.preparation.covariant_multiplets.24=[24,251]",
        ],
    ).requested.loop.preparation
    assert result.fixed_parameters == {"tiny": ("1e-1000", "0")}
    assert result.covariant_multiplets == {"24": (24, 251)}
    result = resolve_config(
        card,
        overrides=["loop.preparation=null", 'loop.preparation.compatibility="new"'],
    ).requested.loop.preparation
    assert result == LoopPreparationConfig(compatibility="new")


def test_unknown_preparation_fields_and_inexact_config_values_reject():
    for payload in ({"unknown": "x"}, {"fixed_parameters": {"mass": 0.1}}):
        with pytest.raises(ConfigurationError):
            resolve_config({"action": "generate", "loop": {"preparation": payload}})


def test_other_configuration_maps_keep_existing_layering():
    result = resolve_config(
        {"action": "generate", "process": {"max_coupling_orders": {"QCD": 2}}},
        overrides=["process.max_coupling_orders.QED=4"],
    ).requested
    assert result.process.max_coupling_orders == {"QCD": 2, "QED": 4}
    assert result.loop.preparation is None
