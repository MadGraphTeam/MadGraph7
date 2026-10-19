# SPDX-License-Identifier: 0BSD
"""Source colour data roundtrips through ordinary raw and compiled model JSON."""

import json
from dataclasses import replace
from types import ModuleType

import pytest

from pyamplicol.models.loop_color_declaration import (
    LOOP_COLOR_DECLARATION_KEY,
    LoopColorDeclaration,
    WholeRuleColorRecipe,
    resolve_loop_color_declaration,
    with_loop_color_declaration,
)
from pyamplicol.models.loop_import import RawLoopModel, capture_raw_loop_model

from .test_loop_model_declaration import _scheme


def declaration():
    return LoopColorDeclaration(
        scheme=replace(_scheme(), dimensional_regularization="HV"),
        formal_symbol="Nc",
        parameter_identities={"CA": "Nc", "CF": "(Nc**2-1)/(2*Nc)"},
        kernel_keys=(("vertices", 7, 0, 0, None),),
        field_keys=((4, "QCD", 2),),
        scheme_parameter_values={"lhv": "1.000000000000000000000000000000"},
        whole_rules=(
            WholeRuleColorRecipe(
                "counterterm_vertices", 8, 1, (("QCD", 4),), "G", "fundamental-dirac"
            ),
        ),
    )


def test_source_attachment_preserves_other_metadata_and_exact_recipe():
    raw = capture_raw_loop_model(ModuleType("colour_data"))
    raw = replace(raw, metadata={**raw.metadata, "provider_note": "retained"})
    assert resolve_loop_color_declaration(raw) is None
    declared = with_loop_color_declaration(raw, declaration())
    restored = RawLoopModel.from_dict(json.loads(json.dumps(declared.to_dict())))
    assert resolve_loop_color_declaration(restored) == declaration()
    assert restored.metadata["provider_note"] == "retained"
    assert restored.collections == raw.collections
    assert restored.capabilities == raw.capabilities
    assert LOOP_COLOR_DECLARATION_KEY not in raw.metadata
    assert declaration().whole_rules[0].key == (
        "counterterm_vertices",
        8,
        1,
        (("QCD", 4),),
    )
    with pytest.raises(TypeError):
        declaration().parameter_identities["CA"] = "3"


def test_provider_tagged_mapping_uses_upstream_serializer():
    module = ModuleType("provider_colour_data")
    setattr(module, LOOP_COLOR_DECLARATION_KEY, declaration().to_dict())
    raw = capture_raw_loop_model(module)
    restored = RawLoopModel.from_dict(json.loads(json.dumps(raw.to_dict())))
    assert resolve_loop_color_declaration(restored) == declaration()


def test_compiled_bundle_invalidates_only_cached_serialization(tmp_path):
    from pyamplicol.models.contracts import CompiledModelIR
    from pyamplicol.models.loading import CompiledModel

    ir = CompiledModelIR(
        name="colour_data",
        orders=(),
        parameters=(),
        particles=(),
        couplings=(),
        propagators=(),
        vertex_terms=(),
        oriented_kernels=(),
        direct_contractions=(),
        closure_contractions=(),
    )
    compiled = CompiledModel(
        source={},
        producer={},
        model={"name": "colour_data"},
        ir=ir,
        parameter_defaults={},
        capabilities={},
        issues=(),
        phase_timings={},
        conversion_seconds=0.0,
        raw_loop_model=capture_raw_loop_model(ModuleType("colour_data")),
    )
    cached = compiled.write(tmp_path / "original.json")
    compiled = replace(compiled, _serialized_path=cached)
    changed = with_loop_color_declaration(compiled, declaration())
    assert changed._serialized_path is changed._prepared_bundle is None
    assert changed.ir is compiled.ir and changed.model is compiled.model
    restored = CompiledModel.from_dict(
        json.loads(changed.write(tmp_path / "declared.json").read_text()),
        validate_fingerprint=False,
    )
    assert resolve_loop_color_declaration(restored.raw_loop_model) == declaration()
    assert resolve_loop_color_declaration(compiled.raw_loop_model) is None


@pytest.mark.parametrize(
    "change",
    [
        {"formal_symbol": "not a scalar"},
        {"kernel_keys": [["vertices", True, 0, 0, None]]},
        {"kernel_keys": [["vertices", 1, 0, 0, None]] * 2},
        {"field_keys": [[0, "QCD", -1]]},
        {"parameter_identities": {"CA": 3}},
        {"scheme_parameter_values": {"lhv": 1.0}},
        {"whole_rules": [{"kind": "guessed-numeric-Nc"}]},
        {"certified": True},
    ],
)
def test_invalid_declaration_is_not_silently_ignored(change):
    with pytest.raises((ValueError, TypeError)):
        LoopColorDeclaration.from_dict(declaration().to_dict() | change)


def test_whole_recipe_never_guesses_a_group_or_correction_order():
    value = declaration().whole_rules[0].to_dict()
    with pytest.raises(ValueError, match=r"duplicate.*order"):
        WholeRuleColorRecipe.from_dict(value | {"order": [["QCD", 2], ["QCD", 2]]})
    with pytest.raises(ValueError, match="recipe"):
        WholeRuleColorRecipe.from_dict(value | {"loop_particles": [[1], [1]]})
    # Actual species and their multiplicities belong to the original source
    # group, rather than a parallel recipe species list that could diverge.
    assert "loop_particles" not in value
