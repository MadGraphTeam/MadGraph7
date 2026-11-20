# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import json
import tomllib
from dataclasses import asdict, replace
from pathlib import Path
from types import ModuleType

import pytest

from pyamplicol.config import (
    ConfigurationError,
    LoopModelConfig,
    LoopSchemeConfig,
    config_to_toml,
    resolve_config,
)
from pyamplicol.generation.loop_artifact import (
    LoopSchemeDeclaration as ArtifactSchemeDeclaration,
)
from pyamplicol.models.contracts import CompiledModelIR
from pyamplicol.models.loading import CompiledModel
from pyamplicol.models.loop_declaration import (
    LOOP_DECLARATION_KEY,
    LoopModelDeclaration,
    LoopSchemeDeclaration,
    resolve_loop_model_declaration,
    with_loop_model_declaration,
)
from pyamplicol.models.loop_import import RawLoopModel, capture_raw_loop_model

pytestmark = pytest.mark.loop_fast


def _scheme() -> LoopSchemeDeclaration:
    return LoopSchemeDeclaration(
        dimensional_regularization="explicit test spin prescription",
        external_states="four dimensional",
        gamma5="explicit test gamma-five prescription",
        renormalization="explicit test renormalization",
        mass_scheme="real masses",
        finite_restoration="none required by this test declaration",
        model_convention="test only; no physics validation claim",
    )


def _declaration() -> LoopModelDeclaration:
    return LoopModelDeclaration(_scheme(), ("QED", "QCD", "QCD"), True, True)


def _raw() -> RawLoopModel:
    return capture_raw_loop_model(ModuleType("declaration_test"))


def _config(declaration: LoopModelDeclaration) -> LoopModelConfig:
    return LoopModelConfig(
        scheme=LoopSchemeConfig(**asdict(declaration.scheme)),
        loop_coupling_orders=declaration.loop_coupling_orders,
        uv_complete=declaration.uv_complete,
        model_r2_complete=declaration.model_r2_complete,
    )


def test_joint_declaration_covers_subsets_without_manufacturing_joint_coverage() -> (
    None
):
    declared = with_loop_model_declaration(_raw(), _declaration())
    for selection in (("QCD",), ("QED",), ("QCD", "QED")):
        result = resolve_loop_model_declaration(
            declared, selected_corrections=selection
        )
        assert result.loop_coupling_orders == ("QCD", "QED")
        assert result.status == "declared-not-certified"
        result.require_declared_completion()
    qcd_only = replace(_declaration(), loop_coupling_orders=("QCD",))
    declared = with_loop_model_declaration(_raw(), qcd_only)
    with pytest.raises(ValueError, match="does not cover selected corrections: QED"):
        resolve_loop_model_declaration(declared, selected_corrections=("QCD", "QED"))


def test_config_roundtrip_and_explicit_whole_override() -> None:
    declaration = _declaration()
    configuration = {
        "action": "generate",
        "process": {"loop_coupling_orders": ["QCD", "QED"]},
        "loop": {
            "model": {
                "scheme": asdict(declaration.scheme),
                "loop_coupling_orders": list(declaration.loop_coupling_orders),
                "uv_complete": True,
                "model_r2_complete": True,
            }
        },
    }
    resolved = resolve_config(configuration)
    assert resolve_config(tomllib.loads(config_to_toml(resolved.requested))) == resolved
    incomplete_source = replace(declaration, uv_complete=False, model_r2_complete=False)
    raw = with_loop_model_declaration(_raw(), incomplete_source)
    chosen = resolve_loop_model_declaration(
        raw, resolved.effective.loop.model, selected_corrections=("QCD", "QED")
    )
    assert chosen == declaration
    assert (
        resolve_loop_model_declaration(
            raw, LoopModelConfig(), selected_corrections=("QCD",)
        )
        == incomplete_source
    )
    with pytest.raises(ConfigurationError, match="partial declarations"):
        LoopModelConfig(uv_complete=True)
    with pytest.raises(ConfigurationError, match="scheme requires explicit"):
        LoopModelConfig(
            loop_coupling_orders=("QCD",), uv_complete=True, model_r2_complete=True
        )


def test_no_declaration_is_inferred_and_completeness_is_not_certification() -> None:
    raw = _raw()
    assert LoopModelConfig().to_declaration() is None
    with pytest.raises(ValueError, match="no explicit scheme/completeness declaration"):
        resolve_loop_model_declaration(raw, selected_corrections=("QCD",))
    incomplete = replace(_declaration(), uv_complete=False, model_r2_complete=False)
    with pytest.raises(
        ValueError, match="complete UV counterterms and complete model R2"
    ):
        incomplete.require_declared_completion()
    with pytest.raises(ValueError, match="complete UV counterterms"):
        incomplete.require_declared_completion(strategy="direct")
    assert (
        replace(incomplete, uv_complete=True).require_declared_completion(
            strategy="direct"
        )
        is None
    )
    invalid = _declaration().to_dict() | {"status": "certified"}
    with pytest.raises(ValueError, match="cannot claim physics certification"):
        LoopModelDeclaration.from_dict(invalid)
    declared = with_loop_model_declaration(raw, _declaration())
    assert declared.capabilities == raw.capabilities
    assert declared.capabilities["uv_complete"] is None
    assert declared.capabilities["model_r2_complete"] is None
    assert ArtifactSchemeDeclaration is LoopSchemeDeclaration


def test_provider_metadata_captures_plain_declaration_without_new_sidecar() -> None:
    module = ModuleType("provider_test")
    setattr(module, LOOP_DECLARATION_KEY, _declaration().to_dict())
    raw = capture_raw_loop_model(module)
    restored = RawLoopModel.from_dict(json.loads(json.dumps(raw.to_dict())))
    assert (
        resolve_loop_model_declaration(restored, selected_corrections=("QCD",))
        == _declaration()
    )


def test_attached_declaration_replaces_cached_model_serialization(
    tmp_path: Path,
) -> None:
    ir = CompiledModelIR(
        name="test",
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
        model={"name": "test"},
        ir=ir,
        parameter_defaults={},
        capabilities={},
        issues=(),
        phase_timings={},
        conversion_seconds=0.0,
        raw_loop_model=_raw(),
    )
    original_path = compiled.write(tmp_path / "before.json")
    compiled = replace(compiled, _serialized_path=original_path)
    declared = with_loop_model_declaration(compiled, _declaration())
    assert declared.raw_loop_model is not compiled.raw_loop_model
    assert declared._serialized_path is None
    restored = CompiledModel.from_dict(
        json.loads(declared.write(tmp_path / "after.json").read_text()),
        validate_fingerprint=False,
    )
    assert (
        resolve_loop_model_declaration(
            restored.raw_loop_model, selected_corrections=("QCD",)
        )
        == _declaration()
    )
    assert LOOP_DECLARATION_KEY not in compiled.raw_loop_model.metadata


@pytest.mark.parametrize(
    "change",
    [
        {"uv_complete": 1},
        {"model_r2_complete": "yes"},
        {"loop_coupling_orders": []},
        {"loop_coupling_orders": "QCD"},
        {"loop_coupling_orders": [" "]},
    ],
)
def test_malformed_source_declarations_are_rejected(change: dict[str, object]) -> None:
    with pytest.raises((TypeError, ValueError)):
        LoopModelDeclaration.from_dict(_declaration().to_dict() | change)
