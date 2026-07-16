# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

import pyamplicol
from pyamplicol._internal.physics.symbols import symbols
from pyamplicol.models import compiler_symbolica as _sym
from pyamplicol.models.contracts import CompiledCouplingRecord, CompiledModelIR
from pyamplicol.models.loading import (
    ModelCompileOptions,
    _load_external_model,
    _loader_restriction_name,
    _model_compiler_source_paths,
    _source_digest,
)


class _SerializedModel:
    def to_json(self, _look: object) -> str:
        return '{"name":"synthetic"}'


def test_model_loading_sanitizes_historical_scalar_environment(
    monkeypatch,
) -> None:
    from ufo_model_loader import commands

    monkeypatch.setenv("UFO_SCALARS_MODEL_N_SCALARS", "91")
    monkeypatch.setenv("UFO_GRAVITY_MODEL_N_POINT_INTERACTIONS", "3,4")
    previous_dont_write_bytecode = sys.dont_write_bytecode
    seen: dict[str, str | bool | None] = {}

    def fake_load_model(*_args, **_kwargs):
        seen["scalars"] = os.environ.get("UFO_SCALARS_MODEL_N_SCALARS")
        seen["gravity"] = os.environ.get("UFO_GRAVITY_MODEL_N_POINT_INTERACTIONS")
        seen["dont_write_bytecode"] = sys.dont_write_bytecode
        return _SerializedModel(), {"lam": 1.0 + 2.0j}

    monkeypatch.setattr(commands, "load_model", fake_load_model)
    payload = _load_external_model(
        Path("synthetic.json"),
        options=ModelCompileOptions(),
    )

    assert seen == {
        "scalars": None,
        "gravity": None,
        "dont_write_bytecode": True,
    }
    assert sys.dont_write_bytecode is previous_dont_write_bytecode
    assert os.environ["UFO_SCALARS_MODEL_N_SCALARS"] == "91"
    assert os.environ["UFO_GRAVITY_MODEL_N_POINT_INTERACTIONS"] == "3,4"
    assert payload["parameter_card"] == {"lam": [1.0, 2.0]}


def test_runtime_domain_symbols_do_not_redefine_symbolica_attributes() -> None:
    _sym._ensure_symbolica()
    complex_symbol = symbols.runtime_domain(
        "synthetic",
        "shared_parameter",
        "complex",
    )
    real_symbol = symbols.runtime_domain(
        "synthetic",
        "shared_parameter",
        "real",
    )
    imaginary_symbol = symbols.runtime_domain(
        "synthetic",
        "shared_parameter",
        "imaginary",
    )

    assert complex_symbol != real_symbol
    assert complex_symbol != imaginary_symbol
    assert real_symbol.is_real()
    assert (-1j * imaginary_symbol).is_real()


def test_model_symbol_registry_preserves_identity_and_is_collision_resistant() -> None:
    _sym._ensure_symbolica()
    model_symbols = symbols.model("sm")

    assert model_symbols.namespace == "model_sm"
    assert model_symbols.symbol("MZ") == model_symbols.expression("UFO::MZ")
    assert model_symbols.symbol("MZ") != model_symbols.symbol("mz")
    assert "UFO::" not in model_symbols.expression_string("UFO::MZ + UFO::aS")
    assert symbols.model("SM-v1").namespace != symbols.model("SM v1").namespace


def test_runtime_model_parameter_uses_the_declared_symbol_owner() -> None:
    _sym._ensure_symbolica()

    assert symbols.runtime_model_parameter("sm", "MZ") == symbols.model("sm").symbol(
        "MZ"
    )
    assert symbols.runtime_model_parameter(
        "sm", "derived_coupling_81"
    ) == symbols.derived_coupling("sm", 81)


def test_compiled_model_rejects_process_global_ufo_scalar_symbols() -> None:
    with pytest.raises(ValueError, match="process-global UFO symbol"):
        CompiledModelIR(
            name="synthetic",
            orders=(),
            parameters=(),
            particles=(),
            couplings=(
                CompiledCouplingRecord(
                    name="GC_1",
                    expression="UFO::aS",
                    resolved_expression="UFO::aS",
                    value=(1.0, 0.0),
                    orders=(),
                ),
            ),
            propagators=(),
            vertex_terms=(),
            oriented_kernels=(),
        )


def test_model_cache_inputs_are_content_based(tmp_path: Path) -> None:
    left = tmp_path / "left" / "model"
    right = tmp_path / "right" / "model"
    left.mkdir(parents=True)
    right.mkdir(parents=True)
    for root in (left, right):
        (root / "__init__.py").write_text("VALUE = 1\n", encoding="utf-8")
        (root / "particles.py").write_text("PARTICLES = []\n", encoding="utf-8")

    assert _source_digest("ufo", left) == _source_digest("ufo", right)

    first = tmp_path / "first restriction.dat"
    second = tmp_path / "second restriction.dat"
    first.write_text("BLOCK MASS\n", encoding="utf-8")
    second.write_text("BLOCK MASS\n", encoding="utf-8")
    first_payload = ModelCompileOptions(restriction=str(first)).canonical_payload()
    second_payload = ModelCompileOptions(restriction=str(second)).canonical_payload()
    assert first_payload == second_payload
    assert str(tmp_path) not in repr(first_payload)


def test_restriction_paths_are_adapted_to_loader_names(tmp_path: Path) -> None:
    ufo = tmp_path / "ufo"
    ufo.mkdir()
    ufo_restriction = ufo / "restrict_no_widths.dat"
    ufo_restriction.write_text("BLOCK DECAY\n", encoding="ascii")
    serialized = tmp_path / "sm.json"
    serialized.write_text("{}", encoding="ascii")
    json_restriction = tmp_path / "restrict_no_widths.json"
    json_restriction.write_text("{}", encoding="ascii")

    assert _loader_restriction_name(ufo, "default") is None
    assert _loader_restriction_name(ufo, "none") == "full"
    assert _loader_restriction_name(ufo, str(ufo_restriction)) == "no_widths"
    assert _loader_restriction_name(serialized, str(json_restriction)) == "no_widths"


def test_restriction_paths_must_follow_loader_layout(tmp_path: Path) -> None:
    ufo = tmp_path / "ufo"
    ufo.mkdir()
    misplaced = tmp_path / "restrict_default.dat"
    misplaced.write_text("BLOCK MASS\n", encoding="ascii")
    malformed = ufo / "default.dat"
    malformed.write_text("BLOCK MASS\n", encoding="ascii")

    with pytest.raises(ValueError, match="stored next to"):
        _loader_restriction_name(ufo, str(misplaced))
    with pytest.raises(ValueError, match="must be named"):
        _loader_restriction_name(ufo, str(malformed))


def test_model_compiler_fingerprint_covers_all_lowering_sources() -> None:
    package_root = Path(pyamplicol.__file__).resolve().parent
    source_paths = _model_compiler_source_paths()
    relative_paths = {
        path.relative_to(package_root).as_posix() for path in source_paths
    }

    model_sources = {
        f"models/{path.name}" for path in (package_root / "models").glob("*.py")
    }
    assert model_sources <= relative_paths
    assert {
        "_internal/physics/parameters.py",
        "_internal/physics/symbols.py",
        "_internal/physics/types.py",
        "processes/core_syntax.py",
    } <= relative_paths
