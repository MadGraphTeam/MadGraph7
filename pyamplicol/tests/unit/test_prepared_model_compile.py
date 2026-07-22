# SPDX-License-Identifier: 0BSD
from __future__ import annotations

from pathlib import Path

import pytest

from pyamplicol.config import EvaluatorConfig, JITConfig
from pyamplicol.models.loading import compile_model_source
from pyamplicol.models.prepared import PreparedModelBundleError
from pyamplicol.models.prepared_catalog import (
    PREPARED_INDEPENDENT_BLOCK_PROOF,
    PreparedKernelCatalog,
    PreparedKernelInput,
    PreparedKernelSpec,
)
from pyamplicol.models.prepared_compile import (
    prepare_model_bundle,
    prepared_symbolica_settings,
)
from pyamplicol.models.recurrence_template import RecurrenceTemplateCatalog


class _FakeJitAdapter:
    def __init__(self, input_len: int = 1, output_len: int = 1) -> None:
        self.input_len = input_len
        self.output_len = output_len

    def artifact_manifest(self, artifact_dir: Path) -> dict[str, object]:
        payload_dir = artifact_dir / "evaluators"
        payload_dir.mkdir(parents=True, exist_ok=True)
        application = payload_dir / "random-application.symjit"
        state = payload_dir / "random-state.evaluator.bin"
        application.write_bytes(b"portable-symjit-application")
        state.write_bytes(b"exact-symbolica-state")
        return {
            "kind": "symjit-application-evaluator",
            "runtime_capability": "symjit.application.complex-f64.v1",
            "backend": "jit",
            "label": "prepared_test",
            "input_len": self.input_len,
            "output_len": self.output_len,
            "application_path": str(application.relative_to(artifact_dir)),
            "application_abi": "symjit-application-storage-v3",
            "element_layout": "complex-f64",
            "batch_layout": "row-major",
            "compiler_type": "native",
            "translation_mode": "indirect",
            "optimization_level": 3,
            "word_bits": 64,
            "endianness": "little",
            "required_defuns": [],
            "evaluator_state_path": str(state.relative_to(artifact_dir)),
            "evaluator_state_runtime_capability": (
                "symbolica.legacy-jit-container.complex-f64.v1"
            ),
            "settings": {"jit_optimization_level": 3},
            "build_timing": {},
        }


def _catalog() -> PreparedKernelCatalog:
    input_contract = PreparedKernelInput(
        role="current",
        component=0,
        symbol="pyamplicol::prepared_test_input",
    )
    return PreparedKernelCatalog(
        model_name="built-in-sm",
        kernels=(
            PreparedKernelSpec(
                kernel_id=0,
                contract_kind="propagator",
                canonical_signature="1" * 64,
                exact_expressions=("pyamplicol::prepared_test_input",),
                inputs=(input_contract,),
                output_layout=("scalar:c0",),
            ),
        ),
        vertex_bindings=(),
        propagator_bindings=(),
        closure_bindings=(),
        model_parameter_kernel_id=None,
    )


def _block_catalog() -> PreparedKernelCatalog:
    return PreparedKernelCatalog(
        model_name="built-in-sm",
        kernels=(
            PreparedKernelSpec(
                kernel_id=0,
                contract_kind="vertex",
                canonical_signature="2" * 64,
                exact_expressions=(
                    "pyamplicol::prepared_block_left+pyamplicol::prepared_block_right",
                ),
                inputs=(
                    PreparedKernelInput(
                        role="left-current",
                        component=0,
                        symbol="pyamplicol::prepared_block_left",
                    ),
                    PreparedKernelInput(
                        role="right-current",
                        component=0,
                        symbol="pyamplicol::prepared_block_right",
                    ),
                ),
                output_layout=("scalar:c0",),
                proof_classes=(PREPARED_INDEPENDENT_BLOCK_PROOF,),
            ),
        ),
        vertex_bindings=(),
        propagator_bindings=(),
        closure_bindings=(),
        model_parameter_kernel_id=None,
    )


@pytest.mark.parametrize("compress", (True, False))
def test_prepared_symbolica_settings_forward_jit_compression(
    compress: bool,
) -> None:
    settings = prepared_symbolica_settings(
        EvaluatorConfig(jit=JITConfig(compress=compress))
    )

    assert settings.jit_compress is compress
    assert settings.to_json_dict()["jit_compress"] is compress


def _native_recurrence_validation(*_args, **_kwargs) -> dict[str, object]:
    return {
        "kind": "pyamplicol-recurrence-template-validation-result",
        "template_input_sha256": "d" * 64,
    }


def _native_result(template_input, authenticated_ids) -> dict[str, object]:
    assert authenticated_ids == [0]
    return {
        "kind": "pyamplicol-recurrence-template-validation-result",
        "schema_version": 1,
        "validation_status": "validated",
        "template_input_abi": "pyamplicol-recurrence-template-input-v1",
        "template_input_schema_version": 1,
        "template_input_sha256": template_input.canonical_digest,
        "catalog_digest": template_input.catalog_digest,
        "compiled_model_digest": template_input.compiled_model_digest,
        "prepared_kernel_pack_digest": template_input.prepared_kernel_pack_digest,
        "prepared_kernel_inventory_verified": True,
        "prepared_kernel_inventory_count": 1,
        "counts": {
            "parameters": 0,
            "current_states": 0,
            "sources": 0,
            "quantum_flows": 0,
            "transitions": 0,
            "propagators": 0,
            "closures": 0,
            "color_contractions": 0,
            "symmetry_proofs": 0,
            "evaluator_bindings": 0,
            "prepared_kernels": 0,
            "referenced_prepared_kernels": 0,
        },
    }


def test_native_recurrence_validation_checks_complete_result_contract(
    monkeypatch,
) -> None:
    import pyamplicol.models.prepared_compile as prepared_compile

    catalog = RecurrenceTemplateCatalog.create(
        compiled_model_digest="a" * 64,
        prepared_kernel_pack_digest="b" * 64,
    )
    module = type("NativeModule", (), {})()
    module._validate_recurrence_template_input_v1 = _native_result
    monkeypatch.setattr(
        prepared_compile.importlib,
        "import_module",
        lambda _name: module,
    )

    result = prepared_compile._validate_native_recurrence_template_input_v1(
        catalog,
        (0,),
    )
    assert result["prepared_kernel_inventory_verified"] is True

    def wrong_result(template_input, authenticated_ids):
        result = _native_result(template_input, authenticated_ids)
        result["schema_version"] = True
        return result

    module._validate_recurrence_template_input_v1 = wrong_result
    with pytest.raises(PreparedModelBundleError, match="schema_version"):
        prepared_compile._validate_native_recurrence_template_input_v1(catalog, (0,))


def test_native_recurrence_validation_requires_matching_extension(
    monkeypatch,
) -> None:
    import pyamplicol.models.prepared_compile as prepared_compile

    catalog = RecurrenceTemplateCatalog.create(
        compiled_model_digest="a" * 64,
        prepared_kernel_pack_digest="b" * 64,
    )
    monkeypatch.setattr(
        prepared_compile.importlib,
        "import_module",
        lambda _name: (_ for _ in ()).throw(ImportError("missing")),
    )

    with pytest.raises(PreparedModelBundleError, match="matching installed"):
        prepared_compile._validate_native_recurrence_template_input_v1(catalog, ())


def test_recurrence_preflight_fails_before_backend_compilation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import pyamplicol.models.prepared_compile as prepared_compile

    compiled = compile_model_source("built-in-sm", use_cache=False)
    compiled_kernel = False

    def fail_recurrence(*_args, **_kwargs):
        raise ValueError("unsupported recurrence semantics")

    def observe_compile(*_args, **_kwargs):
        nonlocal compiled_kernel
        compiled_kernel = True
        raise AssertionError("backend compilation must not run")

    monkeypatch.setattr(
        prepared_compile,
        "build_prepared_kernel_catalog",
        lambda _: _catalog(),
    )
    monkeypatch.setattr(
        prepared_compile,
        "build_recurrence_template_catalog",
        fail_recurrence,
    )
    monkeypatch.setattr(prepared_compile, "_compile_kernel", observe_compile)

    with pytest.raises(ValueError, match="unsupported recurrence semantics"):
        prepare_model_bundle(
            compiled,
            tmp_path / "unsupported-recurrence",
            evaluator=EvaluatorConfig(),
        )
    assert compiled_kernel is False


def test_native_recurrence_preflight_fails_before_backend_compilation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import pyamplicol.models.prepared_compile as prepared_compile

    compiled = compile_model_source("built-in-sm", use_cache=False)
    compiled_kernel = False

    def observe_compile(*_args, **_kwargs):
        nonlocal compiled_kernel
        compiled_kernel = True
        raise AssertionError("backend compilation must not run")

    monkeypatch.setattr(
        prepared_compile,
        "build_prepared_kernel_catalog",
        lambda _: _catalog(),
    )
    monkeypatch.setattr(
        prepared_compile,
        "build_recurrence_template_catalog",
        lambda *_args, compiled_model_digest, prepared_kernel_pack_digest, **_kwargs: (
            RecurrenceTemplateCatalog.create(
                compiled_model_digest=compiled_model_digest,
                prepared_kernel_pack_digest=prepared_kernel_pack_digest,
            )
        ),
    )
    monkeypatch.setattr(
        prepared_compile,
        "_validate_native_recurrence_template_input_v1",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ValueError("native recurrence unavailable")
        ),
    )
    monkeypatch.setattr(prepared_compile, "_compile_kernel", observe_compile)

    with pytest.raises(ValueError, match="native recurrence unavailable"):
        prepare_model_bundle(
            compiled,
            tmp_path / "missing-native-recurrence",
            evaluator=EvaluatorConfig(),
        )
    assert compiled_kernel is False


def test_prepared_compiler_writes_structured_architecture_kernel_pack(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import pyamplicol.models.prepared_compile as prepared_compile

    monkeypatch.setattr(
        prepared_compile,
        "build_prepared_kernel_catalog",
        lambda _: _catalog(),
    )
    monkeypatch.setattr(
        prepared_compile,
        "build_recurrence_template_catalog",
        lambda *_args, compiled_model_digest, prepared_kernel_pack_digest, **_kwargs: (
            RecurrenceTemplateCatalog.create(
                compiled_model_digest=compiled_model_digest,
                prepared_kernel_pack_digest=prepared_kernel_pack_digest,
            )
        ),
    )
    monkeypatch.setattr(
        prepared_compile,
        "_compile_symbolica_outputs",
        lambda *_args, **_kwargs: _FakeJitAdapter(),
    )
    monkeypatch.setattr(
        prepared_compile,
        "_validate_native_recurrence_template_input_v1",
        _native_recurrence_validation,
    )
    compiled = compile_model_source("built-in-sm", use_cache=False)
    progress: list[tuple[str, int, int]] = []

    result = prepare_model_bundle(
        compiled,
        tmp_path / "prepared-built-in",
        evaluator=EvaluatorConfig(),
        progress=lambda label, completed, total: progress.append(
            (label, completed, total)
        ),
    )

    assert result.output.name.endswith(".pyamplicol-model")
    assert result.kernel_count == 1
    assert result.bundle.backend == "jit"
    assert result.bundle.kernel_pack.optimization_settings["jit_compress"] is True
    assert result.bundle.kernel_pack.target["portable"] is False
    assert str(result.bundle.kernel_pack.target["target_triple"]).startswith(
        "symjit-storage-v3-"
    )
    assert result.bundle.kernel_pack.resolver_manifest["model_name"] == "built-in-sm"
    kernel = result.bundle.kernel_pack.kernels[0]
    assert kernel.input_contracts[0]["role"] == "current"
    assert kernel.exact_expressions == ("pyamplicol::prepared_test_input",)
    assert kernel.exact_evaluator_state_path.startswith("kernels/000000/")
    assert all(
        path.startswith("kernels/000000/") for path in kernel.referenced_payload_paths
    )
    assert progress[-1] == ("prepared model complete", 1, 1)
    assert "recurrence_template_validation" in result.phase_timings_seconds


def test_prepared_compiler_emits_independent_block4_jit_variant(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import pyamplicol.models.prepared_compile as prepared_compile

    compiled_shapes: list[tuple[int, int]] = []

    def fake_compile(outputs, parameters, **_kwargs):
        compiled_shapes.append((len(parameters), len(outputs)))
        return _FakeJitAdapter(len(parameters), len(outputs))

    monkeypatch.setattr(
        prepared_compile,
        "build_prepared_kernel_catalog",
        lambda _: _block_catalog(),
    )
    monkeypatch.setattr(
        prepared_compile,
        "build_recurrence_template_catalog",
        lambda *_args, compiled_model_digest, prepared_kernel_pack_digest, **_kwargs: (
            RecurrenceTemplateCatalog.create(
                compiled_model_digest=compiled_model_digest,
                prepared_kernel_pack_digest=prepared_kernel_pack_digest,
            )
        ),
    )
    monkeypatch.setattr(
        prepared_compile,
        "_compile_symbolica_outputs",
        fake_compile,
    )
    monkeypatch.setattr(
        prepared_compile,
        "_validate_native_recurrence_template_input_v1",
        _native_recurrence_validation,
    )
    compiled = compile_model_source("built-in-sm", use_cache=False)

    result = prepare_model_bundle(
        compiled,
        tmp_path / "prepared-block",
        evaluator=EvaluatorConfig(),
    )

    assert compiled_shapes == [(2, 1), (8, 4)]
    (variant,) = result.bundle.kernel_pack.kernel_variants
    assert variant.variant_id == "independent-block-4"
    assert variant.input_lane_stride == 2
    assert variant.output_lane_stride == 1
    assert variant.input_layout == (
        "lane:0:left-current:0",
        "lane:0:right-current:0",
        "lane:1:left-current:0",
        "lane:1:right-current:0",
        "lane:2:left-current:0",
        "lane:2:right-current:0",
        "lane:3:left-current:0",
        "lane:3:right-current:0",
    )
    assert all(
        path.startswith("kernels/000000/variants/independent-block-4/")
        for path in variant.referenced_payload_paths
    )
