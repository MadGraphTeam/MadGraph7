# SPDX-License-Identifier: 0BSD
from __future__ import annotations

from pathlib import Path

from pyamplicol.config import EvaluatorConfig
from pyamplicol.models.loading import compile_model_source
from pyamplicol.models.prepared_catalog import (
    PreparedKernelCatalog,
    PreparedKernelInput,
    PreparedKernelSpec,
)
from pyamplicol.models.prepared_compile import prepare_model_bundle


class _FakeJitAdapter:
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
            "input_len": 1,
            "output_len": 1,
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


def test_prepared_compiler_writes_structured_portable_kernel_pack(
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
        "_compile_symbolica_outputs",
        lambda *_args, **_kwargs: _FakeJitAdapter(),
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
    assert result.bundle.kernel_pack.target["portable"] is True
    assert result.bundle.kernel_pack.resolver_manifest["model_name"] == "built-in-sm"
    kernel = result.bundle.kernel_pack.kernels[0]
    assert kernel.input_contracts[0]["role"] == "current"
    assert kernel.exact_expressions == ("pyamplicol::prepared_test_input",)
    assert kernel.exact_evaluator_state_path.startswith("kernels/000000/")
    assert all(
        path.startswith("kernels/000000/")
        for path in kernel.referenced_payload_paths
    )
    assert progress[-1] == ("prepared model complete", 1, 1)
