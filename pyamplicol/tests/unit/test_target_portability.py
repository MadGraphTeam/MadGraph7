# SPDX-License-Identifier: 0BSD
from __future__ import annotations

from types import SimpleNamespace

import pytest

from pyamplicol.config import CppConfig, EvaluatorBackend, EvaluatorConfig, RunConfig
from pyamplicol.generation import artifact_writer


def _config(*, backend: EvaluatorBackend, native_arch: bool) -> RunConfig:
    return RunConfig(
        action="generate",
        evaluator=EvaluatorConfig(
            backend=backend,
            cpp=CppConfig(native_arch=native_arch),
        ),
    )


def _install_target_info(
    monkeypatch: pytest.MonkeyPatch,
    *,
    triple: str = "x86_64-unknown-linux-gnu",
    features: tuple[str, ...] = ("avx2", "fma", "sse2"),
) -> None:
    rusticol = SimpleNamespace(
        target_info=lambda: SimpleNamespace(
            triple=triple,
            cpu_features=list(features),
        ),
        abi_version=lambda: 7,
    )
    monkeypatch.setattr(
        artifact_writer.importlib,
        "import_module",
        lambda name: rusticol,
    )


@pytest.mark.parametrize("backend", (EvaluatorBackend.JIT, EvaluatorBackend.CPP))
def test_portable_evaluators_declare_baseline_cpu_requirements(
    monkeypatch: pytest.MonkeyPatch, backend: EvaluatorBackend
) -> None:
    _install_target_info(monkeypatch)
    target, c_abi = artifact_writer._target_metadata(
        _config(backend=backend, native_arch=False)
    )
    assert target == {
        "triple": "x86_64-unknown-linux-gnu",
        "cpu_features": [],
    }
    assert c_abi == 7


def test_jit_stays_baseline_when_cpp_native_option_is_irrelevant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_target_info(monkeypatch)
    target, _ = artifact_writer._target_metadata(
        _config(backend=EvaluatorBackend.JIT, native_arch=True)
    )
    assert target["cpu_features"] == []


@pytest.mark.parametrize("backend", (EvaluatorBackend.ASM, EvaluatorBackend.CPP))
def test_native_compiled_evaluators_record_detected_features(
    monkeypatch: pytest.MonkeyPatch, backend: EvaluatorBackend
) -> None:
    _install_target_info(monkeypatch)
    target, _ = artifact_writer._target_metadata(
        _config(backend=backend, native_arch=True)
    )
    assert target["cpu_features"] == ["avx2", "fma", "sse2"]


def test_writer_rejects_noncanonical_rusticol_target_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_target_info(monkeypatch, features=("sse2", "avx2"))
    with pytest.raises(RuntimeError, match="non-canonical"):
        artifact_writer._target_metadata(
            _config(backend=EvaluatorBackend.CPP, native_arch=True)
        )


def test_writer_rejects_targets_outside_the_release_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_target_info(monkeypatch, triple="aarch64-unknown-linux-gnu")
    with pytest.raises(RuntimeError, match="not supported"):
        artifact_writer._target_metadata(
            _config(backend=EvaluatorBackend.JIT, native_arch=False)
        )
