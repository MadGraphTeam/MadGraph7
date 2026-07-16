# SPDX-License-Identifier: 0BSD
"""Runtime adapters for JIT, compiled, and chunked Symbolica evaluators."""

from __future__ import annotations

import struct
import sys
import tempfile
import time
import uuid
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from .._internal.physics.types import NativeEvaluationError
from .._internal.versions import (
    SYMBOLICA_ASM_RUNTIME_CAPABILITY,
    SYMBOLICA_CPP_RUNTIME_CAPABILITY,
    SYMBOLICA_LEGACY_JIT_RUNTIME_CAPABILITY,
    SYMJIT_APPLICATION_ABI,
    SYMJIT_F64_RUNTIME_CAPABILITY,
)
from .execution_schema import aggregate_runtime_capabilities
from .symbolica_helpers import (
    _artifact_path_for_manifest,
    _artifact_path_from_manifest,
    _artifact_subdirectory,
    _complex128_parameter_rows,
    _evaluate_prepared_complex,
    _evaluate_prepared_complex_profiled,
    _safe_symbol_name,
    _symbolica_evaluator_artifact_manifest,
)
from .symbolica_settings import (
    ProgressCallback,
    SymbolicaEvaluatorSettings,
    _report_progress,
)


class _JITSymbolicaEvaluatorAdapter:
    def __init__(
        self,
        evaluator: Any,
        settings: SymbolicaEvaluatorSettings,
        label: str,
        *,
        input_len: int,
        output_len: int,
        progress_callback: ProgressCallback | None = None,
    ) -> None:
        self.input_len = int(input_len)
        self.output_len = int(output_len)
        self.backend = settings.backend
        self.settings = settings.to_json_dict()
        self.label = _safe_symbol_name(label)
        self._source_evaluator = evaluator
        self._progress_callback = progress_callback
        self.application_path: Path | None = None
        self.evaluator_state_path: Path | None = None
        self.build_timing: dict[str, float] = {}

    def evaluate_complex(self, parameter_rows: Any) -> Any:
        return self._source_evaluator.evaluate_complex(parameter_rows)

    def evaluate_complex_profiled(
        self, parameter_rows: Any
    ) -> tuple[Any, tuple[float, float, float]]:
        return self._evaluate_complex_profiled_prepared(
            _complex128_parameter_rows(parameter_rows)
        )

    def supports_complex_profiled(self) -> bool:
        return callable(
            getattr(self._source_evaluator, "evaluate_complex_profiled", None)
        )

    def evaluate(self, parameter_rows: Any) -> Any:
        return self._source_evaluator.evaluate(parameter_rows)

    def _evaluate_complex_prepared(self, parameter_rows: np.ndarray) -> Any:
        return self._source_evaluator.evaluate_complex(parameter_rows)

    def _evaluate_complex_profiled_prepared(
        self, parameter_rows: np.ndarray
    ) -> tuple[Any, tuple[float, float, float]]:
        profile = getattr(self._source_evaluator, "evaluate_complex_profiled", None)
        if not callable(profile):
            raise NativeEvaluationError(
                "this Symbolica build does not expose native complex profiling"
            )
        output, timing = profile(parameter_rows)
        if not isinstance(timing, tuple) or len(timing) != 3:
            raise NativeEvaluationError(
                "Symbolica returned an invalid native complex profile"
            )
        return output, tuple(float(value) for value in timing)

    def materialize(self) -> None:
        self._ensure_jit_compiled()

    def _ensure_jit_compiled(self) -> None:
        dummy = np.ones((1, self.input_len), dtype=np.complex128)
        self._source_evaluator.evaluate_complex(dummy)

    @classmethod
    def from_artifact(
        cls,
        manifest: dict[str, Any],
        artifact_dir: Path,
    ) -> _JITSymbolicaEvaluatorAdapter:
        from symbolica import Evaluator

        instance = cls.__new__(cls)
        instance.input_len = int(manifest["input_len"])
        instance.output_len = int(manifest["output_len"])
        instance.backend = str(manifest["backend"])
        instance.settings = dict(manifest.get("settings", {}))
        instance.label = str(manifest.get("label", "jit_symbolica_evaluator"))
        application_path = manifest.get("application_path")
        instance.application_path = (
            None
            if application_path is None
            else _artifact_path_from_manifest(str(application_path), artifact_dir)
        )
        instance.evaluator_state_path = _artifact_path_from_manifest(
            str(manifest["evaluator_state_path"]),
            artifact_dir,
        )
        instance._source_evaluator = Evaluator.load(
            instance.evaluator_state_path.read_bytes()
        )
        instance._progress_callback = None
        instance.build_timing = {}
        return instance

    def artifact_manifest(self, artifact_dir: Path) -> dict[str, Any]:
        if bool(self.settings.get("jit_direct_translation")):
            raise NativeEvaluationError(
                "direct SymJIT translation cannot be persisted as a portable "
                "process artifact; use indirect translation"
            )
        _report_progress(
            self._progress_callback,
            stage="jit materialize",
            item=self.label,
        )
        materialize_started = time.perf_counter()
        self._ensure_jit_compiled()
        jit_compile_s = time.perf_counter() - materialize_started
        _report_progress(
            self._progress_callback,
            stage="jit materialized",
            item=f"{self.label} {jit_compile_s:.3f}s",
            increment=1,
        )
        evaluator_dir = _artifact_subdirectory(artifact_dir, "evaluators")
        unique = uuid.uuid4().hex
        application_path = evaluator_dir / f"{self.label}_{unique}.symjit"
        evaluator_state_path = evaluator_dir / f"{self.label}_{unique}.evaluator.bin"
        save_started = time.perf_counter()
        application, element_layout = self._export_symjit_application()
        application_path.write_bytes(application)
        application_export_s = time.perf_counter() - save_started
        state_save_started = time.perf_counter()
        evaluator_state_path.write_bytes(self._source_evaluator.save())
        evaluator_save_s = time.perf_counter() - save_started
        evaluator_state_save_s = time.perf_counter() - state_save_started
        self.application_path = application_path
        self.evaluator_state_path = evaluator_state_path
        build_timing = dict(self.build_timing)
        build_timing["jit_materialize_s"] = jit_compile_s
        build_timing["symjit_application_export_s"] = application_export_s
        build_timing["evaluator_save_s"] = evaluator_state_save_s
        build_timing["artifact_manifest_s"] = jit_compile_s + evaluator_save_s
        optimization_level = self.settings.get("jit_optimization_level", 3)
        if isinstance(optimization_level, bool) or not isinstance(
            optimization_level, int
        ):
            raise NativeEvaluationError("invalid SymJIT optimization-level metadata")
        return {
            "kind": "symjit-application-evaluator",
            "runtime_capability": SYMJIT_F64_RUNTIME_CAPABILITY,
            "backend": self.backend,
            "label": self.label,
            "input_len": self.input_len,
            "output_len": self.output_len,
            "application_path": _artifact_path_for_manifest(
                application_path, artifact_dir
            ),
            "application_abi": SYMJIT_APPLICATION_ABI,
            "element_layout": element_layout,
            "batch_layout": "row-major",
            "compiler_type": "native",
            "translation_mode": "indirect",
            "optimization_level": optimization_level,
            "word_bits": struct.calcsize("P") * 8,
            "endianness": sys.byteorder,
            "required_defuns": [],
            "evaluator_state_path": _artifact_path_for_manifest(
                evaluator_state_path, artifact_dir
            ),
            "evaluator_state_runtime_capability": (
                SYMBOLICA_LEGACY_JIT_RUNTIME_CAPABILITY
            ),
            "settings": self.settings,
            "build_timing": build_timing,
        }

    def _export_symjit_application(self) -> tuple[bytes, str]:
        if struct.calcsize("P") * 8 != 64 or sys.byteorder != "little":
            raise NativeEvaluationError(
                "direct SymJIT f64 artifacts require a 64-bit little-endian host"
            )
        exporter = getattr(
            self._source_evaluator,
            "export_symjit_f64_application",
            None,
        )
        if not callable(exporter):
            raise NativeEvaluationError(
                "this Symbolica build cannot export a direct SymJIT f64 application; "
                "install the pinned pyAmpliCol candidate dependency"
            )
        try:
            exported = exporter()
        except Exception as error:
            raise NativeEvaluationError(
                "Symbolica could not export a self-contained SymJIT f64 application; "
                "external evaluator functions are not supported in process artifacts"
            ) from error
        if not isinstance(exported, tuple) or len(exported) != 2:
            raise NativeEvaluationError(
                "Symbolica returned an invalid SymJIT application export"
            )
        application, layout = exported
        if not isinstance(application, bytes) or not application:
            raise NativeEvaluationError(
                "Symbolica returned an empty or non-bytes SymJIT application"
            )
        expected_layout = _symjit_element_layout()
        if layout != expected_layout:
            raise NativeEvaluationError(
                "Symbolica exported an incompatible SymJIT element layout: "
                f"expected {expected_layout!r}, got {layout!r}"
            )
        return application, expected_layout


class _CompiledComplexEvaluatorAdapter:
    def __init__(
        self,
        evaluator: Any,
        settings: SymbolicaEvaluatorSettings,
        label: str,
        *,
        input_len: int,
        output_len: int,
    ) -> None:
        safe_label = _safe_symbol_name(label)
        unique = uuid.uuid4().hex
        function_name = f"pyamplicol_{safe_label}_{unique}"
        self.function_name = function_name
        self.input_len = int(input_len)
        self.output_len = int(output_len)
        self.backend = settings.backend
        self.settings = settings.to_json_dict()
        self.runtime_capability = _compiled_runtime_capability(self.settings)
        self.number_type = "complex"
        self._source_evaluator = evaluator
        self.build_timing: dict[str, float] = {}
        if settings.compiled_output_dir is None:
            self._tmpdir: tempfile.TemporaryDirectory[str] | None = (
                tempfile.TemporaryDirectory(prefix="pyamplicol-symbolica-")
            )
            path = Path(self._tmpdir.name)
        else:
            self._tmpdir = None
            path = Path(settings.compiled_output_dir).expanduser()
            path.mkdir(parents=True, exist_ok=True)
        self.source_path = path / f"{function_name}.cpp"
        self.library_path = path / f"lib{function_name}"
        self.evaluator_state_path: Path | None = path / f"{function_name}.evaluator.bin"
        save = getattr(evaluator, "save", None)
        if callable(save):
            save_started = time.perf_counter()
            self.evaluator_state_path.write_bytes(save())
            self.build_timing["evaluator_save_s"] = time.perf_counter() - save_started
        else:
            self.evaluator_state_path = None
        compile_started = time.perf_counter()
        self._compiled = evaluator.compile(
            function_name,
            str(self.source_path),
            str(self.library_path),
            self.number_type,
            inline_asm=settings.compiled_inline_asm,
            optimization_level=settings.compiled_optimization_level,
            native=settings.compiled_native,
            compiler_path=settings.compiler_path,
            compiler_flags=_compiled_compiler_flags(settings),
        )
        self.build_timing["cxx_compile_s"] = time.perf_counter() - compile_started

    def evaluate_complex(self, parameter_rows: Any) -> Any:
        return self._evaluate_complex_prepared(
            _complex128_parameter_rows(parameter_rows)
        )

    def _evaluate_complex_prepared(self, parameter_rows: np.ndarray) -> Any:
        return self._compiled.evaluate(parameter_rows)

    @classmethod
    def from_artifact(
        cls,
        manifest: dict[str, Any],
        artifact_dir: Path,
    ) -> _CompiledComplexEvaluatorAdapter:
        from symbolica import CompiledComplexEvaluator

        instance = cls.__new__(cls)
        instance._tmpdir = None
        instance.function_name = str(manifest["function_name"])
        instance.input_len = int(manifest["input_len"])
        instance.output_len = int(manifest["output_len"])
        instance.backend = str(manifest["backend"])
        instance.settings = dict(manifest.get("settings", {}))
        instance.runtime_capability = str(
            manifest.get(
                "runtime_capability",
                _compiled_runtime_capability(instance.settings),
            )
        )
        instance.number_type = str(manifest["number_type"])
        instance._source_evaluator = None
        instance.source_path = _artifact_path_from_manifest(
            str(manifest["source_path"]),
            artifact_dir,
        )
        instance.library_path = _artifact_path_from_manifest(
            str(manifest["library_path"]),
            artifact_dir,
        )
        state_path = manifest.get("evaluator_state_path")
        instance.evaluator_state_path = (
            None
            if state_path is None
            else _artifact_path_from_manifest(str(state_path), artifact_dir)
        )
        instance.build_timing = {}
        if instance.number_type != "complex":
            raise NativeEvaluationError(
                f"unsupported compiled evaluator number type: {instance.number_type!r}"
            )
        instance._compiled = CompiledComplexEvaluator.load(
            str(instance.library_path),
            instance.function_name,
            instance.input_len,
            instance.output_len,
        )
        return instance

    def artifact_manifest(self, artifact_dir: Path) -> dict[str, Any]:
        return {
            "kind": "compiled-complex-evaluator",
            "runtime_capability": self.runtime_capability,
            "backend": self.backend,
            "number_type": self.number_type,
            "function_name": self.function_name,
            "input_len": self.input_len,
            "output_len": self.output_len,
            "settings": self.settings,
            "source_path": _artifact_path_for_manifest(
                self.source_path,
                artifact_dir,
            ),
            "library_path": _artifact_path_for_manifest(
                self.library_path,
                artifact_dir,
            ),
            "evaluator_state_path": (
                None
                if self.evaluator_state_path is None
                else _artifact_path_for_manifest(
                    self.evaluator_state_path,
                    artifact_dir,
                )
            ),
            "build_timing": dict(self.build_timing),
        }


def _compiled_compiler_flags(settings: SymbolicaEvaluatorSettings) -> tuple[str, ...]:
    return tuple(settings.compiler_flags)


def _symjit_element_layout() -> str:
    return "complex-f64"


def _compiled_runtime_capability(settings: Mapping[str, object]) -> str:
    if str(settings.get("compiled_inline_asm", "default")) == "none":
        return SYMBOLICA_CPP_RUNTIME_CAPABILITY
    return SYMBOLICA_ASM_RUNTIME_CAPABILITY


class _ChunkedSymbolicaEvaluator:
    def __init__(self, evaluators: tuple[Any, ...]) -> None:
        if not evaluators:
            raise NativeEvaluationError("chunked evaluator needs at least one chunk")
        self._evaluators = evaluators
        self.build_timing: dict[str, float] = {}

    def evaluate_complex(self, parameter_rows: Any) -> Any:
        return np.concatenate(self.evaluate_complex_chunks(parameter_rows), axis=1)

    def evaluate_complex_chunks(self, parameter_rows: Any) -> tuple[Any, ...]:
        prepared_rows = _complex128_parameter_rows(parameter_rows)
        return tuple(
            _evaluate_prepared_complex(evaluator, prepared_rows)
            for evaluator in self._evaluators
        )

    def evaluate_complex_profiled(
        self, parameter_rows: Any
    ) -> tuple[tuple[Any, ...], tuple[float, float, float]]:
        prepared_rows = _complex128_parameter_rows(parameter_rows)
        outputs: list[Any] = []
        profiles: list[tuple[float, float, float]] = []
        for evaluator in self._evaluators:
            output, profile = _evaluate_prepared_complex_profiled(
                evaluator, prepared_rows
            )
            outputs.append(output)
            profiles.append(profile)
        pack_times = sorted(profile[0] for profile in profiles)
        shared_pack_s = pack_times[len(pack_times) // 2]
        return tuple(outputs), (
            shared_pack_s,
            sum(profile[1] for profile in profiles),
            sum(profile[2] for profile in profiles),
        )

    def supports_complex_profiled(self) -> bool:
        return all(
            bool(getattr(evaluator, "supports_complex_profiled", lambda: False)())
            for evaluator in self._evaluators
        )

    @classmethod
    def from_artifact(
        cls,
        manifest: dict[str, Any],
        artifact_dir: Path,
    ) -> _ChunkedSymbolicaEvaluator:
        chunks = manifest.get("chunks")
        if not isinstance(chunks, list):
            raise NativeEvaluationError("chunked evaluator artifact is missing chunks")
        return cls(
            tuple(
                _load_symbolica_evaluator_artifact(chunk, artifact_dir)
                for chunk in chunks
            )
        )

    def artifact_manifest(self, artifact_dir: Path) -> dict[str, Any]:
        worker_counts = []
        for evaluator in self._evaluators:
            settings = getattr(evaluator, "settings", None)
            if isinstance(settings, Mapping):
                workers = settings.get("compiled_chunk_compile_workers")
                if isinstance(workers, int):
                    worker_counts.append(workers)
        workers = min(max(worker_counts, default=1), len(self._evaluators))
        if workers <= 1:
            chunks = [
                _symbolica_evaluator_artifact_manifest(evaluator, artifact_dir)
                for evaluator in self._evaluators
            ]
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [
                    executor.submit(
                        _symbolica_evaluator_artifact_manifest,
                        evaluator,
                        artifact_dir,
                    )
                    for evaluator in self._evaluators
                ]
                chunks = [future.result() for future in futures]
        timing_totals = dict(self.build_timing)
        timing_totals["chunk_count"] = float(len(chunks))
        for chunk in chunks:
            chunk_timing = (
                chunk.get("build_timing") if isinstance(chunk, dict) else None
            )
            if not isinstance(chunk_timing, Mapping):
                continue
            for key, value in chunk_timing.items():
                if isinstance(value, (float, int)):
                    timing_totals[str(key)] = timing_totals.get(str(key), 0.0) + float(
                        value
                    )
        return {
            "kind": "chunked-symbolica-evaluator",
            "chunks": chunks,
            "required_runtime_capabilities": list(
                aggregate_runtime_capabilities(chunks)
            ),
            "build_timing": timing_totals,
        }


def _load_symbolica_evaluator_artifact(
    manifest: Any,
    artifact_dir: Path,
) -> Any:
    if not isinstance(manifest, dict):
        raise NativeEvaluationError("compiled evaluator artifact entry is invalid")
    kind = manifest.get("kind")
    if kind in {"jit-symbolica-evaluator", "symjit-application-evaluator"}:
        return _JITSymbolicaEvaluatorAdapter.from_artifact(manifest, artifact_dir)
    if kind == "compiled-complex-evaluator":
        return _CompiledComplexEvaluatorAdapter.from_artifact(manifest, artifact_dir)
    if kind == "chunked-symbolica-evaluator":
        return _ChunkedSymbolicaEvaluator.from_artifact(manifest, artifact_dir)
    raise NativeEvaluationError(f"unsupported evaluator artifact kind: {kind!r}")
