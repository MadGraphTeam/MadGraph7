# SPDX-License-Identifier: 0BSD
"""Compile process-independent exact catalogs into prepared model bundles."""

from __future__ import annotations

import importlib
import importlib.metadata
import os
import struct
import sys
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import cast

from .._internal.versions import (
    SYMBOLICA_SERIALIZATION_ABI,
    SYMJIT_APPLICATION_ABI,
    package_version,
)
from ..config import EvaluatorConfig
from ..evaluators.symbolica_compile import _compile_symbolica_outputs
from ..evaluators.symbolica_helpers import _symbolica_evaluator_artifact_manifest
from ..evaluators.symbolica_settings import SymbolicaEvaluatorSettings
from .base import Model
from .builtin import BuiltinSMModel
from .external import CompiledUFOModel
from .loading import CompiledModel
from .prepared import (
    PreparedBackend,
    PreparedKernelPack,
    PreparedKernelRecord,
    PreparedModelBundle,
    PreparedModelBundleError,
    load_prepared_model_bundle,
    write_prepared_model_bundle,
)
from .prepared_catalog import (
    PreparedKernelSpec,
    build_prepared_kernel_catalog,
)

PreparedModelProgress = Callable[[str, int, int], None]
_PATH_FIELDS = frozenset(
    (
        "application_path",
        "evaluator_state_path",
        "library_path",
        "payload_path",
        "source_path",
    )
)
_PATH_LIST_FIELDS = frozenset(("payload_paths",))


@dataclass(frozen=True, slots=True)
class PreparedModelBuildResult:
    output: Path
    bundle: PreparedModelBundle
    kernel_count: int
    phase_timings_seconds: Mapping[str, float]


def prepare_model_bundle(
    compiled_model: CompiledModel,
    output: Path,
    *,
    evaluator: EvaluatorConfig,
    progress: PreparedModelProgress | None = None,
) -> PreparedModelBuildResult:
    """Build exactly one eager backend pack and return its validated bundle."""

    started = time.perf_counter()
    model = _runtime_model(compiled_model)
    catalog_started = time.perf_counter()
    catalog = build_prepared_kernel_catalog(model)
    catalog_seconds = time.perf_counter() - catalog_started
    settings = prepared_symbolica_settings(evaluator)
    backend = cast(PreparedBackend, str(evaluator.backend))
    payloads: dict[str, bytes | Path] = {}
    records: list[PreparedKernelRecord] = []
    compile_started = time.perf_counter()

    with tempfile.TemporaryDirectory(prefix="pyamplicol-prepared-model-") as raw:
        staging = Path(raw)
        for index, kernel in enumerate(catalog.kernels):
            if progress is not None:
                progress(
                    f"prepare {kernel.contract_kind} kernel {kernel.kernel_id}",
                    index,
                    len(catalog.kernels),
                )
            record, kernel_payloads = _compile_kernel(
                kernel,
                settings=settings,
                staging=staging / f"kernel-{kernel.kernel_id:06d}",
            )
            records.append(record)
            overlap = payloads.keys() & kernel_payloads.keys()
            if overlap:
                raise PreparedModelBundleError(
                    "prepared kernels produced duplicate payload paths: "
                    + ", ".join(sorted(overlap))
                )
            payloads.update(kernel_payloads)

        compile_seconds = time.perf_counter() - compile_started
        pack = PreparedKernelPack(
            backend=backend,
            optimization_settings=_optimization_metadata(settings),
            producer={
                "distribution": "pyamplicol",
                "version": package_version(),
                "compiled_model_schema": compiled_model.schema_version,
                "model_compiler_version": compiled_model.model_compiler_version,
            },
            dependency_abis={
                "symbolica_serialization": SYMBOLICA_SERIALIZATION_ABI,
                "symjit_application": SYMJIT_APPLICATION_ABI,
                "symbolica_version": _distribution_version("symbolica"),
            },
            provenance={
                "model_name": compiled_model.name,
                "model_source": dict(compiled_model.source),
                "compiled_model_digest": str(
                    compiled_model.source.get("digest", "unavailable")
                ),
                "catalog_kernel_count": len(catalog.kernels),
                "unsupported_variant_count": len(catalog.unsupported_variants),
            },
            target=_prepared_target(backend, evaluator),
            resolver_manifest=catalog.resolver_manifest(),
            kernels=tuple(records),
        )
        bundle_path = write_prepared_model_bundle(
            output,
            compiled_model=compiled_model.to_dict(),
            kernel_pack=pack,
            payloads=payloads,
        )

    bundle = load_prepared_model_bundle(bundle_path)
    timings = {
        "catalog": catalog_seconds,
        "kernel_compilation": compile_seconds,
        "total": time.perf_counter() - started,
    }
    if progress is not None:
        progress("prepared model complete", len(catalog.kernels), len(catalog.kernels))
    return PreparedModelBuildResult(
        output=bundle_path,
        bundle=bundle,
        kernel_count=len(catalog.kernels),
        phase_timings_seconds=timings,
    )


def prepared_symbolica_settings(
    evaluator: EvaluatorConfig,
) -> SymbolicaEvaluatorSettings:
    """Translate the public evaluator configuration into one-kernel settings."""

    optimization = evaluator.optimization
    backend = str(evaluator.backend)
    cores = (
        max(1, os.cpu_count() or 1)
        if optimization.cores == "auto"
        else int(optimization.cores)
    )
    collect_factors = (
        False
        if optimization.collect_factors == "auto"
        else bool(optimization.collect_factors)
    )
    return SymbolicaEvaluatorSettings(
        backend="jit" if backend == "jit" else "compiled-complex",
        iterations=optimization.horner_iterations,
        cpe_iterations=optimization.cpe_iterations,
        n_cores=cores,
        jit_direct_translation=False,
        jit_optimization_level=evaluator.jit.optimization_level,
        max_horner_scheme_variables=optimization.max_horner_variables,
        max_common_pair_cache_entries=optimization.max_common_pair_cache_entries,
        max_common_pair_distance=optimization.max_common_pair_distance,
        collect_factors=collect_factors,
        compiled_inline_asm="default" if backend == "asm" else "none",
        compiled_optimization_level=_cpp_optimization_level(
            evaluator.cpp.optimization
        ),
        compiled_native=evaluator.cpp.native_arch,
        compiler_path=evaluator.cpp.compiler,
        compiler_flags=evaluator.cpp.extra_flags,
        compiled_output_chunk_size=None,
        output_chunk_strategy="uniform",
        output_chunk_autotune_batch_size=evaluator.batch_size,
        compiled_chunk_compile_workers=1,
        compiled_output_dir=None,
    )


def _runtime_model(compiled: CompiledModel) -> Model:
    if compiled.source.get("kind") == "built-in-sm":
        return BuiltinSMModel()
    return CompiledUFOModel(compiled)


def _compile_kernel(
    kernel: PreparedKernelSpec,
    *,
    settings: SymbolicaEvaluatorSettings,
    staging: Path,
) -> tuple[PreparedKernelRecord, dict[str, Path]]:
    from symbolica import Expression

    staging.mkdir(parents=True, exist_ok=True)
    outputs = tuple(Expression.parse(value) for value in kernel.exact_expressions)
    parameters = [Expression.parse(item.symbol) for item in kernel.inputs]
    real_parameters = tuple(
        index
        for index, item in enumerate(kernel.inputs)
        if item.role
        in {
            "left-momentum",
            "right-momentum",
            "momentum",
            "coupling-real",
            "coupling-imag",
        }
    )
    adapter = _compile_symbolica_outputs(
        outputs,
        parameters,
        merge_evaluators_strategy=False,
        verbose_evaluator_build=False,
        real_params=real_parameters,
        symbolica_settings=replace(settings, compiled_output_chunk_size=None),
        jit_compile=True,
        label=f"prepared_{kernel.contract_kind}_{kernel.kernel_id:06d}",
    )
    raw_manifest = _symbolica_evaluator_artifact_manifest(adapter, staging)
    manifest, payloads = _relocate_manifest_payloads(
        raw_manifest,
        staging=staging,
        kernel_id=kernel.kernel_id,
    )
    _validate_backend_manifest(manifest, settings=settings)
    exact_state = manifest.get("evaluator_state_path")
    if not isinstance(exact_state, str) or not exact_state:
        raise PreparedModelBundleError(
            f"prepared kernel {kernel.kernel_id} lacks retained exact evaluator state"
        )
    record = PreparedKernelRecord(
        kernel_id=kernel.kernel_id,
        contract_kind=kernel.contract_kind,
        canonical_signature=kernel.canonical_signature,
        input_arity=kernel.input_arity,
        output_arity=kernel.output_dimension,
        input_layout=tuple(
            f"{item.role}:{item.component}" for item in kernel.inputs
        ),
        input_contracts=tuple(item.to_dict() for item in kernel.inputs),
        output_layout=kernel.output_layout,
        exact_expressions=kernel.exact_expressions,
        exact_evaluator_state_path=exact_state,
        f64_evaluator_manifest=manifest,
    )
    return record, payloads


def _relocate_manifest_payloads(
    manifest: Mapping[str, object],
    *,
    staging: Path,
    kernel_id: int,
) -> tuple[dict[str, object], dict[str, Path]]:
    payloads: dict[str, Path] = {}
    relocated: dict[Path, str] = {}
    counters: dict[str, int] = {}

    def relocate(path: object, field: str) -> str:
        if not isinstance(path, str) or not path:
            raise PreparedModelBundleError(
                f"prepared evaluator {field} must be a nonempty path"
            )
        source = Path(path)
        if not source.is_absolute():
            source = staging / source
        source = source.resolve()
        if not source.is_file() or source.is_symlink():
            raise PreparedModelBundleError(
                f"prepared evaluator payload does not exist: {source}"
            )
        existing = relocated.get(source)
        if existing is not None:
            return existing
        count = counters.get(field, 0)
        counters[field] = count + 1
        stem = field.removesuffix("_path").replace("_", "-")
        suffix = "".join(source.suffixes)
        member = (
            PurePosixPath("kernels")
            / f"{kernel_id:06d}"
            / f"{stem}-{count}{suffix}"
        ).as_posix()
        relocated[source] = member
        payloads[member] = source
        return member

    def visit(value: object) -> object:
        if isinstance(value, Mapping):
            result: dict[str, object] = {}
            for key, child in value.items():
                if key in _PATH_FIELDS:
                    result[str(key)] = (
                        None if child is None else relocate(child, str(key))
                    )
                elif key in _PATH_LIST_FIELDS:
                    if not isinstance(child, Sequence) or isinstance(
                        child, (str, bytes, bytearray)
                    ):
                        raise PreparedModelBundleError(
                            f"prepared evaluator {key} must be a path array"
                        )
                    result[str(key)] = [relocate(item, str(key)) for item in child]
                else:
                    result[str(key)] = visit(child)
            return result
        if isinstance(value, Sequence) and not isinstance(
            value, (str, bytes, bytearray)
        ):
            return [visit(item) for item in value]
        return value

    result = visit(manifest)
    if not isinstance(result, dict):  # pragma: no cover - mapping root invariant
        raise PreparedModelBundleError("prepared evaluator manifest must be an object")
    return result, payloads


def _validate_backend_manifest(
    manifest: Mapping[str, object],
    *,
    settings: SymbolicaEvaluatorSettings,
) -> None:
    if settings.backend != "jit":
        if manifest.get("kind") != "compiled-complex-evaluator":
            raise PreparedModelBundleError(
                "prepared native evaluator has an unexpected manifest kind"
            )
        return
    expected = {
        "kind": "symjit-application-evaluator",
        "application_abi": SYMJIT_APPLICATION_ABI,
        "compiler_type": "native",
        "translation_mode": "indirect",
        "word_bits": 64,
        "endianness": "little",
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise PreparedModelBundleError(
                f"prepared JIT evaluator has incompatible {key}: "
                f"{manifest.get(key)!r}"
            )
    if manifest.get("required_defuns") != []:
        raise PreparedModelBundleError(
            "prepared JIT evaluators must not depend on external functions"
        )


def _optimization_metadata(settings: SymbolicaEvaluatorSettings) -> dict[str, object]:
    result = settings.to_json_dict()
    result["compiled_output_dir"] = None
    result["compiled_output_chunk_size"] = None
    return result


def _prepared_target(
    backend: PreparedBackend,
    evaluator: EvaluatorConfig,
) -> dict[str, object]:
    if backend == "jit":
        if struct.calcsize("P") * 8 != 64 or sys.byteorder != "little":
            raise PreparedModelBundleError(
                "portable prepared JIT packs require a 64-bit little-endian host"
            )
        return {
            "portable": True,
            "word_bits": 64,
            "endianness": "little",
            "target_triple": "portable-symjit-mir",
            "cpu_features": [],
        }
    try:
        rusticol = importlib.import_module("pyamplicol._rusticol")
        info = rusticol.target_info()
    except (AttributeError, ImportError, OSError) as error:
        raise PreparedModelBundleError(
            "native prepared packs require Rusticol target introspection"
        ) from error
    features = (
        sorted(str(item) for item in info.cpu_features)
        if evaluator.cpp.native_arch
        else []
    )
    return {
        "portable": False,
        "word_bits": struct.calcsize("P") * 8,
        "endianness": sys.byteorder,
        "target_triple": str(info.triple),
        "cpu_features": features,
    }


def _distribution_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _cpp_optimization_level(value: str) -> int:
    normalized = value.strip().lower()
    if normalized.startswith("-o"):
        normalized = normalized[2:]
    elif normalized.startswith("o"):
        normalized = normalized[1:]
    if normalized in {"0", "1", "2", "3"}:
        return int(normalized)
    raise PreparedModelBundleError(
        f"unsupported prepared C++ optimization level {value!r}"
    )


__all__ = [
    "PreparedModelBuildResult",
    "prepare_model_bundle",
    "prepared_symbolica_settings",
]
