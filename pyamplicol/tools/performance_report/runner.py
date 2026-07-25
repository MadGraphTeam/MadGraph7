# SPDX-License-Identifier: 0BSD
"""Direct Python-API generation, profiling, and validation for report cells."""

from __future__ import annotations

import hashlib
import importlib
import io
import json
import math
import os
import platform
import stat
import time
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Protocol

from .models import (
    Accuracy,
    CellSpec,
    ExecutionMode,
    ModelKey,
    ResultStatus,
    Workload,
)
from .runtime_evidence import (
    RuntimeEvidenceError,
    established_preimport_runtime_identity,
    loaded_pyamplicol_origin_policy,
    python_package_tree_identity,
)

RELATIVE_TOLERANCE = 1.0e-12
INDEPENDENT_RELATIVE_TOLERANCE = 1.0e-8
ABSOLUTE_TOLERANCE = 1.0e-15
GENERATION_VALIDATION_SEED = 12345
DEFAULT_TARGET_RUNTIME_SECONDS = 5.0
_RECURRENCE_DIRECT_TEMPLATE_ABI = "pyamplicol-recurrence-direct-template-v1"
_RECURRENCE_DIRECT_BACKEND_ABI = "rusticol.recurrence-direct-backend.v1"
_RECURRENCE_DIRECT_CANONICALIZATION_ABI = "pyamplicol-canonical-json-v1"
_RECURRENCE_DIRECT_PAYLOAD_BINDING_ABI = (
    "pyamplicol-recurrence-direct-payload-binding-v1"
)
_RECURRENCE_JIT_SOURCE_APPLICATION_ABI = "symjit-application-storage-v3"
_RECURRENCE_JIT_DIRECT_APPLICATION_ABI = "symjit-direct-application-storage-v1"
_RECURRENCE_JIT_SOURCE_RUNTIME_CAPABILITY = "symjit.application.complex-f64.v1"


class RunnerError(RuntimeError):
    """Raised when a cell cannot satisfy the report measurement contract."""


@dataclass(frozen=True, slots=True)
class RunnerSettings:
    target_runtime_seconds: float = DEFAULT_TARGET_RUNTIME_SECONDS
    batch_size: int = 128
    worker_cores: int = 1
    model_cache_dir: Path | None = None

    def __post_init__(self) -> None:
        if self.target_runtime_seconds <= 0.0:
            raise ValueError("target_runtime_seconds must be positive")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.worker_cores < 1:
            raise ValueError("worker_cores must be positive")


@dataclass(frozen=True, slots=True)
class SelectorContract:
    selected_color_flow_ids: tuple[str, ...]
    selected_color_words: tuple[tuple[int, ...], ...]
    all_flow_helicity_ids: tuple[str, ...]
    all_flow_source_helicities: tuple[tuple[int, int], ...]
    point_digest: str

    def __post_init__(self) -> None:
        if not self.selected_color_flow_ids:
            raise ValueError("selector contract requires a selected color flow")
        if len(self.selected_color_flow_ids) != len(self.selected_color_words):
            raise ValueError("color-flow IDs and words must have equal length")
        if len(set(self.selected_color_flow_ids)) != len(self.selected_color_flow_ids):
            raise ValueError("selector contract color-flow IDs must be unique")
        if len(self.all_flow_helicity_ids) != 1:
            raise ValueError("selector contract requires one fixed helicity")
        if not self.all_flow_source_helicities:
            raise ValueError("selector contract requires source helicities")
        if len(self.point_digest) != 64:
            raise ValueError("selector contract point digest must be SHA-256")

    def as_dict(self) -> dict[str, object]:
        return {
            "selected_color_flow_ids": list(self.selected_color_flow_ids),
            "selected_color_words": [list(word) for word in self.selected_color_words],
            "all_flow_helicity_ids": list(self.all_flow_helicity_ids),
            "all_flow_source_helicities": {
                str(label): value for label, value in self.all_flow_source_helicities
            },
            "point_digest": self.point_digest,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> SelectorContract:
        raw_ids = value.get("selected_color_flow_ids")
        raw_words = value.get("selected_color_words")
        raw_helicities = value.get("all_flow_helicity_ids")
        raw_sources = value.get("all_flow_source_helicities")
        point_digest = value.get("point_digest")
        if (
            not isinstance(raw_ids, Sequence)
            or isinstance(raw_ids, (str, bytes))
            or not isinstance(raw_words, Sequence)
            or isinstance(raw_words, (str, bytes))
            or not isinstance(raw_helicities, Sequence)
            or isinstance(raw_helicities, (str, bytes))
            or not isinstance(raw_sources, Mapping)
            or not isinstance(point_digest, str)
        ):
            raise ValueError("selector contract has an invalid shape")
        return cls(
            selected_color_flow_ids=tuple(str(item) for item in raw_ids),
            selected_color_words=tuple(
                tuple(int(label) for label in word)  # type: ignore[arg-type]
                for word in raw_words
            ),
            all_flow_helicity_ids=tuple(str(item) for item in raw_helicities),
            all_flow_source_helicities=tuple(
                sorted((int(label), int(state)) for label, state in raw_sources.items())
            ),
            point_digest=point_digest,
        )


@dataclass(frozen=True, slots=True)
class GeneratedArtifact:
    path: Path
    process_id: str
    generation_seconds: float
    model_preparation_seconds: float
    model_preparation_reused: bool
    requested_config: Mapping[str, object]
    effective_config: Mapping[str, object]


class RuntimeLike(Protocol):
    @property
    def physics(self) -> object: ...

    def evaluate(
        self,
        momenta: object,
        *,
        helicities: Sequence[str] | None = None,
        color_flows: Sequence[str] | None = None,
        precision: int = 16,
    ) -> Sequence[object]: ...

    def evaluate_resolved(
        self,
        momenta: object,
        *,
        helicities: Sequence[str] | None = None,
        color_flows: Sequence[str] | None = None,
        precision: int = 16,
    ) -> object: ...


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        default=lambda item: item.tolist(),
    ).encode("ascii")


def _regular_file_identity(path: Path) -> tuple[int, str]:
    """Hash one unchanged regular file through a single no-follow descriptor."""

    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_BINARY", 0)
    )
    digest = hashlib.sha256()
    descriptor: int | None = None
    try:
        descriptor = os.open(path, flags)
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise RunnerError(f"identity target is not a regular file: {path}")
        byte_count = 0
        with os.fdopen(descriptor, "rb", closefd=True) as stream:
            descriptor = None
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
                byte_count += len(block)
            after = os.fstat(stream.fileno())
    except OSError as error:
        raise RunnerError(
            f"cannot hash regular file through a checked fd: {path}"
        ) from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
    stable_fields = (
        "st_dev",
        "st_ino",
        "st_mode",
        "st_size",
        "st_mtime_ns",
        "st_ctime_ns",
    )
    if byte_count != before.st_size or any(
        getattr(before, name) != getattr(after, name) for name in stable_fields
    ):
        raise RunnerError(f"identity target changed while it was hashed: {path}")
    return byte_count, digest.hexdigest()


def _sha256_file(path: Path) -> str:
    return _regular_file_identity(path)[1]


def point_digest(points: object) -> str:
    return hashlib.sha256(_canonical_json(points)).hexdigest()


def _real_nonnegative(value: object) -> float:
    number = complex(value)
    if not math.isfinite(number.real) or not math.isfinite(number.imag):
        raise RunnerError("matrix element is not finite")
    if abs(number.imag) > 1.0e-9 * max(abs(number.real), 1.0):
        raise RunnerError(
            f"matrix element has a non-negligible imaginary part: {value}"
        )
    result = float(number.real)
    if result < -ABSOLUTE_TOLERANCE:
        raise RunnerError(f"matrix element is materially negative: {value}")
    return max(result, 0.0)


def _model_source_path(repo_root: Path, model: ModelKey) -> Path | None:
    if model is ModelKey.BUILTIN_SM:
        return None
    if model is ModelKey.UFO_SM:
        return repo_root / "src/pyamplicol/assets/models/json/sm/sm.json"
    if model is ModelKey.SCALAR_CONTACT:
        return repo_root / "src/pyamplicol/assets/models/json/scalars/scalars.json"
    if model is ModelKey.SCALAR_GRAVITY:
        return (
            repo_root
            / "src/pyamplicol/assets/models/json/scalar_gravity/scalar_gravity.json"
        )
    raise RunnerError(f"model {model.value!r} is not supported by process matrices")


def config_values(
    cell: CellSpec,
    settings: RunnerSettings,
    *,
    repo_root: Path,
) -> dict[str, object]:
    """Return complete generation/benchmark settings for one pyAmpliCol cell."""

    measurement = cell.measurement
    if measurement.execution_mode is ExecutionMode.AMPLICOL:
        raise RunnerError("original AmpliCol does not use the pyAmpliCol runner")
    if measurement.model is None:
        raise RunnerError("pyAmpliCol measurement requires an explicit model")
    model_path = _model_source_path(repo_root, measurement.model)
    layout = (
        "all-flow-union" if cell.workload is Workload.ALL_FLOW else "topology-replay"
    )
    values: dict[str, object] = {
        "model": {
            "source": "built-in-sm" if model_path is None else os.fspath(model_path),
            "cache": True,
            "cache_dir": (
                None
                if settings.model_cache_dir is None
                else os.fspath(settings.model_cache_dir)
            ),
        },
        "color": {
            "accuracy": measurement.accuracy.value,
            "lc_flow_layout": layout,
        },
        "generation": {
            "workers": settings.worker_cores,
            "emit_api_bundle": True,
            "validation": {
                "enabled": True,
                "samples": 10,
                "seed": GENERATION_VALIDATION_SEED,
                "relative_tolerance": RELATIVE_TOLERANCE,
                "absolute_tolerance": 1.0e-300,
                "post_build_validation": True,
            },
        },
        "evaluator": {
            "backend": measurement.backend,
            "execution_mode": measurement.execution_mode.value,
            "batch_size": settings.batch_size,
            "output_chunk_size": 512,
            "optimization": {
                "horner_iterations": 10,
                "cpe_iterations": None,
                "cores": settings.worker_cores,
                "max_horner_variables": 1000,
                "max_common_pair_cache_entries": 5_000_000,
                "max_common_pair_distance": 1000,
            },
            "jit": {
                "optimization_level": measurement.jit_optimization_level or 2,
            },
            "cpp": {"optimization": "O3"},
        },
        "benchmark": {
            "target_runtime": settings.target_runtime_seconds,
            "batch_size": settings.batch_size,
            "warmup_runs": 2,
            "minimum_samples": 5,
        },
        "output": {"format": "json", "progress": "off"},
    }
    return values


def _physics_ids(physics: object, name: str) -> tuple[str, ...]:
    return tuple(str(item.id) for item in getattr(physics, name, ()))


def validate_runtime_contract(cell: CellSpec, runtime: RuntimeLike) -> None:
    physics = runtime.physics
    accuracy = str(getattr(physics, "color_accuracy", ""))
    if accuracy != cell.measurement.accuracy.value:
        raise RunnerError(
            f"artifact color accuracy {accuracy!r} does not match "
            f"{cell.measurement.accuracy.value!r}"
        )
    capabilities = set(getattr(physics, "selector_capabilities", ()))
    helicity_ids = _physics_ids(physics, "helicities")
    if not helicity_ids:
        raise RunnerError("artifact exposes no physical helicities")
    if cell.measurement.accuracy is Accuracy.LC:
        color_ids = _physics_ids(physics, "color_flows")
        if not color_ids:
            raise RunnerError("LC artifact exposes no physical color flows")
        missing = {"helicity", "color_flow"} - capabilities
        if missing:
            raise RunnerError(
                "LC artifact does not retain complete runtime selectors: "
                + ", ".join(sorted(missing))
            )
    elif cell.workload is not Workload.CONTRACTED:
        raise RunnerError("NLC/full measurements must use the contracted workload")


def validate_artifact_contract(cell: CellSpec, artifact_path: Path) -> None:
    from pyamplicol._internal.versions import (
        COMPILED_PLANE_ARENA_RUNTIME_CAPABILITY,
        EAGER_DIRECT_ARENA_RUNTIME_CAPABILITY,
        RECURRENCE_DIRECT_ARENA_RUNTIME_CAPABILITY,
    )
    from pyamplicol.artifacts import inspect_artifact

    inspection = inspect_artifact(artifact_path)
    if len(inspection.processes) != 1:
        raise RunnerError("report artifacts must contain exactly one process")
    process = inspection.processes[0]
    if process.execution_mode != cell.measurement.execution_mode.value:
        raise RunnerError(
            f"artifact execution mode {process.execution_mode!r} does not match "
            f"{cell.measurement.execution_mode.value!r}"
        )
    if process.generation_specialized_axes:
        raise RunnerError(
            "report artifacts must retain complete runtime coverage; specialized "
            f"axes: {process.generation_specialized_axes}"
        )
    if process.selected_source_helicities or process.selected_color_sector_ids:
        raise RunnerError("report artifact contains forbidden generation selectors")
    expected_arena_capability = {
        ExecutionMode.COMPILED: COMPILED_PLANE_ARENA_RUNTIME_CAPABILITY,
        ExecutionMode.EAGER: EAGER_DIRECT_ARENA_RUNTIME_CAPABILITY,
        ExecutionMode.RECURRENCE: RECURRENCE_DIRECT_ARENA_RUNTIME_CAPABILITY,
    }[cell.measurement.execution_mode]
    if expected_arena_capability not in set(inspection.runtime_capabilities):
        raise RunnerError(
            "report artifact does not require the final Arena execution lane "
            f"{expected_arena_capability!r}"
        )
    if cell.measurement.accuracy is Accuracy.LC:
        expected_layout = (
            "all-flow-union"
            if cell.workload is Workload.ALL_FLOW
            else "topology-replay"
        )
        if process.lc_flow_layout != expected_layout:
            raise RunnerError(
                f"artifact LC layout {process.lc_flow_layout!r} does not match "
                f"{expected_layout!r}"
            )


def runtime_identity_payload(
    cell: CellSpec,
    runtime: object,
    artifact_path: Path,
    process_id: str,
    *,
    expected_source_revision: str,
) -> dict[str, object]:
    """Return the exact artifact/native-Arena identity used by one report cell."""

    import pyamplicol
    from pyamplicol._internal.versions import (
        COMPILED_PLANE_ARENA_RUNTIME_CAPABILITY,
        COMPILED_PLANE_DIRECT_APPLICATION_ABI,
        EAGER_DIRECT_ARENA_RUNTIME_CAPABILITY,
        EAGER_DIRECT_TABLE_BINDING_ABI,
        NATIVE_COMPILED_DIRECT_APPLICATION_ABI,
        RECURRENCE_DIRECT_ARENA_RUNTIME_CAPABILITY,
        RECURRENCE_RUNTIME_LAYOUT_ABI,
        SYMBOLICA_ASM_RUNTIME_CAPABILITY,
        SYMBOLICA_CPP_RUNTIME_CAPABILITY,
        SYMJIT_APPLICATION_ABI,
        SYMJIT_F64_RUNTIME_CAPABILITY,
        _active_build_info,
        verify_native_module,
    )
    from pyamplicol.artifacts import load_manifest

    manifest = load_manifest(artifact_path, verify_payloads=False)
    matches = [
        process for process in manifest.processes if process.get("id") == process_id
    ]
    if len(matches) != 1:
        raise RunnerError(
            f"report runtime process {process_id!r} is missing or ambiguous"
        )
    process = matches[0]
    loaded_artifact_id = getattr(runtime, "artifact_id", None)
    if loaded_artifact_id != manifest.artifact_id:
        raise RunnerError(
            "native runtime loaded artifact identity does not match the report "
            "artifact manifest"
        )
    try:
        loaded_execution_mode = runtime.execution_mode  # type: ignore[attr-defined]
    except Exception as exc:
        raise RunnerError(
            "native runtime does not expose an authenticated execution mode"
        ) from exc
    expected_execution_mode = cell.measurement.execution_mode.value
    if loaded_execution_mode != expected_execution_mode:
        raise RunnerError(
            f"native runtime loaded execution mode {loaded_execution_mode!r} "
            f"does not match the requested report mode {expected_execution_mode!r}"
        )
    required_capabilities = tuple(
        str(value) for value in process["required_runtime_capabilities"]
    )
    arena_capability, evaluator_abi, source_evaluator_abi = {
        ExecutionMode.EAGER: (
            EAGER_DIRECT_ARENA_RUNTIME_CAPABILITY,
            EAGER_DIRECT_TABLE_BINDING_ABI,
            SYMJIT_APPLICATION_ABI,
        ),
        ExecutionMode.RECURRENCE: (
            RECURRENCE_DIRECT_ARENA_RUNTIME_CAPABILITY,
            RECURRENCE_RUNTIME_LAYOUT_ABI,
            (
                SYMJIT_APPLICATION_ABI
                if cell.measurement.backend == "jit"
                else {
                    "cpp": SYMBOLICA_CPP_RUNTIME_CAPABILITY,
                    "asm": SYMBOLICA_ASM_RUNTIME_CAPABILITY,
                }.get(cell.measurement.backend)
            ),
        ),
    }.get(
        cell.measurement.execution_mode,
        (
            COMPILED_PLANE_ARENA_RUNTIME_CAPABILITY,
            (
                COMPILED_PLANE_DIRECT_APPLICATION_ABI
                if cell.measurement.backend == "jit"
                else NATIVE_COMPILED_DIRECT_APPLICATION_ABI
            ),
            (
                SYMJIT_APPLICATION_ABI
                if cell.measurement.backend == "jit"
                else NATIVE_COMPILED_DIRECT_APPLICATION_ABI
            ),
        ),
    )
    if arena_capability not in required_capabilities:
        raise RunnerError(
            f"report artifact process does not require {arena_capability!r}"
        )
    source_evaluator_runtime_capability = {
        "jit": SYMJIT_F64_RUNTIME_CAPABILITY,
        "cpp": SYMBOLICA_CPP_RUNTIME_CAPABILITY,
        "asm": SYMBOLICA_ASM_RUNTIME_CAPABILITY,
    }.get(cell.measurement.backend)
    if source_evaluator_runtime_capability is None:
        raise RunnerError(
            f"report runtime has unsupported backend {cell.measurement.backend!r}"
        )
    native = importlib.import_module("pyamplicol._rusticol")
    verify_native_module(native)
    native_digest = native.native_build_inputs_sha256()
    target = native.target_info()
    build_info = _active_build_info()
    if not isinstance(build_info, Mapping):
        raise RunnerError("report runtime has no exact candidate build identity")
    build_identity_fields = (
        "schema_version",
        "version",
        "candidate_fingerprint",
        "source_revision",
        "source_checkout",
        "native_build_inputs_sha256",
        "publishable",
    )
    candidate_build_identity = {
        field: build_info.get(field) for field in build_identity_fields
    }
    if (
        candidate_build_identity["source_revision"] != expected_source_revision
        or candidate_build_identity["native_build_inputs_sha256"] != native_digest
        or candidate_build_identity["publishable"] is not False
        or not all(
            candidate_build_identity[field] is not None
            for field in build_identity_fields
        )
    ):
        raise RunnerError(
            "report runtime candidate build does not match the final source revision"
        )
    if not isinstance(native_digest, str) or len(native_digest) != 64:
        raise RunnerError("native runtime build-input identity is invalid")
    native_path = Path(str(native.__file__)).resolve(strict=True)
    native_extension_sha256 = _sha256_file(native_path)
    package_roots = tuple(Path(str(path)) for path in pyamplicol.__path__)
    try:
        preimport_identity = established_preimport_runtime_identity()
        expected_package_tree = preimport_identity.get("python_package_tree")
        expected_native_identity = preimport_identity.get("native_extension")
        if (
            preimport_identity.get("kind") != "pyamplicol-preimport-runtime-identity-v1"
            or not isinstance(expected_package_tree, Mapping)
            or not isinstance(expected_native_identity, Mapping)
        ):
            raise RuntimeEvidenceError(
                "preimport runtime identity has an invalid contract"
            )
        package_tree = python_package_tree_identity(package_roots)
        if dict(package_tree) != dict(expected_package_tree):
            raise RuntimeEvidenceError(
                "report runtime package tree differs from its preimport identity"
            )
        loaded_origin_policy = loaded_pyamplicol_origin_policy(
            package_roots,
            native_extension=native_path,
            expected_package_identity=dict(expected_package_tree),
            expected_native_identity=dict(expected_native_identity),
        )
    except RuntimeEvidenceError as error:
        raise RunnerError(
            "report runtime Python namespace cannot be authenticated"
        ) from error
    candidate_build_identity_sha256 = hashlib.sha256(
        _canonical_json(candidate_build_identity)
    ).hexdigest()
    optimization_identity: dict[str, int] = {}
    direct_codegen_identity: dict[str, object] = {}
    source_jit_identity: dict[str, object] = {}
    if cell.measurement.backend == "jit":
        source_level = cell.measurement.jit_optimization_level
        if (
            isinstance(source_level, bool)
            or not isinstance(source_level, int)
            or source_level not in {0, 1, 2, 3}
        ):
            raise RunnerError(
                "JIT Arena report runtime has no valid source optimization level"
            )
        if (
            cell.measurement.execution_mode
            in {ExecutionMode.EAGER, ExecutionMode.RECURRENCE}
            and source_level != 2
        ):
            raise RunnerError(
                f"{cell.measurement.execution_mode.value} JIT Arena report "
                "runtime must use its authenticated portable O2 source"
            )
        optimization_identity["source_jit_optimization_level"] = source_level
        if cell.measurement.execution_mode is ExecutionMode.COMPILED:
            direct_codegen_identity = _authenticated_direct_codegen_identity(
                manifest,
                process_id=process_id,
                source_optimization_level=source_level,
            )
            optimization_identity["direct_codegen_optimization_level"] = int(
                direct_codegen_identity["optimization_level"]
            )
        elif cell.measurement.execution_mode is ExecutionMode.RECURRENCE:
            source_jit_identity = _authenticated_recurrence_source_identity(
                manifest,
                process_id=process_id,
                source_optimization_level=source_level,
            )
    elif cell.measurement.jit_optimization_level is not None:
        raise RunnerError(
            "non-JIT Arena report runtime unexpectedly declares a JIT "
            "optimization level"
        )
    return {
        "kind": "pyamplicol-report-runtime-identity-v1",
        "artifact_id": manifest.artifact_id,
        "loaded_artifact_id": loaded_artifact_id,
        "artifact_identity_match": True,
        "process_id": process_id,
        "execution_mode": expected_execution_mode,
        "loaded_execution_mode": loaded_execution_mode,
        "backend": cell.measurement.backend,
        "required_arena_capability": arena_capability,
        "expected_evaluator_abi": evaluator_abi,
        "expected_source_evaluator_abi": source_evaluator_abi,
        "expected_source_evaluator_runtime_capability": (
            source_evaluator_runtime_capability
        ),
        "process_required_runtime_capabilities": list(required_capabilities),
        "package_version": pyamplicol.__version__,
        "python_package_tree": package_tree,
        "loaded_module_origin_policy": loaded_origin_policy,
        "artifact_runtime_version": manifest.runtime["engine_version"],
        "source_revision": expected_source_revision,
        "candidate_build_identity": candidate_build_identity,
        "candidate_build_identity_sha256": candidate_build_identity_sha256,
        "native_build_inputs_sha256": native_digest,
        "native_extension": {
            "path": str(native_path),
            "sha256": native_extension_sha256,
            "package_version": str(native.package_version()),
        },
        "native_target": {
            "triple": str(target.triple),
            "cpu_features": [str(value) for value in target.cpu_features],
        },
        **optimization_identity,
        **(
            {"direct_codegen_identity": direct_codegen_identity}
            if direct_codegen_identity
            else {}
        ),
        **({"source_jit_identity": source_jit_identity} if source_jit_identity else {}),
    }


def _authenticated_process_execution(
    manifest: object,
    *,
    process_id: str,
) -> tuple[Mapping[str, object], str, str, Path, Sequence[object]]:
    runtime = getattr(manifest, "runtime", None)
    root = getattr(manifest, "root", None)
    payloads = getattr(manifest, "payloads", None)
    if (
        not isinstance(runtime, Mapping)
        or not isinstance(root, Path)
        or not isinstance(payloads, Sequence)
    ):
        raise RunnerError("artifact exposes no authenticated execution payload")
    relative = runtime.get("evaluator_manifest_path")
    if not isinstance(relative, str) or not relative:
        raise RunnerError("artifact runtime has no evaluator manifest path")
    evaluator_index, _ = _authenticated_json_payload(
        root,
        payloads,
        relative,
        expected_process_id=None,
        label="evaluator index",
    )
    if (
        evaluator_index.get("kind") != "pyamplicol-runtime-execution-set"
        or type(evaluator_index.get("schema_version")) is not int
        or evaluator_index.get("schema_version") != 3
    ):
        raise RunnerError("authenticated evaluator index has an invalid contract")
    entries = evaluator_index.get("processes")
    if isinstance(entries, (str, bytes)) or not isinstance(entries, Sequence):
        raise RunnerError("authenticated evaluator index has no process entries")
    matches = [
        entry
        for entry in entries
        if isinstance(entry, Mapping) and entry.get("process_id") == process_id
    ]
    if len(matches) != 1:
        raise RunnerError(
            f"authenticated evaluator index has no unique entry for {process_id!r}"
        )
    entry_relative = matches[0].get("manifest_path")
    if not isinstance(entry_relative, str) or not entry_relative:
        raise RunnerError("authenticated evaluator index entry has no manifest path")
    entry_logical = PurePosixPath(entry_relative)
    if (
        entry_logical.is_absolute()
        or not entry_logical.parts
        or any(part in {"", ".", ".."} for part in entry_logical.parts)
        or entry_logical.as_posix() != entry_relative
    ):
        raise RunnerError(
            "authenticated evaluator index entry has a noncanonical manifest path"
        )
    execution_relative = (PurePosixPath(relative).parent / entry_logical).as_posix()
    execution, actual_sha256 = _authenticated_json_payload(
        root,
        payloads,
        execution_relative,
        expected_process_id=process_id,
        label="process execution manifest",
    )
    return execution, execution_relative, actual_sha256, root, payloads


def _authenticated_direct_codegen_identity(
    manifest: object,
    *,
    process_id: str,
    source_optimization_level: int,
) -> dict[str, object]:
    """Observe fixed O3 lowering in one artifact-ID-bound process payload."""

    execution, execution_relative, actual_sha256, _, _ = (
        _authenticated_process_execution(manifest, process_id=process_id)
    )

    leaf_count = 0

    def walk(value: object) -> None:
        nonlocal leaf_count
        if isinstance(value, Mapping):
            arena = value.get("compiled_plane_arena")
            if isinstance(arena, Mapping):
                leaves = arena.get("leaves")
                if (
                    isinstance(leaves, (str, bytes))
                    or not isinstance(leaves, Sequence)
                    or not leaves
                ):
                    raise RunnerError(
                        "compiled plane-Arena descriptor has no authenticated leaves"
                    )
                for raw_leaf in leaves:
                    if not isinstance(raw_leaf, Mapping):
                        raise RunnerError(
                            "compiled plane-Arena descriptor has an invalid leaf"
                        )
                    if (
                        raw_leaf.get("optimization_level") != source_optimization_level
                        or raw_leaf.get("direct_codegen_optimization_level") != 3
                    ):
                        raise RunnerError(
                            "compiled plane-Arena leaf optimization identity drifted"
                        )
                    leaf_count += 1
            for child in value.values():
                walk(child)
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            for child in value:
                walk(child)

    walk(execution)
    if leaf_count == 0:
        raise RunnerError("execution payload contains no compiled plane-Arena leaves")
    return {
        "kind": "authenticated-compiled-plane-arena-direct-codegen-v1",
        "optimization_level": 3,
        "source_optimization_level": source_optimization_level,
        "leaf_count": leaf_count,
        "execution_manifest_path": execution_relative,
        "execution_manifest_sha256": actual_sha256,
    }


def _authenticated_recurrence_source_identity(
    manifest: object,
    *,
    process_id: str,
    source_optimization_level: int,
) -> dict[str, object]:
    """Bind recurrence JIT O2 to its exact source and direct-template payloads."""

    execution, execution_relative, execution_sha256, root, payloads = (
        _authenticated_process_execution(manifest, process_id=process_id)
    )
    if execution.get("kind") != "pyamplicol-runtime-recurrence-execution":
        raise RunnerError("authenticated recurrence execution kind is invalid")
    plan = execution.get("plan")
    if not isinstance(plan, Mapping):
        raise RunnerError("authenticated recurrence execution has no plan")
    expected_execution_abis = {
        "direct_template_abi": _RECURRENCE_DIRECT_TEMPLATE_ABI,
        "direct_backend_abi": _RECURRENCE_DIRECT_BACKEND_ABI,
    }
    for field, expected in expected_execution_abis.items():
        if execution.get(field) != expected or plan.get(field) != expected:
            raise RunnerError(
                f"authenticated recurrence execution/plan {field} drifted"
            )
    prepared_kernel_pack_digest = _lowercase_sha256(
        execution.get("prepared_kernel_pack_digest"),
        "authenticated recurrence execution prepared-kernel pack digest",
    )
    direct_template_catalog_digest = _lowercase_sha256(
        execution.get("direct_template_catalog_digest"),
        "authenticated recurrence execution direct-template catalog digest",
    )
    if (
        plan.get("prepared_kernel_pack_digest") != prepared_kernel_pack_digest
        or plan.get("direct_template_catalog_digest") != direct_template_catalog_digest
    ):
        raise RunnerError(
            "authenticated recurrence execution/plan source digests drifted"
        )
    kernel_pack = execution.get("kernel_pack")
    if not isinstance(kernel_pack, Mapping):
        raise RunnerError("authenticated recurrence execution has no kernel pack")
    pack_relative = kernel_pack.get("manifest_path")
    if not isinstance(pack_relative, str) or not pack_relative:
        raise RunnerError("authenticated recurrence execution has no kernel-pack path")
    payload_root = _canonical_recurrence_path(
        kernel_pack.get("payload_root"),
        "authenticated recurrence kernel payload root",
    )
    pack, pack_sha256 = _authenticated_json_payload(
        root,
        payloads,
        pack_relative,
        expected_process_id=None,
        label="recurrence kernel pack",
    )
    settings = pack.get("optimization_settings")
    direct_catalog = pack.get("recurrence_direct_template")
    if not isinstance(settings, Mapping) or not isinstance(direct_catalog, Mapping):
        raise RunnerError("authenticated recurrence kernel pack is incomplete")
    templates = direct_catalog.get("templates")
    if (
        pack.get("backend") != "jit"
        or settings.get("backend") != "jit"
        or settings.get("jit_optimization_level") != source_optimization_level
        or direct_catalog.get("abi") != _RECURRENCE_DIRECT_TEMPLATE_ABI
        or direct_catalog.get("backend") != "jit"
        or direct_catalog.get("backend_abi") != _RECURRENCE_DIRECT_BACKEND_ABI
        or direct_catalog.get("canonicalization_abi")
        != _RECURRENCE_DIRECT_CANONICALIZATION_ABI
        or direct_catalog.get("optimization_level") != source_optimization_level
        or direct_catalog.get("portable") is not True
        or isinstance(templates, (str, bytes))
        or not isinstance(templates, Sequence)
        or not templates
    ):
        raise RunnerError(
            "authenticated recurrence direct-template optimization identity drifted"
        )
    if (
        direct_catalog.get("prepared_kernel_pack_digest") != prepared_kernel_pack_digest
        or direct_catalog.get("catalog_digest") != direct_template_catalog_digest
    ):
        raise RunnerError(
            "authenticated recurrence execution and kernel-pack digests differ"
        )
    if _mapping_digest_without(direct_catalog, "catalog_digest") != (
        direct_template_catalog_digest
    ):
        raise RunnerError(
            "authenticated recurrence direct-template catalog digest is invalid"
        )

    evaluator_members = _authenticated_evaluator_container_members(
        manifest,
        root=root,
        payloads=payloads,
    )
    authenticated_source_payloads: dict[str, str] = {}

    def authenticate_source_payload(relative: str, *, label: str) -> str:
        cached = authenticated_source_payloads.get(relative)
        if cached is not None:
            return cached
        result = _authenticated_recurrence_source_payload(
            root,
            payloads,
            evaluator_members,
            relative,
            label=label,
        )
        authenticated_source_payloads[relative] = result
        return result

    kernels = pack.get("kernels")
    variants = pack.get("kernel_variants")
    if (
        isinstance(kernels, (str, bytes))
        or not isinstance(kernels, Sequence)
        or not kernels
        or isinstance(variants, (str, bytes))
        or not isinstance(variants, Sequence)
    ):
        raise RunnerError(
            "authenticated recurrence kernel pack has an invalid source inventory"
        )
    source_leaves_by_kernel: dict[int, set[tuple[str, str]]] = {}
    source_evaluator_leaf_count = 0
    for index, raw_kernel in enumerate(kernels):
        if not isinstance(raw_kernel, Mapping):
            raise RunnerError(
                f"authenticated recurrence kernel {index} is not an object"
            )
        kernel_id = raw_kernel.get("kernel_id")
        if (
            isinstance(kernel_id, bool)
            or not isinstance(kernel_id, int)
            or kernel_id < 0
            or kernel_id in source_leaves_by_kernel
        ):
            raise RunnerError(
                f"authenticated recurrence kernel {index} has an invalid kernel ID"
            )
        leaves = _authenticated_recurrence_source_leaves(
            raw_kernel.get("f64_evaluator_manifest"),
            source_optimization_level=source_optimization_level,
            payload_root=payload_root,
            authenticate_payload=authenticate_source_payload,
            context=f"authenticated recurrence kernel {index}",
        )
        source_evaluator_leaf_count += len(leaves)
        source_leaves_by_kernel[kernel_id] = set(leaves)
    for index, raw_variant in enumerate(variants):
        if not isinstance(raw_variant, Mapping):
            raise RunnerError(
                f"authenticated recurrence kernel variant {index} is not an object"
            )
        source_evaluator_leaf_count += len(
            _authenticated_recurrence_source_leaves(
                raw_variant.get("f64_evaluator_manifest"),
                source_optimization_level=source_optimization_level,
                payload_root=payload_root,
                authenticate_payload=authenticate_source_payload,
                context=f"authenticated recurrence kernel variant {index}",
            )
        )

    prepared_direct_template_count = 0
    for index, raw_template in enumerate(templates):
        if (
            not isinstance(raw_template, Mapping)
            or raw_template.get("abi") != _RECURRENCE_DIRECT_TEMPLATE_ABI
            or raw_template.get("backend") != "jit"
            or raw_template.get("portable") is not True
            or raw_template.get("optimization_level") != source_optimization_level
        ):
            raise RunnerError(
                f"authenticated recurrence direct-template {index} contract drifted"
            )
        if _mapping_digest_without(raw_template, "semantic_digest") != (
            raw_template.get("semantic_digest")
        ):
            raise RunnerError(
                f"authenticated recurrence direct-template {index} digest is invalid"
            )
        binding = raw_template.get("payload_binding")
        if not isinstance(binding, Mapping):
            raise RunnerError(
                f"authenticated recurrence direct-template {index} has no "
                "payload binding"
            )
        if (
            binding.get("abi") != _RECURRENCE_DIRECT_PAYLOAD_BINDING_ABI
            or _recurrence_payload_binding_digest(binding)
            != binding.get("payload_digest")
        ):
            raise RunnerError(
                f"authenticated recurrence direct-template {index} "
                "payload-binding contract is invalid"
            )
        binding_kind = binding.get("kind")
        if binding_kind == "rusticol-intrinsic":
            if (
                binding.get("source_application_abi") is not None
                or binding.get("direct_application_abi") is not None
                or binding.get("source_application_path") is not None
                or binding.get("source_application_sha256") is not None
                or binding.get("payload_paths") != []
                or not isinstance(binding.get("runtime_template"), str)
                or not binding.get("runtime_template")
            ):
                raise RunnerError(
                    f"authenticated recurrence direct-template {index} "
                    "intrinsic source contract is invalid"
                )
            continue
        if binding_kind != "prepared-direct-call":
            raise RunnerError(
                f"authenticated recurrence direct-template {index} is not executable"
            )
        prepared_direct_template_count += 1
        source_path = _canonical_recurrence_path(
            binding.get("source_application_path"),
            (f"authenticated recurrence direct-template {index} source application"),
        )
        source_sha256 = _lowercase_sha256(
            binding.get("source_application_sha256"),
            (
                f"authenticated recurrence direct-template {index} "
                "source application digest"
            ),
        )
        prepared_kernel_id = binding.get("prepared_kernel_id")
        if (
            binding.get("source_application_abi")
            != _RECURRENCE_JIT_SOURCE_APPLICATION_ABI
            or binding.get("direct_application_abi")
            != _RECURRENCE_JIT_DIRECT_APPLICATION_ABI
            or binding.get("payload_paths") != [source_path]
            or isinstance(prepared_kernel_id, bool)
            or not isinstance(prepared_kernel_id, int)
            or prepared_kernel_id < 0
            or (source_path, source_sha256)
            not in source_leaves_by_kernel.get(prepared_kernel_id, set())
        ):
            raise RunnerError(
                f"authenticated recurrence direct-template {index} source "
                "application is not bound to its prepared kernel"
            )
    if prepared_direct_template_count == 0 or source_evaluator_leaf_count == 0:
        raise RunnerError(
            "authenticated recurrence direct-template pack has no source "
            "application evidence"
        )
    return {
        "kind": "authenticated-recurrence-direct-template-source-v1",
        "optimization_level": source_optimization_level,
        "direct_template_count": len(templates),
        "prepared_direct_template_count": prepared_direct_template_count,
        "source_evaluator_leaf_count": source_evaluator_leaf_count,
        "source_application_abi": _RECURRENCE_JIT_SOURCE_APPLICATION_ABI,
        "direct_application_abi": _RECURRENCE_JIT_DIRECT_APPLICATION_ABI,
        "prepared_kernel_pack_digest": prepared_kernel_pack_digest,
        "direct_template_catalog_digest": direct_template_catalog_digest,
        "execution_manifest_path": execution_relative,
        "execution_manifest_sha256": execution_sha256,
        "kernel_pack_path": pack_relative,
        "kernel_pack_sha256": pack_sha256,
    }


def _lowercase_sha256(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RunnerError(f"{label} is not a lowercase SHA-256")
    return value


def _mapping_digest_without(value: Mapping[str, object], field: str) -> str:
    payload = dict(value)
    payload.pop(field, None)
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def _recurrence_payload_binding_digest(binding: Mapping[str, object]) -> str:
    if binding.get("kind") != "rusticol-intrinsic":
        return _mapping_digest_without(binding, "payload_digest")
    if binding.get("role") == "contribution":
        fields = (
            "abi",
            "destination_operation",
            "contribution_parent_permutation",
            "intrinsic_contract_digest",
            "kind",
            "role",
            "runtime_template",
            "scalar_input_count",
            "scalar_projections",
        )
        return hashlib.sha256(
            _canonical_json({field: binding.get(field) for field in fields})
        ).hexdigest()
    # Source/finalization intrinsics have an opaque Rusticol implementation
    # digest whose derivation includes semantic-catalog fields not duplicated
    # in the payload binding.  Its exact value remains transitively bound by
    # the recomputed template and catalog digests.
    return _lowercase_sha256(
        binding.get("payload_digest"),
        "authenticated recurrence intrinsic payload digest",
    )


def _canonical_recurrence_path(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise RunnerError(f"{label} path is missing")
    logical = PurePosixPath(value)
    if (
        logical.is_absolute()
        or not logical.parts
        or any(part in {"", ".", ".."} for part in logical.parts)
        or logical.as_posix() != value
    ):
        raise RunnerError(f"{label} path is not canonical")
    return value


def _authenticated_evaluator_container_members(
    manifest: object,
    *,
    root: Path,
    payloads: Sequence[object],
) -> Mapping[str, tuple[int, str]]:
    extensions = getattr(manifest, "extensions", None)
    if extensions is None:
        return {}
    if not isinstance(extensions, Mapping):
        raise RunnerError("artifact evaluator payload extensions are invalid")
    raw_container = extensions.get("evaluator_payload_container")
    if raw_container is None:
        return {}
    if not isinstance(raw_container, Mapping):
        raise RunnerError("artifact evaluator payload container is invalid")
    required = {
        "kind",
        "schema_version",
        "storage_abi",
        "path",
        "member_count",
        "unpacked_size_bytes",
        "index_sha256",
    }
    if (
        set(raw_container) != required
        or raw_container.get("kind") != "pyamplicol-evaluator-payload-container"
        or raw_container.get("schema_version") != 1
        or raw_container.get("storage_abi") != "pacbin-v1"
    ):
        raise RunnerError("artifact evaluator payload container contract drifted")
    member_count = raw_container.get("member_count")
    unpacked_size = raw_container.get("unpacked_size_bytes")
    index_sha256 = _lowercase_sha256(
        raw_container.get("index_sha256"),
        "artifact evaluator payload container index digest",
    )
    if (
        isinstance(member_count, bool)
        or not isinstance(member_count, int)
        or member_count < 0
        or isinstance(unpacked_size, bool)
        or not isinstance(unpacked_size, int)
        or unpacked_size < 0
    ):
        raise RunnerError("artifact evaluator payload container inventory is invalid")
    container_path = _canonical_recurrence_path(
        raw_container.get("path"),
        "artifact evaluator payload container",
    )
    data, _ = _authenticated_payload_bytes(
        root,
        payloads,
        container_path,
        expected_process_id=None,
        expected_role="evaluator-state",
        expected_media_type="application/octet-stream",
        label="evaluator payload container",
    )
    try:
        from pyamplicol.generation.evaluator_container import PacbinReader

        with PacbinReader.open(io.BytesIO(data), verify_payloads=True) as reader:
            index = reader.index
            members = tuple(reader.members)
    except (OSError, TypeError, ValueError) as error:
        raise RunnerError(
            "authenticated evaluator payload container is invalid"
        ) from error
    if (
        len(members) != member_count
        or sum(member.length for member in members) != unpacked_size
        or index.index_sha256 != index_sha256
    ):
        raise RunnerError(
            "authenticated evaluator payload container index metadata drifted"
        )
    return {
        member.logical_path: (int(member.kind), member.sha256) for member in members
    }


def _authenticated_recurrence_source_payload(
    root: Path,
    payloads: Sequence[object],
    evaluator_members: Mapping[str, tuple[int, str]],
    relative: str,
    *,
    label: str,
) -> str:
    canonical = _canonical_recurrence_path(relative, label)
    loose = [
        payload for payload in payloads if getattr(payload, "path", None) == canonical
    ]
    packed = evaluator_members.get(canonical)
    if len(loose) + (packed is not None) != 1:
        raise RunnerError(f"{label} payload is missing or ambiguous")
    if loose:
        _, sha256 = _authenticated_payload_bytes(
            root,
            payloads,
            canonical,
            expected_process_id=None,
            expected_role="evaluator-state",
            expected_media_type="application/octet-stream",
            label=label,
        )
        return sha256
    assert packed is not None
    kind, sha256 = packed
    if kind != 1:
        raise RunnerError(f"{label} is not a packed SymJIT application")
    return _lowercase_sha256(sha256, f"{label} packed member digest")


def _authenticated_recurrence_source_leaves(
    value: object,
    *,
    source_optimization_level: int,
    payload_root: str,
    authenticate_payload: object,
    context: str,
) -> list[tuple[str, str]]:
    if not isinstance(value, Mapping):
        raise RunnerError(f"{context} source evaluator is not an object")
    if value.get("kind") == "chunked-symbolica-evaluator":
        chunks = value.get("chunks")
        if (
            isinstance(chunks, (str, bytes))
            or not isinstance(chunks, Sequence)
            or not chunks
        ):
            raise RunnerError(f"{context} chunked source evaluator has no leaves")
        leaves: list[tuple[str, str]] = []
        for index, chunk in enumerate(chunks):
            leaves.extend(
                _authenticated_recurrence_source_leaves(
                    chunk,
                    source_optimization_level=source_optimization_level,
                    payload_root=payload_root,
                    authenticate_payload=authenticate_payload,
                    context=f"{context}.chunks[{index}]",
                )
            )
        return leaves
    if (
        value.get("kind") != "symjit-application-evaluator"
        or value.get("backend") != "jit"
        or value.get("runtime_capability") != _RECURRENCE_JIT_SOURCE_RUNTIME_CAPABILITY
        or value.get("application_abi") != _RECURRENCE_JIT_SOURCE_APPLICATION_ABI
        or value.get("optimization_level") != source_optimization_level
    ):
        raise RunnerError(f"{context} source evaluator leaf contract drifted")
    application_path = _canonical_recurrence_path(
        value.get("application_path"),
        f"{context} source evaluator application",
    )
    payload_path = (
        PurePosixPath(payload_root) / PurePosixPath(application_path)
    ).as_posix()
    if not callable(authenticate_payload):
        raise RunnerError("recurrence source payload authenticator is invalid")
    sha256 = authenticate_payload(  # type: ignore[operator]
        payload_path,
        label=f"{context} source evaluator application",
    )
    if not isinstance(sha256, str):
        raise RunnerError(f"{context} source evaluator digest is invalid")
    return [(application_path, sha256)]


def _authenticated_json_payload(
    root: Path,
    payloads: Sequence[object],
    relative: str,
    *,
    expected_process_id: str | None,
    label: str,
) -> tuple[Mapping[str, object], str]:
    """Read one declared JSON payload through one checked file descriptor."""

    data, actual_sha256 = _authenticated_payload_bytes(
        root,
        payloads,
        relative,
        expected_process_id=expected_process_id,
        expected_role="evaluator-manifest",
        expected_media_type="application/json",
        label=label,
    )
    try:
        value = json.loads(data)
    except json.JSONDecodeError as error:
        raise RunnerError(f"authenticated {label} is invalid JSON") from error
    if not isinstance(value, Mapping):
        raise RunnerError(f"authenticated {label} is not a JSON object")
    return value, actual_sha256


def _authenticated_payload_bytes(
    root: Path,
    payloads: Sequence[object],
    relative: str,
    *,
    expected_process_id: str | None,
    expected_role: str,
    expected_media_type: str,
    label: str,
) -> tuple[bytes, str]:
    """Read exact manifest-declared bytes through one no-follow descriptor."""

    logical = PurePosixPath(relative)
    if (
        logical.is_absolute()
        or not logical.parts
        or any(part in {"", ".", ".."} for part in logical.parts)
        or logical.as_posix() != relative
    ):
        raise RunnerError(f"{label} path is not canonical")
    matches = [
        payload for payload in payloads if getattr(payload, "path", None) == relative
    ]
    if len(matches) != 1:
        raise RunnerError(f"{label} payload is missing or ambiguous")
    payload = matches[0]
    expected_size = getattr(payload, "size_bytes", None)
    expected_sha256 = getattr(payload, "sha256", None)
    if (
        getattr(payload, "role", None) != expected_role
        or getattr(payload, "media_type", None) != expected_media_type
        or getattr(payload, "executable", None) is not False
        or getattr(payload, "process_id", None) != expected_process_id
        or isinstance(expected_size, bool)
        or not isinstance(expected_size, int)
        or expected_size < 0
        or not isinstance(expected_sha256, str)
        or len(expected_sha256) != 64
    ):
        raise RunnerError(f"{label} has an invalid artifact declaration")
    path = root.joinpath(*logical.parts)
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_BINARY", 0)
    )
    descriptor: int | None = None
    try:
        descriptor = os.open(path, flags)
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size != expected_size:
            raise RunnerError(f"{label} does not match its artifact declaration")
        with os.fdopen(descriptor, "rb", closefd=True) as stream:
            descriptor = None
            data = stream.read(expected_size + 1)
            after = os.fstat(stream.fileno())
    except OSError as error:
        raise RunnerError(f"cannot read authenticated {label}") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
    stable_fields = (
        "st_dev",
        "st_ino",
        "st_mode",
        "st_size",
        "st_mtime_ns",
        "st_ctime_ns",
    )
    if any(getattr(before, name) != getattr(after, name) for name in stable_fields):
        raise RunnerError(f"authenticated {label} changed while it was read")
    actual_sha256 = hashlib.sha256(data).hexdigest()
    if len(data) != expected_size or actual_sha256 != expected_sha256:
        raise RunnerError(f"{label} does not match the artifact manifest")
    return data, actual_sha256


def _authenticated_effective_config(artifact_path: Path) -> Mapping[str, object]:
    """Return the exact effective TOML declared by a generated artifact."""

    from pyamplicol.artifacts import load_manifest

    manifest = load_manifest(artifact_path, verify_payloads=False)
    root = getattr(manifest, "root", None)
    payloads = getattr(manifest, "payloads", None)
    configuration = getattr(manifest, "configuration", None)
    if (
        not isinstance(root, Path)
        or not isinstance(payloads, Sequence)
        or not isinstance(configuration, Mapping)
    ):
        raise RunnerError("generated artifact exposes no effective configuration")
    relative = configuration.get("effective_path")
    if not isinstance(relative, str) or not relative:
        raise RunnerError("generated artifact has no effective configuration path")
    data, _ = _authenticated_payload_bytes(
        root,
        payloads,
        relative,
        expected_process_id=None,
        expected_role="configuration-effective",
        expected_media_type="application/toml",
        label="effective configuration",
    )
    try:
        value = tomllib.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise RunnerError(
            "authenticated effective configuration is invalid TOML"
        ) from error
    if not isinstance(value, Mapping):
        raise RunnerError("authenticated effective configuration is not a TOML table")
    return value


def _selector_kwargs(
    cell: CellSpec,
    contract: SelectorContract | None,
) -> dict[str, tuple[str, ...] | None]:
    if cell.measurement.accuracy is not Accuracy.LC:
        return {"helicities": None, "color_flows": None}
    if contract is None:
        raise RunnerError("LC measurement requires a selector contract")
    if cell.workload is Workload.SELECTED_FLOW:
        return {
            "helicities": None,
            "color_flows": contract.selected_color_flow_ids,
        }
    if cell.workload is Workload.ALL_FLOW:
        return {
            "helicities": contract.all_flow_helicity_ids,
            "color_flows": None,
        }
    raise RunnerError("LC measurement has an invalid workload")


def derive_selector_contract(
    runtime: RuntimeLike,
    points: object,
) -> SelectorContract:
    """Select one deterministic flow and one nonzero fixed helicity."""

    physics = runtime.physics
    color_flows = tuple(getattr(physics, "color_flows", ()))
    helicities = tuple(getattr(physics, "helicities", ()))
    particles = tuple(getattr(physics, "external_particles", ()))
    if not color_flows or not helicities or not particles:
        raise RunnerError("LC selector derivation requires complete physical axes")

    selected_flow = color_flows[0]
    resolved = runtime.evaluate_resolved(points, color_flows=(str(selected_flow.id),))
    values = getattr(resolved, "values", ())
    chosen_index: int | None = None
    for helicity_index, _helicity in enumerate(helicities):
        magnitude = sum(
            abs(complex(point[helicity_index][color_index]))
            for point in values
            for color_index in range(len(point[helicity_index]))
        )
        if magnitude > ABSOLUTE_TOLERANCE:
            chosen_index = helicity_index
            break
    if chosen_index is None:
        raise RunnerError("no nonzero fixed-helicity selector exists at report point")
    helicity = helicities[chosen_index]
    labels = tuple(int(particle.label) for particle in particles)
    states = tuple(int(value) for value in helicity.values)
    if len(labels) != len(states):
        raise RunnerError(
            "helicity source-state axis does not match external particles"
        )
    return SelectorContract(
        selected_color_flow_ids=(str(selected_flow.id),),
        selected_color_words=(tuple(int(label) for label in selected_flow.word),),
        all_flow_helicity_ids=(str(helicity.id),),
        all_flow_source_helicities=tuple(zip(labels, states, strict=True)),
        point_digest=point_digest(points),
    )


def validate_selector_contract(
    runtime: RuntimeLike,
    contract: SelectorContract,
    points: object,
) -> None:
    if point_digest(points) != contract.point_digest:
        raise RunnerError("selector contract and measurement point differ")
    physics = runtime.physics
    colors = {
        str(flow.id): tuple(int(label) for label in flow.word)
        for flow in getattr(physics, "color_flows", ())
    }
    for identifier, word in zip(
        contract.selected_color_flow_ids,
        contract.selected_color_words,
        strict=True,
    ):
        if colors.get(identifier) != word:
            raise RunnerError(
                f"artifact does not expose selected physical flow {identifier!r}"
            )
    helicities = {
        str(item.id): tuple(int(value) for value in item.values)
        for item in getattr(physics, "helicities", ())
    }
    labels = tuple(
        int(particle.label) for particle in getattr(physics, "external_particles", ())
    )
    expected_states = dict(contract.all_flow_source_helicities)
    expected = tuple(expected_states[label] for label in labels)
    for identifier in contract.all_flow_helicity_ids:
        if helicities.get(identifier) != expected:
            raise RunnerError(
                f"artifact does not expose selected physical helicity {identifier!r}"
            )


def resolved_sum_validation(
    runtime: RuntimeLike,
    points: object,
    *,
    cell: CellSpec,
    selector_contract: SelectorContract | None,
) -> dict[str, object]:
    selectors = _selector_kwargs(cell, selector_contract)
    optimized = runtime.evaluate(points, **selectors)
    resolved = runtime.evaluate_resolved(points, **selectors)
    totals = tuple(resolved.total())
    if len(optimized) != len(totals):
        raise RunnerError("optimized and resolved evaluations have different lengths")
    maximum_absolute = 0.0
    maximum_relative = 0.0
    for optimized_value, resolved_value in zip(optimized, totals, strict=True):
        absolute = abs(complex(optimized_value) - complex(resolved_value))
        relative = absolute / max(abs(complex(optimized_value)), 1.0e-300)
        maximum_absolute = max(maximum_absolute, absolute)
        maximum_relative = max(maximum_relative, relative)
    passed = (
        maximum_absolute <= ABSOLUTE_TOLERANCE or maximum_relative <= RELATIVE_TOLERANCE
    )
    return {
        "status": (
            ResultStatus.OK.value if passed else ResultStatus.VALIDATION_FAILED.value
        ),
        "maximum_absolute_difference": maximum_absolute,
        "maximum_relative_difference": maximum_relative,
        "relative_tolerance": RELATIVE_TOLERANCE,
        "absolute_tolerance": ABSOLUTE_TOLERANCE,
    }


def pointwise_validation(
    candidate: float,
    baseline: float,
    *,
    relative_tolerance: float = RELATIVE_TOLERANCE,
    absolute_tolerance: float = ABSOLUTE_TOLERANCE,
) -> dict[str, object]:
    if relative_tolerance < 0.0 or absolute_tolerance < 0.0:
        raise ValueError("pointwise tolerances must be non-negative")
    absolute = abs(candidate - baseline)
    relative = absolute / max(abs(baseline), 1.0e-300)
    passed = absolute <= absolute_tolerance or relative <= relative_tolerance
    return {
        "status": (
            ResultStatus.OK.value if passed else ResultStatus.VALIDATION_FAILED.value
        ),
        "candidate": candidate,
        "baseline": baseline,
        "absolute_difference": absolute,
        "relative_difference": relative,
        "relative_tolerance": relative_tolerance,
        "absolute_tolerance": absolute_tolerance,
    }


def _benchmark_measurement(
    benchmark: object,
    *,
    matrix_element: float,
) -> dict[str, object]:
    uncertainty = benchmark.uncertainty
    return {
        "status": ResultStatus.OK.value,
        "wall_seconds_per_point": float(benchmark.wall_time_per_point),
        "execution_seconds_per_point": (
            None
            if benchmark.evaluator_time_per_point is None
            else float(benchmark.evaluator_time_per_point)
        ),
        "matrix_element": matrix_element,
        "sample_count": int(benchmark.sample_count),
        "standard_error_seconds_per_point": float(uncertainty.standard_error),
        "relative_standard_error": float(uncertainty.relative_standard_error),
    }


def profile_runtime(
    runtime: RuntimeLike,
    points: object,
    *,
    cell: CellSpec,
    benchmark_config: object,
    selector_contract: SelectorContract | None,
) -> dict[str, object]:
    from pyamplicol.api import BenchmarkRunner

    validate_runtime_contract(cell, runtime)
    if selector_contract is not None:
        validate_selector_contract(runtime, selector_contract, points)
    selectors = _selector_kwargs(cell, selector_contract)
    values = runtime.evaluate(points, **selectors)
    if not values:
        raise RunnerError("runtime returned no matrix elements")
    selected_config = replace(
        benchmark_config,
        helicity_ids=tuple(selectors["helicities"] or ()),
        color_flow_ids=tuple(selectors["color_flows"] or ()),
    )
    benchmark = BenchmarkRunner(selected_config).run(runtime, points=points)
    result = _benchmark_measurement(
        benchmark,
        matrix_element=_real_nonnegative(values[0]),
    )
    result["resolved_sum_validation"] = resolved_sum_validation(
        runtime,
        points,
        cell=cell,
        selector_contract=selector_contract,
    )
    if result["resolved_sum_validation"]["status"] != ResultStatus.OK.value:
        result["status"] = ResultStatus.VALIDATION_FAILED.value
    return result


def runtime_validation_points(runtime: object) -> object:
    backend = getattr(runtime, "_backend", None)
    operation = getattr(backend, "validation_momenta", None)
    if not callable(operation):
        raise RunnerError("artifact does not retain deterministic validation momenta")
    points = operation()
    if points is None:
        raise RunnerError("artifact validation momenta are unavailable")
    return points


def _single_process_id(artifact_path: Path, fallback: str) -> str:
    manifest_path = artifact_path / "artifact.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        processes = manifest["processes"]
        if isinstance(processes, list) and len(processes) == 1:
            identifier = processes[0]["id"]
            if isinstance(identifier, str) and identifier:
                return identifier
    except (KeyError, OSError, TypeError, json.JSONDecodeError):
        pass
    return fallback


def generate_artifact(
    cell: CellSpec,
    destination: Path,
    *,
    settings: RunnerSettings,
    repo_root: Path,
    prepared_model_path: Path | None = None,
) -> GeneratedArtifact:
    """Generate one complete-coverage artifact and time process generation only."""

    from pyamplicol.api import Generator, ModelSource
    from pyamplicol.config import Action
    from pyamplicol.config.resolver import config_to_dict, resolve_config

    values = config_values(cell, settings, repo_root=repo_root)
    resolution = resolve_config(values, action=Action.GENERATE, base_dir=repo_root)
    assert cell.measurement.model is not None
    source_path = _model_source_path(repo_root, cell.measurement.model)
    source = (
        ModelSource.built_in_sm()
        if source_path is None
        else ModelSource.from_path(source_path)
    )
    model_started = time.perf_counter()
    prepared_execution = cell.measurement.execution_mode in {
        ExecutionMode.EAGER,
        ExecutionMode.RECURRENCE,
    }
    if prepared_model_path is not None:
        if not prepared_model_path.is_file():
            raise RunnerError(f"prepared model does not exist: {prepared_model_path}")
        model = ModelSource.from_path(prepared_model_path)
        preparation_reused = True
    elif prepared_execution and cell.measurement.model is ModelKey.BUILTIN_SM:
        # Omitting the explicit model lets the generation service select the
        # validated wheel-owned built-in-SM JIT O2 prepared pack.
        model = None
        preparation_reused = True
    elif prepared_execution:
        raise RunnerError(
            f"{cell.measurement.model.value} {cell.measurement.execution_mode.value} "
            "generation requires a prepared model path"
        )
    else:
        model = source.compile(
            cache_dir=settings.model_cache_dir,
            use_cache=True,
            require_supported=True,
        )
        preparation_reused = False
    model_seconds = time.perf_counter() - model_started
    generation_started = time.perf_counter()
    Generator(resolution).generate(
        cell.process,
        destination,
        model=model,
        mode="replace",
    )
    generation_seconds = time.perf_counter() - generation_started
    effective_config = _authenticated_effective_config(destination)
    return GeneratedArtifact(
        path=destination,
        process_id=_single_process_id(destination, cell.process),
        generation_seconds=generation_seconds,
        model_preparation_seconds=model_seconds,
        model_preparation_reused=preparation_reused,
        requested_config=config_to_dict(resolution.requested),
        effective_config=effective_config,
    )


def provenance_payload() -> dict[str, object]:
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }


__all__ = [
    "ABSOLUTE_TOLERANCE",
    "INDEPENDENT_RELATIVE_TOLERANCE",
    "RELATIVE_TOLERANCE",
    "GeneratedArtifact",
    "RunnerError",
    "RunnerSettings",
    "SelectorContract",
    "config_values",
    "derive_selector_contract",
    "generate_artifact",
    "point_digest",
    "pointwise_validation",
    "profile_runtime",
    "provenance_payload",
    "resolved_sum_validation",
    "runtime_identity_payload",
    "runtime_validation_points",
    "validate_artifact_contract",
    "validate_runtime_contract",
    "validate_selector_contract",
]
