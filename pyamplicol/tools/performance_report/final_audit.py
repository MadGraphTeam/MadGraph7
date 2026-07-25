# SPDX-License-Identifier: 0BSD
"""Final-SHA structural and numerical audit for checked-in report measurements."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import io
import json
import math
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import tomllib
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from importlib.machinery import EXTENSION_SUFFIXES
from pathlib import Path, PurePosixPath
from typing import Any

from .cache import digest_json, reset_entry
from .catalog import REPORT_CATALOG, ReportCatalog
from .measurement import shared_validation_points
from .models import (
    Accuracy,
    CellSpec,
    ExecutionMode,
    ModelKey,
    ResultStatus,
    Workload,
)
from .publication import publication_measurement_matches_current
from .render import render_all_tables
from .runner import (
    ABSOLUTE_TOLERANCE,
    INDEPENDENT_RELATIVE_TOLERANCE,
    RELATIVE_TOLERANCE,
    SelectorContract,
    point_digest,
    runtime_validation_points,
    validate_runtime_contract,
    validate_selector_contract,
)
from .runtime_evidence import (
    RuntimeEvidenceError,
    established_preimport_runtime_identity,
    loaded_pyamplicol_origin_policy,
    native_extension_in_package,
    preimport_python_runtime_identity,
    python_package_tree_identity,
    source_only_bytecode_policy,
)
from .service import ReportPaths, ReportService, validate_profile_name

_GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_EXPECTED_N4_CELL_COUNT = 742
_LOADED_ORIGIN_OBSERVATION_FIELDS = frozenset(
    {
        "observed_module_count",
        "observations",
        "observations_sha256",
    }
)
_ARENA_CAPABILITY = {
    ExecutionMode.COMPILED: "compiled-plane-arena-v1",
    ExecutionMode.EAGER: "eager-direct-arena-v1",
    ExecutionMode.RECURRENCE: ("rusticol.recurrence-direct-arena.complex-f64.v1"),
}
_COMPILED_DIRECT_ABI = "symjit-direct-application-storage-v3"
_NATIVE_COMPILED_DIRECT_ABI = "pyamplicol-native-compiled-direct-application-v1"
_SYMJIT_APPLICATION_ABI = "symjit-application-storage-v3"
_EAGER_NATIVE_APPLICATION_ABI = "pyamplicol-eager-native-direct-table-v1"
_EAGER_DIRECT_CAPABILITY = "eager-direct-arena-v1"
_EAGER_DIRECT_DESCRIPTOR_ABI = "symjit-direct-table-descriptor-v1"
_EAGER_DIRECT_BINDING_ABI = "symjit-direct-table-binding-v2"
_RECURRENCE_DIRECT_TEMPLATE_ABI = "pyamplicol-recurrence-direct-template-v1"
_RECURRENCE_DIRECT_BACKEND_ABI = "rusticol.recurrence-direct-backend.v1"
_RECURRENCE_DIRECT_CANONICALIZATION_ABI = "pyamplicol-canonical-json-v1"
_RECURRENCE_DIRECT_PAYLOAD_BINDING_ABI = (
    "pyamplicol-recurrence-direct-payload-binding-v1"
)
_RECURRENCE_JIT_DIRECT_APPLICATION_ABI = "symjit-direct-application-storage-v1"
_SOURCE_RUNTIME_CAPABILITY = {
    "jit": "symjit.application.complex-f64.v1",
    "cpp": "symbolica.compiled-cpp.complex-f64.v1",
    "asm": "symbolica.compiled-asm.complex-f64.v1",
}
_MODEL_SOURCE_PATH = {
    ModelKey.UFO_SM: "src/pyamplicol/assets/models/json/sm/sm.json",
    ModelKey.SCALAR_CONTACT: ("src/pyamplicol/assets/models/json/scalars/scalars.json"),
    ModelKey.SCALAR_GRAVITY: (
        "src/pyamplicol/assets/models/json/scalar_gravity/scalar_gravity.json"
    ),
}
_TEX_INPUT_RE = re.compile(r"\\(?:input|include)\s*\{([^{}]+)\}")
_REPORT_PROFILE_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}")
_PUBLICATION_MEMBER_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
_PORTABLE_ARTIFACT_ROOT = "${PYAMPLICOL_REPORT_ARTIFACT_ROOT}"
_PUBLICATION_LINEAGE_KIND = "pyamplicol-report-publication-lineage-v1"
_EXECUTION_TIMING_ABI = "pyamplicol-report-execution-timing-v1"
_COMPILED_ARENA_EXECUTION_TIME_SOURCE = (
    "runtime_profile_core_compiled_direct_arena_orchestration_time"
)
_PAIRED_TIMING_SAMPLE_CONTRACT = "paired_unprofiled_headline_profiled_attribution_v1"
_EXECUTION_TIMING_FIELDS = frozenset(
    {
        "abi",
        "status",
        "ratio_eligible",
        "raw_seconds_per_point",
        "source",
        "compiled_direct_arena_active",
        "sample_count",
        "native_profile_points_per_sample",
        "sample_contract",
    }
)


class FinalAuditError(RuntimeError):
    """The final report evidence does not satisfy its publication contract."""


@dataclass(frozen=True, slots=True)
class ArtifactEvidence:
    """Identity proven directly from one validated process artifact."""

    artifact_id: str
    process_id: str
    runtime_version: str
    runtime_capabilities: tuple[str, ...]
    execution_manifest_path: str
    execution_manifest_sha256: str
    execution_mode: str
    arena_record_count: int
    direct_leaf_count: int
    effective_config: Mapping[str, object]
    source_jit_identity: Mapping[str, object] | None


@dataclass(frozen=True, slots=True)
class _ArtifactReference:
    cell: CellSpec
    measurement: Mapping[str, object]
    path: Path
    process_id: str


@dataclass(frozen=True, slots=True)
class _ReplayObservation:
    matrix_element: float
    resolved_maximum_absolute: float
    resolved_maximum_relative: float


ArtifactAuditor = Callable[[CellSpec, Path, str], ArtifactEvidence]
RuntimeLoader = Callable[[Path, str], object]
SourceAuditor = Callable[[Path, str], Mapping[str, object] | None]
PdfAuditor = Callable[[ReportService], Mapping[str, object]]
RuntimeAuditor = Callable[[str, Path], Mapping[str, object]]


def _mapping(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise FinalAuditError(f"{context} must be an object")
    return value


def _sequence(value: object, context: str) -> Sequence[object]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise FinalAuditError(f"{context} must be an array")
    return value


def _finite_number(value: object, context: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FinalAuditError(f"{context} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise FinalAuditError(f"{context} must be finite")
    if nonnegative and result < 0.0:
        raise FinalAuditError(f"{context} must be non-negative")
    return result


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_relative_path(relative: object, context: str) -> PurePosixPath:
    if not isinstance(relative, str) or not relative:
        raise FinalAuditError(f"{context} path is missing")
    logical = PurePosixPath(relative)
    if (
        logical.is_absolute()
        or not logical.parts
        or any(part in {"", ".", ".."} for part in logical.parts)
        or logical.as_posix() != relative
    ):
        raise FinalAuditError(f"{context} path is not canonical")
    return logical


def _authenticated_payload_bytes(
    artifact: Path,
    manifest: object,
    relative: object,
    *,
    role: str,
    media_type: str,
    process_id: str | None,
    context: str,
) -> tuple[bytes, str, str]:
    """Read one manifest-declared payload through one checked descriptor."""

    logical = _canonical_relative_path(relative, context)
    canonical = logical.as_posix()
    payloads = getattr(manifest, "payloads", None)
    if not isinstance(payloads, Sequence):
        raise FinalAuditError(f"{context} has no artifact payload inventory")
    matches = [
        payload for payload in payloads if getattr(payload, "path", None) == canonical
    ]
    if len(matches) != 1:
        raise FinalAuditError(f"{context} payload is missing or ambiguous")
    payload = matches[0]
    expected_size = getattr(payload, "size_bytes", None)
    expected_sha256 = getattr(payload, "sha256", None)
    if (
        getattr(payload, "role", None) != role
        or getattr(payload, "media_type", None) != media_type
        or getattr(payload, "process_id", None) != process_id
        or getattr(payload, "executable", None) is not False
        or isinstance(expected_size, bool)
        or not isinstance(expected_size, int)
        or expected_size < 0
        or not isinstance(expected_sha256, str)
        or _SHA256_RE.fullmatch(expected_sha256) is None
    ):
        raise FinalAuditError(f"{context} has an invalid artifact declaration")
    path = artifact.joinpath(*logical.parts)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor: int | None = None
    try:
        descriptor = os.open(path, flags)
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size != expected_size:
            raise FinalAuditError(f"{context} does not match its artifact declaration")
        with os.fdopen(descriptor, "rb", closefd=True) as stream:
            descriptor = None
            data = stream.read(expected_size + 1)
    except OSError as error:
        raise FinalAuditError(f"cannot read authenticated {context}") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
    actual_sha256 = hashlib.sha256(data).hexdigest()
    if len(data) != expected_size or actual_sha256 != expected_sha256:
        raise FinalAuditError(f"{context} does not match the artifact manifest")
    return data, canonical, actual_sha256


def _authenticated_json_payload(
    artifact: Path,
    manifest: object,
    relative: object,
    *,
    process_id: str | None,
    context: str,
) -> tuple[dict[str, Any], str, str]:
    data, canonical, sha256 = _authenticated_payload_bytes(
        artifact,
        manifest,
        relative,
        role="evaluator-manifest",
        media_type="application/json",
        process_id=process_id,
        context=context,
    )
    try:
        value = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FinalAuditError(f"authenticated {context} is invalid JSON") from error
    if not isinstance(value, dict):
        raise FinalAuditError(f"authenticated {context} must be an object")
    return value, canonical, sha256


def _authenticated_effective_config(
    artifact: Path,
    manifest: object,
) -> Mapping[str, object]:
    configuration = _mapping(
        getattr(manifest, "configuration", None),
        "artifact.configuration",
    )
    data, _canonical, _sha256 = _authenticated_payload_bytes(
        artifact,
        manifest,
        configuration.get("effective_path"),
        role="configuration-effective",
        media_type="application/toml",
        process_id=None,
        context="effective configuration",
    )
    try:
        value = tomllib.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise FinalAuditError(
            "authenticated effective configuration is invalid TOML"
        ) from error
    return _mapping(value, "authenticated effective configuration")


def _python_package_tree_identity(
    package_roots: Path | Sequence[Path],
) -> dict[str, object]:
    """Translate exact-runtime evidence failures into final-audit failures."""

    try:
        return python_package_tree_identity(package_roots)
    except RuntimeEvidenceError as error:
        raise FinalAuditError(
            "installed pyamplicol namespace cannot be authenticated"
        ) from error


def _validate_python_package_tree_identity(
    value: object,
    *,
    context: str,
) -> Mapping[str, object]:
    identity = _mapping(value, context)
    root = identity.get("root")
    roots = identity.get("roots")
    file_count = identity.get("file_count")
    total_bytes = identity.get("total_bytes")
    digest = identity.get("sha256")
    bytecode_policy = identity.get("bytecode_policy")
    if (
        identity.get("kind") != "pyamplicol-python-package-tree-v2"
        or not isinstance(root, str)
        or not root
        or not Path(root).is_absolute()
        or isinstance(roots, (str, bytes))
        or not isinstance(roots, Sequence)
        or not roots
        or any(
            not isinstance(item, str) or not Path(item).is_absolute() for item in roots
        )
        or len(set(roots)) != len(roots)
        or roots[0] != root
        or isinstance(file_count, bool)
        or not isinstance(file_count, int)
        or file_count < 1
        or isinstance(total_bytes, bool)
        or not isinstance(total_bytes, int)
        or total_bytes < 0
        or not isinstance(digest, str)
        or _SHA256_RE.fullmatch(digest) is None
        or identity.get("member_set_stable") is not True
        or identity.get("namespace_bound_to_root_fd") is not True
        or not isinstance(bytecode_policy, Mapping)
        or bytecode_policy.get("kind") != "pyamplicol-source-only-bytecode-policy-v1"
        or bytecode_policy.get("dont_write_bytecode") is not True
        or bytecode_policy.get("external_pycache_prefix_absent") is not True
        or bytecode_policy.get("package_local_bytecode_eligible") is not False
        or bytecode_policy.get("isolated_startup") is not True
        or bytecode_policy.get("site_initialization") is not False
        or bytecode_policy.get("python_environment_ignored_at_startup") is not True
    ):
        raise FinalAuditError(f"{context} is not a valid package-tree identity")
    return identity


def _validate_loaded_origin_policy(
    value: object,
    *,
    context: str,
) -> Mapping[str, object]:
    policy = _mapping(value, context)
    count = policy.get("observed_module_count")
    observations = policy.get("observations")
    if (
        policy.get("kind") != "pyamplicol-loaded-module-origin-policy-v1"
        or policy.get("all_loaded_origins_authenticated") is not True
        or policy.get("native_image_origin_bound") is not True
        or policy.get("loaded_bytecode_eligible") is not False
        or isinstance(count, bool)
        or not isinstance(count, int)
        or count < 1
        or isinstance(observations, (str, bytes))
        or not isinstance(observations, Sequence)
        or len(observations) != count
        or _SHA256_RE.fullmatch(str(policy.get("observations_sha256"))) is None
        or digest_json(list(observations)) != policy.get("observations_sha256")
    ):
        raise FinalAuditError(f"{context} is not a valid loaded-origin policy")
    return policy


def _stable_runtime_identity(identity: Mapping[str, object]) -> dict[str, object]:
    stable = dict(identity)
    raw_policy = stable.get("loaded_module_origin_policy")
    if isinstance(raw_policy, Mapping):
        stable["loaded_module_origin_policy"] = {
            field: value
            for field, value in raw_policy.items()
            if field not in _LOADED_ORIGIN_OBSERVATION_FIELDS
        }
    return stable


def _audit_runtime_identity_postflight(
    provenance: Mapping[str, object],
    identity: Mapping[str, object],
    *,
    context: str,
) -> None:
    stable_digest = digest_json(_stable_runtime_identity(identity))
    if provenance.get("runtime_identity_stable_sha256") != stable_digest:
        raise FinalAuditError(
            f"{context}.runtime_identity_stable_sha256 does not match"
        )
    if provenance.get("runtime_identity_postflight_stable_sha256") != stable_digest:
        raise FinalAuditError(
            f"{context}.runtime_identity postflight stable SHA-256 differs"
        )
    if provenance.get("runtime_identity_postflight_match") is not True:
        raise FinalAuditError(
            f"{context}.runtime_identity_postflight_match is not true"
        )

    initial_policy = _validate_loaded_origin_policy(
        identity.get("loaded_module_origin_policy"),
        context=f"{context}.runtime_identity.loaded_module_origin_policy",
    )
    postflight_policy = _validate_loaded_origin_policy(
        provenance.get("runtime_identity_postflight_loaded_module_origin_policy"),
        context=(f"{context}.runtime_identity_postflight_loaded_module_origin_policy"),
    )
    stable_initial_policy = {
        field: value
        for field, value in initial_policy.items()
        if field not in _LOADED_ORIGIN_OBSERVATION_FIELDS
    }
    stable_postflight_policy = {
        field: value
        for field, value in postflight_policy.items()
        if field not in _LOADED_ORIGIN_OBSERVATION_FIELDS
    }
    if stable_postflight_policy != stable_initial_policy:
        raise FinalAuditError(f"{context} runtime postflight origin policy changed")
    initial_observations = _sequence(
        initial_policy.get("observations"),
        f"{context}.runtime_identity.loaded_module_origin_policy.observations",
    )
    postflight_observations = _sequence(
        postflight_policy.get("observations"),
        (
            f"{context}."
            "runtime_identity_postflight_loaded_module_origin_policy."
            "observations"
        ),
    )
    postflight_keys = {
        json.dumps(
            observation,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        for observation in postflight_observations
    }
    if any(
        json.dumps(
            observation,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        not in postflight_keys
        for observation in initial_observations
    ):
        raise FinalAuditError(
            f"{context} runtime postflight lost a loaded-module origin"
        )


def _canonical_status(
    absolute: float,
    relative: float,
    *,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> str:
    return (
        ResultStatus.OK.value
        if absolute <= absolute_tolerance or relative <= relative_tolerance
        else ResultStatus.VALIDATION_FAILED.value
    )


def _expected_pointwise_tolerance(cell: CellSpec) -> float:
    if cell.dataset_id.startswith("matrix_recurrence_") or cell.dataset_id.startswith(
        "z_"
    ):
        return INDEPENDENT_RELATIVE_TOLERANCE
    return RELATIVE_TOLERANCE


def _same_float(observed: float, expected: float) -> bool:
    return math.isclose(observed, expected, rel_tol=0.0, abs_tol=0.0)


def _audit_pointwise(
    raw: object,
    *,
    context: str,
    expected_candidate: float,
    expected_baseline: float,
    expected_relative_tolerance: float,
) -> None:
    record = _mapping(raw, context)
    candidate = _finite_number(record.get("candidate"), f"{context}.candidate")
    baseline = _finite_number(record.get("baseline"), f"{context}.baseline")
    absolute = _finite_number(
        record.get("absolute_difference"),
        f"{context}.absolute_difference",
        nonnegative=True,
    )
    relative = _finite_number(
        record.get("relative_difference"),
        f"{context}.relative_difference",
        nonnegative=True,
    )
    relative_tolerance = _finite_number(
        record.get("relative_tolerance"),
        f"{context}.relative_tolerance",
        nonnegative=True,
    )
    absolute_tolerance = _finite_number(
        record.get("absolute_tolerance"),
        f"{context}.absolute_tolerance",
        nonnegative=True,
    )
    recomputed_absolute = abs(candidate - baseline)
    recomputed_relative = recomputed_absolute / max(abs(baseline), 1.0e-300)
    expected_status = _canonical_status(
        recomputed_absolute,
        recomputed_relative,
        absolute_tolerance=absolute_tolerance,
        relative_tolerance=relative_tolerance,
    )
    checks = (
        (
            _same_float(candidate, expected_candidate),
            "candidate does not equal measurement.matrix_element",
        ),
        (
            _same_float(baseline, expected_baseline),
            "baseline does not equal the canonical baseline measurement",
        ),
        (
            _same_float(absolute, recomputed_absolute),
            "absolute difference was not recomputed correctly",
        ),
        (
            _same_float(relative, recomputed_relative),
            "relative difference was not recomputed correctly",
        ),
        (
            _same_float(relative_tolerance, expected_relative_tolerance),
            "relative tolerance is not the catalog contract",
        ),
        (
            _same_float(absolute_tolerance, ABSOLUTE_TOLERANCE),
            "absolute tolerance is not the report contract",
        ),
        (
            record.get("status") == expected_status == ResultStatus.OK.value,
            "stored or recomputed status is not ok",
        ),
    )
    failures = [message for passed, message in checks if not passed]
    if failures:
        raise FinalAuditError(f"{context}: " + "; ".join(failures))


def _audit_resolved_sum(raw: object, *, context: str) -> None:
    record = _mapping(raw, context)
    absolute = _finite_number(
        record.get("maximum_absolute_difference"),
        f"{context}.maximum_absolute_difference",
        nonnegative=True,
    )
    relative = _finite_number(
        record.get("maximum_relative_difference"),
        f"{context}.maximum_relative_difference",
        nonnegative=True,
    )
    relative_tolerance = _finite_number(
        record.get("relative_tolerance"),
        f"{context}.relative_tolerance",
        nonnegative=True,
    )
    absolute_tolerance = _finite_number(
        record.get("absolute_tolerance"),
        f"{context}.absolute_tolerance",
        nonnegative=True,
    )
    expected_status = _canonical_status(
        absolute,
        relative,
        absolute_tolerance=absolute_tolerance,
        relative_tolerance=relative_tolerance,
    )
    if (
        relative_tolerance != RELATIVE_TOLERANCE
        or absolute_tolerance != ABSOLUTE_TOLERANCE
        or record.get("status") != expected_status
        or expected_status != ResultStatus.OK.value
    ):
        raise FinalAuditError(
            f"{context} does not satisfy the recomputed resolved-sum contract"
        )


def _runtime_namespace_paths(
    expected_checkout: Path,
    *,
    search_path: Sequence[str] | None = None,
) -> tuple[tuple[Path, ...], Path]:
    """Resolve source-first roots and only the first native-bearing candidate."""

    try:
        checkout = expected_checkout.expanduser().resolve(strict=True)
        source_package = (checkout / "src" / "pyamplicol").resolve(strict=True)
    except OSError as error:
        raise FinalAuditError(
            "cannot resolve the expected source pyamplicol package"
        ) from error
    if not source_package.is_dir():
        raise FinalAuditError(
            "expected checkout has no source pyamplicol package directory"
        )
    entries = tuple(sys.path if search_path is None else search_path)
    native_package: Path | None = None
    native_extension: Path | None = None
    for raw_entry in entries:
        try:
            entry = Path(raw_entry or os.getcwd()).expanduser().resolve(strict=True)
            candidate = (entry / "pyamplicol").resolve(strict=True)
        except OSError:
            continue
        if not candidate.is_dir():
            continue
        matches = tuple(
            path
            for path in candidate.iterdir()
            if path.is_file()
            and path.name.startswith("_rusticol")
            and path.name.endswith(tuple(EXTENSION_SUFFIXES))
        )
        if not matches:
            continue
        try:
            native = native_extension_in_package(candidate)
        except RuntimeEvidenceError as error:
            raise FinalAuditError(
                "native-bearing pyamplicol candidate is ambiguous"
            ) from error
        native_package = candidate
        native_extension = native
        break
    if native_package is None or native_extension is None:
        raise FinalAuditError(
            "no native-bearing pyamplicol candidate exists on the original "
            "Python search path"
        )
    roots = (
        (source_package,)
        if native_package == source_package
        else (source_package, native_package)
    )
    return roots, native_extension


def _prepare_exact_pyamplicol_namespace(
    expected_checkout: Path,
) -> tuple[object, tuple[Path, ...], Path, Mapping[str, object]]:
    """Authenticate and establish the exact source/native namespace."""

    package_roots, expected_native = _runtime_namespace_paths(expected_checkout)
    already_loaded = any(
        name == "pyamplicol" or name.startswith("pyamplicol.") for name in sys.modules
    )
    try:
        if already_loaded:
            preimport_identity = established_preimport_runtime_identity()
        else:
            preimport_identity = preimport_python_runtime_identity(
                package_roots,
                native_extension=expected_native,
            )
    except RuntimeEvidenceError as error:
        raise FinalAuditError(
            "pyamplicol runtime was not authenticated before import"
        ) from error

    source_parent = str(package_roots[0].parent)
    sys.path[:] = [
        source_parent,
        *(entry for entry in sys.path if entry != source_parent),
    ]
    pyamplicol = importlib.import_module("pyamplicol")
    try:
        package_origin = Path(str(pyamplicol.__file__)).resolve(strict=True)
    except OSError as error:
        raise FinalAuditError(
            "loaded pyamplicol package origin cannot be resolved"
        ) from error
    if package_origin.parent != package_roots[0]:
        raise FinalAuditError(
            "loaded pyamplicol package does not originate in the expected checkout"
        )
    pyamplicol.__path__ = [str(root) for root in package_roots]
    return pyamplicol, package_roots, expected_native, preimport_identity


def _active_runtime_snapshot(
    expected_source_revision: str,
    *,
    expected_checkout: Path,
) -> dict[str, object]:
    """Resolve the active candidate wheel/native identity once for the audit."""

    (
        pyamplicol,
        package_roots,
        expected_native,
        preimport_identity,
    ) = _prepare_exact_pyamplicol_namespace(expected_checkout)
    from pyamplicol._internal.versions import (
        _active_build_info,
        verify_native_module,
    )

    native = importlib.import_module("pyamplicol._rusticol")
    verify_native_module(native)
    build_info = _mapping(
        _active_build_info(), "active pyamplicol candidate build identity"
    )
    fields = (
        "schema_version",
        "version",
        "candidate_fingerprint",
        "source_revision",
        "source_checkout",
        "native_build_inputs_sha256",
        "publishable",
    )
    candidate = {field: build_info.get(field) for field in fields}
    native_digest = native.native_build_inputs_sha256()
    native_version = str(native.package_version())
    source_checkout = candidate["source_checkout"]
    try:
        checkout_matches = isinstance(source_checkout, str) and Path(
            source_checkout
        ).expanduser().resolve(strict=True) == expected_checkout.resolve(strict=True)
    except OSError:
        checkout_matches = False
    if (
        candidate["source_revision"] != expected_source_revision
        or candidate["publishable"] is not False
        or candidate["native_build_inputs_sha256"] != native_digest
        or candidate["version"] != pyamplicol.__version__
        or native_version != pyamplicol.__version__
        or not checkout_matches
        or not isinstance(native_digest, str)
        or _SHA256_RE.fullmatch(native_digest) is None
        or any(candidate[field] is None for field in fields)
    ):
        raise FinalAuditError(
            "active candidate/native runtime is not bound to the expected final SHA"
        )
    native_path = Path(str(native.__file__)).resolve(strict=True)
    if native_path != expected_native:
        raise FinalAuditError(
            "imported native extension is not the first eligible candidate"
        )
    package_tree = _python_package_tree_identity(package_roots)
    expected_package_tree = _mapping(
        preimport_identity.get("python_package_tree"),
        "preimport runtime identity.python_package_tree",
    )
    expected_native_identity = _mapping(
        preimport_identity.get("native_extension"),
        "preimport runtime identity.native_extension",
    )
    if preimport_identity.get(
        "kind"
    ) != "pyamplicol-preimport-runtime-identity-v1" or dict(package_tree) != dict(
        expected_package_tree
    ):
        raise FinalAuditError(
            "active pyamplicol package differs from its preimport identity"
        )
    try:
        loaded_origin_policy = loaded_pyamplicol_origin_policy(
            package_roots,
            native_extension=native_path,
            expected_package_identity=dict(expected_package_tree),
            expected_native_identity=dict(expected_native_identity),
        )
    except RuntimeEvidenceError as error:
        raise FinalAuditError(
            "loaded pyamplicol module origins cannot be authenticated"
        ) from error
    target = native.target_info()
    return {
        "package_version": pyamplicol.__version__,
        "python_package_tree": package_tree,
        "loaded_module_origin_policy": loaded_origin_policy,
        "candidate_build_identity": candidate,
        "candidate_build_identity_sha256": digest_json(candidate),
        "native_build_inputs_sha256": native_digest,
        "native_extension": {
            "path": str(native_path),
            "sha256": _sha256_file(native_path),
            "package_version": native_version,
        },
        "native_target": {
            "triple": str(target.triple),
            "cpu_features": [str(value) for value in target.cpu_features],
        },
    }


def _audit_active_runtime(
    expected_source_revision: str,
    expected_checkout: Path,
) -> Mapping[str, object]:
    return _active_runtime_snapshot(
        expected_source_revision,
        expected_checkout=expected_checkout,
    )


def _git_checked(
    repo_root: Path,
    arguments: Sequence[str],
    *,
    timeout: float = 30.0,
) -> bytes:
    try:
        completed = subprocess.run(
            ("git", *arguments),
            cwd=repo_root,
            check=False,
            capture_output=True,
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise FinalAuditError(
            "cannot authenticate report publication Git history"
        ) from error
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise FinalAuditError(
            "cannot authenticate report publication Git history: "
            f"git {' '.join(arguments)} failed" + (f": {detail}" if detail else "")
        )
    return completed.stdout


def _publication_relative_member(path: str) -> PurePosixPath | None:
    """Return an allowed report member, rejecting executable/source paths."""

    logical = PurePosixPath(path)
    if (
        not logical.parts
        or logical.is_absolute()
        or logical.as_posix() != path
        or any(part in {"", ".", ".."} for part in logical.parts)
        or logical.parts[0] != "docs"
    ):
        return None
    parts = logical.parts
    root_length = 1
    if len(parts) >= 2 and parts[1] == "performance_reports":
        if (
            len(parts) < 4
            or _REPORT_PROFILE_RE.fullmatch(parts[2]) is None
            or ".." in parts[2]
        ):
            return None
        root_length = 3
    relative = PurePosixPath(*parts[root_length:])
    relative_parts = relative.parts
    if not relative_parts:
        return None
    if relative_parts[0] == "results":
        return (
            relative
            if (
                len(relative_parts) == 2
                and _PUBLICATION_MEMBER_RE.fullmatch(relative_parts[1]) is not None
                and relative_parts[1].endswith(".json")
            )
            else None
        )
    if len(relative_parts) != 1:
        return None
    name = relative_parts[0]
    if _PUBLICATION_MEMBER_RE.fullmatch(name) is None:
        return None
    if name in {
        "README.md",
        "architecture-profile.json",
        "architecture_profile.json",
        "final-audit.json",
        "pyAmpliCol.pdf",
        "pyAmpliCol.tex",
        "report-manifest.json",
        "report-workspace.json",
        "report_environment.tex",
    }:
        return relative
    if name.endswith(".manifest.json"):
        return relative
    if name.endswith(".md"):
        return relative
    if name.startswith("section_") and name.endswith(".tex"):
        return relative
    if name.startswith("result_") and name.endswith("_table.tex"):
        return relative
    return None


def _git_tree_entry(
    repo_root: Path,
    revision: str,
    path: str,
) -> tuple[str, str] | None:
    payload = _git_checked(
        repo_root,
        ("ls-tree", "-z", revision, "--", path),
    )
    if not payload:
        return None
    records = tuple(record for record in payload.split(b"\0") if record)
    if len(records) != 1:
        raise FinalAuditError(
            f"publication path {path!r} is ambiguous in Git tree {revision}"
        )
    try:
        metadata, observed_path = records[0].split(b"\t", 1)
        mode, kind, _object_id = metadata.decode("ascii").split(" ", 2)
        decoded_path = os.fsdecode(observed_path)
    except (UnicodeDecodeError, ValueError) as error:
        raise FinalAuditError(
            f"publication path {path!r} has malformed Git tree metadata"
        ) from error
    if decoded_path != path:
        raise FinalAuditError(
            f"publication path {path!r} does not match its Git tree entry"
        )
    return mode, kind


def _require_nonexecutable_publication_blob(
    repo_root: Path,
    revision: str,
    path: str,
) -> None:
    entry = _git_tree_entry(repo_root, revision, path)
    if entry is None:
        return
    mode, kind = entry
    if mode != "100644" or kind != "blob":
        raise FinalAuditError(
            f"publication path {path!r} is executable, a symlink, or not a file"
        )


def _publication_diff(
    repo_root: Path,
    measurement_source_revision: str,
    publication_revision: str,
) -> tuple[dict[str, str], ...]:
    payload = _git_checked(
        repo_root,
        (
            "diff",
            "--name-status",
            "--no-renames",
            "-z",
            measurement_source_revision,
            publication_revision,
            "--",
        ),
        timeout=120.0,
    )
    fields = tuple(field for field in payload.split(b"\0") if field)
    if len(fields) % 2:
        raise FinalAuditError("publication Git diff has malformed name-status data")
    changes: list[dict[str, str]] = []
    for offset in range(0, len(fields), 2):
        try:
            status = fields[offset].decode("ascii")
            path = os.fsdecode(fields[offset + 1])
        except UnicodeDecodeError as error:
            raise FinalAuditError(
                "publication Git diff contains invalid metadata"
            ) from error
        if status not in {"A", "D", "M"}:
            raise FinalAuditError(
                f"publication Git diff has unsupported status {status!r} for {path!r}"
            )
        if _publication_relative_member(path) is None:
            raise FinalAuditError(
                "publication descendant changes a path outside the explicit "
                f"non-executable report allowlist: {path!r}"
            )
        if status != "A":
            _require_nonexecutable_publication_blob(
                repo_root,
                measurement_source_revision,
                path,
            )
        if status != "D":
            _require_nonexecutable_publication_blob(
                repo_root,
                publication_revision,
                path,
            )
        changes.append({"status": status, "path": path})
    return tuple(changes)


def _report_publication_lineage(
    repo_root: Path,
    measurement_source_revision: str,
) -> dict[str, object]:
    """Authenticate a clean report-only descendant of one measured SHA."""

    root = repo_root.expanduser().resolve(strict=False)
    if _GIT_SHA_RE.fullmatch(measurement_source_revision) is None:
        raise FinalAuditError("measurement source revision must be a full Git SHA")
    publication_revision = (
        _git_checked(
            root,
            ("rev-parse", "--verify", "HEAD^{commit}"),
        )
        .decode("ascii")
        .strip()
    )
    if _GIT_SHA_RE.fullmatch(publication_revision) is None:
        raise FinalAuditError("publication revision is not a full Git SHA")
    _git_checked(
        root,
        ("cat-file", "-e", f"{measurement_source_revision}^{{commit}}"),
    )
    status = _git_checked(
        root,
        ("status", "--porcelain=v1", "-z", "--untracked-files=all"),
    )
    if status:
        raise FinalAuditError(
            "publication checkout must be completely clean, including report outputs"
        )
    ancestor = subprocess.run(
        (
            "git",
            "merge-base",
            "--is-ancestor",
            measurement_source_revision,
            publication_revision,
        ),
        cwd=root,
        check=False,
        capture_output=True,
        timeout=30,
    )
    if ancestor.returncode not in {0, 1}:
        raise FinalAuditError("cannot authenticate measurement/publication ancestry")
    if ancestor.returncode != 0:
        raise FinalAuditError(
            "publication revision is not a descendant of the measurement "
            "source revision"
        )
    changes = _publication_diff(
        root,
        measurement_source_revision,
        publication_revision,
    )
    same_revision = publication_revision == measurement_source_revision
    if not same_revision and not changes:
        raise FinalAuditError(
            "later publication revision has no report-only publication diff"
        )
    return {
        "kind": _PUBLICATION_LINEAGE_KIND,
        "schema_version": 1,
        "measurement_source_revision": measurement_source_revision,
        "publication_revision": publication_revision,
        "relationship": ("same-commit" if same_revision else "report-only-descendant"),
        "worktree_clean": True,
        "changed_path_count": len(changes),
        "changed_paths": list(changes),
        "changed_paths_sha256": digest_json(list(changes)),
        "allowed_path_contract": (
            "docs report roots: results/*.json, generated table TeX, report "
            "TeX/Markdown/PDF, architecture metadata, and report manifests; "
            "regular non-executable blobs only"
        ),
        "executable_source_unchanged": True,
    }


def _require_report_source_checkout(
    repo_root: Path,
    expected_source_revision: str,
) -> Mapping[str, object]:
    return _report_publication_lineage(repo_root, expected_source_revision)


def _validated_publication_lineage(
    value: Mapping[str, object] | None,
    *,
    measurement_source_revision: str,
    expected_publication_revision: str | None,
) -> dict[str, object]:
    if value is None:
        # Injectable source auditors used by focused unit tests predate the
        # lineage return value. Production always uses the fail-closed auditor.
        result: dict[str, object] = {
            "kind": _PUBLICATION_LINEAGE_KIND,
            "schema_version": 1,
            "measurement_source_revision": measurement_source_revision,
            "publication_revision": measurement_source_revision,
            "relationship": "same-commit",
            "worktree_clean": True,
            "changed_path_count": 0,
            "changed_paths": [],
            "changed_paths_sha256": digest_json([]),
            "allowed_path_contract": "injected source auditor",
            "executable_source_unchanged": True,
        }
    else:
        result = dict(value)
    publication_revision = result.get("publication_revision")
    changes = result.get("changed_paths")
    changed_count = result.get("changed_path_count")
    if (
        result.get("kind") != _PUBLICATION_LINEAGE_KIND
        or result.get("schema_version") != 1
        or result.get("measurement_source_revision") != measurement_source_revision
        or not isinstance(publication_revision, str)
        or _GIT_SHA_RE.fullmatch(publication_revision) is None
        or result.get("relationship") not in {"same-commit", "report-only-descendant"}
        or result.get("worktree_clean") is not True
        or result.get("executable_source_unchanged") is not True
        or isinstance(changes, (str, bytes))
        or not isinstance(changes, Sequence)
        or isinstance(changed_count, bool)
        or not isinstance(changed_count, int)
        or changed_count != len(changes)
        or result.get("changed_paths_sha256") != digest_json(list(changes))
    ):
        raise FinalAuditError("source auditor returned invalid publication lineage")
    if (
        expected_publication_revision is not None
        and publication_revision != expected_publication_revision
    ):
        raise FinalAuditError(
            "publication checkout HEAD does not equal the explicitly expected "
            "publication revision"
        )
    return result


def _pdf_text_identity(path: Path) -> dict[str, object]:
    try:
        from pypdf import PdfReader

        reader = PdfReader(path)
        if reader.is_encrypted:
            raise FinalAuditError("report PDF is unexpectedly encrypted")
        pages = [page.extract_text() or "" for page in reader.pages]
    except FinalAuditError:
        raise
    except Exception as error:
        raise FinalAuditError(f"cannot extract report PDF text: {path}") from error
    if not pages or not any(page.strip() for page in pages):
        raise FinalAuditError("report PDF has no extractable page text")
    return {
        "page_count": len(pages),
        "text_sha256": digest_json(pages),
        "text_character_count": sum(len(page) for page in pages),
    }


def _pdf_visual_identity(path: Path, output_root: Path) -> dict[str, object]:
    """Render every page with one Poppler binary and hash the resulting pixels."""

    pdftoppm = shutil.which("pdftoppm")
    if pdftoppm is None:
        raise FinalAuditError("pdftoppm is required for the final PDF visual audit")
    output_root.mkdir(parents=True, exist_ok=False)
    prefix = output_root / "page"
    try:
        completed = subprocess.run(
            (
                pdftoppm,
                "-png",
                "-r",
                "96",
                str(path),
                str(prefix),
            ),
            check=False,
            capture_output=True,
            text=True,
            timeout=900,
        )
    except subprocess.TimeoutExpired as error:
        raise FinalAuditError(f"PDF rendering timed out: {path}") from error
    if completed.returncode != 0:
        tail = "\n".join(
            (*completed.stdout.splitlines(), *completed.stderr.splitlines())[-40:]
        )
        raise FinalAuditError(
            f"PDF rendering failed with exit {completed.returncode}: {path}\n{tail}"
        )
    pages = sorted(output_root.glob("page-*.png"))
    if not pages:
        raise FinalAuditError(f"PDF rendering produced no pages: {path}")
    page_hashes = [_sha256_file(page) for page in pages]
    return {
        "page_count": len(page_hashes),
        "page_pixel_sha256": digest_json(page_hashes),
    }


def _tex_without_comments(source: str) -> str:
    return re.sub(r"(?<!\\)%[^\n]*", "", source)


def _audit_tex_table_reachability(
    master_tex: Path,
    canonical_tables: Sequence[Path],
) -> dict[str, object]:
    """Require every rendered table to be reachable from the PDF master."""

    try:
        master = master_tex.expanduser().resolve(strict=True)
    except OSError as error:
        raise FinalAuditError(
            f"report master TeX is unavailable: {master_tex}"
        ) from error
    docs_root = master.parent
    tables: list[Path] = []
    for raw_table in canonical_tables:
        try:
            table = raw_table.expanduser().resolve(strict=True)
            table.relative_to(docs_root)
        except (OSError, ValueError) as error:
            raise FinalAuditError(
                f"canonical report table is unavailable or outside docs: {raw_table}"
            ) from error
        tables.append(table)
    if not tables:
        raise FinalAuditError("report PDF audit has no canonical rendered tables")

    reachable: set[Path] = set()
    pending = [master]
    while pending:
        source = pending.pop()
        if source in reachable:
            continue
        reachable.add(source)
        try:
            text = source.read_text(encoding="utf-8")
        except OSError as error:
            raise FinalAuditError(
                f"cannot read reachable report TeX source: {source}"
            ) from error
        for match in _TEX_INPUT_RE.finditer(_tex_without_comments(text)):
            raw_dependency = match.group(1).strip()
            if not raw_dependency:
                raise FinalAuditError(f"{source} contains an empty TeX dependency")
            dependency = source.parent / raw_dependency
            if dependency.suffix == "":
                dependency = dependency.with_suffix(".tex")
            try:
                dependency = dependency.resolve(strict=True)
                dependency.relative_to(docs_root)
            except (OSError, ValueError) as error:
                raise FinalAuditError(
                    f"{source} has an unavailable or out-of-tree TeX dependency "
                    f"{raw_dependency!r}"
                ) from error
            pending.append(dependency)

    missing = sorted(
        table.relative_to(docs_root).as_posix()
        for table in set(tables).difference(reachable)
    )
    if missing:
        raise FinalAuditError(
            "canonical rendered tables are not reachable from pyAmpliCol.tex: "
            + ", ".join(missing)
        )
    relative_sources = sorted(
        source.relative_to(docs_root).as_posix() for source in reachable
    )
    return {
        "master_path": str(master),
        "reachable_tex_source_count": len(reachable),
        "reachable_tex_sources_sha256": digest_json(relative_sources),
        "reachable_table_source_count": len(set(tables)),
    }


def _audit_pdf(service: ReportService) -> dict[str, object]:
    """Rebuild the PDF and compare both extracted text and rendered pixels."""

    latexmk = shutil.which("latexmk")
    if latexmk is None:
        raise FinalAuditError("latexmk is required for the final PDF audit")
    published = service.paths.docs_dir / "pyAmpliCol.pdf"
    if not published.is_file() or published.stat().st_size == 0:
        raise FinalAuditError("published docs/pyAmpliCol.pdf is missing or empty")
    rendered_tables = render_all_tables(
        service.load_caches(),
        catalog=service.catalog,
    )
    table_paths = [service.paths.docs_dir / name for name in sorted(rendered_tables)]
    tex_dependency_identity = _audit_tex_table_reachability(
        service.paths.docs_dir / "pyAmpliCol.tex",
        table_paths,
    )
    service.paths.artifact_root.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix="final-report-pdf-audit-",
            dir=service.paths.artifact_root,
        )
    )
    try:
        build_docs = staging / "docs"
        shutil.copytree(
            service.paths.docs_dir,
            build_docs,
            ignore=shutil.ignore_patterns(
                "*.aux",
                "*.fdb_latexmk",
                "*.fls",
                "*.log",
                "*.out",
                "*.toc",
                "pyAmpliCol.pdf",
                ".coordination",
            ),
        )
        environment = os.environ.copy()
        environment.update({"LANG": "C", "LC_ALL": "C", "LC_CTYPE": "C"})
        completed = subprocess.run(
            (
                latexmk,
                "-pdf",
                "-interaction=nonstopmode",
                "-halt-on-error",
                "-file-line-error",
                "pyAmpliCol.tex",
            ),
            cwd=build_docs,
            check=False,
            capture_output=True,
            text=True,
            env=environment,
            timeout=900,
        )
        if completed.returncode != 0:
            tail = "\n".join(
                (*completed.stdout.splitlines(), *completed.stderr.splitlines())[-80:]
            )
            raise FinalAuditError(
                f"isolated report PDF rebuild failed with exit "
                f"{completed.returncode}:\n{tail}"
            )
        rebuilt = build_docs / "pyAmpliCol.pdf"
        if not rebuilt.is_file() or rebuilt.stat().st_size == 0:
            raise FinalAuditError("isolated report PDF rebuild produced no PDF")
        published_text = _pdf_text_identity(published)
        rebuilt_text = _pdf_text_identity(rebuilt)
        if published_text != rebuilt_text:
            raise FinalAuditError(
                "published report PDF page text differs from a rebuild of the "
                "audited caches and tables"
            )
        published_visual = _pdf_visual_identity(
            published,
            staging / "published-pages",
        )
        rebuilt_visual = _pdf_visual_identity(
            rebuilt,
            staging / "rebuilt-pages",
        )
        if published_visual != rebuilt_visual:
            raise FinalAuditError(
                "published report PDF rendered pixels differ from a rebuild of "
                "the audited caches and tables"
            )
        table_identity = {path.name: _sha256_file(path) for path in table_paths}
        return {
            "status": ResultStatus.OK.value,
            "published_path": str(published.resolve(strict=True)),
            "published_sha256": _sha256_file(published),
            "rebuilt_sha256": _sha256_file(rebuilt),
            "normalized_page_text": published_text,
            "rendered_pages": published_visual,
            "table_source_count": len(table_identity),
            "table_sources_sha256": digest_json(table_identity),
            "master_tex_dependencies": tex_dependency_identity,
        }
    except subprocess.TimeoutExpired as error:
        raise FinalAuditError("isolated report PDF rebuild timed out") from error
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _expected_evaluator_abis(cell: CellSpec) -> tuple[str, str]:
    if cell.measurement.execution_mode is ExecutionMode.COMPILED:
        if cell.measurement.backend == "jit":
            return _COMPILED_DIRECT_ABI, _SYMJIT_APPLICATION_ABI
        return _NATIVE_COMPILED_DIRECT_ABI, _NATIVE_COMPILED_DIRECT_ABI
    if cell.measurement.execution_mode is ExecutionMode.EAGER:
        source = (
            _SYMJIT_APPLICATION_ABI
            if cell.measurement.backend == "jit"
            else _EAGER_NATIVE_APPLICATION_ABI
        )
        return "symjit-direct-table-binding-v2", source
    source = (
        _SYMJIT_APPLICATION_ABI
        if cell.measurement.backend == "jit"
        else _SOURCE_RUNTIME_CAPABILITY[cell.measurement.backend]
    )
    return "pyamplicol-recurrence-runtime-layout-v2", source


def _audit_runtime_identity(
    cell: CellSpec,
    provenance: Mapping[str, object],
    *,
    expected_source_revision: str,
    active_runtime: Mapping[str, object],
    artifact: ArtifactEvidence | None,
) -> None:
    context = f"{cell.cell_id}.provenance.runtime_identity"
    identity = _mapping(provenance.get("runtime_identity"), context)
    identity_digest = provenance.get("runtime_identity_sha256")
    if identity_digest != digest_json(identity):
        raise FinalAuditError(f"{context} canonical SHA-256 does not match")
    active_package_tree = _validate_python_package_tree_identity(
        active_runtime.get("python_package_tree"),
        context="active_runtime.python_package_tree",
    )
    _validate_loaded_origin_policy(
        active_runtime.get("loaded_module_origin_policy"),
        context="active_runtime.loaded_module_origin_policy",
    )
    _validate_loaded_origin_policy(
        identity.get("loaded_module_origin_policy"),
        context=f"{context}.loaded_module_origin_policy",
    )
    _audit_runtime_identity_postflight(
        provenance,
        identity,
        context=f"{cell.cell_id}.provenance",
    )
    expected_capability = _ARENA_CAPABILITY[cell.measurement.execution_mode]
    expected_abi, expected_source_abi = _expected_evaluator_abis(cell)
    expected_source_runtime_capability = _SOURCE_RUNTIME_CAPABILITY.get(
        cell.measurement.backend
    )
    if expected_source_runtime_capability is None:
        raise FinalAuditError(
            f"{context} has unsupported backend {cell.measurement.backend!r}"
        )
    checks = {
        "kind": "pyamplicol-report-runtime-identity-v1",
        "execution_mode": cell.measurement.execution_mode.value,
        "loaded_execution_mode": cell.measurement.execution_mode.value,
        "backend": cell.measurement.backend,
        "required_arena_capability": expected_capability,
        "expected_evaluator_abi": expected_abi,
        "expected_source_evaluator_abi": expected_source_abi,
        "expected_source_evaluator_runtime_capability": (
            expected_source_runtime_capability
        ),
        "source_revision": expected_source_revision,
        "package_version": active_runtime.get("package_version"),
        "python_package_tree": active_package_tree,
        "candidate_build_identity": active_runtime.get("candidate_build_identity"),
        "candidate_build_identity_sha256": active_runtime.get(
            "candidate_build_identity_sha256"
        ),
        "native_build_inputs_sha256": active_runtime.get("native_build_inputs_sha256"),
        "native_extension": active_runtime.get("native_extension"),
        "native_target": active_runtime.get("native_target"),
    }
    mismatches = [
        field for field, expected in checks.items() if identity.get(field) != expected
    ]
    source_level_field = "source_jit_optimization_level"
    direct_level_field = "direct_codegen_optimization_level"
    if cell.measurement.backend == "jit":
        source_level = identity.get(source_level_field)
        if (
            type(source_level) is not int
            or source_level != cell.measurement.jit_optimization_level
        ):
            mismatches.append(source_level_field)
    elif source_level_field in identity:
        mismatches.append(source_level_field)
    if (
        cell.measurement.execution_mode is ExecutionMode.COMPILED
        and cell.measurement.backend == "jit"
    ):
        direct_level = identity.get(direct_level_field)
        if type(direct_level) is not int or direct_level != 3:
            mismatches.append(direct_level_field)
        direct_identity = _mapping(
            identity.get("direct_codegen_identity"),
            f"{context}.direct_codegen_identity",
        )
        if (
            direct_identity.get("kind")
            != "authenticated-compiled-plane-arena-direct-codegen-v1"
            or direct_identity.get("optimization_level") != 3
            or direct_identity.get("source_optimization_level")
            != cell.measurement.jit_optimization_level
            or type(direct_identity.get("leaf_count")) is not int
            or int(direct_identity["leaf_count"]) <= 0
            or not isinstance(direct_identity.get("execution_manifest_path"), str)
            or not isinstance(direct_identity.get("execution_manifest_sha256"), str)
            or _SHA256_RE.fullmatch(
                str(direct_identity.get("execution_manifest_sha256"))
            )
            is None
        ):
            mismatches.append("direct_codegen_identity")
    else:
        if direct_level_field in identity:
            mismatches.append(direct_level_field)
        if "direct_codegen_identity" in identity:
            mismatches.append("direct_codegen_identity")
    if (
        cell.measurement.execution_mode is ExecutionMode.RECURRENCE
        and cell.measurement.backend == "jit"
    ):
        source_jit_identity = _mapping(
            identity.get("source_jit_identity"),
            f"{context}.source_jit_identity",
        )
        if (
            source_jit_identity.get("kind")
            != "authenticated-recurrence-direct-template-source-v1"
            or source_jit_identity.get("optimization_level")
            != cell.measurement.jit_optimization_level
            or type(source_jit_identity.get("direct_template_count")) is not int
            or int(source_jit_identity["direct_template_count"]) <= 0
            or type(source_jit_identity.get("prepared_direct_template_count"))
            is not int
            or int(source_jit_identity["prepared_direct_template_count"]) <= 0
            or type(source_jit_identity.get("source_evaluator_leaf_count")) is not int
            or int(source_jit_identity["source_evaluator_leaf_count"]) <= 0
            or source_jit_identity.get("source_application_abi")
            != _SYMJIT_APPLICATION_ABI
            or source_jit_identity.get("direct_application_abi")
            != _RECURRENCE_JIT_DIRECT_APPLICATION_ABI
            or _SHA256_RE.fullmatch(
                str(source_jit_identity.get("prepared_kernel_pack_digest"))
            )
            is None
            or _SHA256_RE.fullmatch(
                str(source_jit_identity.get("direct_template_catalog_digest"))
            )
            is None
            or not isinstance(source_jit_identity.get("execution_manifest_path"), str)
            or not isinstance(source_jit_identity.get("kernel_pack_path"), str)
            or not isinstance(source_jit_identity.get("execution_manifest_sha256"), str)
            or _SHA256_RE.fullmatch(
                str(source_jit_identity.get("execution_manifest_sha256"))
            )
            is None
            or not isinstance(source_jit_identity.get("kernel_pack_sha256"), str)
            or _SHA256_RE.fullmatch(str(source_jit_identity.get("kernel_pack_sha256")))
            is None
        ):
            mismatches.append("source_jit_identity")
    elif "source_jit_identity" in identity:
        mismatches.append("source_jit_identity")
    if identity.get("artifact_identity_match") is not True:
        mismatches.append("artifact_identity_match")
    if mismatches:
        raise FinalAuditError(
            f"{context} differs from the active final runtime: "
            + ", ".join(sorted(set(mismatches)))
        )
    if artifact is None:
        return
    artifact_checks = {
        "artifact_id": artifact.artifact_id,
        "loaded_artifact_id": artifact.artifact_id,
        "process_id": artifact.process_id,
        "artifact_runtime_version": artifact.runtime_version,
        "process_required_runtime_capabilities": list(artifact.runtime_capabilities),
    }
    mismatches = []
    measured_effective = _mapping(
        provenance.get("effective_config"),
        f"{cell.cell_id}.provenance.effective_config",
    )
    if dict(measured_effective) != dict(artifact.effective_config):
        mismatches.append("effective_config")
    if (
        cell.measurement.execution_mode is ExecutionMode.COMPILED
        and cell.measurement.backend == "jit"
    ):
        direct_identity = _mapping(
            identity.get("direct_codegen_identity"),
            f"{context}.direct_codegen_identity",
        )
        direct_checks = {
            "execution_manifest_path": artifact.execution_manifest_path,
            "execution_manifest_sha256": artifact.execution_manifest_sha256,
            "leaf_count": artifact.direct_leaf_count,
        }
        mismatches.extend(
            f"direct_codegen_identity.{field}"
            for field, expected in direct_checks.items()
            if direct_identity.get(field) != expected
        )
    if (
        cell.measurement.execution_mode is ExecutionMode.RECURRENCE
        and cell.measurement.backend == "jit"
    ):
        source_jit_identity = _mapping(
            identity.get("source_jit_identity"),
            f"{context}.source_jit_identity",
        )
        if artifact.source_jit_identity is None or dict(source_jit_identity) != dict(
            artifact.source_jit_identity
        ):
            mismatches.append("source_jit_identity")
    elif artifact.source_jit_identity is not None:
        mismatches.append("artifact.source_jit_identity")
    mismatches.extend(
        field
        for field, expected in artifact_checks.items()
        if identity.get(field) != expected
    )
    if mismatches:
        raise FinalAuditError(
            f"{context} differs from the authenticated artifact: "
            + ", ".join(mismatches)
        )


def _audit_effective_config(
    cell: CellSpec,
    provenance: Mapping[str, object],
) -> Mapping[str, object]:
    context = f"{cell.cell_id}.provenance.effective_config"
    effective = _mapping(provenance.get("effective_config"), context)
    _audit_effective_config_mapping(cell, effective, context=context)
    return effective


def _audit_effective_config_mapping(
    cell: CellSpec,
    effective: Mapping[str, object],
    *,
    context: str,
) -> None:
    evaluator = _mapping(effective.get("evaluator"), f"{context}.evaluator")
    color = _mapping(effective.get("color"), f"{context}.color")
    generation = _mapping(effective.get("generation"), f"{context}.generation")
    validation = _mapping(
        generation.get("validation"), f"{context}.generation.validation"
    )
    expected_layout = (
        "all-flow-union" if cell.workload is Workload.ALL_FLOW else "topology-replay"
    )
    mismatches: list[str] = []
    expected = {
        "execution_mode": cell.measurement.execution_mode.value,
        "backend": cell.measurement.backend,
    }
    mismatches.extend(
        f"evaluator.{field}"
        for field, value in expected.items()
        if evaluator.get(field) != value
    )
    if color.get("accuracy") != cell.measurement.accuracy.value:
        mismatches.append("color.accuracy")
    if color.get("lc_flow_layout") != expected_layout:
        mismatches.append("color.lc_flow_layout")
    if cell.measurement.jit_optimization_level is not None:
        jit = _mapping(evaluator.get("jit"), f"{context}.evaluator.jit")
        if jit.get("optimization_level") != cell.measurement.jit_optimization_level:
            mismatches.append("evaluator.jit.optimization_level")
    expected_validation = {
        "enabled": True,
        "samples": 10,
        "seed": 12345,
        "relative_tolerance": RELATIVE_TOLERANCE,
        "absolute_tolerance": 1.0e-300,
        "post_build_validation": True,
    }
    mismatches.extend(
        f"generation.validation.{field}"
        for field, value in expected_validation.items()
        if validation.get(field) != value
    )
    if mismatches:
        raise FinalAuditError(
            f"{context} differs from the report contract: " + ", ".join(mismatches)
        )


def _audit_model_source(
    cell: CellSpec,
    effective: Mapping[str, object],
) -> None:
    context = "authenticated effective configuration.model.source"
    model = _mapping(
        effective.get("model"), "authenticated effective configuration.model"
    )
    source = model.get("source")
    if not isinstance(source, str) or not source:
        raise FinalAuditError(f"{context} is invalid")
    if cell.measurement.model is ModelKey.BUILTIN_SM:
        if source != "built-in-sm":
            raise FinalAuditError(f"{context} differs from the built-in-SM cell")
        return
    relative = _MODEL_SOURCE_PATH.get(cell.measurement.model)
    if relative is None:
        raise FinalAuditError(
            f"{cell.cell_id} has no authenticated model-source contract"
        )
    repo_root = Path(__file__).resolve().parents[2]
    expected = (repo_root / relative).resolve(strict=True)
    try:
        observed = Path(source).expanduser().resolve(strict=True)
    except OSError as error:
        raise FinalAuditError(f"{context} is unavailable: {source}") from error
    if observed != expected:
        raise FinalAuditError(
            f"{context} differs from the exact source for "
            f"{cell.measurement.model.value}"
        )


def _artifact_reference(
    cell: CellSpec,
    measurement: Mapping[str, object],
    *,
    report_paths: ReportPaths | None = None,
) -> _ArtifactReference:
    artifact = _mapping(measurement.get("artifact"), f"{cell.cell_id}.artifact")
    raw_path = artifact.get("path")
    process_id = artifact.get("process_id")
    if not isinstance(raw_path, str) or not raw_path:
        raise FinalAuditError(f"{cell.cell_id}.artifact.path is invalid")
    if not isinstance(process_id, str) or not process_id:
        raise FinalAuditError(f"{cell.cell_id}.artifact.process_id is invalid")
    try:
        if raw_path.startswith(_PORTABLE_ARTIFACT_ROOT):
            if report_paths is None:
                raise FinalAuditError(
                    f"{cell.cell_id}.artifact.path requires report path context"
                )
            suffix = raw_path.removeprefix(_PORTABLE_ARTIFACT_ROOT)
            if not suffix.startswith("/") or suffix == "/":
                raise FinalAuditError(
                    f"{cell.cell_id}.artifact.path has an invalid portable locator"
                )
            logical = _canonical_relative_path(
                suffix.removeprefix("/"),
                f"{cell.cell_id}.artifact",
            )
            path = report_paths.artifact_root.joinpath(*logical.parts).resolve(
                strict=True
            )
            path.relative_to(report_paths.artifact_root.resolve(strict=True))
        else:
            if "${" in raw_path:
                raise FinalAuditError(
                    f"{cell.cell_id}.artifact.path has an unsupported locator"
                )
            path = Path(raw_path).expanduser().resolve(strict=True)
    except OSError as error:
        raise FinalAuditError(
            f"{cell.cell_id}.artifact.path is unavailable: {raw_path}"
        ) from error
    except ValueError as error:
        raise FinalAuditError(
            f"{cell.cell_id}.artifact.path escapes its report artifact root"
        ) from error
    if not path.is_dir():
        raise FinalAuditError(f"{cell.cell_id}.artifact.path is not a directory")
    return _ArtifactReference(cell, measurement, path, process_id)


def _shared_artifact_contract(cell: CellSpec) -> tuple[object, ...]:
    """Fields that must agree when report cells reuse one process artifact."""

    return (
        cell.process,
        cell.n_final,
        cell.process_key,
        cell.measurement,
        cell.workload,
    )


def _expected_legacy_revision(repo_root: Path) -> str:
    lock_path = repo_root / "dependencies" / "contributor-lock.toml"
    try:
        lock = tomllib.loads(lock_path.read_text(encoding="utf-8"))
        section = _mapping(lock.get("legacy_amplicol"), "legacy_amplicol lock")
        revision = section.get("revision")
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise FinalAuditError(
            "cannot authenticate the pinned original-AmpliCol revision"
        ) from error
    if not isinstance(revision, str) or _GIT_SHA_RE.fullmatch(revision) is None:
        raise FinalAuditError("pinned original-AmpliCol revision is invalid")
    return revision


def _audit_below_resolution_execution_timing(
    cell: CellSpec,
    provenance: Mapping[str, object],
    *,
    measurement_sample_count: int,
    context: str,
) -> None:
    """Authenticate the sole permitted unavailable execution submetric."""

    if (
        cell.measurement.execution_mode is not ExecutionMode.COMPILED
        or cell.measurement.backend != "jit"
        or cell.measurement.jit_optimization_level != 3
    ):
        raise FinalAuditError(
            f"{context}.execution_seconds_per_point may be unavailable only "
            "for compiled JIT O3"
        )
    timing = _mapping(
        provenance.get("execution_timing"),
        f"{context}.provenance.execution_timing",
    )
    if set(timing) != _EXECUTION_TIMING_FIELDS:
        raise FinalAuditError(
            f"{context}.provenance.execution_timing fields do not match "
            "the authenticated contract"
        )
    raw_seconds = timing.get("raw_seconds_per_point")
    timing_sample_count = timing.get("sample_count")
    native_points = timing.get("native_profile_points_per_sample")
    if (
        timing.get("abi") != _EXECUTION_TIMING_ABI
        or timing.get("status") != "below_timer_resolution"
        or timing.get("ratio_eligible") is not False
        or isinstance(raw_seconds, bool)
        or not isinstance(raw_seconds, (int, float))
        or not math.isfinite(float(raw_seconds))
        or float(raw_seconds) != 0.0
        or timing.get("source") != _COMPILED_ARENA_EXECUTION_TIME_SOURCE
        or timing.get("compiled_direct_arena_active") is not True
        or isinstance(timing_sample_count, bool)
        or not isinstance(timing_sample_count, int)
        or timing_sample_count < 5
        or timing_sample_count != measurement_sample_count
        or isinstance(native_points, bool)
        or not isinstance(native_points, int)
        or native_points < 1
        or timing.get("sample_contract") != _PAIRED_TIMING_SAMPLE_CONTRACT
    ):
        raise FinalAuditError(
            f"{context}.provenance.execution_timing is not an authenticated "
            "compiled Direct-Arena below-resolution record"
        )


def _audit_measurement(
    cell: CellSpec,
    measurement: Mapping[str, object],
    *,
    baseline: Mapping[str, object] | None,
    expected_source_revision: str,
    expected_legacy_revision: str,
    active_runtime: Mapping[str, object] | None,
    report_paths: ReportPaths | None = None,
) -> _ArtifactReference | None:
    context = cell.cell_id
    if measurement.get("status") != ResultStatus.OK.value:
        raise FinalAuditError(
            f"{context}.measurement.status is not {ResultStatus.OK.value!r}"
        )
    if measurement.get("failure") is not None:
        raise FinalAuditError(f"{context}.measurement has failure metadata")
    matrix_element = _finite_number(
        measurement.get("matrix_element"),
        f"{context}.measurement.matrix_element",
        nonnegative=True,
    )
    for field in ("generation_seconds", "wall_seconds_per_point"):
        _finite_number(
            measurement.get(field),
            f"{context}.measurement.{field}",
            nonnegative=True,
        )
    sample_count = measurement.get("sample_count")
    if (
        isinstance(sample_count, bool)
        or not isinstance(sample_count, int)
        or sample_count < 5
    ):
        raise FinalAuditError(
            f"{context}.measurement.sample_count must be at least five"
        )
    provenance = _mapping(
        measurement.get("provenance"), f"{context}.measurement.provenance"
    )
    if provenance.get("report_source_revision") != expected_source_revision:
        raise FinalAuditError(
            f"{context} was not measured at source SHA {expected_source_revision}"
        )
    validation = _mapping(
        measurement.get("validation"), f"{context}.measurement.validation"
    )
    if validation.get("status") != ResultStatus.OK.value:
        raise FinalAuditError(f"{context}.measurement.validation.status is not ok")

    raw_selector = measurement.get("selector_contract")
    selector: SelectorContract | None = None
    if cell.measurement.accuracy is Accuracy.LC:
        selector = SelectorContract.from_mapping(
            _mapping(raw_selector, f"{context}.measurement.selector_contract")
        )
    elif raw_selector is not None:
        raise FinalAuditError(
            f"{context} contracted-color measurement has a selector contract"
        )

    if cell.measurement.execution_mode is ExecutionMode.AMPLICOL:
        if validation.get("method") != "independent-original-amplicol-oracle":
            raise FinalAuditError(
                f"{context} does not identify the independent AmpliCol oracle"
            )
        stored_point_digest = validation.get("point_digest")
        expected_point_digest = point_digest(shared_validation_points(cell.process))
        if (
            not isinstance(stored_point_digest, str)
            or _SHA256_RE.fullmatch(stored_point_digest) is None
            or stored_point_digest != expected_point_digest
        ):
            raise FinalAuditError(
                f"{context} oracle point digest does not match its physical process"
            )
        if selector is not None and stored_point_digest != selector.point_digest:
            raise FinalAuditError(
                f"{context} oracle and selector use different validation points"
            )
        revision = provenance.get("revision")
        if revision != expected_legacy_revision:
            raise FinalAuditError(
                f"{context} does not use the pinned original-AmpliCol revision"
            )
        return None

    raw_execution_seconds = measurement.get("execution_seconds_per_point")
    if raw_execution_seconds is None:
        _audit_below_resolution_execution_timing(
            cell,
            provenance,
            measurement_sample_count=sample_count,
            context=f"{context}.measurement",
        )
    else:
        execution_seconds = _finite_number(
            raw_execution_seconds,
            f"{context}.measurement.execution_seconds_per_point",
        )
        if execution_seconds <= 0.0:
            raise FinalAuditError(
                f"{context}.measurement.execution_seconds_per_point must be positive"
            )
    if provenance.get("source_revision") != expected_source_revision:
        raise FinalAuditError(
            f"{context} pyAmpliCol source revision is not the measurement SHA"
        )
    if active_runtime is None:
        raise FinalAuditError("pyAmpliCol measurements require an active runtime")
    _audit_effective_config(cell, provenance)
    _audit_resolved_sum(
        validation.get("resolved_sum"),
        context=f"{context}.measurement.validation.resolved_sum",
    )

    scalar = cell.measurement.model in {
        ModelKey.SCALAR_CONTACT,
        ModelKey.SCALAR_GRAVITY,
    }
    if scalar:
        _audit_pointwise(
            validation.get("high_precision"),
            context=f"{context}.measurement.validation.high_precision",
            expected_candidate=matrix_element,
            expected_baseline=_finite_number(
                _mapping(
                    validation.get("high_precision"),
                    f"{context}.measurement.validation.high_precision",
                ).get("baseline"),
                f"{context}.measurement.validation.high_precision.baseline",
            ),
            expected_relative_tolerance=RELATIVE_TOLERANCE,
        )
        if baseline is not None:
            raise FinalAuditError(f"{context} scalar cell has a catalog baseline")
    else:
        if baseline is None:
            raise FinalAuditError(f"{context} has no canonical baseline")
        baseline_value = _finite_number(
            baseline.get("matrix_element"),
            f"{context}.canonical_baseline.matrix_element",
            nonnegative=True,
        )
        _audit_pointwise(
            validation.get("pointwise"),
            context=f"{context}.measurement.validation.pointwise",
            expected_candidate=matrix_element,
            expected_baseline=baseline_value,
            expected_relative_tolerance=_expected_pointwise_tolerance(cell),
        )
        baseline_selector = baseline.get("selector_contract")
        if cell.measurement.accuracy is Accuracy.LC:
            if raw_selector != baseline_selector:
                raise FinalAuditError(
                    f"{context} selector contract differs from canonical baseline"
                )
        elif baseline_selector is not None:
            raise FinalAuditError(
                f"{context} contracted canonical baseline has selectors"
            )

    return _artifact_reference(
        cell,
        measurement,
        report_paths=report_paths,
    )


def _find_process_execution(
    artifact: Path,
    manifest: object,
    process_id: str,
) -> tuple[dict[str, Any], str, str, tuple[str, ...]]:
    runtime = _mapping(manifest.runtime, "artifact.runtime")
    raw_index = runtime.get("evaluator_manifest_path")
    index, index_relative, _index_sha256 = _authenticated_json_payload(
        artifact,
        manifest,
        raw_index,
        process_id=None,
        context="evaluator manifest index",
    )
    if (
        index.get("kind") != "pyamplicol-runtime-execution-set"
        or type(index.get("schema_version")) is not int
        or index.get("schema_version") != 3
    ):
        raise FinalAuditError("artifact evaluator index has an invalid contract")
    matches = [
        item
        for item in _sequence(index.get("processes"), "evaluator index.processes")
        if isinstance(item, Mapping) and item.get("process_id") == process_id
    ]
    if len(matches) != 1:
        raise FinalAuditError(
            f"artifact evaluator index does not uniquely identify {process_id!r}"
        )
    record = matches[0]
    raw_execution = record.get("manifest_path")
    execution_logical = _canonical_relative_path(
        raw_execution,
        "process execution manifest",
    )
    execution_relative = (
        PurePosixPath(index_relative).parent / execution_logical
    ).as_posix()
    execution, execution_relative, execution_sha256 = _authenticated_json_payload(
        artifact,
        manifest,
        execution_relative,
        process_id=process_id,
        context="process execution manifest",
    )
    capabilities = tuple(
        str(value)
        for value in _sequence(
            record.get("required_runtime_capabilities"),
            "evaluator process capabilities",
        )
    )
    return execution, execution_relative, execution_sha256, capabilities


def _walk_compiled_lanes(
    value: object,
    *,
    path: str = "execution",
) -> list[tuple[str, Mapping[str, object]]]:
    result: list[tuple[str, Mapping[str, object]]] = []
    if isinstance(value, Mapping):
        compiled = value.get("compiled")
        if isinstance(compiled, Mapping) and isinstance(
            compiled.get("stage_evaluators"), Mapping
        ):
            result.append((f"{path}.compiled", compiled))
        for key, child in value.items():
            result.extend(_walk_compiled_lanes(child, path=f"{path}.{key}"))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, child in enumerate(value):
            result.extend(_walk_compiled_lanes(child, path=f"{path}[{index}]"))
    return result


def _audit_source_evaluator(
    value: object,
    cell: CellSpec,
    *,
    context: str,
) -> int:
    evaluator = _mapping(value, context)
    kind = evaluator.get("kind")
    if kind == "chunked-symbolica-evaluator":
        chunks = _sequence(evaluator.get("chunks"), f"{context}.chunks")
        if not chunks:
            raise FinalAuditError(f"{context} has no source-evaluator leaves")
        return sum(
            _audit_source_evaluator(
                chunk,
                cell,
                context=f"{context}.chunks[{index}]",
            )
            for index, chunk in enumerate(chunks)
        )
    expected_kind = (
        "symjit-application-evaluator"
        if cell.measurement.backend == "jit"
        else "compiled-complex-evaluator"
    )
    expected_capability = _SOURCE_RUNTIME_CAPABILITY.get(cell.measurement.backend)
    if expected_capability is None:
        raise FinalAuditError(
            f"{cell.cell_id} has unsupported evaluator backend "
            f"{cell.measurement.backend!r}"
        )
    mismatches: list[str] = []
    if kind != expected_kind:
        mismatches.append("kind")
    if evaluator.get("runtime_capability") != expected_capability:
        mismatches.append("runtime_capability")
    if (
        cell.measurement.backend == "jit"
        and evaluator.get("optimization_level")
        != cell.measurement.jit_optimization_level
    ):
        mismatches.append("optimization_level")
    if mismatches:
        raise FinalAuditError(
            f"{context} differs from the {cell.measurement.backend} source "
            "evaluator contract: " + ", ".join(mismatches)
        )
    return 1


def _audit_compiled_execution(
    execution: Mapping[str, object],
    cell: CellSpec,
) -> tuple[int, int]:
    if execution.get("kind") != "pyamplicol-runtime-execution":
        raise FinalAuditError("compiled execution manifest kind is invalid")
    expected_application, expected_source = _expected_evaluator_abis(cell)
    lanes = _walk_compiled_lanes(execution)
    if not lanes:
        raise FinalAuditError("compiled execution contains no executable lanes")
    plane_count = 0
    direct_leaf_count = 0
    for lane_path, compiled in lanes:
        stages = _mapping(
            compiled.get("stage_evaluators"),
            f"{lane_path}.stage_evaluators",
        )
        stage_records = list(_sequence(stages.get("stages"), f"{lane_path}.stages"))
        stage_records.append(
            _mapping(
                stages.get("amplitude_stage"),
                f"{lane_path}.amplitude_stage",
            )
        )
        if not stage_records:
            raise FinalAuditError(f"{lane_path} has no fused stages")
        for index, raw_stage in enumerate(stage_records):
            stage = _mapping(raw_stage, f"{lane_path}.stage[{index}]")
            arena = _mapping(
                stage.get("compiled_plane_arena"),
                f"{lane_path}.stage[{index}].compiled_plane_arena",
            )
            expected = {
                "schema_version": 1,
                "kind": "compiled-plane-arena-stage",
                "application_abi": expected_application,
                "source_application_abi": expected_source,
                "element_layout": "split-complex-component-major",
                "output_operation": "overwrite",
                "output_factor": "identity",
                "input_output_aliasing": "forbidden",
                "output_output_aliasing": "forbidden",
            }
            mismatches = [
                field for field, value in expected.items() if arena.get(field) != value
            ]
            inputs = _sequence(
                arena.get("input_bindings"), f"{lane_path}.arena.input_bindings"
            )
            outputs = _sequence(
                arena.get("output_bindings"),
                f"{lane_path}.arena.output_bindings",
            )
            leaves = _sequence(arena.get("leaves"), f"{lane_path}.arena.leaves")
            if len(inputs) != stage.get("parameter_count"):
                mismatches.append("input_bindings")
            if len(outputs) != stage.get("output_length"):
                mismatches.append("output_bindings")
            if not leaves:
                mismatches.append("leaves")
            source_leaf_count = _audit_source_evaluator(
                stage.get("evaluator"),
                cell,
                context=f"{lane_path}.stage[{index}].evaluator",
            )
            if source_leaf_count != len(leaves):
                mismatches.append("source_evaluator_leaf_count")
            for leaf in leaves:
                leaf_record = _mapping(leaf, f"{lane_path}.arena.leaf")
                if leaf_record.get("source_application_abi") != expected_source:
                    mismatches.append("leaf.source_application_abi")
                if (
                    cell.measurement.backend == "jit"
                    and leaf_record.get("optimization_level")
                    != cell.measurement.jit_optimization_level
                ):
                    mismatches.append("leaf.optimization_level")
                if leaf_record.get("direct_codegen_optimization_level") != 3:
                    mismatches.append("leaf.direct_codegen_optimization_level")
            if mismatches:
                raise FinalAuditError(
                    f"{lane_path} has incompatible plane-Arena metadata: "
                    + ", ".join(sorted(set(mismatches)))
                )
            plane_count += 1
            direct_leaf_count += len(leaves)
        model_parameters = compiled.get("model_parameter_evaluator")
        if model_parameters is not None:
            _assert_model_parameter_not_direct(
                model_parameters,
                context=f"{lane_path}.model_parameter_evaluator",
            )
    return plane_count, direct_leaf_count


def _assert_model_parameter_not_direct(
    value: object,
    *,
    context: str,
) -> None:
    """Reject every compiled DirectApplication representation recursively."""

    if isinstance(value, Mapping):
        if value.get("kind") == "compiled-plane-arena-stage":
            raise FinalAuditError(
                f"{context} model-parameter evaluator entered plane Arena"
            )
        if "compiled_plane_arena" in value or "native_direct_application" in value:
            raise FinalAuditError(
                f"{context} model-parameter evaluator carries "
                "DirectApplication metadata"
            )
        if value.get("application_abi") in {
            _COMPILED_DIRECT_ABI,
            _NATIVE_COMPILED_DIRECT_ABI,
        }:
            raise FinalAuditError(
                f"{context} model-parameter evaluator uses a direct application ABI"
            )
        for name, child in value.items():
            _assert_model_parameter_not_direct(
                child,
                context=f"{context}.{name}",
            )
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, child in enumerate(value):
            _assert_model_parameter_not_direct(
                child,
                context=f"{context}[{index}]",
            )


def _assert_eager_model_parameter_not_direct(
    value: object,
    *,
    context: str,
) -> None:
    """Reject every eager DirectTable representation recursively."""

    if isinstance(value, Mapping):
        if "direct_table" in value:
            raise FinalAuditError(
                f"{context} model-parameter evaluator carries DirectTable metadata"
            )
        direct_markers = {
            _EAGER_DIRECT_CAPABILITY,
            _EAGER_DIRECT_DESCRIPTOR_ABI,
            _EAGER_DIRECT_BINDING_ABI,
            _EAGER_NATIVE_APPLICATION_ABI,
        }
        for name, child in value.items():
            if isinstance(child, str) and child in direct_markers:
                raise FinalAuditError(
                    f"{context}.{name} model-parameter evaluator uses eager "
                    "Direct-Arena metadata"
                )
            _assert_eager_model_parameter_not_direct(
                child,
                context=f"{context}.{name}",
            )
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, child in enumerate(value):
            _assert_eager_model_parameter_not_direct(
                child,
                context=f"{context}[{index}]",
            )


def _audit_eager_execution(
    artifact: Path,
    manifest: object,
    execution: Mapping[str, object],
    cell: CellSpec,
) -> int:
    if execution.get("kind") != "pyamplicol-runtime-eager-execution":
        raise FinalAuditError("eager execution manifest kind is invalid")
    plan = _mapping(execution.get("plan"), "eager execution.plan")
    expected = {
        "eager_plan_abi": "pyamplicol-eager-plan-v3",
        "lowering_input_abi": "pyamplicol-eager-lowering-input-v1",
        "runtime_layout_abi": "pyamplicol-eager-runtime-layout-v1",
    }
    for field, value in expected.items():
        if plan.get(field) != value:
            raise FinalAuditError(f"eager execution.plan.{field} is incompatible")
    if execution.get("eager_plan_abi") != expected["eager_plan_abi"]:
        raise FinalAuditError("eager execution eager_plan_abi is incompatible")
    extensions = _mapping(manifest.extensions, "artifact.extensions")
    prepared = _mapping(
        extensions.get("eager_prepared_pack"),
        "artifact.extensions.eager_prepared_pack",
    )
    if (
        prepared.get("kind") != "pyamplicol-prepared-kernel-pack-identity"
        or prepared.get("schema_version") != 1
        or prepared.get("backend") != cell.measurement.backend
        or prepared.get("eager_kernel_abi") != "pyamplicol-eager-kernel-v1"
        or not isinstance(prepared.get("identity_sha256"), str)
        or _SHA256_RE.fullmatch(str(prepared.get("identity_sha256"))) is None
    ):
        raise FinalAuditError("eager prepared-pack identity is incompatible")
    kernel_pack = _mapping(execution.get("kernel_pack"), "eager kernel_pack")
    raw_pack = kernel_pack.get("manifest_path")
    pack, _pack_path, _pack_sha256 = _authenticated_json_payload(
        artifact,
        manifest,
        raw_pack,
        process_id=None,
        context="eager kernel pack",
    )
    kernels = _sequence(pack.get("kernels"), "eager kernel pack.kernels")
    prepared_kernel_count = prepared.get("kernel_count")
    if (
        isinstance(prepared_kernel_count, bool)
        or not isinstance(prepared_kernel_count, int)
        or prepared_kernel_count < len(kernels)
        or not kernels
    ):
        raise FinalAuditError("eager prepared-pack kernel inventory is invalid")
    direct_count = 0
    expected_source = _expected_evaluator_abis(cell)[1]
    for index, raw_kernel in enumerate(kernels):
        kernel = _mapping(raw_kernel, f"eager kernel[{index}]")
        evaluator = _mapping(
            kernel.get("f64_evaluator_manifest"),
            f"eager kernel[{index}].f64_evaluator_manifest",
        )
        direct = evaluator.get("direct_table")
        if kernel.get("contract_kind") == "model-parameter":
            _assert_eager_model_parameter_not_direct(
                kernel,
                context=f"eager kernel[{index}]",
            )
            continue
        source_leaf_count = _audit_source_evaluator(
            evaluator,
            cell,
            context=f"eager kernel[{index}].f64_evaluator_manifest",
        )
        if source_leaf_count != 1:
            raise FinalAuditError(
                f"eager kernel[{index}] has a non-canonical source leaf count"
            )
        table = _mapping(direct, f"eager kernel[{index}].direct_table")
        expected = {
            "capability": _EAGER_DIRECT_CAPABILITY,
            "descriptor_abi": _EAGER_DIRECT_DESCRIPTOR_ABI,
            "binding_abi": _EAGER_DIRECT_BINDING_ABI,
            "source_application_abi": expected_source,
        }
        mismatches = [
            field for field, value in expected.items() if table.get(field) != value
        ]
        if (
            cell.measurement.backend == "jit"
            and evaluator.get("optimization_level")
            != cell.measurement.jit_optimization_level
        ):
            mismatches.append("optimization_level")
        if mismatches:
            raise FinalAuditError(
                f"eager kernel[{index}] DirectTable identity differs: "
                + ", ".join(mismatches)
            )
        direct_count += 1
    if direct_count == 0:
        raise FinalAuditError("eager artifact contains no direct-Arena kernels")
    return direct_count


def _audit_recurrence_source_pack(
    artifact: Path,
    manifest: object,
    execution: Mapping[str, object],
    cell: CellSpec,
    *,
    execution_manifest_path: str,
    execution_manifest_sha256: str,
) -> Mapping[str, object]:
    if (
        cell.measurement.backend != "jit"
        or cell.measurement.jit_optimization_level != 2
    ):
        raise FinalAuditError(
            "recurrence report cells require an authenticated JIT O2 source pack"
        )
    kernel_pack = _mapping(execution.get("kernel_pack"), "recurrence kernel_pack")
    pack, pack_path, pack_sha256 = _authenticated_json_payload(
        artifact,
        manifest,
        kernel_pack.get("manifest_path"),
        process_id=None,
        context="recurrence prepared kernel pack",
    )
    optimization = _mapping(
        pack.get("optimization_settings"),
        "recurrence prepared kernel pack.optimization_settings",
    )
    plan = _mapping(execution.get("plan"), "recurrence execution.plan")
    expected_execution_abis = {
        "direct_template_abi": _RECURRENCE_DIRECT_TEMPLATE_ABI,
        "direct_backend_abi": _RECURRENCE_DIRECT_BACKEND_ABI,
    }
    for field, expected in expected_execution_abis.items():
        if execution.get(field) != expected or plan.get(field) != expected:
            raise FinalAuditError(f"recurrence execution/plan {field} is incompatible")
    prepared_kernel_pack_digest = _require_sha256(
        execution.get("prepared_kernel_pack_digest"),
        "recurrence execution.prepared_kernel_pack_digest",
    )
    direct_template_catalog_digest = _require_sha256(
        execution.get("direct_template_catalog_digest"),
        "recurrence execution.direct_template_catalog_digest",
    )
    if (
        plan.get("prepared_kernel_pack_digest") != prepared_kernel_pack_digest
        or plan.get("direct_template_catalog_digest") != direct_template_catalog_digest
    ):
        raise FinalAuditError("recurrence execution/plan source digests differ")
    kernel_payload_root = _canonical_relative_path(
        kernel_pack.get("payload_root"),
        "recurrence kernel payload root",
    ).as_posix()
    mismatches = []
    if pack.get("backend") != "jit":
        mismatches.append("backend")
    if optimization.get("backend") != "jit":
        mismatches.append("optimization_settings.backend")
    if optimization.get("jit_optimization_level") != 2:
        mismatches.append("optimization_settings.jit_optimization_level")
    direct_catalog = _mapping(
        pack.get("recurrence_direct_template"),
        "recurrence prepared kernel pack.recurrence_direct_template",
    )
    templates = _sequence(
        direct_catalog.get("templates"),
        "recurrence prepared kernel pack.recurrence_direct_template.templates",
    )
    expected_direct = {
        "abi": _RECURRENCE_DIRECT_TEMPLATE_ABI,
        "backend": "jit",
        "backend_abi": _RECURRENCE_DIRECT_BACKEND_ABI,
        "canonicalization_abi": _RECURRENCE_DIRECT_CANONICALIZATION_ABI,
        "optimization_level": 2,
        "portable": True,
    }
    mismatches.extend(
        f"recurrence_direct_template.{field}"
        for field, expected in expected_direct.items()
        if direct_catalog.get(field) != expected
    )
    if not templates:
        mismatches.append("recurrence_direct_template.templates")
    if (
        direct_catalog.get("prepared_kernel_pack_digest") != prepared_kernel_pack_digest
        or direct_catalog.get("catalog_digest") != direct_template_catalog_digest
    ):
        mismatches.append("recurrence_direct_template.execution_digest_links")
    if _digest_mapping_without(direct_catalog, "catalog_digest") != (
        direct_template_catalog_digest
    ):
        mismatches.append("recurrence_direct_template.catalog_digest")
    if mismatches:
        raise FinalAuditError(
            "recurrence prepared kernel pack does not prove JIT O2: "
            + ", ".join(mismatches)
        )

    evaluator_members = _authenticated_evaluator_container_members(
        artifact,
        manifest,
    )
    authenticated_source_payloads: dict[str, str] = {}

    def authenticate_source_payload(relative: str, *, context: str) -> str:
        cached = authenticated_source_payloads.get(relative)
        if cached is not None:
            return cached
        result = _authenticated_recurrence_source_payload(
            artifact,
            manifest,
            evaluator_members,
            relative,
            context=context,
        )
        authenticated_source_payloads[relative] = result
        return result

    kernels = _sequence(
        pack.get("kernels"),
        "recurrence prepared kernel pack.kernels",
    )
    variants = _sequence(
        pack.get("kernel_variants"),
        "recurrence prepared kernel pack.kernel_variants",
    )
    if not kernels:
        raise FinalAuditError("recurrence prepared kernel pack has no source kernels")
    source_leaves_by_kernel: dict[int, set[tuple[str, str]]] = {}
    source_evaluator_leaf_count = 0
    for index, raw_kernel in enumerate(kernels):
        kernel = _mapping(
            raw_kernel,
            f"recurrence prepared kernel pack.kernels[{index}]",
        )
        kernel_id = kernel.get("kernel_id")
        if (
            isinstance(kernel_id, bool)
            or not isinstance(kernel_id, int)
            or kernel_id < 0
            or kernel_id in source_leaves_by_kernel
        ):
            raise FinalAuditError(
                f"recurrence prepared kernel pack.kernels[{index}] "
                "has an invalid kernel ID"
            )
        leaves = _audit_recurrence_source_evaluator_leaves(
            kernel.get("f64_evaluator_manifest"),
            payload_root=kernel_payload_root,
            authenticate_payload=authenticate_source_payload,
            context=(
                f"recurrence prepared kernel pack.kernels[{index}]"
                ".f64_evaluator_manifest"
            ),
        )
        source_evaluator_leaf_count += len(leaves)
        source_leaves_by_kernel[kernel_id] = set(leaves)
    for index, raw_variant in enumerate(variants):
        variant = _mapping(
            raw_variant,
            f"recurrence prepared kernel pack.kernel_variants[{index}]",
        )
        source_evaluator_leaf_count += len(
            _audit_recurrence_source_evaluator_leaves(
                variant.get("f64_evaluator_manifest"),
                payload_root=kernel_payload_root,
                authenticate_payload=authenticate_source_payload,
                context=(
                    "recurrence prepared kernel pack."
                    f"kernel_variants[{index}].f64_evaluator_manifest"
                ),
            )
        )

    prepared_direct_template_count = 0
    for index, raw_template in enumerate(templates):
        template = _mapping(
            raw_template,
            f"recurrence prepared kernel pack.direct_template[{index}]",
        )
        if (
            template.get("abi") != _RECURRENCE_DIRECT_TEMPLATE_ABI
            or template.get("backend") != "jit"
            or template.get("portable") is not True
            or template.get("optimization_level") != 2
        ):
            raise FinalAuditError(
                f"recurrence direct template[{index}] contract is incompatible"
            )
        if _digest_mapping_without(template, "semantic_digest") != (
            template.get("semantic_digest")
        ):
            raise FinalAuditError(
                f"recurrence direct template[{index}] semantic digest is invalid"
            )
        binding = _mapping(
            template.get("payload_binding"),
            f"recurrence direct template[{index}].payload_binding",
        )
        if (
            binding.get("abi") != _RECURRENCE_DIRECT_PAYLOAD_BINDING_ABI
            or _recurrence_payload_binding_digest(binding)
            != binding.get("payload_digest")
        ):
            raise FinalAuditError(
                f"recurrence direct template[{index}] payload-binding "
                "contract is invalid"
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
                raise FinalAuditError(
                    f"recurrence direct template[{index}] intrinsic source "
                    "contract is invalid"
                )
            continue
        if binding_kind != "prepared-direct-call":
            raise FinalAuditError(
                f"recurrence direct template[{index}] is not executable"
            )
        prepared_direct_template_count += 1
        source_path = _canonical_relative_path(
            binding.get("source_application_path"),
            f"recurrence direct template[{index}] source application",
        ).as_posix()
        source_sha256 = _require_sha256(
            binding.get("source_application_sha256"),
            f"recurrence direct template[{index}] source application digest",
        )
        prepared_kernel_id = binding.get("prepared_kernel_id")
        if (
            binding.get("source_application_abi") != _SYMJIT_APPLICATION_ABI
            or binding.get("direct_application_abi")
            != _RECURRENCE_JIT_DIRECT_APPLICATION_ABI
            or binding.get("payload_paths") != [source_path]
            or isinstance(prepared_kernel_id, bool)
            or not isinstance(prepared_kernel_id, int)
            or prepared_kernel_id < 0
            or (source_path, source_sha256)
            not in source_leaves_by_kernel.get(prepared_kernel_id, set())
        ):
            raise FinalAuditError(
                f"recurrence direct template[{index}] source application "
                "is not bound to its prepared kernel"
            )
    if prepared_direct_template_count == 0 or source_evaluator_leaf_count == 0:
        raise FinalAuditError(
            "recurrence direct-template pack has no source application evidence"
        )
    return {
        "kind": "authenticated-recurrence-direct-template-source-v1",
        "optimization_level": 2,
        "direct_template_count": len(templates),
        "prepared_direct_template_count": prepared_direct_template_count,
        "source_evaluator_leaf_count": source_evaluator_leaf_count,
        "source_application_abi": _SYMJIT_APPLICATION_ABI,
        "direct_application_abi": _RECURRENCE_JIT_DIRECT_APPLICATION_ABI,
        "prepared_kernel_pack_digest": prepared_kernel_pack_digest,
        "direct_template_catalog_digest": direct_template_catalog_digest,
        "execution_manifest_path": execution_manifest_path,
        "execution_manifest_sha256": execution_manifest_sha256,
        "kernel_pack_path": pack_path,
        "kernel_pack_sha256": pack_sha256,
    }


def _require_sha256(value: object, context: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise FinalAuditError(f"{context} must be a lowercase SHA-256")
    return value


def _digest_mapping_without(value: Mapping[str, object], field: str) -> str:
    payload = dict(value)
    payload.pop(field, None)
    return digest_json(payload)


def _recurrence_payload_binding_digest(binding: Mapping[str, object]) -> str:
    if binding.get("kind") != "rusticol-intrinsic":
        return _digest_mapping_without(binding, "payload_digest")
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
        return digest_json({field: binding.get(field) for field in fields})
    # Source/finalization intrinsic digests include semantic-catalog fields
    # which are not duplicated in the payload binding.  The opaque digest is
    # nevertheless bound by the recomputed template and catalog identities.
    return _require_sha256(
        binding.get("payload_digest"),
        "recurrence intrinsic payload digest",
    )


def _authenticated_evaluator_container_members(
    artifact: Path,
    manifest: object,
) -> Mapping[str, tuple[int, str]]:
    extensions = _mapping(
        getattr(manifest, "extensions", None),
        "artifact.extensions",
    )
    raw_container = extensions.get("evaluator_payload_container")
    if raw_container is None:
        return {}
    container = _mapping(
        raw_container,
        "artifact.extensions.evaluator_payload_container",
    )
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
        set(container) != required
        or container.get("kind") != "pyamplicol-evaluator-payload-container"
        or container.get("schema_version") != 1
        or container.get("storage_abi") != "pacbin-v1"
    ):
        raise FinalAuditError("artifact evaluator payload container contract drifted")
    member_count = container.get("member_count")
    unpacked_size = container.get("unpacked_size_bytes")
    index_sha256 = _require_sha256(
        container.get("index_sha256"),
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
        raise FinalAuditError(
            "artifact evaluator payload container inventory is invalid"
        )
    data, _canonical, _sha256 = _authenticated_payload_bytes(
        artifact,
        manifest,
        container.get("path"),
        role="evaluator-state",
        media_type="application/octet-stream",
        process_id=None,
        context="evaluator payload container",
    )
    try:
        from pyamplicol.generation.evaluator_container import PacbinReader

        with PacbinReader.open(io.BytesIO(data), verify_payloads=True) as reader:
            index = reader.index
            members = tuple(reader.members)
    except (OSError, TypeError, ValueError) as error:
        raise FinalAuditError(
            "authenticated evaluator payload container is invalid"
        ) from error
    if (
        len(members) != member_count
        or sum(member.length for member in members) != unpacked_size
        or index.index_sha256 != index_sha256
    ):
        raise FinalAuditError(
            "authenticated evaluator payload container index metadata drifted"
        )
    return {
        member.logical_path: (int(member.kind), member.sha256) for member in members
    }


def _authenticated_recurrence_source_payload(
    artifact: Path,
    manifest: object,
    evaluator_members: Mapping[str, tuple[int, str]],
    relative: str,
    *,
    context: str,
) -> str:
    canonical = _canonical_relative_path(relative, context).as_posix()
    payloads = getattr(manifest, "payloads", None)
    if not isinstance(payloads, Sequence):
        raise FinalAuditError(f"{context} has no artifact payload inventory")
    loose = [
        payload for payload in payloads if getattr(payload, "path", None) == canonical
    ]
    packed = evaluator_members.get(canonical)
    if len(loose) + (packed is not None) != 1:
        raise FinalAuditError(f"{context} payload is missing or ambiguous")
    if loose:
        _data, _canonical, sha256 = _authenticated_payload_bytes(
            artifact,
            manifest,
            canonical,
            role="evaluator-state",
            media_type="application/octet-stream",
            process_id=None,
            context=context,
        )
        return sha256
    assert packed is not None
    kind, sha256 = packed
    if kind != 1:
        raise FinalAuditError(f"{context} is not a packed SymJIT application")
    return _require_sha256(sha256, f"{context} packed member digest")


def _audit_recurrence_source_evaluator_leaves(
    value: object,
    *,
    payload_root: str,
    authenticate_payload: Callable[..., str],
    context: str,
) -> list[tuple[str, str]]:
    evaluator = _mapping(value, context)
    if evaluator.get("kind") == "chunked-symbolica-evaluator":
        chunks = _sequence(evaluator.get("chunks"), f"{context}.chunks")
        if not chunks:
            raise FinalAuditError(f"{context} has no source-evaluator leaves")
        leaves: list[tuple[str, str]] = []
        for index, chunk in enumerate(chunks):
            leaves.extend(
                _audit_recurrence_source_evaluator_leaves(
                    chunk,
                    payload_root=payload_root,
                    authenticate_payload=authenticate_payload,
                    context=f"{context}.chunks[{index}]",
                )
            )
        return leaves
    mismatches = []
    if evaluator.get("kind") != "symjit-application-evaluator":
        mismatches.append("kind")
    if evaluator.get("backend") != "jit":
        mismatches.append("backend")
    if evaluator.get("runtime_capability") != _SOURCE_RUNTIME_CAPABILITY["jit"]:
        mismatches.append("runtime_capability")
    if evaluator.get("application_abi") != _SYMJIT_APPLICATION_ABI:
        mismatches.append("application_abi")
    if evaluator.get("optimization_level") != 2:
        mismatches.append("optimization_level")
    if mismatches:
        raise FinalAuditError(
            f"{context} differs from the JIT O2 source evaluator contract: "
            + ", ".join(mismatches)
        )
    application_path = _canonical_relative_path(
        evaluator.get("application_path"),
        f"{context}.application",
    ).as_posix()
    payload_path = (
        PurePosixPath(payload_root) / PurePosixPath(application_path)
    ).as_posix()
    sha256 = authenticate_payload(
        payload_path,
        context=f"{context}.application",
    )
    return [(application_path, sha256)]


def _audit_recurrence_execution(
    execution: Mapping[str, object],
    cell: CellSpec,
) -> int:
    if execution.get("kind") != "pyamplicol-runtime-recurrence-execution":
        raise FinalAuditError("recurrence execution manifest kind is invalid")
    expected = {
        "builder_input_abi": "pyamplicol-recurrence-builder-input-v2",
        "recurrence_plan_abi": "pyamplicol-recurrence-plan-v2",
        "runtime_layout_abi": "pyamplicol-recurrence-runtime-layout-v2",
        "direct_template_abi": "pyamplicol-recurrence-direct-template-v1",
        "direct_backend_abi": "rusticol.recurrence-direct-backend.v1",
    }
    mismatches = [
        field for field, value in expected.items() if execution.get(field) != value
    ]
    plan = _mapping(execution.get("plan"), "recurrence execution.plan")
    mismatches.extend(
        f"plan.{field}" for field, value in expected.items() if plan.get(field) != value
    )
    summary = _mapping(
        plan.get("inspection_summary"),
        "recurrence execution.plan.inspection_summary",
    )
    arena = _mapping(
        summary.get("direct_arena"),
        "recurrence execution.plan.inspection_summary.direct_arena",
    )
    for field in ("packed_input_bytes", "packed_output_bytes", "scatter_bytes"):
        if arena.get(field) != 0:
            mismatches.append(f"direct_arena.{field}")
    capabilities = set(
        str(value)
        for value in _sequence(
            execution.get("required_runtime_capabilities"),
            "recurrence capabilities",
        )
    )
    expected_color = (
        "rusticol.recurrence-color.lc.v1"
        if cell.measurement.accuracy is Accuracy.LC
        else "rusticol.recurrence-color.contracted.v1"
    )
    if capabilities != {
        _ARENA_CAPABILITY[ExecutionMode.RECURRENCE],
        expected_color,
    }:
        mismatches.append("required_runtime_capabilities")
    if mismatches:
        raise FinalAuditError(
            "recurrence direct-Arena identity differs: "
            + ", ".join(sorted(set(mismatches)))
        )
    row_groups = arena.get("row_group_count")
    if (
        isinstance(row_groups, bool)
        or not isinstance(row_groups, int)
        or row_groups < 1
    ):
        raise FinalAuditError("recurrence direct Arena has no row groups")
    return row_groups


def audit_artifact(
    cell: CellSpec,
    artifact: Path,
    process_id: str,
) -> ArtifactEvidence:
    """Authenticate one artifact and prove the mode-specific Arena ABI."""

    from pyamplicol.artifacts import inspect_artifact, load_manifest

    inspection = inspect_artifact(artifact)
    if inspection.integrity != "verified":
        raise FinalAuditError(f"artifact integrity is not verified: {artifact}")
    # ``inspect_artifact`` has just verified every payload.  Re-load the typed
    # manifest without hashing the potentially large evaluator container twice;
    # every identity-bearing JSON/TOML payload below is then re-read through a
    # checked descriptor against its own manifest size and SHA-256.
    manifest = load_manifest(artifact, verify_payloads=False)
    effective_config = _authenticated_effective_config(artifact, manifest)
    _audit_effective_config_mapping(
        cell,
        effective_config,
        context="authenticated effective configuration",
    )
    _audit_model_source(cell, effective_config)
    process_matches = [
        process for process in manifest.processes if process.get("id") == process_id
    ]
    inspected_matches = [
        process for process in inspection.processes if process.id == process_id
    ]
    if len(process_matches) != 1 or len(inspected_matches) != 1:
        raise FinalAuditError(
            f"artifact process {process_id!r} is missing or ambiguous"
        )
    process = process_matches[0]
    inspected = inspected_matches[0]
    expected_layout = (
        "all-flow-union" if cell.workload is Workload.ALL_FLOW else "topology-replay"
    )
    if (
        process.get("expression") != cell.process
        or process.get("color_accuracy") != cell.measurement.accuracy.value
        or inspected.execution_mode != cell.measurement.execution_mode.value
        or inspected.generation_specialized_axes
        or inspected.selected_source_helicities
        or inspected.selected_color_sector_ids
        or (
            cell.measurement.accuracy is Accuracy.LC
            and inspected.lc_flow_layout != expected_layout
        )
    ):
        raise FinalAuditError(
            f"artifact physics/execution contract differs for {cell.cell_id}"
        )
    (
        execution,
        execution_path,
        execution_sha256,
        indexed_capabilities,
    ) = _find_process_execution(artifact, manifest, process_id)
    capabilities = tuple(
        str(value) for value in process["required_runtime_capabilities"]
    )
    if capabilities != indexed_capabilities:
        raise FinalAuditError("artifact and evaluator-index capabilities differ")
    expected_capability = _ARENA_CAPABILITY[cell.measurement.execution_mode]
    if expected_capability not in capabilities:
        raise FinalAuditError(
            f"artifact does not require final Arena capability {expected_capability}"
        )
    if cell.measurement.execution_mode is ExecutionMode.COMPILED:
        arena_count, direct_leaf_count = _audit_compiled_execution(execution, cell)
        source_jit_identity = None
    elif cell.measurement.execution_mode is ExecutionMode.EAGER:
        arena_count = _audit_eager_execution(artifact, manifest, execution, cell)
        direct_leaf_count = arena_count
        source_jit_identity = None
    else:
        source_jit_identity = _audit_recurrence_source_pack(
            artifact,
            manifest,
            execution,
            cell,
            execution_manifest_path=execution_path,
            execution_manifest_sha256=execution_sha256,
        )
        arena_count = _audit_recurrence_execution(execution, cell)
        direct_leaf_count = 0
    return ArtifactEvidence(
        artifact_id=manifest.artifact_id,
        process_id=process_id,
        runtime_version=str(manifest.runtime["engine_version"]),
        runtime_capabilities=capabilities,
        execution_manifest_path=execution_path,
        execution_manifest_sha256=execution_sha256,
        execution_mode=cell.measurement.execution_mode.value,
        arena_record_count=arena_count,
        direct_leaf_count=direct_leaf_count,
        effective_config=effective_config,
        source_jit_identity=source_jit_identity,
    )


def _selector_kwargs(
    cell: CellSpec,
    measurement: Mapping[str, object],
) -> dict[str, tuple[str, ...] | None]:
    if cell.measurement.accuracy is not Accuracy.LC:
        return {"helicities": None, "color_flows": None}
    contract = SelectorContract.from_mapping(
        _mapping(
            measurement.get("selector_contract"),
            f"{cell.cell_id}.selector_contract",
        )
    )
    if cell.workload is Workload.SELECTED_FLOW:
        return {
            "helicities": None,
            "color_flows": contract.selected_color_flow_ids,
        }
    return {
        "helicities": contract.all_flow_helicity_ids,
        "color_flows": None,
    }


def _real_nonnegative(value: object, context: str) -> float:
    number = complex(value)
    if not math.isfinite(number.real) or not math.isfinite(number.imag):
        raise FinalAuditError(f"{context} is not finite")
    if abs(number.imag) > 1.0e-9 * max(abs(number.real), 1.0):
        raise FinalAuditError(f"{context} has a non-negligible imaginary part")
    result = float(number.real)
    if result < -ABSOLUTE_TOLERANCE:
        raise FinalAuditError(f"{context} is materially negative")
    return max(result, 0.0)


def _resolved_differences(
    optimized: Sequence[object],
    resolved_totals: Sequence[object],
    *,
    context: str,
) -> tuple[float, float]:
    if len(optimized) != len(resolved_totals):
        raise FinalAuditError(f"{context} optimized/resolved lengths differ")
    maximum_absolute = 0.0
    maximum_relative = 0.0
    for candidate, baseline in zip(optimized, resolved_totals, strict=True):
        absolute = abs(complex(candidate) - complex(baseline))
        relative = absolute / max(abs(complex(candidate)), 1.0e-300)
        maximum_absolute = max(maximum_absolute, absolute)
        maximum_relative = max(maximum_relative, relative)
    return maximum_absolute, maximum_relative


def _replay_cell(
    runtime: object,
    reference: _ArtifactReference,
    evidence: ArtifactEvidence,
) -> _ReplayObservation:
    cell = reference.cell
    measurement = reference.measurement
    context = f"{cell.cell_id}.replay"
    if getattr(runtime, "artifact_id", None) != evidence.artifact_id:
        raise FinalAuditError(f"{context} loaded artifact ID differs")
    expected_mode = cell.measurement.execution_mode.value
    loaded_mode = getattr(runtime, "execution_mode", None)
    if loaded_mode != expected_mode or evidence.execution_mode != expected_mode:
        raise FinalAuditError(
            f"{context} loaded execution mode differs from requested {expected_mode!r}"
        )
    provenance = _mapping(
        measurement.get("provenance"),
        f"{context}.provenance",
    )
    runtime_identity = _mapping(
        provenance.get("runtime_identity"),
        f"{context}.provenance.runtime_identity",
    )
    if runtime_identity.get("loaded_execution_mode") != loaded_mode:
        raise FinalAuditError(
            f"{context} stored loaded_execution_mode differs from the runtime"
        )
    validate_runtime_contract(cell, runtime)  # type: ignore[arg-type]
    points = (
        runtime_validation_points(runtime)
        if cell.measurement.model in {ModelKey.SCALAR_CONTACT, ModelKey.SCALAR_GRAVITY}
        else shared_validation_points(cell.process)
    )
    raw_selector = measurement.get("selector_contract")
    if raw_selector is not None:
        validate_selector_contract(
            runtime,  # type: ignore[arg-type]
            SelectorContract.from_mapping(_mapping(raw_selector, context)),
            points,
        )
    selectors = _selector_kwargs(cell, measurement)
    optimized = tuple(
        runtime.evaluate(  # type: ignore[attr-defined]
            points,
            precision=16,
            **selectors,
        )
    )
    if not optimized:
        raise FinalAuditError(f"{context} returned no matrix elements")
    observed_matrix = _real_nonnegative(optimized[0], f"{context}.matrix_element")
    stored_matrix = _finite_number(
        measurement.get("matrix_element"), f"{context}.stored_matrix_element"
    )
    matrix_absolute = abs(observed_matrix - stored_matrix)
    matrix_relative = matrix_absolute / max(abs(stored_matrix), 1.0e-300)
    if matrix_absolute > ABSOLUTE_TOLERANCE and matrix_relative > RELATIVE_TOLERANCE:
        raise FinalAuditError(
            f"{context} no longer reproduces the stored matrix element"
        )
    resolved = runtime.evaluate_resolved(  # type: ignore[attr-defined]
        points,
        precision=16,
        **selectors,
    )
    resolved_totals = tuple(resolved.total())
    maximum_absolute, maximum_relative = _resolved_differences(
        optimized,
        resolved_totals,
        context=context,
    )
    if maximum_absolute > ABSOLUTE_TOLERANCE and maximum_relative > RELATIVE_TOLERANCE:
        raise FinalAuditError(f"{context} optimized/resolved values disagree")
    stored_resolved = _mapping(
        _mapping(measurement.get("validation"), f"{context}.validation").get(
            "resolved_sum"
        ),
        f"{context}.validation.resolved_sum",
    )
    for field, observed in (
        ("maximum_absolute_difference", maximum_absolute),
        ("maximum_relative_difference", maximum_relative),
    ):
        stored = _finite_number(stored_resolved.get(field), f"{context}.{field}")
        if not math.isclose(
            observed,
            stored,
            rel_tol=RELATIVE_TOLERANCE,
            abs_tol=ABSOLUTE_TOLERANCE,
        ):
            raise FinalAuditError(
                f"{context} recomputed {field} differs from stored evidence"
            )
    if cell.measurement.model in {
        ModelKey.SCALAR_CONTACT,
        ModelKey.SCALAR_GRAVITY,
    }:
        high_precision = tuple(
            runtime.evaluate(  # type: ignore[attr-defined]
                points,
                precision=32,
                **selectors,
            )
        )
        if not high_precision:
            raise FinalAuditError(f"{context} precision-32 evaluation is empty")
        high_value = _real_nonnegative(high_precision[0], f"{context}.precision32")
        validation = _mapping(measurement.get("validation"), f"{context}.validation")
        _audit_pointwise(
            validation.get("high_precision"),
            context=f"{context}.validation.high_precision",
            expected_candidate=observed_matrix,
            expected_baseline=high_value,
            expected_relative_tolerance=RELATIVE_TOLERANCE,
        )
    return _ReplayObservation(
        matrix_element=observed_matrix,
        resolved_maximum_absolute=maximum_absolute,
        resolved_maximum_relative=maximum_relative,
    )


def _default_runtime_loader(path: Path, process_id: str) -> object:
    from pyamplicol.api import Runtime

    return Runtime.load(path, process=process_id)


def _entry_map(
    caches: Mapping[str, Mapping[str, object]],
) -> dict[str, Mapping[str, object]]:
    result: dict[str, Mapping[str, object]] = {}
    for cache_name, payload in caches.items():
        entries = _sequence(payload.get("entries"), f"{cache_name}.entries")
        for raw_entry in entries:
            entry = _mapping(raw_entry, f"{cache_name}.entry")
            cell_id = entry.get("cell_id")
            if not isinstance(cell_id, str) or not cell_id:
                raise FinalAuditError(f"{cache_name} contains an invalid cell ID")
            if cell_id in result:
                raise FinalAuditError(f"duplicate cache cell {cell_id!r}")
            result[cell_id] = entry
    return result


def audit_final_report(
    repo_root: Path,
    *,
    expected_source_revision: str,
    expected_publication_revision: str | None = None,
    max_n_final: int = 4,
    expected_cell_count: int | None = _EXPECTED_N4_CELL_COUNT,
    replay: bool = True,
    catalog: ReportCatalog = REPORT_CATALOG,
    service: ReportService | None = None,
    active_runtime: Mapping[str, object] | None = None,
    artifact_auditor: ArtifactAuditor = audit_artifact,
    runtime_loader: RuntimeLoader = _default_runtime_loader,
    source_auditor: SourceAuditor = _require_report_source_checkout,
    runtime_auditor: RuntimeAuditor = _audit_active_runtime,
    verify_render: bool = True,
    verify_pdf: bool = True,
    pdf_auditor: PdfAuditor = _audit_pdf,
) -> dict[str, object]:
    """Audit one cooperative report snapshot under the publication lock."""

    root = repo_root.expanduser().resolve(strict=False)
    report = service or ReportService(ReportPaths.from_repo(root), catalog=catalog)
    with report.store.named_lock("report-writer"):
        return _audit_final_report_locked(
            root,
            expected_source_revision=expected_source_revision,
            expected_publication_revision=expected_publication_revision,
            max_n_final=max_n_final,
            expected_cell_count=expected_cell_count,
            replay=replay,
            catalog=catalog,
            service=report,
            active_runtime=active_runtime,
            artifact_auditor=artifact_auditor,
            runtime_loader=runtime_loader,
            source_auditor=source_auditor,
            runtime_auditor=runtime_auditor,
            verify_render=verify_render,
            verify_pdf=verify_pdf,
            pdf_auditor=pdf_auditor,
        )


def _audit_final_report_locked(
    repo_root: Path,
    *,
    expected_source_revision: str,
    expected_publication_revision: str | None = None,
    max_n_final: int = 4,
    expected_cell_count: int | None = _EXPECTED_N4_CELL_COUNT,
    replay: bool = True,
    catalog: ReportCatalog = REPORT_CATALOG,
    service: ReportService | None = None,
    active_runtime: Mapping[str, object] | None = None,
    artifact_auditor: ArtifactAuditor = audit_artifact,
    runtime_loader: RuntimeLoader = _default_runtime_loader,
    source_auditor: SourceAuditor = _require_report_source_checkout,
    runtime_auditor: RuntimeAuditor = _audit_active_runtime,
    verify_render: bool = True,
    verify_pdf: bool = True,
    pdf_auditor: PdfAuditor = _audit_pdf,
) -> dict[str, object]:
    """Audit every selected record while the report-writer lock is held."""

    if _GIT_SHA_RE.fullmatch(expected_source_revision) is None:
        raise FinalAuditError("expected source revision must be a full Git SHA")
    if (
        expected_publication_revision is not None
        and _GIT_SHA_RE.fullmatch(expected_publication_revision) is None
    ):
        raise FinalAuditError("expected publication revision must be a full Git SHA")
    if max_n_final < 1:
        raise FinalAuditError("max_n_final must be positive")
    root = repo_root.expanduser().resolve(strict=False)
    publication_lineage = _validated_publication_lineage(
        source_auditor(root, expected_source_revision),
        measurement_source_revision=expected_source_revision,
        expected_publication_revision=expected_publication_revision,
    )
    report = service or ReportService(ReportPaths.from_repo(root), catalog=catalog)
    render_result = (
        report.audit()
        if verify_render
        else {"cache_render_match": None, "skipped": True}
    )
    caches = report.load_caches()
    entries = _entry_map(caches)
    cells = tuple(
        sorted(
            (
                cell
                for cell in catalog.measurement_cells()
                if cell.n_final <= max_n_final
            ),
            key=lambda item: item.cell_id,
        )
    )
    if expected_cell_count is not None and len(cells) != expected_cell_count:
        raise FinalAuditError(
            f"catalog selected {len(cells)} cells, expected {expected_cell_count}"
        )
    pyamplicol_cells = tuple(
        cell
        for cell in cells
        if cell.measurement.execution_mode is not ExecutionMode.AMPLICOL
    )
    expected_legacy_revision = (
        _expected_legacy_revision(root)
        if any(
            cell.measurement.execution_mode is ExecutionMode.AMPLICOL for cell in cells
        )
        else ""
    )
    if pyamplicol_cells and active_runtime is None:
        active_runtime = runtime_auditor(
            expected_source_revision,
            root,
        )

    published_measurements: dict[str, Mapping[str, object]] = {}
    measurements: dict[str, Mapping[str, object]] = {}
    errors: list[str] = []
    for cell in cells:
        entry = entries.get(cell.cell_id)
        if entry is None:
            errors.append(f"{cell.cell_id}: cache entry is missing")
            continue
        expected_entry = reset_entry(cell)
        descriptor_mismatches = [
            field
            for field, value in expected_entry.items()
            if field != "measurement" and entry.get(field) != value
        ]
        if descriptor_mismatches:
            errors.append(
                f"{cell.cell_id}: cache descriptor differs: "
                + ", ".join(descriptor_mismatches)
            )
            continue
        published = _mapping(
            entry.get("measurement"),
            f"{cell.cell_id}.published_measurement",
        )
        current = report.store.load_current(cell.cell_id)
        if current is None:
            errors.append(f"{cell.cell_id}: authenticated current result is missing")
            continue
        if not publication_measurement_matches_current(
            published,
            current.result,
            report.paths,
        ):
            errors.append(
                f"{cell.cell_id}: checked-in cache is not the exact portable "
                "projection of authenticated current.result"
            )
            continue
        published_measurements[cell.cell_id] = published
        measurements[cell.cell_id] = _mapping(
            current.result,
            f"{cell.cell_id}.current.result",
        )
    if errors:
        rendered = "\n".join(f"- {message}" for message in errors[:50])
        suffix = (
            ""
            if len(errors) <= 50
            else f"\n- ... and {len(errors) - 50} additional failures"
        )
        raise FinalAuditError(
            "final report publication projection audit failed "
            f"({len(errors)}):\n{rendered}{suffix}"
        )

    references: list[_ArtifactReference] = []
    errors = []
    for cell in cells:
        measurement = measurements[cell.cell_id]
        baseline_cell = catalog.baseline_cell(cell)
        try:
            if baseline_cell is None:
                baseline = None
            else:
                baseline = measurements.get(baseline_cell.cell_id)
                if baseline is None:
                    raise FinalAuditError(
                        f"canonical baseline {baseline_cell.cell_id!r} is missing"
                    )
            reference = _audit_measurement(
                cell,
                measurement,
                baseline=baseline,
                expected_source_revision=expected_source_revision,
                expected_legacy_revision=expected_legacy_revision,
                active_runtime=active_runtime,
                report_paths=report.paths,
            )
            if reference is not None:
                published_reference = _artifact_reference(
                    cell,
                    published_measurements[cell.cell_id],
                    report_paths=report.paths,
                )
                if (
                    published_reference.path != reference.path
                    or published_reference.process_id != reference.process_id
                ):
                    raise FinalAuditError(
                        "portable artifact locator differs from raw current.result"
                    )
                references.append(reference)
        except Exception as error:  # collect every independently actionable cell
            errors.append(f"{cell.cell_id}: {error}")
    if errors:
        rendered = "\n".join(f"- {message}" for message in errors[:50])
        suffix = (
            ""
            if len(errors) <= 50
            else f"\n- ... and {len(errors) - 50} additional failures"
        )
        raise FinalAuditError(
            f"final report structural measurement audit failed ({len(errors)}):\n"
            f"{rendered}{suffix}"
        )

    legacy_cells = tuple(
        cell
        for cell in cells
        if cell.measurement.execution_mode is ExecutionMode.AMPLICOL
    )
    legacy_ids = {cell.cell_id for cell in legacy_cells}
    legacy_agreement_edges: list[tuple[str, str]] = []
    for candidate in cells:
        baseline = catalog.baseline_cell(candidate)
        if baseline is not None and baseline.cell_id in legacy_ids:
            legacy_agreement_edges.append((baseline.cell_id, candidate.cell_id))
    covered_legacy_ids = {baseline for baseline, _candidate in legacy_agreement_edges}
    missing_legacy_agreement = sorted(legacy_ids - covered_legacy_ids)
    if missing_legacy_agreement:
        preview = ", ".join(missing_legacy_agreement[:20])
        suffix = (
            ""
            if len(missing_legacy_agreement) <= 20
            else f", ... ({len(missing_legacy_agreement)} total)"
        )
        raise FinalAuditError(
            "fresh original-AmpliCol oracle cells have no successfully audited "
            f"pointwise-agreement consumer: {preview}{suffix}"
        )

    grouped: dict[tuple[Path, str], list[_ArtifactReference]] = defaultdict(list)
    for reference in references:
        grouped[(reference.path, reference.process_id)].append(reference)
    evidence_by_key: dict[tuple[Path, str], ArtifactEvidence] = {}
    errors = []
    for key, group in grouped.items():
        representative = group[0]
        try:
            evidence = artifact_auditor(
                representative.cell, representative.path, representative.process_id
            )
            evidence_by_key[key] = evidence
            for reference in group:
                if _shared_artifact_contract(
                    reference.cell
                ) != _shared_artifact_contract(representative.cell):
                    raise FinalAuditError(
                        "one artifact path is shared by incompatible report cells"
                    )
                provenance = _mapping(
                    reference.measurement.get("provenance"),
                    f"{reference.cell.cell_id}.provenance",
                )
                assert active_runtime is not None
                _audit_runtime_identity(
                    reference.cell,
                    provenance,
                    expected_source_revision=expected_source_revision,
                    active_runtime=active_runtime,
                    artifact=evidence,
                )
        except Exception as error:
            errors.append(f"{representative.path}: {error}")
    if errors:
        rendered = "\n".join(f"- {message}" for message in errors[:50])
        raise FinalAuditError(
            f"final report artifact/Arena audit failed ({len(errors)}):\n{rendered}"
        )

    replayed: dict[str, _ReplayObservation] = {}
    if replay:
        errors = []
        for key, group in grouped.items():
            path, process_id = key
            try:
                runtime = runtime_loader(path, process_id)
                evidence = evidence_by_key[key]
                for reference in group:
                    replayed[reference.cell.cell_id] = _replay_cell(
                        runtime, reference, evidence
                    )
            except Exception as error:
                errors.append(f"{path} ({process_id}): {error}")
            finally:
                if "runtime" in locals():
                    del runtime
                gc.collect()
        for cell in pyamplicol_cells:
            observation = replayed.get(cell.cell_id)
            baseline_cell = catalog.baseline_cell(cell)
            if observation is None or baseline_cell is None:
                continue
            baseline_observation = replayed.get(baseline_cell.cell_id)
            baseline_value = (
                baseline_observation.matrix_element
                if baseline_observation is not None
                else _finite_number(
                    measurements[baseline_cell.cell_id].get("matrix_element"),
                    f"{baseline_cell.cell_id}.matrix_element",
                )
            )
            absolute = abs(observation.matrix_element - baseline_value)
            relative = absolute / max(abs(baseline_value), 1.0e-300)
            if (
                absolute > ABSOLUTE_TOLERANCE
                and relative > _expected_pointwise_tolerance(cell)
            ):
                errors.append(f"{cell.cell_id}: replay differs from canonical baseline")
        if errors:
            rendered = "\n".join(f"- {message}" for message in errors[:50])
            raise FinalAuditError(
                f"final report numerical replay failed ({len(errors)}):\n{rendered}"
            )
        if len(replayed) != len(pyamplicol_cells):
            raise FinalAuditError(
                "final report replay did not cover every pyAmpliCol measurement: "
                f"{len(replayed)}/{len(pyamplicol_cells)}"
            )

    pdf_result: Mapping[str, object]
    if verify_pdf:
        pdf_result = pdf_auditor(report)
        if pdf_result.get("status") != ResultStatus.OK.value:
            raise FinalAuditError("final PDF auditor did not return status='ok'")
    else:
        pdf_result = {"status": "incomplete", "skipped": True}

    publication_postflight = _validated_publication_lineage(
        source_auditor(root, expected_source_revision),
        measurement_source_revision=expected_source_revision,
        expected_publication_revision=expected_publication_revision,
    )
    if publication_postflight != publication_lineage:
        raise FinalAuditError(
            "report publication lineage changed during final report audit"
        )
    if pyamplicol_cells:
        runtime_postflight = runtime_auditor(expected_source_revision, root)
        _validate_loaded_origin_policy(
            runtime_postflight.get("loaded_module_origin_policy"),
            context="postflight_runtime.loaded_module_origin_policy",
        )
        stable_postflight = dict(runtime_postflight)
        stable_initial = dict(active_runtime or {})
        stable_postflight.pop("loaded_module_origin_policy", None)
        stable_initial.pop("loaded_module_origin_policy", None)
        if stable_postflight != stable_initial:
            raise FinalAuditError(
                "active candidate runtime identity changed during final report audit"
            )
    else:
        runtime_postflight = None

    canonical_publication_scope = (
        catalog is REPORT_CATALOG
        and max_n_final == 4
        and expected_cell_count == _EXPECTED_N4_CELL_COUNT
        and len(cells) == _EXPECTED_N4_CELL_COUNT
    )
    final_gate_complete = (
        replay and verify_render and verify_pdf and canonical_publication_scope
    )
    modes: dict[str, int] = defaultdict(int)
    for cell in cells:
        modes[cell.measurement.execution_mode.value] += 1
    return {
        "kind": "pyamplicol-final-report-audit",
        "schema_version": 2,
        "status": (ResultStatus.OK.value if final_gate_complete else "incomplete"),
        "expected_source_revision": expected_source_revision,
        "measurement_source_revision": expected_source_revision,
        "publication_revision": publication_lineage["publication_revision"],
        "publication_lineage": publication_lineage,
        "maximum_n_final": max_n_final,
        "selected_cell_count": len(cells),
        "mode_counts": dict(sorted(modes.items())),
        "authenticated_current_count": len(cells),
        "portable_publication_projection_count": len(published_measurements),
        "publication_cache_role": "portable-projection-of-current-result",
        "cryptographic_audit_source": "immutable-current-result",
        "numerically_evidenced_cell_count": len(cells),
        "pyamplicol_measurement_count": len(pyamplicol_cells),
        "legacy_fresh_oracle_count": len(legacy_cells),
        "legacy_oracles_with_inbound_agreement": len(covered_legacy_ids),
        "legacy_pointwise_agreement_edge_count": len(legacy_agreement_edges),
        "unique_artifact_count": len(grouped),
        "arena_record_count": sum(
            evidence.arena_record_count for evidence in evidence_by_key.values()
        ),
        "replay_enabled": replay,
        "canonical_publication_scope": canonical_publication_scope,
        "final_gate_complete": final_gate_complete,
        "audit_scope": (
            "final-numerical-pdf-gate"
            if final_gate_complete
            else "diagnostic-incomplete"
        ),
        "pyamplicol_replay_count": len(replayed),
        "replayed_measurement_count": len(replayed),
        "render_audit": render_result,
        "pdf_audit": dict(pdf_result),
        "report_snapshot_lock": "report-writer",
        "source_identity_postflight_match": True,
        "runtime_identity_postflight_match": (
            True if runtime_postflight is not None else None
        ),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit every n<=4 report record at one exact measured source/runtime "
            "SHA and an optional report-only publication descendant. Numerical "
            "replay is mandatory by default."
        )
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--report-profile", type=validate_profile_name)
    parser.add_argument("--expected-source-revision", required=True)
    parser.add_argument(
        "--publication-revision",
        help="require the clean publication checkout to equal this full Git SHA",
    )
    parser.add_argument("--max-n-final", type=int, default=4)
    parser.add_argument(
        "--expected-cell-count",
        type=int,
        default=_EXPECTED_N4_CELL_COUNT,
    )
    replay = parser.add_mutually_exclusive_group()
    replay.add_argument(
        "--replay",
        dest="replay",
        action="store_true",
        default=True,
        help=(
            "load every unique artifact and recompute matrix elements, resolved "
            "totals, selectors, and scalar precision-32 comparisons (default)"
        ),
    )
    replay.add_argument(
        "--structural-only",
        dest="replay",
        action="store_false",
        help="authenticate records and artifacts without numerical replay",
    )
    return parser


def _ensure_exact_cli_python(
    repo_root: Path,
    argv: Sequence[str],
) -> None:
    """Fail closed if this already-imported module was not isolated."""

    del repo_root, argv
    try:
        source_only_bytecode_policy()
    except RuntimeEvidenceError as error:
        raise FinalAuditError(
            "direct tools.performance_report.final_audit was imported before "
            "source-only Python isolation; invoke "
            "'python docs/result_tables.py final-audit ...' or start the module "
            "under an already isolated -I -S -B bootstrap"
        ) from error


def main(argv: Sequence[str] | None = None) -> int:
    raw_arguments = tuple(sys.argv[1:] if argv is None else argv)
    arguments = _parser().parse_args(raw_arguments)
    try:
        _ensure_exact_cli_python(arguments.repo_root, raw_arguments)
        service = ReportService(
            ReportPaths.from_repo(
                arguments.repo_root,
                profile=arguments.report_profile,
            )
        )
        result = audit_final_report(
            arguments.repo_root,
            expected_source_revision=arguments.expected_source_revision,
            expected_publication_revision=arguments.publication_revision,
            max_n_final=arguments.max_n_final,
            expected_cell_count=arguments.expected_cell_count,
            replay=arguments.replay,
            service=service,
        )
    except Exception as error:
        print(f"final report audit failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, allow_nan=False, sort_keys=True))
    return 0 if result["final_gate_complete"] is True else 2


if __name__ == "__main__":
    raise SystemExit(main())
