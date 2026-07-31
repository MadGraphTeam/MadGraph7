#!/usr/bin/env python3
# SPDX-License-Identifier: 0BSD
"""Validate recurrence numerical parity across a frozen baseline and candidate.

The acceptance matrix is deliberately fixed:

* ``d d~ > Z g g g g``;
* ``d d~ > t t~ g g g``;
* ``g g > g g g g``;
* LC ``topology-replay`` and ``all-flow-union`` for every process.

Each baseline/candidate capture runs in its own process tree under the audited
30 GiB watchdog.  The worker reuses the recurrence benchmark's source/runtime
provenance, generation configuration, artifact semantic identity, deterministic
validation fixture, and loaded-artifact verification.  It does not run timing
loops: it loads the generated artifact once and records selected totals plus
every selected resolved component through the public runtime API.

This is developer-only validation tooling.  It does not alter a public API,
CLI, artifact schema, or recurrence ABI.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import signal
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DRIVER_PATH = Path(__file__).resolve()
DRIVER_ROOT = DRIVER_PATH.parents[2]
if str(DRIVER_ROOT) not in sys.path:
    sys.path.insert(0, str(DRIVER_ROOT))

from tools.developer import recurrence_generation_ab_ladder as ladder  # noqa: E402
from tools.developer import recurrence_z6g_benchmark as harness  # noqa: E402

RESULT_KIND = "pyamplicol-recurrence-numerical-ab-parity"
RESULT_SCHEMA = 1
CAPTURE_KIND = "pyamplicol-recurrence-numerical-parity-capture"
CAPTURE_SCHEMA = 1
COMPARISON_KIND = "pyamplicol-recurrence-numerical-parity-comparison"
COMPARISON_SCHEMA = 1
ALLOWED_OUTPUT_PARENT = (
    DRIVER_ROOT / ".artifacts" / "recurrence-generation-opt"
).resolve()
WATCHDOG_PATH = DRIVER_ROOT / "tools" / "ci" / "memory_watchdog.py"
SOURCE_CHECKOUT_ENV = "PYAMPLICOL_RECURRENCE_Z6G_SOURCE_CHECKOUT"
WATCHDOG_LIMIT_GIB = 30.0
ABSOLUTE_TOLERANCE = 1.0e-15
RELATIVE_TOLERANCE = 1.0e-12
DEFAULT_VALIDATION_SAMPLES = 10
DEFAULT_POINT_TILE_SIZE = 1024
DEFAULT_JIT_OPTIMIZATION_LEVEL = 2
DEFAULT_WORKER_TIMEOUT_SECONDS = 2.0 * 60.0 * 60.0
LAYOUTS = ("topology-replay", "all-flow-union")
PROCESS_CASES = (
    ("dd_z_4g", "d d~ > Z g g g g"),
    ("dd_tt_3g", "d d~ > t t~ g g g"),
    ("gg_4g", "g g > g g g g"),
)
EXPECTED_SOURCE_REVISIONS = {
    "baseline": "172e58fd33a3c65563866c50cfbb5e1ddcd7b302",
    "candidate": "4e2b1e02dddde2d55b7250cbd52a93001f09b2c2",
}
_SHA256_LENGTH = 64
_SHA256_DIGITS = frozenset("0123456789abcdef")


class ParityError(RuntimeError):
    """Raised when numerical parity evidence is incomplete or invalid."""


@dataclass(frozen=True, slots=True)
class Variant:
    """One source-bound installed pyAmpliCol runtime."""

    name: str
    python: Path
    checkout: Path
    pythonpath: Path
    prepared_model: Path


@dataclass(frozen=True, slots=True)
class ProcessOutcome:
    """One watchdog-guarded capture process outcome."""

    exit_code: int | None
    timed_out: bool
    wall_seconds: float
    error: str | None


def _utc_now() -> str:
    return dt.datetime.now(dt.UTC).isoformat().replace("+00:00", "Z")


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == _SHA256_LENGTH
        and all(character in _SHA256_DIGITS for character in value)
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _path_identity(path: Path) -> dict[str, object]:
    resolved = path.resolve(strict=True)
    return {
        "path": str(resolved),
        "resolved_path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": _sha256_file(resolved),
    }


def _content_address(value: Mapping[str, object]) -> dict[str, object]:
    result = dict(value)
    if "content_sha256" in result:
        raise ParityError("content-addressed record already contains a digest")
    result["content_sha256"] = _canonical_sha256(result)
    return result


def _validate_content_address(
    value: object,
    *,
    kind: str,
    schema_version: int,
    label: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ParityError(f"{label} is not an object")
    result = dict(value)
    digest = result.pop("content_sha256", None)
    if (
        value.get("kind") != kind
        or isinstance(value.get("schema_version"), bool)
        or not isinstance(value.get("schema_version"), int)
        or value.get("schema_version") != schema_version
        or not _is_sha256(digest)
        or digest != _canonical_sha256(result)
    ):
        raise ParityError(f"{label} failed its content-address contract")
    return dict(value)


def _atomic_write_json(path: Path, value: object) -> None:
    if path.exists():
        raise ParityError(f"refusing to replace existing evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    except (OSError, TypeError, ValueError) as error:
        temporary.unlink(missing_ok=True)
        raise ParityError(f"cannot write evidence JSON: {path}") from error


def _json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ParityError(f"cannot read {label}: {path}") from error
    if not isinstance(value, dict):
        raise ParityError(f"{label} must contain one object")
    return value


def _normalized_process(value: str) -> str:
    return " ".join(value.split()).casefold()


def _validate_process_case(process_key: str, process_expression: str) -> None:
    if not any(
        process_key == expected_key
        and _normalized_process(process_expression)
        == _normalized_process(expected_expression)
        for expected_key, expected_expression in PROCESS_CASES
    ):
        raise ParityError("process is outside the fixed numerical-parity matrix")


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be a positive finite number")
    return parsed


def _resolve_inside_workspace(
    path: Path,
    *,
    label: str,
    regular_file: bool | None = None,
) -> Path:
    try:
        resolved = path.expanduser().resolve(strict=True)
        resolved.relative_to(DRIVER_ROOT.resolve())
    except (OSError, ValueError) as error:
        raise ParityError(f"{label} must exist inside the workspace: {path}") from error
    if regular_file is True and not resolved.is_file():
        raise ParityError(f"{label} is not a regular file: {resolved}")
    if regular_file is False and not resolved.is_dir():
        raise ParityError(f"{label} is not a directory: {resolved}")
    return resolved


def _resolve_python_inside_workspace(path: Path, *, label: str) -> Path:
    """Retain a workspace interpreter symlink whose target may be external."""

    absolute = Path(os.path.abspath(path.expanduser()))
    try:
        absolute.relative_to(DRIVER_ROOT.resolve())
        absolute.parent.resolve(strict=True).relative_to(DRIVER_ROOT.resolve())
    except (OSError, ValueError) as error:
        raise ParityError(f"{label} must exist inside the workspace: {path}") from error
    if not absolute.is_file():
        raise ParityError(f"{label} is not a regular file: {absolute}")
    return absolute


def _resolve_output_root(path: Path) -> Path:
    requested = path.expanduser().resolve()
    try:
        requested.relative_to(ALLOWED_OUTPUT_PARENT)
    except ValueError as error:
        raise ParityError(
            f"output root must remain below {ALLOWED_OUTPUT_PARENT}"
        ) from error
    if requested == ALLOWED_OUTPUT_PARENT:
        raise ParityError("output root must be a child of the artifact parent")
    if requested.exists():
        raise ParityError(f"output root already exists: {requested}")
    return requested


def _variant(
    name: str,
    *,
    python: Path,
    checkout: Path,
    pythonpath: Path,
    prepared_model: Path,
) -> Variant:
    return Variant(
        name=name,
        python=_resolve_python_inside_workspace(python, label=f"{name} Python"),
        checkout=_resolve_inside_workspace(
            checkout,
            label=f"{name} checkout",
            regular_file=False,
        ),
        pythonpath=_resolve_inside_workspace(
            pythonpath,
            label=f"{name} Python path",
            regular_file=False,
        ),
        prepared_model=_resolve_inside_workspace(
            prepared_model,
            label=f"{name} prepared model",
            regular_file=True,
        ),
    )


def _git_stdout(checkout: Path, *arguments: str, label: str) -> str:
    environment = os.environ.copy()
    environment["GIT_OPTIONAL_LOCKS"] = "0"
    completed = subprocess.run(
        ("git", *arguments),
        cwd=checkout,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip()
        raise ParityError(
            f"cannot inspect {label} checkout" + ("" if not detail else f": {detail}")
        )
    return completed.stdout.strip()


def _preflight_variant_source(variant: Variant) -> dict[str, object]:
    expected_revision = EXPECTED_SOURCE_REVISIONS.get(variant.name)
    if expected_revision is None:
        raise ParityError(f"no expected revision is pinned for {variant.name}")
    checkout = variant.checkout.resolve()
    top_level = _git_stdout(
        checkout,
        "rev-parse",
        "--show-toplevel",
        label=variant.name,
    )
    revision = _git_stdout(
        checkout,
        "rev-parse",
        "--verify",
        "HEAD",
        label=variant.name,
    )
    status = _git_stdout(
        checkout,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        label=variant.name,
    )
    if Path(top_level).resolve() != checkout:
        raise ParityError(f"{variant.name} checkout is not its Git worktree root")
    if revision != expected_revision:
        raise ParityError(
            f"{variant.name} revision is {revision}, expected {expected_revision}"
        )
    if status:
        raise ParityError(f"{variant.name} checkout is dirty before the campaign")
    return {
        "checkout": str(checkout),
        "revision": revision,
        "dirty": False,
        "untracked_files_checked": True,
    }


def _complex_payload(value: complex) -> list[float]:
    if not math.isfinite(value.real) or not math.isfinite(value.imag):
        raise ParityError("runtime returned a non-finite complex value")
    return [float(value.real), float(value.imag)]


def _complex_value(value: object, *, label: str) -> complex:
    if (
        not isinstance(value, list)
        or len(value) != 2
        or any(
            isinstance(component, bool)
            or not isinstance(component, (float, int))
            or not math.isfinite(float(component))
            for component in value
        )
    ):
        raise ParityError(f"{label} is not a finite complex pair")
    return complex(float(value[0]), float(value[1]))


def _value_comparison(
    baseline: object,
    candidate: object,
    *,
    absolute_tolerance: float = ABSOLUTE_TOLERANCE,
    relative_tolerance: float = RELATIVE_TOLERANCE,
) -> dict[str, object]:
    baseline_value = _complex_value(baseline, label="baseline value")
    candidate_value = _complex_value(candidate, label="candidate value")
    absolute = abs(candidate_value - baseline_value)
    relative = absolute / max(
        abs(baseline_value),
        abs(candidate_value),
        1.0e-300,
    )
    return {
        "baseline": _complex_payload(baseline_value),
        "candidate": _complex_payload(candidate_value),
        "absolute_difference": absolute,
        "relative_difference": relative,
        "absolute_tolerance": absolute_tolerance,
        "relative_tolerance": relative_tolerance,
        "passes": (absolute <= absolute_tolerance or relative <= relative_tolerance),
    }


def _axis_ids(values: Sequence[object], *, label: str) -> list[str]:
    result = [getattr(value, "id", None) for value in values]
    if (
        not result
        or any(not isinstance(value, str) or not value for value in result)
        or len(set(result)) != len(result)
    ):
        raise ParityError(f"runtime exposes an invalid {label} axis")
    return [str(value) for value in result]


def _select_workload(
    physics: object,
    *,
    layout: str,
) -> tuple[dict[str, tuple[str, ...]], dict[str, object]]:
    color_flows = tuple(getattr(physics, "color_flows", ()))
    helicities = tuple(getattr(physics, "helicities", ()))
    color_ids = _axis_ids(color_flows, label="color-flow")
    helicity_ids = _axis_ids(helicities, label="helicity")
    structural_zero_ids = [
        str(helicity.id)
        for helicity in helicities
        if getattr(helicity, "structural_zero", False) is True
    ]
    if layout == "topology-replay":
        selected_color_id = color_ids[0]
        selectors = {"color_flows": (selected_color_id,)}
        contract = {
            "workload": "single-runtime-selected-flow/helicity-sum",
            "physical_color_ids": color_ids,
            "physical_helicity_ids": helicity_ids,
            "structural_zero_helicity_ids": structural_zero_ids,
            "selected_color_flow_ids": [selected_color_id],
            "selected_helicity_ids": [],
        }
    elif layout == "all-flow-union":
        selected_helicity_id = next(
            (
                helicity_id
                for helicity_id in helicity_ids
                if helicity_id not in set(structural_zero_ids)
            ),
            None,
        )
        if selected_helicity_id is None:
            raise ParityError("runtime exposes no non-structural-zero helicity")
        selectors = {"helicities": (selected_helicity_id,)}
        contract = {
            "workload": "all-flows/runtime-selected-single-helicity",
            "physical_color_ids": color_ids,
            "physical_helicity_ids": helicity_ids,
            "structural_zero_helicity_ids": structural_zero_ids,
            "selected_color_flow_ids": [],
            "selected_helicity_ids": [selected_helicity_id],
        }
    else:
        raise ParityError(f"unsupported LC flow layout: {layout}")
    contract["contract_sha256"] = _canonical_sha256(contract)
    return selectors, contract


def _parameters_identity(artifact: Path) -> dict[str, object]:
    path = artifact / "model" / "parameters.json"
    payload = _json_object(path, label="artifact parameter payload")
    return {
        "file": _path_identity(path),
        "payload": payload,
        "payload_sha256": _canonical_sha256(payload),
    }


def _generation_arguments(arguments: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        mode="recurrence",
        artifact=arguments.artifact,
        prepared_model=arguments.prepared_model,
        process_expression=arguments.process_expression,
        gluon_count=1,
        validation_samples=arguments.validation_samples,
        point_tile_size=arguments.point_tile_size,
        jit_optimization_level=arguments.jit_optimization_level,
        lc_flow_layout=arguments.layout,
        specialize_flow_at_generation=False,
        color_flow="1",
        helicity="1",
        write_mode="error",
    )


def _generation_request(
    *,
    validation_samples: int,
    point_tile_size: int,
    jit_optimization_level: int,
) -> dict[str, object]:
    return {
        "validation_samples": validation_samples,
        "validation_seed": harness.VALIDATION_SEED,
        "point_tile_size": point_tile_size,
        "jit_optimization_level": jit_optimization_level,
        "specialize_flow_at_generation": False,
    }


def _capture_validation(
    runtime: object,
    *,
    artifact: Path,
    process_id: str,
    layout: str,
) -> dict[str, object]:
    points, fixture = harness._validation_fixture(artifact, process_id)
    if not points:
        raise ParityError("generated artifact contains no validation points")
    selectors, selector_contract = _select_workload(
        runtime.physics,
        layout=layout,
    )
    selected = tuple(complex(value) for value in runtime.evaluate(points, **selectors))
    resolved = runtime.evaluate_resolved(points, **selectors)
    resolved_sums = tuple(complex(value) for value in resolved.total())
    if len(selected) != len(points) or len(resolved_sums) != len(points):
        raise ParityError("runtime result count disagrees with validation fixture")
    point_comparisons: list[dict[str, object]] = []
    for point_index, (selected_value, resolved_value) in enumerate(
        zip(selected, resolved_sums, strict=True)
    ):
        comparison = _value_comparison(
            _complex_payload(selected_value),
            _complex_payload(resolved_value),
        )
        point_comparisons.append(
            {
                "point_index": point_index,
                "selected_total": comparison["baseline"],
                "resolved_sum": comparison["candidate"],
                "absolute_difference": comparison["absolute_difference"],
                "relative_difference": comparison["relative_difference"],
                "passes": comparison["passes"],
            }
        )
    resolved_components = [
        [
            _complex_payload(complex(value))
            for helicity_row in point_values
            for value in helicity_row
        ]
        for point_values in resolved.values
    ]
    validation: dict[str, object] = {
        "fixture": {
            **fixture,
            "points": [[list(momentum) for momentum in point] for point in points],
        },
        "selector_contract": selector_contract,
        "selected_totals": [_complex_payload(value) for value in selected],
        "resolved_sums": [_complex_payload(value) for value in resolved_sums],
        "resolved_helicity_ids": list(resolved.helicity_ids),
        "resolved_color_ids": list(resolved.color_ids),
        "resolved_components": resolved_components,
        "point_comparisons": point_comparisons,
        "maximum_absolute_difference": max(
            float(item["absolute_difference"]) for item in point_comparisons
        ),
        "maximum_relative_difference": max(
            float(item["relative_difference"]) for item in point_comparisons
        ),
        "passes": all(bool(item["passes"]) for item in point_comparisons),
    }
    harness._validated_lane_validation_values(
        validation,
        fixture,
        mode="recurrence",
    )
    return validation


def _run_capture_worker(arguments: argparse.Namespace) -> dict[str, object]:
    from pyamplicol import Runtime

    artifact = arguments.artifact.resolve()
    capture_path = arguments.capture_json.resolve()
    prepared_model_path = arguments.prepared_model.resolve(strict=True)
    try:
        artifact.relative_to(ALLOWED_OUTPUT_PARENT)
        capture_path.relative_to(ALLOWED_OUTPUT_PARENT)
        prepared_model_path.relative_to(DRIVER_ROOT.resolve())
    except ValueError as error:
        raise ParityError(
            "capture worker inputs and outputs must remain inside the workspace"
        ) from error
    if artifact.parent != capture_path.parent:
        raise ParityError("capture worker artifact and evidence must share one root")
    if artifact.exists() or capture_path.exists():
        raise ParityError("capture worker refuses to replace existing output")
    if arguments.layout not in LAYOUTS:
        raise ParityError(f"unsupported LC flow layout: {arguments.layout}")
    _validate_process_case(arguments.process_key, arguments.process_expression)

    source_before = harness._git_source_identity()
    runtime_before = harness._runtime_provenance(source_before)
    prepared_model = _path_identity(prepared_model_path)
    arguments.prepared_model = prepared_model_path
    generation_arguments = _generation_arguments(arguments)
    generation = harness._generate_worker(generation_arguments)
    artifact_identity_before = harness._artifact_identity(artifact)
    effective_contract = harness._validate_artifact_contract(
        artifact,
        artifact_identity_before,
        arguments=generation_arguments,
        mode="recurrence",
    )

    process_id = artifact_identity_before.get("process_id")
    if not isinstance(process_id, str) or not process_id:
        raise ParityError("generated artifact has no process identity")
    runtime = Runtime.load(artifact, process=arguments.process_expression)
    loaded_before = harness._loaded_runtime_artifact_verification(
        runtime,
        expected_artifact_id=artifact_identity_before.get("artifact_id"),
        phase="before-numerical-parity-evaluation",
    )
    physics = runtime.physics
    observed_process = getattr(physics, "process", None)
    if not isinstance(observed_process, str) or _normalized_process(
        observed_process
    ) != _normalized_process(arguments.process_expression):
        raise ParityError("loaded runtime process disagrees with the request")
    if getattr(physics, "color_accuracy", None) != "lc":
        raise ParityError("loaded runtime does not expose LC physics")

    validation = _capture_validation(
        runtime,
        artifact=artifact,
        process_id=process_id,
        layout=arguments.layout,
    )
    loaded_after = harness._loaded_runtime_artifact_verification(
        runtime,
        expected_artifact_id=artifact_identity_before.get("artifact_id"),
        phase="after-numerical-parity-evaluation",
    )
    artifact_identity_after = harness._artifact_identity(artifact)
    source_after = harness._git_source_identity()
    runtime_after = harness._runtime_provenance(source_after)
    if artifact_identity_after != artifact_identity_before:
        raise ParityError("artifact changed while numerical parity was evaluated")
    if source_after != source_before:
        raise ParityError("source checkout changed while capture was running")
    if runtime_after != runtime_before:
        raise ParityError("installed runtime changed while capture was running")

    capture = _content_address(
        {
            "kind": CAPTURE_KIND,
            "schema_version": CAPTURE_SCHEMA,
            "variant": arguments.variant,
            "process_key": arguments.process_key,
            "requested_process": arguments.process_expression,
            "observed_process": observed_process,
            "layout": arguments.layout,
            "generation_request": _generation_request(
                validation_samples=arguments.validation_samples,
                point_tile_size=arguments.point_tile_size,
                jit_optimization_level=arguments.jit_optimization_level,
            ),
            "captured_at_utc": _utc_now(),
            "source": source_before,
            "runtime_provenance": runtime_before,
            "prepared_model": prepared_model,
            "generation": generation,
            "artifact": artifact_identity_before,
            "effective_contract": effective_contract,
            "parameters": _parameters_identity(artifact),
            "loaded_artifact_verification": {
                "before_evaluation": loaded_before,
                "after_evaluation": loaded_after,
            },
            "validation": validation,
            "peak_rss_after_evaluation": harness._resource_peak(),
            "tolerances": {
                "absolute": ABSOLUTE_TOLERANCE,
                "relative": RELATIVE_TOLERANCE,
            },
        }
    )
    capture = _validate_capture(
        capture,
        expected_variant=arguments.variant,
        expected_process_key=arguments.process_key,
        expected_process=arguments.process_expression,
        expected_layout=arguments.layout,
        expected_checkout=Path(str(source_before["checkout"])),
        expected_artifact_root=artifact.parent,
        expected_generation_request=_generation_request(
            validation_samples=arguments.validation_samples,
            point_tile_size=arguments.point_tile_size,
            jit_optimization_level=arguments.jit_optimization_level,
        ),
    )
    _atomic_write_json(capture_path, capture)
    return capture


def _validate_file_identity(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ParityError(f"{label} file identity is missing")
    path = value.get("path")
    size_bytes = value.get("size_bytes")
    if (
        not isinstance(path, str)
        or not path
        or isinstance(size_bytes, bool)
        or not isinstance(size_bytes, int)
        or size_bytes < 0
        or not _is_sha256(value.get("sha256"))
    ):
        raise ParityError(f"{label} file identity is invalid")
    resolved_path = value.get("resolved_path")
    if resolved_path is not None and (
        not isinstance(resolved_path, str) or not Path(resolved_path).is_absolute()
    ):
        raise ParityError(f"{label} resolved file identity is invalid")
    return value


def _file_identity_paths(
    value: Mapping[str, Any],
    *,
    label: str,
) -> tuple[Path, ...]:
    raw_paths = [value.get("path")]
    if "resolved_path" in value:
        raw_paths.append(value.get("resolved_path"))
    if any(
        not isinstance(raw, str) or not Path(raw).is_absolute() for raw in raw_paths
    ):
        raise ParityError(f"{label} path is not absolute")
    return tuple(Path(str(raw)).resolve(strict=False) for raw in raw_paths)


def _validate_tree_identity(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ParityError(f"{label} tree identity is missing")
    file_count = value.get("file_count")
    size_bytes = value.get("size_bytes")
    if (
        value.get("algorithm") != "sha256-relative-path-size-content-v1"
        or isinstance(file_count, bool)
        or not isinstance(file_count, int)
        or file_count <= 0
        or isinstance(size_bytes, bool)
        or not isinstance(size_bytes, int)
        or size_bytes <= 0
        or not _is_sha256(value.get("sha256"))
    ):
        raise ParityError(f"{label} tree identity is invalid")
    return value


def _validate_runtime_provenance_record(
    value: Mapping[str, Any],
    *,
    source: Mapping[str, Any],
) -> None:
    required = {
        "interpreter",
        "installed_distribution",
        "active_build_info",
        "native_extension",
        "dependencies",
    }
    if not required.issubset(value):
        raise ParityError("runtime provenance is incomplete")
    interpreter = _validate_file_identity(
        value["interpreter"],
        label="runtime interpreter",
    )
    native = _validate_file_identity(
        value["native_extension"],
        label="native extension",
    )
    active = _validate_file_identity(
        value["active_build_info"],
        label="active build info",
    )
    if (
        not isinstance(interpreter.get("python_version"), str)
        or not isinstance(interpreter.get("implementation"), str)
        or not isinstance(native.get("package_version"), str)
        or not _is_sha256(native.get("build_inputs_sha256"))
        or not isinstance(active.get("payload"), Mapping)
    ):
        raise ParityError("runtime executable provenance is invalid")
    build_payload = active["payload"]
    if (
        build_payload.get("source_checkout") != source.get("checkout")
        or build_payload.get("source_revision") != source.get("revision")
        or build_payload.get("native_build_inputs_sha256")
        != native.get("build_inputs_sha256")
    ):
        raise ParityError("runtime build provenance is not source-bound")

    distribution = value["installed_distribution"]
    if not isinstance(distribution, Mapping):
        raise ParityError("installed distribution provenance is missing")
    _validate_tree_identity(
        distribution.get("distribution_content"),
        label="installed distribution",
    )
    native_modules = distribution.get("native_modules")
    build_info_files = distribution.get("build_info_files")
    if (
        not isinstance(distribution.get("package_version"), str)
        or not isinstance(native_modules, list)
        or not native_modules
        or not isinstance(build_info_files, list)
        or not build_info_files
    ):
        raise ParityError("installed distribution provenance is incomplete")
    validated_native_modules = [
        _validate_file_identity(entry, label="installed native module")
        for entry in native_modules
    ]
    validated_build_info = [
        _validate_file_identity(entry, label="installed build-info")
        for entry in build_info_files
    ]
    if native.get("sha256") not in {
        entry.get("sha256") for entry in validated_native_modules
    } or active.get("sha256") not in {
        entry.get("sha256") for entry in validated_build_info
    }:
        raise ParityError("active runtime files are absent from its distribution")

    dependencies = value["dependencies"]
    if not isinstance(dependencies, Mapping) or not dependencies:
        raise ParityError("runtime dependency provenance is missing")
    for name, dependency in dependencies.items():
        if (
            not isinstance(name, str)
            or not name
            or not isinstance(dependency, Mapping)
            or not isinstance(dependency.get("present"), bool)
            or not isinstance(dependency.get("path"), str)
            or not isinstance(dependency.get("resolved_path"), str)
        ):
            raise ParityError("runtime dependency provenance is invalid")
        if dependency["present"] is True:
            _validate_file_identity(
                dependency,
                label=f"runtime dependency {name}",
            )


def _validate_artifact_identity_record(
    value: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    if (
        not isinstance(value.get("path"), str)
        or not Path(str(value["path"])).is_absolute()
        or not isinstance(value.get("process_id"), str)
        or not value["process_id"]
        or value["process_id"] in {".", ".."}
        or Path(str(value["process_id"])).name != value["process_id"]
        or not isinstance(value.get("producer"), Mapping)
    ):
        raise ParityError("artifact provenance is incomplete")
    artifact_root = Path(str(value["path"])).resolve(strict=False)
    manifest = _validate_file_identity(
        value.get("manifest"),
        label="artifact manifest",
    )
    expected_manifest_path = artifact_root / "artifact.json"
    if any(
        path != expected_manifest_path
        for path in _file_identity_paths(manifest, label="artifact manifest")
    ):
        raise ParityError("artifact manifest path is not bound to the artifact")
    tree = _validate_tree_identity(value.get("tree"), label="artifact")
    payloads = value.get("payloads")
    if not isinstance(payloads, list) or not payloads:
        raise ParityError("artifact payload provenance is missing")
    payload_inventory: dict[str, Mapping[str, Any]] = {}
    for payload in payloads:
        if not isinstance(payload, Mapping):
            raise ParityError("artifact payload provenance is invalid")
        path = payload.get("path")
        size_bytes = payload.get("size_bytes")
        if (
            not isinstance(path, str)
            or not path
            or Path(path).is_absolute()
            or ".." in Path(path).parts
            or not isinstance(payload.get("role"), str)
            or isinstance(size_bytes, bool)
            or not isinstance(size_bytes, int)
            or size_bytes < 0
            or not _is_sha256(payload.get("sha256"))
        ):
            raise ParityError("artifact payload provenance is invalid")
        if path in payload_inventory:
            raise ParityError("artifact payload provenance contains duplicate paths")
        payload_inventory[path] = payload
    if tree.get("file_count") != len(payload_inventory) + 1 or tree.get(
        "size_bytes"
    ) != int(manifest["size_bytes"]) + sum(
        int(payload["size_bytes"]) for payload in payload_inventory.values()
    ):
        raise ParityError(
            "artifact tree is not bound to manifest and payload inventory"
        )

    model_identity = value.get("model_identity")
    if not isinstance(model_identity, Mapping):
        raise ParityError("artifact model identity is missing")
    for field, digest_field in (
        ("manifest", "manifest_sha256"),
        ("common_physics_identity", "common_physics_identity_sha256"),
    ):
        body = model_identity.get(field)
        if not isinstance(body, Mapping) or model_identity.get(
            digest_field
        ) != _canonical_sha256(body):
            raise ParityError("artifact model identity is invalid")
    return payload_inventory


def _validate_artifact_payload_link(
    artifact: Mapping[str, Any],
    payload_inventory: Mapping[str, Mapping[str, Any]],
    file_identity: object,
    *,
    relative_path: str,
    role: str,
    process_id: str | None,
    label: str,
) -> None:
    identity = _validate_file_identity(file_identity, label=label)
    artifact_root = Path(str(artifact["path"])).resolve(strict=False)
    expected_path = (artifact_root / relative_path).resolve(strict=False)
    payload = payload_inventory.get(relative_path)
    if (
        any(
            path != expected_path
            for path in _file_identity_paths(identity, label=label)
        )
        or not isinstance(payload, Mapping)
        or payload.get("role") != role
        or payload.get("process_id") != process_id
        or payload.get("sha256") != identity.get("sha256")
        or payload.get("size_bytes") != identity.get("size_bytes")
    ):
        raise ParityError(f"{label} is not bound to the artifact payload inventory")


def _validate_capture(
    value: object,
    *,
    expected_variant: str,
    expected_process_key: str,
    expected_process: str,
    expected_layout: str,
    expected_checkout: Path | None = None,
    expected_artifact_root: Path | None = None,
    expected_generation_request: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    capture = _validate_content_address(
        value,
        kind=CAPTURE_KIND,
        schema_version=CAPTURE_SCHEMA,
        label=f"{expected_variant} numerical capture",
    )
    source = capture.get("source")
    artifact = capture.get("artifact")
    effective = capture.get("effective_contract")
    validation = capture.get("validation")
    parameters = capture.get("parameters")
    prepared = capture.get("prepared_model")
    generation_request = capture.get("generation_request")
    generation = capture.get("generation")
    runtime_provenance = capture.get("runtime_provenance")
    loaded = capture.get("loaded_artifact_verification")
    semantic = (
        artifact.get("semantic_identity") if isinstance(artifact, Mapping) else None
    )
    coverage = semantic.get("coverage") if isinstance(semantic, Mapping) else None
    reduction_coverage = (
        semantic.get("execution_reduction_coverage")
        if isinstance(semantic, Mapping)
        else None
    )
    model_source = (
        generation.get("model_source") if isinstance(generation, Mapping) else None
    )
    if (
        capture.get("variant") != expected_variant
        or capture.get("process_key") != expected_process_key
        or not isinstance(capture.get("requested_process"), str)
        or _normalized_process(str(capture["requested_process"]))
        != _normalized_process(expected_process)
        or not isinstance(capture.get("observed_process"), str)
        or _normalized_process(str(capture["observed_process"]))
        != _normalized_process(expected_process)
        or capture.get("layout") != expected_layout
        or not isinstance(generation_request, Mapping)
        or set(generation_request)
        != {
            "validation_samples",
            "validation_seed",
            "point_tile_size",
            "jit_optimization_level",
            "specialize_flow_at_generation",
        }
        or isinstance(generation_request.get("validation_samples"), bool)
        or not isinstance(generation_request.get("validation_samples"), int)
        or int(generation_request["validation_samples"]) <= 0
        or isinstance(generation_request.get("validation_seed"), bool)
        or not isinstance(generation_request.get("validation_seed"), int)
        or generation_request.get("validation_seed") != harness.VALIDATION_SEED
        or isinstance(generation_request.get("point_tile_size"), bool)
        or not isinstance(generation_request.get("point_tile_size"), int)
        or int(generation_request["point_tile_size"]) <= 0
        or isinstance(generation_request.get("jit_optimization_level"), bool)
        or not isinstance(generation_request.get("jit_optimization_level"), int)
        or generation_request.get("jit_optimization_level") not in {0, 1, 2, 3}
        or generation_request.get("specialize_flow_at_generation") is not False
        or capture.get("tolerances")
        != {
            "absolute": ABSOLUTE_TOLERANCE,
            "relative": RELATIVE_TOLERANCE,
        }
        or not isinstance(source, Mapping)
        or source.get("dirty") is not False
        or source.get("untracked_files_checked") is not True
        or not isinstance(source.get("checkout"), str)
        or not Path(str(source["checkout"])).is_absolute()
        or not isinstance(source.get("revision"), str)
        or len(str(source["revision"])) != 40
        or any(character not in _SHA256_DIGITS for character in str(source["revision"]))
        or source.get("revision") != EXPECTED_SOURCE_REVISIONS.get(expected_variant)
        or not isinstance(runtime_provenance, Mapping)
        or not isinstance(artifact, Mapping)
        or not _is_sha256(artifact.get("artifact_id"))
        or artifact.get("color_accuracy") != "lc"
        or not isinstance(artifact.get("process_expression"), str)
        or _normalized_process(str(artifact["process_expression"]))
        != _normalized_process(expected_process)
        or not isinstance(semantic, Mapping)
        or artifact.get("semantic_identity_sha256") != _canonical_sha256(semantic)
        or not isinstance(coverage, Mapping)
        or coverage.get("complete_physical_axes") is not True
        or coverage.get("color") != "complete"
        or coverage.get("helicities") != "complete"
        or not isinstance(reduction_coverage, Mapping)
        or reduction_coverage.get("complete") is not True
        or reduction_coverage.get("errors") != []
        or semantic.get("generation_specialized_axes") != []
        or not isinstance(semantic.get("runtime_selector_semantics"), Mapping)
        or semantic["runtime_selector_semantics"].get("generation_specialized_axes")
        != []
        or semantic["runtime_selector_semantics"]["generation_specialized_axes"]
        != semantic["generation_specialized_axes"]
        or not isinstance(effective, Mapping)
        or effective.get("execution_mode") != "recurrence"
        or effective.get("backend") != "jit"
        or isinstance(effective.get("jit_optimization_level"), bool)
        or effective.get("jit_optimization_level")
        != harness.PREPARED_JIT_PORTABLE_OPTIMIZATION_LEVEL
        or effective.get("color_accuracy") != "lc"
        or effective.get("lc_flow_layout") != expected_layout
        or not isinstance(validation, Mapping)
        or validation.get("passes") is not True
        or not isinstance(parameters, Mapping)
        or not isinstance(parameters.get("payload"), Mapping)
        or parameters.get("payload_sha256") != _canonical_sha256(parameters["payload"])
        or not isinstance(prepared, Mapping)
        or not _is_sha256(prepared.get("sha256"))
        or not isinstance(generation, Mapping)
        or generation.get("mode") != "recurrence"
        or generation.get("generation_reused") is not False
        or generation.get("specialized_flow_word") is not None
        or not isinstance(model_source, Mapping)
        or model_source.get("kind") != "explicit-prepared-model"
        or model_source.get("file") != prepared
        or model_source.get("compile_excluded_from_generation") is not True
        or not isinstance(loaded, Mapping)
    ):
        raise ParityError(f"{expected_variant} numerical capture is inconsistent")
    _validate_runtime_provenance_record(runtime_provenance, source=source)
    payload_inventory = _validate_artifact_identity_record(artifact)
    _validate_file_identity(prepared, label="prepared model")
    _validate_artifact_payload_link(
        artifact,
        payload_inventory,
        parameters.get("file"),
        relative_path="model/parameters.json",
        role="model-parameters",
        process_id=None,
        label="artifact parameters",
    )
    if expected_checkout is not None and source.get("checkout") != str(
        expected_checkout.resolve()
    ):
        raise ParityError(f"{expected_variant} capture used the wrong checkout")
    if expected_generation_request is not None and _canonical_sha256(
        generation_request
    ) != _canonical_sha256(expected_generation_request):
        raise ParityError(
            f"{expected_variant} capture used the wrong generation request"
        )
    if expected_artifact_root is not None:
        raw_artifact_path = artifact.get("path")
        if not isinstance(raw_artifact_path, str):
            raise ParityError(f"{expected_variant} capture has no artifact path")
        if Path(raw_artifact_path).resolve() != (
            expected_artifact_root.resolve() / "artifact"
        ):
            raise ParityError(
                f"{expected_variant} artifact is outside its assigned location"
            )
    fixture = validation.get("fixture")
    selector = validation.get("selector_contract")
    try:
        normalized_points = (
            tuple(
                tuple(
                    tuple(float(component) for component in momentum)
                    for momentum in point
                )
                for point in fixture.get("points", ())
            )
            if isinstance(fixture, Mapping)
            else ()
        )
    except (TypeError, ValueError) as error:
        raise ParityError(
            f"{expected_variant} validation fixture is not numerical"
        ) from error
    if (
        not isinstance(fixture, Mapping)
        or not isinstance(fixture.get("points"), list)
        or isinstance(fixture.get("point_count"), bool)
        or not isinstance(fixture.get("point_count"), int)
        or fixture.get("point_count") != len(fixture["points"])
        or fixture.get("point_count") != generation_request["validation_samples"]
        or not fixture["points"]
        or fixture.get("points_sha256") != _canonical_sha256(normalized_points)
        or fixture.get("points_sha256") != _canonical_sha256(fixture.get("points"))
        or not isinstance(selector, Mapping)
        or selector.get("contract_sha256")
        != _canonical_sha256(
            {key: entry for key, entry in selector.items() if key != "contract_sha256"}
        )
    ):
        raise ParityError(f"{expected_variant} fixture or selector is invalid")
    _validate_artifact_payload_link(
        artifact,
        payload_inventory,
        fixture.get("file"),
        relative_path=(f"processes/{artifact['process_id']}/validation-momenta.json"),
        role="validation-momenta",
        process_id=str(artifact["process_id"]),
        label="validation fixture",
    )
    harness._validated_lane_validation_values(
        validation,
        {key: entry for key, entry in fixture.items() if key != "points"},
        mode=expected_variant,
    )
    harness._validate_loaded_runtime_artifact_verification(
        loaded.get("before_evaluation"),
        expected_artifact_id=artifact["artifact_id"],
        phase="before-numerical-parity-evaluation",
    )
    harness._validate_loaded_runtime_artifact_verification(
        loaded.get("after_evaluation"),
        expected_artifact_id=artifact["artifact_id"],
        phase="after-numerical-parity-evaluation",
    )
    _semantic_selector_body(capture)
    _validate_selector_semantics(capture, layout=expected_layout)
    return capture


def _semantic_selector_body(capture: Mapping[str, Any]) -> dict[str, object]:
    try:
        artifact = capture["artifact"]
        validation = capture["validation"]
        semantic = artifact["semantic_identity"]
        body = {
            "artifact_physical_color_flows": semantic["physical_color_flows"],
            "artifact_physical_helicities": semantic["physical_helicities"],
            "artifact_normalization": semantic["normalization"],
            "artifact_manifest_model_identity": semantic["manifest_model_identity"],
            "artifact_runtime_selector_semantics": semantic[
                "runtime_selector_semantics"
            ],
            "artifact_reduction_ordering": semantic["reduction_ordering"],
            "artifact_execution_reduction_identity": semantic[
                "execution_reduction_identity"
            ],
            "artifact_generation_specialized_axes": semantic[
                "generation_specialized_axes"
            ],
            "runtime_selector_contract": validation["selector_contract"],
            "resolved_helicity_ids": validation["resolved_helicity_ids"],
            "resolved_color_ids": validation["resolved_color_ids"],
        }
    except (KeyError, TypeError) as error:
        raise ParityError("capture has incomplete selector semantics") from error
    if any(value is None for value in body.values()):
        raise ParityError("capture has incomplete selector semantics")
    return body


def _ordered_semantic_axis(
    value: object,
    *,
    label: str,
) -> tuple[list[str], list[Mapping[str, Any]]]:
    if not isinstance(value, Mapping):
        raise ParityError(f"capture has no {label} semantic axis")
    count = value.get("count")
    ordered_ids = value.get("ordered_ids")
    ordered_entries = value.get("ordered_entries")
    if (
        isinstance(count, bool)
        or not isinstance(count, int)
        or count <= 0
        or not isinstance(ordered_ids, list)
        or len(ordered_ids) != count
        or any(not isinstance(entry, str) or not entry for entry in ordered_ids)
        or len(set(ordered_ids)) != count
        or not isinstance(ordered_entries, list)
        or len(ordered_entries) != count
        or any(not isinstance(entry, Mapping) for entry in ordered_entries)
        or [entry.get("id") for entry in ordered_entries if isinstance(entry, Mapping)]
        != ordered_ids
    ):
        raise ParityError(f"capture has an invalid {label} semantic axis")
    return list(ordered_ids), list(ordered_entries)


def _validate_selector_semantics(
    capture: Mapping[str, Any],
    *,
    layout: str,
) -> None:
    semantic = capture["artifact"]["semantic_identity"]
    validation = capture["validation"]
    selector = validation["selector_contract"]
    if set(selector) != {
        "workload",
        "physical_color_ids",
        "physical_helicity_ids",
        "structural_zero_helicity_ids",
        "selected_color_flow_ids",
        "selected_helicity_ids",
        "contract_sha256",
    }:
        raise ParityError("capture selector contract has an invalid schema")
    color_ids, _color_entries = _ordered_semantic_axis(
        semantic.get("physical_color_flows"),
        label="color-flow",
    )
    helicity_ids, helicity_entries = _ordered_semantic_axis(
        semantic.get("physical_helicities"),
        label="helicity",
    )
    structural_zero_ids = [
        str(entry["id"])
        for entry in helicity_entries
        if entry.get("structural_zero") is True
    ]
    common_matches = (
        selector.get("physical_color_ids") == color_ids
        and selector.get("physical_helicity_ids") == helicity_ids
        and selector.get("structural_zero_helicity_ids") == structural_zero_ids
    )
    if layout == "topology-replay":
        layout_matches = (
            selector.get("workload") == "single-runtime-selected-flow/helicity-sum"
            and selector.get("selected_color_flow_ids") == color_ids[:1]
            and selector.get("selected_helicity_ids") == []
            and validation.get("resolved_color_ids") == color_ids[:1]
            and validation.get("resolved_helicity_ids") == helicity_ids
        )
    elif layout == "all-flow-union":
        selected_helicity = next(
            (
                helicity_id
                for helicity_id in helicity_ids
                if helicity_id not in set(structural_zero_ids)
            ),
            None,
        )
        layout_matches = (
            selected_helicity is not None
            and selector.get("workload") == "all-flows/runtime-selected-single-helicity"
            and selector.get("selected_color_flow_ids") == []
            and selector.get("selected_helicity_ids") == [selected_helicity]
            and validation.get("resolved_color_ids") == color_ids
            and validation.get("resolved_helicity_ids") == [selected_helicity]
        )
    else:
        layout_matches = False
    if not common_matches or not layout_matches:
        raise ParityError("capture selector contract disagrees with physical axes")


def _comparison_errors(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> list[str]:
    errors: list[str] = []
    baseline_validation = baseline["validation"]
    candidate_validation = candidate["validation"]
    if baseline["prepared_model"]["sha256"] != candidate["prepared_model"]["sha256"]:
        errors.append("prepared model bytes differ")
    if _canonical_sha256(baseline["generation_request"]) != _canonical_sha256(
        candidate["generation_request"]
    ):
        errors.append("generation requests differ")
    if (
        baseline["parameters"]["payload_sha256"]
        != candidate["parameters"]["payload_sha256"]
    ):
        errors.append("default model parameters differ")
    if (
        baseline_validation["fixture"]["points_sha256"]
        != candidate_validation["fixture"]["points_sha256"]
    ):
        errors.append("deterministic phase-space fixtures differ")
    if _canonical_sha256(_semantic_selector_body(baseline)) != _canonical_sha256(
        _semantic_selector_body(candidate)
    ):
        errors.append("physical axes, selectors, normalization, or reduction differ")
    if baseline_validation.get("passes") is not True:
        errors.append("baseline selected/resolved self-comparison failed")
    if candidate_validation.get("passes") is not True:
        errors.append("candidate selected/resolved self-comparison failed")
    return errors


def _compare_captures(
    baseline_value: object,
    candidate_value: object,
    *,
    process_key: str,
    process_expression: str,
    layout: str,
) -> dict[str, object]:
    _validate_process_case(process_key, process_expression)
    if layout not in LAYOUTS:
        raise ParityError(f"unsupported LC flow layout: {layout}")
    baseline = _validate_capture(
        baseline_value,
        expected_variant="baseline",
        expected_process_key=process_key,
        expected_process=process_expression,
        expected_layout=layout,
    )
    candidate = _validate_capture(
        candidate_value,
        expected_variant="candidate",
        expected_process_key=process_key,
        expected_process=process_expression,
        expected_layout=layout,
    )
    errors = _comparison_errors(baseline, candidate)
    baseline_validation = baseline["validation"]
    candidate_validation = candidate["validation"]
    fixtures_match = (
        baseline_validation["fixture"]["points_sha256"]
        == candidate_validation["fixture"]["points_sha256"]
    )
    baseline_selector_sha256 = _canonical_sha256(_semantic_selector_body(baseline))
    candidate_selector_sha256 = _canonical_sha256(_semantic_selector_body(candidate))
    selectors_match = baseline_selector_sha256 == candidate_selector_sha256

    pointwise: list[dict[str, object]] = []
    baseline_totals = baseline_validation["selected_totals"]
    candidate_totals = candidate_validation["selected_totals"]
    if (
        fixtures_match
        and selectors_match
        and len(baseline_totals) == len(candidate_totals)
    ):
        for point_index, (baseline_total, candidate_total) in enumerate(
            zip(baseline_totals, candidate_totals, strict=True)
        ):
            pointwise.append(
                _content_address(
                    {
                        "kind": "pyamplicol-recurrence-numerical-point-comparison",
                        "schema_version": 1,
                        "point_index": point_index,
                        **_value_comparison(baseline_total, candidate_total),
                    }
                )
            )
    else:
        errors.append("selected-total inventories cannot be paired")

    componentwise: list[dict[str, object]] = []
    baseline_components = baseline_validation["resolved_components"]
    candidate_components = candidate_validation["resolved_components"]
    helicity_ids = baseline_validation["resolved_helicity_ids"]
    color_ids = baseline_validation["resolved_color_ids"]
    component_count = len(helicity_ids) * len(color_ids)
    components_pairable = (
        fixtures_match
        and selectors_match
        and len(baseline_components) == len(candidate_components)
        and all(
            len(point) == component_count
            for point in (*baseline_components, *candidate_components)
        )
    )
    if components_pairable:
        for point_index, (baseline_point, candidate_point) in enumerate(
            zip(baseline_components, candidate_components, strict=True)
        ):
            for component_index, (baseline_component, candidate_component) in enumerate(
                zip(baseline_point, candidate_point, strict=True)
            ):
                helicity_index, color_index = divmod(
                    component_index,
                    len(color_ids),
                )
                componentwise.append(
                    _content_address(
                        {
                            "kind": (
                                "pyamplicol-recurrence-numerical-"
                                "resolved-component-comparison"
                            ),
                            "schema_version": 1,
                            "point_index": point_index,
                            "component_index": component_index,
                            "helicity_id": helicity_ids[helicity_index],
                            "color_id": color_ids[color_index],
                            **_value_comparison(
                                baseline_component,
                                candidate_component,
                            ),
                        }
                    )
                )
    else:
        errors.append("resolved-component inventories cannot be paired")

    selector_comparison = _content_address(
        {
            "kind": "pyamplicol-recurrence-numerical-selector-comparison",
            "schema_version": 1,
            "baseline_sha256": baseline_selector_sha256,
            "candidate_sha256": candidate_selector_sha256,
            "passes": selectors_match,
        }
    )
    fixture_comparison = _content_address(
        {
            "kind": "pyamplicol-recurrence-numerical-fixture-comparison",
            "schema_version": 1,
            "baseline_points_sha256": baseline_validation["fixture"]["points_sha256"],
            "candidate_points_sha256": candidate_validation["fixture"]["points_sha256"],
            "point_count": len(baseline_validation["fixture"]["points"]),
            "passes": fixtures_match,
        }
    )
    passes = (
        not errors
        and fixture_comparison["passes"] is True
        and selector_comparison["passes"] is True
        and bool(pointwise)
        and bool(componentwise)
        and all(bool(item["passes"]) for item in pointwise)
        and all(bool(item["passes"]) for item in componentwise)
    )
    return _content_address(
        {
            "kind": COMPARISON_KIND,
            "schema_version": COMPARISON_SCHEMA,
            "process_key": process_key,
            "process_expression": process_expression,
            "layout": layout,
            "baseline_capture_sha256": baseline["content_sha256"],
            "candidate_capture_sha256": candidate["content_sha256"],
            "tolerances": {
                "absolute": ABSOLUTE_TOLERANCE,
                "relative": RELATIVE_TOLERANCE,
            },
            "fixture_comparison": fixture_comparison,
            "selector_comparison": selector_comparison,
            "parameter_comparison": _content_address(
                {
                    "kind": ("pyamplicol-recurrence-numerical-parameter-comparison"),
                    "schema_version": 1,
                    "baseline_sha256": baseline["parameters"]["payload_sha256"],
                    "candidate_sha256": candidate["parameters"]["payload_sha256"],
                    "passes": (
                        baseline["parameters"]["payload_sha256"]
                        == candidate["parameters"]["payload_sha256"]
                    ),
                }
            ),
            "pointwise_comparisons": pointwise,
            "resolved_component_comparisons": componentwise,
            "summary": {
                "point_count": len(pointwise),
                "resolved_component_count": len(componentwise),
                "maximum_absolute_point_difference": max(
                    (float(item["absolute_difference"]) for item in pointwise),
                    default=None,
                ),
                "maximum_relative_point_difference": max(
                    (float(item["relative_difference"]) for item in pointwise),
                    default=None,
                ),
                "maximum_absolute_component_difference": max(
                    (float(item["absolute_difference"]) for item in componentwise),
                    default=None,
                ),
                "maximum_relative_component_difference": max(
                    (float(item["relative_difference"]) for item in componentwise),
                    default=None,
                ),
            },
            "errors": errors,
            "passes": passes,
        }
    )


def _worker_environment(
    variant: Variant,
    capture_root: Path,
) -> tuple[dict[str, str], dict[str, str]]:
    temporary = capture_root / "tmp"
    cache = capture_root / "cache"
    pycache = capture_root / "pycache"
    matplotlib = cache / "matplotlib"
    for path in (temporary, cache, pycache, matplotlib):
        path.mkdir(parents=True, exist_ok=False)
    overrides = {
        SOURCE_CHECKOUT_ENV: str(variant.checkout),
        "MPLCONFIGDIR": str(matplotlib),
        "PYTHONHASHSEED": "0",
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPYCACHEPREFIX": str(pycache),
        "PYTHONPATH": str(variant.pythonpath),
        "SYMBOLICA_HIDE_BANNER": "1",
        "TEMP": str(temporary),
        "TMP": str(temporary),
        "TMPDIR": str(temporary),
        "XDG_CACHE_HOME": str(cache),
    }
    environment = os.environ.copy()
    environment.update(overrides)
    return environment, overrides


def _worker_command(
    variant: Variant,
    *,
    capture_root: Path,
    process_key: str,
    process_expression: str,
    layout: str,
    validation_samples: int,
    point_tile_size: int,
    jit_optimization_level: int,
) -> list[str]:
    artifact = capture_root / "artifact"
    capture_json = capture_root / "capture.json"
    return [
        str(variant.python),
        str(WATCHDOG_PATH),
        "--limit-gib",
        f"{WATCHDOG_LIMIT_GIB:g}",
        "--",
        str(variant.python),
        str(DRIVER_PATH),
        "_worker",
        "--variant",
        variant.name,
        "--process-key",
        process_key,
        "--process-expression",
        process_expression,
        "--layout",
        layout,
        "--prepared-model",
        str(variant.prepared_model),
        "--artifact",
        str(artifact),
        "--capture-json",
        str(capture_json),
        "--validation-samples",
        str(validation_samples),
        "--point-tile-size",
        str(point_tile_size),
        "--jit-optimization-level",
        str(jit_optimization_level),
    ]


def _terminate_process_tree(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
    else:
        process.terminate()
    try:
        process.wait(timeout=5.0)
        return
    except subprocess.TimeoutExpired:
        pass
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            return
    else:
        process.kill()
    process.wait()


def _run_process(
    command: Sequence[str],
    *,
    cwd: Path,
    environment: Mapping[str, str],
    stdout_path: Path,
    stderr_path: Path,
    timeout_seconds: float,
) -> ProcessOutcome:
    started = time.perf_counter()
    process: subprocess.Popen[bytes] | None = None
    try:
        with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
            process = subprocess.Popen(
                list(command),
                cwd=cwd,
                env=dict(environment),
                stdout=stdout,
                stderr=stderr,
                start_new_session=(os.name == "posix"),
            )
            try:
                exit_code = process.wait(timeout=timeout_seconds)
            except subprocess.TimeoutExpired:
                _terminate_process_tree(process)
                return ProcessOutcome(
                    exit_code=None,
                    timed_out=True,
                    wall_seconds=time.perf_counter() - started,
                    error=f"capture timeout after {timeout_seconds:g} seconds",
                )
    except OSError as error:
        if process is not None:
            _terminate_process_tree(process)
        return ProcessOutcome(
            exit_code=None,
            timed_out=False,
            wall_seconds=time.perf_counter() - started,
            error=f"{type(error).__name__}: {error}",
        )
    return ProcessOutcome(
        exit_code=exit_code,
        timed_out=False,
        wall_seconds=time.perf_counter() - started,
        error=None,
    )


def _capture_variant(
    variant: Variant,
    *,
    capture_root: Path,
    process_key: str,
    process_expression: str,
    layout: str,
    validation_samples: int,
    point_tile_size: int,
    jit_optimization_level: int,
    worker_timeout_seconds: float,
) -> tuple[dict[str, Any], dict[str, object]]:
    capture_root.mkdir(parents=True, exist_ok=False)
    environment, overrides = _worker_environment(variant, capture_root)
    stdout_path = capture_root / "stdout.log"
    stderr_path = capture_root / "stderr.log"
    command = _worker_command(
        variant,
        capture_root=capture_root,
        process_key=process_key,
        process_expression=process_expression,
        layout=layout,
        validation_samples=validation_samples,
        point_tile_size=point_tile_size,
        jit_optimization_level=jit_optimization_level,
    )
    started_at = _utc_now()
    outcome = _run_process(
        command,
        cwd=variant.checkout,
        environment=environment,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        timeout_seconds=worker_timeout_seconds,
    )
    finished_at = _utc_now()
    watchdog = ladder.parse_watchdog_log(stderr_path)
    capture_path = capture_root / "capture.json"
    if (
        outcome.timed_out
        or outcome.error is not None
        or outcome.exit_code != 0
        or watchdog.get("terminal_record") != "command-finished"
        or watchdog.get("limit_gib") != WATCHDOG_LIMIT_GIB
        or watchdog.get("limit_exceeded") is not False
        or watchdog.get("child_exit_code") != 0
        or not capture_path.is_file()
    ):
        raise ParityError(
            f"{variant.name}/{process_key}/{layout} capture failed; "
            f"exit={outcome.exit_code}, timed_out={outcome.timed_out}, "
            f"error={outcome.error}, watchdog={watchdog.get('terminal_record')}"
        )
    capture = _validate_capture(
        _json_object(capture_path, label="numerical capture"),
        expected_variant=variant.name,
        expected_process_key=process_key,
        expected_process=process_expression,
        expected_layout=layout,
        expected_checkout=variant.checkout,
        expected_artifact_root=capture_root,
        expected_generation_request=_generation_request(
            validation_samples=validation_samples,
            point_tile_size=point_tile_size,
            jit_optimization_level=jit_optimization_level,
        ),
    )
    invocation = _content_address(
        {
            "kind": "pyamplicol-recurrence-numerical-capture-invocation",
            "schema_version": 1,
            "variant": variant.name,
            "process_key": process_key,
            "layout": layout,
            "started_at_utc": started_at,
            "finished_at_utc": finished_at,
            "wall_seconds": outcome.wall_seconds,
            "command": {
                "argv": command,
                "argv_sha256": _canonical_sha256(command),
            },
            "environment_overrides": overrides,
            "stdout": _path_identity(stdout_path),
            "stderr": _path_identity(stderr_path),
            "capture_file": _path_identity(capture_path),
            "watchdog": watchdog,
        }
    )
    _atomic_write_json(capture_root / "invocation.json", invocation)
    return capture, invocation


def _comparison_file_name(process_key: str, layout: str) -> str:
    return f"{process_key}__{layout.replace('-', '_')}.json"


def _tooling_identity() -> dict[str, object]:
    return {
        "driver": _path_identity(DRIVER_PATH),
        "watchdog": _path_identity(WATCHDOG_PATH),
        "recurrence_harness": _path_identity(
            DRIVER_ROOT / "tools" / "developer" / "recurrence_z6g_benchmark.py"
        ),
        "campaign_ladder": _path_identity(
            DRIVER_ROOT / "tools" / "developer" / "recurrence_generation_ab_ladder.py"
        ),
    }


def _pin_variant_binding(
    bindings: dict[str, dict[str, object]],
    *,
    variant: Variant,
    capture: Mapping[str, Any],
    expected_model: Mapping[str, object],
    expected_generation_request: Mapping[str, object],
    expected_source: Mapping[str, object],
) -> None:
    if (
        _canonical_sha256(capture["prepared_model"])
        != _canonical_sha256(expected_model)
        or _canonical_sha256(capture["source"]) != _canonical_sha256(expected_source)
        or _canonical_sha256(capture["generation_request"])
        != _canonical_sha256(expected_generation_request)
    ):
        raise ParityError(
            f"{variant.name} capture escaped its configured source binding"
        )
    observed_binding = {
        "source": capture["source"],
        "runtime_provenance": capture["runtime_provenance"],
        "prepared_model": capture["prepared_model"],
    }
    expected_binding = bindings.setdefault(variant.name, observed_binding)
    if _canonical_sha256(observed_binding) != _canonical_sha256(expected_binding):
        raise ParityError(
            f"{variant.name} source or runtime changed during the campaign"
        )


def run(arguments: argparse.Namespace) -> dict[str, object]:
    output_root = _resolve_output_root(arguments.output_root)
    baseline = _variant(
        "baseline",
        python=arguments.baseline_python,
        checkout=arguments.baseline_checkout,
        pythonpath=arguments.baseline_pythonpath,
        prepared_model=arguments.baseline_prepared_model,
    )
    candidate = _variant(
        "candidate",
        python=arguments.candidate_python,
        checkout=arguments.candidate_checkout,
        pythonpath=arguments.candidate_pythonpath,
        prepared_model=arguments.candidate_prepared_model,
    )
    preflight_sources = {
        "baseline": _preflight_variant_source(baseline),
        "candidate": _preflight_variant_source(candidate),
    }
    baseline_model = _path_identity(baseline.prepared_model)
    candidate_model = _path_identity(candidate.prepared_model)
    if baseline_model["sha256"] != candidate_model["sha256"]:
        raise ParityError("baseline and candidate prepared-model bytes differ")
    if baseline.checkout == candidate.checkout:
        raise ParityError("baseline and candidate checkouts must be distinct")
    if not WATCHDOG_PATH.is_file():
        raise ParityError(f"audited memory watchdog is missing: {WATCHDOG_PATH}")

    tooling_before = _tooling_identity()
    expected_generation_request = _generation_request(
        validation_samples=arguments.validation_samples,
        point_tile_size=arguments.point_tile_size,
        jit_optimization_level=arguments.jit_optimization_level,
    )
    output_root.mkdir(parents=True, exist_ok=False)
    started_at = _utc_now()
    started = time.perf_counter()
    comparisons: list[dict[str, object]] = []
    capture_records: list[dict[str, object]] = []
    variant_bindings: dict[str, dict[str, object]] = {}
    sequence_index = 0
    for case_index, (process_key, process_expression) in enumerate(PROCESS_CASES):
        for layout in LAYOUTS:
            variants = (
                (baseline, candidate)
                if (case_index + LAYOUTS.index(layout)) % 2 == 0
                else (candidate, baseline)
            )
            captures: dict[str, dict[str, Any]] = {}
            invocations: dict[str, dict[str, object]] = {}
            for variant in variants:
                capture_root = (
                    output_root
                    / "captures"
                    / f"{sequence_index:02d}-{process_key}-{layout}"
                    / variant.name
                )
                capture, invocation = _capture_variant(
                    variant,
                    capture_root=capture_root,
                    process_key=process_key,
                    process_expression=process_expression,
                    layout=layout,
                    validation_samples=arguments.validation_samples,
                    point_tile_size=arguments.point_tile_size,
                    jit_optimization_level=arguments.jit_optimization_level,
                    worker_timeout_seconds=arguments.worker_timeout,
                )
                captures[variant.name] = capture
                invocations[variant.name] = invocation
                expected_model = (
                    baseline_model if variant.name == "baseline" else candidate_model
                )
                _pin_variant_binding(
                    variant_bindings,
                    variant=variant,
                    capture=capture,
                    expected_model=expected_model,
                    expected_generation_request=expected_generation_request,
                    expected_source=preflight_sources[variant.name],
                )
            comparison = _compare_captures(
                captures["baseline"],
                captures["candidate"],
                process_key=process_key,
                process_expression=process_expression,
                layout=layout,
            )
            comparison_path = (
                output_root / "comparisons" / _comparison_file_name(process_key, layout)
            )
            _atomic_write_json(comparison_path, comparison)
            comparisons.append(
                {
                    "process_key": process_key,
                    "process_expression": process_expression,
                    "layout": layout,
                    "passes": comparison["passes"],
                    "comparison_file": _path_identity(comparison_path),
                    "comparison_content_sha256": comparison["content_sha256"],
                }
            )
            capture_records.append(
                {
                    "process_key": process_key,
                    "layout": layout,
                    "order": [variant.name for variant in variants],
                    "baseline_capture_sha256": captures["baseline"]["content_sha256"],
                    "candidate_capture_sha256": captures["candidate"]["content_sha256"],
                    "baseline_invocation_sha256": invocations["baseline"][
                        "content_sha256"
                    ],
                    "candidate_invocation_sha256": invocations["candidate"][
                        "content_sha256"
                    ],
                }
            )
            sequence_index += 1
    tooling_after = _tooling_identity()
    if _canonical_sha256(tooling_after) != _canonical_sha256(tooling_before):
        raise ParityError("validator tooling changed during the campaign")
    final_sources = {
        "baseline": _preflight_variant_source(baseline),
        "candidate": _preflight_variant_source(candidate),
    }
    if _canonical_sha256(final_sources) != _canonical_sha256(preflight_sources):
        raise ParityError("baseline or candidate source changed during the campaign")
    finished_at = _utc_now()
    result = _content_address(
        {
            "kind": RESULT_KIND,
            "schema_version": RESULT_SCHEMA,
            "status": (
                "passed"
                if all(item["passes"] is True for item in comparisons)
                else "failed"
            ),
            "passes": all(item["passes"] is True for item in comparisons),
            "started_at_utc": started_at,
            "finished_at_utc": finished_at,
            "wall_seconds": time.perf_counter() - started,
            "configuration": {
                "processes": [
                    {
                        "key": process_key,
                        "expression": process_expression,
                    }
                    for process_key, process_expression in PROCESS_CASES
                ],
                "layouts": list(LAYOUTS),
                "validation_samples": arguments.validation_samples,
                "validation_seed": harness.VALIDATION_SEED,
                "point_tile_size": arguments.point_tile_size,
                "jit_optimization_level": arguments.jit_optimization_level,
                "worker_timeout_seconds": arguments.worker_timeout,
                "absolute_tolerance": ABSOLUTE_TOLERANCE,
                "relative_tolerance": RELATIVE_TOLERANCE,
                "watchdog_limit_gib": WATCHDOG_LIMIT_GIB,
                "generation_request_sha256": _canonical_sha256(
                    expected_generation_request
                ),
            },
            "tooling": tooling_before,
            "variants": {
                "baseline": {
                    "python": str(baseline.python),
                    "checkout": str(baseline.checkout),
                    "source": preflight_sources["baseline"],
                    "pythonpath": str(baseline.pythonpath),
                    "prepared_model": baseline_model,
                },
                "candidate": {
                    "python": str(candidate.python),
                    "checkout": str(candidate.checkout),
                    "source": preflight_sources["candidate"],
                    "pythonpath": str(candidate.pythonpath),
                    "prepared_model": candidate_model,
                },
            },
            "capture_records": capture_records,
            "variant_bindings": {
                name: {
                    **binding,
                    "binding_sha256": _canonical_sha256(binding),
                }
                for name, binding in sorted(variant_bindings.items())
            },
            "comparisons": comparisons,
            "comparison_count": len(comparisons),
            "passing_comparison_count": sum(
                item["passes"] is True for item in comparisons
            ),
        }
    )
    _atomic_write_json(output_root / "result.json", result)
    return result


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--baseline-python", type=Path, required=True)
    result.add_argument("--candidate-python", type=Path, required=True)
    result.add_argument("--baseline-checkout", type=Path, required=True)
    result.add_argument("--candidate-checkout", type=Path, required=True)
    result.add_argument("--baseline-pythonpath", type=Path, required=True)
    result.add_argument("--candidate-pythonpath", type=Path, required=True)
    result.add_argument("--baseline-prepared-model", type=Path, required=True)
    result.add_argument("--candidate-prepared-model", type=Path, required=True)
    result.add_argument("--output-root", type=Path, required=True)
    result.add_argument(
        "--validation-samples",
        type=_positive_int,
        default=DEFAULT_VALIDATION_SAMPLES,
    )
    result.add_argument(
        "--point-tile-size",
        type=_positive_int,
        default=DEFAULT_POINT_TILE_SIZE,
    )
    result.add_argument(
        "--jit-optimization-level",
        type=int,
        choices=(0, 1, 2, 3),
        default=DEFAULT_JIT_OPTIMIZATION_LEVEL,
    )
    result.add_argument(
        "--worker-timeout",
        type=_positive_float,
        default=DEFAULT_WORKER_TIMEOUT_SECONDS,
    )
    return result


def _worker_parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(add_help=False)
    result.add_argument("--variant", choices=("baseline", "candidate"), required=True)
    result.add_argument("--process-key", required=True)
    result.add_argument("--process-expression", required=True)
    result.add_argument("--layout", choices=LAYOUTS, required=True)
    result.add_argument("--prepared-model", type=Path, required=True)
    result.add_argument("--artifact", type=Path, required=True)
    result.add_argument("--capture-json", type=Path, required=True)
    result.add_argument("--validation-samples", type=_positive_int, required=True)
    result.add_argument("--point-tile-size", type=_positive_int, required=True)
    result.add_argument(
        "--jit-optimization-level",
        type=int,
        choices=(0, 1, 2, 3),
        required=True,
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    values = list(sys.argv[1:] if argv is None else argv)
    try:
        if values[:1] == ["_worker"]:
            arguments = _worker_parser().parse_args(values[1:])
            capture = _run_capture_worker(arguments)
            print(
                json.dumps(
                    {
                        "kind": capture["kind"],
                        "content_sha256": capture["content_sha256"],
                        "passes": capture["validation"]["passes"],
                    },
                    allow_nan=False,
                    sort_keys=True,
                )
            )
            return 0
        arguments = parser().parse_args(values)
        result = run(arguments)
    except (
        ParityError,
        harness.HarnessError,
        ImportError,
        OSError,
        RuntimeError,
        ValueError,
        json.JSONDecodeError,
    ) as error:
        print(f"recurrence-numerical-ab-parity: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "kind": result["kind"],
                "result_json": str(
                    arguments.output_root.expanduser().resolve() / "result.json"
                ),
                "status": result["status"],
                "passes": result["passes"],
                "content_sha256": result["content_sha256"],
            },
            allow_nan=False,
            sort_keys=True,
        )
    )
    return 0 if result["passes"] is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
