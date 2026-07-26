#!/usr/bin/env python3
# SPDX-License-Identifier: 0BSD
"""Audit the four authoritative x86 ``qq -> Z+6g`` recurrence captures.

This command deliberately consumes the complete three-lane output from
``recurrence_z6g_benchmark.py``.  It does not provide a shorter diagnostic
profiling path.  Every stored validation, artifact-semantic, worker, timing,
and interleaving contract is recomputed before the compiled/recurrence
performance ratios are considered.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.developer import (  # noqa: E402
    compiled_mode_matrix_x86 as shard_io,
)
from tools.developer import (  # noqa: E402
    recurrence_z6g_benchmark as benchmark,
)
from tools.developer import (  # noqa: E402
    x86_performance_runtime_bundle as runtime_bundle,
)

RESULT_KIND = "pyamplicol-x86-qq-z6g-recurrence-acceptance"
SCHEMA_VERSION = 1
CONTENT_IDENTITY_ALGORITHM = "sha256-canonical-json-body-v1"
COMPILED_RECURRENCE_RATIO_CEILING = 1.15
PERFORMANCE_BATCH_SIZES = (128, 1024)
REQUIRED_BATCH_SIZES = tuple(benchmark.DEFAULT_BATCH_SIZES)
REQUIRED_MODES = tuple(benchmark.EXECUTION_MODES)
TOPOLOGY_FLOW_ID = "flow:2,4,5,6,7,8,9,1"
UNION_HELICITY_ID = "h:-1,+1,-1,+1,-1,+1,-1,+1,-1"
_GIT_SHA = re.compile(r"[0-9a-f]{40}")

CAPTURE_CONTRACTS: dict[str, dict[str, object]] = {
    "builtin-topology": {
        "model": "builtin",
        "layout": "topology-replay",
        "color_flow": TOPOLOGY_FLOW_ID,
        "helicity": "1",
        "workload": "single-runtime-selected-flow/helicity-sum",
    },
    "builtin-union": {
        "model": "builtin",
        "layout": "all-flow-union",
        "color_flow": "1",
        "helicity": UNION_HELICITY_ID,
        "workload": "all-flows/runtime-selected-single-helicity",
    },
    "ufo-topology": {
        "model": "ufo",
        "layout": "topology-replay",
        "color_flow": TOPOLOGY_FLOW_ID,
        "helicity": "1",
        "workload": "single-runtime-selected-flow/helicity-sum",
    },
    "ufo-union": {
        "model": "ufo",
        "layout": "all-flow-union",
        "color_flow": "1",
        "helicity": UNION_HELICITY_ID,
        "workload": "all-flows/runtime-selected-single-helicity",
    },
}


class AcceptanceError(RuntimeError):
    """Raised when capture provenance or structure cannot be authenticated."""


def _canonical_sha256(value: object) -> str:
    return shard_io._canonical_sha256(value)


def _attach_content_identity(body: Mapping[str, object]) -> dict[str, object]:
    result = dict(body)
    result["content_identity"] = {
        "algorithm": CONTENT_IDENTITY_ALGORITHM,
        "sha256": _canonical_sha256(body),
    }
    return result


def _write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _exact_int(value: object, expected: int) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value == expected


def _positive_number(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and float(value) > 0.0
    )


def _configuration_namespace(configuration: Mapping[str, object]) -> SimpleNamespace:
    """Reconstruct only the arguments used by the harness acceptance functions."""

    return SimpleNamespace(
        batch_size=configuration.get("batch_sizes"),
        color_flow=configuration.get("color_flow_request"),
        generation_only=configuration.get("generation_only"),
        gluon_count=configuration.get("gluon_count"),
        helicity=configuration.get("helicity_request"),
        jit_optimization_level=configuration.get("jit_optimization_level"),
        lc_flow_layout=configuration.get("lc_flow_layout"),
        minimum_samples=configuration.get("minimum_samples"),
        modes=configuration.get("modes"),
        process_expression=None,
        specialize_flow_at_generation=configuration.get(
            "specialize_flow_at_generation"
        ),
        subprocess_samples=configuration.get("subprocess_samples"),
        target_runtime=configuration.get("target_runtime_seconds"),
        warmup_runs=configuration.get("warmup_runs"),
    )


def _recompute_harness_contracts(payload: Mapping[str, Any]) -> None:
    """Re-run the authoritative harness validators over retained evidence."""

    configuration = payload.get("configuration")
    profiles = payload.get("profiles")
    schedule = payload.get("profile_schedule")
    stored_validation = payload.get("validation_summary")
    stored_capture = payload.get("capture_acceptance")
    stored_milestone = payload.get("milestone0_acceptance")
    if (
        not isinstance(configuration, Mapping)
        or not isinstance(profiles, Mapping)
        or not isinstance(schedule, Mapping)
        or not isinstance(stored_validation, Mapping)
        or not isinstance(stored_capture, Mapping)
        or not isinstance(stored_milestone, Mapping)
    ):
        raise AcceptanceError("capture lacks complete harness evidence")
    arguments = _configuration_namespace(configuration)
    try:
        validation = benchmark._pairwise_profile_validation(profiles)
        capture = benchmark._capture_acceptance(
            arguments,
            profiles,
            validation,
            profile_schedule=schedule,
        )
        milestone = benchmark._milestone0_acceptance_manifest(arguments, capture)
    except (
        AttributeError,
        KeyError,
        TypeError,
        ValueError,
        benchmark.HarnessError,
    ) as error:
        raise AcceptanceError("capture evidence cannot be revalidated") from error
    if validation != stored_validation:
        raise AcceptanceError("stored numerical validation is not reproducible")
    if capture != stored_capture:
        raise AcceptanceError("stored capture acceptance is not reproducible")
    if milestone != stored_milestone:
        raise AcceptanceError("stored milestone evidence is not reproducible")
    if (
        capture.get("complete") is not True
        or capture.get("evidence_complete") is not True
        or capture.get("passes") is not True
        or capture.get("authoritative_eligible") is not True
        or capture.get("authoritative_ineligibility_reasons") != []
        or capture.get("generation_specialized_axes_by_mode") != {}
        or capture.get("incomplete_physical_axes") != []
        or validation.get("passes") is not True
    ):
        raise AcceptanceError("capture is not complete authoritative passing evidence")


def _validate_configuration(
    payload: Mapping[str, Any],
    *,
    role: str,
) -> Mapping[str, Any]:
    configuration = payload.get("configuration")
    contract = CAPTURE_CONTRACTS[role]
    if not isinstance(configuration, Mapping):
        raise AcceptanceError(f"{role} has no configuration")
    if (
        configuration.get("modes") != list(REQUIRED_MODES)
        or configuration.get("batch_sizes") != list(REQUIRED_BATCH_SIZES)
        or configuration.get("lc_flow_layout") != contract["layout"]
        or configuration.get("color_flow_request") != contract["color_flow"]
        or configuration.get("helicity_request") != contract["helicity"]
        or not _exact_int(configuration.get("gluon_count"), 6)
        or not _exact_int(configuration.get("jit_optimization_level"), 3)
        or not _exact_int(configuration.get("minimum_samples"), 7)
        or not _exact_int(configuration.get("subprocess_samples"), 7)
        or not _exact_int(configuration.get("warmup_runs"), 2)
        or not _exact_int(configuration.get("point_tile_size"), 1024)
        or not _exact_int(configuration.get("validation_samples"), 10)
        or not _positive_number(configuration.get("target_runtime_seconds"))
        or float(configuration["target_runtime_seconds"]) < 5.0
        or configuration.get("generation_only") is not False
        or configuration.get("allow_diagnostic_incomplete_success") is not False
        or configuration.get("specialize_flow_at_generation") is not False
        or configuration.get("external_watchdog_required_for_long_runs") is not True
    ):
        raise AcceptanceError(f"{role} does not use the authoritative benchmark policy")
    if (
        payload.get("process") != "u u~ > Z g g g g g g"
        or payload.get("process_name") != "uubar_Z_6g"
        or payload.get("workload") != contract["workload"]
    ):
        raise AcceptanceError(f"{role} process/workload identity is wrong")
    return configuration


def _stable_capture_installation(runtime: Mapping[str, Any]) -> dict[str, object]:
    active = runtime.get("active_build_info")
    distribution = runtime.get("installed_distribution")
    native = runtime.get("native_extension")
    if (
        not isinstance(active, Mapping)
        or not isinstance(active.get("payload"), Mapping)
        or not isinstance(distribution, Mapping)
        or not isinstance(native, Mapping)
    ):
        raise AcceptanceError("capture runtime identity is incomplete")
    build_files = distribution.get("build_info_files")
    native_modules = distribution.get("native_modules")
    content = distribution.get("distribution_content")
    if (
        not isinstance(build_files, list)
        or len(build_files) != 1
        or not isinstance(build_files[0], Mapping)
        or not isinstance(native_modules, list)
        or len(native_modules) != 1
        or not isinstance(native_modules[0], Mapping)
        or not isinstance(content, Mapping)
    ):
        raise AcceptanceError("capture installed-distribution identity is incomplete")
    return {
        "package_version": distribution.get("package_version"),
        "build_info": dict(active["payload"]),
        "build_info_sha256": active.get("sha256"),
        "distribution_content": {
            key: content.get(key)
            for key in ("algorithm", "sha256", "file_count", "size_bytes")
        },
        "native_module": {
            key: native_modules[0].get(key)
            for key in ("relative_path", "sha256", "size_bytes")
        },
        "native_runtime": {
            "sha256": native.get("sha256"),
            "size_bytes": native.get("size_bytes"),
            "build_inputs_sha256": native.get("build_inputs_sha256"),
            "package_version": native.get("package_version"),
        },
        "distribution_build_info_sha256": build_files[0].get("sha256"),
    }


def _validate_runtime(
    payload: Mapping[str, Any],
    *,
    role: str,
    expected_revision: str,
    bundle: Mapping[str, Any],
) -> str:
    source = payload.get("source")
    runtime = payload.get("runtime_provenance")
    provenance = payload.get("provenance")
    host = (
        provenance.get("host")
        if isinstance(provenance, Mapping)
        else None
    )
    if (
        not isinstance(source, Mapping)
        or source.get("revision") != expected_revision
        or source.get("dirty") is not False
        or source.get("untracked_files_checked") is not True
        or not isinstance(runtime, Mapping)
        or not isinstance(host, Mapping)
        or host.get("system") != "Linux"
        or host.get("machine") not in {"x86_64", "AMD64"}
    ):
        raise AcceptanceError(f"{role} was not measured from exact clean x86 source")
    installations = bundle.get("installations")
    current = (
        installations.get("current")
        if isinstance(installations, Mapping)
        else None
    )
    if not isinstance(current, Mapping):
        raise AcceptanceError("runtime bundle has no current installation identity")
    observed = _stable_capture_installation(runtime)
    expected_native = current.get("native_module")
    expected_build = current.get("build_info")
    if (
        observed.get("package_version") != current.get("package_version")
        or observed.get("build_info") != expected_build
        or observed.get("build_info_sha256") != current.get("build_info_sha256")
        or observed.get("distribution_build_info_sha256")
        != current.get("build_info_sha256")
        or observed.get("distribution_content")
        != current.get("distribution_content")
        or observed.get("native_module") != expected_native
        or not isinstance(expected_build, Mapping)
        or not isinstance(expected_native, Mapping)
        or observed["native_runtime"].get("sha256")
        != expected_native.get("sha256")
        or observed["native_runtime"].get("size_bytes")
        != expected_native.get("size_bytes")
        or observed["native_runtime"].get("build_inputs_sha256")
        != expected_build.get("native_build_inputs_sha256")
    ):
        raise AcceptanceError(f"{role} runtime does not match the frozen bundle")
    return _canonical_sha256(runtime)


def _validate_model(
    configuration: Mapping[str, Any],
    *,
    role: str,
    expected_revision: str,
    bundle: Mapping[str, Any],
) -> dict[str, object]:
    identities = configuration.get("model_identities")
    if not isinstance(identities, Mapping) or set(identities) != set(REQUIRED_MODES):
        raise AcceptanceError(f"{role} has an incomplete model identity inventory")
    model_kind = CAPTURE_CONTRACTS[role]["model"]
    if model_kind == "builtin":
        if configuration.get("prepared_model_path") is not None:
            raise AcceptanceError(f"{role} unexpectedly uses an explicit model")
        compiled = identities.get("compiled")
        if (
            not isinstance(compiled, Mapping)
            or compiled
            != {
                "kind": "built-in-sm-source",
                "resource_id": None,
                "source_revision": expected_revision,
                "compile_excluded_from_generation": False,
            }
        ):
            raise AcceptanceError(f"{role} compiled lane is not built-in source")
        packaged: dict[str, object] | None = None
        for mode in ("eager", "recurrence"):
            identity = identities.get(mode)
            if (
                not isinstance(identity, Mapping)
                or identity.get("kind") != "packaged-prepared-model"
                or identity.get("resource_id") != benchmark.PREPARED_MODEL_ID
                or identity.get("compile_excluded_from_generation") is not True
                or not _positive_number(identity.get("size_bytes"))
                or not isinstance(identity.get("sha256"), str)
            ):
                raise AcceptanceError(f"{role} {mode} model identity is invalid")
            stable = {
                key: identity.get(key)
                for key in (
                    "kind",
                    "resource_id",
                    "size_bytes",
                    "sha256",
                    "compile_excluded_from_generation",
                )
            }
            if packaged is None:
                packaged = stable
            elif stable != packaged:
                raise AcceptanceError(f"{role} packaged model identities differ")
        assert packaged is not None
        return {"kind": "builtin", "packaged_prepared_model": packaged}
    if not isinstance(configuration.get("prepared_model_path"), str):
        raise AcceptanceError(f"{role} has no explicit UFO prepared model")
    prepared_models = bundle.get("prepared_models")
    expected_file = (
        prepared_models.get("ufo-sm")
        if isinstance(prepared_models, Mapping)
        else None
    )
    if not isinstance(expected_file, Mapping):
        raise AcceptanceError("runtime bundle has no UFO prepared-model identity")
    stable_explicit: dict[str, object] | None = None
    for mode in REQUIRED_MODES:
        identity = identities.get(mode)
        file_identity = identity.get("file") if isinstance(identity, Mapping) else None
        if (
            not isinstance(identity, Mapping)
            or identity.get("kind") != "explicit-prepared-model"
            or identity.get("resource_id") is not None
            or identity.get("compile_excluded_from_generation") is not True
            or not isinstance(file_identity, Mapping)
            or file_identity.get("sha256") != expected_file.get("sha256")
            or file_identity.get("size_bytes") != expected_file.get("size_bytes")
        ):
            raise AcceptanceError(f"{role} {mode} UFO model identity is invalid")
        stable = {
            "kind": identity.get("kind"),
            "resource_id": identity.get("resource_id"),
            "compile_excluded_from_generation": identity.get(
                "compile_excluded_from_generation"
            ),
            "file": {
                "sha256": file_identity.get("sha256"),
                "size_bytes": file_identity.get("size_bytes"),
            },
        }
        if stable_explicit is None:
            stable_explicit = stable
        elif stable != stable_explicit:
            raise AcceptanceError(f"{role} explicit model identities differ")
    assert stable_explicit is not None
    return {"kind": "ufo", "prepared_model": stable_explicit}


def _profile_for_batch(
    payload: Mapping[str, Any],
    *,
    mode: str,
    batch_size: int,
) -> Mapping[str, Any]:
    profiles = payload.get("profiles")
    lane = profiles.get(mode) if isinstance(profiles, Mapping) else None
    measurements = lane.get("profiles") if isinstance(lane, Mapping) else None
    if not isinstance(measurements, list):
        raise AcceptanceError(f"{mode} profile inventory is missing")
    matches = [
        measurement
        for measurement in measurements
        if isinstance(measurement, Mapping)
        and measurement.get("batch_size") == batch_size
    ]
    if len(matches) != 1:
        raise AcceptanceError(f"{mode} batch {batch_size} is not unique")
    return matches[0]


def _paired_cell_evidence(
    payload: Mapping[str, Any],
    *,
    batch_size: int,
    ceiling: float,
) -> dict[str, object]:
    by_mode: dict[str, dict[int, float]] = {}
    for mode in ("compiled", "recurrence"):
        measurement = _profile_for_batch(
            payload,
            mode=mode,
            batch_size=batch_size,
        )
        samples = measurement.get("subprocess_samples")
        if (
            measurement.get("interrupted") is not False
            or not _exact_int(measurement.get("sample_count"), 7)
            or not _exact_int(measurement.get("subprocess_sample_count"), 7)
            or measurement.get("statistics_contract")
            != "subprocess-median-and-raw-mad-v1"
            or not isinstance(samples, list)
            or len(samples) != 7
        ):
            raise AcceptanceError(
                f"{mode} batch {batch_size} is not seven-sample evidence"
            )
        rounds: dict[int, float] = {}
        for sample in samples:
            round_index = sample.get("round") if isinstance(sample, Mapping) else None
            wall = (
                sample.get("wall_seconds_per_point")
                if isinstance(sample, Mapping)
                else None
            )
            if (
                isinstance(round_index, bool)
                or not isinstance(round_index, int)
                or round_index < 0
                or round_index >= 7
                or round_index in rounds
                or not _positive_number(wall)
            ):
                raise AcceptanceError(
                    f"{mode} batch {batch_size} has invalid paired samples"
                )
            rounds[round_index] = float(wall)
        if set(rounds) != set(range(7)):
            raise AcceptanceError(
                f"{mode} batch {batch_size} does not cover seven rounds"
            )
        values = list(rounds.values())
        observed_median = statistics.median(values)
        observed_mad = statistics.median(
            abs(value - observed_median) for value in values
        )
        if (
            not math.isclose(
                float(measurement.get("wall_seconds_per_point_median", math.nan)),
                observed_median,
                rel_tol=1.0e-15,
                abs_tol=0.0,
            )
            or not math.isclose(
                float(measurement.get("wall_seconds_per_point_mad", math.nan)),
                observed_mad,
                rel_tol=1.0e-15,
                abs_tol=0.0,
            )
        ):
            raise AcceptanceError(
                f"{mode} batch {batch_size} headline statistics changed"
            )
        by_mode[mode] = rounds
    ratios = [
        by_mode["compiled"][round_index] / by_mode["recurrence"][round_index]
        for round_index in range(7)
    ]
    ratio_median = statistics.median(ratios)
    ratio_mad = statistics.median(abs(value - ratio_median) for value in ratios)
    upper = ratio_median + 3.0 * ratio_mad
    passes = (
        ratio_mad > 0.0
        and ratio_median <= ceiling
        and upper <= ceiling
    )
    return {
        "batch_size": batch_size,
        "pairing": "same-interleaved-schedule-round-v1",
        "sample_count": len(ratios),
        "compiled_wall_seconds_per_point_by_round": [
            by_mode["compiled"][index] for index in range(7)
        ],
        "recurrence_wall_seconds_per_point_by_round": [
            by_mode["recurrence"][index] for index in range(7)
        ],
        "compiled_over_recurrence_ratios_by_round": ratios,
        "ratio_statistics": {
            "contract": "median-and-raw-mad-v1",
            "median": ratio_median,
            "raw_mad": ratio_mad,
            "upper_three_raw_mad": upper,
        },
        "ceiling": ceiling,
        "passes": passes,
    }


def _checked_bundle(
    path: Path,
    *,
    workflow_run_id: str,
    expected_revision: str,
) -> tuple[dict[str, Any], dict[str, object]]:
    payload, file_identity = shard_io._checked_json(
        path,
        label="x86 runtime bundle manifest",
    )
    try:
        runtime_bundle._require_content_identity(payload)
    except runtime_bundle.BundleError as error:
        raise AcceptanceError("runtime bundle content identity is invalid") from error
    if (
        payload.get("kind") != runtime_bundle.BUNDLE_KIND
        or payload.get("schema_version") != runtime_bundle.SCHEMA_VERSION
        or payload.get("target") != "x86_64-unknown-linux-gnu"
        or payload.get("workflow_run_id") != workflow_run_id
        or payload.get("expected_current_revision") != expected_revision
        or payload.get("passes") is not True
    ):
        raise AcceptanceError("runtime bundle is not bound to this workflow/source")
    return payload, file_identity


def audit(
    *,
    capture_paths: Mapping[str, Path],
    runtime_bundle_manifest: Path,
    workflow_run_id: str,
    expected_current_revision: str,
    ratio_ceiling: float = COMPILED_RECURRENCE_RATIO_CEILING,
) -> dict[str, object]:
    if set(capture_paths) != set(CAPTURE_CONTRACTS):
        raise AcceptanceError("exactly the four canonical capture roles are required")
    if not _positive_number(ratio_ceiling):
        raise AcceptanceError("compiled/recurrence ratio ceiling must be positive")
    bundle, bundle_file = _checked_bundle(
        runtime_bundle_manifest,
        workflow_run_id=workflow_run_id,
        expected_revision=expected_current_revision,
    )
    capture_evidence: dict[str, object] = {}
    runtime_digests: set[str] = set()
    for role in CAPTURE_CONTRACTS:
        payload, file_identity = shard_io._checked_json(
            capture_paths[role],
            label=f"{role} recurrence capture",
        )
        if (
            payload.get("kind") != benchmark.RESULT_KIND
            or payload.get("schema_version") != benchmark.RESULT_SCHEMA
            or payload.get("complete") is not True
            or payload.get("passes") is not True
        ):
            raise AcceptanceError(f"{role} is not a passing benchmark result")
        configuration = _validate_configuration(payload, role=role)
        runtime_digest = _validate_runtime(
            payload,
            role=role,
            expected_revision=expected_current_revision,
            bundle=bundle,
        )
        runtime_digests.add(runtime_digest)
        model = _validate_model(
            configuration,
            role=role,
            expected_revision=expected_current_revision,
            bundle=bundle,
        )
        _recompute_harness_contracts(payload)
        cells = [
            _paired_cell_evidence(
                payload,
                batch_size=batch_size,
                ceiling=float(ratio_ceiling),
            )
            for batch_size in PERFORMANCE_BATCH_SIZES
        ]
        capture_evidence[role] = {
            "capture_file": {
                key: file_identity[key]
                for key in ("size_bytes", "sha256", "canonical_sha256")
            },
            "source_revision": expected_current_revision,
            "runtime_provenance_sha256": runtime_digest,
            "model": model,
            "layout": CAPTURE_CONTRACTS[role]["layout"],
            "workload": CAPTURE_CONTRACTS[role]["workload"],
            "numerical_validation": {
                "contract": "recomputed-recurrence-z6g-pairwise-and-resolved-v1",
                "passes": True,
            },
            "performance_cells": cells,
            "passes": all(cell["passes"] is True for cell in cells),
        }
    if len(runtime_digests) != 1:
        raise AcceptanceError("the four captures used different runtime provenance")
    all_cells = [
        cell
        for capture in capture_evidence.values()
        if isinstance(capture, Mapping)
        for cell in capture["performance_cells"]
    ]
    body = {
        "kind": RESULT_KIND,
        "schema_version": SCHEMA_VERSION,
        "workflow_run_id": workflow_run_id,
        "target": "x86_64-unknown-linux-gnu",
        "expected_current_revision": expected_current_revision,
        "runtime_bundle": {
            "content_sha256": bundle["content_identity"]["sha256"],
            "file_sha256": bundle_file["sha256"],
        },
        "policy": {
            "required_capture_roles": list(CAPTURE_CONTRACTS),
            "required_modes": list(REQUIRED_MODES),
            "required_batch_sizes": list(REQUIRED_BATCH_SIZES),
            "performance_batch_sizes": list(PERFORMANCE_BATCH_SIZES),
            "minimum_target_runtime_seconds_per_worker": 5.0,
            "subprocess_samples_per_cell": 7,
            "native_wall_blocks_per_worker": 7,
            "warmup_runs": 2,
            "numerical_validation_samples": 10,
            "compiled_over_recurrence_ratio_ceiling": float(ratio_ceiling),
            "ratio_gate": "median-plus-three-raw-mad-at-or-below-ceiling-v1",
            "diagnostic_shortcuts_allowed": False,
        },
        "captures": capture_evidence,
        "performance_cell_count": len(all_cells),
        "passes": (
            len(all_cells) == len(CAPTURE_CONTRACTS) * len(PERFORMANCE_BATCH_SIZES)
            and all(cell["passes"] is True for cell in all_cells)
        ),
    }
    return _attach_content_identity(body)


def _git_sha(value: str) -> str:
    if _GIT_SHA.fullmatch(value) is None:
        raise argparse.ArgumentTypeError("must be a lowercase 40-character Git SHA")
    return value


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    for role in CAPTURE_CONTRACTS:
        result.add_argument(f"--{role}", type=Path, required=True)
    result.add_argument("--runtime-bundle-manifest", type=Path, required=True)
    result.add_argument("--workflow-run-id", required=True)
    result.add_argument(
        "--expected-current-revision",
        type=_git_sha,
        required=True,
    )
    result.add_argument(
        "--compiled-recurrence-ratio-ceiling",
        type=float,
        default=COMPILED_RECURRENCE_RATIO_CEILING,
    )
    result.add_argument("--output", type=Path, required=True)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    try:
        result = audit(
            capture_paths={
                role: getattr(arguments, role.replace("-", "_"))
                for role in CAPTURE_CONTRACTS
            },
            runtime_bundle_manifest=arguments.runtime_bundle_manifest,
            workflow_run_id=arguments.workflow_run_id,
            expected_current_revision=arguments.expected_current_revision,
            ratio_ceiling=arguments.compiled_recurrence_ratio_ceiling,
        )
        _write_json_atomic(arguments.output, result)
    except (
        AcceptanceError,
        OSError,
        shard_io.ShardError,
    ) as error:
        print(f"x86-qq-recurrence-acceptance: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0 if result["passes"] is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
