# SPDX-License-Identifier: 0BSD
"""One-cell measurement orchestration over the public pyAmpliCol Python API."""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Mapping
from pathlib import Path

from .agreements import (
    DIRECT_AGREEMENT_FIELD,
    LC_COMMON_COMPONENT_FIELD,
    evaluate_lc_common_component,
)
from .cache import empty_measurement
from .models import (
    Accuracy,
    CellSpec,
    ExecutionMode,
    ModelKey,
    ResultStatus,
    Workload,
)
from .phase_state import WorkerPhaseReporter
from .runner import (
    INDEPENDENT_RELATIVE_TOLERANCE,
    RELATIVE_TOLERANCE,
    GeneratedArtifact,
    RunnerError,
    RunnerSettings,
    SelectorContract,
    _real_nonnegative,
    _selector_kwargs,
    generate_artifact,
    pointwise_validation,
    profile_runtime,
    provenance_payload,
    runtime_identity_payload,
    runtime_validation_points,
    validate_artifact_contract,
)
from .source_identity import (
    ReportSourceIdentityError,
    inspect_report_source,
    require_eligible_report_source,
)


def shared_validation_points(process: str) -> object:
    from pyamplicol.models.builtin.validation import generic_validation_point

    return (
        tuple(
            tuple(float(component) for component in particle.momentum)
            for particle in generic_validation_point(process)
        ),
    )


def _reproduction_momenta(points: object) -> object:
    """Return the exact measured points in portable public-CLI JSON form."""

    to_list = getattr(points, "tolist", None)
    payload = to_list() if callable(to_list) else points
    try:
        encoded = json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as error:
        raise RunnerError(
            "report profiling points cannot be materialized for reproduction"
        ) from error
    return json.loads(encoded)


def source_revision(repo_root: Path, *, require_clean: bool = False) -> str:
    try:
        identity = (
            require_eligible_report_source(repo_root)
            if require_clean
            else inspect_report_source(repo_root)
        )
    except ReportSourceIdentityError as error:
        if require_clean:
            raise RunnerError(
                "report source worktree contains changes outside generated "
                f"report outputs: {error}"
            ) from error
        return "unknown"
    return identity.revision


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _baseline_selector_contract(
    baseline: Mapping[str, object] | None,
) -> SelectorContract | None:
    if baseline is None:
        return None
    raw = baseline.get("selector_contract")
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise RunnerError("baseline selector_contract must be an object")
    return SelectorContract.from_mapping(raw)


def _measurement_selector_contract(
    cell: CellSpec,
    baseline: Mapping[str, object] | None,
    selector_provider: Mapping[str, object] | None,
) -> SelectorContract | None:
    """Choose one canonical LC selector before candidate work starts."""

    if cell.measurement.accuracy is not Accuracy.LC:
        return None
    baseline_contract = _baseline_selector_contract(baseline)
    provider_contract: SelectorContract | None = None
    if selector_provider is not None:
        if selector_provider.get("status") != ResultStatus.OK.value:
            raise RunnerError("LC selector provider is not a successful measurement")
        try:
            provider_contract = _baseline_selector_contract(selector_provider)
        except (TypeError, ValueError) as error:
            raise RunnerError("LC selector provider has an invalid contract") from error
        if provider_contract is None:
            raise RunnerError("LC selector provider has no selector_contract")
    if (
        baseline_contract is not None
        and provider_contract is not None
        and baseline_contract != provider_contract
    ):
        raise RunnerError(
            "LC validation baseline and selected-flow selector provider disagree"
        )
    if baseline_contract is not None:
        return baseline_contract
    return provider_contract


def _baseline_matrix_element(
    baseline: Mapping[str, object] | None,
) -> float | None:
    if baseline is None:
        return None
    if baseline.get("status") != ResultStatus.OK.value:
        raise RunnerError("candidate baseline is not a valid completed measurement")
    value = baseline.get("matrix_element")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RunnerError("candidate baseline has no matrix element")
    return float(value)


def _require_nonzero_lc_all_flow_baseline(
    cell: CellSpec,
    baseline: Mapping[str, object] | None,
) -> None:
    """Reject inherited selector contracts that authenticate only a zero lane."""

    if (
        baseline is None
        or cell.measurement.accuracy is not Accuracy.LC
        or cell.workload is not Workload.ALL_FLOW
    ):
        return
    validation = baseline.get("validation")
    common = (
        validation.get(LC_COMMON_COMPONENT_FIELD)
        if isinstance(validation, Mapping)
        else None
    )
    value = common.get("value") if isinstance(common, Mapping) else None
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise RunnerError(
            "LC all-flow baseline selector is structural zero; remeasure the "
            "baseline with a nonzero fixed-helicity selector before generating "
            "a dependent candidate"
        )


def generated_artifact_from_measurement(
    measurement: Mapping[str, object],
) -> GeneratedArtifact:
    if measurement.get("status") != ResultStatus.OK.value:
        raise RunnerError("only a successful measurement can provide an artifact")
    artifact = measurement.get("artifact")
    provenance = measurement.get("provenance")
    generation_seconds = measurement.get("generation_seconds")
    if (
        not isinstance(artifact, Mapping)
        or not isinstance(provenance, Mapping)
        or isinstance(generation_seconds, bool)
        or not isinstance(generation_seconds, (int, float))
    ):
        raise RunnerError("measurement does not contain reusable artifact metadata")
    path = artifact.get("path")
    process_id = artifact.get("process_id")
    requested = provenance.get("requested_config")
    effective = provenance.get("effective_config")
    preparation_seconds = provenance.get("model_preparation_seconds", 0.0)
    preparation_reused = provenance.get("model_preparation_reused", True)
    generation_command_path = provenance.get("generation_command_path")
    numerical_relation_correctness = provenance.get("numerical_relation_correctness")
    numerical_relation_fallback = provenance.get("numerical_relation_fallback")
    if (
        not isinstance(path, str)
        or not isinstance(process_id, str)
        or not isinstance(requested, Mapping)
        or not isinstance(effective, Mapping)
        or isinstance(preparation_seconds, bool)
        or not isinstance(preparation_seconds, (int, float))
        or not isinstance(preparation_reused, bool)
        or (
            generation_command_path is not None
            and not isinstance(generation_command_path, str)
        )
        or (
            numerical_relation_correctness is not None
            and not isinstance(numerical_relation_correctness, Mapping)
        )
        or (
            numerical_relation_fallback is not None
            and not isinstance(numerical_relation_fallback, Mapping)
        )
    ):
        raise RunnerError("measurement reusable artifact metadata is malformed")
    artifact_path = Path(path).expanduser().resolve(strict=False)
    if not artifact_path.is_dir():
        raise RunnerError(f"reusable artifact directory is missing: {artifact_path}")
    return GeneratedArtifact(
        path=artifact_path,
        process_id=process_id,
        generation_seconds=float(generation_seconds),
        model_preparation_seconds=float(preparation_seconds),
        model_preparation_reused=preparation_reused,
        requested_config=dict(requested),
        effective_config=dict(effective),
        generation_command_path=generation_command_path,
        numerical_relation_correctness=(
            None
            if numerical_relation_correctness is None
            else dict(numerical_relation_correctness)
        ),
        numerical_relation_fallback=(
            None
            if numerical_relation_fallback is None
            else dict(numerical_relation_fallback)
        ),
    )


def _reuse_artifact_for_measurement(
    artifact: GeneratedArtifact,
    *,
    phase_reporter: WorkerPhaseReporter | None,
) -> GeneratedArtifact:
    """Close the local generation phase when no generation work is needed."""

    if phase_reporter is not None:
        # The inherited generation_seconds remains the cost of producing the
        # reusable artifact.  This empty local interval only gives the current
        # supervised worker an authenticated terminal phase state.
        with phase_reporter.generation():
            pass
    return artifact


def _pointwise_tolerance(cell: CellSpec) -> float:
    if cell.dataset_id.startswith("matrix_recurrence_") or cell.dataset_id.startswith(
        "z_"
    ):
        return INDEPENDENT_RELATIVE_TOLERANCE
    return RELATIVE_TOLERANCE


def _stable_runtime_identity(
    identity: Mapping[str, object],
) -> dict[str, object]:
    """Remove only the growing loaded-module observation inventory."""

    stable = dict(identity)
    raw_policy = stable.get("loaded_module_origin_policy")
    if isinstance(raw_policy, Mapping):
        policy = dict(raw_policy)
        for field in (
            "observed_module_count",
            "observations",
            "observations_sha256",
        ):
            policy.pop(field, None)
        stable["loaded_module_origin_policy"] = policy
    return stable


def _runtime_identity_digest(identity: Mapping[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(
            identity,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    ).hexdigest()


def _validate_runtime_identity_postflight(
    initial: Mapping[str, object],
    postflight: Mapping[str, object],
) -> None:
    if _stable_runtime_identity(initial) != _stable_runtime_identity(postflight):
        raise RunnerError(
            "candidate runtime identity changed during report measurement"
        )
    initial_policy = initial.get("loaded_module_origin_policy")
    postflight_policy = postflight.get("loaded_module_origin_policy")
    if not isinstance(initial_policy, Mapping) or not isinstance(
        postflight_policy,
        Mapping,
    ):
        raise RunnerError("report runtime has no loaded-module origin evidence")
    initial_observations = initial_policy.get("observations")
    postflight_observations = postflight_policy.get("observations")
    if not isinstance(initial_observations, list) or not isinstance(
        postflight_observations,
        list,
    ):
        raise RunnerError("report runtime loaded-module evidence is incomplete")
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
        raise RunnerError("report runtime lost an authenticated loaded-module origin")


def _load_runtime(artifact_path: Path, process_id: str) -> object:
    from pyamplicol.api import Runtime

    return Runtime.load(artifact_path, process=process_id)


def _resolution_benchmark_config(effective_config: Mapping[str, object]) -> object:
    from pyamplicol.config import Action
    from pyamplicol.config.resolver import resolve_config

    return resolve_config(
        effective_config,
        action=Action.GENERATE,
        base_dir=Path.cwd(),
    ).effective.benchmark


def measure_pyamplicol_cell(
    cell: CellSpec,
    *,
    artifact_path: Path,
    settings: RunnerSettings,
    repo_root: Path,
    baseline: Mapping[str, object] | None,
    selector_provider: Mapping[str, object] | None = None,
    prepared_model_path: Path | None = None,
    reused_artifact: GeneratedArtifact | None = None,
    phase_reporter: WorkerPhaseReporter | None = None,
) -> dict[str, object]:
    """Generate or retime one complete-coverage pyAmpliCol artifact."""

    _require_nonzero_lc_all_flow_baseline(cell, baseline)
    contract = _measurement_selector_contract(cell, baseline, selector_provider)
    generated = (
        generate_artifact(
            cell,
            artifact_path,
            settings=settings,
            repo_root=repo_root,
            prepared_model_path=prepared_model_path,
            phase_reporter=phase_reporter,
        )
        if reused_artifact is None
        else _reuse_artifact_for_measurement(
            reused_artifact,
            phase_reporter=phase_reporter,
        )
    )
    validate_artifact_contract(cell, generated.path)
    runtime = _load_runtime(generated.path, generated.process_id)
    revision = settings.source_revision_override or source_revision(
        repo_root,
        require_clean=True,
    )
    runtime_identity = runtime_identity_payload(
        cell,
        runtime,
        generated.path,
        generated.process_id,
        expected_source_revision=revision,
    )
    runtime_identity_sha256 = _runtime_identity_digest(runtime_identity)
    points = (
        runtime_validation_points(runtime)
        if cell.measurement.model in {ModelKey.SCALAR_CONTACT, ModelKey.SCALAR_GRAVITY}
        else shared_validation_points(cell.process)
    )
    if cell.measurement.accuracy is Accuracy.LC and contract is None:
        from .runner import derive_selector_contract

        contract = derive_selector_contract(runtime, points)
    if phase_reporter is not None:
        phase_reporter.profiling_started()
    profiling_deadlines = tuple(
        deadline
        for deadline in (
            settings.worker_deadline_monotonic,
            (
                None
                if settings.profiling_time_limit_seconds is None
                else time.monotonic() + settings.profiling_time_limit_seconds
            ),
        )
        if deadline is not None
    )
    profile = profile_runtime(
        runtime,
        points,
        cell=cell,
        benchmark_config=_resolution_benchmark_config(generated.effective_config),
        selector_contract=contract,
        progress=settings.progress,
        profiling_deadline_monotonic=(
            min(profiling_deadlines) if profiling_deadlines else None
        ),
    )
    execution_timing = profile.pop("execution_timing")
    evaluator_total_timing = profile.pop("evaluator_total_timing", None)
    arena_profile_evidence = profile.pop("arena_profile_evidence")
    benchmark_evidence = profile.pop("benchmark_evidence")
    if phase_reporter is not None:
        phase_reporter.validation_started()
    validation: dict[str, object] = {
        "resolved_sum": profile.pop("resolved_sum_validation"),
        DIRECT_AGREEMENT_FIELD: [],
    }
    if cell.measurement.accuracy is Accuracy.LC:
        assert contract is not None
        validation[LC_COMMON_COMPONENT_FIELD] = evaluate_lc_common_component(
            runtime,
            points,
            cell=cell,
            contract=contract,
        )
    scalar = cell.measurement.model in {
        ModelKey.SCALAR_CONTACT,
        ModelKey.SCALAR_GRAVITY,
    }
    requires_high_precision = scalar or (
        baseline is None and cell.measurement.execution_mode is ExecutionMode.RECURRENCE
    )
    if requires_high_precision:
        high_precision = runtime.evaluate(
            points,
            precision=32,
            **_selector_kwargs(cell, contract),
        )
        if not high_precision:
            raise RunnerError("high-precision evaluation returned no values")
        high_precision_value = _real_nonnegative(high_precision[0])
        validation["high_precision"] = pointwise_validation(
            float(profile["matrix_element"]),
            high_precision_value,
        )
    if (
        baseline is None
        and not scalar
        and cell.measurement.execution_mode is not ExecutionMode.RECURRENCE
    ):
        raise RunnerError(
            "non-scalar measurement without a canonical baseline must use "
            "recurrence high-precision validation"
        )
    baseline_value = _baseline_matrix_element(baseline)
    if baseline_value is not None:
        validation["pointwise"] = pointwise_validation(
            float(profile["matrix_element"]),
            baseline_value,
            relative_tolerance=_pointwise_tolerance(cell),
        )
    statuses = {
        str(record.get("status"))
        for record in validation.values()
        if isinstance(record, Mapping)
    }
    validation["status"] = (
        ResultStatus.VALIDATION_FAILED.value
        if ResultStatus.VALIDATION_FAILED.value in statuses
        else ResultStatus.OK.value
    )
    status = str(profile["status"])
    if ResultStatus.VALIDATION_FAILED.value in statuses:
        status = ResultStatus.VALIDATION_FAILED.value
    runtime_identity_postflight = runtime_identity_payload(
        cell,
        runtime,
        generated.path,
        generated.process_id,
        expected_source_revision=revision,
    )
    _validate_runtime_identity_postflight(
        runtime_identity,
        runtime_identity_postflight,
    )
    runtime_identity_stable_sha256 = _runtime_identity_digest(
        _stable_runtime_identity(runtime_identity)
    )
    runtime_identity_postflight_stable_sha256 = _runtime_identity_digest(
        _stable_runtime_identity(runtime_identity_postflight)
    )
    postflight_origin_policy = runtime_identity_postflight.get(
        "loaded_module_origin_policy"
    )
    if not isinstance(postflight_origin_policy, Mapping):
        raise RunnerError(
            "report runtime postflight has no loaded-module origin evidence"
        )

    measurement = empty_measurement()
    measurement.update(profile)
    measurement.update(
        {
            "status": status,
            "generation_seconds": generated.generation_seconds,
            "artifact": {
                "path": str(generated.path),
                "process_id": generated.process_id,
                "policy": "reused" if reused_artifact is not None else "generated",
            },
            "selector_contract": None if contract is None else contract.as_dict(),
            "validation": validation,
            "resources": None,
            "provenance": {
                **provenance_payload(),
                "source_revision": revision,
                "requested_config": dict(generated.requested_config),
                "effective_config": dict(generated.effective_config),
                "model_preparation_seconds": generated.model_preparation_seconds,
                "model_preparation_reused": generated.model_preparation_reused,
                "generation_timer_excludes_model_preparation": True,
                "generation_command_path": generated.generation_command_path,
                **(
                    {
                        "numerical_relation_correctness": dict(
                            generated.numerical_relation_correctness
                        )
                    }
                    if generated.numerical_relation_correctness is not None
                    else {}
                ),
                **(
                    {
                        "numerical_relation_fallback": dict(
                            generated.numerical_relation_fallback
                        )
                    }
                    if generated.numerical_relation_fallback is not None
                    else {}
                ),
                "report_momenta": _reproduction_momenta(points),
                "runtime_profile": benchmark_evidence,
                "execution_timing": execution_timing,
                **(
                    {"evaluator_total_timing": evaluator_total_timing}
                    if evaluator_total_timing is not None
                    else {}
                ),
                "arena_profile_evidence": arena_profile_evidence,
                "runtime_identity": runtime_identity,
                "runtime_identity_sha256": runtime_identity_sha256,
                "runtime_identity_stable_sha256": (runtime_identity_stable_sha256),
                "runtime_identity_postflight_stable_sha256": (
                    runtime_identity_postflight_stable_sha256
                ),
                "runtime_identity_postflight_loaded_module_origin_policy": dict(
                    postflight_origin_policy
                ),
                "runtime_identity_postflight_match": True,
            },
            "failure": None,
        }
    )
    if status != ResultStatus.OK.value:
        measurement["failure"] = {
            "kind": "MeasurementValidationError",
            "message": "candidate or same-artifact numerical validation failed",
        }
    return measurement


def failure_measurement(
    status: ResultStatus,
    error: BaseException | str,
    *,
    resources: Mapping[str, object] | None = None,
) -> dict[str, object]:
    if status in {ResultStatus.NOT_AVAILABLE, ResultStatus.OK}:
        raise ValueError("failure measurement requires a terminal failure status")
    message = str(error)
    failure = empty_measurement()
    failure.update(
        {
            "status": status.value,
            "resources": None if resources is None else dict(resources),
            "failure": {
                "kind": (
                    type(error).__name__ if isinstance(error, BaseException) else None
                ),
                "message": message,
            },
        }
    )
    return failure


def load_measurement(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("measurement file must contain an object")
    return dict(payload)


__all__ = [
    "failure_measurement",
    "file_digest",
    "generated_artifact_from_measurement",
    "load_measurement",
    "measure_pyamplicol_cell",
    "shared_validation_points",
    "source_revision",
]
