# SPDX-License-Identifier: 0BSD
"""Profile-bound completion and resource policies for report campaigns."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum

from .cache import empty_measurement
from .models import Accuracy, CellSpec, ExecutionMode, ResultStatus, Workload
from .source_identity import SOURCE_IDENTITY_SCHEMA, ReportSourceIdentity

CAMPAIGN_POLICY_SCHEMA = "pyamplicol-report-campaign-policy-v1"
POLICY_CENSOR_ABI = "pyamplicol-report-policy-censor-v1"
GENERATION_PHASE_EVIDENCE_ABI = (
    "pyamplicol-report-generation-phase-evidence-v1"
)
WORKER_PHASE_STATE_ABI = "pyamplicol-report-worker-phase-state-v1"
STRICT_POLICY_NAME = "strict-complete-v1"
X86_EPYC_POLICY_NAME = "x86-epyc-v1"
X86_EPYC_PROFILE = "x86_EPYC"
X86_EPYC_WORKERS = 10
X86_EPYC_CELL_CORES = 1
X86_EPYC_TARGET_RUNTIME_SECONDS = 5.0
X86_EPYC_GENERATION_LIMIT_SECONDS = 2.0 * 60.0 * 60.0
X86_EPYC_MEMORY_LIMIT_BYTES = 100_000_000_000

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SOURCE_FIELDS = frozenset(
    {
        "report_source_identity_schema",
        "report_source_revision",
        "report_source_tree",
        "report_measured_source_revision",
        "report_measured_source_tree",
        "report_source_clean",
    }
)
_CENSOR_FIELDS = frozenset(
    {
        "abi",
        "policy",
        "profile",
        "cell_id",
        "kind",
        "generation_limit_seconds",
        "memory_limit_bytes",
        "observed_generation_seconds",
        "observed_rss_bytes",
        "phase_evidence",
        "dependencies",
    }
)
_DEPENDENCY_FIELDS = frozenset({"cell_id", "status", "censor_sha256"})
_GENERATION_PHASE_FIELDS = frozenset(
    {
        "abi",
        "phase_state_abi",
        "configured_timeout_seconds",
        "supervisor_reason",
        "authenticated",
        "run_id",
        "worker_pid",
        "final_sequence",
        "final_phase",
        "generation_started_monotonic_ns",
        "generation_finished_monotonic_ns",
        "generation_elapsed_seconds",
        "final_state_sha256",
        "error",
    }
)


class CampaignPolicyError(ValueError):
    """A report campaign or terminal measurement violates its bound policy."""


class PolicyCensorKind(StrEnum):
    GENERATION_LIMIT = "generation_limit"
    MEMORY_LIMIT = "memory_limit"
    DEPENDENCY = "dependency"


class PolicyMeasurementState(StrEnum):
    SUCCESS = "success"
    GENERATION_LIMIT = "generation_limit"
    MEMORY_LIMIT = "memory_limit"
    DEPENDENCY = "dependency"


@dataclass(frozen=True, slots=True)
class CampaignPolicy:
    name: str
    allow_terminal_censors: bool
    workers: int | None = None
    cell_cores: int | None = None
    target_runtime_seconds: float = X86_EPYC_TARGET_RUNTIME_SECONDS
    generation_limit_seconds: float | None = None
    memory_limit_bytes: int | None = None
    require_symbolica_parallel: bool = False

    def as_manifest(self) -> dict[str, object]:
        return {
            "schema": CAMPAIGN_POLICY_SCHEMA,
            "name": self.name,
            "allow_terminal_censors": self.allow_terminal_censors,
            "workers": self.workers,
            "cell_cores": self.cell_cores,
            "target_runtime_seconds": self.target_runtime_seconds,
            "generation_limit_seconds": self.generation_limit_seconds,
            "memory_limit_bytes": self.memory_limit_bytes,
            "generation_limit_exemptions": (
                "all original-AmpliCol cells and pyAmpliCol compiled/recurrence "
                "LC selected-flow cells"
                if self.allow_terminal_censors
                else "none"
            ),
            "require_symbolica_parallel": self.require_symbolica_parallel,
        }


STRICT_POLICY = CampaignPolicy(
    name=STRICT_POLICY_NAME,
    allow_terminal_censors=False,
)
X86_EPYC_POLICY = CampaignPolicy(
    name=X86_EPYC_POLICY_NAME,
    allow_terminal_censors=True,
    workers=X86_EPYC_WORKERS,
    cell_cores=X86_EPYC_CELL_CORES,
    target_runtime_seconds=X86_EPYC_TARGET_RUNTIME_SECONDS,
    generation_limit_seconds=X86_EPYC_GENERATION_LIMIT_SECONDS,
    memory_limit_bytes=X86_EPYC_MEMORY_LIMIT_BYTES,
    require_symbolica_parallel=True,
)


def campaign_policy(name: str) -> CampaignPolicy:
    if name == STRICT_POLICY_NAME:
        return STRICT_POLICY
    if name == X86_EPYC_POLICY_NAME:
        return X86_EPYC_POLICY
    raise CampaignPolicyError(f"unsupported report campaign policy {name!r}")


def default_campaign_policy(profile: str) -> CampaignPolicy:
    return X86_EPYC_POLICY if profile == X86_EPYC_PROFILE else STRICT_POLICY


def validate_policy_profile(policy: CampaignPolicy, profile: str) -> None:
    if policy is X86_EPYC_POLICY and profile != X86_EPYC_PROFILE:
        raise CampaignPolicyError(
            f"{X86_EPYC_POLICY_NAME} is reserved for profile "
            f"{X86_EPYC_PROFILE!r}"
        )
    if profile == X86_EPYC_PROFILE and policy is not X86_EPYC_POLICY:
        raise CampaignPolicyError(
            f"profile {X86_EPYC_PROFILE!r} requires {X86_EPYC_POLICY_NAME}"
        )


def policy_from_manifest(value: object, *, profile: str) -> CampaignPolicy:
    if not isinstance(value, Mapping):
        raise CampaignPolicyError("workspace campaign_policy must be an object")
    name = value.get("name")
    if not isinstance(name, str):
        raise CampaignPolicyError("workspace campaign_policy has no name")
    policy = campaign_policy(name)
    if dict(value) != policy.as_manifest():
        raise CampaignPolicyError(
            "workspace campaign_policy differs from its canonical definition"
        )
    validate_policy_profile(policy, profile)
    return policy


def generation_limit_exempt(cell: CellSpec) -> bool:
    measurement = cell.measurement
    return measurement.execution_mode is ExecutionMode.AMPLICOL or (
        measurement.execution_mode
        in {ExecutionMode.COMPILED, ExecutionMode.RECURRENCE}
        and measurement.accuracy is Accuracy.LC
        and cell.workload is Workload.SELECTED_FLOW
    )


def generation_limit_for_cell(
    policy: CampaignPolicy,
    cell: CellSpec,
) -> float | None:
    if policy.generation_limit_seconds is None or generation_limit_exempt(cell):
        return None
    return policy.generation_limit_seconds


def validate_campaign_settings(policy: CampaignPolicy, settings: object) -> None:
    """Reject settings that weaken or misstate an architecture policy."""

    if policy is STRICT_POLICY:
        return
    checks = (
        (getattr(settings, "workers", None), policy.workers, "workers"),
        (getattr(settings, "cell_cores", None), policy.cell_cores, "cell_cores"),
        (
            getattr(settings, "target_runtime_seconds", None),
            policy.target_runtime_seconds,
            "target_runtime_seconds",
        ),
        (
            getattr(settings, "max_rss_bytes", None),
            policy.memory_limit_bytes,
            "max_rss_bytes",
        ),
        (
            getattr(settings, "generation_time_limit_seconds", None),
            None,
            "generation_time_limit_seconds",
        ),
        (getattr(settings, "timeout_seconds", None), None, "timeout_seconds"),
        (
            getattr(settings, "campaign_max_rss_bytes", None),
            None,
            "campaign_max_rss_bytes",
        ),
        (
            getattr(settings, "allow_symbolica_parallel", None),
            True,
            "allow_symbolica_parallel",
        ),
    )
    mismatches = [
        f"{field}={observed!r}, expected {expected!r}"
        for observed, expected, field in checks
        if observed != expected
    ]
    if mismatches:
        raise CampaignPolicyError(
            f"{policy.name} settings differ: " + "; ".join(mismatches)
        )


def _canonical_digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    ).hexdigest()


def _finite_number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CampaignPolicyError(f"{field} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise CampaignPolicyError(f"{field} must be finite")
    return number


def _source_provenance(
    source_identity: ReportSourceIdentity | Mapping[str, object],
) -> dict[str, object]:
    provenance = (
        source_identity.provenance()
        if isinstance(source_identity, ReportSourceIdentity)
        else dict(source_identity)
    )
    if set(provenance) != _SOURCE_FIELDS:
        raise CampaignPolicyError("policy censor source identity is malformed")
    return provenance


def dependency_reference(
    cell_id: str,
    measurement: Mapping[str, object],
) -> dict[str, object]:
    provenance = measurement.get("provenance")
    if not isinstance(provenance, Mapping):
        raise CampaignPolicyError(
            f"dependency {cell_id!r} has no policy provenance"
        )
    digest = provenance.get("policy_censor_sha256")
    if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
        raise CampaignPolicyError(
            f"dependency {cell_id!r} has no policy-censor digest"
        )
    return {
        "cell_id": cell_id,
        "status": str(measurement.get("status")),
        "censor_sha256": digest,
    }


def policy_censor_measurement(
    policy: CampaignPolicy,
    profile: str,
    cell: CellSpec,
    *,
    kind: PolicyCensorKind,
    source_identity: ReportSourceIdentity | Mapping[str, object],
    resources: Mapping[str, object] | None,
    observed_generation_seconds: float | None = None,
    observed_rss_bytes: int | None = None,
    phase_evidence: Mapping[str, object] | None = None,
    dependencies: Sequence[Mapping[str, object]] = (),
) -> dict[str, object]:
    if policy is not X86_EPYC_POLICY or not policy.allow_terminal_censors:
        raise CampaignPolicyError("terminal censors require the x86 EPYC policy")
    validate_policy_profile(policy, profile)
    if kind is PolicyCensorKind.GENERATION_LIMIT:
        status = ResultStatus.TIMEOUT
    elif kind is PolicyCensorKind.MEMORY_LIMIT:
        status = ResultStatus.MEMORY_LIMIT
    elif kind is PolicyCensorKind.DEPENDENCY:
        status = ResultStatus.SKIP
    else:  # pragma: no cover - exhaustive StrEnum defense
        raise CampaignPolicyError(f"unsupported policy censor kind {kind!r}")
    record = {
        "abi": POLICY_CENSOR_ABI,
        "policy": policy.name,
        "profile": profile,
        "cell_id": cell.cell_id,
        "kind": kind.value,
        "generation_limit_seconds": policy.generation_limit_seconds,
        "memory_limit_bytes": policy.memory_limit_bytes,
        "observed_generation_seconds": observed_generation_seconds,
        "observed_rss_bytes": observed_rss_bytes,
        "phase_evidence": (
            None if phase_evidence is None else dict(phase_evidence)
        ),
        "dependencies": [dict(item) for item in dependencies],
    }
    provenance = {
        **_source_provenance(source_identity),
        "policy_censor": record,
        "policy_censor_sha256": _canonical_digest(record),
    }
    result = empty_measurement()
    result.update(
        {
            "status": status.value,
            "resources": None if resources is None else dict(resources),
            "provenance": provenance,
            "failure": {
                "kind": "ReportPolicyCensor",
                "message": (
                    "process generation exceeded two hours"
                    if kind is PolicyCensorKind.GENERATION_LIMIT
                    else (
                        "worker process tree exceeded 100 GB RSS"
                        if kind is PolicyCensorKind.MEMORY_LIMIT
                        else "required numerical-agreement dependency was censored"
                    )
                ),
            },
        }
    )
    validate_policy_measurement(
        policy,
        profile,
        cell,
        result,
        expected_source_revision=str(provenance["report_source_revision"]),
        expected_source_tree=str(provenance["report_source_tree"]),
    )
    return result


def _policy_record(
    policy: CampaignPolicy,
    profile: str,
    cell: CellSpec,
    measurement: Mapping[str, object],
    *,
    expected_source_revision: str,
    expected_source_tree: str | None,
) -> tuple[PolicyCensorKind, Mapping[str, object]]:
    provenance = measurement.get("provenance")
    if not isinstance(provenance, Mapping):
        raise CampaignPolicyError("terminal measurement has no provenance")
    if set(provenance) != _SOURCE_FIELDS | {
        "policy_censor",
        "policy_censor_sha256",
    }:
        raise CampaignPolicyError("terminal measurement provenance is not canonical")
    if (
        provenance.get("report_source_identity_schema") != SOURCE_IDENTITY_SCHEMA
        or provenance.get("report_source_revision") != expected_source_revision
        or provenance.get("report_measured_source_revision")
        != expected_source_revision
        or provenance.get("report_source_clean") is not True
    ):
        raise CampaignPolicyError(
            "terminal measurement source identity does not match"
        )
    if expected_source_tree is not None and (
        provenance.get("report_source_tree") != expected_source_tree
        or provenance.get("report_measured_source_tree") != expected_source_tree
    ):
        raise CampaignPolicyError("terminal measurement source tree does not match")
    raw = provenance.get("policy_censor")
    if not isinstance(raw, Mapping) or set(raw) != _CENSOR_FIELDS:
        raise CampaignPolicyError("policy_censor fields do not match the ABI")
    digest = provenance.get("policy_censor_sha256")
    if digest != _canonical_digest(raw):
        raise CampaignPolicyError("policy_censor_sha256 does not match")
    try:
        kind = PolicyCensorKind(str(raw.get("kind")))
    except ValueError as error:
        raise CampaignPolicyError("policy_censor kind is unsupported") from error
    if (
        raw.get("abi") != POLICY_CENSOR_ABI
        or raw.get("policy") != policy.name
        or raw.get("profile") != profile
        or raw.get("cell_id") != cell.cell_id
        or raw.get("generation_limit_seconds")
        != policy.generation_limit_seconds
        or raw.get("memory_limit_bytes") != policy.memory_limit_bytes
    ):
        raise CampaignPolicyError("policy_censor identity or limits differ")
    return kind, raw


def _validate_resources(
    measurement: Mapping[str, object],
    policy: CampaignPolicy,
) -> Mapping[str, object]:
    resources = measurement.get("resources")
    if not isinstance(resources, Mapping):
        raise CampaignPolicyError("x86 EPYC measurement has no resource evidence")
    peak = resources.get("peak_rss_bytes")
    if (
        resources.get("available") is not True
        or resources.get("probe_error") is not None
        or isinstance(peak, bool)
        or not isinstance(peak, int)
        or peak < 0
    ):
        raise CampaignPolicyError(
            "x86 EPYC resource monitoring is unavailable or incomplete"
        )
    assert policy.memory_limit_bytes is not None
    return resources


def _validate_generation_phase(
    value: object,
    policy: CampaignPolicy,
    *,
    expected_reason: str,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != _GENERATION_PHASE_FIELDS:
        raise CampaignPolicyError(
            "generation-phase evidence fields do not match the ABI"
        )
    assert policy.generation_limit_seconds is not None
    run_id = value.get("run_id")
    worker_pid = value.get("worker_pid")
    sequence = value.get("final_sequence")
    started = value.get("generation_started_monotonic_ns")
    finished = value.get("generation_finished_monotonic_ns")
    elapsed = _finite_number(
        value.get("generation_elapsed_seconds"),
        "generation_phase.generation_elapsed_seconds",
    )
    state_digest = value.get("final_state_sha256")
    if (
        value.get("abi") != GENERATION_PHASE_EVIDENCE_ABI
        or value.get("phase_state_abi") != WORKER_PHASE_STATE_ABI
        or value.get("configured_timeout_seconds")
        != policy.generation_limit_seconds
        or value.get("supervisor_reason") != expected_reason
        or value.get("authenticated") is not True
        or not isinstance(run_id, str)
        or not run_id
        or isinstance(worker_pid, bool)
        or not isinstance(worker_pid, int)
        or worker_pid <= 0
        or isinstance(sequence, bool)
        or not isinstance(sequence, int)
        or sequence
        != (1 if expected_reason == "generation_timeout" else 2)
        or value.get("final_phase")
        != (
            "generation"
            if expected_reason == "generation_timeout"
            else "post-generation"
        )
        or isinstance(started, bool)
        or not isinstance(started, int)
        or started <= 0
        or (
            expected_reason == "generation_timeout"
            and finished is not None
        )
        or (
            expected_reason == "completed"
            and (
                isinstance(finished, bool)
                or not isinstance(finished, int)
                or finished < started
            )
        )
        or not isinstance(state_digest, str)
        or _SHA256_RE.fullmatch(state_digest) is None
        or value.get("error") is not None
    ):
        raise CampaignPolicyError(
            "generation-phase evidence is unauthenticated or inconsistent"
        )
    if expected_reason == "generation_timeout":
        if elapsed < policy.generation_limit_seconds:
            raise CampaignPolicyError(
                "generation-phase evidence did not reach its ceiling"
            )
    elif elapsed > policy.generation_limit_seconds:
        raise CampaignPolicyError(
            "successful generation-phase evidence exceeded its ceiling"
        )
    return value


def validate_policy_measurement(
    policy: CampaignPolicy,
    profile: str,
    cell: CellSpec,
    measurement: Mapping[str, object],
    *,
    expected_source_revision: str,
    expected_source_tree: str | None = None,
) -> PolicyMeasurementState:
    """Validate one successful or policy-terminal current measurement."""

    validate_policy_profile(policy, profile)
    status = str(measurement.get("status"))
    if status == ResultStatus.OK.value:
        provenance = measurement.get("provenance")
        if not isinstance(provenance, Mapping):
            raise CampaignPolicyError("successful measurement has no provenance")
        if (
            provenance.get("report_source_identity_schema")
            != SOURCE_IDENTITY_SCHEMA
            or provenance.get("report_source_revision")
            != expected_source_revision
            or provenance.get("report_measured_source_revision")
            != expected_source_revision
            or provenance.get("report_source_clean") is not True
        ):
            raise CampaignPolicyError(
                "successful measurement source identity does not match"
            )
        if expected_source_tree is not None and (
            provenance.get("report_source_tree") != expected_source_tree
            or provenance.get("report_measured_source_tree")
            != expected_source_tree
        ):
            raise CampaignPolicyError(
                "successful measurement source tree does not match"
            )
        if policy is X86_EPYC_POLICY:
            resources = _validate_resources(measurement, policy)
            peak = int(resources["peak_rss_bytes"])
            assert policy.memory_limit_bytes is not None
            if peak > policy.memory_limit_bytes:
                raise CampaignPolicyError(
                    "successful x86 EPYC measurement exceeded its RSS ceiling"
                )
            generation = _finite_number(
                measurement.get("generation_seconds"),
                "generation_seconds",
            )
            limit = generation_limit_for_cell(policy, cell)
            if limit is not None and generation > limit:
                raise CampaignPolicyError(
                    "successful x86 EPYC measurement exceeded its generation ceiling"
                )
            if limit is not None:
                _validate_generation_phase(
                    resources.get("generation_phase"),
                    policy,
                    expected_reason="completed",
                )
        return PolicyMeasurementState.SUCCESS

    if policy is STRICT_POLICY:
        raise CampaignPolicyError(
            "strict report profiles require a successful measurement"
        )
    kind, record = _policy_record(
        policy,
        profile,
        cell,
        measurement,
        expected_source_revision=expected_source_revision,
        expected_source_tree=expected_source_tree,
    )
    failure = measurement.get("failure")
    if (
        not isinstance(failure, Mapping)
        or failure.get("kind") != "ReportPolicyCensor"
        or not isinstance(failure.get("message"), str)
    ):
        raise CampaignPolicyError("terminal measurement failure is not canonical")
    for field in (
        "generation_seconds",
        "wall_seconds_per_point",
        "execution_seconds_per_point",
        "matrix_element",
        "sample_count",
        "standard_error_seconds_per_point",
        "relative_standard_error",
        "artifact",
        "selector_contract",
        "validation",
    ):
        if measurement.get(field) is not None:
            raise CampaignPolicyError(
                f"terminal measurement unexpectedly contains {field}"
            )
    dependencies = record.get("dependencies")
    if not isinstance(dependencies, list):
        raise CampaignPolicyError("policy_censor dependencies must be an array")

    if kind is PolicyCensorKind.GENERATION_LIMIT:
        if status != ResultStatus.TIMEOUT.value or generation_limit_exempt(cell):
            raise CampaignPolicyError(
                "generation censor status or exemption is invalid"
            )
        observed = _finite_number(
            record.get("observed_generation_seconds"),
            "observed_generation_seconds",
        )
        assert policy.generation_limit_seconds is not None
        if observed < policy.generation_limit_seconds:
            raise CampaignPolicyError(
                "generation censor did not reach the two-hour ceiling"
            )
        phase = record.get("phase_evidence")
        _validate_generation_phase(
            phase,
            policy,
            expected_reason="generation_timeout",
        )
        assert isinstance(phase, Mapping)
        if observed != float(phase["generation_elapsed_seconds"]):
            raise CampaignPolicyError(
                "generation censor observed time differs from phase evidence"
            )
        resources = _validate_resources(measurement, policy)
        if resources.get("generation_phase") != phase:
            raise CampaignPolicyError(
                "generation censor phase evidence differs from resources"
            )
        if record.get("observed_rss_bytes") is not None or dependencies:
            raise CampaignPolicyError("generation censor fields are inconsistent")
        return PolicyMeasurementState.GENERATION_LIMIT

    if kind is PolicyCensorKind.MEMORY_LIMIT:
        if status != ResultStatus.MEMORY_LIMIT.value:
            raise CampaignPolicyError("memory censor status is invalid")
        observed = record.get("observed_rss_bytes")
        resources = _validate_resources(measurement, policy)
        assert policy.memory_limit_bytes is not None
        if (
            isinstance(observed, bool)
            or not isinstance(observed, int)
            or observed <= policy.memory_limit_bytes
            or observed != resources.get("peak_rss_bytes")
            or record.get("observed_generation_seconds") is not None
            or dependencies
        ):
            raise CampaignPolicyError("memory censor evidence is inconsistent")
        return PolicyMeasurementState.MEMORY_LIMIT

    if status != ResultStatus.SKIP.value:
        raise CampaignPolicyError("dependency censor status is invalid")
    if (
        record.get("observed_generation_seconds") is not None
        or record.get("observed_rss_bytes") is not None
        or record.get("phase_evidence") is not None
        or measurement.get("resources") is not None
        or not dependencies
    ):
        raise CampaignPolicyError("dependency censor fields are inconsistent")
    seen: set[str] = set()
    for dependency in dependencies:
        if not isinstance(dependency, Mapping) or set(dependency) != _DEPENDENCY_FIELDS:
            raise CampaignPolicyError("dependency censor reference is malformed")
        cell_id = dependency.get("cell_id")
        digest = dependency.get("censor_sha256")
        dep_status = dependency.get("status")
        if (
            not isinstance(cell_id, str)
            or not cell_id
            or cell_id in seen
            or dep_status
            not in {
                ResultStatus.TIMEOUT.value,
                ResultStatus.MEMORY_LIMIT.value,
                ResultStatus.SKIP.value,
            }
            or not isinstance(digest, str)
            or _SHA256_RE.fullmatch(digest) is None
        ):
            raise CampaignPolicyError("dependency censor reference is invalid")
        seen.add(cell_id)
    return PolicyMeasurementState.DEPENDENCY


def policy_status_label(measurement: Mapping[str, object]) -> str | None:
    provenance = measurement.get("provenance")
    if not isinstance(provenance, Mapping):
        return None
    record = provenance.get("policy_censor")
    if not isinstance(record, Mapping):
        return None
    kind = record.get("kind")
    if kind == PolicyCensorKind.GENERATION_LIMIT.value:
        return ">2h"
    if kind == PolicyCensorKind.MEMORY_LIMIT.value:
        return ">100GB"
    if kind == PolicyCensorKind.DEPENDENCY.value:
        dependencies = record.get("dependencies")
        labels: set[str] = set()
        if isinstance(dependencies, list):
            for dependency in dependencies:
                if not isinstance(dependency, Mapping):
                    continue
                status = dependency.get("status")
                if status == ResultStatus.TIMEOUT.value:
                    labels.add(">2h")
                elif status == ResultStatus.MEMORY_LIMIT.value:
                    labels.add(">100GB")
                else:
                    labels.add("blocked")
        detail = "/".join(sorted(labels)) if labels else "blocked"
        return f"dependency {detail}"
    return None


__all__ = [
    "CAMPAIGN_POLICY_SCHEMA",
    "GENERATION_PHASE_EVIDENCE_ABI",
    "POLICY_CENSOR_ABI",
    "STRICT_POLICY",
    "STRICT_POLICY_NAME",
    "WORKER_PHASE_STATE_ABI",
    "X86_EPYC_CELL_CORES",
    "X86_EPYC_GENERATION_LIMIT_SECONDS",
    "X86_EPYC_MEMORY_LIMIT_BYTES",
    "X86_EPYC_POLICY",
    "X86_EPYC_POLICY_NAME",
    "X86_EPYC_PROFILE",
    "X86_EPYC_TARGET_RUNTIME_SECONDS",
    "X86_EPYC_WORKERS",
    "CampaignPolicy",
    "CampaignPolicyError",
    "PolicyCensorKind",
    "PolicyMeasurementState",
    "campaign_policy",
    "default_campaign_policy",
    "dependency_reference",
    "generation_limit_exempt",
    "generation_limit_for_cell",
    "policy_censor_measurement",
    "policy_from_manifest",
    "policy_status_label",
    "validate_campaign_settings",
    "validate_policy_measurement",
    "validate_policy_profile",
]
