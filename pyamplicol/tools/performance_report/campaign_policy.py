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

from tools.ci.memory_watchdog import (
    DARWIN_PHYSICAL_FOOTPRINT_LIMIT_REASON,
    DARWIN_PHYSICAL_FOOTPRINT_PROBE_REASON,
    MEMORY_PROBE_REASON,
    RSS_LIMIT_REASON,
)

from .cache import empty_measurement
from .models import Accuracy, CellSpec, ExecutionMode, ResultStatus, Workload
from .resources import PROCESS_TREE_MEMORY_METRIC_ABI
from .source_identity import SOURCE_IDENTITY_SCHEMA, ReportSourceIdentity

CAMPAIGN_POLICY_SCHEMA = "pyamplicol-report-campaign-policy-v1"
LEGACY_POLICY_CENSOR_ABI = "pyamplicol-report-policy-censor-v2"
POLICY_CENSOR_ABI = "pyamplicol-report-policy-censor-v3"
RESOURCE_FRONTIER_ABI = "pyamplicol-report-resource-frontier-v1"
GENERATION_PHASE_EVIDENCE_ABI = "pyamplicol-report-generation-phase-evidence-v1"
WORKER_PHASE_STATE_ABI = "pyamplicol-report-worker-phase-state-v1"
STRICT_POLICY_NAME = "strict-complete-v1"
MACBOOK_M3_POLICY_NAME = "macbook-m3-v1"
X86_EPYC_POLICY_NAME = "x86-epyc-v1"
MACBOOK_M3_PROFILE = "macbook_M3"
X86_EPYC_PROFILE = "x86_EPYC"
MACBOOK_M3_WORKERS = 1
MACBOOK_M3_CELL_CORES = 1
MACBOOK_M3_TARGET_RUNTIME_SECONDS = 5.0
MACBOOK_M3_MEMORY_LIMIT_BYTES = 30_000_000_000
X86_EPYC_LEGACY_WORKERS = 10
X86_EPYC_WORKERS = 25
X86_EPYC_CELL_CORES = 1
X86_EPYC_TARGET_RUNTIME_SECONDS = 5.0
X86_EPYC_GENERATION_LIMIT_SECONDS = 2.0 * 60.0 * 60.0
X86_EPYC_LEGACY_MEMORY_LIMIT_BYTES = 100_000_000_000
X86_EPYC_MEMORY_LIMIT_BYTES = 80_000_000_000
X86_EPYC_NATIVE_COMPILER_SLOTS = 4

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
_LEGACY_CENSOR_FIELDS = frozenset(
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
        "frontier",
    }
)
_CENSOR_FIELDS = _LEGACY_CENSOR_FIELDS | {
    "observed_guard_bytes",
    "memory_metric_abi",
    "memory_limit_reason",
}
_MEMORY_RESOURCE_FIELDS = frozenset(
    {
        "memory_metric_abi",
        "current_physical_footprint_bytes",
        "peak_physical_footprint_bytes",
        "current_guard_bytes",
        "peak_guard_bytes",
        "memory_limit_bytes",
        "memory_limit_reason",
        "memory_probe_reason",
    }
)
_MEMORY_LIMIT_REASONS = frozenset(
    {
        RSS_LIMIT_REASON,
        DARWIN_PHYSICAL_FOOTPRINT_LIMIT_REASON,
    }
)
_MEMORY_PROBE_REASONS = frozenset(
    {
        MEMORY_PROBE_REASON,
        DARWIN_PHYSICAL_FOOTPRINT_PROBE_REASON,
    }
)
_DEPENDENCY_FIELDS = frozenset({"cell_id", "status", "censor_sha256"})
_FRONTIER_FIELDS = frozenset({"abi", "lane", "root"})
_FRONTIER_LANE_FIELDS = frozenset(
    {
        "process_key",
        "execution_mode",
        "model",
        "accuracy",
        "backend",
        "jit_optimization_level",
        "workload",
        "variant",
    }
)
_FRONTIER_ROOT_FIELDS = frozenset(
    {
        "cell_id",
        "n_final",
        "kind",
        "status",
        "censor_sha256",
    }
)
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
    RESOURCE_FRONTIER = "resource_frontier"


class PolicyMeasurementState(StrEnum):
    SUCCESS = "success"
    GENERATION_LIMIT = "generation_limit"
    MEMORY_LIMIT = "memory_limit"
    DEPENDENCY = "dependency"
    RESOURCE_FRONTIER = "resource_frontier"


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
                if self.generation_limit_seconds is not None
                else "none"
            ),
            "require_symbolica_parallel": self.require_symbolica_parallel,
        }


STRICT_POLICY = CampaignPolicy(
    name=STRICT_POLICY_NAME,
    allow_terminal_censors=False,
)
MACBOOK_M3_POLICY = CampaignPolicy(
    name=MACBOOK_M3_POLICY_NAME,
    allow_terminal_censors=True,
    workers=MACBOOK_M3_WORKERS,
    cell_cores=MACBOOK_M3_CELL_CORES,
    target_runtime_seconds=MACBOOK_M3_TARGET_RUNTIME_SECONDS,
    memory_limit_bytes=MACBOOK_M3_MEMORY_LIMIT_BYTES,
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
    if name == MACBOOK_M3_POLICY_NAME:
        return MACBOOK_M3_POLICY
    if name == X86_EPYC_POLICY_NAME:
        return X86_EPYC_POLICY
    raise CampaignPolicyError(f"unsupported report campaign policy {name!r}")


def default_campaign_policy(profile: str) -> CampaignPolicy:
    if profile == MACBOOK_M3_PROFILE:
        return MACBOOK_M3_POLICY
    if profile == X86_EPYC_PROFILE:
        return X86_EPYC_POLICY
    return STRICT_POLICY


def validate_policy_profile(policy: CampaignPolicy, profile: str) -> None:
    if policy is MACBOOK_M3_POLICY and profile != MACBOOK_M3_PROFILE:
        raise CampaignPolicyError(
            f"{MACBOOK_M3_POLICY_NAME} is reserved for profile {MACBOOK_M3_PROFILE!r}"
        )
    if policy is X86_EPYC_POLICY and profile != X86_EPYC_PROFILE:
        raise CampaignPolicyError(
            f"{X86_EPYC_POLICY_NAME} is reserved for profile {X86_EPYC_PROFILE!r}"
        )
    if profile == X86_EPYC_PROFILE and policy is not X86_EPYC_POLICY:
        raise CampaignPolicyError(
            f"profile {X86_EPYC_PROFILE!r} requires {X86_EPYC_POLICY_NAME}"
        )
    if profile == MACBOOK_M3_PROFILE and policy is not MACBOOK_M3_POLICY:
        raise CampaignPolicyError(
            f"profile {MACBOOK_M3_PROFILE!r} requires {MACBOOK_M3_POLICY_NAME}"
        )


def policy_from_manifest(value: object, *, profile: str) -> CampaignPolicy:
    if not isinstance(value, Mapping):
        raise CampaignPolicyError("workspace campaign_policy must be an object")
    name = value.get("name")
    if not isinstance(name, str):
        raise CampaignPolicyError("workspace campaign_policy has no name")
    policy = campaign_policy(name)
    accepted_manifests = [policy.as_manifest()]
    if policy is X86_EPYC_POLICY:
        legacy_manifest = policy.as_manifest()
        legacy_manifest["workers"] = X86_EPYC_LEGACY_WORKERS
        legacy_manifest["memory_limit_bytes"] = X86_EPYC_LEGACY_MEMORY_LIMIT_BYTES
        accepted_manifests.append(legacy_manifest)
    if dict(value) not in accepted_manifests:
        raise CampaignPolicyError(
            "workspace campaign_policy differs from its canonical definition"
        )
    validate_policy_profile(policy, profile)
    return policy


def generation_limit_exempt(cell: CellSpec) -> bool:
    measurement = cell.measurement
    return measurement.execution_mode is ExecutionMode.AMPLICOL or (
        measurement.execution_mode in {ExecutionMode.COMPILED, ExecutionMode.RECURRENCE}
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
            policy.require_symbolica_parallel,
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
        raise CampaignPolicyError(f"dependency {cell_id!r} has no policy provenance")
    digest = provenance.get("policy_censor_sha256")
    if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
        raise CampaignPolicyError(f"dependency {cell_id!r} has no policy-censor digest")
    return {
        "cell_id": cell_id,
        "status": str(measurement.get("status")),
        "censor_sha256": digest,
    }


def resource_lane_identity(cell: CellSpec) -> dict[str, object]:
    """Return the canonical multiplicity-independent resource lane."""

    measurement = cell.measurement
    return {
        "process_key": cell.process_key,
        "execution_mode": measurement.execution_mode.value,
        "model": None if measurement.model is None else measurement.model.value,
        "accuracy": measurement.accuracy.value,
        "backend": measurement.backend,
        "jit_optimization_level": measurement.jit_optimization_level,
        "workload": cell.workload.value,
        "variant": cell.variant,
    }


def resource_frontier_reference(
    cell: CellSpec,
    source_cell: CellSpec,
    source_measurement: Mapping[str, object],
) -> dict[str, object]:
    """Bind a higher cell directly to its first lower hard-resource censor."""

    if source_cell.n_final >= cell.n_final or resource_lane_identity(
        source_cell
    ) != resource_lane_identity(cell):
        raise CampaignPolicyError(
            "resource-frontier source is not lower in the same lane"
        )
    provenance = source_measurement.get("provenance")
    censor = (
        provenance.get("policy_censor") if isinstance(provenance, Mapping) else None
    )
    digest = (
        provenance.get("policy_censor_sha256")
        if isinstance(provenance, Mapping)
        else None
    )
    if (
        not isinstance(censor, Mapping)
        or censor.get("kind")
        not in {
            PolicyCensorKind.GENERATION_LIMIT.value,
            PolicyCensorKind.MEMORY_LIMIT.value,
        }
        or source_measurement.get("status")
        not in {
            ResultStatus.TIMEOUT.value,
            ResultStatus.MEMORY_LIMIT.value,
        }
        or not isinstance(digest, str)
        or _SHA256_RE.fullmatch(digest) is None
    ):
        raise CampaignPolicyError(
            "resource-frontier source is not a direct hard-resource censor"
        )
    return {
        "abi": RESOURCE_FRONTIER_ABI,
        "lane": resource_lane_identity(cell),
        "root": {
            "cell_id": source_cell.cell_id,
            "n_final": source_cell.n_final,
            "kind": censor["kind"],
            "status": source_measurement["status"],
            "censor_sha256": digest,
        },
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
    observed_guard_bytes: int | None = None,
    memory_metric_abi: str | None = None,
    memory_limit_reason: str | None = None,
    phase_evidence: Mapping[str, object] | None = None,
    dependencies: Sequence[Mapping[str, object]] = (),
    frontier: Mapping[str, object] | None = None,
) -> dict[str, object]:
    if not policy.allow_terminal_censors:
        raise CampaignPolicyError(
            "terminal censors require an architecture campaign policy"
        )
    validate_policy_profile(policy, profile)
    if kind is PolicyCensorKind.GENERATION_LIMIT:
        status = ResultStatus.TIMEOUT
    elif kind is PolicyCensorKind.MEMORY_LIMIT:
        status = ResultStatus.MEMORY_LIMIT
    elif kind in {
        PolicyCensorKind.DEPENDENCY,
        PolicyCensorKind.RESOURCE_FRONTIER,
    }:
        status = ResultStatus.SKIP
    else:  # pragma: no cover - exhaustive StrEnum defense
        raise CampaignPolicyError(f"unsupported policy censor kind {kind!r}")
    legacy_rss_memory_censor = (
        kind is PolicyCensorKind.MEMORY_LIMIT
        and observed_guard_bytes is None
        and memory_metric_abi is None
        and memory_limit_reason is None
    )
    record: dict[str, object] = {
        "abi": (
            LEGACY_POLICY_CENSOR_ABI if legacy_rss_memory_censor else POLICY_CENSOR_ABI
        ),
        "policy": policy.name,
        "profile": profile,
        "cell_id": cell.cell_id,
        "kind": kind.value,
        "generation_limit_seconds": policy.generation_limit_seconds,
        "memory_limit_bytes": policy.memory_limit_bytes,
        "observed_generation_seconds": observed_generation_seconds,
        "observed_rss_bytes": observed_rss_bytes,
        "phase_evidence": (None if phase_evidence is None else dict(phase_evidence)),
        "dependencies": [dict(item) for item in dependencies],
        "frontier": None if frontier is None else dict(frontier),
    }
    if not legacy_rss_memory_censor:
        record.update(
            {
                "observed_guard_bytes": observed_guard_bytes,
                "memory_metric_abi": memory_metric_abi,
                "memory_limit_reason": memory_limit_reason,
            }
        )
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
                        "worker process tree exceeded "
                        f"{int(policy.memory_limit_bytes or 0) / 1_000_000_000:g} "
                        "GB under its authenticated memory guard"
                        if kind is PolicyCensorKind.MEMORY_LIMIT
                        else (
                            "higher multiplicity omitted after a lower-multiplicity "
                            "resource ceiling"
                            if kind is PolicyCensorKind.RESOURCE_FRONTIER
                            else (
                                "required numerical-agreement dependency was censored"
                            )
                        )
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
        or provenance.get("report_measured_source_revision") != expected_source_revision
        or provenance.get("report_source_clean") is not True
    ):
        raise CampaignPolicyError("terminal measurement source identity does not match")
    if expected_source_tree is not None and (
        provenance.get("report_source_tree") != expected_source_tree
        or provenance.get("report_measured_source_tree") != expected_source_tree
    ):
        raise CampaignPolicyError("terminal measurement source tree does not match")
    raw = provenance.get("policy_censor")
    if not isinstance(raw, Mapping):
        raise CampaignPolicyError("policy_censor fields do not match the ABI")
    censor_abi = raw.get("abi")
    expected_fields = (
        _LEGACY_CENSOR_FIELDS
        if censor_abi == LEGACY_POLICY_CENSOR_ABI
        else _CENSOR_FIELDS
    )
    if set(raw) != expected_fields:
        raise CampaignPolicyError("policy_censor fields do not match the ABI")
    digest = provenance.get("policy_censor_sha256")
    if digest != _canonical_digest(raw):
        raise CampaignPolicyError("policy_censor_sha256 does not match")
    try:
        kind = PolicyCensorKind(str(raw.get("kind")))
    except ValueError as error:
        raise CampaignPolicyError("policy_censor kind is unsupported") from error
    if (
        censor_abi not in {LEGACY_POLICY_CENSOR_ABI, POLICY_CENSOR_ABI}
        or raw.get("policy") != policy.name
        or raw.get("profile") != profile
        or raw.get("cell_id") != cell.cell_id
        or raw.get("generation_limit_seconds") != policy.generation_limit_seconds
        or raw.get("memory_limit_bytes") != policy.memory_limit_bytes
    ):
        raise CampaignPolicyError("policy_censor identity or limits differ")
    return kind, raw


def _optional_nonnegative_integer(value: object, field: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CampaignPolicyError(f"{field} is not a non-negative integer")
    return value


def _validate_memory_metric(
    resources: Mapping[str, object],
    policy: CampaignPolicy,
) -> None:
    metric_abi = resources.get("memory_metric_abi")
    present_fields = _MEMORY_RESOURCE_FIELDS.intersection(resources)
    if metric_abi is None:
        if present_fields:
            raise CampaignPolicyError(
                "memory metric fields require an authenticated metric ABI"
            )
        return
    if (
        metric_abi != PROCESS_TREE_MEMORY_METRIC_ABI
        or present_fields != _MEMORY_RESOURCE_FIELDS
    ):
        raise CampaignPolicyError("memory metric ABI or fields are unsupported")

    current_rss = _optional_nonnegative_integer(
        resources.get("current_rss_bytes"),
        "current_rss_bytes",
    )
    peak_rss = _optional_nonnegative_integer(
        resources.get("peak_rss_bytes"),
        "peak_rss_bytes",
    )
    current_physical = _optional_nonnegative_integer(
        resources.get("current_physical_footprint_bytes"),
        "current_physical_footprint_bytes",
    )
    peak_physical = _optional_nonnegative_integer(
        resources.get("peak_physical_footprint_bytes"),
        "peak_physical_footprint_bytes",
    )
    current_guard = _optional_nonnegative_integer(
        resources.get("current_guard_bytes"),
        "current_guard_bytes",
    )
    peak_guard = _optional_nonnegative_integer(
        resources.get("peak_guard_bytes"),
        "peak_guard_bytes",
    )
    if current_rss is None or peak_rss is None:
        raise CampaignPolicyError("memory metric has incomplete RSS evidence")
    expected_current_guard = max(
        current_rss,
        current_physical if current_physical is not None else current_rss,
    )
    expected_peak_guard = max(
        peak_rss,
        peak_physical if peak_physical is not None else peak_rss,
    )
    if current_guard != expected_current_guard or peak_guard != expected_peak_guard:
        raise CampaignPolicyError("memory guard does not match RSS/footprint maxima")

    memory_limit = resources.get("memory_limit_bytes")
    assert policy.memory_limit_bytes is not None
    if (
        isinstance(memory_limit, bool)
        or not isinstance(memory_limit, int)
        or memory_limit != policy.memory_limit_bytes
    ):
        raise CampaignPolicyError("memory metric limit differs from policy")
    reason = resources.get("memory_limit_reason")
    if reason is not None and reason not in _MEMORY_LIMIT_REASONS:
        raise CampaignPolicyError("memory metric reason is unsupported")
    probe_reason = resources.get("memory_probe_reason")
    if probe_reason is not None:
        if probe_reason not in _MEMORY_PROBE_REASONS:
            raise CampaignPolicyError("memory probe reason is unsupported")
        raise CampaignPolicyError(
            "policy measurement has incomplete memory observations"
        )
    if reason == DARWIN_PHYSICAL_FOOTPRINT_LIMIT_REASON and (
        current_physical is None
        or current_physical < current_rss
        or current_guard <= memory_limit
    ):
        raise CampaignPolicyError(
            "physical-footprint limit reason contradicts current evidence"
        )
    if reason == RSS_LIMIT_REASON and (
        current_rss
        < (current_physical if current_physical is not None else current_rss)
        or current_guard <= memory_limit
    ):
        raise CampaignPolicyError("RSS limit reason contradicts current evidence")


def _validate_resources(
    measurement: Mapping[str, object],
    policy: CampaignPolicy,
    *,
    allow_pinned_orphan_unavailable_resources: bool = False,
) -> Mapping[str, object]:
    resources = measurement.get("resources")
    if not isinstance(resources, Mapping):
        raise CampaignPolicyError(
            "architecture-policy measurement has no resource evidence"
        )
    if allow_pinned_orphan_unavailable_resources and dict(resources) == {
        "monitor": "external-cell-supervisor",
        "peak_rss_gib": None,
    }:
        if policy.memory_limit_bytes is None:
            raise CampaignPolicyError("architecture policy has no memory ceiling")
        return resources
    peak = resources.get("peak_rss_bytes")
    if (
        resources.get("available") is not True
        or resources.get("probe_error") is not None
        or isinstance(peak, bool)
        or not isinstance(peak, int)
        or peak < 0
    ):
        raise CampaignPolicyError(
            "architecture-policy resource monitoring is unavailable or incomplete"
        )
    if policy.memory_limit_bytes is None:
        raise CampaignPolicyError("architecture policy has no memory ceiling")
    _validate_memory_metric(resources, policy)
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
        or value.get("configured_timeout_seconds") != policy.generation_limit_seconds
        or value.get("supervisor_reason") != expected_reason
        or value.get("authenticated") is not True
        or not isinstance(run_id, str)
        or not run_id
        or isinstance(worker_pid, bool)
        or not isinstance(worker_pid, int)
        or worker_pid <= 0
        or isinstance(sequence, bool)
        or not isinstance(sequence, int)
        or sequence != (1 if expected_reason == "generation_timeout" else 2)
        or value.get("final_phase")
        != (
            "generation"
            if expected_reason == "generation_timeout"
            else "post-generation"
        )
        or isinstance(started, bool)
        or not isinstance(started, int)
        or started <= 0
        or (expected_reason == "generation_timeout" and finished is not None)
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
    allow_pinned_orphan_unavailable_resources: bool = False,
) -> PolicyMeasurementState:
    """Validate one successful or policy-terminal current measurement."""

    validate_policy_profile(policy, profile)
    status = str(measurement.get("status"))
    if status == ResultStatus.OK.value:
        provenance = measurement.get("provenance")
        if not isinstance(provenance, Mapping):
            raise CampaignPolicyError("successful measurement has no provenance")
        if (
            provenance.get("report_source_identity_schema") != SOURCE_IDENTITY_SCHEMA
            or provenance.get("report_source_revision") != expected_source_revision
            or provenance.get("report_measured_source_revision")
            != expected_source_revision
            or provenance.get("report_source_clean") is not True
        ):
            raise CampaignPolicyError(
                "successful measurement source identity does not match"
            )
        if expected_source_tree is not None and (
            provenance.get("report_source_tree") != expected_source_tree
            or provenance.get("report_measured_source_tree") != expected_source_tree
        ):
            raise CampaignPolicyError(
                "successful measurement source tree does not match"
            )
        if policy.allow_terminal_censors:
            resources = _validate_resources(
                measurement,
                policy,
                allow_pinned_orphan_unavailable_resources=(
                    allow_pinned_orphan_unavailable_resources
                ),
            )
            peak = (
                resources.get("peak_guard_bytes")
                if resources.get("memory_metric_abi") is not None
                else resources.get("peak_rss_bytes")
            )
            if peak is not None:
                assert isinstance(peak, int) and not isinstance(peak, bool)
                assert policy.memory_limit_bytes is not None
                if peak > policy.memory_limit_bytes:
                    raise CampaignPolicyError(
                        "successful architecture measurement exceeded its "
                        "memory ceiling"
                    )
            generation = _finite_number(
                measurement.get("generation_seconds"),
                "generation_seconds",
            )
            limit = generation_limit_for_cell(policy, cell)
            if limit is not None and generation > limit:
                raise CampaignPolicyError(
                    "successful architecture measurement exceeded its "
                    "generation ceiling"
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
    frontier = record.get("frontier")

    if kind is PolicyCensorKind.GENERATION_LIMIT:
        if (
            policy.generation_limit_seconds is None
            or status != ResultStatus.TIMEOUT.value
            or generation_limit_exempt(cell)
        ):
            raise CampaignPolicyError(
                "generation censor status or exemption is invalid"
            )
        observed = _finite_number(
            record.get("observed_generation_seconds"),
            "observed_generation_seconds",
        )
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
        if (
            record.get("observed_rss_bytes") is not None
            or record.get("observed_guard_bytes") is not None
            or record.get("memory_limit_reason") is not None
            or record.get("memory_metric_abi") is not None
            or dependencies
            or frontier is not None
        ):
            raise CampaignPolicyError("generation censor fields are inconsistent")
        return PolicyMeasurementState.GENERATION_LIMIT

    if kind is PolicyCensorKind.MEMORY_LIMIT:
        if status != ResultStatus.MEMORY_LIMIT.value:
            raise CampaignPolicyError("memory censor status is invalid")
        resources = _validate_resources(measurement, policy)
        assert policy.memory_limit_bytes is not None
        legacy_censor = record.get("abi") == LEGACY_POLICY_CENSOR_ABI
        observed_rss = record.get("observed_rss_bytes")
        common_invalid = (
            record.get("observed_generation_seconds") is not None
            or record.get("phase_evidence") is not None
            or dependencies
            or frontier is not None
        )
        if legacy_censor:
            invalid = (
                isinstance(observed_rss, bool)
                or not isinstance(observed_rss, int)
                or observed_rss <= policy.memory_limit_bytes
                or observed_rss != resources.get("peak_rss_bytes")
                or common_invalid
            )
        else:
            observed_guard = record.get("observed_guard_bytes")
            invalid = (
                isinstance(observed_rss, bool)
                or not isinstance(observed_rss, int)
                or observed_rss != resources.get("peak_rss_bytes")
                or isinstance(observed_guard, bool)
                or not isinstance(observed_guard, int)
                or observed_guard <= policy.memory_limit_bytes
                or observed_guard != resources.get("peak_guard_bytes")
                or record.get("memory_metric_abi") != PROCESS_TREE_MEMORY_METRIC_ABI
                or resources.get("memory_metric_abi") != PROCESS_TREE_MEMORY_METRIC_ABI
                or record.get("memory_limit_reason")
                != resources.get("memory_limit_reason")
                or record.get("memory_limit_reason") not in _MEMORY_LIMIT_REASONS
                or common_invalid
            )
        if invalid:
            raise CampaignPolicyError("memory censor evidence is inconsistent")
        return PolicyMeasurementState.MEMORY_LIMIT

    if status != ResultStatus.SKIP.value:
        raise CampaignPolicyError("derived censor status is invalid")
    if (
        record.get("observed_generation_seconds") is not None
        or record.get("observed_rss_bytes") is not None
        or record.get("observed_guard_bytes") is not None
        or record.get("memory_limit_reason") is not None
        or record.get("memory_metric_abi") is not None
        or record.get("phase_evidence") is not None
        or measurement.get("resources") is not None
    ):
        raise CampaignPolicyError("derived censor fields are inconsistent")
    if kind is PolicyCensorKind.RESOURCE_FRONTIER:
        if dependencies or not isinstance(frontier, Mapping):
            raise CampaignPolicyError(
                "resource-frontier censor fields are inconsistent"
            )
        if set(frontier) != _FRONTIER_FIELDS:
            raise CampaignPolicyError("resource-frontier reference is malformed")
        lane = frontier.get("lane")
        root = frontier.get("root")
        if (
            frontier.get("abi") != RESOURCE_FRONTIER_ABI
            or not isinstance(lane, Mapping)
            or set(lane) != _FRONTIER_LANE_FIELDS
            or dict(lane) != resource_lane_identity(cell)
            or not isinstance(root, Mapping)
            or set(root) != _FRONTIER_ROOT_FIELDS
        ):
            raise CampaignPolicyError("resource-frontier identity is invalid")
        cell_id = root.get("cell_id")
        digest = root.get("censor_sha256")
        root_n = root.get("n_final")
        root_kind = root.get("kind")
        root_status = root.get("status")
        if (
            not isinstance(cell_id, str)
            or not cell_id
            or isinstance(root_n, bool)
            or not isinstance(root_n, int)
            or root_n < 1
            or root_n >= cell.n_final
            or root_kind
            not in {
                PolicyCensorKind.GENERATION_LIMIT.value,
                PolicyCensorKind.MEMORY_LIMIT.value,
            }
            or root_status
            not in {
                ResultStatus.TIMEOUT.value,
                ResultStatus.MEMORY_LIMIT.value,
            }
            or not isinstance(digest, str)
            or _SHA256_RE.fullmatch(digest) is None
        ):
            raise CampaignPolicyError("resource-frontier reference is invalid")
        if (
            root_kind == PolicyCensorKind.GENERATION_LIMIT.value
            and root_status != ResultStatus.TIMEOUT.value
        ) or (
            root_kind == PolicyCensorKind.MEMORY_LIMIT.value
            and root_status != ResultStatus.MEMORY_LIMIT.value
        ):
            raise CampaignPolicyError("resource-frontier kind and status differ")
        return PolicyMeasurementState.RESOURCE_FRONTIER
    if frontier is not None or not dependencies:
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
        memory_limit = record.get("memory_limit_bytes")
        if (
            isinstance(memory_limit, bool)
            or not isinstance(memory_limit, int)
            or memory_limit <= 0
            or memory_limit % 1_000_000_000
        ):
            return None
        return f">{memory_limit // 1_000_000_000}GB"
    if kind in {
        PolicyCensorKind.DEPENDENCY.value,
        PolicyCensorKind.RESOURCE_FRONTIER.value,
    }:
        if kind == PolicyCensorKind.RESOURCE_FRONTIER.value:
            raw_frontier = record.get("frontier")
            root = (
                raw_frontier.get("root") if isinstance(raw_frontier, Mapping) else None
            )
            dependencies = [root] if isinstance(root, Mapping) else []
        else:
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
                    memory_limit = record.get("memory_limit_bytes")
                    if (
                        isinstance(memory_limit, int)
                        and not isinstance(memory_limit, bool)
                        and memory_limit > 0
                        and memory_limit % 1_000_000_000 == 0
                    ):
                        labels.add(f">{memory_limit // 1_000_000_000}GB")
                    else:
                        labels.add("blocked")
                else:
                    labels.add("blocked")
        detail = "/".join(sorted(labels)) if labels else "blocked"
        return f"dependency {detail}"
    return None


__all__ = [
    "CAMPAIGN_POLICY_SCHEMA",
    "GENERATION_PHASE_EVIDENCE_ABI",
    "LEGACY_POLICY_CENSOR_ABI",
    "MACBOOK_M3_CELL_CORES",
    "MACBOOK_M3_MEMORY_LIMIT_BYTES",
    "MACBOOK_M3_POLICY",
    "MACBOOK_M3_POLICY_NAME",
    "MACBOOK_M3_PROFILE",
    "MACBOOK_M3_TARGET_RUNTIME_SECONDS",
    "MACBOOK_M3_WORKERS",
    "POLICY_CENSOR_ABI",
    "RESOURCE_FRONTIER_ABI",
    "STRICT_POLICY",
    "STRICT_POLICY_NAME",
    "WORKER_PHASE_STATE_ABI",
    "X86_EPYC_CELL_CORES",
    "X86_EPYC_GENERATION_LIMIT_SECONDS",
    "X86_EPYC_LEGACY_MEMORY_LIMIT_BYTES",
    "X86_EPYC_LEGACY_WORKERS",
    "X86_EPYC_MEMORY_LIMIT_BYTES",
    "X86_EPYC_NATIVE_COMPILER_SLOTS",
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
    "resource_frontier_reference",
    "resource_lane_identity",
    "validate_campaign_settings",
    "validate_policy_measurement",
    "validate_policy_profile",
]
