# SPDX-License-Identifier: 0BSD
"""Relocation-safe identities for split wrapper/measurement workers."""

from __future__ import annotations

import re
from collections.abc import Mapping, MutableMapping

WORKER_HARNESS_ABI = "pyamplicol-report-worker-harness-v1"
WORKER_HARNESS_PROVENANCE_FIELD = "worker_harness"
POLICY_ENTRYPOINT = "docs/arxiv/result_tables.py"
LEGACY_ADAPTER = "tools/performance_report/legacy.py"

_SHA40_RE = re.compile(r"[0-9a-f]{40}")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_FIELDS = frozenset(
    {
        "abi",
        "study_contract_sha256",
        "policy_wrapper_revision",
        "policy_wrapper_tree",
        "policy_entrypoint",
        "policy_entrypoint_sha256",
        "legacy_adapter",
        "legacy_adapter_sha256",
        "measured_source_revision",
        "measured_source_tree",
    }
)


class WorkerHarnessError(ValueError):
    """A worker did not use the authenticated wrapper/measurement split."""


def worker_harness_identity(
    *,
    study_contract_sha256: str,
    policy_wrapper_revision: str,
    policy_wrapper_tree: str,
    policy_entrypoint_sha256: str,
    legacy_adapter_sha256: str,
    measured_source_revision: str,
    measured_source_tree: str,
) -> dict[str, object]:
    """Build a canonical identity without embedding checkout paths."""

    payload: dict[str, object] = {
        "abi": WORKER_HARNESS_ABI,
        "study_contract_sha256": study_contract_sha256,
        "policy_wrapper_revision": policy_wrapper_revision,
        "policy_wrapper_tree": policy_wrapper_tree,
        "policy_entrypoint": POLICY_ENTRYPOINT,
        "policy_entrypoint_sha256": policy_entrypoint_sha256,
        "legacy_adapter": LEGACY_ADAPTER,
        "legacy_adapter_sha256": legacy_adapter_sha256,
        "measured_source_revision": measured_source_revision,
        "measured_source_tree": measured_source_tree,
    }
    validate_worker_harness_identity(payload)
    return payload


def validate_worker_harness_identity(
    value: object,
    *,
    expected: Mapping[str, object] | None = None,
    expected_measured_revision: str | None = None,
    expected_measured_tree: str | None = None,
) -> Mapping[str, object]:
    """Validate one canonical wrapper/measurement identity."""

    if not isinstance(value, Mapping) or set(value) != _FIELDS:
        raise WorkerHarnessError("worker harness fields are not canonical")
    if (
        value.get("abi") != WORKER_HARNESS_ABI
        or value.get("policy_entrypoint") != POLICY_ENTRYPOINT
        or value.get("legacy_adapter") != LEGACY_ADAPTER
    ):
        raise WorkerHarnessError("worker harness ABI or relative paths differ")
    for field in (
        "policy_wrapper_revision",
        "policy_wrapper_tree",
        "measured_source_revision",
        "measured_source_tree",
    ):
        item = value.get(field)
        if not isinstance(item, str) or _SHA40_RE.fullmatch(item) is None:
            raise WorkerHarnessError(
                f"worker harness {field} is not a Git identity"
            )
    for field in (
        "study_contract_sha256",
        "policy_entrypoint_sha256",
        "legacy_adapter_sha256",
    ):
        item = value.get(field)
        if not isinstance(item, str) or _SHA256_RE.fullmatch(item) is None:
            raise WorkerHarnessError(
                f"worker harness {field} is not SHA-256"
            )
    if expected is not None and dict(value) != dict(expected):
        raise WorkerHarnessError(
            "worker harness differs from the authenticated study contract"
        )
    if (
        expected_measured_revision is not None
        and value.get("measured_source_revision")
        != expected_measured_revision
    ):
        raise WorkerHarnessError(
            "worker harness measured source revision differs"
        )
    if (
        expected_measured_tree is not None
        and value.get("measured_source_tree") != expected_measured_tree
    ):
        raise WorkerHarnessError("worker harness measured source tree differs")
    return value


def attach_worker_harness_identity(
    measurement: MutableMapping[str, object],
    identity: Mapping[str, object],
) -> None:
    """Attach an authenticated identity without overwriting another harness."""

    validated = dict(validate_worker_harness_identity(identity))
    provenance = measurement.get("provenance")
    if provenance is not None and not isinstance(provenance, Mapping):
        raise WorkerHarnessError("measurement provenance is invalid")
    existing = (
        provenance.get(WORKER_HARNESS_PROVENANCE_FIELD)
        if isinstance(provenance, Mapping)
        else None
    )
    if existing is not None and existing != validated:
        raise WorkerHarnessError(
            "measurement is already bound to another worker harness"
        )
    measurement["provenance"] = {
        **({} if provenance is None else dict(provenance)),
        WORKER_HARNESS_PROVENANCE_FIELD: validated,
    }


def require_worker_harness_identity(
    measurement: Mapping[str, object],
    *,
    expected: Mapping[str, object] | None = None,
    expected_measured_revision: str | None = None,
    expected_measured_tree: str | None = None,
) -> Mapping[str, object]:
    """Return the authenticated harness carried by a measurement."""

    provenance = measurement.get("provenance")
    if not isinstance(provenance, Mapping):
        raise WorkerHarnessError("measurement has no worker harness provenance")
    return validate_worker_harness_identity(
        provenance.get(WORKER_HARNESS_PROVENANCE_FIELD),
        expected=expected,
        expected_measured_revision=expected_measured_revision,
        expected_measured_tree=expected_measured_tree,
    )


__all__ = [
    "LEGACY_ADAPTER",
    "POLICY_ENTRYPOINT",
    "WORKER_HARNESS_ABI",
    "WORKER_HARNESS_PROVENANCE_FIELD",
    "WorkerHarnessError",
    "attach_worker_harness_identity",
    "require_worker_harness_identity",
    "validate_worker_harness_identity",
    "worker_harness_identity",
]
