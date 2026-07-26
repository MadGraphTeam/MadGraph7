#!/usr/bin/env python3
# SPDX-License-Identifier: 0BSD
"""Audit DirectTable microkernel evidence without importing production code.

The input is an authenticated, normalized campaign document.  This tool is
deliberately independent of the generator and loader that produce the
evidence: declared counts, byte totals, ordering, coverage, numerical results,
and timing gates are recomputed here.

The normalized contract is intentionally strict.  Unknown or missing fields,
compiled-stage-plan v1, incompatible direct ABIs, incomplete odd-tail data,
and malformed islands raise :class:`AcceptanceError`; a well-formed campaign
that misses a landing threshold returns ``passes: false``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

CAMPAIGN_KIND = "pyamplicol-compiled-microkernel-campaign"
CAMPAIGN_SCHEMA_VERSION = 1
RESULT_KIND = "pyamplicol-compiled-microkernel-acceptance"
RESULT_SCHEMA_VERSION = 1
STAGE_PLAN_KIND = "compiled-stage-plan"
STAGE_PLAN_SCHEMA_VERSION = 2

DIRECT_APPLICATION_ABI = "symjit-direct-application-storage-v1"
SOURCE_APPLICATION_ABI = "symjit-application-storage-v3"
DIRECT_TABLE_BINDING_ABI = "symjit-direct-table-binding-v1"
DIRECT_TABLE_DESCRIPTOR_ABI = "symjit-direct-table-descriptor-v1"
DEPENDENCY_REVISION = "89efdb806e7fcd9ac68a9d38f3f2880adf1987d2"
DEPENDENCY_ARCHIVE_SHA256 = (
    "070ff7fc04d5cdc5ab769d7a47b3da04cbc2b97d87136d303180c95b9eb380cd"
)
DEPENDENCY_SOURCE_TREE_SHA256 = (
    "e42d648d995c61881e560aefc50f80a995e86fb24a67ed9b0f0b5a80d6773fcf"
)
DEPENDENCY_CANDIDATE_TREE_SHA256 = (
    "820675246517cd49198495936327768da7a7a1d25f8bf20749c21aad1c2f56da"
)

TARGET_PROCESS = "u u~ > Z+6g"
TARGET_FLOW = "flow:2,4,5,6,7,8,9,1"
TARGET_LAYOUT = "topology-replay"
TARGET_COLOR_ACCURACY = "lc"
TARGET_HELICITY_MODE = "sum"

REQUIRED_BATCHES = (1, 127, 128, 129, 1023, 1024, 1025)
PRIMARY_BATCHES = (128, 1024)
MIN_SAMPLE_PAIRS = 7
MIN_SAMPLE_DURATION_SECONDS = 5.0
MIN_PRIMARY_GAIN = 0.10
MAX_BATCH_ONE_REGRESSION = 0.05
MAX_RESOURCE_REGRESSION = 0.10
MIN_CODE_SIZE_REDUCTION = 0.25
MAX_NON_TARGET_REGRESSION = 0.02
MAD_MULTIPLIER = 3.0
RELATIVE_TOLERANCE = 1.0e-12
ABSOLUTE_TOLERANCE = 1.0e-15

MAX_KERNEL_IDENTITIES = 8
MAX_COMPLEX_INPUTS = 16
MAX_COMPLEX_OUTPUTS = 2
MAX_KERNEL_SOURCE_BYTES = 64 * 1024
MAX_SEMANTIC_ROW_BYTES = 4 * 1024 * 1024
MIN_CENSUS_COVERAGE = 0.50
MAX_PROJECTED_TEXT_FRACTION = 0.25

CENSUS_DENOMINATOR_CONTRACT = (
    "materialized-executable-schedule-repeated-evaluation-groups-v1"
)
TARGET_ACTIVE_NON_SOURCE_CURRENT_SLOTS = 55
TARGET_MATERIALIZED_INTERACTIONS = 203
TARGET_TWO_COMPONENT_CURRENT_SLOTS = 26
TARGET_PROOF_DAG_NON_SOURCE_CURRENT_SLOTS = 1425
TARGET_PROOF_DAG_INTERACTIONS = 8338

REQUIRED_REGRESSION_CASES = frozenset(
    {
        "z6g-lc-union",
        "gg-ttbar-3g-nlc",
        "gg-ttbar-3g-full-color",
        "ddbar-z3g-residual-only",
        "builtin-model",
        "ufo-sm-model",
        "four-quark-lines-lc",
        "four-quark-lines-nlc",
        "four-quark-lines-full-color",
        "mutable-parameters",
        "structural-zeros",
        "global-selectors",
        "per-point-selectors",
        "malformed-plans",
    }
)
REQUIRED_NON_TARGET_PERFORMANCE = frozenset({"z6g-lc-union", "eager"})

_SHA256_LENGTH = 64
_DIAGNOSTIC_KEYS = frozenset(
    {
        "island_count",
        "kernel_count",
        "invocation_count",
        "attachment_count",
        "table_machine_code_bytes",
        "residual_machine_code_bytes",
        "semantic_row_bytes",
        "arena_bytes",
        "warmed_allocation_count",
    }
)


class AcceptanceError(RuntimeError):
    """Raised when evidence is malformed or incomplete."""


def _error(path: str, message: str) -> None:
    raise AcceptanceError(f"{path}: {message}")


def _mapping(value: object, path: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        _error(path, "expected an object")
    if not all(isinstance(key, str) for key in value):
        _error(path, "object keys must be strings")
    return value


def _sequence(value: object, path: str) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        _error(path, "expected an array")
    return value


def _exact_keys(
    value: Mapping[str, object],
    expected: frozenset[str] | set[str],
    path: str,
) -> None:
    actual = set(value)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        _error(path, f"field mismatch; missing={missing}, extra={extra}")


def _integer(
    value: object,
    path: str,
    *,
    minimum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _error(path, "expected an integer")
    if minimum is not None and value < minimum:
        _error(path, f"must be at least {minimum}")
    return value


def _number(
    value: object,
    path: str,
    *,
    minimum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _error(path, "expected a finite number")
    result = float(value)
    if not math.isfinite(result):
        _error(path, "expected a finite number")
    if minimum is not None and result < minimum:
        _error(path, f"must be at least {minimum}")
    return result


def _positive_number(value: object, path: str) -> float:
    result = _number(value, path)
    if result <= 0.0:
        _error(path, "must be greater than zero")
    return result


def _string(value: object, path: str, *, nonempty: bool = True) -> str:
    if not isinstance(value, str):
        _error(path, "expected a string")
    if nonempty and not value:
        _error(path, "must not be empty")
    return value


def _sha256(value: object, path: str) -> str:
    result = _string(value, path)
    if len(result) != _SHA256_LENGTH:
        _error(path, "expected a lowercase SHA-256 digest")
    if any(character not in "0123456789abcdef" for character in result):
        _error(path, "expected a lowercase SHA-256 digest")
    return result


def canonical_sha256(value: object) -> str:
    """Return the canonical JSON SHA-256 used by evidence certificates."""

    try:
        payload = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise AcceptanceError(f"value is not canonical JSON: {exc}") from exc
    return hashlib.sha256(payload).hexdigest()


def _authenticated_digest(value: Mapping[str, object], path: str) -> str:
    declared = _sha256(value.get("sha256"), f"{path}.sha256")
    payload = {key: item for key, item in value.items() if key != "sha256"}
    actual = canonical_sha256(payload)
    if declared != actual:
        _error(path, f"SHA-256 mismatch: declared {declared}, computed {actual}")
    return declared


def _unique_integers(
    value: object,
    path: str,
    *,
    minimum: int | None = 0,
) -> list[int]:
    result = [
        _integer(item, f"{path}[{index}]", minimum=minimum)
        for index, item in enumerate(_sequence(value, path))
    ]
    if len(result) != len(set(result)):
        _error(path, "values must be unique")
    return result


def _unique_strings(value: object, path: str) -> list[str]:
    result = [
        _string(item, f"{path}[{index}]")
        for index, item in enumerate(_sequence(value, path))
    ]
    if len(result) != len(set(result)):
        _error(path, "values must be unique")
    return result


def _dense_ids(values: Sequence[int], path: str) -> None:
    if list(values) != list(range(len(values))):
        _error(path, "IDs must be dense, ordered, and zero-based")


def _validate_diagnostics(value: object, path: str) -> dict[str, int]:
    diagnostics = _mapping(value, path)
    _exact_keys(diagnostics, _DIAGNOSTIC_KEYS, path)
    return {
        key: _integer(diagnostics[key], f"{path}.{key}", minimum=0)
        for key in sorted(_DIAGNOSTIC_KEYS)
    }


def _validate_residual_leaf(
    value: object,
    path: str,
    residual_current_ids: set[int],
) -> tuple[int, list[int]]:
    leaf = _mapping(value, path)
    _exact_keys(
        leaf,
        {
            "leaf_id",
            "current_ids",
            "source_sha256",
            "source_abi",
            "source_bytes",
            "machine_code_bytes",
            "optimization_level",
        },
        path,
    )
    _integer(leaf["leaf_id"], f"{path}.leaf_id", minimum=0)
    current_ids = _unique_integers(leaf["current_ids"], f"{path}.current_ids")
    if not current_ids:
        _error(f"{path}.current_ids", "must not be empty")
    if not set(current_ids) <= residual_current_ids:
        _error(f"{path}.current_ids", "contains a non-residual destination")
    _sha256(leaf["source_sha256"], f"{path}.source_sha256")
    if leaf["source_abi"] != DIRECT_APPLICATION_ABI:
        _error(f"{path}.source_abi", "incompatible DirectApplication ABI")
    _integer(leaf["source_bytes"], f"{path}.source_bytes", minimum=1)
    machine_code_bytes = _integer(
        leaf["machine_code_bytes"],
        f"{path}.machine_code_bytes",
        minimum=0,
    )
    if leaf["optimization_level"] != 3:
        _error(f"{path}.optimization_level", "compiled residuals must use O3")
    return machine_code_bytes, current_ids


def _validate_kernel(value: object, path: str) -> dict[str, object]:
    kernel = _mapping(value, path)
    _exact_keys(
        kernel,
        {
            "kernel_id",
            "motif_sha256",
            "source_sha256",
            "descriptor_sha256",
            "source_abi",
            "binding_abi",
            "descriptor_abi",
            "canonical_input_order",
            "input_permutation",
            "result_signature",
            "mutable_parameter_sha256",
            "coupling_provenance_sha256",
            "selector_domain_sha256",
            "finalizer_sha256",
            "input_complex_count",
            "output_complex_count",
            "source_bytes",
            "machine_code_bytes",
            "optimization_level",
        },
        path,
    )
    kernel_id = _integer(kernel["kernel_id"], f"{path}.kernel_id", minimum=0)
    for field in (
        "motif_sha256",
        "source_sha256",
        "descriptor_sha256",
        "mutable_parameter_sha256",
        "coupling_provenance_sha256",
        "selector_domain_sha256",
        "finalizer_sha256",
    ):
        _sha256(kernel[field], f"{path}.{field}")
    expected_abis = {
        "source_abi": DIRECT_APPLICATION_ABI,
        "binding_abi": DIRECT_TABLE_BINDING_ABI,
        "descriptor_abi": DIRECT_TABLE_DESCRIPTOR_ABI,
    }
    for field, expected in expected_abis.items():
        if kernel[field] != expected:
            _error(f"{path}.{field}", f"expected {expected!r}")
    canonical_order = _unique_strings(
        kernel["canonical_input_order"],
        f"{path}.canonical_input_order",
    )
    if not canonical_order:
        _error(f"{path}.canonical_input_order", "must not be empty")
    permutation = _unique_integers(
        kernel["input_permutation"],
        f"{path}.input_permutation",
    )
    if sorted(permutation) != list(range(len(canonical_order))):
        _error(f"{path}.input_permutation", "must permute every canonical input")
    _string(kernel["result_signature"], f"{path}.result_signature")
    input_count = _integer(
        kernel["input_complex_count"],
        f"{path}.input_complex_count",
        minimum=1,
    )
    output_count = _integer(
        kernel["output_complex_count"],
        f"{path}.output_complex_count",
        minimum=1,
    )
    if input_count > MAX_COMPLEX_INPUTS:
        _error(f"{path}.input_complex_count", "exceeds the initial-slice bound")
    if input_count != len(canonical_order):
        _error(
            f"{path}.input_complex_count",
            "must match the canonical input descriptor count",
        )
    if output_count > MAX_COMPLEX_OUTPUTS:
        _error(f"{path}.output_complex_count", "exceeds the initial-slice bound")
    source_bytes = _integer(
        kernel["source_bytes"],
        f"{path}.source_bytes",
        minimum=1,
    )
    if source_bytes > MAX_KERNEL_SOURCE_BYTES:
        _error(f"{path}.source_bytes", "exceeds 64 KiB")
    machine_code_bytes = _integer(
        kernel["machine_code_bytes"],
        f"{path}.machine_code_bytes",
        minimum=0,
    )
    if kernel["optimization_level"] != 3:
        _error(f"{path}.optimization_level", "microkernels must use O3")
    return {
        "kernel_id": kernel_id,
        "motif_sha256": kernel["motif_sha256"],
        "output_complex_count": output_count,
        "machine_code_bytes": machine_code_bytes,
        "finalizer_sha256": kernel["finalizer_sha256"],
    }


def _validate_certificate(
    value: object,
    path: str,
    *,
    kind: str,
    island_current_ids: list[int],
    order: bool,
) -> dict[int, list[int]]:
    certificate = _mapping(value, path)
    _exact_keys(certificate, {"kind", "rows", "sha256"}, path)
    if certificate["kind"] != kind:
        _error(f"{path}.kind", f"expected {kind!r}")
    rows = _sequence(certificate["rows"], f"{path}.rows")
    parsed: dict[int, list[int]] = {}
    value_field = "evaluation_group_ids" if order else "predecessor_current_ids"
    for index, raw_row in enumerate(rows):
        row_path = f"{path}.rows[{index}]"
        row = _mapping(raw_row, row_path)
        _exact_keys(row, {"current_id", value_field}, row_path)
        current_id = _integer(row["current_id"], f"{row_path}.current_id")
        if current_id in parsed:
            _error(f"{row_path}.current_id", "duplicate certificate current")
        parsed[current_id] = _unique_integers(
            row[value_field],
            f"{row_path}.{value_field}",
            minimum=None if order else 0,
        )
    if set(parsed) != set(island_current_ids):
        _error(f"{path}.rows", "must cover exactly the island destinations")
    if not order:
        island_set = set(island_current_ids)
        for current_id, predecessors in parsed.items():
            conflict = (set(predecessors) & island_set) - {current_id}
            if conflict:
                _error(
                    f"{path}.rows",
                    "dependency certificate groups dependent destinations "
                    f"{current_id} and {sorted(conflict)}",
                )
    _authenticated_digest(certificate, path)
    return parsed


def _validate_island(
    value: object,
    path: str,
    kernels: Mapping[int, Mapping[str, object]],
    table_current_ids: set[int],
) -> dict[str, object]:
    island = _mapping(value, path)
    _exact_keys(
        island,
        {
            "island_id",
            "kernel_id",
            "current_ids",
            "selector_partition_sha256",
            "invocations",
            "attachments",
            "factor_catalog",
            "plane_bindings",
            "dependency_certificate",
            "order_certificate",
            "semantic_row_bytes",
            "arena_bytes",
        },
        path,
    )
    island_id = _integer(island["island_id"], f"{path}.island_id", minimum=0)
    kernel_id = _integer(island["kernel_id"], f"{path}.kernel_id", minimum=0)
    if kernel_id not in kernels:
        _error(f"{path}.kernel_id", "references an unknown kernel")
    kernel = kernels[kernel_id]
    current_ids = _unique_integers(island["current_ids"], f"{path}.current_ids")
    if not current_ids:
        _error(f"{path}.current_ids", "must not be empty")
    if not set(current_ids) <= table_current_ids:
        _error(f"{path}.current_ids", "contains a non-table destination")
    _sha256(
        island["selector_partition_sha256"],
        f"{path}.selector_partition_sha256",
    )

    factor_rows = _sequence(island["factor_catalog"], f"{path}.factor_catalog")
    factor_ids: list[int] = []
    for index, raw_factor in enumerate(factor_rows):
        factor_path = f"{path}.factor_catalog[{index}]"
        factor = _mapping(raw_factor, factor_path)
        _exact_keys(factor, {"factor_id", "factor_sha256"}, factor_path)
        factor_ids.append(
            _integer(factor["factor_id"], f"{factor_path}.factor_id", minimum=0)
        )
        _sha256(factor["factor_sha256"], f"{factor_path}.factor_sha256")
    _dense_ids(factor_ids, f"{path}.factor_catalog.factor_id")

    binding_rows = _sequence(island["plane_bindings"], f"{path}.plane_bindings")
    binding_ids: list[int] = []
    binding_roles: set[str] = set()
    for index, raw_binding in enumerate(binding_rows):
        binding_path = f"{path}.plane_bindings[{index}]"
        binding = _mapping(raw_binding, binding_path)
        _exact_keys(
            binding,
            {"plane_id", "role", "canonical_index", "permutation_index"},
            binding_path,
        )
        binding_ids.append(
            _integer(binding["plane_id"], f"{binding_path}.plane_id", minimum=0)
        )
        role = _string(binding["role"], f"{binding_path}.role")
        if role not in {"current", "momentum", "parameter", "factor"}:
            _error(f"{binding_path}.role", "unknown plane role")
        binding_roles.add(role)
        _integer(
            binding["canonical_index"],
            f"{binding_path}.canonical_index",
            minimum=0,
        )
        _integer(
            binding["permutation_index"],
            f"{binding_path}.permutation_index",
            minimum=0,
        )
    _dense_ids(binding_ids, f"{path}.plane_bindings.plane_id")
    if binding_roles != {"current", "momentum", "parameter", "factor"}:
        _error(f"{path}.plane_bindings", "must bind all four plane roles")

    invocation_rows = _sequence(island["invocations"], f"{path}.invocations")
    invocation_ids: list[int] = []
    invocations: dict[int, tuple[int, int, int]] = {}
    for index, raw_invocation in enumerate(invocation_rows):
        invocation_path = f"{path}.invocations[{index}]"
        invocation = _mapping(raw_invocation, invocation_path)
        _exact_keys(
            invocation,
            {
                "invocation_id",
                "evaluation_group_id",
                "attachment_start",
                "attachment_count",
            },
            invocation_path,
        )
        invocation_id = _integer(
            invocation["invocation_id"],
            f"{invocation_path}.invocation_id",
            minimum=0,
        )
        invocation_ids.append(invocation_id)
        invocations[invocation_id] = (
            _integer(
                invocation["evaluation_group_id"],
                f"{invocation_path}.evaluation_group_id",
            ),
            _integer(
                invocation["attachment_start"],
                f"{invocation_path}.attachment_start",
                minimum=0,
            ),
            _integer(
                invocation["attachment_count"],
                f"{invocation_path}.attachment_count",
                minimum=1,
            ),
        )
    if not invocation_ids:
        _error(f"{path}.invocations", "must not be empty")
    _dense_ids(invocation_ids, f"{path}.invocations.invocation_id")

    attachment_rows = _sequence(island["attachments"], f"{path}.attachments")
    attachment_ids: list[int] = []
    actual_by_current: dict[int, list[tuple[int, str]]] = {
        current_id: [] for current_id in current_ids
    }
    for index, raw_attachment in enumerate(attachment_rows):
        attachment_path = f"{path}.attachments[{index}]"
        attachment = _mapping(raw_attachment, attachment_path)
        _exact_keys(
            attachment,
            {
                "attachment_id",
                "invocation_id",
                "current_id",
                "evaluation_group_id",
                "operation",
                "destination_complex_count",
                "factor_id",
            },
            attachment_path,
        )
        attachment_id = _integer(
            attachment["attachment_id"],
            f"{attachment_path}.attachment_id",
            minimum=0,
        )
        attachment_ids.append(attachment_id)
        invocation_id = _integer(
            attachment["invocation_id"],
            f"{attachment_path}.invocation_id",
            minimum=0,
        )
        if invocation_id not in invocations:
            _error(f"{attachment_path}.invocation_id", "unknown invocation")
        evaluation_group_id = _integer(
            attachment["evaluation_group_id"],
            f"{attachment_path}.evaluation_group_id",
        )
        invocation_group, start, count = invocations[invocation_id]
        if evaluation_group_id != invocation_group:
            _error(
                f"{attachment_path}.evaluation_group_id",
                "does not match its invocation",
            )
        if not start <= attachment_id < start + count:
            _error(
                f"{attachment_path}.attachment_id",
                "lies outside its invocation range",
            )
        current_id = _integer(
            attachment["current_id"],
            f"{attachment_path}.current_id",
            minimum=0,
        )
        if current_id not in actual_by_current:
            _error(f"{attachment_path}.current_id", "unknown island destination")
        operation = _string(
            attachment["operation"],
            f"{attachment_path}.operation",
        )
        if operation not in {"overwrite", "accumulate"}:
            _error(f"{attachment_path}.operation", "unknown write operation")
        destination_count = _integer(
            attachment["destination_complex_count"],
            f"{attachment_path}.destination_complex_count",
            minimum=1,
        )
        if destination_count != kernel["output_complex_count"]:
            _error(
                f"{attachment_path}.destination_complex_count",
                "attachment does not bind the complete kernel destination",
            )
        factor_id = _integer(
            attachment["factor_id"],
            f"{attachment_path}.factor_id",
            minimum=0,
        )
        if factor_id not in set(factor_ids):
            _error(f"{attachment_path}.factor_id", "unknown factor")
        actual_by_current[current_id].append((evaluation_group_id, operation))
    if not attachment_ids:
        _error(f"{path}.attachments", "must not be empty")
    _dense_ids(attachment_ids, f"{path}.attachments.attachment_id")

    expected_start = 0
    for invocation_id in invocation_ids:
        _, start, count = invocations[invocation_id]
        if start != expected_start:
            _error(
                f"{path}.invocations[{invocation_id}].attachment_start",
                "invocation attachment ranges must be contiguous",
            )
        owned = [
            index
            for index, raw_attachment in enumerate(attachment_rows)
            if _mapping(raw_attachment, path)["invocation_id"] == invocation_id
        ]
        if owned != list(range(start, start + count)):
            _error(
                f"{path}.invocations[{invocation_id}]",
                "declared range does not exactly own its attachment rows",
            )
        expected_start += count
    if expected_start != len(attachment_rows):
        _error(f"{path}.invocations", "ranges do not cover every attachment")

    dependency_rows = _validate_certificate(
        island["dependency_certificate"],
        f"{path}.dependency_certificate",
        kind="complete-current-independence-v1",
        island_current_ids=current_ids,
        order=False,
    )
    if set(dependency_rows) != set(current_ids):
        _error(f"{path}.dependency_certificate", "internal coverage mismatch")
    order_rows = _validate_certificate(
        island["order_certificate"],
        f"{path}.order_certificate",
        kind="evaluation-group-order-v1",
        island_current_ids=current_ids,
        order=True,
    )
    for current_id, actual_rows in actual_by_current.items():
        if not actual_rows:
            _error(f"{path}.attachments", f"current {current_id} is unwritten")
        operations = [operation for _, operation in actual_rows]
        if operations[0] != "overwrite":
            _error(f"{path}.attachments", f"current {current_id} is not initialized")
        if any(operation != "accumulate" for operation in operations[1:]):
            _error(
                f"{path}.attachments",
                f"current {current_id} has an invalid later write",
            )
        groups = [group for group, _ in actual_rows]
        if groups != order_rows[current_id]:
            _error(
                f"{path}.order_certificate",
                f"current {current_id} attachment order differs",
            )

    semantic_row_bytes = _integer(
        island["semantic_row_bytes"],
        f"{path}.semantic_row_bytes",
        minimum=0,
    )
    arena_bytes = _integer(
        island["arena_bytes"],
        f"{path}.arena_bytes",
        minimum=0,
    )
    return {
        "island_id": island_id,
        "kernel_id": kernel_id,
        "current_ids": current_ids,
        "evaluation_group_ids": sorted({values[0] for values in invocations.values()}),
        "invocation_count": len(invocation_rows),
        "attachment_count": len(attachment_rows),
        "semantic_row_bytes": semantic_row_bytes,
        "arena_bytes": arena_bytes,
    }


def audit_stage_plan_v2(value: object, path: str = "stage_plan") -> dict[str, object]:
    """Validate and independently summarize one normalized v2 stage plan."""

    plan = _mapping(value, path)
    _exact_keys(
        plan,
        {
            "kind",
            "schema_version",
            "stage_id",
            "direct_application_abi",
            "direct_table_binding_abi",
            "direct_table_descriptor_abi",
            "output_current_ids",
            "table_current_ids",
            "residual_current_ids",
            "residual_leaves",
            "kernels",
            "islands",
            "finalizers",
            "diagnostics",
        },
        path,
    )
    if plan["kind"] != STAGE_PLAN_KIND:
        _error(f"{path}.kind", f"expected {STAGE_PLAN_KIND!r}")
    if plan["schema_version"] != STAGE_PLAN_SCHEMA_VERSION:
        _error(
            f"{path}.schema_version",
            "compiled-stage-plan v2 is required; regenerate the artifact",
        )
    stage_id = _string(plan["stage_id"], f"{path}.stage_id")
    expected_abis = {
        "direct_application_abi": DIRECT_APPLICATION_ABI,
        "direct_table_binding_abi": DIRECT_TABLE_BINDING_ABI,
        "direct_table_descriptor_abi": DIRECT_TABLE_DESCRIPTOR_ABI,
    }
    for field, expected in expected_abis.items():
        if plan[field] != expected:
            _error(f"{path}.{field}", f"expected {expected!r}; regenerate artifact")

    output_ids = _unique_integers(
        plan["output_current_ids"],
        f"{path}.output_current_ids",
    )
    if not output_ids:
        _error(f"{path}.output_current_ids", "must not be empty")
    table_ids = _unique_integers(
        plan["table_current_ids"],
        f"{path}.table_current_ids",
    )
    residual_ids = _unique_integers(
        plan["residual_current_ids"],
        f"{path}.residual_current_ids",
    )
    if set(table_ids) & set(residual_ids):
        _error(path, "a destination may not be split between table and residual")
    if set(table_ids) | set(residual_ids) != set(output_ids):
        _error(path, "table and residual destinations must partition all outputs")

    residual_rows = _sequence(plan["residual_leaves"], f"{path}.residual_leaves")
    residual_leaf_ids: list[int] = []
    residual_machine_code_bytes = 0
    residual_owned: list[int] = []
    for index, raw_leaf in enumerate(residual_rows):
        leaf_path = f"{path}.residual_leaves[{index}]"
        leaf = _mapping(raw_leaf, leaf_path)
        residual_leaf_ids.append(
            _integer(leaf.get("leaf_id"), f"{leaf_path}.leaf_id", minimum=0)
        )
        machine_bytes, owned = _validate_residual_leaf(
            raw_leaf,
            leaf_path,
            set(residual_ids),
        )
        residual_machine_code_bytes += machine_bytes
        residual_owned.extend(owned)
    _dense_ids(residual_leaf_ids, f"{path}.residual_leaves.leaf_id")
    if len(residual_owned) != len(set(residual_owned)):
        _error(f"{path}.residual_leaves", "residual destinations overlap")
    if set(residual_owned) != set(residual_ids):
        _error(f"{path}.residual_leaves", "must own every residual destination")

    kernel_rows = _sequence(plan["kernels"], f"{path}.kernels")
    if len(kernel_rows) > MAX_KERNEL_IDENTITIES:
        _error(f"{path}.kernels", "exceeds eight kernel identities")
    kernels: dict[int, Mapping[str, object]] = {}
    motif_ids: set[object] = set()
    kernel_machine_code_bytes = 0
    kernel_ids: list[int] = []
    for index, raw_kernel in enumerate(kernel_rows):
        kernel_path = f"{path}.kernels[{index}]"
        summary = _validate_kernel(raw_kernel, kernel_path)
        kernel_id = int(summary["kernel_id"])
        kernel_ids.append(kernel_id)
        if summary["motif_sha256"] in motif_ids:
            _error(f"{kernel_path}.motif_sha256", "duplicate canonical motif")
        motif_ids.add(summary["motif_sha256"])
        kernels[kernel_id] = summary
        kernel_machine_code_bytes += int(summary["machine_code_bytes"])
    _dense_ids(kernel_ids, f"{path}.kernels.kernel_id")

    island_rows = _sequence(plan["islands"], f"{path}.islands")
    island_ids: list[int] = []
    island_summaries: list[dict[str, object]] = []
    table_owned: list[int] = []
    for index, raw_island in enumerate(island_rows):
        island_path = f"{path}.islands[{index}]"
        summary = _validate_island(
            raw_island,
            island_path,
            kernels,
            set(table_ids),
        )
        island_ids.append(int(summary["island_id"]))
        island_summaries.append(summary)
        table_owned.extend(summary["current_ids"])  # type: ignore[arg-type]
    _dense_ids(island_ids, f"{path}.islands.island_id")
    if len(table_owned) != len(set(table_owned)):
        _error(f"{path}.islands", "table destinations overlap between islands")
    if set(table_owned) != set(table_ids):
        _error(f"{path}.islands", "must own every table destination")
    if bool(kernel_rows) != bool(island_rows):
        _error(path, "unused kernels and kernel-less islands are forbidden")

    finalizer_rows = _sequence(plan["finalizers"], f"{path}.finalizers")
    finalized: list[int] = []
    island_by_current = {
        current_id: summary
        for summary in island_summaries
        for current_id in summary["current_ids"]  # type: ignore[union-attr]
    }
    for index, raw_finalizer in enumerate(finalizer_rows):
        finalizer_path = f"{path}.finalizers[{index}]"
        finalizer = _mapping(raw_finalizer, finalizer_path)
        _exact_keys(
            finalizer,
            {"current_id", "island_id", "identity_sha256"},
            finalizer_path,
        )
        current_id = _integer(
            finalizer["current_id"],
            f"{finalizer_path}.current_id",
            minimum=0,
        )
        finalized.append(current_id)
        if current_id not in island_by_current:
            _error(f"{finalizer_path}.current_id", "not a table destination")
        expected_island = island_by_current[current_id]
        if finalizer["island_id"] != expected_island["island_id"]:
            _error(
                f"{finalizer_path}.island_id",
                "finalizer must run after its complete island",
            )
        kernel = kernels[int(expected_island["kernel_id"])]
        identity = _sha256(
            finalizer["identity_sha256"],
            f"{finalizer_path}.identity_sha256",
        )
        if identity != kernel["finalizer_sha256"]:
            _error(
                f"{finalizer_path}.identity_sha256",
                "does not match the canonical motif finalizer",
            )
    if len(finalized) != len(set(finalized)) or set(finalized) != set(table_ids):
        _error(f"{path}.finalizers", "must finalize every table destination once")

    declared = _validate_diagnostics(plan["diagnostics"], f"{path}.diagnostics")
    recomputed = {
        "island_count": len(island_rows),
        "kernel_count": len(kernel_rows),
        "invocation_count": sum(
            int(summary["invocation_count"]) for summary in island_summaries
        ),
        "attachment_count": sum(
            int(summary["attachment_count"]) for summary in island_summaries
        ),
        "table_machine_code_bytes": kernel_machine_code_bytes,
        "residual_machine_code_bytes": residual_machine_code_bytes,
        "semantic_row_bytes": sum(
            int(summary["semantic_row_bytes"]) for summary in island_summaries
        ),
        "arena_bytes": sum(int(summary["arena_bytes"]) for summary in island_summaries),
        # This counter comes from the runtime allocation probe, while every
        # other field above is derivable from static plan rows.
        "warmed_allocation_count": declared["warmed_allocation_count"],
    }
    if declared != recomputed:
        _error(
            f"{path}.diagnostics",
            f"declared values differ from recomputed values: {recomputed}",
        )
    return {
        "stage_id": stage_id,
        "diagnostics": recomputed,
        "table_evaluation_group_ids": [
            f"{stage_id}:{group_id}"
            for summary in island_summaries
            for group_id in summary["evaluation_group_ids"]  # type: ignore[union-attr]
        ],
        "residual_only": not table_ids,
        "table_only": not residual_ids,
    }


def _sum_diagnostics(
    summaries: Sequence[Mapping[str, object]],
) -> dict[str, int]:
    return {
        key: sum(
            int(_mapping(summary["diagnostics"], "summary.diagnostics")[key])
            for summary in summaries
        )
        for key in sorted(_DIAGNOSTIC_KEYS)
    }


def _validate_census(
    value: object,
    path: str,
    table_evaluation_group_ids: set[str],
) -> dict[str, float | int]:
    census = _mapping(value, path)
    _exact_keys(
        census,
        {
            "denominator_contract",
            "active_non_source_current_slots",
            "materialized_interaction_count",
            "two_component_current_slots",
            "active_repeated_evaluation_group_ids",
            "eligible_evaluation_group_ids",
            "proof_dag_non_source_current_slots",
            "proof_dag_interaction_count",
            "projected_generated_text_bytes",
            "projected_replaced_text_bytes",
        },
        path,
    )
    if census["denominator_contract"] != CENSUS_DENOMINATOR_CONTRACT:
        _error(
            f"{path}.denominator_contract",
            "coverage denominator must be the materialized executable schedule",
        )
    expected_counts = {
        "active_non_source_current_slots": TARGET_ACTIVE_NON_SOURCE_CURRENT_SLOTS,
        "materialized_interaction_count": TARGET_MATERIALIZED_INTERACTIONS,
        "two_component_current_slots": TARGET_TWO_COMPONENT_CURRENT_SLOTS,
        "proof_dag_non_source_current_slots": (
            TARGET_PROOF_DAG_NON_SOURCE_CURRENT_SLOTS
        ),
        "proof_dag_interaction_count": TARGET_PROOF_DAG_INTERACTIONS,
    }
    for field, expected in expected_counts.items():
        actual = _integer(census[field], f"{path}.{field}", minimum=0)
        if actual != expected:
            _error(
                f"{path}.{field}",
                f"expected frozen target census count {expected}, got {actual}",
            )
    if (
        census["active_non_source_current_slots"]
        == (census["proof_dag_non_source_current_slots"])
        or census["materialized_interaction_count"]
        == (census["proof_dag_interaction_count"])
    ):
        _error(path, "proof-DAG counts are mislabeled as active schedule counts")

    active = set(
        _unique_strings(
            census["active_repeated_evaluation_group_ids"],
            f"{path}.active_repeated_evaluation_group_ids",
        )
    )
    eligible = set(
        _unique_strings(
            census["eligible_evaluation_group_ids"],
            f"{path}.eligible_evaluation_group_ids",
        )
    )
    if not active:
        _error(f"{path}.active_repeated_evaluation_group_ids", "must not be empty")
    if not eligible <= active:
        _error(f"{path}.eligible_evaluation_group_ids", "must be an active subset")
    table_repeated_groups = table_evaluation_group_ids & active
    if eligible != table_repeated_groups:
        _error(
            f"{path}.eligible_evaluation_group_ids",
            "must exactly match repeated groups covered by table invocations",
        )
    coverage = len(eligible) / len(active)
    if coverage < MIN_CENSUS_COVERAGE:
        _error(f"{path}.eligible_evaluation_group_ids", "coverage is below 50%")

    generated = _integer(
        census["projected_generated_text_bytes"],
        f"{path}.projected_generated_text_bytes",
        minimum=0,
    )
    replaced = _integer(
        census["projected_replaced_text_bytes"],
        f"{path}.projected_replaced_text_bytes",
        minimum=1,
    )
    fraction = generated / replaced
    if fraction > MAX_PROJECTED_TEXT_FRACTION:
        _error(
            f"{path}.projected_generated_text_bytes",
            "projected text exceeds 25% of replaced text",
        )
    return {
        "active_repeated_evaluation_group_count": len(active),
        "eligible_evaluation_group_count": len(eligible),
        "coverage": coverage,
        "projected_text_fraction": fraction,
    }


def _validate_artifact_metrics(
    value: object,
    path: str,
) -> dict[str, object]:
    artifact = _mapping(value, path)
    _exact_keys(
        artifact,
        {
            "artifact_sha256",
            "build_identity_sha256",
            "generation_seconds",
            "artifact_bytes",
            "load_seconds",
            "peak_rss_bytes",
            "code_size_metric_available",
            "selected_machine_code_bytes",
            "portable_source_applications",
        },
        path,
    )
    metric_available = artifact["code_size_metric_available"]
    if not isinstance(metric_available, bool):
        _error(f"{path}.code_size_metric_available", "expected a boolean")
    selected_machine_code: int | None
    if metric_available:
        selected_machine_code = _integer(
            artifact["selected_machine_code_bytes"],
            f"{path}.selected_machine_code_bytes",
            minimum=1,
        )
    else:
        if artifact["selected_machine_code_bytes"] is not None:
            _error(
                f"{path}.selected_machine_code_bytes",
                "must be null when the exact machine-code metric is unavailable",
            )
        selected_machine_code = None
    applications = _sequence(
        artifact["portable_source_applications"],
        f"{path}.portable_source_applications",
    )
    portable_source_bytes = 0
    portable_source_digests: list[str] = []
    for index, raw_application in enumerate(applications):
        application_path = f"{path}.portable_source_applications[{index}]"
        application = _mapping(raw_application, application_path)
        _exact_keys(
            application,
            {
                "source_application_sha256",
                "source_application_bytes",
                "source_application_abi",
            },
            application_path,
        )
        digest = _sha256(
            application["source_application_sha256"],
            f"{application_path}.source_application_sha256",
        )
        if digest in portable_source_digests:
            _error(application_path, "portable source payloads must be deduplicated")
        portable_source_digests.append(digest)
        portable_source_bytes += _integer(
            application["source_application_bytes"],
            f"{application_path}.source_application_bytes",
            minimum=1,
        )
        if application["source_application_abi"] != SOURCE_APPLICATION_ABI:
            _error(
                f"{application_path}.source_application_abi",
                f"expected {SOURCE_APPLICATION_ABI!r}",
            )
    if not applications:
        _error(f"{path}.portable_source_applications", "must not be empty")
    return {
        "artifact_sha256": _sha256(
            artifact["artifact_sha256"],
            f"{path}.artifact_sha256",
        ),
        "build_identity_sha256": _sha256(
            artifact["build_identity_sha256"],
            f"{path}.build_identity_sha256",
        ),
        "generation_seconds": _positive_number(
            artifact["generation_seconds"],
            f"{path}.generation_seconds",
        ),
        "artifact_bytes": _integer(
            artifact["artifact_bytes"],
            f"{path}.artifact_bytes",
            minimum=1,
        ),
        "load_seconds": _positive_number(
            artifact["load_seconds"],
            f"{path}.load_seconds",
        ),
        "peak_rss_bytes": _integer(
            artifact["peak_rss_bytes"],
            f"{path}.peak_rss_bytes",
            minimum=1,
        ),
        "code_size_metric_available": metric_available,
        "selected_machine_code_bytes": selected_machine_code,
        "portable_source_application_bytes": portable_source_bytes,
        "portable_source_application_set_sha256": canonical_sha256(
            portable_source_digests
        ),
    }


def _validate_candidate(
    value: object,
    path: str,
) -> tuple[dict[str, object], dict[str, object]]:
    candidate = _mapping(value, path)
    _exact_keys(
        candidate,
        {
            "metrics",
            "stage_plans",
            "diagnostics",
            "census",
        },
        path,
    )
    metrics = _validate_artifact_metrics(candidate["metrics"], f"{path}.metrics")
    stage_rows = _sequence(candidate["stage_plans"], f"{path}.stage_plans")
    if not stage_rows:
        _error(f"{path}.stage_plans", "must not be empty")
    summaries = [
        audit_stage_plan_v2(row, f"{path}.stage_plans[{index}]")
        for index, row in enumerate(stage_rows)
    ]
    stage_ids = [str(summary["stage_id"]) for summary in summaries]
    if len(stage_ids) != len(set(stage_ids)):
        _error(f"{path}.stage_plans", "stage IDs must be unique")
    diagnostics = _validate_diagnostics(
        candidate["diagnostics"],
        f"{path}.diagnostics",
    )
    recomputed = _sum_diagnostics(summaries)
    if diagnostics != recomputed:
        _error(
            f"{path}.diagnostics",
            f"artifact totals differ from stage totals: {recomputed}",
        )
    if diagnostics["semantic_row_bytes"] > MAX_SEMANTIC_ROW_BYTES:
        _error(f"{path}.diagnostics.semantic_row_bytes", "exceeds 4 MiB")
    if diagnostics["kernel_count"] > MAX_KERNEL_IDENTITIES:
        _error(f"{path}.diagnostics.kernel_count", "exceeds eight identities")
    concrete_machine_code_bytes = (
        diagnostics["table_machine_code_bytes"]
        + diagnostics["residual_machine_code_bytes"]
    )
    selected_machine_code_bytes = metrics["selected_machine_code_bytes"]
    if (
        metrics["code_size_metric_available"] is True
        and isinstance(selected_machine_code_bytes, int)
        and selected_machine_code_bytes < concrete_machine_code_bytes
    ):
        _error(
            f"{path}.metrics.selected_machine_code_bytes",
            "is smaller than concrete table plus residual machine code",
        )
    table_groups = {
        str(group)
        for summary in summaries
        for group in summary["table_evaluation_group_ids"]  # type: ignore[union-attr]
    }
    census = _validate_census(
        candidate["census"],
        f"{path}.census",
        table_groups,
    )
    return metrics, {
        "diagnostics": diagnostics,
        "census": census,
        "stage_count": len(summaries),
        "residual_only_stage_count": sum(
            bool(summary["residual_only"]) for summary in summaries
        ),
        "table_only_stage_count": sum(
            bool(summary["table_only"]) for summary in summaries
        ),
    }


def _complex_array(value: object, path: str, expected_length: int) -> list[complex]:
    rows = _sequence(value, path)
    if len(rows) != expected_length:
        _error(path, f"expected {expected_length} complex values")
    result: list[complex] = []
    for index, raw_pair in enumerate(rows):
        pair_path = f"{path}[{index}]"
        pair = _sequence(raw_pair, pair_path)
        if len(pair) != 2:
            _error(pair_path, "complex values must be [real, imag]")
        result.append(
            complex(
                _number(pair[0], f"{pair_path}[0]"),
                _number(pair[1], f"{pair_path}[1]"),
            )
        )
    return result


def _resolved_array(
    value: object,
    path: str,
    expected_length: int,
) -> list[list[complex]]:
    rows = _sequence(value, path)
    if len(rows) != expected_length:
        _error(path, f"expected {expected_length} point rows")
    result: list[list[complex]] = []
    for index, raw_row in enumerate(rows):
        row_path = f"{path}[{index}]"
        entries = _sequence(raw_row, row_path)
        if not entries:
            _error(row_path, "resolved contributions must not be empty")
        result.append(_complex_array(entries, row_path, len(entries)))
    return result


def _close(left: complex, right: complex) -> bool:
    return abs(left - right) <= (ABSOLUTE_TOLERANCE + RELATIVE_TOLERANCE * abs(right))


def _assert_close_arrays(
    left: Sequence[complex],
    right: Sequence[complex],
    path: str,
) -> float:
    if len(left) != len(right):
        _error(path, "array shapes differ")
    maximum = 0.0
    for index, (actual, expected) in enumerate(zip(left, right, strict=True)):
        difference = abs(actual - expected)
        maximum = max(maximum, difference)
        if not _close(actual, expected):
            _error(
                f"{path}[{index}]",
                f"{actual!r} differs from {expected!r}",
            )
    return maximum


def _validate_correctness(
    value: object,
    path: str,
    batch: int,
) -> dict[str, float]:
    correctness = _mapping(value, path)
    _exact_keys(correctness, {"baseline", "candidate"}, path)
    lanes: dict[str, dict[str, object]] = {}
    for lane in ("baseline", "candidate"):
        lane_path = f"{path}.{lane}"
        raw_lane = _mapping(correctness[lane], lane_path)
        _exact_keys(
            raw_lane,
            {"evaluate", "resolved_total", "resolved_contributions"},
            lane_path,
        )
        evaluated = _complex_array(
            raw_lane["evaluate"],
            f"{lane_path}.evaluate",
            batch,
        )
        total = _complex_array(
            raw_lane["resolved_total"],
            f"{lane_path}.resolved_total",
            batch,
        )
        resolved = _resolved_array(
            raw_lane["resolved_contributions"],
            f"{lane_path}.resolved_contributions",
            batch,
        )
        lanes[lane] = {
            "evaluate": evaluated,
            "resolved_total": total,
            "resolved_contributions": resolved,
        }
        _assert_close_arrays(
            evaluated,
            total,
            f"{lane_path}.evaluate_vs_resolved_total",
        )
        _assert_close_arrays(
            [sum(row, start=0j) for row in resolved],
            total,
            f"{lane_path}.resolved_contributions_vs_total",
        )

    baseline = lanes["baseline"]
    candidate = lanes["candidate"]
    total_error = _assert_close_arrays(
        candidate["evaluate"],  # type: ignore[arg-type]
        baseline["evaluate"],  # type: ignore[arg-type]
        f"{path}.candidate_vs_baseline_total",
    )
    baseline_resolved = baseline["resolved_contributions"]
    candidate_resolved = candidate["resolved_contributions"]
    if not isinstance(baseline_resolved, list) or not isinstance(
        candidate_resolved,
        list,
    ):
        _error(path, "internal resolved representation error")
    resolved_error = 0.0
    for point_index, (candidate_row, baseline_row) in enumerate(
        zip(candidate_resolved, baseline_resolved, strict=True)
    ):
        resolved_error = max(
            resolved_error,
            _assert_close_arrays(
                candidate_row,
                baseline_row,
                f"{path}.candidate_vs_baseline_resolved[{point_index}]",
            ),
        )
    return {
        "maximum_total_absolute_error": total_error,
        "maximum_resolved_absolute_error": resolved_error,
    }


def _median_and_mad(values: Sequence[float]) -> tuple[float, float]:
    median = statistics.median(values)
    mad = statistics.median(abs(value - median) for value in values)
    return median, mad


def _validate_measurements(
    value: object,
    path: str,
) -> dict[str, object]:
    rows = _sequence(value, path)
    if len(rows) < 2 * MIN_SAMPLE_PAIRS or len(rows) % 2:
        _error(path, "requires at least seven complete subprocess pairs")
    parsed: list[dict[str, float | int | str]] = []
    for index, raw_row in enumerate(rows):
        row_path = f"{path}[{index}]"
        row = _mapping(raw_row, row_path)
        _exact_keys(
            row,
            {
                "pair_index",
                "sequence_index",
                "lane",
                "duration_seconds",
                "seconds_per_point",
                "warmed_allocation_count",
            },
            row_path,
        )
        lane = _string(row["lane"], f"{row_path}.lane")
        if lane not in {"baseline", "candidate"}:
            _error(f"{row_path}.lane", "expected baseline or candidate")
        duration = _number(
            row["duration_seconds"],
            f"{row_path}.duration_seconds",
            minimum=0.0,
        )
        if duration < MIN_SAMPLE_DURATION_SECONDS:
            _error(f"{row_path}.duration_seconds", "sample ran for under 5 s")
        allocations = _integer(
            row["warmed_allocation_count"],
            f"{row_path}.warmed_allocation_count",
            minimum=0,
        )
        parsed.append(
            {
                "pair_index": _integer(
                    row["pair_index"],
                    f"{row_path}.pair_index",
                    minimum=0,
                ),
                "sequence_index": _integer(
                    row["sequence_index"],
                    f"{row_path}.sequence_index",
                    minimum=0,
                ),
                "lane": lane,
                "duration_seconds": duration,
                "seconds_per_point": _positive_number(
                    row["seconds_per_point"],
                    f"{row_path}.seconds_per_point",
                ),
                "warmed_allocation_count": allocations,
            }
        )
    if [row["sequence_index"] for row in parsed] != list(range(len(parsed))):
        _error(path, "sequence indices must be ordered, dense, and zero-based")
    pair_count = len(parsed) // 2
    pairs: list[tuple[float, float]] = []
    first_lane = str(parsed[0]["lane"])
    second_lane = "candidate" if first_lane == "baseline" else "baseline"
    for pair_index in range(pair_count):
        first = parsed[2 * pair_index]
        second = parsed[2 * pair_index + 1]
        if first["pair_index"] != pair_index or second["pair_index"] != pair_index:
            _error(path, "pair indices must be dense and align with sequence order")
        expected_order = (
            (first_lane, second_lane)
            if pair_index % 2 == 0
            else (second_lane, first_lane)
        )
        if (first["lane"], second["lane"]) != expected_order:
            _error(path, "subprocess lane order must alternate between pairs")
        by_lane = {str(first["lane"]): first, str(second["lane"]): second}
        pairs.append(
            (
                float(by_lane["baseline"]["seconds_per_point"]),
                float(by_lane["candidate"]["seconds_per_point"]),
            )
        )
    baseline_values = [pair[0] for pair in pairs]
    candidate_values = [pair[1] for pair in pairs]
    changes = [baseline - candidate for baseline, candidate in pairs]
    baseline_median, baseline_mad = _median_and_mad(baseline_values)
    candidate_median, candidate_mad = _median_and_mad(candidate_values)
    change_median, change_mad = _median_and_mad(changes)
    return {
        "pair_count": pair_count,
        "baseline_median_seconds_per_point": baseline_median,
        "baseline_mad_seconds_per_point": baseline_mad,
        "candidate_median_seconds_per_point": candidate_median,
        "candidate_mad_seconds_per_point": candidate_mad,
        "paired_change_median_seconds_per_point": change_median,
        "paired_change_mad_seconds_per_point": change_mad,
        "relative_gain": (baseline_median - candidate_median) / baseline_median,
        "paired_change_beyond_three_mad": (change_median > MAD_MULTIPLIER * change_mad),
        "candidate_warmed_allocation_count": max(
            int(row["warmed_allocation_count"])
            for row in parsed
            if row["lane"] == "candidate"
        ),
    }


def _validate_workload(value: object, path: str) -> None:
    workload = _mapping(value, path)
    _exact_keys(
        workload,
        {
            "process",
            "color_accuracy",
            "lc_flow_layout",
            "selected_flow",
            "helicity_mode",
            "source_sha256",
            "model_sha256",
            "point_set_sha256",
            "runtime_target_sha256",
            "host_sha256",
        },
        path,
    )
    expected = {
        "process": TARGET_PROCESS,
        "color_accuracy": TARGET_COLOR_ACCURACY,
        "lc_flow_layout": TARGET_LAYOUT,
        "selected_flow": TARGET_FLOW,
        "helicity_mode": TARGET_HELICITY_MODE,
    }
    for field, expected_value in expected.items():
        if workload[field] != expected_value:
            _error(f"{path}.{field}", f"expected {expected_value!r}")
    for field in (
        "source_sha256",
        "model_sha256",
        "point_set_sha256",
        "runtime_target_sha256",
        "host_sha256",
    ):
        _sha256(workload[field], f"{path}.{field}")


def _validate_dependency(value: object, path: str) -> None:
    dependency = _mapping(value, path)
    _exact_keys(
        dependency,
        {
            "revision",
            "archive_sha256",
            "source_tree_sha256",
            "candidate_tree_sha256",
            "local_patch_count",
            "direct_application_abi",
            "direct_table_binding_abi",
            "direct_table_descriptor_abi",
        },
        path,
    )
    expected = {
        "revision": DEPENDENCY_REVISION,
        "archive_sha256": DEPENDENCY_ARCHIVE_SHA256,
        "source_tree_sha256": DEPENDENCY_SOURCE_TREE_SHA256,
        "candidate_tree_sha256": DEPENDENCY_CANDIDATE_TREE_SHA256,
        "local_patch_count": 0,
        "direct_application_abi": DIRECT_APPLICATION_ABI,
        "direct_table_binding_abi": DIRECT_TABLE_BINDING_ABI,
        "direct_table_descriptor_abi": DIRECT_TABLE_DESCRIPTOR_ABI,
    }
    for field, expected_value in expected.items():
        if dependency[field] != expected_value:
            _error(f"{path}.{field}", f"expected {expected_value!r}")


def _gate(
    gates: dict[str, dict[str, object]],
    name: str,
    passed: bool,
    **details: object,
) -> None:
    gates[name] = {"passes": bool(passed), **details}


def _validate_regressions(
    functional_value: object,
    performance_value: object,
    gates: dict[str, dict[str, object]],
) -> None:
    functional_rows = _sequence(functional_value, "campaign.regression_cases")
    seen: set[str] = set()
    for index, raw_case in enumerate(functional_rows):
        path = f"campaign.regression_cases[{index}]"
        case = _mapping(raw_case, path)
        _exact_keys(case, {"name", "passed", "evidence_sha256"}, path)
        name = _string(case["name"], f"{path}.name")
        if name in seen:
            _error(f"{path}.name", "duplicate regression case")
        seen.add(name)
        if not isinstance(case["passed"], bool):
            _error(f"{path}.passed", "expected a boolean")
        _sha256(case["evidence_sha256"], f"{path}.evidence_sha256")
        _gate(gates, f"regression:{name}", bool(case["passed"]))
    missing = sorted(REQUIRED_REGRESSION_CASES - seen)
    if missing:
        _error("campaign.regression_cases", f"missing required cases: {missing}")

    performance_rows = _sequence(
        performance_value,
        "campaign.non_target_performance",
    )
    performance_seen: set[str] = set()
    for index, raw_lane in enumerate(performance_rows):
        path = f"campaign.non_target_performance[{index}]"
        lane = _mapping(raw_lane, path)
        _exact_keys(
            lane,
            {"name", "baseline_seconds_per_point", "candidate_seconds_per_point"},
            path,
        )
        name = _string(lane["name"], f"{path}.name")
        if name in performance_seen:
            _error(f"{path}.name", "duplicate performance lane")
        performance_seen.add(name)
        baseline = [
            _positive_number(
                item,
                f"{path}.baseline_seconds_per_point[{item_index}]",
            )
            for item_index, item in enumerate(
                _sequence(
                    lane["baseline_seconds_per_point"],
                    f"{path}.baseline_seconds_per_point",
                )
            )
        ]
        candidate = [
            _positive_number(
                item,
                f"{path}.candidate_seconds_per_point[{item_index}]",
            )
            for item_index, item in enumerate(
                _sequence(
                    lane["candidate_seconds_per_point"],
                    f"{path}.candidate_seconds_per_point",
                )
            )
        ]
        if len(baseline) < MIN_SAMPLE_PAIRS or len(candidate) < MIN_SAMPLE_PAIRS:
            _error(path, "requires at least seven samples in each lane")
        baseline_median, baseline_mad = _median_and_mad(baseline)
        candidate_median, _ = _median_and_mad(candidate)
        relative_regression = (candidate_median - baseline_median) / baseline_median
        absolute_regression = candidate_median - baseline_median
        passed = (
            relative_regression <= MAX_NON_TARGET_REGRESSION
            or absolute_regression <= MAD_MULTIPLIER * baseline_mad
        )
        _gate(
            gates,
            f"non_target_performance:{name}",
            passed,
            relative_regression=relative_regression,
            absolute_regression_seconds_per_point=absolute_regression,
            baseline_mad_seconds_per_point=baseline_mad,
        )
    missing_performance = sorted(REQUIRED_NON_TARGET_PERFORMANCE - performance_seen)
    if missing_performance:
        _error(
            "campaign.non_target_performance",
            f"missing required lanes: {missing_performance}",
        )


def audit_campaign(value: object) -> dict[str, object]:
    """Audit a complete baseline/candidate campaign."""

    campaign = _mapping(value, "campaign")
    _exact_keys(
        campaign,
        {
            "kind",
            "schema_version",
            "content_sha256",
            "dependency",
            "workload",
            "baseline",
            "candidate",
            "batches",
            "regression_cases",
            "non_target_performance",
        },
        "campaign",
    )
    if campaign["kind"] != CAMPAIGN_KIND:
        _error("campaign.kind", f"expected {CAMPAIGN_KIND!r}")
    if campaign["schema_version"] != CAMPAIGN_SCHEMA_VERSION:
        _error("campaign.schema_version", "unsupported campaign schema")
    declared_digest = _sha256(
        campaign["content_sha256"],
        "campaign.content_sha256",
    )
    actual_digest = canonical_sha256(
        {key: item for key, item in campaign.items() if key != "content_sha256"}
    )
    if declared_digest != actual_digest:
        _error(
            "campaign.content_sha256",
            f"declared {declared_digest}, computed {actual_digest}",
        )
    _validate_dependency(campaign["dependency"], "campaign.dependency")
    _validate_workload(campaign["workload"], "campaign.workload")
    baseline = _validate_artifact_metrics(campaign["baseline"], "campaign.baseline")
    candidate, candidate_summary = _validate_candidate(
        campaign["candidate"],
        "campaign.candidate",
    )

    gates: dict[str, dict[str, object]] = {}
    resource_fields = (
        "generation_seconds",
        "artifact_bytes",
        "load_seconds",
        "peak_rss_bytes",
    )
    for field in resource_fields:
        baseline_value = float(baseline[field])
        candidate_value = float(candidate[field])
        ratio = candidate_value / baseline_value
        _gate(
            gates,
            f"resource:{field}",
            ratio <= 1.0 + MAX_RESOURCE_REGRESSION,
            candidate_to_baseline_ratio=ratio,
        )
    code_size_available = (
        baseline["code_size_metric_available"] is True
        and candidate["code_size_metric_available"] is True
    )
    if code_size_available:
        machine_code_ratio = float(candidate["selected_machine_code_bytes"]) / float(
            baseline["selected_machine_code_bytes"]
        )
        _gate(
            gates,
            "selected_machine_code_reduction",
            machine_code_ratio <= 1.0 - MIN_CODE_SIZE_REDUCTION,
            metric_available=True,
            candidate_to_baseline_ratio=machine_code_ratio,
        )
    else:
        _gate(
            gates,
            "selected_machine_code_reduction",
            False,
            metric_available=False,
            reason=(
                "exact residual portable-source-application machine-code "
                "shape is unavailable from the pinned SymJIT API"
            ),
        )
    diagnostics = _mapping(
        candidate_summary["diagnostics"],
        "candidate_summary.diagnostics",
    )
    batches = _mapping(campaign["batches"], "campaign.batches")
    if set(batches) != {str(batch) for batch in REQUIRED_BATCHES}:
        _error(
            "campaign.batches",
            f"must contain exactly batches {list(REQUIRED_BATCHES)}",
        )
    batch_summaries: dict[str, object] = {}
    measured_candidate_allocations = 0
    workload = _mapping(campaign["workload"], "campaign.workload")
    for batch in REQUIRED_BATCHES:
        path = f"campaign.batches.{batch}"
        batch_value = _mapping(batches[str(batch)], path)
        _exact_keys(
            batch_value,
            {"point_set_sha256", "correctness", "measurements"},
            path,
        )
        point_digest = _sha256(
            batch_value["point_set_sha256"],
            f"{path}.point_set_sha256",
        )
        if point_digest != workload["point_set_sha256"]:
            _error(
                f"{path}.point_set_sha256",
                "must match the campaign point-set identity",
            )
        correctness = _validate_correctness(
            batch_value["correctness"],
            f"{path}.correctness",
            batch,
        )
        measurements = _validate_measurements(
            batch_value["measurements"],
            f"{path}.measurements",
        )
        measured_candidate_allocations = max(
            measured_candidate_allocations,
            int(measurements["candidate_warmed_allocation_count"]),
        )
        batch_summaries[str(batch)] = {
            "correctness": correctness,
            "performance": measurements,
        }
        relative_gain = float(measurements["relative_gain"])
        if batch in PRIMARY_BATCHES:
            _gate(
                gates,
                f"primary_gain:batch_{batch}",
                (
                    relative_gain >= MIN_PRIMARY_GAIN
                    and bool(measurements["paired_change_beyond_three_mad"])
                ),
                relative_gain=relative_gain,
                paired_change_beyond_three_mad=measurements[
                    "paired_change_beyond_three_mad"
                ],
            )
        if batch == 1:
            regression = -relative_gain
            _gate(
                gates,
                "batch_1_regression",
                regression <= MAX_BATCH_ONE_REGRESSION,
                relative_regression=regression,
            )
    warmed_allocations = max(
        int(diagnostics["warmed_allocation_count"]),
        measured_candidate_allocations,
    )
    _gate(
        gates,
        "zero_warmed_allocations",
        warmed_allocations == 0,
        warmed_allocation_count=warmed_allocations,
    )
    _validate_regressions(
        campaign["regression_cases"],
        campaign["non_target_performance"],
        gates,
    )

    return {
        "kind": RESULT_KIND,
        "schema_version": RESULT_SCHEMA_VERSION,
        "campaign_sha256": declared_digest,
        "passes": all(bool(gate["passes"]) for gate in gates.values()),
        "gates": gates,
        "candidate": candidate_summary,
        "code_size_diagnostics": {
            "baseline_metric_available": baseline["code_size_metric_available"],
            "candidate_metric_available": candidate["code_size_metric_available"],
            "baseline_portable_source_application_bytes": baseline[
                "portable_source_application_bytes"
            ],
            "candidate_portable_source_application_bytes": candidate[
                "portable_source_application_bytes"
            ],
            "portable_source_metric_is_machine_code": False,
        },
        "batches": batch_summaries,
    }


def _reject_duplicate_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise AcceptanceError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def load_campaign(path: Path) -> Mapping[str, object]:
    """Load JSON while rejecting duplicate keys and non-finite constants."""

    def reject_constant(value: str) -> object:
        raise AcceptanceError(f"non-finite JSON constant {value!r}")

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AcceptanceError(f"cannot load {path}: {exc}") from exc
    return _mapping(value, str(path))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        help="write the acceptance result as canonical JSON",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = audit_campaign(load_campaign(args.campaign))
    except AcceptanceError as exc:
        print(f"compiled-microkernel-acceptance: {exc}", file=sys.stderr)
        return 2
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        sys.stdout.write(rendered)
    else:
        args.output.write_text(rendered, encoding="utf-8")
    return 0 if result["passes"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
