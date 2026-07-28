#!/usr/bin/env python3
"""Produce a source-bound, independent numerical truth witness for one cell.

The producer is deliberately not a benchmark.  It evaluates one retained
validation point and records no wall-clock measurements.  The candidate's
native lane is checked against two precision>=50 executions:

* the candidate artifact's retained exact evaluator, and
* a distinct reference artifact carrying an authenticated pre-optimization or
  independent-oracle semantics contract.

The second artifact is mandatory.  A candidate exact evaluator is useful, but
is not accepted as its own independent reference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, localcontext
from pathlib import Path
from typing import Any, Protocol

from pyamplicol.generation.structural_source_proof import (
    ROLE as STRUCTURAL_SOURCE_PROOF_ROLE,
)
from pyamplicol.generation.structural_source_proof import (
    validate_generation_structural_proof,
)
from tools.performance_report.models import Accuracy, ExecutionMode, Workload

SCHEMA = "pyamplicol-final-source-numerical-truth-v1"
REFERENCE_SCHEMA = "pyamplicol-independent-reference-semantics-v1"
REFERENCE_ROLES = {
    "independent-oracle",
    "pre-optimization-reference",
}
MINIMUM_PRECISION = 50
DEFAULT_PRECISION = 80
DEFAULT_EXACT_AGREEMENT_DIGITS = 50
DEFAULT_NATIVE_RELATIVE_TOLERANCE = Decimal("1e-10")
DEFAULT_NATIVE_ABSOLUTE_TOLERANCE = Decimal("1e-14")
_REVISION = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"[0-9a-f]{64}")


class TruthProducerError(RuntimeError):
    """Independent final-source numerical truth could not be established."""


class _Resolved(Protocol):
    values: Sequence[Sequence[Sequence[complex | Decimal]]]
    helicity_ids: Sequence[str]
    color_ids: Sequence[str]
    color_accuracy: str


class _Runtime(Protocol):
    artifact_id: str
    execution_mode: str

    def evaluate_resolved(
        self,
        momenta: object,
        *,
        helicities: Sequence[str] | None,
        color_flows: Sequence[str] | None,
        precision: int,
    ) -> _Resolved: ...


@dataclass(frozen=True)
class _Artifact:
    root: Path
    manifest: Mapping[str, Any]
    artifact_id: str
    process_id: str
    source_revision: str
    color_accuracy: str
    execution_path: str
    execution_sha256: str
    validation_path: str
    validation_sha256: str
    structural_proof_path: str | None
    structural_proof_sha256: str | None


def canonical_sha256(domain: str, value: object) -> str:
    encoded = json.dumps(
        {"domain": domain, "value": value},
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def comparison_sha256(payload: Mapping[str, Any]) -> str:
    value = dict(payload)
    value.pop("comparison_sha256", None)
    return canonical_sha256("final-source-numerical-truth-v1", value)


def _read_json(path: Path, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise TruthProducerError(f"cannot read {label} {path}: {error}") from error
    if not isinstance(value, Mapping):
        raise TruthProducerError(f"{label} {path} must contain a JSON object")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _required_text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise TruthProducerError(f"{label} must be a non-empty string")
    return value


def _required_sha(value: object, label: str) -> str:
    text = _required_text(value, label)
    if _SHA256.fullmatch(text) is None:
        raise TruthProducerError(f"{label} must be a lowercase SHA-256")
    return text


def _required_revision(value: object, label: str) -> str:
    text = _required_text(value, label)
    if _REVISION.fullmatch(text) is None:
        raise TruthProducerError(f"{label} must be a full lowercase Git revision")
    return text


def _required_int(value: object, label: str, *, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise TruthProducerError(f"{label} must be an integer >= {minimum}")
    return value


def _confined(root: Path, relative: str, label: str) -> Path:
    if not relative:
        raise TruthProducerError(f"{label} path is empty")
    resolved_root = root.resolve()
    path = (resolved_root / relative).resolve()
    try:
        path.relative_to(resolved_root)
    except ValueError as error:
        raise TruthProducerError(f"{label} escapes its artifact: {relative}") from error
    return path


def _payload_record(
    manifest: Mapping[str, Any],
    *,
    path: str | None = None,
    role: str | None = None,
    process_id: str | None = None,
) -> Mapping[str, Any]:
    payloads = manifest.get("payloads")
    if not isinstance(payloads, list):
        raise TruthProducerError("artifact manifest has no payload inventory")
    matches = [
        record
        for record in payloads
        if isinstance(record, Mapping)
        and (path is None or record.get("path") == path)
        and (role is None or record.get("role") == role)
        and (process_id is None or record.get("process_id") == process_id)
    ]
    if len(matches) != 1:
        description = path or f"role={role!r}, process={process_id!r}"
        raise TruthProducerError(
            f"artifact must declare exactly one payload for {description}"
        )
    return matches[0]


def _artifact(
    root: Path,
    *,
    source_revision: str,
    process_id: str | None,
    label: str,
    require_structural_proof: bool,
) -> _Artifact:
    resolved = root.resolve()
    manifest = _read_json(resolved / "artifact.json", f"{label} manifest")
    artifact_id = _required_sha(manifest.get("artifact_id"), f"{label} artifact_id")
    producer = manifest.get("producer")
    if not isinstance(producer, Mapping):
        raise TruthProducerError(f"{label} producer identity is absent")
    actual_revision = _required_revision(
        producer.get("git_revision"),
        f"{label} producer.git_revision",
    )
    if actual_revision != source_revision:
        raise TruthProducerError(
            f"{label} source revision {actual_revision} does not equal "
            f"{source_revision}"
        )
    selected = (
        process_id
        if process_id is not None
        else _required_text(
            manifest.get("default_process_id"),
            f"{label} default_process_id",
        )
    )
    processes = manifest.get("processes")
    if not isinstance(processes, list):
        raise TruthProducerError(f"{label} manifest has no process inventory")
    process = next(
        (
            record
            for record in processes
            if isinstance(record, Mapping) and record.get("id") == selected
        ),
        None,
    )
    if process is None:
        raise TruthProducerError(f"{label} process {selected!r} is absent")
    accuracy = _required_text(
        process.get("color_accuracy"),
        f"{label} process color_accuracy",
    )
    execution = _payload_record(
        manifest,
        role="evaluator-manifest",
        process_id=selected,
    )
    execution_path = _required_text(
        execution.get("path"),
        f"{label} execution path",
    )
    execution_sha = _required_sha(
        execution.get("sha256"),
        f"{label} execution digest",
    )
    validation = _payload_record(
        manifest,
        role="validation-momenta",
        process_id=selected,
    )
    validation_path = _required_text(
        validation.get("path"),
        f"{label} validation path",
    )
    validation_sha = _required_sha(
        validation.get("sha256"),
        f"{label} validation digest",
    )
    for path, expected, name in (
        (execution_path, execution_sha, "execution"),
        (validation_path, validation_sha, "validation momenta"),
    ):
        concrete = _confined(resolved, path, f"{label} {name}")
        if not concrete.is_file() or _sha256_file(concrete) != expected:
            raise TruthProducerError(f"{label} {name} payload is absent or changed")
    structural_path: str | None = None
    structural_sha: str | None = None
    if require_structural_proof:
        structural = _payload_record(
            manifest,
            role=STRUCTURAL_SOURCE_PROOF_ROLE,
            process_id=selected,
        )
        structural_path = _required_text(
            structural.get("path"),
            f"{label} structural proof path",
        )
        structural_sha = _required_sha(
            structural.get("sha256"),
            f"{label} structural proof digest",
        )
        concrete = _confined(
            resolved,
            structural_path,
            f"{label} structural proof",
        )
        if not concrete.is_file() or _sha256_file(concrete) != structural_sha:
            raise TruthProducerError(
                f"{label} structural proof payload is absent or changed"
            )
        native_inputs = _required_sha(
            producer.get("native_build_inputs_sha256"),
            f"{label} producer.native_build_inputs_sha256",
        )
        execution_payload = _read_json(
            _confined(resolved, execution_path, f"{label} execution"),
            f"{label} execution",
        )
        try:
            validate_generation_structural_proof(
                _read_json(concrete, f"{label} structural proof"),
                artifact_root=resolved,
                expected_process_id=selected,
                expected_source_revision=actual_revision,
                expected_native_build_inputs_sha256=native_inputs,
                expected_execution_path=execution_path,
                expected_execution_sha256=execution_sha,
                execution=execution_payload,
            )
        except ValueError as error:
            raise TruthProducerError(
                f"{label} generation structural proof does not recompute"
            ) from error
    return _Artifact(
        root=resolved,
        manifest=manifest,
        artifact_id=artifact_id,
        process_id=selected,
        source_revision=actual_revision,
        color_accuracy=accuracy,
        execution_path=execution_path,
        execution_sha256=execution_sha,
        validation_path=validation_path,
        validation_sha256=validation_sha,
        structural_proof_path=structural_path,
        structural_proof_sha256=structural_sha,
    )


def _canonical_decimal(value: object, label: str) -> str:
    if isinstance(value, bool):
        raise TruthProducerError(f"{label} cannot be boolean")
    try:
        if isinstance(value, Decimal):
            decimal = value
        elif isinstance(value, float):
            if not math.isfinite(value):
                raise TruthProducerError(f"{label} is not finite")
            decimal = Decimal(format(value, ".17g"))
        elif isinstance(value, (int, str)):
            decimal = Decimal(value)
        else:
            raise TruthProducerError(f"{label} is not a decimal scalar")
    except InvalidOperation as error:
        raise TruthProducerError(f"{label} is not a valid decimal") from error
    if not decimal.is_finite():
        raise TruthProducerError(f"{label} is not finite")
    if decimal == 0:
        return "0"
    rendered = format(decimal, "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return rendered


def _validation_payload(artifact: _Artifact, label: str) -> dict[str, Any]:
    raw = _read_json(
        _confined(artifact.root, artifact.validation_path, f"{label} validation"),
        f"{label} validation",
    )
    points = raw.get("points")
    if (
        raw.get("available") is not True
        or not isinstance(points, list)
        or len(points) != 1
    ):
        raise TruthProducerError(
            f"{label} must retain exactly one available validation point"
        )
    particles = points[0]
    if not isinstance(particles, list) or not particles:
        raise TruthProducerError(f"{label} validation point is empty")
    canonical_particles: list[dict[str, Any]] = []
    for particle_index, particle in enumerate(particles):
        if not isinstance(particle, Mapping):
            raise TruthProducerError(
                f"{label} validation particle {particle_index} is malformed"
            )
        pdg = particle.get("pdg")
        momentum = particle.get("momentum")
        if (
            not isinstance(pdg, int)
            or isinstance(pdg, bool)
            or not isinstance(momentum, list)
            or len(momentum) != 4
        ):
            raise TruthProducerError(
                f"{label} validation particle {particle_index} is malformed"
            )
        canonical_particles.append(
            {
                "pdg": pdg,
                "momentum": [
                    _canonical_decimal(
                        component,
                        f"{label} validation particle {particle_index} component",
                    )
                    for component in momentum
                ],
            }
        )
    return {"points": [canonical_particles]}


def _momenta(payload: Mapping[str, Any]) -> tuple[tuple[tuple[str, ...], ...], ...]:
    points = payload["points"]
    assert isinstance(points, list)
    return tuple(
        tuple(tuple(str(value) for value in particle["momentum"]) for particle in point)
        for point in points
    )


def _reference_contract(
    reference: _Artifact,
    contract_relative_path: str,
) -> dict[str, Any]:
    record = _payload_record(reference.manifest, path=contract_relative_path)
    if record.get("role") != "independent-reference-semantics":
        raise TruthProducerError(
            "reference contract must use role 'independent-reference-semantics'"
        )
    expected_contract_sha = _required_sha(
        record.get("sha256"),
        "reference contract payload digest",
    )
    contract_path = _confined(
        reference.root,
        contract_relative_path,
        "reference contract",
    )
    if (
        not contract_path.is_file()
        or _sha256_file(contract_path) != expected_contract_sha
    ):
        raise TruthProducerError("reference contract is absent or changed")
    raw = _read_json(contract_path, "reference contract")
    semantics = raw.get("semantics")
    if (
        raw.get("schema") != REFERENCE_SCHEMA
        or raw.get("status") != "materialized"
        or raw.get("source_revision") != reference.source_revision
        or raw.get("artifact_id") != reference.artifact_id
        or raw.get("process_id") != reference.process_id
        or semantics not in REFERENCE_ROLES
    ):
        raise TruthProducerError(
            "reference contract is stale or does not certify independent semantics"
        )
    payloads = raw.get("semantic_payloads")
    if not isinstance(payloads, list) or not payloads:
        raise TruthProducerError("reference contract has no semantic payloads")
    normalized_payloads: list[dict[str, str]] = []
    content_ids: set[str] = set()
    for index, item in enumerate(payloads):
        if not isinstance(item, Mapping):
            raise TruthProducerError(f"reference semantic payload {index} is malformed")
        relative = _required_text(
            item.get("path"),
            f"reference semantic payload {index} path",
        )
        expected = _required_sha(
            item.get("sha256"),
            f"reference semantic payload {index} digest",
        )
        declared = _payload_record(reference.manifest, path=relative)
        if declared.get("sha256") != expected:
            raise TruthProducerError(
                f"reference semantic payload {relative} disagrees with manifest"
            )
        concrete = _confined(reference.root, relative, "reference semantic payload")
        if not concrete.is_file() or _sha256_file(concrete) != expected:
            raise TruthProducerError(
                f"reference semantic payload {relative} is absent or changed"
            )
        if expected in content_ids:
            raise TruthProducerError(
                "reference semantic payload inventory repeats a content identity"
            )
        content_ids.add(expected)
        normalized_payloads.append({"path": relative, "sha256": expected})
    normalized = {
        "schema": REFERENCE_SCHEMA,
        "status": "materialized",
        "semantics": semantics,
        "source_revision": reference.source_revision,
        "artifact_id": reference.artifact_id,
        "process_id": reference.process_id,
        "contract_sha256": expected_contract_sha,
        "semantic_payloads": normalized_payloads,
    }
    normalized["semantic_identity_sha256"] = canonical_sha256(
        "independent-reference-semantics-v1",
        normalized,
    )
    return normalized


def _selector_payload(
    *,
    workload: Workload,
    helicity_ids: Sequence[str],
    color_ids: Sequence[str],
) -> dict[str, Any]:
    helicities = list(helicity_ids)
    colors = list(color_ids)
    if (
        len(set(helicities)) != len(helicities)
        or len(set(colors)) != len(colors)
        or any(not value for value in (*helicities, *colors))
    ):
        raise TruthProducerError("selector IDs must be non-empty and unique")
    if workload is Workload.SELECTED_FLOW:
        if len(helicities) != 1 or len(colors) != 1:
            raise TruthProducerError(
                "selected-flow truth requires exactly one helicity and color selector"
            )
    elif helicities or colors:
        raise TruthProducerError(
            f"{workload.value} truth must not apply resolved-axis selectors"
        )
    payload: dict[str, Any] = {
        "workload": workload.value,
        "helicity_ids": helicities,
        "color_ids": colors,
    }
    payload["selector_sha256"] = canonical_sha256(
        "final-source-selectors-v1",
        payload,
    )
    return payload


def _axes(resolved: _Resolved) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "color_accuracy": str(resolved.color_accuracy),
        "helicity_ids": [str(value) for value in resolved.helicity_ids],
        "color_ids": [str(value) for value in resolved.color_ids],
    }
    payload["resolved_axes_sha256"] = canonical_sha256(
        "final-source-resolved-axes-v1",
        payload,
    )
    return payload


def _exact_values(resolved: _Resolved, label: str) -> list[list[list[str]]]:
    output: list[list[list[str]]] = []
    for point_index, point in enumerate(resolved.values):
        rows: list[list[str]] = []
        for helicity_index, row in enumerate(point):
            values: list[str] = []
            for color_index, value in enumerate(row):
                if not isinstance(value, Decimal):
                    raise TruthProducerError(
                        f"{label}[{point_index},{helicity_index},{color_index}] "
                        "is not independently materialized Decimal output"
                    )
                values.append(
                    _canonical_decimal(
                        value,
                        f"{label}[{point_index},{helicity_index},{color_index}]",
                    )
                )
            rows.append(values)
        output.append(rows)
    return output


def _native_values(
    resolved: _Resolved,
    label: str,
) -> list[list[list[list[str]]]]:
    output: list[list[list[list[str]]]] = []
    for point_index, point in enumerate(resolved.values):
        rows: list[list[list[str]]] = []
        for helicity_index, row in enumerate(point):
            values: list[list[str]] = []
            for color_index, value in enumerate(row):
                complex_value = complex(value)
                if not math.isfinite(complex_value.real) or not math.isfinite(
                    complex_value.imag
                ):
                    raise TruthProducerError(
                        f"{label}[{point_index},{helicity_index},{color_index}] "
                        "is not finite"
                    )
                values.append(
                    [
                        _canonical_decimal(complex_value.real, f"{label} real"),
                        _canonical_decimal(complex_value.imag, f"{label} imaginary"),
                    ]
                )
            rows.append(values)
        output.append(rows)
    return output


def _flatten_exact(values: object, label: str) -> list[Decimal]:
    if not isinstance(values, list) or not values:
        raise TruthProducerError(f"{label} must contain at least one point")
    flattened: list[Decimal] = []
    try:
        for point in values:
            if not isinstance(point, list) or not point:
                raise TruthProducerError(f"{label} has an empty helicity axis")
            for row in point:
                if not isinstance(row, list) or not row:
                    raise TruthProducerError(f"{label} has an empty color axis")
                for value in row:
                    if not isinstance(value, str):
                        raise TruthProducerError(
                            f"{label} exact values must be decimal strings"
                        )
                    decimal = Decimal(value)
                    if not decimal.is_finite():
                        raise TruthProducerError(f"{label} contains non-finite values")
                    flattened.append(decimal)
    except InvalidOperation as error:
        raise TruthProducerError(f"{label} contains invalid decimals") from error
    return flattened


def _flatten_native(values: object, label: str) -> list[tuple[Decimal, Decimal]]:
    if not isinstance(values, list) or not values:
        raise TruthProducerError(f"{label} must contain at least one point")
    flattened: list[tuple[Decimal, Decimal]] = []
    try:
        for point in values:
            if not isinstance(point, list) or not point:
                raise TruthProducerError(f"{label} has an empty helicity axis")
            for row in point:
                if not isinstance(row, list) or not row:
                    raise TruthProducerError(f"{label} has an empty color axis")
                for value in row:
                    if (
                        not isinstance(value, list)
                        or len(value) != 2
                        or not all(isinstance(item, str) for item in value)
                    ):
                        raise TruthProducerError(
                            f"{label} native values must be [real, imaginary] strings"
                        )
                    pair = (Decimal(value[0]), Decimal(value[1]))
                    if not all(item.is_finite() for item in pair):
                        raise TruthProducerError(f"{label} contains non-finite values")
                    flattened.append(pair)
    except InvalidOperation as error:
        raise TruthProducerError(f"{label} contains invalid decimals") from error
    return flattened


def _agreement(
    candidate_exact: object,
    candidate_native: object,
    reference_exact: object,
    *,
    precision: int,
    exact_agreement_digits: int,
    native_relative_tolerance: Decimal,
    native_absolute_tolerance: Decimal,
) -> dict[str, Any]:
    candidate = _flatten_exact(candidate_exact, "candidate_exact.values")
    native = _flatten_native(candidate_native, "candidate_native.values")
    reference = _flatten_exact(reference_exact, "reference_exact.values")
    if len(candidate) != len(reference) or len(native) != len(reference):
        raise TruthProducerError("candidate/reference resolved value shapes differ")
    if exact_agreement_digits > precision - 5:
        raise TruthProducerError(
            "exact agreement digits must leave at least five guard digits"
        )
    exact_tolerance = Decimal(1).scaleb(-exact_agreement_digits)
    exact_max_absolute = Decimal(0)
    exact_max_relative = Decimal(0)
    native_max_absolute = Decimal(0)
    native_max_relative = Decimal(0)
    native_max_imaginary = Decimal(0)
    exact_passes = True
    native_passes = True
    with localcontext() as context:
        context.prec = max(precision + 20, 100)
        for exact_value, (native_real, native_imaginary), reference_value in zip(
            candidate,
            native,
            reference,
            strict=True,
        ):
            exact_absolute = abs(exact_value - reference_value)
            exact_relative = exact_absolute / max(
                abs(exact_value),
                abs(reference_value),
                Decimal(1).scaleb(-precision),
            )
            native_absolute = abs(native_real - reference_value)
            native_relative = native_absolute / max(
                abs(native_real),
                abs(reference_value),
                Decimal("1e-300"),
            )
            exact_max_absolute = max(exact_max_absolute, exact_absolute)
            exact_max_relative = max(exact_max_relative, exact_relative)
            native_max_absolute = max(native_max_absolute, native_absolute)
            native_max_relative = max(native_max_relative, native_relative)
            native_max_imaginary = max(
                native_max_imaginary,
                abs(native_imaginary),
            )
            exact_passes = exact_passes and (
                exact_absolute <= exact_tolerance or exact_relative <= exact_tolerance
            )
            native_passes = (
                native_passes
                and (
                    native_absolute <= native_absolute_tolerance
                    or native_relative <= native_relative_tolerance
                )
                and abs(native_imaginary) <= native_absolute_tolerance
            )
    return {
        "component_count": len(reference),
        "exact_agreement_decimal_digits": exact_agreement_digits,
        "exact_absolute_tolerance": _canonical_decimal(
            exact_tolerance,
            "exact tolerance",
        ),
        "native_relative_tolerance": _canonical_decimal(
            native_relative_tolerance,
            "native relative tolerance",
        ),
        "native_absolute_tolerance": _canonical_decimal(
            native_absolute_tolerance,
            "native absolute tolerance",
        ),
        "candidate_exact_vs_reference": {
            "maximum_absolute_difference": _canonical_decimal(
                exact_max_absolute,
                "exact maximum absolute difference",
            ),
            "maximum_relative_difference": _canonical_decimal(
                exact_max_relative,
                "exact maximum relative difference",
            ),
            "passes": exact_passes,
        },
        "candidate_native_vs_reference": {
            "maximum_absolute_difference": _canonical_decimal(
                native_max_absolute,
                "native maximum absolute difference",
            ),
            "maximum_relative_difference": _canonical_decimal(
                native_max_relative,
                "native maximum relative difference",
            ),
            "maximum_imaginary_magnitude": _canonical_decimal(
                native_max_imaginary,
                "native maximum imaginary magnitude",
            ),
            "passes": native_passes,
        },
        "passes": exact_passes and native_passes,
    }


def _value_record(
    *,
    precision: int,
    executor: str,
    values: object,
    domain: str,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "precision_decimal_digits": precision,
        "executor": executor,
        "values": values,
    }
    record["values_sha256"] = canonical_sha256(domain, values)
    return record


def validate_witness_payload(
    raw: Mapping[str, Any],
    *,
    expected_source_revision: str | None = None,
    expected_cell_id: str | None = None,
    expected_mode: str | None = None,
    expected_accuracy: str | None = None,
    expected_workload: str | None = None,
    expected_candidate_artifact_id: str | None = None,
    expected_structural_proof_sha256: str | None = None,
) -> dict[str, Any]:
    """Recompute every numerical digest and agreement in a serialized witness."""

    payload = dict(raw)
    if payload.get("schema") != SCHEMA or payload.get("status") != "ok":
        raise TruthProducerError("numerical truth witness schema/status is invalid")
    revision = _required_revision(payload.get("source_revision"), "source_revision")
    if expected_source_revision is not None and revision != expected_source_revision:
        raise TruthProducerError("numerical truth source revision mismatch")
    for name, expected in (
        ("cell_id", expected_cell_id),
        ("mode", expected_mode),
        ("accuracy", expected_accuracy),
        ("workload", expected_workload),
    ):
        actual = _required_text(payload.get(name), name)
        if expected is not None and actual != expected:
            raise TruthProducerError(f"numerical truth {name} mismatch")
    precision = _required_int(
        payload.get("precision_decimal_digits"),
        "precision_decimal_digits",
        minimum=MINIMUM_PRECISION,
    )
    candidate_artifact = payload.get("candidate_artifact")
    reference_artifact = payload.get("reference_artifact")
    if not isinstance(candidate_artifact, Mapping) or not isinstance(
        reference_artifact,
        Mapping,
    ):
        raise TruthProducerError("candidate/reference artifact identity is absent")
    candidate_id = _required_sha(
        candidate_artifact.get("artifact_id"),
        "candidate artifact_id",
    )
    reference_id = _required_sha(
        reference_artifact.get("artifact_id"),
        "reference artifact_id",
    )
    if candidate_id == reference_id:
        raise TruthProducerError(
            "candidate and independent reference repeat one content identity"
        )
    if (
        expected_candidate_artifact_id is not None
        and candidate_id != expected_candidate_artifact_id
    ):
        raise TruthProducerError("candidate artifact identity mismatch")
    structural_proof_sha256 = _required_sha(
        candidate_artifact.get("structural_proof_sha256"),
        "candidate structural proof digest",
    )
    _required_text(
        candidate_artifact.get("structural_proof_path"),
        "candidate structural proof path",
    )
    if (
        expected_structural_proof_sha256 is not None
        and structural_proof_sha256 != expected_structural_proof_sha256
    ):
        raise TruthProducerError("candidate structural proof identity mismatch")
    if candidate_artifact.get("source_revision") != revision or (
        reference_artifact.get("source_revision") != revision
    ):
        raise TruthProducerError("artifact identity is not source-bound")
    contract = reference_artifact.get("semantics_contract")
    if (
        not isinstance(contract, Mapping)
        or contract.get("schema") != REFERENCE_SCHEMA
        or contract.get("status") != "materialized"
        or contract.get("semantics") not in REFERENCE_ROLES
        or contract.get("source_revision") != revision
        or contract.get("artifact_id") != reference_id
    ):
        raise TruthProducerError("independent reference semantics are not materialized")
    semantic_payloads = contract.get("semantic_payloads")
    if not isinstance(semantic_payloads, list) or not semantic_payloads:
        raise TruthProducerError("independent reference semantic payloads are absent")
    content_ids: set[str] = set()
    for index, item in enumerate(semantic_payloads):
        if not isinstance(item, Mapping):
            raise TruthProducerError(f"reference semantic payload {index} is malformed")
        _required_text(item.get("path"), f"semantic payload {index} path")
        content_id = _required_sha(
            item.get("sha256"),
            f"semantic payload {index} digest",
        )
        if content_id in content_ids:
            raise TruthProducerError(
                "reference semantic payloads repeat one content identity"
            )
        content_ids.add(content_id)
    semantic_normalized = dict(contract)
    declared_semantic_identity = _required_sha(
        semantic_normalized.pop("semantic_identity_sha256", None),
        "reference semantic identity",
    )
    if declared_semantic_identity != canonical_sha256(
        "independent-reference-semantics-v1",
        semantic_normalized,
    ):
        raise TruthProducerError("reference semantic identity digest mismatch")

    momenta = payload.get("validation_momenta")
    selectors = payload.get("selectors")
    axes = payload.get("resolved_axes")
    if not all(isinstance(value, Mapping) for value in (momenta, selectors, axes)):
        raise TruthProducerError("momenta/selectors/resolved axes are absent")
    momenta_map = dict(momenta)
    declared_momenta = _required_sha(
        momenta_map.pop("validation_momenta_sha256", None),
        "validation momenta digest",
    )
    if declared_momenta != canonical_sha256(
        "final-source-validation-momenta-v1",
        momenta_map,
    ):
        raise TruthProducerError("validation momenta digest mismatch")
    selector_map = dict(selectors)
    declared_selectors = _required_sha(
        selector_map.pop("selector_sha256", None),
        "selector digest",
    )
    if declared_selectors != canonical_sha256(
        "final-source-selectors-v1",
        selector_map,
    ):
        raise TruthProducerError("selector digest mismatch")
    axes_map = dict(axes)
    declared_axes = _required_sha(
        axes_map.pop("resolved_axes_sha256", None),
        "resolved axes digest",
    )
    if declared_axes != canonical_sha256(
        "final-source-resolved-axes-v1",
        axes_map,
    ):
        raise TruthProducerError("resolved axes digest mismatch")
    if axes_map.get("color_accuracy") != payload.get("accuracy"):
        raise TruthProducerError("resolved axes accuracy mismatch")

    records: dict[str, Mapping[str, Any]] = {}
    for name, domain in (
        ("candidate_native", "final-source-candidate-native-values-v1"),
        ("candidate_exact", "final-source-candidate-exact-values-v1"),
        ("reference_exact", "final-source-reference-exact-values-v1"),
    ):
        record = payload.get(name)
        if not isinstance(record, Mapping):
            raise TruthProducerError(f"{name} record is absent")
        record_precision = _required_int(
            record.get("precision_decimal_digits"),
            f"{name} precision",
            minimum=16,
        )
        if name == "candidate_native":
            if record_precision != 16:
                raise TruthProducerError("candidate native precision must be 16")
        elif record_precision != precision:
            raise TruthProducerError(f"{name} precision mismatch")
        values = record.get("values")
        declared = _required_sha(record.get("values_sha256"), f"{name} values digest")
        if declared != canonical_sha256(domain, values):
            raise TruthProducerError(f"{name} values digest mismatch")
        records[name] = record

    agreement = payload.get("agreement")
    if not isinstance(agreement, Mapping):
        raise TruthProducerError("numerical agreement payload is absent")
    exact_digits = _required_int(
        agreement.get("exact_agreement_decimal_digits"),
        "exact agreement digits",
        minimum=1,
    )
    try:
        native_relative = Decimal(
            _required_text(
                agreement.get("native_relative_tolerance"),
                "native relative tolerance",
            )
        )
        native_absolute = Decimal(
            _required_text(
                agreement.get("native_absolute_tolerance"),
                "native absolute tolerance",
            )
        )
    except InvalidOperation as error:
        raise TruthProducerError("native tolerance is not decimal") from error
    recomputed = _agreement(
        records["candidate_exact"].get("values"),
        records["candidate_native"].get("values"),
        records["reference_exact"].get("values"),
        precision=precision,
        exact_agreement_digits=exact_digits,
        native_relative_tolerance=native_relative,
        native_absolute_tolerance=native_absolute,
    )
    if dict(agreement) != recomputed or recomputed["passes"] is not True:
        raise TruthProducerError("numerical agreement payload does not recompute")
    declared_comparison = _required_sha(
        payload.get("comparison_sha256"),
        "comparison_sha256",
    )
    if declared_comparison != comparison_sha256(payload):
        raise TruthProducerError("comparison_sha256 does not authenticate the witness")
    return payload


def build_witness(
    candidate_root: Path,
    reference_root: Path,
    *,
    reference_contract_path: str,
    cell_id: str,
    source_revision: str,
    mode: ExecutionMode,
    accuracy: Accuracy,
    workload: Workload,
    helicity_ids: Sequence[str] = (),
    color_ids: Sequence[str] = (),
    candidate_process_id: str | None = None,
    reference_process_id: str | None = None,
    precision: int = DEFAULT_PRECISION,
    exact_agreement_digits: int = DEFAULT_EXACT_AGREEMENT_DIGITS,
    native_relative_tolerance: Decimal = DEFAULT_NATIVE_RELATIVE_TOLERANCE,
    native_absolute_tolerance: Decimal = DEFAULT_NATIVE_ABSOLUTE_TOLERANCE,
    runtime_loader: Callable[..., _Runtime] | None = None,
) -> dict[str, Any]:
    """Materialize and authenticate one structural-only comparison witness."""

    revision = _required_revision(source_revision, "source_revision")
    if precision < MINIMUM_PRECISION:
        raise TruthProducerError(
            f"comparison precision must be >= {MINIMUM_PRECISION} decimal digits"
        )
    candidate = _artifact(
        candidate_root,
        source_revision=revision,
        process_id=candidate_process_id,
        label="candidate",
        require_structural_proof=True,
    )
    reference = _artifact(
        reference_root,
        source_revision=revision,
        process_id=reference_process_id,
        label="reference",
        require_structural_proof=False,
    )
    if candidate.artifact_id == reference.artifact_id:
        raise TruthProducerError(
            "candidate and independent reference artifacts have one content identity"
        )
    if candidate.color_accuracy != accuracy.value or (
        reference.color_accuracy != accuracy.value
    ):
        raise TruthProducerError("candidate/reference color accuracy mismatch")
    reference_contract = _reference_contract(reference, reference_contract_path)
    candidate_momenta = _validation_payload(candidate, "candidate")
    reference_momenta = _validation_payload(reference, "reference")
    if candidate_momenta != reference_momenta:
        raise TruthProducerError(
            "candidate/reference validation momenta or external PDG order differ"
        )
    selector_payload = _selector_payload(
        workload=workload,
        helicity_ids=helicity_ids,
        color_ids=color_ids,
    )
    if runtime_loader is None:
        from pyamplicol import Runtime

        runtime_loader = Runtime.load
    candidate_runtime = runtime_loader(
        candidate.root,
        process=candidate.process_id,
    )
    reference_runtime = runtime_loader(
        reference.root,
        process=reference.process_id,
    )
    if candidate_runtime.artifact_id != candidate.artifact_id or (
        reference_runtime.artifact_id != reference.artifact_id
    ):
        raise TruthProducerError("loaded runtime artifact identity mismatch")
    if candidate_runtime.execution_mode != mode.value:
        raise TruthProducerError("candidate runtime execution mode mismatch")
    selectors = {
        "helicities": tuple(helicity_ids) or None,
        "color_flows": tuple(color_ids) or None,
    }
    points = _momenta(candidate_momenta)
    try:
        candidate_native_resolved = candidate_runtime.evaluate_resolved(
            points,
            precision=16,
            **selectors,
        )
        candidate_exact_resolved = candidate_runtime.evaluate_resolved(
            points,
            precision=precision,
            **selectors,
        )
        reference_exact_resolved = reference_runtime.evaluate_resolved(
            points,
            precision=precision,
            **selectors,
        )
    except Exception as error:
        raise TruthProducerError(
            "independent precision comparison could not be materialized"
        ) from error
    candidate_axes = _axes(candidate_exact_resolved)
    if _axes(candidate_native_resolved) != candidate_axes or (
        _axes(reference_exact_resolved) != candidate_axes
    ):
        raise TruthProducerError("candidate/reference resolved axes differ")
    candidate_native_values = _native_values(
        candidate_native_resolved,
        "candidate native",
    )
    candidate_exact_values = _exact_values(
        candidate_exact_resolved,
        "candidate exact",
    )
    reference_exact_values = _exact_values(
        reference_exact_resolved,
        "reference exact",
    )
    agreement = _agreement(
        candidate_exact_values,
        candidate_native_values,
        reference_exact_values,
        precision=precision,
        exact_agreement_digits=exact_agreement_digits,
        native_relative_tolerance=native_relative_tolerance,
        native_absolute_tolerance=native_absolute_tolerance,
    )
    if agreement["passes"] is not True:
        raise TruthProducerError(
            "candidate does not agree with the independent reference"
        )
    momenta_record: dict[str, Any] = {
        "candidate_payload_sha256": candidate.validation_sha256,
        "reference_payload_sha256": reference.validation_sha256,
        **candidate_momenta,
    }
    momenta_record["validation_momenta_sha256"] = canonical_sha256(
        "final-source-validation-momenta-v1",
        momenta_record,
    )
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "ok",
        "cell_id": cell_id,
        "source_revision": revision,
        "mode": mode.value,
        "accuracy": accuracy.value,
        "workload": workload.value,
        "precision_decimal_digits": precision,
        "validation_momenta": momenta_record,
        "selectors": selector_payload,
        "resolved_axes": candidate_axes,
        "candidate_artifact": {
            "artifact_id": candidate.artifact_id,
            "source_revision": revision,
            "process_id": candidate.process_id,
            "execution_mode": candidate_runtime.execution_mode,
            "execution_payload_path": candidate.execution_path,
            "execution_payload_sha256": candidate.execution_sha256,
            "structural_proof_path": candidate.structural_proof_path,
            "structural_proof_sha256": candidate.structural_proof_sha256,
        },
        "reference_artifact": {
            "artifact_id": reference.artifact_id,
            "source_revision": revision,
            "process_id": reference.process_id,
            "execution_mode": reference_runtime.execution_mode,
            "execution_payload_path": reference.execution_path,
            "execution_payload_sha256": reference.execution_sha256,
            "semantics_contract": reference_contract,
        },
        "candidate_native": _value_record(
            precision=16,
            executor=f"native-{mode.value}",
            values=candidate_native_values,
            domain="final-source-candidate-native-values-v1",
        ),
        "candidate_exact": _value_record(
            precision=precision,
            executor=f"retained-exact-{mode.value}",
            values=candidate_exact_values,
            domain="final-source-candidate-exact-values-v1",
        ),
        "reference_exact": _value_record(
            precision=precision,
            executor=str(reference_contract["semantics"]),
            values=reference_exact_values,
            domain="final-source-reference-exact-values-v1",
        ),
        "agreement": agreement,
    }
    payload["comparison_sha256"] = comparison_sha256(payload)
    return validate_witness_payload(
        payload,
        expected_source_revision=revision,
        expected_cell_id=cell_id,
        expected_mode=mode.value,
        expected_accuracy=accuracy.value,
        expected_workload=workload.value,
        expected_candidate_artifact_id=candidate.artifact_id,
        expected_structural_proof_sha256=candidate.structural_proof_sha256,
    )


def _atomic_write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="ascii") as stream:
            json.dump(
                payload,
                stream,
                ensure_ascii=True,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-artifact", type=Path, required=True)
    parser.add_argument("--reference-artifact", type=Path, required=True)
    parser.add_argument("--reference-contract-path", required=True)
    parser.add_argument("--cell-id", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument(
        "--mode",
        choices=[item.value for item in ExecutionMode],
        required=True,
    )
    parser.add_argument(
        "--accuracy",
        choices=[item.value for item in Accuracy],
        required=True,
    )
    parser.add_argument(
        "--workload",
        choices=[item.value for item in Workload],
        required=True,
    )
    parser.add_argument("--helicity-id", action="append", default=[])
    parser.add_argument("--color-id", action="append", default=[])
    parser.add_argument("--candidate-process-id")
    parser.add_argument("--reference-process-id")
    parser.add_argument("--precision", type=int, default=DEFAULT_PRECISION)
    parser.add_argument(
        "--exact-agreement-digits",
        type=int,
        default=DEFAULT_EXACT_AGREEMENT_DIGITS,
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    try:
        payload = build_witness(
            arguments.candidate_artifact,
            arguments.reference_artifact,
            reference_contract_path=arguments.reference_contract_path,
            cell_id=arguments.cell_id,
            source_revision=arguments.source_revision,
            mode=ExecutionMode(arguments.mode),
            accuracy=Accuracy(arguments.accuracy),
            workload=Workload(arguments.workload),
            helicity_ids=arguments.helicity_id,
            color_ids=arguments.color_id,
            candidate_process_id=arguments.candidate_process_id,
            reference_process_id=arguments.reference_process_id,
            precision=arguments.precision,
            exact_agreement_digits=arguments.exact_agreement_digits,
        )
    except TruthProducerError as error:
        print(f"final-source numerical truth unavailable: {error}")
        return 2
    _atomic_write(arguments.output, payload)
    print(
        json.dumps(
            {
                "cell_id": payload["cell_id"],
                "comparison_sha256": payload["comparison_sha256"],
                "precision_decimal_digits": payload["precision_decimal_digits"],
                "status": payload["status"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
