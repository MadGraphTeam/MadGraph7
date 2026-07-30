# SPDX-License-Identifier: 0BSD
"""Authenticated high-precision current reuse for recurrence Direct plans.

The first native lowering is an unpublished baseline.  This module evaluates
that exact plan at disjoint Decimal/Symbolica probe domains, certifies only
equal, opposite, and zero relations, and emits canonical evidence for a
second native lowering.  No binary64 runtime result participates in discovery.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from decimal import ROUND_HALF_EVEN, Decimal, localcontext
from fractions import Fraction
from math import copysign, isfinite
from typing import TYPE_CHECKING, Literal, cast

from pyamplicol.api.errors import ArtifactError
from pyamplicol.models.base import Model
from pyamplicol.runtime.recurrence_exact._execution import (
    _evaluate_contracted_point,
    _evaluate_replay_point,
    _evaluate_union_point,
)
from pyamplicol.runtime.recurrence_exact._plan import _RecurrenceExactPlan
from pyamplicol.runtime.recurrence_exact._plan_v2 import (
    DIRECT_NONE_U32,
    _Current,
    _RecurrenceExactSectionsV1,
)
from pyamplicol.runtime.symbolica_exact import _ComplexDecimal

from .dag_equivalence import (
    NUMERICAL_CURRENT_RELATION_WARNING,
    NUMERICAL_CURRENT_RELATION_WARNING_CODE,
)
from .numerical_candidate_index import (
    NumericalObservationCandidateIndex,
    build_numerical_observation_candidate_index,
    numerical_observation_tolerance_window_ids,
)
from .validation import ValidationPointRecord, build_process_validation_point

if TYPE_CHECKING:
    from pyamplicol.processes.ir import CanonicalProcessIR

_CAPTURE_ABI = "pyamplicol-recurrence-current-observation-capture-v2"
_EVIDENCE_ABI = "pyamplicol-recurrence-numerical-current-evidence-v3"
_SOURCE_ABI = "pyamplicol-recurrence-numerical-current-source-v3"
_WARMUP_ABI = "pyamplicol-recurrence-numerical-current-warmup-v1"
_RELATION_SET_ABI = "pyamplicol-authenticated-numerical-current-relation-set-v2"
_OBSERVATION_BATCH_ABI = "pyamplicol-recurrence-current-observation-batch-v2"
_PARAMETER_CONTEXT_ABI = "pyamplicol-recurrence-parameter-context-v1"
_PARAMETER_SCHEMA_ABI = "pyamplicol-recurrence-runtime-parameter-schema-v2"
_RELATION_OBSERVATION_ABI = "pyamplicol-recurrence-relation-observation-v2"
_DECISION_CHAIN_ABI = "pyamplicol-recurrence-numerical-decision-chain-v1"
_REJECTION_CHAIN_ABI = (
    "pyamplicol-recurrence-rejected-numerical-decision-chain-v1"
)
_PERSISTED_EVIDENCE_ABI = (
    "pyamplicol-recurrence-numerical-persisted-evidence-v1"
)
_MAX_RAW_EVIDENCE_MEMORY_BYTES = 1 << 30
# The producer-side peak ends when the canonical bytes have been encoded and
# the complete Decimal graphs are detached.  The per-scalar allowance covers
# the Python Decimal/tuple capture graph and encoder temporaries; the row
# allowance covers capture dictionaries and rows.  Native independently
# authenticates this shape limit, then applies its separate streaming-consumer
# bound without materializing observation JSON or exact-rational graphs.
_RAW_EVIDENCE_SCALAR_RESIDENT_BYTES = 640
_RAW_EVIDENCE_ROW_RESIDENT_BYTES = 512
_RAW_EVIDENCE_FIXED_RESERVE_BYTES = 32 << 20
_RAW_EVIDENCE_WIRE_PEAK_COPIES = 2
_RAW_STREAM_METADATA_COPIES = 2
_RAW_STREAM_BYTES_PER_ROW_OFFSETS = 16
_RAW_STREAM_BYTES_PER_TEXT_REFERENCE = 16
_RAW_STREAM_BYTES_PER_CURRENT_INDEX = 512
_RAW_STREAM_BYTES_PER_RATIONAL = 320
_RAW_STREAM_BYTES_PER_METADATA_TOKEN = 80
_RAW_STREAM_PARAMETER_RATIONAL_COPIES = 4
_MIN_RAW_EVIDENCE_WIRE_BYTES = 1 << 20
# This global ceiling is shared with the native pre-metadata lexical budget.  A
# concrete capture may receive a smaller limit from its scalar/row geometry.
_MAX_RAW_EVIDENCE_BYTES = 157_853_696
_MAX_RAW_EVIDENCE_DECIMAL_BYTES = 16_384
_MAX_PERSISTED_EVIDENCE_BYTES = 64 << 20
_RECURRENCE_CERTIFICATE_ALGORITHM = (
    "authenticated-independent-recursive-decimal-raw-probes-v2"
)
_MAX_REJECTED_DIAGNOSTICS = 32
_MAX_RAW_NUMERICAL_HYPOTHESES = 1_000_000
_CANDIDATE_INDEX_ALGORITHM = "complete-contract-anchor-tolerance-window-v1"

_RelationKind = Literal["equal", "opposite", "zero"]
_Mode = Literal["diagnostic", "certified-reuse"]
_Point = tuple[tuple[Decimal, Decimal, Decimal, Decimal], ...]
_CurrentContract = tuple[object, ...]


class _CanonicalJsonStreamValue:
    """A value that writes canonical JSON without first building a JSON graph."""

    def iter_canonical_json_chunks(self) -> Iterator[bytes]:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class _ObservationValuesStream(_CanonicalJsonStreamValue):
    values: Sequence[_ComplexDecimal]

    def iter_canonical_json_chunks(self) -> Iterator[bytes]:
        yield b"["
        for index, (real, imaginary) in enumerate(self.values):
            if index:
                yield b","
            yield b"["
            yield from _iter_canonical_json_chunks(_decimal_string(real))
            yield b","
            yield from _iter_canonical_json_chunks(_decimal_string(imaginary))
            yield b"]"
        yield b"]"


@dataclass(frozen=True, slots=True)
class _ObservationRowsStream(_CanonicalJsonStreamValue):
    observations: Mapping[int, Sequence[_ComplexDecimal]]
    dimensions: Mapping[int, int]

    def iter_canonical_json_chunks(self) -> Iterator[bytes]:
        yield b"["
        for index, current_id in enumerate(sorted(self.observations)):
            if index:
                yield b","
            yield from _iter_canonical_json_chunks(
                {
                    "current_id": current_id,
                    "dimension": self.dimensions[current_id],
                    "values": _ObservationValuesStream(
                        self.observations[current_id]
                    ),
                }
            )
        yield b"]"


@dataclass(frozen=True, slots=True)
class _DimensionRowsStream(_CanonicalJsonStreamValue):
    dimensions: Mapping[int, int]

    def iter_canonical_json_chunks(self) -> Iterator[bytes]:
        yield b"["
        for index, (current_id, dimension) in enumerate(
            sorted(self.dimensions.items())
        ):
            if index:
                yield b","
            yield from _iter_canonical_json_chunks(
                {"current_id": current_id, "dimension": dimension}
            )
        yield b"]"


@dataclass(frozen=True, slots=True)
class _CertificateRowsStream(_CanonicalJsonStreamValue):
    certificates: Sequence[RecurrenceNumericalCurrentCertificate]

    def iter_canonical_json_chunks(self) -> Iterator[bytes]:
        yield b"["
        for index, certificate in enumerate(self.certificates):
            if index:
                yield b","
            yield from _iter_canonical_json_chunks(certificate.to_json_dict())
        yield b"]"


@dataclass(frozen=True, slots=True)
class _MappingRowsStream(_CanonicalJsonStreamValue):
    certificates: Sequence[RecurrenceNumericalCurrentCertificate]

    def iter_canonical_json_chunks(self) -> Iterator[bytes]:
        yield b"["
        for index, certificate in enumerate(self.certificates):
            if index:
                yield b","
            yield from _iter_canonical_json_chunks(_mapping_payload(certificate))
        yield b"]"


@dataclass(frozen=True, slots=True)
class RecurrenceCurrentObservationCapture:
    """Complete point-major exact current observations."""

    precision_digits: int
    points: tuple[ValidationPointRecord, ...]
    point_sha256s: tuple[str, ...]
    kinematic_sha256s: tuple[str, ...]
    kinematic_binary64: tuple[tuple[str, ...], ...]
    selector_contexts: tuple[object, ...]
    context_sha256s: tuple[str, ...]
    parameter_contexts: tuple[tuple[Decimal, ...], ...]
    parameter_context_sha256s: tuple[str, ...]
    observations: Mapping[int, tuple[_ComplexDecimal, ...]]
    current_dimensions: Mapping[int, int]
    runtime_parameter_schema_sha256: str
    source_semantics_sha256: str
    observation_batch_sha256: str
    capture_contract_sha256: str
    context_policy: str
    full_current_count: int | None = None
    complete_observations_retained: bool = True

    @property
    def point_count(self) -> int:
        return len(self.points)

    @property
    def current_count(self) -> int:
        return (
            len(self.observations)
            if self.full_current_count is None
            else self.full_current_count
        )

    def detached_for_replay(
        self,
        current_ids: Sequence[int],
    ) -> RecurrenceCurrentObservationCapture:
        """Retain only rows needed by the persisted certificate replay."""

        selected = tuple(sorted(set(current_ids)))
        if any(current_id not in self.observations for current_id in selected):
            raise ValueError(
                "certificate replay references an absent recurrence current"
            )
        return replace(
            self,
            observations={
                current_id: self.observations[current_id]
                for current_id in selected
            },
            full_current_count=self.current_count,
            complete_observations_retained=False,
        )

    def to_provenance_dict(self) -> dict[str, object]:
        return {
            "abi": _CAPTURE_ABI,
            "precision_digits": self.precision_digits,
            "point_count": self.point_count,
            "point_sha256s": list(self.point_sha256s),
            "kinematic_sha256s": list(self.kinematic_sha256s),
            "parameter_contexts": [
                [_decimal_string(value) for value in context]
                for context in self.parameter_contexts
            ],
            "parameter_context_sha256s": list(self.parameter_context_sha256s),
            "context_sha256s": list(self.context_sha256s),
            "points": [point.to_mapping() for point in self.points],
            "current_count": self.current_count,
            "runtime_parameter_schema_sha256": (
                self.runtime_parameter_schema_sha256
            ),
            "source_semantics_sha256": self.source_semantics_sha256,
            "observation_batch_sha256": self.observation_batch_sha256,
            "capture_contract_sha256": self.capture_contract_sha256,
            "context_policy": self.context_policy,
            "complete_current_components": True,
            "point_major": True,
            "current_dimensions_sha256": _canonical_sha256(
                {
                    str(current_id): dimension
                    for current_id, dimension in sorted(self.current_dimensions.items())
                }
            ),
            "evaluator": "recurrence-direct-plan-decimal-symbolica-exact",
        }

    def to_evidence_dict(self) -> dict[str, object]:
        if not self.complete_observations_retained:
            raise ValueError(
                "detached recurrence observations cannot recreate raw evidence"
            )
        return {
            **self.to_provenance_dict(),
            "kinematic_binary64": [list(row) for row in self.kinematic_binary64],
            "selector_contexts": list(self.selector_contexts),
            "current_dimensions": [
                {"current_id": current_id, "dimension": dimension}
                for current_id, dimension in sorted(self.current_dimensions.items())
            ],
            "observations": [
                {
                    "current_id": current_id,
                    "dimension": self.current_dimensions[current_id],
                    "values": [
                        [_decimal_string(real), _decimal_string(imaginary)]
                        for real, imaginary in self.observations[current_id]
                    ],
                }
                for current_id in sorted(self.observations)
            ],
        }

    def to_certificate_replay_dict(
        self,
        current_ids: Sequence[int],
    ) -> dict[str, object]:
        """Return deduplicated raw rows needed to replay applied certificates."""

        selected = tuple(sorted(set(current_ids)))
        if any(current_id not in self.observations for current_id in selected):
            raise ValueError(
                "certificate replay references an absent recurrence current"
            )
        observation_rows: list[object] = []
        observation_bytes = 0
        for current_id in selected:
            row = {
                "current_id": current_id,
                "dimension": self.current_dimensions[current_id],
                "values": [
                    [_decimal_string(real), _decimal_string(imaginary)]
                    for real, imaginary in self.observations[current_id]
                ],
            }
            observation_bytes += len(_canonical_json_bytes(row)) + 1
            if observation_bytes > _MAX_PERSISTED_EVIDENCE_BYTES // 2:
                raise ValueError(
                    "recurrence certificate replay rows exceed their compact "
                    "persisted-evidence budget"
                )
            observation_rows.append(row)
        return {
            "abi": "pyamplicol-recurrence-certificate-capture-replay-v1",
            "precision_digits": self.precision_digits,
            "points": [point.to_mapping() for point in self.points],
            "kinematic_binary64": [list(row) for row in self.kinematic_binary64],
            "selector_contexts": list(self.selector_contexts),
            "parameter_contexts": [
                [_decimal_string(value) for value in context]
                for context in self.parameter_contexts
            ],
            "point_sha256s": list(self.point_sha256s),
            "kinematic_sha256s": list(self.kinematic_sha256s),
            "selector_context_sha256s": list(self.context_sha256s),
            "parameter_context_sha256s": list(
                self.parameter_context_sha256s
            ),
            "runtime_parameter_schema_sha256": (
                self.runtime_parameter_schema_sha256
            ),
            "source_semantics_sha256": self.source_semantics_sha256,
            "certificate_current_ids": list(selected),
            "observations": observation_rows,
            "full_batch_commitment": {
                "current_count": self.current_count,
                "current_dimensions_sha256": _canonical_sha256(
                    {
                        str(current_id): dimension
                        for current_id, dimension in sorted(
                            self.current_dimensions.items()
                        )
                    }
                ),
                "observation_batch_sha256": self.observation_batch_sha256,
                "capture_contract_sha256": self.capture_contract_sha256,
            },
        }


@dataclass(frozen=True, slots=True)
class RecurrenceNumericalCurrentCertificate:
    """Canonical candidate plus independent-verification evidence."""

    current_id: int
    representative_id: int | None
    execution_representative_id: int
    relation_kind: _RelationKind
    factor: tuple[int, int]
    source_semantics_sha256: str
    precision_digits: int
    seed: int
    relative_tolerance: float
    absolute_tolerance: float
    candidate_probe_count: int
    verification_probe_count: int
    current_dimension: int
    runtime_parameter_schema_sha256: str
    candidate_capture_sha256: str
    verification_capture_sha256: str
    candidate_maximum_absolute_residual: Fraction
    candidate_maximum_relative_residual: Fraction
    candidate_maximum_tolerance_ratio: Fraction
    verification_maximum_absolute_residual: Fraction
    verification_maximum_relative_residual: Fraction
    verification_maximum_tolerance_ratio: Fraction
    candidate_observations_sha256: str
    verification_observations_sha256: str
    probe_contract_sha256: str
    proof_sha256: str

    def to_json_dict(self) -> dict[str, object]:
        return {
            "algorithm": _RECURRENCE_CERTIFICATE_ALGORITHM,
            "proof_kind": "authenticated-numerical",
            "relation_kind": self.relation_kind,
            "current_id": self.current_id,
            "representative_id": self.representative_id,
            "execution_representative_id": self.execution_representative_id,
            "factor_integer": list(self.factor),
            "source_semantics_sha256": self.source_semantics_sha256,
            "precision_digits": self.precision_digits,
            "seed": self.seed,
            "relative_tolerance_binary64": self.relative_tolerance.hex(),
            "absolute_tolerance_binary64": self.absolute_tolerance.hex(),
            "candidate_probe_count": self.candidate_probe_count,
            "verification_probe_count": self.verification_probe_count,
            "current_dimension": self.current_dimension,
            "runtime_parameter_schema_sha256": (
                self.runtime_parameter_schema_sha256
            ),
            "candidate_capture_sha256": self.candidate_capture_sha256,
            "verification_capture_sha256": self.verification_capture_sha256,
            "candidate_maximum_absolute_residual": _fraction_string(
                self.candidate_maximum_absolute_residual
            ),
            "candidate_maximum_relative_residual": _fraction_string(
                self.candidate_maximum_relative_residual
            ),
            "candidate_maximum_tolerance_ratio": _fraction_string(
                self.candidate_maximum_tolerance_ratio
            ),
            "verification_maximum_absolute_residual": _fraction_string(
                self.verification_maximum_absolute_residual
            ),
            "verification_maximum_relative_residual": _fraction_string(
                self.verification_maximum_relative_residual
            ),
            "verification_maximum_tolerance_ratio": _fraction_string(
                self.verification_maximum_tolerance_ratio
            ),
            "candidate_observations_sha256": (self.candidate_observations_sha256),
            "verification_observations_sha256": (self.verification_observations_sha256),
            "probe_contract_sha256": self.probe_contract_sha256,
            "proof_sha256": self.proof_sha256,
        }


@dataclass(frozen=True, slots=True)
class RecurrenceNumericalCurrentWarmupResult:
    """One unpublished-baseline discovery transaction."""

    requested_mode: _Mode
    color_accuracy: str
    source_semantics_sha256: str
    runtime_parameter_schema: Mapping[str, object]
    candidate_capture: RecurrenceCurrentObservationCapture
    verification_capture: RecurrenceCurrentObservationCapture
    certificates: tuple[RecurrenceNumericalCurrentCertificate, ...]
    evidence_json: bytes
    discovery_report: Mapping[str, object]
    application_validation: Mapping[str, object]

    @property
    def applied_relation_count(self) -> int:
        return len(self.certificates) if self.requested_mode == "certified-reuse" else 0

    @property
    def warning_required(self) -> bool:
        return self.applied_relation_count > 0

    def with_application_validation(
        self,
        payload: Mapping[str, object],
    ) -> RecurrenceNumericalCurrentWarmupResult:
        return replace(self, application_validation=dict(payload))

    def to_json_dict(self) -> dict[str, object]:
        if not self.certificates:
            state = "no_certified_numerical_relation"
        elif self.requested_mode == "certified-reuse":
            state = "authenticated-numerical-applied"
        else:
            state = "authenticated-numerical-diagnostic-only"
        warning = _warning_payload(self.warning_required)
        applied_current_ids = (
            tuple(
                current_id
                for certificate in self.certificates
                for current_id in (
                    certificate.current_id,
                    certificate.representative_id,
                )
                if current_id is not None
            )
            if self.requested_mode == "certified-reuse"
            else ()
        )
        persisted_evidence = {
            "abi": _PERSISTED_EVIDENCE_ABI,
            "generation_raw_evidence_bytes": len(self.evidence_json),
            "raw_evidence_retained": False,
            "runtime_parameter_schema": dict(self.runtime_parameter_schema),
            "runtime_parameter_schema_sha256": (
                self.candidate_capture.runtime_parameter_schema_sha256
            ),
            "candidate_capture": self.candidate_capture.to_certificate_replay_dict(
                applied_current_ids
            ),
            "verification_capture": (
                self.verification_capture.to_certificate_replay_dict(
                    applied_current_ids
                )
            ),
            "full_census": {
                "inspected_current_count": self.discovery_report[
                    "inspected_current_count"
                ],
                "tested_hypothesis_count": self.discovery_report[
                    "tested_hypothesis_count"
                ],
                "candidate_index": dict(
                    cast(
                        Mapping[str, object],
                        self.discovery_report["candidate_index"],
                    )
                ),
                "numerical_candidate_count": self.discovery_report[
                    "numerical_candidate_count"
                ],
                "verification_rejected_count": self.discovery_report[
                    "verification_rejected_count"
                ],
                "uncertified_candidate_count": self.discovery_report[
                    "verification_rejected_count"
                ],
                "certified_relation_count": len(self.certificates),
                "rejected_hypothesis_count": self.discovery_report[
                    "rejected_hypothesis_count"
                ],
                "decision_sha256": self.discovery_report["decision_sha256"],
                "rejection_decision_sha256": self.discovery_report[
                    "rejection_decision_sha256"
                ],
                "certificate_set_sha256": _certificate_set_sha256(
                    self.certificates
                ),
            },
        }
        persisted_encoded = _bounded_canonical_json_bytes(
            persisted_evidence,
            byte_limit=_MAX_PERSISTED_EVIDENCE_BYTES,
            label="recurrence persisted numerical evidence",
        )
        persisted_evidence["measured_payload_bytes"] = len(persisted_encoded)
        persisted_evidence["sha256"] = _canonical_sha256(persisted_evidence)
        _bounded_canonical_json_bytes(
            persisted_evidence,
            byte_limit=_MAX_PERSISTED_EVIDENCE_BYTES,
            label="recurrence persisted numerical evidence",
        )
        return {
            "schema_version": 1,
            "abi": _WARMUP_ABI,
            "requested_mode": self.requested_mode,
            "state": state,
            "scope": {
                "execution_mode": "recurrence",
                "color_accuracy": self.color_accuracy,
                "representation": "recurrence-direct-plan-v2",
            },
            "source_semantics": {
                "abi": _SOURCE_ABI,
                "sha256": self.source_semantics_sha256,
            },
            "candidate_capture": self.candidate_capture.to_provenance_dict(),
            "verification_capture": (self.verification_capture.to_provenance_dict()),
            "application_capture": (
                self.application_validation.get("application_capture")
            ),
            "application_validation": dict(self.application_validation),
            "persisted_numerical_evidence": persisted_evidence,
            "discovery": dict(self.discovery_report),
            "application": {
                "schema_version": 1,
                "abi": _RELATION_SET_ABI,
                "requested_mode": self.requested_mode,
                "state": state,
                "certificate_replay": {
                    "algorithm": _RECURRENCE_CERTIFICATE_ALGORITHM,
                    "status": (
                        "verified"
                        if self.certificates
                        else "no_certified_numerical_relation"
                    ),
                    "certificate_set_sha256": _certificate_set_sha256(
                        self.certificates
                    ),
                },
                "certified_relation_count": len(self.certificates),
                "applied_relation_count": self.applied_relation_count,
                "relation_kind_counts": {
                    kind: sum(
                        certificate.relation_kind == kind
                        for certificate in self.certificates
                    )
                    for kind in ("equal", "opposite", "zero")
                },
                "certificates": [
                    certificate.to_json_dict() for certificate in self.certificates
                ],
                "mappings": [
                    _mapping_payload(certificate) for certificate in self.certificates
                ],
                "warning": warning,
            },
            "certified_relation_count": len(self.certificates),
            "applied_relation_count": self.applied_relation_count,
            "warning": warning,
        }


def run_recurrence_numerical_current_warmup(
    plan: _RecurrenceExactPlan,
    *,
    candidate_points: Sequence[ValidationPointRecord],
    verification_points: Sequence[ValidationPointRecord],
    mode: _Mode,
    color_accuracy: str,
    precision_digits: int,
    seed: int,
    relative_tolerance: float,
    absolute_tolerance: float,
) -> RecurrenceNumericalCurrentWarmupResult:
    """Capture, discover, certify, and encode a second-pass relation set."""

    if mode not in {"diagnostic", "certified-reuse"}:
        raise ValueError(
            "recurrence numerical current warm-up requires diagnostic or "
            "certified-reuse mode"
        )
    _validated_tolerances(relative_tolerance, absolute_tolerance)
    relative_tolerance = float(relative_tolerance)
    absolute_tolerance = float(absolute_tolerance)
    geometry = _validate_raw_evidence_geometry(
        plan.sections,
        candidate_probe_count=len(candidate_points),
        verification_probe_count=len(verification_points),
        runtime_parameter_count=len(plan.runtime_parameter_schema),
    )
    source_digest = recurrence_numerical_source_semantics_sha256(plan.sections)
    runtime_parameter_schema = _runtime_parameter_schema_payload(plan)
    runtime_parameter_schema_sha256 = _canonical_sha256(runtime_parameter_schema)
    runtime_defaults = _runtime_parameter_defaults(plan)
    probe_policies = tuple(
        row.probe_policy for row in plan.runtime_parameter_schema
    )
    candidate_parameter_contexts = _build_parameter_probe_contexts(
        runtime_defaults,
        probe_policies=probe_policies,
        precision_digits=precision_digits,
        seed=seed,
        domain="candidate-current-parameter-probes-v1",
        count=len(candidate_points),
        include_defaults=True,
    )
    verification_parameter_contexts = _build_parameter_probe_contexts(
        runtime_defaults,
        probe_policies=probe_policies,
        precision_digits=precision_digits,
        seed=seed,
        domain="independent-verification-current-parameter-probes-v1",
        count=len(verification_points),
        include_defaults=False,
    )
    candidate = capture_recurrence_current_observations(
        plan,
        candidate_points,
        precision_digits=precision_digits,
        source_semantics_sha256=source_digest,
        seed=seed,
        domain="candidate-current-probes-v1",
        parameter_contexts=candidate_parameter_contexts,
        runtime_parameter_schema_sha256=runtime_parameter_schema_sha256,
    )
    verification = capture_recurrence_current_observations(
        plan,
        verification_points,
        precision_digits=precision_digits,
        source_semantics_sha256=source_digest,
        seed=seed,
        domain="independent-verification-current-probes-v1",
        parameter_contexts=verification_parameter_contexts,
        runtime_parameter_schema_sha256=runtime_parameter_schema_sha256,
    )
    _validate_independent_captures(candidate, verification)
    certificates, discovery = _discover_relations(
        plan.sections,
        candidate,
        verification,
        source_semantics_sha256=source_digest,
        precision_digits=precision_digits,
        seed=seed,
        relative_tolerance=relative_tolerance,
        absolute_tolerance=absolute_tolerance,
        color_accuracy=color_accuracy,
    )
    evidence = _evidence_payload(
        plan.sections,
        mode=mode,
        source_semantics_sha256=source_digest,
        certificates=certificates,
        discovery=discovery,
        precision_digits=precision_digits,
        seed=seed,
        relative_tolerance=relative_tolerance,
        absolute_tolerance=absolute_tolerance,
        probe_count=candidate.point_count,
        verification_probe_count=verification.point_count,
        runtime_parameter_schema=runtime_parameter_schema,
        candidate=candidate,
        verification=verification,
    )
    _validate_raw_evidence_scalar_widths(candidate, verification)
    encoded = _bounded_canonical_json_bytes(
        evidence,
        byte_limit=geometry[2],
        label="recurrence numerical raw evidence",
    )
    _validate_raw_evidence_memory_upper_bound(
        len(encoded),
        scalar_count=geometry[0],
        row_count=geometry[1],
        byte_limit=geometry[2],
    )
    replay_current_ids = (
        tuple(
            current_id
            for certificate in certificates
            for current_id in (
                certificate.current_id,
                certificate.representative_id,
            )
            if current_id is not None
        )
        if mode == "certified-reuse"
        else ()
    )
    candidate = candidate.detached_for_replay(replay_current_ids)
    verification = verification.detached_for_replay(replay_current_ids)
    # The streamed evidence mapping owns observation stream wrappers pointing
    # at the complete captures.  Drop it before returning so CPython can
    # release every non-replay Decimal and tuple before native lowering.
    del evidence
    return RecurrenceNumericalCurrentWarmupResult(
        requested_mode=mode,
        color_accuracy=color_accuracy,
        source_semantics_sha256=source_digest,
        runtime_parameter_schema=runtime_parameter_schema,
        candidate_capture=candidate,
        verification_capture=verification,
        certificates=certificates,
        evidence_json=encoded,
        discovery_report=discovery,
        application_validation={
            "status": (
                "pending-second-pass-validation"
                if mode == "certified-reuse" and certificates
                else "not-required-no-applied-relations"
            ),
            "checked_current_count": 0,
            "checked_component_count": 0,
            "maximum_absolute_residual": None,
            "maximum_relative_residual": None,
            "maximum_tolerance_ratio": None,
            "application_capture": None,
        },
    )


def build_recurrence_numerical_current_probe_points(
    process: CanonicalProcessIR,
    model: Model,
    *,
    process_id: str,
    seed: int,
    candidate_count: int,
    verification_count: int,
) -> tuple[
    tuple[ValidationPointRecord, ...],
    tuple[ValidationPointRecord, ...],
]:
    """Build disjoint deterministic physical points for recurrence probes."""

    if (
        type(seed) is not int
        or seed < 0
        or type(candidate_count) is not int
        or candidate_count < 2
        or type(verification_count) is not int
        or verification_count < 2
    ):
        raise ValueError("recurrence numerical probe-point contract is invalid")

    def points(domain: str, count: int) -> tuple[ValidationPointRecord, ...]:
        return tuple(
            build_process_validation_point(
                process,
                model,
                process_id=process_id,
                seed=_domain_seed(seed, domain=domain, index=index),
            )
            for index in range(count)
        )

    candidate = points("candidate-current-probes-v1", candidate_count)
    verification = points(
        "independent-verification-current-probes-v1",
        verification_count,
    )
    if any(not point.available for point in (*candidate, *verification)):
        errors = tuple(
            point.error for point in (*candidate, *verification) if not point.available
        )
        raise ValueError(
            "recurrence numerical probe-point generation failed: "
            + "; ".join(str(error) for error in errors)
        )
    candidate_hashes = {_kinematic_sha256(point) for point in candidate}
    verification_hashes = {_kinematic_sha256(point) for point in verification}
    if (
        len(candidate_hashes) != len(candidate)
        or len(verification_hashes) != len(verification)
        or not candidate_hashes.isdisjoint(verification_hashes)
    ):
        raise ValueError("recurrence numerical probe domains are not independent")
    return candidate, verification


def validate_recurrence_numerical_current_application(
    baseline: RecurrenceNumericalCurrentWarmupResult,
    applied_plan: _RecurrenceExactPlan,
    *,
    baseline_plan: _RecurrenceExactPlan,
    precision_digits: int,
    seed: int,
    relative_tolerance: float,
    absolute_tolerance: float,
) -> RecurrenceNumericalCurrentWarmupResult:
    """Fail closed unless every second-pass current replays on verification."""

    if not baseline.applied_relation_count:
        return baseline.with_application_validation(
            {
                "status": "not-required-no-applied-relations",
                "checked_current_count": 0,
                "checked_component_count": 0,
                "maximum_absolute_residual": None,
                "maximum_relative_residual": None,
                "maximum_tolerance_ratio": None,
                "application_capture": None,
            }
        )
    reference = capture_recurrence_current_observations(
        baseline_plan,
        baseline.verification_capture.points,
        precision_digits=precision_digits,
        source_semantics_sha256=baseline.source_semantics_sha256,
        seed=seed,
        domain="independent-verification-current-probes-v1",
        parameter_contexts=baseline.verification_capture.parameter_contexts,
        runtime_parameter_schema_sha256=(
            baseline.verification_capture.runtime_parameter_schema_sha256
        ),
    )
    _validate_recaptured_reference(
        baseline.verification_capture,
        reference,
    )
    applied = capture_recurrence_current_observations(
        applied_plan,
        baseline.verification_capture.points,
        precision_digits=precision_digits,
        source_semantics_sha256=baseline.source_semantics_sha256,
        seed=seed,
        domain="independent-verification-current-probes-v1",
        parameter_contexts=baseline.verification_capture.parameter_contexts,
        runtime_parameter_schema_sha256=(
            baseline.verification_capture.runtime_parameter_schema_sha256
        ),
    )
    validation = _validate_applied_observations(
        reference,
        applied,
        relative_tolerance=relative_tolerance,
        absolute_tolerance=absolute_tolerance,
    )
    validation["application_capture"] = applied.to_provenance_dict()
    return baseline.with_application_validation(validation)


def recurrence_numerical_current_opt_out_report(
    sections: _RecurrenceExactSectionsV1,
    *,
    color_accuracy: str,
) -> dict[str, object]:
    """Return stable provenance without evaluating a single exact current."""

    return {
        "schema_version": 1,
        "abi": _WARMUP_ABI,
        "requested_mode": "off",
        "state": "disabled-by-user",
        "scope": {
            "execution_mode": "recurrence",
            "color_accuracy": color_accuracy,
            "representation": "recurrence-direct-plan-v2",
        },
        "source_semantics": {
            "abi": _SOURCE_ABI,
            "sha256": recurrence_numerical_source_semantics_sha256(sections),
        },
        "candidate_capture": None,
        "verification_capture": None,
        "application_capture": None,
        "application_validation": {
            "status": "disabled-by-user",
            "checked_current_count": 0,
            "checked_component_count": 0,
            "maximum_absolute_residual": None,
            "maximum_relative_residual": None,
            "maximum_tolerance_ratio": None,
            "application_capture": None,
        },
        "discovery": None,
        "application": None,
        "certified_relation_count": 0,
        "applied_relation_count": 0,
        "warning": _warning_payload(False),
    }


def capture_recurrence_current_observations(
    plan: _RecurrenceExactPlan,
    points: Sequence[ValidationPointRecord],
    *,
    precision_digits: int,
    source_semantics_sha256: str,
    seed: int,
    domain: str,
    parameter_contexts: Sequence[Sequence[Decimal]] | None = None,
    runtime_parameter_schema_sha256: str | None = None,
) -> RecurrenceCurrentObservationCapture:
    """Evaluate every current before Direct-Arena stage recycling."""

    records = tuple(points)
    if (
        type(precision_digits) is not int
        or precision_digits < 80
        or len(records) < 2
        or any(not isinstance(point, ValidationPointRecord) for point in records)
        or any(not point.available for point in records)
        or len({point.process_id for point in records}) != 1
    ):
        raise ValueError("recurrence current capture point contract is invalid")
    sections = plan.sections
    dimensions = {
        current.semantic_id: current.component_count for current in sections.currents
    }
    observations: dict[int, list[_ComplexDecimal]] = {
        current_id: [] for current_id in dimensions
    }
    point_sha256s = tuple(_canonical_sha256(point.to_mapping()) for point in records)
    kinematic_binary64 = tuple(_kinematic_binary64(point) for point in records)
    kinematic_sha256s = tuple(_canonical_sha256(row) for row in kinematic_binary64)
    if len(set(point_sha256s)) != len(records) or len(set(kinematic_sha256s)) != len(
        records
    ):
        raise ValueError("recurrence current capture requires distinct physical points")
    runtime_defaults = _runtime_parameter_defaults(plan)
    schema_payload = _runtime_parameter_schema_payload(plan)
    actual_schema_sha256 = _canonical_sha256(schema_payload)
    if (
        runtime_parameter_schema_sha256 is not None
        and runtime_parameter_schema_sha256 != actual_schema_sha256
    ):
        raise ValueError("recurrence runtime parameter schema digest drifted")
    resolved_parameter_contexts = (
        tuple(runtime_defaults for _record in records)
        if parameter_contexts is None
        else tuple(tuple(values) for values in parameter_contexts)
    )
    if (
        len(resolved_parameter_contexts) != len(records)
        or any(
            len(values) != len(runtime_defaults)
            or any(
                not isinstance(value, Decimal) or not value.is_finite()
                for value in values
            )
            for values in resolved_parameter_contexts
        )
        or any(
            row.probe_policy == "derived-overwritten-fixed-zero-v1"
            and any(
                values[row.runtime_slot] != runtime_defaults[row.runtime_slot]
                for values in resolved_parameter_contexts
            )
            for row in plan.runtime_parameter_schema
        )
    ):
        raise ValueError(
            "recurrence parameter contexts do not cover every flattened runtime slot"
        )
    parameter_context_sha256s = tuple(
        _canonical_sha256(
            {
                "abi": _PARAMETER_CONTEXT_ABI,
                "domain": domain,
                "point_index": point_index,
                "point_sha256": point_sha256,
                "runtime_parameter_schema_sha256": actual_schema_sha256,
                "values": [_decimal_string(value) for value in values],
            }
        )
        for point_index, (point_sha256, values) in enumerate(
            zip(point_sha256s, resolved_parameter_contexts, strict=True)
        )
    )
    context_payloads: list[object] = []
    working_precision = precision_digits + 16
    with localcontext() as context:
        context.prec = working_precision
        context.rounding = ROUND_HALF_EVEN
        for point_index, (record, parameter_context) in enumerate(
            zip(records, resolved_parameter_contexts, strict=True)
        ):
            parameters = plan.resolve_model_parameters(
                parameter_context,
                working_precision,
            )
            point = cast(
                _Point,
                tuple(
                    tuple(Decimal.from_float(float(value)) for value in momentum)
                    for momentum in record.four_vectors
                ),
            )
            captured, context_payload = _capture_one_point(
                plan,
                point,
                parameters.prepared,
                working_precision,
                selector_seed=_domain_seed(
                    seed,
                    domain=domain,
                    index=point_index,
                ),
            )
            context_payloads.append(context_payload)
            if set(captured) != set(dimensions):
                raise ValueError("recurrence exact current capture is incomplete")
            for current_id, dimension in dimensions.items():
                values = captured[current_id]
                if len(values) != dimension:
                    raise ValueError(
                        f"recurrence current {current_id} capture width "
                        f"{len(values)} does not match {dimension}"
                    )
                observations[current_id].extend(values)
    frozen = {current_id: tuple(values) for current_id, values in observations.items()}
    if any(
        not real.is_finite() or not imaginary.is_finite()
        for values in frozen.values()
        for real, imaginary in values
    ):
        raise ValueError(
            "recurrence exact current capture produced a non-finite component"
        )
    context_sha256s = tuple(_canonical_sha256(payload) for payload in context_payloads)
    batch_digest = _canonical_sha256(
        {
            "abi": _OBSERVATION_BATCH_ABI,
            "source_semantics_sha256": source_semantics_sha256,
            "runtime_parameter_schema_sha256": actual_schema_sha256,
            "point_sha256s": list(point_sha256s),
            "selector_context_sha256s": list(context_sha256s),
            "parameter_context_sha256s": list(parameter_context_sha256s),
            "currents": _ObservationRowsStream(frozen, dimensions),
        }
    )
    context_policy = _context_policy(sections.strategy)
    capture_contract_sha256 = _canonical_sha256(
        {
            "abi": _CAPTURE_ABI,
            "precision_digits": precision_digits,
            "source_semantics_sha256": source_semantics_sha256,
            "runtime_parameter_schema_sha256": actual_schema_sha256,
            "point_sha256s": list(point_sha256s),
            "kinematic_sha256s": list(kinematic_sha256s),
            "selector_context_sha256s": list(context_sha256s),
            "parameter_context_sha256s": list(parameter_context_sha256s),
            "current_dimensions": _DimensionRowsStream(dimensions),
            "observation_batch_sha256": batch_digest,
            "context_policy": context_policy,
        }
    )
    return RecurrenceCurrentObservationCapture(
        precision_digits=precision_digits,
        points=records,
        point_sha256s=point_sha256s,
        kinematic_sha256s=kinematic_sha256s,
        kinematic_binary64=kinematic_binary64,
        selector_contexts=tuple(context_payloads),
        context_sha256s=context_sha256s,
        parameter_contexts=resolved_parameter_contexts,
        parameter_context_sha256s=parameter_context_sha256s,
        observations=frozen,
        current_dimensions=dimensions,
        runtime_parameter_schema_sha256=actual_schema_sha256,
        source_semantics_sha256=source_semantics_sha256,
        observation_batch_sha256=batch_digest,
        capture_contract_sha256=capture_contract_sha256,
        context_policy=context_policy,
    )


def recurrence_numerical_source_semantics_sha256(
    sections: _RecurrenceExactSectionsV1,
) -> str:
    """Bind probes to the baseline semantic schedule and current contracts."""

    return _canonical_sha256(_source_semantics_payload(sections))


def _source_semantics_payload(
    sections: _RecurrenceExactSectionsV1,
) -> dict[str, object]:
    contracts = _current_contracts(sections)
    if sections.strategy == "topology-replay":
        selector_schedule: object = {
            "policy": _context_policy(sections.strategy),
            "replay_targets": [
                {
                    "target_index": target_index,
                    "public_flow_id": target.public_flow_id,
                    "representative_id": target.representative_id,
                    "selector_domain_id": target.selector_domain_id,
                }
                for target_index, target in enumerate(sections.replay_targets)
            ],
        }
    elif sections.strategy == "all-flow-union":
        selector_schedule = {
            "policy": _context_policy(sections.strategy),
            "resolved_helicities": [
                {
                    "helicity_index": helicity_index,
                    "helicity_id": helicity.helicity_id,
                    "selector_domain_id": helicity.selector_domain_id,
                }
                for helicity_index, helicity in enumerate(
                    sections.resolved_helicities
                )
            ],
        }
    else:
        selector_schedule = {
            "policy": _context_policy(sections.strategy),
            "fixed_source_schedule": True,
        }
    return {
        "abi": _SOURCE_ABI,
        "process_id": sections.process_id,
        "strategy": sections.strategy,
        "schedule_semantic_digest": sections.semantic_digest,
        "baseline_runtime_layout_digest": sections.runtime_layout_digest,
        "selector_schedule": selector_schedule,
        "currents": [
            {
                "current_id": current.semantic_id,
                "is_source": current.source_row != DIRECT_NONE_U32,
                "contract": _plain_contract(contracts[current.semantic_id]),
            }
            for current in sections.currents
        ],
    }


def _capture_one_point(
    plan: _RecurrenceExactPlan,
    point: _Point,
    prepared_parameters: Sequence[_ComplexDecimal],
    precision: int,
    *,
    selector_seed: int,
) -> tuple[dict[int, tuple[_ComplexDecimal, ...]], object]:
    sections = plan.sections
    if sections.strategy == "topology-replay":
        if not sections.replay_targets:
            raise ArtifactError("topology replay has no exact probe target")
        target_index = selector_seed % len(sections.replay_targets)
        target = sections.replay_targets[target_index]
        captured = _capture_execution(
            sections,
            lambda observer: _evaluate_replay_point(
                plan,
                point,
                target,
                prepared_parameters,
                precision,
                current_observer=observer,
            ),
        )
        return captured, {
            "strategy": sections.strategy,
            "target_index": target_index,
            "public_flow_id": target.public_flow_id,
            "representative_id": target.representative_id,
            "selector_domain_id": target.selector_domain_id,
        }
    if sections.strategy == "all-flow-union":
        if not sections.resolved_helicities:
            raise ArtifactError("all-flow union has no exact probe helicity")
        by_domain: dict[int, list[int]] = defaultdict(list)
        for index, helicity in enumerate(sections.resolved_helicities):
            by_domain[helicity.selector_domain_id].append(index)
        captures: dict[int, dict[int, tuple[_ComplexDecimal, ...]]] = {}
        selected: list[dict[str, int]] = []
        for domain_id, indices in sorted(by_domain.items()):
            position = selector_seed % len(indices)
            helicity_index = indices[position]
            helicity = sections.resolved_helicities[helicity_index]
            captures[domain_id] = _capture_execution(
                sections,
                lambda observer, helicity=helicity: _evaluate_union_point(
                    plan,
                    point,
                    helicity,
                    prepared_parameters,
                    precision,
                    current_observer=observer,
                ),
            )
            selected.append(
                {
                    "selector_domain_id": domain_id,
                    "helicity_index": helicity_index,
                    "helicity_id": helicity.helicity_id,
                }
            )
        default_capture = captures[min(captures)]
        merged = {
            current.semantic_id: captures.get(
                current.selector_domain_id,
                default_capture,
            )[current.semantic_id]
            for current in sections.currents
        }
        return merged, {
            "strategy": sections.strategy,
            "selector_domain_helicities": selected,
        }
    captured = _capture_execution(
        sections,
        lambda observer: _evaluate_contracted_point(
            plan,
            point,
            prepared_parameters,
            precision,
            current_observer=observer,
        ),
    )
    return captured, {
        "strategy": sections.strategy,
        "fixed_source_schedule": True,
    }


def _capture_execution(
    sections: _RecurrenceExactSectionsV1,
    operation: object,
) -> dict[int, tuple[_ComplexDecimal, ...]]:
    captured: dict[int, tuple[_ComplexDecimal, ...]] = {}

    def observe(
        current_id: int,
        values: tuple[_ComplexDecimal, ...],
    ) -> None:
        if current_id in captured:
            raise ArtifactError(
                f"recurrence current {current_id} was observed more than once"
            )
        captured[current_id] = values

    if not callable(operation):  # pragma: no cover - internal type contract
        raise TypeError("recurrence capture operation is not callable")
    operation(observe)
    expected = {current.semantic_id for current in sections.currents}
    if set(captured) != expected:
        missing = sorted(expected - set(captured))
        raise ArtifactError(
            f"recurrence current observation missed semantic currents {missing[:8]}"
        )
    return captured


def _build_raw_numerical_candidate_index(
    member_ids: Sequence[int],
    observations: Mapping[int, Sequence[_ComplexDecimal]],
) -> NumericalObservationCandidateIndex[Fraction]:
    return build_numerical_observation_candidate_index(
        member_ids,
        observations,
        normalize=Fraction,
    )


def _raw_numerical_tolerance_window_ids(
    index: NumericalObservationCandidateIndex[Fraction],
    current_values: Sequence[_ComplexDecimal],
    *,
    relation_kind: Literal["equal", "opposite"],
    current_id: int,
    relative_tolerance: Fraction,
    absolute_tolerance: Fraction,
) -> tuple[int, ...]:
    return numerical_observation_tolerance_window_ids(
        index,
        current_values,
        relation_kind=relation_kind,
        current_id=current_id,
        relative_tolerance=relative_tolerance,
        absolute_tolerance=absolute_tolerance,
        normalize=Fraction,
    )


def _discover_relations(
    sections: _RecurrenceExactSectionsV1,
    candidate: RecurrenceCurrentObservationCapture,
    verification: RecurrenceCurrentObservationCapture,
    *,
    source_semantics_sha256: str,
    precision_digits: int,
    seed: int,
    relative_tolerance: float,
    absolute_tolerance: float,
    color_accuracy: str,
) -> tuple[
    tuple[RecurrenceNumericalCurrentCertificate, ...],
    dict[str, object],
]:
    relative, absolute = _validated_tolerances(
        relative_tolerance,
        absolute_tolerance,
    )
    relative_fraction = Fraction(relative)
    absolute_fraction = Fraction(absolute)
    contracts = _current_contracts(sections)
    members_by_contract: dict[_CurrentContract, list[int]] = defaultdict(list)
    for current in sections.currents:
        members_by_contract[contracts[current.semantic_id]].append(
            current.semantic_id
        )
    candidate_indexes = {
        contract: _build_raw_numerical_candidate_index(
            member_ids,
            candidate.observations,
        )
        for contract, member_ids in members_by_contract.items()
    }
    prior_by_contract: dict[_CurrentContract, list[int]] = defaultdict(list)
    certificates: list[RecurrenceNumericalCurrentCertificate] = []
    rejected: list[dict[str, object]] = []
    nearest: tuple[Fraction, Fraction, Fraction, int, int, str] | None = None
    tested = 0
    theoretical_pair_hypotheses = 0
    screened_pair_hypotheses = 0
    zero_hypotheses = 0
    candidates = 0
    verification_rejected = 0
    rejected_hypotheses = 0
    decision_chain = _canonical_sha256(
        {
            "abi": _DECISION_CHAIN_ABI,
            "source_semantics_sha256": source_semantics_sha256,
            "candidate_capture_sha256": candidate.capture_contract_sha256,
            "verification_capture_sha256": verification.capture_contract_sha256,
        }
    )
    rejection_chain = _canonical_sha256(
        {
            "abi": _REJECTION_CHAIN_ABI,
            "source_semantics_sha256": source_semantics_sha256,
            "candidate_capture_sha256": candidate.capture_contract_sha256,
            "verification_capture_sha256": verification.capture_contract_sha256,
        }
    )

    def record_decision(
        *,
        current_id: int,
        representative_id: int | None,
        relation_kind: _RelationKind,
        candidate_residuals: tuple[Fraction, Fraction, Fraction],
        verification_residuals: tuple[Fraction, Fraction, Fraction] | None,
        selected: bool,
    ) -> None:
        nonlocal decision_chain
        decision_chain = hashlib.sha256(
            bytes.fromhex(decision_chain)
            + _canonical_json_bytes(
                {
                    "current_id": current_id,
                    "representative_id": representative_id,
                    "relation_kind": relation_kind,
                    "candidate_maximum_absolute_residual": _fraction_string(
                        candidate_residuals[0]
                    ),
                    "candidate_maximum_relative_residual": _fraction_string(
                        candidate_residuals[1]
                    ),
                    "candidate_maximum_tolerance_ratio": _fraction_string(
                        candidate_residuals[2]
                    ),
                    "candidate_accepted": candidate_residuals[2] <= 1,
                    "verification_maximum_absolute_residual": (
                        None
                        if verification_residuals is None
                        else _fraction_string(verification_residuals[0])
                    ),
                    "verification_maximum_relative_residual": (
                        None
                        if verification_residuals is None
                        else _fraction_string(verification_residuals[1])
                    ),
                    "verification_maximum_tolerance_ratio": (
                        None
                        if verification_residuals is None
                        else _fraction_string(verification_residuals[2])
                    ),
                    "verification_accepted": (
                        None
                        if verification_residuals is None
                        else verification_residuals[2] <= 1
                    ),
                    "selected": selected,
                }
            )
        ).hexdigest()

    def record_rejected(
        *,
        current_id: int,
        representative_id: int | None,
        relation_kind: _RelationKind,
        residuals: tuple[Fraction, Fraction, Fraction],
        reason: str,
    ) -> None:
        nonlocal nearest, rejected_hypotheses, rejection_chain
        rejected_hypotheses += 1
        rejected_row = {
            "current_id": current_id,
            "representative_id": representative_id,
            "relation_kind": relation_kind,
            "reason": reason,
            "maximum_absolute_residual": _fraction_string(residuals[0]),
            "maximum_relative_residual": _fraction_string(residuals[1]),
            "maximum_tolerance_ratio": _fraction_string(residuals[2]),
        }
        rejection_chain = hashlib.sha256(
            bytes.fromhex(rejection_chain)
            + _canonical_json_bytes(rejected_row)
        ).hexdigest()
        item = (
            residuals[2],
            residuals[0],
            residuals[1],
            current_id,
            -1 if representative_id is None else representative_id,
            relation_kind,
        )
        if nearest is None or item < nearest:
            nearest = item
        if len(rejected) < _MAX_REJECTED_DIAGNOSTICS:
            rejected.append(rejected_row)

    for current in sections.currents:
        contract = contracts[current.semantic_id]
        prior = prior_by_contract[contract]
        if current.source_row != DIRECT_NONE_U32:
            prior.append(current.semantic_id)
            continue
        equal_representatives = set(
            _raw_numerical_tolerance_window_ids(
                candidate_indexes[contract],
                candidate.observations[current.semantic_id],
                relation_kind="equal",
                current_id=current.semantic_id,
                relative_tolerance=relative_fraction,
                absolute_tolerance=absolute_fraction,
            )
        )
        opposite_representatives = set(
            _raw_numerical_tolerance_window_ids(
                candidate_indexes[contract],
                candidate.observations[current.semantic_id],
                relation_kind="opposite",
                current_id=current.semantic_id,
                relative_tolerance=relative_fraction,
                absolute_tolerance=absolute_fraction,
            )
        )
        theoretical_pair_hypotheses += 2 * len(prior)
        screened_pair_hypotheses += len(equal_representatives) + len(
            opposite_representatives
        )
        zero_hypotheses += 1
        if (
            screened_pair_hypotheses + zero_hypotheses
            > _MAX_RAW_NUMERICAL_HYPOTHESES
        ):
            raise ValueError(
                "recurrence numerical candidate index exceeds the explicit "
                "authenticated screened-hypothesis budget"
            )
        hypotheses: list[tuple[_RelationKind, int | None]] = [
            ("zero", None),
            *[
                (relation_kind, representative_id)
                for representative_id in sorted(
                    equal_representatives | opposite_representatives
                )
                for relation_kind in (
                    *(("equal",) if representative_id in equal_representatives else ()),
                    *(
                        ("opposite",)
                        if representative_id in opposite_representatives
                        else ()
                    ),
                )
            ],
        ]
        accepted = False
        for relation_kind, representative_id in hypotheses:
            tested += 1
            candidate_residuals = _relation_residuals(
                relation_kind,
                candidate.observations[current.semantic_id],
                (
                    None
                    if representative_id is None
                    else candidate.observations[representative_id]
                ),
                relative_tolerance=relative,
                absolute_tolerance=absolute,
            )
            if candidate_residuals[2] > 1:
                record_decision(
                    current_id=current.semantic_id,
                    representative_id=representative_id,
                    relation_kind=relation_kind,
                    candidate_residuals=candidate_residuals,
                    verification_residuals=None,
                    selected=False,
                )
                record_rejected(
                    current_id=current.semantic_id,
                    representative_id=representative_id,
                    relation_kind=relation_kind,
                    residuals=candidate_residuals,
                    reason="candidate-observations-not-equal",
                )
                continue
            candidates += 1
            verification_residuals = _relation_residuals(
                relation_kind,
                verification.observations[current.semantic_id],
                (
                    None
                    if representative_id is None
                    else verification.observations[representative_id]
                ),
                relative_tolerance=relative,
                absolute_tolerance=absolute,
            )
            if verification_residuals[2] > 1:
                verification_rejected += 1
                record_decision(
                    current_id=current.semantic_id,
                    representative_id=representative_id,
                    relation_kind=relation_kind,
                    candidate_residuals=candidate_residuals,
                    verification_residuals=verification_residuals,
                    selected=False,
                )
                record_rejected(
                    current_id=current.semantic_id,
                    representative_id=representative_id,
                    relation_kind=relation_kind,
                    residuals=verification_residuals,
                    reason="independent-verification-rejected-candidate",
                )
                continue
            if relation_kind == "zero":
                execution_representative_id = (
                    prior[0] if prior else current.semantic_id
                )
            else:
                if representative_id is None:
                    raise AssertionError(
                        "non-zero recurrence relation lacks a representative"
                    )
                execution_representative_id = representative_id
            certificates.append(
                _build_certificate(
                    current=current,
                    representative_id=representative_id,
                    execution_representative_id=(execution_representative_id),
                    relation_kind=relation_kind,
                    source_semantics_sha256=source_semantics_sha256,
                    candidate=candidate,
                    verification=verification,
                    candidate_residuals=candidate_residuals,
                    verification_residuals=verification_residuals,
                    precision_digits=precision_digits,
                    seed=seed,
                    relative_tolerance=relative_tolerance,
                    absolute_tolerance=absolute_tolerance,
                )
            )
            record_decision(
                current_id=current.semantic_id,
                representative_id=representative_id,
                relation_kind=relation_kind,
                candidate_residuals=candidate_residuals,
                verification_residuals=verification_residuals,
                selected=True,
            )
            accepted = True
            break
        prior.append(current.semantic_id)
        if accepted:
            continue

    nearest_payload = None
    if nearest is not None:
        ratio, absolute_residual, relative_residual, current_id, rep, kind = nearest
        nearest_payload = {
            "current_id": current_id,
            "representative_id": None if rep < 0 else rep,
            "relation_kind": kind,
            "maximum_absolute_residual": _fraction_string(absolute_residual),
            "maximum_relative_residual": _fraction_string(relative_residual),
            "maximum_tolerance_ratio": _fraction_string(ratio),
        }
    certificate_tuple = tuple(certificates)
    decision_sha256 = _canonical_sha256(
        {
            "abi": _DECISION_CHAIN_ABI,
            "chain_tail_sha256": decision_chain,
            "tested_hypothesis_count": tested,
            "theoretical_pair_hypothesis_count": (
                theoretical_pair_hypotheses
            ),
            "screened_pair_hypothesis_count": screened_pair_hypotheses,
            "zero_hypothesis_count": zero_hypotheses,
            "numerical_candidate_count": candidates,
            "verification_rejected_count": verification_rejected,
            "rejected_hypothesis_count": rejected_hypotheses,
            "certified_relation_count": len(certificate_tuple),
        }
    )
    rejection_sha256 = rejection_chain
    rejected_diagnostics = {
        "total_rejected_hypothesis_count": rejected_hypotheses,
        "retained_count": len(rejected),
        "truncated": rejected_hypotheses > len(rejected),
        "truncation_policy": (
            f"first-{_MAX_REJECTED_DIAGNOSTICS}-in-canonical-decision-order-v1"
        ),
        "retained_sha256": _canonical_sha256(
            {
                "abi": (
                    "pyamplicol-recurrence-rejected-numerical-diagnostics-v1"
                ),
                "rows": rejected,
            }
        ),
        "full_decision_sha256": decision_sha256,
        "full_rejection_sha256": rejection_sha256,
    }
    report: dict[str, object] = {
        "schema_version": 1,
        "state": (
            "certified_numerical_relations"
            if certificate_tuple
            else "no_certified_numerical_relation"
        ),
        "scope": {
            "execution_mode": "recurrence",
            "color_accuracy": color_accuracy,
            "representation": "recurrence-direct-plan-v2",
            "lc_flow_layout": sections.strategy,
        },
        "source_semantics": {
            "abi": _SOURCE_ABI,
            "sha256": source_semantics_sha256,
        },
        "probe_contract": {
            "algorithm": _RECURRENCE_CERTIFICATE_ALGORITHM,
            "precision_digits": precision_digits,
            "seed": seed,
            "relative_tolerance_binary64": relative_tolerance.hex(),
            "absolute_tolerance_binary64": absolute_tolerance.hex(),
            "candidate_point_sha256s": list(candidate.point_sha256s),
            "verification_point_sha256s": list(verification.point_sha256s),
            "candidate_context_sha256s": list(candidate.context_sha256s),
            "verification_context_sha256s": list(verification.context_sha256s),
            "candidate_parameter_context_sha256s": list(
                candidate.parameter_context_sha256s
            ),
            "verification_parameter_context_sha256s": list(
                verification.parameter_context_sha256s
            ),
            "runtime_parameter_schema_sha256": (
                candidate.runtime_parameter_schema_sha256
            ),
            "candidate_capture_sha256": candidate.capture_contract_sha256,
            "verification_capture_sha256": verification.capture_contract_sha256,
            "candidate_observation_batch_sha256": (candidate.observation_batch_sha256),
            "verification_observation_batch_sha256": (
                verification.observation_batch_sha256
            ),
            "deterministic": True,
            "independent_verification": True,
            "current_dimension_bound": True,
        },
        "inspected_current_count": len(sections.currents),
        "structurally_proven_current_count": 0,
        "tested_hypothesis_count": tested,
        "candidate_index": {
            "algorithm": _CANDIDATE_INDEX_ALGORITHM,
            "completeness": "complete-within-configured-tolerance",
            "contract_count": len(candidate_indexes),
            "exhaustive_fallback_contract_count": (
                len(candidate_indexes) if relative_fraction >= 1 else 0
            ),
            "theoretical_pair_hypothesis_count": (
                theoretical_pair_hypotheses
            ),
            "screened_pair_hypothesis_count": screened_pair_hypotheses,
            "zero_hypothesis_count": zero_hypotheses,
            "screened_hypothesis_budget": _MAX_RAW_NUMERICAL_HYPOTHESES,
            "budget_classification": "within-authenticated-budget",
            "nearest_rejected_scope": (
                "zero-and-tolerance-window-screened-hypotheses"
            ),
        },
        "numerical_candidate_count": candidates,
        "verification_rejected_count": verification_rejected,
        "rejected_hypothesis_count": rejected_hypotheses,
        "decision_sha256": decision_sha256,
        "rejection_decision_sha256": rejection_sha256,
        "certified_numerical_relation_count": len(certificate_tuple),
        "certificates": [
            certificate.to_json_dict() for certificate in certificate_tuple
        ],
        "rejected_candidates": rejected,
        "rejected_candidate_diagnostics": rejected_diagnostics,
        "nearest_rejected_hypothesis": nearest_payload,
        "warning": _warning_payload(False),
    }
    return certificate_tuple, report


def _build_certificate(
    *,
    current: _Current,
    representative_id: int | None,
    execution_representative_id: int,
    relation_kind: _RelationKind,
    source_semantics_sha256: str,
    candidate: RecurrenceCurrentObservationCapture,
    verification: RecurrenceCurrentObservationCapture,
    candidate_residuals: tuple[Fraction, Fraction, Fraction],
    verification_residuals: tuple[Fraction, Fraction, Fraction],
    precision_digits: int,
    seed: int,
    relative_tolerance: float,
    absolute_tolerance: float,
) -> RecurrenceNumericalCurrentCertificate:
    factor = {
        "equal": (1, 0),
        "opposite": (-1, 0),
        "zero": (0, 0),
    }[relation_kind]
    candidate_digest = _relation_observation_sha256(
        current.semantic_id,
        representative_id,
        relation_kind,
        candidate,
    )
    verification_digest = _relation_observation_sha256(
        current.semantic_id,
        representative_id,
        relation_kind,
        verification,
    )
    probe_contract = {
        "algorithm": _RECURRENCE_CERTIFICATE_ALGORITHM,
        "source_semantics_sha256": source_semantics_sha256,
        "current_id": current.semantic_id,
        "representative_id": representative_id,
        "execution_representative_id": execution_representative_id,
        "relation_kind": relation_kind,
        "precision_digits": precision_digits,
        "seed": seed,
        "candidate_domain": "candidate-current-probes-v1",
        "verification_domain": ("independent-verification-current-probes-v1"),
        "relative_tolerance_binary64": relative_tolerance.hex(),
        "absolute_tolerance_binary64": absolute_tolerance.hex(),
        "candidate_probe_count": candidate.point_count,
        "verification_probe_count": verification.point_count,
        "current_dimension": current.component_count,
        "runtime_parameter_schema_sha256": (
            candidate.runtime_parameter_schema_sha256
        ),
        "candidate_capture_sha256": candidate.capture_contract_sha256,
        "verification_capture_sha256": verification.capture_contract_sha256,
        "candidate_observations_sha256": candidate_digest,
        "verification_observations_sha256": verification_digest,
    }
    probe_digest = _canonical_sha256(probe_contract)
    proof_payload = {
        **probe_contract,
        "proof_kind": "authenticated-numerical",
        "factor_integer": list(factor),
        "candidate_maximum_absolute_residual": _fraction_string(
            candidate_residuals[0]
        ),
        "candidate_maximum_relative_residual": _fraction_string(
            candidate_residuals[1]
        ),
        "candidate_maximum_tolerance_ratio": _fraction_string(
            candidate_residuals[2]
        ),
        "verification_maximum_absolute_residual": _fraction_string(
            verification_residuals[0]
        ),
        "verification_maximum_relative_residual": _fraction_string(
            verification_residuals[1]
        ),
        "verification_maximum_tolerance_ratio": _fraction_string(
            verification_residuals[2]
        ),
        "probe_contract_sha256": probe_digest,
    }
    return RecurrenceNumericalCurrentCertificate(
        current_id=current.semantic_id,
        representative_id=representative_id,
        execution_representative_id=execution_representative_id,
        relation_kind=relation_kind,
        factor=factor,
        source_semantics_sha256=source_semantics_sha256,
        precision_digits=precision_digits,
        seed=seed,
        relative_tolerance=relative_tolerance,
        absolute_tolerance=absolute_tolerance,
        candidate_probe_count=candidate.point_count,
        verification_probe_count=verification.point_count,
        current_dimension=current.component_count,
        runtime_parameter_schema_sha256=(
            candidate.runtime_parameter_schema_sha256
        ),
        candidate_capture_sha256=candidate.capture_contract_sha256,
        verification_capture_sha256=verification.capture_contract_sha256,
        candidate_maximum_absolute_residual=candidate_residuals[0],
        candidate_maximum_relative_residual=candidate_residuals[1],
        candidate_maximum_tolerance_ratio=candidate_residuals[2],
        verification_maximum_absolute_residual=verification_residuals[0],
        verification_maximum_relative_residual=verification_residuals[1],
        verification_maximum_tolerance_ratio=verification_residuals[2],
        candidate_observations_sha256=candidate_digest,
        verification_observations_sha256=verification_digest,
        probe_contract_sha256=probe_digest,
        proof_sha256=_canonical_sha256(proof_payload),
    )


def _evidence_payload(
    sections: _RecurrenceExactSectionsV1,
    *,
    mode: _Mode,
    source_semantics_sha256: str,
    certificates: tuple[RecurrenceNumericalCurrentCertificate, ...],
    discovery: Mapping[str, object],
    precision_digits: int,
    seed: int,
    relative_tolerance: float,
    absolute_tolerance: float,
    probe_count: int,
    verification_probe_count: int,
    runtime_parameter_schema: Mapping[str, object],
    candidate: RecurrenceCurrentObservationCapture,
    verification: RecurrenceCurrentObservationCapture,
) -> dict[str, object]:
    numerical_candidate_count = _required_report_integer(
        discovery,
        "numerical_candidate_count",
    )
    verification_rejected_count = _required_report_integer(
        discovery,
        "verification_rejected_count",
    )
    tested_hypothesis_count = _required_report_integer(
        discovery,
        "tested_hypothesis_count",
    )
    rejected_hypothesis_count = _required_report_integer(
        discovery,
        "rejected_hypothesis_count",
    )
    return {
        "abi": _EVIDENCE_ABI,
        "requested_mode": mode,
        "schedule_semantic_digest": sections.semantic_digest,
        "baseline_runtime_layout_digest": sections.runtime_layout_digest,
        "source_semantics": _source_semantics_payload(sections),
        "source_semantics_sha256": source_semantics_sha256,
        "runtime_parameter_schema": dict(runtime_parameter_schema),
        "runtime_parameter_schema_sha256": (
            candidate.runtime_parameter_schema_sha256
        ),
        "candidate_capture": _capture_evidence_payload(candidate),
        "verification_capture": _capture_evidence_payload(verification),
        "certificate_algorithm": _RECURRENCE_CERTIFICATE_ALGORITHM,
        "certificate_set_sha256": _certificate_set_sha256(certificates),
        "precision_digits": precision_digits,
        "probe_count": probe_count,
        "verification_probe_count": verification_probe_count,
        "relative_tolerance_binary64": relative_tolerance.hex(),
        "absolute_tolerance_binary64": absolute_tolerance.hex(),
        "seed": seed,
        "candidate_index": dict(
            cast(Mapping[str, object], discovery["candidate_index"])
        ),
        "numerical_candidate_count": numerical_candidate_count,
        "verification_rejected_count": verification_rejected_count,
        "rejected_hypothesis_count": rejected_hypothesis_count,
        "tested_hypothesis_count": tested_hypothesis_count,
        "decision_sha256": str(discovery["decision_sha256"]),
        "rejection_decision_sha256": str(
            discovery["rejection_decision_sha256"]
        ),
        "certificates": _CertificateRowsStream(certificates),
        "mappings": _MappingRowsStream(certificates),
    }


def _capture_evidence_payload(
    capture: RecurrenceCurrentObservationCapture,
) -> dict[str, object]:
    """Return a small mapping whose large rows are streamed during encoding."""

    return {
        "abi": _CAPTURE_ABI,
        "precision_digits": capture.precision_digits,
        "point_count": capture.point_count,
        "point_sha256s": capture.point_sha256s,
        "kinematic_sha256s": capture.kinematic_sha256s,
        "parameter_contexts": tuple(
            tuple(_decimal_string(value) for value in context)
            for context in capture.parameter_contexts
        ),
        "parameter_context_sha256s": capture.parameter_context_sha256s,
        "context_sha256s": capture.context_sha256s,
        "points": tuple(point.to_mapping() for point in capture.points),
        "current_count": capture.current_count,
        "runtime_parameter_schema_sha256": (
            capture.runtime_parameter_schema_sha256
        ),
        "source_semantics_sha256": capture.source_semantics_sha256,
        "observation_batch_sha256": capture.observation_batch_sha256,
        "capture_contract_sha256": capture.capture_contract_sha256,
        "context_policy": capture.context_policy,
        "complete_current_components": True,
        "point_major": True,
        "current_dimensions_sha256": _canonical_sha256(
            {
                str(current_id): dimension
                for current_id, dimension in sorted(
                    capture.current_dimensions.items()
                )
            }
        ),
        "evaluator": "recurrence-direct-plan-decimal-symbolica-exact",
        "kinematic_binary64": capture.kinematic_binary64,
        "selector_contexts": capture.selector_contexts,
        "current_dimensions": _DimensionRowsStream(capture.current_dimensions),
        "observations": _ObservationRowsStream(
            capture.observations,
            capture.current_dimensions,
        ),
    }


def _required_report_integer(
    report: Mapping[str, object],
    key: str,
) -> int:
    value = report.get(key)
    if type(value) is not int:
        raise ValueError(f"recurrence numerical report {key!r} is not an integer")
    return value


def _mapping_payload(
    certificate: RecurrenceNumericalCurrentCertificate,
) -> dict[str, object]:
    return {
        "current_id": certificate.current_id,
        "representative_id": certificate.representative_id,
        "execution_representative_id": (certificate.execution_representative_id),
        "relation_kind": certificate.relation_kind,
        "factor_integer": list(certificate.factor),
        "current_dimension": certificate.current_dimension,
        "certificate_proof_sha256": certificate.proof_sha256,
        "candidate_observations_sha256": (certificate.candidate_observations_sha256),
        "verification_observations_sha256": (
            certificate.verification_observations_sha256
        ),
    }


def _current_contracts(
    sections: _RecurrenceExactSectionsV1,
) -> tuple[_CurrentContract, ...]:
    finalization_executors: dict[int, int] = {}
    for group in sections.row_groups:
        if group.role == 2:
            for row_id in range(group.row_start, group.row_start + group.row_count):
                if row_id in finalization_executors:
                    raise ArtifactError(
                        "recurrence finalization row belongs to multiple groups"
                    )
                finalization_executors[row_id] = group.executor_id
    factors = tuple(
        (
            str(factor.real_numerator),
            str(factor.real_denominator),
            str(factor.imaginary_numerator),
            str(factor.imaginary_denominator),
        )
        for factor in sections.exact_factors
    )
    result = []
    for current in sections.currents:
        if current.finalization_row == DIRECT_NONE_U32:
            finalization = (
                DIRECT_NONE_U32,
                DIRECT_NONE_U32,
                ("1", "1", "0", "1"),
            )
        else:
            try:
                row = sections.finalizations[current.finalization_row]
                executor_id = finalization_executors[current.finalization_row]
                factor = factors[row.exact_factor_id]
            except (IndexError, KeyError) as exc:
                raise ArtifactError(
                    f"recurrence current {current.semantic_id} has an invalid "
                    "finalization contract"
                ) from exc
            finalization = (executor_id, row.momentum_form_id, factor)
        result.append(
            (
                current.node_kind,
                current.stage,
                current.state_template_id,
                current.component_count,
                current.momentum_form_id,
                current.selector_domain_id,
                *finalization,
            )
        )
    return tuple(result)


def _validate_independent_captures(
    candidate: RecurrenceCurrentObservationCapture,
    verification: RecurrenceCurrentObservationCapture,
) -> None:
    if (
        candidate.precision_digits != verification.precision_digits
        or candidate.source_semantics_sha256 != verification.source_semantics_sha256
        or candidate.runtime_parameter_schema_sha256
        != verification.runtime_parameter_schema_sha256
        or candidate.current_dimensions != verification.current_dimensions
        or set(candidate.observations) != set(verification.observations)
        or candidate.point_count < 2
        or verification.point_count < 2
        or not set(candidate.point_sha256s).isdisjoint(verification.point_sha256s)
        or not set(candidate.kinematic_sha256s).isdisjoint(
            verification.kinematic_sha256s
        )
        or len(candidate.parameter_context_sha256s) != candidate.point_count
        or len(verification.parameter_context_sha256s)
        != verification.point_count
        or not set(candidate.parameter_context_sha256s).isdisjoint(
            verification.parameter_context_sha256s
        )
    ):
        raise ValueError(
            "recurrence candidate and verification captures are not independent"
        )
    for current_id, dimension in candidate.current_dimensions.items():
        if (
            len(candidate.observations[current_id]) != candidate.point_count * dimension
            or len(verification.observations[current_id])
            != verification.point_count * dimension
        ):
            raise ValueError(f"recurrence current {current_id} probe geometry drifted")


def _validate_applied_observations(
    reference: RecurrenceCurrentObservationCapture,
    applied: RecurrenceCurrentObservationCapture,
    *,
    relative_tolerance: float,
    absolute_tolerance: float,
) -> dict[str, object]:
    if (
        reference.precision_digits != applied.precision_digits
        or reference.point_sha256s != applied.point_sha256s
        or reference.kinematic_sha256s != applied.kinematic_sha256s
        or reference.kinematic_binary64 != applied.kinematic_binary64
        or reference.selector_contexts != applied.selector_contexts
        or reference.context_sha256s != applied.context_sha256s
        or reference.parameter_contexts != applied.parameter_contexts
        or reference.parameter_context_sha256s
        != applied.parameter_context_sha256s
        or reference.runtime_parameter_schema_sha256
        != applied.runtime_parameter_schema_sha256
        or reference.current_dimensions != applied.current_dimensions
        or set(reference.observations) != set(applied.observations)
    ):
        raise ValueError("recurrence application validation provenance drifted")
    relative, absolute = _validated_tolerances(
        relative_tolerance,
        absolute_tolerance,
    )
    maximum_absolute = Fraction(0)
    maximum_relative = Fraction(0)
    maximum_ratio = Fraction(0)
    checked = 0
    for current_id in sorted(reference.observations):
        before = reference.observations[current_id]
        after = applied.observations[current_id]
        if len(before) != len(after):
            raise ValueError(
                f"recurrence current {current_id} application width drifted"
            )
        residuals = _pair_residuals(
            before,
            after,
            sign=1,
            relative_tolerance=relative,
            absolute_tolerance=absolute,
        )
        maximum_absolute = max(maximum_absolute, residuals[0])
        maximum_relative = max(maximum_relative, residuals[1])
        maximum_ratio = max(maximum_ratio, residuals[2])
        checked += len(before)
        if residuals[2] > 1:
            raise ValueError(
                "recurrence numerical reuse changed current "
                f"{current_id} beyond its authenticated tolerance"
            )
    return {
        "status": "verified",
        "checked_current_count": len(reference.observations),
        "checked_component_count": checked,
        "maximum_absolute_residual": _fraction_string(maximum_absolute),
        "maximum_relative_residual": _fraction_string(maximum_relative),
        "maximum_tolerance_ratio": _fraction_string(maximum_ratio),
        "reference_observation_batch_sha256": (reference.observation_batch_sha256),
        "applied_observation_batch_sha256": (applied.observation_batch_sha256),
    }


def _validate_recaptured_reference(
    authenticated: RecurrenceCurrentObservationCapture,
    recaptured: RecurrenceCurrentObservationCapture,
) -> None:
    """Bind a post-native baseline recapture to the original full commitment."""

    if (
        authenticated.precision_digits != recaptured.precision_digits
        or authenticated.points != recaptured.points
        or authenticated.point_sha256s != recaptured.point_sha256s
        or authenticated.kinematic_sha256s != recaptured.kinematic_sha256s
        or authenticated.kinematic_binary64 != recaptured.kinematic_binary64
        or authenticated.selector_contexts != recaptured.selector_contexts
        or authenticated.context_sha256s != recaptured.context_sha256s
        or authenticated.parameter_contexts != recaptured.parameter_contexts
        or authenticated.parameter_context_sha256s
        != recaptured.parameter_context_sha256s
        or authenticated.runtime_parameter_schema_sha256
        != recaptured.runtime_parameter_schema_sha256
        or authenticated.source_semantics_sha256
        != recaptured.source_semantics_sha256
        or authenticated.current_dimensions != recaptured.current_dimensions
        or authenticated.current_count != recaptured.current_count
        or authenticated.observation_batch_sha256
        != recaptured.observation_batch_sha256
        or authenticated.capture_contract_sha256
        != recaptured.capture_contract_sha256
        or authenticated.context_policy != recaptured.context_policy
    ):
        raise ValueError(
            "recurrence baseline application reference did not reproduce its "
            "authenticated capture commitment"
        )


def _relation_residuals(
    relation_kind: _RelationKind,
    current: Sequence[_ComplexDecimal],
    representative: Sequence[_ComplexDecimal] | None,
    *,
    relative_tolerance: Decimal,
    absolute_tolerance: Decimal,
) -> tuple[Fraction, Fraction, Fraction]:
    comparison: Sequence[_ComplexDecimal]
    if relation_kind == "zero":
        comparison = tuple((Decimal(0), Decimal(0)) for _ in current)
        sign = 1
    else:
        if representative is None or len(current) != len(representative):
            raise ValueError("recurrence numerical relation width is invalid")
        comparison = representative
        sign = 1 if relation_kind == "equal" else -1
    return _pair_residuals(
        current,
        comparison,
        sign=sign,
        relative_tolerance=relative_tolerance,
        absolute_tolerance=absolute_tolerance,
    )


def _pair_residuals(
    left: Sequence[_ComplexDecimal],
    right: Sequence[_ComplexDecimal],
    *,
    sign: int,
    relative_tolerance: Decimal,
    absolute_tolerance: Decimal,
) -> tuple[Fraction, Fraction, Fraction]:
    if len(left) != len(right):
        raise ValueError("recurrence numerical comparison width is invalid")
    maximum_absolute = Fraction(0)
    maximum_relative = Fraction(0)
    maximum_ratio = Fraction(0)
    relative_fraction = Fraction(relative_tolerance)
    absolute_fraction = Fraction(absolute_tolerance)
    for left_value, right_value in zip(left, right, strict=True):
        right_real = Fraction(right_value[0])
        right_imaginary = Fraction(right_value[1])
        signed = (
            (right_real, right_imaginary)
            if sign == 1
            else (-right_real, -right_imaginary)
        )
        difference = max(
            abs(Fraction(left_value[0]) - Fraction(signed[0])),
            abs(Fraction(left_value[1]) - Fraction(signed[1])),
        )
        scale = max(
            abs(Fraction(left_value[0])),
            abs(Fraction(left_value[1])),
            abs(right_real),
            abs(right_imaginary),
        )
        allowed = absolute_fraction + relative_fraction * scale
        relative = (
            Fraction(0)
            if difference == 0
            else difference / max(scale, absolute_fraction)
        )
        ratio = Fraction(0) if difference == 0 else difference / allowed
        maximum_absolute = max(maximum_absolute, difference)
        maximum_relative = max(maximum_relative, relative)
        maximum_ratio = max(maximum_ratio, ratio)
    return maximum_absolute, maximum_relative, maximum_ratio


def _relation_observation_sha256(
    current_id: int,
    representative_id: int | None,
    relation_kind: _RelationKind,
    capture: RecurrenceCurrentObservationCapture,
) -> str:
    return _canonical_sha256(
        {
            "abi": _RELATION_OBSERVATION_ABI,
            "capture_contract_sha256": capture.capture_contract_sha256,
            "current_id": current_id,
            "representative_id": representative_id,
            "relation_kind": relation_kind,
            "current_dimension": capture.current_dimensions[current_id],
            "current_values": [
                [_decimal_string(real), _decimal_string(imaginary)]
                for real, imaginary in capture.observations[current_id]
            ],
            "representative_values": (
                None
                if representative_id is None
                else [
                    [_decimal_string(real), _decimal_string(imaginary)]
                    for real, imaginary in capture.observations[representative_id]
                ]
            ),
        }
    )


def _runtime_parameter_defaults(
    plan: _RecurrenceExactPlan,
) -> tuple[Decimal, ...]:
    if (
        len(plan.runtime_parameter_schema) != len(plan.runtime_defaults)
        or any(
            row.runtime_slot != index
            for index, row in enumerate(plan.runtime_parameter_schema)
        )
        or any(
            row.runtime_slot >= len(plan.runtime_defaults)
            for row in plan.parameter_projection
        )
    ):
        raise ArtifactError(
            "recurrence runtime parameter schema/defaults are incomplete"
        )
    defaults: list[Decimal] = []
    for row, default in zip(
        plan.runtime_parameter_schema, plan.runtime_defaults, strict=True
    ):
        if row.probe_policy == "derived-overwritten-fixed-zero-v1":
            defaults.append(Decimal(0))
            continue
        if row.probe_policy != "native-template-default-perturbed-v1":
            raise ArtifactError(
                "recurrence runtime parameter schema has an unsupported "
                "numerical probe policy"
            )
        binary64_default = float(default)
        if not isfinite(binary64_default):
            raise ArtifactError(
                "recurrence runtime parameter default is not finite binary64"
            )
        defaults.append(Decimal.from_float(binary64_default))
    return tuple(defaults)


def _runtime_parameter_schema_payload(
    plan: _RecurrenceExactPlan,
) -> dict[str, object]:
    defaults = _runtime_parameter_defaults(plan)
    return {
        "abi": _PARAMETER_SCHEMA_ABI,
        "parameters": [
            {
                "runtime_slot": row.runtime_slot,
                "runtime_name": row.runtime_name,
                "parameter_template_id": row.parameter_template_id,
                "prepared_parameter_id": row.prepared_parameter_id,
                "component": row.component,
                "default_binary64": float(default).hex(),
                "probe_policy": row.probe_policy,
            }
            for row, default in zip(
                plan.runtime_parameter_schema, defaults, strict=True
            )
        ],
    }


def _build_parameter_probe_contexts(
    defaults: tuple[Decimal, ...],
    *,
    probe_policies: tuple[str, ...],
    precision_digits: int,
    seed: int,
    domain: str,
    count: int,
    include_defaults: bool,
) -> tuple[tuple[Decimal, ...], ...]:
    if (
        type(precision_digits) is not int
        or precision_digits < 80
        or type(seed) is not int
        or seed < 0
        or not isinstance(domain, str)
        or not domain
        or type(count) is not int
        or count < 2
        or len(probe_policies) != len(defaults)
        or any(
            policy
            not in {
                "native-template-default-perturbed-v1",
                "derived-overwritten-fixed-zero-v1",
            }
            for policy in probe_policies
        )
        or any(not value.is_finite() for value in defaults)
    ):
        raise ValueError(
            "recurrence numerical parameter-probe contract is invalid"
        )
    contexts: list[tuple[Decimal, ...]] = []
    for probe_index in range(count):
        if include_defaults and probe_index == 0:
            contexts.append(defaults)
            continue
        values: list[Decimal] = []
        for parameter_index, (default, probe_policy) in enumerate(
            zip(defaults, probe_policies, strict=True)
        ):
            if probe_policy == "derived-overwritten-fixed-zero-v1":
                values.append(default)
                continue
            digest = hashlib.sha256(
                (
                    f"{seed}:{domain}:{probe_index}:"
                    f"{parameter_index}"
                ).encode("ascii")
            ).digest()
            signed = int.from_bytes(digest[:8], "big") - (1 << 63)
            if signed == 0:
                signed = 1
            scale = max(abs(Fraction(default)), Fraction(1))
            value = Fraction(default) + scale * Fraction(signed, 1 << 67)
            values.append(_terminating_fraction_decimal(value))
        contexts.append(tuple(values))
    return tuple(contexts)


def _validated_tolerances(
    relative_tolerance: float,
    absolute_tolerance: float,
) -> tuple[Decimal, Decimal]:
    if (
        isinstance(relative_tolerance, bool)
        or not isinstance(relative_tolerance, int | float)
        or isinstance(absolute_tolerance, bool)
        or not isinstance(absolute_tolerance, int | float)
        or not isfinite(relative_tolerance)
        or not isfinite(absolute_tolerance)
        or relative_tolerance < 0
        or absolute_tolerance < 0
        or (
            float(relative_tolerance) == 0.0
            and copysign(1.0, float(relative_tolerance)) < 0
        )
        or (
            float(absolute_tolerance) == 0.0
            and copysign(1.0, float(absolute_tolerance)) < 0
        )
        or (relative_tolerance == 0 and absolute_tolerance == 0)
    ):
        raise ValueError("recurrence numerical current tolerances are invalid")
    return (
        Decimal.from_float(float(relative_tolerance)),
        Decimal.from_float(float(absolute_tolerance)),
    )


def _certificate_set_sha256(
    certificates: Sequence[RecurrenceNumericalCurrentCertificate],
) -> str:
    return _canonical_sha256(
        {
            "abi": _RELATION_SET_ABI,
            "certificates": _CertificateRowsStream(certificates),
            "mappings": _MappingRowsStream(certificates),
        }
    )


def _kinematic_binary64(
    point: ValidationPointRecord,
) -> tuple[tuple[str, ...], ...]:
    return tuple(
        tuple(float(value).hex() for value in momentum)
        for momentum in point.four_vectors
    )


def _kinematic_sha256(point: ValidationPointRecord) -> str:
    return _canonical_sha256(_kinematic_binary64(point))


def _domain_seed(seed: int, *, domain: str, index: int) -> int:
    digest = hashlib.sha256(f"{seed}:{domain}:{index}".encode("ascii")).digest()
    return int.from_bytes(digest[:8], "big")


def _context_policy(strategy: str) -> str:
    return {
        "topology-replay": "seeded-replay-target-per-physical-point-v1",
        "all-flow-union": ("seeded-helicity-per-selector-domain-and-physical-point-v1"),
        "contracted-color-union": "fixed-contracted-source-schedule-v1",
    }[strategy]


def _plain_contract(contract: _CurrentContract) -> list[object]:
    result = []
    for value in contract:
        result.append(list(value) if isinstance(value, tuple) else value)
    return result


def _warning_payload(required: bool) -> dict[str, object]:
    return (
        {
            "required": True,
            "emit": "once-per-generated-artifact",
            "code": NUMERICAL_CURRENT_RELATION_WARNING_CODE,
            "message": NUMERICAL_CURRENT_RELATION_WARNING,
        }
        if required
        else {
            "required": False,
            "emit": "never",
            "code": None,
            "message": None,
        }
    )


def _decimal_string(value: Decimal) -> str:
    if not value.is_finite():
        raise ValueError("non-finite Decimal cannot be canonicalized")
    if value == 0:
        return "0"
    sign, digits, exponent = value.as_tuple()
    if not isinstance(exponent, int):
        raise ValueError("non-finite Decimal cannot be canonicalized")
    digit_count = len(digits)
    decimal_point = digit_count + exponent
    if exponent >= 0:
        fixed_length = digit_count + exponent
    elif decimal_point > 0:
        fixed_length = digit_count + 1
    else:
        # ``format(value, "f")`` would emit ``0.``, followed by
        # ``-decimal_point`` leading zeroes and all coefficient digits.
        fixed_length = 2 - exponent
    fixed_length += sign
    if fixed_length > _MAX_RAW_EVIDENCE_DECIMAL_BYTES:
        raise ValueError(
            "finite Decimal fixed-point encoding exceeds the raw evidence "
            "scalar boundary"
        )
    text = format(value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return "0" if text in {"-0", "+0", ""} else text


def _fraction_string(value: Fraction) -> str:
    return (
        str(value.numerator)
        if value.denominator == 1
        else f"{value.numerator}/{value.denominator}"
    )


def _terminating_fraction_decimal(value: Fraction) -> Decimal:
    denominator = value.denominator
    twos = 0
    fives = 0
    while denominator % 2 == 0:
        denominator //= 2
        twos += 1
    while denominator % 5 == 0:
        denominator //= 5
        fives += 1
    if denominator != 1:
        raise ValueError("recurrence parameter context is not a finite decimal")
    places = max(twos, fives)
    scaled = value.numerator
    scaled *= 2 ** (places - twos)
    scaled *= 5 ** (places - fives)
    sign = "-" if scaled < 0 else ""
    digits = str(abs(scaled))
    if places == 0:
        return Decimal(f"{sign}{digits}")
    digits = digits.rjust(places + 1, "0")
    return Decimal(f"{sign}{digits[:-places]}.{digits[-places:]}")


def _canonical_sha256(value: object) -> str:
    digest = hashlib.sha256()
    for chunk in _iter_canonical_json_chunks(value):
        digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_bytes(value: object) -> bytes:
    if isinstance(value, _CanonicalJsonStreamValue):
        return b"".join(_iter_canonical_json_chunks(value))
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _iter_canonical_json_chunks(value: object) -> Iterator[bytes]:
    """Yield the unique ASCII encoding accepted at the native boundary."""

    if isinstance(value, _CanonicalJsonStreamValue):
        yield from value.iter_canonical_json_chunks()
        return
    if value is None:
        yield b"null"
        return
    if value is True:
        yield b"true"
        return
    if value is False:
        yield b"false"
        return
    if type(value) is int:
        yield str(value).encode("ascii")
        return
    if isinstance(value, str):
        yield json.dumps(value, ensure_ascii=True).encode("ascii")
        return
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("canonical JSON mappings require string keys")
        yield b"{"
        for index, key in enumerate(sorted(value)):
            if index:
                yield b","
            yield json.dumps(key, ensure_ascii=True).encode("ascii")
            yield b":"
            yield from _iter_canonical_json_chunks(value[key])
        yield b"}"
        return
    if isinstance(value, Sequence) and not isinstance(
        value,
        str | bytes | bytearray,
    ):
        yield b"["
        for index, item in enumerate(value):
            if index:
                yield b","
            yield from _iter_canonical_json_chunks(item)
        yield b"]"
        return
    if isinstance(value, float):
        raise TypeError(
            "canonical recurrence evidence uses binary64 hex strings, not "
            "JSON float spellings"
        )
    raise TypeError(f"unsupported canonical JSON value {type(value).__name__}")


def _bounded_canonical_json_bytes(
    value: object,
    *,
    byte_limit: int,
    label: str,
) -> bytes:
    """Encode canonically while failing before an oversized string exists."""

    if byte_limit <= 0:
        raise ValueError(f"{label} byte limit is invalid")
    encoded = bytearray()
    for chunk in _iter_canonical_json_chunks(value):
        next_size = len(encoded) + len(chunk)
        if next_size > byte_limit:
            raise ValueError(
                f"{label} exceeds the explicit {byte_limit}-byte canonical "
                "boundary"
            )
        encoded.extend(chunk)
    return bytes(encoded)


def _validate_raw_evidence_canonical_size(
    size: int,
    *,
    byte_limit: int = _MAX_RAW_EVIDENCE_BYTES,
) -> None:
    if (
        size < 0
        or byte_limit <= 0
        or byte_limit > _MAX_RAW_EVIDENCE_BYTES
        or size > byte_limit
    ):
        raise ValueError(
            "recurrence numerical raw evidence exceeds the canonical share "
            f"of its {_MAX_RAW_EVIDENCE_MEMORY_BYTES}-byte memory envelope "
            f"(size={size}, byte_limit={byte_limit})"
        )


def _raw_evidence_wire_byte_limit(
    *,
    scalar_count: int,
    row_count: int,
) -> int:
    """Return the per-shape wire ceiling inside the combined 1 GiB envelope."""

    if scalar_count < 0 or row_count < 0:
        raise ValueError("recurrence raw-evidence memory geometry is invalid")
    resident_without_wire = _raw_evidence_memory_upper_bound(
        0,
        scalar_count=scalar_count,
        row_count=row_count,
    )
    minimum_resident = resident_without_wire + (
        _MIN_RAW_EVIDENCE_WIRE_BYTES * _RAW_EVIDENCE_WIRE_PEAK_COPIES
    )
    if minimum_resident > _MAX_RAW_EVIDENCE_MEMORY_BYTES:
        raise ValueError(
            "recurrence raw-evidence capture geometry leaves less than the "
            f"required {_MIN_RAW_EVIDENCE_WIRE_BYTES}-byte canonical wire "
            f"reserve inside its {_MAX_RAW_EVIDENCE_MEMORY_BYTES}-byte memory "
            f"envelope (scalars={scalar_count}, rows={row_count}, "
            f"resident_without_wire={resident_without_wire})"
        )
    return min(
        _MAX_RAW_EVIDENCE_BYTES,
        (
            _MAX_RAW_EVIDENCE_MEMORY_BYTES - resident_without_wire
        )
        // _RAW_EVIDENCE_WIRE_PEAK_COPIES,
    )


def _validate_raw_evidence_geometry(
    sections: _RecurrenceExactSectionsV1,
    *,
    candidate_probe_count: int,
    verification_probe_count: int,
    runtime_parameter_count: int,
) -> tuple[int, int, int]:
    """Reject unsafe capture geometry before allocating Decimal observations."""

    if (
        candidate_probe_count <= 0
        or verification_probe_count <= 0
        or runtime_parameter_count < 0
    ):
        raise ValueError("recurrence raw-evidence probe geometry is invalid")
    current_count = len(sections.currents)
    component_count = sum(
        current.component_count for current in sections.currents
    )
    point_count = candidate_probe_count + verification_probe_count
    row_count = current_count * 2 + runtime_parameter_count
    scalar_count = (
        2
        * component_count
        * point_count
        + runtime_parameter_count * point_count
    )
    try:
        byte_limit = _raw_evidence_wire_byte_limit(
            scalar_count=scalar_count,
            row_count=row_count,
        )
    except ValueError as error:
        raise ValueError(
            "recurrence raw-evidence capture geometry exceeds the explicit "
            f"{_MAX_RAW_EVIDENCE_MEMORY_BYTES}-byte memory envelope "
            f"(currents={current_count}, components={component_count}, "
            f"candidate_points={candidate_probe_count}, "
            f"verification_points={verification_probe_count}, "
            f"runtime_parameters={runtime_parameter_count}, "
            f"scalars={scalar_count}, rows={row_count}, "
            f"reason={error})"
        ) from error
    return scalar_count, row_count, byte_limit


def _validate_raw_evidence_scalar_widths(
    candidate: RecurrenceCurrentObservationCapture,
    verification: RecurrenceCurrentObservationCapture,
) -> None:
    """Bound every scalar before starting the raw evidence byte buffer."""

    for capture_name, capture in (
        ("candidate", candidate),
        ("verification", verification),
    ):
        for current_id, values in capture.observations.items():
            for component_index, (real, imaginary) in enumerate(values):
                for part_name, value in (("real", real), ("imaginary", imaginary)):
                    if len(_decimal_string(value)) > _MAX_RAW_EVIDENCE_DECIMAL_BYTES:
                        raise ValueError(
                            f"{capture_name} current {current_id} component "
                            f"{component_index} {part_name} exceeds the raw "
                            "evidence scalar boundary"
                        )


def _raw_evidence_memory_upper_bound(
    size: int,
    *,
    scalar_count: int,
    row_count: int,
) -> int:
    if size < 0 or scalar_count < 0 or row_count < 0:
        raise ValueError("recurrence raw-evidence memory geometry is invalid")
    return (
        _RAW_EVIDENCE_FIXED_RESERVE_BYTES
        + scalar_count * _RAW_EVIDENCE_SCALAR_RESIDENT_BYTES
        + row_count * _RAW_EVIDENCE_ROW_RESIDENT_BYTES
        + size * _RAW_EVIDENCE_WIRE_PEAK_COPIES
    )


def _raw_streaming_consumer_memory_upper_bound(
    *,
    raw_byte_count: int,
    metadata_byte_count: int,
    metadata_structural_token_count: int,
    current_count: int,
    component_count: int,
    maximum_dimension: int,
    candidate_probe_count: int,
    verification_probe_count: int,
    runtime_parameter_count: int,
) -> int:
    """Mirror the native streaming-consumer envelope for parity tests."""

    values = (
        raw_byte_count,
        metadata_byte_count,
        metadata_structural_token_count,
        current_count,
        component_count,
        maximum_dimension,
        candidate_probe_count,
        verification_probe_count,
        runtime_parameter_count,
    )
    if any(value < 0 for value in values):
        raise ValueError("recurrence raw streaming geometry is invalid")
    observation_row_count = current_count * 2
    candidate_scalar_references = (
        component_count * candidate_probe_count * 2
    )
    transient_rational_count = (
        maximum_dimension
        * max(candidate_probe_count, verification_probe_count)
        * 4
    )
    parameter_rational_count = (
        runtime_parameter_count
        * (candidate_probe_count + verification_probe_count)
        * _RAW_STREAM_PARAMETER_RATIONAL_COPIES
    )
    return (
        raw_byte_count * _RAW_EVIDENCE_WIRE_PEAK_COPIES
        + metadata_byte_count * _RAW_STREAM_METADATA_COPIES
        + metadata_structural_token_count
        * _RAW_STREAM_BYTES_PER_METADATA_TOKEN
        + observation_row_count * _RAW_STREAM_BYTES_PER_ROW_OFFSETS
        + candidate_scalar_references
        * _RAW_STREAM_BYTES_PER_TEXT_REFERENCE
        + current_count * _RAW_STREAM_BYTES_PER_CURRENT_INDEX
        + transient_rational_count * _RAW_STREAM_BYTES_PER_RATIONAL
        + parameter_rational_count * _RAW_STREAM_BYTES_PER_RATIONAL
        + _RAW_EVIDENCE_FIXED_RESERVE_BYTES
    )


def _validate_raw_evidence_memory_upper_bound(
    size: int,
    *,
    scalar_count: int,
    row_count: int,
    byte_limit: int = _MAX_RAW_EVIDENCE_BYTES,
) -> None:
    _validate_raw_evidence_canonical_size(size, byte_limit=byte_limit)
    if (
        _raw_evidence_memory_upper_bound(
            size,
            scalar_count=scalar_count,
            row_count=row_count,
        )
        > _MAX_RAW_EVIDENCE_MEMORY_BYTES
    ):
        raise ValueError(
            "recurrence numerical raw evidence exceeds its explicit "
            f"{_MAX_RAW_EVIDENCE_MEMORY_BYTES}-byte resident memory envelope"
        )


def _synthetic_raw_evidence_bytes(
    current_count: int,
    *,
    component_count: int,
    candidate_probe_count: int,
    verification_probe_count: int,
    decimal_characters: int,
) -> int:
    """Measure the canonical raw-observation footprint without allocating it."""

    if (
        current_count < 0
        or component_count <= 0
        or candidate_probe_count <= 0
        or verification_probe_count <= 0
        or decimal_characters <= 0
    ):
        raise ValueError("synthetic raw-evidence geometry is invalid")
    return _synthetic_raw_evidence_bytes_by_dimension_counts(
        {component_count: current_count},
        candidate_probe_count=candidate_probe_count,
        verification_probe_count=verification_probe_count,
        decimal_characters=decimal_characters,
    )


def _synthetic_raw_evidence_bytes_by_dimension_counts(
    dimension_counts: Mapping[int, int],
    *,
    candidate_probe_count: int,
    verification_probe_count: int,
    decimal_characters: int,
) -> int:
    """Measure a mixed-dimension canonical raw-observation footprint."""

    if (
        not dimension_counts
        or candidate_probe_count <= 0
        or verification_probe_count <= 0
        or decimal_characters <= 0
        or any(
            dimension <= 0 or count < 0
            for dimension, count in dimension_counts.items()
        )
    ):
        raise ValueError("synthetic raw-evidence geometry is invalid")
    dimensions = tuple(
        dimension
        for dimension, count in sorted(dimension_counts.items())
        for _ in range(count)
    )
    current_count = len(dimensions)
    scalar = "1" + "2" * (decimal_characters - 1)

    def array_size(point_count: int) -> int:
        total = 2 + max(0, current_count - 1)
        for current_id, dimension in enumerate(dimensions):
            values = [[scalar, f"-{scalar}"]] * (dimension * point_count)
            total += len(
                _canonical_json_bytes(
                    {
                        "current_id": current_id,
                        "dimension": dimension,
                        "values": values,
                    }
                )
            )
        return total

    skeleton_size = len(
        _canonical_json_bytes(
            {
                "candidate_capture": {"observations": []},
                "verification_capture": {"observations": []},
            }
        )
    )
    return (
        skeleton_size
        + array_size(candidate_probe_count)
        + array_size(verification_probe_count)
        - 4
    )


__all__ = [
    "RecurrenceCurrentObservationCapture",
    "RecurrenceNumericalCurrentCertificate",
    "RecurrenceNumericalCurrentWarmupResult",
    "build_recurrence_numerical_current_probe_points",
    "capture_recurrence_current_observations",
    "recurrence_numerical_current_opt_out_report",
    "recurrence_numerical_source_semantics_sha256",
    "run_recurrence_numerical_current_warmup",
    "validate_recurrence_numerical_current_application",
]
