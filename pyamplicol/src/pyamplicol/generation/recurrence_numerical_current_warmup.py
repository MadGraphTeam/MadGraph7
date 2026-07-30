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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from decimal import ROUND_HALF_EVEN, Decimal, localcontext
from math import isfinite
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
    NUMERICAL_CURRENT_RELATION_CERTIFICATE_ALGORITHM,
    NUMERICAL_CURRENT_RELATION_WARNING,
    NUMERICAL_CURRENT_RELATION_WARNING_CODE,
)
from .validation import ValidationPointRecord, build_process_validation_point

if TYPE_CHECKING:
    from pyamplicol.processes.ir import CanonicalProcessIR

_CAPTURE_ABI = "pyamplicol-recurrence-current-observation-capture-v1"
_EVIDENCE_ABI = "pyamplicol-recurrence-numerical-current-evidence-v1"
_SOURCE_ABI = "pyamplicol-recurrence-numerical-current-source-v1"
_WARMUP_ABI = "pyamplicol-recurrence-numerical-current-warmup-v1"
_RELATION_SET_ABI = "pyamplicol-authenticated-numerical-current-relation-set-v1"
_MAX_REJECTED_DIAGNOSTICS = 32

_RelationKind = Literal["equal", "opposite", "zero"]
_Mode = Literal["diagnostic", "certified-reuse"]
_Point = tuple[tuple[Decimal, Decimal, Decimal, Decimal], ...]
_CurrentContract = tuple[object, ...]
_NONZERO_RELATION_KINDS: tuple[_RelationKind, ...] = ("equal", "opposite")


@dataclass(frozen=True, slots=True)
class RecurrenceCurrentObservationCapture:
    """Complete point-major exact current observations."""

    precision_digits: int
    points: tuple[ValidationPointRecord, ...]
    point_sha256s: tuple[str, ...]
    kinematic_sha256s: tuple[str, ...]
    context_sha256s: tuple[str, ...]
    observations: Mapping[int, tuple[_ComplexDecimal, ...]]
    current_dimensions: Mapping[int, int]
    source_semantics_sha256: str
    observation_batch_sha256: str
    context_policy: str

    @property
    def point_count(self) -> int:
        return len(self.points)

    @property
    def current_count(self) -> int:
        return len(self.observations)

    def to_provenance_dict(self) -> dict[str, object]:
        return {
            "abi": _CAPTURE_ABI,
            "precision_digits": self.precision_digits,
            "point_count": self.point_count,
            "point_sha256s": list(self.point_sha256s),
            "kinematic_sha256s": list(self.kinematic_sha256s),
            "context_sha256s": list(self.context_sha256s),
            "points": [point.to_mapping() for point in self.points],
            "current_count": self.current_count,
            "source_semantics_sha256": self.source_semantics_sha256,
            "observation_batch_sha256": self.observation_batch_sha256,
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
    candidate_maximum_absolute_residual: Decimal
    candidate_maximum_relative_residual: Decimal
    candidate_maximum_tolerance_ratio: Decimal
    verification_maximum_absolute_residual: Decimal
    verification_maximum_relative_residual: Decimal
    verification_maximum_tolerance_ratio: Decimal
    candidate_observations_sha256: str
    verification_observations_sha256: str
    probe_contract_sha256: str
    proof_sha256: str

    def to_json_dict(self) -> dict[str, object]:
        return {
            "algorithm": NUMERICAL_CURRENT_RELATION_CERTIFICATE_ALGORITHM,
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
            "candidate_maximum_absolute_residual": _decimal_string(
                self.candidate_maximum_absolute_residual
            ),
            "candidate_maximum_relative_residual": _decimal_string(
                self.candidate_maximum_relative_residual
            ),
            "candidate_maximum_tolerance_ratio": _decimal_string(
                self.candidate_maximum_tolerance_ratio
            ),
            "verification_maximum_absolute_residual": _decimal_string(
                self.verification_maximum_absolute_residual
            ),
            "verification_maximum_relative_residual": _decimal_string(
                self.verification_maximum_relative_residual
            ),
            "verification_maximum_tolerance_ratio": _decimal_string(
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
            "discovery": dict(self.discovery_report),
            "application": {
                "schema_version": 1,
                "abi": _RELATION_SET_ABI,
                "requested_mode": self.requested_mode,
                "state": state,
                "certificate_replay": {
                    "algorithm": (NUMERICAL_CURRENT_RELATION_CERTIFICATE_ALGORITHM),
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
    source_digest = recurrence_numerical_source_semantics_sha256(plan.sections)
    candidate = capture_recurrence_current_observations(
        plan,
        candidate_points,
        precision_digits=precision_digits,
        source_semantics_sha256=source_digest,
        seed=seed,
        domain="candidate-current-probes-v1",
    )
    verification = capture_recurrence_current_observations(
        plan,
        verification_points,
        precision_digits=precision_digits,
        source_semantics_sha256=source_digest,
        seed=seed,
        domain="independent-verification-current-probes-v1",
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
    )
    encoded = _canonical_json_bytes(evidence)
    return RecurrenceNumericalCurrentWarmupResult(
        requested_mode=mode,
        color_accuracy=color_accuracy,
        source_semantics_sha256=source_digest,
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
    applied = capture_recurrence_current_observations(
        applied_plan,
        baseline.verification_capture.points,
        precision_digits=precision_digits,
        source_semantics_sha256=baseline.source_semantics_sha256,
        seed=seed,
        domain="independent-verification-current-probes-v1",
    )
    validation = _validate_applied_observations(
        baseline.verification_capture,
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
    kinematic_sha256s = tuple(_kinematic_sha256(point) for point in records)
    if len(set(point_sha256s)) != len(records) or len(set(kinematic_sha256s)) != len(
        records
    ):
        raise ValueError("recurrence current capture requires distinct physical points")
    runtime_defaults = _runtime_parameter_defaults(plan)
    context_payloads: list[object] = []
    working_precision = precision_digits + 16
    with localcontext() as context:
        context.prec = working_precision
        context.rounding = ROUND_HALF_EVEN
        parameters = plan.resolve_model_parameters(
            runtime_defaults,
            working_precision,
        )
        for point_index, record in enumerate(records):
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
            "source_semantics_sha256": source_semantics_sha256,
            "point_sha256s": list(point_sha256s),
            "context_sha256s": list(context_sha256s),
            "currents": [
                {
                    "current_id": current_id,
                    "dimension": dimensions[current_id],
                    "values": [
                        [_decimal_string(real), _decimal_string(imaginary)]
                        for real, imaginary in frozen[current_id]
                    ],
                }
                for current_id in sorted(frozen)
            ],
        }
    )
    return RecurrenceCurrentObservationCapture(
        precision_digits=precision_digits,
        points=records,
        point_sha256s=point_sha256s,
        kinematic_sha256s=kinematic_sha256s,
        context_sha256s=context_sha256s,
        observations=frozen,
        current_dimensions=dimensions,
        source_semantics_sha256=source_semantics_sha256,
        observation_batch_sha256=batch_digest,
        context_policy=_context_policy(sections.strategy),
    )


def recurrence_numerical_source_semantics_sha256(
    sections: _RecurrenceExactSectionsV1,
) -> str:
    """Bind probes to the baseline semantic schedule and current contracts."""

    contracts = _current_contracts(sections)
    return _canonical_sha256(
        {
            "abi": _SOURCE_ABI,
            "process_id": sections.process_id,
            "strategy": sections.strategy,
            "schedule_semantic_digest": sections.semantic_digest,
            "baseline_runtime_layout_digest": sections.runtime_layout_digest,
            "currents": [
                {
                    "current_id": current.semantic_id,
                    "is_source": current.source_row != DIRECT_NONE_U32,
                    "contract": _plain_contract(contracts[current.semantic_id]),
                }
                for current in sections.currents
            ],
        }
    )


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
    contracts = _current_contracts(sections)
    prior_by_contract: dict[_CurrentContract, list[int]] = defaultdict(list)
    certificates: list[RecurrenceNumericalCurrentCertificate] = []
    rejected: list[dict[str, object]] = []
    nearest: tuple[Decimal, Decimal, Decimal, int, int, str] | None = None
    tested = 0
    candidates = 0
    verification_rejected = 0

    def record_rejected(
        *,
        current_id: int,
        representative_id: int | None,
        relation_kind: _RelationKind,
        residuals: tuple[Decimal, Decimal, Decimal],
        reason: str,
    ) -> None:
        nonlocal nearest
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
            rejected.append(
                {
                    "current_id": current_id,
                    "representative_id": representative_id,
                    "relation_kind": relation_kind,
                    "reason": reason,
                    "maximum_absolute_residual": _decimal_string(residuals[0]),
                    "maximum_relative_residual": _decimal_string(residuals[1]),
                    "maximum_tolerance_ratio": _decimal_string(residuals[2]),
                }
            )

    for current in sections.currents:
        contract = contracts[current.semantic_id]
        prior = prior_by_contract[contract]
        if current.source_row != DIRECT_NONE_U32:
            prior.append(current.semantic_id)
            continue
        hypotheses: list[tuple[_RelationKind, int | None]] = [("zero", None)]
        hypotheses.extend(
            (kind, representative_id)
            for representative_id in prior
            for kind in _NONZERO_RELATION_KINDS
        )
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
            "maximum_absolute_residual": _decimal_string(absolute_residual),
            "maximum_relative_residual": _decimal_string(relative_residual),
            "maximum_tolerance_ratio": _decimal_string(ratio),
        }
    certificate_tuple = tuple(certificates)
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
            "algorithm": NUMERICAL_CURRENT_RELATION_CERTIFICATE_ALGORITHM,
            "precision_digits": precision_digits,
            "seed": seed,
            "relative_tolerance_binary64": relative_tolerance.hex(),
            "absolute_tolerance_binary64": absolute_tolerance.hex(),
            "candidate_point_sha256s": list(candidate.point_sha256s),
            "verification_point_sha256s": list(verification.point_sha256s),
            "candidate_context_sha256s": list(candidate.context_sha256s),
            "verification_context_sha256s": list(verification.context_sha256s),
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
        "numerical_candidate_count": candidates,
        "verification_rejected_count": verification_rejected,
        "certified_numerical_relation_count": len(certificate_tuple),
        "certificates": [
            certificate.to_json_dict() for certificate in certificate_tuple
        ],
        "rejected_candidates": rejected,
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
    candidate_residuals: tuple[Decimal, Decimal, Decimal],
    verification_residuals: tuple[Decimal, Decimal, Decimal],
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
        "algorithm": NUMERICAL_CURRENT_RELATION_CERTIFICATE_ALGORITHM,
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
        "candidate_observations_sha256": candidate_digest,
        "verification_observations_sha256": verification_digest,
    }
    probe_digest = _canonical_sha256(probe_contract)
    proof_payload = {
        **probe_contract,
        "proof_kind": "authenticated-numerical",
        "factor_integer": list(factor),
        "candidate_maximum_absolute_residual": _decimal_string(candidate_residuals[0]),
        "candidate_maximum_relative_residual": _decimal_string(candidate_residuals[1]),
        "candidate_maximum_tolerance_ratio": _decimal_string(candidate_residuals[2]),
        "verification_maximum_absolute_residual": _decimal_string(
            verification_residuals[0]
        ),
        "verification_maximum_relative_residual": _decimal_string(
            verification_residuals[1]
        ),
        "verification_maximum_tolerance_ratio": _decimal_string(
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
    return {
        "abi": _EVIDENCE_ABI,
        "requested_mode": mode,
        "schedule_semantic_digest": sections.semantic_digest,
        "baseline_runtime_layout_digest": sections.runtime_layout_digest,
        "source_semantics_sha256": source_semantics_sha256,
        "certificate_algorithm": (NUMERICAL_CURRENT_RELATION_CERTIFICATE_ALGORITHM),
        "certificate_set_sha256": _certificate_set_sha256(certificates),
        "numerical_candidate_count": numerical_candidate_count,
        "verification_rejected_count": verification_rejected_count,
        "tested_hypothesis_count": tested_hypothesis_count,
        "mappings": [_mapping_payload(certificate) for certificate in certificates],
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
            factor.real_numerator,
            factor.real_denominator,
            factor.imaginary_numerator,
            factor.imaginary_denominator,
        )
        for factor in sections.exact_factors
    )
    result = []
    for current in sections.currents:
        if current.finalization_row == DIRECT_NONE_U32:
            finalization = (
                DIRECT_NONE_U32,
                DIRECT_NONE_U32,
                (1, 1, 0, 1),
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
        or candidate.current_dimensions != verification.current_dimensions
        or set(candidate.observations) != set(verification.observations)
        or candidate.point_count < 2
        or verification.point_count < 2
        or not set(candidate.point_sha256s).isdisjoint(verification.point_sha256s)
        or not set(candidate.kinematic_sha256s).isdisjoint(
            verification.kinematic_sha256s
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
        or reference.context_sha256s != applied.context_sha256s
        or reference.current_dimensions != applied.current_dimensions
        or set(reference.observations) != set(applied.observations)
    ):
        raise ValueError("recurrence application validation provenance drifted")
    relative, absolute = _validated_tolerances(
        relative_tolerance,
        absolute_tolerance,
    )
    maximum_absolute = Decimal(0)
    maximum_relative = Decimal(0)
    maximum_ratio = Decimal(0)
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
        "maximum_absolute_residual": _decimal_string(maximum_absolute),
        "maximum_relative_residual": _decimal_string(maximum_relative),
        "maximum_tolerance_ratio": _decimal_string(maximum_ratio),
        "reference_observation_batch_sha256": (reference.observation_batch_sha256),
        "applied_observation_batch_sha256": (applied.observation_batch_sha256),
    }


def _relation_residuals(
    relation_kind: _RelationKind,
    current: Sequence[_ComplexDecimal],
    representative: Sequence[_ComplexDecimal] | None,
    *,
    relative_tolerance: Decimal,
    absolute_tolerance: Decimal,
) -> tuple[Decimal, Decimal, Decimal]:
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
) -> tuple[Decimal, Decimal, Decimal]:
    if len(left) != len(right):
        raise ValueError("recurrence numerical comparison width is invalid")
    maximum_absolute = Decimal(0)
    maximum_relative = Decimal(0)
    maximum_ratio = Decimal(0)
    for left_value, right_value in zip(left, right, strict=True):
        signed = right_value if sign == 1 else (-right_value[0], -right_value[1])
        difference = max(
            abs(left_value[0] - signed[0]),
            abs(left_value[1] - signed[1]),
        )
        scale = max(
            abs(left_value[0]),
            abs(left_value[1]),
            abs(right_value[0]),
            abs(right_value[1]),
        )
        allowed = absolute_tolerance + relative_tolerance * scale
        relative = (
            Decimal(0)
            if difference == 0
            else difference / max(scale, absolute_tolerance)
        )
        ratio = Decimal(0) if difference == 0 else difference / allowed
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
            "current_id": current_id,
            "representative_id": representative_id,
            "relation_kind": relation_kind,
            "point_sha256s": list(capture.point_sha256s),
            "context_sha256s": list(capture.context_sha256s),
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
    if any(
        row.runtime_slot >= len(plan.runtime_defaults)
        for row in plan.parameter_projection
    ):
        raise ArtifactError("recurrence runtime parameter defaults are incomplete")
    return plan.runtime_defaults


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
            "certificates": [
                certificate.to_json_dict() for certificate in certificates
            ],
            "mappings": [_mapping_payload(certificate) for certificate in certificates],
        }
    )


def _kinematic_sha256(point: ValidationPointRecord) -> str:
    return _canonical_sha256(
        [[float(value).hex() for value in momentum] for momentum in point.four_vectors]
    )


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
    normalized = value.normalize()
    return "0" if normalized == 0 else str(normalized)


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


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
